# GB200 单节点 C2C 与后台 RDMA 干扰测试

这个示例使用 PyTorch `torch.distributed` 和 `torchrun`，在一台包含 4 张 GPU
的 GB200 节点上测试：当 GPU 之间持续进行 NCCL IB/GDRDMA 通信时，同一节点的
host-to-device（H2D）和 device-to-host（D2H）拷贝带宽是否下降。

测试只支持单节点、4 个 torchrun worker。它不使用 `srun`，也不使用两节点
rendezvous。

## 测试原理

Python 程序使用一个 NCCL 默认进程组产生后台 GPU buffer `all_reduce`，使用两个
独立的 Gloo 进程组完成 barrier、ready 和 stop 控制。每个 rank 在自己的 CUDA
stream 上执行 H2D/D2H 拷贝，另一个 CUDA stream 的 NCCL collective 在后台线程
持续运行：

1. 先测没有 RDMA 负载时的 H2D、D2H baseline。
2. 启动 4-rank NCCL `all_reduce`，等待首个 collective 完成并预热。
3. 在 RDMA 仍运行时再次测量 H2D、D2H concurrent 带宽。
4. rank 0 汇总所有 rank 的均值，并计算：

   `drop_percent = (baseline_mean_gb_s - concurrent_mean_gb_s) / baseline_mean_gb_s * 100`

wrapper 设置 `NCCL_NET=IB`，并设置 `NCCL_P2P_DISABLE=1`、`NCCL_NVB_DISABLE=1`、`NCCL_PXN_DISABLE=1`、`NCCL_SHM_DISABLE=1`
和 `NCCL_NVLS_ENABLE=0`，禁止 NCCL 使用 GPU P2P/NVLink、共享内存和 NVLink
Switch 回退。只有 NCCL 日志同时确认 `NET/IB` 和 `GDRDMA` 时，wrapper 才会接受
本次运行结果。

### 单节点为什么仍然有 TCP/Gloo

单节点不等于完全不需要 TCP。`torchrun` 的 `c10d` rendezvous 需要一个
TCPStore 来让 4 个 worker 相互发现；程序中的两个 Gloo 进程组也需要 TCP socket，
分别用于 phase barrier、停止后台线程和汇总 CPU 对象。这些是控制面，不是被测的
GPU 数据面。真正的后台 all-reduce 在默认 NCCL 进程组和 CUDA stream 上执行，数据面
仍由 `NCCL_NET=IB` 选择 RDMA。

wrapper 将控制面固定到本机 IPv4 loopback：

```text
GLOO_SOCKET_IFNAME=lo
NCCL_SOCKET_IFNAME=lo
NCCL_SOCKET_FAMILY=AF_INET
torchrun --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:0 ...
```

这里的 `NCCL_SOCKET_IFNAME=lo` 只约束 NCCL 的 socket bootstrap；它不会把
`NCCL_NET=IB` 的 collective 改成 TCP。运行后应在 `nccl-*.log` 中看到 `NET/IB` 和
`GDRDMA`，并且没有 `NET/Socket`。因此不应为了消除控制面的 TCP 而删除 Gloo：将
控制组改成 NCCL 会把控制 collective 也放进 GPU/NCCL 流量，改变被测负载，并可能与
后台 all-reduce 发生 collective 顺序冲突。

如果看到 `TCP client failed to connect/validate to host`，先看错误前缀和第一条失败：

- `TCPStore.cpp` 通常表示 torchrun rendezvous；检查 `run.log` 中的启动参数和
  `torchrun.log`，当前 wrapper 应使用 `127.0.0.1:0`，不会依赖 DLC 注入的
  `MASTER_ADDR`/`MASTER_PORT`。
- `[gloo/transport/tcp]` 表示 Gloo 控制组的 socket；确认 wrapper 的 `lo` 配置
  没有被后续脚本覆盖。
- 如果还没有生成 `nccl-*.log`，失败发生在 NCCL 初始化之前，TCP 报错通常是首个
  rank 退出后其它 rank 的连带错误。
- 如果第一条错误是 `numactl`、`set_mempolicy` 或权限错误，说明容器不允许
  `numarun` 的 NUMA 内存绑定；可以临时用 `NUMARUN_MEMBIND=0` 定位问题，确认
  启动链路后再恢复默认的 `1`。

## `torchrun` 与 `numarun` 的调用层次

`.cache/numarun` 是一个依赖 `LOCAL_RANK` 和 `LOCAL_WORLD_SIZE` 的 worker 包装器。
因此必须让 `torchrun` 先创建 worker，再由每个 worker 执行 `numarun`：

```text
torchrun --no-python numarun python gb200_c2c_rdma_benchmark.py ...
```

不能写成 `numarun torchrun ...`，因为外层 `numarun` 启动时还没有 rank 环境变量，
不会进行 NUMA 绑定。`--no-python` 必须保留，因为 `numarun` 是 shell wrapper，
不是 Python 文件。

wrapper 默认导出 `NUMARUN_MEMBIND=1`，让 `numarun` 同时执行 CPU 绑定和 NUMA
内存绑定。它按照 `LOCAL_RANK` 和 CPU socket 数量分配 CPU/NUMA；运行前应确认
该启发式映射符合本机 GPU、CPU socket 和 HCA 拓扑。

## 运行条件

- 单个 GB200 节点，至少 4 张可用 GPU。
- `torchrun`、`python`、`nvidia-smi`、`numactl` 和 `numarun` 可用。
- PyTorch 同时包含 NCCL 和 Gloo。
- 具有可用的 ConnectX HCA 和支持 GPUDirect RDMA 的 NCCL。可以通过 `--hca` 指定一个
  HCA、NCCL HCA 前缀或逗号分隔的列表；省略时由 NCCL 自动选择可用 HCA。
- 强制 IB 的单节点路径必须得到硬件和 NCCL 网络插件支持。如果 NCCL 无法让
  同节点 rank 通过 HCA 互通，程序会失败；不能把 Socket 或 NVLink 结果当作
  RDMA 结果。
- 每次运行使用新的输出目录，避免历史 NCCL 日志干扰路径检查。

## 推荐调用

```bash
bash examples/gb200/rdma-backgrad/run_gb200_c2c_rdma_benchmark.sh \
  --gpus 0,1,2,3 \
  --hca mlx5_bond_0
```

如果不指定 `--hca`，wrapper 不会设置或覆盖 `NCCL_IB_HCA`；如果调用环境中也没有该
变量，则由 NCCL 自动选择 HCA：

```bash
bash examples/gb200/rdma-backgrad/run_gb200_c2c_rdma_benchmark.sh \
  --gpus 0,1,2,3
```

需要生成 Nsight Systems trace 时，在命令末尾增加 `--nsys`：

```bash
bash examples/gb200/rdma-backgrad/run_gb200_c2c_rdma_benchmark.sh \
  --gpus 0,1,2,3 \
  --hca mlx5_bond_0 \
  --nsys
```

未传入 `--output-dir` 时，脚本自动使用
`results/gb200/rdma-backward/c2c-rdma-<timestamp>`；也可以显式传入自定义目录。

## 检查 `mlx5_bond_0` 是否存在

先列出本机识别到的 RDMA HCA：

```bash
ls -1 /sys/class/infiniband
```

如果输出包含 `mlx5_bond_0`，说明该 HCA 设备存在。也可以直接检查：

```bash
test -d /sys/class/infiniband/mlx5_bond_0 \
  && echo "mlx5_bond_0 exists" \
  || echo "mlx5_bond_0 missing"
```

确认 HCA 端口和链路状态：

```bash
ls -1 /sys/class/infiniband/mlx5_bond_0/ports
ibdev2netdev                 # 如果系统安装了 rdma-core
ibv_devinfo -d mlx5_bond_0        # 查看端口 state、active_mtu 等信息
cat /sys/class/infiniband/mlx5_bond_0/ports/1/state
```

如果端口目录不是 `1`，把最后一条命令中的端口号替换成实际值。输出应显示端口为
`ACTIVE`。如果 `mlx5_bond_0` 不存在，请从
`ls -1 /sys/class/infiniband` 的实际名称中选择 HCA，并把它传给 `--hca`。
wrapper 不会预先检查 HCA 路径，最终是否使用了目标 HCA 以 NCCL 日志为准。

还应检查 HCA 与 GPU 的 NUMA 归属是否合理：

```bash
cat /sys/class/infiniband/mlx5_bond_0/device/numa_node
nvidia-smi -i 0 --query-gpu=pci.bus_id --format=csv,noheader
```

`numarun` 会按 rank 绑定 CPU 和 NUMA 内存，但最终的 GPU/HCA 邻接关系仍应结合
机器拓扑确认。

如果 `numarun` 只存在于仓库的 `.cache/numarun`，wrapper 会自动使用该文件
作为 fallback。若希望直接手工调用 torchrun，先确保 PATH 中的 `numarun` 是可执行
文件，然后执行：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NUMARUN_MEMBIND=1
export NCCL_IB_HCA='=mlx5_bond_0'
export NCCL_IB_DISABLE=0
export NCCL_MNNVL_ENABLE=0
export NCCL_NET=IB
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_NET_GDR_C2C=1
export NCCL_P2P_DISABLE=1
export NCCL_NVB_DISABLE=1
export NCCL_PXN_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH
export NCCL_DEBUG_FILE=/tmp/gb200-c2c-rdma-manual/nccl-%h-%p.log

mkdir -p /tmp/gb200-c2c-rdma-manual
export GLOO_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=lo
export NCCL_SOCKET_FAMILY=AF_INET
torchrun --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:0 \
  --nnodes=1 --nproc-per-node=4 --no-python \
  numarun python examples/gb200/rdma-backgrad/gb200_c2c_rdma_benchmark.py \
  --c2c-buffer-mib 512 \
  --rdma-buffer-mib 256 \
  --warmup-iterations 5 \
  --copy-iterations 20 \
  --rdma-warmup-seconds 3 \
  --rdma-ready-timeout-seconds 120 \
  --output /tmp/gb200-c2c-rdma-manual/result.json
```

上面的 `NCCL_IB_HCA='=mlx5_bond_0'` 使用 NCCL 的精确匹配前缀。使用多个 HCA 时，
例如：`export NCCL_IB_HCA='=mlx5_bond_0,=mlx5_bond_1'`。

wrapper 支持的参数：

| 参数 | 默认值 | 作用 |
| --- | ---: | --- |
| `--gpus` | `0,1,2,3` | 暴露给 4 个 rank 的物理 GPU 列表，必须正好 4 张 |
| `--hca` | NCCL 自动选择 | ConnectX HCA 名称、NCCL 前缀或逗号分隔列表 |
| `--c2c-buffer-mib` | `512` | H2D/D2H pinned host 和 GPU buffer 大小 |
| `--rdma-buffer-mib` | `256` | NCCL all-reduce GPU buffer 大小 |
| `--warmup-iterations` | `5` | 每个拷贝方向的预热次数 |
| `--copy-iterations` | `20` | 每个拷贝方向的计时次数 |
| `--rdma-warmup-seconds` | `3` | concurrent 测量前保持 RDMA 的时间 |
| `--rdma-ready-timeout-seconds` | `120` | 首个 RDMA collective 的超时时间 |
| `--nsys` | 关闭 | 在 torchrun 外层追踪 launcher 和所有 worker，生成一个进程树报告 |
| `--output-dir` | `results/gb200/rdma-backward/c2c-rdma-<timestamp>` | 日志和 rank 0 JSON 目录 |

## 强制 RDMA 配置

| 环境变量 | 值 | 目的 |
| --- | --- | --- |
| `NCCL_NET` | `IB` | 选择 NCCL IB 网络后端 |
| `NCCL_IB_DISABLE` | `0` | 开启 IB |
| `NCCL_IB_HCA` | 传入 `--hca` 时设置，否则保留原值 | 选择 ConnectX HCA；未设置时由 NCCL 自动选择 |
| `NCCL_P2P_DISABLE` | `1` | 禁止 GPU P2P，包括 NVLink/PCI P2P |
| `NCCL_NVB_DISABLE` | `1` | 禁止经中间 GPU 的同节点 NVLink 路径 |
| `NCCL_PXN_DISABLE` | `1` | 禁止经 NVLink 和中间 GPU 使用非本地 NIC |
| `NCCL_SHM_DISABLE` | `1` | 禁止同机共享内存传输 |
| `NCCL_NVLS_ENABLE` | `0` | 禁止 NVLink Switch collective |
| `NCCL_MNNVL_ENABLE` | `0` | 禁止 MNNVL 路径 |
| `NCCL_NET_GDR_LEVEL` | `PHB` | 允许 PHB 范围的 GPU Direct RDMA |
| `NCCL_NET_GDR_C2C` | `1` | 开启 C2C GPU Direct RDMA |
| `GLOO_SOCKET_IFNAME` | `lo` | 将 Gloo 控制组固定到本机 loopback |
| `NCCL_SOCKET_IFNAME` | `lo` | 将 NCCL socket bootstrap 固定到本机 loopback |
| `NCCL_SOCKET_FAMILY` | `AF_INET` | 使用 IPv4 loopback，避免容器 IPv6/主机名解析问题 |

环境变量只是启动配置，最终以 NCCL 日志为准。省略 `--hca` 只会放开 HCA 选择，
wrapper 仍然强制 `NCCL_NET=IB`、GDRDMA 和其他 RDMA 相关配置。运行失败或日志出现
Socket 回退时，wrapper 不会报告成功结果。

`NCCL_PXN_DISABLE=1` 会拒绝通过 NVLink 和中间 GPU 把数据转到非本地 HCA；因此如果
某个 GPU 没有可用的直连 HCA/C2C 路径，NCCL 可能初始化失败。这是严格验证“数据面
走 RDMA 且不借助 NVLink”的预期结果，应根据 NCCL `GRAPH` 日志检查实际拓扑，而不是
把失败改成 Socket 或 NVLink 回退。

例如，`--hca mlx5_bond` 会交给 NCCL 做前缀匹配；如果需要只选择指定设备，可以使用
`--hca '=mlx5_bond_0,=mlx5_bond_1,=mlx5_bond_2'`。

这里的 RDMA 路径指后台 NCCL `all_reduce` 的数据传输；torchrun rendezvous 和
Gloo 控制组使用本机 loopback TCP，它们不属于被测的 GPU 数据流量。

## 输出与判读

输出目录包含：

- `result.json`：rank 0 汇总结果。
- `run.log`：wrapper、torchrun worker 和传输检查的完整运行日志。
- `torchrun.log`：4 个 worker 的标准输出和错误输出。
- `nccl-<host>-<pid>.log`：NCCL 初始化、网络拓扑和实际传输路径。
- 使用 `--nsys` 时还会生成一个 `nsys.nsys-rep`（旧版可能是 `nsys.qdrep`）进程树报告。

`result.json` 关键字段示例：

```json
{
  "summary": {
    "h2d": {
      "baseline_mean_gb_s": 420.0,
      "concurrent_mean_gb_s": 360.0,
      "drop_percent": 14.2857
    },
    "d2h": {
      "baseline_mean_gb_s": 415.0,
      "concurrent_mean_gb_s": 402.0,
      "drop_percent": 3.1325
    }
  },
  "rdma_covers_entire_concurrent_c2c": true,
  "transport": {
    "NCCL_NET": "IB",
    "NCCL_P2P_DISABLE": "1",
    "NCCL_NVB_DISABLE": "1",
    "NCCL_PXN_DISABLE": "1",
    "NCCL_SHM_DISABLE": "1",
    "GLOO_SOCKET_IFNAME": "lo",
    "NCCL_SOCKET_IFNAME": "lo",
    "NCCL_SOCKET_FAMILY": "AF_INET"
  }
}
```

- `drop_percent > 0` 表示 concurrent 带宽低于 baseline。
- `drop_percent < 0` 表示该次运行中 concurrent 更快，应结合重复运行和测量波动判断。
- `rdma_covers_entire_concurrent_c2c` 是后台线程首尾时间戳的粗粒度覆盖判断，不能
  替代 Nsight 时间线对每个 DMA 操作的精确重叠分析。
- `ranks[].rdma.payload_gbit_s` 是完成的 tensor payload 速率，不是网卡 wire-level
  速率；all-reduce 算法、协议和 NCCL 版本都会影响它。
- 后台 all-reduce 每轮会等待完成并同步 stream，循环可能存在短暂 host 间隔，因此
  不能把 `payload_gbit_s` 解读为网卡始终满速。

建议在同一节点、相同 HCA 和相同参数下重复运行多次。程序当前不计算标准差，也不
同时发起 H2D 与 D2H，所以结论限于“两个方向依次测量时的相对带宽变化”。

## DLC 节点环境检查

在 DLC 容器中运行 benchmark 前，可以先执行节点诊断脚本，确认容器看到的环境、NVMe
挂载、RDMA 网卡和设备节点：

```bash
bash examples/gb200/rdma-backgrad/check_dlc_node_environment.sh
```

脚本会自动在仓库根目录创建
`results/gb200/rdma-backward/dlc-node-<timestamp>/node-environment.log`，不需要传入参数
或手动创建输出目录。

脚本按区块输出以下信息：

- `printenv` 的全部环境变量，便于记录 DLC 注入的任务和分布式参数。
- `df -h` 的全部文件系统，以及 `/dev/nvme*` 挂载数量；如果存在 `lsblk`，还会输出
  NVMe 设备摘要。
- `ip addr show` 的完整地址信息和 RDMA 相关接口摘要；如果安装了
  `ibdev2netdev` 或 `rdma`，还会输出 HCA 到网卡的映射和链路状态，因此不会只依赖
  网卡命名（RoCE 网卡可能命名为 `eth*` 或 `enp*`）。
- `/dev` 下的 NVMe 命名空间、`/dev/infiniband` 条目，以及 `/sys/class/nvme`、
  `/sys/class/infiniband` 下的控制器、HCA、端口状态和 NUMA 归属。
- `lspci` 的 NVMe、InfiniBand、Mellanox/NVIDIA 设备摘要（命令存在时）。

`df -h` 统计的是已挂载的 NVMe 文件系统，不等于物理 NVMe 数量；脚本同时报告 `/dev`
块设备数、`/sys` 的 NVMe controller/namespace 数量，避免把挂载数量误认为物理盘数量。
诊断脚本是只读检查，某个可选命令（例如 `ibv_devinfo` 或 `lspci`）不存在时会标记为
unavailable 并继续输出其他信息。

## Nsight Systems

默认不启动 `nsys`。传入 `--nsys` 后，wrapper 将 `nsys profile` 放在 `torchrun`
外层，让单节点的 launcher 和 4 个 worker 进入同一个进程树报告：

```text
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --cuda-trace-scope=process-tree \
  --output results/.../nsys \
  torchrun --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:0 ...
```

Python 程序通过 `torch.cuda.nvtx.range_push/range_pop` 标记
`phase_baseline`、`phase_rdma_warmup`、`phase_concurrent`、H2D/D2H timed copy
以及每次 `rdma_all_reduce`。打开 `.nsys-rep` 后可以观察 CUDA DMA、NCCL kernel/stream
和 concurrent 阶段的重叠关系；RDMA 是否实际使用 IB/GDRDMA 仍以 NCCL 日志为准。
`--nsys` 要求 `nsys` 已加入 `PATH`。

脚本会把 `NSYS_TMPDIR` 默认设为输出目录下的隐藏临时目录，避免容器的 `/tmp` 空间
不足或不可写。运行结束后脚本会检查 `.nsys-rep`/`.qdrep` 是否确实生成；如果只留下
`.qdstrm`，说明采集完成但报告转换没有完成，可以使用同版本 `nsys import` 转换。

如果 `--nsys` 后没有报告，按以下顺序检查：

- `run.log` 是否出现 `nsys is required`、参数不支持、权限或 `permission denied`；
- `torchrun.log` 的第一条错误，尤其是 rendezvous、Gloo、NCCL 或 `numarun` 错误；
  worker 提前退出、超时或收到 SIGTERM 时，Nsight 可能来不及 finalize 报告；
- `test -w <output-dir>`、`df -h <output-dir>` 和 `df -h /tmp`；
- `nsys --version`，确认 DLC 容器中挂载的是 Nsight Systems CLI，而不是只有 Python
  环境；
- `find <output-dir> -maxdepth 1 -name 'nsys*' -o -name '*.qdstrm'`，确认是否生成了
  中间文件。

## 本地检查

```bash
bash -n examples/gb200/rdma-backgrad/run_gb200_c2c_rdma_benchmark.sh
python3 -m py_compile examples/gb200/rdma-backgrad/gb200_c2c_rdma_benchmark.py
uv run python -m pytest tests/unit_tests/scripts/performance/test_gb200_c2c_rdma_benchmark.py
```

单元测试只覆盖带宽计算、汇总和环境校验等硬件无关逻辑，不能替代真实 NCCL
IB/GDRDMA 运行。
