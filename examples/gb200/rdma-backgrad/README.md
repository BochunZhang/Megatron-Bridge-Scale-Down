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

wrapper 设置 `NCCL_NET=IB`，并设置 `NCCL_P2P_DISABLE=1`、`NCCL_SHM_DISABLE=1`
和 `NCCL_NVLS_ENABLE=0`，禁止 NCCL 使用 GPU P2P/NVLink、共享内存和 NVLink
Switch 回退。只有 NCCL 日志同时确认 `NET/IB` 和 `GDRDMA` 时，wrapper 才会接受
本次运行结果。

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
- 具有可用的 ConnectX HCA 和支持 GPUDirect RDMA 的 NCCL。`NCCL_IB_HCA`
  可以是一个 HCA，也可以是逗号分隔的 HCA 列表；应根据本机拓扑选择。
- 强制 IB 的单节点路径必须得到硬件和 NCCL 网络插件支持。如果 NCCL 无法让
  同节点 rank 通过 HCA 互通，程序会失败；不能把 Socket 或 NVLink 结果当作
  RDMA 结果。
- 每次运行使用新的输出目录，避免历史 NCCL 日志干扰路径检查。

## 推荐调用

```bash
bash examples/gb200/rdma-backgrad/run_gb200_c2c_rdma_benchmark.sh \
  --gpus 0,1,2,3 \
  --hca mlx5_bond_0 \
  --output-dir results/gb200/rdma-backward/c2c-rdma-$(date +%s)
```

需要生成 Nsight Systems trace 时，在命令末尾增加 `--nsys`：

```bash
bash examples/gb200/rdma-backgrad/run_gb200_c2c_rdma_benchmark.sh \
  --gpus 0,1,2,3 \
  --hca mlx5_bond_0 \
  --nsys \
  --output-dir results/gb200/rdma-backward/c2c-rdma-$(date +%s)
```

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
wrapper 也会检查 `/sys/class/infiniband/<HCA>` 是否存在。

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
export NCCL_SHM_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH
export NCCL_DEBUG_FILE=/tmp/gb200-c2c-rdma-manual/nccl-%h-%p.log

mkdir -p /tmp/gb200-c2c-rdma-manual
torchrun --standalone --nnodes=1 --nproc-per-node=4 --no-python \
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
| `--hca` | 必填 | ConnectX HCA 名称或逗号分隔列表 |
| `--c2c-buffer-mib` | `512` | H2D/D2H pinned host 和 GPU buffer 大小 |
| `--rdma-buffer-mib` | `256` | NCCL all-reduce GPU buffer 大小 |
| `--warmup-iterations` | `5` | 每个拷贝方向的预热次数 |
| `--copy-iterations` | `20` | 每个拷贝方向的计时次数 |
| `--rdma-warmup-seconds` | `3` | concurrent 测量前保持 RDMA 的时间 |
| `--rdma-ready-timeout-seconds` | `120` | 首个 RDMA collective 的超时时间 |
| `--nsys` | 关闭 | 为每个 torchrun worker 生成 Nsight Systems trace |
| `--output-dir` | `/tmp/gb200-c2c-rdma-<user>` | 日志和 rank 0 JSON 目录 |

## 强制 RDMA 配置

| 环境变量 | 值 | 目的 |
| --- | --- | --- |
| `NCCL_NET` | `IB` | 选择 NCCL IB 网络后端 |
| `NCCL_IB_DISABLE` | `0` | 开启 IB |
| `NCCL_IB_HCA` | 用户指定 | 选择 ConnectX HCA |
| `NCCL_P2P_DISABLE` | `1` | 禁止 GPU P2P，包括 NVLink/PCI P2P |
| `NCCL_SHM_DISABLE` | `1` | 禁止同机共享内存传输 |
| `NCCL_NVLS_ENABLE` | `0` | 禁止 NVLink Switch collective |
| `NCCL_MNNVL_ENABLE` | `0` | 禁止 MNNVL 路径 |
| `NCCL_NET_GDR_LEVEL` | `PHB` | 允许 PHB 范围的 GPU Direct RDMA |
| `NCCL_NET_GDR_C2C` | `1` | 开启 C2C GPU Direct RDMA |

环境变量只是启动配置，最终以 NCCL 日志为准。运行失败或日志出现 Socket 回退时，
wrapper 不会报告成功结果。

这里的 RDMA 路径指后台 NCCL `all_reduce` 的数据传输；torchrun rendezvous 和
Gloo 控制组仍可能使用本机 TCP，它们不属于被测的 GPU 数据流量。

## 输出与判读

输出目录包含：

- `result.json`：rank 0 汇总结果。
- `torchrun.log`：4 个 worker 的标准输出和错误输出。
- `nccl-<host>-<pid>.log`：NCCL 初始化、网络拓扑和实际传输路径。
- 使用 `--nsys` 时还会生成 `nsys-<pid>.nsys-rep` 等 Nsight Systems 报告文件。

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
    "NCCL_SHM_DISABLE": "1"
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

## Nsight Systems

默认不启动 `nsys`。传入 `--nsys` 后，wrapper 会让每个 worker 执行类似下面的
命令，并用 `%p` 按进程 ID 区分报告，避免 4 个 rank 覆盖同一个文件：

```text
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --output results/.../nsys-%p python gb200_c2c_rdma_benchmark.py ...
```

Python 程序通过 `torch.cuda.nvtx.range_push/range_pop` 标记
`phase_baseline`、`phase_rdma_warmup`、`phase_concurrent`、H2D/D2H timed copy
以及每次 `rdma_all_reduce`。打开 `.nsys-rep` 后可以观察 CUDA DMA、NCCL kernel、
RDMA stream 和 concurrent 阶段的重叠关系。`--nsys` 要求 `nsys` 已加入 `PATH`。

## 本地检查

```bash
bash -n examples/gb200/rdma-backgrad/run_gb200_c2c_rdma_benchmark.sh
python3 -m py_compile examples/gb200/rdma-backgrad/gb200_c2c_rdma_benchmark.py
uv run python -m pytest tests/unit_tests/scripts/performance/test_gb200_c2c_rdma_benchmark.py
```

单元测试只覆盖带宽计算、汇总和环境校验等硬件无关逻辑，不能替代真实 NCCL
IB/GDRDMA 运行。
