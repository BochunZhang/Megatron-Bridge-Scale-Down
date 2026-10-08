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
2. 单独运行固定次数的 4-rank NCCL `all_reduce`，记录 CUDA event 单次完成时间；首个
   communicator warmup 不计入 alone 统计。
3. 启动后台 NCCL `all_reduce`，等待首个 collective 完成并预热。
4. 在 RDMA 仍运行时再次测量 H2D、D2H concurrent 带宽，同时记录与 C2C 时间窗口
   重叠的 all-reduce CUDA event 完成时间。
5. rank 0 汇总所有 rank 的均值，并计算：

   `drop_percent = (baseline_mean_gb_s - concurrent_mean_gb_s) / baseline_mean_gb_s * 100`

wrapper 设置 `NCCL_NET=IB`，并设置 `NCCL_P2P_DISABLE=1`、`NCCL_NVB_DISABLE=1`、`NCCL_PXN_DISABLE=1`、`NCCL_SHM_DISABLE=1`
和 `NCCL_NVLS_ENABLE=0`，禁止 NCCL 使用 GPU P2P/NVLink、共享内存和 NVLink
Switch 回退。只有 NCCL 日志同时确认 `NET/IB` 和 `GDRDMA` 时，wrapper 才会接受
本次运行结果。

baseline 只包含 H2D/D2H；alone 只包含 all-reduce；concurrent 才是 all-reduce 与
H2D/D2H 的重叠测试。H2D 和 D2H 在每个阶段内仍然顺序执行。

默认 C2C buffer 为 512 MiB、每个方向计时 20 次，因此每个方向传输 10 GiB；RDMA
buffer 为 256 MiB，alone 默认计时 20 次 all-reduce。可以用
`--rdma-alone-iterations` 调整 alone 样本数。

## P2P send/recv 与 C2C 竞争测试

如果要测试点对点流量，使用新增的 `run_gb200_c2c_with_gpu_rdma_p2p_benchmark.sh`：

```bash
bash examples/gb200/rdma-backgrad/run_gb200_c2c_with_gpu_rdma_p2p_benchmark.sh \
  --gpus 0,1,2,3 \
  --hca mlx5_bond_0
```

不传 `--hca` 时保持 NCCL 的 HCA 自动选择；默认输出目录为
`results/gb200/rdma-backward/c2c-gpu-rdma-p2p-$(date +%s)`，脚本会自动保存
`run.log`、`torchrun.log`、NCCL 日志和 `result.json`。参数和原 benchmark 相同地使用
`torchrun`、`numarun`、loopback Gloo 控制组、严格的 `NCCL_NET=IB`/GDRDMA 配置，
也支持 `--nsys`。

P2P wrapper 的主要参数为：

| 参数 | 默认值 | 作用 |
| --- | ---: | --- |
| `--c2c-buffer-mib` | `512` | H2D/D2H pinned host 和 GPU buffer 大小 |
| `--p2p-buffer-mib` | `256` | 每次 send/recv 的 GPU payload 大小 |
| `--warmup-iterations` | `5` | C2C 和 P2P warmup 次数 |
| `--copy-iterations` | `20` | baseline C2C 计时次数 |
| `--p2p-iterations` | `20` | alone 和 concurrent 在 warmup 后的 P2P 计时次数 |

P2P 拓扑固定为单向 pair `0 -> 1`、`2 -> 3`：

1. `baseline` 在每个 rank 上分别测 H2D 和 D2H；send rank（0、2）的 D2H baseline
   用于 send 对比，recv rank（1、3）的 H2D baseline 用于 recv 对比。
2. `p2p_alone` 只运行匹配的 NCCL P2P send/recv，没有 H2D/D2H；前
   `--warmup-iterations` 个 P2P 操作用于 communicator warmup，不计入统计。
3. `p2p_send_d2h` 中只有 send rank 执行 D2H，recv rank 只执行匹配的 irecv，得到
   send 与 D2H 的直接竞争结果。前 `--warmup-iterations` 个 P2P/拷贝组合只用于 warmup。
4. `p2p_recv_h2d` 中只有 recv rank 执行 H2D，send rank 只执行匹配的 isend，得到
   recv 与 H2D 的直接竞争结果，warmup 规则相同。

P2P 脚本使用一个 Gloo 进程组做 phase barrier 和结果汇总。每个 P2P 操作都通过
`dist.batch_isend_irecv` 发起，并在专用 CUDA stream 上等待
`request.wait()` 和 stream 完成。P2P 的每个 phase 都先执行相同数量的 warmup，再开始
CUDA event 计时；因此 alone 与 concurrent 不会因为 phase 的第一个通信操作而产生额外偏差。
warmup 结束后还会让四个 rank 在 Gloo barrier 对齐，以消除 send/recv 两侧 C2C warmup
工作量不同造成的启动偏移。
`result.json` 的 `p2p_summary.send` 比较
`p2p_send_alone` 与 `p2p_send_with_d2h`，`p2p_summary.recv` 比较
`p2p_recv_alone` 与 `p2p_recv_with_h2d`。每一项都报告：

- `alone_mean_completion_ms`、`with_d2h_mean_completion_ms` 或
  `with_h2d_mean_completion_ms`：P2P stream 上 CUDA event 测得的单次 send/recv 完成时间，
  可与 nsys 中一次完整 P2P CUDA/NCCL activity 的起止范围比较；不要直接和单个
  `ncclDevKernel` 的 duration 或重叠 kernel duration 求和比较；
- `alone_mean_host_completion_ms`、`with_d2h_mean_host_completion_ms` 或
  `with_h2d_mean_host_completion_ms`：主机墙钟观测的单次完成时间，包含 Python 发起、
  `Work.wait()` 和 stream 同步开销，用于诊断调度开销，不能直接和 nsys kernel duration 对比；
- `alone_mean_bandwidth_gb_s`、`with_d2h_mean_bandwidth_gb_s` 或
  `with_h2d_mean_bandwidth_gb_s`：每个 sender/receiver rank 的完成 tensor payload 带宽，按
  十进制 GB/s 计算，不是网卡 wire-level 带宽；
- `completion_slowdown_percent`：有对应 C2C 拷贝时完成时间的增加比例；
- `bandwidth_drop_percent`：有对应 C2C 拷贝时 P2P payload 带宽的下降比例。

P2P API 中的 `send`/`recv` 是 NCCL 的两端操作；`NCCL_P2P_DISABLE=1` 禁止的是
NVLink/PCI 的直接 GPU P2P transport，配合 `NCCL_NET=IB`、`NCCL_SHM_DISABLE=1` 和
GDRDMA 配置后，实际路径仍必须以 NCCL 日志中的 `NET/IB`、`GDRDMA` 为准。脚本发现
Socket fallback 时会失败退出。两组 pair 的汇总是每个参与 rank 的 per-peer 数值；不把
两个 pair 的 payload 相加成单个 wire-rate。

## CPU RDMA Verbs P2P 与 C2C 竞争测试

CPU 背景流量由 Bash 启动的两个 linux-rdma perftest `ib_write_bw` 进程提供：
GPU0 对应的 HCA 接收，GPU3 对应的 HCA 发送。后台流量启动 10 秒后，Bash 在
GPU0 上启动单个 Python 进程测量 D2H/H2D C2C 带宽：

```bash
bash examples/gb200/rdma-background/run_gb200_c2c_with_cpu_rdma_p2p_benchmark.sh \
  --rdma-size-mib 256
```

GPU/HCA 绑定固定如下，不接受 `--gpus` 参数：

| GPU | HCA | 后台进程角色 | C2C 测量 |
| --- | --- | --- | --- |
| 0 | `mlx5_bond_0` | server / receiver | D2H 和 H2D |
| 3 | `mlx5_bond_3` | client / sender | 无 |

Bash 在每次运行时读取实时 `rdma link show` 和 `ip addr`，解析 HCA 对应的网卡、
bond master 和 IPv4 地址；GPU 所在 NUMA 节点从 PCI sysfs 读取。
不读取预先保存的拓扑或地址文本，不调用 `nvidia-smi topo`，也不固定 IP 地址。

两个 `ib_write_bw` 进程都绑定到对应 GPU 所在的 NUMA 节点，但 RDMA WRITE 的
buffer 使用 host memory。`-R` 选择 RDMA CM，`-d` 和 `--bind_source_ip` 分别固定
HCA 和源 IP；`-s` 固定消息大小，`--run_infinitely` 让 GPU3 对应的 sender 持续
向 GPU0 对应的 receiver 发送数据。Bash 先启动 receiver，留出 1 秒打开端点，
再启动 sender；从 sender 启动开始固定等待 10 秒，并每秒检查两个进程是否存活，
然后运行 Python。Python 完成后停止并回收两个后台进程。
启动失败、Python 失败或收到终止信号时也会清理后台进程。

Python 只在 GPU0 上测量有背景流量时的 `d2h` 和 `h2d` 带宽，结果写入
`result.json`；不再测量无背景 baseline 或四个方向/角色组合，也不计算
`drop_percent`。这一模式不使用 Gloo、NCCL 或 `torchrun`。

可用参数如下：

- `--c2c-buffer-mib`、`--warmup-iterations`、`--copy-iterations`：C2C buffer 大小和拷贝次数。
- `--ib-write-bw`：perftest 可执行文件。
- `--rdma-size-mib`、`--rdma-qp`、`--rdma-tx-depth`、`--rdma-report-interval`、
  `--rdma-port`：固定大小的后台 RDMA 流量参数。
- `--nsys`：用 Nsight Systems 采集 GPU0 上的 Python C2C 测量。
- `--output-dir`：输出目录。

默认输出目录是 `results/gb200/rdma-background/c2c-cpu-rdma-p2p-<timestamp>`，
其中包含 `run.log`、`c2c.log`、`rdma_logs/gpu0.recv.log`、
`rdma_logs/gpu3.send.log` 和 `result.json`。运行需要 CUDA 版 PyTorch、Python、
`ib_write_bw`、`ip`、`rdma`、`nvidia-smi`、`numactl` 和 `setsid`；`setsid` 为各进程
建立独立进程组，以便统一清理子进程。启用 `--nsys` 时还需要 Nsight Systems。

如果机器支持 Gloo ibverbs，旧实现仍保留在
`gb200_c2c_with_cpu_rdma_gloo_p2p_benchmark.py` 及对应的
`run_gb200_c2c_with_cpu_rdma_gloo_p2p_benchmark.sh`；它不适用于当前开发机。

### 单节点为什么仍然有 TCP/Gloo

以下说明适用于 GPU all-reduce、GPU P2P 和旧的 Gloo ibverbs 模式。
单节点不等于完全不需要 TCP。这些 wrapper 的 `torchrun` `c10d` rendezvous 都需要
TCPStore 来让 4 个 worker 相互发现。原有 GPU all-reduce 和 GPU P2P 脚本的 Gloo
进程组还使用 TCP socket 做 phase barrier、停止/同步和 CPU 对象汇总；这些是控制面，
不是被测的 GPU 数据面。真正的 GPU 通信在默认 NCCL 进程组和 CUDA stream 上执行，
数据面仍由 `NCCL_NET=IB` 选择 RDMA。旧的 Gloo ibverbs wrapper 的数据组使用
`IBVERBS`。这些分布式模式的 rendezvous 都使用 TCPStore。CPU perftest wrapper
由 Bash 管理进程，RDMA 连接由 `ib_write_bw` 直接建立，不需要 TCPStore 或 Gloo。

分布式 wrapper 将控制面固定到本机 IPv4 loopback：

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
CPU 背景实验是独立模式：新的 perftest 结果不读取或验证 NCCL 日志，`result.json`
中的 `background_backend` 会标出 `perftest/ib_write_bw`；旧 Gloo 版本则标出
`gloo/ibverbs`。

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

以下调用层次仅适用于分布式模式；CPU perftest wrapper 直接按 PCI sysfs 的 NUMA
节点绑定进程。`.cache/numarun` 是一个依赖 `LOCAL_RANK` 和 `LOCAL_WORLD_SIZE` 的 worker 包装器。
因此必须让 `torchrun` 先创建 worker，再由每个 worker 执行 `numarun`：

```text
torchrun --no-python numarun python gb200_c2c_with_gpu_rdma_allreduce_benchmark.py ...
```

不能写成 `numarun torchrun ...`，因为外层 `numarun` 启动时还没有 rank 环境变量，
不会进行 NUMA 绑定。`--no-python` 必须保留，因为 `numarun` 是 shell wrapper，
不是 Python 文件。

wrapper 默认导出 `NUMARUN_MEMBIND=1`，让 `numarun` 同时执行 CPU 绑定和 NUMA
内存绑定。它按照 `LOCAL_RANK` 和 CPU socket 数量分配 CPU/NUMA；运行前应确认
该启发式映射符合本机 GPU、CPU socket 和 HCA 拓扑。

## 运行条件

以下是 GPU RDMA 分布式模式的运行条件；CPU perftest 模式的依赖见上面的独立说明。

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
bash examples/gb200/rdma-backgrad/run_gb200_c2c_with_gpu_rdma_allreduce_benchmark.sh \
  --gpus 0,1,2,3 \
  --hca mlx5_bond_0
```

如果不指定 `--hca`，wrapper 不会设置或覆盖 `NCCL_IB_HCA`；如果调用环境中也没有该
变量，则由 NCCL 自动选择 HCA：

```bash
bash examples/gb200/rdma-backgrad/run_gb200_c2c_with_gpu_rdma_allreduce_benchmark.sh \
  --gpus 0,1,2,3
```

需要生成 Nsight Systems trace 时，在命令末尾增加 `--nsys`：

```bash
bash examples/gb200/rdma-backgrad/run_gb200_c2c_with_gpu_rdma_allreduce_benchmark.sh \
  --gpus 0,1,2,3 \
  --hca mlx5_bond_0 \
  --nsys
```

未传入 `--output-dir` 时，脚本自动使用
`results/gb200/rdma-backward/c2c-gpu-rdma-allreduce-<timestamp>`；也可以显式传入自定义目录。

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
export NCCL_DEBUG_FILE=/tmp/gb200-c2c-gpu-rdma-allreduce-manual/nccl-%h-%p.log

mkdir -p /tmp/gb200-c2c-gpu-rdma-allreduce-manual
export GLOO_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=lo
export NCCL_SOCKET_FAMILY=AF_INET
torchrun --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:0 \
  --nnodes=1 --nproc-per-node=4 --no-python \
  numarun python examples/gb200/rdma-backgrad/gb200_c2c_with_gpu_rdma_allreduce_benchmark.py \
  --c2c-buffer-mib 512 \
  --rdma-buffer-mib 256 \
  --warmup-iterations 5 \
  --copy-iterations 20 \
  --rdma-warmup-seconds 3 \
  --rdma-ready-timeout-seconds 120 \
  --output /tmp/gb200-c2c-gpu-rdma-allreduce-manual/result.json
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
| `--rdma-alone-iterations` | `20` | alone 阶段的计时 all-reduce 次数；首个 warmup 不计入 |
| `--nsys` | 关闭 | 在 torchrun 外层追踪 launcher 和所有 worker，生成一个进程树报告 |
| `--output-dir` | `results/gb200/rdma-backward/c2c-gpu-rdma-allreduce-<timestamp>` | 日志和 rank 0 JSON 目录 |

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
  "rdma_completion": {
    "alone_mean_completion_ms": 3.2,
    "concurrent_mean_completion_ms": 4.1,
    "alone_mean_host_completion_ms": 3.5,
    "concurrent_mean_host_completion_ms": 4.6,
    "slowdown_percent": 28.125,
    "alone_mean_iterations": 20.0,
    "concurrent_mean_iterations": 24.0
  },
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

P2P 脚本的汇总结构如下；`send` 使用 sender ranks 0、2，`recv` 使用 receiver ranks
1、3：

```json
{
  "p2p_topology": {
    "pairs": [[0, 1], [2, 3]],
    "sender_ranks": [0, 2],
    "receiver_ranks": [1, 3]
  },
  "p2p_summary": {
    "send": {
      "alone_mean_completion_ms": 4.0,
      "with_d2h_mean_completion_ms": 4.8,
      "alone_mean_host_completion_ms": 4.3,
      "with_d2h_mean_host_completion_ms": 5.2,
      "completion_slowdown_percent": 20.0,
      "alone_mean_bandwidth_gb_s": 64.0,
      "with_d2h_mean_bandwidth_gb_s": 53.3,
      "bandwidth_drop_percent": 16.7,
      "copy_mean_bandwidth_gb_s": 410.0
    },
    "recv": {
      "alone_mean_completion_ms": 4.1,
      "with_h2d_mean_completion_ms": 5.0,
      "alone_mean_host_completion_ms": 4.4,
      "with_h2d_mean_host_completion_ms": 5.5,
      "completion_slowdown_percent": 22.0,
      "alone_mean_bandwidth_gb_s": 62.4,
      "with_h2d_mean_bandwidth_gb_s": 51.2,
      "bandwidth_drop_percent": 17.9,
      "copy_mean_bandwidth_gb_s": 398.0
    }
  }
}
```

`ranks[]` 中的 `p2p_send_alone`、`p2p_send_with_d2h`、`p2p_recv_alone` 和
`p2p_recv_with_h2d` 包含每个 rank 的 `peer_rank`、迭代数、CUDA event 平均完成时间、
主机平均完成时间、传输字节数和 payload 带宽；不属于该 role 的字段为 `null`。
`send_d2h_copy` 和 `recv_h2d_copy`
记录对应竞争阶段的 C2C 带宽。

- `drop_percent > 0` 表示 concurrent 带宽低于 baseline。
- `drop_percent < 0` 表示该次运行中 concurrent 更快，应结合重复运行和测量波动判断。
- `rdma_covers_entire_concurrent_c2c` 是后台线程首尾时间戳的粗粒度覆盖判断，不能
  替代 Nsight 时间线对每个 DMA 操作的精确重叠分析。
- `rdma_completion` 的 `alone_mean_completion_ms` 和
  `concurrent_mean_completion_ms` 是 CUDA event 完成时间；对应的
  `alone_mean_host_completion_ms` 和 `concurrent_mean_host_completion_ms` 是主机观测时间。
  `slowdown_percent > 0` 表示 concurrent 下设备侧 all-reduce 变慢。concurrent 只统计与
  C2C 窗口重叠的 collective，因此两者的迭代数可能不同，不能直接比较总 elapsed。
- 每个 rank 的 `rdma_alone.average_all_reduce_ms` 和
  `rdma_concurrent.average_all_reduce_ms` 是 CUDA event 完成时间；同一对象中的
  `host_average_all_reduce_ms` 可用于检查 Python/同步开销和 rank 间差异。
- 每个 rank 的 `rdma` 是 concurrent 阶段后台线程从启动到停止的完整诊断统计，包含
  C2C 窗口外的 all-reduce；比较 alone 与 concurrent 时应使用上面的
  `rdma_completion` 或对应的 `rdma_alone`/`rdma_concurrent` 字段。
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
`phase_baseline`、`phase_rdma_alone`、`phase_rdma_warmup`、`phase_concurrent`、H2D/D2H timed copy
以及每次 `rdma_all_reduce`。打开 `.nsys-rep` 后可以观察 CUDA DMA、NCCL kernel/stream
和 concurrent 阶段的重叠关系；RDMA 是否实际使用 IB/GDRDMA 仍以 NCCL 日志为准。
`--nsys` 要求 `nsys` 已加入 `PATH`。

P2P 脚本额外标记 `phase_p2p_alone`、`phase_p2p_send_d2h`、`phase_p2p_recv_h2d`、
`p2p_send`、`p2p_recv` 及对应 C2C copy ranges，可在同一时间线上检查指定方向的
P2P 与 D2H/H2D 是否重叠。

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
bash -n examples/gb200/rdma-backgrad/run_gb200_c2c_with_gpu_rdma_allreduce_benchmark.sh
bash -n examples/gb200/rdma-backgrad/run_gb200_c2c_with_gpu_rdma_p2p_benchmark.sh
bash -n examples/gb200/rdma-background/run_gb200_c2c_with_cpu_rdma_p2p_benchmark.sh
bash -n examples/gb200/rdma-backgrad/run_gb200_c2c_with_cpu_rdma_gloo_p2p_benchmark.sh
python3 -m py_compile examples/gb200/rdma-backgrad/gb200_c2c_with_gpu_rdma_allreduce_benchmark.py
python3 -m py_compile examples/gb200/rdma-backgrad/gb200_c2c_with_gpu_rdma_p2p_benchmark.py
python3 -m py_compile examples/gb200/rdma-background/gb200_c2c_with_cpu_rdma_p2p_benchmark.py
python3 -m py_compile examples/gb200/rdma-backgrad/gb200_c2c_with_cpu_rdma_gloo_p2p_benchmark.py
uv run python -m pytest tests/unit_tests/scripts/performance/test_gb200_c2c_with_gpu_rdma_allreduce_benchmark.py
uv run python -m pytest tests/unit_tests/scripts/performance/test_gb200_c2c_with_gpu_rdma_p2p_benchmark.py
uv run python -m pytest tests/unit_tests/scripts/performance/test_gb200_c2c_with_cpu_rdma_p2p_benchmark.py
uv run python -m pytest tests/unit_tests/scripts/performance/test_gb200_c2c_with_cpu_rdma_p2p_launcher.py
uv run python -m pytest tests/unit_tests/scripts/performance/test_gb200_c2c_with_cpu_rdma_gloo_p2p_benchmark.py
```

单元测试只覆盖带宽计算、汇总和环境校验等硬件无关逻辑，不能替代真实 NCCL
IB/GDRDMA 运行。
