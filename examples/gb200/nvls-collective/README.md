# GB200：基于 torch 的 AG / RS 吞吐对比（ring vs NVLS）

日期：2026-10-13。这个目录用 **torch** 在 4/8/16/32 卡上测 all-gather（AG）、reduce-scatter（RS）
以及两者同时发起的吞吐，并对照 ring 与 NVLS 两种算法。已有的
[`nvls-base-ccl`](../nvls-base-ccl/README.md) 是同题目的 C++/MPI 版本，测的是 NCCL 与 CUDA
multicast 原语本身；这里测的是训练代码真正调用的 `torch.distributed` 路径。

## 回答的三个问题

NVLS 的 collective 在 NCCL 里有**两套独立实现**，本目录把两套都当一等公民与 ring 对照：

| 实现 | 选择方式 | 数据面 |
| --- | --- | --- |
| 经典 NVLS | `NCCL_ALGO=NVLS`，普通 device buffer | multicast + SHARP，nvls.cc 注册路径 |
| 对称内存（symmetric memory） | 不开 `NCCL_ALGO`，改用 window 注册 | NCCL symmetric kernels（`AllGather_STMC` / `ReduceScatter_LDMC`），本地 LSA/`multimem` 路径，见 `src/sym_kernels.cc`、`src/device/symmetric/` |

两者在内核、同步协议、buffer 要求上都不同，训练侧（FSDP/DP 通信）在 2.28+ 走的是后者，
所以只测 NVLS 不足以代表实际训练路径。

| 问题 | 场景 | case 名 |
| --- | --- | --- |
| 分别发起 AG、RS，和同时发起 AG+RS 的吞吐 | 单组（同一批 rank）依次跑 AG、RS、AG+RS | `single` |
| ring 与 nvls 方案对比 | 同一矩阵用 `NCCL_ALGO=Ring` 和 `NCCL_ALGO=NVLS` 各跑一遍 | 两次启动 |
| 16 卡下同时开启两个 8 group | 16 卡切成两个 8 rank 组，每组单独跑，再两组同时跑 | `split` |

`single` 的三种负载都在同一个 job 里测，`ag+rs` 用两个 NCCL communicator + 两条 CUDA stream
同时发射，因此它能和单独测的 AG、RS 直接比较：

```text
speedup          = (T_AG + T_RS) / T_AG+RS     # 1 = 完全串行，理想重叠 = (T_AG+T_RS)/max(T_AG,T_RS)
overlap_fraction = (T_AG + T_RS - T_AG+RS) / min(T_AG, T_RS)   # 1 = 短的操作被完全隐藏，可为负
```

两次启动的结果由 `collective_metrics.py` 合成 `compare.md` 三张表，不需要手工对表。

## 为什么必须三次启动

`NCCL_ALGO` 在 communicator 创建时读取并固化，进程内无法改；对称内存路径还要在创建
communicator 之后注册 window，所以三种实现是两个不同类别的配置，必须分成三个独立的
torchrun job，用不同的 `--master-port` 串行执行，结果都留在同一个输出目录下。

launcher 为每个实现固定了：

| 变量 | ring | nvls | symm | 作用 |
| --- | --- | --- | --- | --- |
| `NCCL_ALGO` | `Ring` | `NVLS` | 空（=不设） | 前两者强制算法；symm 不设，否则会屏蔽掉 window 分发 |
| `NCCL_NVLS_ENABLE` | `0` | `1` | `1` | ring 不申请 multicast buffer；symm 保留 multicast（MC）内核可选 |
| `NCCL_WIN_ENABLE` | 不设 | 不设 | `1` | 允许 `ncclCommWindowRegister`，这是 symm 路径的开关 |
| `NCCL_SYM_KERNEL` | 不设 | 不设 | 可选 | `--symm-kernel ag\|rs` 固定到单个内核，便于定性 |
| `NCCL_LAUNCH_ORDER_IMPLICIT` | `1` | `1` | `1` | 两个 communicator 并发时保证 kernel 发射顺序一致 |
| `NCCL_MNNVL_ENABLE` | 多机才设 | 多机才设 | 多机才设 | 跨 tray 强制走 NVLink fabric，禁止静默回退到 IB |
| `NCCL_DEBUG_SUBSYS` | `INIT,ENV` | `INIT,ENV` | 同上 | 保留 NVLS 初始化与资源建立记录 |

symm 的实现细节（都按 2.28.9 源码确认）：

- 开关：`comm->symmetricSupport = isAllCudaP2p && NCCL_WIN_ENABLE && cuMemEnable`（`src/init.cc:1376`）。
- 入选条件：collective 的收/发 buffer 都要落在**带 `NCCL_WIN_COLL_SYMMETRIC` 标志**的 window 里
  （`src/scheduler/symmetric_sched.cc:36`）。torch 侧就是 `pg.register_mem_pool(pool, symm=True)`
  → `ncclCommWindowRegister(..., NCCL_WIN_COLL_SYMMETRIC)`（`NCCLUtils.cpp:503`），所以负载
  buffer 必须分配在 `torch.cuda.MemPool()` 里，本程序对 `--algo symm` 就是这样分配的。
- 注册是**集合操作**：`ncclDevrWindowRegisterInGroup` 结尾有 `bootstrapBarrier`
  （`dev_runtime.cc`），所以每个 size 都要先全组注册、用完再 `deregister_mem_pool` 才能释放内存。
- 实现范围：`ncclSymkImplemented` 里 AllGather 恒可；ReduceScatter 只支持 sum + 浮点
  （`sym_kernels.cc`）。bf16/fp32 的 sum 都在范围内。
- 单机限制：`if (comm->nNodes > 1) kmask &= ~kernelMask_LSA`（`sym_kernels.cc`）。2.28.9 的
  对称内核走本地 shared-address（LSA）路径，**多 OS 主机的 communicator 会被整体摘掉**，
  剩下能用的只有 GIN 变体，而 GIN 变体是 2.30.4 才加的
  （`AllGather_RailRing_LsaSTMC`、`ReduceScatter_RailA2A_LsaLDMC`）。
  所以：本目录在 2.28.9 上只有**单托盘 4 卡**是真正的平铺对称内存结果；8/16/32 卡要么换
  2.30.4，要么把结果记为"跨主机退化"，程序会把实测 `nNodes/MNNVL` 写进 JSONL 并在
  `nNodes > 1` 时打警告，不接受未标注的跨主机对称内存结论。

强制的算法不可用时 NCCL 会打 `Error : no algorithm/protocol available for function ...` 并让
collective 失败（见 `src/enqueue.cc:1859`），不会静默换算法。`NCCL_MAX_NVLS_ARITY` 为 32
（`src/include/device.h:207`），所以 AG/RS 的 NVLS 组最多 32 个 rank；再大就只能是 ring 或
NVLS_TREE，本实验到 32 卡为止。

一个例外要自己判断：NCCL 2.28.9 在**跨主机**（`comm->nNodes > 1`）时要求 `collNetSupport == 1`
才允许 `NCCL_ALGO_NVLS`（`src/enqueue.cc:1803`），跨主机 AG/RS 的 NVLS 主要走 `NVLS_TREE`
（arity 3）。若这样，跨 tray 的 `nvls` 就不是纯平铺 NVLS，比较时要在报告里写明。

注意 `comm->nNodes` 按 `ringRecv[0]` 分组（`init.cc:1095`），MNNVL 下同一 NVLink domain 的多台
OS 主机仍是多个 node，`NCCL_MNNVL_ENABLE=1` 只影响 P2P/拓扑，不会把 nNodes 变成 1。

## 运行

单机 4 卡冒烟（先确认 CUDA/NCCL/多卡可用）：

```bash
bash examples/gb200/nvls-collective/run_nvls_collective_benchmark.sh \
  --sizes-mib 64 --trials 2 --warmup 2
```

一个 NVL72 domain 内 8 个 tray（32 卡，每 tray 4 卡）跑完整矩阵，**每台节点执行同一条命令**：

```bash
bash examples/gb200/nvls-collective/run_nvls_collective_benchmark.sh --nnodes 8
```

DLC 会注入 `MASTER_ADDR` 和 `RANK`/`NODE_RANK`，所以不用再传 `--master-addr`、`--node-rank`；
其他环境手动加上这两个参数。

单独把对称内存那一档和 ring 对比（2.28.9 下只在单托盘 4 卡有意义）：

```bash
bash examples/gb200/nvls-collective/run_nvls_collective_benchmark.sh \
  --algos ring,symm --group-sizes 4 --sizes-mib 16,64,256,1024
```

16 卡（4 节点 × 4 卡）时只看问题 3：

```bash
bash examples/gb200/nvls-collective/run_nvls_collective_benchmark.sh \
  --nnodes 4 --group-sizes 8 --split-totals 16 --split-layouts contiguous,interleaved
```

`--group-sizes` 默认取 4/8/16/32 中不超过 `nnodes × gpus-per-node` 的部分，`--split-totals`
默认在 ≥16 卡时取 16，否则不跑 split。`--dry-run` 只打印命令和将要使用的输出目录，不创建目录，
也不要求 GPU 或 torch。

### 输出目录（不需要指定）

固定为仓库内 `results/gb200/nvls-collective/gpu-num-<总卡数>/run-<NN>`，`NN` 是该卡数下已有
`run-*` 的最大编号加一（`run-01`、`run-02`…，不回填空号）。由 [`run_dir.py`](run_dir.py) 决定：

1. node 0 用 `mkdir` 原子地占下下一个 `run-NN`（同时启动的两个 job 不会拿到同一个），并写一个
   `.node0` 标记文件。
2. 多节点时 node 0 在 `MASTER_ADDR:<master-port − 1>`（默认 29610）起一个 `TCPStore`，把路径发给
   其他节点，并等所有节点确认收到后才继续。不能让每个节点各自扫描目录：晚启动的节点会看到 node 0
   刚建好的 `run-01`，于是自己选 `run-02`。
3. 其他节点检查能否看到 `.node0`，看不到说明仓库不在共享存储上，直接报错退出。

### 网络接口（GLOO_SOCKET_IFNAME）

launcher 读取 `ip -o -4 addr show up`，只认处于 up 且有 IPv4 地址的接口：

| 场景 | 判定 | 优先 | 优先接口不可用时 |
| --- | --- | --- | --- |
| DSW | 总卡数 ≤ 4 | `eth0` | 退到 `bond1` 并打警告 |
| DLC | 总卡数 > 4 | `bond1` | 退到 `eth0` 并打警告 |

两个都没有就报错退出。选中的接口同时设给 `GLOO_SOCKET_IFNAME` 和 `NCCL_SOCKET_IFNAME`
（NCCL 的 bootstrap 也走 TCP）。如果启动前已经 export 了 `GLOO_SOCKET_IFNAME` 或
`NCCL_SOCKET_IFNAME`，以导出的值为准。
launcher 固定打开 `NCCL_DEBUG_SUBSYS=INIT,ENV,TUNING`：`Algo ... proto ...` 和
`[Symmetric] ... Kernel ...` 都是 `INFO(NCCL_TUNING, ...)`，只在 comm rank 0 上打印；程序从每个
进程的 `NCCL_DEBUG_FILE` 按偏移增量读取，把 init 行归到对应的 communicator、把内核行归到对应
case，结果写进 JSONL。这是逐 case 证明实际内核的唯一依据，不要只看 `NVLS supported`。

### 为每个测试生成 nsys trace

```bash
bash examples/gb200/nvls-collective/run_nvls_collective_benchmark.sh \
  --group-sizes 4 --sizes-mib 64,256 --trials 3 --nsys
```

`--nsys` 用 `torchrun --no-python` 给**每个 worker** 套一个 `nsys profile`：

```text
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=repeat \
  --output=<algo>/nsys/rank%q{RANK} python nvls_collective_benchmark.py ...
```

- 每个 case（实现 × 组 × 负载 × 大小 × isolated/concurrent）开一个
  `cudaProfilerStart/Stop` 窗口，只包住计时 trial，warmup、buffer 分配、window 注册不在内。
  `repeat` 让 nsys 每个窗口写一份独立报告，所以**每个测试 × 每个 rank 一份 trace**。
- **所有 rank 对每个 case 都开窗口**，包括该 case 闲置的 rank（如 split isolated 时另一组），
  这样各 rank 的第 k 份报告都对应同一个 case。
- 报告里最外层 NVTX range 是 case 标签（如 `symm/single4/ag+rs/256MiB`），内层是 `trial0..N`；
  `<algo>/nsys/index.tsv` 按采集顺序列出 `capture_index、label、scenario、phase、start_unix、end_unix`。
  nsys 不同版本给 repeat 报告编号的文件名格式不一定相同，**以 NVTX 标签和时间窗为准**。
- `--nsys` 跑出的数字带 `profiled=true`，`compare.md` 顶部会标注；trace 只用来看内核名、两条
  stream 是否真的重叠、两个 8 组是否同时在跑，**结论表必须用不带 `--nsys` 的那次**。

多机可以改成 Slurm 的 `srun`（`#SBATCH --ntasks-per-node=4`、`--gpus-per-node=4`），
把每条 `uv run ... torch.distributed.run` 换成 `srun` 即可；`torch.distributed.run` 在
Slurm 分配内也能用，`--master-addr` 取第一个节点。

### 用 SQL 查 AG / RS 的 kernel 开销

`--nsys` 跑完后 node 0 自动执行 [`analyse_nsys.py`](analyse_nsys.py)：等其他节点的报告落盘，对每份
报告跑 `nsys export --type sqlite`，再执行 [`nsys_collective_kernels.sql`](nsys_collective_kernels.sql)
里的两条查询。也可以手动跑：

```bash
uv run --no-sync python examples/gb200/nvls-collective/analyse_nsys.py \
  --output-dir results/gb200/nvls-collective/gpu-num-4/run-01
# 或者对单份报告直接用 sqlite3：
nsys export --type=sqlite --output=rank0.1.sqlite rank0.1.nsys-rep
sqlite3 -header -csv rank0.1.sqlite < examples/gb200/nvls-collective/nsys_collective_kernels.sql
```

查询的归属规则：

| 步骤 | 表 | 规则 |
| --- | --- | --- |
| 找 case 和 trial | `NVTX_EVENTS` + `StringIds` | 外层 range 名形如 `symm/single4/ag+rs/256MiB`，内层为 `trial0..N`；`text` 与 `textId` 两种导出格式都支持 |
| kernel → 发射它的 API | `CUPTI_ACTIVITY_KIND_KERNEL` ⋈ `CUPTI_ACTIVITY_KIND_RUNTIME` | `correlationId` 相同且同一进程（`globalTid >> 24`） |
| kernel → trial | 上面两者 | **发射时刻**落在 trial 的 NVTX range 内（与 `mbench/build_csv.py` 相同），不看 GPU 何时执行 |
| 只保留集合通信 | `StringIds.value` | 名字以 `nccl` 开头且含 `AllGather` / `ReduceScatter`；`_sleep` 延迟 kernel、warmup 都被排除 |
| AG∩RS 重叠 | 同 trial 的 AG × RS kernel 两两求交 | 同一 collective 在自己的 stream 上串行，所以两两交集之和就是精确的重叠时长 |

输出：`<algo>/nsys/kernels.csv`（每份报告 × trial × op × kernel 一行），以及 `nsys_kernels.md`：每个 case
和组给出 AG / RS 每次迭代的 kernel 时间、AG∩RS 重叠、`overlap_fraction = 重叠 / 较短一方`、kernel 跨度，
以及同一次运行的 CUDA-event makespan 和二者之比。规则与计时表相同：trial 内取组内最慢 rank，再对
trial 取中位数。

读这张表要注意两点：

- **经典 NCCL kernel 名不能说明算法**。2.28.9 的 `best_kernel()`（`src/device/generate.py`）对每个
  collective 只特化一个 kernel，AllGather 一律是 `ncclDevKernel_AllGather_RING_LL`，内部再经函数表
  分发到 NVLS/Ring。所以 nvls 运行里看到 `*_RING_LL` 是正常的，算法以 `compare.md` 2b 的 TUNING 日志为准。
  对称内存 kernel 的名字（`ncclSymkDevKernel_AllGather_STMC` 等）则直接带变体。
- kernel 时长包含等待对端到达的时间，不是纯传输时间；数据来自 profile 运行，只和同一次的 CUDA-event 对照。

## 计时与正确性

每个 case 的输入按**小整数**构造，保证 bf16 下任何归约顺序都精确：

- AG：rank `r` 的 chunk 值固定为 `r + 1`，输出每个 rank 都逐元素校验 chunk 顺序。
- RS：组内偶数 rank 输入 1、奇数 rank 输入 2，任意顺序的部分和都是小整数（≤ 组大小 + 组大小/2），
  输出每个 rank 校验一次。

计时不依赖 CPU 同步：先在当前 stream 上插入 `torch.cuda._sleep`（约 `--launch-delay-ms`，默认
5 ms）当起点，让所有 rank 的 NCCL kernel 都已入队，再 `start.record()`；`--iters` 次发射背靠背
执行，结束事件记录在各自 stream 上，当前 stream 用 `wait_event` 汇合后计时。默认模式
（`TORCH_NCCL_USE_COMM_NONBLOCKING=0`）下 NCCL kernel 直接跑在当前 stream 上，所以这段就是
NCCL 真实占用 GPU 的时间，不掺 Python 入队开销。**多机是异步的**：慢 rank 会占住自己的
communicator 而别的 rank 继续入队，所以这里测的是 GPU 侧 kernel 时间，不是端到端墙钟；
`enqueue_ms` 会记录 Python 入队时间，超过 `--launch-delay-ms` 的 case 标 `[HOST-BOUND]`，
此时延迟不再被完全吸收，需要调大 `--launch-delay-ms` 或调小 `--iters`。

每个 trial 前有 Gloo barrier 对齐各组起点，取 `--trials` 次的**中位数**；每个 trial 内部取组内
**最慢 rank** 的 event 时间，不用平均掩盖慢 rank。`busbw = size × (n−1)/n / t`，十进制 GB/s，
`size` 指 AG 的完整输出 / RS 的完整输入（每 rank 的 shard 为 `size/n`）。带宽都不乘 GPU 数，
所以 `single` 里可以直接比 AG、RS 和 AG+RS。`aggregate busbw` 是**单个组**内所有操作的 busbw
之和、用该组自己的 makespan 计算（AG+RS 时 = 2 × size × (n−1)/n / T_AG+RS）；`split` 每组各出一行，
两组的总量需要把 A、B 两行相加。这些都是从 payload 推出来的值，不是 NVSwitch wire bytes。

## 对比表格（`compare.md`）

`collective_metrics.py` 把三次启动的 `results.jsonl` 合并成以下几张表，每格都是 trial 中位数：

| 表 | 回答 | 每行 | 关键列 |
| --- | --- | --- | --- |
| 1 | AG / RS / AG+RS | 实现 × 组大小 × 消息 | AG busbw、RS busbw、AG+RS 中各自 busbw、pair aggregate、`speedup`、`overlap_fraction` |
| 1b | 三个实现的绝对吞吐 | 组大小 × 消息 × 负载 | ring、nvls、symm 的 aggregate busbw，及最高者 |
| 2 | 相对 ring | 组大小 × 消息 × 负载 | nvls、nvls/ring、symm、symm/ring |
| 2b | 实际执行的内核 | 实现 × 组 | 实测 nNodes、日志里的 `Algo/proto` 或对称内核名 |
| 3 | 两个 8 组同跑 | 实现 × 布局 × 消息 × 负载 × 组 | isolated busbw、concurrent busbw、`slowdown = T_concurrent / T_isolated` |

没有记录的格子显示 `-`，不插值。2b 里 `nNodes > 1` 的 symm 行不能当平铺对称内存结论。

## 输出与验收

```text
results/gb200/nvls-collective/gpu-num-<W>/run-<NN>/
  .node0                                       # node 0 建目录时写入，用于检查共享存储
  ring/
    results.jsonl  metadata.json               # 只由全局 rank 0 写
    torchrun-node<k>.log                       # 每个节点一份
    nccl-<host>-<pid>.log                      # 每个进程一份
    nsys/{rank<R>*.nsys-rep, index.tsv, kernels.csv}   # 仅 --nsys
  nvls/ ...
  symm/ ...                                    # window 注册的对称内存路径
  compare.md                                   # node 0 合并三个 results.jsonl
  nsys_kernels.md                              # 仅 --nsys，node 0 用 SQL 汇总
```

### 多节点数据怎么汇到同一个文件

所有节点写同一个 run 目录（上面的 TCPStore 握手保证路径一致），但**结果文件只有一个写入者**：

1. torchrun 用默认的 static rendezvous，全局 rank = `node_rank × gpus_per_node + local_rank`
   （`torch/distributed/run.py` 把 `rdzv_configs["rank"]` 设为 `node_rank`，
   `elastic/agent/server/api.py` 里 `base_global_rank = group_rank * local_world_size`），
   所以全局 rank 0 一定在 node 0 上。
2. 每个 case 结束时，所有 rank 通过 Gloo 默认组执行 `dist.all_gather_object`，把各自的 trial 计时、
   校验结果和从本进程 NCCL 日志读到的内核证据交给 rank 0。TUNING 行只在每个 communicator 的
   rank 0 上打印，比如 split 的 B 组在全局 rank 8 上，也是靠这一步汇到 rank 0。
3. rank 0 计算统计量，**追加**到 `<algo>/results.jsonl`；`metadata.json`、`nsys/index.tsv` 也只由
   rank 0 写。其他 rank 不写这些文件，所以不存在多节点并发写同一文件的问题。
4. 每个进程只写自己名字的文件：`nccl-<host>-<pid>.log`、`nsys/rank<R>*.nsys-rep`
   （`R` 是 torchrun 设置的全局 `RANK`），以及每个节点的 `torchrun-node<k>.log`。
5. 三个实现是三次独立的 torchrun，结果在 `ring/`、`nvls/`、`symm/` 三个 `results.jsonl` 里。
   node 0 的 torchrun 只有在 rank 0 退出后才返回，而 rank 0 退出前已经写完 `results.jsonl`，
   所以随后执行的 `collective_metrics.py` 读到的是完整数据，合并成一份 `compare.md`；合并时会
   检查每个文件里的 `algo` 与目录一致。
6. `--nsys` 时，各节点的 nsys 报告可能比 node 0 晚落盘，`analyse_nsys.py` 会等到报告数达到
   `world_size × case 数` 且文件大小不再变化（最多 300 秒）后，再逐份导出 SQLite 汇总。

`results.jsonl` 每行一个 case：`scenario`、`phase`、`total_ranks`、`layout`、`group_index`、
`workload`、`group_size`、`ranks`、`size_bytes`、`makespan_ms`、`ag_ms`/`rs_ms`、`host_bound`、
`check_passed`、`trial_makespan_ms` 以及各类带宽。另外每个 case 带 `nccl_algo`（经典路径的
`Algo/proto`）、`nccl_kernel`（对称路径的内核名）和实测的 `n_nodes`/`mnnvl`——这三项是从
launcher 的 `NCCL_DEBUG_FILE` 里读出来的实测值，用来证明某个 case 真的走了哪条实现；
三项都为空的 case 在报告里只显示 `-`，不得当作对应实现的证据。`compare.md` 的 2b 区会把
每个实现的实测内核列表列出来。`metadata.json` 记录世界大小、每 rank 的
GPU UUID/PCI bus ID/主机名、torch/CUDA/NCCL 版本和全部 `NCCL_*` 环境，便于解释 rank 到
NVLink 拓扑的映射。已有输出目录会报 `FileExistsError`，重复实验用新目录。

接受结论前检查：

1. `check_passed` 全为 true，且 metadata 里的 NCCL 版本、GPU 映射与计划一致。
2. 日志里有 NVLS/MNNVL 初始化记录；经典路径需要 `Algo=NVLS ... Proto=Simple`，对称内存路径需要
   `[Symmetric]: ... -> Kernel AllGather_STMC/ReduceScatter_LDMC`。两种证据不能互相替代：
   只看 `NVLS supported` 或只看 window 注册成功都不能证明某个 collective 用了哪条实现。
3. `split` 的 A、B 两组都用 8 rank 组自己的 `T_concurrent / T_isolated` 比较；两组没有共同 GPU
   仍可能共享交换机链路和控制器，减速本身不能定位到 SHARP ALU。
4. 对照不同 `--trials`、`--iters` 和消息大小重复一次，确认带宽平台稳定；`[HOST-BOUND]` 的 case
   不参与结论。

## 本地检查（无 GPU）

```bash
uv run --no-sync python -m pytest \
  --confcutdir=tests/unit_tests/examples \
  tests/unit_tests/examples/test_gb200_nvls_collective.py \
  tests/unit_tests/examples/test_gb200_nvls_analyse_nsys.py \
  tests/unit_tests/examples/test_gb200_nvls_run_dir.py -q
bash -n examples/gb200/nvls-collective/run_nvls_collective_benchmark.sh
uv run --no-sync ruff check examples/gb200/nvls-collective
```

单测覆盖算法族/布局/负载的解析与默认值、ring/nvls/symm 三套环境约束（含 symm 必须不设
`NCCL_ALGO`、必须开 `NCCL_WIN_ENABLE`、`NCCL_SYM_KERNEL` 必须与内核名一致）、NCCL 日志探针
（algo/kernel/nNodes/MNNVL）、两个 8 组划分、带宽与重叠公式、报告四张表以及两个入口只用
标准库可导入（torch 除外）。
它们不需要 GPU、NCCL 或 NVLink，也**不能**验证 CUDA/NCCL 真实行为、带宽或 NVLS 资源。
本目录的程序必须在上面说的 GB200 环境里编译/运行才能真正出数。
