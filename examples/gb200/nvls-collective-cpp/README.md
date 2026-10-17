# GB200：纯 C++ 测 NCCL 2.32.3 的 AG / RS（ring / NVLS / 对称内核 / TMA / CE）

日期：2026-10-17。这是 [`../nvls-collective`](../nvls-collective/README.md) 的 C++ 版本：场景、计时口径、
实现校验和输出目录结构都与它对齐，但不经过 torch，直接调用 NCCL 2.32.3（源码
`megatron-bridge/nccl/nccl-2.32.3`），并新增两组对照：

- **TMA + `multimem.st` AllGather**（`AllGather_TmaSTMC`）+ 对称内核里基于 `multimem.ld_reduce` 的
  ReduceScatter（`ReduceScatter_LDMC`）。
- **CE（copy engine）`multimem` AllGather** + 同一个 `ReduceScatter_LDMC`。

以下行为都按 2.32.3 源码确认。

## 回答的问题

| 问题 | 场景 | case 名 |
| --- | --- | --- |
| 分别发起 AG、RS、AR，和同时发起 AG+RS 的吞吐 | 单组（同一批 rank）依次跑 AG、RS、AR、AG+RS | `single` |
| 各实现对比（ring / NVLS / symmetric / TMA / CE） | 同一矩阵每个 `--impl` 各跑一遍 | 每个实现一次启动 |
| 16 卡下同时开启两个 8 group | 16 卡切成两个 8 rank 组，两组同时跑同一负载 | `split` 的 `AB-concurrent` |
| 两个独立组同时：A 做 AG、B 做 RS | 同一 split，A 组 all-gather 的同时 B 组 reduce-scatter | `split` 的 `AB-cross` |

`single` 的 `ag+rs` 用两个 communicator + 两条 CUDA stream 同时发射，与单独测的 AG、RS 直接比较：

```text
speedup          = (T_AG + T_RS) / T_AG+RS     # 1 = 完全串行，理想重叠 = (T_AG+T_RS)/max(T_AG,T_RS)
overlap_fraction = (T_AG + T_RS - T_AG+RS) / min(T_AG, T_RS)   # 1 = 短的操作被完全隐藏，可为负
```

## 实现与选择方式

2.32 新增了逐次调用的 `ncclCollConfig_t`（`ncclAllGatherConfig` / `ncclReduceScatterConfig`）。程序每次
调用都带一个 config：`algSelection` 固定内核，`CTAPolicy = NCCL_CTA_POLICY_ZERO` 把 AllGather 送到
copy engine，`forceAlgSelection = 1`（默认）让不可用的选择直接报错，而不是悄悄回退。所以**不再需要**
`NCCL_ALGO` / `NCCL_SYM_KERNEL` 这些进程级开关，launcher 反而会把它们 unset。

| `--impl` | buffer | AllGather | ReduceScatter | AllReduce | 测量哪些 workload |
| --- | --- | --- | --- | --- | --- |
| `ring` | `cudaMalloc` | `RING` | `RING` | `RING` | ag, rs, ag+rs, ar |
| `nvls` | `cudaMalloc` | `NVLS_SIMPLE` | `NVLS_SIMPLE` | `NVLS_SIMPLE` | ag, rs, ag+rs, ar |
| `symmetric` | `ncclMemAlloc` + window | `SYMK_STMC` (寄存器 `multimem.st`) | `SYMK_LDMC` (`multimem.ld_reduce`) | `SYMK_RSxLDMC_AGxSTMC` | ag, rs, ag+rs, ar |
| `symmetric-tma` | 同上 | `SYMK_TmaSTMC` (TMA multicast) | `SYMK_LDMC` | — | ag, ag+rs |
| `copy-engine` | 同上 | `CTAPolicy=ZERO` (CE 写 multicast) | `SYMK_LDMC` | — | ag, ag+rs |

`symmetric-tma` 和 `copy-engine` 只测 AllGather 和 AG+RS，因为它们的 ReduceScatter 跟 `symmetric` 完全一样
（都是 `SYMK_LDMC`），单独测 RS 没有意义。AllReduce 同样在 `symmetric` 里已经覆盖。

- **TmaSTMC**（`src/device/symmetric/primitives.cuh` `bcastMultimem<.., EnableTma=true>`）：每个 warp 的
  lane 0 先 `cp.async.bulk` 把输入块搬进 shared memory，再 `cp.async.bulk` 从 smem 写到 multicast 地址，
  SM 不再逐寄存器发 `multimem.st`。
- **CE**（`src/ce_coll.cc` `ncclCeAllGather`）：`ncclCeAllGatherUseMulticast` 为真时一条
  `cudaMemcpyBatchAsync` 写本 rank 分片的 multicast 地址，交换机负责扇出；为假时 N−1 份 unicast 拷贝。
  2.32 的选择由 `NCCL_CE_COLL_AG_MULTICAST_THRESHOLD`（≥0 时绕过 cost model，分片 ≤ 阈值走 MC）决定，
  launcher 对 `ce` 设 2^62、对 `ce-uc` 设 0。同步旗标在有 LSA multimem 时也走 multicast，日志里显示为
  `[Copy Engine] ... -> cudaMemcpy; CE synchronization with NVLS`。
- NCCL 没有 CE ReduceScatter（`ncclCeImplemented` 只有 AllGather / AlltoAll / Scatter / Gather），所以
  CE 组的 RS 只能是对称内核。

**symm 下的 copy engine 分支**：2.32 仍保留 Blackwell 上 `count > NCCL_SYM_CE_THRESHOLD`（元素，默认 8Mi）
的 AllGather 自动走 CE 的分支（`src/enqueue/enqueue.cc`），但只在**没有** `algSelection` 时生效。`stmc` /
`tma` 有选择，不受影响；`symm` 没有选择，所以和 `../nvls-collective` 一样设
`NCCL_SYM_CE_THRESHOLD=9223372036854775807`，程序也要求它不小于最大 payload，否则拒绝启动。

## 什么时候可用（每组运行时检查）

对称内核和 CE 都要求 communicator 是**一个 LSA team**（所有 rank 互相 load/store 可达），STMC / TMA /
LDMC / CE-MC 还要求 **multimem**。程序在每个组建好后用 `ncclCommQueryProperties` 读 `nLsaTeams`、
`multimemSupport`、`deviceApiSupport`，任一不满足就带原因退出，不出半截结果：

- `multimemSupport = nvlsSupport && !p2pCrossClique`（`dev_runtime.cc`）。跨 NVLink clique 的 P2P
  会关掉 multimem。
- `nLsaTeams = nRanks / lsaSize`；`lsaSize` 在单 NVL domain 且开了 cross-clique 时扩到全组，否则按
  节点分组取 gcd（`computeLsaSize`）。多托盘组是不是一个 LSA team 要看实测，结果记在 `n_lsa_teams`。
- TMA 另需 `minCompCap >= 100`、`maxSharedMemOptin` 足够、`NCCL_SYM_TMA_ENABLE != 0`
  （`ncclSymkTmaAvailable`），以及 16 B 对齐的 buffer（`ncclMemAlloc` 满足）。
- CE 需要驱动 ≥ 12.5（`ncclCeImplemented`）。

单托盘 4 卡一定能跑全部实现；8/16/32 卡上 `stmc`/`tma`/`ce` 能不能跑由实测 `nLsaTeams`/`multimem`
决定，不能就会明确报错，`ring`/`nvls` 不受影响。

## 安装与编译

```bash
# 1. NCCL 2.32.3：拉源码到 megatron-bridge/nccl/nccl-2.32.3（已存在且版本对则复用），按 sm_100 编到 build/，
#    并检查 TMA / STMC / LDMC / CE / per-call config 都编进去了、没有被环境或 nccl.conf 关掉。
bash install/install-nccl-2.32.3.sh
# 2. torch 自带的 NCCL 是什么版本（只读，不建 communicator）
bash install/probe-torch-nccl-version.sh
```

本目录的 benchmark 通过 rpath + `LD_LIBRARY_PATH` 用 `nccl/nccl-2.32.3/build/lib/libnccl.so`，不碰 torch
加载的那份；启动时还会比对 `ncclGetVersion()` 和编译时的 `NCCL_VERSION_CODE`，不一致直接退出。
launcher 默认先在 node 0 `make`，也可以手动：

```bash
make -C examples/gb200/nvls-collective-cpp NCCL_HOME=nccl/nccl-2.32.3/build CUDA_HOME=/usr/local/cuda
```

## 运行

单托盘 4 卡，测 TMA 和 copy-engine：

```bash
bash examples/gb200/nvls-collective-cpp/run_nvls_collective_benchmark.sh \
  --impls symmetric-tma,copy-engine --sizes-mib 64,256,1024 --trials 3
```

全部实现；16 卡起默认带 split（16 卡上 8+8，32 卡上 16+16，含 cross）：

```bash
bash examples/gb200/nvls-collective-cpp/run_nvls_collective_benchmark.sh             # 1 托盘
bash examples/gb200/nvls-collective-cpp/run_nvls_collective_benchmark.sh --nnodes 4  # 每个节点执行同一条
```

每个节点为每张 GPU 起一个 `nvls_collective_bench` 进程（`RANK = node_rank × gpus_per_node + local_rank`），
NCCL 自己通过 `NCCL_COMM_ID=<master-addr>:<master-port + i>` 建 bootstrap，不需要 torchrun 或 MPI，
所以全局 rank 0 一定在 node 0 上。每个实现单独一轮进程：`NCCL_NVLS_ENABLE` 和 CE 阈值是进程级参数，
只读一次。某个 rank 失败时 launcher 杀掉同轮其余 rank 并报错退出。

`MASTER_ADDR`/`RANK` 注入、run id 握手、`eth0`/`bond1` 接口选择都和 `../nvls-collective` 相同；
`--sizes-mib` 默认 16,32,64,128,256,512,1024,2048（完整张量，AG 输出 / RS 输入）；`--group-sizes` 默认取 4/8/16/32 中
不超过总卡数的部分；`--split-totals` 默认取 16/32 中放得下的最大值（16 卡 → 8+8，32 卡 → 16+16），布局默认
只有 `contiguous`。split 的半组大小 `T/2` 必须在 `--group-sizes` 里（它提供 single 参照），默认值满足这个条件；
自定义时程序会检查。

### nsys

`--nsys` 给每个进程套 `nsys profile --capture-range=cudaProfilerApi --capture-range-end=repeat
--output=<impl>/nsys/rank%q{RANK}`。每个 case 一个 `cudaProfilerStart/Stop` 窗口，只包住计时 trial；
**所有 rank 对每个 case 都开窗口**，最外层 NVTX range 是 `<impl>/<case>`，内层 `trial0..N`；rank 0 写
`<impl>/nsys/index.tsv`。跑完后 node 0 执行 [`../nvls-collective/analyse_nsys.py`](../nvls-collective/analyse_nsys.py)
（只用 Python 标准库，与测量无关）：按 NVTX 标签把报告和 SQLite 导出挪到
`<impl>/cases/<case>/<impl>_<case>_rank<RR>.{nsys-rep,sqlite}`，再跑
[`nsys_collective_kernels.sql`](../nvls-collective/nsys_collective_kernels.sql) 写 `nsys_kernels.md`。
CE AllGather 是 memcpy，不是 kernel，所以 `ce`/`ce-uc` 在 SQL 表里只有 RS 的 kernel 时间。
`--nsys` 的数字带 `profiled=true`，只用来看 trace，结论表用不带 `--nsys` 的那次。

## 计时与正确性

输入按**小整数**构造（AG 分片 k = k+1，RS 偶数 rank 1、奇数 rank 2），bf16 下任何归约顺序都精确，
每个 case 都逐元素校验输出。

每个 trial 在主 stream 上依次入队：

1. `spinKernel`（约 `--launch-delay-ms`，默认 5 ms），让后面所有 launch 在 GPU 开跑前都已入队；
2. **同步起点（start gate）**：在覆盖整个 scenario 所有 rank 的独立 communicator 上做一次 1 个 fp32 元素的
   all-reduce。没有共同 communicator 的 A、B 两组会在几微秒内同时离开它；
3. 记 start 事件，然后 `--iters` 次背靠背发射，结束事件记在各自 stream 上，主 stream 汇合后计时。

world communicator 上的 barrier 只负责让各 rank 进入同一个 trial（8 字节 all-reduce，不计时），真正的起点
是 GPU 上的 gate。因为所有组从同一时刻计时：

```text
concurrency = min(T_A, T_B) / max(T_A, T_B)   # 每个 trial 一个值，报告取中位数；只在 AB-concurrent / AB-cross 有值
```

| 量 | 定义 |
| --- | --- |
| 单 rank 每次迭代时间 | `t_rank = (该 rank 所有 op 的结束事件 − start) / iters`；`ag_ms`/`rs_ms` 是各 op 自己 stream 的结束事件 |
| 每个 trial 的时间 `t` | 组内最慢 rank 的 `t_rank` |
| 完成时间 | 表 0 给出各 trial `t` 的 mean / median / min / max（`mean_makespan_ms`、`makespan_ms`、`trial_makespan_ms`） |
| `size` | 完整张量字节数：AG 每卡的接收（输出）总量，RS 每卡的发送（输入）总量，AR 每卡原地归约的总量；AG/RS 每卡 shard = `size/n`，AR 不分片 |
| algbw | `size / t`，十进制 GB/s |
| busbw | AG/RS：`size × (n−1)/n / t`（nccl-tests 定义）；AR：`2 × size × (n−1)/n / t`（nccl-tests 的 all-reduce 定义），因此可以和 AG/RS 的 busbw 直接比 |
| aggregate busbw | 单个组内所有 op 的 busbw 之和，用该组 makespan 算 |

`enqueue_ms` 记录主机入队时间，超过 `--launch-delay-ms` 的 case 标 `[HOST-BOUND]`，不参与结论。

### 实现校验

每个组的第一个 rank（= 两个 communicator 的 comm rank 0，TUNING 行只在它上面打）拿本 case 期间自己进程
日志新增的行检查：

1. 每个 op 的行数恰好 `warmup + trials × iters`，每行字节数等于 `size / n`；
2. 实现符合本轮：`tma` 只接受 `Kernel AllGather_TmaSTMC`，`ce` 只接受 `[Copy Engine] ... CE synchronization
   with NVLS`，`symm` 只接受对称内核，等等（`bench_core.h` `impls()`）；
3. gate 的 comm rank 0（scenario 的第一个 rank）恰好 `trials` 行 `AllReduce: 4 Bytes`；world barrier 是
   8 字节，被忽略；不能出现本负载以外的 collective。

任一 rank 结果错或任一组不符，所有 rank 一起报错退出，`case.json` 写明违规项。

## 对比表格（`compare.md`）

`collective_report` 把每个实现的 `results.jsonl` 合并成以下几张表，格式与 `../nvls-collective/collective_metrics.py`
相同（另加 2a），除表 0 外每格都是 trial 中位数：

| 表 | 回答 | 每行 | 关键列 |
| --- | --- | --- | --- |
| 0 | 每个测试的完成时间 | 实现 × case × 组 | mean / median / min / max ms、由 mean 算的 algbw / busbw、concurrency、**kernel** |
| 1 | AG / RS / AR / AG+RS | 实现 × 组大小 × 消息 | AG busbw、RS busbw、AR busbw、AG+RS 中各自 busbw、pair aggregate、`speedup`、`overlap_fraction`、**AG / RS / AR kernel** |
| 1b | 各实现的绝对吞吐 | 组大小 × 消息 × 负载 | 每个实现的 aggregate busbw 与最高者 |
| 2 | 相对 ring | 组大小 × 消息 × 负载 | 每个实现的 aggregate busbw 与 /ring |
| 2a | **TMA / CE 相对 symmetric** | 组大小 × 消息 × 负载 | symmetric-tma、copy-engine 的 aggregate busbw 与 /symmetric |
| 2b | 实际执行的内核 | 实现 × 组 | 实测 nNodes、nLsaTeams、multimem、逐 launch 校验过的实现 |
| 3 | 两个 T/2 组同跑同一负载 | 实现 × 布局 × 消息 × 负载 × 组 | isolated busbw（取自 `single-n<T/2>`）、concurrent busbw、`slowdown = T_concurrent / T_isolated`、concurrency |
| 4 | A 组 AG、B 组 RS 同时跑 | 实现 × 布局 × 消息 | A 的 AG 与 B 的 RS 各自的 isolated / cross busbw、cross ms、slowdown，`A+B cross busbw`、concurrency |

表 0、表 1 的 kernel 列来自 `nccl_impl`（benchmark 逐 launch 核对过的 TUNING 日志），按 NCCL 2.32.3 的代码生成器
换算成实际运行的函数名（`bench_core.h` 的 `kernelName`）：对称路径是 `ncclSymkDevKernel_<Coll>_<algo>[_sum_<ty>]`，
ring/nvls 是经典路径分发到的设备函数 `ncclDevFunc_<Coll>[_Sum_<ty>]_<ALGO>_<PROTO>`，copy-engine 的 AllGather
是 cudaMemcpy，没有 kernel。表 1 里 AG+RS 同时发起时换了实现会在括号里注明 `pair：...`。

表 3、表 4 的 isolated 参照都是 `single-n<T/2>` 里同大小、同负载的 case（表 4：A 对 `ag`、B 对 `rs`），
所以 `T/2` 必须出现在 `--group-sizes` 中；split 场景本身只跑 AB-concurrent 和 AB-cross，不单独跑 A、B。
表 4 的 cross case 只在 `--workloads` 同时包含 `ag` 和 `rs` 时运行（默认包含）。

## 输出

```text
results/gb200/nvls-collective-cpp/gpu-num-<W>/<yyyymmdd-hhmmss>/
  <impl>/                                      # ring nvls symm stmc tma ce ce-uc
    results.jsonl  metadata.json               # 只由全局 rank 0 写，schema 与 ../nvls-collective 相同
    bench-node<k>.log                          # 每个节点一份（该节点所有 rank 的 stdout/stderr）
    nccl-raw/nccl-<host>-<pid>.log             # 每个进程一份，整个进程的原始 NCCL 日志
    comm/<impl>_<scenario>_rank<RR>.nccl.log   # 每个 scenario 的 communicator 建立/连接/销毁日志
    cases/<case>/                              # 每个测试一个目录
      case.json                                # rank 0：该 case 的结果记录，或实现校验失败的原因
      <impl>_<case>_rank<RR>.nccl.log          # 每个参与 rank：本 case 期间的 NCCL 日志切片
      <impl>_<case>_rank<RR>.nsys-rep / .sqlite  # 仅 --nsys，analyse_nsys.py 归档
    nsys/{index.tsv, kernels.csv}              # 仅 --nsys
  compare.md                                   # node 0 合并所有 results.jsonl
  nsys_kernels.md                              # 仅 --nsys
```

命名与 `../nvls-collective/collective_metrics.py` 相同（`bench_core.h` 的 `scenarioName` / `caseName` /
`rankFile`）：

| 项 | 格式 | 例子 |
| --- | --- | --- |
| scenario | `single-n<n>` / `split-n<T>-<layout>` | `single-n16`、`split-n16-contiguous` |
| case | `<scenario>[-<part>]_<workload>_<dtype>_<size>MiB`，part 为 `AB-concurrent`/`AB-cross`；cross 的 workload 固定为 `agA-rsB` | `single-n4_ag_bf16_1024MiB`、`split-n16-contiguous-AB-cross_agA-rsB_bf16_1024MiB` |
| 每 rank 文件 | `<impl>_<case>_rank<RR><suffix>`，rank 补零到至少两位 | `tma_single-n4_ag_bf16_1024MiB_rank03.nccl.log` |

`results.jsonl` 每行一个 (case, 组)：`algo`（= impl）、`scenario`、`phase`（single 为 isolated，split 为 concurrent / cross）、
`total_ranks`、`layout`、`group_index`、`case`/`case_dir`、`workload`、`group_size`、`ranks`、`size_bytes`、
`makespan_ms`、`mean_makespan_ms`、`trial_makespan_ms`、`ag_ms`/`rs_ms`、各类带宽、`concurrency`、
`host_bound`，以及实测证据 `nccl_impl`、`n_nodes`、`mnnvl`、`n_lsa_teams`、`multimem_support`。
`metadata.json` 记录世界大小、NCCL/CUDA 版本、`NCCL_*` 环境和每 rank 的主机/GPU/PCI/UUID。

## 本地检查（无 GPU）

```bash
make -C examples/gb200/nvls-collective-cpp selftest   # TUNING 日志解析、选择标签、命名、JSON、带宽
bash -n examples/gb200/nvls-collective-cpp/run_nvls_collective_benchmark.sh
bash examples/gb200/nvls-collective-cpp/run_nvls_collective_benchmark.sh --dry-run --nnodes 4 --master-addr 10.0.0.1
```

selftest 用的是 2.32.3 源码里的日志格式串，验证的是解析逻辑，**不能**代替 GB200 上的真实运行。
