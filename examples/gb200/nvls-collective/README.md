# GB200：基于 torch 的 AG / RS / AR 吞吐对比（ring vs NVLS）

日期：2026-10-13。这个目录用 **torch** 在 4/8/16/32 卡上测 all-gather（AG）、reduce-scatter（RS）、
all-reduce（AR）以及 AG、RS 同时发起的吞吐，并对照 ring 与 NVLS 两种算法。已有的
[`nvls-base-ccl`](../nvls-base-ccl/README.md) 是同题目的 C++/MPI 版本，测的是 NCCL 与 CUDA
multicast 原语本身；这里测的是训练代码真正调用的 `torch.distributed` 路径。

## 回答的三个问题

NVLS 的 collective 在 NCCL 里有**两套独立实现**，本目录把两套都当一等公民与 ring 对照：

| 实现 | 选择方式 | 数据面 |
| --- | --- | --- |
| 经典 NVLS | `NCCL_ALGO=NVLS`，普通 device buffer | multicast + SHARP，nvls.cc 注册路径 |
| 对称内存（symmetric memory） | 不开 `NCCL_ALGO`，改用 window 注册 | NCCL symmetric kernels（`AllGather_STMC` / `ReduceScatter_LDMC` / `AllReduce_RSxLDMC_AGxSTMC`），本地 LSA/`multimem` 路径，见 `src/sym_kernels.cc`、`src/device/symmetric/` |

两者在内核、同步协议、buffer 要求上都不同，训练侧（FSDP/DP 通信）在 2.28+ 走的是后者，
所以只测 NVLS 不足以代表实际训练路径。

| 问题 | 场景 | case 名 |
| --- | --- | --- |
| 分别发起 AG、RS，和同时发起 AG+RS 的吞吐 | 单组（同一批 rank）依次跑 AG、RS、AG+RS | `single` |
| 单独 all-reduce 的吞吐 | 同一单组跑原地 AR（`dist.all_reduce`，整块张量） | `single` 的 `ar` |
| ring 与 nvls 方案对比 | 同一矩阵用 `NCCL_ALGO=Ring` 和 `NCCL_ALGO=NVLS` 各跑一遍 | 两次启动 |
| 两个半组同时跑同一负载 | T 卡切成两个 T/2 组（16 卡 8+8，32 卡 16+16）同时跑；单独跑的参照直接取 `single-n<T/2>` | `split` 的 `AB-concurrent` |
| 两个独立组同时：A 做 AG、B 做 RS | 同一 split，A 组 all-gather 的同时 B 组 reduce-scatter，参照同样取 `single-n<T/2>` 的 ag、rs | `split` 的 `AB-cross` |

split 不再单独跑 A 组或 B 组：`single-n<T/2>` 用的是 rank 0..T/2−1，和 contiguous 的 A 组完全相同，
B 组是另一批同样大小、同样拓扑（每 tray 4 卡、同一 NVL domain）的 rank，单独跑的结果不该有区别。
所以 `--split-totals` 的每个 T 都要求 T/2 在 `--group-sizes` 里，否则程序拒绝启动。
interleaved 布局（偶/奇 rank）与 single 的 rank 集合不同，用它时 isolated 参照是近似值。

`single` 的四种负载都在同一个 job 里测，每组 AG、RS、AR 各有自己的 NCCL communicator 和 CUDA stream；
`ag+rs` 在两个 communicator 上同时发射，因此它能和单独测的 AG、RS 直接比较：

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
| `NCCL_SYM_KERNEL` | 不设 | 不设 | 可选 | `--symm-kernel ag\|rs\|ar` 固定到单个内核（`AllGather_STMC` / `ReduceScatter_LDMC` / `AllReduce_RSxLDMC_AGxSTMC`），便于定性；只能配同名的单个 `--workloads`，因为它是单内核掩码，其他 op 会没有对称内核可用 |
| `NCCL_SYM_CE_THRESHOLD` | 不设 | 不设 | `INT64_MAX` | 关掉 AllGather 的 copy engine 分支，保证 symm 测的是对称内核（见下文） |
| `NCCL_LAUNCH_ORDER_IMPLICIT` | `1` | `1` | `1` | 两个 communicator 并发时保证 kernel 发射顺序一致 |
| `NCCL_MNNVL_ENABLE` | 多机才设 | 多机才设 | 多机才设 | 跨 tray 强制走 NVLink fabric，禁止静默回退到 IB |
| `NCCL_DEBUG_SUBSYS` | `INIT,ENV,TUNING` | 同左 | 同左 | INIT/ENV 记录 communicator 建立，TUNING 是逐 launch 的算法/内核行，实现校验靠它 |

symm 的实现细节（都按 2.28.9 源码确认）：

- 开关：`comm->symmetricSupport = isAllCudaP2p && NCCL_WIN_ENABLE && cuMemEnable`（`src/init.cc:1376`）。
- 入选条件：collective 的收/发 buffer 都要落在**带 `NCCL_WIN_COLL_SYMMETRIC` 标志**的 window 里
  （`src/scheduler/symmetric_sched.cc:36`）。torch 侧就是 `pg.register_mem_pool(pool, symm=True)`
  → `ncclCommWindowRegister(..., NCCL_WIN_COLL_SYMMETRIC)`（`NCCLUtils.cpp:503`），所以负载
  buffer 必须分配在 `torch.cuda.MemPool()` 里，本程序对 `--algo symm` 就是这样分配的。
- 这个 MemPool 必须用 NCCL 的分配器：`torch.cuda.MemPool(backend.mem_allocator)`（走 `ncclMemAlloc`）。
  NCCL 注册 window 时只按 window 起始地址取**一个** cuMem allocation（`cuMemGetAddressRange` +
  `cuMemRetainAllocationHandle`，`dev_runtime.cc`），只把这一段映射给 peer、绑定到 multicast。
  默认 `torch.cuda.MemPool()` 在 `expandable_segments:True` 下一个 segment 由多个 cuMem handle 拼成，
  注册不报错，但第一个 handle 之后的内容 peer 看不到，symmetric AG/RS 静默写错，表现为
  `wrong result on ranks [...]`；不开 expandable segments 时则是 `cudaMalloc` 内存，注册直接失败。
- 注册是**集合操作**：`ncclDevrWindowRegisterInGroup` 结尾有 `bootstrapBarrier`
  （`dev_runtime.cc`），所以每个 size 都要先全组注册、用完再 `deregister_mem_pool` 才能释放内存。
- 实现范围：`ncclSymkImplemented` 里 AllGather 恒可；ReduceScatter 和 AllReduce 只支持 sum + 浮点
  （不含 fp64，`sym_kernels.cc`）。multicast 归约（`*LDMC*`）对 bf16/fp16/fp32 的 sum 都开放（`ncclSymkMask`）。
- **symm 的 AR 只测 `AllReduce_RSxLDMC_AGxSTMC`**，其他三个对称 AR 内核（`AllReduce_AGxLL_R`、
  `AllReduce_AGxLLMC_R`、`AllReduce_RSxLD_AGxST`）出现在任何一次 launch 里都算该 case 失败。
  AR 没有 copy engine 实现，不受 `NCCL_SYM_CE_THRESHOLD` 影响。见下节“symm AR 为什么不会回退”。
- 跨 tray：对称内核要求整个 communicator 都在一个 LSA team 里（`ncclSymkMask`，`sym_kernels.cc`）。
  MNNVL 生效时 NCCL 把整个 NVL clique 的拓扑合并成一个 system（`graph/topo.cc:1543`
  `ncclTopoFuseXml`），ring 搜索覆盖所有 tray，于是 `nNodes = 1`，8/16/32 卡也能走对称内核。
  是否真的走到了不靠推断：每个 case 都按 TUNING 日志逐 launch 校验（见“实现校验”），
  回退到经典内核的 case 直接失败，实测 `nNodes/MNNVL` 也写进 JSONL。

### symm AR 为什么不会回退（NCCL 2.29.7）

一次 AR 走到 `AllReduce_RSxLDMC_AGxSTMC` 要过三关，任一关不满足 NCCL 都会**静默**改走别的实现：

| 关卡 | 源码 | 条件 | 本程序如何保证 |
| --- | --- | --- | --- |
| 进入对称调度 | `enqueue.cc:397`、`symmetric_sched.cc:34` `ncclSymkAvailable` | `symmetricSupport`（`NCCL_WIN_ENABLE`、cuMem、LSA team 覆盖全部 rank）、`isAllDirectNvlink`、sum + 非 fp64 浮点、掩码非空 | launcher 设 `NCCL_WIN_ENABLE=1`；跨 tray 依赖 MNNVL 让 `nNodes = 1`，否则 LSA team 小于组，AR 没有 Gin 内核，掩码为空 |
| 掩码里保留 RSxLDMC_AGxSTMC | `sym_kernels.cc` `ncclSymkMask` | `hasLsaMultimem = nvlsSupport && LSA team > 2 rank`（`ncclSymkInitOnce`，第一次注册 symmetric window 时执行）；sum + bf16/fp16/fp32 有 LDMC；`NCCL_SYM_KERNEL` 未屏蔽 | launcher 设 `NCCL_NVLS_ENABLE=1`；程序拒绝任何小于 3 rank 的组（`SYMM_AR_MIN_RANKS`） |
| 选中它而不是别的 | `ncclSymkPickKernel` | 收发 buffer 都在 symmetric window 里，否则只剩 LL 内核；在掩码里时由代价模型决定 | `ar_buf` 原地（收 = 发），分配在 `ncclMemAlloc` 的 MemPool 里并注册到 `ar_pg`；按 `queryModel_lsa` 逐项复算，n = 4/8/16/32、16 MiB–2 GiB 全部选 RSxLDMC_AGxSTMC（2 GiB 时 LL 内核本来就被去掉） |

注册之后 `symmetric_sched.cc` 对收发都已注册的任务不再走 `needFallback`。所以剩下可能出问题的都是
运行时条件（NVLS 资源不足、`nNodes > 1`、注册失败等），程序不靠推断：每个 AR case 都逐 launch 核对
TUNING 日志，只要有一行不是 `AllReduce [Symmetric]: <size> Bytes -> Kernel AllReduce_RSxLDMC_AGxSTMC`
（包括回退后的 `AllReduce: ... Algo NVLS/RING ...`），该 case 就失败。

ring / nvls 两轮的 AR 同样不会混入别的实现：`NCCL_WIN_ENABLE` 默认是 1，但它们的 buffer 没有注册
window，`ncclSymkPickKernel` 只留 LL 内核，接着 `needFallback = !NCCL_SYM_NOWIN_ENABLE && 未注册` 为真，
回到经典内核，再由 `NCCL_ALGO` 限定为 Ring 或 NVLS；强制算法不可用时 NCCL 直接报错，不会换算法，
实现校验也只认 `RING/*`、`NVLS/*`（`NVLS_TREE` 不算 NVLS）。launcher 不设 `NCCL_SYM_NOWIN_ENABLE`。

### symm 下的 AllGather 会被 NCCL 改走 copy engine

NCCL 2.29.7 在挑对称内核**之前**就有一个分支（`src/enqueue.cc:2966`）：

```c
} else if (ceAvailable && comm->symmetricSupport && info->coll == ncclFuncAllGather
           && info->count > ncclParamSymCeThreshold()   // NCCL_SYM_CE_THRESHOLD，默认 8*1024*1024
           && comm->minCompCap >= 100 && comm->isAllDirectNvlink) {
  NCCLCHECK(ceCollTaskAppend(...));   // cudaMemcpy batch，不是 kernel
```

- `info->count` 是 AG 每卡的 sendcount（元素数），bf16 下即“每卡 shard > 16 MiB”就走 CE；
  另要求 `nNodes == 1`、recv buffer 已注册 window（`ncclCeAvailable`，`src/ce_coll.cc:100`）。
- 这个分支不看 `NCCL_SYM_KERNEL`，所以固定 `AllGather_STMC` 也拦不住。
- ReduceScatter 没有 CE 实现（`ncclCeImplemented` 只有 AG/AlltoAll/Scatter/Gather）。

默认 sizes（16–2048 MiB）下若不处理，bf16 时 n=4 的 ≥128 MiB、n=8 的 ≥256 MiB、n=16 的 ≥512 MiB、n=32 的 ≥1024 MiB AG 都会是 CE。另外 2048 MiB 时 `nRanks × shard = 2 GiB`，NCCL 会去掉对称 LL 内核（LL 用 32 位计数，`sym_kernels.cc` 的 `nBusBytes >= 2<<30`），只剩 ST/STMC。launcher 给
symm 设 `NCCL_SYM_CE_THRESHOLD=9223372036854775807`，程序也要求它不小于最大 payload 字节数，
否则拒绝启动；万一仍出现 `AllGather [Copy Engine]` 行，实现校验会让该 case 失败。

### 实现校验

TUNING 行在 2.29.7 里**每次 launch 都打、只在 comm rank 0 上打**，debug 文件是行缓冲
（`debug.cc` 的 `setlinebuf`），格式有三种：

| 路径 | 日志 | 记为 |
| --- | --- | --- |
| 经典 kernel | `AllGather: <n> Bytes -> Algo NVLS proto Simple ...`（`enqueue.cc:793`） | `NVLS/Simple` |
| 对称 kernel | `AllGather [Symmetric]: <n> Bytes -> Kernel AllGather_STMC ...`（`symmetric_sched.cc:288`） | `Symmetric/AllGather_STMC` |
| copy engine | `AllGather [Copy Engine]: <n> Bytes -> cudaMemcpy...`（`enqueue.cc:1555`） | `CopyEngine` |

`<n>` 是每卡 count × 元素大小：AG/RS 是 shard（AG 的 sendcount、RS 的 recvcount），AR 没有分片，
是整块张量。每个 case 结束时，
各组的第一个 rank（= 两个 communicator 的 comm rank 0）拿本 case 期间自己进程日志新增的行检查：

1. 每个 op 的行数恰好等于 `warmup + trials × iters`，即 warmup 和计时区间里的每一次 launch（AR 另加 1 次正确性校验 launch）；
2. 每行字节数等于该 op 的每卡 payload：AG/RS 为 `size / n`，AR 为 `size`；
3. 实现符合本轮：ring 只能 `Ring/*`，nvls 只能 `NVLS/*`，symm 只能 `SYMM_KERNELS` 里的对称内核
   （固定了 `NCCL_SYM_KERNEL` 时只能是它）；`CopyEngine` 和回退到经典内核都算错；
4. 不能出现本负载以外的 collective。start gate 也是 AllReduce，靠 4 字节这个大小区分：gate comm rank 0
   每个 trial 恰好一行 `AllReduce: 4 Bytes`，`ar` case 的 AllReduce 行必须是 `size` 字节。

任何一条不满足，所有 rank 一起报错退出，`case.json` 写明违规项，报错信息给出日志目录。
通过的 case 把实测值写进 `nccl_impl`（如 `AllGather=Symmetric/AllGather_STMC`）。程序同时要求
`NCCL_DEBUG=INFO`、`NCCL_DEBUG_SUBSYS` 含 `TUNING`、`NCCL_DEBUG_FILE` 含 `%p`，缺任何一个都拒绝启动。

强制的算法不可用时 NCCL 会打 `Error : no algorithm/protocol available for function ...` 并让
collective 失败（见 `src/enqueue.cc:1859`），不会静默换算法。`NCCL_MAX_NVLS_ARITY` 为 32
（`src/include/device.h:207`），所以 AG/RS 的 NVLS 组最多 32 个 rank；再大就只能是 ring 或
NVLS_TREE，本实验到 32 卡为止。AR 不受这个上限约束（`enqueue.cc:1914` 只对非 AllReduce 检查
`localRanks > NCCL_MAX_NVLS_ARITY`）。

一个例外要自己判断：NCCL 2.28.9 在**跨主机**（`comm->nNodes > 1`）时要求 `collNetSupport == 1`
才允许 `NCCL_ALGO_NVLS`（`src/enqueue.cc:1803`），跨主机 AG/RS 的 NVLS 主要走 `NVLS_TREE`
（arity 3）。若这样，跨 tray 的 `nvls` 就不是纯平铺 NVLS，比较时要在报告里写明。

注意 `comm->nNodes` 按 `ringRecv[0]` 分组（`init.cc:1174`）。MNNVL 下拓扑先按 clique 合并
（`graph/topo.cc:1543`），同一 NVLink domain 的多台 OS 主机通常落在同一个 ring 里，`nNodes = 1`；
以 `comm/` 日志和 JSONL 里实测的 `n_nodes` 为准。

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

单独把对称内存那一档和 ring 对比（跨 tray 是否可用看实现校验和实测 `n_nodes`）：

```bash
bash examples/gb200/nvls-collective/run_nvls_collective_benchmark.sh \
  --algos ring,symm --group-sizes 4
```

16 卡（4 节点 × 4 卡）时只看两组同时（8+8）：

```bash
bash examples/gb200/nvls-collective/run_nvls_collective_benchmark.sh \
  --nnodes 4 --group-sizes 8 --split-totals 16 --split-layouts contiguous,interleaved
```

`--group-sizes` 默认取 4/8/16/32 中不超过 `nnodes × gpus-per-node` 的部分，`--split-totals`
默认取 16/32 中不超过总卡数的最大值（16 卡切 8+8，32 卡切 16+16），不足 16 卡不跑 split。`--dry-run` 只打印命令和将要使用的输出目录，不创建目录，
也不要求 GPU 或 torch。

### 输出目录（不需要指定）

固定为仓库内 `results/gb200/nvls-collective/gpu-num-<总卡数>/<yyyymmdd-hhmmss>`，时间戳由
[`scripts/scale-down/common/run_timestamp_and_logs.sh`](../../scripts/scale-down/common/run_timestamp_and_logs.sh)
的 `resolve_run_time` 给出——**和 scale-down 实验脚本用的是同一个 run id**：

1. 单节点（`--nnodes 1`）直接用本机时钟打一个戳，不需要握手。
2. 多节点时 node 0 打戳后在 `MASTER_ADDR:MASTER_PORT`（launcher 设为 `master-port − 1`，默认
   29610）起一个 `TCPStore`，其他节点阻塞直到读到同一个值，node 0 等所有节点确认后才继续。
   各节点自己格式化 `date` 会把一次逻辑运行散到几个时间戳里，因为 fan-out 要几分钟。
3. 时间戳天然唯一，不需要扫描目录，所以同一次实验重跑会落到新目录；要写进指定 id 就
   `export RUN_TIME=<id>`（`--dry-run` 也走本机时钟，不握手）。

握手是纯 CPU 的，必须在任何 CUDA context 建立之前完成；它经 Gloo 引导，所以 `GLOO_SOCKET_IFNAME`
要先定好（见下节）。launcher 只复用 `resolve_run_time` 这一个函数，不调 `setup_log_dir`，所以日志
仍然写在 `<output_dir>/<algo>/` 下，也不会生成 `node.json`。

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

- 每个 case（实现 × scenario × 负载 × 大小）开一个
  `cudaProfilerStart/Stop` 窗口，只包住计时 trial，warmup、buffer 分配、window 注册不在内。
  `repeat` 让 nsys 每个窗口写一份独立报告，所以**每个测试 × 每个 rank 一份 trace**。
- **所有 rank 对每个 case 都开窗口**，包括该 case 闲置的 rank（如 single-n8 时的 rank 8..31），
  这样各 rank 的第 k 份报告都对应同一个 case。
- 报告里最外层 NVTX range 是 case 标签 `<algo>/<case>`（如 `symm/all_gather+reduce_scatter-256MiB-bf16-single-n4`），
  内层是 `trial0..N`，在 start gate **之后**才打开：gate 本身是 AllReduce，放进 trial range 会被当成
  `ar` 的 kernel；`<algo>/nsys/index.tsv` 按采集顺序列出 `capture_index、label、scenario、phase、start_unix、end_unix`。
- nsys 把第 K 个窗口写成 `<algo>/nsys/rank<R>.<K>.nsys-rep`。**每个 pass 跑完**，node 0 就执行
  `analyse_nsys.py --algo <algo> --file-only`：按 `index.tsv` 的第 K 行把报告改名挪到
  `<algo>/cases/<algo>-<case>/<algo>-<case>.rank_<RR>.nsys-rep`（如
  `nvls/cases/nvls-all_reduce-1024MiB-bf16-single-n16/nvls-all_reduce-1024MiB-bf16-single-n16.rank_01.nsys-rep`），
  和该 case 的 NCCL 日志放在一起。归档只改名、不导出，所以很快；后面的 pass 失败也不会留下一堆
  `rank<R>.<K>` 的报告。分析时再核对每份报告的外层 NVTX 标签和它所在的目录一致，编号对不上会报错。
- 闲置 rank 的窗口里没有任何 CUDA kernel（它只在 Gloo barrier 上等），nsys 导出的 SQLite 就没有
  `CUPTI_ACTIVITY_KIND_KERNEL` 表。这类报告照常归档，分析时视为“没有 kernel”跳过。
- `--nsys` 跑出的数字带 `profiled=true`，`compare.md` 顶部会标注；trace 只用来看内核名、两条
  stream 是否真的重叠、两个 8 组是否同时在跑，**结论表必须用不带 `--nsys` 的那次**。

多机可以改成 Slurm 的 `srun`（`#SBATCH --ntasks-per-node=4`、`--gpus-per-node=4`），
把每条 `uv run ... torch.distributed.run` 换成 `srun` 即可；`torch.distributed.run` 在
Slurm 分配内也能用，`--master-addr` 取第一个节点。

### 用 SQL 查 AG / RS / AR 的 kernel 开销

所有 pass 跑完后，node 0 执行 [`analyse_results.sh`](analyse_results.sh)：先用 `collective_metrics.py`
写 `compare.md`，再（仅 `--nsys`）执行 [`analyse_nsys.py`](analyse_nsys.py)：归档剩下的报告，对每份
报告跑 `nsys export --type sqlite`，再执行 [`nsys_collective_kernels.sql`](nsys_collective_kernels.sql)
里的查询（`label` 用来核对归档，`kernels`、`overlap` 用来统计）。

后处理和测量是分开的：`analyse_results.sh` 失败时 launcher 只打印 warning 和重跑命令，退出码仍由
benchmark 本身决定，已经跑完的测量不会因为后处理出错而作废。任何一步失败，后面的步骤照样执行；
`analyse_nsys.py` 里某个 pass 出错也不影响其他 pass 写进 `nsys_kernels.md`，最后以非零退出码报告。
手动重跑（已归档的报告和 SQLite 导出会复用）：

```bash
bash examples/gb200/nvls-collective/analyse_results.sh \
  --output-dir results/gb200/nvls-collective/gpu-num-4/run-01
# 只做 nsys 这一步，或者只归档某个 pass：
uv run --no-sync python examples/gb200/nvls-collective/analyse_nsys.py \
  --output-dir results/gb200/nvls-collective/gpu-num-4/run-01 [--algo nvls --file-only]
# 或者对单份报告直接用 sqlite3：
nsys export --type=sqlite --output=rank0.1.sqlite rank0.1.nsys-rep
sqlite3 -header -csv rank0.1.sqlite < examples/gb200/nvls-collective/nsys_collective_kernels.sql
```

查询的归属规则：

| 步骤 | 表 | 规则 |
| --- | --- | --- |
| 找 case 和 trial | `NVTX_EVENTS` + `StringIds` | 外层 range 名形如 `symm/all_gather+reduce_scatter-256MiB-bf16-single-n4`，内层为 `trial0..N`；`text` 与 `textId` 两种导出格式都支持 |
| kernel → 发射它的 API | `CUPTI_ACTIVITY_KIND_KERNEL` ⋈ `CUPTI_ACTIVITY_KIND_RUNTIME` | `correlationId` 相同且同一进程（`globalTid >> 24`） |
| kernel → trial | 上面两者 | **发射时刻**落在 trial 的 NVTX range 内（与 `mbench/build_csv.py` 相同），不看 GPU 何时执行 |
| 只保留集合通信 | `StringIds.value` | 名字以 `nccl` 开头且含 `AllGather` / `ReduceScatter` / `AllReduce`，分别记为 `ag`/`rs`/`ar`；`_sleep` 延迟 kernel、warmup、trial range 外的 start gate 都被排除 |
| 只算本负载的 op | `analyse_nsys.py` | 每行只统计 case 的 workload 里有的 op；C++ 版的 trial range 仍包含 gate，AG/RS case 里的 gate AllReduce 由此剔除 |
| AG∩RS 重叠 | 同 trial 的 AG × RS kernel 两两求交 | 同一 collective 在自己的 stream 上串行，所以两两交集之和就是精确的重叠时长 |

输出：`<algo>/nsys/kernels.csv`（每份报告 × trial × op × kernel 一行），以及 `nsys_kernels.md`：每个 case
和组给出 AG / RS / AR 每次迭代的 kernel 时间、AG∩RS 重叠、`overlap_fraction = 重叠 / 较短一方`、kernel 跨度，
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
- AR：`dist.all_reduce` 是原地操作，每次 launch 都会把上一次的结果再求和一遍，k 次后每个元素是
  `(n + n/2)·n^(k−1)`，bf16 几十次就溢出。所以 AR 用独立 buffer（输入同 RS），在 warmup **之前**单独发
  一次并逐元素校验 `= n + n/2`，然后清零；warmup 和计时都在全零数据上跑（零求和仍为零，NCCL 耗时与数值
  无关），结束时再校验 buffer 仍全为 0。这一次校验 launch 计入实现校验的 AllReduce 行数（`warmup + trials × iters + 1`）。

计时不依赖 CPU 同步。每个 trial 在当前 stream 上依次入队：

1. `torch.cuda._sleep`（约 `--launch-delay-ms`，默认 5 ms），让后面所有 launch 在 GPU 开跑前都已入队；
2. **同步起点（start gate）**：在覆盖整个 scenario 所有 rank 的独立 NCCL communicator 上做一次
   1 个 fp32 元素的 all-reduce。all-reduce 要等所有 rank 都进入才会在任何 rank 上完成，所以没有共同
   communicator 的 A、B 两组会在几微秒内同时离开它；
3. `start.record()`，然后 `--iters` 次发射背靠背执行，结束事件记录在各自 stream 上，当前 stream 用 `wait_event` 汇合后计时。默认模式
（`TORCH_NCCL_USE_COMM_NONBLOCKING=0`）下 NCCL kernel 直接跑在当前 stream 上，所以这段就是
NCCL 真实占用 GPU 的时间，不掺 Python 入队开销。**多机是异步的**：慢 rank 会占住自己的
communicator 而别的 rank 继续入队，所以这里测的是 GPU 侧 kernel 时间，不是端到端墙钟；
`enqueue_ms` 会记录 Python 入队时间，超过 `--launch-delay-ms` 的 case 标 `[HOST-BOUND]`，
此时延迟不再被完全吸收，需要调大 `--launch-delay-ms` 或调小 `--iters`。

Gloo barrier 只负责让各 rank 进入同一个 trial，真正的起点是 GPU 上的 start gate；gate 自己的耗时
不计入（start 记录在 gate 之后），它的 AllReduce 也被实现校验单独计数（gate comm rank 0 每个 trial 恰好一行
`AllReduce: 4 Bytes`）。因为所有组从同一时刻计时，各组的结束时间可以直接比较：

```text
concurrency = min(T_A, T_B) / max(T_A, T_B)   # 每个 trial 一个值，报告取中位数
```

它是较长那一组的时间里另一组也在传数据的比例，1 表示两组全程同时在跑；它只在多组 case
（`AB-concurrent`、`AB-cross`）里有值。

#### 时间和吞吐怎么换算

| 量 | 定义 |
| --- | --- |
| 单 rank 每次迭代时间 | `t_rank = (该 rank 所有 op 的结束事件 − start) / iters`；`ag_ms`/`rs_ms` 是各 op 自己 stream 的结束事件 |
| 每个 trial 的时间 `t` | 组内最慢 rank 的 `t_rank`，不用平均掩盖慢 rank |
| 完成时间 | 表 0 给出各 trial `t` 的 mean / median / min / max（`mean_makespan_ms`、`makespan_ms`、`trial_makespan_ms`） |
| `size` | 完整张量字节数：AG 每卡的接收（输出）总量，RS 每卡的发送（输入）总量，AR 每卡原地归约的总量；AG/RS 每卡 shard = `size/n`，AR 不分片 |
| algbw | `size / t`，十进制 GB/s（1 GB = 10⁹ B） |
| busbw | AG/RS：`size × (n−1)/n / t`，nccl-tests 的 AG/RS 定义，等于每卡实际从别的卡收到（AG）或发给别的卡（RS）的字节数 / t；AR：`2 × size × (n−1)/n / t`，nccl-tests 的 all-reduce 定义（等价于一次 RS + 一次 AG），因此可以和 AG/RS 的 busbw 直接比 |
| aggregate busbw | 单个组内所有 op 的 busbw 之和，用该组 makespan 算；AG+RS 时 = 2 × size × (n−1)/n / T |

例：8 卡组、1 GiB、AG 用时 3 ms → algbw = 1.0737 GB / 0.003 s = 357.9 GB/s，
busbw = 357.9 × 7/8 = 313.2 GB/s。表 1–4 用 trial 中位数换算，表 0 用 trial 均值换算。
带宽都不乘 GPU 数；`split` 每组各出一行，两组总量要把 A、B 相加（表 4 已给出 `A+B cross busbw`）。
这些都是从 payload 推出来的值，不是 NVSwitch wire bytes。

## 对比表格（`compare.md`）

`collective_metrics.py` 把三次启动的 `results.jsonl` 合并成以下几张表，除表 0 外每格都是 trial 中位数：

| 表 | 回答 | 每行 | 关键列 |
| --- | --- | --- | --- |
| 0 | 每个测试的完成时间 | 实现 × case × 组 | mean / median / min / max ms、由 mean 算的 algbw / busbw、concurrency、**kernel** |
| 1 | AG / RS / AR / AG+RS | 实现 × 组大小 × 消息 | AG busbw、RS busbw、AR busbw、AG+RS 中各自 busbw、pair aggregate、`speedup`、`overlap_fraction`、**AG / RS / AR kernel** |
| 1b | 三个实现的绝对吞吐 | 组大小 × 消息 × 负载 | ring、nvls、symm 的 aggregate busbw，及最高者 |
| 2 | 相对 ring | 组大小 × 消息 × 负载 | nvls、nvls/ring、symm、symm/ring |
| 2b | 实际执行的内核 | 实现 × 组 | 实测 nNodes、日志里的 `Algo/proto` 或对称内核名 |
| 3 | 两个 T/2 组同跑同一负载 | 实现 × 布局 × 消息 × 负载 × 组 | isolated busbw（取自 `single-n<T/2>`）、concurrent busbw、`slowdown = T_concurrent / T_isolated`、concurrency |
| 4 | A 组 AG、B 组 RS 同时跑 | 实现 × 布局 × 消息 | A 的 AG 与 B 的 RS 各自的 isolated / cross busbw、cross ms、slowdown，`A+B cross busbw`、concurrency |

表 3、表 4 的 isolated 参照都是 `single-n<T/2>` 里同大小、同负载的 case（表 4：A 对 `ag`、B 对 `rs`），
所以 cross case 只在 `--workloads` 同时包含 `ag` 和 `rs` 时运行（默认包含；`--symm-kernel` 固定内核时不跑）。

表 0、表 1 的 kernel 列来自 `nccl_impl`（benchmark 逐 launch 核对过的 TUNING 日志），按 NCCL 2.29.7 的代码生成器
换算成实际运行的函数名：

| 路径 | 日志 | 报告里的名字 | 说明 |
| --- | --- | --- | --- |
| symm | `AllGather [Symmetric]: ... Kernel AllGather_STMC` | `ncclSymkDevKernel_AllGather_STMC` | 每个变体一个 kernel，launch 的就是它（`src/device/symmetric/generate.py`） |
| symm | `ReduceScatter [Symmetric]: ... Kernel ReduceScatter_LDMC`（bf16） | `ncclSymkDevKernel_ReduceScatter_LDMC_sum_bf16` | 归约类带 `_sum_<ty>` |
| ring/nvls | `AllGather: ... Algo NVLS proto SIMPLE` | `ncclDevFunc_AllGather_NVLS_SIMPLE` | 经典路径每个 collective 只 launch 一个特化 kernel（`ncclDevKernel_AllGather_RING_LL`，nsys 显示这个名字），再按 funcId 分发到该设备函数（`best_kernel`、`ncclKernelMain`） |
| ring/nvls | `ReduceScatter: ... Algo RING proto LL128`（bf16） | `ncclDevFunc_ReduceScatter_Sum_bf16_RING_LL128` | 同上 |

同一个 op 在一个 case 里出现多种实现时全部列出；表 1 里 AG+RS 同时发起时换了实现会在括号里注明 `pair：...`。
没有日志证据显示 `-`。

没有记录的格子显示 `-`，不插值。2b 里 `nNodes > 1` 的 symm 行不能当平铺对称内存结论。

## 输出与验收

```text
results/gb200/nvls-collective/gpu-num-<W>/<yyyymmdd-hhmmss>/
  ring/
    results.jsonl  metadata.json               # 只由全局 rank 0 写
    torchrun-node<k>.log                       # 每个节点一份
    nccl-raw/nccl-<host>-<pid>.log             # 每个进程一份，整个进程的原始 NCCL 日志
    comm/ring-<scenario>.rank_<RR>.nccl.log    # 每个 scenario 的 communicator 建立/连接/销毁日志
    cases/ring-<case>/                         # 每个测试一个目录，如 ring-all_reduce-1024MiB-bf16-single-n16/
      case.json                                # rank 0：该 case 的结果记录，或实现校验失败的原因
      ring-<case>.rank_<RR>.nccl.log           # 每个参与 rank：本 case 期间的 NCCL 日志切片
      ring-<case>.rank_<RR>.nsys-rep / .sqlite # 仅 --nsys，每个 pass 后归档 / 分析时导出
    nsys/{index.tsv, kernels.csv}              # 仅 --nsys；归档前的 rank<R>.<K>.nsys-rep 也在这里
  nvls/ ...
  symm/ ...                                    # window 注册的对称内存路径
  compare.md                                   # node 0 合并三个 results.jsonl
  nsys_kernels.md                              # 仅 --nsys，node 0 用 SQL 汇总
```

命名规则（`collective_metrics.py` 的 `scenario_name` / `case_name` / `rank_file`）：

| 项 | 格式 | 例子 |
| --- | --- | --- |
| scenario | `single-n<n>` / `split-n<T>-<layout>` | `single-n16`、`split-n16-contiguous` |
| case | `<workload>-<size>MiB-<dtype>-<scenario>[-<part>]`，workload 写全称（`all_gather`、`reduce_scatter`、`all_reduce`、`all_gather+reduce_scatter`；cross 固定为 `all_gather_A+reduce_scatter_B`），part 为 `AB-concurrent`/`AB-cross` | `all_gather-1024MiB-bf16-single-n16`、`all_gather+reduce_scatter-256MiB-bf16-split-n16-contiguous-AB-concurrent`、`all_gather_A+reduce_scatter_B-1024MiB-bf16-split-n16-contiguous-AB-cross` |
| case 目录 | `<algo>-<case>`（`case_dir_name`），算法打头，拷出 run 目录也能看出测的是什么 | `nvls-all_reduce-1024MiB-bf16-single-n16` |
| 每 rank 文件 | `<algo>-<case>.rank_<RR><suffix>`，rank 补零到至少两位 | `nvls-all_reduce-1024MiB-bf16-single-n16.rank_01.nsys-rep` |

`<size>` 是完整张量：AG 的接收（输出）总量、RS 的发送（输入）总量、AR 的原地归约总量；AG/RS 每卡 shard = `size / n`。

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
4. 每个进程只写自己名字的文件：`nccl-raw/nccl-<host>-<pid>.log`、`comm/` 与 `cases/<algo>-<case>/` 下带
   `.rank_<RR>` 的日志、`nsys/rank<R>.<K>.nsys-rep`（`R` 是 torchrun 设置的全局 `RANK`），以及每个节点的
   `torchrun-node<k>.log`。`case.json` 只由 rank 0 写。
5. 三个实现是三次独立的 torchrun，结果在 `ring/`、`nvls/`、`symm/` 三个 `results.jsonl` 里。
   node 0 的 torchrun 只有在 rank 0 退出后才返回，而 rank 0 退出前已经写完 `results.jsonl`，
   所以随后执行的 `collective_metrics.py` 读到的是完整数据，合并成一份 `compare.md`；合并时会
   检查每个文件里的 `algo` 与目录一致。
6. `--nsys` 时，各节点的 nsys 报告可能比 node 0 晚落盘，`analyse_nsys.py` 会等到报告数达到
   `world_size × case 数` 且文件大小不再变化（最多 300 秒）后再归档；超时就先归档已有的，
   之后再跑一次 `analyse_results.sh` 会把晚到的补上（按编号 K 归档，与先后顺序无关）。

`results.jsonl` 每行一个 case：`scenario`、`phase`、`total_ranks`、`layout`、`group_index`、
`workload`、`group_size`、`ranks`、`size_bytes`、`makespan_ms`、`ag_ms`/`rs_ms`/`ar_ms`、`host_bound`、
`check_passed`、`trial_makespan_ms` 以及各类带宽，`case`/`case_dir` 指向该测试的目录。另外每个 case
带 `nccl_impl`（逐 launch 校验过的实现）、`nccl_algo`（经典路径的 `Algo/proto`）、`nccl_kernel`
（对称内核名或 `CopyEngine`）和实测的 `n_nodes`/`mnnvl`——都是从 launcher 的 `NCCL_DEBUG_FILE`
里读出来的实测值。`compare.md` 的 2b 区会把
每个实现的实测内核列表列出来。`metadata.json` 记录世界大小、每 rank 的
GPU UUID/PCI bus ID/主机名、torch/CUDA/NCCL 版本和全部 `NCCL_*` 环境，便于解释 rank 到
NVLink 拓扑的映射。已有输出目录会报 `FileExistsError`，重复实验用新目录。

接受结论前检查：

1. `check_passed` 全为 true，且 metadata 里的 NCCL 版本、GPU 映射与计划一致。
2. 实现由程序逐 launch 校验（见“实现校验”），跑完即保证；人工复核时看 `cases/<algo>-<case>/` 里 comm rank 0
   的日志切片。只看 `NVLS supported` 或只看 window 注册成功都不能证明某个 collective 用了哪条实现。
3. `split` 的 A、B 两组都和 `single-n<T/2>` 比 `T_concurrent / T_isolated`；两组没有共同 GPU
   仍可能共享交换机链路和控制器，减速本身不能定位到 SHARP ALU。
4. 对照不同 `--trials`、`--iters` 和消息大小重复一次，确认带宽平台稳定；`[HOST-BOUND]` 的 case
   不参与结论。

## 本地检查（无 GPU）

```bash
uv run --no-sync python -m pytest \
  --confcutdir=tests/unit_tests/examples \
  tests/unit_tests/examples/test_gb200_nvls_collective.py \
  tests/unit_tests/examples/test_gb200_nvls_analyse_nsys.py -q
bash -n examples/gb200/nvls-collective/run_nvls_collective_benchmark.sh
bash examples/gb200/nvls-collective/run_nvls_collective_benchmark.sh --dry-run
uv run --no-sync ruff check examples/gb200/nvls-collective
```

单测覆盖算法族/布局/负载的解析与默认值、ring/nvls/symm 三套环境约束（含 symm 必须不设
`NCCL_ALGO`、必须开 `NCCL_WIN_ENABLE`、`NCCL_SYM_KERNEL` 必须与内核名一致）、NCCL 日志探针
（algo/kernel/nNodes/MNNVL）、两个半组划分、带宽与重叠公式、报告四张表以及两个入口只用
标准库可导入（torch 除外）。
它们不需要 GPU、NCCL 或 NVLink，也**不能**验证 CUDA/NCCL 真实行为、带宽或 NVLS 资源。
本目录的程序必须在上面说的 GB200 环境里编译/运行才能真正出数。
