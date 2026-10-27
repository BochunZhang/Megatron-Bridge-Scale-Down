# GB200：基于 torch 的 AG / RS / AR 吞吐对比（ring / NVLS / 对称内存 / 对称 P2P / TMA / copy engine）

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

对称内核本身还有一套不用 multicast 的 P2P 方案：`symm-p2p` pass 用同样的 window，但设
`NCCL_NVLS_ENABLE=0`，NCCL 只能选单播 load/store 的 `AllGather_ST` / `ReduceScatter_LD` /
`AllReduce_RSxLD_AGxST`（小尺寸为 LL 变体）。它与 ring 每卡收发的字节数相同，用来区分
“直接全互连 vs 逐跳 ring”和“multicast vs 单播”两个因素（见下文“symm-p2p”）。

在对称内存之上，NCCL 2.32.3 还有两种 AllGather 实现，本目录各用一个 pass 测（见下文
“symm-tma 与 copy-engine”）：

| 实现 | AllGather | ReduceScatter | 测哪些负载 |
| --- | --- | --- | --- |
| `symm-tma` | `AllGather_TmaSTMC`：TMA `cp.async.bulk` 搬进 smem，再写 multicast 地址 | `ReduceScatter_LDMC` | ag、ag+rs |
| `copy-engine` | copy engine 一次 `cudaMemcpy` 写本 rank 分片的 multicast 地址，交换机扇出 | `ReduceScatter_LDMC` | ag、ag+rs |

NCCL 没有 TMA multicast 或 CE 的 reduce-scatter，所以两者的 RS 都用对称内核里的 multicast 实现
`ReduceScatter_LDMC`，单独测 RS 和 AR 没有新信息（symm 已覆盖），只测 AG 和 AG+RS。
single、split（AB-concurrent、AB-cross）场景与其他实现完全相同。

| 问题 | 场景 | case 名 |
| --- | --- | --- |
| 分别发起 AG、RS，和同时发起 AG+RS 的吞吐 | 单组（同一批 rank）依次跑 AG、RS、AG+RS | `single` |
| 单独 all-reduce 的吞吐 | 同一单组跑原地 AR（`dist.all_reduce`，整块张量） | `single` 的 `ar` |
| ring 与 nvls 方案对比 | 同一矩阵用 `NCCL_ALGO=Ring` 和 `NCCL_ALGO=NVLS` 各跑一遍 | 两次启动 |
| 对称内核的 P2P 方案与 ring、multicast 对比 | `symm-p2p` 跑与 symm 相同的全部负载，只是关掉 multicast | 一次启动，表 2c |
| TMA / copy engine 的 AllGather 与 symm 对比 | `symm-tma`、`copy-engine` 各跑一遍 ag、ag+rs 的 single 和 split | 两次启动，表 2a |
| 两个半组同时跑同一负载 | T 卡切成两个 T/2 组（16 卡 8+8，32 卡 16+16）同时跑；单独跑的参照直接取 `single-n<T/2>` | `split` 的 `AB-concurrent` |
| 两个独立组同时：A 做 AG、B 做 RS | 同一 split，A 组 all-gather 的同时 B 组 reduce-scatter，参照同样取 `single-n<T/2>` 的 ag、rs | `split` 的 `AB-cross` |
| 链路被 RS 占着时，半路加入的 AG 能否跑满 | `--late-join`：同一组 rank 先发 4S 的 RS，跑到约 1/4 时再发 S 的 AG | `late-join`，表 5 |

split 默认关闭（`--split-totals none`），要测时显式传 `--split-totals 16` 或 `32`。

split 不再单独跑 A 组或 B 组：`single-n<T/2>` 用的是 rank 0..T/2−1，和 contiguous 的 A 组完全相同，
B 组是另一批同样大小、同样拓扑（每 tray 4 卡、同一 NVL domain）的 rank，单独跑的结果不该有区别。
所以 `--split-totals` 的每个 T 都要求 T/2 在 `--group-sizes` 里，否则程序拒绝启动。
interleaved 布局（偶/奇 rank）与 single 的 rank 集合不同，用它时 isolated 参照是近似值。

`single` 的四种负载都在同一个 job 里测，每组 AG、RS、AR 各有自己的 NCCL communicator 和 CUDA stream；
`ag+rs` 在两个 communicator 上同时发射，因此它能和单独测的 AG、RS 直接比较。single 的 `ag+rs` 每次迭代之间
两条 stream 汇合一次（`lockstep`，记录里 `"lockstep": true`）：main 等 AG、RS 两条 stream，两条 stream 再等 main，
所以 ag[i] 和 rs[i] 同时起步。不汇合时，`NCCL_LAUNCH_ORDER_IMPLICIT=1` 让同一张 GPU 上的 NCCL kernel 按发射顺序
依次"开始"（下一个要等上一个的 CTA 全部开始执行），ag[i] 要等 rs[i-1] 开始，快的 op 最多领先一次迭代就被卡住，
最后剩一段慢的 op 单独跑，测到的就不是持续重叠。计时口径不变：仍是一个起点 event 到各 op 最后一次结束，除以
迭代数，汇合的 µs 级开销计入其中。单独的 AG、RS 和 split 的 `ag+rs` 仍是连续发射：


```text
speedup          = (T_AG + T_RS) / T_AG+RS     # 1 = 完全串行，理想重叠 = (T_AG+T_RS)/max(T_AG,T_RS)
overlap_fraction = (T_AG + T_RS - T_AG+RS) / min(T_AG, T_RS)   # 1 = 短的操作被完全隐藏，可为负
```

各次启动的结果由 `collective_metrics.py` 合成一份 `compare.md`，不需要手工对表。

## 为什么每个实现单独启动

`NCCL_ALGO` 在 communicator 创建时读取并固化，进程内无法改；对称内存路径还要在创建
communicator 之后注册 window，`NCCL_SYM_KERNEL`、`NCCL_SYM_CE_THRESHOLD` 等也是进程级的，
所以六种实现必须分成六个独立的 torchrun job，用不同的 `--master-port` 串行执行，结果都留在同一个输出目录下。

launcher 为每个实现固定了（`symm-p2p` 简写为 p2p，`symm-tma` 简写为 tma，`copy-engine` 简写为 ce）：

| 变量 | ring | nvls | symm | p2p | tma | ce | 作用 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `NCCL_ALGO` | `Ring` | `NVLS` | 空（=不设） | 空 | 空 | 空 | 前两者强制算法；window 路径不设，否则会屏蔽掉 window 分发 |
| `NCCL_NVLS_ENABLE` | `0` | `1` | `1` | `0` | `1` | `1` | ring 不申请 multicast buffer；p2p 借它关掉对称内核的 multicast；其余保留 multicast（MC）内核 / CE multicast 可选 |
| `NCCL_WIN_ENABLE` | 不设 | 不设 | `1` | `1` | `1` | `1` | 允许 `ncclCommWindowRegister`，这是 window 路径的开关 |
| `NCCL_SYM_KERNEL` | 不设 | 不设 | 可选 | 不设 | `allgather:AllGather_TmaSTMC;reducescatter:ReduceScatter_LDMC` | `reducescatter:ReduceScatter_LDMC` | symm：`--symm-kernel ag\|rs\|ar` 固定到单个内核（`AllGather_STMC` / `ReduceScatter_LDMC` / `AllReduce_RSxLDMC_AGxSTMC`），只能配同名的单个 `--workloads`；`--symm-kernel mc`（NCCL ≥ 2.32）按 func 固定为 `allgather:AllGather_STMC;reducescatter:ReduceScatter_LDMC`，可配任意 workload，`run_symm_cta_sweep.sh` 默认使用；tma、ce 按 func 分别固定（见下文） |
| `NCCL_SYM_CE_THRESHOLD` | 不设 | 不设 | `INT64_MAX` | `INT64_MAX` | `INT64_MAX` | `0` | symm/p2p/tma 关掉 AllGather 的 copy engine 分支；ce 反过来让每个 AllGather 都走 copy engine |
| `NCCL_CE_COLL_AG_MULTICAST_THRESHOLD` | 不设 | 不设 | 不设 | 不设 | 不设 | `2^62` | ce：分片不超过它就写 multicast 地址，而不是 N−1 份 unicast 拷贝 |
| `NCCL_LAUNCH_ORDER_IMPLICIT` | `1` | `1` | `1` | `1` | `1` | `1` | 两个 communicator 并发时保证 kernel 发射顺序一致 |
| `NCCL_MNNVL_ENABLE` | 多机才设 | 多机才设 | 多机才设 | 多机才设 | 多机才设 | 多机才设 | 跨 tray 强制走 NVLink fabric，禁止静默回退到 IB |
| `NCCL_DEBUG_SUBSYS` | `INIT,ENV,TUNING` | 同左 | 同左 | 同左 | 同左 | 同左 | INIT/ENV 记录 communicator 建立，TUNING 是逐 launch 的算法/内核行，实现校验靠它 |

`--symm-kernel` 只传给 symm pass。p2p、tma、ce 的环境由 benchmark 逐项核对，任何一项不符都拒绝启动，
不会像 symm 的 `NCCL_SYM_CE_THRESHOLD` 那样自动改写：这几个变量本身就决定了跑的是哪种 AllGather。

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

### symm-tma 与 copy-engine（NCCL 2.32.3）

以下都按 2.32.3 源码确认。

**怎么选中实现。** torch 不能逐次传 `ncclCollConfig_t`（C++ 版 `../nvls-collective-cpp` 用的是它），只能用进程级变量：

- `NCCL_SYM_KERNEL` 在 2.32 里和 `NCCL_ALGO` 用同一个解析器（`src/tuning/cost_model.cc` 的 `parseList`），
  支持 `<func>:<kernel>;...`。列出的 func **只能**用列出的内核，经典算法也一并关掉；没列出的 func（AllReduce，
  也就是 start gate）不受影响。所以 tma 写 `allgather:AllGather_TmaSTMC;reducescatter:ReduceScatter_LDMC`，
  ce 只固定 RS。
- TMA 内核另需 `ncclSymkTmaAvailable`（`minCompCap >= 100`、`maxSharedMemOptin` 足够、`NCCL_SYM_TMA_ENABLE`
  默认 1）和收发 offset 16 B 对齐（`ncclMemAlloc` 的 MemPool 满足），`*STMC`/`*LDMC` 需要 `hasLsaMultimem`
  （`NCCL_NVLS_ENABLE=1`、LSA team > 2 rank）。不满足时被固定的 func 没有任何实现可用，NCCL 直接报
  `No algorithm/protocol nor symKernelId available`，不会静默换内核。
- CE 分支在挑内核**之前**（`src/enqueue/enqueue.cc`）：`count > NCCL_SYM_CE_THRESHOLD`、recv buffer 在 window 里、
  Blackwell、全直连 NVLink 时 AllGather 走 `ceCollTaskAppend`。ce 设阈值 0，所以每个 AllGather 都走 CE；
  symm、tma 设 `INT64_MAX` 把这个分支关掉。
- CE 内部走 multicast 还是 unicast 由 `ncclCeAllGatherUseMulticast`（`src/ce_coll.cc`）决定：有 `hasLsaMultimem` 时，
  `NCCL_CE_COLL_AG_MULTICAST_THRESHOLD >= 0` 就绕过 cost model，分片 ≤ 阈值走 multicast（一条
  `cudaMemcpyBatchAsync` 写 multicast 地址），否则 N−1 份 unicast。launcher 设 `2^62`（与 C++ 版相同）。

**communicator 必须提前建好。** 固定了 AG/RS 之后，`_connect` 原来那种用普通 buffer 的小 AG/RS 会失败：
没有经典算法可退，multicast 内核又要等第一次注册 symmetric window（`ncclSymkInitOnce` 设置 `hasLsaMultimem`）
才可选。而 torch 的 `register_mem_pool` 要求 communicator 已存在，communicator 默认又是第一次 collective 才建。
所以这两个 pass 用 `dist.new_group(..., device_id=...)` 让 torch 在建组时就创建 NCCL communicator，
`_connect` 改为在一个 4 KiB/rank 的已注册 window 上做第一次 AG 和 RS（`_connect_windows`），跑的就是被固定的实现。
ring、nvls、symm 的建组方式不变。

**校验。** CE 的 TUNING 行不区分 multicast 和 unicast，只在有 `hasLsaMultimem` 时追加
`; CE synchronization with NVLS`。benchmark 把这种行记为 `CopyEngine+NVLSSync`，ce 只接受它；再加上程序要求
`NCCL_CE_COLL_AG_MULTICAST_THRESHOLD` 不小于最大 payload，二者合起来才说明数据是写 multicast 地址的。
tma 的 AG 只接受 `Symmetric/AllGather_TmaSTMC`，两者的 RS 只接受 `Symmetric/ReduceScatter_LDMC`。
组小于 3 rank（含 split 的半组）时没有 multicast，程序拒绝启动。

**split。** AB-concurrent 跑 ag、ag+rs；`--workloads` 同时含 `ag`、`rs` 时照样跑 AB-cross（A 组用本 pass 的 AG，
B 组用 `ReduceScatter_LDMC`）。因为本 pass 不测单独的 rs，表 4 里 B 的 isolated 为 `-`，可参照 symm 中
RS kernel 同为 `ReduceScatter_LDMC` 的行。

**nsys。** CE AllGather 是 memcpy 不是 kernel，`nsys_kernels.md` 里 ce 的 AG 没有 kernel 时间，只有 RS 的。

### symm-p2p：关掉 multicast 的对称内核（NCCL 2.32.3）

**怎么选中。** 对称路径的开关 `comm->symmetricSupport` 只看 `isAllCudaP2p`、`NCCL_WIN_ENABLE`、cuMem
和 LSA team，与 NVLS 无关；multicast 内核则要 `hasLsaMultimem = ncclNvlsSymmetricMultimemEnabled(comm)
&& LSA team > 2 rank && !p2pCrossClique`，前者要求 `comm->nvlsSupport`。`NCCL_NVLS_ENABLE=0` 让
`ncclNvlsInit` 保持 `nvlsSupport = 0`，于是 `ncclSymkMask` 去掉所有 `*STMC` / `*LDMC` 内核，cost model 只能在
单播内核里选，无需 `NCCL_SYM_KERNEL`：

| op | 单播对称内核（`SYMM_P2P_KERNELS`） | 数据怎么走 |
| --- | --- | --- |
| AG | `AllGather_ST`、`AllGather_LL`、`AllGather_TmaST` | 每卡把自己的 shard 直接 store 进其余 n−1 卡的 window |
| RS | `ReduceScatter_LD`、`ReduceScatter_LL`、`ReduceScatter_TmaLD` | 每卡从其余 n−1 卡的 window 直接 load 自己那段再归约 |
| AR | `AllReduce_RSxLD_AGxST`、`AllReduce_AGxLL_R`、`AllReduce_RSxTmaLD_AGxTmaST` | RSxLD + AGxST 串起来；`AGxLL_R` 只用于小尺寸 |

`NCCL_SYM_CE_THRESHOLD` 仍设 `INT64_MAX`，让 AllGather 不走 copy engine。benchmark 要求 `NCCL_NVLS_ENABLE=0`、
`NCCL_WIN_ENABLE=1`、`NCCL_ALGO` 与 `NCCL_SYM_KERNEL` 不设、CE 阈值不小于最大 payload，不自动改写；
逐 launch 校验时任何 `*MC` 内核、经典内核或 copy engine 都让 case 失败。p2p 不需要 multicast，所以没有
symm 那样“组至少 3 rank”的限制，建组和连接方式与 symm 默认相同。

**与 ring 的通信量。** 设完整张量 S、n 卡，只数 NVLink 上每卡发出（= 收到）的有效字节：

| op | ring | symm-p2p（ST/LD） | symm（multicast） |
| --- | --- | --- | --- |
| AG | (n−1)/n·S，n−1 步逐跳转发 | (n−1)/n·S，一步直接写 n−1 个 peer | 发 S/n（`multimem.st` 由交换机扇出），收 (n−1)/n·S |
| RS | (n−1)/n·S，n−1 步逐跳“收-加-发” | (n−1)/n·S，一步直接读 n−1 个 peer | 收 S/n（`multimem.ld_reduce` 在交换机归约），发 (n−1)/n·S |
| AR | 2(n−1)/n·S（RS + AG） | `RSxLD_AGxST`：2(n−1)/n·S | RS 收 + AG 发各 S/n |

所以 AG、RS、AR 上 symm-p2p 的总通信量与 ring **相等**，都等于 busbw 定义里的那一份，busbw 可以直接比。
差别不在字节数而在路径：ring 每个分片要经 n−1 跳，每跳多一次本地 HBM 读写和步间同步，链路上同一时刻
只用到 1 个前驱 / 1 个后继；p2p 在 NVSwitch 上一步全互连，没有中间转发。两个例外：LL 协议（`*_LL`、
`AllReduce_AGxLL_R`）每 8 字节带 8 字节 flag，线上字节约翻倍；`AllReduce_AGxLL_R` 是每卡把整块 S 发给每个
peer 再本地归约，每卡收 (n−1)·S，远多于 ring，只在小尺寸被选中。multicast 把一侧缩到 S/n，这正是
symm 相对 p2p 的增益来源。

表 2c 并排列出 ring、symm-p2p、symm：`symm-p2p/ring` 是相同字节数下直接全互连 vs 逐跳 ring，
`symm/symm-p2p` 是同一套 window 上 multicast 的增益。

### 实现校验

TUNING 行在 2.29.7 里**每次 launch 都打、只在 comm rank 0 上打**，debug 文件是行缓冲
（`debug.cc` 的 `setlinebuf`），格式有三种：

| 路径 | 日志 | 记为 |
| --- | --- | --- |
| 经典 kernel | `AllGather: <n> Bytes -> Algo NVLS proto Simple ...`（`enqueue.cc:793`） | `NVLS/Simple` |
| 对称 kernel | `AllGather [Symmetric]: <n> Bytes -> Kernel AllGather_STMC ...`（`symmetric_sched.cc:288`） | `Symmetric/AllGather_STMC` |
| copy engine | `AllGather [Copy Engine]: <n> Bytes -> cudaMemcpy`（2.32.3 `ce_coll.cc`） | `CopyEngine` |
| copy engine + multimem | `AllGather [Copy Engine]: <n> Bytes -> cudaMemcpy; CE synchronization with NVLS` | `CopyEngine+NVLSSync` |

`<n>` 是每卡 count × 元素大小：AG/RS 是 shard（AG 的 sendcount、RS 的 recvcount），AR 没有分片，
是整块张量。每个 case 结束时，
各组的第一个 rank（= 两个 communicator 的 comm rank 0）拿本 case 期间自己进程日志新增的行检查：

1. 每个 op 的行数恰好等于 `warmup + trials × iters`，即 warmup 和计时区间里的每一次 launch（AR 另加 1 次正确性校验 launch）；
2. 每行字节数等于该 op 的每卡 payload：AG/RS 为 `size / n`，AR 为 `size`；
3. 实现符合本轮：ring 只能 `Ring/*`，nvls 只能 `NVLS/*`，symm 只能 `SYMM_KERNELS` 里的对称内核
   （固定了 `NCCL_SYM_KERNEL` 时只能是它），`CopyEngine` 和回退到经典内核都算错；symm-p2p 只能是
   `SYMM_P2P_KERNELS` 里的单播对称内核，出现 `*MC` 内核即失败；symm-tma 的 AG 只能是
   `Symmetric/AllGather_TmaSTMC`，copy-engine 的 AG 只能是 `CopyEngine+NVLSSync`，二者的 RS 只能是
   `Symmetric/ReduceScatter_LDMC`（`PINNED_IMPLEMENTATIONS`）；
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

只测 TMA 和 copy engine 的 AllGather（带上 symm 作参照，表 2a 给出相对 symm 的比值）：

```bash
bash examples/gb200/nvls-collective/run_nvls_collective_benchmark.sh \
  --algos symm,symm-tma,copy-engine --sizes-mib 64,256,1024 --trials 3
```

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

`--group-sizes` 默认是 `nnodes × gpus-per-node`（一组包含所有 rank），要测更小的组就显式列出，如 `4,8,16`；`--split-totals`
默认 `none`（不跑 split），要测两组同时就传 16（8+8）或 32（16+16）。`--dry-run` 只打印命令和将要使用的输出目录，不创建目录，
也不要求 GPU 或 torch。`--launch-delay-ms`（默认 5）是每个 trial 前的 GPU sleep，case 被标 `[HOST-BOUND]` 时调大。

### late join：RS 跑到 1/4 时加入 AG（`--late-join`）

```bash
bash examples/gb200/nvls-collective/run_nvls_collective_benchmark.sh \
  --nnodes 8 --algos ring,nvls,symm --late-join
```

问题：一个 RS 已经在用 NVLink，同一组 rank 再发一个 AG，AG 能不能拿到它单独跑时的带宽？两者合起来能不能超过
单个 op？multicast 下 AG（`multimem.st`）主要占入方向、RS（`ld_reduce`）主要占出方向
（`documents/gb200/03-nccl-symmetric-nvls-kernel.md` 6.1 节），理论上可以叠加，但同时发起的 AG+RS 实测基本串行
（同文 6.3 节）。late join 让 RS 先独占一段时间，再看 AG 加入后的分配。

`--late-join` 只跑这个场景（不跑 single，也不能和 split 同时开），对 `--group-sizes` 的每个 n、`--sizes-mib`
的每个 S，依次跑三个 case（scenario `late-join-n<n>`）：

| phase | case | 做什么 |
| --- | --- | --- |
| `rs-isolated` | `reduce_scatter-<4S>MiB-...-late-join-n<n>` | 单独 RS(4S)，得到 `T_RS iso` |
| `ag-isolated` | `all_gather-<S>MiB-...-late-join-n<n>` | 单独 AG(S)，得到 `T_AG iso` |
| `join` | `reduce_scatter_x4+late_all_gather-<S>MiB-...-late-join-n<n>` | RS(4S) 先发；AG 的 stream 先 GPU sleep `--join-fraction × T_RS iso`（默认 0.25），再发 AG(S) |

- **计时**：三个 case 都逐迭代计时（`_run_joined`）。每次迭代先在 main stream 上记一个起点，两条 op stream 都等它，
  迭代结束 main 等两条 stream 汇合，所以每次迭代都是"RS 从发射开始、AG 中途加入"，参照和 join 用同样的口径。
  AG 的 stream 在 sleep 之后、发 AG 之前记录 `ag_begin`：`ag_begin_ms` 是实测的放行时刻，`join_fraction` 是它相对
  `T_RS iso` 的比例，不依赖 sleep 的换算是否准确。sleep 的周期数用的是程序启动时标定的 `_sleep` 速率。
- **前提：所有 launch 在迭代开始前都已入队。** 否则 GPU 等 host，sleep 结束时 AG 还没发出来，AG 实际晚于
  `ag_begin` 才开始。launcher 在 `--late-join` 下把默认值改成 `--iters 5 --trials 10`，入队约 2–3 ms，
  在 5 ms 的 `--launch-delay-ms` 之内；被标 `[HOST-BOUND]`（表 5 的 †）的行不要用，调大 `--launch-delay-ms` 重跑。
- **默认大小**：S = 64,256,512 MiB（RS 为 256、1024、2048 MiB）。按 03 号文档 symm 8–32 卡的数据，RS(4S) 为
  0.41–0.44 / 1.58–1.73 / 3.14–3.45 ms，AG 约在 0.1 / 0.4 / 0.8 ms 放行，单独跑时 AG 为 0.11 / 0.39 / 0.76 ms，正常情况下 AG 在 RS 一半左右
  就结束，整段 AG 都落在 RS 里。可用 `--sizes-mib` 改。
- **实现校验**照常：RS 每行必须是 `4S/n` 字节，AG 每行 `S/n` 字节（`check_implementation` 的 `op_sizes`）。
- 所有实现都能跑：symm-tma、copy-engine 的 RS 是 `ReduceScatter_LDMC`，copy-engine 的 AG 不占 SM，
  可以用来区分 SM / CTA 竞争和链路竞争。

表 5（`compare.md`）读法：

| 列 | 含义 |
| --- | --- |
| AG / RS slowdown | join 下 AG 从放行到结束的时间 / `T_AG iso`；RS 从发射到结束的时间 / `T_RS iso` |
| AG / RS algbw overlap | overlap 窗口 = [AG 放行, min(AG 结束, RS 结束)]。窗口外假设各自按单独跑的速率前进，剩下的字节数在窗口内完成，除以窗口时长 |
| capacity | 窗口内 AG 速率 / AG 单独速率 + RS 速率 / RS 单独速率：**2 = 两者互不影响、都跑满；1 = 分时共用同一瓶颈** |
| AG share | AG 那一项 / capacity：0.5 是平分，远小于 0.5 说明先跑的 RS 占着资源，后来的 AG 抢不到 |
| AG in RS | AG 是否在 RS 结束前完成；否则 AG 后半段是单独跑的，窗口只覆盖它的一部分 |

**同时发射（`--join-fraction 0`）**：AG 的 stream 不 sleep，每次迭代 RS(4S) 和 AG(S) 从同一个起点出发，
迭代结束两条 stream 汇合后才开始下一次（与 single `ag+rs` 的 lockstep 一样，但每次迭代单独计时），overlap 窗口就是
[0, min(AG 结束, RS 结束)]：AG 整段落在 RS 里时，窗口内 AG algbw 就是它在 overlap 下的实测带宽，不依赖估算；
RS 的窗口内带宽仍按上面的方法推算。join 列为 0（没有 `ag_begin` 事件）。不同 `--join-fraction` 的 case 名相同，
放在不同的运行目录里比较。

```bash
bash examples/gb200/nvls-collective/run_nvls_collective_benchmark.sh --algos symm --late-join --join-fraction 0
```

capacity 和窗口内带宽都是从完成时间推出来的估算。直接证据是 `--nsys` 的 NVLink TX/RX 采样：在 nsys 时间线上
对齐 RS、AG 两个 kernel，看 AG 加入前后入方向（RX）和出方向（TX）各自是否上升。

### 分别指定 AG / RS / AR / P2P 的 CTA 数（`--cta-configs`）

```bash
bash examples/gb200/nvls-collective/run_nvls_collective_benchmark.sh --algos symm --group-sizes 4 \
  --sizes-mib 1024 --workloads ag+write,rs+write,ag+rs \
  --cta-configs 'default;ag=3,rs=3;p2p=32;ag=3,rs=6' --nsys
```

问题：AG/RS 和 P2P（或者 AG 和 RS）同时跑时，给它们分配不同数量的 SM，流量怎么变化？

**怎么限制。** NCCL 每个 channel 一个 CTA（thread block），CTA 数在 communicator 创建时由 `ncclConfig_t` 的
`minCTAs` / `maxCTAs` 决定。torch 通过 `dist.new_group(..., pg_options=ProcessGroupNCCL.Options())` 的
`options.config.min_ctas / max_ctas` 把它传给这个 process group 下的**每一个** NCCL communicator（包括 symm
懒初始化时 torch 为 send/recv 单独建的 2 卡 communicator）。benchmark 里 AG、RS、AR、AG2、RS2、P2P 本来就是
各自独立的 group，所以**同一个 case 里每个 op 可以用不同的 CTA 数**，例如 `ag+rs` 里 AG 3 个、RS 6 个。

| 键 | 作用于 | 设置方式 | 说明 |
| --- | --- | --- | --- |
| `ag=N` | AG 的 comm（`ag+ag` 的第二个 AG 也跟随） | `min_ctas = max_ctas = N` | 经典 kernel：channel 数 = N；symm：对称 kernel 的 CTA 数 = N |
| `rs=N` | RS 的 comm（`rs+rs` 的第二个 RS 也跟随） | 同上 | |
| `ar=N` | AR 的 comm | 同上 | |
| `p2p=N` | read / write 的 comm，N ∈ 1,2,4,8,16,32 | N ≤ 16：`max_ctas = N`；N = 32：`min_ctas = 64` | 见下 |

- 每个 `;` 分隔的条目是一组配置；`default` 是 NCCL 自己选。每组配置都会**重建所有 communicator**，把所有 case
  再跑一遍（同一个 torchrun 进程内循环，不是多次调用 Python），case 名追加 `-cta-ag3-rs6-p2p32` 这样的后缀，
  `comm/` 日志也带同样的后缀。结果写进同一个 `results.jsonl`，每行带 `cta_config`、`requested_ctas`、
  `observed_ctas`、`predicted_p2p_ctas`。
- **symm 的上限**：原版 NCCL 的 multicast 对称 kernel（`AllGather_STMC`、`AllGather_TmaSTMC`、`ReduceScatter_LDMC`、
  `AllReduce_RSxLDMC_AGxSTMC`）在 Blackwell 上最多 `divUp(32, n)` 个 CTA（2.32.3 `tuning/sym_model/lsa.cc`
  `maxBlocksLsa`），`minCTAs/maxCTAs` 突破不了：4 卡最多 8（默认 6），8 卡 4，16 卡 2，32 卡 1，超出的请求会被静默截断。
  `install/install-nccl-2.32.3-cta.sh` 编出的 nccl-2.32.3-cta（横幅 `NCCL version 2.32.3-cta+...`）允许大于上限的
  `minCTAs` 生效，最多 64，所以 benchmark 不再在启动前拒绝超过上限的 `ag=/rs=/ar=`。用原版 NCCL 跑这类配置时，
  下面的校验会以「requested N CTAs, NCCL launched 4」让 case 失败。进程级的 `NCCL_SYM_CTAS` 同样能突破上限，但它同时作用于
  所有对称 kernel，所以不能和 `ag=/rs=/ar=` 一起用（可以和 `p2p=` 一起用）。
- **P2P**：send/recv 的 CTA 数由两层决定（`paths.cc`、`enqueue.cc`）：comm 的 P2P channel 数
  `pow2Up(nChannels)`，以及每次 send 按大小在 `[起始值, 每 peer 上限 32]` 之间翻倍（同 tray 每个 channel 最多
  32 × 512 KiB = 16 MiB，跨 tray 每个 channel 一个 256 KiB chunk）。4 卡 1024 MiB 时每次 send 256 MiB，默认 16 个；
  `max_ctas = N` 把它压到 N；要翻倍到 32 必须让 comm 有 64 个 P2P channel，即 `min_ctas = 64`。消息太小时达不到目标值，
  `predicted_p2p_ctas` 会如实给出推算值。
- **校验**：集合通信每次 launch 的 CTA 数都从 comm rank 0 的 TUNING 日志读（对称 kernel 的 `nchannels`，经典 kernel 的
  `channel{Lo..Hi}`），和请求值不一致、或者 symm 因为 CTA 上限换掉了 multicast kernel（比如 `AllGather_STMC` 变成
  `AllGather_ST`），case 直接失败——否则比较的是 kernel 而不是 CTA 数。NCCL 不为 send/recv 打 CTA 日志，P2P 的 CTA 数是
  按源码推算的（compare.md 里写成 `~N`），**实测值看 `--nsys` 后 `nsys_kernels.md` 的 `CTAs (gridX)` 列**。
- 进程级的 `NCCL_MIN_CTAS`、`NCCL_MAX_CTAS`、`NCCL_{MIN,MAX}_NCHANNELS`、`NCCL_{MIN,MAX}_NRINGS`、
  `NCCL_NVLS_NCHANNELS`、`NCCL_{MIN,MAX}_P2P_NCHANNELS` 会覆盖每个 comm 的 config。GB200 镜像默认导出了
  `NCCL_MIN_CTAS=4`、`NCCL_MIN_NCHANNELS=16`（见历次 `metadata.json`），所以 launcher 在 `--cta-configs` 含非 default
  条目时用 `env -u` 把它们从 worker 环境里去掉，并打 warning：**这次运行里的 `default` 是 NCCL 自己的选择，和以前
  带着这两个变量跑出来的结果不完全同口径**，对比请用同一次运行里的 default。直接调用 benchmark 时它们仍会被拒绝。

**AG / RS CTA 扫描（`run_symm_cta_sweep.sh`）**：用 `LD_PRELOAD` 换上 nccl-2.32.3-cta，在 symm 上跑
`--workloads ar,ag,rs,ag+rs --sizes-mib 1024 --nsys`，`--cta-configs` 是 `default` 加上所有 `1 <= ag <= rs <= --max-ctas`
（默认 8，共 37 组）。每个节点运行同一条命令：

```bash
bash examples/gb200/nvls-collective/run_symm_cta_sweep.sh --nnodes 8 --group-sizes 32
```

- `--venv` 默认是 `.venv/python3.12-torch2.13-nccl2.32.3`，`--nccl-lib` 默认是
  `nccl/nccl-2.32.3-cta/build/lib/libnccl.so.2`；`--` 之后的参数原样传给 `run_nvls_collective_benchmark.sh`，并覆盖上面的默认值。
- 只有不带 `nvidia-nccl-cu13` 的 torch（`install/install-torch-2.13.0-nccl-2.32.3.sh` 编出的）能被 `LD_PRELOAD` 干净替换：
  官方 wheel 在 `import torch` 时会按路径再加载一份 NCCL。所以启动前会检查 python 里只映射了 preload 的那一份 libnccl，
  跑完后会检查 `<run>/*/nccl-raw/` 里每个进程的横幅都是 `NCCL version 2.32.3-cta+...`（worker 由 nsys 拉起），否则以失败退出。
  copy-engine 的 AG 不用 SM，不接受 `ag=`。
- 读结果：`compare.md` 表 1–5 只用 `default` 的结果，新增的**表 6** 把每个非 default 配置和同一 case 的 default
  配对，每个 op 给出 `CTA 数 / 每次迭代 ms (相对 default)`。

**AllGather_STMC 暂停扫描（`run_symm_cta_pause_sweep.sh`）**：人为让 AG 的 multimem store 变慢，看 ag+rs 重叠时 RS
的时间怎么变。只扫暂停值，AG / RS 的 CTA 数保持 NCCL 默认（`--cta-configs default`）。先编出带暂停的 NCCL（clone `v2.32.3-1` 到 `nccl/nccl-2.32.3-cta-with-pause`，依次打
`nccl-2.32.3-cta.patch` 和 `nccl-2.32.3-cta-pause.patch`，横幅为 `NCCL version 2.32.3-cta-pause+cuda...`）：

```bash
./install/install-nccl-2.32.3-cta.sh --with-pause
bash examples/gb200/nvls-collective/run_symm_cta_pause_sweep.sh --nnodes 8 --group-sizes 32 --pause-cycles 0,500,1000,2000,4000
```

- 这个 patch 新增 `NCCL_SYM_AG_STMC_PAUSE_CYCLES`（默认 0，即不暂停）：`bcastMultimem` 主循环里每个 warp 每发完一块 4 KiB
  （8 × 32 lane × 16 B）的 multimem store 后，按 `clock64()` 自旋这么多个 SM 周期，再发下一块；首尾的零头不暂停。
  只有 `AllGather_STMC` / `AllGather_TmaSTMC` 用到这个参数，`ReduceScatter_LDMC` 和 `AllReduce_RSxLDMC_AGxSTMC` 不变，所以默认
  workload 是 `ag,rs,ag+rs`，rs 作为对照。
- 为什么用 `clock64()` 而不用 `%globaltimer`：`clock64()` 每个 SM 周期加 1，暂停长度能精确到几个周期；`%globaltimer`
  虽然以 ns 为单位，但多久跳一次由 GPU 和驱动决定（挂上 profiler 后还可能变）。跳动间隔为 T 时，p ns 的自旋实际持续
  `(⌈p/T⌉·T − T, ⌈p/T⌉·T]`，所有小于 T 的 p 都分不出来。代价是换成 ns 要除以 SM 频率：暂停 ns = 周期数 / SM 频率，
  GB200 降频时同样的周期数会更长。可以扫描前用 `nvidia-smi -lgc <MHz>,<MHz>` 锁频（需要 root），或事后从 nsys GPU metrics
  的 GPC Clock Frequency 读实际频率。`run_globaltimer_probe.sh` 会测 `%globaltimer` 的跳动间隔、各 SM 的频率和这段自旋的实际周期数。
- 早先的 `%globaltimer` 版 patch 读的是 `NCCL_SYM_AG_STMC_PAUSE_NS`，横幅同样是 `2.32.3-cta-pause`，扫描脚本会因为库里没有
  `NCCL_SYM_AG_STMC_PAUSE_CYCLES` 而拒绝它。用旧 patch 打过的 `nccl/nccl-2.32.3-cta-with-pause` 需要先
  `git -C nccl/nccl-2.32.3-cta-with-pause checkout -- .` 还原，再重新运行 `install-nccl-2.32.3-cta.sh --with-pause`。
- NCCL 每个进程只读一次这个变量，所以每个暂停值单独启动一次 launcher，结果目录带上暂停标记：
  `results/gb200/nvls-collective/gpu-num-<N>/<id>-pause<cycles>cyc/`。整次扫描共用一个 `<id>`（导出 `RUN_TIME` 可以指定；
  多节点时由 node 0 生成并下发）。node 0 在每个目录写 `pause.json`，各 algo 的 `metadata.json` 也记录了这个变量；跑完后检查每个
  NCCL 进程的横幅是 `2.32.3-cta-pause`、日志里有 `NCCL_SYM_AG_STMC_PAUSE_CYCLES set by environment to <cycles>.`。
  node 0 还会把归档好的 nsys 报告改名，带上暂停值：`<algo>-<case>.pause<cycles>cyc.rank_<RR>.nsys-rep`（case 目录名不变），
  拷出运行目录也能看出是哪个暂停值；`analyse_nsys.sh` 照常分析这些报告。
- 每个 case 都走 symmetric 的 multicast kernel（`AllGather_STMC` / `ReduceScatter_LDMC`），不走 unicast（P2P）kernel：
  `--symm-kernel mc` 固定 `NCCL_SYM_KERNEL`、`NCCL_NVLS_ENABLE=1`，`NCCL_SYM_TMA_ENABLE=0` 排除 `AllGather_TmaSTMC`，
  benchmark 按 TUNING 日志逐次检查，跑了别的 kernel 的 case 直接判失败。`--algos`、`--symm-kernel`、`--cta-configs`
  不能透传。
- `--nnodes` 默认 1（单 tray 4 GPU）。torch 要求和 `--` 透传与 `run_symm_cta_sweep.sh` 相同；`--pause-cycles 0` 就是同一个库下的基线。

**看实际用了多少 SM。** CTA 数 ≈ 占用的 SM 数（CTA 数远小于 SM 数时调度器会把它们分散到不同 SM；一个 SM 理论上能放多个
CTA，所以只是近似）。证据从强到弱：

1. nsys `CUPTI_ACTIVITY_KIND_KERNEL.gridX`：每个 kernel 的 grid 大小，`nsys_kernels.md` 的 `CTAs (gridX)` 列（含 P2P 的
   `ncclDevKernel_SendRecv`）。
2. nsys GPU metrics 的 `SMs Active`：整卡随时间的 SM 占用比例，重叠阶段约等于（AG CTA + P2P CTA）/ SM 总数；在
   `gpu_metrics.csv` 里（`nsys_gpu_metrics.md` 只列 NVLink / DRAM / C2C）。
3. 集合通信：TUNING 日志的 `nchannels` / `channel{Lo..Hi}`，即 `results.jsonl` 的 `observed_ctas`。
4. ncu 的 `LaunchStats` / `Occupancy`：Grid Size 和每个 SM 能放的 CTA 数；ncu 串行重放，看不到重叠。

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

`--nsys` 用 `torchrun --no-python` 给**每个 worker** 套一个 `nsys profile`（经
[`nsys_worker.sh`](nsys_worker.sh) 补上本 rank 的 GPU id，见下文“NVLink / DRAM 吞吐曲线”）：

```text
nsys profile --gpu-metrics-devices=<本 rank 的 GPU> \
  --trace=cuda,nvtx --sample=none --cpuctxsw=none --gpu-metrics-frequency=50000 \
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

所有 pass 跑完后，node 0 只执行 [`analyse_results.sh`](analyse_results.sh)：用 `collective_metrics.py`
写 `compare.md`，几秒就结束。nsys 的分析**不在作业里做**，作业结束后单独执行
[`analyse_nsys.sh`](analyse_nsys.sh)（它调用 [`analyse_nsys.py`](analyse_nsys.py)）：归档剩下的报告，对每份
报告跑 `nsys export --type sqlite`，再执行 [`nsys_collective_kernels.sql`](nsys_collective_kernels.sql)
里的查询（`label` 用来核对归档，`kernels`、`overlap` 用来统计，`gpu_metrics` 统计 NVLink / DRAM 吞吐，
见下文）。

为什么拆开：32 卡、默认参数下每个 pass 有 5376 份报告，node 0 串行导出三个 pass 要两个多小时。
如果放在 launcher 里，其他节点早就退出、进入作业里的下一条命令，在 `MASTER_ADDR:<master-port - 1>`
的 RUN_TIME 握手上等 node 0，最终 `could not agree on RUN_TIME` 超时失败。`analyse_nsys.sh` 只用 CPU
和 `nsys`，在任何能访问结果目录的机器上都能跑；Python 版和 C++ 版的结果目录都支持，可以一次传多个
`--output-dir`。

后处理和测量是分开的：`analyse_results.sh` 失败时 launcher 只打印 warning 和重跑命令，退出码仍由
benchmark 本身决定，已经跑完的测量不会因为后处理出错而作废。`analyse_nsys.py` 里某个 pass 出错也
不影响其他 pass 写进 `nsys_kernels.md`，最后以非零退出码报告。已归档的报告和比报告新的 SQLite 导出
会复用，所以中断后重跑会接着做：

```bash
bash examples/gb200/nvls-collective/analyse_results.sh \
  --output-dir results/gb200/nvls-collective/gpu-num-4/run-01
# 作业结束后分析 nsys 报告（可重复 --output-dir，C++ 版的结果目录也可以）：
bash examples/gb200/nvls-collective/analyse_nsys.sh \
  --output-dir results/gb200/nvls-collective/gpu-num-4/run-01
# 只归档某个 pass，或者只分析某个 pass：
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
| 只保留通信 kernel | `StringIds.value` | 名字以 `nccl` 开头且含 `AllGather` / `ReduceScatter` / `AllReduce` / `SendRecv`，分别记为 `ag`/`rs`/`ar`/`p2p`；`_sleep` 延迟 kernel、warmup、trial range 外的 start gate 都被排除 |
| 每次 launch 的 CTA 数 | `CUPTI_ACTIVITY_KIND_KERNEL.gridX` | grid 大小 = NCCL channel 数 = CTA 数，每个 op 列出出现过的值（`nsys_kernels.md` 的 `CTAs (gridX)` 列） |
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

### NVLink / DRAM 吞吐曲线（nsys GPU metrics）

`--nsys` 同时打开 nsys 的 GPU Metrics 采样：硬件计数器按固定频率采样，单 pass、不串行化 kernel、
不重放，所以 AG/RS 并发、两组同时跑这些场景也能看。打开报告后，时间线上每张 GPU 下面多出
`NVLink bytes received / transmitted`、`DRAM Read / Write Bandwidth`、`SMs Active` 等行，和
`ncclSymkDevKernel_*` 对齐显示。

| 选项 | 默认 | 说明 |
| --- | --- | --- |
| `--nsys-freq HZ\|high` | 50000 | GPU metrics 采样频率，50 kHz 即每 20 µs 一个点。数字超出 nsys 的合法区间 10..200000 时钳位到边界（打一条 warning）；`high` 取最高的 200000（每 5 µs 一个点）。频率太高时 nsys 报告的 Diagnostics 页会出现 “Buffer overflow”，要调低 |
| `--nsys-gpu-metrics-set SET` | 不传，由 nsys 选第一个支持该 GPU 的 metric set | 可选值和 GPU 编号在运行目录的 `nsys-gpu-metrics.node<N>.txt` 里（launcher 跑 `--gpu-metrics-devices=help` 和 `--gpu-metrics-set=help` 存下的）。默认 set 没有 NVLink 行时，`analyse_nsys.py` 会打 warning，换一个含 NVLink 的 set |

实现要点：

- **每个 rank 只采自己那张卡。** GPU metrics 按设备采样，一张卡只能有一个 nsys 会话在采，
  同一节点 4 个 worker 都用 `all` 会冲突。GPU id 要等 torchrun 设好 `LOCAL_RANK` 之后才知道，所以
  worker 是 [`nsys_worker.sh`](nsys_worker.sh)：它执行
  `nsys profile --gpu-metrics-devices=<GPU> <launcher 给的 nsys 选项> python ...`，`<GPU>` 是
  `LOCAL_RANK`（设了数字型 `CUDA_VISIBLE_DEVICES` 时取其中第 `LOCAL_RANK` 项）。
- nsys 的 GPU 编号和 `nvidia-smi` 一样按 PCI bus 排序；launcher 在 `--nsys` 时设
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`，让 CUDA 的 `cuda:LOCAL_RANK` 用同一个顺序。核对方法：
  `metadata.json` 里每个 rank 的 `pci_bus_id` 应和 `nsys-gpu-metrics.node<N>.txt` 里同编号 GPU 的 PCI 地址一致。
- **权限和冲突**：GPU Metrics 需要 root 或 `NVreg_RestrictProfilingToAdminUsers=0`；和 DCGM 的 profiling
  模块（如 dcgm-exporter 的 `DCGM_FI_PROF_*`）、ncu 抢同一组计数器，节点上跑着 DCGM profiling 时先
  `dcgmi profile --pause`，跑完再 `--resume`。

**读数的含义**（nsys User Guide 的 GPU Metrics → Available metrics 一节）：

- `NVLink bytes received / transmitted` = `nvlrx__bytes` / `nvltx__bytes` 占峰值的百分比，**含协议开销**，是本卡
  **18 条链路的合计**，nsys 看不到单条链路（单条链路用下面的 `--ncu` 看 `Nvlink_Tables`）。GB200 每方向
  峰值约 900 GB/s，所以 GB/s ≈ % × 9。
- `DRAM Read / Write Bandwidth` = HBM 接口在读 / 写的周期占比。
- 两者都是**整卡**计数器，包含别的 GPU 经 NVSwitch 读写本卡 HBM 的流量：multicast 下 `ld_reduce`
  的归约读、对端 `multimem.st` 扇出的写，都会记在被访问的那张卡上。
- 取值是 0..100 的整数采样。

`analyse_nsys.sh` 对有 GPU metrics 的报告多跑一条 `gpu_metrics` 查询：每个 trial 的 GPU 窗口取该 rank
本 trial **第一个集合通信 kernel 开始到最后一个结束**（`iters` 次背靠背迭代，含其间的空隙），对窗口内
每个 metric 的采样求平均和最大值。输出：

- `<algo>/nsys/gpu_metrics.csv`：每份报告 × trial × metric 一行（set 里的所有 metric）。
- `nsys_gpu_metrics.md`：只列名字含 NVLink / DRAM / C2C 的 metric；每个 case、组、metric 给出组内各
  rank 的平均 / 最小 / 最大、窗口内单个采样的峰值，再对 trial 取中位数。rank 最小和最大差得多，说明
  组内某张卡的链路或 HBM 明显比其他卡忙。`min samples` 太少（< 5）时只看量级。

旧的 `--nsys` 运行（本选项加入前）没有 GPU metric 表，分析照常、不出 `nsys_gpu_metrics.md`。

### 用 Nsight Compute 看 kernel 细节（--ncu）

```bash
bash examples/gb200/nvls-collective/run_nvls_collective_benchmark.sh \
  --algos symm --workloads ag --sizes-mib 1024 --ncu
```

`--ncu` 给每个 worker 套一个 `ncu`（和 `--nsys` 互斥：两者用同一组计数器），每个 rank 写一份
`<algo>/ncu/rank<R>.ncu-rep`，用 `ncu-ui` 打开。

**profile 什么。** 用 range replay（`--replay-mode=range`），只采每个 case **trial0** 这个 NVTX range：

| 选项 | 作用 |
| --- | --- |
| `--replay-mode=range` | 把 range 内的 CUDA API 调用和 kernel 整段录下，每个计数器 pass 在**同一进程内**重放一次（必要时保存/恢复显存）；指标属于整个 range，不是单个 kernel，range 内的 kernel 照常并发 |
| `--nvtx --nvtx-include=trial0/` | range 的边界 = trial0 这个 push/pop range；start gate 在 range 打开前发射，被排除 |

`--ncu` 下 benchmark 每个 case 只跑 **1 个 trial × 1 次 iteration**（忽略 `--trials`/`--iters`），warmup
照常在 range 之外跑一次、不被 profile。每个 case 的 trial0 被重放 pass 数那么多次，所以**限制
`--sizes-mib` 和 section 数**。

不用 app-range：app-range 每个 pass 重启整个 worker 进程，lockstep 不支持它（只支持 kernel/range），
4 卡实测卡在第二个 range（watchdog 480 s 心跳超时）。

**为什么要 lockstep。** symmetric kernel 在 LSA barrier（`multimem.red`）上等所有对端，单个 rank 自己重放
会卡死。所以用 `--communicator=tcp`（server 在 `--master-addr`，端口 `--ncu-port + algo 序号`，默认从
49217 起）加 `--lockstep-kernel-launch --lockstep-nvtx-include=trial0/`，所有 rank 同步进入、同步重放同一个 range。lockstep 要等所有
ncu 实例，因此 `--ncu` 只允许**一个包含全部 rank 的组**：

- `--group-sizes` 必须等于总卡数（不传时自动设成总卡数）。
- 不跑 split，否则闲置的那半边永远不发 kernel，lockstep 会一直等。

**失真控制。** `--clock-control=none`、`--cache-control=none`：ncu 默认锁 base 频率、每个 pass 前刷 L2，
这里都关掉，让 kernel 在和计时运行相同的频率和缓存状态下跑。即使这样，range 的时长包含重放开销和等待
最慢 rank 的时间。时间和带宽以不带 profiler 的
运行和 nsys 为准，ncu 只用来看 kernel 内部。

**默认 section**（`--ncu-sections` 可改，逗号分隔）：

| section | 看什么 |
| --- | --- |
| `SpeedOfLight` | 计算 / 显存吞吐占峰值 |
| `MemoryWorkloadAnalysis*` | 显存图：HBM、L2、peer（经 NVLink 访问对端）的字节数 |
| `Nvlink` / `Nvlink_Tables` / `Nvlink_Topology` | NVLink 收发总量，以及**每条链路**的收发量和对端；用来看 18 条链路是否均匀 |
| `C2CLink` | Grace ↔ Blackwell C2C 链路 |
| `LaunchStats` / `Occupancy` / `SchedulerStats` / `WarpStateStats` | CTA 数、占用率、warp 停顿原因 |
| `SourceCounters` | 按 SASS 指令的执行次数和停顿采样：`multimem.ld_reduce` 的延迟体现在**使用其结果**的那条指令的 `long_scoreboard` 上；`multimem.st` 体现为 `lg_throttle`、fence 上的 `membar` 和退出时的 `drain` |
| `PmSampling` | kernel 运行期间的指标时间线（多 pass 拼出来） |

命令行查看：

```bash
ncu --import <algo>/ncu/rank0.ncu-rep --page details --section Nvlink_Tables
ncu --import <algo>/ncu/rank0.ncu-rep --page raw --csv | grep -i -E 'nvlrx__bytes|nvltx__bytes|dram__bytes'
```

各节以 `ncu --list-sections` 为准，GB200 上可用的 NVLink 指标用
`ncu --query-metrics | grep -i -E "nvl|ctc"` 确认。

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

`collective_metrics.py` 把各次启动的 `results.jsonl` 合并成以下几张表，除表 0 外每格都是 trial 中位数：

| 表 | 回答 | 每行 | 关键列 |
| --- | --- | --- | --- |
| 0 | 每个测试的完成时间 | 实现 × case × 组 | mean / median / min / max ms、由 mean 算的 algbw / busbw、concurrency、**kernel** |
| 1 | AG / RS / AR / AG+RS | 实现 × 组大小 × 消息 | AG busbw、RS busbw、AR busbw、AG+RS 中各自 busbw、pair aggregate、`speedup`、`overlap_fraction`、**AG / RS / AR kernel** |
| 1b | 各实现的绝对吞吐 | 组大小 × 消息 × 负载 | ring、nvls、symm、symm-p2p、symm-tma、copy-engine 的 aggregate busbw，及最高者 |
| 2 | 相对 ring | 组大小 × 消息 × 负载 | 每个实现的 aggregate busbw 与 /ring |
| 2a | TMA / CE 相对 symm | 组大小 × 消息 × 负载（只有 tma/ce 测过的 ag、ag+rs） | symm、symm-tma、copy-engine 的 aggregate busbw 与 /symm |
| 2b | 实际执行的内核 | 实现 × 组 | 实测 nNodes、日志里的 `Algo/proto` 或对称内核名 |
| 2c | 对称 P2P 相对 ring 与 multicast | 组大小 × 消息 × 负载（symm-p2p 测过的） | ring、symm-p2p、symm 的 aggregate busbw，`symm-p2p/ring`、`symm/symm-p2p` |
| 3 | 两个 T/2 组同跑同一负载 | 实现 × 布局 × 消息 × 负载 × 组 | isolated busbw（取自 `single-n<T/2>`）、concurrent busbw、`slowdown = T_concurrent / T_isolated`、concurrency |
| 4 | A 组 AG、B 组 RS 同时跑 | 实现 × 布局 × 消息 | A 的 AG 与 B 的 RS 各自的 isolated / cross busbw、cross ms、slowdown，`A+B cross busbw`、concurrency |
| 5 | late join（仅 `--late-join`） | 实现 × 组大小 × AG 大小 | 放行时刻、AG / RS 的 isolated 与 join 时间、slowdown、overlap 窗口内 algbw、capacity、AG share |
| 6 | CTA 配置（仅 `--cta-configs` 有非 default 条目） | 实现 × case × CTA 配置 | 每个 op 的 CTA 数（P2P 为推算 `~N`）、每次迭代 ms 及相对 default 的比值、makespan / default |

表 1–5 只用 `default` CTA 配置的结果；表 0 列出所有 case（含各 CTA 配置）。

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
      ring-<case>.rank_<RR>.nsys-rep / .sqlite # 仅 --nsys，每个 pass 后归档 / analyse_nsys.sh 导出
    nsys/{index.tsv, kernels.csv}              # 仅 --nsys；归档前的 rank<R>.<K>.nsys-rep 也在这里
    nsys/gpu_metrics.csv                       # 仅 --nsys：每份报告 × trial × GPU metric 的窗口均值
    ncu/rank<R>.ncu-rep                        # 仅 --ncu：每个 rank 一份，trial0 的集合通信 kernel
  nvls/ ...
  symm/ ...                                    # window 注册的对称内存路径
  symm-p2p/ ...                                # 同样的 window，关掉 multicast 的单播对称内核
  symm-tma/ ...                                # AllGather_TmaSTMC + ReduceScatter_LDMC
  copy-engine/ ...                             # CE multicast AllGather + ReduceScatter_LDMC
  compare.md                                   # node 0 合并各 pass 的 results.jsonl
  nsys_kernels.md                              # 仅 --nsys，作业结束后 analyse_nsys.sh 用 SQL 汇总
  nsys_gpu_metrics.md                          # 仅 --nsys：NVLink / DRAM / C2C 吞吐，同上
  nsys-gpu-metrics.node<N>.txt                 # 仅 --nsys：nsys 的 GPU 编号和可选 metric set
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
5. 每个实现是一次独立的 torchrun，结果在 `ring/`、`nvls/`、`symm/`、`symm-p2p/`、`symm-tma/`、`copy-engine/` 各自的 `results.jsonl` 里。
   node 0 的 torchrun 只有在 rank 0 退出后才返回，而 rank 0 退出前已经写完 `results.jsonl`，
   所以随后执行的 `collective_metrics.py` 读到的是完整数据，合并成一份 `compare.md`；合并时会
   检查每个文件里的 `algo` 与目录一致。
6. `--nsys` 时，各节点的 nsys 报告可能比 node 0 晚落盘，`analyse_nsys.py` 会等到报告数达到
   `world_size × case 数` 且文件大小不再变化（最多 300 秒）后再归档；超时就先归档已有的，
   之后的 `analyse_nsys.sh` 会把晚到的补上（按编号 K 归档，与先后顺序无关）。

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

## P2P push / pull 单独 profile（`run_p2p_push_pull_benchmark.sh`）

只测两张卡之间的单向 NCCL P2P，不跑任何 collective。限定在一个 4-GPU 节点（GB200 一个 tray）上执行
（`nvidia-smi -L` 不是 4 张卡就直接退出），但每个 workload 只启动 2 个 rank：rank 0 是 sender，rank 1 是
receiver，对应 `--gpus S,R`（默认 `0,1`）。两个 workload 各起一次 torchrun，因为 read/write 在 P2P
channel 连接时就由 `NCCL_P2P_READ_ENABLE` 决定，之后不会变：

| workload | `NCCL_P2P_READ_ENABLE` | 搬数据的 SM（NVLink initiator） | 另一侧 |
| --- | --- | --- | --- |
| `push` | 0（P2P write） | sender：把数据 store 到 receiver 侧的 staging buffer（sender 的 NVLink TX） | receiver 在本地把 staging buffer 拷进用户 buffer |
| `pull` | 1（P2P read） | receiver：从 sender 侧的 staging buffer load（receiver 的 NVLink RX） | sender 在本地把用户 buffer 拷进自己的 staging buffer |

连接后 sender 的 NCCL INIT 日志必须是 `via P2P/...`，且 `/read` 只出现在 pull 中，否则作业直接失败。

```bash
# GPU0 -> GPU1，1 GiB，push 和 pull，默认带 nsys
bash examples/gb200/nvls-collective/run_p2p_push_pull_benchmark.sh

# 只跑 pull，GPU2 -> GPU3，两个大小
bash examples/gb200/nvls-collective/run_p2p_push_pull_benchmark.sh --workloads pull --gpus 2,3 --sizes-mib 256,1024
```

每个 GPU、每个消息大小各生成一个 nsys 报告，只覆盖计时的 trials，并在本 GPU 上采样 GPU metrics
（NVLink RX/TX、DRAM 等）。跑完后报告会重命名，从文件名就能看出 GPU 的角色：

```
results/gb200/nvls-collective/p2p-push-pull/<yyyymmdd-hhmmss>/
  nsys-reports.md                                   # 所有报告：workload / GPU / 角色 / 是否 initiator
  push/nsys/push.1024MiB.gpu0.sender.initiator.nsys-rep     # push P2P 的 sender
  push/nsys/push.1024MiB.gpu1.receiver.nsys-rep
  pull/nsys/pull.1024MiB.gpu0.sender.nsys-rep
  pull/nsys/pull.1024MiB.gpu1.receiver.initiator.nsys-rep   # pull P2P 的 receiver
  <workload>/results.jsonl, metadata.json, torchrun.log, nccl-raw/, nsys/index.tsv
```

报告里最外层的 NVTX range 也是这个名字，并附带这张卡上 SM 在做什么，例如
`push.1024MiB.gpu0.sender.initiator | NVLink initiator: SMs store into the receiver's staging buffer`；
每个 trial 是一个 `trial<i>` range。`results.jsonl` 里有两侧各自的 trial 时间，algbw 按 receiver 的时间计算。
`--no-nsys` 时只计时。

## 本地检查（无 GPU）

```bash
uv run --no-sync python -m pytest \
  --confcutdir=tests/unit_tests/examples \
  tests/unit_tests/examples/test_gb200_nvls_collective.py \
  tests/unit_tests/examples/test_gb200_nvls_analyse_nsys.py \
  tests/unit_tests/examples/test_gb200_p2p_push_pull.py \
  tests/unit_tests/examples/test_gb200_symm_cta_pause_sweep.py -q
bash -n examples/gb200/nvls-collective/run_nvls_collective_benchmark.sh
bash examples/gb200/nvls-collective/run_nvls_collective_benchmark.sh --dry-run
bash examples/gb200/nvls-collective/run_p2p_push_pull_benchmark.sh --dry-run
uv run --no-sync ruff check examples/gb200/nvls-collective
```

单测覆盖算法族/布局/负载的解析与默认值、六套环境约束（含 symm 必须不设
`NCCL_ALGO`、必须开 `NCCL_WIN_ENABLE`、`NCCL_SYM_KERNEL` 必须与内核名一致；symm-p2p 必须
`NCCL_NVLS_ENABLE=0`；symm-tma/copy-engine 的
`NCCL_SYM_KERNEL`、CE 阈值和 multicast 阈值）、tma/ce 逐 launch 的实现校验、NCCL 日志探针
（algo/kernel/nNodes/MNNVL）、两个半组划分、带宽与重叠公式、报告四张表、nsys GPU metric 查询与汇总、
launcher `--nsys` / `--ncu` 的 dry-run 命令行和选项校验，以及两个入口只用标准库可导入（torch 除外）。
它们不需要 GPU、NCCL 或 NVLink，也**不能**验证 CUDA/NCCL 真实行为、带宽或 NVLS 资源。
本目录的程序必须在上面说的 GB200 环境里编译/运行才能真正出数。
