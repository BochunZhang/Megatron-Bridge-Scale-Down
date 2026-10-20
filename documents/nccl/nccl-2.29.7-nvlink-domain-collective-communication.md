# 基于 nccl-2.29.7 分析 nvlink-domain 的 collective-communication 实现

日期：2026-10-15（2026-10-16 新增 3.7 节）。

源码版本：NVIDIA NCCL **v2.29.7**，本地路径 `Megatron-Scale-Down/nccl-2.29.7`。下文的源码位置均相对 `nccl-2.29.7/src`。例外：3.7 节中标有【2.32.3】的内容（TMA kernel、CE 多播 AllGather）来自 NCCL v2.32.3，源码位置相对 `nccl-2.32.3/src`。

相关文档：

- [NCCL 2.29.7 中基于 NVLS 的 AllGather / ReduceScatter / AllReduce 实现](nccl-2.29.7-nvls-collectives.zh-CN.md)
- [NVLS 资源](nccl-nvls-resources.zh-CN.md)
- [PTX 9.4 multimem 指令](ptx-9.4-multimem.zh-CN.md)

场景约束：通信组的所有 rank 都在**同一个 NVL72**（同一个 MNNVL clique）内。此时 NCCL 把整个 clique 视为一个节点，即 `nNodes == 1`，并且 `localRanks` 等于通信组大小 P（2 ≤ P ≤ 72）。本文不讨论跨 NVLink 域的 RDMA 和 GIN 路径，也不讨论 Tree、PAT 等算法。

本文结论均来自源码阅读，**没有 GPU 实测**。

---

## 1. 实现总览

本文把 NVLink 域内的集合通信实现分为 4 类：

| 类别 | 一句话描述 | 数据由谁搬运 |
|---|---|---|
| **P2P（Ring）** | GPU 之间通过点对点通信（NVLink ld/st 访问对端 FIFO）组成环，逐步转发 | SM |
| **Legacy NVLS kernel** | 基于 multimem PTX 实现，使用**普通内存**，即 NCCL 内部的 NVLS 中转缓冲区 | SM 发出 multimem 指令，NVSwitch 负责复制或归约 |
| **Symmetric NVLS kernel** | 基于**对称内存**（symmetric window）实现，既可以用单播 `st` / `ld`，也可以用多播 `multimem.st` / `multimem.ld_reduce` | SM（多播版本由 NVSwitch 负责复制或归约） |
| **Copy Engine（CE）** | 由 host 提交 memcpy 任务，Copy Engine 负责拷贝，不占用 SM | CE |

### 1.1 实现明细表

表中符号说明：
- P 表示通信组大小。
- 「注册」指用 `ncclCommWindowRegister(..., NCCL_WIN_COLL_SYMMETRIC)` 把缓冲区注册为 symmetric window。
- 「MC」表示多播（multicast）。

| 类别 | 集合操作 | 实现（kernel / 函数） | 算法模式 | 数据平面指令 | 发起者 | 协议 | 缓冲区要求 | 用户缓冲区零拷贝 | 支持的 op / 类型 | NVL72 上的条件或限制 | 关键源码 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **P2P（Ring）** | AllReduce | `RunWorkColl<AllReduce, RING, *>` | Ring：先 RS 后 AG，共 2(P-1) 步 | 单播 ld/st，写对端 FIFO | SM | LL / LL128 / SIMPLE | 无；可选注册 recvbuff（`ncclCommRegister`） | 默认否，经过 FIFO 中转；注册后 SIMPLE 直接写下游 rank 的 recvbuff | 全部 | 无特殊限制；步数随 P 线性增长；fp8 且 P > 8 时基本不会选 Ring | `device/all_reduce.h` |
| | AllGather | `RunWorkColl<AllGather, RING, *>` | Ring，共 P-1 步 | 单播 ld/st | SM | LL / LL128 / SIMPLE | 同 AllReduce | 同 AllReduce | 全部 | 同上 | `device/all_gather.h` |
| | ReduceScatter | `RunWorkColl<ReduceScatter, RING, *>` | Ring，共 P-1 步 | 单播 ld/st，在 SM 内归约 | SM | LL / LL128 / SIMPLE | 无 | 否，kernel 没有开启 Direct，始终经过 FIFO | 全部 | 同上 | `device/reduce_scatter.h` |
| | （可选变体） | `NCCL_P2P_USE_CUDA_MEMCPY=1` | 同 Ring | SM 写本地 `ceDevBuff`，由 proxy 线程调用 `cudaMemcpyAsync` 跨卡拷贝 | SM 加 proxy 驱动的 CE | 只有 SIMPLE 走 CE | 无；用户缓冲区注册会被强制关闭 | 否 | 全部 | SM 占用不减少，每个 step 都需要 host 轮询；依赖 host SHM，**跨 tray 很可能不可用**【推断】 | `transport/p2p.cc:121, 538, 829` |
| **Legacy NVLS** | AllReduce | `RunWorkColl<AllReduce, NVLS, SIMPLE>` | scatter → `ld_reduce` + `st` → gather | `multimem.ld_reduce` + `multimem.st` | SM，NVSwitch 归约和复制 | **只有 SIMPLE** | 无（MNNVL 下不能注册） | **否**：MNNVL 下 `nvlsRegSupport = 0`，必须经过 NVLS 的 UC/MC 中转缓冲区 | sum：int32/64、uint32/64、f16、bf16、f32、f64；min/max：整数、f16、bf16（`ncclNvlsSupported`） | P 不受 32 的限制，但 NVLS head 最多 32 个 | `device/all_reduce.h:389`、`init.cc:1024`、`include/device.h:566` |
| | AllGather | `RunWorkColl<AllGather, NVLS, SIMPLE>` | bcast（`multimem.st`）→ gather | `multimem.st` | SM，NVSwitch 复制 | 只有 SIMPLE | 无 | 否 | 全部 | **要求 P ≤ 32**，整个 72 卡的通信组用不了 | `device/all_gather.h:176`、`enqueue.cc:1914` |
| | ReduceScatter | `RunWorkColl<ReduceScatter, NVLS, SIMPLE>` | scatter → `ld_reduce` | `multimem.ld_reduce` | SM，NVSwitch 归约 | 只有 SIMPLE | 无 | 否 | 同 AllReduce | **要求 P ≤ 32** | `device/reduce_scatter.h`、`enqueue.cc:1914` |
| **Symmetric NVLS** | AllReduce | `AllReduce_AGxLL_R` | one-shot：LL 方式广播给所有 rank，各自在本地归约 | 单播 `st`（LL） | SM | LL 风格 | send/recv 至少注册一个；都不注册时默认回退（`NCCL_SYM_NOWIN_ENABLE=0`） | 否，经过 LL 缓冲区 | sum；f16、bf16、f32、fp8（不支持 f64） | 总线字节数 < 2GB；不能合并多个 work | `device/symmetric/all_reduce.cuh:382` |
| | | `AllReduce_AGxLLMC_R` | 同上，LL 数据写到多播地址 | 对 MC 地址的普通 `st.relaxed.sys`（LL） | SM，NVSwitch 复制 | LL 风格 | 同上 | 否 | 同上 | 同上，并要求 `nvlsSupport` 且 P > 2 | 同上 |
| | | `AllReduce_RSxLD_AGxST` | two-shot：从各对端 `ld` 后在本地归约，再 `st` 到各对端 | 单播 `ld` + `st` | SM | SIMPLE 风格 | send 和 recv **都要注册** | 是 | 同上 | — | `all_reduce.cuh:265` |
| | | `AllReduce_RSxLDMC_AGxSTMC` | two-shot：`ld_reduce` 后接 `multimem.st` | `multimem.ld_reduce` + `multimem.st` | SM，NVSwitch 归约和复制 | SIMPLE 风格 | send 和 recv 都要注册 | 是 | LDMC 支持的类型子集（fp8 需要 sm100） | 每卡最多 `divUp(32, P)` 个 CTA，**P=72 时只有 1 个** | `all_reduce.cuh:350` |
| | AllGather | `AllGather_ST` | direct：`st` 到每个对端 | 单播 `st` | SM | SIMPLE 风格 | recv 要注册 | 是 | 全部 | — | `device/symmetric/all_gather.cuh:167` |
| | | `AllGather_LL` | direct，LL 格式 | 单播 `st`（LL） | SM | LL 风格 | send/recv 至少注册一个；都不注册时默认回退（`NCCL_SYM_NOWIN_ENABLE=0`） | 否 | 全部 | 总线字节数 < 2GB；不能合并多个 work | `all_gather.cuh:296` |
| | | `AllGather_STMC` | direct：一次 `multimem.st` 写到所有 rank | `multimem.st` | SM，NVSwitch 复制 | SIMPLE 风格 | recv 要注册 | 是 | 全部 | P=72 时只有 1 个 CTA | `all_gather.cuh:195`、`primitives.cuh:234` |
| | | `AllGather_LLMC` | direct，LL 数据写到多播地址 | 对 MC 地址的普通 `st.relaxed.sys`（LL） | SM，NVSwitch 复制 | LL 风格 | send/recv 至少注册一个；都不注册时默认回退（`NCCL_SYM_NOWIN_ENABLE=0`） | 否 | 全部 | 同 `AllGather_LL` | `include/nccl_device/impl/ll_a2a__funcs.h:75` |
| | ReduceScatter | `ReduceScatter_LD` | direct：从每个对端 `ld` 后在本地归约 | 单播 `ld` | SM | SIMPLE 风格 | send 要注册 | 是 | sum；f16、bf16、f32、fp8 | — | `device/symmetric/reduce_scatter.cuh:231` |
| | | `ReduceScatter_LL` | LL 格式交换后在本地归约 | 单播 `st`（LL） | SM | LL 风格 | send/recv 至少注册一个；都不注册时默认回退（`NCCL_SYM_NOWIN_ENABLE=0`） | 否 | 同上 | 总线字节数 < 2GB；不能合并多个 work | `reduce_scatter.cuh:395` |
| | | `ReduceScatter_LDMC` | direct：一次 `multimem.ld_reduce` | `multimem.ld_reduce` | SM，NVSwitch 归约 | SIMPLE 风格 | send 要注册 | 是 | LDMC 支持的类型子集 | P=72 时只有 1 个 CTA | `reduce_scatter.cuh:312` |
| **Copy Engine** | AllReduce | — | — | — | — | — | — | — | **不支持**（CE 只能拷贝，不能归约） | — | `ce_coll.cc:80` |
| | AllGather | `ncclCeAllGather` | direct push：每个 rank 直接写给其余 P-1 个 rank，不转发 | `cudaMemcpyBatchAsync`（CUDA 12.8 及以上），同步用 `cuStreamBatchMemOp`；同步 flag 写到多播地址 | **CE**（host 提交） | 无协议概念 | recv 要注册（send 可以不注册） | 是 | 全部 | driver ≥ 12.5；`isAllDirectNvlink`；触发方式：`NCCL_CTA_POLICY_ZERO`，或 Blackwell 上每 rank 的 count > `NCCL_SYM_CE_THRESHOLD`（默认 8M **个元素**）时自动选中 | `ce_coll.cc`、`enqueue.cc:2919, 2966` |
| | ReduceScatter | — | — | — | — | — | — | — | **不支持** | — | `ce_coll.cc:80` |

### 1.2 表注

1. **数据量与 SM 占用**
   - 只有 CE 的 AllGather 完全不占 SM。
   - 其余所有实现，包括 P2P CE 变体，数据都由 SM 发起。
2. **协议**
   - LL / LL128 / SIMPLE 是 Legacy 框架（P2P、Legacy NVLS）里的协议枚举。
   - Symmetric kernel 不使用这个枚举。表中的「LL 风格 / SIMPLE 风格」是按实现方式做的对应：LL 风格是 16B 里放 8B 数据加 flag；SIMPLE 风格是批量读写，用 barrier 同步。
   - 所有 multimem 路径都不使用 LL128。
3. **Legacy NVLS 在 NVL72 上的两个限制**
   - AllGather / ReduceScatter 要求 P ≤ 32（`enqueue.cc:1914`）。72 卡的通信组只能用 NVLS 做 AllReduce。
   - MNNVL 下不能注册缓冲区（`init.cc:1024`），所以数据一定会经过中转缓冲区。
4. **Symmetric kernel 的回退规则**（`scheduler/symmetric_sched.cc:104`）
   - send 和 recv 都注册时，直接使用 symmetric kernel，不回退。
   - 否则只剩 LL 类 kernel 可选。如果 Legacy 代价模型选出的协议不是 LL，就回退到 Legacy kernel。
5. **Symmetric 多播版本的前提**：`nvlsSupport && P > 2`（`sym_kernels.cc` 中的 `hasLsaMultimem`）。
6. **CE 的其他集合操作**：CE 还支持 AlltoAll / Scatter / Gather，不在本文范围内。

---

## 2. 四类实现详解

### 2.1 P2P（Ring）

| collective | kernel | instruction | engine | protocol | buffer |
|:---|:---|:---|:---|:---|:---|
| all-reduce | `RunWorkColl<AllReduce, RING, *>` | 单播 ld/st，写对端 FIFO | SM | LL / LL128 / SIMPLE | 默认经过 FIFO 中转；recvbuff 注册后，SIMPLE 可直接写下游 rank 的 recvbuff |
| all-gather | `RunWorkColl<AllGather, RING, *>` | 单播 ld/st | SM | LL / LL128 / SIMPLE | 同 all-reduce |
| reduce-scatter | `RunWorkColl<ReduceScatter, RING, *>` | 单播 ld/st，在 SM 内归约 | SM | LL / LL128 / SIMPLE | 始终经过 FIFO 中转（kernel 没有开启 Direct） |

**通信模型**

- 传统 P2P 算法由 **SM 发起通信**，每个 rank 只和它在环上的**相邻 rank** 通信：从上游（`ring.prev`）接收，向下游（`ring.next`）发送。一个集合操作要在环上走 P-1 步（AllGather / ReduceScatter）或 2(P-1) 步（AllReduce）。
- 环的顺序由拓扑搜索决定（`graph/search.cc`）。NVL72 内所有 GPU 都通过 NVSwitch 互联、彼此等距，所以环上的"相邻"只是逻辑顺序，不是物理上更近。

**同时有几个 channel 在通信**

- **1 个 channel = 1 个 CTA = 1 个 SM。** kernel 的 grid 大小等于本次使用的 channel 数，每个 CTA 按 `blockIdx.x` 找到自己的 channelId（`device/common.h:355`）。每个 channel 有自己的一条环和一套连接（FIFO），处理数据中属于自己的那一段。
- **communicator 初始化时确定 channel 上限 `comm->nChannels`：**
  1. 拓扑搜索得到环的条数。Blackwell 上每条环的目标带宽从 90 GB/s 开始往下试（`sm100SpeedArrayIntra`，`search.cc:987`），上限是 `MAXCHANNELS/2 = 32`（`init.cc:1093`）。
  2. `ncclTopoDupChannels` 在 `bwIntra ≥ 50` 时把环复制一份（`search.cc:963`）。
  3. `ncclTopoPostset` 再无条件翻倍，上限是 `MAXCHANNELS = 64`（`connect.cc:453`）。
  4. 最后受 `NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS` 和 comm config 的 `minCTAs` / `maxCTAs` 约束（`connect.cc:491`）。
- **每次调用再按消息大小缩减**（`enqueue.cc` 的 `topoGetAlgoInfo`）：
  ```cpp
  nc = comm->nChannels;  nt = maxThreads[RING][proto];
  while (nBytes < nc * nt * threadThreshold && nc >= 2) nc--;      // 先减 channel
  while (nBytes < nc * nt * threadThreshold && nt % 128 == 0) nt /= 2;  // 再减线程
  if (proto == SIMPLE) nt += WARP_SIZE;  // SIMPLE 多一个 warp
  ```

  | 协议 | `nt` 上限 | `threadThreshold`（每线程字节数） | 每个 channel 至少需要的数据量（P=72） |
  |:---|:---|:---|:---|
  | SIMPLE | 512（再加 1 个 warp，共 544） | 64 | 512 × 64 = 32 KB |
  | LL | 512 | 8 × P = 576 | 512 × 576 = 288 KB |
  | LL128 | 640 | 8 | 640 × 8 = 5 KB |

- **NVL72 上的典型值【推断】**：
  - 每张 GPU 的 NVLink 带宽约为 18 × 40.1 ≈ 722 GB/s（`SM100_NVLINK_BW`），拓扑搜索大约能找到 8 条 90 GB/s 的环。
  - 经过 Dup 和 Postset 两次翻倍后，`nChannels` 约为 32，即大消息时 Ring 占用约 32 个 SM。
  - 实际值以 `NCCL_DEBUG=INFO` 初始化日志中的 `coll channels` 为准。
- 用户缓冲区注册后，如果 channel 数在 (16, 24] 区间，会被压到 16（`register/coll_reg.cc:465`）。

**协议选择逻辑（NVL72 域内，`nNodes == 1`）**

1. **候选协议**：Ring 的 LL、LL128、SIMPLE 三种都可用。
   - LL128 默认为"条件启用"（`tuning.cc:480`）。在 NVL72 上它的条件都满足：Blackwell、`NCCL_LL128_C2C=1`、`typeIntra ≤ PATH_NVB`、所有 GPU 的 compute capability 相同。所以 LL128 默认开启。
2. **代价模型**：对每个 (algorithm, protocol) 估算时间，选最小的（`tuning.cc:625`）。
   ```
   time = latency + nBytes / (1000 × bandwidth)     // 单位：us，nBytes 单位为字节，bandwidth 单位为 GB/s
   ```
   - nBytes 的含义（`ncclFuncMaxSendRecvCount`，`include/enqueue.h:37`），都是**单个 rank 上一个缓冲区的大小**，不是所有 rank 的数据量之和：
     - AllReduce：`count × eltSize`，即 sendbuff（= recvbuff）的大小。
     - AllGather：`P × count × eltSize`，即每个 rank 的 recvbuff（输出）大小。
     - ReduceScatter：`P × count × eltSize`，即每个 rank 的 sendbuff（输入）大小。
   - **latency** = `baseLat[proto] + nsteps × nvlinkLat[proto]`，其中 nsteps = 2(P-1)（AR）或 P-1（AG/RS）。

     | 协议 | baseLat (us) | 每步 NVLink 延迟 (us) | AR，P=72 | AG/RS，P=72 |
     |:---|:---|:---|:---|:---|
     | LL | 6.6 | 0.6 | 91.8 | 49.2 |
     | LL128 | 14.0 | 1.9 | 283.8 | 148.9 |
     | SIMPLE | 8.4 | 3.4 | 491.2 | 249.8 |

   - **bandwidth**：设拓扑搜索得到的总带宽为 `busBw = nChannels × bwIntra`，各协议的带宽如下，最后都要乘以 `P / nsteps` 换算成算法带宽：
     - LL：`min(282, 0.5 × busBw)`，其中 282 GB/s 是 Blackwell 单节点的 `llMaxBws`。
     - LL128：`min(0.92 × busBw, nChannels × 40)`。
     - SIMPLE：`busBw`。
3. **结果（示意）【推断】**：假设 `busBw ≈ 720 GB/s`，按上面的公式可以算出：
   - AR：消息小于约 48 MB 时选 LL，约 48 MB 到 870 MB 选 LL128，再往上选 SIMPLE。
   - AG/RS：nBytes 小于约 50 MB 时选 LL，约 50 MB 到 840 MB 选 LL128，再往上选 SIMPLE。这里的 nBytes 是 AG 输出缓冲区（RS 输入缓冲区）的大小。换算成每个 rank 贡献的数据量（`count × eltSize` = nBytes / 72），切换点约为 0.7 MB 和 11.7 MB。

   要注意两点：
   - 这只是 Ring 内部三种协议之间的比较。实际调度时 Ring 还要和 NVLS（AR）、Tree，以及更早决定的 symmetric kernel 和 CE 竞争。
   - P=72 时 Ring 的延迟随步数线性增长（AR SIMPLE 约 490 us），所以 Ring 实际会被选中的消息范围比上面的结果更窄。
4. **特殊规则**：fp8 数据类型且 P > 8 时，Ring 的估算时间会乘以 1024（`enqueue.cc:1924`），以避开多步归约带来的精度损失。所以 NVL72 上 fp8 的 AR / RS 基本不会选 Ring。
5. **手动指定**：可以用 `NCCL_ALGO=Ring` 和 `NCCL_PROTO=LL|LL128|Simple` 强制指定，也可以用 tuner plugin 覆盖。

**数据如何通过 NVLink 到达对端**

一个 channel 上，每对相邻 rank 之间有一条 send→recv 连接，由 `transport/p2p.cc` 建立。

1. **FIFO 在接收端的显存里。** 下游 rank 在自己的 GPU 上分配接收缓冲区（`ncclRecvMem`），里面包含三个协议各自的环形 FIFO，每个 FIFO 有 `NCCL_STEPS = 8` 个槽：

   | 协议 | FIFO 总大小 | 每槽大小 |
   |:---|:---|:---|
   | SIMPLE | 4 MB | 512 KB |
   | LL | 512 KB | 64 KB |
   | LL128 | 约 4.7 MB | 约 600 KB |

2. **上游 rank 把这块显存映射到自己的虚拟地址空间。**
   - 下游 rank 用 cuMem 分配内存，导出成一个可共享的 handle 交给上游。
   - 上游用 `cuMemImportFromShareableHandle` 导入，再用 `cuMemAddressReserve` + `cuMemMap` + `cuMemSetAccess` 映射到本地地址（`p2p.cc:285-300`）。
   - 之后上游 SM 对这个地址执行普通的 `st` 指令，数据就会经过 NVLink、NVSwitch 写进下游 GPU 的显存。
   - Blackwell 默认是**写模式**（`NCCL_P2P_WRITE`）：只有 sm80 才默认读模式（`graph/paths.cc:375`）。
3. **MNNVL 下用的是 fabric handle。**
   - 单机内，cuMem handle 是 POSIX 文件描述符，只能通过 Unix socket 在同一台主机的进程间传递。
   - NVL72 的 72 张 GPU 分布在 18 台主机（compute tray）上，文件描述符无法跨主机传递。所以 MNNVL 会把 handle 类型强制设为 `CU_MEM_HANDLE_TYPE_FABRIC`（`mnnvl.cc:86`）。
   - fabric handle 是一段普通字节，可以经 bootstrap 网络发给另一台主机。只要两张 GPU 在同一个 NVLink fabric 里，对方就能导入并映射。
   - 映射完成后，访问方式和单机完全一样：SM 用 ld/st 直接访问远端显存。
4. **流控**：
   - **SIMPLE**：发送端先轮询 `head`，确认下游已经消费了旧的槽；然后写数据，执行 `__threadfence_system`，再写下游的 `tail`。接收端轮询 `tail`，确认数据到齐后再读。CTA 里专门有一个额外的 warp，用来让 threadfence 和数据拷贝重叠（`prims_simple.h:597`）。
   - **LL / LL128**：flag 和数据写在同一行里（LL 每 16B 有 8B 数据；LL128 每 128B 有 120B 数据）。接收端只要轮询数据行本身，不需要单独的 head/tail 握手，所以延迟更低，但有效带宽打了折扣。
5. **注册用户缓冲区（Direct 模式）**：
   - 前提：用户调用 `ncclCommRegister(recvbuff)`，或者在 CUDA Graph capture 中使用 `NCCL_GRAPH_REGISTER`。
   - 这样 NCCL 会把 recvbuff 通过 IPC 映射给环上的 prev/next 两个 rank（`register/coll_reg.cc:277`）。
   - 之后 SIMPLE 协议在 AR / AG kernel 里会直接把数据写到下游 rank 的 recvbuff 中的目标偏移位置（`DirectWrite`，`prims_simple.h:557`）。这省掉了"写进 FIFO，下游再拷到 recvbuff"中间那一次拷贝。
   - RS kernel 的 Primitives 模板参数是 `Direct=0`，所以 RS 一定经过 FIFO。LL / LL128 也不走 Direct。

**同步与流控（P2P Ring 不需要全局 barrier）**

Ring 在操作开始前**不做全局同步**（没有 barrier 让所有 rank 等齐），同步是逐步嵌在每一次发送或接收里的。假设 P=4，GPU 0 的 kernel 先启动，GPU 1 还没启动，不会出错，GPU 0 最多在自旋等待。

SIMPLE 协议的流控基于两个单调递增的 step 计数：

- `tail`：发送方把一个槽写完后，写入接收方显存里的 `tail`，通知接收方有新数据。
- `head`：接收方消费了一个槽后，写回发送方显存里的 `head`，通知发送方这个槽可以复用。

发送方写一个槽之前，先检查 FIFO 还有没有空位（[prims_simple.h:116](../nccl-2.29.7/src/device/prims_simple.h#L116)）：

```cpp
while (connStepCache + NCCL_STEPS < step + StepPerSlice)
    connStepCache = *head;   // 自旋轮询 head，等接收方释放空间
```

最多可以领先 `NCCL_STEPS = 8` 个槽。接收方消费完一个槽后写 `head`，发送方看到 `head` 前进，才继续写。

LL / LL128 的流控逻辑类似，发送方同样要等接收方释放槽（[prims_ll.h:57](../nccl-2.29.7/src/device/prims_ll.h#L57) 的 `waitSend`）。区别是接收方不等 `tail`，而是轮询数据行里的 flag，flag 值绑定到 step 编号，所以不同操作的 flag 不重叠，不会读到上一次残留的数据。

step 计数在 kernel 结束时保存回 `conn->step`（`prims_simple.h:707`），下一次操作从这个值接着数，所以多次操作之间也不需要额外握手。

**什么时候会有问题**：不是先后启动的问题，而是顺序和参数必须一致。如果某个 rank 永远不调用这次集合操作，或者各 rank 传入的 count 不同，发送方会在 `head` / flag 上永久自旋，形成 hang。

Direct 模式在操作开始时还有一次额外的**指针交换握手**（`setDataPtrs`，`prims_simple.h:744`）：接收方先把自己 recvbuff 的地址写进 `ptrExchange` 槽，发送方轮询该槽直到读到地址，再开始写数据。这保证了接收方的 kernel 已经启动、recvbuff 已经可以接收写入，才会有数据写进去。

**FIFO 与 Direct 模式的选择**

通常情况下走**FIFO 模式**。Direct 模式要同时满足以下两个条件：

1. recvbuff 注册成功（`register/coll_reg.cc:277`）。注册有两个来源：
   - 用户显式调用 `ncclCommRegister(recvbuff)`（`NCCL_LOCAL_REGISTER` 默认为 1）。PyTorch 等框架默认不调用，需要专门开启。
   - 在 CUDA Graph capture 中自动注册（`NCCL_GRAPH_REGISTER` 默认为 1，`enqueue.cc:277`）。
2. 用户缓冲区的底层内存能导出成对端可导入的 cuMem handle。

第二个条件取决于内存的分配方式和 handle 类型：

| 场景 | handle 类型 | 能否走 Direct |
|:---|:---|:---|
| 同一 tray 内，`cudaMalloc` 分配 | legacy cudaIpc | 能，同一主机内有效 |
| 同一 tray 内，cuMem 分配并申请了 POSIX FD handle | POSIX FD | 能 |
| 跨 tray（MNNVL），`cudaMalloc` 分配 | legacy cudaIpc 不能跨 OS | **不能**，退回 FIFO |
| 跨 tray（MNNVL），cuMem 但没申请 FABRIC handle | — | **不能**，导出失败，静默退回 FIFO（`p2p.cc:1034`） |
| 跨 tray（MNNVL），`ncclMemAlloc` 或 cuMem 并申请了 FABRIC handle | FABRIC | **能** |

NCCL 内部的 FIFO 缓冲区是 NCCL 自己分配的，`requestedHandleTypes` 已经包含了当前的 handle 类型（MNNVL 下是 FABRIC），所以 FIFO 模式总是可用。注册失败时 NCCL 不报错，直接退回 FIFO。

**验证方法**：`NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=REG`，注册失败时会打印 `failed to IPC register userbuff ... type FABRIC`（`transport/p2p.cc:1121`）。

只有 AR 和 AG 的 SIMPLE 协议能走 Direct（`DirectWrite`），RS 一定经过 FIFO，LL / LL128 也不走 Direct。

#### 2.1.1 AllReduce

**kernel**：`RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_RING, PROTO>`，内部调用 `runRing`（`device/all_reduce.h:14`）。三种协议共用同一份算法代码，只是 Primitives 的模板参数不同。

**SM 数量**：
- 每张 GPU 用 `nc` 个 CTA（= `nc` 个 SM），见上文的 channel 计算。
- 每个 CTA 的线程数：SIMPLE 最多 544，LL 最多 512，LL128 最多 640。

**数据划分**：
- `ncclCollCbdPart` 把整个缓冲区按 channel 切成连续的段，每个 channel 处理 `channelCount` 个元素。
- 每个 channel 每轮处理 `loopCount = P × chunkCount` 个元素：把这一段再切成 P 个 chunk，环上每个 rank 负责归约其中一个。
- chunk 大小：
  - SIMPLE：`stepSize × ALLREDUCE_CHUNKSTEPS = 512 KB × 4 = 2 MB`，每个 slice 是 1 MB。
  - LL：32 KB 数据。
  - LL128：约 560 KB 数据。
- 最后一轮剩余数据不足时，chunkCount 会缩小。

**每个 CTA 的执行逻辑（伪代码）**，其中 r 是本 rank 在环上的位置 `ring->index`：

```text
prims = Primitives(recvPeer = ring.prev, sendPeer = ring.next, sendbuff, recvbuff)
for elemOffset in [0, channelCount) step P*chunkCount:          # 每轮处理 P 个 chunk
    # ---------- 阶段 1：ReduceScatter，共 P-1 步 ----------
    c = (r - 1) mod P
    prims.directSend(chunk c)                                    # 第 0 步：把自己的 chunk c 发给 next
    for j in 2 .. P-1:                                           # 中间 P-2 步
        c = (r - j) mod P
        prims.directRecvReduceDirectSend(chunk c)                # 收 prev 的部分和，加上本地 sendbuff[c]，发给 next
    c = r
    prims.directRecvReduceCopyDirectSend(chunk c, postOp=true)   # 第 P-1 步：得到 chunk r 的最终结果
                                                                 # 写到本地 recvbuff[c]，同时发给 next
    # ---------- 阶段 2：AllGather，共 P-1 步 ----------
    for j in 1 .. P-2:
        c = (r - j) mod P
        prims.directRecvCopyDirectSend(chunk c)                  # 收到最终结果，写入 recvbuff[c]，再转发给 next
    c = (r + 1) mod P
    prims.directRecv(chunk c)                                    # 最后一步：只收，不再转发
```

**每个原语在 SM 上做什么**（以 SIMPLE 为例）：
1. 线程角色：
   - `tid 0`：等待 recv（轮询 prev 写给本 rank 的 `tail`）。
   - `tid 1`：等待 send（轮询 next 写回来的 `head`）。
   - 最后几个线程：在本步完成后写 `tail` 和 `head`，通知对端。
2. 所有 worker 线程一起执行 `reduceCopy`：
   - 用 16B 向量 `ld` 读取 recv FIFO 的槽和本地 `sendbuff`；
   - 在寄存器里归约；
   - 用 16B 向量 `st` 写到 next 的 FIFO 槽（Direct 模式下直接写 next 的 recvbuff），并按需写本地 recvbuff。
3. 每个 slice 结束时，CTA 内做一次 barrier，然后推进 step。

**数据量**：每个 rank 发出和收到的数据量都是 `2(P-1)/P × N`，算法带宽 = busBw × P / (2(P-1))。

#### 2.1.2 AllGather

**kernel**：`RunWorkColl<ncclFuncAllGather, T, RedOp, NCCL_ALGO_RING, PROTO>`，内部调用 `runRing`（`device/all_gather.h:14`）。
- SIMPLE 使用 `ProtoSimple<ALLGATHER_CHUNKSTEPS/ALLGATHER_SLICESTEPS, ALLGATHER_SLICESTEPS>`，chunk 为 2 MB、slice 为 1 MB。
- `isNetOffload` 分支只在跨节点、每节点 1 个 rank 时启用，NVL72 域内用不到。

**SM 数量**：和 AllReduce 相同，`nc` 个 CTA。

**数据划分**：
- 每个 rank 的输入有 `count` 个元素，按 channel 切成 `partCount`，每个 channel 每次处理 `chunkCount` 个元素。
- 输出缓冲区中，rank k 的数据位于 `recvbuff + k × count`。

**每个 CTA 的执行逻辑（伪代码）**，其中 `ringRanks[0]` 是本 rank，`ringRanks[j]` 是环上往上游数第 j 个 rank：

```text
prims = Primitives(recvPeer = ring.prev, sendPeer = ring.next, sendbuff, recvbuff)
for elemOffset in [0, partCount) step chunkCount:
    off(k) = partOffset + elemOffset + k * count                 # rank k 的数据在输出缓冲区中的偏移
    # 第 0 步：发送自己的数据
    if in-place: prims.directSend(off(self))                     # 数据已经在 recvbuff 中
    else:        prims.directCopySend(sendbuff → recvbuff[off(self)]，同时发给 next)
    # 中间 P-2 步：转发
    for j in 1 .. P-2:
        k = ringRanks[P - j]
        prims.directRecvCopyDirectSend(off(k))                   # 从 prev 收到 rank k 的数据，写入 recvbuff，再发给 next
    # 最后一步：只收
    k = ringRanks[1]
    prims.directRecv(off(k))
```

**SM 上的工作**：
- 没有归约，只有拷贝：`ld` recv FIFO → `st` 本地 recvbuff，加上 `st` next 的 FIFO。
- Direct 模式下，上游已经直接写进了本 rank 的 recvbuff，本 rank 只要再转发给 next 即可，省掉一次本地拷贝。

**数据量**：每个 rank 发出和收到的数据量都是 `(P-1) × count`。

#### 2.1.3 ReduceScatter

**kernel**：`RunWorkColl<ncclFuncReduceScatter, T, RedOp, NCCL_ALGO_RING, PROTO>`，内部调用 `runRing`（`device/reduce_scatter.h:14`）。
- Primitives 的模板参数是 `Direct=0`，所以**不使用 Direct 模式**，数据一定经过 FIFO。
- SIMPLE 的 chunk 和 slice 大小与 AllGather 相同（2 MB / 1 MB）。

**SM 数量**：和 AllReduce 相同，`nc` 个 CTA。

**数据划分**：
- 每个 rank 的输入有 `P × count` 个元素，其中第 k 段 `sendbuff + k × count` 是要归约给 rank k 的部分。
- 输出是 `recvbuff`，共 `count` 个元素。

**每个 CTA 的执行逻辑（伪代码）**：

```text
prims = Primitives(recvPeer = ring.prev, sendPeer = ring.next, sendbuff, recvbuff)   # Direct = 0
for elemOffset in [0, channelCount) step chunkCount:
    off(k) = gridOffset + elemOffset + k * count
    # 第 0 步：从"离自己最远的下游 rank"那一段开始
    k = ringRanks[P - 1]
    prims.send(sendbuff[off(k)])                                 # 把本地这一段写进 next 的 FIFO
    # 中间 P-2 步：收部分和，加上本地数据，继续发
    for j in 2 .. P-1:
        k = ringRanks[P - j]
        prims.recvReduceSend(off(k))                             # FIFO 中的部分和 + sendbuff[off(k)] → next 的 FIFO
    # 最后一步：得到属于本 rank 的最终结果
    k = ringRanks[0]                                             # 即本 rank
    prims.recvReduceCopy(off(k) → recvbuff[gridOffset + elemOffset], postOp=true)
```

**SM 上的工作**：每一步都要 `ld` recv FIFO 和本地 sendbuff，在寄存器里归约，再 `st` 到 next 的 FIFO。最后一步写到本地 recvbuff，`postOp` 会处理 `ncclAvg` 这类需要事后除以 P 的操作。

**数据量**：每个 rank 发出和收到的数据量都是 `(P-1) × count`。和 AllGather 一样，每一步的数据都要先进入 FIFO，下游才能在 SM 里完成归约。

### 2.2 Legacy NVLS kernel

| collective | kernel | instruction | engine | protocol | buffer |
|:---|:---|:---|:---|:---|:---|
| all-reduce | `RunWorkColl<AllReduce, NVLS, SIMPLE>` | `multimem.ld_reduce`（SM scatter→NVSwitch 归约）+ `multimem.st`（NVSwitch bcast）| SM，NVSwitch 归约和复制 | **只有 SIMPLE** | 无；MNNVL 下不能注册（`init.cc:1024`），数据经过 NVLS 中转缓冲区 |
| all-gather | `RunWorkColl<AllGather, NVLS, SIMPLE>` | `multimem.st`（Bcast warp 写到 MC 地址），按 int8 字节拷贝，不受数据类型限制 | SM，NVSwitch 复制 | 只有 SIMPLE | 无；**要求 P ≤ 32**（`enqueue.cc:1914`） |
| reduce-scatter | `RunWorkColl<ReduceScatter, NVLS, SIMPLE>` | `multimem.ld_reduce`（Reduce warp 读 MC 地址，结果直接写 recvbuff） | SM，NVSwitch 归约 | 只有 SIMPLE | 无；**要求 P ≤ 32** |

**通信模型**

Legacy NVLS 的 NVLink Switch（NVLS）算法不走环路，而是利用 NVSwitch 的 multicast/reduce 硬件在**一轮通信**内完成广播或归约。kernel 通过两类虚拟 peer 访问 NVLS 中转缓冲区（`connectNvls`，`graph/connect.cc:242`）：

- `nvls->up[h]`（h = 0..nHeads-1）：指向所有 head 的 FIFO，由 scatter/gather warp 使用。访问的是 **UC 地址**，即只读写本卡的那份物理内存。
- `nvls->down`：指向本 rank 负责的 head（`headRank`），访问的是 **MC 地址**，用 `multimem.ld_reduce` / `multimem.st` 一次作用于所有卡。AllReduce 中由 Reduce warp 同时使用 `recv[0]` 和 `send[0]`；AllGather 中由 Bcast warp 使用 `send[0]`（只有 B 方向）；ReduceScatter 中由 Reduce warp 使用 `recv[0]`（只有 A 方向）。

**head 与 channel**：
- head 由拓扑搜索决定，NVL72 上最多 32 个（`NCCL_MAX_NVLS_ARITY`）。单节点（`nNodes == 1`，即 NVL72 的场景）时 `nvlsGraph->minChannels = maxChannels = min(32, P)`（`graph/search.cc:1065-1073`），也就是每张 GPU 都当 head（P > 32 时为 32 个）。这里 NVLS graph 中的 "channel" 实际上就是 head。
- 运行时的 NVLS channel（CTA）是另一个概念：每个 channel 都连接全部 nHeads 个 head，拓扑完全相同，只是处理的数据段和使用的 FIFO 不同。

**SM 数量**：每个 channel 一个 CTA。CTA 数为 `comm->nvlsChannels`，**与 Ring 的 `nChannels` 无关**，由架构决定（`transport/nvls.cc:149-200`）：SM90 为 16，SM100 单 NVLink 域为 24，SM100 跨域为 32；可被 `NCCL_NVLS_NCHANNELS` 覆盖。单节点时直接使用 `nc = nvlsChannels`（`enqueue.cc:1991-1997`）。每个 CTA 的线程数固定为 `NCCL_MAX_NTHREADS = 640`（`tuning.cc:239`），不随消息大小缩减。

**MNNVL 下不能注册用户缓冲区**：`init.cc:1024` 中 `if (comm->MNNVL) comm->nvlsRegSupport = 0`。NVL72 整体是 MNNVL，所以数据一定经过 NVLS 内部的 `ucBuff`（单播）和 `mcBuff`（多播）中转缓冲区，不会直接写用户内存。

#### 2.2.1 AllReduce

**kernel**：`RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_NVLS, NCCL_PROTO_SIMPLE>`（`device/all_reduce.h:389`）。本节只讨论 `oneNode` 分支（`all_reduce.h:409`），前提如下：

- 所有 rank 在同一个 MNNVL clique 内，所以 `nNodes == 1`、`oneNode == true`。这里的"单节点"指一个 NVLink 域，而不是一个 compute tray（见 3.6）。
- `nvls.out == -1`，即 `hasOut == false`，不经过 CollNet/SHARP。
- MNNVL 下 `nvlsRegSupport = 0`（`init.cc:1024`），所以 `regUsed == 0`，数据一定经过 NVLS 的中转 FIFO。
- 下文以 sum、bf16 为例，P 表示通信组大小，N 表示每个 rank 的输入字节数。

**一句话概括**：每个 rank 把 sendbuff 切成 nHeads 份，写进本卡的中转 FIFO（Scatter）。每个 head 用 `multimem.ld_reduce` 让 NVSwitch 对所有卡上"属于自己的那一份"求和，再用 `multimem.st` 让 NVSwitch 把结果广播回所有卡（Reduce）。最后每个 rank 从本卡 FIFO 把结果拷进 recvbuff（Gather）。本质上是 ReduceScatter + AllGather，规约和复制都在交换机里完成。

**1. 关键参数**

| 参数 | 含义 | 决定因素 | 源码 |
|:---|:---|:---|:---|
| `nHeads` | 参与分工规约的 head 数，每个 head 负责 1/nHeads 的数据 | 拓扑搜索：单 NVLink 域时 `minChannels = maxChannels = min(32, P)`，即**强制取满**，每张 GPU 都当 head（P > 32 时只有 32 个 head）。找不到满足条件的方案时直接关闭 NVLS | `graph/search.cc:1065-1073`、`init.cc:1273` |
| `headRank` | 本 rank 是第几个 head，-1 表示不是 head | `nvlsHeads[]` 中的位置 | `graph/connect.cc:249-260` |
| `nvlsChannels` | 运行时的 CTA 数，每个 CTA 一条独立流水线 | 只与架构和是否跨 NVLink 域有关：SM90 为 16，SM100 单域为 24，SM100 跨域为 32；可被 `NCCL_NVLS_NCHANNELS` 覆盖。**与 P 无关** | `transport/nvls.cc:149-200` |
| `chunkSize` | 每个 head 每个 step 处理的数据量 | 上限 `NCCL_NVLS_CHUNKSIZE`（默认 128KB），再按 `concurrentOps = nHeads` 和**单个 channel** 的字节数逐级减到 64K / 32K / 16K（`calcCollChunking` 按 channel 调用，传入 `nChannels=1`） | `enqueue.cc:711-719, 2153-2167` |
| FIFO | 每条 8 个 slot，slot 跨度 128KB | 每卡共 `nHeads × 2 × nvlsChannels` 条，在通信器初始化时一次性分配 | `transport/nvls.cc:314-340` |

注意区分两个"channel"：拓扑搜索中的 NVLS graph channel 实际上就是 head（数量为 nHeads）；运行时的 NVLS channel 是 CTA（数量为 nvlsChannels）。

**2. 缓冲区与连接**

每个 head h 对应一个虚拟 peer `nRanks + 1 + h`（`connectNvls`，`graph/connect.cc:242`）。每个 channel 中：

- `nvls.up[h] = nRanks + 1 + h`：所有 head，供 Scatter/Gather 使用。
- `nvls.down = nRanks + 1 + headRank`：本 rank 负责的 head，供 Reduce 使用。

NVLS 内存由 `cuMulticastCreate` + `cuMemCreate` + `cuMulticastBindMem` 创建（`nvlsAllocateMem`，`transport/nvls.cc:224`）。每张卡在同一块逻辑缓冲区上有两个虚拟地址：

- **UC（unicast）**：只指向本卡的物理内存，使用普通 `ld`/`st`。
- **MC（multicast）**：同时指向组内所有卡同一偏移处的物理内存，使用 `multimem.*` 访问，由 NVSwitch 负责规约或复制。

**`ucBuff` 和 `mcBuff` 的关系**：每张卡上只有一份物理内存，`ucBuff` 和 `mcBuff` 是映射到它的两个不同虚拟地址。

```c
// transport/nvls.cc:224 nvlsAllocateMem
cuMulticastCreate / cuMemImportFromShareableHandle(&mcHandle)  // multicast 对象：只是一张"成员表"，本身没有存储
cuMulticastAddDevice(mcHandle, myDev)
cuMemCreate(&ucHandle, ucsize, 本卡)                            // 唯一一次物理分配
cuMemMap(ucBuff → ucHandle)                                     // ucBuff：本卡物理内存的 VA
cuMulticastBindMem(mcHandle, 0, ucHandle, 0, ucsize)            // 把本卡物理内存绑定进 multicast 对象
cuMemMap(mcBuff → mcHandle)                                     // mcBuff：multicast 对象的 VA
```

```text
                 GPU0 物理    GPU1 物理    ...   GPU7 物理
offset x:        [ p0 ]       [ p1 ]             [ p7 ]
GPU1: ucBuff + x ──▶ p1                    （只指向本卡）
GPU1: mcBuff + x ──▶ p0, p1, ..., p7       （指向所有成员卡的 offset x，包括本卡）
```

| | ucBuff | mcBuff |
|:---|:---|:---|
| 物理显存 | 占用 `ucsize` | **不占用**，只占一段虚拟地址空间 |
| 指向 | 本卡物理内存 | 所有成员卡上同一偏移的物理内存 |
| 访问方式 | 普通 `ld` / `st` | `multimem.ld_reduce`（读所有份并规约）、`multimem.st`（写所有份）；NCCL 还对 MC 地址使用普通 `st` 来广播 credit |

各卡上的这份内存**不会自动保持一致**，multicast 对象并不是一个跨卡一致的共享变量：

- 通过 UC 写（Scatter 的 `st.global`）只改变本卡那一份。所以 A 区在各卡上的内容各不相同（各自的输入），正好供 `ld_reduce` 求和。
- 通过 MC 写（Reduce 的 `multimem.st`）由交换机写到每张卡的同一偏移。所以 B 区在各卡上的内容全部相同（规约结果），各卡再通过 UC 读取本卡这一份。
- credit 内存（`ucCredit` / `mcCredit`）的结构相同：Scatter/Gather 通过 UC 读写本卡的计数器，head 通过 MC 用 `min` 读所有卡、用 `st` 写所有卡。

所以下文的显存估算只计算 UC 的物理分配；`ucsize` 和 `mcsize` 分别按 `ucgran` / `mcgran` 向上对齐，实际占用可能略大于理论值。

对每个 head h、每个 channel c，有 A（reduce 方向）和 B（bcast 方向）两条 FIFO，各自配有一对 credit 计数器（head/tail）。4 个 connection 的映射如下（`transport/nvls.cc:344-356, 437-471`）：

| connection | 数据 FIFO | credit | 使用者 | min-poll |
|:---|:---|:---|:---|:---|
| `send[1]` | UC A | UC A_cred | 所有 rank 的 Scatter：写本卡 | 否 |
| `recv[0]` | MC A | MC A_cred | head 的 Reduce：`ld_reduce` 读所有卡 | **是** |
| `send[0]` | MC B | MC B_cred | head 的 Reduce：`multimem.st` 写所有卡 | **是** |
| `recv[1]` | UC B | UC B_cred | 所有 rank 的 Gather：读本卡 | 否 |

`ncclConnInfo` 存放的是 FIFO 基址、head/tail 计数器地址、stepSize、flags 等描述信息；真正的数据和计数器在 `ucBuff/mcBuff` 和 `ucCredit/mcCredit` 中。

正常情况下所有 FIFO 都会被用到：本卡 `A[h][c]` 由本卡 Scatter 写入，由 head h 通过 MC 读取；本卡 `B[h][c]` 由 head h 通过 MC 写入，由本卡 Gather 读取。只有在小消息（step 数少于 8，或部分 channel 分不到数据）、尾块（部分 head 的 `nelem = 0`），以及 chunk 小于 128KB（slot 只用了前半部分）时，才会有空闲的 slot。

**3. CTA 内的线程分工**

每个 CTA 有 `NCCL_MAX_NTHREADS = 640` 个线程，即 20 个 warp。warp 数的计算见 `all_reduce.h:394-397`；在 `hasOut == false`、`regUsed == 0`、`P > 6` 时为 **Scatter 8 + Gather 7 + Reduce 5**，与 P = 8 还是 16 无关。

| tid | warp | 组 | worker | 特殊 lane（N_h = nHeads） |
|:---|:---|:---|:---|:---|
| 0–31 | 0 | Scatter | 是 | lane 0..N_h-1：WaitSend[h = lane] |
| 32–223 | 1–6 | Scatter | 是 | — |
| 224–255 | 7 | Scatter | **否** | 最后 N_h 个 lane：PostSend[h]；其余 lane 只参与 barrier |
| 256–287 | 8 | Gather | 是 | lane 0..N_h-1：WaitRecv[h] |
| 288–447 | 9–13 | Gather | 是 | — |
| 448–479 | 14 | Gather | 是 | 最后 N_h 个 lane：PostRecv[h]（同时参与拷贝） |
| 480–511 | 15 | Reduce | 是 | lane 0：WaitRecv(down)；lane 1：WaitSend(down) |
| 512–607 | 16–18 | Reduce | 是 | — |
| 608–639 | 19 | Reduce | **否** | lane 30：PostRecv(down)；lane 31：PostSend(down) |

- 角色分配来自 Primitives 构造函数（`device/prims_simple.h:597, 619-623`）：`MaxSend > 0` 且 `nthreads ≥ 96` 时，最后一个 warp 不做 worker（用于让 fence 与拷贝重叠）。Gather 没有 send，所以 7 个 warp 都是 worker。
- named barrier（`prims_simple.h:62, 69`）：Scatter 用 15/14，Gather 用 13/13，Reduce 用 11/10（分别是全组/仅 worker）。
- 非 head rank（P > 32 时）的 Reduce 组直接空跑结束。

**4. 每个 step 的执行流程**

数据切分：

- `ncclCollCbdPart` 把数据按 channel 切成连续的段。
- 每个 channel 每轮迭代（一个 step）处理 `loopCount = nHeads × chunkSize`，head h 负责其中第 h 个 chunk（`elemOffset + h × chunkSize`）。也就是说，按 head 的切分是**逐 step 交错**进行的。
- 最后一轮使用 `lastChunkSize = alignUp(divUp(rem, nHeads), 16KB/sizeof(T))`。
- 协议为 `ProtoSimple<1, 1>`，每轮每个角色前进 1 个 step，使用 slot `step % 8`。

```text
# ---------- Scatter（所有 rank，warp 0-7）：本卡 HBM → 本卡 HBM ----------
warp0 lane h:   自旋 ld.volatile.global.u64 [UC A_cred[h].head]，直到 head + 8 ≥ step + 1
                # 等待 FIFO[h] 的 slot[step%8] 被 head h 释放；N_h 个 lane 各等一条 FIFO
bar.sync 14     # N_h 个 slot 全部可写后才放行
warp0-6:        for h in 0..N_h-1:
                    8 × { ld.volatile.global.v2.b64 [sendbuff + off + h*chunk]
                          st.global.v2.b64     [UC A_data[h].slot] }      # 每轮一个 4KB hunk
bar.red.or 15
warp7 lane:     fence.acq_rel.sys ; st.relaxed.sys.global.u64 [UC A_cred[h].tail], step

# ---------- Reduce（仅 head k，warp 15-19）：唯一跨卡的组 ----------
warp15 lane0:   自旋 multimem.ld_reduce.acquire.sys.global.min.u64 [MC A_cred[k].tail]
                直到 min ≥ step + 1           # ① 所有 rank 都已写好 A slot（数据可读）
warp15 lane1:   自旋 multimem.ld_reduce.acquire.sys.global.min.u64 [MC B_cred[k].head]
                直到 min + 8 ≥ step + 1       # ② 所有 rank 的 Gather 都已读完 B slot（目标可写）
bar.sync 10
warp15-18:      每个 4KB hunk、每个线程:
                    8 × acc = multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2 [MC A_data[k].slot]
                    8 × multimem.st.global.v4.f32 [MC B_data[k].slot], acc
bar.sync 11
warp19 lane30:  st.relaxed.sys.global.u64 [MC A_cred[k].head], step     # ③ 释放 A slot
warp19 lane31:  fence.acq_rel.sys
                st.relaxed.sys.global.u64 [MC B_cred[k].tail], step     # ④ 通知 B slot 已送达

# ---------- Gather（所有 rank，warp 8-14）：本卡 HBM → 本卡 HBM ----------
warp8 lane h:   自旋 ld.volatile.global.u64 [UC B_cred[h].tail]，直到 tail ≥ step + 1
bar.sync 13
warp8-14:       for h in 0..N_h-1:
                    8 × { ld.volatile.global.v2.b64 [UC B_data[h].slot]
                          st.global.v2.b64     [recvbuff + off + h*chunk] }
bar.red.or 13
warp14 lane:    st.relaxed.sys.global.u64 [UC B_cred[h].head], step
```

要点：

- Scatter 和 Gather 只访问本卡显存，**跨卡流量只出现在 Reduce 组**。
- Reduce 组在 SM 上**不做任何加法**：`reduceCopy` 中 `nSrcs = 1`，SM 只做"load MC → store MC"；P 路求和在 NVSwitch 中完成（bf16/fp16 以 `.acc::f32` 累加，fp8 在 sm100 上以 `.acc::f16` 累加，见 `device/reduce_kernel.h:844-857`）。
- `ld_reduce` 读 A，`st` 写 B，这是两条不同的 FIFO。拿到规约结果只说明 A slot 可以释放；B slot 能否写入需要单独的 ② 号握手。
- 归约结果从寄存器出来后直接 `multimem.st` 到所有卡的 B slot，这一步就是 AllGather 的跨卡传输，不会先写本卡 HBM 再由别的线程发起 AllGather。与 Symmetric 版本的对比见 3.7.5。
- Scatter 最多可以领先 Reduce 8 个 step（FIFO 深度）。刚开始时 `head = 0`，前 8 个 step 不会阻塞。

**5. 同步汇总**

每个 channel、每个 step，在 head k 上有 4 次跨卡同步：

| # | 执行者 | 指令 | 含义 |
|:---|:---|:---|:---|
| ① | warp15 lane0 | `multimem.ld_reduce.acquire.sys.global.min.u64 [MC A_cred.tail]` | 所有 rank 都已 scatter：数据可读 |
| ② | warp15 lane1 | `multimem.ld_reduce.acquire.sys.global.min.u64 [MC B_cred.head]` | 所有 rank 都已 gather：目标可写 |
| ③ | warp19 lane30 | `st.relaxed.sys.global.u64 [MC A_cred.head]` | 释放 A slot 给所有 rank 的 Scatter |
| ④ | warp19 lane31 | `fence.acq_rel.sys` + `st.relaxed.sys.global.u64 [MC B_cred.tail]` | 通知所有 rank 的 Gather：结果已送达 |

- credit 是单调递增的 step 计数器，对 P 张卡取 `min` 就得到最慢 rank 的进度。**一条指令完成一次 P 方"是否都到齐"的检查**（`prims_simple.h:96-100`，由 `NCCL_NVLS_MIN_POLL` 开启）。
- `.acquire` 与生产者一侧的 `fence.acq_rel.sys + st` 构成 release–acquire 关系。之后的大批量数据使用 `.relaxed`，顺序由这一次 acquire 保证。
- ③④ 是对 MC 地址的普通 `st`，由交换机复制到所有 GPU（`postPeer`，`prims_simple.h:174`）。
- **`multimem.st` 结束后需要通知**，就是 ④。必须先 `fence` 再写 tail，否则 tail 可能先于数据到达，Gather 会读到旧数据。④ 是单向通知，不等待回复；Gather 写本卡 `B_cred.head` 作为"回执"，由 head 在 8 个 step 之后复用 slot 时通过 ② 统一收集。
- 本地同步：Scatter 读本卡 `A_cred.head`、写本卡 `A_cred.tail`；Gather 读本卡 `B_cred.tail`、写本卡 `B_cred.head`。

**6. Reduce 组的流水：`ld_reduce` 与 `st` 如何重叠**

`reduceCopyPacks`（`device/common_kernel.h:34`）的参数为 `Unroll = 8`（`ncclCollUnroll`，sm100）、`BytePerPack = 16`，一个 warp 每轮处理一个 4KB hunk。warp w 处理第 w, w+4, w+8, … 个 hunk：

```text
每个线程每轮（同一个 hunk）:
  for u in 0..7: acc[u] = multimem.ld_reduce ... [src + u*512 + lane*16]   # 8 条独立 load，背靠背发射
  for u in 0..7: multimem.st ... [dst + u*512 + lane*16], acc[u]           # 按顺序发射，st[u] 只等 acc[u]
  src/dst 跳到本 warp 的下一个 hunk
```

| 粒度 | 是否重叠 | 原因 |
|:---|:---|:---|
| 同一个 16B 数据的 ld 与 st | 否 | 数据依赖 |
| 同一 hunk 内，后面的 ld 与前面的 st | 是 | load 异步完成，`st[u]` 只等 `acc[u]` |
| 本 hunk 的 st 与下一个 hunk 的 ld | 是 | `multimem.st` 是 posted write，不阻塞后续发射 |
| 本 hunk 的 ld 与下一个 hunk 的 ld（同一 warp） | 否 | 下一批 ld 要等 st7 发出，而 st7 要等 acc7 |
| 不同 warp 之间 | 是 | 4 个 worker warp 同时出发、各自独立；某个 warp 等待 load 时，调度器发射其他 warp 的指令。warp 15–18 通常分布在 4 个不同的 SMSP 上【推断】 |
| 同一 CTA 的相邻 step | 否 | step 结尾有 `bar.sync` + `fence.acq_rel.sys`，需要排空 store |
| 不同 CTA 之间 | 是 | 24 个 channel 完全独立，处在不同相位，可以填补 step 边界的气泡 |

所以，每个 warp 是一条"按顺序发射、异步完成"的小流水线，4 个 warp 是 4 条这样的流水线并行运行，**并不依赖 warp 之间的接力**。单个 warp 在两个 hunk 之间会短暂没有 load 在途，由其他 warp 填补。一个 CTA 最多约有 4 warps × 4KB = 16KB 的 `ld_reduce` 请求在途。

Scatter → Reduce → Gather 三个角色之间，是通过 FIFO 实现的另一层流水：三者分别处理不同的 slot，本地 HBM 拷贝与 NVLink multimem 传输可以并行。

只给 Reduce 分配 5 个 warp 的原因是：head 只处理 1/nHeads 的数据，Reduce 的工作量只有 Scatter/Gather 的 1/nHeads。`P ≤ 6` 时 Reduce 用 7 个 warp，`P > 6` 时只用 5 个，把 warp 让给本地拷贝量更大的 Scatter/Gather。

**7. 流量与负载均衡**

nHeads = P 时，每张 GPU 在每个方向上的 NVLink 流量约为 `(P+1)/P × N`（逻辑流量，不计协议开销）：

| 方向 | 来源 | 流量 |
|:---|:---|:---|
| GPU → Switch | 本卡 A 区被所有 head 的 `ld_reduce` 读取 | N |
| GPU → Switch | 本卡作为 head 发出的 `multimem.st` | N/P |
| Switch → GPU | 本卡作为 head 收到的 `ld_reduce` 结果（已规约） | N/P |
| Switch → GPU | 所有 head 的 `multimem.st` 广播到本卡 B 区 | N |

- Ring AllReduce 每个方向约为 `2(P-1)/P × N`，所以 NVLS 的链路流量约为 Ring 的一半。
- 负载均衡靠 head 分片：每张卡只规约 1/nHeads，而被读取和接收广播的量（各为 N）在所有 rank 上都一样。这就是拓扑搜索要求单 NVLink 域内 head 数取满的原因（源码注释 *"pull evenly from all GPUs"*）。
- P > 32 时只有 32 个 head：占主导的两项（各为 N）仍然对称，head 只多承担约 N/32。
- 本卡 HBM 还要承担 Scatter 和 Gather 的拷贝（读写各 2N），这是 MNNVL 下 `regUsed = 0` 的固有开销。
- NVL72 中每张 GPU 的 NVLink 都连到 NVLink Switch Tray，tray 内的 GPU 之间也没有直连，所以跨 tray 和 tray 内的路径相同。

**8. 示例：P = 8 与 P = 16（GB200 MNNVL，128MiB bf16）**

| | P = 8 | P = 16 |
|:---|:---|:---|
| nHeads / nvlsChannels | 8 / 24 | 16 / 24 |
| 每卡 FIFO 数 / 显存 | 384 条 / 384MB | 768 条 / 768MB |
| chunkSize（每 channel 约 5.33MiB，concurrentOps = nHeads = 8 / 16） | 64KB | 32KB |
| 每 step 的 `loopCount` | 8 × 64KB = 512KB | 16 × 32KB = 512KB |
| 每个 channel 的 step 数 | ≈ 11 | ≈ 11 |
| Scatter/Gather 每 step 的外层 head 循环 | 8 次，每个 head 16 个 hunk | 16 次，每个 head 8 个 hunk（7 个 warp 分 8 个 hunk，负载不均） |
| Reduce 每个线程每 step 的 multimem 指令 | 32 次 ld_reduce + 32 次 st | 16 次 + 16 次 |
| 每条 multimem 指令的扇入/扇出 | 8 | 16 |
| NVLink 每个方向的流量 | 1.125 × 每 step 数据量 | 1.0625 × |

P = 16 时链路流量略低，但每个 step 的有效工作量减半，而握手、fence、barrier 的固定开销不变，min-poll 的扇入也更大。所以在中等消息大小下，P = 16 更难达到峰值带宽【推断】。

**数据量**：每个 rank 读入 N（sendbuff）、写出 N（recvbuff）；NVLink 每个方向约 `(P+1)/P × N`。步数不随 P 增长。

#### 2.2.2 AllGather

**kernel**：`RunWorkColl<ncclFuncAllGather, T, RedOp, NCCL_ALGO_NVLS, NCCL_PROTO_SIMPLE>`（`device/all_gather.h:176`）。本节只讨论 `oneNode && !regUsed` 分支（`all_gather.h:259-288`），前提与 2.2.1 相同：

- 所有 rank 在同一个 MNNVL clique 内，`nNodes == 1`、`oneNode == true`，不走 NVLS + IB SHARP 分支（`all_gather.h:318`）。
- MNNVL 下 `nvlsRegSupport = 0`（`init.cc:1024`），所以 `regUsed == 0`，数据一定经过 NVLS 中转 FIFO。
- **要求 P ≤ 32**：除 AllReduce 外，NVLS 算法要求 `localRanks <= NCCL_MAX_NVLS_ARITY = 32`（`enqueue.cc:1914`）。P > 32 时 Legacy NVLS 不能做 AllGather。
- **不限数据类型**：AllGather 在入队时被改写为字节拷贝，`count *= elementSize`、`datatype = ncclInt8`（`enqueue.cc:2611-2615`），所以 kernel 中 `T = int8_t`、`RedOp = FuncCopy`（`device/generate.py:383, 392`）。NVLS 资格判断对 AllGather 跳过 `ncclNvlsSupported` 的类型检查（`enqueue.cc:421`）。下文的"bf16"只影响字节数。
- P 表示通信组大小，N 表示每个 rank 的输入字节数（sendbuff），recvbuff 为 P·N。

**一句话概括**：每个 rank 用 `multimem.st` 把自己的 sendbuff 写到"自己作为 head 的那条 B FIFO"的 MC 地址，NVSwitch 把它复制到所有卡的同一偏移（Bcast）；每个 rank 再从本卡的 P 条 B FIFO（UC）把 P 份数据拷进 recvbuff 的对应位置（Gather）。只用到 AllReduce 中的 B 方向（广播一半），A 方向的 FIFO 完全不用。

**1. 关键参数**

| 参数 | 含义 | 决定因素 | 源码 |
|:---|:---|:---|:---|
| `nHeads` | head 数 | 单 NVLink 域时 `minChannels = maxChannels = min(32, P)`。P ≤ 32，所以 **nHeads = P，每个 rank 都是 head**，`headRank` 是本 rank 在 `nvlsHeads[]` 中的位置 | `graph/search.cc:1065-1073`、`graph/connect.cc:249-260` |
| `nvlsChannels` | CTA 数 | SM100 单 NVLink 域为 24，与 P 无关 | `transport/nvls.cc:149-200`、`enqueue.cc:1991-1997` |
| 每 channel 的数据量 | 每个 rank 的那一段（count 个元素）被 `ncclCollCbdPart` 按 channel 切成连续的段 `[gridOffset, gridOffset + channelCount)`；所有 P 段使用相同的切分 | 以 cell 为单位均分：`trafficPerByte = P`，`cellSize = alignUp(32KB / P, 16)` | `enqueue.cc:93-100, 651-701`、`include/device.h:330-355` |
| `chunkSize` | 每个 step 每个 rank 广播的数据量（也是每条 B FIFO 每个 slot 中的有效数据量） | 与 AllReduce 走**同一个分支**（`enqueue.cc:2156-2166`；2154 行的分支只在 `NCCL_NVLS_REG_BUFFER` 时生效）：上限 `NCCL_NVLS_CHUNKSIZE = 128KB`，再按 `concurrentOps = nChannels × nHeads` 和 nBytes 逐级降到 64K / 32K / 16K。注意 `calcCollChunking` 是**按 channel 调用的**，传入 `nChannels = 1`、`nBytes = P × 该 channel 每 rank 字节数`（`globalBytesPerElement = elemSize × P`），所以实际 `concurrentOps = nHeads = P` | `enqueue.cc:705-721, 2153-2167`、`include/enqueue.h:37-39` |
| FIFO | 每条 8 个 slot，slot 跨度 128KB（`stepSize = nvlsChunkSize`） | 与 AllReduce 共用同一组 NVLS 缓冲区（每卡 `nHeads × 2 × nvlsChannels` 条），AllGather 只用其中的 B 半边 | `transport/nvls.cc:314-356, 436-471` |

**2. 使用的 connection**

| connection | 数据 FIFO | credit | 使用者 | min-poll |
|:---|:---|:---|:---|:---|
| `send[0]`（peer = `nvls.down`，即本 rank 的 head） | MC B[headRank] | MC B_cred[headRank] | Bcast：`multimem.st` 写所有卡；WaitSend 读 head，PostSend 写 tail | **是**（`transport/nvls.cc:471`） |
| `recv[1]`（peer = `nvls.up[h]`，h = 0..P-1） | UC B[h] | UC B_cred[h] | Gather：读本卡；WaitRecv 读 tail，PostRecv 写 head | 否 |
| `send[1]` / `recv[0]`（A 方向） | — | — | **不使用** | — |

- Bcast 构造 `Primitives(..., recvPeers = NULL, sendPeers = &nvls->down, ..., group = 3 × MaxGroupWidth = 6, connIndexRecv = 0, connIndexSend = 0)`（`all_gather.h:279-281`），所以走 `send[0]`。
- Gather 构造 `Primitives(..., recvPeers = nvls->up, sendPeers = NULL, ..., group = 0, connIndexRecv = 1, connIndexSend = 1)`（`all_gather.h:267-269`），所以走 `recv[1]`。
- `send[0].buffs = mcBuff + ((h·2+1)·nChannels + c)·buffSize`，`recv[1].buffs = ucBuff + ((h·2+1)·nChannels + c)·buffSize`（`transport/nvls.cc:355-356`）：两者是同一块 B 区的 MC / UC 视图。
- **结果位置依赖 head 编号与 rank 编号一致**：Gather 把 `up[h]` 的数据放到 `recvbuff + h × count`（见第 4 步），而 rank r 广播到的是 `B[headRank_r]`。所以只有 `nvlsHeads[h] == h` 时结果才落在 rank h 的位置。单节点搜索时第 c 个 NVLS graph channel 的 head 就是第 c 个 GPU（`graph/search.cc:793-795`），也就是要求拓扑中 GPU 的顺序与 rank 顺序一致【推断：源码中未看到显式检查】。

**3. CTA 内的线程分工**

每个 CTA 640 线程（20 个 warp）。`nThreadsGather = 8 × 32 = 256`，`nThreadsBcast = 640 − 0 − 256 = 384`（`all_gather.h:251-257`），与 P 无关。

| tid | warp | 组 | Fan | worker | 特殊 lane |
|:---|:---|:---|:---|:---|:---|
| 0–31 | 0 | Gather | `FanAsymmetric<32, 0>` | 是 | lane 0..P-1：WaitRecv[h = lane] |
| 32–223 | 1–6 | Gather | 同上 | 是 | — |
| 224–255 | 7 | Gather | 同上 | 是 | 最后 P 个 lane（tid 256-P..255）：PostRecv[h]（同时参与拷贝） |
| 256–287 | 8 | Bcast | `FanAsymmetric<0, 1>` | 是 | lane 0（tid 256）：WaitSend(down) |
| 288–607 | 9–18 | Bcast | 同上 | 是 | — |
| 608–639 | 19 | Bcast | 同上 | **否** | lane 31（tid 639）：PostSend(down)；其余 lane 只参与 barrier |

- 角色分配（`device/prims_simple.h:597, 619-624`）：Gather 的 `MaxSend = 0`，所以 `nworkers = nthreads = 256`，8 个 warp 都拷贝；`nrecv = P`（`up[]` 在 P 之后填 -1，`graph/connect.cc:256-257`），tid < P 为 WaitRecv，`nthreads - nrecv ≤ tid` 为 PostRecv。Bcast 的 `MaxSend = 1` 且 384 ≥ 96，所以 `nworkers = 352`（warp 8–18），最后一个 warp 19 不做 worker；tid 0 为 WaitSend，tid `nthreads-1` 为 PostSend。
- named barrier（`prims_simple.h:59-94`，`bar = 15 - group`）：Gather group 0 → 全组和 worker 都用 **15**（`nworkers == nthreads`）；Bcast group 6 → 全组 **9**、仅 worker **8**。

**4. 每个 step 的执行流程**

数据切分：

- 每个 channel 处理每个 rank 段内的 `[gridOffset, gridOffset + channelCount)`，按 `chunkCount` 前进，每轮一个 step（`all_gather.h:270-286`）。
- 一个 step 中：每个 rank 广播自己的 `chunkCount` 字节；每个 rank 的 Gather 收齐 P 个 chunk，写入 recvbuff 的 P 个位置，即 **每 step 写出 P × chunk**。
- 协议为 `ProtoSimple<1, 1>`（Bcast 为 `ProtoSimple<1, 1, COLL_UNROLL, 0, 1>`，`MultimemDsts = 1`），每个 step 使用 slot `step % 8`。

以 rank r（headRank = r）为例：

```text
# ---------- Bcast（rank r，warp 8-19）：本卡 HBM → NVSwitch → 所有卡的 B[r] ----------
warp8 lane0:    自旋 multimem.ld_reduce.acquire.sys.global.min.u64 [MC B_cred[r].head]
                直到 min + 8 ≥ step + 1        # ① 所有 P 张卡的 Gather 都已读完 B[r] 的这个 slot
                dsts[0] = MC B[r].slot
bar.sync 8      # 352 线程（warp 8-18）
warp8-18:       每个 4KB hunk、每个线程:
                    8 × ld.volatile.global.v2.b64 [sendbuff + off]            # 本卡 HBM
                    8 × multimem.st.global.v4.f32 [MC B[r].slot]              # 一次写入 P 张卡
bar.sync 9      # 384 线程（含 warp 19）
warp19 lane31:  fence.acq_rel.sys
                st.relaxed.sys.global.u64 [MC B_cred[r].tail], step           # ② 通知所有卡：B[r] slot 已送达

# ---------- Gather（每个 rank，warp 0-7）：本卡 HBM → 本卡 HBM ----------
warp0 lane h:   自旋 ld.volatile.global.u64 [UC B_cred[h].tail]，直到 tail ≥ step + 1
                srcs[h] = UC B[h].slot        # P 个 lane 各等一条 FIFO（等 head h 的 ②）
bar.sync 15     # 256 线程
warp0-7:        for h in 0..P-1:
                    8 × { ld.volatile.global.v2.b64 [UC B[h].slot]
                          st.global.v2.b64      [recvbuff + h*count + off] }
bar.red.or 15
warp7 末 P lane: st.relaxed.sys.global.u64 [UC B_cred[h].head], step          # 本卡回执，不加 fence
```

要点：

- **跨卡的只有 Bcast 组**：`multimem.st` 写数据、①读 credit、②写 credit。Gather 只访问本卡显存（UC B 和 recvbuff）。
- Bcast 在 SM 上只做"本卡 load → MC store"；复制由 NVSwitch 完成，本卡（rank r）自己的 B[r] 也通过这条路径收到数据，Gather 再把它拷进 `recvbuff + r × count`。即使是 in-place（`sendbuff == recvbuff + rank × count`），本卡这一段也会被重新写一遍（值相同）。
- Gather 的 `skip = -1`、`shift = 0`（`all_gather.h:273`）：P 个 head 按 0..P-1 顺序处理，不跳过自己（`prims_simple.h:468-477`）。
- 拷贝的 pack 大小：`T = int8_t`、`MultimemSrcs = 0`，所以 `BigPackSize = 16`；指针 16B 对齐时走 16B pack + `Unroll = 8`（`common_kernel.h:222-252`）。不对齐或尾部不足 16B 时退回 1 字节 pack，此时 `multimem_st_global<1>` 实际是**对 MC 地址的普通 `st.global.b8`**（`device/op128.h:383-385`）【推断：依赖普通 st 写 MC 地址也会被交换机复制，与 credit 广播同一机制】。

**5. 同步汇总**

每个 channel、每个 step，rank r（作为 head r）上有 **2 次跨卡同步**：

| # | 执行者 | 指令 | 作用的 credit | 含义 |
|:---|:---|:---|:---|:---|
| ① | Bcast warp8 lane0（tid 256） | `multimem.ld_reduce.acquire.sys.global.min.u64` | **MC** `B_cred[r].head`（读 P 张卡并取 min） | 所有 rank 的 Gather 都已释放 B[r] 的这个 slot：目标可写 |
| ② | Bcast warp19 lane31（tid 639） | `fence.acq_rel.sys` + `st.relaxed.sys.global.u64` | **MC** `B_cred[r].tail`（写到 P 张卡） | 通知所有 rank 的 Gather：B[r] slot 的数据已送达 |

本地同步（只读写本卡的 UC credit）：

| 执行者 | 指令 | credit | 含义 |
|:---|:---|:---|:---|
| Gather warp0 lane h | `ld.volatile.global.u64` 自旋 | UC `B_cred[h].tail` | 等待 head h 的 ② 到达本卡 |
| Gather warp7 末 P lane | `st.relaxed.sys.global.u64`（无 fence） | UC `B_cred[h].head` | 回执：本卡已读完 B[h] slot；head h 在 8 个 step 后复用 slot 时通过 ① 统一收集 |

- **`multimem.st` 之后需要 fence + 通知**，就是 ②：`postPeer` 中 `Send && RolePostSend && dataStored` 时先 `fence_acq_rel_sys()` 再 `st_relaxed_sys_global(tail)`（`prims_simple.h:174-182`）。没有 fence 的话，tail 可能比数据先到达某张卡，Gather 会用 `ld.volatile` 读到旧数据。
- Gather 的 PostRecv 是纯 Recv，不加 fence（`prims_simple.h:177`），只需保证读在写 head 之前完成：读出的值已在 `bar.red.or` 之前被 `st.global` 使用【推断：WAR 顺序由数据依赖 + barrier 保证】。
- **min-poll 只在 `send[0]` 上生效**：`send[0].conn.flags |= NCCL_NVLS_MIN_POLL`（`transport/nvls.cc:471`），Bcast 的 WaitSend 在 `loadSendConn` 中据此置 `NvlsMinPolling`（`prims_simple.h:550-554`），`loadStepValue` 改用 `multimem.ld_reduce.acquire.sys.global.min.u64`（`prims_simple.h:96-107`）。Gather 用的 `recv[1]` 没有这个标志，WaitRecv 用 `ld.volatile.global.u64` 读本卡。构造函数里 WaitSend 还会先做一次 min-poll 作为初值（`prims_simple.h:554`）。
- 与 AllReduce 相比少了 A 方向的两次握手（AllReduce 中的 ①③）：AllGather 不需要"所有 rank 都已 scatter"，因为数据源就是本卡 sendbuff。
- step 计数器在 kernel 结束时由 PostSend/PostRecv 线程写回 `conn->step`（`prims_simple.h:707`），下一次使用同一 connection 的集合通信（AllGather 或 AllReduce）从该值继续。

**6. 流水与重叠**

- Bcast 与 Gather 通过 8 深的 B FIFO 解耦：rank r 的 Bcast 最多领先"P 张卡中最慢的那个 Gather"8 个 step（①）；rank r 的 Gather 在 step s 需要等齐所有 P 个 head 的 step s（P 个 WaitRecv lane）。所以整个组按最慢的 rank 前进，可以容忍最多 8 个 step 的相位差。128MiB 时每个 channel 只有约 11 个 step（见第 8 小节），FIFO 深度覆盖了大部分传输。
- 每个 step 中，Bcast 只处理 1 个 chunk，Gather 要拷 P 个 chunk。Bcast 有 11 个 worker warp，Gather 只有 8 个，**SM 上的主要工作量在 Gather 的本地拷贝**【推断】。
- Bcast 组内 `ld.volatile` 与 `multimem.st` 的重叠方式与 2.2.1 第 6 小节的 Reduce 组相同（`reduceCopyPacks`，`Unroll = 8`，一个 warp 每轮 4KB hunk）。区别是 load 来自本卡 HBM，延迟比 `multimem.ld_reduce` 短。
- 24 个 channel 相互独立，可以填补 step 边界（`bar.sync` + `fence.acq_rel.sys`）的气泡。

**7. 流量与负载均衡**

每张 GPU 的 NVLink 逻辑流量（不计协议开销和 credit）：

| 方向 | 来源 | 流量 |
|:---|:---|:---|
| GPU → Switch | 本卡 Bcast 的 `multimem.st`（只发一份） | N |
| Switch → GPU | 所有 P 个 head 的广播写入本卡 B 区（含自己那份） | P·N |

- Ring AllGather 每个方向约为 `(P-1)·N`。NVLS 的发送方向降为 N，但接收方向是 `P·N`，比 Ring 多了自己那一份【推断：按 multicast 语义本卡也是成员，自己的数据会经交换机回到本卡】。接收方向是瓶颈，所以按链路字节数算，NVLS AllGather 并不比 Ring 少；它的优势在于没有 P-1 步逐跳转发，以及 SM 不用参与转发。
- 本卡 HBM：读 sendbuff N，NVLink 写入 B 区 P·N，Gather 读 B 区 P·N、写 recvbuff P·N，合计约 `3P·N + N`。这是 MNNVL 下 `regUsed = 0` 的固有开销（注册路径下 Bcast 用 `directSend` 直接写到已注册的 recvbuff，Gather 只用于同步，见 `all_gather.h:289-316`【推断：写的是 recvbuff 的 MC 映射】）。
- 负载均衡：nHeads = P，每个 rank 恰好是自己数据的 head，Bcast 与 Gather 的工作量在所有 rank 上完全对称。

**8. 示例：P = 8 与 P = 16（GB200 MNNVL，recvbuff 共 128MiB，bf16）**

channel 切分：`trafficBytes = P·N = 128MiB`，24 个 channel 中前 23 个各 171 个 cell、最后一个 163 个 cell（`enqueue.cc:653-687`）；每个 cell 的流量为 `cellSize × P = 32KB`。

| | P = 8 | P = 16 |
|:---|:---|:---|
| N（每 rank sendbuff） | 16MiB | 8MiB |
| nHeads / nvlsChannels | 8 / 24 | 16 / 24 |
| cellSize | 4KB | 2KB |
| 每 channel 每 rank 的数据量 | 684KB | 342KB |
| chunkSize（`calcCollChunking` 的 nBytes ≈ 5.34MB，concurrentOps = 8 / 16） | 64KB | 32KB |
| 每 step 写入 recvbuff 的数据量（P × chunk） | 512KB | 512KB |
| 每个 channel 的 step 数 | 11 | 11 |
| Bcast 每 step（11 个 worker warp） | 64KB = 16 个 hunk，5 个 warp 各 2 个、6 个各 1 个 | 32KB = 8 个 hunk，3 个 warp 空闲 |
| Gather 每 step（8 个 warp） | 8 个 head × 16 个 hunk，每个 warp 每个 head 2 个 | 16 个 head × 8 个 hunk，每个 warp 每个 head 1 个 |
| WaitRecv / PostRecv lane 数 | 8 / 8 | 16 / 16 |
| 每条 `multimem.st` 的扇出 / ① 的扇入 | 8 / 8 | 16 / 16 |
| NVLink 接收 / 发送（每卡） | 128MiB / 16MiB | 128MiB / 8MiB |
| Ring 对照（每个方向 `(P-1)·N`） | 112MiB | 120MiB |

总 recvbuff 相同时，P = 16 的每 step 有效工作量与 P = 8 相同（512KB），但 Bcast 的 chunk 减半、Gather 每 step 要轮询和处理 16 条 FIFO，`multimem.st` 扇出和 ① 的扇入也更大【推断：每 step 固定开销更高】。

**数据量**：每个 rank 读入 N（sendbuff）、写出 P·N（recvbuff）；NVLink 发送 N、接收 P·N。step 数不随 P 增长。

#### 2.2.3 ReduceScatter

**kernel**：`RunWorkColl<ncclFuncReduceScatter, T, RedOp, NCCL_ALGO_NVLS, NCCL_PROTO_SIMPLE>`（`device/reduce_scatter.h:145`）。本节只讨论 `oneNode && !regUsed` 分支（`reduce_scatter.h:228-257`），前提与 2.2.1 相同：

- 所有 rank 在同一个 MNNVL clique 内，`nNodes == 1`、`oneNode == true`，`nThreadsNetRecv = 0`，不走 multi-node 分支（`reduce_scatter.h:291`）。
- MNNVL 下 `nvlsRegSupport = 0`（`init.cc:1024`），所以 `regUsed == 0`。
- **要求 P ≤ 32**（`enqueue.cc:1914`，同 AllGather）。
- **支持的类型 / 运算**（`ncclNvlsSupported`，`include/device.h:566-581`；`enqueue.cc:421`）：sum 支持 int32/uint32/int64/uint64/f16/bf16/f32/f64；min/max 支持 int32/uint32/int64/uint64/f16/bf16。不支持 fp8、不支持 avg（`ncclDevPreMulSum` / `ncclDevSumPostDiv`）、不支持 prod。
- 下文以 sum、bf16 为例。P 表示通信组大小，N 表示每个 rank 的输出字节数（recvbuff），sendbuff 为 P·N。

**一句话概括**：每个 rank 把 sendbuff 的 P 段分别拷进本卡 P 条 A FIFO（UC，Scatter）；head k 用 `multimem.ld_reduce` 读所有卡上 A[k] 的同一个 slot，由 NVSwitch 求和，再用普通 `st.global` 直接写进本卡 recvbuff（Reduce）。只用到 AllReduce 中的 A 方向（规约一半），B 方向的 FIFO 完全不用，也**没有 `multimem.st`**。

**1. 关键参数**

| 参数 | 含义 | 决定因素 | 源码 |
|:---|:---|:---|:---|
| `nHeads` | head 数 | 单 NVLink 域且 P ≤ 32 时 nHeads = P，每个 rank 都是 head，负责规约属于自己的那一段 | `graph/search.cc:1065-1073`、`graph/connect.cc:249-260` |
| `nvlsChannels` | CTA 数 | SM100 单 NVLink 域为 24，与 P 无关 | `transport/nvls.cc:149-200`、`enqueue.cc:1991-1997` |
| 每 channel 的数据量 | 每个输出段（count 个元素）被 `ncclCollCbdPart` 按 channel 切成连续的段 `[gridOffset, gridOffset + channelCount)`；sendbuff 的 P 段使用相同切分 | 以 cell 为单位均分：`trafficPerByte = P`，`cellSize = alignUp(32KB / P, 16)` | `enqueue.cc:93-100, 651-701`、`include/device.h:330-355` |
| `chunkSize` | 每个 step、每个 head 规约的数据量（也是每条 A FIFO 每个 slot 中的有效数据量） | 与 AllReduce / AllGather 走**同一个分支**（`enqueue.cc:2156-2166`）：上限 128KB，按 `concurrentOps` 和 nBytes 降到 64K / 32K / 16K。`calcCollChunking` 按 channel 调用，`nChannels = 1`、`nBytes = P × 该 channel 每段字节数`，所以 `concurrentOps = nHeads = P` | `enqueue.cc:705-721, 2153-2167`、`include/enqueue.h:37-39` |
| FIFO | 每条 8 个 slot，slot 跨度 128KB | 与 AllReduce 共用同一组 NVLS 缓冲区，ReduceScatter 只用其中的 A 半边 | `transport/nvls.cc:314-356, 436-471` |

**2. 使用的 connection**

| connection | 数据 FIFO | credit | 使用者 | min-poll |
|:---|:---|:---|:---|:---|
| `send[1]`（peer = `nvls.up[h]`，h = 0..P-1） | UC A[h] | UC A_cred[h] | Scatter：写本卡；WaitSend 读 head，PostSend 写 tail | 否 |
| `recv[0]`（peer = `nvls.down`，即本 rank 的 head） | MC A[headRank] | MC A_cred[headRank] | Reduce：`multimem.ld_reduce` 读所有卡；WaitRecv 读 tail，PostRecv 写 head | **是**（`transport/nvls.cc:456`） |
| `send[0]` / `recv[1]`（B 方向） | — | — | **不使用** | — |

- Scatter 构造 `Primitives(..., recvPeers = NULL, sendPeers = nvls->up, sendbuff, NULL, ..., group = 0, connIndexRecv = 1, connIndexSend = 1)`（`reduce_scatter.h:237-239`），所以走 `send[1]`。
- Reduce 构造 `Primitives(..., recvPeers = &nvls->down, sendPeers = NULL, NULL, recvbuff, ..., group = 3 × MaxGroupWidth = 6, connIndexRecv = 0, connIndexSend = 0)`（`reduce_scatter.h:249-251`），所以走 `recv[0]`。
- `send[1].buffs = ucBuff + (h·2·nChannels + c)·buffSize`，`recv[0].buffs = mcBuff + (h·2·nChannels + c)·buffSize`（`transport/nvls.cc:351-352`）：同一块 A 区的 UC / MC 视图。
- 与 AllGather 一样，Scatter 把 `sendbuff + h × count` 送进 A[h]，head k 把 A[k] 的规约结果写进**自己的** recvbuff，所以要求 head k 就是 rank k【推断：依赖拓扑中 GPU 顺序与 rank 顺序一致，源码中未看到显式检查】。

**3. CTA 内的线程分工**

每个 CTA 640 线程（20 个 warp）。`nThreadsScatter = 8 × 32 = 256`，`nThreadsReduce = 640 − 0 − 256 = 384`（`reduce_scatter.h:221-226`），与 P 无关。注意 `reduce_scatter.h:218-220` 的注释（"7 or 5 warps to reduce"）已经过时，实际是 Scatter 8 个 warp、Reduce 12 个 warp。

| tid | warp | 组 | Fan | worker | 特殊 lane |
|:---|:---|:---|:---|:---|:---|
| 0–31 | 0 | Scatter | `FanAsymmetric<0, 32>` | 是 | lane 0..P-1：WaitSend[h = lane] |
| 32–223 | 1–6 | Scatter | 同上 | 是 | — |
| 224–255 | 7 | Scatter | 同上 | **否** | 最后 P 个 lane（tid 256-P..255）：PostSend[h]；其余 lane 只参与 barrier |
| 256–287 | 8 | Reduce | `FanAsymmetric<1, 0>` | 是 | lane 0（tid 256）：WaitRecv(down) |
| 288–607 | 9–18 | Reduce | 同上 | 是 | — |
| 608–639 | 19 | Reduce | 同上 | 是 | lane 31（tid 639）：PostRecv(down)（同时参与规约） |

- 角色分配（`device/prims_simple.h:597, 619-624`）：Scatter 的 `MaxSend = 32 > 0` 且 256 ≥ 96，所以 `nworkers = 224`，warp 7 不做 worker；`nrecv = 0`、`nsend = P`，tid < P 为 WaitSend，`nthreads - nsend ≤ tid` 为 PostSend。Reduce 的 `MaxSend = 0`，所以 12 个 warp 都是 worker；tid 0 为 WaitRecv，tid `nthreads-1` 为 PostRecv。
- named barrier（`prims_simple.h:59-94`）：Scatter group 0 → 全组 **15**、仅 worker **14**；Reduce group 6 → 全组和 worker 都用 **9**。

**4. 每个 step 的执行流程**

数据切分：

- 每个 channel 处理每个输出段内的 `[gridOffset, gridOffset + channelCount)`，按 `chunkCount` 前进，每轮一个 step（`reduce_scatter.h:240-256`）。
- 一个 step 中：每个 rank 的 Scatter 把 P 段各 `chunkCount` 个元素写进 P 条 A FIFO，即 **每 step 读 P × chunk**；每个 head 规约 1 个 chunk，写进 recvbuff。
- 协议：Scatter 为 `ProtoSimple<1, 1>`，Reduce 为 `ProtoSimple<1, 1, COLL_UNROLL, 1, 0>`（`MultimemSrcs = 1`），每个 step 使用 slot `step % 8`。

以 rank r（headRank = r）为例：

```text
# ---------- Scatter（每个 rank，warp 0-7）：本卡 HBM → 本卡 HBM ----------
warp0 lane h:   自旋 ld.volatile.global.u64 [UC A_cred[h].head]，直到 head + 8 ≥ step + 1
                dsts[h] = UC A[h].slot        # 等待 head h 释放 A[h] 的这个 slot（head h 的 ② 写到本卡）
bar.sync 14     # 224 线程（warp 0-6）
warp0-6:        for h in 0..P-1:
                    8 × { ld.volatile.global.v2.b64 [sendbuff + h*count + off]
                          st.global.v2.b64      [UC A[h].slot] }
bar.red.or 15   # 256 线程（含 warp 7）
warp7 末 P lane: fence.acq_rel.sys
                st.relaxed.sys.global.u64 [UC A_cred[h].tail], step           # 本卡 tail，由 head h 的 ① 跨卡读取

# ---------- Reduce（rank r，warp 8-19）：唯一跨卡的组 ----------
warp8 lane0:    自旋 multimem.ld_reduce.acquire.sys.global.min.u64 [MC A_cred[r].tail]
                直到 min ≥ step + 1           # ① 所有 P 张卡都已写好 A[r] 的这个 slot
                srcs[0] = MC A[r].slot
bar.sync 9      # 384 线程
warp8-19:       每个 4KB hunk、每个线程:
                    8 × acc = multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2 [MC A[r].slot]
                    8 × st.global.v2.b64 [recvbuff + off], acc                # 直接写本卡 recvbuff
bar.sync 9
warp19 lane31:  st.relaxed.sys.global.u64 [MC A_cred[r].head], step           # ② 释放 A[r] slot 给所有卡的 Scatter（无 fence）
```

要点：

- **跨卡的只有 Reduce 组**：`multimem.ld_reduce` 读数据、①读 credit、②写 credit。Scatter 只访问本卡显存。
- Reduce 组在 SM 上**不做加法**：`reduceCopy` 的 `nSrcs = 1`、`MultimemSrcs = 1`（`prims_simple.h:279-285`、`common_kernel.h:90-92`），P 路求和在 NVSwitch 中完成（bf16 以 `.acc::f32` 累加，`device/reduce_kernel.h:843-847, 898, 1009`）。`postOp = false`（`prims.recv(offset, nelem)`，`reduce_scatter.h:255`）。
- 与 AllReduce 的差别：规约结果不再 `multimem.st` 回所有卡，而是由 head 用普通 `st.global` 写进本卡 recvbuff。所以没有 B 方向，也没有"结果已送达"的通知。
- Scatter 的 `skip = -1`、`shift = 0`（`reduce_scatter.h:243`）：包括自己那一段在内的 P 段都送进 A FIFO（`prims_simple.h:448-460`），自己那一段也由交换机和其他卡的数据一起规约。
- Scatter 读 sendbuff 时 `PreOpSrcs = 1`（`prims_simple.h:441`），但 sum/min/max 的 preOp 都是恒等操作。

**5. 同步汇总**

每个 channel、每个 step，rank r（作为 head r）上有 **2 次跨卡同步**：

| # | 执行者 | 指令 | 作用的 credit | 含义 |
|:---|:---|:---|:---|:---|
| ① | Reduce warp8 lane0（tid 256） | `multimem.ld_reduce.acquire.sys.global.min.u64` | **MC** `A_cred[r].tail`（读 P 张卡并取 min） | 所有 rank 的 Scatter 都已写好 A[r] slot：数据可读 |
| ② | Reduce warp19 lane31（tid 639） | `st.relaxed.sys.global.u64`（**无 fence**） | **MC** `A_cred[r].head`（写到 P 张卡） | 释放 A[r] slot 给所有 rank 的 Scatter |

本地同步（只读写本卡的 UC credit）：

| 执行者 | 指令 | credit | 含义 |
|:---|:---|:---|:---|
| Scatter warp0 lane h | `ld.volatile.global.u64` 自旋 | UC `A_cred[h].head` | 等待 head h 的 ② 到达本卡 |
| Scatter warp7 末 P lane | `fence.acq_rel.sys` + `st.relaxed.sys.global.u64` | UC `A_cred[h].tail` | 本卡 A[h] slot 已写好；由 head h 的 ① 跨卡收集 |

- **生产者一侧的 fence 在 Scatter 上**：Scatter 用 `st.global` 写本卡 A slot，PostSend 先 `fence.acq_rel.sys` 再写 tail（`ScatterGatherOp` 中 `fenceNeeded` 经 `barrierAny` 汇总后传给 `postPeer`，`prims_simple.h:456-458, 480-481, 177-179`）。它与 head 的 ①（`.acquire`）构成 release–acquire 关系，之后的 `multimem.ld_reduce.relaxed` 大批量读取由这一次 acquire 保证顺序。每个 PostSend lane 各自执行一次 fence，所以 warp 7 每 step 有 P 个 lane 同时做 fence。
- **② 不需要 fence**：PostRecv 是纯 Recv，`postPeer` 只在 `Send && RolePostSend` 时 fence（`prims_simple.h:177`）。要保证的只是"读完再释放"，而 `ld_reduce` 的结果已在 `bar.sync 9` 之前被 `st.global` 使用【推断：WAR 顺序由数据依赖 + barrier 保证】。
- **没有 `multimem.st`，也就没有"写完后 fence + 通知"的环节**。结果写在本卡 recvbuff，kernel 结束后由流顺序保证可见。
- **min-poll 只在 `recv[0]` 上生效**：`recv[0].conn.flags |= NCCL_NVLS_MIN_POLL`（`transport/nvls.cc:456`），Reduce 的 WaitRecv 在 `loadRecvConn` 中据此置 `NvlsMinPolling`（`prims_simple.h:501-505`），`loadStepValue` 改用 `multimem.ld_reduce.acquire.sys.global.min.u64`（`prims_simple.h:96-107`）。Scatter 用的 `send[1]` 没有这个标志，WaitSend 用 `ld.volatile.global.u64` 读本卡。
- Reduce 的 PostRecv 在构造函数中还会执行一次 `*connStepPtr = step`（`prims_simple.h:497-500`），即在 kernel 开始时向 MC `A_cred[r].head` 写一次当前 step（"Return credits in case we rounded up"）。
- 与 AllReduce 相比少了 B 方向的两次握手（AllReduce 中的 ②④）。

**6. 流水与重叠**

- Scatter 与 Reduce 通过 8 深的 A FIFO 解耦：rank r 的 Scatter 在 A[h] 上最多领先 head h 的 Reduce 8 个 step；head r 的 Reduce 在 step s 要等齐所有 P 张卡的 Scatter 的 step s（①）。整个组按最慢的 rank 前进。
- 每个 step 中，Scatter 要拷 P 个 chunk（7 个 worker warp），Reduce 只处理 1 个 chunk（12 个 warp）。但 Reduce 的每条 `multimem.ld_reduce` 要等 P 张卡的数据经交换机汇总后返回，延迟远高于本地 load【推断】，所以分给它更多 warp 来提高在途请求数：一个 CTA 最多约 12 warps × 4KB = 48KB 的 `ld_reduce` 在途（受每 step hunk 数限制，见第 8 小节）。
- Reduce 组内 `ld_reduce` 与 `st.global` 的重叠方式与 2.2.1 第 6 小节相同（`reduceCopyPacks`，`Unroll = 8`，每 warp 每轮 4KB hunk，warp 之间独立）。
- 24 个 channel 相互独立，填补 step 边界的气泡。

**7. 流量与负载均衡**

每张 GPU 的 NVLink 逻辑流量（不计协议开销和 credit）：

| 方向 | 来源 | 流量 |
|:---|:---|:---|
| GPU → Switch | 本卡 A 区被所有 P 个 head 的 `ld_reduce` 读取（含自己那份） | P·N |
| Switch → GPU | 本卡作为 head 收到的 `ld_reduce` 结果（已规约） | N |

- Ring ReduceScatter 每个方向约为 `(P-1)·N`。NVLS 的接收方向降为 N，但发送方向是 `P·N`，比 Ring 多了自己那一份【推断：本卡也是 multicast 成员，自己那份数据也由交换机读取】。与 AllGather 方向相反、规模相同。
- 本卡 HBM：Scatter 读 sendbuff P·N、写 A 区 P·N，交换机读 A 区 P·N，Reduce 写 recvbuff N，合计约 `3P·N + N`。
- 负载均衡：nHeads = P，每个 rank 恰好规约自己那一段，Scatter 与 Reduce 的工作量在所有 rank 上完全对称。

**8. 示例：P = 8 与 P = 16（GB200 MNNVL，sendbuff 共 128MiB，bf16 sum）**

channel 切分与 AllGather 相同：`trafficBytes = P·N = 128MiB`，24 个 channel 中前 23 个各 171 个 cell、最后一个 163 个 cell。

| | P = 8 | P = 16 |
|:---|:---|:---|
| N（每 rank recvbuff） | 16MiB | 8MiB |
| nHeads / nvlsChannels | 8 / 24 | 16 / 24 |
| cellSize | 4KB | 2KB |
| 每 channel 每段的数据量 | 684KB | 342KB |
| chunkSize（`calcCollChunking` 的 nBytes ≈ 5.34MB，concurrentOps = 8 / 16） | 64KB | 32KB |
| 每 step 读取 sendbuff 的数据量（P × chunk） | 512KB | 512KB |
| 每个 channel 的 step 数 | 11 | 11 |
| Scatter 每 step（7 个 worker warp） | 8 个 head × 16 个 hunk，每个 head 上 2 个 warp 做 3 个、5 个 warp 做 2 个 | 16 个 head × 8 个 hunk，每个 head 上 1 个 warp 做 2 个、6 个 warp 做 1 个 |
| Reduce 每 step（12 个 warp） | 16 个 hunk，4 个 warp 各 2 个、8 个各 1 个 | 8 个 hunk，4 个 warp 空闲 |
| Reduce 每线程每 step 的 `ld_reduce` 数 | 2 个 hunk 时 16 次，1 个 hunk 时 8 次 | 8 次（空闲 warp 为 0） |
| WaitSend / PostSend lane 数（PostSend 各做一次 fence） | 8 / 8 | 16 / 16 |
| 每条 `ld_reduce` 的扇入 / ① 的扇入 | 8 / 8 | 16 / 16 |
| NVLink 发送 / 接收（每卡） | 128MiB / 16MiB | 128MiB / 8MiB |
| Ring 对照（每个方向 `(P-1)·N`） | 112MiB | 120MiB |

总 sendbuff 相同时，P = 16 的每 step 有效工作量与 P = 8 相同，但 Reduce 的 chunk 减半，12 个 warp 中有 4 个空闲，在途 `ld_reduce` 减少；同时 Scatter 每 step 要处理 16 条 FIFO、16 次 fence，`ld_reduce` 和 ① 的扇入也更大【推断：P = 16 时更难达到峰值带宽】。

**数据量**：每个 rank 读入 P·N（sendbuff）、写出 N（recvbuff）；NVLink 发送 P·N、接收 N。step 数不随 P 增长。

---

### 2.3 Symmetric NVLS kernel

这一类 kernel 使用 symmetric memory（对称内存）：用户用 `ncclMemAlloc`（cuMem）分配缓冲区，再用 `ncclCommWindowRegister(..., NCCL_WIN_COLL_SYMMETRIC)` 注册成 window。注册后，kernel 可以**直接读写对端的用户缓冲区**，不经过任何中转 FIFO。

**symmetric window 与 Legacy NVLS 的 `ucBuff` / `mcBuff` 的区别**

注册过程（`dev_runtime.cc`）：

```c
cuMemRetainAllocationHandle(&memHandle, userPtr)                 // :676  取得用户缓冲区的物理分配
cuMemExportToShareableHandle(&fabricHandle, memHandle)           // :198  导出（MNNVL 下是 fabric handle）
// 每个 rank：导入所有对端的 handle，映射到一段连续的平铺 VA
cuMemAddressReserve(&lsaFlatBase, lsaSize * bigSize)             // :205
for r in lsaTeam: cuMemImportFromShareableHandle(...); cuMemMap(lsaFlatBase + r*bigSize + bigOffset, ...)  // :219-226
// 多播：把用户缓冲区本身绑定进 team 的 multicast 对象
cuMulticastBindMem(mcHandle, bigOffset, memHandle, 0, size)      // :247
cuMemMap(mcAddr → mcHandle)                                      // :319-320
```

对端 r 的地址 = `lsaFlatBase + r × bigSize + bigOffset + offset`（`dev_runtime.cc:1474`）；多播地址 = `mcAddr + bigOffset + offset`。

| | Legacy NVLS `ucBuff` / `mcBuff` | symmetric window |
|:---|:---|:---|
| 内存归属 | NCCL 内部的中转缓冲区，初始化时按 `nHeads × 2 × nChannels × 1MB` 分配 | **用户缓冲区本身**（sendbuff / recvbuff） |
| 地址视图 | UC：**只有本卡**这一份；MC：所有卡 | **每个对端各一个单播地址**（P 个，可以直接 `ld`/`st` 对端内存）+ MC 地址 |
| 数据拷贝 | 需要 sendbuff → FIFO → recvbuff 的本地中转 | 零拷贝，直接读写用户缓冲区 |
| 访问组织 | 固定大小的 FIFO，按 step 循环复用 slot | 整个缓冲区一次性可见，没有 slot 复用 |
| 同步方式 | 每个 step 都有 credit（head/tail）握手 | kernel 首尾的 LSA barrier，或 LL 包内的 epoch flag（见下文） |
| MNNVL 下 | `nvlsRegSupport = 0`，不能把用户缓冲区注册进来 | 可以注册（使用 fabric handle） |

两者的共同点：MC 地址都只是一个虚拟地址视图，背后是各卡各自的物理内存，**各卡的副本不会自动保持一致**（见 2.2.1 第 2 小节）。

**数据平面**

- **单播（ST / LD）**：每个 rank 用普通 `st` 写 P-1 个对端的 LSA 地址，或用普通 `ld` 从 P-1 个对端读取。ReduceScatter 在 SM 内归约，AllGather 由 SM 一对一推送。
- **多播（STMC / LDMC）**：每个 rank 用 `multimem.st` 写 MC 地址，NVSwitch 把数据复制到所有 rank（AllGather）；或用 `multimem.ld_reduce` 读 MC 地址，NVSwitch 归约（ReduceScatter）。
- **LL / LLMC**：数据和 epoch flag 打包写到对端的 LL scratch 缓冲区。LLMC 对 MC 地址使用的是**普通 `st.relaxed.sys.v4.u32`**，不是 `multimem.st`（`include/nccl_device/impl/ll_a2a__funcs.h:83`）。

**同步：没有 FIFO，但仍然需要同步**

Symmetric kernel 没有 Legacy 那种逐 step 的 credit 流控，但并不是"不需要同步"。与 Legacy 在同步对象上的区别见下文"与 2.2 Legacy NVLS 的对比"。按 kernel 类型分为三种（详见 2.3.2 第 3 小节）：

| 类型 | 同步方式 | 作用 |
|:---|:---|:---|
| ST / LD / RSxLD_AGxST | kernel 首尾各一次**单播 LSA barrier**：P-1 个线程各向一个对端的 `inbox` 写入 epoch+1，再并行轮询本卡 inbox | 开头：确认对端已进入 kernel，它的 recvbuff 可写 / sendbuff 已就绪；结尾：发布写到对端的数据，或确认大家都已读完我的 sendbuff |
| STMC / LDMC / RSxLDMC_AGxSTMC | kernel 首尾各一次**多播 LSA barrier**：thread0 执行 `multimem.red.{release\|relaxed}.sys.add.u32 [MC inbox], 1`，然后轮询本卡计数直到 ≥ epoch+P（`include/nccl_device/impl/lsa_barrier__funcs.h:70-72`） | 同上 |
| LL / LLMC | **没有 barrier**。每个 16B 包是 `{data_lo, epoch, data_hi, epoch}`，接收方轮询本卡 scratch 中 flag 等于当前 epoch 的包；scratch 按 epoch 奇偶分成两个半区轮流使用 | 逐包同步，数据和 flag 在同一个 8B 半行里，不需要 fence |

- barrier 是**每个 CTA 一个**（`index = blockIdx.x`），只同步各 rank 上 blockIdx 相同的 CTA，不是整卡 barrier。
- epoch 单调递增，在 kernel 结束时写回，下一次调用接着使用，所以连续两次调用之间不需要清零。
- kernel 中间（首尾 barrier 之间）没有跨 rank 同步。

| collective | kernel | instruction | engine | protocol | buffer |
|:---|:---|:---|:---|:---|:---|
| all-reduce | `AllReduce_RSxLD_AGxST` | 单播 `ld`（RS 阶段）+ 单播 `st`（AG 阶段） | SM | SIMPLE 风格 | send 和 recv **都要注册** |
| | `AllReduce_RSxLDMC_AGxSTMC` | `multimem.ld_reduce` + `multimem.st` | SM，NVSwitch 归约和复制 | SIMPLE 风格 | 同上；P=72 时每卡 **1 个 CTA** |
| | `AllReduce_AGxLL_R` | 单播 LL `st`（广播给所有人，本地归约） | SM | LL 风格 | 见下方回退规则 |
| | `AllReduce_AGxLLMC_R` | 对 MC 地址的普通 LL `st` | SM，NVSwitch 复制 | LL 风格 | 同上 |
| all-gather | `AllGather_ST` | 单播 `st` | SM | SIMPLE 风格 | recv 要注册 |
| | `AllGather_STMC` | `multimem.st` | SM，NVSwitch 复制 | SIMPLE 风格 | recv 要注册；P=72 时每卡 **1 个 CTA** |
| | `AllGather_LL` | 单播 LL `st` | SM | LL 风格 | 见下方回退规则 |
| | `AllGather_LLMC` | 对 MC 地址的普通 LL `st` | SM，NVSwitch 复制 | LL 风格 | 见下方回退规则 |
| reduce-scatter | `ReduceScatter_LD` | 单播 `ld`，SM 内归约 | SM | SIMPLE 风格 | send 要注册 |
| | `ReduceScatter_LDMC` | `multimem.ld_reduce` | SM，NVSwitch 归约 | SIMPLE 风格 | send 要注册；P=72 时每卡 **1 个 CTA** |
| | `ReduceScatter_LL` | 单播 LL `st` + 本地归约 | SM | LL 风格 | 见下方回退规则 |

**可用性与注册**：`ncclSymkAvailable` 只检查 `isAllDirectNvlink`、op/类型和 kernel 掩码（`sym_kernels.cc:558`）；注册情况在 `ncclSymkPickKernel` 中按 `winRegType` 过滤（`sym_kernels.cc:581-586`）：AllReduce 只有 send 和 recv 都注册时才考虑非 LL kernel；AllGather 要求 recv 已注册；ReduceScatter 要求 send 已注册。否则只剩 LL 类可选。多个 work 合并（`nWorks > 1`）时不能使用 LL 类。

**NVL72 上 multimem 版的 CTA 限制**：SM100 上 `nMaxBlocksNvls = divUp(32, P)`（`sym_kernels.cc:319`），P = 8 / 16 / 72 时分别为 4 / 2 / **1**。STMC / LDMC / RSxLDMC_AGxSTMC 在 P=72 时整张 GPU 只跑 1 个 CTA。代价模型中这几个 kernel 的 `busBytes` 故意少算、并乘以 `busMultiplier = 0.55 × P`，源码注释写明是为了让 MC 版能胜过非 MC 版（`sym_kernels.cc:355, 367`），而不是为了补偿 CTA 数。实际能否打满带宽需要实测。

**回退规则**（`scheduler/symmetric_sched.cc:97-117`）：
- send 和 recv 都注册时，直接使用选出的 symmetric kernel，不回退。
- 否则只剩 LL 类 kernel 可选：
  - send 和 recv 都没注册时，默认（`NCCL_SYM_NOWIN_ENABLE=0`）一律回退；
  - 只注册了一个时，查询 Legacy 代价模型，如果它选出的协议不是 LL 就回退。
- 回退目标是 Legacy 调度器选出的算法（不一定是 Ring）。
- 另外，Blackwell 上 recv 已注册、每 rank count 超过 `NCCL_SYM_CE_THRESHOLD` 的 AllGather，会在入队阶段直接选择 CE，不进入 symmetric kernel（`enqueue.cc:2966`，见 2.4.2）。

**与 2.2 Legacy NVLS 的对比（两者都使用 multimem 的 kernel）**

Legacy NVLS（2.2）与 Symmetric MC（2.3 中的 `RSxLDMC_AGxSTMC`、`STMC`、`LDMC`）使用**同一套硬件原语**：`multimem.ld_reduce` 让 NVSwitch 规约，`multimem.st` 让 NVSwitch 复制。AllReduce 的每卡 NVLink 逻辑流量也相同，都约为 `(P+1)/P × N`。差别不在指令，而在三件事：**数据放在哪、工作怎么组织、怎么同步**。

- Legacy：数据先进 NCCL 的中转 FIFO，CTA 内按角色分成流水线，每个 step 做一轮 credit 握手。
- Symmetric：直接对用户缓冲区的 MC 地址操作，所有 warp 做同样的工作，整个 kernel 只在首尾各有一次 barrier。

数据路径：

| collective | Legacy NVLS（2.2） | Symmetric MC（2.3） |
|:---|:---|:---|
| AllReduce | sendbuff →（Scatter，本地拷贝）→ UC A FIFO → head：`ld_reduce` MC A → `multimem.st` MC B →（Gather，本地拷贝）→ recvbuff | `RSxLDMC_AGxSTMC`：`ld_reduce` **MC(sendbuff)** → `multimem.st` **MC(recvbuff)**，由同一个线程完成 |
| AllGather | Bcast warp：读 sendbuff → `multimem.st` MC B FIFO → Gather：从本卡 P 条 UC B FIFO 拷贝到 recvbuff | `STMC`：读本卡 sendbuff → `multimem.st` **MC(recvbuff + rank·N)**（`primitives.cuh:234`） |
| ReduceScatter | Scatter：sendbuff → UC A FIFO → Reduce warp：`ld_reduce` MC A → `st` 本卡 recvbuff | `LDMC`：`ld_reduce` **MC(sendbuff + rank·N)** → `st` 本卡 recvbuff（`reduce_scatter.cuh:333`） |

也就是说，Symmetric 版本相当于把 Legacy 中 Reduce/Bcast 那一组单独取出来，直接作用在用户缓冲区上，省掉了 Scatter 和 Gather 两次本地中转。

差异汇总：

| 方面 | Legacy NVLS | Symmetric MC |
|:---|:---|:---|
| multimem 操作的地址 | NCCL 内部 FIFO 的 MC 地址（`mcBuff`） | 用户缓冲区注册后得到的 MC 地址 |
| 本地 HBM 中转 | 有。MNNVL 下 `regUsed = 0`，AllReduce 额外读写约 2N | 无，零拷贝 |
| CTA 内分工 | 按角色分组（Scatter / Reduce 或 Bcast / Gather）。AllReduce 的 20 个 warp 中只有 4 个 worker warp 发 multimem 指令 | 不分角色，每个 CTA 512 个线程（16 个 warp）全部发 multimem 指令 |
| CTA 数 | `nvlsChannels`，SM100 单域为 24，与 P 无关 | `divUp(32, P)`：P = 8 / 16 / 72 时为 4 / 2 / **1** |
| 数据分工 | 按 head 划分，最多 32 个。每个 step 处理 `nHeads × chunk`，head h 取第 h 个 chunk | 按 4KB chunk 轮流分配：全局 warp 号 `W = rank + P × (block + nBlocks × warp)`，chunk 归 `W mod P` 号 rank，所有 P 个 rank 均分，**没有 32 的上限** |
| 流控 | 8 个 slot 的 FIFO 循环复用；每个 step 结尾要 `fence` 并更新 credit | 无 FIFO、无 step，chunk 之间不需要 fence 或 barrier |
| 跨卡同步次数 | 每个 channel、每个 step、每个 head：AllReduce 4 次，AllGather 和 ReduceScatter 各 2 次 | 每个 CTA 在 kernel 首尾各 1 次 `multimem.red.add` barrier |
| P > 32 时 | AllReduce 可用但只有 32 个 head；AllGather / ReduceScatter 不可用（`enqueue.cc:1914`） | 都可用，但每卡只有 1 个 CTA |
| 缓冲区要求 | 不需要注册 | 必须注册：AllReduce 要求 send 和 recv 都注册，AllGather 要求 recv 注册，ReduceScatter 要求 send 注册；另外要求 P > 2 |
| 额外显存 | 每卡 `nHeads × 2 × 24 × 1MB`（P = 8 时 384MB） | 无中转缓冲区，只有很小的 barrier inbox |

**两者的同步为什么不同：同步的对象不同，而不是"一个要同步、一个不要"**

Symmetric MC **同样需要同步**，只是同步点只有 kernel 首尾。不能理解为"用户缓冲区里的数据已经准备好，所以不用同步"。每张 GPU 只能保证自己 stream 上的先后顺序，并不知道别的 GPU 执行到哪了：

- 开头的 barrier：确认所有 rank 都已进入这个 kernel。此时每个 rank 的 sendbuff 才算就绪（它可能是前一个 kernel 刚算出来的），对端的 recvbuff 才可以被覆盖（对端可能还在使用它）。
- 结尾的 barrier：AllReduce 和 AllGather 用 release 把写给别人的数据发布出去；ReduceScatter 确认别人都已读完我的 sendbuff，我才能返回。
- LL 类 kernel 没有 barrier，改为每个 16B 包带 epoch flag 逐包同步。

Legacy 之所以需要逐 step 的 credit，主要原因不是"SM 拷贝可能还没完成"，而是 **FIFO 只有 8 个 slot，必须循环复用**。每个 step 要回答两个问题：

| 问题 | 含义 | 对应的同步 |
|:---|:---|:---|
| slot 里的数据好了没有？ | 对应"SM 拷贝是否完成" | Scatter 先 `fence`，再写 `A.tail`；Reduce 用 `ld_reduce.min` 轮询所有卡的 `A.tail` |
| slot 能不能被覆盖了？ | 与拷贝无关，是 slot 复用带来的 | Reduce 读完后写 `A.head` 归还 A slot；Gather 读完后写 `B.head`，Reduce 用 `ld_reduce.min` 确认后才写 B slot |

Symmetric 没有 slot 复用：缓冲区在整个 kernel 期间固定，每个字节只写一次、读一次，所以不需要第二类同步，第一类也不需要逐 step 进行。

| | 同步的对象 | 次数 |
|:---|:---|:---|
| Legacy | 每个 step：数据好了没有，以及 slot 能不能复用 | 约等于 step 数 × 每 step 的握手次数 |
| Symmetric | 整个 kernel：各 rank 是否都已进入、写出的数据是否已发布 | 每个 CTA 固定 2 次（LL 类为逐包 flag） |

实际取舍（NVL72 + MNNVL）：
- Legacy 的优势是用户无需改动、缓冲区不用注册；24 个 CTA 能同时发出更多请求，在途量更大。代价是额外的本地拷贝、大量的 step 级握手，以及最多 32 个 head。
- Symmetric 的优势是零拷贝、同步次数极少、所有 rank 均分工作。代价是缓冲区必须用 `ncclMemAlloc` 分配并注册为 window，而且 CTA 数被压到 `divUp(32, P)`，P 越大，单卡能同时发出的 multimem 请求越少。P = 72 时只有 1 个 SM，能否打满 NVLink 需要实测。

#### 2.3.1 AllReduce

**kernel**：Symmetric AllReduce 共有 4 个 kernel，入口都在 `device/symmetric/all_reduce.cuh`，由 `generate.py` 为每个 (op, type) 生成 `__global__ ncclSymkDevKernel_AllReduce_<algo>_<red>_<ty>`，参数是 4KB 的 `__grid_constant__` 结构 `ncclSymkDevWorkArgs4K`（`device/symmetric/generate.py:164`）。

| kernel | 入口 | 算法 | 数据平面 |
|:---|:---|:---|:---|
| `AllReduce_AGxLL_R` | `all_reduce.cuh:467` → `_impl:382` | one-shot：LL 格式 AllGather 到 scratch，每个 rank 本地归约全部数据 | 单播 `st.relaxed.sys.v4` 写 P 个 rank 的 LL scratch |
| `AllReduce_AGxLLMC_R` | `all_reduce.cuh:472` → `_impl:382` | 同上 | **一次**对 MC 地址的 `st.relaxed.sys.v4`（不是 `multimem.st`），NVSwitch 复制到 P 个 rank 的 scratch |
| `AllReduce_RSxLD_AGxST` | `all_reduce.cuh:265` | two-shot（RS + AG 按 chunk 融合）：单播 `ld` 读 P 个 rank 的 sendbuff，SM 归约，单播 `st` 写 P 个 rank 的 recvbuff | 单播，zero-copy |
| `AllReduce_RSxLDMC_AGxSTMC` | `all_reduce.cuh:350` | two-shot（RS + AG 按 chunk 融合）：`multimem.ld_reduce` 读、`multimem.st` 写 | 多播，zero-copy，NVSwitch 归约和复制 |

前提与记号：

- 所有 rank 在同一个 NVL72 MNNVL clique 内，`nNodes == 1`，LSA team = world，`lsaSize == nRanks == P`。GIN 类 kernel 被 `ncclSymkMask` 排除（`sym_kernels.cc:553-554`：`needGin = lsa.nRanks < nRanks` 为假时去掉 `kernelMask_Gin`）。
- 以 sum、bf16 为例。N 表示每个 rank 的输入字节数（AllReduce 的输入和输出都是 N）。
- 下文"UC 地址"指 LSA flat VA 中某个 rank 的单播地址；"MC 地址"指 multicast 对象的 VA。

**一句话概括**：小消息走 LL（one-shot：每个 rank 把自己的 N 字节以"8B 数据 + 8B flag"的 LL 格式广播给所有人，每个 rank 自己把 P 份加起来，靠逐包 flag 同步，**没有 barrier**）；大消息走 RSxLDMC_AGxSTMC（每个 rank 负责交错分布的 1/P 数据：`multimem.ld_reduce` 让 NVSwitch 求和，结果直接 `multimem.st` 广播进所有 rank 的 recvbuff，kernel 首尾各一次 LSA barrier）。RSxLD_AGxST 是没有 multicast 时（P = 2 等）的单播版 two-shot。

**1. kernel 选择**

**(1) 候选集合**（`ncclSymkMask`，`sym_kernels.cc:512-556`；`ncclSymkPickKernel`，`sym_kernels.cc:568-613`）

| 条件 | 效果 | 源码 |
|:---|:---|:---|
| `isAllDirectNvlink`，sum，浮点且非 f64 | 否则整个 symmetric 路径不可用 | `sym_kernels.cc:485-510, 558-566` |
| `hasLsaMultimem = nvlsSupport && lsa.nRanks > 2` | 否则去掉 LLMC、RSxLDMC。**P = 2 时没有多播 kernel** | `sym_kernels.cc:414, 540` |
| LDMC 的类型支持：bf16/f16/f32 sum 可以；fp8 要求 `compCap >= 100` | 否则去掉 RSxLDMC | `sym_kernels.cc:518-541` |
| `nBytes >= 2GB` | 去掉 LL 类（LL 用 32 位下标） | `sym_kernels.cc:544-546` |
| `nWorks > 1`（同一 group 里多个同类 AllReduce 合并成一次 launch） | 去掉 LL 类 | `sym_kernels.cc:577-578` |
| send/recv **不是都注册**（`winRegType != ncclSymSendRegRecvReg`） | **只剩 LL 类** | `sym_kernels.cc:580-581` |
| `NCCL_SYM_KERNEL=<名字>` | 只留该 kernel，`forced = true` | `sym_kernels.cc:102-125, 575` |

**(2) 代价模型**（`queryModel_lsa`，`sym_kernels.cc:314-404`；sm100 参数）

```text
time(nBlocks) = baseLat + softplus(busBytes / bw - 1)
bw            = softmin(nBlocks × smBw × busMultiplier, peakBw)      # smBw = 55 GB/s 每 CTA
```

| kernel | busBytes | busMultiplier | nMaxBlocks | baseLat | peakBw |
|:---|:---|:---|:---|:---|:---|
| AGxLL_R | `P × N × 9`（`LL_BusFactor = 9`："2X the bytes, plus some processing, plus no unrolling"） | 1 | 64 | 11 µs | 600 GB/s |
| AGxLLMC_R | `P × N × 9` | 1.1（注释 "To beat non-MC LL"） | 64 | 11 µs | 600 GB/s |
| RSxLD_AGxST | `2N(P-1)/P` | 1 | 64 | 19.5 µs | 600 GB/s |
| RSxLDMC_AGxSTMC | `N/P + N` | **P** | **`nMaxBlocksNvls = divUp(32, P)`** | 19.5 µs | **1000 GB/s** |

（`sym_kernels.cc:315, 319, 328-342, 389-391`。`nMaxBlocks` 还要再与 `comm->config.maxCTAs`（默认 64，`init.cc:2115`）取 min，`sym_kernels.cc:373`。）

- 每个 kernel 先算 `nMaxBlocks` 时的时间，再从 `minCTAs`（默认 1）往上找**第一个**时间不超过它 1.025 倍的 CTA 数（`sym_kernels.cc:393-403`）。
- 多个 kernel 之间按 `time × (1 + 0.025 × nBlocks)` 比较（每多一个 SM 罚 2.5%，`sym_kernels.cc:594-606`）。
- 每个 CTA 固定 16 个 warp，即 **512 线程**（`*nWarps = 16`，`sym_kernels.cc:611`；`plan->threadPerBlock = nWarps × 32`，`scheduler/symmetric_sched.cc:184`），不随消息大小变化。
- `nMaxBlocksNvls = divUp(32, P)`：P = 8 → 4，P = 16 → 2，P = 72 → **1**。乘积 P × nBlocks ≈ 32，相当于把整个 NVLS 域里同时发 multimem 的 CTA 总数限制在 32 左右【推断：源码没有注释说明意图】。

把上表代入模型（bf16，send/recv 都已注册），选择结果如下（数值由公式计算得到，不是实测）：

| P | LL 类 → RSxLDMC 的切换点 | 小消息的 kernel / CTA 数 | 大消息 RSxLDMC 的 CTA 数 |
|:---|:---|:---|:---|
| 2 | 无 MC：AGxLL_R → **RSxLD** 约 234KiB | AGxLL_R，1~ 个 | RSxLD，3~13 个 |
| 8 | 约 52KiB | AGxLLMC_R，2~10 个 | ≤ 512KiB：1；1~2MiB：2；≥ 4MiB：3（上限 4） |
| 16 | 约 26KiB | AGxLLMC_R，3~10 个 | ≤ 2MiB：1；≥ 4MiB：2（上限 2） |
| 72 | 约 6KiB | AGxLLMC_R，7~10 个 | 恒为 1 |

- 有多播时 **RSxLD_AGxST 永远选不上**：RSxLDMC 的 busMultiplier = P、peakBw = 1000，而 RSxLD 的 busBytes 约为 2N。只有 P = 2、类型不支持 LDMC，或用 `NCCL_SYM_KERNEL` 强制时才会用到 RSxLD。
- LLMC 与 LL 的模型几乎相同，LLMC 靠 busMultiplier = 1.1 在平局时胜出，所以有多播时小消息基本都是 AGxLLMC_R。

**(3) 注册与回退**（`scheduler/symmetric_sched.cc:89-138`）

| winRegType | 可选 kernel | 是否回退到 Legacy |
|:---|:---|:---|
| send 和 recv 都注册（`ncclSymSendRegRecvReg`） | 4 个都可以 | **永不回退**（`:104-105`）；即使大消息也走 symmetric |
| 只注册一边 | 只有 LL 类 | 若 legacy 调优器（`ncclGetAlgoInfo`）选出的协议不是 LL，则回退（`:110-116`） |
| 都没注册 | 只有 LL 类 | 默认 `NCCL_SYM_NOWIN_ENABLE=0` 时**无条件回退**（`:109`）；一个线程管多张 GPU 时也回退（`:107`）；`NCCL_SYM_KERNEL` 强制时不回退 |

"回退"是把 task 放回 legacy 任务队列，由 legacy 调优器在 Ring / NVLS / Tree 等之间重新选择（`:121-137`），不一定是 Ring。

**2. 内存**

**(1) 用户缓冲区：symmetric window**

用户用 `ncclMemAlloc` 分配，或用 `ncclCommWindowRegister(..., NCCL_WIN_COLL_SYMMETRIC)` 注册（`dev_runtime.cc:1067`）。注册流程（`ncclDevrWindowRegisterInGroup`，`dev_runtime.cc:638`）：

```text
ncclCommRegister(userPtr)                                    # :653 本地注册
ncclSymkInitOnce(comm)                                       # :657 第一次出现 SYMMETRIC window 时创建 LL scratch / barrier
cuMemGetAddressRange → 要求只跨 1 个物理段，偏移 4KB 对齐     # :661-674
cuMemRetainAllocationHandle(&memHandle, base)                # :676 取回用户那块 cuMem 物理内存的 handle
symMemoryObtain:                                             # :402
  bigOffset = ncclSpaceAlloc(bigSpace, size)                 # :428 在 [0, bigSize) 中分配偏移
  symMemoryMapLsaTeam:                                       # :183
    bootstrapIntraNodeAllGather(handle)                      # :201 MNNVL 下交换的是 CUmemFabricHandle
    cuMemImportFromShareableHandle + cuMemMap 到
      lsaFlatBase + r × bigSize + bigOffset   (r = 0..P-1)   # :225-227
  symBindTeamMemory: cuMulticastBindMem(mcHandle, bigOffset, memHandle, 0, size)   # :247
symWindowCreate:                                             # :516
  win.lsaFlatBase = devr.lsaFlatBase + bigOffset             # :545
  win.mcOffset4K  = bigOffset >> 12                          # :546
  win.stride4G    = bigSize >> 32                            # :547
bootstrapBarrier                                             # 注册是集合操作
```

- `bigSize = max(各 rank 的 totalGlobalMem)` 向上对齐到 4GB（`dev_runtime.cc:110-117`）。`lsaFlatBase` 处预留 `P × bigSize` 的 VA（`:205`）。
- LSA team 的 multicast 对象只建一次，大小为 `bigSize`（`symTeamObtain`，`dev_runtime.cc:309, 318-325`）。每个 window 的物理页按 `bigOffset` 绑定进去。
- 设备侧地址计算（`include/nccl_device/impl/core__funcs.h`）：

| 用途 | 地址 | 源码 |
|:---|:---|:---|
| rank r 的 UC 地址 | `win.lsaFlatBase + r × stride4G × 4GB + offset` | `core__funcs.h:134-140`（`ncclSymPtr::peerPtr(team, r)`） |
| 本 rank 的 UC 地址 | 同上，r = 自己（LSA flat VA 里自己那一份，是用户 VA 的别名） | `core__funcs.h:105-110` |
| MC 地址 | `mcBasePtr + mcOffset4K × 4KB + offset` | `core__funcs.h:144-148`（`ncclSymPtr::multimemPtr`） |
| 未注册的指针 | `ncclSymPtr` 的 window 为空，`offset` 就是原始指针 | `ptr__funcs.h:95-101`，`sym_kernels.cc:636-638` |

和 Legacy NVLS 的对比：Legacy 的 `ucBuff/mcBuff` 是 NCCL 自己分配的中转 FIFO，multicast 对象按 FIFO 大小创建，只映射本卡的 UC；Symmetric 的 window 就是**用户 buffer 本身**，所有 P 个 rank 的物理页都映射进本卡的 flat VA（可以直接 `ld/st` 对端），同时绑定到一个覆盖 `bigSize` 的共享 multicast 对象。

**(2) NCCL 内部资源：resource window**

`ncclSymkInitOnce`（`sym_kernels.cc:406-475`）通过 `ncclDevrCommCreateInternal` 申请两类资源，它们被合并到**一个**由 NCCL 自己 `cuMemCreate` 的 symmetric window（resource window）里（`dev_runtime.cc:934, 994-1008`），初始化时整块清零（`:1008`）。这块内存同样映射进 LSA flat VA，并绑定到 LSA multicast 对象。

| 资源 | 大小（每卡） | 布局 | 使用者 |
|:---|:---|:---|:---|
| LSA barrier | `(3 × 64 + 64 × P) × 4B`（`nccl_device/lsa_barrier.cc:18`；`nBarriers = ncclSymkMaxBlocks = 64`，`sym_kernels.cc:416`） | `[epochMC[64]] [epochUC[64]] [mcInbox[64]] [ucInbox[64][P]]`（`lsa_barrier__types.h:30-43`） | RSxLD（单播）、RSxLDMC（多播） |
| LL A2A scratch | `64 × (1 + 2 × nSlots) × 16B`，`nSlots = P × 512`（`nccl_device/ll_a2a.cc:13, 23`；`sym_kernels.cc:419-421`） | 每个 block：`[16B 头，x 存 epoch-2] [半区 0：nSlots × 16B] [半区 1：nSlots × 16B]`（`ll_a2a__types.h:32-34`） | AGxLL_R、AGxLLMC_R |

| P | barrier | LL scratch |
|:---|:---|:---|
| 8 | 2.75KB | ≈ 8.0MiB |
| 16 | 4.75KB | ≈ 16.0MiB |
| 72 | 18.75KB | ≈ 72.0MiB |

- barrier 和 LL scratch 都是**按 blockIdx 分槽**（最多 64 个 CTA），同一个 comm 上所有 symmetric kernel 的第 b 个 CTA 共用第 b 个槽。
- **没有 FIFO**：LD/LDMC 直接读写用户 buffer；LL 的 scratch 是"每个 epoch 一个半区"的邮箱，不是环形 FIFO，也没有 head/tail credit。

**3. 线程与数据切分（公共部分）**

**(1) host 侧切分**（`ncclSymmetricTaskScheduler`，`scheduler/symmetric_sched.cc:163-296`）：以 1KB 为一个 cell（`NCCL_SYM_KERNEL_CELL_SIZE`，`sym_kernels.h:19`），每个 channel 分 `cellPerChannel = divUp(divUp(count, nBlocks), cellCount)` 个 cell（`:203`），实际 launch 的 CTA 数是用到的 channel 数（`:279, 283`），可能小于模型给出的 nBlocks。

**(2) device 侧切分：两类 kernel 完全不同**

| | LD / LDMC（`forEachWork`） | LL（`singleWork`） |
|:---|:---|:---|
| CTA 看到的范围 | 单个 task 时整个 task 跨所有 channel，进入 `getWorkRangeFused`，**每个 CTA 都拿到 `[0, nElts)`**（`primitives.cuh:116-131, 146-147`） | 只拿到 host 侧分给自己的连续一段 `[indexLo, indexHi)`（`primitives.cuh:92-113, 164-175`） |
| 再切分 | 全局 warp 号 `W = rank + P × (block + nBlocks × warp)`（`flattenIx`，`all_reduce.cuh:283-287, 368-372`；`primitives.cuh:41-46`）；warp W 处理第 W、W + P·nBlocks·16、… 个 chunk | 每轮（epoch）512 个线程各取一个 8B pack，即每个 CTA 每轮 4KB |
| 结果 | rank r 负责 chunk 号 ≡ r (mod P) 的数据：**按 chunk 交错**，不是连续的 `[r·N/P, (r+1)·N/P)` | 每个 rank 处理**全部** N（one-shot），CTA 之间按连续段分 |

chunk 大小：RSxLD 为 2KB（16B pack × 4 unroll × 32 lane，`all_reduce.cuh:222, 30`）；RSxLDMC 为 4KB（16B pack × 8 unroll × 32 lane，`all_reduce.cuh:306, 312-313, 318`）。

**4. LSA barrier 的实现**（`include/nccl_device/impl/lsa_barrier__funcs.h`）

`ncclLsaBarrierSession<ncclCoopCta>(cta, comm, ncclTeamTagLsa(), blockIdx.x, multimem)`：每个 CTA 有自己的 barrier（index = blockIdx.x），只和**其他 rank 上同号的 CTA** 同步。`ncclCoopCta::sync()` 就是 `__syncthreads()`（`coop.h:152`）。

```text
# 构造（:29-30）：所有线程读本卡 UC 的 epoch
epoch = state[(multimem ? 0 : 1) × 64 + index]

# arrive(order)（:63-84）
__syncthreads()
if multimem:                                                # RSxLDMC 使用
    tid 0:  multimem.red.release.sys.add.u32 [MC mcInbox[index]], 1     # order = release 时
            multimem.red.relaxed.sys.add.u32 [MC mcInbox[index]], 1     # order = relaxed 时
            # 一条指令，NVSwitch 对 P 张卡（含自己）的 mcInbox[index] 各 +1
else:                                                        # RSxLD 使用
    tid i（i < P-1）: peer = i + (rank <= i)
            atomic_ref(peer 的 ucInbox[index][rank]).store(epoch + 1, order)
            # 远端 UC 写：st.relaxed.sys / st.release.sys【推断：libcu++ 的典型展开】

# wait(order)（:89-118）—— 注意 acquireOrderOf(release) == relaxed（utility.h:309-313）
if multimem:
    tid 0:  loop: got = ld.relaxed.sys [本卡 UC mcInbox[index]]
                  until int32(got - (epoch + P)) >= 0       # 计数器到达 epoch + P
            epoch += P
else:
    tid i（i < P-1）: loop: got = ld.relaxed.sys [本卡 ucInbox[index][peer]]
                         until int32(got - (epoch + 1)) >= 0
    所有线程: epoch += 1
__syncthreads()

# 析构（:46-57）：tid 0 把 epoch 写回 state[...]，__syncthreads()，供下一次 kernel 继续使用
```

要点：

- **push + 本地轮询**：arrive 写的是对端（UC 版）或所有卡（MC 版）的 inbox，wait 只轮询本卡显存，轮询不产生 NVLink 流量。
- UC 版每个 rank 的 P-1 个线程各写一个对端，所以一次 arrive 产生 P-1 次跨卡写；MC 版只需 1 条 `multimem.red`。UC 版每个 (owner, sender) 有独立槽，写的是绝对值 `epoch + 1`；MC 版所有 sender 共享一个计数器，靠累加。
- **单调计数 + 有符号差比较**：计数器只增不减，判断用 `got - target` 的有符号解释，可以容忍 32 位回绕，也容忍快的 rank 已经进入下一轮 barrier 并多加了一次（此时 `got > target`，照样放行）。快的 rank 最多领先一轮：它要通过下一轮 barrier，必须等慢的 rank 也到达下一轮。
- **epoch 跨 kernel 持久化**：epoch 存在 resource window 的 `state[0..127]`（MC 版和 UC 版分开），由析构函数写回、下次构造时读出。resource window 初始化时清零（`dev_runtime.cc:1008`），所以所有 rank 的 epoch 从 0 开始，并且每次调用都推进相同的量，始终一致。
- 所有 wait 都是 **relaxed** 加载。release 语义只出现在结尾 barrier 的 arrive 一侧。

**5. 本节使用的 PTX 一览**

| 指令 | 出现位置 | 访问对象 |
|:---|:---|:---|
| `ld.global.v4` / `st.global.v4`（普通，16B） | RSxLD 读写 sendbuff/recvbuff【推断：C++ 解引用编译为普通 ld/st】 | 本卡与对端的 UC |
| `multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2` | RSxLDMC 主循环（`reduce_kernel.h:888-901, 844-852`） | sendbuff 的 MC |
| `multimem.ld_reduce...add.acc::f32.bf16x2`（取 4B 后截出 2B） | RSxLDMC 尾部（`reduce_kernel.h:920-932`） | sendbuff 的 MC |
| `multimem.st.global.v4.f32` | RSxLDMC 主循环（`op128.h:399-403`） | recvbuff 的 MC |
| `st.global.b16`（**普通 st 写 MC 地址**） | RSxLDMC 中 bf16 的尾部元素（`op128.h:387-389`） | recvbuff 的 MC |
| `st.relaxed.sys.v4.u32 {d0, epoch, d1, epoch}` | LL / LLMC 的 bcast（`ll_a2a__funcs.h:54, 83, 110, 135`） | LL scratch 的 UC（LL）或 MC（LLMC） |
| `ld.relaxed.sys.v4.u32`（sm90 上为 `ld.acquire.sys`） | LL recv 轮询（`ll_a2a__funcs.h:184, 192`） | 本卡 LL scratch |
| `multimem.red.{release,relaxed}.sys.add.u32` | MC barrier arrive（`lsa_barrier__funcs.h:70, 72`） | resource window 的 MC |
| `st` / `ld` `.{relaxed,release}.sys.u32`（libcu++ `atomic_ref`） | UC barrier（`lsa_barrier__funcs.h:80-81, 108-111`） | 对端 / 本卡 UC |

---

##### 2.3.1.1 `AllReduce_AGxLL_R` / `AllReduce_AGxLLMC_R`

两个 kernel 共用 `ncclSymkRun_AllReduce_AGxLL_R_impl`（`all_reduce.cuh:382-464`），唯一的区别是 `multimem` 参数（`:468, 473`），只影响 `lla2a.bcast` 的写法。

**1. 一句话概括**

每个 rank 把自己的 N 字节按 8B 一个 pack，包装成 16B 的 LL 包（数据中间插入 epoch flag），写进**所有 P 个 rank**（包括自己）的 LL scratch；然后每个 rank 从本卡 scratch 收齐 P 份，在 SM 上用 fp32 求和，写进本卡 recvbuff。本质是 AllGather（LL）+ 本地 Reduce，所以叫 `AGxLL_R`。sendbuff 只在本卡读，recvbuff 只在本卡写，**用户 buffer 不需要注册**。

选择条件：小消息（有多播时 P = 8 约 < 52KiB，P = 16 约 < 26KiB，P = 72 约 < 6KiB）；只注册一边或都不注册时只能选它们（还要过回退检查，见 1(3)）；`N < 2GB` 且 group 内只有 1 个 task。

**2. 关键参数**

| 参数 | 值 | 源码 |
|:---|:---|:---|
| 每个 CTA 的线程数 `tn` | 512（`ncclSymkMaxThreads`） | `all_reduce.cuh:410`，`sym_kernels.h:22` |
| CTA 数 | 代价模型给出，一般 2~13；实际 launch 数 ≤ 模型值（按 1KB cell 切分） | `sym_kernels.cc:393-403`，`symmetric_sched.cc:203` |
| pack | `BytePack<8>`，bf16 时 4 个元素 | `all_reduce.cuh:394-396` |
| 每轮（epoch）每个 CTA 处理 | `min(nPacks, 512)` 个 pack，最多 4KB | `all_reduce.cuh:416, 432-434` |
| 累加类型 | bf16 → fp32（`ncclSymkAccumType<FuncSum, bf16, false>`） | `primitives.cuh:208`，`all_reduce.cuh:391` |
| LL 槽位 | sender s 的第 t 个 pack 放在 `slot = s × nIterPacks + t` | `all_reduce.cuh:418-420` |

**3. 使用的内存**

- sendbuff / recvbuff：`inputPtr.localPtr()` / `outputPtr.localPtr()`（`all_reduce.cuh:403-404`），只访问本卡。
- LL scratch（resource window）：第 blockIdx 个 block 槽，当前半区为 `epoch & 1`（`ll_a2a__types.h:32-34`）。
  - LL：写对端的 UC 地址 `ncclGetResourceBufferPeerPointer`（`ll_a2a__funcs.h:105, 130`）。
  - LLMC：写 MC 地址 `ncclGetResourceBufferMultimemPointer`（`ll_a2a__funcs.h:78`）。
  - 读：总是本卡 UC（`ll_a2a__funcs.h:171`）。

**4. 执行流程**

```text
# 每个 rank 上的 CTA b（512 线程），处理自己那一段连续数据
lla2a = LLA2ASession(block=b)                       # epoch = header.x + 2（ll_a2a__funcs.h:25-28）
while nPacks > 0:                                    # 每轮 = 1 个 epoch
  nIter = min(nPacks, 512)
  if t < nIter:
    inp = ld.global.v2 [本卡 sendbuff + t*8]         # 8B
    # ---- bcast（ll_a2a__funcs.h:73-152）----
    LL  : for r in rank, rank+1, ..., rank-1:        # P 次，含自己；每 8 个 rank 展开一次
            st.relaxed.sys.v4.u32 [UC scratch(r)[half][rank*nIter + t]], {inp.lo, epoch, inp.hi, epoch}
    LLMC: st.relaxed.sys.v4.u32 [MC scratch[half][rank*nIter + t]], {inp.lo, epoch, inp.hi, epoch}
            # 1 条普通 st 写 MC 地址，由 NVSwitch 复制到 P 张卡【推断：依赖实现行为】
    # ---- recvReduce（ll_a2a__funcs.h:241-267），Unroll = 8 ----
    for s0 in 0, 8, 16, ... < P:
      loop:                                          # 自旋直到这 8 个槽全部到齐
        v[k] = ld.relaxed.sys.v4.u32 [本卡 scratch[half][(s0+k)*nIter + t]], k = 0..7
      until 所有 v[k].y == epoch && v[k].w == epoch
      acc(fp32) += bf16→f32({v[k].x, v[k].z})
    st.global.v2 [本卡 recvbuff + t*8], f32→bf16(acc)
  # ---- endEpoch（ll_a2a__funcs.h:272-285）----
  if epoch >= 2^32-2: __syncthreads(); 本卡当前半区清零
  __syncthreads()
  epoch += (epoch == 2^32-1) ? 3 : 1                 # 跳过 0 和 1
  half = epoch & 1
# 析构：tid 0 写 header.x = epoch - 2（ll_a2a__funcs.h:35-38）
```

**5. 同步**

LL 类 kernel **不创建 `ncclLsaBarrierSession`**，开头和结尾都没有 barrier。全部同步由 LL 包里的 flag 完成。

| 机制 | 实现 | 源码 |
|:---|:---|:---|
| 逐包同步 | 16B 包为 `{data_lo, epoch, data_hi, epoch}`，一次 `st.v4` 写出。接收方轮询直到 `.y` 和 `.w` 都等于当前 epoch。flag 交错放在两个 8B 半部里，即使 16B 写入不是原子的，每个 8B 半部也各自带有 flag【推断：依赖 8B 访问的原子性】 | `ll_a2a__funcs.h:54, 215-216` |
| flag 的值 | 就是 epoch。每个 CTA 一个 epoch 计数器，每轮 +1；跨 kernel 存在 scratch 的 header 里。初值：scratch 清零 → header 0 → epoch 从 2 开始，所以清零的内存永远不会被误认为有效包 | `ll_a2a__funcs.h:25-28, 35-38, 283`，`dev_runtime.cc:1008` |
| memory order | 写和读都是 `.relaxed.sys`（sm90 上读用 `.acquire.sys`）。数据和 flag 在同一次访问里，不需要额外 fence | `ll_a2a__funcs.h:181-196` |
| 复用保护（双缓冲） | epoch e 用半区 `e & 1`。rank A 在 epoch e 写 B 的半区 (e&1) 之前，A 一定已经收齐了 B 在 epoch e-1 发出的包；B 发 epoch e-1 的包是在 B 的 `endEpoch(e-2)` 的 `__syncthreads()` 之后，此时 B 的所有线程都已读完半区 (e&1) 里 epoch e-2 的数据。所以同一个半区不会被提前覆盖。前提是每轮每个 rank 都给每个 rank 至少发一个包、也从每个 rank 收一个包（`ll_a2a.h:46-49`）；这里 t = 0 的线程每轮都会参与 | `ll_a2a__funcs.h:272-285` |
| 回绕保护 | epoch 接近 2^32 时把用完的半区清零，回绕后跳过 0 和 1 | `ll_a2a__funcs.h:273-283` |
| 跨调用 | epoch 跨 kernel 连续，同一套双缓冲论证同样适用于"上一次调用的最后一轮"和"这一次调用的第一轮"。用户 buffer 只在本卡访问，不需要首尾 barrier | — |

为什么没有 FIFO 和 credit 也是正确的：每一轮的数据量（≤ P × 512 个槽）正好能装进一个半区，发送方至多领先接收方一轮，而 flag 同时起到"数据到达"和"槽可用"两种作用（由上面的双缓冲论证保证）。

**6. 流水**

- CTA 内各轮严格串行：每轮结尾有 `__syncthreads()`，每轮的代价约为一次 NVLink 单向延迟加上自旋。
- 跨 rank：双缓冲允许快的 rank 先发下一轮，与慢的 rank 收当前轮重叠。
- CTA 之间完全独立。
- 一个线程的 8 个槽读取同时发出（Unroll = 8），P = 16 时分 2 批，P = 72 时分 9 批。

**7. NVLink 流量**（每张 GPU，逻辑字节数）

| | AGxLL_R | AGxLLMC_R |
|:---|:---|:---|
| GPU → Switch | `2N × (P-1)`（16B 包装 8B 数据，发给 P-1 个对端） | `2N`（每个包只发 1 次） |
| Switch → GPU | `2N × (P-1)` | `2N × P`（含自己那份经交换机回写【推断】） |
| 每个 pack 的 SM store 指令数 | P | 1 |
| 本卡 HBM | 读 scratch `2N × P`，加上读 N、写 N | 同左 |
| SM 归约 | 每个 rank 都对全部 P × N 字节求和（冗余计算） | 同左 |

LL 的流量是 two-shot 的约 P 倍，只适合延迟主导的小消息。代价模型里的 `busBytes = 9 × P × N` 反映了这一点。

**8. 示例：P = 8 与 P = 16（N = 16KiB bf16，send/recv 已注册）**

| | P = 8 | P = 16 |
|:---|:---|:---|
| 模型选择 | AGxLLMC_R，nBlocks = 9 | AGxLLMC_R，nBlocks = 10 |
| 实际 CTA 数（`cellPerChannel = 2`） | 8 | 8 |
| 每个 CTA | 2KB = 256 个 pack，1 个 epoch，256 个线程干活 | 同左 |
| 每个 CTA 每轮占用的 scratch | 8 × 256 × 16B = 32KB | 16 × 256 × 16B = 64KB |
| 每个线程的 recv | 8 个槽，1 批 | 16 个槽，2 批 |
| GPU → Switch / Switch → GPU | 32KB / 256KB | 32KB / 512KB |
| 换成 AGxLL_R 时每卡发出 | 224KB | 480KB |

P 翻倍时，接收量和 SM 归约量都翻倍，所以 LL 的适用上限随 P 下降（52KiB → 26KiB → 6KiB）。

---

##### 2.3.1.2 `AllReduce_RSxLD_AGxST`

**1. 一句话概括**

two-shot 的单播版：每个 warp 负责一个 2KB chunk，先用普通 `ld` 读出 P 个 rank（含自己）sendbuff 中该 chunk 的数据，在寄存器里以 fp32 求和，然后**立即**用普通 `st` 把结果写到 P 个 rank（含自己）recvbuff 的同一位置。RS 和 AG 按 chunk 融合在同一个线程里，**两阶段之间没有 barrier，也没有写本地中间结果**。只有 kernel 开头（拆成 arrive / wait 两半）和结尾各有一次单播 LSA barrier。

选择条件：send 和 recv 都必须注册；有多播时代价模型不会选它；P = 2（`hasLsaMultimem` 为假）时约从 234KiB 起被选中；也可以用 `NCCL_SYM_KERNEL=AllReduce_RSxLD_AGxST` 强制。

**2. 关键参数**

| 参数 | 值 | 源码 |
|:---|:---|:---|
| 线程数 | 512 / CTA | `sym_kernels.cc:611` |
| CTA 数 | 上限 64，模型通常给 3~13 | `sym_kernels.cc:318, 335-336` |
| 全局 warp 号 | `W = rank + P × (block + nBlocks × warp)`，共 `wn = P × nBlocks × 16` 个 | `all_reduce.cuh:283-287, 24-25` |
| 主阶段 | 16B pack，`UnrollPacks = 4`，`UnrollPeers = 2`；warp 每次处理 4 × 32 × 16B = **2KB**；总量按 `8KB × P × nBlocks` 的整数倍截取 | `all_reduce.cuh:221-236` |
| 次阶段 | 4B pack，`UnrollPacks = 4`，`UnrollPeers = 4`，处理主阶段剩下的部分 | `all_reduce.cuh:239-255` |
| 尾部 | 逐元素，`UnrollPeers = 8`，由全部 `P × nBlocks × 512` 个线程分摊 | `all_reduce.cuh:259-261, 129-201` |
| 累加类型 | bf16 → fp32 | `all_reduce.cuh:271` |

**3. 使用的内存**

- sendbuff / recvbuff 必须是 symmetric window。所有访问都通过 `peerPtr(world, r)`（`all_reduce.cuh:38, 52, 77, 108`），即 LSA flat VA 中 rank r 的 UC 地址；访问自己的数据时也走 flat VA 中自己那一份。
- barrier：resource window 中的 `ucInbox[blockIdx][P]` 和 `epochUC[blockIdx]`。
- 不使用 LL scratch，没有任何中转缓冲区。

**4. 执行流程**

```text
# rank r 的 CTA b，warp k，lane l；W = r + P*(b + nBlocks*k)
bar = LsaBarrierSession(index=b, multimem=false)
bar.arrive(relaxed):                                  # all_reduce.cuh:276
    __syncthreads()
    tid i < P-1: st.relaxed.sys.u32 [peer_i 的 ucInbox[b][r]], epoch+1      # 远端 UC
# ---- 主阶段 allreduceDeep<16B>（all_reduce.cuh:13-127）----
c = W                                                 # chunk 号，2KB 一个
acc0[0..3] = ld.global.v4 [UC_r(sendbuff) + c*2KB + u*512 + l*16]   # 先读本 rank 的数据，可与 barrier 等待重叠
bar.wait(relaxed):                                    # all_reduce.cuh:42
    tid i < P-1: 自旋 ld.relaxed.sys.u32 [本卡 ucInbox[b][peer_i]] >= epoch+1
    __syncthreads()
loop:
    # RS：从 r+1 开始轮转读取，各 rank 起点不同，分散对端压力
    tmp = ld.global.v4 [UC_{r+1}(sendbuff) + ...] × 4;  acc1 = f32(acc0) + f32(tmp)
    for 剩下的 P-2 个 rank，每次 2 个:
        tmp[2][4] = ld.global.v4 [UC_peer(sendbuff) + ...]  # 8 条 16B load 同时在途
        acc1 += f32(tmp)
    acc0 = bf16(acc1)
    # AG：从自己开始，写到 P 个 rank
    for q in r, r+1, ..., r-1:
        st.global.v4 [UC_q(recvbuff) + c*2KB + u*512 + l*16], acc0          # q != r 时为远端写
    c += wn
    if 还有: acc0 = ld.global.v4 [UC_r(sendbuff) + c*2KB ...]              # 预取下一轮本地数据
# ---- 次阶段（4B pack）与尾部（逐元素）：结构相同 ----
bar.sync(release):                                    # all_reduce.cuh:295
    __syncthreads()
    tid i < P-1: st.release.sys.u32 [peer_i 的 ucInbox[b][r]], epoch+1
    tid i < P-1: 自旋 ld.relaxed.sys.u32 [本卡 ucInbox[b][peer_i]] >= epoch+1
    __syncthreads()
# 析构：tid 0 写回 epochUC[b]
```

**5. 同步**

| 位置 | 实现 | 作用 |
|:---|:---|:---|
| 开头 arrive（relaxed） | P-1 个线程各向一个对端的 `ucInbox[b][r]` 写 `epoch+1` | 告诉对端"我已进入 kernel"：我的 stream 上之前的工作（生成 sendbuff、读取上一轮 recvbuff）都已完成 |
| 开头 wait（relaxed），**推迟到第一次访问对端之前** | P-1 个线程各轮询本卡的一个槽 | 保证访问对端 sendbuff/recvbuff 时，对端已经进入 kernel。在等待之前先发出本地 load（`all_reduce.cuh:34-42`），把 barrier 延迟和本地读重叠起来 |
| 中间 | **无** | 每个 chunk 只属于一个 rank（chunk 号 mod P），只由一个线程读 P 份、写 P 份；输出是最终位置，每个字节只写一次；输入在 kernel 内不会被别人修改。所以不需要 credit，RS 和 AG 之间也不需要 barrier |
| 结尾 sync（arrive 为 release，wait 为 relaxed） | 同上，store 改为 `.release.sys` | ① 我退出前，所有 rank 对我 recvbuff 的写都已完成；② 我退出前，所有 rank 都已读完我的 sendbuff，用户之后可以改写 |

- 每个 CTA 只和其他 rank 上的同号 CTA 同步就够了：第 b 号 CTA 处理的 chunk 是 `{W : (W / P) mod nBlocks == b}`，各 rank 的第 b 号 CTA 合起来恰好覆盖这一批 chunk 的全部读写，而 kernel 要等所有 CTA 都结束才算完成。
- in-place（sendbuff == recvbuff）也安全：chunk c 只有 owner 会读写，而且 owner 的同一个线程先读完 P 份再写 P 份。
- memory order：开头用 relaxed，因为它只传递"进入 kernel"这一控制信息，之前 kernel 写下的数据靠 kernel 边界保证可见【推断】。结尾的 release 让本 CTA 之前的远端 store（先经过 `__syncthreads()` 汇聚）在 inbox 更新之前对外可见；wait 一侧是 relaxed，读者真正读数据是在 kernel 结束之后，依赖 kernel 完成时的全局可见性【推断】。
- 连续两次调用：下一次调用开头的 barrier 保证"对端已经结束上一次调用并开始了这一次"，epoch 单调递增，不会混淆。

**6. 流水**

- warp 内：本地预取 → 2 个对端 × 4 个 16B 同时在途 → 归约 → P 次 posted store → 预取下一轮本地数据。store 不阻塞后续 load。
- warp 之间、CTA 之间完全独立，没有角色分工。

**7. NVLink 流量**（每张 GPU，每个方向约 `2(P-1)/P × N`，与 Ring 相同）

| 方向 | 来源 | 流量 |
|:---|:---|:---|
| Switch → GPU | 我读取对端的数据（load 响应） | `(P-1)/P × N` |
| Switch → GPU | 对端把结果写进我的 recvbuff | `(P-1)/P × N` |
| GPU → Switch | 对端读取我的 sendbuff | `(P-1)/P × N` |
| GPU → Switch | 我把结果写到对端 | `(P-1)/P × N` |

SM 归约量为每个 rank `N`（P 份 × N/P），没有冗余。

**8. 示例：P = 8 与 P = 16（强制使用，N = 128MiB bf16）**

| | P = 8 | P = 16 |
|:---|:---|:---|
| 模型 nBlocks（上限 64） | 13 | 13 |
| 全局 warp 数 `wn` | 8 × 13 × 16 = 1664 | 16 × 13 × 16 = 3328 |
| 每个 rank 负责 | 16MiB（chunk 号 ≡ r mod 8） | 8MiB（chunk 号 ≡ r mod 16） |
| 每个 chunk 的 load / store | 8 份 load、8 次 store | 16 份 load、16 次 store |
| 每个方向的 NVLink 流量 | 1.75N | 1.875N |

---

##### 2.3.1.3 `AllReduce_RSxLDMC_AGxSTMC`

**1. 一句话概括**

two-shot 的多播版：每个 warp 负责一个 4KB chunk，用 `multimem.ld_reduce` 读 sendbuff 的 MC 地址，由 NVSwitch 把 P 个 rank 上同一位置的数据求和后返回；SM 不做任何加法，直接用 `multimem.st` 把结果写到 recvbuff 的 MC 地址，由 NVSwitch 复制到 P 个 rank。RS 和 AG 在同一个线程里按 chunk 融合。kernel 首尾各有一次**多播** LSA barrier。

选择条件：send 和 recv 都必须注册；`hasLsaMultimem`（P > 2）；bf16 sum 支持 LDMC。有多播时，中等以上的消息（P = 8 约 ≥ 52KiB，P = 16 约 ≥ 26KiB，P = 72 约 ≥ 6KiB）都选它。

**2. 关键参数**

| 参数 | 值 | 源码 |
|:---|:---|:---|
| 线程数 | 512 / CTA | `sym_kernels.cc:611` |
| CTA 数上限 | `divUp(32, P)`：P = 8 → 4，P = 16 → 2，P = 72 → 1 | `sym_kernels.cc:319, 341` |
| pack | 16B（`LoadMultimem_BigPackSize`：浮点 sum 且 `sizeof(T) < 8` 时为 16） | `reduce_kernel.h:973-984`，`all_reduce.cuh:306` |
| unroll | `UnrollPacks = 128 / 16 = 8`，warp 每次处理 8 × 32 × 16B = **4KB** | `all_reduce.cuh:312-313` |
| 全局 warp 号 | 与 RSxLD 相同，`W = rank + P × (block + nBlocks × warp)` | `all_reduce.cuh:368-372` |
| 尾部 | 不足 4KB 的部分逐元素处理：`ld_reduce` 标量 bf16x2（读对齐的 4B 后截出 2B），`st.global.b16` | `all_reduce.cuh:341-346` |
| 累加 | 由交换机完成，`.acc::f32`；SM 侧 `Red<bf16>` 只用来选指令 | `all_reduce.cuh:356`，`reduce_kernel.h:844-852` |
| 每个 CTA 在途的 `ld_reduce` | 16 warp × 4KB = 64KB | — |

**3. 使用的内存**

- sendbuff / recvbuff：`input.multimemPtr(lsaMultimem)` 和 `output.multimemPtr(lsaMultimem)`（`all_reduce.cuh:374`），即 `mcBasePtr + (bigOffset >> 12) × 4KB + offset`。整个 kernel 的数据面**只用 MC 地址**。
- barrier：resource window 中的 `mcInbox[blockIdx]`（arrive 用其 MC 地址，wait 用本卡 UC 地址）和 `epochMC[blockIdx]`。
- 不使用 LL scratch，没有中转缓冲区。

**4. 执行流程**

```text
# rank r 的 CTA b，warp k，lane l；W = r + P*(b + nBlocks*k)；wn = P*nBlocks*16
bar = LsaBarrierSession(index=b, multimem=true)
bar.sync(relaxed):                                    # all_reduce.cuh:362，完整的 arrive + wait
    __syncthreads()
    tid 0: multimem.red.relaxed.sys.add.u32 [MC mcInbox[b]], 1
    tid 0: 自旋 ld.relaxed.sys.u32 [本卡 mcInbox[b]] 直到 >= epoch + P;  epoch += P
    __syncthreads()
c = W                                                 # chunk 号，4KB 一个
while c < nChunks:                                    # all_reduce.cuh:321-334
    for u in 0..7:  v[u] = multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2
                           [MC(sendbuff) + c*4KB + u*512 + l*16]     # 8 条独立 load 同时发出
    for u in 0..7:  multimem.st.global.v4.f32 [MC(recvbuff) + c*4KB + u*512 + l*16], v[u]
    c += wn
# 尾部（< 4KB）：每个元素由一个线程处理
    x = multimem.ld_reduce...add.acc::f32.bf16x2 [MC(sendbuff) + (off & ~3)]，截取 2B
    st.global.b16 [MC(recvbuff) + off], x              # 普通 st 写 MC 地址
bar.sync(release):                                    # all_reduce.cuh:378
    __syncthreads()
    tid 0: multimem.red.release.sys.add.u32 [MC mcInbox[b]], 1
    tid 0: 自旋 ld.relaxed.sys.u32 [本卡 mcInbox[b]] 直到 >= epoch + P;  epoch += P
    __syncthreads()
# 析构：tid 0 写回 epochMC[b]
```

**5. 同步**

| 位置 | 实现 | 作用 |
|:---|:---|:---|
| 开头 sync（relaxed） | 每个 rank 的每个 CTA 由 tid 0 发 1 条 `multimem.red.relaxed.sys.add.u32`，NVSwitch 把 P 张卡上的 `mcInbox[b]` 各加 1；tid 0 轮询本卡计数器直到 `>= epoch + P`，即同号 CTA 全部到齐 | 保证所有 rank 都已进入 kernel，之后才能 `ld_reduce` 对端 sendbuff、`multimem.st` 对端 recvbuff。这里不拆成两半，因为没有可以提前做的本地工作（`ld_reduce` 本身就要读所有 rank） |
| 中间 | **无** | 与 RSxLD 相同：每个 chunk 只属于一个 rank，`ld_reduce` 和 `st` 由同一个线程完成，输出是最终位置 |
| 结尾 sync（arrive 为 release，wait 为 relaxed） | 先 `__syncthreads()` 汇聚本 CTA 所有线程的 `multimem.st`，再由 tid 0 发 `multimem.red.release.sys` | ① 计数器到达 `epoch + P` 时，所有 rank 的同号 CTA 都已发完 `multimem.st`，并且 release 保证这些写先于计数器增量对外可见；② 所有 rank 都已读完我的 sendbuff |

- release 的范围：`multimem.red.release.sys` 本身只对发出它的线程（tid 0）有 release 语义。其他线程的 `multimem.st` 是通过前面的 `__syncthreads()` 排在它之前的（CTA 内的 barrier，再加上 release 的累积性）【推断：依据 PTX 内存模型】。
- 读者一侧是 relaxed 加载，没有 acquire。真正读取 recvbuff 的是 kernel 之后的工作，依赖 kernel 结束时的可见性【推断】。
- 计数器只增不减。快的 rank 如果已经进入下一次调用的开头 barrier 并多加了 1，慢的 rank 用 `>=` 判断，照样能放行。epoch 每次加 P，并持久化在 `epochMC[b]` 中。
- 与 Legacy NVLS 的对比：Legacy 每个 step 在 head 上有 4 次跨卡握手（`ld_reduce.min` 轮询 credit、写 credit）；这里整个 kernel 只有 2 次 barrier，每次每个 CTA 只发 1 条 `multimem.red`，数据量与同步次数无关。

**6. 流水**

与 Legacy NVLS 的 Reduce 组（2.2.1 第 6 节）相同：每个 warp 是"8 条 `ld_reduce` 背靠背发出 → 按顺序 8 条 `multimem.st`（posted）"的小流水线，16 个 warp 并行。不同之处：

- 没有 Scatter/Gather 角色，也没有本地 HBM 拷贝，512 个线程全部用于 multimem。`multimem.st` 直接写入所有卡的用户 recvbuff，接收端零搬运（见 3.7.5）。
- 每个 CTA 在途 64KB（Legacy 约 16KB/CTA × 24 CTA）。但 CTA 数被 `divUp(32, P)` 压得很少：P = 8 时最多 4 个，P = 72 时只有 1 个。
- 没有 step 边界，chunk 之间不需要 fence 或 barrier。

**7. NVLink 流量**（每张 GPU，每个方向约 `(P+1)/P × N`，与 Legacy NVLS 相同，约为 Ring 的一半）

| 方向 | 来源 | 流量 |
|:---|:---|:---|
| GPU → Switch | 本卡 sendbuff 被所有 owner 的 `ld_reduce` 读取 | N |
| GPU → Switch | 本卡作为 owner 发出的 `multimem.st` | N/P |
| Switch → GPU | 本卡 `ld_reduce` 拿回的归约结果 | N/P |
| Switch → GPU | 所有 owner 的 `multimem.st` 写进本卡 recvbuff | N |

- 负载均衡：chunk 按 `W mod P` 轮流分给 rank，粒度为 4KB，每个 rank 负责的量相差不超过一个 chunk。所以 P = 72 时也是 72 个 rank 均分，不存在 Legacy NVLS "最多 32 个 head" 的不对称。
- 本卡 HBM：只有交换机对 sendbuff 的读和对 recvbuff 的写，各 N。没有 Legacy 的 Scatter/Gather 拷贝（Legacy 在 MNNVL 下还要多读写 2N）。

**8. 示例：P = 8 与 P = 16（N = 128MiB bf16）**

| | P = 8 | P = 16 | P = 72 |
|:---|:---|:---|:---|
| `nMaxBlocksNvls` / 模型选择 | 4 / 3 | 2 / 2 | 1 / 1 |
| 每卡 SM 数 | 3 | 2 | 1 |
| 全局 warp 数 `wn` | 8 × 3 × 16 = 384 | 16 × 2 × 16 = 512 | 72 × 1 × 16 = 1152 |
| 4KB chunk 总数 | 32768 | 32768 | 32768 |
| 每个 warp 的迭代次数 | ≈ 85 | 64 | ≈ 28 |
| 每个 rank 负责 | 16MiB | 8MiB | ≈ 1.78MiB |
| 每卡在途 `ld_reduce` | 192KB | 128KB | 64KB |
| `ld_reduce` 的扇入 / `st` 的扇出 | 8 | 16 | 72 |
| 每个方向的 NVLink 流量 | 1.125N | 1.0625N | ≈ 1.014N |
| 模型估计时间（非实测） | ≈ 159µs | ≈ 151µs | ≈ 145µs |

P 增大时，链路流量略有下降，每卡 CTA 数却减少，单卡在途请求也随之减少。能否在 P = 16 / 72 时只用 1~2 个 SM 跑满 NVLink，取决于交换机侧 `ld_reduce` 的延迟和单 SM 能维持的在途量，需要实测【推断】。

---

**汇总**

| | AGxLL_R | AGxLLMC_R | RSxLD_AGxST | RSxLDMC_AGxSTMC |
|:---|:---|:---|:---|:---|
| 算法 | one-shot | one-shot | two-shot（融合） | two-shot（融合） |
| 需要注册 | 否（全都不注册时默认回退） | 否（同左） | send 和 recv 都要 | send 和 recv 都要 |
| CTA 数上限（P = 8/16/72） | 64 | 64 | 64 | 4 / 2 / 1 |
| 中转缓冲区 | LL scratch（P × 512 槽 × 2 半区 × 64 block） | 同左 | 无 | 无 |
| 首尾 barrier | **无** | **无** | 单播 inbox（开头拆成两半） | 多播 `multimem.red` |
| 中间同步 | 每个 16B 包带 epoch flag | 同左 | 无 | 无 |
| 归约 | SM，fp32，每个 rank 归约全部数据 | 同左 | SM，fp32，每个 rank 归约 1/P | NVSwitch，`.acc::f32` |
| 每卡每个方向的 NVLink 流量 | 约 2(P-1)N | 出 2N，入约 2PN | 2(P-1)/P·N | (P+1)/P·N |

#### 2.3.2 AllGather

Symmetric AllGather 有四个 kernel：`AllGather_ST`、`AllGather_STMC`、`AllGather_LL`、`AllGather_LLMC`（`device/symmetric/all_gather.cuh:167, 195, 327, 331`）。`AllGather_RailRing_LsaSTMC` 是跨 NVLink 域的 GIN kernel：`needGin = lsa.nRanks < nRanks` 为假时会被屏蔽（`sym_kernels.cc:553-554`），本节不讨论。

下文中 P 表示通信组大小，N 表示每个 rank 的 sendbuff 字节数（recvbuff 为 P·N）。AllGather 在入队时会把 count 换算成字节、把 datatype 改成 `ncclInt8`（`enqueue.cc:2611-2614`），所以 kernel 里的元素类型 `T` 是 `char`，与用户使用 bf16 还是其他类型无关。

**一句话概括**

| kernel | 算法 | 数据平面 | 同步 |
|:---|:---|:---|:---|
| `AllGather_ST` | 每个 rank 把自己的 N 字节用单播 `st` 直接写进 P-1 个对端 recvbuff 的第 rank 段 | 单播 16B `st.global`（NVLink） | 首尾各一次**单播** LSA barrier |
| `AllGather_STMC` | 每个 rank 用 `multimem.st` 把 N 字节写到 recvbuff 第 rank 段的**多播地址**，由 NVSwitch 复制到 P 张卡 | `multimem.st.global.v4.f32` | 首尾各一次**多播** LSA barrier（`multimem.red`） |
| `AllGather_LL` | 每 4KB 一轮：把 8B 数据加上 epoch flag 打包成 16B，写进 P 个 rank 的 LL scratch；各 rank 轮询本卡 scratch 的 flag 后拷进 recvbuff | 单播 `st.relaxed.sys.v4.u32` | **没有 barrier**，靠逐 16B 的 epoch flag，加上按 epoch 奇偶切换的双缓冲 |
| `AllGather_LLMC` | 同 LL，但 16B 的 LL 行只写一次，写到 scratch 的**多播地址** | 对 MC 地址执行普通 `st.relaxed.sys.v4.u32`（**不是** `multimem.st`） | 同 LL |

**0. 选择规则（host 侧）**

1. **可用性**：`ncclSymkAvailable` 只检查 `isAllDirectNvlink`、op/type 是否实现（AllGather 全部实现），以及 kernel 掩码是否非空（`sym_kernels.cc:485-510, 558-566`）。它**不检查注册**。此外前提是 `comm->symmetricSupport`（`enqueue.cc:397`）。
2. **多播前提**：`hasLsaMultimem = nvlsSupport && lsa.nRanks > 2`（`sym_kernels.cc:414`）。条件不满足时 STMC 和 LLMC 被屏蔽（`sym_kernels.cc:540`）。
3. **LL 的限制**：`nBusBytes = P·N ≥ 2GiB` 时屏蔽 LL 类 kernel（LL 用 32 位 int 计数，`sym_kernels.cc:544-546`）；`nWorks > 1`（group 内合并了多个 AllGather）时也屏蔽 LL（`sym_kernels.cc:577-578`）。
4. **注册**：只有在 recvbuff 已注册的情况下（`SendRegRecvReg` 或 `SendNonregRecvReg`），ST/STMC 才保留；否则只剩 LL 类（`sym_kernels.cc:583`）。ST/STMC 只通过 `input.localPtr()` 读本卡的 sendbuff，所以 **sendbuff 不需要注册**（`all_gather.cuh:25`、`primitives.cuh:238`）。
5. **代价模型**（`queryModel_lsa`，`sym_kernels.cc:314-404`；sm100 的参数为 `smBw = 55GB/s`、`peakBw = 600GB/s`、`baseLat` LL=8.5us、非 LL=13us）：

   | kernel | busBytes | busMultiplier | CTA 上限 |
   |:---|:---|:---|:---|
   | `AllGather_LL` | `9·P·N`（`LL_BusFactor = 9`） | 1 | 64 |
   | `AllGather_LLMC` | `9·P·N` | 1.1（注释："To beat non-MC LL"） | 64 |
   | `AllGather_ST` | `(P-1)·N` | 1 | 64 |
   | `AllGather_STMC` | `(P-1)·N`（注释承认应为 `P·N`，故意写小以便赢过非 MC） | `0.55·P` | `divUp(32, P)` |

   对每个 kernel 先算出用满 CTA 上限时的耗时，然后选出耗时不超过它 1.025 倍的最少 CTA 数（`sym_kernels.cc:393-403`）；kernel 之间按 `time × (1 + 0.025 × nBlocks)` 比较（`sym_kernels.cc:594-606`）。CTA 上限还受 `NCCL_MIN_CTAS` / `NCCL_MAX_CTAS` 约束（默认 1 / 64），`NCCL_SYM_CTAS` 可以直接把 CTA 数固定下来（`sym_kernels.cc:373-377`）。
6. **回退规则**（`scheduler/symmetric_sched.cc:104-121`）：
   - send 和 recv 都注册：不回退。
   - 选中的不是 LL kernel（ST/STMC，即 recv 已注册但 send 没有）：也不回退。
   - 选中 LL kernel，且 send 和 recv 都没注册：默认（`NCCL_SYM_NOWIN_ENABLE=0`，`symmetric_sched.cc:15`）**一律回退** Legacy。
   - 选中 LL kernel，只注册了一侧：查询 Legacy 代价模型，只要它选出的协议不是 LL 就回退。回退到的是 Legacy 调度器选出的算法（Ring / NVLS / …），不一定是 Ring。
   - 设置了 `NCCL_SYM_KERNEL` 强制指定 kernel 时，跳过 Legacy 查询。
7. **CE 抢先**：Blackwell 上，如果 recv 已注册，且 `count > NCCL_SYM_CE_THRESHOLD`（默认 8M 个**用户元素**，bf16 即 N > 16MiB），AllGather 在入队阶段就直接走 CE，不会进入 symmetric kernel 的选择流程（`enqueue.cc:31, 2966`）。

按上述公式手算的选择结果如下（bf16，send 和 recv 都注册，GB200，NVLS 可用；**只是模型输出，未经实测**）：

| N（每 rank） | P = 8 | P = 16 | P = 72 |
|:---|:---|:---|:---|
| 4KB | LLMC，模型 5 个 CTA（实际 grid 为 4，见下文） | LLMC，模型 7 个（实际 grid 4） | STMC，1 个 |
| 16KB | LLMC，9 个（实际 grid 8） | STMC，1 个 | STMC，1 个 |
| 64KB | STMC，2 个 | STMC，1 个 | STMC，1 个 |
| ≥ 256KB | STMC，3 个 | STMC，2 个（≥ 1MiB 时） | STMC，1 个 |
| 不可用 MC 时的 ≥ 1MiB | ST，12～13 个 | ST，12～13 个 | ST，12～13 个 |

LL → STMC 的切换点大约在 P·N ≈ 256KB 附近（P=8 时约为 32KB，P=16 时约为 16KB，P=72 时约为 4KB），因为 LL 的 busBytes 正比于 P·N。

**1. 公共参数：CTA、线程、数据切分**

| 参数 | 值 | 源码 |
|:---|:---|:---|
| 每个 CTA 的线程数 | 固定 16 个 warp = 512 线程（`nWarps = 16`） | `sym_kernels.cc:611`、`symmetric_sched.cc:184`、`sym_kernels.h:22` |
| CTA 数上限 | 非 MC 为 `ncclSymkMaxBlocks = 64`；STMC 为 `divUp(32, P)`：P=8 时为 4，P=16 时为 2，P=72 时为 **1** | `sym_kernels.h:21`、`sym_kernels.cc:319, 357` |
| host 切分粒度 | 1KB 的 cell（`NCCL_SYM_KERNEL_CELL_SIZE`）。每个 channel 分到 `cellPerChannel = ⌈⌈total/nBlocks⌉/cellCount⌉` 个 cell，实际 grid 为用到的 channel 数（`channelMask`），所以小消息的 grid 可能比模型给出的 CTA 数少 | `sym_kernels.h:19`、`symmetric_sched.cc:203, 283` |
| ST/STMC 的 CTA 间切分 | `forEachWork`：一个 work 被拆到多个 channel 上时（"fused"），每个 CTA 拿到的都是**整个 work 范围**，再用 `t = lane + 32·(block + nBlocks·warp)` 给全局线程编号。全局 warp id 为 `block + nBlocks·warp`，因此相邻的 warp 粒度数据块**轮流**分给不同 CTA，而不是每个 CTA 处理一段连续区间 | `primitives.cuh:116-131, 134-161`、`all_gather.cuh:181-184, 208-211` |
| LL 的 CTA 间切分 | `singleWork`：每个 CTA 处理一段**连续**的 cell 区间 `[indexLo, indexHi)` | `primitives.cuh:92-113, 164-175` |

**2. 内存：地址如何得到**

| 地址 | 来源 | 指向 |
|:---|:---|:---|
| 用户 recvbuff 的单播对端地址 `output.lsaPtr(r)` | `lsaFlatBase + r·stride4G + offset`（`core__funcs.h:114-119`），其中窗口的 `lsaFlatBase` 已加上 `bigOffset`（`dev_runtime.cc:545`） | rank r 的物理显存；r == 本 rank 时是本卡 |
| 用户 recvbuff 的多播地址 `output.multimemPtr(lsaMultimem)` | `mcBasePtr + mcOffset4K·4096 + offset`（`core__funcs.h:144-148`、`dev_runtime.cc:546`）；物理页由 `cuMulticastBindMem(mcHandle, bigOffset, …)` 绑定（`dev_runtime.cc:247`） | 所有成员卡同一偏移处的物理显存（包括本卡） |
| 未注册的 buffer | `ncclSymPtr` 的 window 为 null 时，`localPtr()` 直接返回原始指针（`ptr__funcs.h:95-101`） | 本卡 |
| LSA barrier 状态 | 通信器私有的 resource window，大小为 `(3·nB + nB·P)·4B`，其中 `nB = 64`（每个 CTA 一个 barrier）。初始化时清零 | `nccl_device/lsa_barrier.cc:18`、`dev_runtime.cc:934, 1008`、`sym_kernels.cc:416` |
| LL scratch（`lsaLLA2A`） | 同一个 resource window。共 `64 × (1 + 2·nSlots) × 16B`，`nSlots = 512·P`（`ncclLLA2ACalcSlots(P·512, 8)`）。**每卡约 P MiB**：P=8 时 8MiB，P=16 时 16MiB，P=72 时 72MiB | `sym_kernels.cc:419-422`、`nccl_device/ll_a2a.cc:12-14, 23` |

barrier 状态区的布局（`lsa_barrier__types.h:30-43`、`lsa_barrier__funcs.h:30`）：

```text
uint32 state[]:
  [0      , nB)          多播 barrier 的 epoch（每个 CTA 一个，跨 kernel 持久化）
  [nB     , 2nB)         单播 barrier 的 epoch
  [2nB    , 3nB)         多播 inbox：mcInbox(index)。通过 MC 地址 red.add，再通过 UC 地址轮询本卡这一份
  [3nB    , 3nB + nB·P)  单播 inbox：ucInbox(owner, peer) = owner 卡上的 [index][peer]
```

LL scratch 中每个 CTA 一段（`ll_a2a__funcs.h:25-27`、`ll_a2a__types.h:32-34`）：

```text
block b: [ header line (uint4, .x = epoch-2) | half0: nSlots × uint4 | half1: nSlots × uint4 ]
         当前 epoch 使用的半区 = (epoch & 1)
```

**没有 FIFO**：ST/STMC 直接写对端的用户 recvbuff（零拷贝）；LL/LLMC 只有一块按 epoch 奇偶切换的双缓冲 scratch，没有 head/tail credit。

**3. 同步机制详解（四个 kernel 共用）**

*(a) `ncclLsaBarrierSession`：每个 CTA 一个 barrier*

构造时 `index = blockIdx.x`，`team = LSA team`（NVL72 场景下就是全部 P 个 rank）（`lsa_barrier__funcs.h:36-41`、`all_gather.cuh:169-171, 197-199`）。所以这**不是整卡 barrier**，而是"各 rank 上 blockIdx 相同的 CTA 之间的 barrier"。这要求所有 rank 选出相同的 CTA 数，而代价模型是确定性的，所以能满足。

```text
构造:  epoch = state[(multimem?0:1)·nB + index]          # 本卡读，lsa_barrier__funcs.h:29-30
析构:  thread0: state[(multimem?0:1)·nB + index] = epoch;  __syncthreads   # :46-58，跨 kernel 持久化

arrive(order):                                           # :63-84
  __syncthreads                                          # CTA 内所有线程此前的写，先在 CTA 范围内排好序
  多播版:  thread0:  multimem.red.{release|relaxed}.sys.add.u32 [MC mcInbox(index)], 1   # :70 / :72
           → NVSwitch 把 +1 送到 P 张卡上 mcInbox(index) 这一份（包括本卡）
  单播版:  thread i (i < P-1), peer = i + (rank ≤ i):
           atomic_ref(peer 卡上的 ucInbox[index][myRank]).store(epoch+1, releaseOrderOf(order))
           → st.{release|relaxed}.sys.u32，经 NVLink 写对端【推断：atomic_ref 默认 system scope】

wait(order):                                             # :89-118
  多播版:  只有 thread0 参与：
           spin ld.{acquire|relaxed}.sys [本卡 UC mcInbox(index)]，直到 got ≥ epoch + P（回绕安全的比较 :99）
           epoch += P                                    # :101
  单播版:  thread i (i < P-1) 并行地轮询本卡 ucInbox[index][peer_i]，直到 ≥ epoch + 1
           epoch += 1                                    # :115
  __syncthreads                                          # :117，放行整个 CTA
  轮询每 10000 次检查一次 abortFlag（utility.h:80-88）
```

- **计数器从不清零**：多播 inbox 是单调递增的累加计数器，每轮 barrier 加 P；单播 inbox 存的是"对端最新的 epoch"，用 ≥ 比较。epoch 保存在本卡 `state` 中，跨 kernel 延续，所以**连续两次调用之间不需要重置，也不会把上一次的到达计成这一次的**。
- **内存序参数**：`releaseOrderOf(release) = release`，`acquireOrderOf(release) = relaxed`（`utility.h:309-318`）。所以 `bar.sync(release)` 的 arrive 带 release，wait 却只是 **relaxed 轮询**；`bar.sync(relaxed)` 两侧都是 relaxed。
- **代价**：单播版每次 arrive 是 P-1 个远端 store，wait 是 P-1 个线程并行轮询；多播版每次只有 1 条 `multimem.red` 和 1 个线程轮询，与 P 无关。

*(b) ST / STMC 的两次 barrier 各自保证什么*

| 位置 | ST（单播 barrier） | STMC（多播 barrier） | 作用 |
|:---|:---|:---|:---|
| 开头 | `bar.arrive(relaxed)`（`all_gather.cuh:174`）；`bar.wait(relaxed)` **推迟**到第一批本地 load 发出之后（`all_gather.cuh:37, 160`） | `bar.sync(relaxed)`，arrive 后立即 wait（`all_gather.cuh:202`） | **WAR 保护**：对端 CTA 已经进入本 kernel，说明对端 stream 上之前读写 recvbuff 的工作都已完成，现在可以覆盖它的 recvbuff。这里不需要发布任何数据，所以用 relaxed |
| 结尾 | `bar.sync(release)`（`all_gather.cuh:192`） | `bar.sync(release)`（`all_gather.cuh:217`） | **RAW 发布**：arrive 带 release，保证本 CTA 此前写到对端的数据，先于 inbox 被对端看到。wait 返回后，本 CTA 负责的那部分数据，所有 rank 都已写进本卡 |

- 结尾 wait 是 relaxed，kernel 内也不再读取收到的数据。数据的消费者是 stream 上后续的 kernel，依赖的是 kernel 边界的可见性【推断】。
- release 的覆盖范围：arrive 先执行 `__syncthreads`，再由 thread0 发出 release。按照 PTX 内存模型中 release 的累积性，CTA 内其他线程此前的写也被包含在内【推断，基于 PTX 内存模型】。对于 STMC，被 release 的是此前的 `multimem.st`（`.sys` 作用域）。
- **kernel 中间没有任何跨 rank 同步**，也没有逐块流控：只要两端的 CTA 都已进入 kernel，就可以一次写完全部数据。

*(c) LL 协议：逐 16B 的 epoch flag，加双缓冲*

```text
LL 行（16B）:  { x = data[0:4], y = epoch, z = data[4:8], w = epoch }      # asm "{%1,%3,%2,%3}", ll_a2a__funcs.h:54/83
发送:  st.relaxed.sys.v4.u32  [peer 的 scratch 或 MC scratch + (slotsOffset + slot)·16]
接收:  ld.relaxed.sys.v4.u32  [本卡 scratch + (slotsOffset + slot)·16]     # sm90 上是 ld.acquire（:184），sm100 上是 relaxed（:192）
       直到 y == epoch && w == epoch                                       # :215-216
```

- **flag 怎么来**：不需要额外的 flag 字段，flag 就是 CTA 自己维护的 `epoch`。构造时 `epoch = header.x + 2`，析构时写回 `epoch - 2`（`ll_a2a__funcs.h:25-28, 34-39`）；每处理完一轮（一个 4KB 迭代），`endEpoch` 执行 `__syncthreads; epoch += 1`（`:282-283`）。
- **为什么不需要 fence**：数据和 flag 在同一个 8B 半行里（{x,y}、{z,w}）。即使 16B 的 store 被拆成两次 8B 写入，每一半也各自带着 flag，接收方分别校验，所以不依赖 16B 原子性【推断：依赖 8B 写不会被撕裂】。
- **为什么不需要 head/tail credit（没有 FIFO 仍然正确）**：epoch e 使用半区 `e & 1`（`ll_a2a__types.h:33`）。发送方在 epoch e+2 会再次写同一个半区；而它能进入 e+2，说明 e+1 已经结束，也就是**已经收到每个对端在 e+1 发来的数据**。对端只有在完成 `endEpoch(e)` 之后才会发送 e+1 的数据（`endEpoch` 前有 `__syncthreads`，所有线程都已读完 e 的数据），所以 e 半区已经没人再读，可以覆盖。接口注释要求每个 epoch 内必须"对每个 peer 至少 send 一次，并且至少从每个 peer 收一次"（`ll_a2a.h:46-50`）。AllGather 的 `bcast` 发给所有 rank，接收时读取所有 rank 的槽，满足这一约定。
- **旧数据不会被误认**：e-2 留下的行，flag 是 e-2 ≠ e。scratch 初始为 0，epoch 从 2 开始，回绕时跳过 0 和 1（`:283`）。快回绕时（`epoch ≥ -2u`）先清零本 CTA 的 slot（`:273-281`）。
- **连续两次调用**：epoch 按 CTA 持久化在 header 中，下一次调用从 e+1 继续，半区自然交替，上面的论证跨 kernel 同样成立。AllReduce / AllGather / ReduceScatter 的 LL kernel 共用这块 scratch 和 header（`handler.lsaLLA2A`），各 rank 执行的 LL kernel 序列相同，所以各 rank 上同一个 CTA 的 epoch 保持一致。
- **不需要开头和结尾的 barrier**：LL kernel 只读本卡 sendbuff、写本卡 recvbuff，跨卡只写对端的 scratch，而 scratch 的复用已经由 epoch 保证。所以 LL kernel **完全没有 LSA barrier**，这是它在小消息时延迟低的主要原因（模型中 baseLat 为 8.5us，非 LL 为 13us）。

**4. `AllGather_ST`**

参数（`bcast`，`all_gather.cuh:109-165`）：

| 阶段 | 条件 | 粒度 |
|:---|:---|:---|
| 16B 主体 | `(input.offset - output.offset) % 16 == 0` | 16B pack，每个 warp 每次迭代处理 `4 pack × 32 lane × 16B = 2KB`，`UnrollPeers = 2`。块数按 8KB（4 个 warp）计算，**向下取整到 nBlocks 的倍数**（`:125-128`） |
| 4B 主体 | 剩余部分且 4B 对齐 | 4B pack，`UnrollPeers = 4`（`:142-157`） |
| 头尾 | 不足一个块的部分 | 逐字节，`UnrollPeers = 8`（`bcastEnds`，`:76-106`） |

```text
# 每个 CTA（512 线程），全局线程号 t = lane + 32·(block + nBlocks·warp)
arrive: __syncthreads; thread i<P-1: st.relaxed.sys.u32 [peer_i 的 ucInbox[blk][me]], epoch+1   # 远端 UC
for 本 warp 负责的每个 2KB 块 (w, w+wn, ...):
    4 × ld.global.v4  [本卡 sendbuff + off]                  # 本卡 HBM；第一批在 wait 之前发出
    (仅第一次) wait: thread i<P-1 自旋 ld.relaxed.sys [本卡 ucInbox[blk][peer_i]] ≥ epoch+1; __syncthreads
    r = rank (+1 if in-place); 依次遍历 P 个（in-place 时 P-1 个）rank，每次 2 个:
        4 × st.global.v4  [lsaPtr(r) → rank r 的 recvbuff + rank·N + off]   # 远端 UC（r==rank 时为本卡）
    下一个块的 4 × ld.global.v4（在上一批 st 之后发出，st 是 posted write，不阻塞）
end: __syncthreads; thread i<P-1: st.release.sys.u32 [peer_i 的 ucInbox[blk][me]], epoch+2
     thread i<P-1 自旋 ld.relaxed.sys [本卡 ucInbox[blk][peer_i]] ≥ epoch+2; __syncthreads
     thread0: state[nB+blk] = epoch                                  # 析构，持久化
```

- 起始对端为 `rank+1` 或 `rank`，各 rank 错开，避免所有 rank 同时写同一个对端（`all_gather.cuh:41-43, 88-90`）。
- in-place（`sendbuff == recvbuff + rank·N`，且位于同一窗口）时跳过向自己的拷贝（`:114, :41`）。
- **流水**：每个 lane 一次最多有 4 pack × 2 peer = 8 个 16B store 在途，下一块的 load 与在途 store 重叠；第一批 load 与开头 barrier 的等待重叠。
- **观察**：16B 主体按 nBlocks 的倍数向下取整，所以每个 CTA 都有 ≤ nBlocks×8KB 的数据退化到 4B 或 1B 路径。CTA 数为 12 时，这部分最多 96KB，对 MB 级消息影响很小。

**5. `AllGather_STMC`**

参数（`bcastMultimem`，`primitives.cuh:233-281`）：16B pack，`UnrollPacks = 8`，每个 warp 每次迭代处理 `8 × 32 × 16B = 4KB`，块数**不**按 CTA 取整。前缀和后缀逐字节处理，此时 `multimem_st_global<1>` 退化成对 MC 地址的普通 `st.global.b8`（`op128.h:383-384`）。如果 `(input - output) % 16 != 0`，整段都按逐字节处理（`primitives.cuh:268-271`）。

```text
sync(relaxed, MC):  __syncthreads; thread0: multimem.red.relaxed.sys.add.u32 [MC mcInbox[blk]], 1
                    thread0 自旋 ld.relaxed.sys [本卡 mcInbox[blk]] ≥ epoch+P; epoch += P; __syncthreads
for 本 warp 负责的每个 4KB 块:
    8 × ld.global.v4                   [本卡 sendbuff + off]                     # 本卡 HBM
    8 × multimem.st.global.v4.f32      [MC(recvbuff) + rank·N + off]             # 一次写，NVSwitch 复制到 P 张卡（含本卡）
sync(release, MC):  __syncthreads; thread0: multimem.red.release.sys.add.u32 [MC mcInbox[blk]], 1
                    thread0 自旋 ld.relaxed.sys [本卡 mcInbox[blk]] ≥ epoch+P; __syncthreads
```

- SM 只需要发出 N 字节，与 P 无关；复制由 NVSwitch 完成。所以 CTA 上限 `divUp(32, P)` 随 P 减小：P=72 时只有 1 个 CTA（16 个 warp，每个 lane 最多 8 个 16B 的 `multimem.st` 在途）。代价模型认为此时已经足够（`busMultiplier = 0.55·P`）。实际能否打满入向带宽**需要实测**【推断】。
- 与 ST 不同，开头的 barrier 是立即 wait，没有预取重叠（`all_gather.cuh:202`）。

**6. `AllGather_LL` / `AllGather_LLMC`**

参数（`allgather_LL_body`，`all_gather.cuh:221-294`；会话构造在 `:298-300`，`maxElts = 512`）：

| 项 | 值 |
|:---|:---|
| 每个 CTA 每轮 | 本卡最多 512 个 8B pack，即 4KB（`nIterPacks = min(nPacks, 512)`） |
| 发送 | 线程 t < nIterPacks 读 8B，`bcast` 到槽 `nIterPacks·rank + t` |
| 接收 | 槽 `i ∈ [0, P·nIterPacks)` 平均分给 512 个线程，`peer = i / nIterPacks`，`pack = i % nIterPacks`；每次 4 个槽（`Unroll = 4`）一起轮询 |
| 写回 | 本卡 `recvbuff + peer·N + pack·8`，普通 `st.global`（`storePack`） |
| 对齐 | N、input、output 都 8B 对齐时走 `BytePack<8>` 特化（`:313-319`） |

```text
# 每个 CTA 处理连续区间 [indexLo, indexHi)；epoch = header.x + 2；half = epoch & 1
loop 每 4KB:
  t < nIterPacks:  8B ld.global [本卡 sendbuff + t·8]                               # 本卡 HBM
     LL  : for r in (rank, rank+1, …) 共 P 个，每次 8 个:
               st.relaxed.sys.v4.u32 [rank r 的 UC scratch[half][nIterPacks·rank + t]], {lo,ep,hi,ep}   # P-1 次远端 + 1 次本卡
     LLMC:     st.relaxed.sys.v4.u32 [MC scratch[half][nIterPacks·rank + t]], {lo,ep,hi,ep}          # 1 次，交换机复制到 P 张卡
  all t: for 分到的 P·nIterPacks/512 个槽（每次 4 个）:
     spin ld.relaxed.sys.v4.u32 [本卡 UC scratch[half][i]] 直到 .y == .w == epoch            # 本卡轮询
     8B st.global [本卡 recvbuff + peer·N + pack·8]                                         # 本卡 HBM
  endEpoch: __syncthreads; epoch += 1; half ^= 1
析构: thread0: header.x = epoch - 2
```

- **LLMC 不是 `multimem.st`**：`bcast` 的多播分支对 `bufmc` 执行的是普通 `st.relaxed.sys.v4.u32`（`ll_a2a__funcs.h:78-87`）。PTX 文档规定 MC 地址只能用 `multimem` 指令访问（见 [PTX 9.4 multimem 指令](ptx-9.4-multimem.zh-CN.md)），NCCL 在这里依赖的是"对 MC 地址的普通 store 也会被复制到所有成员"这一实现行为【推断】。
- 不需要注册：只读写本卡用户 buffer，跨卡只写 scratch。默认情况下，两侧都不注册会被回退（见第 0 节），所以实际常见的是"已注册但消息很小"的场景。
- **流水**：同一个 CTA 内各轮之间被 `endEpoch` 的 `__syncthreads` 串行化；不同 CTA 相互独立。轮内的远端 store 是 posted 的，发完立即开始轮询。

**7. NVLink 流量与负载均衡（每卡，逻辑字节）**

| kernel | 出向（GPU → Switch） | 入向（Switch → GPU） | SM 侧 HBM |
|:---|:---|:---|:---|
| ST | (P-1)·N | (P-1)·N | 读 N（非 in-place 时还要写本卡 N） |
| STMC | **N** | P·N（含本卡副本经交换机写回【推断】） | 读 N |
| LL | 2·(P-1)·N（每 8B 数据占 16B） | 2·(P-1)·N | 读 N，写本卡 scratch 2N，读 scratch 2PN，写 recvbuff PN |
| LLMC | **2N** | 2·P·N【推断】 | 同 LL |

- 所有 rank 的角色完全对称：每个 rank 发出自己那一份，接收其余 P-1 份，所以天然负载均衡，不需要像 Legacy NVLS 那样按 head 分片。
- 入向流量对四个 kernel 都至少是 (P-1)·N，是 AllGather 的下限。STMC 的优势在于出向流量和 SM 发出的指令数都从 (P-1)·N 降到 N，代价是入向多出本卡那一份 N。
- 单播 ST 每张卡同时向 P-1 个对端写。各 rank 的起始对端错开（rank+1），避免热点。

**8. P = 8 与 P = 16 的对比**

例 1：N = 1MiB（bf16 时 512K 个元素），send 和 recv 都注册，NVLS 可用 → STMC。

| | P = 8 | P = 16 | P = 72 |
|:---|:---|:---|:---|
| CTA 数（模型）/ 上限 | 3 / 4 | 2 / 2 | 1 / 1 |
| 全局 warp 数 | 48 | 32 | 16 |
| 4KB 块数 / 每个 warp 的迭代数 | 256 / ≈ 5.3 | 256 / 8 | 256 / 16 |
| 每个 lane 每次迭代 | 8 × `ld.v4` + 8 × `multimem.st.v4` | 同左 | 同左 |
| 出向 / 入向 | 1MiB / 8MiB | 1MiB / 16MiB | 1MiB / 72MiB |
| barrier | 每个 CTA 首尾各 1 条 `multimem.red` + 1 个线程轮询 | 同左 | 同左 |

如果 NVLS 不可用（只能用 ST）：P=8 和 P=16 时模型都给 12 个 CTA（192 个 warp）。每个 2KB 块，P=8 时要做 7 组对端 store，P=16 时要做 15 组；出向流量从 7MiB 增加到 15MiB。barrier 中远端 store 和轮询线程从 7 个增加到 15 个。

例 2：N = 4KB → LLMC（P=8 和 P=16 都是），grid 为 4（4 个 1KB cell）。

| | P = 8 | P = 16 |
|:---|:---|:---|
| 每个 CTA 每轮 | 128 pack（1KB），1 轮 | 同左 |
| 发送线程 / 每个线程的 MC store 数 | 128 / 1 | 128 / 1 |
| 接收的槽数（每个线程） | 1024（2 个） | 2048（4 个） |
| 入向（LL 行） | 2·8·4KB = 64KB | 128KB |
| scratch 占用（整个通信器，每卡） | 8MiB | 16MiB |

P 翻倍时，STMC 的 SM 侧工作量不变，但 CTA 上限减半，入向流量翻倍；ST 和 LL 的 SM 侧工作量（store 或轮询的次数）随 P 线性增长。所以 P 越大，代价模型越早从 LL 切换到 STMC（P=8 时约 32KB，P=16 时约 16KB，P=72 时约 4KB）。

#### 2.3.3 ReduceScatter

Symmetric ReduceScatter 在 NVLink 域内有三个 kernel：`ReduceScatter_LD`、`ReduceScatter_LDMC`、`ReduceScatter_LL`（`device/symmetric/reduce_scatter.cuh:231, 312, 395`）。`ReduceScatter_RailA2A_*` 是 GIN kernel，在 `nNodes == 1` 时被屏蔽（`sym_kernels.cc:553-554`），本节不讨论。

下文中 P 表示通信组大小，N 表示每个 rank 的 **recvbuff 字节数**（sendbuff 为 P·N，第 k 段归约后写到 rank k）。以 sum、bf16 为例。CTA、线程数、cell 切分、LSA barrier 以及 LL 协议的实现，与 2.3.2 第 1～3 节相同，这里只列出差异。

**一句话概括**

| kernel | 算法 | 数据平面 | 归约在哪里做 | 同步 |
|:---|:---|:---|:---|:---|
| `ReduceScatter_LD` | 每个 rank 用单播 `ld` 从 P 个 rank 的 sendbuff 读出第 rank 段，在寄存器里求和，写本卡 recvbuff | 单播 16B `ld.global`（NVLink） | SM，f32 累加 | 首尾各一次**单播** barrier，均为 relaxed |
| `ReduceScatter_LDMC` | 对 sendbuff 第 rank 段的**多播地址**发出 `multimem.ld_reduce`，NVSwitch 读取 P 份并求和后返回，再写本卡 recvbuff | `multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2` | NVSwitch，f32 累加 | 首尾各一次**多播** barrier，均为 relaxed |
| `ReduceScatter_LL` | 每 4KB 一轮：把 sendbuff 的第 k 段用 LL 行写进 rank k 的 scratch，再从本卡 scratch 收齐 P 份后求和 | 单播 `st.relaxed.sys.v4.u32`（LL 行） | SM，f32 累加 | 无 barrier，依靠 epoch flag 和双缓冲 |

**0. 选择规则（host 侧）**

1. **类型**：只支持 sum 和浮点类型，不支持 f64（`sym_kernels.cc:485-510`）。实际只生成了 f32、f16、bf16、e4m3、e5m2 的 kernel（`device/symmetric/generate.py:57`）。
2. **LDMC 的前提**：`hasLsaMultimem`（`nvlsSupport && P > 2`），且类型支持 `multimem.ld_reduce`。bf16、f16、f32 都满足；fp8 还要求 `compCap ≥ 100`（`sym_kernels.cc:518-541`），并且 fp8 的 LDMC kernel 只针对 sm100a/101a/120a 等编译（`generate.py:116-119`）。LD 和 LL 的 fp8 只要求 sm90 及以上（`generate.py:113-115`）。
3. **注册**：LD 和 LDMC 读对端 sendbuff，要求 **send 已注册**（`SendRegRecvReg` 或 `SendRegRecvNonreg`）。recvbuff 只在本卡写入（`output.localPtr()`），可以不注册。否则只剩 LL（`sym_kernels.cc:586`）。
4. **LL 的限制**：`P·N ≥ 2GiB`，或者 `nWorks > 1` 时屏蔽 LL（`sym_kernels.cc:544-546, 577-578`）。回退规则与 2.3.2 第 0 节相同（`symmetric_sched.cc:104-121`）：只要选中的是 LL，且 send 和 recv 不是都已注册，就可能回退到 Legacy；两侧都没注册时，默认一律回退。
5. **代价模型**（`sym_kernels.cc:360-370, 389`）：

   | kernel | busBytes | busMultiplier | CTA 上限 | baseLat（sm100） |
   |:---|:---|:---|:---|:---|
   | `ReduceScatter_LL` | `9·P·N` | 1 | 64 | 11us |
   | `ReduceScatter_LD` | `(P-1)·N` | 1 | 64 | 13us |
   | `ReduceScatter_LDMC` | `(P-1)·N`（注释说明应为 `P·N`，故意写小以便赢过非 MC） | `0.55·P` | `divUp(32, P)` | 13us |

按公式手算的选择结果（bf16，send 已注册，GB200，NVLS 可用；只是模型输出，未经实测）：

| N（每 rank 输出） | P = 8 | P = 16 | P = 72 |
|:---|:---|:---|:---|
| 4KB | LL，模型 5 个 CTA（实际 grid 4） | LDMC，1 个 | LDMC，1 个 |
| 16KB | LDMC，1 个 | LDMC，1 个 | LDMC，1 个 |
| 64KB | LDMC，2 个 | LDMC，1 个 | LDMC，1 个 |
| ≥ 256KB | LDMC，3 个 | LDMC，2 个 | LDMC，1 个 |
| 不可用 MC 时的 1MiB | LD，12 个 | LD，12 个 | LD，12 个 |

LL → LDMC 的切换点：P=8 时约 8KB，P=16 时约 4KB，P=72 时约 768B。比 AllGather 更早切换，因为 RS 的 LL 基础延迟为 11us，高于 AG 的 8.5us。

**1. 内存**

| 访问对象 | 地址 | 本卡还是对端 |
|:---|:---|:---|
| LD：对端 sendbuff 的第 rank 段 | `input.peerPtr(world, r)`，即 `lsaFlatBase + (lsaRank + r - rank)·stride4G + offset`（`core__funcs.h:134-140`）；`input` 已加上 `rank·N` | r == rank 时为本卡，其余经 NVLink 访问对端 HBM |
| LDMC：sendbuff 第 rank 段的 MC 地址 | `input.multimemPtr(lsaMultimem) + rank·N`（`reduce_scatter.cuh:333`） | 由 NVSwitch 读取全部 P 张卡（包括本卡） |
| recvbuff | `output.localPtr()`（`reduce_scatter.cuh:96, 333`） | 本卡 |
| LL scratch | 与 AllGather 共用 `lsaLLA2A`，每卡约 P MiB | 发送写对端（或本卡），接收读本卡 |

同样**没有 FIFO**：LD/LDMC 直接读对端的用户 sendbuff（零拷贝）；LL 只使用双缓冲的 scratch。

**2. 同步**

| 位置 | LD（单播 barrier） | LDMC（多播 barrier） | 保证的内容 |
|:---|:---|:---|:---|
| 开头 | `bar.arrive(relaxed)`（`reduce_scatter.cuh:239`）；`bar.wait(relaxed)` 推迟到第一批**本卡**的 load 发出之后（`:40, :223`） | `bar.sync(relaxed)`：`multimem.red.relaxed.sys.add.u32` 加上 thread0 轮询本卡计数 ≥ epoch+P（`:322`） | **RAW**：对端 CTA 已进入本 kernel，说明它 stream 上之前写 sendbuff 的工作都已完成，现在可以读它的 sendbuff |
| 结尾 | `bar.sync(relaxed)`（`:257`） | `bar.sync(relaxed)`（`:337`） | **WAR**：所有 rank 都已读完我的 sendbuff，返回后用户可以覆盖它 |

- 与 AllGather 不同，RS 结尾用的是 **relaxed**：跨卡的只有读，结果写在本卡 recvbuff，没有需要 release 的远端写入。读出的值已经被使用（参与归约并写回），所以在执行 arrive 之前，这些读一定已经完成【推断】。
- 开头只用 relaxed 来保证 RAW：对端 sendbuff 是由之前的 kernel 写入的，可见性依赖 kernel 边界，而不是 barrier 的 acquire【推断】。
- barrier 的实现、epoch 的持久化、为什么连续调用不会误计数：见 2.3.2 第 3 节 (a)。LL 的 flag、双缓冲和 epoch：见 2.3.2 第 3 节 (c)。
- **kernel 中间没有跨 rank 同步**。

**3. `ReduceScatter_LD`**

参数（`reduce`，`reduce_scatter.cuh:169-228`）：

| 阶段 | 粒度 | 取整 |
|:---|:---|:---|
| 16B 主体（`(input.offset - output.offset) % 16 == 0`） | 每个 warp 每次迭代处理 `4 pack × 32 × 16B = 2KB`，`UnrollPeers = 2`（`:190`） | 块（8KB）数**向下取整到 `P·nBlocks` 的倍数**（`:193`） |
| 4B 主体 | 2KB 一块，`UnrollPeers = 4`（`:207`） | 同上（`:210`） |
| 头尾 | 逐元素（bf16 为 2B），`UnrollPeers = 8`（`reduceEnds`，`:112-167`） | — |

```text
# 全局线程号 t = lane + 32·(block + nBlocks·warp)；累加类型 Acc = float（primitives.cuh:208）
arrive(relaxed): __syncthreads; thread i<P-1: st.relaxed.sys.u32 [peer_i 的 ucInbox[blk][me]], epoch+1
for 本 warp 负责的每个 2KB 块:
    acc0 = 4 × ld.global.v4 [本卡 sendbuff + rank·N + off]            # 本卡；第一批在 wait 之前发出（:33-38）
    (仅第一次) wait(relaxed): P-1 个线程轮询本卡 ucInbox ≥ epoch+1; __syncthreads
    r = rank+1:  4 × ld.global.v4 [rank r 的 sendbuff + rank·N + off]   # 远端 UC
                 acc1 = f32(acc0) + f32(tmp)                          # :45-56
    其余 P-2 个 rank，每次 2 个:
                 8 × ld.global.v4 [rank r, r+1 的 sendbuff + rank·N + off]   # 远端 UC，每个 lane 8 个在途
                 acc1 += f32(tmp)                                     # :61-90
    4 × st.global.v4 [本卡 recvbuff + off], bf16(acc1)                # 本卡（:95-96）
    下一个块的本卡 acc0 预取（:104-107）
sync(relaxed): __syncthreads; P-1 个远端 st.relaxed + P-1 个线程轮询
```

- 读对端的顺序从 `rank+1` 开始依次轮转，各 rank 错开，因此任意时刻 P 个 rank 读的是不同的对端，负载均衡（`:45, :58`）。
- **流水**：主循环是 `#pragma unroll 1`。每组 load 之后紧跟着依赖它们的加法，所以一个 warp 处理一个 2KB 块时，大约要经过 `1 + ⌈(P-2)/2⌉` 轮串行的 NVLink 往返（P=8 时 4 轮，P=16 时 8 轮，P=72 时 36 轮）。需要靠大量 warp（模型给出 12～13 个 CTA）来掩盖延迟【推断】。
- **观察：取整粒度会随 P 变粗**。16B 主体按 `P·nBlocks × 8KB` 向下取整：P=8、12 个 CTA 时粒度为 768KB，P=16 时为 1.5MiB。所以当 N = 1MiB、P = 16 时，16B 主体**一块都不做**：768KB 走 4B 路径，256KB 逐元素处理。P = 8 时，768KB 走 16B，192KB 走 4B，64KB 逐元素。对性能的影响需要实测【推断】。

**4. `ReduceScatter_LDMC`**

参数（`reduceMultimem`，`reduce_scatter.cuh:260-309`）：`BytePerPack = LoadMultimem_BigPackSize<FuncSum<bf16>> = 16`（`reduce_kernel.h:980-985`），`UnrollPacks = 8`，每个 warp 每次迭代处理 4KB。块数不按 CTA 取整，余下部分逐元素处理：对 bf16 先对齐到 4B 做 `ld_reduce`，再取出其中一半（`reduce_kernel.h:932-944`）。归约函数的元素类型就是 bf16（`ncclSymkAccumType<…, nvls=true>` 不提升精度，`reduce_scatter.cuh:317`），累加精度由指令的 `.acc::f32` 决定（`reduce_kernel.h:847`）。

```text
sync(relaxed, MC):  __syncthreads; thread0: multimem.red.relaxed.sys.add.u32 [MC mcInbox[blk]], 1
                    thread0 自旋 ld.relaxed.sys [本卡 mcInbox[blk]] ≥ epoch+P; epoch += P; __syncthreads
for 本 warp 负责的每个 4KB 块:
    8 × multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2  [MC(sendbuff) + rank·N + off]
                                         # 每条：NVSwitch 读 P 张卡的同一 16B，求和后返回 16B
    8 × st.global.v4  [本卡 recvbuff + off]
sync(relaxed, MC):  同上
```

- SM 只接收 N 字节的结果，不做加法；每条 `ld_reduce` 的扇入为 P。每个 CTA 最多有 `512 × 8 × 16B = 64KB` 的 `ld_reduce` 请求在途。
- CTA 上限为 `divUp(32, P)`：P=8 时 4（模型选 3），P=16 时 2，P=72 时 **1**。估算如下【推断】：每卡出向的上限约等于 NVLink 单向带宽，需要的结果速率约为"出向带宽 / P"，P 越大越小，所以 CTA 数随 P 减少基本合理。P=72 时，1 个 CTA 能否打满需要实测。

**5. `ReduceScatter_LL`**

参数（`ncclSymkRun_ReduceScatter_LL_body`，`reduce_scatter.cuh:341-392`；会话不带 multimem，`:397-399`，所以**没有 LLMC 版本**）：

| 项 | 值 |
|:---|:---|
| 每个 CTA 每轮 | 输出 ≤ 512 个 8B pack（4 个 bf16 一个 pack，共 4KB），输入为 P × 4KB |
| 发送 | 槽 `i ∈ [0, P·nIterPacks)` 分给 512 个线程，`peer = i / nIterPacks`，`pack = i % nIterPacks`；读本卡 `sendbuff + peer·N + pack·8`，`send(peer, slot = rank·nIterPacks + pack)` |
| 接收 | 只有线程 t < nIterPacks 参与：`recvReduce<Unroll=8>` 读取槽 `t + j·nIterPacks`（j = 0..P-1，即来自 rank j 的数据），每次轮询 8 个槽 |
| 累加 | f32（`Red<float>`，`:400`；`AccPack`，`:347`），最后转回 bf16 写本卡 recvbuff |

```text
loop 每 4KB 输出:
  all t: for 分到的 P·nIterPacks/512 个 (peer, pack):
      8B ld.global [本卡 sendbuff + peer·N + pack·8]
      st.relaxed.sys.v4.u32 [rank peer 的 UC scratch[half][rank·nIterPacks + pack]], {lo,ep,hi,ep}   # P-1 次远端 + 1 次本卡
  t < nIterPacks: 对 j = 0..P-1，每 8 个一组:
      spin ld.relaxed.sys.v4.u32 [本卡 UC scratch[half][t + j·nIterPacks]] 直到 .y == .w == epoch
      acc += f32(data)
      8B st.global [本卡 recvbuff + t·8], bf16(acc)
  endEpoch: __syncthreads; epoch += 1; half ^= 1
```

- 每个 epoch 都满足"向每个 peer 发送、从每个 peer 接收"的约定（`ll_a2a.h:46-50`），所以 2.3.2 中关于双缓冲安全性的论证在这里同样成立。
- 接收端每个线程要轮询 P 个槽，并做 P 次加法。P=72 时，nIterPacks 个线程每个都要处理 72 个槽，这是 LL 在大 P 下很早就被 LDMC 取代的原因之一【推断】。

**6. NVLink 流量与负载均衡（每卡，逻辑字节）**

| kernel | 出向（GPU → Switch） | 入向（Switch → GPU） | SM 侧 |
|:---|:---|:---|:---|
| LD | (P-1)·N（响应对端的读） | (P-1)·N（读回数据） | 读本卡 N，写 N；(P-1)·N 次 f32 加法 |
| LDMC | P·N（交换机为 P 个请求方各读一份，含本卡【推断】） | **N**（已经归约好的结果） | 只做 load 和 store |
| LL | 2·(P-1)·N | 2·(P-1)·N | 读 P·N，写 scratch 2PN，读 scratch 2PN，写 N |

- 三个 kernel 中所有 rank 的角色都对称，每个 rank 只负责自己那一段的归约，天然负载均衡。
- LDMC 把入向流量和 SM 的读取量从 (P-1)·N 降到 N，归约由 NVSwitch 完成；出向仍然是 P·N 量级。它与 STMC 正好镜像：STMC 是出向 N、入向 P·N。

**7. P = 8 与 P = 16 的对比**

例 1：N = 1MiB（每 rank 输出 512K 个 bf16），send 已注册，NVLS 可用 → LDMC。

| | P = 8 | P = 16 | P = 72 |
|:---|:---|:---|:---|
| CTA 数（模型）/ 上限 | 3 / 4 | 2 / 2 | 1 / 1 |
| 全局 warp 数 / 每个 warp 的 4KB 迭代数 | 48 / ≈ 5.3 | 32 / 8 | 16 / 16 |
| 每个 lane 每次迭代 | 8 × `ld_reduce`（扇入 8）+ 8 × `st` | 8 × `ld_reduce`（扇入 16）+ 8 × `st` | 扇入 72 |
| 出向 / 入向 | 8MiB / 1MiB | 16MiB / 1MiB | 72MiB / 1MiB |
| barrier | 每个 CTA 首尾各 1 条 `multimem.red.relaxed` | 同左 | 同左 |

例 2：同样的 N，NVLS 不可用 → LD，12 个 CTA（两种 P 相同）。

| | P = 8 | P = 16 |
|:---|:---|:---|
| 每个 2KB 块的串行往返轮数 | 4 | 8 |
| 16B / 4B / 逐元素的数据量 | 768KB / 192KB / 64KB | 0 / 768KB / 256KB |
| 入向 = 出向 | 7MiB | 15MiB |
| barrier（每个 CTA、每次） | 7 个远端 st + 7 个线程轮询 | 15 + 15 |

例 3：N = 4KB。P=8 时选 LL（grid 4，每个 CTA 1 轮，每轮输出 1KB，即 128 个 pack）：发送端每个线程 2 次 16B 远端 store，128 个接收线程每个轮询 8 个槽。P=16 时代价模型已经改选 LDMC（1 个 CTA）。

P 翻倍时：LDMC 的 SM 侧工作量不变，但扇入翻倍、出向流量翻倍、CTA 上限减半；LD 的串行往返轮数和入向流量大约翻倍；LL 接收端每个线程的轮询和加法次数翻倍。所以 P 越大，代价模型越倾向于选 LDMC，并且更早从 LL 切换过去。

---

### 2.4 Copy Engine

| collective | kernel | instruction | engine | protocol | buffer |
|:---|:---|:---|:---|:---|:---|
| all-reduce | — | — | — | — | **不支持**（CE 不能做归约） |
| all-gather | `ncclCeAllGather` | `cudaMemcpyBatchAsync` / `cudaMemcpyAsync`；同步 flag 写到 MC 地址 | **CE**（无 SM） | 无协议 | recv **要注册**（send 可以不注册） |
| reduce-scatter | — | — | — | — | **不支持** |

**通信模型**

CE collective 是唯一真正不占 SM 的实现。数据传输由 Copy Engine 完成，host 代码直接提交 memcpy 任务，中间不需要 SM 线程参与。

AllGather 的算法是 **direct push（one-shot）**，不是 ring：每个 rank 把自己的 sendbuff 数据**直接**写到其余 P-1 个 rank 的 recvbuff，不经过中转，不转发。和 Ring 的 P-1 步相比，CE AllGather 只有一轮 memcpy 加两次 barrier。

**触发条件**（`enqueue.cc:2919-2966`）：

1. 用户在 comm config 里设置 `CTAPolicy = NCCL_CTA_POLICY_ZERO`。
2. 自动触发：Blackwell（sm100+）、`isAllDirectNvlink`、`symmetricSupport` 都成立，并且每 rank 的元素个数 `count > NCCL_SYM_CE_THRESHOLD`（默认 `8M` 个元素，`enqueue.cc:31`）。

自动触发只对 AllGather 有效，且 count 阈值是**元素个数**不是字节数（bf16 时相当于每 rank 超过 16MB）。两种触发方式都以 `ceAvailable` 为前提：`nNodes == 1`、recvbuff 已注册为 symmetric window、driver ≥ 12.5。这个判断发生在 `collTaskAppend` 之前，所以一旦满足条件，CE 会抢在 symmetric kernel、NVLS、Ring 之前被选中（详见 2.4.2）。

#### 2.4.1 AllReduce

不支持。CE 只能拷贝，无法做归约。

#### 2.4.2 AllGather

**入口**：`ncclLaunchCeColl → ncclCeAllGather`（`ce_coll.cc:685, 472`）。没有 device kernel，所有工作都是 host 线程向**用户 stream** 提交的 CUDA 异步操作。前提如下：

- 所有 rank 在同一个 MNNVL clique 内，`nNodes == 1`，LSA team 覆盖全部 P 个 rank（见 3.1），所以 `ncclDevrGetLsaRankPtr` 可以直接用全局 rank 号作为 LSA rank。
- `comm->nvlsSupport == 1`（GB200 NVL72 上 multicast 可用且 NVLS 拓扑搜索成功），同步走 multicast 分支 `ncclPrepMCSync`。若 NVLS 被关闭（`NCCL_NVLS_ENABLE=0`，或 `init.cc:1273` 因 NVLS graph 为 0 而清零），会退回单播分支 `ncclPrepUCSync`，下文会单独说明。
- 运行时为 CUDA ≥ 12.8、非 graph capture、非 legacy null stream，即走 `cudaMemcpyBatchAsync` 路径。
- P 表示通信组大小，N 表示每个 rank 的字节数（`chunkBytes = nElts × eltSize`，`ce_coll.cc:475`）。

**一句话概括**：一次 one-shot direct push。每个 rank 先做一次 P 方 barrier，确认所有对端都已到达本次调用；然后用一次 `cudaMemcpyBatchAsync` 让 Copy Engine 把自己的 sendbuff 直接写到其余 P-1 个 rank 的 recvbuff 中第 `myRank` 段（通过 symmetric window 的 LSA 单播地址），再做一次 P 方 barrier，确认所有 rank 推给我的数据都已写完。两次 barrier 都是"向 multicast 地址写一个 4 字节序号 + `cuStreamBatchMemOp` 等待 P-1 个 flag"，整条路径**不发射任何 NCCL kernel**。

**0. 选择条件**

`taskAppend`（`enqueue.cc:2885`）中有两个入口，都要求 `ceAvailable` 为真：

| 入口 | 条件 | 源码 |
|:---|:---|:---|
| 强制（任何 CE 支持的 collective） | `comm->config.CTAPolicy & NCCL_CTA_POLICY_ZERO`（comm config 或 `NCCL_CTA_POLICY=ZERO/2`） | `enqueue.cc:2920`、`init.cc:75-112` |
| 自动（仅 AllGather） | 未设置 ZERO，且 `info->count > NCCL_SYM_CE_THRESHOLD`（默认 `8*1024*1024`，**单位是原始 datatype 的元素个数**），`comm->minCompCap >= 100`，`comm->isAllDirectNvlink` | `enqueue.cc:31, 2966` |

`ncclCeAvailable`（`ce_coll.cc:100-118`）：

| 条件 | 源码 |
|:---|:---|
| driver ≥ 12.5（`ncclCeImplemented`，只接受 AllGather / AlltoAll / Scatter / Gather） | `ce_coll.cc:81-98` |
| `comm->nNodes == 1` | `ce_coll.cc:105` |
| `comm->symmetricSupport`（`isAllCudaP2p && NCCL_WIN_ENABLE && cuMem && LSA 覆盖全部 rank`） | `ce_coll.cc:109`、`init.cc:1475` |
| recvbuff 在 `NCCL_WIN_COLL_SYMMETRIC` window 中（`ncclSymSendRegRecvReg` 或 `ncclSymSendNonregRecvReg`）；sendbuff 不要求注册 | `ce_coll.cc:113`、`sym_kernels.cc:644-661` |

补充说明：

- `isAllDirectNvlink` 指所有 GPU 对之间的最长路径类型为 `PATH_NVL`（`graph/paths.cc:971-975`），NVL72 下成立。
- 阈值按元素数比较：bf16 时相当于每 rank **N > 16MB**，fp32 时 N > 32MB。代码注释写的"size > 8MB"（`enqueue.cc:2967`）只对 1 字节类型成立。
- 自动入口在 `collTaskAppend` 之前判断，所以只要满足条件，CE 会**抢在** symmetric kernel / NVLS / Ring 之前被选中，不经过 tuner 的代价模型。
- 入队后，AllGather 的 count 被换算成字节、datatype 改为 `ncclInt8`（`enqueue.cc:2665-2669`），所以 `ceCollArgs->nElts` 就是 N、`eltSize = 1`（`enqueue.cc:1544-1545`）。
- 每个 CE 任务单独成为一个 plan（`enqueue.cc:1538-1561`），由 `doLaunches` 调用 `ncclLaunchCeColl` 而不是 `ncclLaunchKernel`（`group.cc:337-338`）。

**1. 关键参数**

| 参数 | 值 | 源码 |
|:---|:---|:---|
| stream | `comm->planner.streams->stream`，即用户传入的 stream（多个用户 stream 时为第一个，其余通过 event 汇合进来）。**不使用 NCCL 内部的 deviceStream / 独立 CE stream** | `ce_coll.cc:687`、`enqueue.cc:1605-1615` |
| memcpy 个数 `numOps` | out-of-place：P（1 个本地 + P-1 个远端）；in-place（`sendbuff == recvbuff + rank*N`）：P-1 | `ce_coll.cc:488-504` |
| 每个 memcpy 的大小 | N，不按 channel 切分 | `ce_coll.cc:491, 502` |
| 提交顺序 | op0 = 本地拷贝（若有），之后依次为 `(rank+1)%P, (rank+2)%P, …` | `ce_coll.cc:496-497` |
| memcpy 属性 | 1 个 attr 作用于所有 op：`srcAccessOrder = cudaMemcpySrcAccessOrderStream`，`flags = cudaMemcpyFlagPreferOverlapWithCompute` | `ce_coll.cc:371-375` |
| 大消息切块 `intraBatchSync` | `numOps > 8 && N × numOps ≥ 512MiB` 时开启 | `ce_coll.cc:20-22, 507` |
| 切块大小 | `chunkSize = 512MiB / numOps`，`numRounds = ceil(N / chunkSize)`；所有块仍在**同一次** `cudaMemcpyBatchAsync` 中提交 | `ce_coll.cc:377-432` |
| 使用几个 CE | NCCL 不指定。一次 batch 内的 op 如何分配到物理 CE 由 driver 决定【推断】 | — |

`srcAccessOrder = Stream` 表示源数据按 stream 顺序访问，即 sendbuff 可以由同一 stream 上前面的 kernel 产出。`PreferOverlapWithCompute` 是给 driver 的提示：优先选择不与 SM 计算争抢的拷贝方式（即用 CE 而不是 SM copy kernel）【推断：语义来自 CUDA 文档，driver 实际实现不可见】。

**2. 使用的内存**

*数据：recvbuff 所在的 symmetric window*

AllGather 结果在 recvbuff 中的布局是 `[rank0 的 N | rank1 的 N | … | rank(P-1) 的 N]`。rank k 需要把自己的 sendbuff 写到**每个对端** recvbuff 的第 k 段：

```c
// ce_coll.cc:477, 498-499
myRecvBuff = recvBuff + rank * N;                       // 本 rank 的段（本地地址）
offset     = myRecvBuff - recvWin->userPtr;             // 在 window 内的偏移，各 rank 相同
ncclDevrGetLsaRankPtr(comm, recvWin, offset, targetRank, &peerRecvBuff);
// dev_runtime.cc:1461-1478
peerRecvBuff = lsaFlatBase + targetRank * bigSize + recvWin->bigOffset + offset;
```

`lsaFlatBase + r × bigSize + bigOffset` 是 rank r 的这块物理显存在本进程中的单播映射（`symMemoryMapLsaTeam`，见 3.1）。CE 对这个地址的写会经 NVLink / NVSwitch 落到 rank r 的 HBM。数据**只用单播地址**，不使用 multicast：每个对端各发一份。

源地址 `mySendBuff` 是普通本地地址，不需要注册。

*同步 flag：CE 专用的 symmetric window*

第一次使用 CE 时，`ceCollTaskAppend` 把 `ncclCeInit` 放进 `ceInitTaskQueue`（`enqueue.cc:2641-2646`），在 `ncclGroupEnd` 中、launch 之前执行（`group.cc:272-276`）：

```c
// ce_coll.cc:24-49 ncclCeInit
size = alignUp(P*4, 16) * 2;
ncclMemAlloc(&ceDevBase, size);                     // cuMemCreate + cuMemMap，按 granularity 向上取整；不清零
ncclDevrWindowRegisterInGroup(comm, ceDevBase, size, NCCL_WIN_COLL_SYMMETRIC, &win);   // 集合操作
readyPtr    = ceDevBase + 0;                        // uint32_t ready[P]
complPtr    = ceDevBase + alignUp(P*4, 16);         // uint32_t compl[P]
ceSeqNum = 0; useCompletePtr = false;
```

```text
每张 GPU 上同一偏移处（symmetric）:
  ready[0] ready[1] ... ready[P-1] | pad | compl[0] compl[1] ... compl[P-1]
  ready[r] / compl[r]：只由 rank r 写入（通过 MC 写到所有 GPU 的同一位置）
                       每张 GPU 只读本地这一份
```

multicast 地址在第一次同步时惰性获得：

```c
// ce_coll.cc:136-138, dev_runtime.cc:1494-1508
dst      = &ready[myRank]  或 &compl[myRank];
mcDstPtr = tm->mcBasePtr + ceSyncWin->bigOffset + (dst - ceSyncWin->userPtr);
```

`symTeamObtain(..., multimem=true)` 在第一次调用时创建 LSA team 的 multicast 对象：rank 0 `ncclNvlsGroupCreate`，通过 `bootstrapIntraNodeBroadcast` 分发 handle，各 rank `cuMulticastAddDevice` + `cuMemMap` 一段 `bigSize` 的 MC VA，并把所有已有的 symmetric memory 用 `cuMulticastBindMem` 绑定到 `bigOffset` 处（`dev_runtime.cc:265-345`）。这一步包含 bootstrap 通信，只发生一次。

| | P = 8 | P = 16 |
|:---|:---|:---|
| ready 区 / compl 区偏移 | 0 / 32B | 0 / 64B |
| 逻辑大小 | 64B | 128B |
| 物理占用 | 按 `cuMemGetAllocationGranularity(RECOMMENDED)` 取整（通常为 2MB）【推断】 | 同左 |

**3. 执行流程**

host 侧（用户线程，在 `ncclGroupEnd → doLaunches` 中执行）：

```text
# ncclCeAllGather（ce_coll.cc:472-521），stream = 用户 stream
ncclMemOpSync(stream)                       # ① 前置 barrier，见第 4 节
ops = []
if recvbuff + rank*N != sendbuff:
    ops += (src=sendbuff, dst=recvbuff + rank*N, N)               # 本地 D2D
for r in 1..P-1:
    t = (rank + r) % P
    ops += (src=sendbuff, dst=lsaFlatBase + t*bigSize + bigOffset + rank*N, N)   # 远端 D2D
intraBatchSync = len(ops) > 8 and N*len(ops) >= 512MiB

# ncclCeLaunchBatchOps（ce_coll.cc:326-469）
if 正在 graph capture 或 stream 为 legacy null stream 或 driver < 12.8:
    for i, op in ops:
        cudaMemcpyAsync(op.dst, op.src, N, cudaMemcpyDeviceToDevice, stream)
        if intraBatchSync and (i+1) % 8 == 0 and i+1 < len(ops):
            ncclMemOpSync(stream)           # 每 8 个 memcpy 插一次全局 barrier
elif not intraBatchSync:
    cudaMemcpyBatchAsync(dsts, srcs, sizes, count=len(ops),
                         attrs=[{srcAccessOrder=Stream, flags=PreferOverlapWithCompute}],
                         attrIdxs=[0], numAttrs=1, stream)
else:
    chunk = 512MiB / len(ops); rounds = ceil(N / chunk)
    for round in 0..rounds-1:
        for i in 0..len(ops)-1:
            idx = (i + round) % len(ops)             # 每轮把起点旋转一位
            追加 ops[idx] 的 [round*chunk, min(N, (round+1)*chunk)) 一段
    cudaMemcpyBatchAsync(全部 len(ops)*rounds 段, 同样的 attr, stream)   # 仍是一次调用，不插 barrier

ncclMemOpSync(stream)                       # ② 后置 barrier
return                                      # host 立即返回，不等待完成
```

GPU 侧（stream 上的执行顺序）：

```text
[前序 kernel] → ① MC 写 4B flag → ① 等 P-1 个 flag → [CE: P 或 P-1 个 memcpy] → ② MC 写 4B flag → ② 等 P-1 个 flag → [后续 kernel]
```

- 数据：CE 从本卡 HBM 读 sendbuff，经 NVLink → NVSwitch 写到对端 HBM 的 recvbuff 第 `rank` 段。本卡读 sendbuff 的次数等于 op 数（P 或 P-1），每个 op 单独读一遍。
- flag：4 字节 H2D memcpy 写到 MC 地址，由 NVSwitch 复制到所有 P 张卡（包括自己）的 `ready[rank]` / `compl[rank]`。等待由 `cuStreamBatchMemOp(CU_STREAM_MEM_OP_WAIT_VALUE_32)` 完成，即 stream 前端轮询本卡显存【推断：硬件上由 host interface / front-end 执行，不占 SM】。

**是否占 SM**：NCCL 没有为 CE plan 发射任何 kernel（`group.cc:337-342` 走 `ncclLaunchCeColl` 分支，`uploadWork` 对 `isCeColl` 直接返回，`enqueue.cc:1215`）。数据由 CE 搬运，flag 写用 memcpy，等待用 stream memop，**NCCL 层面不占 SM**。driver 是否在某些情况下用 SM copy kernel 实现 `cudaMemcpyBatchAsync`，NCCL 无法控制，只通过 `PreferOverlapWithCompute` 给出提示【推断】。

**4. 同步（重点）**

`ncclMemOpSync`（`ce_coll.cc:208-259`）在 NVLS 可用时调用 `ncclPrepMCSync`（`ce_coll.cc:120-163`）：

```text
# ---------- 一次 ncclMemOpSync（rank k，非 graph capture）----------
seq = ++comm->ceColl.ceSeqNum                                  # host 端计数器，每次 sync 加 1
arr = useCompletePtr ? compl : ready                           # 两个数组交替使用
mc  = MC 地址(&arr[k])

cudaMemcpyAsync(mc, &seq, 4, cudaMemcpyHostToDevice, stream)   # 写：NVSwitch 把 seq 广播到所有 GPU 的 arr[k]
cuStreamBatchMemOp(stream, P-1 个 op, flags=0):                # 等：一次调用提交 P-1 个等待
    for r != k:
        { CU_STREAM_MEM_OP_WAIT_VALUE_32, address=&arr[r]（本卡单播地址）,
          value=seq, flags=CU_STREAM_WAIT_VALUE_EQ }
useCompletePtr = !useCompletePtr
```

| 项目 | 实现 | 源码 |
|:---|:---|:---|
| 写 flag 的 API | `cudaMemcpyAsync(mcDstPtr, &currentSeq, 4, cudaMemcpyHostToDevice, stream)`；源是 host 栈变量（可分页内存） | `ce_coll.cc:141-146` |
| 写到哪里 | `arr[myRank]` 的 **multicast** 地址，一次写到达所有 P 张卡 | `ce_coll.cc:136-138` |
| 写的值 | 单调递增的 `ceSeqNum`（各 rank 调用次数相同，所以序号一致） | `ce_coll.cc:127, 130` |
| 等待 API | `cuStreamBatchMemOp`，每批最多 255 个 op（`ncclCuStreamBatchMemOp` 自动拆分） | `ce_coll.cc:247`、`misc/cudawrap.cc:317-330` |
| 等待条件 | P-1 个 `WAIT_VALUE_32`，`arr[r] == seq`（`CU_STREAM_WAIT_VALUE_EQ`），读本卡单播地址；不等自己的 slot | `ce_coll.cc:149-157` |
| 等待 flag | `cuStreamBatchMemOp` 的 flags = 0，waitValue.flags 也没有 `CU_STREAM_WAIT_VALUE_FLUSH` | `misc/cudawrap.cc:324`、`ce_coll.cc:155` |

*① 前置 barrier 的含义：对端 recvbuff 可写*

rank r 的 flag 写是排在它的 stream 上的，所以 `arr[r] == seq` 成立说明 rank r 的 stream 已经执行到本次 AllGather，它在这之前提交的所有工作（比如还在读上一轮 recvbuff 内容的 kernel）都已完成。等齐 P-1 个 flag 之后，本 rank 才能安全地向所有对端的 recvbuff 写入。同一个 stream 上的顺序语义也保证了本卡 sendbuff 已经被前序 kernel 写好。这个 barrier **只覆盖用户 stream 上的工作**；如果用户在其他 stream 上读写 recvbuff，需要自行同步。

*② 后置 barrier 的含义：所有推给我的数据都已落地*

CE 是 push 模型：数据由发送方写入，接收方的 stream 看不到"对端拷贝结束"这一事件。rank r 的 ② flag 排在它的 `cudaMemcpyBatchAsync` 之后，所以 `arr[r] == seq` 意味着 rank r 推给所有人（包括我）的数据都已完成。等齐 P-1 个 flag 之后，本卡 recvbuff 的 P 段都已就绪，后续 kernel 可以读取。

*memory order：数据先于 flag 可见*

NCCL 没有插入任何显式 fence 或 flush，正确性依赖两层 CUDA stream 语义：

1. 发送方：同一 stream 上，后一个操作（② 的 flag memcpy）在前一个操作（数据 batch memcpy）**完成**之后才开始。只要"完成"包括远端写已被确认，flag 就不可能先于数据到达【推断：依赖 driver 在 stream 相邻操作之间插入 sys 级 membar / semaphore release，NCCL 源码中不可见】。
2. 接收方：`WAIT_VALUE` 满足之后，同一 stream 上的后续 kernel 才开始执行，并能看到已到达本卡 HBM 的数据。没有使用 `CU_STREAM_WAIT_VALUE_FLUSH`（该标志针对 GPUDirect RDMA 这类远端写的可见性）。NVLink 写已经直接落在本卡显存中，所以不需要它【推断】。

数据走单播路径，flag 走 multicast 路径，两者经过不同的交换机路径。之所以不会乱序，靠的是上面第 1 条的"先完成、后发起"，而不是同一路径上的 FIFO 顺序【推断】。

*连续两次调用之间如何避免 flag 冲突*

两个机制：

- **序号**：每次 sync `++ceSeqNum`，等待条件是 `== seq`，旧值不会误满足新等待。
- **ready / compl 交替**：`useCompletePtr` 每次 sync 后翻转（`ce_coll.cc:250`）。不发生 intra-batch sync 时，每次 AllGather 恰好 2 次 sync，所以 ① 总是用 ready、② 总是用 compl，序号分别为奇数和偶数。

为什么需要两个数组？等待条件是 `EQ` 而不是 `GEQ`。如果只有一个数组，快的 rank B 可能在慢的 rank A 观察到 `arr[B] == s` 之前就写入 `s+1`，A 会永远等不到 `s`。使用两个数组后，B 下一次写同一个数组（序号 `s+2`）之前，必须先通过中间那次 sync（另一个数组，序号 `s+1`）。这需要 A 的 `s+1` flag，而 A 的 stream 只有在 A 已经通过序号 `s` 的等待之后才会写出 `s+1`。所以 B 覆盖 `arr[B]` 时，A 一定已经读到了旧值。

```text
rank A stream:  [W ready=s][wait ready==s]  [copy]  [W compl=s+1][wait compl==s+1]  ...
rank B stream:  [W ready=s][wait ready==s]  [copy]  [W compl=s+1][wait compl==s+1]  [W ready=s+2] ...
                                                                         ▲
                     B 写 ready=s+2 之前必须看到 A 的 compl=s+1，而 A 写 compl=s+1 时已经通过了 ready==s
```

*CUDA graph capture 时*

capture 时每次 replay 的值必须固定，所以改用常量 `GRAPH_SYNC_VALUE = 1`（`ce_coll.cc:16, 130-132`）作为写入值和等待值，并在同一个 `cuStreamBatchMemOp` 的等待之后追加 P 个 `CU_STREAM_MEM_OP_WRITE_VALUE_32`，把本卡 `arr[0..P-1]` 全部清零（`ce_coll.cc:235-244`），供下一次 replay 使用。清零的安全性与上面的"两个数组交替"是同一个论证。源指针改为 static 常量，也是为了让 graph 中的 memcpy 节点在 replay 时源地址仍然有效【推断】。

*单播回退（`nvlsSupport == 0`）*

`ncclPrepUCSync`（`ce_coll.cc:165-205`）把一次 MC 写换成 P-1 次 `cudaMemcpyAsync(H2D, 4B)`，分别写到每个对端 `arr[myRank]` 的 LSA 单播地址，等待部分完全相同。每次 sync 多出 P-2 个小 memcpy，P = 16 时为 15 次。

*intra-batch sync 带来的额外 barrier*

只在 memcpy 逐个提交的路径（graph capture / legacy stream / driver < 12.8）中，`intraBatchSync` 为真时，每 8 个 memcpy 插入一次 `ncclMemOpSync`（`ce_coll.cc:361-363, 456-458`）。这样每次 AllGather 的 sync 次数可能是奇数，ready/compl 与"前置/后置"的对应关系会错开，但各 rank 的 sync 序列完全一致，上面的交替论证仍然成立。`cudaMemcpyBatchAsync` 路径中的 intraBatchSync 只做切块和重排，**不插 barrier**。

**5. 流水与重叠**

| 层面 | 是否重叠 | 说明 |
|:---|:---|:---|
| 一次 batch 内的 P-1 个远端 memcpy | 可能 | NCCL 只提交一次 batch；op 是否在多个 CE 上并行执行由 driver 决定【推断】 |
| 不同 rank 之间的目标分布 | 错开 | 所有 rank 的第 i 个 op 指向 `rank+i`，是 P 个 GPU 上的一个循环移位置换。若 CE 按提交顺序推进，每个时刻每张卡大约只被一个对端写入，避免多对一的 incast【推断：意图来自顺序设计，源码无注释】 |
| 大消息切块（P > 8） | 交错 | 每个 op 被切成 ≤ `512MiB/numOps` 的块，按轮交错、每轮起点旋转一位。每一轮内各 rank 第 i 块的目标仍是同一个移位置换。源码注释说明这是为了"improve CE collective performance with large scale"（`ce_coll.cc:18`），具体收益机制未说明 |
| 数据拷贝与 barrier | 不重叠 | ① → 拷贝 → ② 在同一 stream 上严格串行 |
| 与用户 stream 上的 kernel | 不重叠 | 操作放在用户 stream 上，与前后 kernel 串行 |
| 与其他 stream 上的计算 kernel | 可以重叠 | 不占 SM；`PreferOverlapWithCompute` 提示 driver 选择适合与计算并行的拷贝方式。HBM 与 NVLink 带宽仍然共享 |
| 同一 group 中的多个 CE AllGather | 不重叠 | 每个任务是一个 plan，按顺序提交到同一 stream，各自带两次 barrier |

**6. 每张 GPU 的流量**

| 方向 / 介质 | 内容 | 流量 |
|:---|:---|:---|
| NVLink 出（GPU → Switch） | 推给 P-1 个对端的数据 | (P-1)·N |
| NVLink 入（Switch → GPU） | P-1 个对端推来的数据 | (P-1)·N |
| NVLink flag | 每次 sync 发出 1 个 4B MC 写，收到 P 个 4B 写（含自己） | 每次调用 2 × (4B 出 + 4P B 入)，可忽略 |
| HBM 读 | sendbuff 被每个 op 读一遍 | P·N（out-of-place）/ (P-1)·N（in-place） |
| HBM 写 | 对端写入 + 本地拷贝 | (P-1)·N + N（out-of-place） |

- NVLink 流量与 Ring AllGather 相同，都是 (P-1)·N，是 AllGather 每方向的理论下限。
- 与 symmetric NVLS 的 `multimem.st` 版本（2.3.2）相比，CE 每张卡出方向多发 P-1 倍：NVLS 只发 N，由交换机复制。CE 的优势是零 SM 和一次 host 调用，而不是链路效率。
- 时间下限约为 `(P-1)·N / B_nvl`（B_nvl 为单卡单向 NVLink 带宽），再加上两次 barrier 的固定延迟【推断】。

**7. 示例：P = 8 与 P = 16（GB200 MNNVL，bf16，每 rank 64MiB，out-of-place）**

count = 32M 个元素 > 8M，所以满足自动选择 CE 的阈值（N > 16MB）。

| | P = 8 | P = 16 |
|:---|:---|:---|
| recvbuff 大小（需在 symmetric window 中） | 8 × 64MiB = 512MiB | 16 × 64MiB = 1GiB |
| numOps（1 本地 + P-1 远端） | 8 | 16 |
| `intraBatchSync` | 否（numOps = 8，不大于 8） | 是（16 > 8 且 16 × 64MiB = 1GiB ≥ 512MiB） |
| 每个 op 的大小 | 64MiB | 切成 512MiB/16 = 32MiB，2 轮 |
| `cudaMemcpyBatchAsync` 的 count | 8（一次调用） | 32（一次调用） |
| 每次 sync 的 `WAIT_VALUE_32` 个数 | 7 | 15 |
| 每次 sync 的 flag 写 | 1 次 MC 写（UC 回退时 7 次） | 1 次 MC 写（UC 回退时 15 次） |
| 每次调用的 barrier 数 | 2 | 2（batch 路径下切块不增加 barrier） |
| flag 区逻辑大小 | 64B | 128B |
| NVLink 每方向流量 | 7 × 64MiB = 448MiB | 15 × 64MiB = 960MiB |
| HBM 读 sendbuff | 8 × 64MiB | 16 × 64MiB |

- P = 16 时切块的触发点：out-of-place 为 N ≥ 32MiB，in-place 为 N ≥ 512MiB/15 ≈ 34.1MiB。P ≤ 8 时 numOps ≤ 8，**永远不切块**。
- bf16 在 P = 16 时，16MB < N < 32MiB 的消息会走 CE 但不切块；N ≥ 32MiB 时切块。
- 两次 barrier 的固定开销基本不随 P 变化（一次 MC 写，加上等待 P-1 个本地 flag），但 barrier 的完成时间由最慢的 rank 决定。P 越大，被单个慢 rank 拖住的概率越高【推断】。
- 数据量随 P 线性增长（(P-1)·N），时间下限也线性增长；SM 占用始终为零。

**数据量**：每个 rank 读入 N（sendbuff，被 CE 读 P 次），写出 P·N（recvbuff）；NVLink 每方向 (P-1)·N。没有 step 的概念，一次 batch 加两次 barrier。

#### 2.4.3 ReduceScatter

不支持。CE 的 `ncclCeImplemented` 只注册了 AllGather / AlltoAll / Scatter / Gather 四种操作（`ce_coll.cc:80`），ReduceScatter 不在其中。

---

## 3. NCCL 相关概念

### 3.1 Symmetric Memory

**概念：同一块逻辑缓冲区在所有 LSA 成员的虚拟地址空间里占据相同的偏移位置。**

用 `ncclMemAlloc` 分配的缓冲区，或者用 `ncclCommWindowRegister(..., NCCL_WIN_COLL_SYMMETRIC)` 注册的用户缓冲区，都属于 symmetric window。每个 window 有一个全局唯一的 `bigOffset`（在 `[0, bigSize)` 区间内）。任意 rank 只要知道 `bigOffset`，就能通过本地的 `lsaFlatBase + peer * bigSize + bigOffset` 算出对端的虚拟地址，无需在 kernel 启动时传入指针数组。

**LSA team：能直接 ld/st 访问的 rank 集合。**

`ncclDevrInitOnce`（`dev_runtime.cc:77`）在初始化时扫描 `comm->rankToNode` 数组，计算每台主机上连续 rank 的数量，取所有主机 rank 数的最大公约数作为 `lsaSize`。每个 rank 的 LSA team 成员列表写在 `devr->lsaRankList`，成员 `i` 的全局 rank 号为 `comm->rank + (i - lsaSelf)`（`dev_runtime.cc:98`）。NVL72 场景下，整个 clique 被 MNNVL 合并为单节点（`nNodes == 1`），因此 `lsaSize == nRanks`，LSA team 覆盖所有 72 个 rank。

**单播映射（lsaFlatBase + cuMemMap）**

`symMemoryMapLsaTeam`（`dev_runtime.cc:183`）负责把每个 window 映射到本 rank 的虚拟地址空间：

1. 各 rank 将本地的 `CUmemGenericAllocationHandle`（或跨主机时的 `CUmemFabricHandle`）通过 `bootstrapIntraNodeAllGather` 分发给 LSA team 内所有成员（`dev_runtime.cc:200`）。
2. 首次映射时，用 `cuMemAddressReserve` 预留一块大小为 `lsaSize × bigSize` 的连续虚拟地址区域，起始地址存入 `devr->lsaFlatBase`（`dev_runtime.cc:205`）。
3. 对 team 内的每个 peer `r`，用 `cuMemImportFromShareableHandle` 导入对方的 handle，再用 `cuMemMap` 把对方的物理页映射到 `lsaFlatBase + r * bigSize + bigOffset`（`dev_runtime.cc:225`），最后调用 `cuMemSetAccess` 开放读写权限（`dev_runtime.cc:227`）。

映射完成后，rank A 访问 `lsaFlatBase + B * bigSize + bigOffset` 即等价于访问 rank B 的显存，底层通过 NVLink 完成搬运。

**多播映射（cuMulticastBindMem + multimemPtr）**

当 `comm->nvlsSupport` 为真且 LSA team 大小 > 1 时，`symTeamObtain`（`dev_runtime.cc:264`）还会创建一个 `CUmulticastObject`。每个 window 注册时，`symBindTeamMemory`（`dev_runtime.cc:241`）调用 `cuMulticastBindMem(tm->mcHandle, mem->bigOffset, mem->memHandle, 0, mem->size, 0)`（`dev_runtime.cc:247`）把物理页绑定到多播对象的相同偏移位置。多播对象的基地址存入 `tm->mcBasePtr`，kernel 通过 `ncclDevrGetLsaTeamPtrMC` 把 `bigOffset` 加上 `mcBasePtr` 得到多播地址。对这个地址发出 `multimem.st` 时，NVSwitch 硬件会把数据广播给所有成员；对这个地址发出 `multimem.ld_reduce` 时，NVSwitch 在路径上完成归约。

**启用条件**（`init.cc:1475`）

```cpp
comm->symmetricSupport = comm->isAllCudaP2p && ncclParamWinEnable() && ncclCuMemEnable() &&
  (comm->globalGinSupport != NCCL_GIN_CONNECTION_NONE || (ncclTeamLsa(comm).nRanks == comm->nRanks));
```

四个条件：所有 rank 之间都有 P2P 访问路径（`isAllCudaP2p`）、`NCCL_WIN_ENABLE` 未被关闭、cuMem 可用，以及 GIN 可用或 LSA team 覆盖全部 rank 之一成立。NVL72 场景下第三个条件（LSA team 覆盖全部 rank）通常成立，所以即使没有 GIN 也能开启 symmetric memory。

**与 Legacy P2P FIFO 的区别**

| | Legacy P2P | Symmetric Memory |
|---|---|---|
| 寻址方式 | kernel 启动时传入指针；环上每对 rank 独立映射 | `lsaFlatBase + peer * bigSize + offset`；一次初始化，所有 rank 均可访问 |
| 数据路径 | 先写进接收端 FIFO，接收端再拷到 recvbuff | 直接写/读对端 recvbuff（zero-copy） |
| 多播 | 不支持 | 支持 multimem.st / multimem.ld_reduce |
| 归约 | SM 在本地做 | 可由 NVSwitch 在路径上完成（LDMC）|
| 同步 | FIFO head/tail 或 LL flag | 非 LL kernel：首尾 LSA barrier（单播 inbox 或多播 `multimem.red`）；LL kernel：没有 barrier，靠包内 epoch flag |

### 3.2 Copy Engine

**CE（Copy Engine）是 GPU 上独立于 SM 的硬件 DMA 单元。** 它专门做数据搬运，执行期间不占用任何 SM。NCCL 的 CE 路径由**用户线程**在 `ncclGroupEnd → doLaunches → ncclLaunchCeColl` 中直接把搬运任务提交到用户 stream（`group.cc:337`、`ce_coll.cc:687`），没有 proxy 线程参与，GPU 上也不运行 kernel。

**支持的集合操作**（`ce_coll.cc:81`）

CE 路径只支持无归约的集合操作，CUDA driver 12.5 及以上版本才启用：

```
AllGather、AlltoAll、Scatter、Gather → 支持
AllReduce、ReduceScatter → 不支持（CE 只能拷贝，不能做归约）
```

**CUDA API**

NCCL 用两套 API 发起搬运，根据 CUDA runtime 版本自动选择（`ce_coll.cc:368`）：

- `cudaMemcpyBatchAsync`（CUDA 12.8 及以上，`CUDART_VERSION >= 12080` 且 `driverVersion >= 12080`）：一次调用提交一批拷贝：out-of-place 时共 P 个（包括 sendbuff → 本地 recvbuff 的那一个），in-place 时 P-1 个（`ce_coll.cc:488-504`），减少 host 调用开销。
- `cudaMemcpyAsync`（CUDA 12.5-12.7）：逐个目的地提交，每个对端单独调用一次。

**同步机制**

CE AllGather 需要两次全局同步：第一次确认所有 rank 都已执行到本次调用（对端 recvbuff 可写）后才开始拷贝；第二次确认所有推给本 rank 的数据都已落地。host 调用会立即返回，只是用户 stream 上的后续工作要等第二次同步完成。同步通过以下两种 API 配合实现（`ce_coll.cc:120`），详见 2.4.2：

1. 每个 rank 把自己的 ready/complete flag 写到**多播地址**（`cudaMemcpyAsync` 写到 `mcDstPtr`），NVSwitch 一次广播给所有 rank（`ce_coll.cc:141`）。
2. 每个 rank 用 `cuStreamBatchMemOp` 提交 `CU_STREAM_MEM_OP_WAIT_VALUE_32`（`CU_STREAM_WAIT_VALUE_EQ`）操作，在 CUDA stream 上原地等待其余 P-1 个 rank 的 flag 等于本次的序号（`ce_coll.cc:149-157`）。
3. flag 值是每次同步递增的 `ceSeqNum`；ready 和 complete 两组 flag 数组按 `useCompletePtr` 每次交替使用（`ce_coll.cc:250`），避免快 rank 覆盖慢 rank 还没读到的值。graph capture 时固定写 1，等待后再清零。

这样同步完全在 GPU stream 上推进，host 不需要轮询，CE 搬运本身也不占 SM。

**AllGather 触发条件**（`ce_coll.cc:100`、`enqueue.cc:2919`）

- `ncclCeAvailable`：driver ≥ 12.5、`nNodes == 1`、`symmetricSupport` 开启，且 recvbuff 已注册为 symmetric window（`ncclSymSendRegRecvReg` 或 `ncclSymSendNonregRecvReg`）。
- 在此之上，CE 会在以下两种情况之一被调度器选中（`enqueue.cc:2966`）：
  - 用户设置了 `NCCL_CTA_POLICY_ZERO`（强制 zero CTA 模式）。
  - Blackwell 上每个 rank 的元素数超过 `NCCL_SYM_CE_THRESHOLD`（默认 8M 个元素），自动选中 CE，此时 SM kernel 路径预计饱和。

**与 SM 发起通信的对比**

| | SM kernel | CE AllGather |
|---|---|---|
| SM 占用 | 每个 channel 1 个 CTA | 零 SM |
| 每步 rank 数 | Ring：逐跳转发；STMC：一次广播 | 每个 rank 直接向其余 P-1 个 rank 发送（one-shot direct push） |
| 归约支持 | 是（SM 内做） | 否 |
| 延迟 | 受 SM 调度影响 | 受 DMA 引擎队列深度影响 |
| 适用场景 | 通用；MFU 敏感训练 | 大消息 AllGather；SM 已饱和的场景 |

### 3.3 PTX

**PTX（Parallel Thread Execution）是 CUDA 虚拟指令集架构，是 NCCL device kernel 里使用的汇编级指令层。** NCCL 的 symmetric kernel 直接通过 `asm volatile` 嵌入 PTX，绕过 CUDA intrinsic 的限制，精确控制多播行为。

#### 3.3.1 `multimem.st.global.*`：多播 store

**语义**：SM 向一个多播地址发出写请求，NVSwitch 把数据原样复制到该多播对象的所有成员 GPU。

实现在 `device/op128.h:391`，以 BytePack 大小为模板参数特化：

```ptx
// 4 字节
multimem.st.global.b32 [%0], %1;
// 8 字节
multimem.st.global.b64 [%0], %1;
// 16 字节（NCCL 最常用的宽度）
multimem.st.global.v4.f32 [%0], {%1,%2,%3,%4};
```

注意 1 字节和 2 字节无对应的 `multimem.st`，退回普通 `st.global`（`op128.h:384,388`）。

使用位置：Symmetric AllGather 的 `AllGather_STMC`（`device/symmetric/all_gather.cuh:195`）和带多播标志的 AllReduce AllGather 阶段（`device/op128.h:508`）。

#### 3.3.2 `multimem.ld_reduce.global.*.add.*`：多播 load-reduce

**语义**：SM 向一个多播地址发出读归约请求，NVSwitch 在路径上对所有成员 GPU 的对应地址做求和，把结果返回给发起方。一次读操作相当于 P 个 rank 同时参与的归约，不需要 P-1 次显式数据搬运。

实现在 `include/nccl_device/impl/multimem__funcs.h`，按数据类型特化：

```ptx
// f32，单元素
multimem.ld_reduce.global.add.f32 %0, [%1];
// f16x2（两个 f16 打包为 32 位）
multimem.ld_reduce.global.add.acc::f32.f16x2 %0, [%1];
// fp8 e4m3，4 个元素打包（sm_100 才支持，multimem__funcs.h:288）
multimem.ld_reduce.global.add.acc::f16.e4m3x4 %0, [%1];
// f64 单元素（multimem__funcs.h:65）
multimem.ld_reduce.global.add.f64 %0, [%1];
```

**数据类型支持**（`multimem__funcs.h` 的模板特化范围）：f32、f16x2、bf16x2、fp8 e4m3（sm_100+）均有原生向量指令；f64 有 1 元素特化，但注释明确没有 128 位向量版本（`multimem__funcs.h:74`），多个 f64 用拆解循环处理，性能较差。`int8_t` 和 `uint8_t` 被静态断言显式禁止（`multimem__funcs.h:30`）。

使用位置：Symmetric ReduceScatter 的 `ReduceScatter_LDMC`（`device/symmetric/reduce_scatter.cuh:312`）以及带多播标志的 AllReduce ReduceScatter 阶段。

#### 3.3.3 `multimem.red.release.sys.add.u32`：多播 reduce（barrier 用）

**语义**：SM 向多播地址发出一个原子加 1 操作，并附带 `release` 语义（保证此前的写操作对所有 rank 可见）。NVSwitch 把这个原子操作广播到多播对象的所有成员，每个 rank 的 inbox 计数器都加 1。

实现在 `include/nccl_device/impl/lsa_barrier__funcs.h:70`：

```ptx
multimem.red.release.sys.add.u32 [%0], 1;   // release 语义
multimem.red.relaxed.sys.add.u32 [%0], 1;   // relaxed 语义（无 release）
```

使用位置：`ncclLsaBarrierSession::arrive` (`lsa_barrier__funcs.h:63`)，即 Symmetric kernel 里所有 rank 完成一个阶段后用来互相通知的 LSA barrier arrive 路径。多播版本（`this->multimem == true`，即 team 有多播地址时）只需要一条指令就能让所有 rank 的 inbox 同时加 1，替代了单播路径下的 `nRanks-1` 次逐对 atomic_ref store。

#### 3.3.4 与普通 `st/ld` 的区别

普通的 `st.global` 和 `ld.global` 只操作单一物理地址对应的 GPU 显存。当目标地址是多播地址时，必须改用 `multimem.*` 指令，NVSwitch 才会识别出这是多播请求。如果对多播地址用普通 `st`，按 PTX ISA 属于未定义行为（规定多播地址必须通过 `multimem` 指令访问）。但 NCCL 实际上依赖了这种用法：Legacy NVLS 用普通 `st.relaxed.sys` 写 MC credit 来广播（`postPeer`），`AllGather_LLMC` / `AllReduce_AGxLLMC_R` 用普通 `st.relaxed.sys.v4.u32` 写 MC 地址，NVLS AllGather 的尾部（不足 16B）也会退化成对 MC 地址的普通 `st`（`op128.h:383-385`）。这依赖的是硬件实现行为，而不是 PTX 规范的保证。反之，对单播地址用 `multimem.st`，NVSwitch 会退化为普通点对点写。

#### 3.3.5 在 Symmetric kernel 里的使用位置

| kernel 名 | 使用的指令 | 作用 |
|---|---|---|
| `AllGather_STMC` | `multimem.st.global.v4.f32` | 一次广播写到所有 rank 的对应位置 |
| `AllGather_LLMC` | 对 MC 地址的普通 `st.relaxed.sys.v4.u32`（LL 格式，`ll_a2a__funcs.h:83`） | LL 协议下的多播写 |
| `ReduceScatter_LDMC` | `multimem.ld_reduce.global.add.*` | 一次归约读，由 NVSwitch 做求和 |
| `AllReduce_RSxLDMC_AGxSTMC` | 两者均用 | RS 阶段用 LDMC，AG 阶段用 STMC |
| LSA barrier（multimem 路径） | `multimem.red.release.sys.add.u32` | 阶段间同步的 arrive 操作 |

### 3.4 GIN

**GIN（GPU Interconnect Network）是 NCCL 中用于跨 NVLink 域通信的机制。** 它依赖一个独立的网络插件（`ncclNet`），通过 GDR（GPU Direct RDMA）在不同 NVLink 域（rail）之间传输数据，对应 `NCCL_GIN_CONNECTION_RAIL` 和 `NCCL_GIN_CONNECTION_FULL` 两种连接模式（`init.cc:1471`）。启用条件包括 GIN 网络插件存在、NIC 未融合（`!globalNicFused`）、GDR 可用（`globalCuMemGdrSupport`）。

**在本文场景（NVL72 单 clique 内）GIN 不涉及。**

NVL72 是单个 NVLink 域：72 张 Blackwell GPU 通过 9 个 NVSwitch 全互联，所有 rank 在同一个 fabric clique 里，无需跨域通信。MNNVL 初始化时把整个 clique 合并为 `nNodes == 1`（`graph/topo.cc:1543`），symmetric memory 的启用条件走的是 `ncclTeamLsa(comm).nRanks == comm->nRanks` 分支，而非 `globalGinSupport`（`init.cc:1476`）。因此，调度器在 NVL72 上不会启用 GIN 相关路径，`comm->globalGinSupport` 保持 `NCCL_GIN_CONNECTION_NONE`。

**GIN 相关 kernel（仅在多域场景使用）**

`sym_kernels.cc` 中定义的 GIN kernel 掩码包含：
- `ReduceScatter_RailA2A_LsaLD`：在 rail 之间做 all-to-all 数据交换后，在本地 LSA 域内用单播 `ld` 完成归约。
- `ReduceScatter_RailA2A_LsaLDMC`：同上，LSA 域内归约改用多播 `multimem.ld_reduce`。
- `AllGather_RailRing_LsaSTMC`：先在 LSA 域内用多播 `multimem.st` 做 AllGather，再通过 GIN 以 ring 方式跨 rail 广播。

这三个 kernel 均在 `kernelMask_Gin`（`sym_kernels.cc:80`）中标记，仅当 `comm->globalGinSupport != NCCL_GIN_CONNECTION_NONE` 时才会被调度器选中，NVL72 单域场景下永远不会走到这条路径。

### 3.5 Channel 搜索机制

Channel 搜索是 `ncclCommInitRank` 时一次性完成的操作，它为每种 legacy 算法（Ring、Tree、NVLS 等）独立搜索一套通信拓扑，结果存储在 `comm->graphs[NCCL_ALGO_*]` 里。之后所有集合操作都直接查询这些结果，搜索本身和哪种集合操作无关。

#### 3.5.1 适用范围

| 算法 | 搜索模式 | 结果 |
|:---|:---|:---|
| Ring（AG/RS/AR） | `NCCL_TOPO_PATTERN_RING`：找哈密顿环 | 每个 channel 一条 GPU 顺序 |
| Tree（AR） | `NCCL_TOPO_PATTERN_TREE` / `SPLIT_TREE` / `BALANCED_TREE`：找树型 | 每个 channel 一棵树 |
| NVLS（AG/RS/AR） | `NCCL_TOPO_PATTERN_NVLS`：找 head GPU 集合 | NVLS head 列表 |

Symmetric kernel 和 CE collective 不依赖这套搜索，它们直接通过 symmetric window 的 LSA 映射找到对端地址。

调用入口（[init.cc:1091-1131](../nccl-2.29.7/src/init.cc#L1091)）：
```cpp
ringGraph->pattern = NCCL_TOPO_PATTERN_RING;
NCCLCHECKGOTO(ncclTopoCompute(comm->topo, ringGraph), ret, fail);
// 同理 treeGraph、nvlsGraph ...
```

#### 3.5.2 外层：速度迭代

`ncclTopoCompute`（`graph/search.cc`）的外层循环按带宽从高到低依次尝试：

Blackwell 节点内的速度候选（`sm100SpeedArrayIntra`）为：90、80、70、60、50、45、40、30、24、20、19、18 GB/s。

外层每次以一个固定的目标带宽 `bwIntra` 驱动内层深度优先搜索。如果搜到的 channel 数达到上限，或者总带宽已经饱和（`nChannels × bwIntra ≥ totalBw`），则停止；否则降低目标带宽继续搜，直到超时（`NCCL_SEARCH_GLOBAL_TIMEOUT`）。

同时有两趟搜索（pass 1 和 pass 2）：第一趟先尝试 `sameChannels=1`（所有 channel 的 GPU 顺序相同，只起点不同），失败则改为 `sameChannels=0`（每条 channel 独立搜索）。NVL72 这种全互联系统通常满足 `sameChannels=1`，搜索可以通过 REPLAY 模式加速。

#### 3.5.3 内层：哈密顿环的深度优先搜索（Ring 模式）

`ncclTopoSearchRecGpu` 是递归的深度优先搜索：从起始 GPU 出发，每步选一个还没被本 channel 用过的 GPU（用 `used` 位掩码标记），走满所有 P 个 GPU 后回到起点，找到一条哈密顿环（[search.cc:577](../nccl-2.29.7/src/graph/search.cc#L577)）。

**下一跳候选的排序由 `ncclTopoSearchNextGpuSort` 决定（[search.cc:243](../nccl-2.29.7/src/graph/search.cc#L243)）：**

1. 先按 `intraNhops`（跳数）和 `intraBw`（带宽）对所有未使用 GPU 排序。
2. 有 NVSwitch 时，额外调整：把 `(index+1) % P` 和 `(index-1) % P` 提到候选列表最前面（Ring 模式优先尝试 `nextGpu`，即编号加一的 GPU），并把 `*countPtr` 限制为 2，让搜索优先探索这两个方向（[search.cc:282-313](../nccl-2.29.7/src/graph/search.cc#L282)）。

这里的 `index` 是当前 GPU 在系统 GPU 列表里的下标，不是 rank 编号，也不随 channel 变化。

**NVSwitch 为什么优先"邻居"**

NVSwitch 上所有 GPU 对之间的路径代价相同（跳数相同、带宽相同），标准排序给出的是相同分数的候选列表，搜索方向没有明确偏好。显式把 `index±1` 提前，是为了让同一 channel 的环保持局部连续性，降低 NVSwitch 内部端口的竞争。

#### 3.5.4 多 channel 之间的偏移：REPLAY 模式

找到第一条 channel 后，`ncclTopoSearchRec`（[search.cc:715](../nccl-2.29.7/src/graph/search.cc#L715)）对后续 channel 使用 `FORCED_ORDER_REPLAY`：

```cpp
if (graph->nChannels > 0 && graph->sameChannels == 1) {
    ncclTopoReplayGetGpu(system, graph, -1, &g);  // 起始 GPU = 上一条 channel 的起始 GPU + 1
    ncclTopoSearchTryGpu(..., FORCED_ORDER_REPLAY, ...);
}
```

`ncclTopoReplayGetGpu` 在 REPLAY 模式下，每一步都从上一条 channel 同步骤的 GPU 编号加 1（取模 P）。这样第 k 条 channel 的环就是第 0 条环整体偏移 k 格的版本。

**举例（P=32）**：

假设搜索找到 channel 0 的环为：`0 → 1 → 2 → ... → 31 → 0`

那么 channel 1 为：`1 → 2 → 3 → ... → 31 → 0 → 1`

从 GPU 0 的角度看：

| channel | ring.prev（上游） | ring.next（下游） |
|:---|:---|:---|
| channel 0 | 31 | 1 |
| channel 1 | 0 作为 channel 1 的第 31 步，所以 GPU 0 的 prev 是 31（不变），但 GPU 0 在 channel 1 里的位置不同 | — |

更直观的描述：channel k 里，GPU g 的下游是 `(g + 1) mod P` 的下游是 `(g - 1 + k) mod P`（这是简化模型，实际取决于环的排列，不是简单的线性偏移）。关键结论是**各 channel 的上下游不同**，于是并发传输时不同 channel 使用的 NVLink 端口会自然错开。

#### 3.5.5 DupChannels 和 Postset 翻倍

搜索完成后还有两次翻倍：

1. **`ncclTopoDupChannels`（[search.cc:962](../nccl-2.29.7/src/graph/search.cc#L962)）**：
   - 在 Blackwell 上 `bwIntra ≥ 50 GB/s` 时把 channel 数翻倍，每条环原样复制，`bwIntra` 减半（让代价模型看到每条环的带宽降低了）。
   - 这样总带宽不变，但 channel 数更多，有利于小消息分散。
   - NVLS pattern 不参与这一步。
2. **`ncclTopoPostset`（[graph/connect.cc:453](../nccl-2.29.7/src/graph/connect.cc#L453)）**：
   - 进一步把 `nChannels` 翻倍，上限 `MAXCHANNELS = 64`，同样是原样复制环，两个复制的 channel 共享相同的 ring.prev / ring.next，但使用不同的连接资源（FIFO）并发运行。

两次翻倍后，ring.prev / ring.next 对于编号相差 `nOrigChannels` 的两个 channel 是相同的，差异只在 FIFO 槽和 SM 线程上。

#### 3.5.6 搜索结果的读取

搜索结果写到 `graph->intra[c * ngpus + step] = rank`，其中 `c` 是 channel 编号，`step` 是步骤编号（0 到 P-1），`rank` 是这步访问的 GPU rank。`ncclTopoPostset` 从这个数组里提取 `ring.prev` 和 `ring.next` 并写回每个 channel 的 `ring` 结构，供 kernel 里的 `ncclShmem.channel.ring` 使用。

设置 `NCCL_DEBUG=INFO` 时，NCCL 会打印每条 channel 的详细结果，包括 GPU 顺序、bwIntra 和 nChannels，可以用来验证搜索结论。

### 3.6 MNNVL

MNNVL（Multi-Node NVLink）**不是另一条数据通道**。它是一套软件机制，让分属不同操作系统的进程能够把彼此的显存映射到自己的地址空间。在 NVL72 里，同一 tray 内的通信和跨 tray 的通信走的是同一条物理路径。

#### 3.5.1 物理层：都经过 NVSwitch

这一小节是 GB200 NVL72 的硬件知识，不是从 NCCL 源码里得出的。

- 每个 compute tray（node）上有 2 颗 Grace CPU 和 4 张 Blackwell GPU。tray 内**没有 NVSwitch**，GPU 之间也**没有点对点直连的 NVLink**。
- 每张 GPU 的 18 条 NVLink 全部连到 9 个 NVSwitch tray 上。
- 不管是同一 tray 内的两张 GPU，还是不同 tray 的 GPU，数据都走 GPU → NVLink → NVSwitch → NVLink → GPU，跳数和带宽都一样。
- GPU 和本 tray 的 CPU 之间是 NVLink-C2C，这条链路和 GPU 之间的通信无关。

#### 3.5.2 软件层：区别在于显存能不能互相映射

GPU 要通过 ld/st 直接访问对端显存，前提是先把对端显存映射到本进程的虚拟地址空间。这需要把 cuMem handle 交给对方进程。问题出在通信组跨操作系统时，这个 handle 怎么传：

| 场景 | handle 类型 | 怎么传递 | 是否需要 MNNVL |
|:---|:---|:---|:---|
| 通信组里的 GPU 都在同一个 tray（同一个 OS） | `CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR` | 通过 Unix socket 传文件描述符，只能在同一主机内 | 不需要 |
| 通信组跨多个 tray（多个 OS） | `CU_MEM_HANDLE_TYPE_FABRIC` | fabric handle 是一段普通字节，可以通过 bootstrap 网络发给其他主机，对方导入后映射 | **需要** |

跨 tray 的情况还依赖驱动层的 IMEX 服务来授权 fabric handle 的导入和导出。

#### 3.5.3 NCCL 中的判断逻辑

1. **何时启用**（`init.cc:988`）：
   ```cpp
   if ((nNodes > 1 && ncclParamMNNVLEnable() != 0 && p2pLevel != 0) || ncclParamMNNVLEnable() == 1)
     ncclMnnvlCheck(comm);
   ```
   - 这里的 `nNodes` 是按 hostHash 统计的主机数。
   - 只有通信组跨了多台主机，才会去检查 MNNVL。如果只在一个 tray 内（最多 4 卡），不会进入这条路径。
   - `NCCL_MNNVL_ENABLE` 的默认值是 2，表示自动判断；设为 1 强制启用，设为 0 关闭。
2. **启用后做什么**（`mnnvl.cc`）：
   - 用 NVML 读取每张 GPU 的 fabric clusterUuid 和 cliqueId。值相同的 rank 属于同一个 clique，也就是同一个 NVLink 域。
   - cliqueId 可以用 `NCCL_MNNVL_CLIQUE_ID` 覆盖。
   - 然后设置 `ncclCuMemHandleType = CU_MEM_HANDLE_TYPE_FABRIC` 和 `comm->MNNVL = 1`（`mnnvl.cc:86`）。
3. **对拓扑的影响**（`graph/topo.cc:1543`）：
   - 整个 clique 的拓扑被合并成一个节点：`localRanks = clique.ranks`。
   - 因此之后 `comm->nNodes == 1`，`localRanks` 等于通信组大小 P。
   - 这就是本文场景约束里"整个 NVL72 被当作一个节点"的来源。
   - 它也带来一个后果：Legacy NVLS 的 AG/RS 要求 `localRanks ≤ 32`，这里的 `localRanks` 就是 P。

#### 3.5.4 启用 MNNVL 后的副作用

这些副作用只在通信组跨 tray 时出现：

- **Legacy NVLS 不能注册用户缓冲区**：`init.cc:1024` 中有 `if (comm->MNNVL) comm->nvlsRegSupport = 0`。数据一定经过 NVLS 中转缓冲区（见 2.2）。
- **P2P CE 变体很可能不可用【推断】**：`NCCL_P2P_USE_CUDA_MEMCPY` 依赖 host SHM 段，跨 tray 时无法共享。
- **P2P 连接日志不同**：P2P transport 的连接日志会显示 `via P2P/MNNVL`，不再是 `P2P/CUMEM`（`transport/p2p.cc:411`）。可以据此确认是否启用了 MNNVL。

#### 3.5.5 小结

|  | 同一 tray 内（≤4 卡） | 跨 tray（同一个 NVL72） |
|:---|:---|:---|
| 物理路径 | NVLink 加 NVSwitch | NVLink 加 NVSwitch，和左边相同 |
| 是否启用 MNNVL | 否 | 是 |
| cuMem handle | POSIX FD | FABRIC |
| NCCL 眼中的节点数 | 1 | 1（clique 合并后） |
| `localRanks` | 本 tray 的 rank 数 | 整个 clique 的 rank 数（= P） |

跨 node 的数据并不是"借助 MNNVL 传输"。数据在物理上本来就能直达，MNNVL 解决的是控制面的问题：让不同 OS 中的进程能把彼此的显存映射到自己的地址空间。

### 3.7 数据落点：谁发起访存、数据经过哪里、接收端是否还要拷贝

本节把第 2 章各实现的数据路径，按"访存发起者 × 访存语义"统一整理，回答两个问题：

1. 跨卡传输结束后，数据落在哪里（寄存器、shared memory、FIFO 还是用户缓冲区）？
2. 接收端或发起端还需不需要再做一次搬运？

TMA 相关 kernel 和 CE 的多播 AllGather 在 2.29.7 中**不存在**，本节引用的是 **NCCL 2.32.3**（本地路径 `Megatron-Scale-Down/nccl-2.32.3`），用【2.32.3】标出，对应的源码位置相对 `nccl-2.32.3/src`。其余源码位置仍相对 `nccl-2.29.7/src`。

#### 3.7.1 三种访存发起者

| 发起者 | 指令 / API | 发起端的数据中转 | 粒度 | 发起者如何得知完成 |
|:---|:---|:---|:---|:---|
| **SM + LSU** | `ld.global` / `st.global` / `multimem.ld_reduce` / `multimem.st` | 线程**寄存器** | 每个线程每条指令 4–16B，一个 warp 一次 512B | load：寄存器依赖（scoreboard）；store：posted write，不等完成，要保证顺序必须 `fence` |
| **TMA**【2.32.3】 | `cp.async.bulk`：`cuda::device::memcpy_async_tx`（global → shared），`ptx::cp_async_bulk`（shared → global） | **shared memory**，每个 warp 一个 tile | 每个 warp 只有 lane 0 发一条指令，一次搬一个 tile（`UnrollPacks × 32 × 16B`） | global → smem：mbarrier 按字节计数（`barrier_arrive_tx` + `bar.wait`）；smem → global：`cp.async.bulk.wait_group.read` 只保证 smem 已被读走，**不保证远端写完成** |
| **CE** | `cudaMemcpyBatchAsync` / `cudaMemcpyAsync`，由 host 提交 | 无（CE 内部完成） | 整块缓冲区 | stream 顺序；跨 rank 用 flag + `cuStreamBatchMemOp(WAIT_VALUE_32)` |

三者的能力边界：

- 只有 SM 能发 `multimem.ld_reduce`，所以**只有 SM 能利用 NVSwitch 做归约**。
- TMA 和 CE 都可以把 MC 地址当作写目标，实现多播写；但它们都**没有归约语义**。所以 2.32.3 中不存在 `TmaLDMC`（`sym_kernels.cc:23` 的 `kernelMask_LDMC` 里没有任何 TMA kernel），CE 也不支持 AllReduce 和 ReduceScatter。

#### 3.7.2 四种访存语义

| 访存 | 方向 | 数据来源 | 数据落点 | 谁把数据写进目标 HBM |
|:---|:---|:---|:---|:---|
| `ld.global`（单播，读对端 UC/LSA 地址） | 拉取 | 一张卡的 HBM | 发起线程的**寄存器** | 发起者必须再发一条 store |
| `st.global`（单播，写对端） | 推送 | 发起线程的寄存器 | 目标卡的 HBM | NVLink 写直接落地 |
| `multimem.ld_reduce`（MC 地址） | 拉取，交换机归约 | P 张卡同一偏移处的数据 | 发起线程的**寄存器**（已归约） | 发起者必须再发一条 `st` 或 `multimem.st` |
| `multimem.st`（MC 地址） | 推送，交换机复制 | 发起线程的寄存器 | P 张卡同一偏移处的 HBM | 交换机直接写入 |

TMA load 与 `ld` 同类（落点是 smem），TMA store 和 CE 拷贝与 `st` 同类（落点是目标 HBM）。

**核心规律**：

- **推送式**（`st`、`multimem.st`、TMA store、CE 拷贝）：只要写目标就是最终地址（用户 recvbuff），**接收端零搬运**，也不占用接收端的 SM。
- **拉取式**（`ld`、`multimem.ld_reduce`、TMA load）：结果只存在于发起者片上（寄存器或 smem），**必须由发起者再执行一次 store 才能进入 HBM**。这次 store 不可省略，但它的目标可以直接是最终位置：
  - ReduceScatter：store 到本卡 recvbuff（`st.global`）。
  - AllReduce：store 本身就是 AllGather 的 `multimem.st`，直接写到所有卡的 recvbuff（见 3.7.5）。
- 所以"`ld_reduce` 需要额外一次 copy"严格来说是：发起者要**多发射一次 store**（占用 SM 指令和寄存器），但本卡 HBM 的写流量与推送式相同（都是 N）。真正"额外"的 HBM → HBM 拷贝，只出现在经过 FIFO 中转的 Legacy NVLS 路径中（Scatter、Gather）。

#### 3.7.3 各实现的数据路径对照

P 为通信组大小。N 的含义：AllGather 为每个 rank 的 sendbuff 大小，ReduceScatter 为每个 rank 的 recvbuff 大小，AllReduce 为缓冲区大小。

| 实现 | 跨卡访存 | 发起者 | 发送 / 发起端路径 | 接收端路径 | 接收端搬运 | 本卡 HBM → HBM 中转拷贝 |
|:---|:---|:---|:---|:---|:---|:---|
| Legacy NVLS AG（2.2.2） | `multimem.st` | SM | sendbuff → 寄存器 → MC B[r] slot | B[h] slot → 寄存器 → recvbuff（Gather） | **有** | Gather：P·N |
| Legacy NVLS RS（2.2.3） | `multimem.ld_reduce` | SM | Scatter：sendbuff → A[h] slot | head：MC A[r] → 寄存器 → `st` recvbuff | 寄存器 → recvbuff（必需） | Scatter：P·N |
| Legacy NVLS AR（2.2.1） | 两者都用 | SM | Scatter：sendbuff → A | head：MC A → 寄存器 → `multimem.st` MC B；Gather：B → recvbuff | **有**（Gather） | Scatter N + Gather N |
| Sym `AllGather_STMC`（2.3.2） | `multimem.st` | SM | sendbuff → 寄存器 → MC recvbuff | — | **无** | 无 |
| Sym `AllGather_TmaSTMC`【2.32.3】 | TMA 写 MC 地址 | TMA（每个 warp 的 lane 0） | sendbuff → smem → `cp.async.bulk` → MC recvbuff | — | **无** | 无 |
| Sym `ReduceScatter_LDMC`（2.3.3） | `multimem.ld_reduce` | SM | MC sendbuff → 寄存器 → `st` 本卡 recvbuff | — | 寄存器 → recvbuff（必需） | 无 |
| Sym `ReduceScatter_TmaLD`【2.32.3】 | TMA 单播拉取（**非多播**） | TMA + SM | 各对端 sendbuff → smem（TMA）→ 寄存器，在 SM 上归约 → smem → TMA store 本卡 recvbuff | — | smem → recvbuff（必需） | 无 |
| Sym `AllReduce_RSxLDMC_AGxSTMC`（2.3.1.3） | 两者都用 | SM | MC sendbuff → 寄存器 → `multimem.st` MC recvbuff | — | **无** | 无 |
| Sym `AllReduce_RSxTmaLD_AGxTmaST`【2.32.3】 | TMA 单播（**非多播**） | TMA + SM | 各对端 sendbuff → smem → 寄存器，在 SM 上归约 → smem → TMA 单播写 P 张卡的 recvbuff | — | **无** | 无 |
| CE AllGather，单播分支（2.4.2） | CE 单播写 | CE | sendbuff → P-1 张对端卡的 recvbuff；out-of-place 时另有本地自拷贝 | — | **无** | out-of-place 自拷贝 N（语义要求，不是中转） |
| CE AllGather，多播分支【2.32.3】 | CE 写 MC 地址 | CE | sendbuff → MC recvbuff，一次拷贝覆盖所有 P 张卡，**包括自己** | — | **无** | 无（没有自拷贝） |

补充说明：

- **`AllGather_TmaSTMC`**【2.32.3】（`device/symmetric/all_gather.cuh:317-341`，数据路径在 `device/symmetric/primitives.cuh:327-336, 388-399` 的 `tmaLoadStoreMc`）：

  ```text
  lane 0:  memcpy_async_tx(smem ← sendbuff + cursor, tile)        # TMA load，mbarrier 计字节
           bar.wait(barrier_arrive_tx(bar, 1, tile))                # 等 TMA load 完成
           cp.async.bulk.global.shared(MC(recvbuff) + cursor ← smem, tile)   # 写 MC 地址，NVSwitch 复制到 P 张卡
           cp.async.bulk.commit_group; cp.async.bulk.wait_group.read 0      # 只等 smem 被读走
  ```

  与 SM 版 STMC 的区别只是发送端中转从寄存器换成 smem、从 32 个 lane 发指令换成 lane 0 一条 bulk 指令；同步（首尾 LSA barrier）完全相同。启用条件：`minCompCap >= 100`、`NCCL_SYM_TMA_ENABLE=1`（默认开启），并且每个 CTA 的 smem 能容纳 16 个 warp × `ncclTmaShmemScratchWarpSize()`（`sym_kernels.cc:133-141`）。

- **CE AllGather 多播分支**【2.32.3】（`ce_coll.cc:638-707`）：`ncclDevrGetLsaTeamPtrMC` 取得 `recvbuff[rank]` 的 MC 地址，只提交 1 次 `sendbuff → MC` 拷贝。源码注释为 *"a single write to the multicast pointer covers every rank in the LSA team (including self), so no self-copy and no incast"*。是否使用多播由 `ncclCeAllGatherUseMulticast`（`ce_coll.cc:77-98`）决定：需要 `hasLsaMultimem`；设置了 `NCCL_CE_COLL_AG_MULTICAST_THRESHOLD` 时，每 rank 字节数不超过阈值才走多播；否则由 tuning 模型在 `ncclCeMethodId_AllGather_MC` 和单播之间选择。
- **每卡发送方向的流量**：STMC、TmaSTMC 和 CE 多播分支都是 N（由交换机复制）；CE 单播分支是 (P-1)·N。

#### 3.7.4 "数据已就绪"如何得知

| 情形 | 机制 | 是否占用 SM |
|:---|:---|:---|
| `ld` / `ld_reduce` 的结果进入寄存器 | 硬件寄存器依赖（scoreboard）。load 发射不阻塞，warp 在第一条使用结果的指令处停顿，调度器切换到其他 warp | 是（仅发起者） |
| TMA load 进入 smem【2.32.3】 | mbarrier 按字节计数。AllGather 路径（`init(&bar, 1)`）只有 lane 0 arrive 并 `bar.wait`，其余 lane 不参与；RS / AR 路径（`init(&bar, WARP_SIZE)`，`reduce_scatter.cuh:52-55, 91-95`）32 个 lane 都 arrive（lane 0 带上 tx 字节数）并 `bar.wait`，之后各 lane 直接读 smem | 是（仅发起者） |
| Legacy：推送进 B FIFO（AG / AR） | 发送方 `fence.acq_rel.sys` + 写 MC `B_cred.tail`；接收方 Gather 轮询本卡 UC `B_cred.tail` | 是（两端） |
| Legacy：拉取 A FIFO 之前（RS / AR） | head 用 `multimem.ld_reduce.acquire.min` 轮询 MC `A_cred.tail`，确认所有卡都已写好 A slot | 是 |
| Symmetric（SM / TMA） | 没有逐 step 通知。首 barrier 保证所有 rank 已就绪；尾 barrier 中发送方发 `multimem.red.release`，接收方的 kernel 要等齐 P 个 arrive 才能结束 | 是（每个 CTA 的 tid 0） |
| CE | `ncclMemOpSync`：CE 把 4B flag 写到 MC 地址（无多播时逐个写 UC 地址），stream 用 `WAIT_VALUE_32` 等待 | **否** |

【推断】`AllGather_TmaSTMC` 的尾部 barrier 只有 `multimem.red.release`（`include/nccl_device/impl/lsa_barrier__funcs.h:62`）。在它之前，我没有找到 `cp.async.bulk.wait_group 0`（非 `.read` 版本）或 `fence.proxy.async`。TMA store 属于 async proxy，"async proxy 的写入先于 generic proxy 的 release 可见"这一保证，可能依赖硬件或 CTA 退出语义，而不是代码中的显式操作。需要进一步核实。

#### 3.7.5 AllReduce：ReduceScatter 的结果如何进入 AllGather

两种 NVLS AllReduce 中，`multimem.ld_reduce` 的结果都**从寄存器直接以 `multimem.st` 发出**，中间**不会先写到本卡 HBM**，也没有另一组线程再读出来发起 AllGather。区别只在于 `multimem.st` 的落点。

**Legacy NVLS**（`device/all_reduce.h:388-455`，`oneNode` 分支）：

Reduce 组构造 `Primitives<..., FanSymmetric<1>, Direct=1, ProtoSimple<1, 1, COLL_UNROLL, 1, 1>>`，即 `MultimemSrcs = 1`、`MultimemDsts = 1`。recv 和 send 都是 `&nvls->down`，每个 step 调用 `prims.directRecvDirectSend(offset, offset, nelem)`。`regUsed == 0` 时两端指针都落在 FIFO（`prims_simple.h:131-170`），一次 `reduceCopy` 完成：

```text
# Reduce 组（head k）：同一个线程、同一组寄存器
acc[u] = multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2 [MC A[k].slot]   # P 张卡的 A[k] → 交换机求和 → 寄存器
multimem.st.global.v4.f32 [MC B[k].slot], acc[u]                                       # 寄存器 → 交换机复制 → 所有卡的 B[k] slot（HBM）

# Gather（每张卡）：本卡 HBM → 寄存器 → 本卡 HBM
ld.volatile.global [UC B[h].slot] → st.global [recvbuff + h*chunk]
```

- "把归约结果从寄存器写进 HBM 的 FIFO slot"这一步，**就是 `multimem.st`，也就是 AllGather 的跨卡传输本身**。它写的是所有 P 张卡上的 B[k] slot，而不只是本卡。
- 结果落在 B FIFO 而不是 recvbuff，原因是 MNNVL 下 `nvlsRegSupport = 0`（`init.cc:1024`），用户缓冲区没有 MC 映射。所以每张卡还要 Gather 一次 B → recvbuff。这一步加上前面的 Scatter（sendbuff → A），就是 Legacy 的两次额外 HBM → HBM 拷贝。
- 同步：写 B 之前要通过 ② 确认所有卡的 Gather 已读完 B slot（min-poll `B_cred.head`）；写完后执行 ④，即 `fence.acq_rel.sys` + 写 MC `B_cred.tail`（见 2.2.1 第 5 小节）。

**Symmetric NVLS `AllReduce_RSxLDMC_AGxSTMC`**（`device/symmetric/all_reduce.cuh:299-379`，2.32.3 中 `all_reduce.cuh:438-507` 逻辑相同）：

```text
# 全局 warp W 负责 4KB chunk c（W = rank + P*(block + nBlocks*warp)，chunk 按 W 轮转）
tmp[u] = multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2 [MC(sendbuff) + c*4KB + u*512 + lane*16]
multimem.st.global.v4.f32 [MC(recvbuff) + c*4KB + u*512 + lane*16], tmp[u]
```

- 同一个线程、同一组寄存器，中间没有 HBM 落地，也**没有任何软件同步**，只依赖寄存器依赖。
- `multimem.st` 的落点就是**用户 recvbuff 的 MC 映射**，交换机直接写入所有卡的最终位置。所以没有 Scatter、没有 Gather，接收端零搬运。
- in-place（`sendbuff == recvbuff`）同样安全：每个 4KB chunk 只有一个 owner，owner 先 `ld_reduce` 读取所有卡上的 chunk c，再 `multimem.st` 写回所有卡的 chunk c；其他 owner 不访问 chunk c【推断：依据切分方式】。
- 换句话说，3.7.2 中"`ld_reduce` 之后必需的那次 store"在 AllReduce 中被 AllGather 吸收了：它就是 `multimem.st`。

**对比**

| | Legacy NVLS AllReduce | Symmetric `RSxLDMC_AGxSTMC` |
|:---|:---|:---|
| `ld_reduce` 读取 | MC A FIFO（Scatter 写入的中转副本） | MC 用户 sendbuff |
| 归约结果的出口 | 寄存器 → `multimem.st` | 寄存器 → `multimem.st` |
| `multimem.st` 写到 | 所有卡的 B FIFO | 所有卡的用户 recvbuff |
| 是否先存本卡 HBM 再发起 AG | 否 | 否 |
| 额外的本卡 HBM → HBM 拷贝 | Scatter（sendbuff → A）+ Gather（B → recvbuff） | 无 |
| 每卡 HBM 流量（≈） | 6N：Scatter 读写 2N + 交换机读 A N + 交换机写 B N + Gather 读写 2N | 2N：交换机读 sendbuff N + 交换机写 recvbuff N |
| 同步 | 每 step、每个 head 4 次跨卡握手 | kernel 首尾各一次 LSA barrier |

【2.32.3】中的 `AllReduce_RSxTmaLD_AGxTmaST`（`device/symmetric/all_reduce.cuh:150-250, 433-436`）不使用多播：lane 0 用 TMA 把各对端 sendbuff 的 tile 拉进 smem，全 warp 在 SM 上用 fp32 归约，结果写回 smem，执行 `fence.proxy.async.shared::cta` 后，由 lane 0 对 P 张卡的 recvbuff 各发一次 `cp.async.bulk` 单播写入。归约结果同样不经过本卡 HBM 中转，只是出口从 `multimem.st` 一次写换成了 P 次 TMA 单播写。
