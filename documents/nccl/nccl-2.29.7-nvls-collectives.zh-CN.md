# NCCL 2.29.7 中基于 NVLS 的 AllGather / ReduceScatter / AllReduce 实现

日期：2026-10-12。

源码版本：NVIDIA NCCL **v2.29.7-1**，commit `b91894bd5b190c874d98a017f93f5daa515b65d0`，本地路径 `Megatron-Scale-Down/nccl-2.29.7`。下文的源码位置均相对该目录。

相关文档：

- [PTX 9.4 multimem 指令](ptx-9.4-multimem.zh-CN.md)：`multimem.ld_reduce` / `st` / `red` 的语义、流量模型、成员语义。本文默认读者已了解。
- [NVLS 资源](nccl-nvls-resources.zh-CN.md)：multicast group 配额、交换机数据面资源、staging 显存。

场景约束：通信组在一个 NVL72（MNNVL clique）或一台 NVSwitch 单机内，`nNodes == 1`，**不走 RDMA**。

本文结论均来自源码阅读，**没有 GPU 实测**。标【推断】的内容是根据代码做的推理，源码里没有直接说明。

---

## 0. 结论速览

### 0.1 实现路径一览

| 路径 | collective | kernel | multimem 指令 | 适用条件 |
|---|---|---|---|---|
| **经典 NVLS** | AG | `RunWorkColl<AllGather, NVLS, SIMPLE>` | `st` | AG/RS 要求 `localRanks ≤ 32` |
| | RS | `RunWorkColl<ReduceScatter, NVLS, SIMPLE>` | `ld_reduce` | 同上 |
| | AR | `RunWorkColl<AllReduce, NVLS, SIMPLE>` | `ld_reduce` + `st` | P 不受 32 限制，但最多 32 个 head |
| **Symmetric** | AG | `AllGather_STMC` / `AllGather_LLMC` | `st` | recv 注册为 symmetric window（LLMC 不要求） |
| | RS | `ReduceScatter_LDMC` | `ld_reduce` | send 注册为 symmetric window |
| | AR | `AllReduce_RSxLDMC_AGxSTMC`（大消息）/ `AllReduce_AGxLLMC_R`（小消息） | `ld_reduce` + `st` / `st` | send、recv 都注册（LLMC_R 可回退） |
| **CE**（不是 NVLS） | AG | `cudaMemcpyBatchAsync` | 无（只有同步 flag 写 MC） | recv 注册；`NCCL_CTA_POLICY=ZERO` 或大消息自动选中 |
| **GIN + LSA** | AG / RS | `*_Rail*_Lsa*MC` | `st` / `ld_reduce` | 只在 comm 跨多个 NVLink 域时；NVL72 内被屏蔽 |
| **Device API** | 用户自定义 | 用户 kernel | `st` / `ld_reduce` / `red` | `ncclDevCommCreate(lsaMultimem=true)` |

### 0.2 核心结论

- **NVLS 数据面全部由 SM 执行 multimem 指令完成**。CE 不参与 NVLS 数据面；CE 也做不了归约。
- **AG** = 每个 rank 对自己 shard 的 MC 地址执行 `multimem.st`（push 广播）。
- **RS** = 每个 rank 对“自己负责的输出 shard”的 MC 地址执行 `multimem.ld_reduce`（pull 归约）。
- **AR** = RS + AG，但**由同一组线程按 chunk 流水执行**：对一个 pack 先 `ld_reduce` 再 `st`。RS 主要占 TX，AG 主要占 RX，两个方向同时被用满。
- **head** 是经典 NVLS 路径中的概念：负责一段数据的归约 / 广播的 rank。Symmetric 路径没有 head。详见 1.2 节。

---

## 1. 术语

### 1.1 术语表

| 术语 | 含义 | 出处 |
|---|---|---|
| **NVLS** | NVLink SHARP。利用 NVSwitch 的 multicast 和交换机内归约实现 collective | — |
| **multicast object / team / group** | 绑定在一组 GPU 物理内存上的多播对象。NCCL 日志里叫 “Multicast group”，Device API 里叫 team 的 multimem 句柄。一个 object 占用交换机上一个 group 配额 | `src/transport/nvls.cc:59` |
| **UC / MC 地址** | UC 只指向本卡副本；MC 指向全部副本，只能用 multimem 指令访问 | 见 PTX 文档 1.1 节 |
| **MNNVL / clique** | Multi-Node NVLink。NVL72 跨 tray 的 GPU 组成一个 clique，NCCL 把它融合成一个拓扑：`nNodes = 1`，`localRanks = clique.size` | `src/graph/topo.cc:1543` |
| **arity** | 经典 NVLS 中 scatter/gather 一次能连接的 peer 数上限，`NCCL_MAX_NVLS_ARITY = 32` | `src/include/device.h:210` |
| **head** | 经典 NVLS 中“负责一段数据的 multimem 操作”的 rank，见 1.2 节 | `src/include/device.h:212-220` |
| `nHeads` | head 数，= min(32, P)（单机） | 同上 |
| `headRank` | 本 rank 是第几个 head，`0..nHeads-1`；**不是 head 时为 -1** | 同上 |
| **虚拟 peer** `up[h]` / `down` | `up[h] = nRanks+1+h`，代表“第 h 个 head 的 staging”；`down = nRanks+1+headRank`，代表“自己作为 head 的 staging”。它们不是真实 rank 号，只是 `channel->peers[]` 的下标 | `src/graph/connect.cc:254-256` |
| **NVLS channel** | 经典 NVLS kernel 的 CTA 数（每个 rank 上）。SM90 默认 16，SM100 单机 / NVL72 默认 24，SM100 多节点 32 | `src/transport/nvls.cc:149-200` |
| **chunk / `chunkSize`** | 一个 head 在一轮中处理的元素数，上限 `NCCL_NVLS_CHUNKSIZE`（默认 128 KB），小消息时缩到 64K/32K/16K | `src/enqueue.cc:2153-2167` |
| **step / slot / `NCCL_STEPS`** | staging FIFO 有 8 个 slot，每个 slot 大小 = chunkSize。生产者和消费者最多相差 8 步 | `src/include/device.h:26` |
| **staging buffer** | 经典 NVLS 中 NCCL 内部分配的 multicast 中转区。每个 (head, channel) 有两段：**reduce staging**（归约输入）和 **bcast staging**（广播输出） | `src/transport/nvls.cc:314-360` |
| **credit（FIFO 的 head / tail 计数）** | 生产者写 `tail`、消费者写 `head`，用来控制 FIFO 流水。**注意：这里的 head 是 FIFO 指针，与 head rank 无关**，见 1.4 节 | `src/transport/nvls.cc:436-471` |
| `NCCL_NVLS_MIN_POLL` | 连接标志。带这个标志的连接用 `multimem.ld_reduce.min` 轮询 credit，即“等所有 rank 中最慢的那个” | `src/include/device.h:120`，`src/device/prims_simple.h:96-107` |
| `regUsed` | 用户 buffer 已注册进 multicast object，kernel 直接对用户 buffer 的 MC 地址执行 multimem，不需要 staging 拷贝。**MNNVL 下禁用** | `src/init.cc:1024` |
| `oneNode` | `nNodes == 1`。NVL72 内恒为真 | — |
| `ProtoSimple<…, MultimemSrcs, MultimemDsts>` | Simple 协议原语的模板参数。`MultimemSrcs=1` 表示第 0 个源用 `ld_reduce` 读；`MultimemDsts=1` 表示第 0 个目的用 `multimem.st` 写 | `src/device/common_kernel.h:83-162` |
| **LSA / LSA team** | Load-Store Accessible：comm 中可以用 NVLink load/store 直接访问的 rank 集合。NVL72 内等于整个 comm | `src/dev_runtime.cc:77-100` |
| **symmetric window** | 用 `ncclCommWindowRegister(..., NCCL_WIN_COLL_SYMMETRIC)` 注册的 buffer。所有 rank 在同一偏移处有对应的内存，且整个 LSA team 共享一个 multicast object | `src/dev_runtime.cc:264-345` |
| **LSA barrier** | Device API 的 barrier。multimem 版本用 `multimem.red.add` 让所有副本的计数器 +1，再轮询本地副本 | `src/include/nccl_device/impl/lsa_barrier__funcs.h:63-80` |
| **CE** | Copy Engine，执行 `cudaMemcpy*`，不占 SM | `src/ce_coll.cc` |
| **GIN** | GPU-Initiated Networking，Device API 中**面向网卡**的部分（GDAKI 或 CPU Proxy 后端） | `src/include/nccl_device/gin.h` |

Symmetric kernel 的命名规则（`src/sym_kernels.cc:16-32`）：

| 片段 | 含义 |
|---|---|
| `ST` / `STMC` | 用 store 写对端：unicast（逐 peer）/ multicast（`multimem.st`） |
| `LD` / `LDMC` | 用 load 读对端：unicast 后在 SM 上归约 / multicast（`multimem.ld_reduce`，交换机归约） |
| `LL` / `LLMC` | LL 协议（8B 数据 + flag）：unicast / multicast 广播 |
| `RSxLDMC_AGxSTMC` | AllReduce = RS（LDMC）× AG（STMC） |
| `AGxLLMC_R` | AllReduce = AG（LLMC）后本地 Reduce |
| `Rail…_Lsa…` | 跨 NVLink 域时：rail 方向走 GIN（网卡），LSA 域内走 multimem |

### 1.2 head 详解

**一句话**：head 是“负责一段数据的 multimem 操作”的 rank。每轮把数据切成 nHeads 段，head h 负责第 h 段。

- 在 AllReduce 中，head h 对第 h 段先 `ld_reduce` 再 `st`。
- 在 ReduceScatter 中，head h 对第 h 段 `ld_reduce`，结果写到本地输出。
- 在 AllGather 中，head h 把自己的输入 `st` 到第 h 段。

**head 的来历**：这个名字来自 CollNet Direct 与多节点 NVLS+SHARP。`struct ncclNvls` 中 `nHeads` 的注释是 “Number of parallel N<->1<->net operations”，`headRank` 的注释是 “-1 if I'm not a head rank (no local NIC)”（`src/include/device.h:214-215`）。多节点时，head 是负责把本机归约结果发给网卡的 rank。**单机 NVLS 中 head 与网卡无关**，只是 multimem 操作的发起者。

**head 怎么选出来**：

1. NVLS 图搜索中，**一个 graph channel 对应一个 head**，`maxChannels = min(NCCL_MAX_NVLS_ARITY=32, nGPU)`（`src/graph/search.cc:1065`）。单机时还强制 `minChannels = maxChannels`（`search.cc:1070-1073`，注释 “Force intra-node NVLS algorithm to pull evenly from all GPUs”）。
2. 单机搜索第 c 个 channel 时，直接从拓扑中第 c 张 GPU 开始：`ncclTopoSearchTryGpu(..., g = graph->nChannels)`（`search.cc:794`、`338-341`）。这张 GPU 被写到 `intra[c*ngpus + 0]`（`search.cc:597`）。
3. `ncclTopoSearchTryNvls`（`search.cc:387-417`）检查 NVS↔GPU 双向带宽：**head 那张卡记 2 倍带宽，其余卡记 1 倍**。因为 head 除了被读、被写之外，还要自己发起 multimem 操作。
4. `intra[c*localRanks + 0]` 去重后成为 `nvlsHeads[]`（`src/graph/connect.cc:75-90`）。`connectNvls` 中若 `nvlsHeads[h] == 本 rank`，则 `headRank = h`（`connect.cc:249-251`）。

结果：**head 就是拓扑顺序中的前 min(32, P) 张 GPU**。P ≤ 32 时每个 rank 都是 head。

**head 与 rank 的对应**【推断】：经典 AG 的 gather 把第 h 个 head 的数据写到 `recvbuff + h*count`，经典 RS 的 scatter 把 `sendbuff + h*count` 交给第 h 个 head（第 3、4 节）。所以 AG/RS 的正确性隐含要求“第 h 个 head 就是 rank h”。注册 buffer 的 RS 路径直接用 `rank*count` 作偏移（`src/device/reduce_scatter.h:279`），也说明作者假定二者一致。源码中没有找到显式检查；常见机器上拓扑 GPU 顺序与 rank 顺序一致。AllReduce 只按 head 编号切 chunk，不依赖这个对应关系。

**head 与 NVLS channel 是两个正交的维度**：

| 概念 | 数量 | 作用 |
|---|---|---|
| head（NVLS graph channel） | min(32, P) | **按数据段**分工：第 h 段由 head h 负责 |
| NVLS channel（CTA） | 16 / 24 / 32 | **按消息区间**分工：每个 CTA 处理消息的一个区间（`ncclCollCbdPart`） |

每个 CTA 处理一个区间；在这个区间内，每轮再切成 nHeads 个 chunk，分给 nHeads 个 head。所以同时在途的 multimem 操作数为 `nvlsChannels × nHeads`（`src/enqueue.cc:2163`），staging 显存也按这个乘积分配。

### 1.3 为什么最多只有 32 个 head

**代码层面的硬限制**：`NCCL_MAX_NVLS_ARITY = 32` 是编译期常量，以下地方都依赖它：

| 位置 | 作用 |
|---|---|
| `src/graph/search.cc:1065` | NVLS graph channel 数（= head 数）上限为 `min(32, nGPU)` |
| `src/include/device.h:216` `int up[NCCL_MAX_NVLS_ARITY]` | 每个 channel 最多记录 32 个 head 虚拟 peer |
| `src/device/all_reduce.h:419,431` 等 `FanAsymmetric<0/32, 32/0>` | scatter / gather 的 Primitives 最多只能连 32 个 peer |
| `src/device/common.h:34-35` `srcs/dsts[NCCL_MAX_ARITY+1]` | sm90+ 上 `NCCL_MAX_ARITY = NCCL_MAX_NVLS_ARITY`，shared memory 中的指针数组按 32 分配 |

**为什么这样设计**【推断】：

1. **head 数不影响链路带宽**。归约发生在交换机内，每张卡都要被读一遍、被写一遍，与它是不是 head 无关。head 只多出 1/nHeads 的“自环”流量（第 9 节）。把 head 从 32 增加到 72，链路不会更快。
2. **staging 显存随 head 数线性增长**：`nHeads × nvlsChannels × 2 × 1 MiB`。NVL72 上 32 个 head 已经是每卡 1.5 GiB；72 个 head 会到约 3.4 GiB（见资源文档）。
3. **粒度变粗**。每轮处理 `nHeads × chunkSize` 个元素（`src/device/all_reduce.h:412`）。head 越多，小消息越难喂满所有 head。
4. **scatter / gather 要逐个处理 nHeads 个 peer**（`src/device/prims_simple.h:446-458, 466-475`）。head 越多，每个 peer 分到的 slice 越小，循环开销越大。

**为什么 AllReduce 允许 P > 32，而 AG / RS 不允许**（`src/enqueue.cc:1914`）：

- **AllReduce** 按 head 编号切 chunk，与 rank 编号无关。所有 rank 都会 gather 到全部 nHeads 段结果，所以 32 个 head 足以覆盖全部数据。
- **ReduceScatter** 中，rank r 的输出就是第 r 段，必须由 rank r 用自己的 `down` 去做 `ld_reduce`（`src/device/reduce_scatter.h:240, 247-256`）。这要求**每个 rank 都是 head**，所以 P 不能超过 32。AllGather 同理：每个 rank 都要用自己的 `down` 广播自己的 shard。

### 1.4 staging 与 credit 的布局

每个 (head h, channel c) 有两段 staging，每段 `chunkSize × NCCL_STEPS = 1 MiB`；每段各配一对 credit 计数（`head` / `tail`），credit 也放在 multicast 内存里（`src/transport/nvls.cc:344-357, 436-471`）：

| 连接 | 方向 | 数据 buffer | 使用者 | 等待方式 | 通知方式 |
|---|---|---|---|---|---|
| `send[1]` | reduce：UC 写入 | 本卡 UC reduce staging | scatter warp | 读本卡 UC `head` | 写本卡 UC `tail` |
| `recv[0]` | reduce：MC 读取 | MC reduce staging | head 的 reduce warp | `ld_reduce.acquire.sys.min.u64` 读 MC `tail`，即**所有 rank 中最慢的进度** | 写 MC `head`，交换机复制到所有 rank |
| `send[0]` | bcast：MC 写入 | MC bcast staging | head 的 bcast / reduce warp | 同上，读 MC `head` 取 min | 写 MC `tail`，复制到所有 rank |
| `recv[1]` | bcast：UC 读取 | 本卡 UC bcast staging | gather warp | 读本卡 UC `tail` | 写本卡 UC `head` |

要点：

- 通知用的是 `postPeer` 中对 MC 地址的普通 `st.relaxed.sys`（`src/device/prims_simple.h:174-182`），交换机会把它复制到所有副本。
- 因此 head 开始处理第 k 步之前，已确认**所有 rank 都写完了第 k 步的输入**，并且**所有 rank 都取走了第 k−8 步的输出**。
- 再次强调：表中的 `head` / `tail` 是 **FIFO 指针**，与 “head rank” 同名但无关。

### 1.5 每个 CTA 的 warp 分工

每个经典 NVLS CTA 有 640 个线程（`NCCL_MAX_NTHREADS`，20 个 warp），按角色分组、流水并行：

| collective | 非注册 buffer | 注册 buffer |
|---|---|---|
| AG（`all_gather.h:250-257`） | gather 8 warp + bcast 12 warp | gather `roundUp(nHeads*4, 32)` 个线程（只做同步）+ bcast 其余 |
| RS（`reduce_scatter.h:220-226`） | scatter 8 warp + reduce 12 warp | scatter `roundUp(nHeads*4, 32)` 个线程（只做同步）+ reduce 其余 |
| AR（`all_reduce.h:394-407`） | P ≤ 6：scatter 7 + gather 6 + reduce 7；否则 8 + 7 + 5 | scatter 1 + gather 1 + reduce 18 |

角色含义：

| 角色 | 做什么 | 访问哪里 |
|---|---|---|
| **scatter** | 把 `sendbuff` 的第 h 段拷到本卡第 h 个 head 的 UC reduce staging | 只访问本卡显存 |
| **gather** | 把本卡第 h 个 head 的 UC bcast staging 拷到 `recvbuff` 的第 h 段 | 只访问本卡显存 |
| **bcast** | 对 MC bcast staging（或 MC recvbuff）执行 `multimem.st` | 发起 NVLink 事务 |
| **reduce** | 对 MC reduce staging（或 MC sendbuff）执行 `multimem.ld_reduce`；AR 中紧接着 `multimem.st` | 发起 NVLink 事务 |

scatter / gather 存在的原因：**用户 buffer 没有映射进 multicast object**，必须先拷到 staging（multicast object 背后的物理页）才能被 `ld_reduce` 读到；广播结果也只能先落在 staging，再拷回用户 buffer。注册 buffer 后，这两次拷贝就省掉了。

---

## 2. Host 调用路径选择

```text
ncclAllGather / ncclReduceScatter / ncclAllReduce
  └─ taskAppend (src/enqueue.cc:2885)
      ├─ [CE]  CTAPolicy=ZERO，或（AG、symmetric、count>8M、sm100、全直连 NVLink）
      │        → ceCollTaskAppend → cudaMemcpyBatchAsync   （不是 NVLS，只支持 AG/A2A/Scatter/Gather）
      └─ collTaskAppend
           └─ ncclPrepareTasks (src/enqueue.cc:397)
               ├─ [Symmetric] ncclMakeSymmetricTaskList → ncclSymkPickKernel
               │        → AllGather_STMC / ReduceScatter_LDMC / AllReduce_RSxLDMC_AGxSTMC ...
               └─ [经典] ncclGetAlgoInfo → NCCL_ALGO_NVLS + SIMPLE
                        → RunWorkColl<..., NVLS, SIMPLE>
```

### 2.1 经典 NVLS 的前置条件

| 条件 | 源码 |
|---|---|
| `CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED` 且 `NCCL_NVLS_ENABLE≠0` | `src/transport/nvls.cc:156-205` |
| op / dtype（RS、AR）：i32/u32/i64/u64/f16/bf16 支持 Sum、MinMax；f32/f64 只支持 Sum。AG 不受限制 | `src/include/device.h:566`，`src/enqueue.cc:421` |
| **AG/RS 要求 `localRanks ≤ 32`**；AR 不受此限制 | `src/enqueue.cc:1914` |
| 只支持 SIMPLE 协议；`nNodes>1` 时必须有 CollNet | `src/graph/tuning.cc:281`，`src/enqueue.cc:1915` |
| 单机时 NVLS_TREE、CollNet+NVLS 被禁用 | `src/graph/tuning.cc:467,472` |
| **MNNVL 下禁用经典 NVLS 的用户 buffer 注册**（`nvlsRegSupport=0`），只能走 staging 路径 | `src/init.cc:1024`，`src/transport/nvls.cc:484` |

对 NVL72 的影响：

- 通信组 > 32 卡时，AG/RS 不能走经典 NVLS，只能走 symmetric kernel 或 CE；AR 仍可走经典 NVLS（最多 32 个 head）。
- 通信组 ≤ 32 卡时可以走经典 NVLS，但在 MNNVL 下总是走“staging + 本地拷贝”的非注册路径。

### 2.2 Symmetric 路径的前置条件

- buffer 通过 `ncclCommWindowRegister(..., NCCL_WIN_COLL_SYMMETRIC)` 注册。Megatron 中由 `megatron/core/nccl_allocator.py` 的 `register_mem_pool(pool, symm=True)` 完成。
- `isAllDirectNvlink`。
- `symmetricSupport`（NVL72 内 LSA team 等于整个 comm 即可，不需要 GIN）。
- `hasLsaMultimem = nvlsSupport && lsa.nRanks > 2`（`src/sym_kernels.cc:414`）。

相关源码：`src/sym_kernels.cc:414, 512-565`，`src/init.cc:1475`。

---

## 3. 经典 NVLS AllGather

源码：`src/device/all_gather.h:250-317`（`oneNode` 分支）。前提：P ≤ 32，**每个 rank 都是 head**。

### 3.1 非注册 buffer

```text
rank r（headRank = r）：
  bcast  warp：prims.send(offset, nelem)                               (all_gather.h:285)
               ProtoSimple<…, MultimemSrcs=0, MultimemDsts=1>，sendPeer = down
               ld 本地 sendbuff ──► multimem.st( MC bcastStaging[r] )
               └─ NVSwitch 复制 ──► 每张卡的 UC bcastStaging[r]
  gather warp：prims.gather(offset, nHeads*count, nelem, count, -1, 0) (all_gather.h:273)
               for h in 0..nHeads-1:
                 本卡 UC bcastStaging[h] ──普通拷贝──► recvbuff[h*count + offset]
```

- 每个 rank 只广播自己的 shard；收到所有 head 的广播后，gather warp 把它们拷进 `recvbuff` 对应位置。
- 每个字节在接收方多一次 HBM 读写（gather 拷贝）。

### 3.2 注册 buffer（单机、非 MNNVL）

```text
bcast warp：prims.directSend(inpOffset, outOffset = inpOffset + rank*count, nelem)   (all_gather.h:314)
            multimem.st( MC(recvbuff) + rank*count )  ── 一次完成广播，直接落到所有 rank 的 recvbuff
gather warp：只做同步（scatter/gather 的 nelem = 0）
```

---

## 4. 经典 NVLS ReduceScatter

源码：`src/device/reduce_scatter.h:228-290`（`oneNode` 分支）。前提：P ≤ 32，**每个 rank 都是 head**。

### 4.1 非注册 buffer

```text
rank r（headRank = r）：
  scatter warp：prims.scatter(offset, nHeads*count, nelem, count, -1, 0)   (reduce_scatter.h:243)
                for h in 0..nHeads-1:
                  sendbuff[h*count + offset] ──普通写──► 本卡 UC reduceStaging[h]
  reduce  warp：prims.recv(offset, nelem)                                  (reduce_scatter.h:255)
                ProtoSimple<…, MultimemSrcs=1, MultimemDsts=0>，recvPeer = down
                multimem.ld_reduce( MC reduceStaging[r] )
                └─ NVSwitch 读取 P 张卡上的第 r 段并求和 ──► 寄存器 ──普通写──► 本地 recvbuff
```

- 这是 **pull** 模式：rank r 只拉取自己那一段的归约结果。
- 每个字节在发送方多一次 HBM 读写（scatter 拷贝）。

### 4.2 注册 buffer（单机、非 MNNVL）

```text
reduce warp：prims.directRecvCopy(inpOffset = outOffset + rank*count, outOffset, nelem)   (reduce_scatter.h:284)
             multimem.ld_reduce( MC(sendbuff) + rank*count ) ──► 本地 recvbuff
scatter warp：只做同步
```

### 4.3 AG 与 RS 数据流对照

```text
AG:  rank r: ld 本地 send_r ──(SM)──► multimem.st MC[r 段] ──NVSwitch 复制──► 每卡 recv[r 段]
RS:  每卡 send[q 段] 都在 HBM ◄──NVSwitch 读取 P 份并求和── multimem.ld_reduce MC[q 段]（由 rank q 发起）
     ──► 寄存器 ──(SM)──► rank q 本地 recv
```

---

## 5. 经典 NVLS AllReduce

源码：`src/device/all_reduce.h:388-455`（`oneNode` 分支）。P 不受 32 限制，nHeads = min(32, P)。

### 5.1 chunk 划分

- 每个 CTA 处理消息的一个区间（`ncclCollCbdPart`）。区间内每轮处理 `loopCount = nHeads × chunkSize` 个元素（`all_reduce.h:412`）。
- head h 负责其中第 h 个 chunk：`chunkOffset = elemOffset + headRank*chunkSize`。
- 最后一轮不满时，`lastChunkSize = alignUp(divUp(remCount, nHeads), 16384/sizeof(T))`。

### 5.2 非注册 buffer 的数据流（P=4，以 head h 负责的 chunk 为例）

```text
① Scatter（每个 rank，本地 HBM）：      send_r[chunk h] ──st──► 本卡 UC reduceStaging[h]
② Reduce（仅 head h 发起）：            multimem.ld_reduce( MC reduceStaging[h] )
                                         └─NVSwitch 读取 4 份副本并求和─► head h 的寄存器
③ Bcast（head h 发起，同一线程紧接着）： multimem.st( MC bcastStaging[h] ) ─NVSwitch 复制─► 4 张卡的 UC bcastStaging[h]
④ Gather（每个 rank，本地 HBM）：       本卡 UC bcastStaging[h] ──copy──► recv_r[chunk h]
```

- 四段之间通过 credit（1.4 节）流水推进。
- 代价：每个字节在本卡多 2 次 HBM 读写（①、④）。
- NVL72（MNNVL）上经典路径总是走这种 staging 模式。

### 5.3 “先 `ld_reduce` 再 `multimem.st`”是什么意思

调用链：

```text
RunWorkColl<AllReduce, NVLS, SIMPLE>::run      (src/device/all_reduce.h:440-454)
  Primitives<..., ProtoSimple<1,1,UNROLL, MultimemSrcs=1, MultimemDsts=1>>(recvPeer=&down, sendPeer=&down, connIndex 0/0)
  └─ prims.directRecvDirectSend(offset, offset, nelem)       (src/device/prims_simple.h:879-881)
     └─ genericOp<1,1,1,1,-1,-1>
        └─ reduceCopy<..., MultimemSrcs=1, ..., MultimemDsts=1, ...>(srcs[0], dsts[0])   (prims_simple.h:277-284)
           └─ reduceCopyPacks                                  (src/device/common_kernel.h:83-162)
                acc = applyLoadMultimem(srcs[0])  → multimem.ld_reduce.relaxed.sys.global.add[.acc::f32].v4.*   (reduce_kernel.h:898)
                multimem_st_global(dsts[0], acc)  → multimem.st.global.v4.f32                                   (src/device/op128.h:399-402)
```

- `srcs[0]` 是 `recv[0]` 的 buffer，即 head h 的 **MC reduce staging**；`dsts[0]` 是 `send[0]` 的 buffer，即 **MC bcast staging**（`src/transport/nvls.cc:350-355`）。注册 buffer 时换成 `MC(sendbuff)` / `MC(recvbuff)`。
- 同一个线程对同一个 16B pack 依次执行两条指令：
  1. **`multimem.ld_reduce [MC_reduce + off]`**：SM 只发出一次读请求。交换机读取每张卡上的副本（即各 rank 在 scatter 阶段写进去的数据），求和，把**一份**结果返回到 head 的寄存器。等价于对这 16B 做 Reduce，根在 head。
  2. **`multimem.st [MC_bcast + off], acc`**：SM 只发出一次写，交换机把它复制到每张卡（包括 head 自己）。等价于对这 16B 做 Broadcast。
- 合起来就是对这 16B 的 **AllReduce = Reduce-to-head + Broadcast-from-head**。归约结果只在寄存器里停留，不先写到 head 的显存。
- 从整体看：所有 head 的 `ld_reduce` 合起来是 ReduceScatter，所有 head 的 `st` 合起来是 AllGather。二者按 pack 交错执行，所以 RS 和 AG 天然重叠。

### 5.4 非 head GPU（只在 P > 32 时出现）在做什么

非 head GPU 的 reduce warp 直接退出（`all_reduce.h:440` 要求 `headRank != -1`），只剩 scatter 和 gather：

**Scatter（喂料）**，`ScatterGatherOp<..., Send=1>`，`prims_simple.h:437-458`：

```text
for 每轮 elemOffset:
  for h in 0..nHeads-1:
    等本卡 UC credit：slot h 的 reduce staging 有空位
    sendbuff[elemOffset + h*chunkSize : +chunkSize]  ──普通 ld/st──►  本卡 UC reduceStaging[h]
  写本卡 UC tail：第 k 步就绪
```

**Gather（收料）**，`ScatterGatherOp<..., Recv=1>`，`prims_simple.h:459-476`：

```text
for 每轮 elemOffset:
  for h in 0..nHeads-1:
    等本卡 UC tail：head h 已把第 k 步结果 multimem.st 到这里（tail 也由 head 经 MC 广播写入）
    本卡 UC bcastStaging[h]  ──普通 ld/st──►  recvbuff[elemOffset + h*chunkSize : +chunkSize]
  写本卡 UC head：归还 slot（head 会用 ld_reduce.min 读到所有 rank 的这个值）
```

两步都只访问本卡显存。它链路上的流量是被动产生的：

- head 执行 `ld_reduce` 时，交换机读取本卡 UC reduce staging，占用本卡 **TX**。
- head 执行 `st` 时，交换机写入本卡 UC bcast staging，占用本卡 **RX**。
- head 的 credit 轮询与通知，也由交换机替它读写本卡的计数。

| 角色 | 本卡 SM 的工作 | 本卡发起的 NVLink 事务 | 本卡链路流量（总量 S） |
|---|---|---|---|
| head | scatter + gather + 对 1/nHeads 的数据执行 `ld_reduce` / `st` | `ld_reduce`、`st`、MC credit | TX ≈ RX ≈ S + S/nHeads |
| 非 head | 只做 scatter + gather | **无** | TX ≈ RX ≈ S（被动） |

注册 buffer 时，scatter / gather 的 `nelem = 0`（`all_reduce.h:425, 437`），只剩 credit 握手。这时非 head GPU 的 SM 几乎不做事。

---

## 6. Symmetric kernel

Symmetric kernel 直接对用户 window 的 MC 地址执行 multimem：**没有 head、没有 staging、没有逐 step 的 credit**，只在首尾各做一次 LSA barrier。

### 6.1 共同结构

```text
bar.sync(multimem LSA barrier)          // multimem.red 计数，确保所有 rank 的输入就绪
for 本线程负责的元素：                   // 线程按 (warp lane, block, warp) 交错编号（flattenIx）
    <multimem 操作>
bar.sync(release)                       // 确保所有 rank 的写入可见后才返回
```

- CTA 数：Blackwell 上 `nMaxBlocksNvls = ceil(32 / nRanks)`（`src/sym_kernels.cc:319`），8 卡约 4 个 CTA，72 卡 1 个 CTA（每 CTA 512 线程）。**SM 占用远低于经典 NVLS 的 24 个 channel。**
- 没有 32 rank 的 arity 限制。
- 可以用 `NCCL_SYM_KERNEL=<kernel 名>` 强制选择（`src/sym_kernels.cc:102-122`），用 `NCCL_SYM_CTAS` 调 CTA 数。

### 6.2 各 kernel

| kernel | 逻辑 | 源码 |
|---|---|---|
| `AllGather_STMC` | 每个线程读本地 input 的 16B pack，`multimem.st` 写到 `output.multimemPtr + rank*nAllElts`。**rank r 只广播自己的 shard** | `src/device/symmetric/all_gather.cuh:195-215`，`primitives.cuh:234` |
| `ReduceScatter_LDMC` | 对 `input.multimemPtr + rank*nAllElts` 执行 `ld_reduce`，普通写入本地 output（`reduceMultimem`）。**rank r 只拉取自己的 shard** | `src/device/symmetric/reduce_scatter.cuh:261, 312-335` |
| `AllReduce_RSxLDMC_AGxSTMC` | 全局线程（跨 rank → block → warp 交错编号）平均瓜分整个 `[0, S)`；每个线程 `tmp = ld_reduce(MC(input)+i)`，紧接着 `st(MC(output)+i, tmp)` | `src/device/symmetric/all_reduce.cuh:299-380` |
| `AllReduce_AGxLLMC_R` | 每个线程把 8B 输入以 LL 格式（数据 + flag）`multimem.st` 广播到所有 rank 的 LL slot（`lla2a.bcast`），再从本卡 P 个 slot 读出，在 SM 上 fp32 累加后写 output。**不用交换机 ALU** | `src/device/symmetric/all_reduce.cuh:382-475`，`src/include/nccl_device/impl/ll_a2a__funcs.h:73-` |
| `AllGather_LLMC` | 小消息 AG：LL 格式 + `multimem.st` | `src/device/symmetric/all_gather.cuh` |

与经典路径对比，`AllReduce_RSxLDMC_AGxSTMC` 中“由谁负责哪段”不再按 head 划分，而是**所有 rank 的所有线程平均分**，因此 P > 32 时也没有 head 不对称的问题。

### 6.3 选用条件与限制

| kernel | 条件 |
|---|---|
| `ReduceScatter_LDMC` | Sum，dtype f32/f16/bf16/fp8（不支持 f64）；**send 已注册** |
| `AllGather_STMC` | **recv 已注册** |
| `AllReduce_RSxLDMC_AGxSTMC` | Sum，dtype f32/f16/bf16/fp8（`src/sym_kernels.cc:485-510`）；**send 与 recv 都已注册**，否则只能用 LL kernel（`src/sym_kernels.cc:580-581`） |
| `AllReduce_AGxLLMC_R` | 小消息；nBusBytes ≥ 2 GB 时禁用 LL kernel。代价模型 `busBytes = P·S·9`，`busMultiplier = 1.1`（`src/sym_kernels.cc:331-334`） |

- `AllReduce_RSxLDMC_AGxSTMC` 的代价模型 peak 取 1000 GB/s，其他 symmetric kernel 取 600（`src/sym_kernels.cc:391`），说明 NCCL 认为它能同时用满两个方向。
- 在 NVL72（MNNVL）上，symmetric 路径是让 AR/AG/RS **免去 staging 拷贝**的唯一 host API 路径。

---

## 7. CE collective（对照：不是 NVLS）

源码：`src/ce_coll.cc`。

- **支持的 collective**：只有 AllGather、AlltoAll、Scatter、Gather，驱动 ≥ 12.5（`ncclCeImplemented`，`:81`）。**没有 RS、AR、Reduce。**
- **可用条件**（`ncclCeAvailable`，`:100`）：`nNodes == 1`、`symmetricSupport`、recv buffer 已注册为 symmetric window。
- **什么时候被选中**（`src/enqueue.cc:2919-2968`）：
  1. 设置了 `NCCL_CTA_POLICY=ZERO`（或 comm config），且 CE 可用。
  2. **自动**：AG、CE 可用、每 rank 的**元素数** > `NCCL_SYM_CE_THRESHOLD`（默认 8M）、`minCompCap >= 100`、`isAllDirectNvlink`。这个判断发生在 symmetric kernel 选择之前，所以 **GB200 上已注册 window 的大块 AG 默认走 CE**。

数据面（`ncclCeAllGather`，`:472`）：

```text
ncclMemOpSync(ready)                       // 跨 rank 握手
batch = [ send → 本地 recv[rank]（仅 out-of-place） ]
      + [ send → peer r 的 recv[rank]，r = rank+1 … rank+P-1 ]  // ncclDevrGetLsaRankPtr 取 peer 的 UC 地址
ncclCeLaunchBatchOps(batch)                // Copy Engine 执行
ncclMemOpSync(complete)
```

- CUDA ≥ 12.8 时调用 `cudaMemcpyBatchAsync`（`:326`），属性为 `srcAccessOrderStream` 与 `cudaMemcpyFlagPreferOverlapWithCompute`；graph capture 或 legacy null stream 下回退为逐个 `cudaMemcpyAsync`。
- `numOps > 8` 且总量 ≥ 512MB 时开启 `intraBatchSync`，分轮拷贝、轮间同步，用来控制拥塞。
- **push / unicast 模型**：每个 rank 把自己的 shard 分别写给 P−1 个 peer。
- 同步（`ncclMemOpSync`，`:208`）：支持 NVLS 时，`ncclPrepMCSync`（`:120`）用 `cudaMemcpyAsync(H2D)` 把序号写到 MC 地址，再用 `cuStreamBatchMemOp` 的 `WAIT_VALUE_32` 在本地等其他 slot。这是 CE 中**唯一**用到 multicast 的地方，规划日志里的 “CE synchronization with NVLS”（`src/enqueue.cc:1554`）指的就是它。
- **SM 占用为 0**。

CE 能否改成 NVLS：广播可以（CE 写 MC 地址会被复制，见 PTX 文档第 5 节），但 NCCL 没有实现；RS/AR 不行，因为归约只能由 SM 指令触发。

---

## 8. GPU 发起：Device API 与 GIN

### 8.1 GIN 管什么

GIN（GPU-Initiated Networking）是 Device API 中**面向网卡**的部分，后端是 GDAKI（GPU 直接写 NIC doorbell）或 CPU Proxy。原语是 `put` / `putValue` / `signal` / `counter`（`src/include/nccl_device/gin.h`），对端用 `teamRankToGinRank` 映射到 rail 上的 NIC rank（`impl/gin__funcs.h:31`）。启用要求网络插件支持 GIN 与 GDR（`src/init.cc:1470-1473`）。

**GIN 不发 multimem 指令，也不经过 NVSwitch multicast。**

### 8.2 NCCL 内置的 “GIN + NVLS” kernel

| kernel | NVLS 部分 | GIN 部分 |
|---|---|---|
| `AllGather_RailRing_LsaSTMC`（`src/device/symmetric/all_gather_gin.cuh`） | LSA 域内用 `bcastMultimem`（`multimem.st`）广播自己的数据和从 rail 收到的数据 | 1 个 warp 沿 rail ring 执行 `gin.put` + `SignalInc`，其余 warp `waitSignal` 后再广播 |
| `ReduceScatter_RailA2A_LsaLDMC`（`reduce_scatter_gin.cuh`） | stage0 用 `reduceLsaBatch(...multimem)` 先做 LSA 域内归约，写到 outbox；stage1 的 `reduceLsa` 对本 rank 的 shard 执行 `ld_reduce` | stage0 的 GIN warp 把 outbox `postSends` 到 rail peer；stage1 `waitRecvs` 后在 smem 中累加，写到 output |

这两个 kernel 只有在 `ncclTeamLsa(comm).nRanks < comm->nRanks` 时才可能被选中（`src/sym_kernels.cc:553-554`）：

```cpp
bool needGin = ncclTeamLsa(comm).nRanks < comm->nRanks;
kmask &= needGin ? kernelMask_Gin : ~kernelMask_Gin;
```

**整个 comm 都在一个 NVL72 内时，GIN kernel 被屏蔽**，实际选中的是第 6 节的 kernel。

### 8.3 GPU 侧发起 NVLS：Device API（LSA + multimem）

```cpp
// host
ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
reqs.lsaMultimem     = true;          // 为 LSA team 创建 multicast object（symTeamObtain）
reqs.lsaBarrierCount = nCTAs;
ncclDevCommCreate(comm, &reqs, &devComm);
ncclCommWindowRegister(comm, buf, bytes, &win, NCCL_WIN_COLL_SYMMETRIC);

// device：在用户 kernel 中
ncclLsaBarrierSession<ncclCoopCta> bar{ncclCoopCta(), devComm, ncclTeamTagLsa(), blockIdx.x, /*multimem=*/true};
bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);
ncclMultimemCopy     (coop, localSrc, ncclSymPtr<T>{recvWin, off + rank*N}, N, devComm.lsaMultimem); // AG：multimem.st
ncclMultimemReduceSum(coop, ncclSymPtr<T>{sendWin, off + rank*N}, localDst, N, devComm.lsaMultimem); // RS：multimem.ld_reduce
bar.sync(ncclCoopCta(), cuda::memory_order_release);
```

接口位置：`src/include/nccl_device/core.h:82,131,173,243`，`reduce_copy.h:60-109`。

### 8.4 三种发起方式对照

| 维度 | Host（经典 NVLS） | Host（Symmetric） | GPU（Device API；跨域时加 GIN） |
|---|---|---|---|
| 数据面指令 | `multimem.st` / `ld_reduce` | 相同 | **相同**（共用 `multimem__funcs.h`） |
| 执行单元 | SM | SM | SM；GIN 的 put 由 NIC 执行 |
| multicast object | NCCL 内部 staging，或注册的用户 buffer | 用户 symmetric window（LSA team MC） | 同 symmetric window，可通过 `teamRequirementsList` 申请额外 team |
| 同步 | 逐 step 的 credit，`ld_reduce.min` 轮询 | 首尾各一次 LSA barrier | 用户自己放置 barrier / signal |
| 调度方 | NCCL | NCCL | **用户**：可以与计算融合、只算部分 shard、只发给部分 rank |
| 是否需要 GIN | 否 | 否 | NVL72 内不需要 |
| 32 rank 限制 | AG/RS 有 | 无 | 无 |

**在 NVL72 内，“GPU 发起的 NVLS”就是 Device API 的 LSA multimem 路径**，与 host 侧 symmetric kernel 的指令和数据流完全相同，区别只在由谁写 kernel、谁控制同步和切分。

---

## 9. 带宽分析

### 9.1 记号与模型

- P：通信组大小；N：每个 rank 的 shard 字节数（RS 的输入、AG 的输出都是 P·N）；S：AllReduce 的总字节数。
- 只统计 payload，不计请求包和 flag。
- 采用 PTX 文档第 3 节的**自环假设**：`ld_reduce` 也读发起者自己的副本，`st` 也写回发起者自己。需要用 DCGM 的 `DCGM_FI_PROF_NVLINK_TX_BYTES` / `RX_BYTES` 实测确认。

### 9.2 每卡链路字节数

| 实现 | RS 的 TX / RX | AG 的 TX / RX | RS 与 AG 叠加后（P=4） |
|---|---|---|---|
| Ring（SM） | (P−1)N / (P−1)N | (P−1)N / (P−1)N | **6N / 6N** |
| NVLS（`ld_reduce` / `st`） | P·N / N | N / P·N | **5N / 5N**（两方向互补） |
| CE AG（unicast）+ NVLS RS | — | (P−1)N / (P−1)N | **7N / 4N**（TX 先饱和） |
| 理论下界（交换机内归约，无自环） | (P−1)N / N | N / (P−1)N | **4N / 4N** |

NVLS 比下界多出的 N（TX、RX 各一份）就是 RS 和 AG 各自的自环。P=4 时占 25%，P=64 时约 1.6%。

### 9.3 实现对比

| 维度 | CE collective | Ring | NVLS |
|---|---|---|---|
| 执行单元 | Copy Engine（无 kernel） | SM | SM |
| 支持的 collective | AG / A2A / Scatter / Gather | 全部 | AG / RS / AR |
| 能否归约 | 不能 | 能（SM 上） | 能（交换机内） |
| 数据路径 | P−1 次 unicast push | 环上逐跳转发 | 一次 multicast store / reduce-load |
| 每卡 TX / RX（AG） | (P−1)N / (P−1)N | (P−1)N / (P−1)N | N / P·N |
| 步数 | 1 | P−1，时延随 P 线性增长 | O(1) |
| 同步 | stream memop wait-value，flag 经 MC 广播 | 逐 step FIFO | barrier（symmetric）或 credit（经典） |
| buffer 要求 | recv 必须注册 | 无 | 经典无要求；symmetric 需要注册 |
| 优点 | 不占 SM，适合与计算重叠 | 通用 | 步数少、带宽高 |

Ring 的实现：RS 为 `send` → (P−2) 次 `recvReduceSend` → `recvReduceCopy`（`src/device/reduce_scatter.h:14-56`）；AG 为 `directSend/CopySend` → (P−2) 次 `directRecvCopyDirectSend` → `directRecv`（`src/device/all_gather.h:14-80`）。

### 9.4 AllReduce

| 阶段 | TX | RX |
|---|---|---|
| RS：本卡副本被所有 head（或 symmetric 中的所有线程）读取 | S | — |
| RS：本卡发起的 `ld_reduce` 拿到的结果 | — | S/P |
| AG：本卡把自己那份结果 `st` 出去 | S/P | — |
| AG：收到所有广播 | — | S |
| **合计** | **(1 + 1/P)·S** | **(1 + 1/P)·S** |
| Ring AllReduce | 2(P−1)/P·S | 2(P−1)/P·S |
| 理论下界 | S | S |

- **大消息、P 较大时 NVLS 约比 Ring 快 2 倍**，与下界只差 1/P。NCCL 的代价模型因此把 AllReduce 的单 channel 带宽乘以 2（`src/graph/tuning.cc:292-294`，注释 “AllReduce pipelines two operations”）。
- NVLS AllReduce 天然是“RS ∥ AG、速率相同”的理想配比（9.5 节），由 kernel 内的流水自动实现。
- **P > 32 时经典路径 head 不对称**：每个 head 的 TX/RX 为 S + S/32，非 head 为 S；SM 工作也全部集中在 head 上。Symmetric 路径没有这个问题。
- 延迟模型：`latency = intraLat`，不随 P 增长（`src/graph/tuning.cc:392-394`）；Ring 为 2(P−1) 跳。

### 9.5 AG 与 RS 并发时的理想带宽分配

模型：

- 每卡 TX、RX 带宽各为 B。GB200 NVLink5 标称每方向约 900 GB/s；NCCL 代价模型取 720 GB/s（`src/sym_kernels.cc:167`）。
- N_a：每 rank 的 param shard 字节数（AG）；N_r：每 rank 的 grad shard 字节数（RS）；k = N_r / N_a。
- 流体模型：两条 collective 同时在跑时链路不空闲。完成时间 T = max(TX 总量, RX 总量) / B。

每传 1 字节 shard，每卡产生的链路字节数：

| 算法 | TX | RX |
|---|---|---|
| NVLS AG（`st`） | 1 | P |
| NVLS RS（`ld_reduce`） | P | 1 |
| Ring / CE / 直连 AG 或 RS | P−1 | P−1 |
| 理论下界：AG / RS | 1 / P−1 | P−1 / 1 |

**等量情形（k = 1）**：

| 方案 | 每卡 TX | 每卡 RX | T（单位 N/B） | P=8、N=100MB、B=720GB/s |
|---|---|---|---|---|
| NVLS AG、RS **串行** | (P+1)N | (P+1)N | **2P** = 16 | 2.22 ms |
| Ring AG ∥ Ring RS | 2(P−1)N | 2(P−1)N | 2(P−1) = 14 | 1.94 ms |
| CE AG ∥ NVLS RS | (2P−1)N | P·N | 2P−1 = 15 | 2.08 ms |
| **NVLS AG ∥ NVLS RS** | (P+1)N | (P+1)N | **P+1** = 9 | **1.25 ms** |
| 理论下界 | P·N | P·N | P = 8 | 1.11 ms |

- NVLS AG 主要占 RX，NVLS RS 主要占 TX。**并发时间约为串行的一半**。
- NVLS 串行时每个方向有一半时间空闲，**甚至比 Ring 并发还慢**。NVLS 的收益几乎完全来自 AG 与 RS 的重叠。
- 理想分配：两条 collective 以**相同的 shard 速率**推进、同时结束，r_a = r_r = B/(P+1)。TX 上 AG 占 1/(P+1)、RS 占 P/(P+1)；RX 反之。**不是“各分一半带宽”**。哪条先结束，剩下那条就只能用一个方向。

**不等量情形（k ≠ 1）**：

| 配置 | k | 纯 NVLS 并发的 T（单位 N_a/B） | 瓶颈 |
|---|---|---|---|
| bf16 param + bf16 grad | 1 | P+1 | 平衡 |
| bf16 param + fp32 grad（`grad_reduce_in_fp32=True`） | 2 | 1+2P（P=8 时 17） | TX；RX 只用到 (P+2)/(2P+1) |
| fp8 param gather + fp32 grad | 4 | 1+4P（P=8 时 33） | TX 严重偏高 |

1. **T ≈ max(N_a + P·N_r, P·N_a + N_r) / B，只有乘以 P 的那一项起决定作用**。k > 1 时瓶颈是 P·N_r，压缩 param（减小 N_a）几乎没用，应减小 grad 在链路上的字节数。
2. **让 grad 以 bf16 上链路（k=1）收益最大**：P=8 时 T 从 17 降到 9。NVLS 处理 bf16 时交换机内用 fp32 累加、只舍入一次；数值风险主要来自本地 grad 舍入为 bf16，以及 `grad_reduce_in_fp32=False` 时微批间的 bf16 累加。必须先验证收敛。折中做法是本地 `main_grad` 保持 fp32、通信前转 bf16，代价是多一次 HBM 读写。
3. **把一部分 RS 改走 Ring / 直连可以重新平衡**：设比例为 α，TX 系数变为 P−α，RX 系数变为 1+α(P−2)。两者相等时 α* = (k−1)/k，T = (kP−k+2)·N_a/B，只比 TX 下界多 N_a。P=8、k=2 时 T 从 17 降到 16（约 6%），收益不大。
4. k < 1 时对称：把 β = 1−k 的 AG 改走 CE / unicast。CE 不占 SM，这是 CE AG 合理的使用场景。

怎样逼近理想分配：

1. **按速率匹配分配发起资源**：两条 stream 上按 per-CTA 吞吐分配 CTA 数（`NCCL_SYM_CTAS`、comm config 的 `minCTAs/maxCTAs`）；或用 Device API 写**融合 kernel**，按 chunk 交替发出 AG 的 `st` 和 RS 的 `ld_reduce`，每轮字节数之比固定为 N_a : N_r。
2. **bucket 对齐**：参与重叠的 AG bucket 与 RS bucket 在元素数上大致成 1/k 的比例，否则重叠窗口一结束，链路就退回单向。
3. **只在 P ≥ 3 时用 NVLS**。P=2 时 NVLS 的 AG RX、RS TX 都是 2N，而点对点直传只需 N；NCCL 也只在 `lsa.nRanks > 2` 时启用 symmetric multimem kernel。
4. **不要用 unicast 去“补”**：每字节 TX+RX 之和，NVLS 是 P+1，unicast 是 2(P−1)。P ≥ 4 时混入 unicast 只会更差。

---

## 10. 自定义设计

### 10.1 “通信组 4 卡，GPU1–3 用 NVLS 归约，结果只给 GPU0”

**Host API 为什么做不到**：

- `ncclReduce(root=0)` 只有 RING 算法（`src/device/generate.py:91`），**不走 NVLS**。
- `ncclReduceScatter`：每个 rank 都会得到一个 shard。让 GPU0 贡献 0 时 shard 0 正确，但 GPU1–3 也白算了 shard 1–3。
- multicast 的语义是“全部成员参与”（PTX 文档第 4 节）。排除 GPU0 的贡献只有两种办法：**让它贡献单位元**，或者**换成不含 GPU0 的 multicast object**。

**方案 A：4 卡 team + GPU0 贡献 0，由 GPU0 拉取（最简单）**

```cpp
__global__ void reduce123_to_0(ncclDevComm dc, ncclWindow_t sendWin, size_t off, T* out0, size_t N) {
  ncclLsaBarrierSession<ncclCoopCta> bar{ncclCoopCta(), dc, ncclTeamTagLsa(), blockIdx.x, true};
  bar.sync(ncclCoopCta(), cuda::memory_order_acquire);   // GPU1–3 的数据已就绪；GPU0 的段已清零
  if (dc.rank == 0) {
    // 按 blockIdx 切分 [0, N)
    ncclMultimemReduceSum(ncclCoopCta(), sendWin, off + blk_off, out0 + blk_off, blk_n, dc.lsaMultimem);
  }
  bar.sync(ncclCoopCta(), cuda::memory_order_release);   // 之后 GPU1–3 才能覆写 send
}
```

- 只有 GPU0 执行 `ld_reduce`。链路：GPU1–3 各出 N；GPU0 出 N（读自己的 0，浪费）+ 入 N。
- 只适用于有单位元的运算（Sum 用 0，Max 用 -inf，Min 用 +inf）。所有 4 卡都要进入 kernel 参与 barrier。

**方案 B：{1,2,3} 3 卡 multicast + LSA 写给 GPU0（GPU0 不浪费带宽）**

1. 父 comm：4 卡，用于 barrier 和 GPU0 输出 window 的 LSA unicast 访问。
2. `ncclCommSplit(parent, rank==0 ? NCCL_SPLIT_NOCOLOR : 1, rank, &sub, ...)` 得到 3 卡子 comm。子 comm 的 `ncclDevCommCreate(reqs.lsaMultimem=true)` 满足 `lsa.nRanks=3 > 2`，会创建只含 GPU1–3 的 multicast object。
3. `sendbuf` 只在子 comm 中注册为 symmetric window；GPU0 的 `out` 只在父 comm 中注册。
4. kernel 由 4 卡共同启动：

```cpp
// GPU1–3：子 comm 中的 rank s∈{0,1,2}，各自负责 N/3
bar_parent.sync(...);
T* dst0 = (T*)ncclGetLsaPointer(outWin_parent, outOff + s*N/3, /*lsaPeer=*/0);  // GPU0 输出的 NVLink 直写地址
ncclMultimemReduceSum(coop, ncclSymPtr<T>{sendWin_sub, sendOff + s*N/3}, dst0, N/3, devComm_sub.lsaMultimem);
//   ↑ 对 {1,2,3} 执行 multimem.ld_reduce，结果用普通 st 经 NVLink 写到 GPU0
bar_parent.sync(..., release);
// GPU0：只参与 barrier
```

- 链路：GPU1–3 各出 N + N/3；GPU0 **入 N、出 0**，达到下界。
- 代价：多一个 sub comm、多一个 multicast object（消耗全局配额），以及两层 devComm 的管理。
- 也可以不 split，在父 comm 的 `teamRequirementsList` 中申请 `ncclTeam{nRanks=3, rank=r-1, stride=1}` 的 multimem team（`symTeamObtain` 支持 `rank + i*stride` 形式的等距子集，`src/dev_runtime.cc:264-340`）。但 GPU0 不在 team 中，各 rank 的 `teamRequirementsList` 不对称；**这种用法没有文档保证，建议优先 `ncclCommSplit`**。

**不推荐：`multimem.red`**。它会写到全部 4 个副本，写流量放大 4 倍，浮点 `red` 只有 `.add` 且顺序不确定。只想加到 GPU0 时，用 unicast 的 `red.global.add` 原子写 GPU0 地址更合理。

| 方案 | 能否实现 | 由谁执行 ld_reduce | GPU0 入 / 出 | 复杂度 |
|---|---|---|---|---|
| Host `ncclReduce` / `ncclReduceScatter` | 不能高效实现 | — | — | 低 |
| A：4 卡 MC + GPU0 贡献 0 | **能** | GPU0 | N / N | 低 |
| B：{1,2,3} MC + LSA 写给 GPU0 | **能** | GPU1–3 分片 | N / 0 | 中 |

### 10.2 “排除自身”的分步 NVLS ReduceScatter

**设计（P=4）**：

```text
step k（k = 0..3，处理要 scatter 给 GPU k 的 shard k）：
  GPU j≠k 组成 3 卡 multicast 组 MC_k，对 shard k 执行 NVLS 归约：partial_k = Σ_{j≠k} grad_j[k]
  partial_k 发送给 GPU k
  GPU k 在本地计算 out_k = partial_k + grad_k[k]       // GPU k 不向交换机发送自己的 shard k
```

目标：(1) 省掉标准 NVLS RS 中 GPU k 自读 shard k 的 TX；(2) step k 中 GPU k 的 TX 空闲，可以让给 AG。

**可行的数据流只有一种**（由 PTX 文档 4.1 节的两条硬约束决定）：MC_k 的 3 个成员各自对 N/3 执行 `ld_reduce`，再用 unicast `st` 写到 GPU k 的 relay buffer；GPU k 等待后在本地做最后一次 add。

**带宽核算**（每个 step）：

| | GPU k（目标） | GPU j≠k（各卡） |
|---|---|---|
| 标准 NVLS RS（4 卡 MC，k 拉取） | TX N（自读），RX N | TX N，RX 0 |
| 排除自身（3 卡 MC + relay） | TX 0，RX N（relay） | TX N + N/3（relay），RX N/3（`ld_reduce` 结果） |

4 个 step 累加后每卡：

| 方案 | RS 的 TX | RS 的 RX | 叠加 NVLS AG 后 TX / RX |
|---|---|---|---|
| 标准 NVLS RS | 4N | N | 5N / **5N** |
| 排除自身 + relay | 3·(N + N/3) = **4N** | 3·(N/3) + N = **2N** | 5N / **6N** |
| 一般 P | 标准 P·N / N；排除自身 P·N / 2N | | 标准 (P+1)N / (P+1)N；排除自身 (P+1)N / (P+2)N |

- **TX 是零和的**：GPU k 省下的 N，被其他 3 卡的 relay（3 × N/3）补回来。
- **RX 多了 N**：`ld_reduce` 结果要先回到成员，再转发一次给 k。叠加 AG 后 RX 是 6N，**和 Ring 持平**。
- 逐 step 看也不均衡：GPU k 在 step k 的 RX 是 2N，其他卡只有 4N/3。
- 额外开销：每 step 一次跨卡同步，GPU k 多一次 HBM 读写。

**结论：可以实现，但不能节省带宽。**标准 NVLS RS 已经是“3 卡归约后交给目标卡”的最优 NVLS 实现，代价是 N 字节的自环 TX，在当前指令集下无法消除。

**若仍要实现（实验用）**：

- **方案 1：Device API + 4 个 `ncclCommSplit` 子 comm**。{0,2,3}、{0,1,3} 不是等差序列，`ncclTeam` 只能表示 `{nRanks, rank, stride}`（`src/dev_runtime.cc:281`），所以只能 split。

  ```cpp
  // host：parent 为 4 卡 DP comm
  for (int k = 0; k < 4; k++)
    ncclCommSplit(parent, rank == k ? NCCL_SPLIT_NOCOLOR : 0, rank, &excl[k], &cfg);   // rank==k 时 excl[k] 为 NULL
  // grad（P·N）、relay（N）、out（N）用 ncclMemAlloc 分配
  // parent：注册 relay window；ncclDevCommCreate 时设置 lsaBarrierCount
  // excl[k]（k != rank）：注册 grad window；ncclDevCommCreate 时设置 reqs.lsaMultimem = true

  // device
  for (int k = 0; k < 4; k++) {
    if (rank != k) {
      int s = excl_rank[k];                                   // 本卡在 excl[k] 中的 rank，0..2
      T* dst = (T*)ncclGetLsaPointer(relayWin, 0, lsaRankOf(k)) + s*N/3;
      ncclMultimemReduceSum(ncclCoopCta(), ncclSymPtr<T>{gradWin[k], (k*N + s*N/3)*sizeof(T)},
                            dst, N/3, devExcl[k].lsaMultimem);
    }
    parentBar.sync(ncclCoopCta(), cuda::memory_order_acq_rel);
    if (rank == k) local_add(out, relay, grad + k*N, N);
  }
  ```

  注意：同一块 grad 要在 3 个子 comm 中重复注册为 window，需验证是否允许；每个子 comm 的 multicast object 绑定整个 symmetric VA（`src/dev_runtime.cc:309`）；每个 DP 组需要 P 个额外 multicast object，P=72 时基本不可行。

- **方案 2：直接用 CUDA Multicast API**（流程见 PTX 文档 1.2 节）：每个 k 建一个 3 成员的 multicast object，只用 `cuMulticastBindAddr` 绑定 shard k 那一段（与 NCCL 注册 NVLS buffer 的方式相同，`src/transport/nvls.cc:628`），kernel 中内联 `multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2`，再普通 `st` 到 GPU k。同步可复用父 comm 的 LSA barrier。
- **方案 3：只做功能对照**：用 10.1 节方案 A，或直接调用 `ncclReduceScatter`。

**想要“和 AG 重叠、节省带宽”，更有效的办法**：

1. **标准 NVLS RS + NVLS AG 并发**（9.5 节）：注册 symmetric window 让 RS 选 `ReduceScatter_LDMC`、AG 选 `AllGather_STMC`；调大 `NCCL_SYM_CE_THRESHOLD` 防止大块 AG 走 CE；两条 stream 并发。
2. **融合 kernel**：用 Device API 按 chunk 交替执行 AG 的 `st` 和 RS 的 `ld_reduce`，首尾各一次 barrier。
3. **不占 SM 的 multicast AG**（需实测）：`cudaMemcpyAsync(ncclGetLsaMultimemDevicePointer(recvWin, rank·N), send, N, D2D)`，前后用 `cuStreamBatchMemOp` 握手（参照 `ncclMemOpSync`）。流量与 `AllGather_STMC` 相同，但 0 个 CTA。
4. **把 NVLS 组做大**：自环开销是 1/P。

---

## 11. 对 Megatron 的意义

| 场景 | 建议 |
|---|---|
| TP / SP 的 AllReduce（小消息） | 走 `AllReduce_AGxLLMC_R` 或经典 NVLS，主要收益是延迟不随 P 增长 |
| DP 梯度 AllReduce（未用 distributed optimizer，大消息） | NVLS 约比 Ring 快 2 倍；NVL72 上建议注册 symmetric window 走 `AllReduce_RSxLDMC_AGxSTMC`，避免 staging 拷贝和 head 不对称 |
| FSDP forward：只有 param AG | **CE AG**：RX 更少（(P−1) 对比 P），且不占 SM。GB200 上大块已注册 AG 默认就走 CE |
| FSDP backward：param 重新 AG ∥ grad RS | **NVLS AG + NVLS RS**，并调大 `NCCL_SYM_CE_THRESHOLD`，防止 backward 的 AG 被分到 CE |
| 优化器之后的 param AG（ZeRO-1 / distributed optimizer） | CE AG |

推论：

- **AllReduce 自动获得 RS ∥ AG 的带宽互补；FSDP 必须自己安排重叠，才能达到同样的链路利用率。**
- 在 NVLS 下，backward 中重新做一次 param AG 几乎不占 TX：它用的 RX 本来就被 RS 闲置，主要代价是 SM 和时延。
- 如果 forward 之后不 reshard，backward 只剩 RS（只占 TX），forward 只剩 AG（只占 RX），这是 NVLS 最差的组合。
- ZeRO-1 / ZeRO-2 中 RS 和 AG 处在不同阶段，NVLS 的互补用不上，单独执行时 Ring/CE 反而更省：(P−1) 对比 P。

---

## 12. 配置与验证

### 12.1 常用开关

| 用途 | 方法 |
|---|---|
| 强制经典路径 | `NCCL_ALGO=NVLS NCCL_PROTO=Simple`（AG/RS 只在 localRanks ≤ 32 时有效） |
| 关闭 NVLS | `NCCL_NVLS_ENABLE=0`（不分配资源）或 `NCCL_ALGO=^NVLS`（只影响算法选择） |
| 经典 NVLS 的 CTA 数 | `NCCL_NVLS_NCHANNELS`（NVL72 默认 24），`ncclConfig_t.nvlsCTAs` |
| 经典 NVLS 的 chunk | `NCCL_NVLS_CHUNKSIZE`（默认 128 KB） |
| 强制 symmetric kernel | 注册 window，并设置 `NCCL_SYM_KERNEL=AllGather_STMC` / `ReduceScatter_LDMC` / `AllReduce_RSxLDMC_AGxSTMC` / `AllReduce_AGxLLMC_R` |
| symmetric 的 CTA 数 | `NCCL_SYM_CTAS`；`ncclConfig_t.minCTAs/maxCTAs` |
| 让大块 AG 不走 CE | 调大 `NCCL_SYM_CE_THRESHOLD`（单位：每 rank 元素数） |
| 强制 CE | `NCCL_CTA_POLICY=ZERO` |

### 12.2 验证清单

1. **确认 NVLS / MNNVL 状态**：`NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NVLS,TUNING`。检查 `NVLS multicast support is available ... (NVLS_NCHANNELS 24)`、`MNNVL 1 cliqueSize ...`、`nNodes 1 localRanks N`，以及 `NVLS comm ... headRank %d nHeads %d`（`src/transport/nvls.cc:337`）。
2. **确认选中的实现**：TUNING 日志中的 `NVLS/Simple` 或 symmetric kernel 名称；Nsight 中 kernel 名为 `ncclDevKernel_AllReduce_Sum_bf16_NVLS_SIMPLE` 或 `ncclSymk_*`；CE 路径在时间线上是 memcpy / batch memop，没有 NCCL kernel。
3. **SM 占用**：经典 NVLS 约 24 个 CTA；symmetric 为 ceil(32/P) 个 CTA；CE 为 0。
4. **链路流量**：用 DCGM 的 `DCGM_FI_PROF_NVLINK_TX_BYTES` / `RX_BYTES` 分别测 AG、RS 单独与并发三种情况，核对每卡是否为 RS 约 P·N / N、AG 约 N / P·N，从而验证自环假设与 9.5 节的模型。

---

## 13. 关键源码索引

| 主题 | 位置 |
|---|---|
| `struct ncclNvls`（head 定义） | `src/include/device.h:210-220` |
| head 选择（图搜索） | `src/graph/search.cc:387-417, 794, 1063-1073` |
| head 去重与 `headRank` | `src/graph/connect.cc:75-90, 242-262` |
| NVLS CTA 数默认值 | `src/transport/nvls.cc:149-200` |
| staging / credit 布局 | `src/transport/nvls.cc:314-360, 384-471` |
| chunk 大小 | `src/enqueue.cc:2153-2167` |
| 算法过滤（arity 32） | `src/enqueue.cc:1914` |
| 经典 NVLS AG | `src/device/all_gather.h:175-318` |
| 经典 NVLS RS | `src/device/reduce_scatter.h:144-290` |
| 经典 NVLS AR | `src/device/all_reduce.h:388-520` |
| Simple 原语（scatter/gather、credit、multimem 路径） | `src/device/prims_simple.h:96-107, 174-300, 420-480, 495-590` |
| multimem ld_reduce / st | `src/device/reduce_kernel.h:824-1018`，`src/device/op128.h:374-410` |
| reduceCopy 中的 multimem 分支 | `src/device/common_kernel.h:83-162` |
| CE collective | `src/ce_coll.cc:81-160, 208, 326, 472-520`，`src/enqueue.cc:2919-2968` |
| Symmetric kernel 选择 | `src/sym_kernels.cc:16-32, 314-565`，`src/scheduler/symmetric_sched.cc:17` |
| Symmetric AG/RS/AR | `src/device/symmetric/all_gather.cuh:195`，`reduce_scatter.cuh:261-340`，`all_reduce.cuh:299-475`，`primitives.cuh:234` |
| GIN + LSA kernel | `src/device/symmetric/all_gather_gin.cuh`，`reduce_scatter_gin.cuh` |
| Device API | `src/include/nccl_device/{core.h,reduce_copy.h,gin.h,lsa_barrier.h}`，`impl/multimem__funcs.h` |
| multicast team 创建 | `src/dev_runtime.cc:264-345, 900-930` |
| LSA team 大小 | `src/dev_runtime.cc:77-100` |
| 代价模型 | `src/graph/tuning.cc:281-300, 392-394`，`src/sym_kernels.cc:167, 319-391` |
