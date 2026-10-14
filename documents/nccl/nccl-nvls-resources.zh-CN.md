# NVLS 资源：multicast group 配额、交换机数据面、GPU 端资源与 NCCL 配置

日期：2026-10-12。源码基于 NCCL **v2.29.7-1**（本地 `Megatron-Scale-Down/nccl-2.29.7`），并对照 NCCL **v2.32.3** 的变更。

相关文档：

- [PTX 9.4 multimem 指令](ptx-9.4-multimem.zh-CN.md)
- [NCCL 2.29.7 NVLS AG / RS / AR 实现](nccl-2.29.7-nvls-collectives.zh-CN.md)：文中的 head、staging、symmetric window、LSA team 等术语见该文第 1 节。

每条结论都标注了证据等级：

| 等级 | 含义 |
|---|---|
| 【官方】 | NVIDIA 文档或演讲 |
| 【源码】 | NCCL 源码 |
| 【社区】 | issue、论文 |
| 【推断】 | 由上述证据推理得出，**需要实测** |

NVIDIA 没有公开 NVSwitch 内部 SHARP 缓冲区的组织方式和容量，因此和“缓存”相关的结论只能推断。

---

## 0. 资源分层

NVLS 涉及三个层面的资源，性质不同：

| 层面 | 资源 | 分配时机 | 耗尽时的表现 |
|---|---|---|---|
| **控制面：组（slot）** | multicast group，即交换机中的组表项与地址映射 | **静态**：在 `cuMulticastCreate`、`AddDevice`、`Bind` 时由 Fabric Manager 分配，comm 销毁时才释放 | 初始化或注册**失败**：`cuMulticastBindMem` 返回 CUDA error 2（out of memory），新驱动还可能返回 `CUDA_ERROR_MULTICAST_RESOURCE_FULL` |
| **数据面：交换机** | crossbar 带宽（multicast 复制）、SHARP ALU 吞吐、SHARP scratch SRAM（归约中间状态）、事务跟踪表项 | **动态**：按请求即时占用 | **背压**：变慢，不丢包（NVLink 是基于 credit 的无损链路）【推断】 |
| **GPU 端** | SM（发起 multimem 指令的 CTA）、GPU 端口的 TX/RX 带宽、HBM，以及 staging buffer 显存 | kernel 运行时占用；staging 在 init 或首次使用时分配 | 变慢；或显存不足 |

### 0.1 证据

- **【官方】Hot Chips 34（NVLink4 NVSwitch）**：
  - 交换机内部有 SHARP Controller、SHARP ALU、SHARP Scratch SRAM 三类模块，与 XBAR 和 Port Logic 并列；Port Logic 中也包含 SHARP 加速器。
  - “**SHARP Controller can manage up to 128 SHARP groups in parallel**”。
  - “XBAR BW uprated to carry additional SHARP-related exchanges”。
  - 每颗芯片有 400 GFLOPS 的 FP32 SHARP 算力。
  - 示意图：读归约时，交换机读取 N 份数据并在交换机内求和，结果作为 1 次 reduced read 返回发起者；多播写时，发起者只写 1 次，交换机复制 N 份。
- **【官方】GB200 NVL Partition Guide 第 9 章**：
  - “There is a hardware limit of **1024 total multicast groups** that can be on the fabric at a time”，这是**全局资源，所有 partition 共享**。
  - 管理员可以按 partition 预留 `multicastGroupsLimit`，取值必须为 0 或 4 的倍数。
  - Default Partition 默认占用全部 1024 个。
  - 建议按 `floor(3.55·N_GPU)·4` 分配，例如 64 卡分配 908 个；≤4 卡的 partition 不需要 multicast。
  - 当前用量可以通过 `GetPartitionInfoList.numAllocatedMulticastGroups` 查询。
- **【官方】Blackwell NVLink Switch（Hot Chips 2024）**：每颗芯片 72 个端口，7.2 TB/s，SHARP 算力 3.6 TFLOPS。NVL72 共有 18 颗芯片，每张 GPU 用 18 条 link 分别连到这 18 颗芯片。
- **【社区】NVIDIA/nccl#2077**（H200 单机，NCCL 2.29.7）：
  - Fabric Manager 日志显示 multicast group 0–127 分配完之后报错 `all the NVSwitch multicast resources/slot ids are used`。
  - 每个子 comm 大约创建 10 个 group，直到 comm 销毁才释放。
  - `NCCL_NVLS_NCHANNELS=1`、`NCCL_NVLS_TREE_ENABLE=0` 都无法减少 slot 消耗，只有 `NCCL_NVLS_ENABLE=0` 有效。
  - NVIDIA 维护者的答复：为避免部分 rank 静默回退、部分 rank 继续用 NVLS 导致 hang，2.29 起资源分配失败会直接报错，不再静默回退。之后在 2.32 中加入了按 comm 关闭的开关，并减少了 slot 消耗。
- **【社区】论文**：
  - SCIN（arXiv 2603.28239）中的 NVLS 类模型：每个交换机端口有 320 KB 归约表（1 KB 一个表项），crossbar 有 2× 内部加速比。它引用原始 NVLS 研究（Klenk et al., ISCA 2020）的结论：NVLS 的性能对归约缓冲区容量敏感，需要约 2 倍于理论最小值的容量才能跑满带宽；multicast 仍可能在 crossbar 上造成拥塞。**这些是论文作者的建模参数，不是 NVIDIA 公布的规格。**
  - NVIDIA 专利 US 11502867（“Injection limiting and wave synchronization for scalable in-network computation”）提到：multicast 复制会加剧拥塞；归约请求与 multicast 写会相互干扰；可以用 credit 限制网络中未完成的操作数。

---

## 1. GPU 端资源：staging 显存、credit 与 SM

### 1.1 经典 NVLS 的 staging 显存【源码】

每个 rank 为经典 NVLS 分配的 buffer（`src/transport/nvls.cc:327-335`）：

```text
buffSize        = NCCL_NVLS_CHUNKSIZE × NCCL_STEPS          = 128 KB × 8 = 1 MiB
nvlsPerRankSize = nvlsChannels × 2 × buffSize               // 每个 head：reduce staging + bcast staging
nvlsTotalSize   = nvlsPerRankSize × nHeads
```

| 配置 | nvlsChannels | nHeads | 每卡 staging |
|---|---|---|---|
| H100 / H200 8 卡 | 16 | 8 | 8 × 16 × 2 × 1 MiB = **256 MiB** |
| NVL72 内 32 卡 comm | 24 | 32 | 32 × 24 × 2 × 1 MiB = **1.5 GiB** |
| NVL72 内 64 / 72 卡 comm（仅 AR 可用） | 24 | 32（上限） | **1.5 GiB** |
| 假如允许 72 个 head | 24 | 72 | 约 3.4 GiB |

- 这是 UC 物理内存，每个 comm 各自一份，**comm 越多占用越多**。
- 在**第一次使用 NVLS 算法时**才分配（`ncclNvlsBufferSetup`，`src/group.cc:163`、`src/init.cc:1385`）。
- 可以用 `NCCL_NVLS_NCHANNELS` / `ncclConfig_t.nvlsCTAs`（减少 channel）或 `NCCL_NVLS_CHUNKSIZE`（减小 chunk）调小。
- staging 显存随 head 数线性增长，是 head 上限定为 32 的原因之一【推断】（见实现文档 1.3 节）。

### 1.2 credit

- 每个 (head, channel) 两对 credit，每对 64 字节：`creditSize = nChannels × 2 × 64 × nHeads`（`src/transport/nvls.cc:403`），也是一个 multicast object。
- 在 comm 初始化时分配（`ncclNvlsSetup`，`src/init.cc:1368`）。

### 1.3 SM

| 路径 | CTA 数（每 rank） | 每 CTA 线程 |
|---|---|---|
| 经典 NVLS | `nvlsChannels`：SM90 16，SM100 单机 / NVL72 24，SM100 多节点 32（`src/transport/nvls.cc:149-200`） | 640 |
| Symmetric（Blackwell） | `ceil(32 / nRanks)`（`src/sym_kernels.cc:319`） | 512 |
| CE collective | 0 | — |

---

## 2. NVLS AG 和 NVLS RS 是否占用相同的资源？能否同时处理广播和归约？

### 2.1 资源对照

| 资源 | NVLS AG（`multimem.st`） | NVLS RS（`multimem.ld_reduce`） | 是否共享 |
|---|---|---|---|
| multicast group（slot） | 占用 param / recv buffer 所在的 group | 占用 grad / send buffer 所在的 group | **可以是同一个，也可以不同**。NCCL symmetric 路径下，一个 comm 的所有 window 共用同一个 LSA team（`src/dev_runtime.cc:264-345`） |
| GPU 端口 TX | 1×N_a | P×N_r | 共享端口，主要占用方向相反 |
| GPU 端口 RX | P×N_a | 1×N_r | 同上 |
| crossbar | 写请求被复制 P 份，复制带宽需求大 | 读请求扇出 P 份；读响应汇聚到 ALU；结果只回 1 份 | 共享 |
| SHARP ALU | 不使用 | 每个输入元素约做一次加法 | 只有 RS 使用 |
| Scratch SRAM / 事务跟踪 | 写入需要完成跟踪（release 语义），状态较少【推断】 | **每个未完成的 ld_reduce 都要占用部分和与等待响应的表项** | RS 是主要消耗方 |
| SM | 读本地数据后发起 st | 发起 ld_reduce，再写回本地 | 共享 SM |

### 2.2 能否同时处理：可以

- **【源码】同一个 group、同一条指令流中交替执行归约和广播**：经典 NVLS AllReduce 的 reduce warp 使用 `ProtoSimple<1,1,COLL_UNROLL,/*MultimemSrcs=*/1,/*MultimemDsts=*/1>`，调用 `directRecvDirectSend`（`src/device/all_reduce.h:440-454`）：对 `nvls->down` 先 `multimem.ld_reduce`，再对同一 group `multimem.st`。多个 chunk 流水执行，同一时刻交换机里同时存在归约请求和广播写。
- symmetric 路径的 `AllReduce_RSxLDMC_AGxSTMC` 也是先 ld_reduce 再 st。
- **【官方】** 交换机能同时管理 128 个（NVLink4）SHARP group，AG 和 RS 使用不同 group 并发也在支持范围内。
- **代价是性能干扰，不是功能限制**【推断】：
  - 两者共享 crossbar 和事务跟踪资源。专利明确提到归约请求和 multicast 写会相互干扰。
  - AG 主要占 RX、RS 主要占 TX，GPU 端口不是冲突点。冲突主要发生在**交换机内部**（crossbar 复制 + ALU + SRAM）。
- **算力估算**【推断】：
  - NVL72 上每颗芯片要处理约 72 × 720 GB/s ÷ 18 ≈ 2.9 TB/s 的入口流量。按 bf16 计约 1.44 T 元素/s，每个元素约 1 次加法，约 1.4 TFLOP/s。
  - 这低于 3.6 TFLOPS，因此 **ALU 在线速下不太可能成为瓶颈**。前提是 3.6 TFLOPS 指的就是所用的数据格式，需要实测确认。
  - ALU 需求只取决于**入口总字节数**，与 GPU 如何分组无关。

**验证方法**：在同一组 GPU 上分别测量 AG 单独、RS 单独、AG 与 RS 并发三种情况下的 busBW，以及 DCGM 的 NVLink TX/RX。如果并发时间明显高于 `max(TX, RX)/B` 的模型，说明交换机内部的 crossbar 或 SRAM 是瓶颈。

---

## 3. 多个组同时用 NVLS 做 RS，会不会出现资源分配问题？

以 64 卡拆成两组、每组 32 卡为例，要分两层讨论。

### 3.1 控制面（slot）：会出问题，但发生在初始化阶段，不是运行时抢占

- group 在 comm 初始化或 buffer 注册时**静态分配**，运行时不会被其他组“挤掉”。
- 但 slot 池是**全局共享**的：GB200 fabric 共 1024 个，H100/H200 每颗 NVSwitch 128 个。在同一个 partition 内先到先得，**先建 comm 的一方可能耗尽 slot，后建的一方初始化失败**。在 NCCL 2.29 中这是硬错误。

**【源码】NCCL 2.29.7 中每个 comm 消耗多少 multicast object：**

| 来源 | 数量 | 何时创建 | 源码位置 |
|---|---|---|---|
| 经典 NVLS 的 credit | 1 | comm 初始化（`ncclNvlsSetup`） | `src/transport/nvls.cc:431`、`src/init.cc:1368` |
| 经典 NVLS 的 staging buffer | 1 | 首次使用 NVLS 算法时 | `src/transport/nvls.cc:340`、`src/group.cc:163` |
| 用户 buffer 注册（NVLS reg） | 每个注册的 buffer 1 个 | 注册时。**MNNVL 下禁用**（`src/init.cc:1024`） | `src/transport/nvls.cc:536-640` |
| symmetric / CE / Device API 的 LSA team | 每个不同的 team 1 个，与该 comm 的所有 window 共用 | 首次注册 symmetric window 或调用 `ncclDevCommCreate(lsaMultimem)` 时 | `src/dev_runtime.cc:264-345, 657` |
| Device API 的 `teamRequirementsList` | 每个 team 1 个 | `ncclDevCommCreate` | `src/dev_runtime.cc:921-930` |

- 子 comm 可以复用父 comm 的经典 NVLS 资源，条件是 `parent->shareResources && parent->localRanks == comm->localRanks`（`src/transport/nvls.cc:384`）。
- **在 MNNVL 上把 64 卡拆成 2×32，`localRanks` 从 64 变为 32，条件不满足，无法复用。**
- 自定义设计（例如实现文档 10.2 节“排除自身”的 RS）每个 DP 组需要 P 个额外的 multicast object，P 大时基本不可行。

**Megatron 场景的估算**【推断】：

- Megatron 会创建很多 process group：TP、DP、CP、EP、PP、DP×CP、embedding 等。只要一个 comm 的 GPU 数 ≥2 且支持 NVLS，它就会占用 2 个以上的 group。消耗量等于**所有 comm 的总和**：一个 group 由组内成员共享，不按 rank 倍增。
- 例如 64 卡、TP=4 时有 16 个 TP comm，约 32 个以上 group；再加上各种 DP/CP/EP comm，整体规模在数十到一两百个。相比 GB200 的 1024 个通常够用，但要注意三种情况：
  1. 与其他作业共享 Default Partition。
  2. comm 泄漏，例如反复创建、销毁 process group（参见 Megatron-LM#6897）。
  3. 大量注册 symmetric window 或 Device API team。

### 3.2 数据面（交换机缓存、ALU、crossbar）：运行时动态共享，表现为降速

- **NVLink 是基于 credit 的无损链路**。归约表项不够时，交换机不会接收新请求，压力回传给发起端的 SM。结果是**变慢，而不是丢包或出错**【推断，依据是链路层 credit 机制和专利中的 credit 限流方案】。
- **两组 GPU 不相交时，会不会抢同一块 SRAM？**
  - 每张 GPU 在每颗交换芯片上占用不同的端口。如果归约表像 ISCA'20 / SCIN 模型那样**按入口端口划分**（每张 GPU 发起的 ld_reduce 只占自己端口的表项），两组之间**不会争用归约缓冲区**，只共享 crossbar 和 ALU。
  - 如果 SRAM 是**芯片级共享池**，流量大的一组会挤占另一组的表项，使后者吞吐下降。但流量控制机制使它仍会继续推进，不会饿死。
  - **NVIDIA 没有公开具体是哪一种，需要实测。**
- **ALU**：算力需求取决于全芯片的入口总流量。64 卡分成 2×32 和合成 1×64，入口流量相同，ALU 压力也相同；并且如 2.2 节所估，线速下 ALU 仍有余量。
- **crossbar**：两组之间没有共享端口。一组 AG 的 multicast 复制只会流向本组成员的出口端口，crossbar 内部有加速比，两组之间基本隔离【推断】。

**验证方法**：64 卡分为 A、B 两组，测 4 种情况：

1. A 单独 RS。
2. B 单独 RS。
3. A、B 同时 RS。
4. A 用很多 CTA（例如 `NCCL_SYM_CTAS=8`）、B 用很少 CTA 同时 RS。

如果第 3 种情况下两组的 busBW 与单独运行相同，说明数据面在组之间隔离良好；如果第 4 种情况下 B 明显变慢，说明存在芯片级共享资源的争用。

### 3.3 另外两个约束

- **【源码】** 经典 NVLS 的 AG/RS 要求 `localRanks ≤ 32`（`src/enqueue.cc:1914`）。因此 64 卡单个 comm 的 AG/RS 只能使用 symmetric kernel；拆成 2×32 后可以使用经典 NVLS。
- **显存**：拆成多个 comm 后，每个 comm 各有一份 1.5 GiB 的 staging（1.1 节）。

---

## 4. NCCL 中能否配置占用的 multicast 资源？

只能控制**是否使用**以及**创建多少个 group**，**不能**为 comm 预留交换机内的 SRAM、ALU 或带宽份额。数据面只能通过“发起强度”（CTA 数）间接控制。

| 想控制的对象 | 方法 | 版本 | 说明 |
|---|---|---|---|
| 全局关闭 NVLS | `NCCL_NVLS_ENABLE=0` | 2.17+ | 不分配任何 NVLink SHARP 资源。默认值 2：不支持时不报错，但**资源分配失败会报错**（2.27/2.28 会静默回退，2.29 起恢复为报错） |
| **按 comm 关闭 host 侧 NVLS** | `ncclConfig_t.nvlsHostMode`，取值：`ncclNvlsHostModeDisableTransport`（经典 NVLS/NVLS_TREE、buffer 注册、父子资源共享）、`ncclNvlsHostModeDisableSymmetricMultimem`（symmetric kernel 与 CE 的 multimem）、`ncclNvlsHostModeDisable`（全部关闭） | **2.32.3+**（commit `b194d66`） | Device API 不受影响，仍由 `ncclDevCommCreate` 的 `lsaMultimem` 控制。同一版本还 “optimizes NVLS slot consumption”。**2.29.7 没有这个选项。** |
| 让子 comm 复用父 comm 的 NVLS 资源 | `ncclConfig_t.splitShare=1` 或 `NCCL_COMM_SPLIT_SHARE_RESOURCES=1`，再调用 `ncclCommSplit` | 2.18+ | 仅当父子 `localRanks` 相同时生效（`src/transport/nvls.cc:384`） |
| 不创建 symmetric 的 LSA team | `NCCL_WIN_ENABLE=0`；或不注册 `NCCL_WIN_COLL_SYMMETRIC` window | — | 同时会失去 symmetric kernel 和 CE collective |
| 不为用户 buffer 创建 MC object | `NCCL_LOCAL_REGISTER=0`；或不调用 `ncclCommRegister` | — | MNNVL 下本来就不做 NVLS 注册 |
| Device API 的 team | `ncclDevCommRequirements.lsaMultimem`、`teamRequirementsList[].multimem` | 2.28+ | 按需申请 |
| 经典 NVLS 的 channel 数（SM 与 staging 显存） | `NCCL_NVLS_NCHANNELS` / `ncclConfig_t.nvlsCTAs`；`NCCL_NVLS_CHUNKSIZE`（默认 128 KB） | — | **不影响 slot 数**（#2077 已验证），只影响显存和发起强度 |
| symmetric kernel 的 CTA 数 | `NCCL_SYM_CTAS`；`ncclConfig_t.minCTAs/maxCTAs` | — | 间接限制发起强度，相当于粗粒度的 QoS |
| 某个 comm 不用 NVLS 算法（但仍分配资源） | `NCCL_ALGO=^NVLS` | — | 只影响算法选择，不释放已分配的 slot |
| 跨 partition / 作业隔离 slot | 管理员通过 NMX-C / NVOS 为 partition 设置 `multicastGroupsLimit` | GB200 | 需要先删除 Default Partition；用量通过 `numAllocatedMulticastGroups` 监控 |

### 4.1 对 Megatron 场景的建议

1. **只给需要 NVLS 的 comm 开启**，例如做 FSDP AG/RS 的 DP comm。TP/EP/CP 等小组（≤4 卡）关闭 NVLS：
   - 升级到 NCCL 2.32.3+ 后，在创建这些 process group 时传入 `nvlsHostMode=ncclNvlsHostModeDisable`。PyTorch 的 `ProcessGroupNCCL.Options().config` 需要相应版本支持该字段。
   - 停留在 2.29.7 时只能全局关闭，或者靠控制 comm 数量与 window 注册来减少消耗。
2. **避免 comm 泄漏**：不再使用的 process group 要显式 `destroy_process_group`；测试和多次初始化场景下尤其要注意。
3. **监控**：
   - 开启 `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=NVLS`，日志中的 `NVLS Creating Multicast group nranks %d size %zu` 每出现一次就对应一个 group（`src/transport/nvls.cc:59`）；`NVLS comm ... headRank %d nHeads %d ... nvlsTotalSize %zu`（`src/transport/nvls.cc:337`）给出 staging 显存大小。
   - 在交换机侧查看 `numAllocatedMulticastGroups`，以及 Fabric Manager / NMX-C 日志。
4. **数据面公平性**：如果实测发现两组 RS 之间存在干扰，可以用各组的 CTA 数（`NCCL_SYM_CTAS` 或 comm config）调节发起强度。

---

## 参考资料

- NVIDIA, *The NVLink-Network Switch*, Hot Chips 34 (2022)：https://hc34.hotchips.org/assets/program/conference/day2/Network%20and%20Switches/NVSwitch%20HotChips%202022%20r5.pdf
- NVIDIA, *GB200 NVL Partition User Guide*（第 9 章 Multi-Cast Support）：https://docs.nvidia.com/multi-node-nvlink-systems/partition-guide-v1-2.pdf
- NVIDIA Mission Control, NVLink Partition Management：https://docs.nvidia.com/mission-control/docs/systems-administration-guide/2.1.0/nvlink-partition-management.html
- Blackwell NVLink Switch 规格（Hot Chips 2024 报道）：https://convergedigest.com/think-of-nvidia-blackwell-as-a-platform/
- NCCL 环境变量文档（`NCCL_NVLS_ENABLE`）：https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
- NVIDIA/nccl#2077，NVLS multicast slot exhaustion：https://github.com/NVIDIA/nccl/issues/2077
- pytorch/pytorch#178749：https://github.com/pytorch/pytorch/issues/178749
- NCCL commit `b194d66`（nvlsHostMode）：https://github.com/NVIDIA/nccl/commit/b194d662b5d05e78e15008b295a86cb9fada6c88
- NCCL Releases（v2.32.3、v2.31.2）：https://github.com/NVIDIA/nccl/releases
- Megatron-LM#6897（process group 泄漏）：https://github.com/NVIDIA/Megatron-LM/issues/6897
- CUDA Driver API, Multicast Object Management：https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MULTICAST.html
- SCIN, arXiv 2603.28239：https://arxiv.org/pdf/2603.28239
- Klenk et al., *An In-Network Architecture for Accelerating Shared-Memory Multiprocessor Collectives*, ISCA 2020：https://dblp1.uni-trier.de/rec/conf/isca/KlenkJTD20.html
- NVIDIA 专利 US 11502867：https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11502867
