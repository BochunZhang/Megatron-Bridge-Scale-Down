# NVLS 上 All-Gather / Reduce-Scatter 重叠与 FSDP3 集成分析

**日期：2026-10-08**
**结论等级：架构可行；GB200 的域级资源上限已公开为 1,024 multicast teams。“完美重叠”、每 ASIC 资源映射和 scratch SRAM 容量仍需实测或进一步规格。**

**官方手册入口：** [NVIDIA GB200 NVL Partition User Guide v1.2，第 9 章（PDF 第 56–57 页）](https://docs.nvidia.com/multi-node-nvlink-systems/partition-guide-v1-2.pdf#page=56) 明确给出 1,024 个全局 multicast teams、分区预留和已用资源查询方法。文末“官方手册与资料边界”汇总公开手册以及 NVOnline 中已确认存在的系统手册；不应将“未查到内部规格”表述成“NVIDIA 没有官方手册”。

## 摘要

NVLS 的 multicast memory 和 NVLink SHARP 足以表达这两类原语：`multimem.st` 把一个源写入 multicast object 的所有副本，`multimem.ld_reduce` 从所有副本读取并在 NVSwitch 中归约，结果返回发起指令的 GPU。[CUDA multicast memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html)、[PTX multimem](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)、[NCCL Device API](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html) 都描述了这个语义。

因此，四卡上让 GPU0 发起一段 broadcast，同时让 GPU0 发起另一段 reduce-load，**在端点链路方向上有并发空间**。但这不能推出 NVSwitch 内部会把两种操作以各自峰值同时执行。两者仍可能共享 multicast group、crossbar、credit/VC、SHARP 控制器和归约数据通路；NCCL 还可能因为同一 communicator 的集体操作顺序而串行化。公开资料没有给出“broadcast 与 NVLS reduce 完全独立”的保证。

SC'24 的带宽最优 Allgather 结论可以迁移为一个实验假设，而不能原样移植为 NVLS 实现。该论文假设全双工、non-blocking 网络：multicast Allgather 受接收路径限制，in-network Reduce-Scatter 受发送路径限制，理论 speedup 为 `2 - 2/P`，四卡为 1.5 倍，规模增大才趋近 2 倍。[SC'24 论文](https://arxiv.org/abs/2408.13356)。NVLS 是单 NVLink 域内的硬件 multicast/SHARP，省去了论文中的 SmartNIC、UDP multicast 可靠性和 fat-tree 调度；跨 NVLink 域仍需 NVLS 加 IB/SHARP 的分层算法。

FSDP3 的关键工程动作是为 AG 和 RS 使用相同 rank 集合的独立 communicator/process group，并使用对称内存、持久 buffer、事件和正确的 collective 顺序。当前 Bridge 的 decentralized `ProcessGroupCollection` 创建了 `dp_cp`，但没有把 `dp_cp_ag` 填进去；MCore FSDP adapter 已经会读取 `pg_collection.dp_cp_ag` 并传入 `fsdp_group_ag`。因此，直接在现有路径上打开两个 stream 不足以证明 wire-level overlap，需要先补齐独立 AG PG，再逐步接入 NCCL symmetric kernels 或自定义 multimem kernel。

## 先澄清四卡数据流

用户描述的“GPU0 广播，GPU1/2/3 把数据发给 GPU0 归约”可以作为一个定向 microbenchmark，但它不是完整 FSDP Allgather/Reduce-Scatter 的一般形态：

- 完整 Allgather 中，每个 rank 都要把自己的 shard 写入所有 rank，因此每个 GPU 都会在不同 chunk 上发起 `multimem.st`。
- 完整 Reduce-Scatter 中，每个 rank 对自己负责的输出 offset 发起 `multimem.ld_reduce`；每个 GPU 既可能是 operand 的提供者，也可能是另一个 offset 的结果接收者。
- `multimem.ld_reduce` 是 pull 语义：GPU0 必须发起 load，NVSwitch 才会读取 multicast object 的副本并返回结果。GPU1、2、3 没有一个独立的“push 到 NVLS、GPU0 稍后收取”的公开 PTX 指令。

所以“GPU0 的上行与 GPU1--3 的上行方向相反”只说明端点端口没有立即冲突；完整 FSDP 的所有 rank、所有 chunk 仍会同时使用两种方向以及同一 NVSwitch 内部资源。

## SC'24 能否在 NVLS 上复现

### 可以复现的部分

SC'24 的核心构造是“广播组成 Allgather”：不同 root 的 buffer 通过多播在网络中只经过一次链路。NVLS 对应的最小实现是：

1. 为参与 rank 创建一个 multicast object，每个 GPU 绑定一个等大的物理副本。
2. 将每个 root 的 shard 放在确定的 offset，root 发 `multimem.st`。
3. 轮换 root/offset，得到 Allgather 的所有 shard。
4. 对另一条 stream 的 RS chunk 发 `multimem.ld_reduce`，通过独立 communicator 保证 NCCL 的操作序列不互相阻塞。

单 NVLink 域内无需实现 SC'24 的软件可靠 multicast：NVLink/NVSwitch 是受控的 lossless fabric，NCCL/CUDA 负责 group 建立、映射和同步。SC'24 的三层并行思想仍可借鉴：多个 chunk/channel、发送与接收方向、避免多个 root 同时形成 incast。

### 不能原样复现的部分

SC'24 的 1.5 倍（四卡）是 full-duplex NIC 加 non-blocking fat-tree 模型的上界，不能当作 NVSwitch 内部的保证。论文实现依赖 CPU/UCC、BlueField DPA、RDMA multicast、分段重组和软件可靠性；其开源实现也要求 InfiniBand/OpenSM multicast groups。[论文代码](https://github.com/spcl/muliticast-based-allgather)。NVLS 没有同样的 SmartNIC 调度接口，且多播和 SHARP 共用 NVSwitch 资源。

跨节点时，NVLS 只能覆盖节点内的 rank；节点间需要 IB/NDR SHARP、GIN 或普通 NCCL 传输。因此应做分层 Allgather/Reduce-Scatter，而不是把 SC'24 的单层 multicast 拓扑直接替换成 NVLS。

## 八个问题的回答

### 1. Broadcast 和 SHARP 是否依赖 multimem？两者能否流水

“依赖”需要分层理解：NVLink SHARP 是 NVSwitch 的硬件能力；`multimem.st`、`multimem.ld_reduce` 和 `multimem.red` 是 CUDA/PTX 暴露给程序的访问方式；NCCL 的 NVLS symmetric kernels、LSA pointer 和 user-buffer 机制是软件封装。NCCL 不同版本和不同消息大小可能选择 symmetric kernel、staging 或 fallback，不能把所有 NVLS 路径都等同于一条 `multimem` 指令。

广播的 `multimem.st` 主要使用 multicast replication；归约的 `multimem.ld_reduce`/`multimem.red` 才使用 SHARP reduction datapath。故不能简单说两者都占用同一组 reduction ALU；更稳妥的说法是它们共享 multicast team/address、NVSwitch crossbar、credits 和部分控制器资源，是否存在独立的内部流水要由目标 NVLS 代际实测。

“NVLS 完全不能 pipeline broadcast 与 reduce”过于绝对。NCCL 的 NVLS transport 源码为每个 channel 分出 `Reduce UC -> MC` 与 `Broadcast MC -> UC` 两个 buffer half，并在一个 NVLS domain 建立共享 MC group。[NCCL `nvls.cc`](https://raw.githubusercontent.com/NVIDIA/nccl/master/src/transport/nvls.cc)。这说明软件有方向分离和并发设计空间；同时，共享 MC group、crossbar、credits 及 SHARP 控制器意味着仍会争用。要获得真正的 overlap，至少需要：

- AG、RS 使用独立 communicator/PG，且所有 rank 按同一顺序创建和调用；
- 不复用仍在通信中的 symmetric buffer、signal pad 或 chunk；
- 使用 Nsight Systems/Nsight Compute 观察 NVLS kernel、NVLink TX/RX、SM 和等待时间；
- 比较 `T(AG+RS)` 与 `max(T_AG,T_RS)`，而不是只看两个 CUDA stream 是否有时间区间交叠。

还要注意 Symmetric Memory API 的约束：同一个 group 的 multimem collective 需要从单一 CUDA stream 发出；把同一个 group 的 AG、RS 仅换到两个 stream 可能导致顺序错误或 barrier 死锁。[PyTorch Symmetric Memory](https://docs.pytorch.org/docs/main/symmetric_memory.html)。因此，AG/RS 的 stream 级并发必须建立在独立 group/handle 或经过验证的自定义 barrier 之上，不能只依赖 stream overlap。

TokenWeave 的结果不能作为“两个 NVLS collective 可同时满速”的证明。它把 `ld_reduce -> 寄存器计算 -> st` 融合在一个 kernel，并用另一个 token split 做流水；论文明确指出把 RS 和 AG 拆开会增加开销，在小消息上可能超过 50%。[TokenWeave](https://arxiv.org/abs/2505.11329)。这是融合/分块调度的证据，不是独立 AG 与 RS 同时占满交换机的证据。

### 2. NVLS 做 reduce 时吞吐会下降多少

没有一个适用于所有 GPU、消息大小和 NCCL 版本的固定百分比。应先明确带宽口径：对四卡、输出大小为 `S` 的 reduce，`ld_reduce` 逻辑上消费四个 operand（若 root 用 identity，则有效贡献是三个）并返回 `S`；broadcast 逻辑上从一个源注入 `S`，通过 multicast 生成多个副本。两者的“逻辑字节数”不同，直接比较 NCCL busbw 容易把归约的 N 路输入误算成降速。

硬件归约不会必然比普通链路慢到某个固定比例，但它要经过 SHARP 归约 datapath、控制器、credits 和 chunk 调度；端到端还包含对称内存注册、barrier、kernel launch 以及结果写回。NVIDIA 对第三代 NVSwitch 公布过 H100 系统约 450 GB/s reduction bandwidth；这是平台峰值，不是 GB200/NVLS Gen5 的保证。[NVIDIA NVSwitch 说明](https://developer.nvidia.com/blog/upgrading-multi-gpu-interconnectivity-with-the-third-generation-nvidia-nvswitch/)。公开资料没有 GB200 的通用 NVLS reduce 峰值，也没有“相对 broadcast 下降 X%”的规格。

作为量级参考，NCCL 的 H100 issue 报告在某个 Reduce-Scatter 测试中记录 ring 约 344.5 GB/s、NVLS 约 305.7 GB/s，约低 11.3%；另一版本/消息尺寸的差异约 12%。这是特定软件、拓扑和测试口径的观察，不是硬件定律，但说明“启用 NVLS 就一定更快”也不成立。[NCCL issue #1047](https://github.com/NVIDIA/nccl/issues/1047)。

建议报告两个值：

```text
logical_reduce_bw = output_bytes / elapsed
operand_bw        = participant_count * output_bytes / elapsed
```

并分别测单独 AG、单独 RS、同一 PG 的 AG+RS、独立 PG 的 AG+RS。小消息通常受 launch/barrier/波量化影响，大消息才接近链路或 SHARP datapath 上限；TokenWeave 的小 tile 带宽下降和 RS+AG 拆分开销也说明不能从单一峰值外推所有尺寸。

### 3. `multimem.slot` 是 NCCL、GPU 还是 NVSwitch 概念

公开 CUDA PTX 和 NCCL Device API 没有名为 `multimem.slot` 的标准符号。这里至少有三种容易混淆的“slot”：

1. **NVSwitch/Fabric Manager 的 multicast team/group 资源**：创建和绑定 multicast object 时隐式消耗，属于 NVSwitch fabric 的有限资源。
2. **NCCL 的软件资源**：channel、credit/data/UB slice、barrier 或队列槽位，用于调度和同步。
3. **应用自己的 chunk/流水槽**：例如 FSDP double buffer。

因此，如果资料中的 slot 指“可创建多少个 multicast team”，它是 fabric/control-plane 资源，NCCL 只是申请和复用；如果指 NCCL 的 queue slot，则不能用来推断 NVSwitch 硬件带宽。NCCL GIN 文档中 `nSlots`、barrier state 等就是软件同步/队列概念。[NCCL GIN 与 symmetric memory](https://github.com/NVIDIA/nccl/blob/master/docs/contrib/GIN/NCCL_Gin_and_Symmetric_Memory/NCCL_Gin_and_Symmetric_Memory.md)。

### 4. GB200 和 NVLS Gen5 有多少 slot

需要区分公开的“team/group 配额”和每个 GPU 的硬件 slot：

- H100/NVSwitch 第三代资料称 SHARP controller 可并行管理最多 128 个 SHARP groups；H200 上也有 NCCL/PyTorch issue 报告观察到 128 multicast slots 的硬上限。这些是 Hopper 代/实测报告，不能直接当作 GB200 规格。[NVIDIA 博客](https://developer.nvidia.com/blog/upgrading-multi-gpu-interconnectivity-with-the-third-generation-nvidia-nvswitch/)、[PyTorch issue #178749](https://github.com/pytorch/pytorch/issues/178749)、[NCCL issue #2077](https://github.com/NVIDIA/nccl/issues/2077)。
- GB200/GB300 NVLink partition 文档公开的是 **NVLink domain/partition 的 multicast team 配额，最多 1,024**；它是全局资源，按 partition 配置共享，不是“每个 GPU 1,024 个 slot”，也不是每个数据流能并行 1,024 路。[Mission Control NVLink partition](https://docs.nvidia.com/mission-control/docs/systems-administration-guide/2.1.0/nvlink-partition-management.html)、[GB200 partition guide](https://docs.nvidia.com/multi-node-nvlink-systems/partition-guide-v1-2.pdf)。
- 目前没有公开文档给出“NVLS Gen5 每个 GPU/每个 NVSwitch ASIC 的 multimem.slot 精确数量”。如果要按 slot 做 FSDP 子组容量规划，应向平台/NVSwitch 固件文档确认，不能把 1,024 team 配额写成吞吐并行度。

### 5. slot 是否决定 reduce/broadcast 吞吐，需要多少 slot 才能占满带宽

不直接决定。slot/team 是能否同时存在多个 multicast object/communicator 的资源；单个 object 可以承载很大的对称 buffer 和多个 offset。NCCL 当前 NVLS 代码按 channel 分配数据、credit 和 user-buffer slice；源码中 SM90 默认 16 channels，Blackwell 单节点配置为 24、跨节点为 32（具体版本和环境变量可能变化）。channel/CTA、chunk size、GPU 到 NVSwitch 链路、crossbar、SHARP ALU、HBM 和消息大小共同决定吞吐。

“需要多少 slot 才能占满带宽”没有公开公式。通常应先用一个 communicator/team、足够大的消息和默认 channel 配置测单操作 line-rate；增加 slot 主要支持多 communicator/多租户并发，不能把一个 reduce 自动变成 N 倍带宽。过多 FSDP 子 communicator 反而可能在初始化阶段耗尽 multicast 资源，出现 `cuMulticastBindMem` 失败。

### 6. NVLS 能否把 3 个 GPU 聚合后写入第 4 个 GPU

可以实现，但需要准确理解触发方式。建立四卡 multicast team 后，让 GPU1、2、3 把 operand 写到各自 replica，GPU0 发起 `multimem.ld_reduce`，结果会返回 GPU0；GPU0 的 replica 可预先填充 identity（加法为 0），从而排除它自己的贡献。另一种做法是创建只含 GPU1、2、3 的 team，再由 GPU0 负责接收路径，但这会引入额外 team/映射管理。

PTX 没有在 `ld_reduce` 上传入“排除某个成员”的 mask；`multimem.red` 的语义是把结果写回所有 replica，也不是 root-only。NCCL Device API 也要求 multimem reduce 的 local rank 参与该 team 的源/目的语义。[NCCL multimem ReduceSumCopy](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_reducecopy.html)。NCCL 的常规 symmetric AllReduce 通常是 reduce-load 加 multicast-store；NCCL RFE #2072 提出的“先让部分 GPU reduce，再由 root 做最后 reduce 和 broadcast”正是一个需要额外 multicast group 或 fused/masked primitive 的优化建议，不代表现成 NCCL 已支持。[NCCL RFE #2072](https://github.com/NVIDIA/nccl/issues/2072)。

因此，用户设想的单目的 reduce 是可做的自定义 kernel/microbenchmark；不能直接把现有标准 `multimem_reduce_scatter` 当成“3 到 1”的 push API。

### 7. ATP 的多租户场景是否适用于 64 GPU、两个 DP 组

在“资源竞争”层面适用，在“协议语义”层面不等价。ATP 面向多 job、多 rack 的异步 in-network aggregation：交换机聚合器动态分配，资源不足时可回退，设计了公平和拥塞处理。[ATP/NSDI'21](https://www.usenix.org/system/files/nsdi21-lao.pdf)、[ATP repository](https://github.com/in-ATP/ATP)。

两个 DP 组如果共享同一个 NVLink domain/partition，会竞争 multicast team quota、crossbar、credits、SHARP controller/ALU 和 GPU/NVLink 端口。一个组计算更慢、进入通信的 duty cycle 更低时，另一个组可能在 work-conserving 情况下得到更多 fabric 时间；但这不是 ATP 式的 QoS 或公平保证，需要用两组不同速度的 workload 实测。

差异在于 NCCL/FSDP 是 strict collective：同一 communicator 的 rank 必须按相同顺序到达，缺少成员时通常等待或报错，没有 ATP 的“部分聚合后继续”和端到端回退。ATP 论文还指出 ring AllReduce 的梯度在交换机不会相遇；NVLS/SHARP 的星型/多播路径才使“交换机内聚合”的类比成立。

### 8. NVLS reduce 占用哪些资源，按什么分组

可以按五层理解：

1. **GPU 端**：发送/接收 NVLink 端口、HBM/L2 访问、发起 multimem 的 SM/CTA、事件和 signal pad。
2. **NVSwitch 端口与 crossbar**：每个方向的链路、虚拟通道和 credit；端点方向互补不代表 crossbar 没有竞争。
3. **SHARP/NVLS datapath**：multicast group context、归约 ALU/SRAM、控制器以及返回路径。
4. **NCCL communicator/channel**：每个 NVLS domain 的 MC group、`nHeads`/rail 划分、每个 channel 的 reduce/broadcast slices、credit/data/UB 分区、chunk 和 CTA 数量。
5. **Fabric Manager partition 配额**：多播 team/group 的全局上限和 partition 规则；同一 partition 中的多个 DP 组共享，Hopper 的某些分区模式与 GB200/GB300 的全局配额规则不同。

资源首先按 NVLink domain/partition 和 multicast team 分组，其次按 communicator/channel 分片；不是每个 GPU 独占一个“reduce slot”。这也是为什么独立 AG PG 能改善操作排序，却不能保证获得独立的 SHARP 硬件实例。

## FSDP3 集成方案与难点

### 当前 Bridge/Megatron 路径

Bridge 的 FSDP3 实验 README 已明确将 `optim_grads_params` 的 AG/RS overlap 作为测量对象，并指出 `communication-unit-size` 同时影响 AG 和 RS 的预取/队列。[FSDP3 overlap README](/Users/zhangbochun/Desktop/AI-Infra/Megatron/Megatron-Scale-Down/megatron-bridge/examples/scale-down/01-analyse/04-fsdp3-comm/README.md)。

MCore 的 adapter 已读取 `dp_cp_ag`/`expt_dp_ag` 并传给 `fsdp_group_ag`（[mcore_fsdp_adapter.py](/Users/zhangbochun/Desktop/AI-Infra/Megatron/Megatron-Scale-Down/megatron-bridge/3rdparty/Megatron-LM/megatron/core/distributed/fsdp/mcore_fsdp_adapter.py:396)）；经典 MPU 路径有 `create_all_gather_groups()`，其目的就是用同 rank 集合的独立 communicator 支持 AG/RS overlap。当前 Bridge decentralized 初始化在 `ProcessGroupCollection` 中只填 `dp_cp`/`expt_dp`，没有填 AG 对应字段（[initialize.py](/Users/zhangbochun/Desktop/AI-Infra/Megatron/Megatron-Scale-Down/megatron-bridge/src/megatron/bridge/training/initialize.py:534)）。这应先作为集成缺口处理。

由于 Bridge 已经传入自建 `pg_collection`，MCore `get_model` 中“没有 `pg_collection` 才自动调用 `create_all_gather_groups()`”的兜底路径不会替它创建 AG group；因此必须在 Bridge 初始化阶段显式创建并 attach `dp_cp_ag`/`expt_dp_ag`，而不是期待默认初始化补齐。

这与 PyTorch FSDP2/FSDP3 的公开说明一致：AG 和 RS 即使在不同 CUDA stream，默认仍共享一个 NCCL communicator，一次只能处理一个 collective；启用 separate reduce-scatter process group 才允许网络支持时并行。[PyTorch FSDP `set_separate_reduce_scatter_group`](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)。

### 建议的分阶段实现

**阶段 A：建立可归因的基线。**

- 同一 PG、标准 NCCL NVLS：单独 AG、单独 RS、按现有 FSDP3 预取运行。
- `NCCL_NVLS_ENABLE=0` 做 fallback 对照。
- 检查 recipe 是否覆盖了环境变量；当前 Bridge 的部分 GB200 Llama 性能 recipe 显式把 `NCCL_NVLS_ENABLE` 设为 `0`，复现实验必须覆盖该设置并从 `NCCL_DEBUG=INFO` 日志确认实际算法。
- 记录 AG/RS kernel 名、通信 stream、step time、`ag_rs_exposed_pct`，并检查 NVLink TX/RX 和 SM 使用。

**阶段 B：独立 AG communicator。**

- 为 `dp_cp` 和 expert DP 创建完全相同 rank 集合的 `dp_cp_ag`/`expt_dp_ag`；所有 rank 按统一顺序创建。
- decentralized HyperCommGrid 不能简单对同一维度再次 `create_pg`；需要用 `grid._gen_rank_enum(["dp", "cp"])` 加 `torch.distributed.new_subgroups_by_enumeration`（或建立独立 view/grid）创建新的 NCCL group，并确保每个 rank 的创建顺序一致。
- 让 AG 使用 `fsdp_group_ag`，RS 使用原 group；保持每个 communicator 内部顺序不变。

**阶段 C：对称内存与 NCCL symmetric kernels。**

- 使用 NCCL user buffers/symmetric memory，保证地址、大小、16B 对齐、注册和生命周期满足 NVLS 要求；普通 FSDP flat parameter 的 `cudaMalloc` 地址不能自动成为 multicast object。
- 遵守每个 symmetric-memory group 的 stream 约束；若 AG 与 RS 要在不同 stream 同时发射，应使用独立 group/handle，并为 signal pad、block-id 和事件建立独立命名空间。
- FSDP double buffer、prefetch event、梯度归约完成事件必须与通信 buffer 解耦，不能在 AG/RS 未完成时复用。
- 先使用 NCCL 的 symmetric AG/RS kernel（例如 LLMC/STMC/LDMC 变体），只有在基线证明硬件路径可重叠后才写自定义 multimem extension。

**阶段 D：自定义定向 kernel（若确实需要 3 到 1）。**

- 为 AG 和 RS 使用不同 chunk/offset、signal-pad namespace 和 stream；不要让两个 pipeline 共享同一个 barrier slot。
- 对 root identity、归约结果写回、错误处理、rank 不一致和异常退出做完整验证。
- 评估每个额外 FSDP subcommunicator 消耗的 multicast team；优先复用 parent NVLS resources，避免多子组触发 slot exhaustion。

### 主要难点

1. **语义和顺序**：FSDP 的 AG/RS 是动态 bucket、prefetch 和 autograd 依赖，不是两个可以任意交错的裸 kernel。
2. **内存**：multimem 需要 symmetric/multicast object；注册、对齐、barrier、buffer 生命周期可能抵消 staging copy 的收益。
3. **共享硬件**：独立 PG 解决 communicator 排队，不会创建独立 NVSwitch crossbar/SHARP 实例。
4. **粒度**：chunk 太小会受到 launch、barrier 和波量化影响；太大则暴露通信尾部。`communication-unit-size` 同时影响 AG 和 RS，不能当作单独开关。
5. **容量**：过多 FSDP unit/PG 可能耗尽 multicast team，而不是在运行时平滑降速。
6. **跨域**：NVLS 只在一个 NVLink domain 内有效；多节点 DP 必须联合 IB/SHARP/GIN，且需要重新分析层次化拓扑。

## 建议的最小实验矩阵

在目标 GB200/NVLS Gen5 系统上，先做四卡、再做完整 NVL 域：

| Case | AG PG | RS PG | AG/RS 数据 | 目的 |
|---|---|---|---|---|
| A | 同一 PG | 同一 PG | 标准 NCCL | 现有 FSDP 基线 |
| B | 独立 PG | 独立 PG | 标准 NCCL symmetric | 分离 communicator 排队影响 |
| C | 独立 PG | 独立 PG | GPU0 `st` + GPU0 `ld_reduce` | 用户提出的方向互补 microbenchmark |
| D | 独立 PG | 独立 PG | 完整 FSDP 多 root/chunk | 检验真实 AG/RS，而非单 root 特例 |
| E | 独立 PG | 独立 PG | NVLS off | 网络/软件 fallback 对照 |

每个 case 变更消息大小、chunk size、channel/CTA 配置、stream 间距和 1/2 个并发 DP 组。至少记录：`T_AG`、`T_RS`、`T_both`、logical/operand bandwidth、NVLink 每端口 TX/RX、NVLS kernel 的 SM 使用、NCCL 初始化是否出现 multicast bind 错误，以及 FSDP step time。只有当 `T_both` 接近 `max(T_AG,T_RS)` 且端口/SHARP 计数显示没有明显降频时，才可以称为“接近完美 overlap”。

可用以下指标区分两种极限：`T_ideal = max(T_AG, T_RS)`，`T_serial = T_AG + T_RS`，`overlap_efficiency = (T_AG + T_RS) / T_both`。效率接近 2 且 `T_both` 接近 `T_ideal` 才接近理想并行；效率接近 1 说明两种操作基本串行。若并发后每个 collective 的 bus bandwidth 都下降，仍不能只凭时间线把它判为成功 overlap。

## 最终判断

- **能否做？** 能。NVLS 原语足以表达 broadcast、reduce 和由它们组成的 AG/RS；3 GPU 聚合到第 4 GPU 也能通过 root 发起 `ld_reduce` 实现。
- **能否像 SC'24 一样保证近 2 倍或四卡 1.5 倍？** 不能直接保证。SC'24 的公式依赖全双工 NIC 和 non-blocking 网络；NVLS 的共享 switch/SHARP 资源需要实测。
- **能否“完美 overlap”？** 只能作为目标，不是硬件承诺。正确顺序是独立 PG → symmetric NCCL baseline → 定向 multimem kernel → 真实 FSDP trace 和 NVLink/NVSwitch counter。
- **slot 是否是吞吐旋钮？** 不是。slot/team 是并发对象和资源配额；吞吐由链路、crossbar、SHARP datapath、channel/chunk/CTA、SM/HBM 和消息大小共同决定。

## 补充：multicast slot、SHARP 资源和 TokenWeave 实现审计

本节针对 NCCL issue #2077 中的 “128 multicast slots” 重新澄清硬件资源含义，并补充 NVSwitch Gen4/Gen5 的公开资料边界。这里的 slot 确实应理解为 NVSwitch/fabric 中用于 multicast team/group 的硬件资源，而不是普通 NCCL queue slot；但它仍不等于 SHARP 数据缓存容量，也不等于可达到的带宽倍数。

### Gen4/Gen5 的命名必须先固定

公开资料存在两套代际命名，不能把它们混写：

| 产品/资料口径 | GPU 链路 | Switch 口径 | 公开的关键能力 |
|---|---|---|---|
| Hopper/H100 | NVLink 4 | NVIDIA Fabric Manager 文档通常称第三代 NVSwitch；Hot Chips 2022 的演讲标题称 “4th-Generation NVSwitch Chip” | 64 个 NVLink4 端口、嵌入式 SHARP ALU、SHARP controller、scratch SRAM；演讲材料写 controller 可管理最多 128 个 SHARP groups |
| Blackwell/GB200 | NVLink 5，单 GPU 1.8 TB/s 双向 | Hot Chips 2024 称第四代 NVLink Switch Chip/Tray | GB200 NVL72 的 72-GPU 域、每个 switch chip 7.2 TB/s 全互联双向，SHARPv4；没有公开每 ASIC 的 slot 或 scratch SRAM 字节数 |

因此，“NVLS Gen5”如果是指 NVLink5/GB200，应使用 **Blackwell、第四代 NVLink Switch chip** 这套公开名称。Hopper 资料中的 128 不能直接套用到 GB200。参考：[Hot Chips 2022 H100/NVSwitch](https://hc34.hotchips.org/assets/program/conference/day2/Network%20and%20Switches/NVSwitch%20HotChips%202022%20r5.pdf)、[Hot Chips 2024 Blackwell/GB200](https://hc2024.hotchips.org/assets/program/conference/day1/64_HC2024.NVIDIA.TirumalaWong.pdf)、[GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/)。

### 公开资料能确认哪些 SHARP 资源

Hot Chips 的 NVLink4/NVSwitch 图把数据路径画成 `SHARP Controller`、`SHARP ALU` 和 `SHARP Scratch SRAM`，并说明 ALU 支持逻辑运算、整数以及 FP16/32/64/BF16 等类型。[Hot Chips 2022](https://hc34.hotchips.org/assets/program/conference/day2/Network%20and%20Switches/NVSwitch%20HotChips%202022%20r5.pdf) 能支持以下资源分层：

1. **端口和 crossbar**：承载 GPU↔switch 的 ingress/egress、路由、virtual channel 和 credit。
2. **multicast team/group 上下文**：保存成员、地址映射、复制/归约路径的控制状态。NVIDIA Fabric Manager 称这类有限资源为 **Multicast Slots**；H100/HGX 的 slot 集合与 NVLink partition 独立，B200 代则由 partition 创建时按 8/4/2/1 GPU 规模分配上限。[Fabric Manager User Guide](https://docs.nvidia.com/datacenter/tesla/pdf/fabric-manager-user-guide.pdf)
3. **SHARP controller 和 ALU**：对流经该 group 的 chunk 做归约及调度。broadcast 需要复制和转发，但不需要执行加法 ALU。
4. **SHARP scratch SRAM**：保存归约流水中的暂存/累加状态。公开硬件图没有给出每个 switch、每个 group 或每个 ALU 的 SRAM 字节数，也没有公开组间切分算法。
5. **GPU 侧 multicast object 和 NCCL buffer**：这是 CUDA/NCCL 在 GPU HBM 中分配和映射的内存，不能当成 switch scratch SRAM。

CUDA Driver API 明确 `cuMulticastBindMem()` 的 `size` 是绑定到 multicast object 的内存大小；资源不足时返回 `CUDA_ERROR_MULTICAST_RESOURCE_FULL` 或 OOM。[CUDA Multicast Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__MULTICAST.html) 因此，NCCL 日志中的 `size 2097152`、`size 536870912` 或更大值表示一次 GPU 侧 object/buffer 的绑定请求，不是 “switch 内部有 2 MiB/512 MiB 缓存”。NCCL issue #2077 报告的 128 是 group/slot 上限的实测现象；它没有给出每个 slot 的 scratch SRAM 容量。[NCCL issue #2077](https://github.com/NVIDIA/nccl/issues/2077)

本次检索的公开资料未给出“Gen4/Gen5 的 SHARP scratch SRAM 总共多少 MiB、每个 slot 分多少字节、按 rank 还是按 channel 分配”。这些应标记为 **未找到公开依据的硬件细节**；这不证明 NVIDIA 内部或受限手册没有相关规格，也不能用 NCCL 的 GPU buffer 大小反推。

### multicast 会不会占用 SHARP 资源

会占用 **multicast team/group slot、地址映射、复制路径、crossbar、credit 和控制状态**。CUDA 文档说明 multicast object 有多个物理副本，`multimem.st` 通过 NVLink SHARP 相关路径把写入复制到副本；`multimem.ld_reduce` 则读取副本并执行归约。[CUDA Multicast Memory](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/04-special-topics/virtual-memory-management.html)

但“占用 multicast slot”不等于“占用同样多的 reduce ALU”。更准确的资源关系是：

- `multimem.st`：需要 multicast group/team、复制树、crossbar、credits 和 GPU 侧 object；通常不需要做算术归约的 SHARP ALU。
- `multimem.ld_reduce`/`multimem.red`：除上述 group/复制基础设施外，还需要 SHARP controller、归约 ALU、暂存状态以及结果返回路径。
- AG 与 RS 共享 team/group 和部分 fabric 资源，即使软件上使用不同 communicator，也不会凭空得到两套 switch crossbar 或两套 SHARP ALU。

这解释了为什么两个 stream 可以在端点方向上重叠，却不能由 “GPU0 上行与 GPU1--3 上行相反” 推导出 switch 内部一定无冲突。

### NCCL 中容易混淆的另外几类 slot/buffer

当前 NCCL NVLS transport 将 NVLS domain 的数据拆成 channel/head/chunk，并为 reduce 和 broadcast 分出不同的方向/half。公开源码中的默认量包括：SM90 通常 16 channels，Blackwell 单节点通常 24、跨节点通常 32（版本和环境变量可调整）；默认 chunk size 为 128 KiB，`NCCL_STEPS` 为 8。[NCCL `nvls.cc`](https://raw.githubusercontent.com/NVIDIA/nccl/master/src/transport/nvls.cc)、[NCCL `device.h`](https://github.com/NVIDIA/nccl/blob/master/src/include/device.h)

以源码的内存公式表示，设 `B = chunk_size × NCCL_STEPS`，则每 rank 的 GPU 侧数据缓冲近似为：

```text
data_gpu_bytes_per_rank = nChannels × 2 × B × nHeads
credit_gpu_bytes        = nChannels × 2 × 64 × nHeads
```

例如 H100 的一个 NCCL 日志曾出现 `nChannels=16`、`nHeads=8`、`buffSize=1 MiB`、`nvlsPerRankSize=32 MiB`，随后为一个 multicast object 申请约 256/512 MiB 的 GPU 侧空间；另一个配置甚至达到 GiB 级。它们是 staging/credit/UB 的 HBM 映射，**不是 switch SRAM**。[NCCL issue #1584](https://github.com/NVIDIA/nccl/issues/1584)、[NCCL issue #1259](https://github.com/NVIDIA/nccl/issues/1259)

此外，NCCL 还有 `NCCL_MAX_NVLS_ARITY=32` 等 kernel 数组容量，以及 PAT reduce-scatter 中按 wave 使用的 FIFO slot。这些是软件 kernel 的 fan-in/队列容量；不能与 NCCL issue #2077 的 fabric multicast group slot 混为一谈。

### 32 个和 64 个 rank 需要多少资源才能打满出口

没有公开的公式可以写成“32 rank 要 32 个 slot，64 rank 要 64 个 slot”。一个参与 32 rank 或 64 rank 的 reduce-scatter communicator 通常只需要一个 multicast team/group；rank 数增加的是每个输出 chunk 的 fan-in、总 operand 流量和调度压力，而不是按 rank 线性增加 group slot。

可以用下式理解测量目标。若每个 rank 的本地输入大小为 `M`，参与者为 `P`，每个 rank 最终接收约 `M/P`：

```text
T >= max(T_ingress, T_SHARP(P), T_egress, T_control)
T_ingress  ≈ M / B_gpu_tx       # 每个 rank 的输入发送压力
T_egress   ≈ (M/P) / B_gpu_rx   # 每个 rank 的结果接收压力
```

`T_SHARP(P)` 由未公开的 controller/ALU/scratch 服务率、fan-in、chunk 和 credit 决定。若固定每 rank 的 `M`，从 32 增到 64 主要增加交换机的总归约输入和 fan-in 延迟；若固定每 rank 的输出 shard 大小 `S`，则每 rank 输入变成 `P×S`，总 operand 流量随 `P` 增大。两种实验的结论不能混用。

工程上应保持一个 team，使用足够大的对齐消息，先调默认 `nChannels`、chunk 和 NCCL CTAs，观察每 GPU 的 NVLink TX/RX、NVLS bus bandwidth、kernel occupancy 和完成时间。当端口已接近 line-rate 或 `T_SHARP` 成为瓶颈时，再增加 slot 不会提高单个 collective 的带宽；额外 slot 主要用于多个 communicator/租户并发。对 64 rank，可能需要更多 channel/chunk 来隐藏 fan-in latency，但没有证据表明必须把 slot 数翻倍。

NCCL `reduce_scatter.h` 的注释提到按 rank 数在 GPU kernel 中分配 5/7 个 warp 来填充 NVLS 带宽；这属于 **GPU 端 warp/CTA heuristic**，不是 NVSwitch slot 分配规则。[NCCL `reduce_scatter.h`](https://github.com/NVIDIA/nccl/blob/master/src/device/reduce_scatter.h)

### TokenWeave 对这个问题能证明什么

已检查 `/Users/zhangbochun/Desktop/AI-Infra/Megatron/tokenweave` 的实现，结论是它证明了 GPU 端的融合和流水策略，不提供 NVSwitch slot 或 SHARP scratch 的查询：

- [tokenweave_multimem_utils.cuh](/Users/zhangbochun/Desktop/AI-Infra/Megatron/tokenweave/csrc/tokenweave_multimem_utils.cuh:250) 的 reduce-scatter 用 `multimem.ld_reduce` 从 symmetric replicas 读取并写本地 shard；`WORLD_SIZE` 改变 shard 几何和 fan-in，不会直接改变硬件 slot 数。
- [tokenweave_fused_kernels.cu](/Users/zhangbochun/Desktop/AI-Infra/Megatron/tokenweave/csrc/tokenweave_fused_kernels.cu:82) 的融合 kernel 顺序执行 `ld_reduce → 本地 residual/LayerNorm → multimem.st`，并通过 signal pad/barrier 连接阶段；这不是两个独立 NVLS collective 同时发射。
- [triton_comm.py](/Users/zhangbochun/Desktop/AI-Infra/Megatron/tokenweave/vllm/distributed/triton_comm/triton_comm.py:329) 的 `multimem_all_gather_async` 是把本地 shard 用 `cudaMemcpyAsync` 拷贝到 multicast pointer 的 copy stream；它与前面的 reduce/barrier/compute 形成流水，但没有调用硬件 slot 查询接口。
- [tokenweave_utils.py](/Users/zhangbochun/Desktop/AI-Infra/Megatron/tokenweave/vllm/model_executor/models/tokenweave_utils.py:214) 的 overlap 路径使用两个 CUDA stream 和 staging buffer；`MAX_CTAS=2..32` 是 GPU kernel CTA 数，不是 multicast slot。
- staging buffer 由 `symm_mem.empty()` 和 `symm_mem.rendezvous()` 创建，位于 GPU 对称内存/HBM；它不是 NVSwitch 内部 scratch SRAM。[AddTokenWeave.md](/Users/zhangbochun/Desktop/AI-Infra/Megatron/tokenweave/docs/AddTokenWeave.md:247)

因此，TokenWeave 可以用来验证 “RS/本地计算/AG 分块之间的流水” 和 “融合是否减少 launch/barrier/copy 开销”，不能用来推断 “AG 与 RS 各自占多少 NVSwitch slot” 或 “64 rank 需要多少 slot”。

### 用户能否在 NVSwitch 上增加逻辑行为

生产 NVSwitch 没有面向用户的 datapath/RTL/SHARP-ALU 编程接口。用户可以在 GPU 端编写 CUDA/Triton/NCCL/NVSHMEM kernel，使用公开的 `multimem.ld_reduce`、`multimem.st`、`multimem.red` 组合本地计算、分块、barrier 和结果写回；NCCL Device API 也支持固定类型和固定归约语义的 multimem 操作。[NCCL Device API](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html)、[NCCL ReduceCopy API](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_reducecopy.html)

Fabric Manager/NVLSM/NVOS 管理员可以配置 NVLink partition、路由、监控、资源上限和固件生命周期；这些是控制面，不能增加新的 switch-side operator、改变 SHARP ALU 的内部调度或加载用户函数。[DGX GB200 software](https://docs.nvidia.com/dgx/dgxgb200-user-guide/software.html)、[NVOS User Manual](https://docs.nvidia.com/networking/display/NVIDIANVOSUserManualforNVLinkSwitchesv25-02-2223.2223.pdf)

所以可行的扩展边界是：

- **可由用户完成**：GPU 端 fused RS/AG、本地预处理/后处理、chunk 和 stream 调度、多个 multicast object 的组织、partition/配额配置（需管理员权限）。
- **需要 NVIDIA 硬件/固件支持**：新的交换机算子、任意用户代码、改变 scratch SRAM 分配、增加 multicast slot 数、定义新的 group 内部状态机。

如果需要交换机内执行非加法的自定义算子，现实路径是向 NVIDIA 提交 CUDA/NCCL/Fabric Manager feature request，或在 FPGA/研究型交换机上实现；不能通过 FSDP、NCCL 环境变量或普通 CUDA kernel 把逻辑写入生产 NVSwitch。

### 对原八个问题的两处精确化

- **关于 3→1 reduce**：若 GPU0 不是参与 operand 的 rank，不能用一个只含 GPU1/2/3 的 `ld_reduce` 指令直接把结果“推送”到 GPU0；公开语义要求发起 load 的 local rank 参与 team。可行方案是四卡 team 中让 GPU0 的 replica 放 identity，再由 GPU0 发起 `ld_reduce`，或由一个参与者取回结果后再做一次 GPU→GPU copy。
- **关于 Gen5 slot 数**：现有报告中引用的 128 应限定为 Hopper/H200 issue 和公开演示的上限；GB200/Blackwell 公开文档给出的是最多 1,024 个 NVLink partition multicast teams 的域级配额，不能写成每 GPU 或每 ASIC 的物理 slot 数，也不能用来推导带宽。

## multicast group 的详细生命周期：硬件语义、NCCL 和 TokenWeave

### 1. controller、ALU、scratch SRAM 和 multicast group 的关系

NVIDIA 对 Hopper 代 NVLink4 NVSwitch 的公开架构图给出了 `SHARP Controller`、`SHARP ALU (Hopper)` 和 `SHARP Scratch SRAM` 三个硬件块。资料还明确写出：ALU 支持 logical、min/max、add 以及 S/U integer、FP16、FP32、FP64、BF16，controller 最多可并行管理 128 个 SHARP groups；但没有给出 scratch SRAM 的字节容量、bank 数、每 group 配额或调度算法。[Hot Chips 2022 NVLink-Network Switch](https://www.hc34.hotchips.org/assets/program/conference/day2/Network%20and%20Switches/NVSwitch%20HotChips%202022%20r5.pdf)、[NVIDIA NVSwitch 技术博客](https://developer.nvidia.com/blog/upgrading-multi-gpu-interconnectivity-with-the-third-generation-nvidia-nvswitch/)

四者的职责可以按下面的层次理解：

| 层次 | 作用 | 用户可见的配置 | 不能从公开资料得到的部分 |
|---|---|---|---|
| multicast group/team context | 保存成员、group 生命周期、映射和复制/归约路径状态；对应 Fabric Manager 所说的有限 Multicast Slots | `numDevices`、成员 GPU、group 生命周期 | 一个 group 消耗多少 controller 状态/slot 微资源 |
| SHARP Controller | 为 group/chunk 调度归约、复制、credit 和完成状态 | 只能间接通过 CUDA/NCCL collective 使用 | 具体调度公平性、并发优先级、每 group 状态大小 |
| SHARP ALU | 对流中的元素执行支持的 reduce operator | 数据类型和 operator 由 PTX/NCCL API 限定 | 每 ASIC 的 ALU 数、每 operator 的并发度 |
| SHARP Scratch SRAM | 保存归约流水的部分结果/暂存状态，使数据按 chunk 流过 switch | 没有公开用户 API | 总容量、每 group/chunk 分配、是否动态共享 |
| crossbar/port/credit | 承载 ingress、egress、路由和 backpressure | topology/partition 由系统软件管理 | 每 group 对 VC/credit 的精确配额 |

scratch SRAM 应理解为 **流式归约的内部暂存**，而不是完整 tensor 的缓存。CUDA multicast object 的 payload 在 GPU 侧由每个成员 GPU 的物理内存 backing；一个 N-GPU team 对应 N 个本地物理副本。[CUDA Programming Guide: Multicast Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html)

因此，以下数值不能互相推导：

- `128 multicast slots`：是 Hopper/H200 上观察到的 group/context 数量级，不是 128 个 SRAM 缓存。
- `CUmulticastObjectProp::size` 或 NCCL 日志中的 `size 536870912`：是每 GPU 绑定到 multicast object 的 GPU 侧内存大小，不是 switch SRAM。
- NCCL 的 `nChannels × nHeads × chunk`：是 GPU 侧 staging/credit/data 布局，不是 SHARP ALU 或 scratch SRAM 的数量。

### 2. CUDA 侧如何建立一个 multicast group

CUDA Driver API 给出的必要步骤是固定的：

```text
query CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED
    ↓
cuMulticastCreate(numDevices=N, size=per_device_object_size)
    ↓
把 mcHandle 共享给控制 N 个 GPU 的进程
    ↓
每个参与 GPU 调用 cuMulticastAddDevice(mcHandle, dev)
    ↓  （所有 GPU 都 AddDevice 完成后才能继续）
每个 GPU:
    cuMemCreate(local_physical_memory)
    cuMulticastBindMem(mcHandle, mcOffset, local_mem, memOffset, size)
    ↓
每个 GPU:
    cuMemAddressReserve
    cuMemMap(multicast VA, mcHandle)
    cuMemSetAccess(multicast VA, local GPU, READWRITE)
    ↓
GPU kernel 使用映射后的 multicast VA 执行 multimem.st / multimem.ld_reduce
```

`numDevices` 在 `cuMulticastCreate` 时确定，`cuMulticastAddDevice` 对每个对象只允许一次；加入的 GPU 必须属于同一个 clique。所有成员加入完成前，bind 和 map 会阻塞，或可能返回状态错误。绑定的 `size`、`mcOffset` 和地址必须满足 `cuMulticastGetGranularity` 的对齐要求；资源不足时 API 可以返回 `CUDA_ERROR_MULTICAST_RESOURCE_FULL` 或 OOM。[CUDA Driver API: Multicast Object Management](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__MULTICAST.html)

group 本身并没有“root”字段：

- `multimem.st [mc_va], value`：发起该指令的 GPU 将 value 写入 group 的所有副本，适合 broadcast 或 all-gather 中某个 rank 写自己的 offset。
- `multimem.ld_reduce ... [mc_va]`：发起 load 的 GPU 是结果 executor；switch 读取所有成员副本并把归约结果返回该 GPU。若要让 GPU0 得到 GPU1/2/3 的和，可以让 GPU0 也成为 team 成员并把对应副本初始化为加法 identity 0。
- `multimem.red`：是写回 multicast 映射的归约形式；它也没有一个公开的 root-only mask。

释放时必须保证所有 GPU 不再使用该 object，再执行 `cuMulticastUnbind`、unmap/release；GB200 NVL72 软件 release notes 还记录了大量 object 同时清理期间新 object 创建可能暂时返回 `CUDA_ERROR_SYSTEM_NOT_READY`，建议重试时创建新的 multicast team/handle，而不是复用失败的 handle。[DGX GB200 NVL72 Release Notes](https://docs.nvidia.com/pdf/nvidia-dgx-gb200-nvl72-systems-rack-system-software-and-firmware-1-0-release-notes.pdf)

### 3. 当前 NCCL NVLS 如何设置和复用 multicast group

下面是基于当前 NCCL `src/transport/nvls.cc` 的实际生命周期；不同 NCCL release 可能存在“credit/data 分成多个 object”与“一个 object 切片”的差异，必须以运行时版本源码和 `NCCL_DEBUG=INFO` 日志为准。

#### 3.1 能力探测和软件调优

`ncclNvlsInit()` 在 `NCCL_NVLS_ENABLE` 允许时检查 CUDA driver 是否有 `cuMulticastCreate`，并查询 `CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED`。`ncclNvlsTuning()` 决定 `nvlsChannels`、`nvlsChunkSize` 和 tree chunk 上限；这些是 NCCL 软件布局参数，不是硬件 multicast slot 数。[NCCL `nvls.cc` 初始化与调优](https://github.com/NVIDIA/nccl/blob/master/src/transport/nvls.cc)

#### 3.2 一个 NVLS domain 的 group 和三个逻辑分区

当前 master 的 `ncclNvlsSetup()` 有一条非常关键的注释：**一个 NVLS domain 建立一个共享 MC group，credit、data、UB 是同一个 group 内的不同 slice**。代码先计算：

```text
creditSize = nChannels × 2 × 64 × nHeads
buffSize   = nvlsChunkSize × NCCL_STEPS
dataSize   = nChannels × 2 × buffSize × nHeads
ubSize     = ncclNvlsUbSize(comm)
```

然后提交三个 request：`{creditSize, dataSize, ubSize}`，由 `ncclMcGroupBuildPartitions()` 分配到 `creditPartition`、`dataPartition` 和可选的 `ubPartition`。这三个 partition 不是三个 NVSwitch slot；它们是同一 MC group 内的地址/内存区间。[NCCL `nvls.cc` group 构建](https://github.com/NVIDIA/nccl/blob/master/src/transport/nvls.cc)、[NCCL `transport.h` 的 `ncclNvlsSharedRes`](https://github.com/NVIDIA/nccl/blob/master/src/include/transport.h)

随后对每个 `head` 和 `channel` 建立两套方向：

```text
Reduce:    UC → MC       send[1] / recv[0]
Broadcast: MC → UC       recv[1] / send[0]
```

也就是说，NCCL 的 reduce 和 broadcast 在软件连接上有两个 half，但它们挂在同一个 MC group、同一个 NVLS domain，并共享 switch 侧 group/crossbar/credit 资源。它们不是两个自动隔离的 SHARP 实例。[NCCL `nvls.cc` channel/head 布局](https://github.com/NVIDIA/nccl/blob/master/src/transport/nvls.cc)

#### 3.3 data object 的绑定通常延后到第一次需要时

`ncclNvlsBufferSetup()` 根据 `nHeads` 和 `nChannels` 计算 `nvlsTotalSize`，再为当前 GPU 分配 unicast physical memory，映射成 UC 地址，并通过 `ncclMcPartitionBindMem()` 绑定到前面保留的 data partition。绑定前源码显式调用 intra-node barrier，绑定完成后还要等所有 GPU 的 MC mapping 完成，避免某些 rank 先访问未完成映射的地址。

这解释了 NCCL 日志中“初始化 communicator 成功，但第一次选中 NVLS 算法时才出现 multicast object/buffer 申请”的现象。旧版 issue #2360 记录了 credit object 在 communicator 初始化时建立、data object 在第一次 NVLS collective 时惰性建立；当前 master 已把 credit/data 组织成单个共享 MC group 的 slice，版本差异要特别注意。[NCCL issue #2360](https://github.com/NVIDIA/nccl/issues/2360)

#### 3.4 communicator 之间何时复用 group

`ncclNvlsSetup(comm, parent)` 只有在以下条件满足时才复用 parent 的 `nvlsResources`：

```text
parent != nullptr
parent NVLS enabled
parent->shareResources == true
parent->localRanks == comm->localRanks
```

复用时会继承 parent 的 `nChannels`、chunk size 和 tree chunk size，并增加 resource reference count；否则创建新的 shared resource 和新的 MC group。释放时只有 reference count 归零才 unbind/destroy。[NCCL `nvls.cc` resource sharing/free](https://github.com/NVIDIA/nccl/blob/master/src/transport/nvls.cc)

这对 AG/RS overlap 很关键：两个完全独立初始化的 ProcessGroup/NCCL communicator **不能假设**会自动复用同一个 group，也不能假设一定得到两个独立硬件 datapath。它们可能多占一个 slot，也可能仍争用同一 NVSwitch 物理资源。创建许多子 communicator 是 NCCL issue #2077 中 slot 耗尽的直接风险之一。[NCCL issue #2077](https://github.com/NVIDIA/nccl/issues/2077)

### 4. TokenWeave 的 group 设置和实际执行语义

TokenWeave 没有直接调用 `cuMulticastCreate`，而是复用 PyTorch symmetric memory：

```python
staging = symm_mem.empty((CHUNK_SIZE, hidden_size), dtype=..., device="cuda")
hdl = symm_mem.rendezvous(staging, group=get_device_group())
```

`rendezvous()` 是高层封装，内部负责形成同一 rank group 的 symmetric/multicast handle，并暴露 `hdl.multicast_ptr`、`hdl.signal_pad_ptrs_dev`、`hdl.rank` 和 `hdl.world_size`。TokenWeave 代码本身没有设置 slot ID、SHARP group ID、scratch SRAM 大小或 resource quota。

源码中的操作是：

```text
multimem_reduce_scatter_kernel:
  barrier(all ranks)
  ptr = multicast_ptr + rank * numel_per_rank
  multimem.ld_reduce(..., ptr)       # 每个 rank 取回自己负责 shard
  local store
  barrier(all ranks)

multimem_all_gather_kernel:
  barrier(all ranks)
  local load
  ptr = multicast_ptr + rank * numel_per_rank
  multimem.st(..., ptr)               # 写入所有成员副本
  barrier(all ranks)
```

其中 `RANK * numel_per_rank` 是对象内的 **offset**，不是新建 multicast group。只要 buffer 在创建时足够大，AG/RS、不同 token split 和不同 chunk 都可以放在同一 object 的不同 offset 中。

TokenWeave 的实际 overlap 路径还有两个容易误读的地方：

- [triton_comm.py](/Users/zhangbochun/Desktop/AI-Infra/Megatron/tokenweave/vllm/distributed/triton_comm/triton_comm.py:329) 中的 `multimem_all_gather_async()` 调用的是 `cudaMemcpyAsync(DeviceToDevice)`，目的地址是 `multicast_ptr + offset + rank * nbytes_per_rank`；它是 copy engine 写 multicast object 的路径，并不是 PTX `multimem.st`。
- [tokenweave_fused_kernels.cu](/Users/zhangbochun/Desktop/AI-Infra/Megatron/tokenweave/csrc/tokenweave_fused_kernels.cu:129) 的 fused kernel 在同一 kernel 内按 `ld_reduce → residual/RMSNorm → multimem.st` 顺序执行；[tokenweave_utils.py](/Users/zhangbochun/Desktop/AI-Infra/Megatron/tokenweave/vllm/model_executor/models/tokenweave_utils.py:214) 的第一层路径是 `RS → barrier(channel=7) → async AG copy → barrier(channel=9)`。

所以 TokenWeave 的 group 配置原则是：**一个 device group、一个 rendezvous handle、一个 multicast object，多种 offset；用软件 barrier 保证阶段顺序**。这能实现 RS/本地计算/AG 的粗粒度流水，但不能证明两个独立操作在 NVSwitch 内部同时满速。

### 5. 如果要让 FSDP3 的 AG 和 RS 使用独立 multicast group

若目标是主动测试两个独立 group 的资源争用，可以按下面方式设计；这不是现有 TokenWeave 的实现：

```python
# all ranks execute these calls in the same order
ag_pg = dist.new_group(ranks=dp_ranks)
rs_pg = dist.new_group(ranks=dp_ranks)

ag_storage = symm_mem.empty(ag_shape, dtype=dtype, device="cuda")
rs_storage = symm_mem.empty(rs_shape, dtype=dtype, device="cuda")

ag_hdl = symm_mem.rendezvous(ag_storage, group=ag_pg)
rs_hdl = symm_mem.rendezvous(rs_storage, group=rs_pg)
```

随后：

1. AG kernel 只能访问 `ag_hdl.multicast_ptr`，RS kernel 只能访问 `rs_hdl.multicast_ptr`；不要把两个 handle 的 `signal_pad_ptrs_dev`、offset 或 barrier channel 混用。
2. 两个 group 的 rank 集合、创建顺序和 collective 顺序必须在所有 process 上一致；CUDA multicast API 要求成员属于同一个 clique，PyTorch symmetric memory 还要求 group 内的参与者一致。
3. 每个 group 都可能消耗一个新的 multicast resource。不能因为两个 group 的成员完全相同，就假设它们自动共享 hardware group；共享需要由上层 runtime/NCCL 明确实现。
4. 若只需要不同 buffer/offset，而不需要两个 group 的独立生命周期，应优先使用一个 handle 和分离 offset，减少 slot 压力；但同一 symmetric-memory group 的 stream/barrier 约束仍然必须遵守。
5. 先用单独 AG、单独 RS、同 group 顺序执行和双 group 并发四组测试，观察 `cuMulticastBindMem`/`CUDA_ERROR_MULTICAST_RESOURCE_FULL`、NVLink TX/RX、NVLS kernel 时间以及 `T_both`。双 group 只解决 communicator/group 排队问题，不保证 SHARP ALU、scratch SRAM 或 crossbar 独立。

对 FSDP3 来说，推荐的实验路径是：创建相同 rank 集合的独立 AG/RS process group → 为每个 group 配置独立 symmetric buffer/handle → 固定创建顺序 → 再测 overlap。若生产目标是节省 multicast slot，则使用一个 shared NVLS resource 和 offset 分区更合理；若目标是证明硬件并行，则必须接受额外 group 的资源成本，并以运行时日志和性能 counter 验证。

### 6. 配置检查清单

在目标机器上可以按下列顺序确认 group 是否正确：

```text
1. 确认 driver、CUDA、Fabric Manager、NVLSM 版本匹配。
2. 确认所有 GPU 位于同一 NVLink domain/partition/clique。
3. 开启 NCCL_DEBUG=INFO，记录 NVLS support、nHeads、nChannels、chunkSize、group 创建和 bind 日志。
4. 确认每个 multicast object 的 numDevices 等于实际参与 rank 数。
5. 确认所有成员先 AddDevice，再 BindMem，再 Map/SetAccess。
6. 检查 size、offset、地址满足 multicast granularity；避免每个 chunk 单独创建 object。
7. 正确初始化 reduce identity，并让 ld_reduce 的 executor 属于 team。
8. 释放时先同步所有 GPU，再 unbind/unmap/release；失败 handle 不复用。
9. 对多个 PG 统计 group 数量，避免 communicator 创建数量接近 Fabric Manager 的 slot 上限。
```

最终要点是：**multicast group 是 NVSwitch/driver 资源上下文，multicast object 是 GPU 侧地址和物理副本，NCCL partition 是该 object 内的 GPU buffer 布局，TokenWeave offset/barrier 是应用级调度。** 配置 AG/RS 时首先决定是“一个 group 多个 offset”还是“多个 group 两套 handle”；这个选择决定 slot 消耗，但不决定 SHARP ALU 的实际吞吐，后者仍需在目标 NVSwitch 代际上测量。

## 7. 其他利用 NVLS/SHARP 的项目

公开实现大致分成三层：底层 CUDA multicast object；由 NCCL、NVSHMEM、PyTorch Symmetric Memory 管理的库级抽象；以及 Megatron、TensorRT-LLM、FlashInfer、MoonEP、vLLM 等面向模型或通信算子的封装。它们都使用同一类 NVSwitch 能力，但“group”的含义不同，不能把 API 名称直接等同为一个硬件 multicast slot。

| 项目 | NVLS/SHARP 用法 | multicast group 的管理方式 | 对本问题的启示 |
|---|---|---|---|
| [NVIDIA multi-gpu-programming-models](https://github.com/NVIDIA/multi-gpu-programming-models/tree/master/multi_node_p2p) | `-use_mc_red` 用 multicast object 和 `multimem` 做 device-side barrier/allreduce | 应用直接调用 `cuMulticastCreate`、`cuMulticastAddDevice`、`cuMulticastBindMem`、map/access；root 导出 fabric handle，其他 rank import | 最清楚的低层参考实现；一个 object 可以承载一个团队的多 GPU 副本，但 object 大小是 GPU 侧映射大小，不是 NVSwitch SRAM |
| [NCCL multimem kernel-fusion 示例](https://github.com/NVIDIA/nccl/blob/master/docs/examples/07_kernel_fusion/README.md#02-multimem) | 单 kernel 内做 ReduceScatter → 本地 RMSNorm → AllGather | 通过 `lsaMultimem=true` 为 device communicator 请求 multimem；kernel 获取 LSA multimem pointer 和 barrier 状态 | 证明 SHARP load/reduce、GPU 计算和 multicast store 可以在一个协议内融合；它不是两个独立硬件 group 并发的证据 |
| [NVSHMEM](https://docs.nvidia.com/nvshmem/api/latest/gen/api/setup.html#nvshmemx-mc-ptr) | NVLS ReduceScatter/Allgather/Allreduce；`nvshmemx_mc_ptr(team, dest)` 返回 multimem 地址 | 应用只提供 symmetric pointer 和 NVSHMEM team；库内部创建/复用 multicast 资源，团队成员必须一致 | team 是上层参与者集合；复制 team、并发 team 或不同 locality domain 可能消耗额外 NVLS 资源 |
| [PyTorch Symmetric Memory](https://docs.pytorch.org/docs/main/symmetric_memory.html) / NCCL SymK | `symm_mem.rendezvous` 后，标准 all-reduce/all-gather/reduce-scatter 可选择 multimem/NVLS kernel | `rendezvous(tensor, group)` 形成 symmetric handle；同一 group 的 signal pad/collective 要在同一 CUDA stream 上使用；每个 group 还有独立的 CFT endpoint 集 | PyTorch 不暴露 slot ID；AG/RS 是否共享硬件资源取决于 group、NCCL resource sharing 和 buffer 生命周期 |
| [Megatron Core NVLSAllGatherVDispatcher](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/token_dispatcher_inference.py) | MoE inference 的 variable-count NVLS AllGather-V/ReduceScatter-V，支持 fused RS+add+norm+AG | 按 EP process group 建立多个 symmetric buffer（AGV、RSV、metadata 等）；后端负责 rendezvous | 这是 NVLS 在不等 token 数量下的实际模型集成案例；“多个 buffer”不代表一定创建多个硬件 group，需看 symmetric-memory/NCCL 后端是否复用 |
| [TensorRT-LLM `ipcNvlsMemory`](https://nvidia.github.io/TensorRT-LLM/_cpp_gen/runtime.html#ipcNvlsAllocate) | 为指定 ranks 分配 NVLS multicast memory，供 TP/allreduce 使用 | `ipcNvlsAllocate(size, ranks, groupComm)` 创建与 TP group 对齐的分配；`ipcNvlsFabricUsable` 额外做 fabric/IMEX 探测 | 静态 `MULTICAST_SUPPORTED` 不足以保证运行；生产代码需要 capability probe 和 fallback |
| [FlashInfer MNNVL](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/comm/mnnvl.py) | Python/CUDA Driver API 实现 MNNVL multicast buffer 和 multimem allreduce | 每个 communicator 先分配 UC physical memory，再用 fabric handle 建 multicast object；持久化 signal pad、UC/MC VA 和 rank pointer 数组 | 适合研究/自定义 fused kernel；创建、barrier、unbind 的生命周期必须由上层严格管理 |
| [MoonEP](https://github.com/MoonshotAI/MoonEP/blob/master/csrc/nvl_shared_buffer.cuh) | EP shared buffer 用 `multimem.st` 一次 fan-out | 每个 EP group 创建一个持久化 MC object，把已有的每-rank NVL chunk bind 到同一 object；额外主要是 VA mapping | 说明 multicast 可以复用已有显存，不需要为 MC 再复制一份物理 payload；对象应按 buffer 生命周期缓存 |
| [vLLM custom allreduce](https://docs.vllm.ai/en/latest/api/vllm/distributed/device_communicators/custom_all_reduce/) / [SGLang](https://github.com/sgl-project/sglang/issues/41882) | 使用 PyTorch symmetric memory 或 MNNVL，无法使用时回退 NCCL | group 通常是 TP/DP process group；应用层不直接分配 slot | 下游框架的关键问题是 topology/capability 检查和 fallback，而不是重新实现 NVSwitch 控制面 |

这组项目给出两个稳定结论。第一，生产实现通常把 multicast object 做成长生命周期资源，并在 object 内按 offset、channel、buffer 或 signal pad 分区；很少为每个 collective chunk 重新创建 object。第二，使用者几乎都把 NVLS 作为一个单 NVLink clique 内的能力：参与 rank 必须能加入同一个 multicast team，跨 clique、跨不兼容 partition 或只满足静态属性但 fabric 未初始化时，应回退到 NCCL/IB 路径。

## 8. multicast group 的准确分层

### 8.1 CUDA multicast object：地址和物理副本

CUDA VMM 的 multicast object 是最底层的地址对象。官方编程指南定义，一个包含 `N` 个 GPU 的 multicast team 有 `N` 个本地物理 replica；每个进程先创建或导入同一个 opaque handle，再执行以下顺序：

```text
cuMulticastCreate(numDevices=N, size, handleType)
  -> share/import handle
  -> every rank cuMulticastAddDevice()
  -> barrier
  -> each rank cuMemCreate() local physical allocation
  -> cuMulticastBindMem() at the same mcOffset
  -> reserve/map multicast VA and set access
  -> multimem.st / multimem.ld_reduce / multimem.red
```

`cuMulticastAddDevice` 对一个 object 是永久成员关系；所有 device 完成 AddDevice 后才能 bind 或 map，且成员必须属于同一个 clique。[CUDA Driver API multicast](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__MULTICAST.html) 还要求 offset、size 和地址满足 multicast granularity。这个 API 层没有“slot ID”参数；slot 由驱动、Fabric Manager 和 NVSwitch 资源管理器分配。

### 8.2 NVSwitch Multicast Slot：有限的硬件资源配额

Fabric Manager 文档把 NVSwitch 中可用于 multicast/reduction 的有限资源称为 **Multicast Slots**。[官方资源分配说明](https://docs.nvidia.com/hgx-platforms/fabric-manager-user-guide/index.html#fabric-manager-multicast-nvlink-sharp-resource-allocation) 对 H100 及以后明确说明：H100 partition 通常拥有独立的完整 slot 集合；B200/B300 的 partition 共享 slot 集合，并按 partition 大小给出上限（8-GPU 全部、4-GPU 不超过 50%、2-GPU 不超过 25%、1-GPU 为 0）。

因此，CUDA object 的 `size`、NCCL 的 `dataSize`、PyTorch tensor 的字节数都不能直接换算为 slot 数量。slot 更接近“并行 multicast context/team 的硬件配额”，而不是 payload SRAM 容量。NCCL issue [#2077](https://github.com/NVIDIA/nccl/issues/2077) 展示了 H200 上 128 个 slot 被大量 subcommunicator 消耗后，新的 group/bind 失败；这说明 slot 会受到 communicator/group 生命周期影响，但该 issue 不能推出所有 NVSwitch 代际都固定为 128。

GB200 的系统级上限有更直接的官方依据：[GB200 NVL Partition User Guide v1.2，第 9 章](https://docs.nvidia.com/multi-node-nvlink-systems/partition-guide-v1-2.pdf#page=56) 规定总共 1,024 multicast teams，由所有 partitions 共享；管理员通过 `multicastGroupsLimit` 预留配额，取值须为 0 或 4 的倍数。第 9.3 节还说明可以通过 `GetPartitionInfoList.allocatedMulticastGroups` 查询当前已用 groups。这是 group 容量及分区管理规格，不是每 ASIC SRAM 容量，也不是单个 collective 满带宽所需 group 数。

### 8.3 NCCL MC group：一个 NVLS domain 内的共享对象

当前 NCCL NVLS transport 通常为一个 NVLS domain 建立共享 MC group，然后在该 group 内切分 software partitions：`credit`、`data` 和可选 `UB`。每个 channel/head 再把 UC→MC 的 reduce slice 与 MC→UC 的 broadcast slice 放到不同 offset。源码见 [NCCL `src/transport/nvls.cc`](https://github.com/NVIDIA/nccl/blob/master/src/transport/nvls.cc)。

所以 NCCL 中的以下对象不能混为一谈：

```text
NVSwitch hardware multicast slot     = 有限硬件 team/context 配额
CUDA multicast object / MC group     = 一个可绑定多个 offset 的地址对象
NCCL credit/data/UB partition        = object 内的 GPU buffer 布局
nChannels / nHeads / NCCL steps      = 软件并发与流水参数
LSA barrier slots / nSlots           = GPU 侧协议状态和 payload slot
```

当 parent communicator 开启 resource sharing 且 local rank 集合匹配时，NCCL 可以复用 NVLS shared resources；否则会创建新的 group。NCCL 的 reuse 逻辑因此是决定 slot 压力的关键之一，而不是 AG/RS 的 tensor size 本身。[NCCL `transport.h`](https://github.com/NVIDIA/nccl/blob/master/src/include/transport.h) 中的 `ncclNvlsSharedRes` 也把 `mcGroup` 与 credit/data/UB partition 一起保存，反映了这一层次关系。

### 8.4 PyTorch、NVSHMEM 与模型框架的 group

PyTorch `ProcessGroup`、NVSHMEM `team`、Megatron EP/TP group 和 TensorRT-LLM `groupComm` 都是**参与者集合和控制面抽象**。它们可能各自拥有一个或多个 CUDA multicast object，也可能通过 NCCL 复用同一个 object。不能仅从“创建了两个 process group”断言硬件一定有两个独立 slot；反过来，如果库无法识别两个 group 可共享，也不能假设它们会自动复用。

对 AG/RS overlap，推荐明确记录四层 ID：

```text
logical group:       AG/RS 的 rank 集合和 collective 顺序
CUDA object:         multicast handle、numDevices、size、mcOffset
NCCL resource:       shared MC group、credit/data/UB partition、channels/heads
hardware resource:   Fabric Manager 分配的 multicast slot、SHARP datapath/VC
```

这样才能解释“两个 operation 是否真的并发”：它们可以使用不同 logical group，却仍共享同一 CUDA object、同一 NCCL resource，甚至同一 SHARP controller/ALU；也可以因独立 object 而增加 slot 压力，却仍在 NVSwitch crossbar 或归约单元上竞争。

## 9. 对 FSDP3/TokenWeave 实验的直接建议

1. 先分别运行 AG、RS、同一 group 顺序执行、两个独立 group 并发四组基准；同时记录 NCCL `NVLS Creating Multicast group` 日志、`cuMulticastBindMem` 错误、NVLink TX/RX 和 kernel 时间。
2. 对照 [NCCL 对称内存示例](https://github.com/NVIDIA/nccl/tree/master/docs/examples/05_symmetric_memory)，让 AG/RS buffer 在 communicator 初始化阶段注册并复用；不要在每一层或每个 chunk 创建新的 process group/object。
3. 若需要证明硬件 group 并行，必须验证两个 handle 的 `multicast_ptr`、group 生命周期和 NCCL resource 是否真的不同；仅仅把 AG、RS 放到两个 CUDA stream 不足以证明 NVSwitch 内部独立执行。
4. 若目标是减少 slot 压力，优先采用一个 symmetric buffer/object 加 offset 分区；若目标是研究资源隔离，再使用两个 group/object，并将 `CUDA_ERROR_MULTICAST_RESOURCE_FULL` 作为可观测的资源上限信号。
5. 在 GB200/B200 等目标平台上分别测试 32-rank、64-rank 和多 DP group；公开资料没有给出“参与 rank 数量到需要多少 slot”的公式。rank 增长通常增加 payload、channel/partition 和端点流量，但不必然线性增加 hardware slot；是否能打满出口带宽应由 message size、nChannels、SHARP ALU、scratch/credit 水位和 topology counter 实测。

## 10. 官方手册与资料边界

NVIDIA 有正式的编程、管理和系统手册。以下公开手册应优先于项目 issue，用于确定资源限制和 API 契约：

| 官方资料 | 直接回答的内容 | 没有由该资料确定的内容 |
|---|---|---|
| [GB200 NVL Partition User Guide v1.2，第 9 章](https://docs.nvidia.com/multi-node-nvlink-systems/partition-guide-v1-2.pdf#page=56) | GB200 的 1,024 全局 multicast teams、分区预留上限、已用 group 查询 | 单 ASIC slot 映射、scratch SRAM 字节容量、满带宽所需 slot 数 |
| [Mission Control — NVLink Partition Management](https://docs.nvidia.com/mission-control/docs/systems-administration-guide/2.2.0/nvlink-partition-management.html#multi-cast-support) | GB200/GB300 的 team 上限及 `mcast-limit` CLI；默认分区与多个租户分区配置 | CUDA allocation 与硬件内部数据通路的精确对应 |
| [Fabric Manager User Guide](https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/index.html#fabric-manager-multicast-nvlink-sharp-resource-allocation) | HGX/DGX NVSwitch 的 multicast slots 和 partition 配额；FM/NVLSM 管理 | 不同平台的 slot 数不能只按 GPU 世代相互外推 |
| [NVOS User Manual for NVLink Switches v25.02.4282](https://docs.nvidia.com/networking/display/nvidia-nvos-user-manual-for-nvlink-switches-v25-02-4282.pdf) | 交换机管理、NVLink 接口、计数器、partition、固件及相关系统手册索引 | 管理命令不能据此理解为任意 SHARP ALU 或 SRAM 编程接口 |
| [CUDA Driver API — Multicast Object Management](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__MULTICAST.html) 与 [CUDA Programming Guide — Multicast Memory Sharing](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html#multicast-memory-sharing) | team/object 创建、成员、显存绑定、粒度、映射及资源错误 | 内部缓存组织和控制器仲裁策略 |
| [NVIDIA Hot Chips 2022 NVSwitch 架构演讲，第 14 页](https://hc34.hotchips.org/assets/program/conference/day2/Network%20and%20Switches/NVSwitch%20HotChips%202022%20r5.pdf#page=14) | Hopper/NVLink4 的 Controller、ALU、Scratch SRAM 方框图；最多并行管理 128 SHARP groups | 没有标注 SRAM 大小、每组配额或 AG/RS 同时满速保证；这是官方架构演讲，并非完整硬件编程手册 |

此外，NVOS 手册的 References 表明确列出以下 NVOnline 文档，足以确认这些系统手册存在：

[NVIDIA NVLink NVOS 官方文档入口](https://networking-docs.nvidia.com/software/switch-software/nvidiatab-nvos-nvlink) 也明确说明，NVLink NVOS Software User Manual 随 NVLink software bundle 通过 NVOnline 提供。NVOnline 中的具体访问权限及文档内容需以账户授权为准。

| 文档名称 | NVOnline Document ID |
|---|---|
| NVIDIA NVLink GB200 Switch Systems User Manual | 1115337 |
| GB200 NVL Technical Overview | 1114121 |
| GB200 NVL72 Rack System Specification and Integration Guide | 1117886 |

出处：[NVOS v25.02.4282，§3.3.2.1 References，PDF 第 52 页](https://docs.nvidia.com/networking/display/nvidia-nvos-user-manual-for-nvlink-switches-v25-02-4282.pdf#page=52)。本次未取得这些 NVOnline 文档的全文，不能断言其是否包含 SHARP scratch SRAM 或内部调度规格。需要向 NVIDIA 查询时，应提供上述文档名称/ID，并明确索取 NVLink5 NVSwitch 的 SHARP 微架构和性能资源规格。

同时，InfiniBand/Quantum 的 [SHARP Deployment Guide](https://docs.nvidia.com/networking/display/sharpv3110/Using%2BNVIDIA%2BSHARP%2Bwith%2BNVIDIA%2BNCCL) 是另一套正式资料，其 Aggregation Manager、SAT tree、buffer 规格不能直接用于 NVLink/NVSwitch。GB200 分区配额的 suggested sizing 也只是容量预留建议，不能解释为 32/64-rank reduce-scatter 打满带宽的资源需求公式。

## 11. team、group、slot 与独立计算资源

**不能把一个 multicast slot 当成一套独占的 SHARP 计算单元。** 公开资料明确存在有限 multicast 资源，并展示了 SHARP Controller、ALU、Scratch SRAM，但没有公布 slot 到 ALU、SRAM bank 或计算带宽的对应关系。前文的“group/context 配额”是资源规划层面的解释，不是对硬件表项结构或物理一一映射的断言。

| 名词 | 所在层次与含义 | 不应由此推导的结论 |
|---|---|---|
| CUDA multicast team | 参与同一个 multicast object 的 GPU 成员集合；例如 `{GPU0,GPU1,GPU2,GPU3}` | 4 个成员需要 4 个 slot；相同成员集合的新 object 必然自动复用资源 |
| CUDA multicast object | 由 handle 标识的地址对象；各 GPU 绑定本地物理副本，映射为 multicast VA | object 的 `size` 是交换机 SRAM 容量；一份导入 handle 就增加一份独立 object |
| multicast group / SHARP group | 必须结合出处：NCCL 的 MC group 通常指 multicast 资源封装；架构资料中的 SHARP group 是 controller 管理的组 | 不同文档中的 group 必然是同一个 ID 命名空间或同一物理实体 |
| multicast slot | Fabric Manager 对有限 NVSwitch multicast resources 的称呼，可在分区管理中限制可用数量 | 一个 slot 是一个 ALU、一个独占 scratch SRAM bank、一个包缓存，或一个 NCCL channel |
| NVLink partition | 管理员设置的硬件隔离域，并为其中应用预留 multicast-team 容量 | 为分区预留 512 teams 就预留了一半 SHARP 算力 |
| NCCL channel / credit-data-UB partition / 软件 slot | 通信协议的并行通道、GPU buffer 分区与同步状态 | 每个软件条目对应一个交换机硬件 slot |
| CFT multicast logical endpoint | CUDA 提供的另一种面向通信资源的寻址与完成通知抽象 | endpoint 数、VMM object 数与物理 multicast slot 数相等 |

CUDA team/object 的依据见 [CUDA Multicast Memory Sharing](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html#multicast-memory-sharing)；slot 的正式用词见 [Fabric Manager 资源分配章节](https://docs.nvidia.com/hgx-platforms/fabric-manager-user-guide/index.html#fabric-manager-multicast-nvlink-sharp-resource-allocation)。两者并未规定“一个 CUDA object 恰好消耗一个单 ASIC 物理 slot”。尤其不能把跨进程 import 同一 object、对同一 object 多次映射 VA、多个独立 object、以及库内部共享 object 当作同一种行为。

### 独立身份、独立状态和独占执行资源是不同性质

不同 group 的成员、绑定和请求必须在逻辑上可区分；这不说明每个 group 的协议状态、部分和、等待队列都固定占据独立物理 SRAM。即使某些状态按 group 保存，也不说明计算 ALU 和 scratch 读写端口按 group 独占。固定分区、共享池、动态分配、分 bank 或时间复用等实现均不能仅凭 block diagram 确定。

[Hopper NVSwitch 架构图，第 14 页](https://hc34.hotchips.org/assets/program/conference/day2/Network%20and%20Switches/NVSwitch%20HotChips%202022%20r5.pdf#page=14) 说明 controller 可并行管理最多 128 SHARP groups，并单独画出 ALU、Scratch SRAM 与 crossbar；它没有声称有 128 套 ALU/SRAM。类似地，GB200 的 1,024 teams 是域级资源容量，不代表 1,024 条独立满速归约流水线。

所以需要分别判断三件事：

1. **容量**：还能创建/保留多少 multicast teams，是否已耗尽分区配额。
2. **进展**：多个 group 的请求能否同时挂起和推进，是否受软件 stream/barrier 或队头阻塞限制。
3. **吞吐**：多个操作的总速率是否受相同 ALU、SRAM 端口、crossbar、NVLink 端口或 credit 限制。

容量隔离不能作为吞吐隔离的证据；支持并发推进也不能作为每个 group 享有独立峰值带宽的证据。工程建模应将独占算力视为未得到保证，是否存在专用子资源仍需更细的硬件资料或实验确认。

### 对 AG/RS overlap 的含义

AG 使用 group A、RS 使用 group B，可以在软件协议设计正确时分离成员/缓冲区/同步状态，但不能据此判定 A 和 B 拥有独立计算资源。`multimem.st` 不要求做输入间的加法归约，仍需要 multicast 的复制、路由和传输；`multimem.ld_reduce` 还需要归约执行。因此二者有重叠空间，也可能竞争公共路径，具体比例不能从 slot 数计算。

若单独执行耗时为 `T_AG`、`T_RS`，两个独立且已就绪操作的理想完全重叠耗时是 `max(T_AG,T_RS)`。实测 `T_both` 才能判断效果；它同时反映硬件竞争和软件同步，不能仅凭一次性能下降归因于共享 ALU。一个支持 64-rank 的 team 并不因参与者是 64 而自动需要 64 slots，32→64 ranks 也不推出 slot 需求翻倍。
