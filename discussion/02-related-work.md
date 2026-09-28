# RL 时间平衡与训练侧 Offload 相关工作

本文整理截至 **2026-09-21** 检索到的论文和正式技术报告。相关工作分为三条主线：

1. **RL 系统侧的时间平衡**：缩短 rollout 的长尾、重叠 rollout 与 training，或在两个阶段之间动态调度 GPU；
2. **训练侧的状态卸载**：将 activation、model weight、gradient、optimizer state 等从 GPU 移到 CPU、NVMe/SSD 或其他分层介质。
3. **非规则 GPU 资源的训练**：通过非对称 shard、异构 pipeline 或弹性 pipeline 使用不满足规则 mesh 整除关系的 GPU 集合。

前两类工作分别直接讨论 rollout/training 的流水线效率和训练显存墙，通常不会同时解决另一侧的问题；第三类工作则关注如何使用不满足规则整除关系的 GPU 集合。对于本项目，较有价值的方向是将三类方法组合起来：先用时间模型确定训练侧可以占用多少 GPU，再用非对称并行或分层 offload 将训练模型放入这部分 GPU。

## 论文总表

| 方向 | 论文 | 介质/资源组织 | 主要卸载或调度对象 | 方案设计 | 对本项目的启示与代价 |
| --- | --- | --- | --- | --- | --- |
| RL 时间平衡 | [ReaL: Efficient RLHF Training of Large Language Models with Parameter Reallocation](https://arxiv.org/abs/2406.14088)（MLSys 2025） | 训练集群 GPU；不同 workload 可使用不同并行布局 | actor、reference、reward、rollout/training workload 的参数与并行资源 | 将 RLHF 执行表示为带资源和并行策略的数据流图；用轻量 cost estimator 搜索 execution plan；运行时重新分配参数和并行策略，而不是所有阶段固定使用同一套 TP/PP。 | 直接对应“训练侧需要很大 PP、但并不持续需要同等算力”的问题。代价是参数重分布和布局切换有通信开销，需要准确的时间模型。 |
| RL 时间平衡 | [HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256)（EuroSys 2025） | colocated 或 disaggregated GPU；训练/生成间切换 | actor 模型参数布局 | 通过 hybrid controller 编排 RLHF dataflow；设计 **3D-HybridEngine**，在 training 和 generation 阶段对 actor 参数进行 reshard，减少重复副本和通信。 | 可作为“同一批 GPU 在 rollout 和 training 之间复用”的基础。它主要优化阶段切换，不会自动解决 2.4T 模型的状态容量问题。 |
| RL 时间平衡 | [StreamRL: Scalable, Heterogeneous, and Elastic RL for LLMs with Disaggregated Stream Generation](https://arxiv.org/abs/2504.15930)（2025） | 异构、可弹性伸缩的 disaggregated GPU 池 | rollout 请求流、reward、training stream | 打破传统“整批 rollout 完成后再训练”的阶段边界；以 stream generation 持续产出样本并与 training 重叠；用 output-length ranker 识别长尾样本，再进行 skewness-aware dispatch/scheduling。 | 适合把 15 分钟 rollout 看成持续数据流，而不是一次性 barrier；需要处理异步样本的版本和 staleness。 |
| RL 时间平衡 | [AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning](https://arxiv.org/abs/2505.24298)（2025） | rollout worker 与 trainer 解耦的异步 GPU 池 | rollout batch、训练更新、样本 staleness | rollout worker 持续生成，trainer 收集到 batch 即更新；通过 workload balance 控制 staleness；采用 staleness-enhanced/decoupled PPO、interruptible generation 和 dynamic batching。 | 与“训练侧可以变慢，只要不慢过 rollout”最接近：可通过 worker 比例把两侧服务率调到接近。代价是 PPO 目标、版本同步和收敛稳定性更复杂。 |
| RL 时间平衡 | [RollPacker: Mitigating Long-Tail Rollouts for Fast, Synchronous RL Post-Training](https://arxiv.org/abs/2509.21009)（NSDI 2026） | 同步 RL；rollout 资源可弹性调整 | 长响应、rollout round、reward 和 training stream | **tail batching** 将长响应集中到少数 long rounds，使多数 short rounds 不被长尾拖慢；配合 elastic rollout parallelism、动态 reward 调度和 stream trainer。 | 如果 15 分钟主要由少量超长 response 造成，这种方法可先缩短 rollout 端长尾。它保持同步语义，但不直接减少训练模型的显存需求。 |
| RL 时间平衡 | [APRIL: Active Partial Rollouts in Reinforcement Learning to Tame Long-Tail Generation](https://arxiv.org/abs/2509.18521)（2025） | rollout worker 池 | 未完成的 partial response | 预留多于目标数量的 rollout 请求；达到目标 response 数后提前结束当前轮；未完成 response 不丢弃，而是放入后续轮次继续生成。 | 可减少 rollout 阶段因少数长样本造成的 GPU 空闲；需要保证 partial response 的续接、优势估计和数据去重语义正确。 |
| RL 时间平衡 | [BiDiRL: Bidirectional Resource Scheduling for Disaggregated and Asynchronous RL Post-Training](https://arxiv.org/abs/2607.09207)（2026 预印本） | disaggregated rollout/training GPU 池；支持 hot-switch | 两个资源池之间的 GPU 配额 | 先用 time-performance model 做静态资源划分，使 rollout/training 粗粒度平衡；再用 hot-switch runtime 和 bidirectional scheduler 在运行时让瓶颈阶段临时借用另一阶段的空闲 GPU。 | 与本项目的目标最直接：训练完成过早时不必永久保留全部训练 GPU，训练变慢时也可临时借用 rollout 空闲资源。需要低开销的模型布局切换和严格的资源隔离。 |
| CPU：activation | [vDNN: Virtualized Deep Neural Networks for Scalable, Memory-Efficient Neural Network Design](https://arxiv.org/abs/1602.08124)（MICRO 2016） | GPU + CPU DRAM | activation/feature map | 运行时内存管理器把 GPU 和 CPU 内存虚拟化为统一容量；根据层的访问顺序将 activation 异步 swap out/in，并进行预取。 | 说明 activation 可以通过 CPU 扩容显存；适合长序列和大 batch。缺点是 PCIe/NVLink 带宽会成为瓶颈，且只解决 activation 不解决参数/optimizer state。 |
| CPU：activation | [SuperNeurons: Dynamic GPU Memory Management for Training Deep Neural Networks](https://arxiv.org/abs/1801.04380)（2018） | GPU + host memory | activation、临时 tensor | 结合 liveness analysis、unified tensor pool 和 cost-aware recomputation；按 tensor 生命周期选择保留、交换或重计算。 | 提供“offload 与 recompute 联合选择”的思路，可用来按 rollout 时间预算选择训练侧额外计算量；原工作面向通用 DNN，需重新适配 Transformer/PP。 |
| CPU：activation | [Capuchin: Tensor-based GPU Memory Management for Deep Learning](https://www.microsoft.com/en-us/research/?p=690597)（ASPLOS 2020） | GPU + CPU memory | activation tensor | 运行时跟踪 tensor access pattern；联合使用 tensor eviction/prefetch 和 recomputation，在规则访问模式下提前规划迁移。 | 适合做 layer/tensor 粒度的动态策略；可与 activation checkpointing 组合，但运行时规划和同步会增加复杂度。 |
| CPU：activation | [SwapAdvisor: Pushing Deep Learning Beyond the GPU Memory Limit via Smart Swapping](https://par.nsf.gov/servlets/purl/10191573)（ASPLOS 2020） | GPU + CPU memory | 任意 dataflow tensor | 对 operator scheduling、memory allocation、swap decision 三个维度联合优化，用 genetic algorithm 搜索 swap plan，并尽量重叠通信和计算。 | 比固定的 layerwise swap 更通用，可作为训练侧 offload planner 的参考；搜索成本和对稳定 dataflow 的依赖较高。 |
| CPU：model/gradient/optimizer | [ZeRO-Offload: Democratizing Billion-Scale Model Training](https://arxiv.org/abs/2101.06840)（ATC 2021） | GPU + CPU DRAM | gradient、optimizer state；参数主要留在 GPU | 基于 ZeRO 分片 optimizer state；把 gradient 和 optimizer computation 放到 CPU，使用高效 CPU Adam、延迟参数更新和最小化 GPU↔CPU 传输。 | 是降低训练 GPU 常驻状态的基础方案，尤其适合 optimizer state 占主导的场景；CPU Adam 和 PCIe 传输会拉长 step time。 |
| CPU：model/gradient/optimizer | [PatrickStar: Parallel Training of Pre-trained Models via Chunk-based Memory Management](https://arxiv.org/abs/2108.05818)（2021） | GPU + CPU heterogeneous memory | parameter、gradient、optimizer state | 将模型状态组织为固定大小 chunk；用 warm-up 统计动态决定 chunk 在 CPU/GPU 的位置；按计算顺序预取，并支持 hybrid Adam。 | 粒度从“整个模型”细化到 chunk，适合减少大 PP；需要 allocator、prefetch 和分布式通信协同。 |
| CPU：gradient/optimizer | [Ratel（LoHan 接收版本）](https://ieeexplore.ieee.org/abstract/document/11113169/)（ICDE 2025） | GPU + CPU memory（受限主存） | gradient、activation | 将 holistic offloading traffic 纳入优化目标；采用 active gradient offloading 和 traffic-aware activation swapping，在有限主存下联合安排数据搬运。 | 说明不能只优化某一类 tensor；应同时建模 gradient、activation 和 CPU↔GPU 链路，否则 offload 后可能因带宽竞争变慢。详见 [2025-ratel](related-work/2025-ratel/README.md)。 |
| CPU：model/gradient/optimizer | [ZenFlow: Enabling Stall-Free Offloading Training via Asynchronous Updates](https://arxiv.org/abs/2505.12242)（2025） | GPU + CPU memory | 重要/不重要参数的 gradient 与 update | 在 GPU 上原地更新重要参数；不重要梯度在 CPU 异步累积和更新；用轻量 gradient selection 利用重要梯度的空间/时间局部性，避免全局同步。 | 直接针对“GPU 等 CPU update”的空泡，适合把训练额外时间控制在 rollout 窗口内；需要研究异步 update 对 RL policy freshness 和收敛的影响。 |
| CPU：model/optimizer | [SuperOffload: Unleashing the Power of Large-Scale LLM Training on Superchips](https://arxiv.org/abs/2509.21271)（2025） | GH200：Hopper GPU + Grace CPU + NVLink-C2C | weight、optimizer computation、gradient transfer | 面向 superchip 设计 adaptive weight offloading、bucketization repartitioning、superchip-aware casting；用 speculation-then-validation、GPU/CPU optimizer partition 和 GraceAdam 隐藏 CPU update。 | 对 GB200/GH200 类紧耦合 GPU-CPU 很有参考价值；其收益依赖 NVLink-C2C、CPU 架构和专用 Adam，不能直接外推到 PCIe 集群。 |
| CPU：model/gradient/optimizer | [MegaTrain: Full Precision Training of 100B+ Parameter Large Language Models on a Single GPU](https://arxiv.org/abs/2604.05091)（2026 预印本） | GPU 作为 transient compute cache；host CPU memory 保存持久状态 | parameter、gradient、optimizer state | 将持久训练状态全部放在 host memory；按 layer 流式加载参数、流式卸载 gradient；用 pipelined double buffer 重叠预取/计算/卸载，以 stateless layer template 避免持久 autograd graph；optimizer 在 CPU 执行。 | 是“减少 GPU 数量而接受更长训练时间”最极端的实现范式；对 2.4T 规模，CPU 内存容量、互联带宽和总计算时间仍是主要限制。 |
| CPU/NVMe：model state | [ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://arxiv.org/abs/2104.07857)（2021） | GPU + CPU DRAM + NVMe | 分片 parameter、gradient、optimizer state；activation 主要卸载到 CPU | 在 ZeRO-3 分片基础上建立多级内存层次；用 memory-centric tiling 降低 operator working set；通过 Infinity engine 同时重叠 NVMe↔CPU、CPU↔GPU、GPU 计算和 CPU optimizer。 | 最接近“模型状态卸载到 CPU/NVMe 以降低 PP”的参考基线；NVMe 带宽和 I/O 调度决定可接受的训练 slowdown。 |
| NVMe：activation | [FlashNeuron: SSD-Enabled Large-Batch Training of Very Deep Neural Networks](https://www.usenix.org/conference/fast21/presentation/bae)（FAST 2021） | GPU + NVMe SSD，GPU 直连 SSD 数据路径 | activation/feature map | 通过 offloading scheduler 选择要写入 SSD 的中间数据，采用压缩格式；在 backward 前预取，并尽量不增加 forward evaluation 时间。 | 适合 activation memory 主导、希望增大 micro-batch 的场景；SSD 写带宽、寿命和恢复延迟需要纳入系统设计。 |
| NVMe：activation | [SSDTrain: An Activation Offloading Framework to SSDs for Faster Large Language Model Training](https://arxiv.org/abs/2408.10013)（2024） | GPU + CPU + NVMe SSD | activation | 采用 adaptive activation offloading；用 tensor deduplication 和 forwarding 减少 I/O；将 SSD transfer 与计算完全重叠。 | 面向 PyTorch、Megatron、DeepSpeed，适合作为 Transformer activation offload 的工程参考；不能单独解决 2.4T 参数和 optimizer state。 |
| NVMe + near-storage compute | [Smart-Infinity: Fast Large Language Model Training using Near-Storage Processing on a Real System](https://arxiv.org/abs/2403.06664)（2024） | GPU + CPU + NVMe/near-storage accelerator | parameter update、gradient | 用 near-storage accelerator 执行 SmartUpdate，把参数更新放到存储侧以减少往返流量；用 data-transfer handler 固定内存占用并重叠传输；在多设备时压缩 gradient。 | 当 NVMe 带宽成为瓶颈时，可通过“把计算移向数据”减少 I/O；需要额外硬件或 near-storage 能力，工程门槛高。 |
| GPU/CPU/SSD：MoE | [MoESys: A Distributed and Efficient Mixture-of-Experts Training and Inference System](https://arxiv.org/abs/2205.10034)（2022） | GPU、CPU memory、SSD hierarchical storage | sparse expert parameter/state | 训练侧采用 Elastic MoE、2D prefetch 和 fused communication，让层级存储中的 expert state 预取与计算并行；同时利用 MoE 稀疏性降低实际搬运量。 | 对 2.4T MoE 模型尤其相关：可按 expert 热度进行分层存储；需要处理 expert routing 带来的不可预测访问。 |
| DRAM/block device：分布式训练 | [SpeedLoader: An I/O Efficient Scheme for Heterogeneous and Distributed LLM Operation](https://proceedings.neurips.cc/paper_files/paper/2024/file/3d3a9e085540c65dd3e5731361f9320e-Abstract-Conference.html)（NeurIPS 2024） | host DRAM 和 block device；分布式 shard | model state 与 shard | 重新设计 heterogeneous hardware 上的 dataflow，减少 sharded model training 中不必要的 tensor communication，并以 I/O 路径为中心安排训练和推理。 | 可作为多节点训练侧 offload 的通信优化参考；重点不是某一种介质，而是减少跨层级数据移动。 |
| CXL memory：其他介质 | [Efficient Tensor Offloading for Large Deep-Learning Model Training based on Compute Express Link](https://sc24.supercomputing.org/proceedings/tech_paper/tech_paper_pages/pap287.html)（SC 2024） | GPU + CPU DRAM + CXL memory pool | tensor/parameter transfer | 用 CXL cache-coherent domain 扩展 accelerator memory；针对粗粒度迁移和未改变字节的冗余传输，设计 update coherence 以减少不必要搬运。 | CXL 可能比 NVMe 更适合作为“容量扩展层”，但收益取决于拓扑和协议；需要评估 GB200 的 CXL/NVLink 内存层级。 |

## 任意数量 GPU 与非对称并行

### 1. 规则 Megatron mesh 的整除约束

Megatron 的常规 TP/PP/CP/EP/DP 组合要求并行组大小和 world size 满足整除关系。对 MoE，attention 和 expert layer 可以使用不同但重叠的 mesh；一个更准确的最小设备数表达是：

```text
min_gpus = PP × max(TP × CP, EP × ETP)
DP       = world_size / (TP × PP × CP)
EDP      = world_size / (PP × EP × ETP)
```

这里不能把 `PP × TP × CP × EP × ETP` 当作设备数，因为 attention mesh 和 expert mesh 在同一个 PP stage 内复用 GPU。MoE Parallel Folding 正是通过这种 mesh 解耦突破传统的 `EP ≤ DP` 限制，但它仍然要求每个 mesh 的 DP/EDP 是整数，并没有把规则 rank grid 变成任意设备集合。[MoE Parallel Folding](https://arxiv.org/abs/2504.14960)

例如，在假设 `TP=PP=ETP=1` 时，`CP=32`、`EP=32`、`DP=EDP=2` 对应 64 张 GPU。将 world size 改为 72 后，`72/32=2.25`，现有通信组无法直接建立；保持这些并行度时，下一个规则规模通常是 96，而不是 72。CP 只沿 sequence/activation 维切分，不能单独解决参数、梯度和 optimizer state 的容量问题。

### 2. 论文中放松整除约束的方案

| 论文/系统 | 放松的约束 | 关键机制 | 报告结果与适用边界 |
| --- | --- | --- | --- |
| [HAP: SPMD DNN Training on Heterogeneous GPU Clusters with Automated Program Synthesis](https://arxiv.org/abs/2401.05965)（EuroSys 2024） | 不同 GPU 使用不同 tensor shard 比例；MoE expert 不要求简单 padding 到设备数的倍数 | 用程序合成搜索 distributed instruction，并将 shard ratio 优化写成线性规划；对不等长 All-Gather/Grouped Broadcast 和 sufficient-factor broadcasting 做联合选择 | 在 benchmark 上相对最佳基线最高 217% 加速；BERT-MoE 实验中可平滑使用非规则设备数，最高报告 64% 加速。实验模型和规模明显小于本项目。 |
| [FlashFlex / HexiScale](https://arxiv.org/abs/2409.01143)（MLSys 2026 版本为 HexiScale） | 每个 pipeline stage 可以有不同 GPU 数、TP degree、layer 数和 batch/chunk 数 | 将异构资源上的非对称 3D parallel allocation 建模为约束优化，并用 hierarchical graph partitioning 生成布局 | 7B–30B LLM 上，HexiScale 相比异构基线达到 1.5–2.4× 吞吐；需要独立进程启动和非对称通信组，不能直接作为 Megatron 配置项。 |
| [Cephalo: Harnessing Heterogeneous GPU Clusters for Training Transformer Models](https://doi.org/10.1145/3721145.3730418)（ICS 2025） | 计算分配和训练状态分片不再绑定；每张 GPU 可保存从 0% 到 100% 的状态 | 在 FSDP 上实现 uneven sharding，按 GPU 算力分配 batch，联合配置 gradient accumulation、recompute 和 CPU offload | 最多 64 GPU、7B 模型上最高比异构基线高 10×；不等长 collective 的运行时开销最高约 25%，因此需要 profile 和调度器。 |
| [Varuna: Scalable, Low-cost Training of Massive Deep Learning Models](https://www.microsoft.com/en-us/research/publication/varuna-scalable-low-cost-training-of-massive-deep-learning-models/)（EuroSys 2022） | 不要求固定的超集群拓扑，资源数量变化时可重选 pipeline stage 数和布局 | 自动识别 cut point，按网络和显存约束划分 pipeline；用动态、语义保持的 reconfiguration 适应 spot/preemption | 在普通网络和低优先级 VM 上训练到 200B；相对其他 pipeline 方法最高 26% 改善。适合资源弹性，不等于任意非均匀 TP。 |
| [Oobleck: Resilient Distributed Training of Large Models Using Pipeline Templates](https://arxiv.org/abs/2309.08125)（SOSP 2023） | 预先生成多个不同大小的 pipeline template，用组合覆盖可用节点数 | 将逻辑等价但物理异构的 pipeline template 作为“资源硬币”，运行时用动态规划选取覆盖全部剩余节点的组合 | 主要目标是故障恢复和资源不闲置；可借鉴其“预生成多个可行布局”的思想，但不能替代大模型状态 offload。 |

这些系统的共同点是：它们改变了 collective 的语义或调度方式，而不是把一个普通 NCCL group 简单扩容。要支持 64→72，至少需要处理不等长 All-Gather/Reduce-Scatter、不同 stage 的 micro-batch 数、梯度归一化、参数版本一致性和 checkpoint 映射；对 MoE 还要处理 expert routing 的不均匀 token 数。

### 3. 将额外 GPU 用作状态层，而不是计算 rank

另一条路线是不强求 72 张 GPU 都加入同一个计算 mesh，而是保持 64-rank compute group，将额外 GPU 或节点作为参数/optimizer state 的热缓存或 staging layer。相关工作包括：

- **Cephalo** 把 compute assignment 与 state sharding 解耦，说明“GPU 负责计算”和“GPU 保存多少状态”可以是两个独立优化变量；
- **ZeRO-Infinity** 在 GPU、CPU 和 NVMe 之间分片并预取参数、梯度和 optimizer state，目标是突破 GPU memory wall；
- **LuWu/DisDP** 更进一步将 optimizer/state 和 collective 放到独立的 in-network 服务，计算 GPU 只保留当前 working set。

这条路线对本项目更现实，但额外 GPU 带来的容量必须按字节数核算。若按问题中的 **24T 参数、每参数 10 bytes**，仅训练状态就约为 240 TB；64→72 只增加 12.5% 的 GPU 数量，远不能靠 HBM 解决一个数量级的容量缺口。因此实际系统仍需要 CPU DRAM、NVMe 或 CXL memory 作为主容量层，额外 GPU 只能缓存当前 layer/expert 的热 working set。

### 4. 72 张 GPU 对效率的上限

如果把 64 张 GPU 扩展到 72 张，并且假设完全线性、没有通信和调度开销，计算时间最多缩短到：

```text
T72 = T64 × 64/72 = 0.889 × T64
```

也就是 5 分钟变成约 4.44 分钟，理想收益只有 12.5%。因此只要非对称 collective、pipeline imbalance、额外同步和数据搬运合计造成超过约 11.1% 的开销，72 张 GPU 的 step time 就可能不如规则的 64 张配置。对于背景中 rollout 约 15 分钟、训练约 5 分钟的 RL 周期，继续把训练压到 4.44 分钟本身并不能缩短同步迭代；更有价值的是让训练使用更少的计算 GPU，并把释放出来的 GPU 分配给 rollout，或让 offload 后的训练时间接近 15 分钟。

## 方案设计归纳

### 1. Rollout/training 时间平衡的主要路线

现有 RL 工作大致形成四种路线：

1. **静态资源比例优化**：ReaL 和 BiDiRL 根据 workload 的时间模型搜索 rollout/training 的 GPU 划分。适合在运行前确定“训练侧最多可以使用多少 GPU”。
2. **异步解耦和流式重叠**：StreamRL、AReaL 将 rollout 和 training 变成持续运行的生产者/消费者，消除整批 barrier；代价是样本 staleness 和算法稳定性问题。
3. **长尾调度**：RollPacker 和 APRIL 处理 response length 的长尾，减少 rollout 内部空泡；这更适合解决“rollout 本身被少量长样本拖慢”的问题。
4. **运行时资源借用**：BiDiRL 通过 hot-switch 让瓶颈阶段临时借用另一个阶段的空闲 GPU，最接近“训练 5 分钟完成后，释放资源；训练变慢时再按需借回”的目标。

### 2. CPU offload 的共同设计模式

CPU 方案并不是简单地把 tensor 复制到主存，而是围绕以下机制组织：

- **分层粒度**：从 activation tensor（vDNN、Capuchin、SwapAdvisor）到 chunk（PatrickStar），再到参数/梯度/优化器分片（ZeRO、MegaTrain）；
- **预取和双缓冲**：在当前 layer 计算时搬运下一层所需状态，隐藏 H2D/D2H 延迟；
- **计算位置选择**：ZeRO-Offload 和 SuperOffload 将 optimizer computation 放到 CPU；ZenFlow 则只把部分 update 放到 CPU，避免 GPU 等待；
- **联合优化**：Ratel（LoHan 接收版本）表明 gradient、activation 和传输链路必须一起调度，单独优化某个内存池可能只是把瓶颈转移到 PCIe/NVLink；
- **动态状态布局**：PatrickStar、MegaTrain 只把当前计算所需的热状态物化到 GPU，其余状态保留在 CPU。

### 3. NVMe/SSD 与其他介质的共同设计模式

NVMe 的容量远大于 GPU/CPU，但延迟和带宽更差，因此论文普遍采用以下策略：

- 将 **cold state** 或 activation checkpoint 放到 SSD，而不是每一步无条件交换所有 tensor；
- 以 chunk、bucket 或批量 I/O 进行迁移，避免大量小 tensor 造成随机 I/O；
- 通过 compression、deduplication、forwarding 减少写入量；
- 用 GPU direct I/O、预取和计算/I/O overlap 隐藏读写延迟；
- 在 Smart-Infinity 中进一步把 update 放到 near-storage accelerator，减少参数来回搬运；
- 对 MoE 使用 expert sparsity 和热度预测，只预取可能被路由到的 expert。

CXL memory 处在 CPU DRAM 与 SSD 之间：容量和延迟可能优于 NVMe，但实际收益高度依赖互联拓扑、coherence 协议和 NUMA 放置，不能简单等同于“更快的 CPU offload”。

## 对 2.4T RL 训练侧方案的直接启示

### 建议的目标函数

可以把训练侧 offload 的目标写成资源和时间约束问题：

```text
minimize    number of training GPUs
subject to  memory(model, optimizer, activation | GPU, CPU, NVMe) is feasible
            T_train(offload, GPU_count) <= T_rollout + scheduling_margin
            numerical behavior and RL convergence remain acceptable
```

对于同步 RL，单轮时间近似为 `T_rollout + T_train`；对于异步 RL，吞吐更接近由较慢阶段的服务率决定。因而“训练从 5 分钟变为接近 15 分钟”并不是退化目标，而是可以接受的资源-时间交换，只要不会超过 rollout 的生产速率太多。

### 可能的组合架构

1. **上层调度器**：借鉴 ReaL/BiDiRL，用 profiling 得到不同 GPU 数、PP/TP 配置和 offload 比例下的 `T_rollout`、`T_train`，选择接近 15 分钟的训练配置。
2. **GPU 热工作集**：只保留当前 pipeline stage、当前 layer 或当前 expert 的参数、activation 和 gradient；使用 double buffer 和 prefetch。
3. **CPU 温状态层**：优先存放 optimizer state、非当前 stage 的参数分片和 activation checkpoint；用 ZenFlow/SuperOffload 类异步 update 隐藏 CPU 计算。
4. **NVMe 冷状态层**：当 CPU 容量不足时，存放低频访问的 activation 或参数/optimizer shard；使用 SSDTrain/ZeRO-Infinity 类批量 I/O 和计算重叠。
5. **RL 版本控制**：对异步 rollout 明确 policy version、最大 staleness 和 checkpoint 边界；offload 只能改变状态位置和时间，不能破坏一次 update 内的参数一致性。

### 需要特别注意的限制

- **Offload 解决的是容量，不会消除训练 FLOPs。** 对 2.4T 模型，减少 GPU 数量后训练可能远超 15 分钟；需要先用时间模型确认计算量仍可接受。
- **Activation offload 与 model/optimizer offload 的收益不同。** activation 主要释放长序列、大 batch 的峰值显存；model/optimizer offload 更适合降低 PP 和参数常驻容量。
- **带宽是第一约束。** 如果 H2D/D2H 或 NVMe I/O 无法与计算重叠，训练可能从 5 分钟膨胀到超过 rollout 周期，反而降低 RL 吞吐。
- **PP 兼容性需要单独验证。** 一些 layer-level activation offload 实现对 PP 有限制；在需要较大 PP 的 2.4T 场景，应优先评估参数、梯度和 optimizer state 的分片卸载。
- **RL 的异步性会放大一致性问题。** 需要同时测量最终收敛、policy staleness、训练吞吐和 GPU 利用率，不能只看单步时间或显存峰值。
