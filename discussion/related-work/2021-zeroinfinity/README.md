# ZeRO-Infinity

## 版本与来源

- **论文题目**：ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning
- **发表版本**：SC21（International Conference for High Performance Computing, Networking, Storage and Analysis），2021，Proceedings 论文，pp. 1–14。
- **目录中的 PDF**：`zeroinfinity-sc21.pdf` 为 arXiv 论文稿（对应 SC21 论文内容）；另保留 `zeroinfinity-sc21-slides.pdf` 作为官方演示材料，避免混淆论文与 slides。
- **来源**：[SC21 官方页面](https://sc21.supercomputing.org/presentation/index-id%3Dpap464%26sess%3Dsess174.html)、[arXiv](https://arxiv.org/abs/2104.07857)、[DeepSpeed 资料页](https://www.deepspeed.ai/)。

## 论文总结

ZeRO-Infinity 将 GPU、CPU 内存和 NVMe 组成分层存储，目标是在不修改模型代码的前提下训练远超单机 GPU 显存容量的模型。它建立在 ZeRO-3 参数/梯度/优化器分片之上，并加入面向 NVMe 的内存中心分块和流水线。

主要机制：

1. **三类状态分片**：参数、梯度和优化器状态按数据并行分片，只有计算当前层所需的分片进入 GPU。
2. **Memory-centric tiling**：把大参数张量切成可独立搬运和计算的 tile，降低单次 GPU 缓存需求。
3. **多级预取/回写**：重叠 NVMe↔CPU、CPU↔GPU、GPU 计算以及 CPU optimizer，减少介质延迟暴露在关键路径上。
4. **大规模可扩展性**：论文展示了训练万亿级参数模型的可行性，并将 NVMe 作为容量扩展而非仅用于 checkpoint。

## 对 RL 训练侧 offload 的启示

- ZeRO-Infinity 给出了训练侧“模型状态放在 CPU/NVMe、GPU 只做当前工作集”的基础架构，适合减少 RL 训练副本所需的 GPU 数量。
- 参数和优化器状态可放在 CPU；极大模型的冷参数或 checkpoint 可放在 NVMe，按 rollout 所需层/阶段预取。
- 分层流水线可以把训练预取安排在 rollout 运行期间，但必须为 rollout 的随机访问和训练的顺序层访问分别设计缓存策略。

## 局限与工程注意事项

- NVMe 带宽、队列深度和 SSD 寿命会成为瓶颈；若多个训练副本共享盘，需要做带宽隔离。
- 分片和 tile 调度增加通信/同步复杂度，PP 较大时还要处理 stage 间的参数可见性。
- 论文主要关注大模型训练容量，不直接解决 RL 中 rollout 权重版本、策略滞后或训练/生成调度问题。
