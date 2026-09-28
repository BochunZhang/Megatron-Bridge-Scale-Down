# Ratel（LoHan 的接收版本）

## 版本与来源

- **论文题目**：Ratel: Optimizing Holistic Data Movement to Fine-tune 100B Model on a Consumer GPU
- **与 LoHan 的关系**：LoHan 是早期预印本/项目名称，接收版本更名为 Ratel。
- **接收版本**：IEEE 41st International Conference on Data Engineering（ICDE 2025），DOI `10.1109/ICDE65448.2025.00029`，pp. 1–15。
- **目录中的 PDF**：`ratel-icde25.pdf`，作者公开的 ICDE 版本。
- **来源**：[IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/11113169/)、[作者 PDF](https://wangzeke.github.io/doc/Ratel-ICDE25.pdf)。

## 论文总结

Ratel 针对消费级 GPU 显存不足以微调百亿/千亿参数模型的问题，把参数、激活值、梯度和优化器状态在 GPU、主存和 SSD 之间的搬运视为一个整体的数据移动优化问题。系统不只决定“哪些张量卸载”，还联合决定卸载顺序、预取时机、梯度落盘位置和重计算策略。

核心设计包括：

1. **梯度主动卸载**：在反向传播过程中尽早把梯度移出 GPU，使 CPU 优化器执行、CPU/SSD 访问与后续 GPU 反向计算重叠。
2. **激活交换与重计算协同**：根据层的计算/传输代价选择保存激活或重计算，避免所有激活都写入慢速介质。
3. **整体流量建模**：同时考虑 PCIe、CPU 内存带宽、SSD 带宽和 GPU kernel 的竞争，按关键路径调度搬运，而不是分别优化参数或激活。
4. **消费级硬件落地**：在 RTX 4090 加 256 GB 主存的机器上支持 175B 模型微调；论文报告 13B 场景相对代表性基线最高约 2.32 倍吞吐提升。

## 对 RL 训练侧 offload 的启示

- Ratel 的“整体数据移动”视角适合 RL 中 rollout 与训练的异步流水线：可以把训练侧的预取、梯度回写和 optimizer step 安排到 rollout 产生下一批样本的时间窗口内。
- 梯度主动卸载可以减少训练 GPU 的峰值显存，使 PP/DP 配置缩小，把释放出的 GPU 交给 rollout；这正对应“训练时间变长但与 rollout 对齐”的目标。
- 激活保存/重计算不应固定为单一策略。RL 的序列长度、micro-batch 和奖励模型阶段变化较大，需要按阶段估计传输与重计算代价。

## 局限与工程注意事项

- 方案依赖主机内存、SSD 和 PCIe 带宽；当 rollout 与训练共享 I/O 或 CPU 资源时，独立测得的收益可能下降。
- 消费级单机实验不能直接代表多节点训练，需要额外处理 NUMA、跨节点通信和 checkpoint 一致性。
- 调度器需要知道张量生命周期和关键路径，接入现有 Megatron/Transformer Engine 时会涉及 autograd、通信和优化器边界改造。
