# SpeedLoader

## 版本与来源

- **论文题目**：SpeedLoader: An I/O Efficient Scheme for Heterogeneous and Distributed LLM Operation
- **发表版本**：NeurIPS 2024，Main Conference Track，DOI `10.52202/079017-1092`。
- **目录中的 PDF**：`speedloader-neurips24.pdf`，NeurIPS 官方 Conference PDF。
- **来源**：[NeurIPS 论文页](https://proceedings.neurips.cc/paper_files/paper/2024/hash/3d3a9e085540c65dd3e5731361f9320e-Abstract-Conference.html)、[官方 PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/3d3a9e085540c65dd3e5731361f9320e-Paper-Conference.pdf)。

## 论文总结

SpeedLoader 针对异构、分布式 LLM 运行中模型状态位于主机 DRAM 或块设备、I/O 逐渐成为主瓶颈的问题，重新安排模型加载、激活卸载和反向传播。

1. **有效 batch 级调度**：将多个 sub-batch 组织成一个 effective batch，使模型参数在一轮中只需加载两次，而不是每个 sub-batch 重复加载。
2. **激活主机缓存**：前向产生的激活卸载到主机，反向时按需取回；通过重写计算图和插入 no-op 保持反向依赖正确。
3. **通信与计算重叠**：使用 pinned host memory，并以两个 CUDA stream 同时执行参数/激活交换和 GPU 计算。
4. **分布式一致性**：减少模型加载和梯度同步次数，适配异构 GPU/CPU/NVMe 环境。

论文报告训练最高约 3–30 倍加速、最高约 51% MFU；推理场景报告约 1.5–2.35 倍提升。

## 对 RL 训练侧 offload 的启示

- SpeedLoader 的 effective-batch 思路可把多轮 rollout 样本聚合为一次训练更新，减少模型权重在训练 GPU 与主机之间的往返。
- 激活放入 pinned host memory 后，训练 GPU 可以在 rollout 阶段释放显存；下一次训练只恢复所需 sub-batch 的激活。
- 双 stream 重叠是实现“训练耗时匹配 rollout 耗时”的关键：一个 stream 做计算，另一个 stream 预取参数/激活并回写梯度。

## 局限与工程注意事项

- effective batch 会增加样本等待时间；RL 中需限制策略数据的新鲜度，不能盲目扩大聚合窗口。
- 激活主机缓存占用大量 DRAM，长序列或高并发 rollout 可能超过容量；必要时要增加 NVMe 分层和压缩/重计算。
- 图重写和 no-op 方案需要与现有 autograd、重计算、流水并行实现仔细集成。
