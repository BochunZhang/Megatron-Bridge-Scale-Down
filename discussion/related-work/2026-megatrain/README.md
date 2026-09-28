# MegaTrain

## 版本与来源

- **论文题目**：MegaTrain: Full Precision Training of 100B+ Parameter Large Language Models on a Single GPU
- **接收状态**：COLM 2026 accepted papers（poster）。当前未找到公开的 COLM 最终排版 PDF；目录中的 `megatrain-colm26.pdf` 是 arXiv 论文稿 `2604.05091`，可作为接收工作的公开版本。
- **来源**：[COLM 2026 accepted papers](https://colm.eventhosts.cc/Conferences/2026/AcceptedPapers)、[arXiv](https://arxiv.org/abs/2604.05091)。

## 论文总结

MegaTrain 目标是在单个 GPU 上进行 100B+ 模型的全精度训练。它把 GPU 从“保存完整模型”转变为“执行当前层的高速缓存”，将持久状态放在主机内存：

1. **持久状态主机化**：参数、优化器状态和累积梯度存放在 CPU 内存，GPU 只保留当前层的临时工作集。
2. **逐层流式执行**：前向前预取当前层参数，反向后立即把梯度回写主机；双缓冲在计算时搬运下一层。
3. **多 CUDA stream pipeline**：参数预取、GPU 计算、梯度卸载并行，减少 PCIe/NVLink 传输暴露时间。
4. **无持久 autograd 图**：使用无状态 layer template 重建必要的反向依赖，避免整网 autograd 图占用内存。

论文报告单个 H200 加 1.5 TB 主机内存可训练最高约 120B 模型；在 14B 场景相对 DeepSpeed ZeRO-3 CPU offload 最高约 1.84 倍，且展示了 GH200 上 7B、512K 上下文训练。

## 对 RL 训练侧 offload 的启示

- MegaTrain 提供最直接的“缩小训练 GPU 集群”路径：把每个训练副本的持久状态迁到主机，释放 GPU 给 rollout。
- 逐层双缓冲可以与 rollout 进行阶段级流水：rollout 期间预取下一训练 batch 的前几层，训练期间生成侧使用独立 GPU。
- 无状态层模板适合把训练模型拆成可调度单元，但需要在 checkpoint、梯度累积和参数版本切换处增加显式状态管理。

## 局限与工程注意事项

- 单 GPU 结果依赖极高的主机内存带宽和 GPU↔CPU 链路；多 GPU/多节点下需要重新设计参数分片和通信。
- 全精度 optimizer 状态传输量很大，训练吞吐会随上下文长度和 batch size 变化；必须用 rollout 时间作为调度目标而非只看单 step 延迟。
- 论文是 arXiv 公开稿，COLM 最终版本未公开，引用时应同时标明 COLM 2026 接收状态。
