# SuperOffload

## 版本与来源

- **论文题目**：SuperOffload: Unleashing the Power of Large-Scale LLM Training on Superchips
- **接收/发表版本**：ASPLOS 2026，Proceedings of the 31st ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 1，pp. 249–264，DOI `10.1145/3760250.3762217`。
- **目录中的 PDF**：`superoffload-arxiv.pdf` 为作者 arXiv 稿（`2509.21271`）；官方 ACM 最终排版版本未公开下载，因此在此明确标注为 accepted-work 的公开稿。
- **来源**：[ASPLOS 2026 program](https://www.asplos-conference.org/asplos2026/program/)、[项目主页](https://supercomputing-system-ai-lab.github.io/projects/superoffload/)、[arXiv](https://arxiv.org/abs/2509.21271)。

## 论文总结

SuperOffload 面向 GH200 等 CPU-GPU superchip，建立在 ZeRO Stage 3 上，重点优化 CPU Adam 与 GPU 反向传播之间的同步瓶颈。

1. **Speculation-Then-Validation（STV）**：CPU 先推测执行 Adam 更新，GPU 反向完成后再验证梯度裁剪、NaN 等条件；验证失败时回滚并重执行，成功时隐藏 CPU optimizer 延迟。
2. **细粒度 bucket 化**：把参数/梯度分成小 bucket，部分 bucket 保留在 GPU，避免下一迭代因等待尾部 bucket 而停顿。
3. **Superchip-aware casting**：在 GPU 侧完成精度转换，再通过 NVLink-C2C 传输高精度数据，利用 superchip 的高带宽互联。
4. **拓扑感知执行**：结合 NUMA 绑定和内存带宽分区（MPAM）降低 CPU 线程与 GPU 访问的相互干扰。

项目报告相对 ZeRO-Offload 最高约 4 倍提升，在单个 GH200 上支持 GPT-OSS-20B/Qwen3-14B 等模型，并达到约 600 TFLOPS 的训练吞吐。

## 对 RL 训练侧 offload 的启示

- STV 很适合作为 rollout 期间的后台 optimizer：只要验证条件不依赖下一批 rollout，即可把 CPU 更新从训练关键路径移走。
- bucket 尾部保留 GPU 的思想可用于“rollout-ready 参数集合”：优先把下一次生成所需的层或 adapter 权重保留/更新完毕。
- GH200/NVLink-C2C 说明介质选择必须结合硬件拓扑；在 PCIe 机器上直接复制 SuperOffload 的调度参数可能失效。

## 局限与工程注意事项

- STV 需要可回滚的 optimizer 状态和确定性的验证条件；梯度裁剪、混合精度异常或 RL 特有的动态 loss 会增加回滚频率。
- 该工作主要优化 CPU↔GPU，而不是 NVMe 容量扩展；超大模型仍需 ZeRO-Infinity 类分层存储。
- 公开 PDF 是 arXiv 稿，最终 ACM 版本可能包含实现和实验细节差异，应以 DOI 版本为最终引用依据。
