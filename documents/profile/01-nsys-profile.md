观察到 nsys-profile 和 memory-profile 对 qwen3.5-9b 模型训练的影响
- 关闭 profile, 全部稳定在 880TFlops 上下
- 全部 profile, 吞吐稳定在 440TFlops+, 最后一个 iteration 下降到 420TFlops+
- 部分 profile, 前面几个 iteration 的吞吐在 600TFlops+, profile 后面两个轮次吞吐大幅下降