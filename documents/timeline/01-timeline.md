Observation
1. RL Rollout 需要大量时间产生训练样本, 且产生样本的时间相对固定, e.g. 产生 1024 个 seq-length=128k 的样本需要 15min 
2. RL 训练一个大模型需要占用较大内存, 通常使用 PP + EP 来容纳模型.
   - PP 需要足够多的样本来填充流水线, 且面临严重的 memory 不均匀
   - EP 需要 dispatch 和 combine, 带来额外的通信成本
   - 总结: 为来容纳完整 Qwen3.8-2.4T 模型, 需要足够多的 GPU 撑起足够大的 PP 和 EP, 但 GPU 跑不满
3. RL 训练长文本时 (256k), 需要 CP 来切分序列

- PP -> PP bubble -> 利率低
- 额外的 GPU

问题描述
1. 细粒度管理 HBM 存储的内容, 通过 offload 减少 memory 占用, 减少 GPU 需求.
2. 目标: 计算给定 GPU 数量下最高效的 offload 策略, 或给定 Rollout 时间下, 最少的 GPU 占用

解决方案
1. 采用 double buffer 策略, 在 GPU 上维护两个 layer 的参数 / 激活值 / 梯度, 降低单次 forward 后在 GPU 驻留的 memory
2. 采用 offload + recompute forward 和 backward 过程中的 peak memory, 后者能降低 cpu memory 占用
3. 仿真结果表明, double-buffer 策略能够将 GPU 驻留的 memory 占用降低到 2-10G (fsdp-2/3 能进一步降低 memory 占用), 需要用剩下的 memory 加速训练
   - 增大 micro-batch, 降低 cpu-overhead

待讨论问题
1. Qwen3.8-2.4T 及后续模型采用 GatedDeltaNet + GatedAttention 的混合架构, GatedDeltaNet 在 CP 下的训练表现不佳
   - GatedDeltaNet 需要前序 chunk 的 St(记忆), 导致训练过程中存在前后的依赖关系, 导致 CP bubble
   - GatedDeltaNet chunk 内部的计算规模较小, 计算效率低, 增加 mbs 提高计算效率
   - 选择合适的计算 chunk size (chunk size 越大, chunk 内部的计算开销越大, 但是保存的激活值越小)


