# GB200 Qwen3.8 selective activation offload 对比计划

## 1. 测试背景

本实验用于排查 selective activation offload 的瓶颈位置，而不是直接给 dense 和 sparse/MoE 模型做绝对性能排名。重点是回答以下问题：

- 哪个 MLP/MoE 激活张量占据了峰值 GPU memory；
- D2H/H2D copy 是否能与 forward/backward 计算重叠；
- offload 带来的显存收益是否被 copy、同步等待或 CPU staging 开销抵消；
- 对便宜算子执行 recompute，是否比保存或搬运其激活更划算；
- dense MLP 与 sparse MoE 在相同实验约束下，是否存在不同的主要 memory/transfer 瓶颈。

测试对象为 Qwen/Qwen3.8 的两个 text 模型：27B dense 模型和 35B-A3B sparse/MoE 模型。两者都采用 Gated DeltaNet 与 Gated Attention 交替的层结构。Gated DeltaNet 路径不暴露 `qkv_linear` 和 `core_attn` 的 offload/recompute 接口，因此本轮不把 attention 作为比较对象，只隔离 MLP（dense）和 MoE expert MLP（sparse）的 selective offload/recompute 影响。attention layer 的影响另列为后续 TODO。

当前 recipe 中 35B 配置仍需在实现阶段确认并切换到 Qwen/Qwen3.8 对应的 35B-A3B checkpoint/config；实验结果必须记录最终展开后的 model id 和 provider config，不能仅依据 recipe 函数名判断模型身份。

## 2. 试验约束

### 2.1 硬件、模型和训练设置

- 硬件固定为 GB200；两个模型使用相同的节点/GPU 数、序列长度、global batch size、micro batch size、训练步数和数据模式。
- 模型固定为 Qwen/Qwen3.8 27B dense 与 Qwen/Qwen3.8 35B-A3B sparse/MoE。并行度沿用各自 Qwen3.8 GB200 recipe；记录 TP、PP、CP、EP、ETP、DP 和 sequence parallel 的最终值。
- 第一轮固定 BF16，用于建立容易归因的机制基线；第二轮在完全相同的矩阵上切换 MXFP8，用于确认精度、量化 metadata、padding 和 grouped GEMM 是否改变 offload 结论。
- 所有配置均关闭 CUDA graph：`cuda_graph_impl="none"`，并清空 `cuda_graph_modules`。本计划不把 graph capture/replay 的影响混入 selective offload 结论。
- 本轮只研究 fine-grained activation offloading 和 selective recompute；不启用 layer-level `cpu_offloading`，不启用 full-layer recompute，不改变 optimizer offload、TP/PP/EP 或 dispatcher 来制造额外变量。
- 每个配置先 warmup 20 步，再采样 100 步，独立重复 3 次。若配置校验失败、训练出现 NaN/Inf 或显存不足，保留失败日志和最终 config，不把该配置标记为性能结果。

### 2.2 运行与目录

代码放在 `examples/scale-down/01-analyse/01-fine-grained-offload-compare/`，按以下职责拆分公共参数、模型适配、训练入口、profile 配置和结果汇总；不依赖 `srun`，统一使用：

```bash
uv run python -m torch.distributed.run --nproc_per_node=<N> <script.py> ...
```

所有输出写入项目内 `results/`，目录至少包含：

```text
results/<model>/<precision>/<experiment>/<repeat>/
  config.json
  environment.json
  train.log
  profile/
  memory/
  summary.json
```

启动时将 `HF_HOME`、`HF_HUB_CACHE`、`TRANSFORMERS_CACHE` 统一指向 `<repo>/.cache/huggingface`，将 NeMo cache 指向 `<repo>/.cache/nemo`。每次运行同时保存实际 model id、数据 manifest、缓存路径、GPU/驱动/Transformer Engine 版本和完整 CLI override，避免不同缓存或软件版本造成不可解释的差异。

## 3. 实验目标

### 3.1 基于 Qwen3.8 对比 dense 和 sparse 模型在 MLP/MoE 卸载部分的效率

#### 3.1.1 试验目标

对比两个 Qwen3.8 模型在 feed-forward 路径上的显存-吞吐折中，并定位 selective offload 的主要瓶颈：

- dense 模型测量 dense MLP 的整体 recompute，以及新增的 activation-only offload/recompute 路径；
- sparse 模型测量 expert FC1 输入、MoE activation 输入和整个 MoE recompute 的组合影响；
- 对每个配置分别记录 peak HBM、tokens/s、step time、D2H/H2D bytes、copy duration、同步等待和 recompute duration；
- 只在 loss/梯度健康且与 baseline 数值差异在容差内时，比较性能收益。

这里的 dense 与 sparse 对比是机制对比。MoE 额外包含 router、dispatch、permute、expert load balance 和 grouped GEMM，不能把端到端绝对 tokens/s 直接解释为 dense 或 sparse 架构的普遍优劣。

#### 3.1.2 试验矩阵

每一轮精度（BF16、随后 MXFP8）都包含各自的无 offload baseline。所有配置都使用 `cuda_graph_impl="none"`；未实现的 dense `mlp_act` 组合在代码改造完成前只记录为 pending，不得提前宣称已验证。

除 baseline 外，表中出现 `recompute_modules` 的配置都必须显式设置 `recompute_granularity="selective"`；`recompute_method` 和 `recompute_num_layers` 保持为 `None`。`None` 表示该实验不执行 recompute。

**Dense 27B：**

Dense 的 recompute 实验都表示对整个 `mlp` 模块进行重算。`layernorm` 用于重算 norm；本实验关注的是进入 MLP 的 `mlp_norm`。`recompute-1` 与 `recompute-2` 的差别，是后者额外卸载 `mlp_norm` 的输入，从而进一步释放该输入激活。

Dense 的 offload 实验都采用“MLP activation offload + activation recompute”的组合方案。`offload-1` 与 `offload-2` 的差别，是后者额外卸载 `mlp_norm` 的输入。

Dense 模型中，`mlp_norm` 的输出就是 `mlp_fc1` 的输入。开启 `layernorm` 重算后，`mlp_norm` 输出可以在反向阶段重新生成，因此没有必要再单独卸载 `mlp_fc1` 的输入，本轮不增加 dense `mlp_fc1` offload 支持。

| 实验 | `recompute_modules` | `offload_modules` | 目的 |
|---|---|---|---|
| baseline | `None` | `None` | 无 offload/recompute 的基线 |
| recompute-1 | `['layernorm', 'mlp']` | `None` | 整个 MLP 和 norm 重算，作为纯 recompute 方案 |
| recompute-2 | `['layernorm', 'mlp']` | `['mlp_norm']` | 在 recompute-1 基础上进一步卸载 `mlp_norm` 输入 |
| offload-1 | `['layernorm', 'mlp_act']` | `['mlp_act']` | 通过 activation 重算与 offload 处理 MLP，保留 `mlp_norm` 输入 |
| offload-2 | `['layernorm', 'mlp_act']` | `['mlp_norm', 'mlp_act']` | 在 offload-1 基础上进一步卸载 `mlp_norm` 输入 |

primary comparison 为 `baseline -> recompute-1 -> recompute-2 -> offload-1 -> offload-2`。其中 recompute-1/2 用于测量整个 MLP 重算的代价以及 `mlp_norm` 输入是否仍需保留；offload-1/2 用于测量只重算 activation 并卸载 FC1 输出的方案，避免重算 FC1/FC2，并验证额外卸载 `mlp_norm` 输入是否带来显著收益。

**Sparse/MoE 35B-A3B：**

MoE 的 recompute 实验都表示对整个 `moe` 模块进行重算。`recompute-1` 与 `recompute-2` 的差别，是后者额外卸载 `mlp_norm` 的输入，从而进一步释放该输入激活。MoE 的 offload 实验都采用“expert FC1/activation offload + activation recompute”的组合方案处理整个 MoE 模块；`offload-1` 与 `offload-2` 的差别，是后者额外卸载 `mlp_norm` 的输入。

与 dense MLP 不同，MoE 中 `mlp_norm` 的输出还要经过 router 才进入 expert MLP。因此 offload + recompute 与 full recompute 覆盖的显存范围不同：前者只处理 `expert_fc1`/`moe_act` 等 expert MLP 激活，不处理 router 激活；后者重算整个 `moe` forward，同时覆盖 router 激活，但会引入额外的 router 计算开销。

MoE forward 还包含 shared expert。本轮关闭 shared expert 的独立重计算，不在 `recompute_modules` 中加入 `shared_experts`；shared expert 仍参与正常 forward。后续若单独测试 shared expert recompute，必须先禁用 `moe_shared_expert_overlap`；对于 `recompute=['moe']` 的全模块 checkpoint，shared expert 是否被 checkpoint 覆盖必须通过 trace 确认并单独报告。

| 实验 | `recompute_modules` | `offload_modules` | 目的 |
|---|---|---|---|
| baseline | `None` | `None` | 无 offload/recompute 的基线 |
| recompute-1 | `['layernorm', 'moe']` | `None` | 整个 MoE 和 norm 重算，作为纯 recompute 方案 |
| recompute-2 | `['layernorm', 'moe']` | `['mlp_norm']` | 在 recompute-1 基础上进一步卸载 `mlp_norm` 输入 |
| offload-1 | `['layernorm', 'moe_act']` | `['expert_fc1', 'moe_act']` | 通过 expert FC1/activation offload 与 activation 重算处理 MoE，保留 `mlp_norm` 输入 |
| offload-2 | `['layernorm', 'moe_act']` | `['mlp_norm', 'expert_fc1', 'moe_act']` | 在 offload-1 基础上进一步卸载 `mlp_norm` 输入 |

`offload-1/2` 中的 `moe_act` offload 语义是卸载 FC1 输出，不是卸载 activation 输出；`moe_act` recompute 只重算 activation，不重算 FC1。`fused_group_mlp` 作为整段 fused expert MLP 的另一种机制，若纳入实验必须另设组，不能与 `expert_fc1` 或 `moe_act` offload 同时使用。

Sparse/MoE 的 primary comparison 为 `baseline -> recompute-1 -> recompute-2 -> offload-1 -> offload-2`。最终报告同时给出每个配置相对本模型 baseline 的显存变化、吞吐变化和 copy/recompute 归因，不能只报告单一最佳配置。

#### 3.1.3 代码改造要求

当前 Megatron-LM 的 fine-grained offload allowlist 支持 `mlp_norm`、`expert_fc1`、`moe_act` 和 `fused_group_mlp`，但不支持 dense MLP 的 `mlp_act`；selective recompute allowlist 也只有 dense `mlp`，没有 `mlp_act`。因此 `offload-1/2` 需要先完成代码改造，再加入正式矩阵：

1. 为 dense MLP 增加明确的 `mlp_act` offload hook，语义定义为 FC1 输出，即 activation 函数的输入；不能复用 MoE `moe_act` 名称而产生歧义。
2. 增加 dense `mlp_act` selective recompute，只重算 activation（包括 gated SwiGLU/GeGLU 所需的 bias、interleave、scale 处理），不重算 FC1 或 FC2。
3. 更新 TransformerConfig 的字段说明、allowlist 和组合校验；覆盖 transformer-engine fused activation、BF16、MXFP8、CUDA graph 关闭路径，并补充单元测试验证保存/释放/重算的张量语义。
4. 通过 memory snapshot 或 saved-tensor trace 确认 `offload-1/2` 实际卸载/重算的张量、dtype、元素数和 copy bytes；不能仅凭张量拓扑推断节省量。

### 3.2 TODO：对比 attention layer 的影响

待完成 attention 专项实验，至少包括：

- 确认 Gated Attention 层是否实际暴露 `qkv_linear`、`core_attn` 或 `attn_proj` 的 fine-grained hook，以及 Gated DeltaNet 层的对应限制；
- 将交替层按 layer type 分桶，避免把不支持 attention offload 的 Gated DeltaNet 层错误计入同一平均值；
- 在不改变 3.1 MLP/MoE 约束的前提下，增加 attention baseline、单项 offload 和可用的 selective recompute 组合；
- 单独报告 attention 的 copy/recompute 开销，并评估它与 MLP/MoE offload 同时开启时的相互影响。

## 4. 观测指标与判据

先确认配置校验通过、loss/梯度无 NaN/Inf，再比较性能。每个 step/stage 记录 forward、backward、recompute、optimizer 时间；模块级 D2H/H2D bytes、duration、stream、event、同步点和 overlap；GPU peak/allocated/reserved memory；CPU、HBM、NVLink/PCIe 带宽；SM、Tensor Core、DRAM 利用率；MoE router imbalance、expert token counts 和 grouped GEMM occupancy。

copy 时间占比高、重叠率低、CPU staging 队列出现空洞或链路带宽饱和，判定为 transfer 瓶颈；显存下降小但 step time 明显增加，优先归因于 transfer 或同步；重算时间增加而 copy bytes 显著下降，则归因于 recompute/compute 瓶颈。开启 memory history 时保存 snapshot，并使用现有 snapshot 工具重放峰值分配。

## 5. 交付物

每项实验保存完整 config、环境/GPU 信息、训练日志、profile trace、显存 snapshot 和汇总 CSV/Markdown。最终报告给出：

- dense 与 sparse 各自相对 baseline 的显存-吞吐 Pareto；
- BF16 与 MXFP8 的机制差异；
- expert FC1、MoE activation、dense MLP 和 dense activation 的瓶颈归因；
- offload-1/2 代码改造前后的行为和数值验证；
- 推荐默认配置、不适用条件，以及 attention 专项实验的 TODO 清单。
