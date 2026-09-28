# GB200 Qwen3.5 fine-grained activation offload 对比计划

## 1. 测试背景

本实验用于排查 fine-grained activation offload 的瓶颈位置，而不是直接给 dense 和 sparse/MoE 模型做绝对性能排名。重点是回答以下问题：

- 哪个 MLP/MoE 激活张量占据了峰值 GPU memory；
- D2H/H2D copy 是否能与 forward/backward 计算重叠；
- offload 带来的显存收益是否被 copy、同步等待或 CPU staging 开销抵消；
- dense MLP 与 sparse MoE 在相同实验约束下，是否存在不同的主要 memory/transfer 瓶颈。

测试对象为 Qwen/Qwen3.5 的两个 text 模型：9B dense 模型，以及基于 35B-A3B 配置缩减为 20 层、64 个 routed experts 的 sparse/MoE 模型。两者都采用 Gated DeltaNet 与 Gated Attention 交替的层结构。Gated DeltaNet 路径不暴露 `qkv_linear` 和 `core_attn` 的 offload 接口，因此本轮不把 attention 作为比较对象，只隔离 MLP（dense）和 MoE expert MLP（sparse）的 fine-grained offload 影响。attention layer 的影响另列为后续 TODO。

当前测试分别使用 `Qwen/Qwen3.5-9B-Base` 和 `Qwen/Qwen3.5-35B-A3B-Base` 的 checkpoint/config；实验结果必须记录最终展开后的 model id 和 provider config，不能仅依据 recipe 函数名判断模型身份。

## 2. 试验约束

### 2.1 硬件、模型和训练设置

- 硬件固定为 GB200；两个模型使用相同的节点/GPU 数、序列长度、global batch size、micro batch size、训练步数和数据模式。
- 模型固定为 Qwen/Qwen3.5 9B dense，以及由 Qwen/Qwen3.5 35B-A3B 配置缩减为 20 层、64 个 routed experts 的 sparse/MoE 模型。并行度沿用各自 Qwen3.5 GB200 recipe；记录 TP、PP、CP、EP、ETP、DP 和 sequence parallel 的最终值。
- 第一轮固定 BF16，用于建立容易归因的机制基线；第二轮在完全相同的矩阵上切换 MXFP8，用于确认精度、量化 metadata、padding 和 grouped GEMM 是否改变 offload 结论。
- 所有配置均关闭 CUDA graph：`cuda_graph_impl="none"`，并清空 `cuda_graph_modules`。本计划不把 graph capture/replay 的影响混入 fine-grained offload 结论。
- 本轮只研究 fine-grained activation offloading；不启用 activation recompute、layer-level `cpu_offloading`，不改变 optimizer offload 或 TP/PP/EP。Dense 模型保持 recipe 默认 dispatcher；MoE 模型对 `alltoall` 与 `flex + hybridep` 做完整因子对比，除此之外的配置保持一致。
- 每个配置运行 10 个 iteration，全部 iteration 都纳入汇总，每个配置只运行 1 次，与 `run_qwen35_9b.sh` 的默认训练设置保持一致。若配置校验失败、训练出现 NaN/Inf 或显存不足，保留失败日志和最终 config，不把该配置标记为性能结果。

### 2.2 运行与目录

本对比的 benchmark 入口和结果收集器放在 `examples/scale-down/01-analyse/01-fine-grained-offload/02-offload-on-dense-or-expert-model/`。通用训练入口仍由 `01-mlp-scope-pretrain/run_pretrain_fsdp1.sh` 维护，新目录通过相对软链接复用该脚本；不依赖 `srun`，统一使用：

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

### 3.1 基于 Qwen3.5 对比 dense 和 sparse 模型在 MLP/MoE 卸载部分的效率

#### 3.1.1 试验目标

对比两个 Qwen3.5 模型在 feed-forward 路径上的显存-吞吐折中，并定位 fine-grained offload 的主要瓶颈：

- dense 模型测量 `mlp_norm` 与 `mlp_act` activation offload 的组合影响；
- sparse 模型测量 `mlp_norm`、`expert_fc1` 与 `moe_act` activation offload 的组合影响；
- 对每个配置分别记录 peak HBM、tokens/s、step time、D2H/H2D bytes、copy duration 和同步等待；
- 只在 loss/梯度健康且与 baseline 数值差异在容差内时，比较性能收益。

这里的 dense 与 sparse 对比是机制对比。MoE 额外包含 router、dispatch、permute、expert load balance 和 grouped GEMM，不能把端到端绝对 tokens/s 直接解释为 dense 或 sparse 架构的普遍优劣。

#### 3.1.2 试验矩阵

每一轮精度（BF16、随后 MXFP8）都包含各自的无 offload baseline。所有配置都使用 `cuda_graph_impl="none"`。本轮只比较 baseline 与纯 fine-grained activation offload；所有 case 都设置 `recompute_granularity=None`、`recompute_modules=None`，避免将重算开销混入 offload 性能差异。

**Dense 9B：**

Dense offload 同时覆盖 MLP norm 输入和 MLP activation，直接测量将这两类激活转移到 CPU 后的显存收益与传输开销。

| 实验 | `recompute_modules` | `offload_modules` | 目的 |
|---|---|---|---|
| baseline | `None` | `None` | 无 offload/recompute 的基线 |
| offload | `None` | `['mlp_norm', 'mlp_act']` | 纯 MLP activation offload，不执行重算 |

**Sparse/MoE 35B-A3B：**

MoE offload 同时覆盖 MLP norm 输入、expert FC1 相关激活和 MoE activation。router 和 shared expert 仍按 baseline 路径执行，不启用任何 activation recompute。

| 实验 | `recompute_modules` | `offload_modules` | 目的 |
|---|---|---|---|
| baseline | `None` | `None` | 无 offload/recompute 的基线 |
| offload | `None` | `['mlp_norm', 'expert_fc1', 'moe_act']` | 纯 MoE activation offload，不执行重算 |

上述两个 MoE case 分别使用 `alltoall` 和 `flex + hybridep` 运行。dispatcher 对比必须使用相同的模型、精度、batch、路由、并行度、CUDA graph、offload 设置和 profiling 后端。HybridEP 固定记录 `moe_flex_dispatcher_num_sms` 和 `NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN`；若运行环境缺少 HybridEP 包，记为环境失败，不能作为性能样本。

统一入口为：

```bash
examples/scale-down/01-analyse/01-fine-grained-offload/02-offload-on-dense-or-expert-model/benchmark_mlp_offload.sh
```

默认每个 case 运行 10 个 iteration、运行 1 次，不额外丢弃 warmup step。性能矩阵使用 `--profile none`；需要定位 D2H/H2D copy 或同步等待时，用同一入口通过 `--case <name>` 选择代表性 case，并用 `--profile nsys` 或 `--profile torch` 重跑。汇总结果写入 `results/01-analyse/01-fine-grained-offload/benchmarks/<run-time>/`。

#### 3.1.3 实现验证要求

当前 Megatron-LM 已支持 dense `mlp_act` 以及 MoE `expert_fc1`、`moe_act` 的 fine-grained offload。本轮需要验证这些 hook 在 BF16、MXFP8 和 CUDA graph 关闭路径中实际生效，并通过 memory snapshot 或 saved-tensor trace 确认卸载张量、dtype、元素数和 copy bytes；不能仅凭配置值推断节省量。若启用了不兼容的 Transformer Engine fused MLP 路径，测试必须失败并明确报告，不能静默退化成 baseline。

### 3.2 TODO：对比 attention layer 的影响

待完成 attention 专项实验，至少包括：

- 确认 Gated Attention 层是否实际暴露 `qkv_linear`、`core_attn` 或 `attn_proj` 的 fine-grained hook，以及 Gated DeltaNet 层的对应限制；
- 将交替层按 layer type 分桶，避免把不支持 attention offload 的 Gated DeltaNet 层错误计入同一平均值；
- 在不改变 3.1 MLP/MoE 约束的前提下，增加 attention baseline 和单项 offload；
- 单独报告 attention 的 copy 开销，并评估它与 MLP/MoE offload 同时开启时的相互影响。

## 4. 观测指标与判据

先确认配置校验通过、loss/梯度无 NaN/Inf，再比较性能。每个 step/stage 记录 forward、backward、optimizer 时间；模块级 D2H/H2D bytes、duration、stream、event、同步点和 overlap；GPU peak/allocated/reserved memory；CPU、HBM、NVLink/PCIe 带宽；SM、Tensor Core、DRAM 利用率；MoE router imbalance、expert token counts 和 grouped GEMM occupancy。

copy 时间占比高、重叠率低、CPU staging 队列出现空洞或链路带宽饱和，判定为 transfer 瓶颈；显存下降小但 step time 明显增加，优先归因于 transfer 或同步。开启 memory history 时保存 snapshot，并使用现有 snapshot 工具重放峰值分配。

## 5. 交付物

每项实验保存完整 config、环境/GPU 信息、训练日志、profile trace、显存 snapshot 和汇总 CSV/Markdown。最终报告给出：

- dense 与 sparse 各自相对 baseline 的显存-吞吐 Pareto；
- BF16 与 MXFP8 的机制差异；
- expert FC1、MoE activation、dense MLP activation 的瓶颈归因；
- baseline/offload 的行为和数值验证；
- 推荐默认配置、不适用条件，以及 attention 专项实验的 TODO 清单。
