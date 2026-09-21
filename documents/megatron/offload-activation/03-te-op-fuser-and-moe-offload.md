# TE Op-Fuser 与 MoE Fine-Grained Offload

本文说明 Transformer Engine（TE）op-fuser 与 fine-grained activation
offload 的关系，重点覆盖 MoE expert MLP 的三个 offload 选项：

```text
expert_fc1
moe_act
fused_group_mlp
```

本文讨论的是 `model.fine_grained_activation_offloading`，不是 layer-level
`model.cpu_offloading`。两者是不同的 activation offload 机制；layer-level
`cpu_offloading` 不能与 fine-grained activation offloading 同时开启。

## 1. 三个容易混淆的配置

### 1.1 `moe_grouped_gemm`

```python
cfg.model.moe_grouped_gemm = True
```

多个 local expert 的 GEMM 使用 TE/MCore 的 grouped GEMM 路径。它主要改变
expert GEMM 的执行方式，不等于 FC1、activation、FC2 三段都由 op-fuser
融合，也不等于 expert 权重一定保存为一个连续的 GroupedTensor 参数。

### 1.2 `use_transformer_engine_op_fuser`

```python
cfg.model.use_transformer_engine_op_fuser = True
```

启用 TE operation fuser 后，`TEGroupedMLP` 可以构造如下 fused sequence：

```text
GroupedLinear FC1 -> activation -> GroupedLinear FC2
```

当前实现中，`TEGroupedMLP` 会在初始化阶段检查 fused path 是否支持当前
配置；不支持时直接触发：

```text
Fused GroupedMLP is not supported for this configuration.
```

### 1.3 `moe_single_grouped_weight`

```python
cfg.model.moe_single_grouped_weight = True
```

这个选项改变的是 expert 权重的参数布局：

```text
False: weight0, weight1, ..., weightN
True:  weight  # 一个 GroupedTensor 参数
```

它不是 op-fuser 的同义词。当前 Megatron-Core 的训练实现要求：

```text
moe_single_grouped_weight=True
    => moe_grouped_gemm=True
    => use_transformer_engine_op_fuser=True
```

此外还要求 Transformer Engine >= 2.14，并且主权重路径为支持的 BF16/FP16、
MXFP8 或 NVFP4 等路径。相关校验在：

```text
3rdparty/Megatron-LM/megatron/core/transformer/transformer_config.py
```

如果没有开启 op-fuser，配置校验会拒绝 `moe_single_grouped_weight`，因为
当前非 op-fuser TE GroupedLinear 训练路径会按 expert 拆分 grouped parameter，
不支持 single-grouped-weight 训练。

## 2. Op-fuser 与 MoE offload 支持矩阵

假设：

```text
fine_grained_activation_offloading=True
use_transformer_engine_op_fuser=True
```

| `offload_modules` 选项 | 与 TE op-fuser 的关系 | 说明 |
|---|---|---|
| `expert_fc1` | **不支持** | 需要在 FC1 边界单独 offload 输入；当前 fused sequence 没有可用的独立边界 |
| `moe_act` | **不支持** | 需要在 activation 边界单独 offload FC1 输出；当前 fused sequence 不支持该局部边界 |
| `fused_group_mlp` | **支持** | offload 整个 fused grouped MLP 的输入，适配 op-fuser 路径 |
| `attn_norm`、`qkv_linear`、`core_attn`、`attn_proj` | 通常不直接阻断 | 它们不属于 `TEGroupedMLP` 的局部 expert FC1/activation 检查，但仍需满足各自 offload 和 CUDA Graph 约束 |
| `mlp_norm`、`mlp_act` | 通常不直接阻断 | dense MLP 边界与 MoE fused grouped MLP 是不同模块；实际组合仍需通过全局配置校验 |

代码中的直接判断是：只要 `offload_expert_fc1` 或 `offload_moe_act` 为真，
`_is_fused_impl_supported()` 就返回 `False`：

```python
if self.offload_expert_fc1 or self.offload_moe_act:
    return False
```

因此以下配置会失败：

```text
use_transformer_engine_op_fuser=True
offload_modules=[expert_fc1]
```

以及：

```text
use_transformer_engine_op_fuser=True
offload_modules=[moe_act]
```

这不是说 `moe_grouped_gemm` 不能与 `expert_fc1`/`moe_act` offload 共存。
它们可以在 **非 op-fuser** 的 TE GroupedMLP 路径上共存：

```text
moe_grouped_gemm=True
use_transformer_engine_op_fuser=False
offload_modules=[expert_fc1, moe_act]
```

此时 grouped GEMM 仍然有效，但 FC1、activation、FC2 不会走 TE op-fuser
的 fused sequence。

## 3. `fused_group_mlp` 的正确用法

`fused_group_mlp` 是为 op-fuser 路径提供的整体 offload 边界。它的语义是：

```text
offload fused grouped MLP 的输入
    ↓
GroupedLinear FC1 -> activation -> GroupedLinear FC2
    ↓
提交整个 fused group 的 offload/release
```

典型配置为：

```python
cfg.model.fine_grained_activation_offloading = True
cfg.model.use_transformer_engine_op_fuser = True
cfg.model.offload_modules = ["fused_group_mlp"]
```

源码中的 fused forward 会围绕整个 grouped MLP 建立 offload manager，并在
fused ops 执行后提交 group offload，而不是在 FC1 或 activation 中间插入
独立的 partial offload hook。

`fused_group_mlp` 有两个硬约束：

1. 必须开启 `use_transformer_engine_op_fuser=True`；
2. 不能与 `expert_fc1` 或 `moe_act` 同时出现在 `offload_modules` 中。

因此下面的组合无效：

```text
offload_modules=[fused_group_mlp, expert_fc1]
offload_modules=[fused_group_mlp, moe_act]
```

配置校验会报告 fused group 与 partial MoE offload 不能组合。

## 4. `expert_fc1` 的非 fused 路径

在非 op-fuser 路径中，`expert_fc1` offload 的调用边界是：

```python
with expert_fc1_manager as permuted_local_hidden_states:
    fc1_output, bias_parallel = linear_fc1(
        permuted_local_hidden_states, tokens_per_expert
    )
```

随后 MCore 对该 group 提交 offload，并释放或恢复对应的输入 tensor。这个
边界需要 FC1 作为独立的可调用模块存在，因此不能直接套用将 FC1、activation、
FC2 封装成一个 TE op-fuser sequence 的实现。

在 BF16 等非 MXFP8/NVFP4 路径中，代码还会让 FC1 保存原始输入，以避免 offload
和量化/转换后的输入之间产生额外的 CPU copy 负担。这个行为与 partial
`expert_fc1` offload 的生命周期管理相关。

## 5. 与 `moe_single_grouped_weight` 的关系

推荐把两种目标分开测试：

### 5.1 Partial expert activation offload

```text
moe_grouped_gemm=True
use_transformer_engine_op_fuser=False
moe_single_grouped_weight=False
offload_modules=[expert_fc1, moe_act]
```

这个组合适合测试 FC1 输入、MoE activation 的细粒度 offload。expert 权重
通常仍按 `weight0`、`weight1` 等 per-expert 参数注册。

### 5.2 Fused grouped MLP 与 single grouped weight

```text
moe_grouped_gemm=True
use_transformer_engine_op_fuser=True
moe_single_grouped_weight=True
offload_modules=[fused_group_mlp]
```

这个组合适合测试：

- TE fused grouped MLP；
- single grouped weight 参数布局；
- 整个 fused grouped MLP 的 activation offload。

它不能把 `expert_fc1` 或 `moe_act` 再作为独立 partial offload 边界加入。

### 5.3 只开启 `moe_single_grouped_weight`

以下配置在当前实现中不成立：

```text
moe_single_grouped_weight=True
use_transformer_engine_op_fuser=False
```

配置校验会直接报错，而不是自动回退到非 op-fuser 的 single grouped weight
训练路径。

## 6. 相关全局约束

以下约束不专属于 op-fuser，但会影响组合测试：

- `fine_grained_activation_offloading` 不能与 layer-level `cpu_offloading`
  同时开启；
- layer-level `cpu_offloading` 需要 PP=1，且不能与 activation recompute 或
  CUDA Graph 组合；
- selective recompute 的整个 `moe` 模块不能再同时 offload 其内部的
  `expert_fc1`、`moe_act` 或 `fused_group_mlp`；
- `moe_paged_stash=True` 时，不能再配置 `expert_fc1`、`moe_act` 或
  `fused_group_mlp`，因为 paged stash 已经接管这些 activation；
- `moe_single_grouped_weight=True` 还受 TE 版本、量化 recipe、GroupedLinear
  API 和模型并行配置限制。

## 7. 快速判断规则

```text
想卸载 FC1 输入或 FC1 输出之间的 activation？
    -> 使用 expert_fc1 / moe_act
    -> 关闭 TE op-fuser

想使用 TE fused grouped MLP？
    -> 开启 use_transformer_engine_op_fuser
    -> 使用 fused_group_mlp 作为整体 offload 边界

想让 expert 权重成为一个 GroupedTensor？
    -> 开启 moe_single_grouped_weight
    -> 必须同时开启 use_transformer_engine_op_fuser
```

源码参考：

- `3rdparty/Megatron-LM/megatron/core/transformer/moe/experts.py`
- `3rdparty/Megatron-LM/megatron/core/transformer/transformer_config.py`
- `documents/megatron/offload-activation/02-selective.md`
