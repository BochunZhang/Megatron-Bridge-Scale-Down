# GatedDeltaNet 激活值 Offload 设计

本文定义 GatedDeltaNet 在 Megatron-Core 中接入 fine-grained activation offload 的首版方案。
范围限定为普通训练路径：

- 只讨论 `training=True` 的 forward/backward；
- 不启用 activation recompute；
- 不讨论具体 GDN/FLA backend 的实现和保存张量清单；
- 不覆盖 inference、CUDA graph 以及 layer-level `cpu_offloading`。

本文只设计 Megatron-Core 的 group 边界和调用方式。普通训练配置必须继续满足
Megatron-Core 对 fine-grained activation offload 的既有运行时要求；CUDA graph 和其他
offload 模式需要单独设计和验证。

当前实现入口为：

- `3rdparty/Megatron-LM/megatron/core/ssm/gated_delta_net/gdn.py`
- `3rdparty/Megatron-LM/megatron/core/ssm/gated_delta_net/common.py`
- `3rdparty/Megatron-LM/megatron/core/pipeline_parallel/fine_grained_activation_offload.py`

GatedDeltaNet 当前没有自己的 fine-grained offload group。因此开启
`fine_grained_activation_offloading` 时，GDN 内部算子为 backward 保存的 activation 不会
自动进入 GDN 专用的 offload group。

## 1. Group 语义

新增三个独立的 group 名称：

```text
gdn_in_proj
gdn_core_attn
gdn_out_proj
```

| group | 覆盖范围 | 主要目的 |
| --- | --- | --- |
| `gdn_in_proj` | `in_proj` | offload `in_proj` backward 所需的输入 `hidden_states` |
| `gdn_core_attn` | CP→HP、split/reshape、conv1d、Q/K preparation、`g`/`beta`、gated delta rule、gated output norm、HP→CP | 捕获这段计算中由 autograd 保存的 activation |
| `gdn_out_proj` | `out_proj` | offload `out_proj` backward 所需的输入 `norm_out` |

推荐配置：

```python
config.fine_grained_activation_offloading = True
config.offload_modules = [
    "gdn_in_proj",
    "gdn_core_attn",
    "gdn_out_proj",
]
```

只需要 GDN 主体 activation 时，可以只配置 `gdn_core_attn`。首版沿用标准 attention 的
配置约束：如果启用 `gdn_out_proj`，同时启用 `gdn_core_attn`。这个约束是为了让 GDN
projection 和 core 的 group 顺序固定，避免只实现一半 group 时出现不同的生命周期策略。

三个 group 的开关应只由配置决定，实际 forward 中还要加 `self.training` 门控：

```python
self.offload_gdn_in_proj = (
    self.config.fine_grained_activation_offloading
    and "gdn_in_proj" in self.config.offload_modules
)
self.offload_gdn_core = (
    self.config.fine_grained_activation_offloading
    and "gdn_core_attn" in self.config.offload_modules
)
self.offload_gdn_out_proj = (
    self.config.fine_grained_activation_offloading
    and "gdn_out_proj" in self.config.offload_modules
)
```

## 2. 普通训练 forward 的 group 边界

GDN 的计算顺序为：

```text
hidden_states
  -> in_proj
  -> CP→HP all-to-all
  -> split / reshape
  -> causal conv1d
  -> Q/K preparation
  -> g / beta
  -> gated delta rule
  -> gated output norm
  -> HP→CP all-to-all
  -> out_proj
```

group 的生命周期必须与这三个边界一致：

```text
start gdn_in_proj(hidden_states)
  in_proj
commit gdn_in_proj(qkvzba)

start gdn_core_attn(qkvzba)
  CP→HP ... gated output norm ... HP→CP
commit gdn_core_attn(norm_out)

start gdn_out_proj(norm_out)
  out_proj
commit gdn_out_proj(out)
```

group manager 只在训练态启用：

```python
gdn_core_manager = off_interface(
    self.offload_gdn_core and self.training,
    qkvzba,
    "gdn_core_attn",
)
```

其他两个 group 使用相同的 `and self.training` 门控。这样 eval forward 不会创建训练态
offload group，也不会让训练迭代的 group 序列混入非训练调用。

## 3. 为什么 group 能捕获 GDN 内部 activation

`FineGrainedActivationOffloadingInterface` 在 group scope 内安装
`torch.autograd.graph.saved_tensors_hooks`。在 scope 内执行的 autograd operation，只要通过
PyTorch 的保存机制记录 backward 所需的 Tensor，就会被当前 group 记录，并在 backward
需要时 reload。

因此 `gdn_core_attn` 应覆盖从 CP→HP 到 HP→CP 的完整计算范围。这样可以统一捕获：

- CP all-to-all、conv1d 和 Q/K preparation 的 saved tensors；
- gated delta rule 实现保存的 backward inputs；
- gated output norm 及其前后的 autograd saved tensors。

这里不把某个 backend 的 `ctx.save_for_backward()` 清单写入设计，也不根据某个 backend
推导固定的显存公式。不同实现可以保存不同的 Tensor；正确性标准是所有需要 backward
的 Tensor 都能在对应 group 中完成保存和 reload。

## 4. 代码修改方案

### 4.1 `transformer_config.py`

文件：

```text
3rdparty/Megatron-LM/megatron/core/transformer/transformer_config.py
```

在 `offload_modules` 的说明和 `allowed_modules` 中加入：

```text
gdn_in_proj
gdn_core_attn
gdn_out_proj
```

增加配置约束：

```python
if "gdn_out_proj" in self.offload_modules and "gdn_core_attn" not in self.offload_modules:
    raise ValueError(
        "gdn_out_proj cannot be set without gdn_core_attn for the initial GDN offload path."
    )
```

不要在这里复制 layer-level `cpu_offloading` 或其他既有约束；这些约束继续由现有配置校验负责。CUDA graph 组合不属于本文首版范围，若目标分支允许该组合，应在实现
完成前显式拒绝，避免把未接入的 GDN group 当作已支持的 graph offload。

### 4.2 `common.py`

文件：

```text
3rdparty/Megatron-LM/megatron/core/ssm/gated_delta_net/common.py
```

在 `_GDNBase.__init__()` 中初始化三个配置开关，位置和命名与标准 attention 的 offload
开关保持一致。开关本身不读取运行时 global process group，也不负责创建 manager。

### 4.3 `gdn.py`：`in_proj`

增加：

```python
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
```

将原来的 `in_proj` 包在 group 中：

```python
gdn_in_proj_manager = off_interface(
    self.offload_gdn_in_proj and self.training,
    hidden_states,
    "gdn_in_proj",
)
with gdn_in_proj_manager as hidden_states:
    qkvzba, _ = self.in_proj(hidden_states)
qkvzba = gdn_in_proj_manager.group_offload(
    qkvzba,
    forced_released_tensors=[],
)
```

`qkvzba` 要作为下一个 `gdn_core_attn` group 的输入继续使用，因此这里不手动释放它。

### 4.4 `gdn.py`：GDN core

在 `in_proj` group commit 后，创建一个从 `qkvzba` 开始的 core group，并把以下现有代码
放入同一个 `with` 块：

- CP→HP all-to-all；
- transpose、split、reshape 和 causal conv1d；
- `_prepare_input_for_gated_delta_rule()`；
- `_compute_g_and_beta()`；
- `self.gated_delta_rule()`；
- `_gated_norm_and_a2a()`。

结构如下：

```python
gdn_core_manager = off_interface(
    self.offload_gdn_core and self.training,
    qkvzba,
    "gdn_core_attn",
)
with gdn_core_manager as qkvzba:
    # 原有的 CP→HP、conv、输入准备、gated delta rule 和 norm/a2a 代码
    norm_out = self._gated_norm_and_a2a(
        core_attn_out,
        gate,
        thd_cp_a2a_inv,
        batch,
        seq_len,
        packed_seq_params,
    )

norm_out = gdn_core_manager.group_offload(
    norm_out,
    forced_released_tensors=[query, key, value, core_attn_out, gate],
)
```

强制释放列表只放在 forward 中已经完成最后一次使用的 Tensor：

- `query`、`key`、`value`：在 gated delta rule 调用完成后不再被后续 forward 使用；
- `core_attn_out`、`gate`：在 `_gated_norm_and_a2a()` 返回后不再被后续 forward 使用。

不要把 group 的输出 `norm_out` 放进这个列表；它是下一个 `gdn_out_proj` group 的输入。
也不要为了估算内存而手动 resize 任何 autograd saved tensor。saved-tensor hook 负责 saved
activation 的 D2H/H2D 生命周期，force release 只处理仍被 Python 引用但已完成最后一次
forward 使用的 live Tensor。

### 4.5 `gdn.py`：`out_proj`

将原来的 `out_proj` 包在独立 group 中：

```python
gdn_out_proj_manager = off_interface(
    self.offload_gdn_out_proj and self.training,
    norm_out,
    "gdn_out_proj",
)
with gdn_out_proj_manager as norm_out:
    out, out_bias = self.out_proj(norm_out)
out = gdn_out_proj_manager.group_offload(
    out,
    forced_released_tensors=[norm_out],
)
```

`norm_out` 是 `out_proj` backward 所需的输入，在 `out_proj` forward 返回后完成最后一次
forward 使用，因此可以在 group commit 后 force release。`out` 作为 group commit 的输出
返回给上层，不应加入 release 列表。

## 5. 普通训练路径的约束

本方案依赖以下前提：

1. 模型已经初始化 fine-grained offload 的 chunk handler；GDN 不能脱离模型的 offload
   生命周期单独创建 manager。
2. 每个 group 的 start/commit 顺序在 warmup 和 steady state 中保持一致；三个 GDN group
   的名称必须在每个训练 iteration 中按相同顺序出现。
3. 首版不承诺 CUDA graph 下的 event、stream 和 TE replay 同步；该组合需要先补充
   TransformerLayer 集成和专门测试。

## 6. 测试和验证

### 6.1 配置测试

上游 Megatron-LM 的配置测试应放在：

```text
tests/unit_tests/transformer/test_transformer_config.py
```

至少覆盖：

- 三个新名称在 `fine_grained_activation_offloading=True` 时通过校验；
- `gdn_out_proj` 单独配置时触发依赖错误；
- 默认配置不受影响；
- 与既有 `cpu_offloading` 配置的互斥校验仍然生效。

### 6.2 普通训练数值对齐

使用相同参数和输入分别运行：

```text
offload disabled
offload_modules=["gdn_core_attn"]
offload_modules=["gdn_in_proj", "gdn_core_attn", "gdn_out_proj"]
```

比较 forward 输出、bias、hidden states gradient，以及 `in_proj`、`out_proj`、`A_log` 和
`dt_bias` 的 gradient。测试至少覆盖普通 `bshd` 和 packed `thd` 输入；允许正常的低精度
数值误差，但不能出现 NaN、shape mismatch 或 backward deadlock。

### 6.3 确认测试确实发生 offload

单独实例化 GDN 并做一次 forward/backward 不足以证明 offload 生效。测试必须使用完整
模型，或显式初始化 `PipelineOffloadManager` 的 chunk handler，并完成：

1. reset offload singleton；
2. warmup forward/backward，建立 group 序列；
3. reset 触发 warmup summary；
4. 再运行 steady-state forward/backward；
5. 检查 `gdn_in_proj`、`gdn_core_attn`、`gdn_out_proj` 的 offload summary 和实际 reload。

测试配置应把 `min_offloaded_tensor_size` 调低到能覆盖小型 fixture 的值，并使用至少两个
相同名称的 group occurrence；offload manager 可能保留最后一个 group，单层 fixture 不能
作为唯一验证依据。

不要把某个理论 saved payload 当作整个 group 的固定字节数。实际 summary 取决于算子保存
的 Tensor、`min_offloaded_tensor_size`、group 保留策略和 `activation_offload_fraction`。

## 7. 文件修改清单

| 文件 | 修改 |
| --- | --- |
| `3rdparty/Megatron-LM/megatron/core/transformer/transformer_config.py` | 注册三个 GDN group 名称和依赖校验 |
| `3rdparty/Megatron-LM/megatron/core/ssm/gated_delta_net/common.py` | 初始化三个 offload 开关 |
| `3rdparty/Megatron-LM/megatron/core/ssm/gated_delta_net/gdn.py` | 引入 offload interface，包裹三个 group |
| `tests/unit_tests/transformer/test_transformer_config.py` | 新 group 名称和非法组合的配置校验 |
| GDN focused test | 普通训练路径的 forward/backward、packed 输入和实际 D2H/H2D 验证 |

本文件不要求修改 FLA backend 或 `fine_grained_activation_offload.py` 的通用实现。实际
Python 改动应在对应 Megatron-LM 上游仓库完成；本文件记录设计和验证要求。

