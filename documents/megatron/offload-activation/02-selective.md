# Fine-grained Offload 与 Selective Recomputation

本文只讨论 Transformer layer 内的 module-level activation 处理，不讨论 layer-level `cpu_offloading` 的完整实现。本文中的 **activation** 指某个module 在 forward 中生成、并可能被 autograd 为 backward 保存的内容，**不包括该 module 的输入**。

## 参考文档

- [Megatron-LM: Fine-Grained Activation Offloading](../../../3rdparty/Megatron-LM/docs/user-guide/features/fine_grained_activation_offloading.md)
- [Megatron Bridge: Activation Recompute Skill](../../../skills/nemo-mbridge-perf-activation-recompute/SKILL.md)

源码中的配置定义和实现入口：

- `3rdparty/Megatron-LM/megatron/core/transformer/transformer_config.py`
- `3rdparty/Megatron-LM/megatron/core/transformer/transformer_layer.py`
- `3rdparty/Megatron-LM/megatron/core/transformer/attention.py`
- `3rdparty/Megatron-LM/megatron/core/transformer/multi_latent_attention.py`
- `3rdparty/Megatron-LM/megatron/core/transformer/moe/experts.py`
- `3rdparty/Megatron-LM/megatron/core/tensor_parallel/random.py`

## 1. 基础概念

### 1.1 Selective recompute

Selective recompute 在 `recompute_granularity="selective"` 下通过
`recompute_modules` 选择 module。Megatron 当前允许：

```text
core_attn, moe_act, layernorm, mla_up_proj,
mlp, moe, shared_experts, gdn_norm_out
```

其中：

| 类型 | module | 行为 |
|---|---|---|
| Normal checkpoint | `core_attn`, `mlp`, `moe`, `shared_experts` | 保存 module 输入；forward 不保存内部中间 activation；backward 重新执行 module |
| Discard-output checkpoint | `layernorm`, `moe_act`, `mla_up_proj`, `gdn_norm_out` | 保存 module 输入；释放 module 输出；backward 根据输入重建输出 |

| Module | What it recomputes | Compute cost | Memory savings |
|---|---|---|---|
| `core_attn` | attention softmax/dropout/QKV dot product | low (Flash Attention already recomputes internally) | moderate |
| `layernorm` | layer normalization | negligible (~0%) | negligible |
| `mlp` | full FFN block | high (~16% on Llama3 70B, hidden=28672) | ~3 GB |
| `moe` | MoE expert dispatch | varies | varies |
| `moe_act` | MoE activation functions | low | small |
| `shared_experts` | shared expert layers | moderate | moderate |
| `mla_up_proj` | Multi-Latent Attention up projection | moderate | moderate |

`CheckpointWithoutOutput` 的输出释放和 backward hook 逻辑见
`random.py` 中的 `CheckpointWithoutOutput`。例如 `layernorm` 的输出会在
self-attention 或 MLP 使用之后被释放，而不是把 layernorm 的输入丢掉。

### 1.2 Fine-grained activation offload

Fine-grained offload 在 `fine_grained_activation_offloading=True` 且配置
`offload_modules` 时启用。当前支持：

```text
attn_norm, qkv_linear, core_attn, attn_proj,
mlp_norm, expert_fc1, moe_act, fused_group_mlp
```

配置语义是“offload module input”。实现上，module group 通过
`saved_tensors_hooks` 捕获该范围内 autograd 保存的 CUDA tensors，在 forward
结束后异步 D2H，在 backward 到达该 group 前 H2D reload。因此 offload 列
中的“输入”是该机制的边界语义；具体传输对象仍受 tensor 大小、TE
`do_not_offload`、offload fraction 等过滤条件影响。

## 2. Module activation

下列四张表严格按“module 生成的内容”描述 activation。`input` 单独列出，
activation 列只写该 module 的输出或内部生成值。表格中的合并单元格表示同一
module 或同一个 discard-output checkpoint 覆盖多行 activation；offload 列不
默认合并，因为一个 offload module 只对应某个具体的后继输入边界。

### Self-attention

<table>
<thead><tr><th>module</th><th>input</th><th>activation</th><th>recompute</th><th>offload</th><th>备注与配置要求</th></tr></thead>
<tbody>
<tr><td rowspan="2"><code>attn_norm</code></td><td rowspan="2"><code>mlp.output</code><br>[seq_len × batch, hidden_size]</td><td><code>attn_norm.rstdevs</code><br>[seq_len × batch]</td><td rowspan="2"><code>attn_norm</code>（discard）<br>从 <code>mlp.output</code> 重建两项输出</td><td></td><td rowspan="2"><code>layernorm</code> recompute；<code>attn_norm</code> offload 的边界输入是 <code>mlp.output</code>；IdentityOp 时跳过</td></tr>
<tr><td><code>attn_norm.output</code><br>[seq_len × batch, hidden_size]</td><td><code>qkv_linear</code></td></tr>
<tr><td rowspan="3"><code>qkv_linear</code></td><td rowspan="3"><code>attn_norm.output</code><br>[seq_len × batch, hidden_size]</td><td><code>qkv_linear.query</code><br>[seq_len × batch, num_query_heads, head_dim]</td><td rowspan="3">无独立 selective recompute；full-layer recompute 可间接重算</td><td><code>core_attn</code></td><td rowspan="3"><code>qkv_linear</code> offload 处理输入，而不是直接卸载 Q/K/V 输出</td></tr>
<tr><td><code>qkv_linear.key</code><br>[seq_len × batch, num_kv_heads, head_dim]</td><td><code>core_attn</code></td></tr>
<tr><td><code>qkv_linear.value</code><br>[seq_len × batch, num_kv_heads, head_dim]</td><td><code>core_attn</code></td></tr>
<tr><td rowspan="2"><code>core_attn</code></td><td rowspan="2"><code>qkv_linear.query/key/value</code></td><td><code>core_attn.scores/softmax/dropout</code><br>[batch, heads, seq_len, seq_len]</td><td rowspan="2"><code>core_attn</code>（normal）<br>保存 Q/K/V，重建内部 attention activation</td><td></td><td rowspan="2">fused attention 可能已内部重算；offload <code>attn_proj</code> 时必须同时 offload <code>core_attn</code></td></tr>
<tr><td><code>core_attn.output</code><br>[seq_len × batch, hidden_size]</td><td><code>attn_proj</code></td><td><code>attn_proj</code></td></tr>
<tr><td><code>attn_proj</code></td><td><code>core_attn.output</code><br>[seq_len × batch, hidden_size]</td><td><code>attn_proj.output</code><br>[seq_len × batch, hidden_size]</td><td>无独立 selective recompute</td><td><code>attn_proj</code></td><td><code>attn_proj</code> offload 处理输入；必须与 <code>core_attn</code> offload 配置</td></tr>
<tr><td><code>attn_bda</code></td><td><code>attn_proj.output</code> + residual</td><td><code>attn.output</code><br>[seq_len × batch, hidden_size]</td><td>无</td><td><code>mlp_norm</code></td><td>bias/dropout/add 不是独立 selective module；其输出成为 MLP norm 输入</td></tr>
</tbody>
</table>

### MLA-attention

<table>
<thead><tr><th>module</th><th>input</th><th>activation</th><th>recompute</th><th>offload</th><th>备注与配置要求</th></tr></thead>
<tbody>
<tr><td rowspan="2"><code>attn_norm</code></td><td rowspan="2"><code>mlp.output</code><br>[seq_len × batch, hidden_size]</td><td><code>attn_norm.rstdevs</code><br>[seq_len × batch]</td><td rowspan="2"><code>attn_norm</code>（discard）<br>从 <code>mlp.output</code> 重建两项输出</td><td></td><td rowspan="2">MLA 仍经过 TransformerLayer attention norm；IdentityOp 时跳过；CUDA Graph scope 可能禁用 offload</td></tr>
<tr><td><code>attn_norm.output</code><br>[seq_len × batch, hidden_size]</td><td><code>qkv_linear</code></td></tr>
<tr><td rowspan="2"><code>qkv_linear</code></td><td rowspan="2"><code>attn_norm.output</code><br>[seq_len × batch, hidden_size]</td><td><code>qkv_linear.q_compressed</code><br>[seq_len × batch, q_lora_rank]</td><td rowspan="2">无独立 selective recompute；full-layer recompute 可间接重算</td><td></td><td rowspan="2">MLA 的 down-projection/compressed latent 在此阶段生成；<code>qkv_linear</code> offload 处理输入而不是直接卸载输出</td></tr>
<tr><td><code>qkv_linear.kv_compressed</code><br>[seq_len × batch, kv_lora_rank]</td><td></td></tr>
<tr><td rowspan="3"><code>mla_up_proj</code></td><td rowspan="3"><code>qkv_linear.q_compressed/kv_compressed</code>、position input</td><td><code>mla_up_proj.query</code><br>[seq_len × batch, query_heads, qk_head_dim]</td><td rowspan="3"><code>mla_up_proj</code>（discard）<br>释放并重建 query/key/value 输出</td><td><code>core_attn</code></td><td rowspan="3">需要 <code>multi_latent_attention=True</code>；没有同名 fine-grained offload group</td></tr>
<tr><td><code>mla_up_proj.key</code><br>[seq_len × batch, kv_heads, qk_head_dim]</td><td><code>core_attn</code></td></tr>
<tr><td><code>mla_up_proj.value</code><br>[seq_len × batch, kv_heads, v_head_dim]</td><td><code>core_attn</code></td></tr>
<tr><td rowspan="2"><code>core_attn</code></td><td rowspan="2"><code>query/key/value</code></td><td><code>core_attn.scores/softmax/dropout</code><br>[batch, heads, seq_len, seq_len]</td><td rowspan="2"><code>core_attn</code>（normal）</td><td></td><td rowspan="2">fused attention 可能已内部重算</td></tr>
<tr><td><code>core_attn.output</code><br>[seq_len × batch, hidden_size]</td><td><code>attn_proj</code></td><td><code>attn_proj</code></td></tr>
<tr><td><code>attn_proj</code></td><td><code>core_attn.output</code><br>[seq_len × batch, hidden_size]</td><td><code>attn_proj.output</code><br>[seq_len × batch, hidden_size]</td><td>无独立 selective recompute</td><td><code>attn_proj</code></td><td>必须与 <code>core_attn</code> offload 配置</td></tr>
<tr><td><code>attn_bda</code></td><td><code>attn_proj.output</code> + residual</td><td><code>attn.output</code><br>[seq_len × batch, hidden_size]</td><td>无</td><td><code>mlp_norm</code></td><td>输出进入 pre-MLP norm</td></tr>
</tbody>
</table>

### Dense

<table>
<thead><tr><th>module</th><th>input</th><th>activation</th><th>recompute</th><th>offload</th><th>备注与配置要求</th></tr></thead>
<tbody>
<tr><td rowspan="2"><code>mlp_norm</code></td><td rowspan="2"><code>attn.output</code><br>[seq_len × batch, hidden_size]</td><td><code>mlp_norm.rstdevs</code><br>[seq_len × batch]</td><td rowspan="2"><code>mlp_norm</code>（discard）<br>即 <code>layernorm</code> recompute</td><td></td><td rowspan="2"><code>mlp_norm</code> offload 处理输入；IdentityOp 时跳过；CUDA Graph 边界可能禁用</td></tr>
<tr><td><code>mlp_norm.output</code><br>[seq_len × batch, hidden_size]</td><td></td></tr>
<tr><td><code>mlp.fc1</code></td><td><code>mlp_norm.output</code><br>[seq_len × batch, hidden_size]</td><td><code>mlp.fc1.output</code><br>[seq_len × batch, ffn_hidden_size]；gated MLP 可能包含两路结果</td><td><code>mlp</code>（normal）间接丢弃并重算</td><td></td><td>当前没有 dense <code>mlp.fc1</code> fine-grained offload group</td></tr>
<tr><td><code>mlp.act</code></td><td><code>mlp.fc1.output</code></td><td><code>mlp.act.output</code><br>[seq_len × batch, ffn_hidden_size/2]（gated 时依实现而定）</td><td><code>mlp</code>（normal）间接重算</td><td></td><td>当前没有 dense <code>mlp_act</code> offload 或独立 selective recompute</td></tr>
<tr><td><code>mlp.fc2</code></td><td><code>mlp.act.output</code></td><td><code>mlp.fc2.output</code><br>[seq_len × batch, hidden_size]</td><td><code>mlp</code>（normal）间接重算</td><td></td><td>输出进入 bias/dropout/add；没有 dense <code>mlp.fc2</code> offload group</td></tr>
<tr><td><code>mlp_bda</code></td><td><code>mlp.fc2.output</code> + residual</td><td><code>mlp.output</code><br>[seq_len × batch, hidden_size]</td><td>无</td><td><code>attn_norm</code>（下一层）</td><td>下一层 <code>attn_norm</code> offload 处理该 layer output 作为输入；原始中间值可回收</td></tr>
</tbody>
</table>

### MoE

<table>
<thead><tr><th>module</th><th>input</th><th>activation</th><th>recompute</th><th>offload</th><th>备注与配置要求</th></tr></thead>
<tbody>
<tr><td rowspan="2"><code>mlp_norm</code></td><td rowspan="2"><code>attn.output</code><br>[seq_len × batch, hidden_size]</td><td><code>mlp_norm.rstdevs</code><br>[seq_len × batch]</td><td rowspan="2"><code>mlp_norm</code>（discard）</td><td></td><td rowspan="2">MoE CUDA Graph scope 可能禁用 norm offload/recompute</td></tr>
<tr><td><code>mlp_norm.output</code><br>[seq_len × batch, hidden_size]</td><td></td></tr>
<tr><td><code>shared_experts</code>（可选分支）</td><td><code>mlp_norm.output</code></td><td><code>shared_experts.output</code><br>[seq_len × batch, hidden_size]</td><td><code>shared_experts</code>（normal）</td><td></td><td>没有 shared-expert fine-grained offload 名称；<code>shared_experts</code> recompute 与 <code>moe_shared_expert_overlap</code> 互斥，同时开启会报错</td></tr>
<tr><td rowspan="2"><code>route</code> / <code>dispatch</code></td><td rowspan="2"><code>mlp_norm.output</code><br>[seq_len × batch, hidden_size]</td><td><code>routing_probs</code>、<code>routing_map</code><br>[token, top_k] / model-specific</td><td rowspan="2"><code>moe</code>（normal）可整体重算 route、preprocess、dispatch、expert、combine</td><td></td><td rowspan="2">没有 route/dispatch 独立 offload group</td></tr>
<tr><td><code>dispatch.output</code> = <code>D</code><br>[dispatched_tokens, hidden_size]</td><td><code>expert_fc1</code></td></tr>
<tr><td><code>expert_fc1</code></td><td><code>D</code><br>[dispatched_tokens, hidden_size]</td><td><code>expert_fc1.output</code> = <code>E1</code><br>[dispatched_tokens, ffn_hidden_size]</td><td>无独立 recompute</td><td><code>moe_act</code></td><td><code>expert_fc1</code> offload 实际处理输入 <code>D</code>；<code>moe_act</code> offload 处理本行生成的 <code>E1</code></td></tr>
<tr><td><code>moe_act</code></td><td><code>E1</code><br>[dispatched_tokens, ffn_hidden_size]</td><td><code>moe_act.output</code> = <code>Eact</code><br>[dispatched_tokens, ffn_hidden_size/2]（gated 时依实现而定）</td><td><code>moe_act</code>（discard）<br>保存 <code>E1</code>，丢弃并重建 <code>Eact</code></td><td></td><td>需要 <code>moe_grouped_gemm=True</code>；<code>moe_act</code> offload 与 recompute 可同时配置，分别处理 <code>E1</code> 与 <code>Eact</code></td></tr>
<tr><td><code>expert_fc2</code></td><td><code>Eact</code></td><td><code>expert_fc2.output</code> = <code>E2</code><br>[dispatched_tokens, hidden_size]</td><td>无独立 recompute</td><td></td><td><code>E2</code> 是 combine 输入；combine 完成后原始 <code>E2</code> 可回收，后续使用 combine 输出</td></tr>
<tr><td rowspan="2"><code>combine</code> / <code>postprocess</code></td><td rowspan="2"><code>E2</code> + <code>shared_experts.output</code></td><td><code>combine.output</code> = <code>C</code><br>[seq_len × batch, hidden_size]</td><td rowspan="2"><code>moe</code>（normal）可整体间接重算</td><td></td><td rowspan="2">没有独立 combine offload group；combine backward 负责把梯度传回 <code>E2</code></td></tr>
<tr><td><code>moe.output</code>（加上 shared expert 后）<br>[seq_len × batch, hidden_size]</td><td><code>mlp_bda</code></td></tr>
<tr><td><code>mlp_bda</code></td><td><code>moe.output</code> + residual</td><td><code>mlp.output</code><br>[seq_len × batch, hidden_size]</td><td>无</td><td><code>attn_norm</code>（下一层）</td><td>下一层 attention norm 可处理该 layer output 作为输入</td></tr>
</tbody>
</table>

## 四类模型路径

### Self-attention

标准路径为：

```text
H -> attn_norm -> N -> qkv_linear -> Q,K,V
  -> core_attn -> O -> attn_proj -> P
```

这里最容易混淆的是 `N`：

```text
attn_norm recompute:
    保存 H，释放输出 N

qkv_linear offload:
    offload qkv_linear 的输入 N
```

二者都能降低 `N` 的 GPU 常驻量，但一个靠重新执行 layernorm，一个靠
CPU 存储。`core_attn` 则是另一种边界：recompute 主要丢弃 attention 内部
score/softmax/dropout 等中间 activation；offload 处理其 Q/K/V 输入。

### MLA attention

MLA 额外包含压缩 latent、up projection 和 RoPE applying。当前 selective
recompute 提供 `mla_up_proj`，使用 discard-output checkpoint；fine-grained
offload 没有同名 group，MLA 的输入仍可通过 `qkv_linear`、`core_attn` 和
`attn_proj` 的 offload 边界处理。

### Dense MLP

```text
M -> fc1 -> activation -> fc2 -> bias/dropout/add
```

当前 fine-grained offload 没有 dense `fc1`、activation、`fc2` 三个 group。
选择性 recompute 只有整个 `mlp`，因此 FC1 和 activation 的 activation 只能
通过整个 MLP 的 normal checkpoint 间接节省。`mlp_norm` 的 recompute 只处理
其输出 `M`，不能等价替代整个 MLP recompute。

### MoE

```text
M -> route/preprocess/dispatch -> expert_fc1 -> moe_act -> expert_fc2
  -> combine/postprocess
```

最重要的三个 activation 边界是：

```text
D    = dispatch 输出 = expert_fc1 输入
E1   = expert_fc1 输出 = moe_act 输入
Eact = moe_act 输出 = expert_fc2 输入
E2   = expert_fc2 输出 = combine 输入
```

对应关系为：

```text
expert_fc1 offload: D
moe_act offload:    E1
moe_act recompute:  Eact
combine 完成后:     E2 可回收
```

因此 `moe_act` 的 offload 和 recompute 处理的是不同 activation：前者搬运
输入 `E1`，后者丢弃输出 `Eact`。如果选择 `moe` 整体 recompute，则不能再
对其内部的 `expert_fc1`、`moe_act` 或 `fused_group_mlp` 配置 offload，源码
会直接拒绝这种组合。

## 关键配置冲突

- fine-grained offload 需要 `fine_grained_activation_offloading=True` 和非空 `offload_modules`。
- selective recompute 需要 `recompute_granularity="selective"` 和 `recompute_modules`。
- `attn_proj` offload 必须同时有 `core_attn` offload。
- `fused_group_mlp` 需要 Transformer Engine op fuser，且不能与 `expert_fc1` 或 `moe_act` offload 同时使用。
- `moe` 整体 recompute 不能与 MoE 内部 `expert_fc1`、`moe_act`、`fused_group_mlp` offload 同时使用。
- `moe_act` recompute 需要 grouped GEMM；`mla_up_proj` recompute 需要 MLA；`layernorm`/`moe_act` 的 FP8 支持还受 Transformer Engine 版本和 FP8 recipe 约束。
- **`shared_experts` recompute 与 `moe_shared_expert_overlap` 互斥**：当 `moe_shared_expert_overlap=True` 时，不能配置 `recompute_modules` 包含 `shared_experts`，源码会直接抛出 `ValueError`。
- fine-grained offload 与 layer-level `cpu_offloading` 互斥；它们不是同一套 offload manager。
- CUDA Graph 会改变部分 boundary 的可用性：`attn_norm`、`mlp_norm` 可能被禁用，图内 offload 还要求满足对应 PyTorch/Transformer Engine 版本。

## CUDA Graph 与 Offload、Recompute 的互斥关系

CUDA Graph 通过 `cuda_graph_modules` 配置捕获范围，可选值包括：`attn`、`mlp`、`moe`、`moe_router`、`moe_preprocess`、`mamba`。空列表表示整层捕获。

### 1. CUDA Graph 实现方式

| 实现 (`cuda_graph_impl`) | 说明 |
|--------------------------|------|
| `local` | 逐层本地实现，支持 partial MoE graph |
| `transformer_engine` | TE 实现，不支持 partial MoE graph |
| `full_iteration` | 整迭代捕获，不支持 per-layer scope |
| `none` | 禁用 CUDA Graph |

### 2. MLP（密集层）的约束

#### 2.1 Scope 限制

- `mlp` scope **仅适用于密集层**（非 MoE），纯 MoE 模型中配置会报错
- `mlp_norm` offload 的可用性受 cuda graph scope 影响：

| CUDA Graph 配置 | `mlp_norm` offload 状态 |
|-----------------|------------------------|
| `attn` 在 scope，但 `mlp` 不在 | 不支持（边界冲突） |
| `attn` 不在 scope，但 `mlp` 在 | 不支持（边界冲突） |
| 空 scope（整层捕获） | 条件支持，需 torch>=2.9.0, TE>=2.14.0 |

#### 2.2 Recompute 冲突

`mlp` recompute 与 hidden dropout 在 full cudagraph 下冲突：

```
报错条件：full_cudagraph=True + hidden_dropout!=0 + "mlp" in recompute_modules
"hidden dropout is not supported with graphed MLP recomputation."
```

**约束逻辑**：CUDA Graph 内的随机性（dropout）与 activation checkpoint 的重执行机制不兼容。

### 3. MoE 的约束

#### 3.1 Scope 互斥

| 冲突组合 | 说明 |
|----------|------|
| `moe` + `moe_router` | 不能同时存在，互斥 |
| `moe_preprocess` 无 `moe_router` | `moe_preprocess` 依赖 `moe_router` |
| `moe` + fine-grained offload | Token-drop MoE 不支持 offloading |

#### 3.2 Partial CUDA Graph 与 Recompute

启用 `moe_router` 或 `moe_preprocess` 时自动进入 partial mode：

```python
self.use_partial_cudagraphs = True
self.moe_layer_recompute = (
    self.config.recompute_granularity == 'selective'
    and "moe" in self.config.recompute_modules
    and self.config.cuda_graph_impl == "local"
)
```

**注意**：`transformer_engine` impl 下 `"moe"` recompute 与 `moe_router` cuda graph **不兼容**。

#### 3.3 Offload 支持矩阵

| `cuda_graph_impl` | 支持的 Offload Modules | 说明 |
|-------------------|------------------------|------|
| `local` | 仅 `expert_fc1`, `moe_act`, `fused_group_mlp` | Partial MoE offload 专用 |
| `transformer_engine` | 全部 | 推荐用于复杂组合 |
| `full_iteration` | 全部 | 整层粒度 |

**Token-drop MoE 限制**：`cuda_graph_modules` 包含 `moe` 时，fine-grained offload 会直接报错。

#### 3.4 Shared Experts 特殊约束

GTP + local cuda graph + fp8/fp4 + shared_experts recompute 冲突：

```
报错条件：gtp_weight_remat_size > 1
        + cuda_graph_impl="local"
        + (fp8 或 fp4)
        + moe_shared_expert_intermediate_size 不为 None
        + 非 moe_shared_expert_overlap
        + (full_cudagraph 或 moe/moe_router 在 scope)
        + "shared_experts" in recompute_modules

错误信息："GTP + local CUDA graphs cannot recompute shared_experts under fp8/fp4"
```

**原因**：`te_checkpoint` 需要 `.backward()`，但 local fwd-graph warmup 使用 `.grad()`。

### 4. 最佳实践建议

1. **MoE + Recompute**：优先使用 `cuda_graph_impl="local"` 而非 `"transformer_engine"`，可同时启用 `moe_router` cudagraph 和 `"moe"` recompute

2. **Fine-grained Offload + CUDA Graph**：优先使用 `transformer_engine` 或 `full_iteration` impl，local impl 的 offload 支持有限

3. **避免冲突组合**：
   - 不要同时配置 `moe` 和 `moe_router` cuda graph
   - 不要在 `moe` cuda graph 下使用 activation offloading
   - 注意 `mlp_norm` offload 与 cuda graph scope 的边界冲突
   - GTP 场景下避免 `shared_experts` recompute + local cuda graph + fp8/fp4
