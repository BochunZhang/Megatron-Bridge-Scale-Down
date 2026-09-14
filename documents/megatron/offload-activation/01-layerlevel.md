# Layer-level CPU activation offload

## 机制概览

Layer-level offload 以 Transformer layer 为调度边界，把该 layer 前向过程中为反向传播保存的 activation 异步复制到 CPU pinned memory；反向传播需要它们之前，再异步复制回 GPU。计算仍在 GPU 上执行，CPU 主要承担 activation 存储。

这里的“layer-level”描述的是 **offload/reload 的调度范围**，并不表示把整个 `nn.Module` 调用 `layer.cpu()`，也不表示无条件搬走 layer 的所有中间张量、参数或权重。

## Megatron 的实现路径

Megatron-Core 在 `TransformerBlock` 初始化时调用 `get_cpu_offload_context(...)`，取得：

- `offload_context`：包住每个 Transformer layer 的 forward，捕获 autograd 为 backward 保存的 tensors。
- `group_prefetch_offload_commit_async`：在 layer forward 结束后提交 offload，并参与后续 prefetch/reload 调度。

关键源码：

- `3rdparty/Megatron-LM/megatron/core/transformer/transformer_block.py:298`
- `3rdparty/Megatron-LM/megatron/core/transformer/transformer_block.py:663`
- `3rdparty/Megatron-LM/megatron/core/extensions/transformer_engine.py:3331`

Megatron 的 `transformer_engine.py` 是适配层，实际 CPU activation offload 由 Transformer Engine 的 `get_cpu_offload_context` 执行。底层使用 saved-tensor hooks、CPU pinned buffers、专用 CUDA stream 和 CUDA events 管理 D2H/H2D 传输。

典型前向结构如下：

```python
with offload_context:
    hidden_states = layer(hidden_states)

hidden_states = prefetch_offload_commit_async(hidden_states)
```

## Offload 时序

Layer `i` 的 forward 完成后，提交函数会在 offload stream 上排队 GPU→CPU 拷贝，并记录完成 event。该提交通常是异步的，主计算流可以继续执行后续 layer：

```text
Layer i forward
    ↓
捕获 saved tensors
    ↓
异步 D2H 到 pinned CPU memory
    ↓
Layer i+1 继续计算
```

这不是同步的 `tensor.cpu()`。但如果需要释放 GPU storage、处理引用关系或满足后续依赖，局部仍可能等待 D2H event。实际收益取决于 CPU/GPU 链路带宽、pinned memory、offload stream，以及后续计算能否覆盖传输时间。

## Reload / prefetch 时序

反向传播通常按 layer 的逆序执行。系统会在目标 layer backward 到达之前，提前在 H2D stream 上恢复它的 saved tensors，并记录 reload event：

```text
Backward Layer i+1
    ↓
预取 Layer i 的 H2D
    ↓
等待 reload event（如仍未完成）
    ↓
Layer i backward 使用恢复的 tensors
```

因此，prefetch 是“目标 layer backward 之前启动”，而不是等 backward 已经开始后才同步加载。若 H2D 没能赶在 backward 到达前完成，计算流会在真正访问 tensor 时等待对应 event。

## 实际卸载对象

Layer-level 的调度范围覆盖整个 Transformer layer，但实际搬运对象由 autograd/TE 的 saved-tensor 策略决定，通常必须满足：

- 是 CUDA tensor；
- 不是 `Parameter`；
- 被 autograd 保存给 backward；
- 没有被 TE 标记为 `do_not_offload`；
- 满足实现中的大小、生命周期和调度条件。

因此，默认配置：

```python
cfg.model.cpu_offloading = True
cfg.model.cpu_offloading_num_layers = K
cfg.model.cpu_offloading_activations = True
cfg.model.cpu_offloading_weights = False
```

表示对指定数量的 Transformer layers 启用 activation offload，并默认保留权重在 GPU。`cpu_offloading_weights=True` 时，TE 可以把符合条件的权重也纳入 offload，但仍不是把整个 Python module 搬到 CPU。

## 与 fine-grained activation offload 的区别

两者都使用异步 D2H/H2D 和使用前 event 等待，但控制粒度不同：

| 项目 | Layer-level | Fine-grained |
|---|---|---|
| 调度单位 | Transformer layer | 层内 module/group |
| 主要实现 | Transformer Engine context，由 Megatron 适配 | MCore `PipelineOffloadManager` / `ChunkOffloadHandler` |
| 选择方式 | `cpu_offloading_num_layers` | `offload_modules`、大小阈值、fraction |
| 典型模块选择 | 不能单独指定 attention 或 MLP 子模块 | 可选 `core_attn`、`qkv_linear`、`expert_fc1` 等 |
| Pipeline parallelism | 当前校验要求 `pipeline_model_parallel_size == 1` | 支持 PP 和 interleaved PP |
| 重计算 | 当前 layer-level activation offload 不能与 `recompute_granularity` 组合 | 可与选择性重计算组合，但有模块级约束 |
| 同时启用 | 与 fine-grained 互斥 | 与 layer-level 互斥 |

Fine-grained 的 `activation_offload_fraction` 是 eligible groups 的比例，不是 activation 字节比例；它会根据 group 顺序、大小阈值、PP rank 策略等选择实际 offload group。

## 当前代码的兼容性边界

Transformer config 当前校验包括：

- `cpu_offloading_num_layers` 必须落在合法范围；
- layer-level activation offload 要求 PP=1；
- layer-level activation offload 不能与 activation recomputation 同时使用；
- fine-grained 与 `cpu_offloading` 互斥；
- 普通 CUDA Graph 路径不支持 CPU offload，当前代码对 `full_iteration` 路径保留了例外。

相关校验位于：

`3rdparty/Megatron-LM/megatron/core/transformer/transformer_config.py`

这些是当前实现的工程约束，不是 CPU offload 原理本身的必然限制。

