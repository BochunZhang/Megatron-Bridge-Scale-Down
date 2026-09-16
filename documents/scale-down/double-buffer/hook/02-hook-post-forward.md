# Hook: _post_forward

定义在 `megatron_fsdp.py:871`，是 `_register_fsdp_hooks` 内部的闭包函数。在 FSDP unit module 的 forward 执行完毕后触发，负责释放 all-gather 恢复的完整参数 buffer，降低训练峰值显存。同时配合 activation recomputation 场景，通过 lazy release 避免不必要的重复通信。

## 1. 注册信息

### 1.1 注册等级

仅在 `data_parallel_sharding_strategy != "no_shard"` 时才有实际意义（因为 `no_shard` 不做 all-gather/release）。实际注册逻辑不显式检查 strategy，但 hook 内部调用的 `release_bucket` 仅在参数已被 all-gather 到 unsharded buffer 时才会执行释放。

典型配置：`optim_grads_params`（完全分片策略）。

### 1.2 注册对象

**仅注册在 FSDP unit modules 上**，即 `isinstance(module, tuple(fsdp_unit_modules))` 为 True 的模块。

注册时的筛选逻辑（L955-974）：

```python
fsdp_modules = []
for name, module in root_module.named_modules():
    # 跳过已注册 fsdp_module 的子模块
    if any(is_submodule(module, fsdp_module) for fsdp_module in fsdp_modules):
        continue

    if isinstance(module, tuple(fsdp_unit_modules)):
        fsdp_modules.append(module)
        self.forward_hooks[f"release module {name} parameters"] = (
            module.register_forward_hook(_post_forward, prepend=False)
        )
```

**为什么只注册在 FSDP unit 上**：参数 buffer 以 FSDP unit 为粒度分配 bucket，一个 bucket 对应一个 unit 的所有参数。释放必须以 bucket 为最小单位，无法部分释放子模块的参数。因此 `_post_forward` 在 unit 级别触发，一次性释放整个 unit 的参数 buffer。

对应 GPTModel 结构：

```
GPTModel
├── embedding (LanguageModelEmbedding)          ← 注册 _post_forward
├── decoder
│   ├── layers[0] (TransformerLayer)            ← 注册 _post_forward
│   │   ├── self_attention                      ← 跳过（子模块）
│   │   └── mlp                                 ← 跳过（子模块）
│   ├── layers[1] (TransformerLayer)            ← 注册 _post_forward
│   └── ...
└── output_layer                                ← 不匹配类型，跳过
```

### 1.3 注册方式

通过 PyTorch 的 `module.register_forward_hook(_post_forward, prepend=False)` 注册。

- **`register_forward_hook`**：在 module 的 `forward()` 执行完毕后调用
- **`prepend=False`**：追加到 hook 列表末尾，保证在所有其他 forward hook 之后执行
- 签名：`hook(module, input, output) -> output`，可以修改输出（此处原样返回）

### 1.4 执行时间

| 场景 | 触发时机 | `_training_state` 输入值 |
|------|----------|--------------------------|
| 正常 forward pass | 每个 FSDP unit 的 `forward()` 完成后 | `FORWARD` |
| Activation recomputation forward | backward 阶段重计算 forward 时 | `PRE_BACKWARD` |

在一次完整的训练 iteration 中，每个 FSDP unit 的 `_post_forward` 至少触发一次（正常 forward），如果启用了 activation recomputation 则在 backward 阶段再触发一次（重计算 forward）。

## 2. 执行流程

### 2.1 核心逻辑

```python
@torch.compiler.disable
def _post_forward(module: nn.Module, input: Any, output: Any):
    if module._training_state == TrainingState.PRE_BACKWARD:
        lazy_release = True
    else:
        lazy_release = False
        module._training_state = TrainingState.IDLE

    assert isinstance(module, tuple(fsdp_unit_modules))

    release_module_parameters(module, bwd=False, lazy=lazy_release)
    return output
```

### 2.2 状态判断与 lazy_release 决策

**正常 forward 路径**：

1. `_pre_forward_param_unshard` 将状态设为 `FORWARD`
2. module 执行 forward 计算
3. `_post_forward` 检测到状态不是 `PRE_BACKWARD` → `lazy_release = False`
4. 状态回到 `IDLE`
5. 立即释放参数 buffer

**Activation recomputation 路径**：

1. `_root_pre_backward` 将所有 FSDP unit 状态设为 `PRE_BACKWARD`
2. PyTorch autograd 触发 activation recomputation，重新执行 forward
3. `_post_forward` 检测到 `PRE_BACKWARD` → `lazy_release = True`
4. **不修改状态**（保持 `PRE_BACKWARD`）
5. 参数标记为"可延迟释放"，但暂不释放

### 2.3 release_module_parameters 流程

```python
def release_module_parameters(module, bwd, lazy=False, *unused):
    for param in module.parameters():
        bucket_id = self.param_and_grad_buffer.param_to_param_group[param]
        self.all_gather_pipeline.release_bucket(bucket_id, bwd, lazy=lazy)

    if not self.ddp_config.keep_fp8_transpose_cache:
        release_params_fp8_transpose_cache(module.parameters())
```

步骤：
1. 遍历 unit 内所有参数（递归）
2. 通过 `param_to_param_group` 映射找到每个参数对应的 bucket ID
3. 调用 `all_gather_pipeline.release_bucket` 释放对应 bucket
4. 如果未启用 `keep_fp8_transpose_cache`，清理 FP8 转置缓存

### 2.4 release_bucket 的两种行为

```python
def release_bucket(self, bucket_id, bwd, lazy: bool = False):
    bucket_key = self.get_bucket_key(bucket_id, bwd)
    if self.bucket_status[bucket_key] == BucketStatus.EMPTY:
        return

    if lazy:
        self.bucket_can_be_released[bucket_key] = True
        return

    self.wait_bucket_ready(bucket_id, bwd, empty_ok=True)
    buf = self.buffer.parameter_groups[bucket_id].model_weight_buffer
    buf.free_bucket_storage()
    self.bucket_status[bucket_key] = BucketStatus.EMPTY
```

| 模式 | 行为 | 状态变化 |
|------|------|----------|
| `lazy=False` | 等待通信完成 → 释放 `model_weight_buffer` 存储 | → `EMPTY` |
| `lazy=True` | 仅标记 `bucket_can_be_released = True` | 不变 |

lazy 模式下，实际释放推迟到 pipeline 下次需要分配新 buffer 时通过 `recycle_unused_buckets()` 执行：

```python
def recycle_unused_buckets(self):
    for bucket_key, can_be_released in self.bucket_can_be_released.items():
        if can_be_released:
            bucket_id, is_transpose_weight = bucket_key[0], bucket_key[1]
            self.release_bucket(bucket_id, is_transpose_weight)
            self.bucket_can_be_released[bucket_key] = False
```

### 2.5 非 FSDP unit 模块的 FP8 缓存释放

对于非 FSDP unit 的模块，当满足以下条件时注册独立的 FP8 缓存释放 hook：

- `keep_fp8_transpose_cache = False`
- `data_parallel_sharding_strategy == "optim_grads_params"`

```python
elif (
    not self.ddp_config.keep_fp8_transpose_cache
    and self.ddp_config.data_parallel_sharding_strategy == "optim_grads_params"
):
    self.forward_hooks[f"remove module {name} fp8 transpose cache"] = (
        module.register_forward_hook(_release_module_fp8_transpose_cache, prepend=False)
    )
```

hook 实现：

```python
@torch.compiler.disable
def _release_module_fp8_transpose_cache(module: nn.Module, *unused):
    release_params_fp8_transpose_cache(module.parameters(recurse=False))
```

注意此 hook 只清理浅参数（`recurse=False`）的 FP8 转置缓存，不涉及 bucket 释放。这是因为非 unit 模块的参数 buffer 释放由其所属的 FSDP unit 的 `_post_forward` 统一处理。

## 3. 通信 Overlap

### 3.1 lazy release 与 activation recomputation 的协作

Activation recomputation（梯度检查点）的执行流程：

```
Forward pass:      [Unit A forward] → [Unit B forward] → ...
                        ↓                    ↓
                   _post_forward          _post_forward
                   (lazy=False,           (lazy=False,
                    立即释放)              立即释放)

Backward pass:     _root_pre_backward (设置所有 unit 为 PRE_BACKWARD)
                        ↓
                   [Unit B recompute forward] → [Unit B backward]
                        ↓
                   _post_forward
                   (lazy=True, 延迟释放)
```

### 3.2 为什么需要 lazy release

在 activation recomputation 中，重计算 forward 和 backward 紧密相连：

1. 重计算 forward 需要参数（已通过 `_pre_backward_param_unshard` 获取）
2. 紧接的 backward 同样需要这些参数来计算梯度
3. 如果在重计算 forward 后立即释放，backward 就需要再次 all-gather — 造成冗余通信

通过 `lazy_release = True`：
- 参数 buffer 在重计算 forward 后**不释放**
- backward 直接使用已在显存中的参数
- 释放推迟到后续 unit 需要分配新 buffer 时自动回收

### 3.3 时序图

```
Timeline (activation recomputation for Unit B):
─────────────────────────────────────────────────────────────────
compute:  │ B recompute fwd │ B backward (需要相同参数) │
ag_stream:│ AG(B) ──────────│──── 参数保持可用 ────────│→ recycle
─────────────────────────────────────────────────────────────────
                             ↑                          ↑
                     _post_forward(lazy=True)     recycle_unused_buckets
                     不释放，标记可回收             实际释放
```

对比如果不使用 lazy release：

```
Timeline (无 lazy — 冗余通信):
─────────────────────────────────────────────────────────────────
compute:  │ B recompute fwd │ wait │ B backward │
ag_stream:│ AG(B) │ release │ AG(B) again! │
─────────────────────────────────────────────────────────────────
                              ↑
                        不必要的重复 all-gather
```

### 3.4 内存与通信的权衡

lazy release 的代价是参数 buffer 在显存中多停留一段时间（从重计算 forward 结束到 backward 结束）。但由于 backward 本身就需要这些参数，这段时间内参数被有效利用，不构成真正的"浪费"。收益是避免了一次完整的 all-gather 通信，节省了网络带宽和 GPU 等待时间。
