# Hook: _pre_backward_param_unshard

定义位置：`megatron_fsdp.py:808`

在 backward 计算到达某个模块之前，对该模块的参数执行 all-gather（unshard）。forward 结束后参数已被 reshard/释放，backward 需要完整参数才能计算梯度，因此需要在 backward 开始前重新 gather。

## 1. 注册信息

### 1.1 注册等级

适用于以下 `data_parallel_sharding_strategy`：

| 策略 | 是否注册 |
|------|----------|
| `optim_grads_params` | 是 |
| `optim_grads` | 是 |
| `optim` | 是 |
| `no_shard` | 是（但执行时 `all_gather_and_wait_parameters_ready` 内部直接 return） |

注册函数 `_register_pre_backward_param_unshard_hook` 没有策略判断条件（不像 pre-forward hook 有 `!= "no_shard"` 的 guard），所有策略都会注册该 hook。但在 `no_shard` 策略下，`all_gather_and_wait_parameters_ready` 第一行即 early return：

```python
if self.data_parallel_sharding_strategy == "no_shard":
    return
```

### 1.2 注册对象

源码位置：`megatron_fsdp.py:955-978`

遍历 `root_module.named_modules()`，根据 `enable_fine_grained_param_gather_hook` 配置分两种注册模式：

**正常模式** (`enable_fine_grained_param_gather_hook = False`)：

仅注册在 **FSDP unit modules** 上：

```python
if isinstance(module, tuple(fsdp_unit_modules)):
    if not self.enable_fine_grained_param_gather_hook:
        _register_pre_backward_param_unshard_hook(module)
```

`fsdp_unit_modules` 是用户传入的类列表，典型值为 `[TransformerLayer, LanguageModelEmbedding]`。

**Fine-grained 模式** (`enable_fine_grained_param_gather_hook = True`)：

注册在**所有 module** 上（包括非 FSDP unit 的子模块）：

```python
if self.enable_fine_grained_param_gather_hook:
    _register_pre_backward_param_unshard_hook(module)
```

此代码在 `is_submodule` 跳过检查之前执行，所以所有子模块都会被注册。

**对比表：**

| 模式 | 注册对象 | backward 时 gather 的参数范围 |
|------|----------|------|
| 正常模式 | 仅 FSDP unit modules | `module.parameters()` — 递归收集所有子模块参数 |
| Fine-grained | 所有 modules | `module.parameters(recurse=False)` — 仅直接持有的参数 |

正常模式以 FSDP unit 为粒度做一次大 all-gather；fine-grained 模式以每个子模块为粒度做多次小 all-gather，实现更细的通信-计算 overlap。

### 1.3 注册方式

注册入口：

```python
def _register_pre_backward_param_unshard_hook(module):
    self.backward_pre_hooks[f"all-gather {module._get_name()} parameters"] = (
        create_custom_backward_hook(module, _pre_backward_param_unshard)
    )
```

核心机制是 `create_custom_backward_hook`，它通过 **forward hook + register_multi_grad_hook** 的组合实现 backward pre-hook：

```python
def create_custom_backward_hook(module, custom_backward_handler):
    @torch.compiler.disable
    def forward_hook(_module, inputs, output):
        # 1. view_as 创建新计算图节点，避免输入输出共享同一 tensor
        output = tree_map(
            lambda t: t.view_as(t) if torch.is_tensor(t) else t, output
        )

        # 2. 收集输出中的所有 tensor
        output_list = []
        if isinstance(output, torch.Tensor):
            output_list = [output]
        elif isinstance(output, (tuple, list)):
            output_list = [t for t in output if isinstance(t, torch.Tensor)]

        # 3. 在输出 tensor 上注册 grad hook
        torch.autograd.graph.register_multi_grad_hook(
            output_list,
            lambda grads: custom_backward_handler(_module, grads),
            mode="any"
        )
        return output

    # 注册 forward hook（每次 forward 都会重新挂载 grad hook）
    return module.register_forward_hook(forward_hook)
```

**三层间接调用链：**

```
_register_pre_backward_param_unshard_hook(module)
  → create_custom_backward_hook(module, _pre_backward_param_unshard)
    → module.register_forward_hook(forward_hook)        # 永久注册
      → forward_hook: 每次 forward 执行时
        → register_multi_grad_hook(output_list, handler, mode="any")
          → backward 梯度到达 output tensor 时触发 _pre_backward_param_unshard
```

**关键设计细节：**

1. **为什么不直接用 `register_full_backward_pre_hook`？** PyTorch 的 module backward hook 时机不够精确。通过在 output tensor 上注册 grad hook，能在梯度流到该模块输出的瞬间触发，确保 all-gather 在 backward 计算前完成。

2. **`view_as` 的作用：** 对输出 tensor 执行 `t.view_as(t)` 产生新的计算图节点，避免输入输出共享同一 tensor 导致无法区分层边界。

3. **`mode="any"` 的含义：** 只要任一输出 tensor 收到梯度就立即触发 unshard，不需要等所有输出都有梯度。这保证了最早时机的参数恢复。

4. **每次 forward 重新注册：** forward hook 是永久注册的，每次 forward 执行时都会在新的 output tensor 上重新注册 `register_multi_grad_hook`，因为每次 forward 产生的 output tensor 是不同的计算图节点。

### 1.4 执行时间

触发时机：**backward pass 中，梯度到达某模块的输出 tensor 时**。

具体来说，在 backward 计算图从 loss 向输入方向传播梯度的过程中，当梯度计算到达某个模块的输出节点（即该模块 forward 输出的 tensor）时，`register_multi_grad_hook` 触发 `_pre_backward_param_unshard`。此时该模块的 backward 计算尚未开始，参数正好可以在 backward kernel 执行前被 all-gather 恢复。

执行顺序（单个模块视角）：

```
forward: 模块计算 → forward_hook 在 output 上注册 grad hook
backward: 梯度到达 output → _pre_backward_param_unshard 触发
        → all-gather 参数 → 模块 backward 计算 → 梯度继续传播
```

## 2. 执行流程

### 2.1 Hook 本体逻辑

```python
@torch.compiler.disable
def _pre_backward_param_unshard(module: nn.Module, *unused):
    # Step 1: 标记训练状态为 PRE_BACKWARD
    module._training_state = TrainingState.PRE_BACKWARD

    # Step 2: 确定需要 gather 的参数列表
    if isinstance(module, tuple(fsdp_unit_modules)):
        param_list = list(module.parameters())         # FSDP unit: 递归收集
    else:
        param_list = list(module.parameters(recurse=False))  # 非 FSDP unit: 仅自身

    # Step 3: fine-grained 模式覆盖为非递归
    if self.enable_fine_grained_param_gather_hook:
        param_list = list(module.parameters(recurse=False))

    # Step 4: all-gather 并等待参数就绪
    self.all_gather_and_wait_parameters_ready(
        param_list, prefetch_order=PrefetchOrder.BACKWARD_PASS_ORDER, bwd=True
    )
```

**各步骤详解：**

1. **设置 `TrainingState.PRE_BACKWARD`：** 通知其他 hook（如 `_post_forward`）当前处于 backward 阶段，避免在 activation recomputation 期间错误释放参数。

2. **参数列表收集：** FSDP unit 递归收集所有子模块参数（因为整个 unit 的参数是一起 shard 的）；非 FSDP unit 仅收集直接持有的参数。

3. **Fine-grained 覆盖：** 在 fine-grained 模式下，无论模块类型都只收集直接参数，实现更细粒度的 gather。

4. **all-gather 与等待：** 调用 `all_gather_and_wait_parameters_ready` 发起通信并等待完成。

### 2.2 all_gather_and_wait_parameters_ready 内部流程

源码位置：`megatron_fsdp.py:413-466`

```python
def all_gather_and_wait_parameters_ready(
    self, params, prefetch=True,
    prefetch_order=PrefetchOrder.FORWARD_PASS_ORDER,
    wait_bucket_ready=True, bwd=False,
):
    # 1. no_shard 策略直接返回
    if self.data_parallel_sharding_strategy == "no_shard":
        return

    # 2. 发起异步 all-gather（含 prefetch）
    ag_pipeline = self.all_gather_pipeline
    ag_pipeline.all_gather_params(
        params=params,
        prefetch=prefetch,
        prefetch_order=prefetch_order,
        suggested_AG_prefetch_size=self.suggested_AG_prefetch_size,
        outer_fsdp_group_param_gather=(...),
        bwd=bwd,
    )

    # 3. 等待当前模块的 bucket 就绪
    if wait_bucket_ready:
        for param in params:
            bucket_id = self.param_and_grad_buffer.param_to_param_group[param]
            ag_pipeline.wait_bucket_ready(bucket_id, bwd)
            # FP8 参数：创建 transpose cache
            if bwd and is_float8tensor(param):
                fp8_create_transpose_cache(param)

    # 4. 设置参数标记
    for param in params:
        param.grad_added_to_main_grad = False
        param.__fsdp_param__ = True
        param.overwrite_main_grad = True
```

**关键行为（`bwd=True` 时的区别）：**

- `prefetch_order=BACKWARD_PASS_ORDER`：prefetch 方向从高 bucket_id 向低 bucket_id 搜索（backward 计算顺序是 forward 的逆序）
- `bwd=True`：影响 `get_bucket_key` 的返回值。如果参数组存在 `transpose_weight_buffer`（mxfp8），`bucket_key = (bucket_id, True)` 用于区分 forward 和 backward 的 buffer
- FP8 参数在 backward 等待就绪后会创建 transpose cache，供 backward 矩阵乘法使用

### 2.3 all_gather_pipeline.all_gather_params 内部流程

源码位置：`param_and_grad_buffer.py:3447-3627`

核心步骤：

1. **确定 bucket 列表：** 通过 `param_to_param_group` 映射参数到 bucket_id，排序去重
2. **标记不可释放：** `bucket_can_be_released[key] = False`，防止 prefetch 期间被回收
3. **Prefetch 扩展（BACKWARD_PASS_ORDER）：** 从当前 bucket 向低 id 方向搜索下一个 bucket，累计不超过 `suggested_AG_prefetch_size`（默认 500MB）
4. **过滤已就绪 bucket：** 只对状态为 `EMPTY` 的 bucket 发起 all-gather
5. **异步 all-gather：** 在 `ag_stream` 上执行 coalesced NCCL all-gather，通过 `_coalescing_manager` 聚合同一 data-parallel group 的多个 bucket

```python
# Backward prefetch 方向：从高 bucket_id 向低搜索
# (forward 是从低向高)
if prefetch_order == PrefetchOrder.BACKWARD_PASS_ORDER:
    bucket_id = ag_buckets[-1] - 1
    for i in reversed(ag_buckets[:-1]):
        if i != bucket_id:
            break
        bucket_id -= 1
```

### 2.4 与 activation recomputation 的交互

`_root_pre_backward` hook 在 backward 开始时将所有 FSDP unit 标记为 `PRE_BACKWARD`。这样在 recompute forward 期间：

- `_post_forward` 检测到 `module._training_state == PRE_BACKWARD` 后不会释放参数（设置 `lazy_release = True`）
- 避免不必要的 reshard + all-gather 循环
- 延迟释放的 bucket 在下一次 `all_gather_params` 调用时由 `recycle_unused_buckets()` 统一执行

## 3. 通信 Overlap

### 3.1 Backward Prefetch 机制

backward prefetch 是实现 all-gather 与 backward 计算 overlap 的核心机制。当某个模块的 `_pre_backward_param_unshard` 触发时：

1. **当前模块的参数 all-gather**：等待完成（同步），确保 backward 计算有数据
2. **下一个模块的参数 prefetch**：异步发起，与当前模块的 backward 计算重叠

```
时间线（正常模式，Layer N → Layer N-1 → Layer N-2）：

Layer N:  [all-gather(N)] [wait(N)] [backward compute(N)]
Layer N-1:                     [prefetch(N-1) ←────────overlap────────→] [wait(N-1)] [backward(N-1)]
Layer N-2:                                                                    [prefetch(N-2) ←──overlap──→] ...
```

### 3.2 Prefetch 方向与大小控制

- **方向：** `BACKWARD_PASS_ORDER` 从高 bucket_id 向低 bucket_id 搜索。bucket 的编号顺序与 forward 计算顺序一致，因此 backward 计算是从高到低。
- **大小限制：** `suggested_AG_prefetch_size`（默认 500MB）控制 prefetch 的总数据量，避免过度占用显存。
- **Double buffer 限制：** 如果启用 `fsdp_double_buffer`，prefetch 不能超过 2 个 FSDP unit 的覆盖范围。

### 3.3 异步通信流

all-gather 操作在独立的 `ag_stream`（CUDA stream）上执行：

```python
all_gather_stream = self.ag_stream if self.ag_stream is not None else torch.cuda.current_stream()
all_gather_stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(all_gather_stream):
    with _coalescing_manager(dp_group, async_ops=async_param_gather) as coalescing_event:
        for bucket_id in buckets:
            self.async_bucket_gather(bucket_id, bwd)
```

- `ag_stream.wait_stream(current_stream)`：确保 all-gather 在当前计算完成后才开始
- `wait_bucket_ready` 中的 `param_gather_event.wait()`：在 default stream 上等待 ag_stream 完成
- prefetch 的 bucket 不需要立即 wait，只有当 backward 真正需要该模块参数时才同步

### 3.4 Fine-grained 模式的 Overlap 优势

在 fine-grained 模式下，每个子模块都有独立的 backward pre-hook，实现更细粒度的 overlap：

```
正常模式（FSDP unit 粒度）：
  [all-gather entire layer] [wait] [backward: linear_qkv → linear_proj → layernorm → ...]

Fine-grained 模式（子模块粒度）：
  [AG(linear_qkv)] [wait] [bwd(linear_qkv)]
                      [AG(linear_proj) ←overlap→] [wait] [bwd(linear_proj)]
                                                    [AG(layernorm) ←overlap→] ...
```

Fine-grained 模式可以减少峰值显存（不需要一次性 gather 整个 layer 的所有参数），代价是更多的通信调用次数。
