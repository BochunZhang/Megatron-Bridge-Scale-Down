# Hook: _root_pre_backward

**源码位置**: `megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py:831`

`_root_pre_backward` 是整个 backward pass 中第一个执行的 FSDP hook。它作为 `_register_fsdp_hooks` 中的闭包定义，负责在反向传播开始前完成全局状态初始化：将所有 FSDP unit 切换到 PRE_BACKWARD 状态、标记 forward all-gather bucket 可释放、收集需要梯度处理的参数集合，以及排队 backward 结束后的清理回调。

## 1. 注册信息

### 1.1 注册等级

Root module 级别。该 hook 属于全局性的 backward 初始化逻辑，不针对某个特定子模块，而是在 backward 开始时对整个模型进行状态设置。

### 1.2 注册对象

注册在两类对象上：

**第一类：`named_modules()` 遍历中与 root 参数数量相同的 module（line 1017-1024）**

```python
for name, module in root_module.named_modules():
    if len(list(module.parameters())) != len(list(root_module.parameters())):
        continue
    self.backward_pre_hooks[f"{name} _root_pre_backward"] = \
        create_custom_backward_hook(module, _root_pre_backward)
```

筛选条件是 `module 参数总数 == root_module 参数总数`。满足条件的有：
- root_module 自身（`named_modules()` 第一项，`name=""`）
- 不增减参数的透传包装层（如 `Sequential` 外壳）

深层子模块参数数必然小于 root 总参数数，会被跳过。

**第二类：循环外兜底注册（line 1025-1027）**

```python
self._root_pre_backward_hook_handle = create_custom_backward_hook(
    module, _root_pre_backward
)
```

此处 `module` 是循环结束后的最后一个迭代变量，即 `named_modules()` DFS 遍历的最后一个子模块（树中最深最右的叶子节点）。

**多点注册的设计意图**：在 pipeline parallelism 等场景中，用户可能不直接调用 `root_module.forward()`，而是调用子模块的 forward。如果只在 root 上注册，hook 的 forward_hook 不会被触发，backward 中就没有 grad hook 被注入。多点注册 + 幂等 flag 解决了这个问题。

### 1.3 注册方式

通过 `create_custom_backward_hook` 实现双重注册机制：

```python
def create_custom_backward_hook(module, custom_backward_handler):
    def forward_hook(_module, inputs, output):
        output = tree_map(lambda t: t.view_as(t) if torch.is_tensor(t) else t, output)
        output_list = []
        if isinstance(output, torch.Tensor):
            output_list = [output]
        elif isinstance(output, (tuple, list)):
            output_list = [t for t in output if isinstance(t, torch.Tensor)]
        torch.autograd.graph.register_multi_grad_hook(
            output_list, lambda grads: custom_backward_handler(_module, grads), mode="any"
        )
        return output
    return module.register_forward_hook(forward_hook)
```

工作流程：
1. 在目标 module 上注册 `register_forward_hook`
2. 每次 forward 时，对 output tensor 执行 `view_as` 创建新视图（确保可区分来源）
3. 在 output tensor 上调用 `register_multi_grad_hook(output_list, handler, mode="any")`
4. 反向传播梯度到达该 tensor 时，handler（即 `_root_pre_backward`）被触发

因此 hook 的实际触发依赖于目标 module 的 forward 被调用且其输出参与了 backward 计算图。

### 1.4 执行时间

在 backward pass 最开始阶段触发。当 loss 的梯度反向传播，到达 root module 输出 tensor 时，`register_multi_grad_hook` 触发 `_root_pre_backward`。由于 root 是计算图的最外层，它的输出 tensor 是第一个收到梯度的，因此这是整个 backward 中第一个执行的 FSDP hook。

幂等 flag `_root_pre_backward_hook_issued` 确保无论哪个注册点先触发，逻辑体只执行一次。flag 在 `_root_post_backward` 结束时重置为 `False`。

## 2. 执行流程

### 2.1 幂等性保护

```python
if self._root_pre_backward_hook_issued:
    return
self._root_pre_backward_hook_issued = True
```

由于 hook 注册在多个 module 上，此 flag 确保整个 backward 中逻辑体只执行一次。

### 2.2 设置 FSDP unit 为 PRE_BACKWARD 状态

```python
if self.ddp_config.data_parallel_sharding_strategy == "optim_grads_params":
    for module in root_module.modules():
        if isinstance(module, tuple(fsdp_unit_modules)):
            module._training_state = TrainingState.PRE_BACKWARD
```

仅在参数分片策略（ZeRO-3 / `optim_grads_params`）下执行。遍历所有 module，只设置 FSDP unit modules（如 `TransformerLayer`、`LanguageModelEmbedding`）的状态。

**作用**：通知 `_post_forward` hook 当前处于 backward 阶段。在 activation recomputation（梯度检查点）期间会重放 forward，`_post_forward` 检测到 `PRE_BACKWARD` 状态后选择 `lazy_release = True`，延迟释放参数，避免释放后 backward 又需要重新 all-gather。

### 2.3 标记 forward all-gather bucket 可释放

```python
ag_pipeline = self.all_gather_pipeline
for bucket_id in range(ag_pipeline.num_buckets):
    group = self.param_and_grad_buffer.parameter_groups[bucket_id]
    if group.fsdp_unit_id is not None:
        ag_pipeline.bucket_can_be_released[
            ag_pipeline.get_bucket_key(bucket_id, bwd=False)
        ] = True
```

将前向阶段 all-gather 的 bucket（`bwd=False`）标记为可释放。这些 bucket 在 forward 期间存放了 unshard 后的完整参数，进入 backward 后 forward 路径不再需要它们。

条件 `group.fsdp_unit_id is not None` 确保只标记属于 FSDP unit 的 bucket，非分片参数组（如非 FSDP unit module 的直属参数）不受影响。

**释放时机**：不立即释放，而是在后续 backward 中某个子模块需要 all-gather 新参数时（`async_bucket_gather` → `recycle_unused_buckets`），实际回收这些标记为可释放的 bucket，实现 just-in-time 显存回收。

### 2.4 收集需要梯度处理的参数

```python
self._params_require_handle_grad = set()
for param_group in self.param_and_grad_buffer.parameter_groups:
    if not param_group.requires_grad:
        continue
    self._params_require_handle_grad |= set(param_group.params)
    for param in param_group.params:
        param.grad_added_to_main_grad = False
```

两件事同时完成：

1. **初始化 `_params_require_handle_grad` 集合**：收集所有 `requires_grad=True` 的参数组中的参数。在 backward 过程中，每个参数的 `post_accumulate_grad_hook` 执行 `_grad_acc` 后会从该集合中移除自身。backward 结束后，集合中剩余的参数由 `_root_post_backward` 兜底处理。

2. **重置 `grad_added_to_main_grad` 标志**：将所有参数的该标志设为 `False`，表示本轮 backward 尚未将 `param.grad` 搬运到 `main_grad`。该标志防止同一参数在一轮 backward 中重复执行梯度累加。

### 2.5 排队 _root_post_backward 回调

```python
torch.autograd.Variable._execution_engine.queue_callback(_root_post_backward)
```

利用 PyTorch autograd execution engine 的 callback 队列，在整个 backward 所有 autograd 节点执行完毕后触发 `_root_post_backward`。该回调负责：

1. 对 `_params_require_handle_grad` 中剩余参数执行 `_grad_acc`（兜底处理遗漏梯度）
2. 根据条件（sharding 策略 / last microbatch / auto_sync）触发 `reduce_gradients`
3. 重置 `_root_pre_backward_hook_issued = False`，为下一个 microbatch 做准备

## 3. 通信 Overlap

`_root_pre_backward` 通过以下机制使能 backward 阶段的通信-计算 overlap pipeline：

### 3.1 状态切换启用 backward AG pipeline

设置所有 FSDP unit 为 `PRE_BACKWARD` 状态后，后续 `_pre_backward_param_unshard` hook 可以正确地为各子模块发起 backward all-gather。同时 `_post_forward`（在 activation recomputation 中重放时）知道不应立即释放参数，避免了 "释放 → 重新 all-gather" 的额外通信开销。

### 3.2 forward bucket 释放为 backward AG 腾出空间

标记 forward bucket 可释放后，backward 阶段的 all-gather pipeline 可以在需要新 bucket 时回收这些空间。这实现了 forward 到 backward 的 bucket 复用：

```
Forward AG bucket (已完成前向) → 标记可释放 → backward 需要时回收 → 用于 backward AG
```

如果不做此标记，backward all-gather 将无法分配 bucket，导致 pipeline stall。

### 3.3 生命周期时序

```
Forward 结束（output tensor 被挂载 grad hook）
  ↓
Backward 开始，梯度到达 root 输出 tensor
  ↓
_root_pre_backward 触发（仅一次）
  ├─ 设置所有 FSDP unit → PRE_BACKWARD
  ├─ 标记 forward AG buckets 可释放
  ├─ 初始化 _params_require_handle_grad
  ├─ 重置 grad_added_to_main_grad
  └─ 排队 _root_post_backward
  ↓
各子模块 backward hooks 依次执行：
  _pre_backward_param_unshard → backward 计算 → grad_acc → reduce
  ↓
Backward 结束
  ↓
_root_post_backward 触发
  ├─ 兜底处理遗漏梯度
  ├─ 条件性 reduce-scatter
  └─ 重置 _root_pre_backward_hook_issued = False
```
