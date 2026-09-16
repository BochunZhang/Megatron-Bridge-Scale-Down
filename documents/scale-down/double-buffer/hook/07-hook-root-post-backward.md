# Hook: \_root\_post\_backward

**源码位置**: `megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py:764`

`_root_post_backward` 是整个 backward pass 的终结器（finalizer）。它在 autograd engine 完成所有反向计算节点后执行，负责处理遗留梯度、触发最终的 reduce-scatter 通信、重置训练状态，以及在 `model_auto_sync` 模式下完成梯度同步。

## 1. 注册信息

### 1.1 注册等级

Root-level callback。该 hook 不绑定于任何特定模块，而是作为 autograd engine 级别的全局回调，在整个 backward pass 的所有 autograd 节点执行完毕后统一调用。

### 1.2 注册对象

不直接注册在任何 module 上。它由 `_root_pre_backward` 在执行时动态排队：

```python
# _root_pre_backward 中（line 868）
torch.autograd.Variable._execution_engine.queue_callback(_root_post_backward)
```

由于 `_root_pre_backward` 受幂等 flag `_root_pre_backward_hook_issued` 保护，每次 backward pass 中 `_root_post_backward` 只会被 queue 一次。

### 1.3 注册方式

通过 `torch.autograd.Variable._execution_engine.queue_callback` 注册。

这是 PyTorch autograd engine 提供的底层 API，其工作机制为：
- 将一个 callable 加入 autograd engine 内部的 **回调队列**（callback queue）
- 该队列中的所有回调在当前 backward pass 的 **最后一步** 统一执行
- 执行时机是 autograd engine 完成所有 `AccumulateGrad` / `Function.backward()` 节点之后
- 回调按 FIFO 顺序执行（先 queue 的先执行）
- 不依赖任何特定 tensor 的梯度计算，是 "全局完成" 语义

与 module hook（如 `register_full_backward_hook`）不同，`queue_callback` 不绑定到任何模块的输入/输出 tensor 上，因此能保证在所有模块级 backward hook 之后执行。

### 1.4 执行时间

在整个 backward pass 的 **最末尾** 执行。时序如下：

```
backward 开始
  → _root_pre_backward（全局初始化 + queue_callback）
  → 各 FSDP unit 的 _pre_backward_param_unshard（all-gather 参数）
  → autograd 计算各节点梯度
  → 各参数的 post_accumulate_grad_hook（逐参数处理梯度 + 流水线 reduce）
  → 各 FSDP unit 的 _post_backward_release_module（reshard 参数）
  → autograd engine 完成所有节点
  → _root_post_backward（兜底梯度 + 最终 reduce + 状态重置）
```

关键特征：`_root_post_backward` 保证在所有 per-parameter 和 per-module backward hook 之后执行，是 backward pass 中执行顺序最靠后的 FSDP hook。

## 2. 执行流程

`_root_post_backward` 的完整执行逻辑分为 4 个阶段：

### 2.1 阶段一：补漏梯度处理（Gradient Catch-All）

```python
ordered_params = sorted(
    list(self._params_require_handle_grad), key=lambda p: self.param_to_name[p]
)
for param in ordered_params:
    _grad_acc(param)
```

**作用**：对 `_params_require_handle_grad` 中仍然残留的参数执行梯度累积。

**背景**：`_params_require_handle_grad` 在 `_root_pre_backward` 中被初始化为所有需要梯度的参数集合。正常流程中，`post_accumulate_grad_hook`（即 `_process_post_backward_gradients`）会逐参数处理梯度并通过 `discard` 从集合中移除。但以下情况可能导致残留：

- shared parameters（带 `_is_shared=True` 属性），被 `_process_post_backward_gradients` 跳过
- 因 autograd 图结构原因 `post_accumulate_grad_hook` 未被触发的参数

**排序的意义**：按参数名称排序确保跨 rank 的确定性执行顺序，避免 reduce-scatter 因顺序不一致导致死锁。

**`_grad_acc(param)` 核心逻辑**：

| buffer 类型 | 操作 | 语义 |
|------------|------|------|
| sharded（`is_data_distributed=True`） | `param.main_grad.copy_(param.grad)` | 写入 main_grad bucket，等待 reduce-scatter |
| unsharded | `param.main_grad.add_(param.grad)` | 梯度累积（多 microbatch 场景） |

### 2.2 阶段二：触发剩余梯度 reduce 通信

```python
grad_reduce_every_bprop = self.ddp_config.data_parallel_sharding_strategy in [
    "optim_grads", "optim_grads_params",
]
is_last_microbatch = getattr(self, "is_last_microbatch", False)
if grad_reduce_every_bprop or is_last_microbatch or self.model_auto_sync:
    self.grad_reduce_pipeline.reduce_gradients(
        ordered_params,
        suggested_queue_capacity=self.suggested_RS_queue_capacity,
        outer_fsdp_group_grad_reduce=(
            self.dist_index.use_hybrid_fsdp
            and (is_last_microbatch or self.model_auto_sync)
        ),
    )
    self.grad_reduce_pipeline.reset()
```

**reduce 条件判断**：

| 条件 | 含义 | 触发场景 |
|------|------|----------|
| `grad_reduce_every_bprop` | 梯度分片策略（optim_grads / optim_grads_params） | 每次 backward 都必须 reduce，因为梯度 buffer 是分片的 |
| `is_last_microbatch` | 最后一个 microbatch | 梯度累积完成，需要 reduce 后进入 optimizer step |
| `model_auto_sync` | 自动同步模式 | 每次 backward 都 reduce，简化用户侧逻辑 |

三个条件为 OR 关系，满足任一即触发。

**`reduce_gradients` 流程**：

1. 按 bucket_id 排序参数（确保跨 rank 一致性）
2. 将参数标记为 grad ready，加入对应 bucket 的 `bucket_grad_ready_params`
3. 当某 bucket 的所有参数梯度就绪时，发起异步 reduce-scatter
4. 通过 `suggested_queue_capacity` 控制并发 reduce 操作的 buffer 上限

**`outer_fsdp_group_grad_reduce` 参数**：在 hybrid FSDP（多 FSDP group）场景中，仅在 `is_last_microbatch` 或 `model_auto_sync` 时额外触发外层 DP group 的 all-reduce。

**`grad_reduce_pipeline.reset()` 逻辑**：

```python
def reset(self):
    self.wait_for_previous_grad_reduce(0)  # 等待所有异步 reduce 完成
    # 断言所有 bucket 的 grad_ready_params 为空（全部已处理）
    for bucket_id, grad_ready_params in enumerate(self.bucket_grad_ready_params):
        assert len(grad_ready_params) == 0
    # 释放所有 bucket 的存储
    for bucket_id, _ in self.bucket_status.items():
        gbuf = self.get_fsdp_buffer(bucket_id)
        gbuf.free_bucket_storage()
        self.bucket_status[bucket_id] = BucketStatus.EMPTY
```

reset 的核心作用：
- 同步等待所有进行中的异步 reduce-scatter 操作完成
- 验证所有参数梯度已被正确处理
- 释放梯度 bucket 的临时存储，回收显存

### 2.3 阶段三：状态重置

```python
self._root_pre_backward_hook_issued = False
self.microbatch_count += 1
```

- **重置 `_root_pre_backward_hook_issued`**：允许下一次 backward pass 重新触发 `_root_pre_backward`，重新初始化训练状态
- **递增 `microbatch_count`**：跟踪当前 optimization step 内的 microbatch 序号，影响后续 forward 中的 outer FSDP group all-gather 行为（仅 `microbatch_count == 0` 时触发）

### 2.4 阶段四：可选的自动梯度同步

```python
if self.model_auto_sync:
    self.finish_grad_sync()
```

当 `model_auto_sync=True` 时，自动调用 `finish_grad_sync()` 完成完整的梯度同步流程：

```python
def finish_grad_sync(self):
    # 1. 等待异步 reduce-scatter 完成
    self.synchronize_gradient_reduce()
    # 2. 将 reduced 梯度挂载到 optimizer 参数的 .grad 属性
    self.attach_grad_to_optimizer_state()
    # 3. 同步参数 all-gather（overlap_param_gather 模式下）
    if self.ddp_config.overlap_param_gather:
        self.synchronize_param_gather()
    # 4. 替换模块参数为分布式 optimizer 参数
    self._replace_param_with_distributed_if_needed()
    # 5. 重置 microbatch 计数器
    self.microbatch_count = 0
```

`finish_grad_sync()` 的各子步骤：

| 子步骤 | 方法 | 作用 |
|--------|------|------|
| 1 | `synchronize_gradient_reduce()` | 等待所有异步 reduce-scatter/all-reduce 完成 |
| 2 | `attach_grad_to_optimizer_state()` | 调用 `param_and_grad_buffer.update_main_grads()`，将 buffer 中的梯度关联到 optimizer 参数 |
| 3 | `synchronize_param_gather()` | 等待 backward 期间发起的参数 all-gather 完成，并重置 all-gather pipeline |
| 4 | `_replace_param_with_distributed_if_needed()` | 将模块中的 raw 参数替换为 DTensor 分布式参数 |
| 5 | 重置 `microbatch_count = 0` | 标记新 optimization step 开始 |

## 3. 通信 Overlap

### 3.1 延迟同步与 Overlap 收益

在 **非** `model_auto_sync` 模式下，`_root_post_backward` 仅发起异步 reduce-scatter 通信并 reset pipeline，但 **不等待** 通信完成。这带来了关键的性能优化：

```
┌─────────────────────────────────────────────────────────────────────┐
│ model_auto_sync = False（默认，高性能模式）                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  microbatch N backward:                                             │
│    _root_post_backward → reduce_gradients (async) + reset           │
│                                  │                                  │
│  microbatch N+1 forward:         │ ← reduce-scatter 与 forward 重叠 │
│    _pre_forward_param_unshard    │                                  │
│    module.forward()              │                                  │
│                                  ▼                                  │
│  ... 直到 is_last_microbatch 或手动调用 finish_grad_sync()           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

异步 reduce-scatter 在后台 CUDA stream 上执行，与下一个 microbatch 的 forward 计算并行。用户在最后一个 microbatch 结束后手动调用 `finish_grad_sync()` 来同步所有通信。

### 3.2 model_auto_sync 对 Overlap 的影响

当 `model_auto_sync=True` 时，`_root_post_backward` 在发起 reduce 后 **立即** 调用 `finish_grad_sync()`，这会：

1. **阻塞当前 CUDA stream** 等待所有 reduce-scatter 完成
2. 执行 `synchronize_param_gather()` 等待参数 all-gather 完成
3. 重置 `microbatch_count = 0`

```
┌─────────────────────────────────────────────────────────────────────┐
│ model_auto_sync = True（简化模式，牺牲性能）                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  microbatch N backward:                                             │
│    _root_post_backward → reduce_gradients → reset                   │
│                        → finish_grad_sync() ← 阻塞等待通信完成       │
│                                                                     │
│  下一次 forward:   ← 所有通信已完成，无法 overlap                     │
│    _pre_forward_param_unshard                                       │
│    module.forward()                                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

源码注释明确说明了这一取舍：

```python
# If model_auto_sync is enabled, we automatically synchronize gradients
# so the user does not have to call finish_grad_sync() manually. However,
# this will reduce training performance when using delayed optimization
# techniques such as gradient accumulation, because asynchronous gradient
# reduce-scatter calls can be overlapped with subsequent compute.
```

### 3.3 梯度累积场景下的 Overlap 策略

在多 microbatch 梯度累积场景中，通信 overlap 的关键决策点在于 reduce 条件：

| 策略 | 非最后 microbatch 行为 | 最后 microbatch 行为 |
|------|----------------------|---------------------|
| `optim_grads` / `optim_grads_params` | 每次 backward 都 reduce（梯度分片要求） | reduce + finish_grad_sync |
| `optim`（仅 optimizer 分片） | 不 reduce，仅累积梯度 | reduce + finish_grad_sync |

对于 `optim_grads` / `optim_grads_params` 策略：
- 每次 backward 的 reduce-scatter 通信可以与下一个 microbatch 的 forward 重叠
- `reset()` 中的 `wait_for_previous_grad_reduce(0)` 会等待上一次 reduce 完成，但新发起的 reduce 仍可与后续计算重叠

对于 `optim` 策略：
- 中间 microbatch 不发起任何通信，梯度在 unsharded buffer 中累积
- 最后 microbatch 发起一次 all-reduce（或 reduce-scatter），然后 finish_grad_sync 同步完成

### 3.4 microbatch_count 与下一次迭代的联动

`microbatch_count` 递增影响下一次 forward 中 outer FSDP group 的 all-gather 行为：

```python
# all_gather_and_wait_parameters_ready 中
outer_fsdp_group_param_gather=(
    self.dist_index.use_hybrid_fsdp
    and self.ddp_config.outer_dp_sharding_strategy != "no_shard"
    and (self.microbatch_count == 0 or self.model_auto_sync)
)
```

- `microbatch_count == 0`：新 optimization step 的第一个 microbatch，需要从 outer DP group all-gather 最新权重
- `microbatch_count > 0`：同一 step 的后续 microbatch，权重未变，无需重复 all-gather

`finish_grad_sync()` 会将 `microbatch_count` 重置为 0，标记新 step 开始。而在非 auto_sync 模式下，`_root_post_backward` 仅递增计数器，依赖用户后续手动调用 `finish_grad_sync()` 来重置。
