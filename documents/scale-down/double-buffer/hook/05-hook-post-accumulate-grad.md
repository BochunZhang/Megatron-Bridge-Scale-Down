# Hook: post_accumulate_grad (_process_post_backward_gradients)

## 概述

`_process_post_backward_gradients` 是 Megatron FSDP 中处理 backward 梯度的核心 hook。它通过 PyTorch 的 `register_post_accumulate_grad_hook` 机制注册在每个需要梯度的参数上，当 autograd 引擎为某个参数完成梯度累积后立即触发。

**核心职责**：
1. 将 `param.grad` 搬运到 `ParamAndGradBuffer` 管理的 `main_grad` 桶中
2. 条件性地触发异步 reduce-scatter 操作，实现梯度通信与 backward 计算的 overlap

**源码位置**: `megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py`
- hook 函数定义: line 615
- 注册逻辑: line 1003-1012

## 1. 注册信息

### 1.1 注册等级

此 hook 在**所有 sharding strategy** 下均注册，包括：
- `optim_grads_params` (ZeRO-3, 参数+梯度+优化器状态均分片)
- `optim_grads` (ZeRO-2, 梯度+优化器状态分片)
- `no_shard` (仅数据并行, 不分片)

注册逻辑中没有对 `data_parallel_sharding_strategy` 的条件过滤：

```python
# line 1003-1012 — 无 strategy 条件判断，所有策略都注册
if isinstance(module, tuple(fsdp_unit_modules)):
    grad_acc_param_list = list(module.parameters())
else:
    grad_acc_param_list = list(module.parameters(recurse=False))

for param in grad_acc_param_list:
    self.grad_acc_hooks[f"grad_acc and reduce for {self.param_to_name[param]}"] = (
        param.register_post_accumulate_grad_hook(
            lambda p: _process_post_backward_gradients([p])
        )
    )
```

### 1.2 注册对象

Hook 注册在**每个需要梯度的参数**（`param` 级别），但参数的收集范围由所属 module 类型决定。

遍历逻辑（line 955-1012）通过 `root_module.named_modules()` 遍历所有模块：

```python
fsdp_modules = []
for name, module in root_module.named_modules():
    # 已注册 FSDP unit 的子模块跳过
    if any(is_submodule(module, fsdp_module) for fsdp_module in fsdp_modules):
        continue

    if isinstance(module, tuple(fsdp_unit_modules)):
        fsdp_modules.append(module)
        grad_acc_param_list = list(module.parameters())           # 递归收集
    else:
        grad_acc_param_list = list(module.parameters(recurse=False))  # 仅浅层
```

**参数收集规则**：

| Module 类型 | 参数收集方式 | 典型例子 |
|------------|-------------|---------|
| FSDP unit module | `module.parameters()` 递归取所有子模块参数 | `TransformerLayer`, `LanguageModelEmbedding` |
| 非 FSDP unit（且不是任何 FSDP unit 的子模块） | `module.parameters(recurse=False)` 仅直属参数 | root module, `final_layernorm`, `output_layer` |
| FSDP unit 的子模块 | 不注册（被父 FSDP unit 覆盖） | `self_attention`, `mlp`, `linear_qkv` 等 |

这种设计保证每个参数恰好注册一次 hook，不会重复触发。FSDP unit 作为梯度管理的基本单元，其所有子参数统一收集；而非 FSDP unit 的模块只收集自身直属参数，避免与 FSDP unit 的参数重叠。

### 1.3 注册方式

使用 PyTorch 的 `Tensor.register_post_accumulate_grad_hook` API：

```python
param.register_post_accumulate_grad_hook(
    lambda p: _process_post_backward_gradients([p])
)
```

**PyTorch 机制说明**（来自 `torch/_tensor.py` line 679）：

- 此 hook 注册在**叶子张量**（leaf tensor，即 `grad_fn is None` 的参数）上
- 在该张量的所有梯度累积完成后触发（即 `.grad` 字段已更新）
- hook 签名: `hook(param: Tensor) -> None`
- hook 操作的是参数本身而非梯度，可以 in-place 修改 `.grad` 字段
- 在 `no_grad` 模式下执行（除非 `create_graph=True`）

与旧式 `grad_acc.register_hook` 的区别：`register_post_accumulate_grad_hook` 在梯度完全累积后才触发（考虑了梯度累加的情况），语义更清晰。

### 1.4 执行时间

在 backward 过程中，当 autograd 引擎为某个参数**完成梯度累积**后立即触发。

具体时序：
1. Autograd 引擎按拓扑逆序遍历计算图
2. 对每个叶子参数，计算并累积其梯度到 `param.grad`
3. 梯度累积完成后，触发该参数上注册的 `post_accumulate_grad_hook`
4. Hook 内执行梯度搬运和条件性 reduce-scatter

这意味着 hook 的触发顺序与 backward 计算顺序一致——后面层的参数先触发，前面层的参数后触发。

## 2. 执行流程

### 调用链总览

```
param.register_post_accumulate_grad_hook
  → _process_post_backward_gradients([param])    (line 615)
    → 过滤 shared parameters                     (line 650)
    → _grad_acc(param)                           (line 652)
    → 判断是否触发 reduce                         (line 656-662)
    → grad_reduce_pipeline.reduce_gradients()    (条件触发, line 666)
    → _params_require_handle_grad.discard(param) (line 677)
```

### 2.1 过滤 shared parameters

```python
# line 650
param_list = [p for p in param_list if not getattr(p, "_is_shared", False)]
```

Shared parameters（如 embedding 权重共享）的 `_is_shared` 属性在初始化时标记（line 1280）。这些参数的梯度处理被延迟到 `_root_post_backward` 兜底 hook 中统一处理，确保共享参数的梯度在所有使用点累积完毕后再进行 reduce。

### 2.2 _grad_acc 函数：梯度搬运到 main_grad

```python
def _grad_acc(param):  # line 546
    group_id = self.param_and_grad_buffer.param_to_param_group[param]
    group = self.param_and_grad_buffer.parameter_groups[group_id]
    if not group.requires_grad:
        return

    # 选择梯度 buffer：hybrid FSDP 用 hsdp_gbuf，否则用 main_grad_buffer
    gbuf = group.hsdp_gbuf if group.hsdp_gbuf else group.main_grad_buffer

    if gbuf.is_data_distributed:
        # === Sharded 路径：覆盖写入 ===
        if not param.grad_added_to_main_grad:
            if self.ddp_config.fsdp_double_buffer:
                self.grad_reduce_pipeline._enforce_double_buffer_limit([group_id])
            param.main_grad = param.get_main_grad()
            if param.grad is not None:
                param.main_grad.copy_(to_local_if_dtensor(param.grad))
                del param.grad
            else:
                param.main_grad.zero_()
    else:
        # === Unsharded 路径：累加 ===
        if not param.grad_added_to_main_grad:
            if param.grad is not None:
                param.main_grad = param.get_main_grad()
                param.main_grad.add_(to_local_if_dtensor(param.grad))
                del param.grad

    # 清理已被 TE 直接写入的多余 grad
    if param.grad_added_to_main_grad and param.grad is not None:
        del param.grad

    param.grad_added_to_main_grad = False
```

**两条路径对比**：

| | Sharded (`is_data_distributed=True`) | Unsharded (`is_data_distributed=False`) |
|---|---|---|
| 操作 | `main_grad.copy_(param.grad)` 覆盖 | `main_grad.add_(param.grad)` 累加 |
| 原因 | 每个 microbatch 独立 reduce-scatter，不需跨 microbatch 累加 | 只在最后一个 microbatch reduce，需要跨 microbatch 累加 |
| double-buffer | 写入前 `_enforce_double_buffer_limit`，确保最多 2 个 unit 的 buffer 同时存活 | 不涉及 |
| grad 为 None | `main_grad.zero_()` 写零 | 跳过（不分配 buffer） |

**`grad_added_to_main_grad` 标志**：当 Transformer Engine (TE) 的融合 kernel 直接将梯度写入 `main_grad` 时，会将此标志置为 True。此时 `_grad_acc` 跳过搬运操作，仅清理多余的 `param.grad`。

### 2.3 判断是否触发 reduce

```python
# line 656-662
grad_reduce_every_bprop = self.ddp_config.data_parallel_sharding_strategy in [
    "optim_grads",
    "optim_grads_params",
]
is_last_microbatch = getattr(self, "is_last_microbatch", False)

if grad_reduce_every_bprop or is_last_microbatch or self.model_auto_sync:
    self.grad_reduce_pipeline.reduce_gradients(...)
```

**三个触发条件**（OR 关系）：

| 条件 | 含义 | 场景 |
|------|------|------|
| `grad_reduce_every_bprop` | 梯度分片策略下每次 backward 都 reduce | ZeRO-2/3: `optim_grads` 或 `optim_grads_params` |
| `is_last_microbatch` | 最后一个 microbatch | 梯度累积的最后一步，所有 microbatch 梯度累加完毕 |
| `model_auto_sync` | 自动同步模式 | 用户未使用 `no_sync()` 上下文时的默认行为 |

在 ZeRO-2/3 下，`grad_reduce_every_bprop=True`，因此每个 microbatch 的 backward 都会触发 reduce-scatter。这是因为分片梯度需要通过 reduce-scatter 归约到各 rank 负责的 shard 上，不能跨 microbatch 累加未归约的梯度。

### 2.4 grad_reduce_pipeline.reduce_gradients 流程

```python
# line 666-673
self.grad_reduce_pipeline.reduce_gradients(
    param_list,
    suggested_queue_capacity=self.suggested_RS_queue_capacity,
    outer_fsdp_group_grad_reduce=(
        self.dist_index.use_hybrid_fsdp
        and (is_last_microbatch or self.model_auto_sync)
    ),
)
```

`ReduceScatterPipeline.reduce_gradients()`（`param_and_grad_buffer.py` line 3048）的执行流程：

1. **按 bucket 排序参数**：确保 reduce-scatter 操作按确定性顺序执行，避免 NCCL 死锁
2. **标记参数就绪**：将参数加入对应 bucket 的 `bucket_grad_ready_params` 集合
3. **检查 bucket group 就绪**：当一个 bucket group 中所有参数的梯度都就绪时，触发 reduce
4. **流量控制**：通过 `wait_for_previous_grad_reduce(suggested_queue_capacity)` 等待前序通信完成
5. **执行异步 reduce-scatter**：在独立的 CUDA stream 上执行通信

```python
def reduce_gradients(self, params, suggested_queue_capacity=None, ...):
    params = sorted(params, key=lambda x: self.buffer.param_to_param_group[x])
    for param in params:
        bucket_id = self.buffer.param_to_param_group[param]
        # 标记参数梯度就绪
        self.bucket_grad_ready_params[bucket_id].add(param)
        # 检查整个 bucket group 是否就绪
        bucket_group = self.get_ready_bucket_group_for_reduction(bucket_id)
        if bucket_group:
            # 流量控制：等待队列容量空出
            self.wait_for_previous_grad_reduce(
                suggested_queue_capacity=suggested_queue_capacity
            )
            # 执行 reduce-scatter
            self._bucket_group_gradient_reduce(
                bucket_group, async_op=True,
                outer_fsdp_group_grad_reduce=outer_fsdp_group_grad_reduce,
            )
```

### 2.5 标记参数已处理

```python
# line 675-677
for param in param_list:
    self._params_require_handle_grad.discard(param)
```

每个参数处理完毕后从 `_params_require_handle_grad` 集合中移除。该集合在 `_root_pre_backward` 中初始化为所有需要梯度的参数，`_root_post_backward` 兜底处理集合中剩余的参数（如 shared parameters）。

## 3. 通信 Overlap

### 3.1 Reduce-scatter 与 backward 计算的 overlap

`post_accumulate_grad_hook` 的核心价值在于实现梯度 reduce-scatter 与 backward 计算的流水线并行：

```
时间线：
backward 计算:  [Layer N grad] [Layer N-1 grad] [Layer N-2 grad] ...
reduce-scatter:           [RS Layer N]    [RS Layer N-1]    ...
                          ↑ overlap ↑     ↑ overlap ↑
```

实现机制：
1. **per-parameter 粒度触发**：每个参数梯度计算完成后立即触发 hook，不等待整层完成
2. **bucket group 聚合**：同一 bucket group 的所有参数就绪后才实际发起 reduce-scatter
3. **异步 CUDA stream**：reduce-scatter 在独立的 `side_stream_for_buffer_copy_and_grad_accum` stream 上执行，不阻塞主 stream 上的 backward 计算

```python
# _bucket_group_gradient_reduce 中的 stream 切换 (param_and_grad_buffer.py line 3205-3209)
current_stream = torch.cuda.current_stream()
reduce_scatter_stream = self.rs_stream if self.rs_stream is not None else current_stream
reduce_scatter_stream.wait_stream(current_stream)

with torch.cuda.stream(reduce_scatter_stream):
    # 在独立 stream 上执行 reduce-scatter，不阻塞 backward
    torch.distributed.reduce_scatter_tensor(
        output=grad_shard, input=bucket.data, op=reduce_op, group=dp_group
    )
```

### 3.2 suggested_queue_capacity 机制

`suggested_queue_capacity` 控制同时在进行中的 reduce-scatter 操作的总量（以参数元素数计），实现流量控制：

```python
# 初始化 (megatron_fsdp.py line 374-398)
if data_parallel_sharding_strategy == "optim_grads_params":
    # 默认设为 2 个 FSDP module 的参数量
    suggested_communication_unit_size = total_param_elements // total_fsdp_module * 2
# 下限 1B 元素
suggested_communication_unit_size = max(1_000_000_000, suggested_communication_unit_size)
self.suggested_RS_queue_capacity = suggested_communication_unit_size
```

**流量控制逻辑**（`wait_for_previous_grad_reduce`, line 3092-3116）：

```python
def wait_for_previous_grad_reduce(self, suggested_queue_capacity=None):
    if suggested_queue_capacity is not None:
        queue_space = sum(
            bucket.main_grad_buffer.bucket_index.size
            for _, _, bucket_id in self.grad_reduce_queue
        )
        while queue_space > suggested_queue_capacity:
            grad_reduce_event, free_up_grad_bucket, bucket_id = self.grad_reduce_queue.pop(0)
            grad_reduce_event.wait()      # 同步等待最早的 reduce 完成
            free_up_grad_bucket()          # 释放 bucket 内存
            queue_space -= ...
```

设计目的：
- **显存控制**：限制同时存在的梯度 bucket 数量，防止内存爆炸
- **通信/计算平衡**：允许一定数量的 reduce 操作排队（约 2 个 FSDP unit 的大小），既保持 overlap 又不占用过多显存
- **避免死锁**：确保 reduce-scatter 有序完成，不因队列过深导致资源耗尽

### 3.3 Hybrid FSDP 的 outer_fsdp_group_grad_reduce

在 hybrid FSDP（多级 FSDP 分组）配置下，梯度需要在 outer DP group 间进一步归约：

```python
# line 669-672
outer_fsdp_group_grad_reduce=(
    self.dist_index.use_hybrid_fsdp
    and (is_last_microbatch or self.model_auto_sync)
)
```

**触发条件**：仅在最后一个 microbatch 或 auto_sync 模式下触发 outer group 归约。中间 microbatch 只做 inner group 的 reduce-scatter。

**执行机制**（`param_and_grad_buffer.py` line 3262-3310）：

```python
if outer_fsdp_group_grad_reduce:
    self.outer_fsdp_group_grad_reduce_stream.wait_stream(reduce_scatter_stream)
    outer_fsdp_group = self.buffer.dist_index.get_outer_fsdp_group()
    with torch.cuda.stream(self.outer_fsdp_group_grad_reduce_stream):
        # 在 inner reduce-scatter 之后，进一步跨 outer group 归约
        if outer_dp_sharding_strategy != "no_shard":
            # outer group 也做 reduce-scatter
            torch.distributed.reduce_scatter_tensor(
                output=grad_full_shard, input=gbuf.data,
                op=reduce_op, group=outer_fsdp_group,
            )
        else:
            # outer group 做 all-reduce
            torch.distributed.all_reduce(
                gbuf.data, group=outer_fsdp_group, op=reduce_op
            )
```

通信层次：
1. Inner FSDP group reduce-scatter → 在 `rs_stream` 上执行
2. Outer FSDP group reduce-scatter/all-reduce → 在 `outer_fsdp_group_grad_reduce_stream` 上执行
3. 三个 CUDA stream 保持流水线并行：backward 计算 / inner RS / outer RS
