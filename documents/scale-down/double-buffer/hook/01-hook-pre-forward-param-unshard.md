# Hook: _pre_forward_param_unshard

## 概述

`_pre_forward_param_unshard` 定义在 `megatron_fsdp.py:680`，是 `_register_fsdp_hooks` 内部的闭包函数。

**核心职责**：在每个 module 的 forward 执行前，通过 all-gather 通信将 sharded 参数恢复为完整参数，使 forward 计算能使用完整权重。

## 1. 注册信息

### 1.1 注册等级

在 `_register_fsdp_hooks`（L468）中，通过遍历 `root_module.named_modules()` 进行注册。注册发生在 **MegatronFSDP 初始化阶段**，对整个模型树生效。

前置条件：`data_parallel_sharding_strategy != "no_shard"`，即必须启用参数分片策略（`optim_grads_params`）。

### 1.2 注册对象

注册对象取决于 `enable_fine_grained_param_gather_hook` 配置：

| 模式 | 注册范围 | 参数收集方式 |
|------|----------|-------------|
| `fine_grained=False`（默认） | 所有非 FSDP-unit 子模块的 module + FSDP unit 本身 | FSDP unit: `parameters()` 递归；非 unit: `parameters(recurse=False)` |
| `fine_grained=True` | 所有 module（无跳过） | 始终 `parameters(recurse=False)` |

**默认模式的注册逻辑**（L955-966）：

```python
fsdp_modules = []
for name, module in root_module.named_modules():
    # 跳过已被某个 FSDP unit 包含的子模块（避免重复 backward hook）
    if any(is_submodule(module, fsdp_module) for fsdp_module in fsdp_modules):
        continue
    if not self.enable_fine_grained_param_gather_hook:
        _register_pre_forward_param_unshard_hook(module)
```

注意：`is_submodule` 跳过机制只影响 backward hook 的注册，pre-forward hook 对所有未跳过的 module 都会注册。

### 1.3 注册方式

使用 PyTorch 的 `register_forward_pre_hook` API：

```python
module.register_forward_pre_hook(
    _pre_forward_param_unshard, prepend=True, with_kwargs=True
)
```

| 参数 | 值 | 含义 |
|------|-----|------|
| `prepend` | `True` | 插入到 hook 列表最前面，确保在其他 pre-hook 之前执行 |
| `with_kwargs` | `True` | hook 签名接收 `kwargs`，返回 `(args, kwargs)` |

### 1.4 执行时间

- **Forward pass 开始前**：每次 `module.forward()` 调用前触发
- **Activation recomputation 时**：在 backward 中重计算 forward 时也会触发（此时 `_training_state == PRE_BACKWARD`）
- **触发频率**：每个 microbatch 的每个 module forward 都会触发

## 2. 执行流程

### 2.1 判断训练状态，决定是否启用 prefetch（L684-690）

```python
input_training_state = module._training_state
fsdp_forward_prefetch = True
if input_training_state == TrainingState.PRE_BACKWARD:
    fsdp_forward_prefetch = False
else:
    module._training_state = TrainingState.FORWARD
```

- `PRE_BACKWARD` 状态表示当前处于 activation recomputation 的重计算前向，此时**禁用 prefetch**（避免与 backward 的通信冲突）
- 正常 forward 时将状态切换为 `FORWARD`，**启用 prefetch** 以实现通信重叠

### 2.2 收集需要 unshard 的参数列表（L692-701）

```python
if isinstance(module, tuple(fsdp_unit_modules)):
    param_list = list(module.parameters())           # 递归收集所有参数
else:
    param_list = list(module.parameters(recurse=False))  # 只收集浅参数

if self.enable_fine_grained_param_gather_hook:
    param_list = list(module.parameters(recurse=False))  # 覆盖为浅参数
```

| 条件 | 参数收集方式 | 原因 |
|------|-------------|------|
| FSDP unit module | `parameters()` 递归 | 整个 unit 作为一个通信单元 |
| 非 FSDP unit module | `parameters(recurse=False)` | 减少内存分配，只处理当前层 |
| `fine_grained=True` | 始终 `recurse=False` | 更细粒度的通信重叠 |

### 2.3 执行 all-gather 并等待参数就绪（L704-708）

```python
self.all_gather_and_wait_parameters_ready(
    params=param_list,
    prefetch=fsdp_forward_prefetch,
    prefetch_order=PrefetchOrder.FORWARD_PASS_ORDER,
)
```

`all_gather_and_wait_parameters_ready` 定义在 `megatron_fsdp.py:413`，内部流程：

**步骤 1: 短路检查**

```python
if self.data_parallel_sharding_strategy == "no_shard":
    return
```

**步骤 2: 调用 AllGatherPipeline.all_gather_params**

```python
ag_pipeline.all_gather_params(
    params=params,
    prefetch=prefetch,
    prefetch_order=prefetch_order,
    suggested_AG_prefetch_size=self.suggested_AG_prefetch_size,
    outer_fsdp_group_param_gather=(
        self.dist_index.use_hybrid_fsdp
        and self.ddp_config.outer_dp_sharding_strategy != "no_shard"
        and (self.microbatch_count == 0 or self.model_auto_sync)
    ),
    bwd=bwd,
)
```

`outer_fsdp_group_param_gather` 仅在 HSDP 模式下的第一个 microbatch（或 auto_sync）时为 True。

**步骤 3: 等待每个 bucket 就绪**

```python
if wait_bucket_ready:
    for param in params:
        bucket_id = self.param_and_grad_buffer.param_to_param_group[param]
        ag_pipeline.wait_bucket_ready(bucket_id, bwd)
        if bwd and is_float8tensor(param):
            fp8_create_transpose_cache(param)
```

逐个参数查找 bucket ID，等待对应 bucket 的 NCCL 操作完成。backward 时额外为 FP8 参数创建转置缓存。

**步骤 4: 设置参数属性**

```python
for param in params:
    param.grad_added_to_main_grad = False
    param.__fsdp_param__ = True
    param.overwrite_main_grad = True
```

| 属性 | 作用 |
|------|------|
| `grad_added_to_main_grad = False` | 兼容 TE activation offloading |
| `__fsdp_param__ = True` | 标记为 FSDP 管理的参数 |
| `overwrite_main_grad = True` | 告诉 TE 覆盖而非累加 main_grad |

### 2.4 AllGatherPipeline.all_gather_params 详细流程

定义在 `param_and_grad_buffer.py:3447`。管理 bucket 级别的 all-gather 通信。

**核心状态机**：

```
BucketStatus:
  EMPTY  ──[async_bucket_gather]──>  COMMUNICATING  ──[wait_bucket_ready]──>  READY_TO_USE
    ^                                                                              │
    └──────────────────────────[release_bucket]────────────────────────────────────┘
```

**2.4.1 参数到 Bucket 映射**

```python
ag_buckets = [self.buffer.param_to_param_group[item] for item in params]
ag_buckets = list(sorted(set(ag_buckets)))
```

将参数列表转换为去重排序的 bucket ID 列表。

**2.4.2 Double Buffer 校验**

Double buffer 模式限制同时活跃的 FSDP unit 不超过 2 个，超出则报错。

**2.4.3 标记 bucket 不可释放**

```python
for bucket_id in ag_buckets:
    self.bucket_can_be_released[self.get_bucket_key(bucket_id, bwd)] = False
```

防止正在 all-gather 的 bucket 被 `recycle_unused_buckets` 回收。

**2.4.4 Prefetch 扩展**

当 `prefetch=True` 时，沿 forward 方向扩展 bucket 列表：

```python
while bucket_id is not None:
    prefetch_all_gather_size = sum(扩展后的 bucket sizes) - base_all_gather_size
    if prefetch_all_gather_size >= suggested_AG_prefetch_size:  # 默认 500MB
        break
    if need_skip_prefetch(bucket_id):  # double buffer 限制
        break
    ag_buckets.extend(self.buffer.bucket_to_bucket_group[bucket_id])
    bucket_id = next_bucket_id(ag_buckets)
```

- **FORWARD_PASS_ORDER**: 从当前 bucket 向后（bucket_id + 1）搜索
- **BACKWARD_PASS_ORDER**: 从当前 bucket 向前（bucket_id - 1）搜索
- 终止条件：累计 prefetch 大小达到阈值，或超出 double buffer 容量

**2.4.5 过滤已就绪 bucket**

```python
ag_buckets = [
    bucket_id for bucket_id in ag_buckets
    if self.bucket_status[self.get_bucket_key(bucket_id, bwd)] == BucketStatus.EMPTY
]
```

只对状态为 `EMPTY` 的 bucket 发起 all-gather，跳过已在通信中或已就绪的。

**2.4.6 按 bucket group 分组执行 coalesced all-gather**

```python
for _, buckets in bucket_group_to_buckets.items():
    all_gather_stream.wait_stream(torch.cuda.current_stream())
    dp_group = self.get_fsdp_buffer(buckets[0]).data_parallel_group
    with torch.cuda.stream(all_gather_stream):
        with _coalescing_manager(dp_group, async_ops=True) as coalescing_event:
            for bucket_id in buckets:
                self.async_bucket_gather(bucket_id, bwd)
```

同一 bucket group（同一 data-parallel group）的 bucket 被合并为一次 coalesced NCCL 调用。

**2.4.7 async_bucket_gather 单 bucket 操作**（`param_and_grad_buffer.py:3719`）

```python
def async_bucket_gather(self, bucket_id, bwd):
    self.bucket_status[bucket_key] = BucketStatus.COMMUNICATING
    wbuf = self.get_fsdp_buffer(bucket_id, bwd)
    self.recycle_unused_buckets()           # 回收不再需要的 bucket（腾出显存）
    bucket = wbuf.fetch_bucket(set_param_data=True)  # 分配 unsharded bucket
    param_gather_event = torch.distributed.all_gather_into_tensor(
        output_tensor=bucket.data,
        input_tensor=wbuf.get_shard_from_local_buffer(),
        group=wbuf.data_parallel_group,
        async_op=True,
    )
    self.param_gather_event_map[bucket_key] = (param_gather_event, mark_bucket_ready_to_use)
```

关键点：
- `fetch_bucket(set_param_data=True)` 分配完整大小的 tensor，并将每个参数的 `.data` 指向对应切片
- `get_shard_from_local_buffer()` 获取本地 shard 作为 all-gather 输入
- `all_gather_into_tensor` 是异步操作，返回 work handle

**2.4.8 wait_bucket_ready**（`param_and_grad_buffer.py:3629`）

```python
def wait_bucket_ready(self, bucket_id, bwd, empty_ok=False):
    if self.bucket_status[bucket_key] == BucketStatus.READY_TO_USE:
        return  # 已就绪，直接返回
    if self.bucket_status[bucket_key] == BucketStatus.EMPTY:
        raise ValueError(...)  # 异常状态
    param_gather_event, mark_bucket_ready_to_use = self.param_gather_event_map.pop(bucket_key)
    param_gather_event.wait()    # 阻塞直到 all-gather 完成
    mark_bucket_ready_to_use()   # 状态 → READY_TO_USE
```

`param_gather_event.wait()` 在当前 stream 上插入等待，确保后续计算看到完整的参数数据。

### 2.5 返回原始输入（L709）

```python
return args, kwargs
```

Hook 不修改 forward 输入，module 正常执行 forward（此时参数已 unsharded）。

### 2.6 完整调用链路图

```
_pre_forward_param_unshard(module)
│
├── 判断 training state → 决定 prefetch
├── 收集 param_list
│
└── all_gather_and_wait_parameters_ready(params, prefetch, FORWARD_PASS_ORDER)
    │
    ├── all_gather_pipeline.all_gather_params(params, prefetch, ...)
    │   │
    │   ├── params → bucket IDs (去重排序)
    │   ├── [可选] HSDP outer group all-gather
    │   ├── prefetch 扩展 bucket 列表 (≤500MB)
    │   ├── 过滤: 只保留 EMPTY 状态的 bucket
    │   ├── 按 bucket group 分组
    │   │
    │   └── for each group: coalesced all-gather on ag_stream
    │       └── async_bucket_gather(bucket_id)
    │           ├── recycle_unused_buckets()
    │           ├── wbuf.fetch_bucket(set_param_data=True)  ← 参数 .data 重定向
    │           └── all_gather_into_tensor(async_op=True)
    │
    ├── for each param: wait_bucket_ready(bucket_id)
    │   └── param_gather_event.wait() → mark READY_TO_USE
    │
    └── 设置参数属性 (__fsdp_param__, overwrite_main_grad, ...)
```

## 3. 通信 Overlap

### 3.1 Prefetch 机制

Prefetch 是实现通信/计算重叠的核心手段。在处理当前 module 的参数 all-gather 时，额外启动**下一批 bucket** 的 all-gather：

```
Timeline (forward pass):
───────────────────────────────────────────────────────
compute stream: │ Layer N forward │ Layer N+1 forward │
ag_stream:      │ AG(N) │ AG(N+1) prefetch │ AG(N+2) │
───────────────────────────────────────────────────────
```

- Layer N 的 pre-forward hook 触发时，AG(N) + AG(N+1) 在 `ag_stream` 上发起
- Layer N 做 forward 计算的同时，AG(N+1) 的通信并行执行
- Layer N+1 的 pre-forward hook 触发时，bucket 可能已经 `READY_TO_USE`，无需等待

### 3.2 Prefetch 的限制条件

1. **大小限制**：累计 prefetch 不超过 `suggested_AG_prefetch_size`（默认 500MB）
2. **Double buffer 限制**：同时 unshard 的 FSDP unit 不超过 2 个
3. **Activation recomputation 时禁用**：`PRE_BACKWARD` 状态下 `prefetch=False`，避免与 backward 通信冲突

### 3.3 异步流设计

```python
all_gather_stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(all_gather_stream):
    with _coalescing_manager(dp_group, async_ops=True):
        for bucket_id in buckets:
            self.async_bucket_gather(bucket_id, bwd)
```

- all-gather 在独立 `ag_stream` 上执行，不阻塞 compute stream
- `all_gather_stream.wait_stream(current_stream)` 确保之前的计算完成后再开始通信
- `wait_bucket_ready` 中的 `param_gather_event.wait()` 在 compute stream 同步回来

### 3.4 Coalescing 优化

同一 data-parallel group 的多个 bucket 使用 `_coalescing_manager` 合并为一次 NCCL coalesced 操作，减少通信启动开销。

### 3.5 与其他 hook 的协作

| Hook | 时机 | 作用 |
|------|------|------|
| `_pre_forward_param_unshard` | forward 前 | all-gather 恢复完整参数 + prefetch |
| `_post_forward` | forward 后 | reshard / 释放参数内存 |
| `_pre_backward_param_unshard` | backward 前 | 再次 all-gather 用于梯度计算 |
| `_post_backward_release_module` | backward 后 | reduce-scatter 梯度 + 释放参数 |

forward 阶段的 prefetch + post-forward 的及时释放，共同实现了 **"滑动窗口"** 式的内存管理：始终只保持当前计算层 + 下一层的参数在显存中。

