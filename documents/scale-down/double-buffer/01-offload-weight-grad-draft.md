# Double Buffer：`model_weight` 与 `main_grad`

本文总结 FSDP double buffer 和 CPU offload 实现。重点是：

- `model_weight` 如何从 CPU reload 到 GPU，并参与 all-gather；
- `main_grad` 如何在 reduce-scatter 后 offload 到 CPU，并在下一次 backward 或 optimizer step 前 reload；
- 两类 buffer 如何通过固定大小为 2 的 pool 复用 GPU 存储。

文中将 `model_weight_buffer` 简写为 **model weight**，将 `main_grad_buffer` 简写为 **main grad**。`main_weight_buffer` 是 optimizer 的 FP32/master weight，属于另一条 optimizer-state offload 路径，不能与 `model_weight_buffer` 混为一谈。

## 1. 两层“双缓冲”

Estimator 中存在两种相互独立、都由 `FixedPoolAllocator` 管理的临时存储。通信 full-bucket pool 固定为 `size=2`；CPU offload pool 使用 `cpu_offload_pool_size`，默认值也是 2：

| 层次 | 典型 allocator | 存放内容 | 作用 |
| --- | --- | --- | --- |
| 通信 full-bucket double buffer | `model_weight_bucket_alloc`、`main_grad_bucket_alloc` | `total_numel` 大小的 full bucket | 一个 unit 计算时，另一个 unit 可以做 all-gather 或 reduce-scatter |
| CPU offload shard pool（默认 2 槽） | `model_weight_buffer_alloc`、`main_grad_buffer_alloc` | `data_size` 大小的 GPU shard 临时副本 | CPU 常驻、GPU 按需 reload；默认两个槽位限制同时活跃的 FSDP unit 数 |

`fsdp_double_buffer_params=True` 创建 weight/grad 的 full-bucket pool；`cpu_offload_weight=True` 或 `cpu_offload_grad=True` 创建对应的 CPU-offload GPU shard pool。两者可以同时启用，但用途不同：double buffer 只管理临时存储的复用，不改变 all-gather/reduce-scatter 的通信语义。

### 1.1 FixedPoolAllocator 的槽位规则

`FixedPoolAllocator`（`param_and_grad_buffer.py:476-694`）首先找出 bucket布局、dtype 和参数量相同的 FSDP unit 集合，记录为 `fsdp_double_buffer_units`。只有这些 unit 使用固定池；不匹配的 unit（例如大小不同的 embedding 或 non-unit bucket）回退到 `backup_allocator`。

每个槽位由 `(buf_group_id, bucket_offset)` 标识：

```text
buf_group_id = 0 或 1       # 两组物理存储
bucket_offset                # unit 内第几个 bucket
```

`using_buffer` 记录 bucket 当前占用的槽位，`idle_buffer` 记录可用槽位。申请新槽位时如果没有空闲槽位，代码不会直接覆盖正在使用的数据，而是要求调用方先回收一个 `buffer_can_be_release=True` 的 owner；找不到可回收 owner 就抛出 `RuntimeError`。

因此，“最多两个 unit 同时活跃”是 allocator 的实际约束：

- 参数路径由 `AllGatherPipeline._release_buffers_for_slot()` 和 `_release_buckets_for_slot()` 执行回收；
- 梯度路径由 `GradReducePipeline._enforce_double_buffer_limit()` 在分配临时 grad bucket 前检查，并通过 `wait_for_previous_grad_reduce()` 推进队列。

## 2. `DataParallelBuffer` 的状态模型

每个 buffer 都有一份逻辑数据和两份位置状态（`param_and_grad_buffer.py:732-1161`）：

- `data`：当前 GPU shard；CPU offload 时是临时分配的，可能为 `None`；
- `cpu_data`：CPU 持久副本；启用 offload 时一直保留；
- `gpu_status` / `cpu_status`：`INVALID`、`EMPTY`、`COPYING`、`READY`；
- `gpu_ready_event` / `cpu_ready_event`：异步 H2D/D2H 完成事件；
- `buffer_can_be_release`：GPU shard 临时存储是否可以归还 pool；
- `bucket_can_be_release`：full bucket 临时存储是否可以归还 pool。

两个核心判断：

```python
needs_offload = (
    device == "cpu"
    and gpu_status == READY
    and cpu_status == EMPTY
)

needs_gpu_reload = (
    cpu_status in (COPYING, READY)
    and gpu_status in (INVALID, EMPTY)
)
```

`offload_buffer_to_cpu()` 在指定 stream 上执行 GPU→CPU copy，记录事件并把
`cpu_status` 置为 `READY`；`reload_buffer_to_gpu()` 执行 CPU→GPU copy，记录
`gpu_ready_event`。这两个函数只搬运 persistent shard，不负责 full-bucket
all-gather。

## 3. `model_weight` 的创建、offload 与 reload

### 3.1 初始化：CPU 常驻，GPU 只做一次临时拷贝

当 `cpu_offload_weight=True` 且 bucket 属于 FSDP unit 时，`model_weight_buffer` 的 `device` 被设置为 CPU （`param_and_grad_buffer.py:1925-1930`、`2059-2083`）。初始化流程如下：

1. 分配 CPU `cpu_data`，状态为 `cpu_status=EMPTY`；
2. `_swap_param_to_wbuf()`（`param_and_grad_buffer.py:2433-2479`）从 `model_weight_buffer_alloc` 取得一个 GPU shard 临时槽位；
3. 将参数的本地 shard 写入 GPU `data`，并把 `param.data` 暂时绑定到 bucket view；
4. 执行一次 D2H copy，把 shard 保存到 `cpu_data`；
5. 归还 GPU shard 槽位，后续 forward/backward 需要时再 reload。

所以 CPU offload 并不是把参数对象删除，而是让参数的数据存储在每次 hook 触发时重新绑定。

### 3.2 Forward/backward 前：先 reload shard，再 all-gather

FSDP unit 的 pre-forward hook 或 pre-backward hook 调用 `all_gather_and_wait_parameters_ready()`（`megatron_fsdp.py:436-451`、`483-494`）。它将请求 bucket 交给 `AllGatherPipeline.all_gather_params()`（`param_and_grad_buffer.py:3038-3130`）。对每个 bucket 的顺序是：

```text
DEFAULT stream
    │ 记录 sync_default
    ├──────────────► FSDP_RELOAD_WEIGHT stream
    │                  1. 等待 CPU D2H 完成事件
    │                  2. 必要时回收另一个 unit 的 GPU shard 槽位
    │                  3. fetch_buffer() 从 pool 取得 GPU shard
    │                  4. reload_buffer_to_gpu() 发起 H2D
    │                  5. 记录 gpu_ready_event
    │
    └──────────────► FSDP_AG stream
                       等待 reload 完成事件
                       对 ZeRO-3 执行 all-gather 到 full bucket
                       将 param.data 绑定到 full bucket view
```

具体由 `async_buffer_reload()`（`param_and_grad_buffer.py:3215-3250`）完成：

- `cpu_status` 为 `COPYING` 时先等待 `cpu_ready_event`，再视为 `READY`；
- `gpu_status=INVALID` 时调用 `_release_buffers_for_slot()`，只能回收已经 `buffer_can_be_release` 的 owner；
- `fetch_buffer()` 从 `model_weight_buffer_alloc` 获得 GPU shard；
- `reload_buffer_to_gpu()` 在 `FSDP_RELOAD_WEIGHT` stream 上做 H2D。

随后 `async_bucket_gather()`（`param_and_grad_buffer.py:3153-3212`）在 `FSDP_AG` stream 上：

- ZeRO-3 (`is_data_distributed=True`) 从 GPU shard all-gather 到 `model_weight_bucket_alloc` 提供的 full bucket；
- 将每个参数的 `param.data` 重新绑定到 full bucket view；
- 通过 `bucket_ready_event` 发布 full bucket 可用事件。

ZeRO-1/2 的 `model_weight_buffer` 不分片，不需要 all-gather；仍然会先做 CPU→GPU reload，之后直接使用 GPU buffer。也就是说，CPU offload 下“reload”和 “all-gather”是两个连续但独立的步骤。

### 3.3 计算后释放：只释放 GPU 临时存储

post-forward/post-backward hook 调用 `release_module_parameters()`
（`megatron_fsdp.py:454-464`、`342-366`）：

- `lazy=True` 时只设置 `bucket_can_be_release` 或 `buffer_can_be_release`，让后续申请槽位时再回收；
- 真正回收时，`release_bucket()` 释放 full bucket，`release_buffer()` 归还 GPU shard 槽位；
- CPU `cpu_data` 不被释放，因此下次使用仍可 H2D reload。
