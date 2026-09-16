# DeepSpeed ZeRO Stage 3: Param Reload / All-Gather / Grad Reduce-Scatter / Grad Buffer 累加

## 概述

ZeRO Stage 3 将 param、grad、optimizer states 全部按 world_size 切分。每个 rank 只持有 1/N 的参数分区（`param.ds_tensor`）。
计算需要完整参数，因此每个 module 计算前必须通过 all-gather 重建完整参数，计算后释放。

## 源码位置

| 组件 | 文件 |
|------|------|
| Fetch/Prefetch/Release 调度 | `deepspeed/runtime/zero/partitioned_param_coordinator.py` |
| Hook 注册（forward/backward） | `deepspeed/runtime/zero/parameter_offload.py` |
| All-gather 实现 | `deepspeed/runtime/zero/partition_parameters.py` |
| Reduce-scatter + grad buffer | `deepspeed/runtime/zero/stage3.py` |

---

## 一、Param Reload + All-Gather（Prefetch Bucket）

### 核心机制

每个 rank 只存 param 的 1/N 分区。计算前通过 all-gather 从所有 rank 收集分区拼成完整参数。

**Prefetch = 提前发起 all-gather。** 不是"先加载到 GPU 再 all-gather"，而是 prefetch 本身就包含了 all-gather。

### Partition 存储位置

| 配置 | `param.ds_tensor` 位置 | All-gather 内容 |
|------|----------------------|----------------|
| `offload_param = True` | CPU (pinned memory) | CPU→GPU DMA + 跨 rank NCCL all-gather |
| `offload_param = False` | GPU | 仅跨 rank NCCL all-gather |

两种配置共享完全相同的 prefetch 逻辑。

### 时序（Forward 和 Backward 对称）

```
default_stream:        [Module N compute]      [Module N+1 compute]
                            ↑ wait                    ↑ wait
                            │                         │
allgather_stream:  [fetch N: allgather] [prefetch N+1, N+2: allgather] [N+1 已就绪] ...
```

Forward 结束后 param 释放（`data = empty(0)`），backward 到同一 module 时重新 fetch。
一个 training step 中每个参数被 all-gather **两次**（forward 一次，backward 一次）。

### Prefetch 策略

基于 trace 录制-回放：

1. **第一轮 forward+backward**：记录所有 module 的执行顺序 → 构建参数访问队列 `__param_order`
2. **后续每轮**：用 `__param_order` 生成 FIFO 队列 `__param_queue`
3. **当前 module fetch 完成后**：从队列前方取参数发起 prefetch，直到 budget 用完

```python
# partitioned_param_coordinator.py:431-464
max_params_to_prefetch = min(
    self.__max_n_available_params - self.__n_available_params,  # GPU 显存余量
    self.__prefetch_bucket_sz                                    # 配置上限
)
while self.__param_queue and numel_prefetching < max_params_to_prefetch:
    param_in_trace = self.__param_queue.popleft()
    if param.ds_status == NOT_AVAILABLE:
        params_to_prefetch.add(param_in_trace.param)
        numel_prefetching += param_in_trace.param.ds_numel
```

### 默认配置

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `stage3_prefetch_bucket_size` | 5e7 (≈100MB bf16) | 每次 prefetch 的最大 numel |
| `stage3_max_live_parameters` | 1e9 | GPU 上同时驻留参数上限 |
| `stage3_max_reuse_distance` | 1e9 | 复用距离内不释放参数 |

### Prefetch 触发机制

Prefetch 不是独立触发的，而是**搭载在每个 module 的 pre_hook 中**：

```
pre_hook(Module N):
  ├─ ① 发起当前 module 的 all-gather（如果还没就绪）
  ├─ ② wait 当前 module 的参数就绪
  └─ ③ 从 __param_queue 前方取参数，发起 prefetch（async，不 wait）
```

post_hook 只负责 release，不触发 prefetch。

### Prefetch Bucket 的本质：动态 numel 预算，非固定容器

**没有预先划分好的固定 bucket。** "Bucket" 只是一个 numel 上限，每次 prefetch 动态地从 FIFO 队列头部取 param，凑够就打包发一次 all-gather：

```python
# 没有预分配的 bucket 结构，只有一个平铺的参数队列
self.__param_queue = deque(self.__param_order)  # [param_0, param_1, param_2, ...]

# 每次 prefetch 从队列头部动态取
params_to_prefetch = set()
numel_prefetching = 0
while self.__param_queue and numel_prefetching < max_params_to_prefetch:
    param_in_trace = self.__param_queue.popleft()
    params_to_prefetch.add(param_in_trace.param)
    numel_prefetching += param_in_trace.param.ds_numel

# 凑够就发：一次 coalesced all-gather
self.__all_gather_params(params_to_prefetch)
```

一次 prefetch 可能跨越多个 module 的参数边界，也可能只包含一个大 module 的部分参数。

### Prefetch 的两阶段：选参数 → 批量搬运通信

Prefetch 是分两步执行的：

1. **选参数（逐个从队列取，无数据搬运）**：从 `__param_queue` 逐个 pop param 放入集合，累计 numel 直到满足 budget。这一步只是确定"这次 prefetch 搬哪些 param"。

2. **批量执行（一次性 cat + all-gather）**：把选好的所有 param 的 CPU partition 拼接到预分配的连续 GPU buffer，然后发起一次 NCCL all-gather。

```python
# 步骤 1：选参数（仅收集，无数据搬运）
params_to_prefetch = set()
while queue and numel < budget:
    params_to_prefetch.add(queue.popleft().param)

# 步骤 2：批量 cat + all-gather
__all_gather_params(params_to_prefetch):
    # 各 param 的 ds_tensor 在 CPU 内存中是散落的（不连续）
    # 必须拼到一块连续 GPU buffer 才能做一次 NCCL call
    torch.cat([p.ds_tensor.to(GPU) for p in params], out=buffer[rank_slice])
    dist.all_gather_into_tensor(flat_buffer, buffer[rank_slice], async_op=True)
```

Coalesced 的意义：合并多个小参数的通信为一次大通信，减少 NCCL kernel launch 开销。

### All-gather 操作细节

#### Param 的切分方式

每个 param **独立**按 dp_world_size 均匀切分，不是把多个 param 拼成大 buffer 再切：

```python
# partition_parameters.py:1722-1758
tensor_size = aligned_size(param)              # pad 到 world_size 整数倍
partition_size = tensor_size // world_size      # 每个 rank 的分片大小
start = partition_size * rank
param.ds_tensor.copy_(param.view(-1).narrow(0, start, partition_size))
```

```
param W (numel=16M, world_size=8):
  rank 0: ds_tensor = W.view(-1)[0:2M]
  rank 1: ds_tensor = W.view(-1)[2M:4M]
  ...
  rank 7: ds_tensor = W.view(-1)[14M:16M]
```

#### 发送阶段：cat + all-gather

CPU→GPU 和 NCCL 通信是**一个原子操作**，不存在"先加载到 GPU 再 all-gather"的分离：

```python
# partition_parameters.py: _all_gather_dtype
# 1. 预分配连续 buffer（world_size 份 partition 位置）
flat_tensor = torch.empty(partition_sz * world_size, dtype=dtype, device=GPU)
partitions = [flat_tensor.narrow(0, partition_sz * i, partition_sz) for i in range(world_size)]

# 2. 所有 param 的 CPU partition cat 到当前 rank 的位置
torch.cat(
    [p.ds_tensor.to(GPU) for p in params],
    out=partitions[rank_in_group])

# 3. 发起 NCCL all-gather，各 rank 把自己的 slice 广播填满 flat_tensor
handle = dist.all_gather_into_tensor(flat_tensor, partitions[rank], group=pg, async_op=True)
```

All-gather 后 `flat_tensor` 的内存布局（按 rank 分块，非按 param 分块）：

```
flat_tensor:
┌────────────────────┬────────────────────┬─────┬────────────────────┐
│rank0: A₀|B₀|C₀    │rank1: A₁|B₁|C₁    │ ... │rankN: Aₙ|Bₙ|Cₙ    │
└────────────────────┴────────────────────┴─────┴────────────────────┘
 ← partitions[0] →   ← partitions[1] →          ← partitions[N] →
```

#### 重组阶段：wait 后按 param 拆分

All-gather 本身不保证 param 连续。各 param 的连续性由 `AllGatherCoalescedHandle.wait()` 重组：

```python
# partition_parameters.py:752-767
param_offset = 0
for param in self.params:
    partitions_for_this_param = []
    for rank in range(world_size):
        # 从每个 rank 的块中，取出该 param 对应位置的分片
        part = self.partitions[rank].narrow(0, param_offset, ds_tensor_numel)
        partitions_for_this_param.append(part)

    # 拼接各 rank 的分片 → 完整 param
    param.data = torch.cat(partitions_for_this_param).view(param.ds_shape)
    param_offset += ds_tensor_numel
```

可视化重组过程：

```
重组 param A:  取各 rank 块中 offset=0 处的 A₀,A₁,...,Aₙ → cat → 完整 A
重组 param B:  取各 rank 块中 offset=|A| 处的 B₀,B₁,...,Bₙ → cat → 完整 B
重组 param C:  取各 rank 块中 offset=|A|+|B| 处的 C₀,C₁,...,Cₙ → cat → 完整 C
```

### 滑动窗口行为

稳态下每个 pre_hook 都推进一次滑动窗口：

```
pre_hook(M0):  M0 已就绪(wait) → prefetch ≤5e7 numel 的后续 params → 一次 all-gather
post_hook(M0): release M0

pre_hook(M1):  M1 已就绪(wait) → prefetch ≤5e7 numel 的后续 params → 一次 all-gather
post_hook(M1): release M1

...每个 pre_hook 补充一次 prefetch，窗口持续向前滑动
```

GPU 上同时驻留的参数量 ≈ 当前 module 参数 + 已 prefetch 的参数，被 `max_live_parameters` 封顶。
当 GPU 余量不足时 `max_params_to_prefetch` 趋近 0，停止 prefetch，等前面 release 腾出空间。

### Release（计算后释放）

```python
# post_forward_hook / post_backward_hook:
param_coordinator.release_sub_module(sub_module)
    → param.data = torch.empty(0, dtype=param.dtype, device=param.device)  # 释放 GPU 上的完整副本
    # CPU 分区 param.ds_tensor 保持不变
```

---

## 二、Grad 的 Reduce-Scatter

### 触发时机

Backward autograd 计算出 `param.grad`（完整大小 = ds_numel）后，通过注册的 grad accumulator hook 触发 reduce。

**Bucket 满触发条件**（`stage3.py:1448-1451`）：
```python
if bucket.elements + param.ds_numel > reduce_bucket_size and bucket.elements > 0:
    # flush 当前 bucket → 执行 reduce-scatter
    self.__reduce_and_partition_ipg_grads()
# 然后把新 param 的 grad 拷入清空后的 bucket
```

此外，当 backward 最后一个 param hook 触发后，`independent_gradient_partition_epilogue()` 会 flush 剩余梯度。

### 流程

```
param.grad 产生 (GPU, 完整大小)
    │
    │ grad hook: reduce_partition_and_remove_grads
    ▼
__add_grad_to_ipg_bucket():
    param.grad copy→ ipg_bucket.buffer (GPU contiguous buffer)
    │
    │ bucket 满 (elements > reduce_bucket_size) 时触发:
    ▼
__reduce_and_partition_ipg_grads() [在 reduce_and_partition_stream 上]:
    │
    ├─ reduce-scatter (NCCL 通信):
    │    输入: 完整 grad (ds_numel)
    │    输出: 本 rank 的 grad partition (ds_numel / world_size)
    │    每个 rank 拿到的 partition 是所有 rank 对应位置 grad 的均值
    │
    ├─ partition_grads(): 写入 grad buffer (见第三节)
    │
    └─ param.grad = None (释放 GPU 上的完整 grad)
```

### IPG Bucket 结构

```python
# stage3.py:124-145 — IPGBucketZ3
class IPGBucketZ3:
    buffer: Tensor   # 预分配 GPU 连续 buffer, 大小 = reduce_bucket_size (默认 5e8 元素)
    params: list     # 已拷入 buffer 的 param 列表
    elements: int    # 当前已填充的元素数
```

梯度通过 `non_blocking copy` 写入 bucket 的连续 buffer：
```python
# __add_grad_to_ipg_bucket (stage3.py:1461)
bucket.buffer.narrow(0, bucket.elements, param.grad.numel()).copy_(param.grad, non_blocking=True)
param.grad.data = bucket.buffer.narrow(0, bucket.elements, param.grad.numel())
bucket.elements += param.grad.numel()
```

### Reduce-scatter 通信前后的数据重组

默认走 `reduce_scatter=True` 路径（`coalesced_collectives.py:158-218`）：

#### 通信前：交错拼接 (interleave + cat)

```python
# 对每个 tensor 按 world_size 切分
for tensor in tensors:
    chunk_sz = ceil(tensor.numel() / world_sz)
    for rank in range(world_sz):
        chunks_for_rank[rank].append(tensor.narrow(0, rank * chunk_sz, chunk_sz))

# 按 rank 交错排列后拼接为一个 flat buffer
# 布局: [t0_chunk_rank0, t1_chunk_rank0, ..., t0_chunk_rank1, t1_chunk_rank1, ...]
tensor_partition_flat_buffer = torch.cat(interleaved_chunks)
tensor_partition_flat_buffer.div_(world_sz)
```

#### 通信：单次 NCCL reduce-scatter

```python
# 每个 rank 拿到自己那段的归约结果
dist.reduce_scatter_fn(output=my_chunk, input=tensor_partition_flat_buffer, group=group)
```

#### 通信后：解交错提取各 param 的 partition

```python
# my_chunk 中各 tensor 的 partition 是连续排列的
offset = 0
for tensor in tensors:
    chunk_sz = ceil(tensor.numel() / world_sz)
    grad_partition = my_chunk.narrow(0, offset, chunk_sz)  # 仅指针操作
    offset += chunk_sz
```

解交错是纯 `narrow()` 操作，无数据拷贝。

### Reduce-scatter 与计算的 Overlap

```
default_stream:              [...Module N-1 backward...]  [...Module N-2 backward...]
                                                            ↑ 不被阻塞
reduce_and_partition_stream: [reduce-scatter N] [partition_grads N] [reduce-scatter N-1] ...
```

reduce-scatter 和 grad buffer 写入都在独立 stream 上，与后续 module 的 backward 计算并行。

---

## 三、Grad Buffer（Main Grad）的累加

### Grad Buffer 位置

由 `self.device` 决定：

```python
self.device = cuda_device if not self.offload_optimizer else 'cpu'

self.grad_partitions_flat_buffer = torch.zeros(
    sum(p.partition_numel() for p in all_params),
    dtype=self.gradient_accumulation_dtype,
    device=self.device)  # GPU 或 CPU

if self.offload_optimizer_pin_memory:
    self.grad_partitions_flat_buffer = pin_memory(self.grad_partitions_flat_buffer)
```

| 配置 | Grad Buffer 位置 |
|------|-----------------|
| `offload_optimizer = False` | **GPU** |
| `offload_optimizer = True` | **CPU** (pinned memory) |

### 累加逻辑（partition_grads）

```python
# stage3.py:1824-1842
grad_buffer = self.__param_id_to_grad_partition[param.ds_id]  # 目标 buffer

if self.micro_step_id == 0:  # 第一个 micro-batch
    grad_buffer.copy_(grad_partition, non_blocking=True)  # 覆写

elif get_accelerator().on_accelerator(grad_buffer):  # buffer 在 GPU
    grad_buffer.add_(grad_partition.to(self.gradient_accumulation_dtype).view(grad_buffer.shape))

else:  # buffer 在 CPU（offload 场景）
    # CPU→GPU: 加载 grad buffer
    cuda_grad_buffer = grad_buffer.to(grad_partition.device, non_blocking=True)
    # GPU 上累加（CPU add 太慢），含 dtype cast
    cuda_grad_buffer.add_(grad_partition.to(self.gradient_accumulation_dtype).view(cuda_grad_buffer.shape))
    # GPU→CPU: 卸载回 CPU
    grad_buffer.copy_(cuda_grad_buffer, non_blocking=True)
```

### CPU Grad Buffer 的加载/卸载时序

仅在 `offload_optimizer = True` + 多 micro-batch（gradient accumulation）时存在：

```
reduce_and_partition_stream 上:

micro_step 0:
  reduce-scatter → grad_partition (GPU)
  grad_buffer.copy_(grad_partition)                    # GPU→CPU (async, 无需加载)

micro_step 1:
  reduce-scatter → grad_partition (GPU)
  cuda_buf = grad_buffer.to(GPU, non_blocking=True)   # ① 加载 CPU→GPU
  cuda_buf.add_(grad_partition)                        # ② GPU 上求和
  grad_buffer.copy_(cuda_buf, non_blocking=True)      # ③ 卸载 GPU→CPU

micro_step 2: 同 micro_step 1 ...
```

**Overlap**：这三步操作在 `reduce_and_partition_stream` 上执行，与 default_stream 上的 backward 计算并行。
但三步之间本身是串行的（数据依赖）。

### Grad Buffer 在 Optimizer Step 时的消费

```python
# offload 场景: grad 已在 fp32_param.grad (CPU) 中就绪
# 非 offload 场景:
self.averaged_gradients[i] = [self.__param_id_to_grad_partition[p.ds_id] for p in sub_group]
fp32_param.grad = flatten(averaged_gradients[i])  # 直接引用 GPU buffer
optimizer.step()  # 消费 grad
```

---

## 四、全景时序图

```
一个 Training Step (gradient_accumulation_steps = 2):

═══ Micro-batch 0 ════════════════════════════════════════════════════

Forward (Module 0→N):
  allgather_stream:  [allgather M0][prefetch M1,M2][allgather M1 ready]...
  default_stream:    [wait][compute M0][wait][compute M1]...
  每个 module: fetch → compute → release

Backward (Module N→0):
  allgather_stream:  [allgather MN][prefetch MN-1,MN-2]...
  default_stream:    [wait][grad MN][wait][grad MN-1]...
  reduce_partition:  [reduce-scatter MN][copy→buffer][reduce-scatter MN-1]...
  grad buffer:       覆写 (micro_step=0)

═══ Micro-batch 1 ════════════════════════════════════════════════════

Forward + Backward: 同上
  reduce_partition:  [reduce-scatter][加载buffer][GPU add][卸载buffer]...
  grad buffer:       累加 (micro_step=1)

═══ Optimizer Step ═══════════════════════════════════════════════════

for each sub_group:
  offload 模式:
    CPU 上: unscale_grad → DeepSpeedCPUAdam.step() → fp32→fp16 cast
    同步 copy: fp32→fp16 cast 后 .data.copy_() 到 GPU partition (供下次 allgather)

  非 offload 模式:
    GPU 上: flatten grads → assign to fp32.grad → optimizer.step() → fp32→fp16 cast
```

---

## 五、关键设计决策总结

| 问题 | 答案 |
|------|------|
| Param 有 prefetch 吗？ | 有，forward/backward 都有，基于 trace 的 prefetch bucket |
| Prefetch 和 all-gather 是分开的吗？ | 不是，prefetch 就是提前发起 all-gather |
| Param offload 和非 offload 的 prefetch 有区别吗？ | 逻辑完全相同，区别仅在于有无 CPU→GPU DMA 这一步 |
| Grad 有 prefetch 吗？ | 没有，grad 是 backward 的产物，不需要提前加载 |
| Grad buffer 什么时候需要加载/卸载？ | 仅 offload + 多 micro-batch 累加时：加载到 GPU 做 add，再卸载回 CPU |
| Grad 通信和计算有 overlap 吗？ | 有，reduce-scatter 在独立 stream 上与下一个 module 的 backward 并行 |
| Optimizer 支持混合模式吗？ | 支持，通过 `offload_ratio` 可将部分 sub_group 放 GPU（用 backup AdamW），部分放 CPU（用 DeepSpeedCPUAdam） |
