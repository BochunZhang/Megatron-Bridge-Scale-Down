# DeepSpeed Activation Offload + Recompute 分析

## 概述

Activation offload 和 recompute（activation checkpointing）在 DeepSpeed 中是**一体设计**，offload 是 recompute 框架内的一个增强选项，不能独立使用。

核心思想：
- **Recompute**：不保存 forward 中间激活，backward 时重新计算
- **CPU Offload**：连 checkpoint 边界的 inputs 也卸载到 CPU，进一步压缩 GPU 峰值显存

## 源码位置

主实现：`deepspeed/runtime/activation_checkpointing/checkpointing.py`

| 组件 | 位置 |
|------|------|
| `CheckpointFunction.forward` | Line 500 |
| `CheckpointFunction.backward` | Line 581 |
| `partition_activations()` | Line 377 |
| `gather_partitioned_activations()` | Line 266 |
| 配置入口 | Line 1005 (`configure()`) |

## 配置项

```python
PARTITION_ACTIVATIONS = False   # 按 MP rank 切分激活
CPU_CHECKPOINT = False          # 将激活 offload 到 CPU
CONTIGUOUS_CHECKPOINTING = False # 使用预分配连续 CPU buffer
```

三种组合模式：
1. **CPU_CHECKPOINT only**：整个激活 tensor 拷到 CPU
2. **PARTITION_ACTIVATIONS + CPU_CHECKPOINT**：先按 MP 切分（只存 1/mp_size），再拷到 CPU
3. **PARTITION + CPU + CONTIGUOUS**：在预分配的连续 pinned buffer 中存放，避免 page fault

## 数据流时序

### Forward（CheckpointFunction.forward）

```
checkpoint(function, *inputs):
│
├─ 1. 保存 inputs 到 CPU
│     ├─ CPU_CHECKPOINT 模式:
│     │   inputs_cpu = copy_to_device(args, device='cpu')
│     │   → 同步 .to('cpu')，无 async
│     │
│     └─ PARTITION + CPU 模式:
│         partition = args[i].narrow(0, rank*chunk, chunk)  # 只取 1/mp_size
│         partition = partition.cpu()
│
├─ 2. 在 GPU 上执行 forward
│     inputs_cuda = copy_to_device(args, device=cuda_device)
│     outputs = run_function(*inputs_cuda)
│     # 中间激活在 no_grad() 下计算，不会被 autograd 保留
│
├─ 3. 记录 CPU 副本供 backward 使用
│     arg.saved_data = cpu_copy
│
└─ 4. 释放 GPU 上的 inputs
      arg.data = torch.empty([])  # 只剩空壳
```

### Backward（CheckpointFunction.backward）

```
CheckpointFunction.backward(*grads):
│
├─ 1. 从 CPU 恢复 inputs 到 GPU（同步阻塞）
│     for t in ctx.deepspeed_saved_tensors:
│         t.data = t.saved_data.to(t.device)  # CPU→GPU，阻塞等待完成
│         t.saved_data = None
│
├─ 2. 若 PARTITION：all_gather 重建完整激活
│     inputs = gather_partitioned_activations(saved_tensors, device=cuda_device)
│     # dist.all_gather_into_tensor() 跨 MP ranks
│
├─ 3. 恢复 RNG state（保证 recompute 数值一致性）
│     torch.set_rng_state(ctx.fwd_cpu_rng_state)
│     set_cuda_rng_state(ctx.fwd_cuda_rng_state)
│
├─ 4. Recompute forward（重新计算中间激活）
│     with torch.enable_grad():
│         outputs = run_function(*detached_inputs)
│
└─ 5. 正常 backward 计算梯度
      torch.autograd.backward(outputs, grads)
```

## 关键设计决策

### 为什么 reload 没有 prefetch？

与 weight offload 不同，activation reload 是**同步阻塞**的（`.to(device)` 无 `non_blocking`）。原因：

1. **卸载的 tensor 很小**：只存 checkpoint 边界的 inputs（一层的输入 hidden states），不是全部中间激活
2. **Recompute 是主开销**：CPU→GPU memcpy 时间相比重跑整个 forward segment 很小
3. **无计算可重叠**：inputs 拷回后立刻用于 recompute，没有其他工作可以并行
4. **频率低**：每 N 层一个 checkpoint 边界，不像 weight offload 每个 module 都搬

### Recompute 和 Offload 的互补关系

| 方案 | GPU 显存中保留 | 峰值显存 |
|------|---------------|---------|
| 无 checkpoint | 所有中间激活 | 最大 |
| Recompute only | checkpoint inputs 在 GPU | 中等 |
| Recompute + CPU offload | checkpoint inputs 在 CPU，GPU 只有 outputs | 最小 |

Recompute 解决"中间激活太多"→ 不存中间结果，backward 重算。
CPU offload 解决"连 inputs 都不想在 GPU 上留"→ inputs 搬到 CPU，进一步释放显存。

## Contiguous Checkpointing 优化

当 `CONTIGUOUS_CHECKPOINTING = True` 时（需配合 `PARTITION_ACTIVATIONS`）：

```python
# 预分配连续 CPU buffer（所有层共享）
contiguous_data_buffers = [torch.empty(total_size, dtype=dtype, device='cpu', pin_memory=True)]

# Forward 时直接写入 buffer 的对应 offset
buffer[offset:offset+size].copy_(partition)

# 优势：
# 1. 避免每次 malloc/free 的开销
# 2. pin_memory 保证 DMA 效率
# 3. 连续内存避免 page fault
```

## 与 Weight Offload 的对比

| 维度 | Activation Offload | Weight Offload |
|------|-------------------|---------------|
| 卸载对象 | checkpoint 边界 inputs | 参数分区 (ds_tensor) |
| 卸载时机 | Forward 结束时 | 初始化时就在 CPU |
| Reload 时机 | Backward 开始时 | Forward/Backward pre-hook |
| 异步 prefetch | 无（同步阻塞） | 有（allgather_stream + trace-based） |
| 传输频率 | 每 checkpoint 段一次 | 每 module 每次 forward+backward |
| 传输量 | 小（一层 hidden states） | 大（完整参数 all-gather） |
| 额外代价 | Recompute 计算开销 | All-gather 通信 + CPU↔GPU 带宽 |

## FPDT 高级异步方案（参考）

在 `deepspeed/runtime/sequence/fpdt_layer.py` 中有更精细的 stream-based activation offload：

```python
offload_stream = get_accelerator().Stream()

# Forward: 在 offload_stream 上异步 offload
cpu_chunk = torch.empty(chunk.shape, dtype=chunk.dtype, device='cpu', pin_memory=True)
cpu_chunk.copy_(gpu_chunk, non_blocking=True)  # GPU→CPU，不阻塞 compute stream

# Backward: 异步 reload
gpu_chunk.copy_(cpu_chunk, non_blocking=True)   # CPU→GPU
compute_stream.wait_stream(offload_stream)       # 需要数据时才同步
```

这是针对长序列注意力的特殊优化，不是通用 activation checkpoint 路径。
