# DeepSpeed 激活值 Offload 与重计算实现分析

> 基于 `/Users/zhangbochun/Desktop/AI-Infra/Megatron/Megatron-Scale-Down/deepspeed` 工作树分析，代码状态截至 2026-09-16。

## 1. 结论先行

DeepSpeed 把“激活值节省”拆成两个层次：

1. **Activation checkpointing / recompute**：前向只保存 checkpoint 输入，不保存 checkpoint 内部中间激活；反向第一次需要这些中间激活时，用相同输入重新执行一遍前向，再对重算图做反向。这是主要的计算换显存策略。
2. **Activation offload**：checkpoint 输入仍然需要跨越 forward/backward 生命周期保存，因此可以把它们从 GPU 搬到 CPU。当前代码有两条路径：
   - 原生 `checkpointing.py` 的 **CPU checkpoint**：分片模式下先切分再搬 CPU；非分片模式下使用共享的异步 offload engine，在 pinned CPU buffer 和 side stream 上进行 D2H/H2D。
   - `offload_activations.py` 的 **HF 非重入 checkpoint 输入 offload**：通过 `saved_tensors_hooks` 只拦截被 HF `GradientCheckpointingLayer` 标记的 hidden states，其他 autograd 保存张量原样通过。

因此，offload 并没有消除重计算：反向仍然需要恢复 checkpoint 输入，然后重新运行 checkpoint 函数；它只是进一步降低“少量必须保存的输入”的 GPU 占用。

## 2. 代码入口和状态

核心文件：

| 文件 | 作用 |
| --- | --- |
| `deepspeed/runtime/activation_checkpointing/checkpointing.py` | 原生 checkpoint、RNG 保存/恢复、分片、连续缓冲、CPU checkpoint、重算和反向 |
| `deepspeed/runtime/activation_checkpointing/offload_activations.py` | 异步 CPU offload engine、HF 非重入 checkpoint hook、pinned buffer pool |
| `deepspeed/runtime/activation_checkpointing/config.py` | `activation_checkpointing` 配置解析 |
| `tests/unit/runtime/activation_checkpointing/` | 重算、CPU offload、异步 stream 和生命周期测试 |

原生路径由几个全局开关控制：`PARTITION_ACTIVATIONS`、`CPU_CHECKPOINT`、`CONTIGUOUS_CHECKPOINTING`、`SYNCHRONIZE`、`PROFILE_TIME`。配置通过 `deepspeed.checkpointing.configure(...)` 或 DeepSpeed JSON 的 `activation_checkpointing` 节读取。

## 3. 原生 checkpoint 的 forward/backward 时序

### 3.1 Forward：只保留输入

`CheckpointFunction.forward`（重入路径）和 `non_reentrant_checkpoint` 都遵循相同的总体思想：

1. 记录 CPU RNG、设备 RNG 以及 model-parallel RNG tracker 状态。
2. 依据配置处理 checkpoint 输入：
   - 普通模式：直接使用输入；
   - `partition_activations`：每个输入只保留当前 model-parallel rank 的 partition；
   - `cpu_checkpointing`：将 checkpoint 数据放到 CPU，或者注册到异步 offload engine；
   - `contiguous_memory_optimization`：把 partition 写入预分配的连续 buffer。
3. 在 `torch.no_grad()` 下执行 `run_function(*inputs_cuda)`，所以 checkpoint 内部中间激活不会建立正常的 autograd 保存链。
4. 保存少量元数据和 checkpoint 输入，而不是保存内部激活。对输入的 `Tensor`、非 Tensor 参数和输出顺序分别记录，保证反向时可以重建原始调用签名。

`non_reentrant_checkpoint` 使用 `saved_tensors_hooks` 记录前向中真正被 autograd 请求的保存张量，并在第一次 unpack 时触发重算；它可以处理“输入不要求 grad、但模块参数要求 grad”的场景。

### 3.2 Backward：恢复输入、重算、再反向

反向阶段的关键顺序是：

1. 恢复/聚合 checkpoint 输入：
   - 分片输入通过 `gather_partitioned_activations` 聚合；
   - CPU 输入通过同步 H2D 或异步 engine restore；
   - 普通输入直接使用。
2. 对恢复后的输入调用 `detach_variable`，保留原来的 `requires_grad` 语义并建立新的重算图。
3. 临时把 RNG 状态切换为 forward 结束时保存的状态。
4. 在 `torch.enable_grad()` 下重新执行 `run_function`。重入路径会在这次执行中保存内部张量；非重入路径的 `replay_pack` 按第一次 forward 的保存顺序把重算激活放入 holder/storage。
5. 恢复进入 backward 前的 RNG 状态，避免重算改变外层随机数序列。
6. 对重算输出调用 `torch.autograd.backward(output_tensors, grad_tensors)`，把梯度传回原图。
7. 清理保存的 tensor、连续 buffer 和临时引用。`SYNCHRONIZE`/`PROFILE_TIME` 只影响同步与计时，不改变数学结果。

这意味着一次 checkpoint 区域通常需要额外执行一次 forward compute；显存收益来自不保存内部激活，代价是重算 FLOPs 和可能的通信/数据搬运。

## 4. 原生 CPU checkpoint 与 offload

### 4.1 可保存哪些输入

`is_activation_to_checkpoint` 只接受满足以下条件的对象：浮点 Tensor、元素数至少为 model-parallel size、且没有 `no_checkpointing=True`。非浮点 Tensor、太小的 Tensor、标记为不 checkpoint 的 Tensor 和非 Tensor 参数不会进入 activation offload。

### 4.2 分片路径：`partition_activations + cpu_checkpointing`

`partition_activations(args, cpu_checkpoint, contiguous_checkpoint)` 对每个候选 Tensor 执行：

1. `detach().contiguous().view(-1)`；
2. 按 `mp_size` 计算当前 rank 的 `partition_size` 和起始 offset；
3. clone 出本 rank 的 partition；
4. `cpu_checkpoint=True` 时把 partition 放在 CPU，否则留在当前设备。

反向时，原始 Tensor 对象的数据指针被置为空，partition 通过 `saved_data` 保存；`gather_partitioned_activations` 重新建立完整输入后，才进入重算。因此 GPU 保存量近似从完整激活降为 `1 / mp_size`，CPU offload 再把这部分 partition 移出 GPU。

这条路径使用的是显式 partition/gather 逻辑，而不是 `_ActivationOffloadEngine`。当前实现对分片 CPU checkpoint 的 copy 是同步语义；异步 side-stream engine 只用于下面的非分片路径。

### 4.3 非分片路径：异步 D2H/H2D

当 `CPU_CHECKPOINT=True` 且未启用 `PARTITION_ACTIVATIONS` 时，`_get_cpu_offload_engine()` 创建一个共享的 `_ActivationOffloadEngine`（默认 `keep_last_count=1`、`min_offload_bytes=0`）。

Forward 的 `get_offloaded_activations_for_backward`：

1. 对每个候选输入调用 `engine.offload_input(arg.detach())`；重复引用按 `id` 去重。
2. engine 为 Tensor 分配 dense CPU buffer，默认 pin memory。
3. 在独立 side stream 上等待当前 compute stream，然后执行 non-blocking D2H copy。
4. 用 `ds_offload_id` 记录 token，随后把原 Tensor 的 `.data` 换成空 Tensor，从而释放 GPU storage；engine 内的 detached alias 保证 D2H 完成前源 storage 仍然有效。

Backward 的 `restore_offloaded_activations`：

1. 根据 token 在 CPU tracker 中找到 buffer；
2. 在 side stream 上分配 `empty_strided` GPU Tensor 并执行 non-blocking H2D；
3. 记录 event，让 compute stream `wait_event`；
4. 原地写回 `t.data`，保留 autograd Tensor 对象身份和 `requires_grad`；
5. 恢复完成后才执行 checkpoint 重算。

`keep_last_count` 默认保留最近一个输入在 GPU，避免刚完成 forward 时第一个 backward 立刻等待最晚的一次 D2H。`max_fwd_stash_count` 控制仍持有 GPU 引用、等待 side stream 排空的前向窗口；`max_cpu_buffer_pool_count` 控制可复用的 CPU buffer 数量。engine 会用 `record_stream` 和 CUDA event 管理 allocator 生命周期。

## 5. HF 非重入 checkpoint 的定向 offload

`CheckpointHiddenStatesOffload` 继承 `_ActivationOffloadEngine` 和 `saved_tensors_hooks`，目标是 Transformers 的非重入 gradient checkpointing：

1. context `__enter__` 时临时 monkey-patch `GradientCheckpointingLayer.__call__`。
2. 只有处于 training、启用了 gradient checkpointing、且 `args[0]` 是 hidden states 时，才调用 `manager.mark(hidden_states)`。
3. `pack` hook 看到被标记的 Tensor 后才执行 offload，并把它替换为轻量 `_OffloadedTensorRef(tensor_id)`；未标记 Tensor 原样返回。
4. `unpack` hook 在反向需要该 Tensor 时恢复它。`cache=True` 支持 retain-graph 下同一 token 被多次 unpack。
5. context 退出时同步 side stream、回收 pending CPU buffers、清除 tracker，并恢复原始 `GradientCheckpointingLayer.__call__`。嵌套 manager 用线程本地 stack 和 patch-user 计数保护。

它与原生 `checkpointing.py` 的区别是：它不接管整个 checkpoint 函数，也不修改最终 norm/head 或普通模块保存的 Tensor；它只针对 HF 标记为 checkpoint 输入的 hidden states。`backward()` 必须在同一 context 内执行，否则 offload engine 可能已被清理。

## 6. 连续内存优化

`contiguous_memory_optimization=True` 只能与 `partition_activations=True` 一起使用，且必须提供 `number_checkpoints`。实现为每类 partition 预分配 `num_layers` 个固定大小的 buffer，并用 `data_offsets` 循环写入；Tensor shape 元数据另外写入 `contiguous_size_buffers`。

收益有两点：

- 避免大量不同大小的 checkpoint allocation，降低 allocator fragmentation；
- CPU buffer 场景下通过预触碰 mmap page，尽量把 page fault 成本移到实际 copy 之前的窗口。

它要求 checkpoint 形状/大小具有同质性；在 backward 开始时 buffer 引用会被清空。eval 场景若只有 forward 没有 backward，应调用 `deepspeed.checkpointing.reset()`，否则连续 buffer 或 offload engine 的引用会跨 step 累积。

## 7. 配置组合与限制

```json
{
  "activation_checkpointing": {
    "partition_activations": false,
    "cpu_checkpointing": false,
    "contiguous_memory_optimization": false,
    "number_checkpoints": null,
    "synchronize_checkpoint_boundary": false,
    "profile": false
  }
}
```

| 组合 | 实际行为 | 主要代价/限制 |
| --- | --- | --- |
| 全部关闭 | checkpoint 仍可重算，但输入留在 GPU | 只节省 checkpoint 内部激活，不节省输入 |
| `cpu_checkpointing` | 输入搬到 CPU；非分片 CUDA 路径使用异步 pinned side stream | PCIe/NVLink 搬运和 CPU 内存占用 |
| `partition_activations` | 每个 model-parallel rank 只保存自己的 partition，反向再 gather | 需要 model-parallel group；反向有 gather 通信 |
| 两者同时开启 | partition 后把 partition 放 CPU | 当前分片路径不是共享异步 engine 路径 |
| `contiguous_memory_optimization` | partition checkpoint 使用预分配连续 buffer | 必须 partition、必须 `number_checkpoints`、checkpoint 尺寸应同质 |
| `synchronize_checkpoint_boundary` | 每次 checkpoint forward/backward 边界设备同步 | 便于计时/边界一致性，但会损失 overlap |
| `profile` | 记录每个 checkpoint 的 forward/backward 时间 | 只影响计时与日志 |

额外边界条件：

- 同步型 accelerator 没有可用 side stream 时，engine 自动退化为同步 copy。
- CPU、Parameter、Buffer、过小 Tensor、stride 为 0 的重叠 view 会被跳过或原样保存。
- 原生 checkpoint 不支持在该路径中用 `.grad()` 代替 `.backward()`；代码显式检查 `torch.autograd._is_checkpoint_valid()`。
- 随机算子依赖 RNG 状态恢复；如果 checkpoint 函数有外部不可恢复状态，重算仍可能不等价。
- 异步 offload 的 pinned memory 受 `get_accelerator().pin_memory()` 和 `DS_PIN_MEMORY_BACKEND` 影响；未正确注册的 host memory 会使 side-stream copy 退化为同步。

## 8. 计算与性能取舍

可以把单个 checkpoint 区域的成本近似写成：

```text
总 step 成本 = 原始 forward + backward
             + checkpoint forward 重算
             + D2H/H2D 搬运（启用 CPU offload 时）
             + partition gather/scatter 通信（启用分片时）
```

`use_streams=True` 的目标是把 D2H/H2D 放在 compute stream 之外：forward 期间提前搬运较早 checkpoint 输入，backward 期间通过 event 等待恢复完成。它不能消除带宽消耗，只能利用 copy/compute overlap；`keep_last_count` 和 stash window 是首个 backward 延迟与 GPU 峰值之间的调节旋钮。

选择建议：

- GPU 主要被 checkpoint 内部激活占满时，先启用 recompute；
- GPU 仍被长序列 hidden-state checkpoint 输入占满时，再启用 CPU offload；
- 有 model parallel 且输入很大时，考虑 partition，再评估 gather 通信；
- allocator fragmentation 明显时使用 contiguous buffer，并准确设置 checkpoint 数量；
- 对 HF 非重入 checkpoint 优先使用定向 `CheckpointHiddenStatesOffload`，避免把非 checkpoint 激活不必要地搬到 CPU。

## 9. 源码验证点

- `checkpointing.py:387`：partition、CPU 和 contiguous checkpoint 输入的构造。
- `checkpointing.py:501`：共享异步 CPU engine 的选择条件。
- `checkpointing.py:520`：offload 输入、清空 GPU `.data`、记录 token。
- `checkpointing.py:557`：restore 输入并原地恢复 Tensor identity。
- `checkpointing.py:575`：`CheckpointFunction.forward` 的 no-grad 执行和保存策略。
- `checkpointing.py:668`：backward 的恢复、RNG 恢复、重算和 autograd backward。
- `checkpointing.py:1135`：JSON 配置映射到全局开关。
- `offload_activations.py:80`：buffer pool、side stream、tracker 和 keep-last 状态。
- `offload_activations.py:187`：异步 D2H；`offload_activations.py:236`：异步 H2D 和 event 等待。
- `offload_activations.py:391`：HF marker patch；`offload_activations.py:438`：pack/unpack hook。
