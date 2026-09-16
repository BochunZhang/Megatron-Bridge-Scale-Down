# DeepSpeed ZeRO-Infinity、SuperOffload 与三层 Offload 实现分析

本文基于本地 DeepSpeed 源码（`/Users/zhangbochun/Desktop/AI-Infra/Megatron/Megatron-Scale-Down/deepspeed`）整理，重点回答三个问题：

1. ZeRO-Infinity 和代码中的 SuperOffload（用户所称的 “super-infinity” 可能指它）分别卸载什么；
2. 参数、梯度、optimizer state、激活值如何卸载到 CPU；
3. NVMe 如何与 GPU、CPU 组成三层存储并实现预取。

## 1. 先区分名称：ZeRO-Infinity 与 SuperOffload

### 1.1 ZeRO-Infinity 是 ZeRO-3 的异构内存 offload 能力

ZeRO-3 已经把 optimizer states、gradients、parameters 按 data-parallel rank 分片；ZeRO-Infinity 在此基础上允许这些分片驻留在 CPU 或 NVMe，并在模块执行前取回。官方教程明确说 ZeRO-Infinity 可把完整 model state offload 到 CPU/NVMe，而 ZeRO-Offload（典型是 ZeRO-2）主要把 optimizer 和 gradient states 放到 CPU。

源码入口是 `DeepSpeedZeRoOffload`：

- `offload_param_config` 非空且 device 不是 `none` 时，设置 `offload_device` 和 pinned-memory 策略（`deepspeed/runtime/zero/parameter_offload.py:163-173`）；
- 将普通参数转换成 ZeRO 参数，并在初始化时把 `remote_device=self.offload_device` 传给 `zero.Init`（`parameter_offload.py:252-273`）；
- 创建 `PartitionedParameterCoordinator(prefetch_nvme=...)`，当参数 device 为 NVMe 时启用 NVMe 分区预取（`parameter_offload.py:206-218`）；
- forward/backward 的 pre-hook 调用 `fetch_sub_module`，post-hook 调用 `release_sub_module`（`parameter_offload.py:524-599`）。

因此，ZeRO-Infinity 的“卸载对象”不是单一的权重，而是：

| 对象 | 分片/卸载含义 |
| --- | --- |
| 参数 | ZeRO-3 参数分片；空闲分片可位于 CPU 或 NVMe，计算前 all-gather 成完整参数 |
| 梯度 | backward 后按 rank 分片/reduce-scatter；可写入 CPU/NVMe 的 optimizer swapper 路径 |
| optimizer states | fp32 master parameter、momentum、variance 等按分片保存；可在 CPU 或 NVMe 上更新 |
| 激活值 | 不是 ZeRO-Infinity 参数 offload 的必然组成部分；由 activation checkpointing 的独立 CPU offload 机制处理 |

### 1.2 代码中没有独立的 “Super-Infinity” 类

当前仓库实现的相关名称是 `SuperOffloadOptimizer_Stage3` 和 `SuperOffloadCPUOptimizer`，不是 `SuperInfinity`。它继承 ZeRO-3 optimizer（`deepspeed/runtime/superoffload/superoffload_stage3.py:27`），复用 ZeRO-3 的参数分片、梯度分片、CPU/NVMe offload；新增部分主要是：

- 主进程启动一个 `torch.multiprocessing` worker；
- worker 内创建 `DeepSpeedCPUAdam`，接收参数和梯度，执行 optimizer step；
- worker 与 PyTorch 主进程做 CPU core affinity 划分；
- optimizer step 结果异步回传，主进程把更新后的 fp32 参数写回或继续 swap out。

`SuperOffload` 并没有引入新的 NVMe 层级，也不把激活值交给 CPU Adam worker。它优化的是“CPU optimizer computation 的调度与并行”，而不是新的模型状态种类。

## 2. 参数、梯度、optimizer state、激活值的卸载

### 2.1 参数：ZeRO-3 分片 + CPU pinned memory 或 NVMe

参数在模块实际执行前才变为可用：

1. `pre_sub_module_forward_function` / `pre_sub_module_backward_function` 调用 coordinator 的 `fetch_sub_module`；
2. coordinator 计算当前模块缺失参数，提交 all-gather，并等待参数达到 `AVAILABLE`；
3. 模块执行完成后，post-hook 调用 `release_sub_module`，按 `max_live_parameters`、`max_reuse_distance`、持久化阈值决定是否释放；
4. 释放后的参数回到分片状态；如果分片最终位置是 NVMe，则其 `ds_tensor` 数据会被 swapper buffer 代替。

CPU offload 使用 `zero.Init(remote_device=cpu, pin_memory=...)` 预先在 CPU 建立参数存储。pinned CPU buffer 便于 GPU↔CPU 异步 DMA，但会占用不可换出的 host memory。

NVMe 参数由 `AsyncPartitionedParameterSwapper` 管理（`deepspeed/runtime/swap_tensor/partitioned_param_swapper.py:36-100`）：

- 每个 rank 建立自己的 `zero_stage_3/<dtype>params/rank<rank>` 目录；
- 预分配 `buffer_count × buffer_size` 的 buffer pool；
- `swap_out` 通过 `swap_out_tensors` 提交异步写，写完成后释放参数的内存映射；
- `swap_in` 为参数分配 buffer，通过 libaio 或可选 GDS 读入；完成后把 `param.ds_tensor.data` 指向 compute buffer，并把状态置为 `AVAILABLE`；
- `AVAILABLE`、`NOT_AVAILABLE`、`INFLIGHT` 三种状态用于避免重复读写。

### 2.2 梯度：产生在 GPU，分片后异步搬到 CPU/NVMe

ZeRO stage 2/3 的梯度在 backward 中先被 reduce/reduce-scatter，随后落入每个 rank 的 gradient partition。启用 optimizer offload 后：

- CPU offload：gradient partition 通过 pinned buffer 异步复制到 CPU；optimizer computation 也在 CPU 执行；
- NVMe offload：`PartitionedOptimizerSwapper` 的 `gradient_swapper` 对梯度做异步 NVMe write/read（`deepspeed/runtime/swap_tensor/partitioned_optimizer_swapper.py:35-46`）；
- 在需要 optimizer step 或恢复梯度时，`swap_in_optimizer_state` 同时准备 optimizer tensors 和 gradient buffer（`partitioned_optimizer_swapper.py:65-94`）。

梯度不是永久保存成完整模型梯度：它通常只保留当前 rank 的 partition，并在梯度累积边界执行归一化、reduce 后交给 optimizer。参数释放时也会清除 `param.grad`，避免 GPU 引用继续占用显存。

### 2.3 optimizer states：CPU 计算，CPU 或 NVMe 保存

配置 `offload_optimizer.device=cpu` 时，optimizer states 在 CPU 保存并由 `DeepSpeedCPUAdam` 更新；配置为 `nvme` 时，states 以 NVMe 文件为后备，step 前读入 pinned buffers，step 后写回 NVMe。`buffer_count` 至少应覆盖 optimizer state 数量；Adam 常见需要 parameter、gradient、momentum、variance 四类 buffer。

ZeRO-Infinity 的 tile/sub-group 机制由 `sub_group_size` 控制：参数被切成可处理的 tile，减少一次 optimizer step 所需的 CPU/NVMe 工作集。`pipeline_read` 和 `pipeline_write` 允许“读下一 tile / 写上一 tile”和当前 tile 的 optimizer compute 重叠。

### 2.4 SuperOffload：把 CPU Adam 放到独立 worker

`SuperOffloadCPUOptimizer` 启动独立进程（`superoffload_utils.py:183-207`）。主进程通过队列传递：

- fp32 partitioned parameter；
- 对应 gradient；
- parameter group、sub-group、learning rate 以及可选 rollback 标志。

worker 预分配一个 CPU gradient scratch buffer，把 GPU/主进程传来的梯度复制到 CPU，然后调用 `DeepSpeedCPUAdam.step_subgroup`，将更新后的 parameter 返回（`superoffload_utils.py:99-145`）。ZeRO-3 在 backward 完成 sub-group 后异步 `async_step`（`superoffload_stage3.py:166-185`），下一次 `step()` 再等待未完成的结果。这里优化的是 CPU 计算流水线，不改变“参数/梯度/state 可以被 ZeRO offload”的数据模型。

### 2.5 激活值：独立的 autograd saved-tensor CPU offload

激活值由 `deepspeed/runtime/activation_checkpointing/offload_activations.py` 处理，而不是 `offload_param` / `offload_optimizer`：

- `CheckpointHiddenStatesOffload` 使用 `torch.autograd.graph.saved_tensors_hooks`；
- 只标记 gradient-checkpointing layer 的 hidden states，普通 saved tensor 原样保留；
- `_start_offload` 在专用 stream 上把 GPU tensor D2H copy 到 CPU（可选 pinned memory），并记录 device、shape、stride（`offload_activations.py:183-209`）；
- 为避免刚产生的激活还没被第一次 backward 使用就被搬走，`keep_last_count` 个 tensor 留在 GPU；
- `_restore_tensor` 在 backward 消费前从 CPU H2D，使用 event/stream 建立依赖，随后回收 CPU buffer（`offload_activations.py:232-290`）。

因此“offload 激活值”与 ZeRO 的参数 offload 是两条独立路径：前者通过 autograd hooks 保护反向所需的 saved tensors，后者通过模块 hooks 管理参数生命周期。

## 3. GPU → CPU → NVMe 三层与预取

### 3.1 三层数据位置

在 NVMe 配置下，一个参数分片可处于以下位置：

```text
NVMe 文件  --(异步 read, libaio/GDS)-->  CPU pinned buffer
CPU pinned buffer  --(异步 H2D)-->  GPU compute/all-gather buffer
GPU 参数使用完成  --(D2H 或直接 swap buffer)--> CPU/NVMe
```

GPU 只保留当前模块和预取窗口内的参数；NVMe 模式下 CPU 可能保留部分 partition 作为 host-side 工作集，其余可 swap 的分片落在 NVMe。参数 all-gather 仍发生在 data-parallel ranks 之间，NVMe 只负责每个 rank 自己那份 partition 的本地持久化。

### 3.2 参数预取的两级流水

`PartitionedParameterCoordinator.fetch_sub_module` 明确按三步执行：提交当前模块 fetch、提交后续模块 prefetch、等待当前模块参数完成。完成 trace 后：

1. 从 parameter trace queue 取出后续参数；
2. 以 `prefetch_bucket_size` 和 `max_live_parameters` 限制预取元素数；
3. 对普通 CPU/GPU resident 参数直接提交后续 all-gather；
4. 对 `final_location == nvme` 且当前状态为 `NOT_AVAILABLE` 的参数，停止普通队列扫描，把 NVMe 预取交给 `_prefetch_nvme_param_partitions`（`partitioned_param_coordinator.py:420-479`）；
5. NVMe swapper 异步 `swap_in`，把参数标记为 `INFLIGHT`；读取完成后转为 `AVAILABLE`，后续 all-gather 才能使用。

这种“在 CPU/GPU all-gather prefetch 和 NVMe local read 之间保持 trace 顺序”的设计，避免了 NVMe 读乱序导致参数 fetch 顺序与模块执行顺序不一致。

## 3.2.1 只开启 CPU offload 时：prefetch 与计算的重叠

这里的“只开启 CPU offload”需要区分两种常见配置：

- ZeRO-2 + `offload_optimizer.device=cpu`（旧配置 `cpu_offload=true` 的主要含义）：参数仍由各 GPU 保留，CPU 只承载 optimizer states 和梯度；因此没有 ZeRO-3 参数 fetch/prefetch，只有 backward 中的梯度 reduce/scatter、GPU→CPU copy 与 CPU optimizer step 的重叠。
- ZeRO-3 + `offload_param.device=cpu`：参数 partition 在 CPU（通常是 pinned CPU）中保存，模块使用时才 all-gather 到 GPU。

对第二种情况，forward 和 backward 使用完全相同的 coordinator 算法，只是 `forward` 标志决定计时事件名称和 trace 所处方向：

1. 当前模块 pre-hook 调用 `fetch_sub_module`，找出 `ds_status=NOT_AVAILABLE` 的参数；
2. `__all_gather_params` 在 `__allgather_stream` 上提交异步 all-gather。all-gather 的 input 是 `param.ds_tensor`，其存储位于 CPU；pinned memory 允许 DMA 在通信 stream 上非阻塞地把 CPU partition 送到 GPU；
3. 对当前模块，随后在同一 stream/handle 上 wait，保证计算开始前参数为 `AVAILABLE`；
4. 完成一次 trace 后，coordinator 从后续 parameter queue 取出不超过 `prefetch_bucket_size`、且不超过 `max_live_parameters` 的参数，提前提交下一批 all-gather；
5. 当前模块 forward/backward 计算在默认 compute stream 上进行时，下一批 CPU→GPU copy + data-parallel all-gather 在 `__allgather_stream` 上推进。真正需要下一模块参数时才 wait，因此 copy/通信延迟可以被当前模块计算隐藏；
6. post-hook 根据 reuse distance 和持久化规则 partition/release 参数，释放 GPU full parameter，保留 CPU partition。

如果 `overlap_comm=false`，`__allgather_stream` 退化为 default stream，通信与计算基本串行；如果 accelerator 不支持异步 stream，也不会获得上述重叠。`param.record_stream` 和 all-gather handle 的 event wait 用于防止 CPU/GPU buffer 在异步操作完成前被复用。

backward 的区别不在搬运方向，而在模块触发顺序：反向 hook 按反向 trace 访问模块，coordinator 从对应的 backward parameter queue 预取即将执行的模块参数。当前 backward 模块的参数 fetch/wait 可与前一个模块的梯度计算、reduce-scatter（由 ZeRO-3 的 reduce stream 管理）并行；但在某个模块真正读取参数前仍必须完成该模块的 fetch wait。

## 3.2.2 CPU + NVMe 参数 offload 时：何时从 CPU，何时从 NVMe

若配置 `offload_param.device=nvme`，并不是所有参数都必然先从 NVMe 读取。`Init._partition_param` 会按分片大小和 swapper 的 `swappable_tensor` 判断：可 swap 的大分片设置 `final_location=nvme` 并写入 NVMe；不可 swap 的小分片（或持久化参数）直接保留在 CPU/GPU。运行时可按下面规则判断来源：

典型的组合是 `offload_optimizer.device=cpu` + `offload_param.device=nvme`：前者决定 optimizer states/optimizer computation 使用 CPU，后者决定参数 partition 采用上述 CPU/NVMe 混合后备；两者分别作用于不同的 ZeRO 状态，不是把同一 tensor 同时设置成两个 device。

| 运行时状态 | 预取来源 | 触发路径 |
| --- | --- | --- |
| `param.nvme_swapper is None`，或分片不是 `final_location=nvme` | CPU partition | 普通 `__all_gather_params`；all-gather input 直接是 CPU `ds_tensor` |
| `final_location=nvme` 且 `ds_tensor.status=NOT_AVAILABLE` | NVMe 文件 | `_prefetch_nvme_param_partitions` 调用 `nvme_swapper.swap_in(..., async_op=True)` |
| `final_location=nvme` 且 `status=INFLIGHT` | 已经在 NVMe→swap buffer 途中 | 不重复提交 read；需要时 `synchronize_reads()` 等待 |
| NVMe read 完成，`status=AVAILABLE` | CPU-like swap buffer | 普通 all-gather 使用已就绪的 `ds_tensor`，再把完整参数放到 GPU |

forward/backward 的共同时序如下：

```text
模块 N pre-hook
  ├─ 当前参数：CPU ds_tensor 直接 all-gather，或等待/同步 NVMe swap-in
  ├─ 当前参数 wait，模块 N 在 GPU 上计算
  └─ trace 完整时扫描后续 queue
       ├─ CPU resident 参数：提交下一批 GPU all-gather
       └─ NVMe resident 参数：停止普通 queue 扫描，提交异步 NVMe read
模块 N post-hook：释放 GPU full parameter
模块 N+1 pre-hook：复用已完成的 CPU/NVMe 预取结果并 all-gather
```

具体实现上，`fetch_sub_module` 先处理当前模块，再处理预取。普通 CPU/GPU 参数预取会从 queue 中累计到 `prefetch_bucket_size`；遇到 NVMe 参数时会把它放回 queue，随后调用 `__prefetch_nvme_param_partitions`。该函数只扫描 trace 中靠后的 NVMe 参数，最多受两个限制：

- `numel_considered > 2 * numel_in_flight` 时停止，避免 NVMe read 领先当前 all-gather 太远；
- `available_swap_in_buffers()` 用尽时停止，避免无限制占用 swap buffer。

因此，“从 CPU prefetch”指 **CPU 中已经存在的 partition 直接进入异步 GPU all-gather**；“从 NVMe prefetch”指 **先把 partition 异步读入 NVMe swap buffer，使其变成 `AVAILABLE`，之后仍需 GPU all-gather 才能形成模块完整参数**。CPU 与 NVMe 不是二选一的全局模式，而是按每个 parameter partition 的 `final_location` 和当前 status 在同一条 trace 上混合调度。

需要注意：`max_in_cpu` 是 NVMe 参数 offload 的容量配置，但当前运行时是否走 NVMe 的直接判定点是 partition 创建时的 `swappable_tensor`、`final_location` 和 `nvme_swapper` 状态；分析具体版本时应以这些字段为准，而不要仅凭配置名推断某个参数一定会驻留 CPU。

## 3.2.3 两种模式的重叠边界

| 模式 | 可与当前计算重叠的工作 | 必须等待的时刻 |
| --- | --- | --- |
| ZeRO-3 + CPU offload | 后续参数的 CPU→GPU DMA、data-parallel all-gather；backward 中还可与前一模块的梯度归约并行 | 当前模块第一次读取参数前 |
| ZeRO-3 + NVMe parameter offload | 后续 NVMe→swap buffer read；已在 CPU 的参数仍做 CPU→GPU DMA/all-gather | 当前模块参数需先完成 NVMe read，再完成 all-gather |
| ZeRO-2 + optimizer CPU offload | 梯度 reduce/scatter、GPU→CPU 梯度 copy、CPU Adam step（实现/配置允许时） | optimizer step 使用对应 CPU gradient/state 前 |

NVMe read 本身不会直接把完整参数放到 GPU；它只是为下一次 all-gather 准备本 rank 的 partition。这个额外阶段解释了为什么 NVMe 模式需要更早的 trace-driven prefetch，以及为什么 swap buffer 数量和 I/O queue depth 会直接影响计算是否被 stall。

### 3.3 NVMe I/O 的 buffer、队列与完成语义

`AsyncPartitionedParameterSwapper` 在初始化时：

- 创建独立的 read/write AIO handle；
- 根据 `aio.block_size`、`queue_depth`、`single_submit`、`overlap_events`、`intra_op_parallelism` 配置 Linux libaio；
- 按 I/O alignment 对元素数向上取整；
- 使用固定数量的 swap-in buffers，防止异步读无限制地制造 host memory pressure。

`swap_in(async_op=True)` 只提交 read 并记录 inflight 参数；`synchronize_reads()` 等待 AIO 完成，将 buffer 绑定到 `ds_tensor` 并更新可用元素计数。`swap_out` 同理先提交 write，只有 `synchronize_writes()` 确认完成后才能回收 swap buffer。

optimizer NVMe 路径使用类似的 `PartitionedOptimizerSwapper`：参数/optimizer states 和 gradients 使用不同的 swap metadata；pinned tensors 可以直接异步写，非 pinned tensors 先复制到临时 pinned buffer 再写。这样形成：

```text
计算当前 tile       读取下一 tile (NVMe→CPU)       写回上一 tile (CPU→NVMe)
       |------------------- pipeline_read / pipeline_write -------------------|
```

### 3.4 关键配置与实际含义

| 配置 | 作用 |
| --- | --- |
| `zero_optimization.stage=3` | 开启参数分片；参数 NVMe offload 仅对 stage 3 有效 |
| `offload_param.device=cpu|nvme` | 参数分片的后备位置 |
| `offload_param.pin_memory` | CPU 参数 buffer 是否 page-lock，影响异步 H2D/D2H |
| `offload_param.buffer_count/buffer_size` | NVMe 参数 swap buffer 池容量 |
| `offload_param.max_in_cpu` | NVMe 模式下允许保留在 CPU 的参数元素数 |
| `offload_optimizer.device=cpu|nvme` | optimizer states 的后备位置；optimizer compute 仍在 CPU |
| `offload_optimizer.buffer_count` | optimizer state/gradient swap buffer 数量 |
| `pipeline_read/pipeline_write` | tile optimizer 的读/写与计算重叠 |
| `stage3_prefetch_bucket_size` | 当前模块之外可提前 fetch 的参数元素上限 |
| `stage3_max_live_parameters` | GPU/活动参数工作集上限 |
| `aio.*` | NVMe 异步 I/O 块大小、队列深度和并行度 |

## 4. 结论

- ZeRO-Infinity 的核心是 **ZeRO-3 分片 + 参数/梯度/optimizer state 的 CPU/NVMe 异构驻留 + trace 驱动的 fetch/release/prefetch**。
- 当前 DeepSpeed 代码中的 “Super” 实现是 **SuperOffload**：把 CPU Adam 放进独立进程并异步执行，不是另一套名为 Super-Infinity 的 NVMe 引擎。
- 参数 offload 通过模块 hook 和 `PartitionedParameterCoordinator` 管理；梯度和 optimizer states 通过 `PartitionedOptimizerSwapper` 管理；激活值通过 autograd saved-tensor hooks 独立管理。
- 三层预取不是一次性把所有数据搬到 CPU：NVMe 是参数/optimizer 分片的后备层，CPU pinned buffers 是 I/O staging 层，GPU 只保留计算窗口；libaio/GDS、固定 buffer pool、event/stream 和 trace queue 共同实现读写与计算重叠。

## 5. 主要源码索引

- `deepspeed/runtime/zero/config.py`
- `deepspeed/runtime/zero/offload_config.py`
- `deepspeed/runtime/zero/parameter_offload.py`
- `deepspeed/runtime/zero/partitioned_param_coordinator.py`
- `deepspeed/runtime/swap_tensor/partitioned_param_swapper.py`
- `deepspeed/runtime/swap_tensor/partitioned_optimizer_swapper.py`
- `deepspeed/runtime/superoffload/superoffload_stage3.py`
- `deepspeed/runtime/superoffload/superoffload_utils.py`
- `deepspeed/runtime/activation_checkpointing/offload_activations.py`
- `docs/_tutorials/zero.md`
- `docs/_pages/config-json.md`
