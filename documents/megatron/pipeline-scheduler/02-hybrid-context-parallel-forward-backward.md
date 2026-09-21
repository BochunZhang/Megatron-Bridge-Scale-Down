# Hybrid Context Parallel Forward/Backward 调度分析

本文基于当前仓库中 Megatron-LM 子模块的实现，分析
`hybrid_context_parallel_forward_backward` 及其配套的数据调度逻辑，说明调用它的
scheduler，以及相对普通 context-parallel scheduler 的优化、限制和代价。

相关源码：

- `3rdparty/Megatron-LM/megatron/core/pipeline_parallel/hybrid_cp_schedule.py`
- `3rdparty/Megatron-LM/megatron/core/pipeline_parallel/schedules.py`
- `3rdparty/Megatron-LM/megatron/core/datasets/data_schedule.py`
- `3rdparty/Megatron-LM/megatron/core/utils.py`
- `3rdparty/Megatron-LM/megatron/core/parallel_state.py`

## 1. 结论

`hybrid_context_parallel_forward_backward` 是一个**动态 CP、无 pipeline 的
forward/backward 执行 scheduler**。它针对 packed sequence 中每个长度不同的
sub-sample，动态选择参与该 sub-sample 的 CP rank 数量；同一批数据中的不同
sub-sample 可以使用不同大小的 CP group。

当前调用关系只有一条：

```text
get_forward_backward_func()
└── PP == 1
    └── forward_backward_no_pipelining()
        └── hybrid_context_parallel_forward_backward()
```

在 `forward_backward_no_pipelining` 内部，分支优先级是：

```text
overlap_moe_expert_parallel_comm and not forward_only
    -> combined_1f1b_schedule_for_no_pipelining
elif config.hybrid_context_parallel
    -> hybrid_context_parallel_forward_backward
else
    -> 普通 forward_step/backward_step 循环
```

因此，当 MoE expert-parallel overlap 和 HCP 同时打开且处于训练模式时，前者
优先，HCP 函数不会被调用。`get_forward_backward_func` 本身只依据 PP/VPP
选择顶层 scheduler；HCP 分支是在 `forward_backward_no_pipelining` 内部进行的。

配置校验还禁止 HCP 与 pipeline parallelism、CUDA Graph、Megatron FSDP 一起使用，
并要求 single dataloader 与 per-token loss。因此：

- `forward_backward_pipelining_with_interleaving` 不调用该函数；
- `forward_backward_pipelining_without_interleaving` 不调用该函数；
- 这不是 pipeline 1F1B 的替代实现，而是 PP=1 场景下的动态 CP scheduler。

## 2. HCP 的数据准备链路

HCP 的关键不只在 forward/backward 函数，还在数据进入 scheduler 之前的重排。
训练入口在 `args.hybrid_context_parallel` 时用
`HybridCPDataLoaderWrapper` 包装原始 iterator。每次取一个 global batch 后，流程如下：

```text
packed batch
  -> 提取每个 sub-sample 的长度
  -> DP group all-gather 全局长度
  -> BalancedCPScheduler 生成 groups 和 sample_id_groups
  -> DPxCP group all-to-all 重分发 sub-sample
  -> 每个 rank 得到自己在各 group 中负责的 sample ID
```

### 2.1 `BalancedCPScheduler`：按长度选择 CP 大小

`gpus_needed(seq_len)` 根据 `max_seqlen_per_dp_cp_rank` 估算一个 sub-sample 需要的
CP rank 数，并向上取 2 的幂：

```text
needed_cp = max(1, 2 ** ceil(log2(seq_len / max_seqlen_per_dp_cp_rank)))
```

可用的动态 group 因而是 2、4、8 ... 等幂次大小；当所需大小等于整个 DPxCP
group 时，复用完整的 DPxCP group。`get_total_workload` 使用近似成本
`seq_len² / cp_size`，用于比较不同长度和不同 CP 大小下的相对计算量，而不是精确
的 FLOPs 统计。

### 2.2 组内平衡和 deadlock 避免

`next_hdp_group` 依次完成以下工作：

1. 按 CP 大小把 sub-sample 分桶，并尽量让桶的估算工作量接近。
2. 优先放入已有的相同大小 group，否则从空闲 rank 中创建新 group。
3. 按 group 的最大执行时间选择负载较低的 group。
4. 负载超出 slack 时尝试移除最后加入的 sub-sample。
5. 如果存在空闲 rank，则把较小 CP group 扩大到下一个 2 的幂，并移动后续工作，
   让所有 DPxCP rank 都有工作。

一个 group 是一组可以在 CP 域内无额外 barrier 地连续执行的 sub-sample。不能把
任意长度的 sub-sample 无限串在一个 group 中：如果部分 rank 已经切换到新的 CP
大小，其他 rank 仍在旧 group 中，CP collective 的参与者不一致就可能造成死锁。

### 2.3 重分发和动态 CP 元数据

`HybridCPDataLoaderWrapper` 先把 packed sample 按 `cu_seqlens` 拆成独立
sub-sample，再通过 DPxCP group 的 `all_to_all_single` 把 sub-sample 发到目标 rank。
这避免了为了一个短 sub-sample 搬运整个 packed sample。

每个 sub-sample 在执行前会设置 `local_cp_size`。该值是当前 sub-sample 所在
`sample_id_groups[group_id]` 中的参与 rank 数，并以 `int32` 在 TP group 内广播。
随后 `get_batch_on_this_cp_rank`：

1. 用 `local_cp_size` 找到对应的 hybrid DPxCP process group；
2. 用 per-sequence zigzag 方式切分该 sub-sample；
3. 把实际 group 写入 `batch["hybrid_cp_group"]`。

`forward_step` 再把 `local_cp_size` 和 `hybrid_cp_group` 放入
`PackedSeqParams`，让 Transformer Engine 的 varlen attention 使用正确的局部
CP 配置。

## 3. `hybrid_context_parallel_forward_backward` 的执行逻辑

源码入口：
`3rdparty/Megatron-LM/megatron/core/pipeline_parallel/hybrid_cp_schedule.py:477`。

### 3.1 读取并广播全局调度结果

- 只有 TP rank 0 从 `data_iterator` 读取 `(batch, sample_id_groups)`；
- 每个 rank 根据自己的 `hdp_rank`（DPxCP rank）计算各 group 要执行的 sub-sample 数；
- 通过一次全局 barrier 加 TP broadcast 广播 group 数量及每个 group 的 sub-sample 数量；
- 非 TP0 rank 不持有原始 batch，只接收后续所需的 `local_cp_size` 和普通 TP batch
  广播数据。

`num_microbatches` 仍作为参数传给 `forward_step`，但 HCP 实际遍历的是数据侧生成的
group/sub-sample 计划；`current_microbatch` 则按当前 rank 实际执行的 sub-sample
递增。因此它不是普通 scheduler 那种固定数量、固定形状的 microbatch 循环。

### 3.2 前向、反向和同步边界

假设有多个 group，每个 group 在某个 rank 上包含若干 sub-sample，整体流程是：

```text
group 0: F(sample) -> B(sample) -> ...
          group barrier
group 1: F(sample) -> B(sample) -> ...
          group barrier
...
last group: 前 n-1 个 sub-sample 在 no_sync 中执行
            最后一个 sub-sample 在 no_sync 外执行
```

具体实现分三段：

1. **除最后 group 外**：所有 sub-sample 在 `no_sync_func()` 中执行
   `forward_step`，训练模式下紧接 `backward_step`；group 末尾在完整的 DPxCP
   group 上调用 barrier。
2. **最后 group 的前 n-1 个 sub-sample**：仍在 `no_sync_func()` 中执行，并执行
   同样的 forward/backward。
3. **最后 group 的最后一个 sub-sample**：移出 `no_sync_func()` 执行，使最后一次
   backward 按普通训练语义触发梯度同步。

当 `forward_only=True` 时仍会读取和执行同样的分组计划，但跳过 backward；最后
一个 sub-sample 仍作为调度尾部执行。`total_num_tokens` 累加每个 sub-sample 的
`num_tokens`，供外层的 per-token loss 和最终梯度处理使用。

### 3.3 为什么 group 末尾必须 barrier

动态 CP 意味着相邻 sub-sample 可能属于不同大小的 CP group。某个 rank 如果提前
开始下一个 sub-sample，会进入一个只有部分 partner rank 参与的 CP collective。
因此 barrier 的作用不是同步梯度，而是确保整个 DPxCP group 已经完成当前 group，
再一起切换到下一个 `local_cp_size` 和 process group。

## 4. 与普通 scheduler 的对比

这里的“普通 scheduler”指 `forward_backward_no_pipelining` 中不启用 HCP、也不启用
MoE combined 1F1B 的路径。普通路径对每个固定形状 microbatch 使用同一个
`pg_collection.cp`，并直接调用 `forward_step` / `backward_step`。

| 维度 | 普通 scheduler | HCP scheduler | 带来的效果 |
|---|---|---|---|
| CP group | 整个 batch 使用固定 CP group | 每个 sub-sample 按长度选择局部 CP group | 长样本可扩展到更多 rank，短样本不必占用全部 CP rank |
| 输入形状 | 通常是固定序列长度或固定 packed 形状 | 从 packed batch 拆成变长 sub-sample | 减少短样本被长样本 padding 的浪费 |
| DPxCP 负载 | rank 间通常按固定样本划分 | 按 `seq_len² / cp_size` 估算并平衡 | 降低长短样本混合时的 rank 空转和 straggler |
| 数据传输 | 不需要为动态 CP 重排样本 | 使用 all-gather、all-to-all 和 TP broadcast | 把样本送到真正参与其 CP group 的 rank |
| CP 切换 | 无动态切换 | group 之间显式 barrier | 保证 collective 参与者一致，避免 deadlock |
| 梯度同步 | 最后 microbatch 外同步 | 最后 group 的最后 sub-sample 外同步 | 保留 no-sync 累积梯度语义 |
| 主要优化目标 | 简化、稳定的固定形状执行 | 变长 packed sequence 的内存和计算利用率 | 提高有效 token 吞吐和可容纳的序列长度 |

### 4.1 主要收益

1. **按需分配 CP 资源**：长度超过单 rank 能力的样本才扩大 CP group，短样本可
   使用更小 group，避免固定大 CP group 带来的通信和同步开销。
2. **降低 padding 和显存浪费**：sub-sample 在拆分后按真实长度参与调度，避免
   为了同一个 packed batch 中的最长样本给所有 rank 分配同等形状。
3. **改善 DPxCP 负载均衡**：使用近似 attention 工作量进行分桶、分组、裁剪和
   空 rank 填充，减少某些 rank 处理长序列而其他 rank 等待的情况。
4. **保持现有 forward/backward 接口**：外层仍通过 `forward_step`、`backward_step`
   和 `no_sync_func` 工作；动态部分集中在数据重排与 CP group 选择。

### 4.2 不能混淆的优化边界

HCP 主要优化的是**变长 packed sequence 下的资源分配和负载均衡**，不是 MoE
dispatch/combine 的通信隐藏。后者属于 `combined_1f1b` 路径，两者在
`forward_backward_no_pipelining` 中是互斥的分支优先级。

HCP 也不会减少 CP collective 的基本通信量；它通过缩小短样本的参与 group、减少
无效 padding 和等待来改善有效利用率。同时，动态调度本身增加了：

- 长度 all-gather 和样本 all-to-all；
- TP 元数据广播；
- 每个 group 的 DPxCP barrier；
- 拆包、重建 iterator 和 process-group 切换管理。

当样本长度比较均匀、固定 CP 已经能充分利用 GPU，或者 group 数过多导致 barrier
频繁时，HCP 的额外调度成本可能抵消收益。

## 5. 使用限制和前置条件

根据当前参数校验与实现，启用 HCP 至少需要注意：

- `max_seqlen_per_dp_cp_rank` 必须设置，用于计算动态 CP 大小；
- DPxCP 总 rank 数必须为偶数，并且动态 group 大小按 2 的幂构造；
- 需要 packed sequence 的 `cu_seqlens` 元数据；
- 只支持 `dataloader_type=single`；
- 必须启用 `calculate_per_token_loss`；
- 不支持 pipeline parallelism、CUDA Graph 和 Megatron FSDP；
- 当前实现通过 group barrier 保证安全切换，因此不能把它理解为完全无同步的
  异步 scheduler；
- Transformer/attention 实现必须正确消费 `local_cp_size` 和
  `hybrid_cp_group`，否则动态 group 只会改变数据分发而不会改变实际 CP 计算。

## 6. 代码定位速查

| 逻辑 | 文件与位置 |
|---|---|
| HCP 主 scheduler | `pipeline_parallel/hybrid_cp_schedule.py:477` |
| CP 大小估算和负载平衡 | `pipeline_parallel/hybrid_cp_schedule.py:14`、`:104`、`:456` |
| HCP 数据包装和 all-to-all 重排 | `datasets/data_schedule.py:12`、`:267` |
| 顶层 HCP 分支 | `pipeline_parallel/schedules.py:708`、`:780` |
| 动态 CP group 创建 | `parallel_state.py:440`、`:1040` |
| `local_cp_size` 到 CP group 的解析 | `utils.py:2612` |
| HCP 参数限制 | `training/arguments.py:1404` |
