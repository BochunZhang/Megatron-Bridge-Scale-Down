# GatedDeltaNet Context Parallel：基于 FLA 代码的执行逻辑

整理日期：2026-09-15

本文基于工作区中与 `megatron-bridge` 同级的 `flash-linear-attention/` 参考实现。

重点解释 `fla/ops/cp/` 和 `fla/ops/gated_delta_rule/` 的实际调用顺序。这里的 CP（Context Parallel）不是把完整 KV 序列在 rank 之间环形传递，而是：

1. 每个 rank 独立处理自己的连续 token 区间；
2. 将本地 chunk 压缩成一个状态贡献和一个状态转移矩阵；
3. 使用一次同步 `AllGather` 收集这些固定大小的摘要；
4. 每个 rank 在本地重建自己所需的前序状态；
5. 再执行本 rank 的 GatedDeltaNet 主 kernel。

## 1. 代码入口和职责

| 文件 | 作用 |
| --- | --- |
| `fla/ops/gated_delta_rule/chunk.py` | GDN 的公开 chunkwise forward/backward API，以及 CP 分支接入点 |
| `fla/ops/cp/context.py` | 构造 `FLACPContext`，计算本 rank 的序列范围、前后续 rank 数量和卷积边界信息 |
| `fla/ops/cp/chunk_delta_h.py` | 生成本地状态摘要、执行 AllGather、运行 forward/backward merge kernel |
| `fla/ops/cp/comm.py` | `all_gather_into_tensor` 以及卷积/token shift 的邻 rank 数据交换封装 |
| `fla/modules/conv/cp/ops.py` | causal Conv1d 的 CP 前向初始状态和反向梯度修正 |
| `fla/modules/token_shift_cp.py` | token shift 的 CP 前向 cache 和反向梯度修正 |

GDN 的公共入口是 `fla/ops/gated_delta_rule/chunk.py:397` 的 `chunk_gated_delta_rule`。它通过 autograd Function 调用 `chunk_gated_delta_rule_fwd` 和 `chunk_gated_delta_rule_bwd`。

## 2. CP context 的构造

调用方先使用：

```python
cp_context = build_cp_context(
    cu_seqlens_global,
    group=cp_group,
    conv1d_kernel_size=window,
)
```

`cu_seqlens_global` 是切分前的全局累计长度，例如：

```text
[0, length_0, length_0 + length_1, ..., total_tokens]
```

`get_cp_cu_seqlens` 在 `fla/ops/cp/context.py` 中完成以下工作：

### 2.1 计算 rank 的连续 token 区间

```python
part_len = total_tokens // world_size
rank_start = part_len * rank
rank_end = rank_start + part_len
```

随后使用 `searchsorted` 找出与 `[rank_start, rank_end)` 相交的 document，并把全局 `cu_seqlens` 转换为本 rank 的局部 `cu_seqlens`。

这里的参考代码使用整数除法计算 `part_len`，因此上层应明确处理 `total_tokens` 不能被 `world_size` 整除时的尾部 token；不能默认该分片公式会自动覆盖余数。

因此，递归模型要求 CP rank 持有全局 token 流的连续区间。它不能直接使用会交换头尾 chunk 的负载均衡布局，否则前后状态顺序会被破坏。

### 2.2 计算状态链元数据

`FLACPContext` 保存：

- `is_first_rank`：本 rank 是否是当前 document 的第一个处理 rank；
- `is_last_rank`：本 rank 是否是当前 document 的最后一个处理 rank；
- `pre_num_ranks`：当前 rank 前面有多少个属于同一状态链的 rank；
- `post_num_ranks`：当前 rank 后面有多少个属于同一状态链的 rank；
- `pre_num_conv_tokens`：当前 rank 的第一个 document 有多少 token 来自前一个 rank；
- `cu_seqlens` / `cu_seqlens_cpu`：本 rank 的局部变长序列元数据。

这些字段不是简单的 `rank` 和 `world_size` 替代品。一个 rank 可能位于多个 document 的交界处，因此前后状态链需要根据 document 边界判断。

## 3. GDN forward 总调用链

`chunk_gated_delta_rule_fwd` 的执行顺序如下：

```text
raw g
  -> gate cumulative sum
  -> chunk_gated_delta_rule_fwd_intra
       生成 WY/UT 中间量 w, u, A
  -> chunk_gated_delta_rule_fwd_h_pre_process  [CP]
       本地状态摘要 hm
       AllGather(hm)
       本地 prefix merge
       得到 initial_state
  -> chunk_gated_delta_rule_fwd_h
       使用 initial_state 计算 h / v_new
  -> chunk_fwd_o
       使用 q、k、v_new、h 计算输出 o
```

对应代码位于 `fla/ops/gated_delta_rule/chunk.py:51-123`。

### 3.1 Gate 累积

默认情况下，`g` 已经是 log-space gate。代码调用：

```python
g = chunk_local_cumsum(
    g,
    chunk_size=chunk_size,
    scale=RCP_LN2,
    cu_seqlens=cu_seqlens,
    chunk_indices=chunk_indices,
)
```

如果启用 `use_gate_in_kernel=True`，则由 `gdn_gate_chunk_cumsum` 将原始 gate 激活和 chunk 累积融合完成。

### 3.2 chunk 内 WY/UT 预处理

`chunk_gated_delta_rule_fwd_intra` 在 `fla/ops/gated_delta_rule/chunk_fwd.py` 中执行：

1. 计算 chunk 内的 `beta * K K^T` 下三角块；
2. 应用 gate 和 beta 缩放；
3. 对 `(I + A)` 做三角求解；
4. 通过 `recompute_w_u_fwd` 生成 `w` 和 `u`。

`chunk_size=64` 时使用融合的 KKT + solve-tril kernel；`16` 或 `32` 使用数学等价但未完全融合的路径。

这一步只处理本 rank 的 token，不涉及跨 rank 通信。

## 4. Forward CP：本地摘要、AllGather 和 merge

### 4.1 分配摘要和初始状态

`chunk_gated_delta_rule_fwd_h_pre_process` 位于 `fla/ops/cp/chunk_delta_h.py:747`。启用 CP 后：

```python
hm = k.new_zeros(HV, K, V + K, dtype=torch.float32)
initial_state = k.new_zeros(N, HV, K, V, dtype=torch.float32)
```

`hm` 的最后一维分成两部分：

```text
hm[..., :V]   = 本地 chunk 在零输入状态下产生的 state contribution
hm[..., V:]   = 本地 chunk 的 state transition matrix
```

这里的摘要状态使用 FP32。主模型输入可以是 BF16，但状态链反复矩阵乘法时不能频繁降回 BF16，否则误差会累积。

### 4.2 计算本地摘要

只有不是状态链最后一个 rank 时，代码才计算本地 `hm`：

```python
if not context.is_last_rank:
    pre_process_fwd_kernel_merged[grid](
        k=k,
        v=u if v is None else v,
        w=w,
        g=g,
        gk=gk,
        bg=bg,
        u=u,
        hm=hm,
        cu_seqlens=cu_seqlens[-2:],
        ...,
    )
```

最后一个 rank 不再需要把自己的状态贡献发送给后继 rank，因此跳过本地摘要计算；它仍然会参加后面的 collective。

`pre_process_fwd_kernel_merged` 使用一个统一 kernel 同时计算：

- `h` 部分：局部累积 state contribution；
- `m` 部分：局部 transition matrix。

forward 只把本 rank 最后一个 local sequence 的两个边界传给预处理 kernel（`cu_seqlens[-2:]`），因为跨 rank 的 forward 状态只可能从本 rank 的尾部继续到后一个 rank。其他 document 在本 rank 内独立开始和结束。

### 4.3 AllGather 是同步的

随后执行：

```python
ag_hm, _ = all_gather_into_tensor(hm, group=context.group)
```

`fla/ops/cp/comm.py` 中默认 `async_op=False`，底层是同步的 `torch.distributed.all_gather_into_tensor`。因此当前代码的语义是：

```text
所有 rank 完成本地摘要
        ↓
所有 rank 参与一次 AllGather
        ↓
AllGather 返回后继续 merge
```

这不是 `cp0` 计算完立即把完整 state 发送给 `cp1` 的点对点流水线。每个 rank 收到的是所有 rank 的 `hm` 摘要。

### 4.4 Forward merge kernel

非第一个 rank 执行：

```python
merge_fwd_bwd_kernel[grid](
    h=initial_state[0],
    ag_hm=ag_hm,
    pre_or_post_num_ranks=context.pre_num_ranks,
    rank=rank,
    FORWARD=True,
    INTRACARD_MODE=False,
    ...,
)
```

Triton kernel 中的 CP 分支逻辑是：

```python
b_h = 0
for idx in range(num_ranks):
    cur_rank = rank - num_ranks + idx
    b_he = ag_hm[cur_rank, ..., :V]
    b_m  = ag_hm[cur_rank, ..., V:]
    b_h = b_m @ b_h + b_he
store(initial_state)
```

因此：

- 第一个 rank 不需要 merge，初始状态为零或外部给定状态；
- 第 `r` 个 rank 在本地按顺序合并前面的 `pre_num_ranks` 个摘要；
- 最后一个 rank 的 merge 循环通常最长；
- merge 处理的是固定大小的状态矩阵，不是前序 token 或完整 KV。

对于单条 document 覆盖全部 `P` 个 rank 的情况，merge 次数是：

```text
cp0: 0 次
cp1: 1 次
...
cp(P-1): P-1 次
```

当前实现没有在这里使用树形 prefix scan；Triton kernel 直接按 `cur_rank` 顺序循环。因此 rank 越靠后，状态 merge 的本地工作量越大。

### 4.5 使用初始状态执行主 forward

merge 完成后调用公共状态 kernel：

```python
h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
    k=k,
    w=w,
    u=u,
    g=g,
    initial_state=initial_state,
    output_final_state=output_final_state,
    ...,
)
```

这里的 `initial_state` 是本 rank 局部 token 区间开始时的 recurrent state。它包含所有前序 rank 的影响，但主 kernel 只读取本 rank 的 `k/w/u/g`。

CP 公共 API 明确禁止：

```python
initial_state is not None
output_final_state is True
```

因为 CP 模式下初始状态由跨 rank merge 生成，最终状态也不会作为普通单卡 `final_state` 返回给调用方。

最后，`compress_h0` 只保留需要保存用于反向的最小初始状态。对于 CP 中只有第一个 sequence 可能是跨 rank continuation 的事实，不需要保存完整的每个 sequence 初始状态。

## 5. GDN backward CP

反向入口位于 `fla/ops/gated_delta_rule/chunk.py:126-230`，总体顺序为：

```text
重算 w, u
  -> expand_h0
  -> 重算 h / v_new
  -> 计算局部 dv
  -> chunk_gated_delta_rule_bwd_dhu_pre_process [CP]
       生成本地 backward 摘要 dhm
       AllGather(dhm)
       反方向 merge
       得到 dht
  -> chunk_gated_delta_rule_bwd_dhu
  -> chunk_bwd_dqkwg
```

### 5.1 重算前向中间量

为了节省显存，反向先使用保存的 `A` 重算：

```python
w, u = recompute_w_u_fwd(...)
```

如果启用 CP，则先执行：

```python
initial_state = expand_h0(initial_state, context=cp_context)
```

然后用同一个 `chunk_gated_delta_rule_fwd_h` 重建局部 `h` 和 `v_new`。

### 5.2 生成 backward 摘要

`chunk_gated_delta_rule_bwd_dhu_pre_process` 位于 `fla/ops/cp/chunk_delta_h.py:831`：

```python
dhm = q.new_zeros(HV, K, V + K, dtype=torch.float32)
dht = q.new_zeros(N, HV, K, V, dtype=torch.float32)
```

不是第一个 rank 时，先运行 `pre_process_bwd_kernel_merged` 计算本地 backward 摘要。第一个 rank 没有前置状态梯度来源，因此跳过这一步，但仍参与 AllGather。

backward 预处理使用 `cu_seqlens[:2]`，只处理本 rank 第一个 local sequence；这是因为来自后续 rank 的状态梯度只会沿当前状态链回到本 rank 的开头。

### 5.3 反向 AllGather 和逆向 merge

```python
ag_dhm, _ = all_gather_into_tensor(dhm, group=context.group)
```

不是最后一个 rank 时执行 merge：

```python
merge_fwd_bwd_kernel[grid](
    h=dht[-1],
    ag_hm=ag_dhm,
    pre_or_post_num_ranks=context.post_num_ranks,
    rank=rank,
    FORWARD=False,
    ...,
)
```

kernel 采用反向 rank 顺序：

```python
for idx in range(num_ranks):
    cur_rank = rank + num_ranks - idx
    b_h = b_m @ b_h + b_he
```

所以 forward 是向后传播状态，backward 是向前传播状态梯度：

```text
forward:   cp0 -> cp1 -> cp2 -> cp3
backward:  cp3 -> cp2 -> cp1 -> cp0
```

但在当前实现中，这两个方向都通过 AllGather 后的本地 merge 实现，而不是通过 rank 间逐跳 send/recv。

## 6. 通信与阻塞点

### 6.1 GDN recurrent state 的 CP

GDN forward 的主要同步点是：

```text
本地 pre_process kernel
        ↓
同步 AllGather(hm)
        ↓
本地 merge kernel
        ↓
本地 GDN forward kernel
```

当前 FLA 代码的 `all_gather_into_tensor` 默认 `async_op=False`，因此从调用者视角 AllGather 是阻塞的。代码中没有把 `hm` 的通信和后续 merge 通过异步 handle 显式重叠起来。

但是，AllGather 之前各 rank 的本地摘要是并行计算的；阻塞发生在 collective 需要所有参与 rank 都到达之后，而不是 `cp1` 等待 `cp0` 逐 token 或逐 chunk 传完整状态。

### 6.2 merge 的工作不均衡

对单条跨越全部 rank 的 sequence，朴素实现的 merge 工作量随 rank 增长：

```text
rank 0: 0 × (K×K) @ (K×V)
rank 1: 1 × ...
rank 2: 2 × ...
...
rank P-1: (P-1) × ...
```

因此最后 rank 的 prefix merge 最重。它并没有处理更多本地 token，但需要组合更多前序状态变换。

更具体地说，单次 merge 执行：

```python
state = transition_j @ state + contribution_j
```

其中 `transition_j` 大致为 `[K, K]`，`state` 和 `contribution_j` 大致为 `[K, V]`。单次矩阵乘法的主要成本约为 `O(K^2 * V)`；对 `HV` 个 value heads 和 `pre_num_ranks` 个前序 rank，当前 rank 的 prefix merge 成本可近似写为：

```text
O(pre_num_ranks * HV * K^2 * V)
```

因此，在一条 sequence 覆盖全部 `P` 个 rank 的等长场景下：

```text
cp0: pre_num_ranks = 0，initial_state 直接为零
cp1: pre_num_ranks = 1，合并 hm0
cp2: pre_num_ranks = 2，依次合并 hm0、hm1
...
cp(P-1): pre_num_ranks = P-1，合并所有前序摘要
```

这说明“越靠后的 rank，`initial_state` merge 成本越高”是当前实现的直接结果。但需要区分：

- 本地 GDN 主 kernel 仍只处理当前 rank 的 token，等长切分时其工作量大致相同；
- 最后 rank 跳过了为后继 rank 生成 `hm` 的预处理，因为没有后继 rank 使用它；
- 变高的是跨 rank prefix merge 的成本，而不是完整的 GDN forward 成本。

AllGather 返回后，各 rank 才开始自己的 merge，因此后面的 rank 可能更晚完成该阶段，形成潜在的负载不均衡。当前 Triton 实现直接在 merge kernel 内按 rank 顺序循环，没有使用树形 prefix scan。

`pre_num_ranks` 按 document 状态链计算，而不是简单等于全局 rank。对于 packed/variable-length 输入，如果某个 document 没有跨越前面的 rank，该 rank 的 `pre_num_ranks` 可能为 `0` 或较小；因此“rank 编号越大，merge 必然越重”只适用于单条 sequence 连续跨越多个 rank 的场景。

如果要降低该瓶颈，可以把仿射变换 `(M, E)` 作为可结合对象做 hierarchical prefix scan；这属于替换 merge 实现的优化，不是当前 `merge_fwd_bwd_kernel` 的执行方式。

### 6.3 通信量

每个 rank 的 forward 摘要形状大致为：

```text
[HV, K, V + K]，FP32
```

AllGather 后的张量是：

```text
[world_size, HV, K, V + K]
```

通信内容与完整序列长度无关，但与 head 数、key/value state 大小和 CP degree 有关。

## 7. causal Conv1d 和 token shift 的 CP

GDN 前端通常还包含 causal convolution 或 token shift。这两类依赖不是矩阵状态递归，而是很短的邻 token 边界依赖，需要单独处理。

### 7.1 causal Conv1d

`fla/modules/conv/cp/ops.py` 的 forward：

1. 非首 rank 取本地最后 `W-1` 个 token；
2. 所有 rank 通过 `conv_cp_send_recv_fwd` 参加一次 AllGather；
3. 非首 rank 选取前一 rank 的 tail 作为 head cache；
4. 将这些 token 填入 causal Conv1d 的 `initial_state`；
5. 调用原始 `causal_conv1d_fwd`。

第一个 rank 也必须参加 collective，但收到的数据不使用。

反向时，非末 rank 的初始状态梯度通过 `conv_cp_send_recv_bwd` 传回前一 rank，并加到当前 rank 最后 `W-1` 个 token 的梯度上。

### 7.2 token shift

`fla/modules/token_shift_cp.py` 只有一个 token 的边界依赖：

```text
y[t] = x[t-1] - x[t]
```

非首 rank 需要前一 rank 的最后一个 token 作为 cache；反向时把 cache 梯度传回前一 rank 的最后一个 token。

需要注意，`fla/ops/cp/comm.py` 中名为 `send_recv_fwd` / `send_recv_bwd` 的 helper 为了让所有 rank 参与 collective，实际仍使用 `all_gather_into_tensor`，然后从 gathered tensor 中选取邻 rank 数据，并非底层 point-to-point `dist.send` / `dist.recv`。

## 8. packed/variable-length sequence 语义

当传入 `cu_seqlens` 时：

- GDN API 要求 batch size 为 `1`，token 先 flatten，再用 `cu_seqlens` 描述 document 边界；
- `build_cp_context` 从全局 `cu_seqlens` 生成 rank-local `cu_seqlens`；
- 只有跨 rank 的第一个 local sequence 可能需要前一 rank 的状态；
- 其他 document 在自己的起点使用零状态；
- `pre_num_ranks` 和 `post_num_ranks` 按状态链计算，而不是简单按全局 rank 计算。

对于 Conv1d，`pre_num_conv_tokens` 还会限制前一 rank tail 中真正属于当前第一个 document 的 token 数，避免把相邻 document 的 token 混入卷积初始状态。

## 9. 一次完整 forward 的伪代码

下面的伪代码对应当前实现的高层行为：

```python
# 每个 rank 持有自己的连续 token slice
cp = build_cp_context(global_cu_seqlens, group=cp_group, ...)

# GDN chunkwise local work
g = chunk_local_cumsum(g, ...)
w, u, A = chunk_gated_delta_rule_fwd_intra(k, v, g, beta, ...)

# State CP pre-process
hm = zeros([HV, K, V + K], dtype=float32)
if not cp.is_last_rank:
    hm = local_state_summary(k, w, u, g, ...)

# Synchronous collective; every rank receives every summary
all_hm = all_gather_into_tensor(hm, group=cp.group)

# Reconstruct this rank's initial recurrent state
initial_state = zeros(..., dtype=float32)
if not cp.is_first_rank:
    for rank_j in previous_ranks(cp):
        E_j, M_j = split(all_hm[rank_j])
        initial_state = M_j @ initial_state + E_j

# Local GDN state/output computation
h, v_new, _ = chunk_gated_delta_rule_fwd_h(
    k, w, u, g,
    initial_state=initial_state,
    output_final_state=False,
    ...,
)
o = chunk_fwd_o(q, k, v_new, h, g, ...)
```

## 10. 与“cp0 传给 cp1，再传给 cp2”的区别

两种描述分别对应不同的实现思路：

```text
逐跳流水线（概念模型）:
    cp0 --state--> cp1 --state--> cp2 --state--> cp3

当前 FLA CP 实现:
    cp0 --local summary--┐
    cp1 --local summary--├── AllGather ──> 每个 rank 本地 prefix merge
    cp2 --local summary--┤
    cp3 --local summary--┘
```

当前实现仍然保留递归的因果顺序：后续 rank 的初始 state 必须包含前序 rank 的影响。但这种依赖通过 `(M, E)` 摘要组合表达，避免传递完整 token 历史。

## 11. 实现限制和验证重点

实现或移植到其他训练框架时，需要重点检查：

1. CP rank 是否持有连续的全局 token 区间；
2. 每个 document 的 `cu_seqlens` 是否在切分前构造；
3. forward 的 `hm` 和 backward 的 `dhm` 是否使用 FP32 累积；
4. 首 rank/末 rank 是否正确跳过不需要的摘要或 merge；
5. Conv1d/token shift 的边界 cache 是否与 GDN state CP 同时处理；
6. forward merge 使用 `pre_num_ranks`，backward merge 使用 `post_num_ranks`；
7. CP 模式下不要传入外部 `initial_state` 或请求 `output_final_state`；
8. 用 CP=1 和单卡完整序列作为数值基准，分别验证等长、packed 和跨 document 边界场景。
