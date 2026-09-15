# Megatron-Core 中 GatedDeltaNet 的实现

整理日期：2026-09-15

本文说明当前 `3rdparty/Megatron-LM` 中 `GatedDeltaNet` 的实现，重点解释：

- GatedDeltaNet 模块的计算顺序；
- TP、CP 和 HP layout 的关系；
- CP→HP 与 HP→CP All-to-All 的通信范围和 shape 变化；
- forward/backward 中的同步和顺序依赖；
- Megatron-Core 实现与独立 FLA CP 实现的区别。

本文中的 HP 是 **hidden parallel layout**，不是一个额外的并行维度，也不是一个独立的 process group。

## 1. 先区分两条实现路径

同一个 GatedDeltaNet 名称下，需要区分两层代码：

| 路径 | 主要代码 | CP 方式 |
| --- | --- | --- |
| Megatron-Core GDN module | `megatron/core/ssm/gated_delta_net/` | GDN module 外层执行 CP↔HP All-to-All；随后调用 GDN kernel |
| 独立 FLA GDN operator | `fla/ops/gated_delta_rule/` 和 `fla/ops/cp/` | 每个 rank 处理连续 sequence chunk，交换 state summary，并通过 AllGather 生成前序 state |

同目录的 `02-context-parallel.md` 主要描述独立 FLA operator 的 state-summary CP。本文描述 Megatron-Core 中的 `GatedDeltaNet` module，不应把两种 CP 机制直接等同。

Megatron-Core GDN 的构造函数明确说明：`cp_comm_type` 只为 TransformerLayer 兼容而接收，GDN 会忽略它，并使用自身的 All-to-All：

```text
CP layout -> HP layout -> 本地 GDN 计算 -> CP layout
```

参考：`3rdparty/Megatron-LM/megatron/core/ssm/gated_delta_net/common.py` 中 `_GDNBase.__init__` 的参数说明。

## 2. 核心结论

### 2.1 HP 不是独立并行轴

Megatron-Core 中没有 `hp_size`、`hp_rank` 或 `hp_group`。HP 表示张量在一次布局转换后的状态：

```text
CP layout: sequence 被切分，hidden/head 维较宽
HP layout: sequence 完整，hidden/head 维被进一步切分
```

HP 通信使用的是：

```python
cp_group=self.pg_collection.cp
```

因此，HP 是通过 CP group 完成的 hidden/sequence 重排，而不是在 TP group 上建立新的通信域。

### 2.2 GDN 支持 `TP=1, CP>1`，但不支持跳过 HP 重排

下面两件事不同：

```text
TP=1, CP>1
```

和：

```text
CP>1，但不执行 CP->HP / HP->CP
```

前者在当前代码结构上是可行的，前提是 Q/K/V/head 相关维度满足 CP 整除条件。后者不是当前 Megatron-Core GDN 的实现路径：只要 `cp_size > 1`，GDN forward 会执行 CP→HP，输出阶段会执行 HP→CP。

## 3. GatedDeltaNet 模块结构

`GatedDeltaNet` 由以下部分组成：

```text
in_proj       ColumnParallelLinear
conv1d        TP-local depthwise causal convolution
A_log/dt_bias TP-sharded parameters，并在 forward 中按 CP slice 取用
gated delta rule kernel
out_norm      本地 gated RMSNorm
out_proj      RowParallelLinear
```

### 3.1 输入投影

GDN 的 `in_proj` 使用 Column Parallel Linear：

```python
self.in_proj = build_module(
    submodules.in_proj,
    self.hidden_size,
    self.in_proj_dim,
    gather_output=False,
    tp_group=self.pg_collection.tp,
)
```

其输出 section 为：

```text
query
key
value
z / gate
beta
alpha
```

每个 TP rank 只保存 `in_proj_dim / TP` 的输出 feature。由于 `gather_output=False`，输入投影之后不会立即在 TP group 上 All-Gather 完整 Q/K/V。

参考：

- `3rdparty/Megatron-LM/megatron/core/ssm/gated_delta_net/common.py`
- `3rdparty/Megatron-LM/megatron/core/ssm/gated_delta_net/gdn.py`

### 3.2 输出投影

GDN 的 `out_proj` 使用 Row Parallel Linear，并声明输入已经是 parallel layout：

```python
self.out_proj = build_module(
    submodules.out_proj,
    self.v_dim,
    self.hidden_size,
    input_is_parallel=True,
    skip_bias_add=True,
    tp_group=self.pg_collection.tp,
)
```

每个 TP rank 计算自己的输入 feature 分片，随后通过 TP group 的 All-Reduce 合并输出。

## 4. TP、CP 和 token/head 的职责

设：

```text
S: 全局 sequence length
B: batch size
H: 输入 hidden size
T: tensor parallel size
C: context parallel size
M: GDN in_proj 总输出维度
```

下面先假设 `sequence_parallel=False`，这样可以单独观察 TP 和 CP 的职责。

### 4.1 TP 不会自动切 token

TP 主要切权重矩阵和输出 feature 维。

对于 Column Parallel `in_proj`：

```text
输入：每个 TP rank 都看到相同的 token 输入
权重：沿 output feature 维切分
输出：每个 TP rank 得到 [sequence, B, M/T]
```

因此，只有开启 TP 并不意味着 token 沿 sequence 维被切分。

如果额外开启：

```python
sequence_parallel=True
```

才会使用 TP group 在 sequence 维做 Sequence Parallel。Sequence Parallel 与 CP 是两个独立概念：

```text
TP：切 feature/权重；开启 SP 后也可以切 sequence
CP：切 sequence，并在 CP group 内完成上下文相关的重排/通信
HP：CP→HP All-to-All 后得到的 hidden/head layout
```

### 4.2 CP 负责初始 sequence shard

没有 Sequence Parallel 时，CP rank 的输入大致为：

```text
cp rank i: [S/C, B, H]
```

经过 TP-local `in_proj` 后，在固定 TP 坐标下变为：

```text
cp rank i: [S/C, B, M/T]
```

这里的 `cp rank i` 表示某一个 TP/DP/PP 坐标对应的 CP group 中的第 `i` 个 rank。

## 5. CP group 的通信范围

Megatron-Core 使用 `RankGenerator.get_ranks('cp')` 构造 CP group。CP group 只沿 `cp` 轴变化，其他并行坐标保持固定：

```text
固定：TP rank
固定：DP rank
固定：PP rank
固定：其他适用的模型并行坐标
变化：CP rank
```

如果：

```text
TP=2, CP=4
```

某个 PP stage 中可以抽象为：

```text
CP group 0: (tp0, cp0), (tp0, cp1), (tp0, cp2), (tp0, cp3)
CP group 1: (tp1, cp0), (tp1, cp1), (tp1, cp2), (tp1, cp3)
```

CP→HP All-to-All 是两组独立的 4-rank 通信：

```text
(tp0, cp0..cp3) 之间通信
(tp1, cp0..cp3) 之间通信
```

它不会直接在整个 `TP×CP=8` 个 rank 上做一次 All-to-All，也不会跨 TP group 交换数据。

对应代码：

```python
qkvzba = tensor_a2a_cp2hp(
    qkvzba,
    seq_dim=0,
    head_dim=-1,
    cp_group=self.pg_collection.cp,
)
```

参考：`3rdparty/Megatron-LM/megatron/core/parallel_state.py` 中 `get_ranks('cp')` 的 group 创建逻辑，以及 `gdn.py` 中的调用。

## 6. CP→HP All-to-All 的 shape 和数据流

### 6.1 高层 shape

固定一个 TP 坐标，输入是：

```text
CP layout:
[S/C, B, M/T]
```

CP→HP 之后变为：

```text
HP layout:
[S, B, M/(T*C)]
```

底层 helper 的 contract 是：

```text
[global_sequence/CP, B, local_hidden]
    -> All-to-All on CP group
[global_sequence, B, local_hidden/CP]
```

参考：`3rdparty/Megatron-LM/megatron/core/ssm/mamba_context_parallel.py` 中 `_all_to_all_cp2hp`。

### 6.2 source rank 和 destination rank

以 `C=4` 为例。每个 source CP rank 的局部张量可以写成：

```text
source cp0: [sequence chunk 0, hidden slice 0..3]
source cp1: [sequence chunk 1, hidden slice 0..3]
source cp2: [sequence chunk 2, hidden slice 0..3]
source cp3: [sequence chunk 3, hidden slice 0..3]
```

All-to-All 后，destination rank `cpj` 得到所有 source rank 的第 `j` 个 hidden slice：

```text
destination cp0:
  cp0 的 hidden slice 0
  cp1 的 hidden slice 0
  cp2 的 hidden slice 0
  cp3 的 hidden slice 0

destination cp1:
  cp0 的 hidden slice 1
  cp1 的 hidden slice 1
  cp2 的 hidden slice 1
  cp3 的 hidden slice 1
```

这些 slice 沿 sequence 维重新排列后，形成：

```text
cp0: 完整 sequence + hidden/head slice 0
cp1: 完整 sequence + hidden/head slice 1
cp2: 完整 sequence + hidden/head slice 2
cp3: 完整 sequence + hidden/head slice 3
```

所以，All-to-All 不是把某个 rank 的完整 Q/K/V 复制给另一个 rank，而是：

```text
每个 source rank 的局部 sequence
    + 自己的 TP-local hidden/head channels
        -> 切 hidden/head channels
        -> 分发到 CP group 的各个 destination rank
        -> 每个 destination 组装完整 sequence 的部分 channels
```

### 6.3 Q/K/V section 的 permutation

GDN 的 `in_proj` 输出不是一个可以任意整体切分的单一 section，而是：

```text
query | key | value | z | beta | alpha
```

因此，GDN 在 All-to-All 前调用 `_build_head_perm_for_split_sections`，按照每个 section 单独切分 hidden/head 维，再把 section 重新排布，使一次未分段的 All-to-All 等价于对各 section 分别通信。

每个 section 都必须满足：

```text
section_size % CP == 0
```

相关检查位于：

```text
3rdparty/Megatron-LM/megatron/core/ssm/gated_delta_net/common.py
```

中的 `_build_head_perm_for_split_sections`。

## 7. 一个 token 的 Q/K/V 在哪里计算

以 `TP=1, CP=4` 为例：

### CP→HP 之前

```text
cp0:
  token 0 ... token S/4-1 的全部 Q/K/V heads

cp1:
  token S/4 ... token S/2-1 的全部 Q/K/V heads

cp2:
  token S/2 ... token 3S/4-1 的全部 Q/K/V heads

cp3:
  token 3S/4 ... token S-1 的全部 Q/K/V heads
```

这里是“局部 sequence + 全部 head”这一概念；如果 TP>1，则“全部 head”还要改成 TP-local heads。

### CP→HP 之后

```text
cp0: token 0 ... token S-1 的 head slice 0
cp1: token 0 ... token S-1 的 head slice 1
cp2: token 0 ... token S-1 的 head slice 2
cp3: token 0 ... token S-1 的 head slice 3
```

因此你的理解可以精确表述为：

> 每个 CP rank 先对本地 sequence chunk 做本地 `in_proj`。随后，CP group 内的 All-to-All 把每个 source rank 的 hidden/head 子块分发到各个 destination rank，使每个 rank 获得完整 sequence 上的一组 heads 或 hidden channels。

但不能说 TP 自动切了 token。更准确的关系是：

```text
TP：先切 output feature/head 和权重
CP：先切 sequence
CP→HP：在固定 TP 坐标的 CP group 内，把 sequence shard + TP-local heads
        转换为 full sequence + 更小的 head/hidden slice
```

## 8. Megatron-Core GDN forward 流程

当前 `GatedDeltaNet.forward` 的主要顺序如下：

```text
hidden_states
  -> TP ColumnParallel in_proj
       得到 TP-local q/k/v/z/beta/alpha
  -> CP->HP All-to-All
       [S/C, B, M/T] -> [S, B, M/(T*C)]
  -> transpose 和 section split
  -> 本地 causal Conv1d(qkv)
  -> 本地 Q/K 准备、归一化、head reshape
  -> 本地计算 g 和 beta
  -> 本地 gated delta rule kernel
  -> 本地 gated RMSNorm + output gate
  -> HP->CP All-to-All
       [S, B, V/(T*C)] -> [S/C, B, V/T]
  -> TP RowParallel out_proj
       TP All-Reduce
```

代码顺序位于：

```text
3rdparty/Megatron-LM/megatron/core/ssm/gated_delta_net/gdn.py:129-273
```

### 8.1 本地 GDN 计算

CP→HP 完成后，每个 rank 具有：

```text
完整 sequence
部分 Q/K/V heads 或 hidden channels
```

然后在本 rank 内执行：

1. 拆分 `qkvzba` 为 `qkv`、`gate`、`beta`、`alpha`；
2. 对 Q/K/V 执行 depthwise causal convolution；
3. 按本地 head 维度 reshape Q/K/V；
4. 根据 `A_log`、`dt_bias`、`alpha`、`beta` 计算门控量；
5. 调用 `gated_delta_rule` kernel；
6. 对输出做 gated norm。

这里不再需要 CP rank 之间传递完整 token 历史或 recurrent state。对于每个 rank 负责的 heads，序列方向的状态推进由本 rank 的 GDN kernel 完成。

### 8.2 HP→CP 和 output projection

本地 GDN 输出仍然处于 HP layout：

```text
[S, B, V/(T*C)]
```

通过 HP→CP All-to-All 后变为：

```text
[S/C, B, V/T]
```

之后输入 TP Row Parallel `out_proj`。每个 TP rank 计算自己的 partial matrix product，并在 TP group 上进行输出 All-Reduce。

参考：

- `3rdparty/Megatron-LM/megatron/core/ssm/gated_delta_net/common.py` 中 `_gated_norm_and_a2a`；
- `3rdparty/Megatron-LM/megatron/core/ssm/mamba_context_parallel.py` 中 `_all_to_all_hp2cp`；
- `3rdparty/Megatron-LM/megatron/core/tensor_parallel/layers.py` 中 `RowParallelLinear.forward`。

## 9. forward 中的通信和顺序依赖

### 9.1 有 collective 阶段依赖

GDN forward 的主要阶段依赖是：

```text
每个 rank 本地完成 in_proj
        |
        v
CP group 上 CP->HP All-to-All
        |
        v
每个 rank 本地执行完整 sequence 的部分 heads
        |
        v
CP group 上 HP->CP All-to-All
        |
        v
TP group 上 out_proj All-Reduce
```

当前 All-to-All 底层使用 `torch.distributed.all_to_all_single`。所有参与 rank 必须按相同顺序进入 collective，collective 返回后本 rank 才能继续后续计算。

### 9.2 没有 `cp0 完成 GDN 后传给 cp1` 的串行路径

不存在以下执行方式：

```text
cp0 完成完整 GDN
    -> 把 GDN output 或 state 发给 cp1
    -> cp1 才开始 GDN
```

实际是：

```text
cp0 本地 in_proj --┐
cp1 本地 in_proj --├-> CP->HP All-to-All
cp2 本地 in_proj --┤
cp3 本地 in_proj --┘
                         |
                         v
             cp0/cp1/cp2/cp3 并行执行本地 GDN
```

因此存在的是 collective 到达/完成依赖，而不是 CP rank 之间的 GDN 计算串行依赖。

### 9.3 GDN 内部仍有 token 顺序依赖

Gated Delta Rule 的数学状态仍然具有因果关系：

```text
state[t] 依赖 state[t-1]
```

但在 Megatron-Core 的 CP→HP 实现中，一个 rank 在进入 GDN kernel 前已经获得完整 sequence 的局部 heads。因此这个 token 顺序依赖在单个 rank 的 kernel 内处理，不是 `cp0 -> cp1` 的跨 rank state 传递。

这也是 HP layout 的核心作用：

```text
用 CP group 的 hidden/sequence All-to-All
替代跨 CP rank 的 recurrent state 传递
```

## 10. backward 中的通信

CP↔HP All-to-All 通过自定义 autograd function 封装。其 backward 使用逆向 All-to-All：

```text
forward CP->HP  => backward HP->CP
forward HP->CP  => backward CP->HP
```

底层 `_AllToAll.backward` 会交换 forward 的 split 参数：

```text
3rdparty/Megatron-LM/megatron/core/tensor_parallel/mappings.py
```

TP 线性层的反向则遵循标准 Megatron TP 规则：

- Column Parallel Linear 根据配置对输入梯度执行 All-Reduce 或 Sequence Parallel Reduce-Scatter；
- Row Parallel Linear 的前向 partial output 在 TP group 上 All-Reduce；
- 各 rank 根据本地权重分片计算相应 weight gradient。

所以 backward 也不是某个 CP rank 计算完梯度后逐个传给下一个 CP rank，而是通过对应 collective 的反向操作完成梯度布局转换。

## 11. 与 FLA state-summary CP 的区别

独立 FLA CP 实现可以采用另一种算法：

```text
每个 rank 处理自己的连续 sequence chunk
  -> 计算本地 state contribution 和 transition matrix
  -> AllGather 固定大小的 state summary
  -> 每个 rank 本地 prefix merge
  -> 使用本 rank 的 initial state 继续 GDN kernel
```

这种方法的特点是：

- rank 保留连续 sequence chunk；
- 通信的是 state summary，而不是 hidden/head activation；
- 需要显式处理 recurrent state、Conv1d 边界和 document 边界；
- merge 的工作量可能随 rank 在状态链中的位置增加。

而本文描述的 Megatron-Core GDN module 是：

```text
本地 in_proj
  -> CP->HP activation All-to-All
  -> 每个 rank 处理完整 sequence 的部分 heads
  -> HP->CP activation All-to-All
```

两者都可以解决长序列上的并行问题，但通信对象和状态处理方式不同：

| 项目 | Megatron-Core GDN module | 独立 FLA state-summary CP |
| --- | --- | --- |
| rank 的主计算布局 | full sequence + partial hidden/head | local sequence chunk + local heads |
| 主要 CP 通信对象 | Q/K/V/gate 等 activation | recurrent state summary |
| forward CP 通信 | CP→HP、HP→CP All-to-All | state summary AllGather 和本地 merge |
| recurrent state 是否跨 rank显式传递 | 不在该 module 外层显式传递 | 通过 summary merge 重建 |
| `cp_comm_type` | GDN 外层忽略 | 由 FLA CP operator 自己定义 |

## 12. 配置和实现限制

### 12.1 维度整除

CP→HP 需要每个 GDN projection section 都能被 CP 均匀切分：

```text
q section      % CP == 0
k section      % CP == 0
v section      % CP == 0
gate section   % CP == 0
beta section   % CP == 0
alpha section  % CP == 0
```

这也是为什么 GDN 会分别记录 `in_proj_split_sections` 和 `feat_dim_split`，而不是只对整个 `in_proj` 输出做简单 reshape。

### 12.2 `cp_comm_type` 和 HCP

标准 Transformer Engine Attention 支持：

```text
p2p
all_gather
a2a
a2a+p2p
```

但 GDN module 的构造函数明确表示，`cp_comm_type` 对 GDN 外层会被忽略。GDN 使用自己的 CP↔HP All-to-All。

因此：

```text
标准 Attention 的 HCP / a2a+p2p
不等同于
GDN 的 CP->HP / HP->CP
```

### 12.3 HP 没有单独的 checkpoint parallel dimension

GDN 的 checkpoint sharding 仍然主要以 TP 维为依据。HP 是运行时 activation layout，不应被理解为需要单独保存一套 HP checkpoint 或单独建立 HP rank group。

## 13. 总结

Megatron-Core GDN 的并行关系可以概括为：

```text
TP：切权重和 output feature/head
CP：切输入 sequence
CP->HP：在固定 TP 坐标的 CP group 内做 activation All-to-All
       [S/C, B, M/T] -> [S, B, M/(T*C)]
本地 GDN：每个 rank 处理完整 sequence 的部分 heads
HP->CP：恢复 CP layout
TP out_proj：Row Parallel + TP All-Reduce
```

最重要的三点是：

1. TP 本身不切 token；Sequence Parallel 才会让 TP 参与 sequence 维切分。
2. GDN 的 HP 不是独立并行轴，而是 CP group 上 All-to-All 后的 hidden/head 数据布局。
3. CP rank 之间需要在 All-to-All 处同步，但不存在 `cp0 完成 GDN 输出后再驱动 cp1` 的串行计算链；每个 rank 在拿到完整 sequence 的本地 heads 后并行执行 GDN。
