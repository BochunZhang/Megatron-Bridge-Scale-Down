# Combined Forward/Backward 调度逻辑

本文分析当前 Megatron-Core 中 `combined_forward_backward_step` 及其两层封装，说明它们被哪些 scheduler 调用，以及相对普通 scheduler 的优化点和使用边界。

## 1. 结论概览

`combined_forward_backward_step` 不是一个新的顶层 scheduler，而是 `overlap_moe_expert_parallel_comm=True` 时使用的细粒度 1F1B 执行原语。它同时接收：

- 当前 microbatch 的 forward schedule plan；
- 前一个或另一个 microbatch 的 backward schedule plan；
- pipeline steady state 中可选的 PP P2P 前后处理回调。

随后它把 Transformer layer 拆成 attention/router、MoE dispatch、expert MLP、MoE combine 等节点，在计算流和通信流上交错执行。核心目标不是减少通信量，而是把 MoE Expert Parallel 的 All-to-All（A2A）dispatch/combine 通信隐藏在另一 microbatch 的 attention/MLP 计算之下。

直接或间接调用关系如下：

```text
get_forward_backward_func()
├── PP = 1
│   └── forward_backward_no_pipelining()
│       └── combined_1f1b_schedule_for_no_pipelining()
│           └── combined_forward_backward_step()
│
├── PP > 1 且 VPP != None
│   └── forward_backward_pipelining_with_interleaving()
│       └── combined_1f1b_schedule_for_interleaved_pipelining()
│           └── combined_forward_backward_step()
│
└── PP > 1 且 VPP = None
    └── forward_backward_pipelining_without_interleaving()
        └── 不调用 combined_forward_backward_step()
```

两条 combined 路径都要求：

```python
config.overlap_moe_expert_parallel_comm is True
and not forward_only
```

因此，`forward_only=True` 的验证或推理不会进入该路径。

## 2. 三个核心函数的职责

源码入口：

- `3rdparty/Megatron-LM/megatron/core/pipeline_parallel/combined_1f1b.py`
- `3rdparty/Megatron-LM/megatron/core/pipeline_parallel/schedules.py`
- `3rdparty/Megatron-LM/megatron/core/models/common/model_chunk_schedule_plan.py`

### 2.1 `combined_1f1b_schedule_for_no_pipelining`

该函数由 `forward_backward_no_pipelining` 在 PP=1 时调用。虽然没有 pipeline stage，它仍然在 microbatch 维度建立 1F1B：

```text
假设有 4 个 microbatches：

Phase 0: F0
Phase 1: B0 + F1
Phase 2: B1 + F2
Phase 3: B2 + F3
Phase 4: B3
```

执行分为三段：

1. 首个 microbatch 只有 forward，用来产生后续 backward 所需的 schedule plan、loss node 和 activation。
2. 中间 `num_microbatches - 1` 轮同时传入 `f_model` 和 `b_model`，将新 microbatch 的 forward 与前一 microbatch 的 backward 合并调度。这些轮次位于 `no_sync_func()` 内，暂不触发最终梯度同步。
3. 最后一个 microbatch 只有 backward，并在 `no_sync_func()` 外执行，使最后一次梯度同步按普通训练语义发生。

如果使用 Megatron FSDP，该路径会先恢复 raw parameters；在细粒度执行绕过普通 module forward/backward hook 时，还会显式连接每层参数释放 hook。这里的 FSDP 处理主要是维持参数生命周期和内存正确性，不是 combined 调度的主要性能来源。

### 2.2 `combined_1f1b_schedule_for_interleaved_pipelining`

该函数由 `forward_backward_pipelining_with_interleaving` 调用，是既有 interleaved pipeline scheduler 的适配层。它没有重新实现整个 PP scheduler，而是复用后者已有的：

- virtual microbatch 到 model chunk、真实 microbatch 的映射；
- input/output tensor 队列；
- 参数和梯度同步；
- pipeline warmup、steady state、cooldown；
- PP P2P send/recv 回调。

单次调用的顺序是：

```text
forward_step_helper_preprocess
backward_step_helper_preprocess
combined_forward_backward_step
forward_step_helper_postprocess
backward_step_helper_postprocess
```

在 steady state，forward virtual microbatch 和 backward virtual microbatch 会同时存在，形成真正的 combined 1F1B。warmup 只有 forward，cooldown 只有 backward，但仍可通过同一接口传入 `None` 表示缺失的一侧。

开启 EP overlap 时，interleaved scheduler 会额外增加一个 warmup forward microbatch，保证每个 steady-state 组合中的 forward 与 backward 相互独立，避免把存在数据依赖的两项工作错误地并行调度。

当 `overlap_p2p_comm=True` 时，scheduler 还会把异步 PP send/recv 封装为 `pre_forward`、`post_forward`、`pre_backward`、`post_backward` 回调，交给 combined plan 在更合适的层内位置发起或等待。

当前 interleaved combined 路径明确不支持 Megatron FSDP；PP=1 的 non-pipelined combined 路径可以处理 FSDP。

### 2.3 `combined_forward_backward_step`

这是合并 forward/backward 的核心适配函数，可以概括为六步。

#### 第一步：准备执行上下文

- 根据配置建立 autocast context。
- delayed FP8 recipe 使用覆盖整次 combined pass 的 FP8 context；其他 FP8 recipe 在 layer 内细粒度进入 context。
- 设置 first/current microbatch 状态。
- 把 pipeline 输入设置到 forward model。
- PP=1 + FSDP backward 时显式调用根级 `pre_backward()`。

#### 第二步：forward 不立即执行完整 model，而是构造 plan

普通 `forward_step` 调用：

```python
output_tensor, loss_func = forward_step_func(data_iterator, model)
```

combined 路径改为：

```python
f_schedule_plan, loss_func = forward_step_func(
    data_iterator,
    unwrapped_model,
    return_schedule_plan=True,
)
```

返回值必须实现 `AbstractSchedulePlan`。以 GPT 为例，实际类型是 `TransformerModelChunkSchedulePlan`。它描述 model chunk 的节点，而不是已经完成的 forward 输出。

#### 第三步：从旧输出恢复 backward plan

前一轮 forward 结束时，函数会把两项对象暂存在 output tensor 上：

```python
output_tensor.schedule_plan = f_schedule_plan
output_tensor.loss_func = loss_node
```

轮到该 microbatch backward 时，再取回并清空：

```python
b_schedule_plan = b_output_tensor[0].schedule_plan
loss_node = b_output_tensor[0].loss_func
```

如果当前 stage 没有从下游收到 `b_output_tensor_grad`，说明这里需要从 loss 开始反传。函数先对标量 loss 执行 backward，得到进入 model schedule plan 的梯度，然后释放 loss node 状态和已完成使用的输入 storage。

#### 第四步：执行 model-chunk 级 1F1B

核心调用是：

```python
type(f_schedule_plan or b_schedule_plan).run(
    f_schedule_plan,
    b_schedule_plan,
    b_grad=b_grad,
    pre_forward=pre_forward,
    pre_backward=pre_backward,
    post_forward=post_forward,
    post_backward=post_backward,
)
```

`TransformerModelChunkSchedulePlan.run` 从前向第一层和反向最后一层开始配对。假设一个 model chunk 有 4 层：

```text
Phase 0: forward preprocess + backward postprocess
Phase 1: F(layer 0) + B(layer 3)
Phase 2: F(layer 1) + B(layer 2)
Phase 3: F(layer 2) + B(layer 1)
Phase 4: F(layer 3) + B(layer 0)
Phase 5: PP send/recv（如启用相应回调）
Phase 6: 延后的首层 wgrad + forward postprocess + backward preprocess
```

如果 forward/backward plan 的层数不同，先重叠 `min(f_num_layers, b_num_layers)` 层，再串行完成剩余一侧。因此该接口也能覆盖 warmup/cooldown 的纯 forward 或纯 backward。

#### 第五步：执行 layer 级细粒度交错

每个 `TransformerLayerSchedulePlan` 把一层拆为：

```text
pre_dispatch_computation : attention -> norm -> router -> dispatch preprocess
moe_dispatch             : dispatch A2A
mlp                      : local expert MLP
moe_combine              : combine A2A
mtp_post_process         : 可选的 MTP 后处理
```

其中 `pre_dispatch_computation` 和 `mlp` 位于计算流，`moe_dispatch` 和 `moe_combine` 位于独立通信流。`ScheduleNode` 用 CUDA event 在两个流之间表达真正的数据依赖。

同时存在 forward layer 和 backward layer 时，代码中的大致顺序为：

```text
通信流: B combine ───── F dispatch ─ B dispatch ───── F combine
计算流: ─ F pre-dispatch ─ B MLP/dgrad/wgrad ─ F MLP ─ B pre-dispatch
```

这不是简单地把 `forward_step()` 和 `backward_step()` 相邻调用。拆分后的 A2A 与另一 microbatch 的 GEMM/attention 可以在不同 CUDA stream 上同时运行，才构成实际的通信隐藏。

#### 第六步：恢复普通 scheduler 所需的输出语义

- forward 侧执行 loss/postprocess，将 plan 与 loss node 附着到输出，供未来 backward 使用；
- backward 侧读取 pipeline input 的 `.grad`，返回给上游 stage；
- 及时释放 layer/node 引用，避免 schedule plan 跨 microbatch 泄漏；
- PP=1 + FSDP 路径最后调用根级 `post_backward()`。

因此，外层 scheduler 仍然收到与普通 forward/backward helper 对应的 `output_tensor`、`num_tokens` 和 `input_tensor_grad`，不需要理解 layer 内部如何拆分。

## 3. 哪些 scheduler 会调用

| 顶层 scheduler | `get_forward_backward_func` 选择条件 | combined 调用条件 | 调用方式 |
|---|---|---|---|
| `forward_backward_no_pipelining` | `PP == 1` | EP overlap 开启且不是 forward-only | 调用 `combined_1f1b_schedule_for_no_pipelining` |
| `forward_backward_pipelining_with_interleaving` | `PP > 1` 且 `VPP != None` | EP overlap 开启且不是 forward-only | helper wrapper 调用 `combined_1f1b_schedule_for_interleaved_pipelining` |
| `forward_backward_pipelining_without_interleaving` | `PP > 1` 且 `VPP == None` | 不支持 | 不调用 |

需要注意，配置校验要求 `PP > 1` 时必须设置 VPP，才能开启 `overlap_moe_expert_parallel_comm`。因此非 interleaved PP 不只是“当前没有接调用点”，也是配置层面被排除的组合。

直接调用 `combined_forward_backward_step` 的只有两个 helper：

1. `combined_1f1b_schedule_for_no_pipelining`；
2. `combined_1f1b_schedule_for_interleaved_pipelining`。

## 4. 相对普通 scheduler 的优化

### 4.1 从 microbatch 级串行变为 layer/node 级交错

PP=1 的普通 scheduler 对每个 microbatch 执行完整 forward，再执行完整 backward：

```text
F0 -> B0 -> F1 -> B1 -> F2 -> B2
```

combined scheduler 先建立一个 forward，再把后续 forward 与前一 microbatch backward 配对：

```text
F0 -> (B0 + F1) -> (B1 + F2) -> B2
```

interleaved PP 原本已经在 microbatch/model-chunk 级采用 1F1B；combined 路径进一步把每个 paired step 下沉到 layer 和 layer 内节点级，使 EP A2A 与计算真正重叠。

### 4.2 双向隐藏 MoE A2A

普通路径中，一个 microbatch 内部的 dispatch A2A、expert MLP、combine A2A 必须按依赖顺序执行，A2A 容易直接暴露在 critical path 上。

combined 路径利用两个独立 microbatch：

- forward 的 dispatch/combine 通信可被 backward 的 attention/MLP 计算覆盖；
- backward 的 dispatch/combine 通信可被 forward 的 attention/MLP 计算覆盖。

收益来自缩短“暴露的通信时间”，不是减少传输字节，也不改变 EP group 或 token routing。

### 4.3 通信流、CUDA event 与可选高优先级

普通 model forward/backward 主要依赖 autograd 顺序。combined plan 显式为计算节点和通信节点分配不同 CUDA stream，并由共享 CUDA event 建立依赖。`high_priority_a2a_comm_stream=True` 时，A2A 通信流还可使用 CUDA 高优先级，减少通信 launch 被长计算排队阻塞的机会。

高优先级只影响调度优先级，不保证通信带宽，也不保证性能一定提升。

### 4.4 PP P2P 与层内计算进一步重叠

在 interleaved PP 且开启 `overlap_p2p_comm` 时：

- forward 输出的 P2P send/recv 在通信流发起，可与 backward pre-dispatch 计算重叠；
- backward 梯度的 P2P send/recv 发起后，可与延后的首层 weight-gradient 计算重叠。

这是 PP 通信和 combined layer schedule 的协同优化；如果没有开启 PP P2P overlap，对应回调不会提供这一层收益。

### 4.5 可选 delayed wgrad 提供更多可覆盖计算

`delay_wgrad_compute=True` 会把部分 weight-gradient GEMM 从常规 backward 节点中拆出并延后。例如 model chunk 的最后阶段会在 PP 通信发起后执行首层 pre-dispatch wgrad，使通信期间仍有计算可运行。

它是 combined EP overlap 上的额外优化，不是 `combined_forward_backward_step` 的必要条件，也不保证在所有 workload 上继续提速。应先单独验证 plain EP overlap，再 A/B 测试 delayed wgrad。

### 4.6 更积极的生命周期管理

combined 路径会在节点使用完输入后释放部分 tensor storage，在 backward 完成后释放 node/plan 状态，并为 FSDP 补上细粒度 reshard hook。这些机制用于控制更复杂并发调度下的 activation、autograd graph 和参数生命周期。

这不意味着 combined 路径必然降低峰值显存：多 microbatch 并行存活、通信 buffer、dispatcher backend 和可选特性都会影响最终内存，应以实测为准。

## 5. 普通与 combined scheduler 对比

| 维度 | 普通 scheduler | Combined 1F1B scheduler |
|---|---|---|
| model 执行接口 | 调用完整 model forward，由 autograd 完成完整 backward | model 返回 `AbstractSchedulePlan`，调度器执行拆分节点 |
| 重叠粒度 | microbatch/model-chunk 级；层内基本遵循单个图的依赖顺序 | model chunk、layer、layer 内节点三级 |
| EP A2A | dispatch/combine 较容易暴露 | 与另一 microbatch 的 attention/MLP 双向重叠 |
| CUDA stream | 主要使用当前计算流，通信由各实现自行安排 | 显式计算流 + 通信流 + CUDA event |
| PP=1 顺序 | 每个 microbatch 完整 F 后完整 B | 首 F、若干 `(B + F)`、末 B |
| Interleaved PP | 普通 interleaved 1F1B | 保留外层 interleaved 逻辑，内部改为 fine-grained combined plan |
| 非 interleaved PP | 支持普通 1F1B | 不支持 combined EP overlap |
| activation checkpoint | 支持普通 partial/full 机制 | `checkpoint_activations_microbatch` 必须为 `None`；full/MoE recompute 受限 |
| forward-only | 支持 | 自动走回普通路径 |
| 主要目标 | 通用、兼容性优先 | 隐藏 MoE EP A2A，优化通信受限 workload |

## 6. 配置与兼容性边界

当前 `TransformerConfig` 对该路径的主要校验包括：

- PyTorch 至少为 2.6；
- `expert_model_parallel_size > 1`；
- dispatcher 必须是 `alltoall` 或 `flex`；
- 精度必须为 BF16 或 FP16；
- `PP > 1` 时必须配置 VPP；
- 不支持 full activation recompute；
- 不允许设置 `recompute_method` 或 `recompute_num_layers`；
- selective recompute 中不能包含 `moe`；
- 不能同时开启 `moe_shared_expert_overlap`；
- MTP 最多一层；
- Transformer Engine CUDA Graph 不能捕获 `moe` 或 `mlp` scope；
- `delay_wgrad_compute` 依赖 `overlap_moe_expert_parallel_comm`；
- interleaved pipeline combined 路径不支持 Megatron FSDP。

此外，用户的 `forward_step_func` 必须接受 `return_schedule_plan=True`，模型必须实现 `build_schedule_plan` 并返回 `AbstractSchedulePlan`。这使 combined 路径不是对任意 PyTorch model 都透明适用的通用 scheduler。

## 7. 性能判断

理论上，若普通 profile 中 EP dispatch/combine A2A 明显暴露，combined scheduler 能把其中一部分转为与计算并发，从而降低 step time、提高 GPU 利用率。收益通常在 EP 较大、跨节点通信较重、且有足够 microbatches 时更明显。

但加速不是必然的：

- 如果原本是纯计算受限，几乎没有 A2A 可隐藏；
- 如果 microbatch 数过少，首 F 和末 B 的不可重叠边界占比会很高；
- 通信和 GEMM 可能竞争 SM、HBM 或互连资源；
- 小规模 EP 的调度、event 和 stream 开销可能超过隐藏的通信；
- dispatcher 中的 host sync 或 dynamic-shape 路径可能重新串行化执行；
- 数值目标保持一致，但执行和梯度累加顺序改变，不应期待 bitwise 相同。

验证时应固定 dispatcher、routing、batch shape、并行布局、CUDA Graph 和运行时，仅切换 `overlap_moe_expert_parallel_comm`，比较：

1. 稳态 step time 和 model TFLOPS/GPU；
2. A2A 与 GEMM/attention 的时间交集；
3. 暴露通信时间，而不是简单累加各 stream 的 kernel duration；
4. loss、梯度、NaN/skip 情况；
5. 峰值显存。

## 8. 关键源码索引

| 内容 | 文件与位置 |
|---|---|
| 顶层 scheduler 选择 | `pipeline_parallel/schedules.py:get_forward_backward_func` |
| PP=1 combined 分支 | `pipeline_parallel/schedules.py:forward_backward_no_pipelining` |
| interleaved PP combined 分支 | `pipeline_parallel/schedules.py:forward_backward_pipelining_with_interleaving` |
| 三个 combined 函数 | `pipeline_parallel/combined_1f1b.py` |
| schedule node、stream、event 抽象 | `pipeline_parallel/utils.py:ScheduleNode` |
| layer 与 model-chunk 1F1B | `models/common/model_chunk_schedule_plan.py` |
| 模型细粒度 callable | `models/common/fine_grained_callables.py` |
| 配置约束 | `transformer/transformer_config.py` |
| 正确性测试 | `tests/unit_tests/a2a_overlap/test_schedule_layer_1f1b.py` |

上述相对路径均以 `3rdparty/Megatron-LM/megatron/core` 或 `3rdparty/Megatron-LM` 为起点。
