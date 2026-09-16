# Hook: _post_backward_release_module

**定义位置**: `megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py:594`

**功能概述**: 在 FSDP unit 的 backward 计算完成后，释放该模块的 unsharded 参数 buffer，并将模块训练状态转回 IDLE。这是 `optim_grads_params` 策略下显存回收的核心机制。

## 1. 注册信息

### 1.1 注册等级

仅在 `optim_grads_params`（完全参数分片）策略下注册：

```python
if self.ddp_config.data_parallel_sharding_strategy == "optim_grads_params":
    self.forward_pre_hooks[f"module {name} register post-backward hook"] = (
        module.register_forward_pre_hook(
            functools.partial(_register_post_backward_hook, _post_backward_release_module),
            with_kwargs=True,
        )
    )
```

其他策略（`no_shard`、`optim_grads`）下参数不分片或仅梯度分片，backward 后无需释放参数 buffer。

### 1.2 注册对象

仅注册在 **FSDP unit modules** 上：

```python
if isinstance(module, tuple(fsdp_unit_modules)):
```

遍历 `root_module.named_modules()` 时，通过子模块去重逻辑保证嵌套场景下只有最外层 FSDP unit 被注册：

```python
if any(is_submodule(module, fsdp_module) for fsdp_module in fsdp_modules):
    continue
```

典型注册实例（GPT 模型）：

```
GPTModel (root_module)
├── LanguageModelEmbedding   ← 注册
├── TransformerLayer[0]      ← 注册
├── TransformerLayer[1]      ← 注册
├── ...
├── TransformerLayer[N-1]    ← 注册
└── output_layer             ← 不注册（非 FSDP unit）
```

TransformerLayer 内部子模块（SelfAttention、MLP、LayerNorm）不单独注册，参数释放由父层统一处理。

### 1.3 注册方式

采用 **forward pre-hook + autograd Function** 的间接注册方式。不是直接注册 backward hook，而是在每次 forward 前通过 `_register_post_backward_hook` 在计算图中插入一个恒等 autograd 节点。

#### 为什么使用这种方式

PyTorch 的 `register_full_backward_hook` 时机不够精确，且存在与 in-place 操作不兼容的问题。通过在 module 输入 tensor 上插入自定义 autograd Function，能精确控制在 module 的 input gradients 计算完成后立即触发回调。

#### _register_post_backward_hook 流程

```python
@torch.compiler.disable
def _register_post_backward_hook(post_backward_hook, module, args, kwargs):
    # 1. 无梯度模式下跳过
    if not torch.is_grad_enabled():
        return args, kwargs

    # 2. 展平输入，找出 requires_grad=True 的 tensor
    args_kwargs_list = list(args_list) + list(kwargs_list)
    inp_tensors = [obj for obj in args_kwargs_list
                   if torch.is_tensor(obj) and obj.requires_grad]

    # 3. 无需要梯度的 tensor 则跳过
    if len(inp_tensors) == 0:
        return args, kwargs

    # 4. 通过 RegisterFSDPBackwardFunction 插入 autograd 节点
    inp_tensors = RegisterFSDPBackwardFunction.apply(
        functools.partial(post_backward_hook, module), *inp_tensors
    )

    # 5. 将处理后的 tensor 放回原位
    ...
    return args, kwargs
```

#### RegisterFSDPBackwardFunction

```python
class RegisterFSDPBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, post_backward, *inputs: torch.Tensor):
        ctx.post_backward = post_backward
        return inputs

    @staticmethod
    def backward(ctx, *grads: torch.Tensor):
        ctx.post_backward()       # → _post_backward_release_module(module)
        return (None,) + grads    # 梯度原样传递
```

关键特性：
- `forward` 是恒等操作，不修改 tensor 数据
- `backward` 在梯度传递的同时调用保存的回调函数
- 回调通过 `functools.partial(post_backward_hook, module)` 绑定了具体的 module

### 1.4 执行时间

在 module 的 **backward 计算完成后**触发。具体时机：

```
module forward → RegisterFSDPBackwardFunction.forward (透传)
                            ↓ (backward 阶段，反向传播)
module backward 计算梯度 → RegisterFSDPBackwardFunction.backward 触发
                            ↓
                    _post_backward_release_module(module) 执行
```

由于 autograd 节点插入在 module 输入 tensor 上，当反向传播计算完该 module 所有内部节点的梯度、到达输入 tensor 时，该节点的 backward 被调用。此时 module 已完成梯度计算，参数可以安全释放。

## 2. 执行流程

### 2.1 注册阶段（初始化时）

在 `MegatronFSDP.__init__` 中，对每个匹配的 FSDP unit module 注册 forward_pre_hook：

```python
self.forward_pre_hooks[f"module {name} register post-backward hook"] = (
    module.register_forward_pre_hook(
        functools.partial(_register_post_backward_hook, _post_backward_release_module),
        with_kwargs=True,
    )
)
```

这是一个"注册 hook 的 hook"——每次 forward 执行前动态地在计算图中插入 backward 回调节点。

### 2.2 Forward 阶段（动态插入 autograd 节点）

每次 module 执行 forward 前：

1. `_register_post_backward_hook` 作为 forward_pre_hook 被调用
2. 扫描 module 输入 args/kwargs 中所有 `requires_grad=True` 的 tensor
3. 通过 `RegisterFSDPBackwardFunction.apply()` 在这些 tensor 上插入恒等节点
4. 恒等节点的 `ctx` 保存了 `_post_backward_release_module` 回调

### 2.3 Backward 阶段（触发释放）

当反向传播到达插入的 autograd 节点时：

```python
class RegisterFSDPBackwardFunction(torch.autograd.Function):
    @staticmethod
    def backward(ctx, *grads):
        ctx.post_backward()  # 触发 _post_backward_release_module
        return (None,) + grads
```

### 2.4 _post_backward_release_module 执行内容

```python
def _post_backward_release_module(module, *unused):
    # 1. 断言检查
    assert isinstance(module, tuple(fsdp_unit_modules))
    assert self.ddp_config.data_parallel_sharding_strategy == "optim_grads_params"

    # 2. 释放参数
    release_module_parameters(module, bwd=True)

    # 3. 状态转换
    module._training_state = TrainingState.IDLE
```

#### Step 1: release_module_parameters(module, bwd=True)

```python
def release_module_parameters(module, bwd, lazy=False, *unused):
    for param in module.parameters():
        bucket_id = self.param_and_grad_buffer.param_to_param_group[param]
        self.all_gather_pipeline.release_bucket(bucket_id, bwd, lazy=lazy)

    if not self.ddp_config.keep_fp8_transpose_cache:
        release_params_fp8_transpose_cache(module.parameters())
```

- 遍历 module 所有参数（递归），找到对应 bucket_id
- 调用 `release_bucket(bucket_id, bwd=True)` 释放 buffer 内存
- 清理 FP8 transpose cache（如果未配置保留）

#### release_bucket(bucket_id, bwd=True) 行为

```python
def release_bucket(self, bucket_id, bwd, lazy=False):
    bucket_key = self.get_bucket_key(bucket_id, bwd)
    if self.bucket_status[bucket_key] == BucketStatus.EMPTY:
        return

    self.wait_bucket_ready(bucket_id, bwd, empty_ok=True)

    # bwd=True 时：优先释放 transpose_weight_buffer (mxfp8)
    if bwd and param_group.transpose_weight_buffer is not None:
        buf = param_group.transpose_weight_buffer
    else:
        buf = param_group.model_weight_buffer

    buf.free_bucket_storage()
    self.bucket_status[bucket_key] = BucketStatus.EMPTY
```

注意 `bwd=True` 与 `bwd=False` 的区别：
- `bwd=True`（backward 后）：如果存在 `transpose_weight_buffer`（mxfp8 模式），优先释放它；否则释放 `model_weight_buffer`
- `bwd=False`（forward 后）：总是释放 `model_weight_buffer`

#### Step 2: 状态转换

```python
module._training_state = TrainingState.IDLE
```

将模块状态从 `PRE_BACKWARD` 转为 `IDLE`，标识该模块的 backward 工作完成，不再需要 unsharded 参数。

### 2.5 完整时序

```
IDLE
  → _pre_backward_param_unshard: all-gather 参数, 状态 → PRE_BACKWARD
  → backward 计算梯度
  → RegisterFSDPBackwardFunction.backward 触发
  → _post_backward_release_module 执行:
      release_module_parameters(module, bwd=True) — 释放 buffer
      module._training_state = TrainingState.IDLE
```

梯度的 reduce-scatter 由独立的 `register_post_accumulate_grad_hook` 处理，与参数释放解耦。

## 3. 通信 Overlap

### 显存释放与流水线复用

在 `optim_grads_params` 策略下，backward 按逆序逐层执行。每个 FSDP unit 的 backward 完成后立即释放 unsharded 参数副本，释放的显存可被下一个 FSDP unit 的 backward all-gather 复用：

```
Layer N backward:
  [all-gather Layer N params] → [compute gradients] → [release Layer N params]
                                                              ↓ 显存释放
Layer N-1 backward:
  [all-gather Layer N-1 params (复用释放的显存)] → [compute] → [release]
```

这种机制确保 GPU 显存占用峰值不会随模型层数线性增长，而是维持在约 2 个 FSDP unit 的 unsharded 参数量级（当前层计算 + 下一层 prefetch）。

### 与 double buffer 的配合

当启用 `fsdp_double_buffer` 时，`_post_backward_release_module` 的及时释放更为关键。double buffer 机制允许同时最多持有 2 个 bucket 的 unsharded 参数：一个正在计算，一个正在 all-gather。如果释放不及时，会触发 `_enforce_double_buffer_limit` 中的等待逻辑，阻塞计算。

### 与梯度通信的解耦

参数释放（本 hook）和梯度 reduce-scatter（`post_accumulate_grad` hook）是独立的：
- 参数释放：backward 计算完成后立即执行，释放 model weight buffer
- 梯度通信：每个参数梯度计算完成后独立触发，通过 reduce pipeline 异步执行

这种解耦设计允许梯度通信与后续层的参数释放、all-gather 并行执行，最大化通信-计算 overlap。
