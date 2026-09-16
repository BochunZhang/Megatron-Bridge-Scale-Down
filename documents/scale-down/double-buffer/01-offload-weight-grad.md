# Double Buffer：`model_weight` 与 `main_grad`

本文描述 model_weight 和 main_grad 在 GPU/CPU 的存储和管理逻辑。

Transformer Layer 按照 parameter 性质 (dtype, is_expert, ...) 拆分为若干 ParameterGroup，每个 ParameterGroup 里面使用 model_weight_buffer 和 main_grad_buffer 两个 DataParallelBuffer 对象管理参数和梯度。

- ParameterGroup.fsdp_unit 表示其所属的 transformer layer id
- ParameterGroup.index，表示其是 transformed layer 里面的第几个 ParameterGroup
- Allocator 根据 index 从 slot 里面为 ParameterGroup 分配临时 buffer (每个 slot 里面有两个 buffer，一个用计算，一个用于 prefetch)

e.g., Transformer Layer 里面有 bf16 dense、bf16 expert、mxfp8 dense、mxfp8 expert 四类 buffer，他们的 index 依次 0、1、2、3，allocator 从 slot[0]里面为 bf16 dense 分配 buffer，从 slot[3] 里面为 mxfp8 expert 分配 buffer

## 1. `DataParallelBuffer`

`DataParallelBuffer` 管理 cpu、gpu 上存储的 shard 数据和 unshared 数据，包括 gpu_data、cpu_data、gpu_bucket 等字段。

| 字段 | is_distributed | device |
| --- | --- | --- |
| `gpu_data` | 指向 buffer 存储的数据 <br> is_distributed=True 存储 shard data <br> is_distributed=False 存储 unshard data | device=`gpu` 时数据常驻 gpu <br> device=`cpu` 时由 allocator 临时从 slot[i] 里面分配 buffer|
| `cpu_data` | 同上 | device=`cpu` 时数据常驻 cpu <br> device=`gpu` 时无效|
| `gpu_bucket` | 指向 no-shard data, is_distributed=True 时由 allocator 分配 buffer, is_distributed=False 时指向 gpu_data | - |

gpu_bucket 用于通信
- 对于 model_weight, gpu_bucket 作为 all-gather 的输出 buffer, 需要将 param.data 绑定到 gpu_bucket
- 对于 main_grad, gpu_bucket 用于汇总 param.grad, 合并为 continuous buffer 用于 reduce-scatter 通信
- 在开启 offload 的情况 (device=`cpu`), gpu_data 和 gpu_bucket 使用的是不同的 allocator

FSDP 配置含义如下
- FSDP-1 `optim`：只将 optimizer state 分片，包括 `main_weight`、`exp_avg`、`exp_avg_sq`；
- FSDP-2 `optim_grads`：将 optimizer state 和 `main_grad` 分片；
- FSDP-3 `optim_grads_params`：将 optimizer state、`main_grad` 和 `model_weight` 分片。


## 2. `model_weight` 的处理逻辑


### 2.1 prefetch 流水线

| Hook | 主要职责 |
| --- | --- |
| pre-forward hook | 遍历 `layer[i]` 的全部参数, 发起 fetch `layer[i]` & fetch `layer[i+1]`, 等待 fetch `layer[i]` 结束后开始计算|
| post-forward hook | 标记本层从 allocator 申请的临时 buffer 可以被释放掉 |
| pre-backward hook | 遍历 `layer[i]` 的全部参数, 发起 fetch `layer[i]` & fetch `layer[i-1]`, 等待 fetch `layer[i]` 结束后开始计算|
| post-backward hook | 标记本层从 allocator 申请的临时 buffer 可以被释放掉 |


### 2.2 FSDP + offload 配置

| 配置 | `gpu_data` | `gpu_bucket`|
| --- | --- | --- |
| FSDP-1 & 2 | unshard. <br> 模型参数常驻 gpu <br> 不需要 fetch | 指向 `gpu_data`, 指向常驻 gpu 的 buffer |
| FSDP-3 | shard. <br> 模型参数常驻 gpu <br> 不需要 fetch | fetch 前向 allocator 申请临时 buffer, 用作 all-gather 的 recv_buffer |
| FSDP-1 & 2 + offload | unshard. <br> 模型参数常驻 cpu <br> fetch 前向 allocator 申请临时 buffer <br> fetch 时从 `cpu_data` 加载数据 | 指向`gpu_data`, 对应临时 buffer |
| FSDP-3 + offload | shard. <br> 模型参数常驻 cpu <br> fetch 前向 allocator 申请临时 buffer <br> fetch 时从 `cpu_data` 加载数据 | fetch 前向 allocator 申请临时 buffer, 用作 all-gather 的 recv_buffer|



## 3. `main_grad` 的处理逻辑

backward 产生的 `param.grad` 不是连续的通信 buffer。对需要 reduce-scatter 的 FSDP-2/3

1. 将 `param.grad` copy 到连续的 `gpu_bucket`;
2. 对 `gpu_bucket` 原地执行 reduce-scatter;
3. 将得到的本地 shard 累加到 `main_grad_buffer.gpu_data`;
4. 通信完成后释放 `gpu_bucket`;
5. 启用 CPU offload 时，还要把更新后的 GPU buffer D2H 到 `cpu_data`，再释放 GPU slot.


### 3.1 Hook 职责

| Hook | 主要职责 |
| --- | --- |
| root pre-backward hook | 遍历 module, 标记进入 backward 状态 |
| pre-backward hook | 遍历 `layer[i]` 的全部参数, 发起 fetch `layer[i]` & fetch `layer[i-1]`  <br> 等待 fetch `layer[i]` 结束后开始计算 <br> fetch & prefetch 需要等待使用这个 slot 的上一个 ParameterGroup.main_grad_buffer 使用完毕才可以发起通信 |
| `post_accumulate_grad` hook | 每个参数的 `param.grad` 就绪后执行 `_grad_acc`, 将梯度拷贝到连续 buffer <br> 在 bucket group 内全部 param 的梯度都完成拷贝后, 发起 reduce-scatter 或 offload |
| root post-backward hook | 兜底处理未触发参数 hook 的梯度, 提交剩余通信 |

- 对于 model_weight, 在 `layer[i]` 执行前发起 `layer[i+1]` 的 prefetch, 此时 `layer[i-1]` 已经完成, 因此不需要等待 layer[i-1] 是否完成.
- 对于 main_grad, 在 `layer[i]` 执行前发起 `layer[i-1]` 的 prefetch, 不代表 `layer[i+1]` 的 reduce-scatter 或者 offload 完成, 因此要通过 cuda event 来同步, 确保通信完成后, layer[i-1] 才可以使用这个 buffer.

### 3.2 各配置的 buffer 生命周期

#### FSDP-1 + offload

`cpu_data` 中的 `main_grad` 不分片。
- backward 前，pre-backward hook 向 allocator 申请 buffer，将其绑定到 `gpu_data`，发起 reload。reload 前需要等待上一个使用这个 buffer 的那个 ParameterGroup 完成 offload.
- `post_accumulate_grad` hook 将每个 `param.grad` 累加到 `gpu_bucket` (实际指向 `gpu_data`)；
- ParameterGroup 的全部参数的梯度 copy 完成后，把 `gpu_data` D2H 回 `cpu_data`，标记该 GPU buffer 可释放

#### FSDP-2/3
`gpu_data` 中的 `main_grad` 是分片的。
- 第一个参数的 `post_accumulate_grad` hook 触发时，allocator 申请一个 `gpu_bucket`，用于收集 param.grad；
- `post_accumulate_grad` hook 将每个 `param.grad` 拷贝到 `gpu_bucket`；拷贝前需要等待上一个使用这个 buffer 的那个 Parameter Group 完成 reduce-scatter
- ParameterGroup 的全部参数的梯度 copy 完成后，执行 reduce-scatter，结果原地保存
- reduce-scatter 结束后, 将收到的结果累加到 `main_grad_buffer.gpu_data`

#### FSDP-2/3 + offload

`cpu_data` 中的 `main_grad` 是分片的。
- backward 前，pre-backward hook 向 allocator 申请 buffer，将其绑定到 `gpu_data`，发起 reload。reload 前需要等待上一个使用这个 buffer 的那个 ParameterGroup 完成 offload.
- 第一个参数的 `post_accumulate_grad` hook 触发时，allocator 申请一个 `gpu_bucket`，用于收集 param.grad；
- `post_accumulate_grad` hook 将每个 `param.grad` 拷贝到 `gpu_bucket`；拷贝前需要等待上一个使用这个 buffer 的那个 Parameter Group 完成 offload (offload 期间也会用到这个 buffer)
- ParameterGroup 的全部参数的梯度 copy 完成后，执行 reduce-scatter，结果原地保存
- reduce-scatter 结束后, 将收到的结果累加到 `main_grad_buffer.gpu_data`，然后开启 offload


### 3.3 zero-grad

对于 FSDP-1/2/3，需要在 iteration 开始前执行 zero-grad，清空 buffer
在开启 offload 之后，第一个 backward 期间没必要从 cpu reload main_grad，因为我们期望的 grad 时被清空的，我们直接将从 allocator 的申请的 gpu_bucket / gpu_data 情况即可.