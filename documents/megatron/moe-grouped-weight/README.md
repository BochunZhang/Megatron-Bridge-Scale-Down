# MoE Grouped Weight 与 NVTETensor Handle 溢出

本文总结大规模 MoE 训练中 Transformer Engine（TE）tensor handle 数量
溢出的原因，以及编译期和运行时两种处理方式。当前 Bridge checkout 通过
`pyproject.toml` 将 Transformer Engine 固定为源码 revision
`4329ff84bfbdaa778a33cba02a15fb0807c64689`；因此实际行为应以运行环境中构建
的 TE 版本为准。

## 1. 问题背景

在测试 **256 个 expert、40 个 MoE layer** 时，可能出现类似
`NVTETensor` 数量溢出或 tensor handle pool 耗尽的错误。

这里的主要限制不是模型权重本身的显存容量，而是 TE 为 tensor handle 分配的
内部元数据对象数量。TE 的 `NVTETensor` 是指向实际数据的 tensor wrapper，
本身不拥有权重或 activation 的数据内存。因而这个错误应首先按
**tensor handle 数量上限**排查，而不是直接按 CUDA OOM 排查。

在触发该问题的 TE 实现中，静态 tensor handle 容量由类似下面的常量控制：

```c
const size_t MAX_TENSOR_NUM = 20 * 1024 * 1024 / sizeof(Tensor);
```

这表示默认静态容量约为 20 MiB 除以单个 `Tensor` 对象大小。256 expert 和
40 个 MoE layer 会让每个 rank 上需要管理的 expert 参数、GroupedLinear
元数据、量化状态以及 forward/backward 生命周期中的 tensor handle 快速累积，
从而触发该容量限制。

实际数量还会受到下列因素影响：

- `expert_model_parallel_size`：决定每个 rank 上的 local expert 数量；
- FC1、FC2 两个 GroupedLinear；
- BF16、MXFP8 或其他量化路径产生的额外 metadata；
- FSDP/DDP 参数管理和梯度状态；
- 是否启用 `moe_single_grouped_weight`，即是否使用 GroupedTensor 参数布局；
- TE 版本及其 tensor handle 生命周期实现。

因此，`256 × 40` 是容易暴露问题的规模，但不是一个能单独决定精确 handle
数量的公式。

## 2. 当前模型路径中的 grouped weight

需要区分三个概念：

```text
moe_grouped_gemm
    多个 expert GEMM 合并到 grouped GEMM kernel

use_transformer_engine_op_fuser
    将 FC1 -> activation -> FC2 组织成 TE fused op sequence

moe_single_grouped_weight
    将 weight0..weightN 存为一个 GroupedTensor 参数
```

`moe_grouped_gemm=True` 并不等价于所有 expert 权重都变成一个连续参数。
当前配置如果没有打开 `moe_single_grouped_weight`，通常仍然是：

```text
weight0, weight1, ..., weightN
```

只是这些权重会被 grouped GEMM 一起执行。

在当前 Megatron-Core 实现中，`moe_single_grouped_weight=True` 还要求：

- `moe_grouped_gemm=True`；
- Transformer Engine >= 2.14；
- `use_transformer_engine_op_fuser=True`；
- BF16/FP16、MXFP8 或 NVFP4 等受支持的主权重路径。

配置约束见：

- `3rdparty/Megatron-LM/megatron/core/transformer/transformer_config.py`
  中的 `moe_single_grouped_weight` 定义和校验；
- `3rdparty/Megatron-LM/megatron/core/transformer/moe/experts.py`
  中的 TEGroupedMLP fused-path 检查。

`expert_fc1` 或 `moe_act` 的 selective offload 当前只支持非 op-fuser 路径；
如果需要同时使用 fused grouped MLP 和 offload，应使用 `fused_group_mlp`
这个整体 offload 边界，而不是 `expert_fc1`/`moe_act` 的局部边界。

## 3. 解决方案一：重新编译 TE，增大 `MAX_TENSOR_NUM`

可以直接修改 TE 源码中的 `MAX_TENSOR_NUM`，增大静态 tensor handle 容量，
然后重新编译和安装 Transformer Engine。

示意流程：

1. 在实际构建使用的 TE revision 中定位 `MAX_TENSOR_NUM`；
2. 增大其值，例如改为原值的 2 倍或 4 倍；
3. 清理旧的 TE 构建产物；
4. 使用当前 CUDA、PyTorch 和 Python 环境重新构建 TE；
5. 重新启动训练并验证 handle 数量、吞吐和显存。

该方法改变的是 TE 二进制中的固定容量，优点是上限明确，缺点是需要重新构建
和维护自定义 TE 版本。不要只修改 Bridge 的 Python 配置；如果运行时加载的
仍是旧 TE 二进制，修改不会生效。

## 4. 解决方案二：使用运行时 handle pool 环境变量

Transformer Engine 提供运行时环境变量，用于增大内部 handle pool：

```bash
export NVTE_TENSOR_HANDLE_POOL_SIZE_MB=128
export NVTE_GROUPED_TENSOR_HANDLE_POOL_SIZE_MB=128
```

在触发该问题的 TE 实现中，设置这些 pool 大小后，handle 分配会使用运行时
pool 的容量，而不是继续受静态 `MAX_TENSOR_NUM` 预分配上限约束。因此这是
无需重新编译 TE、直接绕开该固定容量限制的运行时方案。

### 4.1 `NVTE_TENSOR_HANDLE_POOL_SIZE_MB`

配置普通 `NVTETensor` handle pool 的容量。当前使用 per-expert
`weight0`/`weight1`/... 参数布局时，通常应优先关注这个变量。

### 4.2 `NVTE_GROUPED_TENSOR_HANDLE_POOL_SIZE_MB`

配置 `NVTEGroupedTensor` handle pool 的容量。当启用
`moe_single_grouped_weight=True`、GroupedLinear single-parameter 路径或其他
GroupedTensor API 时，这个变量更加相关。

这两个变量是 **TE 运行时环境变量，不是编译期常量**。它们不需要重新编译
Transformer Engine，但必须在启动 Python/分布式训练进程之前设置：

```bash
NVTE_TENSOR_HANDLE_POOL_SIZE_MB=128 \
NVTE_GROUPED_TENSOR_HANDLE_POOL_SIZE_MB=128 \
uv run --no-sync python -m torch.distributed.run \
  --standalone --nproc_per_node=4 \
  ...
```

它们扩大的是 TE 的 handle/metadata pool，不会减少 expert 权重的总字节数，
也不会改变 MoE 的 EP、TP、token dispatch 或 grouped GEMM 形状。

## 5. 推荐的排查顺序

建议一次只改变一个变量：

1. 保持模型并行、batch、seq length 和 precision 不变，先设置
   `NVTE_TENSOR_HANDLE_POOL_SIZE_MB=128`；
2. 如果错误仍然指向 GroupedTensor，再设置
   `NVTE_GROUPED_TENSOR_HANDLE_POOL_SIZE_MB=128`；
3. 记录初始化是否成功、峰值显存、step time 和稳定迭代吞吐；
4. 若运行时 pool 足够但仍溢出，再评估重新编译 TE、增加 `MAX_TENSOR_NUM`；
5. 最后再单独比较 per-expert weight 与 `moe_single_grouped_weight` 布局。

不要把扩大 pool 当成权重融合：扩大 pool 只是提高可容纳的 handle 数量；
`moe_single_grouped_weight` 才是参数注册布局的变化，而且会影响 op-fuser、
FSDP/DDP 参数管理、checkpoint key 和 optimizer 状态处理。

## 6. 最小记录项

每次实验至少记录以下配置，避免把不同 handle 路径混在一起：

```text
num_moe_experts
num_moe_layers
expert_model_parallel_size
expert_tensor_parallel_size
moe_grouped_gemm
use_transformer_engine_op_fuser
moe_single_grouped_weight
fine_grained_activation_offloading
offload_modules
NVTE_TENSOR_HANDLE_POOL_SIZE_MB
NVTE_GROUPED_TENSOR_HANDLE_POOL_SIZE_MB
Transformer Engine revision/version
```

最终应将 `MAX_TENSOR_NUM` 溢出、普通 `NVTETensor` pool 耗尽、
`NVTEGroupedTensor` pool 耗尽和 CUDA 显存 OOM 分开报告；它们是不同层次的
资源限制，解决方式也不同。
