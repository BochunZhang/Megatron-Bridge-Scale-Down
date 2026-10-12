# mbench：GB200 单卡 Megatron 模块基准框架

`mbench` 在单张 GPU 上以 **eager 模式**执行 Megatron 训练时真正使用的模块（TE Linear、
`TEGroupedMLP`、`TEDotProductAttention`、`SelfAttention`、`GatedDeltaNet`、`TopKRouter`……），
用 `cuStreamWaitValue32` 闸门 + CUDA Event 测量纯 GPU 时间，并可选用 Nsight Systems + SQL 交叉校验。
不使用 CUDA Graph。

模型相关的配置与启动脚本放在各自目录，例如
[`examples/gb200/micro-batch-size/qwen3.8-2.4T/`](../micro-batch-size/qwen3.8-2.4T/README.md)。

## 目录

| 路径 | 作用 |
| --- | --- |
| `run_mbench.sh` | 以独立进程运行一个 scope（`--scope GEMM\|GatedDeltaNet\|GatedAttention\|MLP\|Experts`）；`--nsys` 时套 `nsys profile` 并导出 SQLite |
| `mbench/run.py` | 读取 YAML、构建 Megatron 模块、逐个 case 计时、写 JSON；`SCOPES` 定义 scope 名与结果目录的对应关系 |
| `mbench/megatron_setup.py` | 单 rank 进程组 + `GPTModelProvider` + 用训练同款 spec 构建模块；挂 `main_grad` 与 DDP 反向 hook |
| `mbench/gate.py` / `gate_watchdog.py` | `cuStreamWaitValue32` 闸门与看门狗进程 |
| `mbench/timing.py` | 轮次（round）计时协议 |
| `mbench/cases.py` | case 描述、FLOPs、EP 路由下本地 expert 的 token 数 |
| `mbench/categories/{gemm,gated_delta_net,gated_attention,mlp,experts}.py` | 五个 scope 的 case 定义 |
| `mbench/build_csv.py` | JSON → 每个 scope 一个 CSV；有 SQLite 时追加 `nsys_*` 列 |

## 模块来源（保证与训练一致）

- 配置：YAML 的 `model` 段原样传给 `megatron.bridge.models.gpt_provider.GPTModelProvider`，
  `finalize()` 后作为 Megatron `TransformerConfig`。并行度全部强制为 1。
- 层 spec：`get_transformer_layer_with_experimental_attention_variant_spec(config)`，即 Qwen3.5/3.8
  Bridge 训练所用的 `transformer_layer_spec`。GDN 层与 attention 层分别取第一个对应层号的 spec，
  `build_module(...)` 构建；MoE 的 router / experts / shared experts 取自同一层 spec 中的
  `MoESubmodules`。MTP `eh_proj` 与 LM head 按 `MultiTokenPredictionLayer` / `GPTModel`
  的构造参数构建（BF16 下 LM head 为 `tensor_parallel.ColumnParallelLinear`）。
- 反向：每个参数分配 fp32 `main_grad`、`grad_added_to_main_grad`，并在 `AccumulateGrad`
  节点上挂与 Megatron DDP `_make_backward_post_hook` 相同逻辑的 hook。因此 TE 层走
  `fuse_wgrad_accumulation`（wgrad GEMM 直接累加进 fp32 `main_grad`），非融合参数
  （router、gate_weight、norm、conv1d、A_log…）由 hook `main_grad.add_`，与训练一致。
- 每个 case 的所有输入张量都 `requires_grad`，所以反向包含 dgrad（等同非首层）。

## 计时协议（`mbench/timing.py`）

```
pre_run_iters 次（不计时、不 gate：自动调优/惰性初始化/allocator 预热，这些需要 host 同步）
repeat until profile_iters 个样本（默认 warmup 3 + profile 10，一轮完成）:
    synchronize
    arm:   cuStreamWaitValue32(stream, flag >= gen)        ← GPU 在此停住
    enqueue: W 次 warmup  → E0 → step → E1 → step → E2 … → EP   （每次迭代一个 NVTX range）
    release: CPU 写 flag = gen                             ← GPU 开始连续执行
    synchronize；样本 = Ei - E(i-1)
```

- 由于 GPU 直到全部任务提交完毕才开始执行，Event 之间不可能出现 GPU 等 CPU 的空泡：
  **测到的是纯 GPU 时间（含 kernel 之间的 GPU 端调度间隙，不含 CPU launch 开销）**。
- 一个 step = 一次训练式迭代：`fwd` 为 autograd 记录下的前向（保存激活后丢弃），
  `fwdbwd` 为前向 + `torch.autograd.backward`（固定的随机 `grad_output`）。
- 闸门实现：pinned（`cuMemHostRegister`）的 `/dev/shm` 共享页作为 flag，通过 ctypes 调用
  `libcuda.so.1` 的 `cuStreamWaitValue32`，不需要额外依赖。
- 安全措施：
  - 提交期间启用 `torch.cuda.set_sync_debug_mode("error")`。若被测算子内部有 host 同步，
    立即报错并对该 case 改为 **不 gate** 测量，CSV 的 `gated=False` 标出。
  - CUDA launch 队列有限。一轮放不下时 CPU 会在提交中阻塞；看门狗进程超时后替 CPU 释放
    flag 并标记 fired，该轮整体丢弃，轮内迭代数减半重试（`discarded_rounds`、`iters_per_round`）。
- 每轮都有 `warmup_iters` 次 gated warmup，用于 GPU 在空等后恢复时钟。

## Nsys 交叉校验（`--nsys`）

`run_mbench.sh --nsys` 以 `nsys profile --trace=cuda,nvtx --sample=none` 运行，并
`nsys export --type=sqlite` 得到 `<output>/nsys/<category>.sqlite`。`build_csv.py` 用 SQL：

1. 读 `NVTX_EVENTS` 中 `mbench|cat=..|case=..|mbs=..|mode=..|phase=..|round=..|iter=..` 的 range；
   只用 JSON 中记录为“接受”的 round。
2. 取该 range 时间窗口内、同一进程（任意线程，含 autograd 反向线程）的
   `CUPTI_ACTIVITY_KIND_RUNTIME` 调用，按 `correlationId` 连接到
   `CUPTI_ACTIVITY_KIND_KERNEL/MEMSET/MEMCPY`。这样即使 GPU 执行发生在 range 结束很久之后，
   归属仍然正确。
3. 每次迭代计算：
   - `nsys_iter_ms` = 本迭代最后一个 GPU 活动结束 − 上一迭代最后一个 GPU 活动结束（与 Event 同口径）；
   - `nsys_span_ms` = 首个开始到最后结束；
   - `nsys_kernel_union_ms` = 区间并集（GPU 忙碌时间）；`nsys_kernel_sum_ms` = 简单求和。
4. 每个 case 取中位数，并给出 `nsys_vs_event_ratio`、`nsys_gpu_busy_ratio`、前 5 个 kernel 的时间占比。

Nsys 追踪本身对 GPU 有少量开销，正式数字以 `event_*` 为准，`nsys_*` 用于校验和解释。

## CSV 列（`<output>/csv/<category>.csv`）

| 列 | 含义 |
| --- | --- |
| `case`, `megatron_module`, `micro_batch_size`, `mode` | 用例标识与被执行的 Megatron 类 |
| 参数列 | 每个 scope 不同：GEMM 的 `m,k,n,groups,rows_per_group`；attention 的头数/head_dim；GDN 的头数/维度；experts 的 rows/路由等 |
| `flops`, `tflops_median` | 该迭代的模型 FLOPs（`flops_scope` 说明范围）与按中位时间计算的 TFLOP/s |
| `event_{median,mean,min,p90,max,std}_ms`, `samples` | CUDA Event 单次迭代时间统计 |
| `gated`, `iters_per_round`, `discarded_rounds` | 计时是否在闸门下完成、实际轮大小、丢弃轮数 |
| `peak_memory_gib`, `sm_clock_mhz`, `power_w` | 峰值显存；case 结束时的 SM 时钟与功耗（需 pynvml） |
| `nsys_*` | 见上节；仅当存在对应 SQLite 时出现 |

## 新增模型

1. 新建 `examples/gb200/micro-batch-size/<model>/mbench.yaml`（`model` 段为 `GPTModelProvider` 字段）。
2. 复制该目录下的 `_env.sh`、`run_all_benchmarks.sh`、`build_csv.sh`，修改 `RESULTS_ROOT`。
3. 结构不同于 GDN/attention 混合 + MoE 的模型需要在 `mbench/categories/` 中新增 case 定义，
   并在 `mbench/run.py` 的 `SCOPES` 与 `run_mbench.sh` 的 scope 映射中注册。
