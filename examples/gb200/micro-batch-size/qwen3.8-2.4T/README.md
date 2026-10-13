# Qwen3.8 2.4T：GB200 单卡算子 / 模块效率测试（micro-batch-size 扫描）

目标：在单张 GB200 上，用 **Megatron 训练时实际执行的模块与 kernel**，测量 Qwen3.8 2.4T 各类
计算在不同 micro-batch size / 序列长度（`tokens = seq × mbs`，默认 seq = 4096）下的耗时与 TFLOP/s。计时屏蔽 CPU 开销，
不使用 CUDA Graph。

测量框架（计时协议、Nsys 校验、CSV 列定义）见
[`examples/gb200/benchmark/README.md`](../../benchmark/README.md)。

## 1. 文件

| 文件 | 作用 |
| --- | --- |
| `mbench.yaml` | 模型结构（`GPTModelProvider` 字段）、扫描范围、计时参数 |
| `run_all_benchmarks.sh` | 依次运行各 scope（每个 scope 一个进程），最后生成 CSV；支持 `--seq`、`--scope`、`--nsys` |
| `build_csv.sh` | 只重新生成某个序列长度的 CSV（`results/.../seq<len>/csv/<category>.csv`） |
| `_env.sh` | 路径与 `PYTHONPATH`（含同级 `flash-linear-attention` checkout，若存在） |

## 2. 模型配置与假设

| 项 | 取值 |
| --- | --- |
| 层 | 92 层：69 × GatedDeltaNet + 23 × gated attention（`linear_attention_freq: 4`），每层 MoE |
| hidden | 8192，RMSNorm（zero-centered gamma），SiLU/SwiGLU，无 bias |
| Gated attention | 64 Q 头、4 KV 头、head_dim 256，QK RMSNorm，输出门，RoPE（`rotary_percent 0.25`） |
| GatedDeltaNet | 16 key 头、128 value 头、head_dim 128/128，conv kernel 4 |
| MoE | 512 experts，top-10，expert FFN 2048，shared expert FFN 2048 + sigmoid gate，router fp32、`global_aux_loss` |
| 词表 | 248320（不共享 embedding） |
| 精度 | BF16 参数与激活，fp32 `main_grad`，`gradient_accumulation_fusion: true` |
| 序列 / mbs | 默认 4096（`--seq` 可改为 16k / 64k / 256k …）；mbs = 1, 2, 4, 8 |
| 并行假设 | TP = CP = 1，正式 EP = 64：每 rank 8 个本地 expert |

说明：
- 结构参数沿用此前测试的形状；Qwen3.8 无公开 HF config，若与实际不符请修改 `mbench.yaml` 的 `model` 段。
- attention 输出门由 Megatron `_apply_output_gate` 实现，固定为 **sigmoid**（旧脚本里写的
  swish 在 Megatron 中并不存在）。
- 单卡执行，不做任何通信（MoE dispatch/combine、TP/CP 集合通信都不在测试范围内）。

## 3. 实验清单

实验分为 5 个 scope，用 `--scope` 选择，默认全部执行。每个 scope 输出一个结果目录和一个 CSV：

| `--scope` | 结果目录 / CSV | 内容 |
| --- | --- | --- |
| `GEMM` | `gemm` | 所有线性层 |
| `GatedDeltaNet` | `gated_delta_net` | GDN kernel 与整个 `GatedDeltaNet` 模块 |
| `GatedAttention` | `gated_attention` | core attention 与整个 `SelfAttention` 模块 |
| `MLP` | `mlp` | shared expert（`SharedExpertMLP`）；Qwen3.8 没有 dense MLP 层，shared expert 就是每个 token 都经过的 MLP |
| `Experts` | `experts` | `TopKRouter` 与 routed experts（`TEGroupedMLP`） |

每个 case 都对 `mbs ∈ {1,2,4,8}` × `mode ∈ {fwd, fwdbwd}` 测量。

### 3.1 `GEMM`：所有线性层（经由其所属的 Megatron 模块执行）

| case | Megatron 模块 | M | K | N | 每次迭代出现次数 |
| --- | --- | --- | --- | --- | --- |
| `gdn_in_proj` | `TELayerNormColumnParallelLinear`（含融合 RMSNorm） | tokens | 8192 | 37120 | 69 |
| `gdn_out_proj` | `TERowParallelLinear` | tokens | 16384 | 8192 | 69 |
| `attention_linear_qkv` | `TELayerNormColumnParallelLinear`（含融合 RMSNorm） | tokens | 8192 | 34816 | 23 |
| `attention_linear_proj` | `TERowParallelLinear` | tokens | 16384 | 8192 | 23 |
| `router_gating` | `TopKRouter.gating`（`router_gating_linear`，fp32 输出） | tokens | 8192 | 512 | 92 |
| `shared_expert_fc1` | `TEColumnParallelLinear` | tokens | 8192 | 4096 | 92 |
| `shared_expert_fc2` | `TERowParallelLinear` | tokens | 2048 | 8192 | 92 |
| `shared_expert_gate` | `F.linear(x, SharedExpertMLP.gate_weight)` | tokens | 8192 | 1 | 92 |
| `mtp_eh_proj` | `TEColumnParallelLinear` | tokens | 16384 | 8192 | 每个 MTP 层 1 次 |
| `output_layer` | `tensor_parallel.ColumnParallelLinear` | tokens | 8192 | 248320 | 1 |
| `routed_expert_fc1` | `TEColumnParallelGroupedLinear`，8 组 | 8 × 1.25·tokens | 8192 | 4096 | 92 |
| `routed_expert_fc2` | `TERowParallelGroupedLinear`，8 组 | 8 × 1.25·tokens | 2048 | 8192 | 92 |

- 输入布局与训练一致（`[s, b, k]`）。`fwdbwd` 包含 dgrad 和 wgrad。TE 层的 wgrad 直接累加进
  fp32 `main_grad`；LM head 在缺少 APEX `fused_weight_gradient_mlp_cuda` 时退化为非融合 wgrad，
  CSV 的 `fused_wgrad_accumulation` 列会标出。
- routed expert 的 GEMM 使用均匀路由：每个本地 expert 收到 `tokens·topk·EP/num_experts = 1.25·tokens` 行。

### 3.2 `GatedAttention`

| case | 执行内容 |
| --- | --- |
| `core_attention` | 只执行 `TEDotProductAttention`（causal，GQA 64/4，hd 256，`sbhd`）。具体后端由 TE 按 `attention_backend: auto` 选择，可从 `nsys_top_kernels` 确认 |
| `self_attention` | 整个 Megatron `SelfAttention`：融合 input RMSNorm 的 `linear_qkv`、QK norm、RoPE（fused）、core attention、sigmoid 输出门、`linear_proj` |

### 3.3 `GatedDeltaNet`

| case | 执行内容 |
| --- | --- |
| `gated_delta_rule` | `GatedDeltaNet.gated_delta_rule`（FLA `chunk_gated_delta_rule`），输入按 Megatron 的方式准备：q/k 先 L2 归一化并 `repeat_interleave` 到 128 头，g 为 fp32，beta 为 bf16，`initial_state=None` |
| `gated_delta_net` | 整个 Megatron `GatedDeltaNet`：融合 RMSNorm 的 `in_proj`、causal conv1d、L2 norm、g/beta 计算、kernel、门控 RMSNorm、`out_proj` |

### 3.4 `MLP`

| case | 执行内容 |
| --- | --- |
| `shared_experts` | `SharedExpertMLP`：fc1 → SwiGLU → fc2，× sigmoid(gate) |

### 3.5 `Experts`

| case | 执行内容 |
| --- | --- |
| `router` | `TopKRouter.forward`：fp32 gating GEMM、softmax top-10、global aux loss |
| `routed_experts_uniform` | `TEGroupedMLP`（8 个本地 expert：grouped fc1 → 加权 SwiGLU → grouped fc2），均匀路由 |
| `routed_experts_multinomial` | 同上，每 rank 的 token 数取自以 `seed=2026` 对 512 个 expert 做等概率多项式抽样后 rank 0 的 8 个 expert，带有自然的不均衡 |

`tokens_per_expert` 按 alltoall dispatcher 的方式以 host tensor 传入，因此 dispatcher 的
D2H 同步、permute/unpermute 与通信不在计时范围内。

## 4. 测量方法（概要）

1. 每个 scope 在一个独立进程中运行：单 rank NCCL 进程组，Megatron 并行状态全部为 1。
2. 每个 case 先执行 `pre_run_iters`（默认 1）次，不计时也不加闸门，用于 cuBLAS/cuDNN/Triton
   自动调优和惰性初始化。这些操作需要 host 同步，不能放在闸门下。
3. 计时：GPU 被 `cuStreamWaitValue32` 停住，CPU 提交 `warmup_iters`（默认 3）次 warmup 和
   `profile_iters`（默认 10）次计时迭代（每次计时迭代后记录一个 CUDA Event），然后释放，GPU 连续执行。
   默认 3 + 10 = 13 次在一轮内完成；单次迭代时间 = 相邻 Event 之差，共 10 个样本，报告中位数、p90 等。
4. 由于 GPU 在全部任务提交后才开始执行，测量结果只含 GPU 时间，不含 CPU launch 开销。
   如果提交过程中出现 host 同步或 launch 队列被占满，会自动检测：前者改为不 gate 测量并标记，
   后者丢弃该轮并缩小轮大小。
5. `--nsys` 时，再用 Nsys SQLite 按 `correlationId` 把每次迭代的 kernel 归属出来，独立重算迭代时间
   （`nsys_iter_median_ms`），与 Event 结果交叉对比（`nsys_vs_event_ratio` 应接近 1）。

## 5. 运行

```bash
# 全部 scope，仅 Event 计时
examples/gb200/micro-batch-size/qwen3.8-2.4T/run_all_benchmarks.sh

# 全部 scope + nsys trace + SQL 交叉校验列
examples/gb200/micro-batch-size/qwen3.8-2.4T/run_all_benchmarks.sh --nsys

# 只跑部分 scope（空格或逗号分隔，不区分大小写）/ 部分 mbs
examples/gb200/micro-batch-size/qwen3.8-2.4T/run_all_benchmarks.sh --scope "GatedDeltaNet GatedAttention"
examples/gb200/micro-batch-size/qwen3.8-2.4T/run_all_benchmarks.sh --scope GEMM --micro-batch-sizes 1,8

# 长序列（k = 1024，也可直接写 token 数）：结果写入 seq16k/、seq64k/、seq256k/ …
examples/gb200/micro-batch-size/qwen3.8-2.4T/run_all_benchmarks.sh --seq 16k

# 多个序列长度一次跑完（空格或逗号分隔），含小于 1k 的 token 数
examples/gb200/micro-batch-size/qwen3.8-2.4T/run_all_benchmarks.sh --seq "256,1k" --scope GEMM
examples/gb200/micro-batch-size/qwen3.8-2.4T/run_all_benchmarks.sh --seq 256k --scope "GEMM GatedDeltaNet" --micro-batch-sizes 1

# 单个 scope、单个 case（底层入口）
examples/gb200/benchmark/run_mbench.sh \
  --config examples/gb200/micro-batch-size/qwen3.8-2.4T/mbench.yaml \
  --scope GatedAttention --cases core_attention --seq 64k \
  --output-dir results/gb200/micro-batch-size/qwen3.8_2.4T/seq64k --nsys

# 重新整理某个序列长度的 CSV（默认 4k）
examples/gb200/micro-batch-size/qwen3.8-2.4T/build_csv.sh --seq 64k
```

要求：仓库 `uv` 环境（PyTorch、Transformer Engine、megatron-core、flash-linear-attention、numpy、PyYAML）；
`--nsys` 需要 `nsys` 在 `PATH` 中。

## 6. 输出

```
results/gb200/micro-batch-size/qwen3.8_2.4T/seq<len>/     例如 seq4k、seq16k、seq256k
  gemm/ gated_delta_net/ gated_attention/ mlp/ experts/   每个 (case, mbs, mode) 一个 JSON
  nsys/<category>.nsys-rep / .sqlite                       仅 --nsys
  csv/gemm.csv  csv/gated_delta_net.csv  csv/gated_attention.csv  csv/mlp.csv  csv/experts.csv
```

长序列 × 大 mbs 的组合可能超出单卡显存（例如 256k × mbs 8 的 LM head 输出就超过 1 TB）：这类 case
不会中断整个 scope，而是写一条 `status: oom` 的记录，在 CSV 中保留该行、计时列为空。

整模型单次迭代的计算时间可以用 `gemm.csv` 中的单层耗时 × 出现次数加权求和来估算。

## 7. 已知局限

- 只覆盖 BF16。若正式训练使用 MXFP8，需要在 `GPTModelProvider` 字段中打开对应的 FP8 配置，
  并补充量化相关的计时。
- 只测孤立模块：输入在 L2 中常驻，没有通信 kernel 争抢 SM 和带宽，所以结果是单卡计算时间的**上界**。
- `fwd` 是训练式前向（autograd 记录、保存激活），不是推理前向。
- RoPE 频率表在计时之外预先计算（`GPTModel` 每次前向计算一次，开销很小）。
- 本仓库所在机器没有 GPU，这套脚本只做过静态检查和 CPU 侧单元验证（CSV/SQL 解析、看门狗），
  首次在 GB200 上运行时请先只跑单个 case 确认。
