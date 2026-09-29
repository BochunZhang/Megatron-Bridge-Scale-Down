# Megatron 与 DeepSpeed Offload 对比实验

本目录是 offload 与 recompute 对比实验的 Megatron 侧入口。实验固定模型、并行方式、序列长度和 batch 预算，扫描 activation 策略和 micro-batch size，观察 GPU/CPU pinned memory、单次迭代耗时及训练吞吐量的变化。

所有实验都使用 Megatron FSDP，并固定：

```text
ddp.data_parallel_sharding_strategy=optim_grads_params
```

该分片策略对 optimizer state、gradient 和 parameter 分片，对应 DeepSpeed ZeRO-3 的比较层级。当前 runner 不支持 parameter CPU offload，因此 DeepSpeed 的 `param_cpu` 不进入本矩阵。

## 脚本组成

- `pretrain_experiment.sh`：展开并运行完整实验矩阵。
- `experiment_manifest.py`：分别构建 DeepSpeed 与 Megatron override 后的有效模型配置，校验架构字段并记录训练、并行契约。
- `summarize.py`：查找结果树中的 `summary.json`，结合相邻的 `config.json` 生成 `summary.csv`。
- `collect_results.py`：读取每个测试最后 4 个完整 iteration，生成吞吐与 peak memory 的 XLSX 汇总。
- `run_pretrain_fsdp1.sh`：本实验目录中的单次四卡 Megatron 训练 runner；它由 fine-grained offload 实验的 runner 复制并在本目录独立维护。

脚本内部使用 `uv run --no-sync`。运行前需要：

- 从仓库根目录执行本文命令；
- 4 块可见的 NVIDIA GPU；
- 已准备好的 `uv` 环境和项目依赖；
- Hugging Face 模型配置可从本地缓存读取，或者运行环境可以访问 Hugging Face。

## 默认实验矩阵

### 固定配置

| 配置项 | 默认值 |
| --- | --- |
| GPU 数量 | 4 |
| Sequence length | 4096 |
| Precision | BF16 |
| Seed | 42 |
| 每 GPU batch 预算 | 16 |
| Global batch size | 64，即 `16 × 4` |
| Micro-batch size | 1、2、4 |
| CUDA graph | 关闭 |
| Profiler | 关闭（固定为 `none`） |
| FSDP 分片 | `optim_grads_params` |
| Train iterations | 10 |
| Warmup steps | 3 |

模型轴包含：

| 名称 | 模型 | 并行设置 |
| --- | --- | --- |
| `dense` | Qwen/Qwen3.5-9B-Base | 默认 dispatcher |
| `expert` | Qwen/Qwen3.5-35B-A3B-Base | 64 experts、EP=4、HybridEP |

DeepSpeed 和 Megatron 都从上述同一个 Hugging Face Base config 构建 text backbone。这里的“模型配置对齐”指两侧分别应用完 override 后，再比较有效配置，而不只是比较模型 ID。Expert 模型的具体过程为：

```text
DeepSpeed: HF text config + num_experts=64
Megatron:  对应 recipe provider + model.num_moe_experts=64
```

每次非 dry-run 启动时，`experiment_manifest.py` 会实际构建这两个结果，并把它们规范化为相同字段名。校验范围包括 layer 数、hidden size、attention/KV head、head dim、dense/MoE FFN size、shared expert size、expert 数、top-k、vocab、完整 layer type 序列、GDN 参数、MTP、RoPE、norm epsilon、初始化、attention dropout、embedding sharing 和 attention bias。任意字段不同都会在开始训练前报错；两侧的最终值和 mismatch 会写入 `model_manifest.json`。

当前共同的有效模型配置为：

| 字段 | Dense | Expert |
| --- | ---: | ---: |
| Transformer layers | 32 | 40 |
| Hidden size | 4096 | 2048 |
| Attention heads | 16 | 16 |
| KV heads | 4 | 2 |
| Dense FFN intermediate size | 12288 | 不适用 |
| Routed expert FFN intermediate size | 不适用 | 512 |
| Shared expert intermediate size | 不适用 | 512 |
| Experts | 不适用 | 64（两侧均从原始 256 覆盖） |
| Experts per token | 不适用 | 8 |
| Vocabulary size | 248320 | 248320 |
| MTP layers | 1 | 1 |

两侧都保留原生 layer 数和 `3 × linear attention + 1 × full attention` 的层模式。DeepSpeed 对比入口会拒绝 `--apply_model_shape_overrides true` 和非 64 的 `--moe_num_experts`，避免生成无法与 Megatron 对齐的测试。框架专属的 kernel、dispatcher、FSDP/ZeRO、recompute 和 offload 配置不属于模型架构字段，会作为实验策略单独记录。

### Activation 策略

| 策略 | Dense 与 Expert 的共同设置 | 模型相关设置 |
| --- | --- | --- |
| `baseline` | 不启用 recompute，不启用细粒度 activation offload | 无 |
| `recompute` | `recompute_granularity=full`、`recompute_method=uniform`、`recompute_num_layers=1` | 每次重计算一个完整 transformer layer，即逐层重计算 |
| `recompute_offload` | `recompute_granularity=selective`，同时启用细粒度 activation offload | Dense：recompute `[layernorm,mlp_act]`，offload `[mlp_norm,mlp_act]`；Expert：recompute `[layernorm,moe_act]`，offload `[mlp_norm,expert_fc1,moe_act]` |

### Optimizer 策略

当前实验固定使用 `optimizer_none`：optimizer 完全保留在 GPU，`optimizer_cpu_offload=false`、`optimizer_offload_fraction=0.0`，且不启用 optimizer D2H/H2D overlap。

原计划中的 `optimizer_cpu_090`、`optimizer_cpu_075` 和 `optimizer_cpu_100` 暂时不进入矩阵，因为当前环境不能同时启用 Megatron FSDP 和 CPU optimizer。入口脚本中的原策略列表保留为注释，待该组合受支持后再恢复；当前也不开放 `--optimizer-strategies` 参数，避免意外启动不受支持的组合。

默认矩阵共有：

```text
2 models × 3 MBS × 3 activation strategies × 1 GPU optimizer strategy = 18 runs
```

`--repeats` 默认为 1；增大它会按比例增加运行次数。

## 运行命令

以下命令均从仓库根目录执行。

### 查看完整矩阵，不启动训练

```bash
./examples/scale-down/01-analyse/03-megatron-vs-deepspeed/pretrain_experiment.sh \
  --dry-run
```

Dry-run 会打印每个组合展开后的 recompute、activation offload、GPU optimizer 和 FSDP 分片配置，并在结果根目录生成本次矩阵的文本清单；它不会加载模型或启动训练。

### 运行默认完整矩阵

```bash
./examples/scale-down/01-analyse/03-megatron-vs-deepspeed/pretrain_experiment.sh
```

默认命令运行上述 18 个组合。由于默认每个组合只有 10 个 iteration，这个配置更适合验证组合能否运行和定位 OOM，不足以形成稳定的性能结论。正式性能测试应提高 `--train-iters` 和 `--repeats`，并仅统计 warmup 之后的 iteration。

### 完整实验前运行 test 矩阵

```bash
RUN_TIME="20260929-test" \
./examples/scale-down/01-analyse/03-megatron-vs-deepspeed/pretrain_experiment.sh \
  --test
```

`--test` 固定使用 `MBS=1` 和 GPU optimizer。模型轴仍包含 Dense 和 Expert，activation 轴仍包含 `baseline`、`recompute` 和 `recompute_offload`，因此默认共运行 6 个组合。该模式会覆盖命令行传入的 `--micro-batch-sizes`，其余参数（例如 `--train-iters`、`--models` 和 `--activation-strategies`）仍然有效。

`RUN_TIME` 未设置时，入口会使用当前时间自动生成批次名，格式为 `YYYYmmdd-HHMMSS`。需要给批次指定稳定名称时，像上面一样设置环境变量；主脚本不提供 `--run-time` 参数。

test 矩阵结束后会自动生成 XLSX。需要根据已有日志重新生成时，将同一个批次名传给结果收集脚本：

```bash
uv run --no-sync python \
  examples/scale-down/01-analyse/03-megatron-vs-deepspeed/collect_results.py \
  --results-root results/01-analyse/03-megatron-vs-deepspeed \
  --run-time "20260929-test" \
  --output results/01-analyse/03-megatron-vs-deepspeed/megatron-vs-deepspeed-20260929-test.xlsx
```

生成文件为：

```text
results/01-analyse/03-megatron-vs-deepspeed/megatron-vs-deepspeed-20260929-test.xlsx
```

### 运行单一组合

下面只运行 Dense、MBS=1、逐层重计算和 GPU optimizer 的一个组合：

```bash
./examples/scale-down/01-analyse/03-megatron-vs-deepspeed/pretrain_experiment.sh \
  --models "dense" \
  --activation-strategies "recompute" \
  --micro-batch-sizes "1" \
  --train-iters 20 \
  --warmup-steps 3 \
  --repeats 1
```

### 运行裁剪后的 smoke 矩阵

下面同时检查 Dense 和 Expert 的 baseline 与 recompute+offload，optimizer 固定在 GPU：

```bash
./examples/scale-down/01-analyse/03-megatron-vs-deepspeed/pretrain_experiment.sh \
  --models "dense expert" \
  --activation-strategies "baseline recompute_offload" \
  --micro-batch-sizes "1" \
  --train-iters 10 \
  --warmup-steps 3 \
  --repeats 1
```

### 指定结果目录

```bash
RUN_TIME="20260929-benchmark" \
./examples/scale-down/01-analyse/03-megatron-vs-deepspeed/pretrain_experiment.sh \
  --models "dense" \
  --activation-strategies "recompute_offload" \
  --micro-batch-sizes "1" \
  --train-iters 10 \
  --warmup-steps 3 \
  --results-root "$PWD/results/01-analyse/03-megatron-vs-deepspeed"
```

当前矩阵入口支持以下选项：

```text
--models
--activation-strategies
--micro-batch-sizes
--precision bf16|fp8mx
--train-iters
--warmup-steps
--repeats
--per-gpu-batch-size
--results-root
--test
--dry-run
```

`--per-gpu-batch-size` 必须能被每个 MBS 整除。Global batch size 始终由 `per_gpu_batch_size × 4` 计算，因此改变 MBS 只改变 gradient accumulation 形态，不改变每个 optimizer step 的 token 数。

## 观测指标

矩阵默认向底层 runner 传入：

```text
--record-memory-usage true
--memory-usage-start-step <warmup_steps>
```

因此正常矩阵和 `--test` 模式都默认开启 GPU 与 CPU pinned memory 记录，不需要额外参数。peak counter 在 warmup 结束前重置，之后每个 logging interval 的数值写入该 run 的 `train.log`：

```text
<results-root>/<model-id>/<result-path-name>/<run-time>/train.log
```

训练日志从指定 step 开始记录以下原始字节指标：

```text
cuda_memory_allocated_bytes
cuda_peak_memory_allocated_bytes
host_memory_allocated_bytes
host_peak_memory_allocated_bytes
```

- `cuda_memory_allocated_bytes`：当前 PyTorch CUDA allocator 已分配显存。
- `cuda_peak_memory_allocated_bytes`：warmup 后 CUDA allocator 峰值。
- `host_memory_allocated_bytes`：当前 CUDA pinned-host allocator 分配量。
- `host_peak_memory_allocated_bytes`：warmup 后 pinned-host allocator 峰值。

`host_memory_*` 与 DeepSpeed 侧的 `torch.cuda.host_memory_stats()` 口径一致，但不是训练进程 RSS，也不是机器总 CPU memory。

`train.log` 每个 logging interval 还包含 `elapsed time per iteration (ms)` 和 Megatron 计算的
`throughput per GPU (TFLOP/s/GPU)`。在本实验固定 global batch size 和 sequence length 后，还可按下式计算每次
optimizer iteration 的全局 token 吞吐量：

```text
tokens/s = global_batch_size × sequence_length / iteration_time_seconds
```

因此默认 GBS=64、sequence length=4096 时，每个 iteration 处理 262144 tokens。比较吞吐时应排除 warmup，并对稳态 iteration 取均值和波动范围。

本实验固定使用 `--profile none`，不会启动 nsys、PyTorch profiler、memory snapshot 或 `nvidia-smi`
采样。四项 memory byte 指标直接来自训练循环中的 PyTorch allocator 计数器，因此不依赖 profiler。

## 输出目录

默认结果根目录是：

```text
results/01-analyse/03-megatron-vs-deepspeed/
```

其中根目录包含：

```text
model_manifest.json
experiment_summary_<run-time>.txt
summary.csv
megatron-vs-deepspeed-<run-time>.xlsx
```

单次运行目录结构为：

```text
<results-root>/<model-id>/<result-path-name>/<run-time>/
```

`result-path-name` 按下面的固定顺序拼接：

```text
dense:  dtype_<dtype>-mbs_<mbs>-gbs_<gbs>-<test-name>
expert: dtype_<dtype>-mbs_<mbs>-gbs_<gbs>-dispatcher_<dispatcher>-<test-name>
```

其中 `test-name` 为 `<activation>-<optimizer>-r<repeat>`。内部 `RUN_NAME` 分别以 `dense-` 或
`expert-` 开头，runner 会校验该前缀，并将 `model_kind`、`test_name` 和 `result_path_name` 同时写入
`config.json`，供结果收集脚本直接读取。

例如：

```text
results/01-analyse/03-megatron-vs-deepspeed/
  qwen35_text_9b/
    dtype_bf16-mbs_1-gbs_64-recompute-optimizer-none-r1/
      <run-time>/

  qwen35_text_35b_a3b/
    dtype_bf16-mbs_1-gbs_64-dispatcher_hybridep-recompute-offload-optimizer-none-r1/
      <run-time>/
```

每个 run 会写入 `config.yaml`、`config.json`、`command.txt`、`environment.json`、`run_info.txt`、`train.log`、`summary.json`、`gpu_utilization.json` 和 `rank_logs/`。模型名位于上一级目录；组合目录名包含 dtype、MBS、GBS、activation、optimizer 和 repeat，Expert 组合还包含 dispatcher。

`RUN_TIME` 在 `pretrain_experiment.sh` 启动时只生成一次，默认格式为 `YYYYmmdd-HHMMSS`；同一矩阵的所有组合共用该值。主脚本不接受 `--run-time`，需要固定批次名时使用环境变量：

```bash
RUN_TIME="20260929-test" \
./examples/scale-down/01-analyse/03-megatron-vs-deepspeed/pretrain_experiment.sh --test
```

实验结束后，入口脚本会自动生成 `summary.csv` 和 XLSX。手动重新生成时，先更新 CSV：

```bash
uv run --no-sync python \
  examples/scale-down/01-analyse/03-megatron-vs-deepspeed/summarize.py \
  results/01-analyse/03-megatron-vs-deepspeed
```

不指定 `--run-time` 时，XLSX 收集器会从结果目录中发现批次，并选择批次名排序后的最后一个；对于默认时间戳，这就是最近一次运行：

```bash
uv run --no-sync python \
  examples/scale-down/01-analyse/03-megatron-vs-deepspeed/collect_results.py \
  --results-root results/01-analyse/03-megatron-vs-deepspeed
```

输出文件默认为 `megatron-vs-deepspeed-<run-time>.xlsx`。回看指定批次，或者 `RUN_TIME` 使用了自定义名称时，显式传入批次名：

```bash
uv run --no-sync python \
  examples/scale-down/01-analyse/03-megatron-vs-deepspeed/collect_results.py \
  --results-root results/01-analyse/03-megatron-vs-deepspeed \
  --run-time "20260929-test" \
  --output results/01-analyse/03-megatron-vs-deepspeed/megatron-vs-deepspeed-20260929-test.xlsx
```

`summary.csv` 每行对应一个 run，汇总运行状态和配置。XLSX 包含两张表：`Summary` 每个测试组合一行，
`Samples` 保留每个测试最后 4 个完整 iteration。汇总指标包括平均 iteration time、平均 TFLOP/s/GPU、
平均 tokens/s、`GPU Peak (GiB/bytes)` 以及 `CPU Pinned Peak (GiB/bytes)`。`Samples` 中也保留最后 4 个 iteration 各自的 GPU/CPU peak。失败或指标不足的 run 也会保留，
并在 `Status` 与 `Error` 列中说明原因。

## 与 DeepSpeed 结果对比

- 对齐模型、sequence length、global batch size、precision、seed 和 warmup 后再比较。
- Megatron `optim_grads_params` 与 DeepSpeed ZeRO-3 对齐的是分片层级，不代表两者 kernel 或调度实现完全相同。
- DeepSpeed `param_cpu` 没有进入 Megatron 矩阵，因为当前 runner 不支持 parameter CPU offload。
- DeepSpeed activation CPU offload 与本实验的 Megatron selective recompute + fine-grained activation offload 不是 kernel-equivalent；应比较 memory/throughput 趋势，而不是宣称实现等价。
- 同时查看 allocator peak、pinned-host peak、iteration time 和 tokens/s。单次 10-iteration smoke run 只能用于功能与 OOM 检查，不能作为最终性能数据。
