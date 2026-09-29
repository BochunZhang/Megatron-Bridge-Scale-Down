# Offload ON/OFF 性能对比

## 测试目标

### 1. 量化 dense / MoE 开启 offload 后的性能下降幅度

此前测试发现 dense 模型开启 fine-grained activation offload 后性能下降约 5%，expert 模型下降接近 20%。
本测试需要复现该现象，并通过 nsys 找到性能瓶颈的来源 —— 分析是否需要为 moe 的流量模式设计新的 offload 策略。

## 测试日志

### 1. qwen3.5 训练效率偏低

测试 qwen3.5 dense / expert 的训练吞吐为 500 TFlops / 100 TFlops。
此前测试 deepseek (mla) 架构时, 能跑到 1000 TFlops / 700 TFlops，增加 mbs 能够跑到 1100 / 800+ TFlops。
增加 deepseek 模型作为对照

问题定位:
- deepseek 的训练也只有 500TFlops，这是没有安装 flash-attn 导致
- 安装 flash-attn-4[cu13]==4.0.0b11 后训练吞吐恢复到 1000TFlops

### 2. allreduce 通信

通过 nsys 观察到大量 allreduce 通信，增加 nvtx 标记，检查通信来源。

问题定位:
- codex 生成的测试里面，错误的开启了 zero-3 (optim-grads-params)，改成 zero-1 (optim) 即可。

## 测试内容

对比 Qwen Dense、Qwen MoE 以及 DeepSeek 的 dense/expert proxy 在 **开启 / 关闭 fine-grained activation offload** 时的性能差异（step time、tokens/s、显存峰值）：

| 模型 | 选择参数 | scope | offload 模块 |
| --- | --- | --- | --- |
| Qwen Dense (9B/27B) | `--model qwen --scope dense` | dense MLP | `[mlp_norm,mlp_act]` |
| Qwen MoE (35B-A3B, 16 层, 64 experts) | `--model qwen --scope expert` | expert MLP | `[mlp_norm,expert_fc1,moe_act]` |
| DeepSeek-V3 dense proxy (4 层, 32 experts) | `--model deepseek --scope dense` | dense MLP | `[mlp_norm,mlp_act]` |
| DeepSeek-V3 expert proxy (4 层, 32 experts) | `--model deepseek --scope expert` | expert MLP | `[mlp_norm,expert_fc1,moe_act]` |

DeepSeek-V3 的两种 proxy 由脚本 override 构造（不再限制 `--scope dense`）：统一设置 `num_layers=4`、`num_moe_experts=32`、`num_nextn_predict_layers=0`；dense proxy 令 `moe_layer_freq=[0,0,0,0]`（4 层全部走 dense MLP，dispatcher 用 recipe 默认值），expert proxy 令 `moe_layer_freq=[1,1,1,1]`（4 层全部走 MoE）。

扫描维度：

- **MBS**：`1,2,4,8`（默认，必须整除 GBS）
- **精度**：`--dtype bf16`（默认）或 `--dtype mxfp8`
- **dispatcher**（仅 MoE）：`alltoall` / `hybridep`；默认两个都跑，可用 `--dispatcher <alltoall|hybridep>` 只跑其一
- **case**：`baseline`（offload 关）与 `offload`（offload 开）

固定条件：4 GPU FSDP1、seq len 4096、GBS 32、train-iters 10、关闭 recompute / CUDA graph / optimizer offload / checkpoint。

## 文件说明

- `benchmark_mlp_offload.sh` — 矩阵入口，展开所有组合并逐项调用训练脚本
- `run_pretrain_fsdp1.sh` — 软链接到 `../01-mlp-scope-pretrain/`，实际执行 4-GPU FSDP1 训练
- `collect_mlp_offload_results.mjs` — 扫描结果目录，生成 XLSX 吞吐对比表

## 运行指令

前置：仓库根目录 `uv sync`；4 张 GPU 可用（GB200 recipe）；`--profile nsys` 需要 `nsys`；生成 XLSX 需要 Node.js 和 `@oai/artifact-tool` 模块。

```bash
EXPERIMENT_DIR=examples/scale-down/01-analyse/01-fine-grained-offload/02-offload-on-dense-or-expert-model
```

每次启动会自动生成批次标识（`benchmark_id`，即 `RUN_TIME`）。如需把 bf16 / mxfp8 两组吞吐运行合并进同一个 XLSX，给两条 `--profile none` 命令传相同的 `RUN_TIME` 环境变量即可（收集器按 `model/dtype/dispatcher/case/mbs` 区分运行，dtype 不同不会冲突）。

### 测试 1：qwen3.5 dense + expert 综合矩阵（MBS 1,2,4,8）

不带 `--scope` 时 dense 和 expert 都跑；expert 自动覆盖 alltoall 和 hybridep 两个 dispatcher。每个 dtype 分别给出关闭 / 开启 nsys 的指令。

**关闭 nsys（吞吐矩阵，结果进 XLSX）**，每个 dtype 24 个运行（dense 8 + expert 16）：

```bash
# bf16
bash "./examples/scale-down/01-analyse/01-fine-grained-offload/02-offload-on-dense-or-expert-model/benchmark_mlp_offload.sh" \
  --model qwen --dtype bf16 --micro-batch-sizes 1,2,4,8

# mxfp8
bash "./examples/scale-down/01-analyse/01-fine-grained-offload/02-offload-on-dense-or-expert-model/benchmark_mlp_offload.sh" \
  --model qwen --dtype mxfp8 --micro-batch-sizes 1,2,4,8
```

**开启 nsys（定位瓶颈 / allreduce 来源，不进 XLSX）**，每个 dtype 同样 24 个运行：

```bash
# bf16 + nsys
bash "./examples/scale-down/01-analyse/01-fine-grained-offload/02-offload-on-dense-or-expert-model/benchmark_mlp_offload.sh" \
  --model qwen --dtype bf16 --micro-batch-sizes 1,2,4,8 --profile nsys

# mxfp8 + nsys
bash "./examples/scale-down/01-analyse/01-fine-grained-offload/02-offload-on-dense-or-expert-model/benchmark_mlp_offload.sh" \
  --model qwen --dtype mxfp8 --micro-batch-sizes 1,2,4,8 --profile nsys
```

> nsys 只采集 step 7–8（`profile_step_start=7`、`profile_step_end=8`），单 run 开销可控；如只想对代表性 case 采集，可再叠加 `--scope`、`--case`、`--dispatcher`、`--micro-batch-size` 缩小矩阵。

**吞吐矩阵跑完后汇总 TFlops**（只统计 `--profile none` 的成功运行）：

```bash
node examples/scale-down/01-analyse/01-fine-grained-offload/02-offload-on-dense-or-expert-model/analyse_mlp_offload_results.mjs \
  --model qwen
```

### 测试 2：deepseek-v3 dense + expert 综合矩阵（MBS 1,2,4,8）

不带 `--scope` 时 dense proxy 和 expert proxy 都跑；expert 自动覆盖 alltoall 和 hybridep 两个 dispatcher，dense 走 recipe 默认 dispatcher。每个 dtype 分别给出关闭 / 开启 nsys 的指令。

**关闭 nsys（吞吐矩阵，结果进 XLSX）**，每个 dtype 24 个运行（dense 8 + expert 16）：

```bash
# bf16
bash "./examples/scale-down/01-analyse/01-fine-grained-offload/02-offload-on-dense-or-expert-model/benchmark_mlp_offload.sh" \
  --model deepseek --dtype bf16 --micro-batch-sizes 1,2,4,8

# mxfp8
bash "./examples/scale-down/01-analyse/01-fine-grained-offload/02-offload-on-dense-or-expert-model/benchmark_mlp_offload.sh" \
  --model deepseek --dtype mxfp8 --micro-batch-sizes 1,2,4,8
```

**开启 nsys（定位瓶颈 / allreduce 来源，不进 XLSX）**，每个 dtype 同样 24 个运行：

```bash
# bf16 + nsys
bash "./examples/scale-down/01-analyse/01-fine-grained-offload/02-offload-on-dense-or-expert-model/benchmark_mlp_offload.sh" \
  --model deepseek --dtype bf16 --micro-batch-sizes 1,2,4,8 --profile nsys

# mxfp8 + nsys
bash "./examples/scale-down/01-analyse/01-fine-grained-offload/02-offload-on-dense-or-expert-model/benchmark_mlp_offload.sh" \
  --model deepseek --dtype mxfp8 --micro-batch-sizes 1,2,4,8 --profile nsys
```

**吞吐矩阵跑完后汇总 TFlops**（只统计 `--profile none` 的成功运行）：

```bash
node examples/scale-down/01-analyse/01-fine-grained-offload/02-offload-on-dense-or-expert-model/analyse_mlp_offload_results.mjs \
  --model deepseek
```

> 如只想单独跑某一种 proxy，加 `--scope dense` 或 `--scope expert`；expert 还可用 `--dispatcher <alltoall|hybridep>` 只跑其一，并可叠加 `--micro-batch-sizes 1,2` 缩小矩阵（对照 qwen3.5 分析训练效率时的常用组合）。

### 测试 3：qwen3.5 expert 单项测试（MBS 1 + bf16 + hybridep + nsys）

```bash
bash "./examples/scale-down/01-analyse/01-fine-grained-offload/02-offload-on-dense-or-expert-model/benchmark_mlp_offload.sh" \
  --model qwen --scope expert --dtype bf16 --dispatcher hybridep \
  --micro-batch-size 1 --profile nsys
```

运行 2 个 case（baseline / offload）。

### 生成 XLSX 汇总（只收 `--profile none` 的成功运行）

```bash
node "./examples/scale-down/01-analyse/01-fine-grained-offload/02-offload-on-dense-or-expert-model/collect_mlp_offload_results.mjs"
# 默认收集最新批次；回看旧批次加 --run-time <benchmark_id>
# 只校验数据不写 XLSX：加 --dry-run
```

收集器固定读取 iteration 5–9 的 step time，所以 `--train-iters` 不能小于 10。

## 结果存储位置

训练结果根目录（可用 `--results-root <path>` 改）：

```text
result/01-analyse/01-fine-grained-offload/02-offload-on-dense-or-expert-model/
└── <qwen35_text_9b|qwen35_text_35b_a3b|deepseek>/<bf16|mxfp8>/<baseline|offload>/<run-time>/
    ├── config.json / config.yaml   # 最终 recipe 与 override（含 dtype、dispatcher）
    ├── command.txt                 # 实际 distributed 命令
    ├── train.log                   # 训练与 step time 日志
    ├── summary.json                # status=0 表示成功
    ├── gpu_memory/                 # nvidia-smi 显存采样
    ├── memory/snapshot.pickle      # profiling 时的 CUDA memory snapshot
    ├── profile/                    # nsys 输出：nsys-*.nsys-rep（4 个 rank 各一份）
    └── rank_logs/
```

XLSX 汇总输出：

```text
result/01-analyse/01-fine-grained-offload/02-offload-on-dense-or-expert-model/offload-comparison-<run-time>.xlsx
```

包含 `Summary`（每 run 吞吐均值/中位数/波动、相对 baseline 的比例）和 `Samples`（逐 iteration step time、tokens/s、raw result 目录）两个 sheet。

### 按模型汇总 TFlops

```bash
node "./examples/scale-down/01-analyse/01-fine-grained-offload/02-offload-on-dense-or-expert-model/analyse_mlp_offload_results.mjs" \
  --model qwen

node "./examples/scale-down/01-analyse/01-fine-grained-offload/02-offload-on-dense-or-expert-model/analyse_mlp_offload_results.mjs" \
  --model deepseek
```

分析脚本只读取成功且 `profile=none` 的运行。它按 model、dispatcher、MBS、dtype、dense/expert
配对 baseline 和 offload，分别选择各组最新的 baseline 与 offload，并对 iteration 5–9 的
`MODEL_TFLOP/s/GPU` 取平均。结果默认写入：

```text
result/01-analyse/01-fine-grained-offload/02-offload-on-dense-or-expert-model/offload-throughput-<qwen|deepseek>.xlsx
```

表格按 dispatcher、MBS、dtype、dense/expert 排序。性能下降幅度按
`(baseline - offload) / baseline` 计算。

## nsys / NVTX 分析

- `--profile nsys` 会自动设置 `profiling.use_nsys_profiler=true`、`profiling.nvtx_ranges=true`、`profiling.record_memory_history=true`，并用 `nsys profile -s none -t cuda,nvtx --capture-range=cudaProfilerApi` 启动训练，只采集 step 7–8，4 个 rank 都记录。
- 用 Nsight Systems GUI 打开 `<run-dir>/profile/*.nsys-rep`：
  - 在时间轴上把 NCCL allreduce / reduce-scatter / all-gather kernel 与 NVTX range（forward、backward、optimizer、offload D2H/H2D 等）对齐，即可回答"allreduce 是哪个操作产生的"（目的 3）；
  - 对比 baseline 与 offload 的时间轴，找出无法被计算 overlap 的 copy / 通信段（目的 1）；
  - 对比 qwen3.5 与 deepseek 的 kernel 间隙和通信占比，定位效率差距来源（目的 2）。
- profiling 运行不会被 XLSX 收集器纳入吞吐比较；显存细节看 `memory/snapshot.pickle` 和 `gpu_memory/`。

## 如何解读

- baseline 与 offload 必须在相同模型、dtype、MBS、GBS、dispatcher 下对比；不要把不同架构的绝对 tokens/s 当成 dense/MoE 的普遍排名。
- 先看 `summary.json` 和日志有无 NaN/Inf，再比 step time、tokens/s、显存。
- 显存峰值看各 run 目录的 `gpu_memory/`；XLSX 只汇总吞吐。
- 本矩阵只覆盖 MLP/MoE activation offload；attention offload 是 `plan.md` 中的后续 TODO。
