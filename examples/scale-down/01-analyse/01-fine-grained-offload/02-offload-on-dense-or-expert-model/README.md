# Offload ON/OFF 性能对比：Qwen Dense / Qwen MoE / DeepSeek-V3

## 这个实验测什么

对比 **3 个模型** 在 **开启 / 关闭 fine-grained activation offload** 时的性能差异（step time、tokens/s、显存峰值）：

| 模型 | 选择参数 | scope | offload 模块 |
| --- | --- | --- | --- |
| Qwen Dense (9B/27B) | `--model qwen --scope dense` | dense MLP | `[mlp_norm,mlp_act]` |
| Qwen MoE (35B-A3B, 16 层, 64 experts) | `--model qwen --scope expert` | expert MLP | `[mlp_norm,expert_fc1,moe_act]` |
| DeepSeek-V3 (8 层, 64 experts, MoE-only) | `--model deepseek` | expert MLP | `[mlp_norm,expert_fc1,moe_act]` |

扫描维度：

- **MBS**：`1,2,4,8`（必须整除 GBS）
- **精度**：`--dtype bf16`（默认）或 `--dtype mxfp8`
- **dispatcher**（仅 MoE 模型）：`alltoall` 和 `hybridep` 各跑一遍
- **case**：`baseline`（offload 关）与 `offload`（offload 开）

固定条件：4 GPU FSDP1、seq len 4096、GBS 32、train-iters 10、关闭 recompute / CUDA graph / optimizer offload / checkpoint，`--profile none`。

## 文件说明

- `benchmark_mlp_offload.sh` — 矩阵入口，展开所有组合并逐项调用训练脚本
- `run_pretrain_fsdp1.sh` — 软链接到 `../01-mlp-scope-pretrain/`，实际执行 4-GPU FSDP1 训练
- `collect_mlp_offload_results.mjs` — 扫描结果目录，生成 XLSX 吞吐对比表

## 运行指令

前置：仓库根目录 `uv sync`；4 张 GPU 可用（GB200 recipe）；生成 XLSX 需要 Node.js 和 `@oai/artifact-tool` 模块。

```bash
EXPERIMENT_DIR=examples/scale-down/01-analyse/01-fine-grained-offload/02-offload-on-dense-or-expert-model
```

### 1. 先 dry-run 查看矩阵（不启动训练）

```bash
bash "$EXPERIMENT_DIR/benchmark_mlp_offload.sh" --dry-run --micro-batch-sizes 1,2,4,8
```

### 2. 三个模型 × bf16 × MBS 1,2,4,8

```bash
# Qwen Dense 9B：每 MBS 2 个运行（baseline/offload），共 8 个
bash "$EXPERIMENT_DIR/benchmark_mlp_offload.sh" \
  --model qwen --scope dense --dense-model 9b \
  --dtype bf16 --micro-batch-sizes 1,2,4,8

# Qwen MoE：每 MBS 4 个运行（2 dispatcher × 2 case），共 16 个
bash "$EXPERIMENT_DIR/benchmark_mlp_offload.sh" \
  --model qwen --scope expert \
  --dtype bf16 --micro-batch-sizes 1,2,4,8

# DeepSeek-V3：每 MBS 4 个运行，共 16 个
bash "$EXPERIMENT_DIR/benchmark_mlp_offload.sh" \
  --model deepseek \
  --dtype bf16 --micro-batch-sizes 1,2,4,8
```

### 3. 三个模型 × mxfp8 × MBS 1,2,4,8

只把 `--dtype` 换成 `mxfp8`，其余参数保持一致（便于和 bf16 对照）：

```bash
bash "$EXPERIMENT_DIR/benchmark_mlp_offload.sh" \
  --model qwen --scope dense --dense-model 9b \
  --dtype mxfp8 --micro-batch-sizes 1,2,4,8

bash "$EXPERIMENT_DIR/benchmark_mlp_offload.sh" \
  --model qwen --scope expert \
  --dtype mxfp8 --micro-batch-sizes 1,2,4,8

bash "$EXPERIMENT_DIR/benchmark_mlp_offload.sh" \
  --model deepseek \
  --dtype mxfp8 --micro-batch-sizes 1,2,4,8
```

> 不带 `--scope` 时 `--model qwen` 会同时跑 dense + expert。`--scope dense` 只对 Qwen 有效（DeepSeek 是 MoE-only）。每次 benchmark 启动时会自动生成批次标识，不需要额外传参。

### 4. 生成 XLSX 汇总（默认读取最新批次）

```bash
node "$EXPERIMENT_DIR/collect_mlp_offload_results.mjs"
# 只校验数据不写 XLSX：加 --dry-run
```

benchmark 完成后会打印自动生成的批次标识和上述收集命令。只有在需要回看旧批次时，才使用 collector 自身的可选 `--run-time <id>` 参数。

收集器只接受 `profile=none` 且 `summary.json` 成功的运行，固定读取 iteration 5–9 的 step time（所以 `--train-iters` 不能小于 10）。

## 结果存储位置

训练结果根目录（可用 `--results-root <path>` 改）：

```text
results/01-analyse/01-offload-on-dense-and-expert-model/
└── <model>/<dtype>/<run-name__参数标签>/<run-time>/
    ├── config.json / config.yaml   # 最终 recipe 与 override
    ├── command.txt                 # 实际 distributed 命令
    ├── train.log                   # 训练与 step time 日志
    ├── summary.json                # status=0 表示成功
    ├── gpu_memory/                 # nvidia-smi 显存采样
    └── rank_logs/
```

XLSX 汇总输出：

```text
results/01-analyse/01-offload-on-dense-and-expert-model/offload-comparison-<run-time>.xlsx
```

包含 `Summary`（每 run 的吞吐均值/中位数/波动、相对 baseline 的比例）和 `Samples`（逐 iteration step time、tokens/s、raw result 目录）两个 sheet。

## 如何解读

- baseline 与 offload 必须在相同模型、dtype、MBS、GBS、dispatcher 下对比；不要把不同架构的绝对 tokens/s 当成 dense/MoE 的普遍排名。
- 先看 `summary.json` 和日志有无 NaN/Inf，再比 step time、tokens/s、显存。
- 显存峰值看各 run 目录的 `gpu_memory/`；XLSX 只汇总吞吐。
- 本矩阵只覆盖 MLP/MoE activation offload；attention offload 是 `plan.md` 中的后续 TODO。
- 需要定位 copy/同步开销时，对代表性 case 单独用 `--profile nsys` 或 `--profile torch` 重跑（profiling 运行不进 XLSX）。
