# Offload ON/OFF 性能对比：Qwen Dense / Qwen MoE / DeepSeek-V3

## 测试目的

1. **量化 dense / MoE 开启 offload 后的性能下降幅度。** 此前测试发现 dense 模型开启 fine-grained activation offload 后性能下降很大，本测试要复现该现象，并通过 nsys 找到性能瓶颈的来源——例如哪里的 D2H/H2D copy 或通信无法被计算 overlap。
2. **分析 qwen3.5 训练效率偏低的原因。** 测试发现 qwen3.5 dense / expert 的训练效率仅约 500 TFlops / 100 TFlops，而此前测试 deepseek 架构时能跑到 1000 TFlops / 700 TFlops。因此增加 deepseek 架构的对照测试，用于分析 qwen3.5 效率低在哪里。
3. **定位 allreduce 通信的来源。** profiling 时观察到很多 allreduce 通信；`--profile nsys` 会自动开启 NVTX 标记（`profiling.nvtx_ranges=true`，nsys 以 `-t cuda,nvtx` 采集），通过时间轴上 NVTX range 与 NCCL kernel 的对应关系，确认每个 allreduce 是由哪个操作产生的（FSDP grad reduce / param gather / MoE dispatch-combine 等）。

## 测试内容

对比 **3 个模型** 在 **开启 / 关闭 fine-grained activation offload** 时的性能差异（step time、tokens/s、显存峰值）：

| 模型 | 选择参数 | scope | offload 模块 |
| --- | --- | --- | --- |
| Qwen Dense (9B/27B) | `--model qwen --scope dense` | dense MLP | `[mlp_norm,mlp_act]` |
| Qwen MoE (35B-A3B, 16 层, 64 experts) | `--model qwen --scope expert` | expert MLP | `[mlp_norm,expert_fc1,moe_act]` |
| DeepSeek-V3 (8 层, 64 experts, MoE-only) | `--model deepseek` | expert MLP | `[mlp_norm,expert_fc1,moe_act]` |

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

### 测试 2：deepseek-v3 效率单项测试（MBS 1,2 + nsys）

对照 qwen3.5 分析训练效率。按以下顺序执行 4 条指令（bf16 两个 dispatcher → mxfp8 两个 dispatcher），每条运行 4 个 case（baseline/offload × mbs 1,2）：

```bash
# 1. bf16 + alltoall
bash "./examples/scale-down/01-analyse/01-fine-grained-offload/02-offload-on-dense-or-expert-model/benchmark_mlp_offload.sh" \
  --model deepseek --dtype bf16 --dispatcher alltoall \
  --micro-batch-sizes 1,2 --profile nsys

# 2. bf16 + hybridep
bash "./examples/scale-down/01-analyse/01-fine-grained-offload/02-offload-on-dense-or-expert-model/benchmark_mlp_offload.sh" \
  --model deepseek --dtype bf16 --dispatcher hybridep \
  --micro-batch-sizes 1,2 --profile nsys

# 3. mxfp8 + alltoall
bash "./examples/scale-down/01-analyse/01-fine-grained-offload/02-offload-on-dense-or-expert-model/benchmark_mlp_offload.sh" \
  --model deepseek --dtype mxfp8 --dispatcher alltoall \
  --micro-batch-sizes 1,2 --profile nsys

# 4. mxfp8 + hybridep
bash "./examples/scale-down/01-analyse/01-fine-grained-offload/02-offload-on-dense-or-expert-model/benchmark_mlp_offload.sh" \
  --model deepseek --dtype mxfp8 --dispatcher hybridep \
  --micro-batch-sizes 1,2 --profile nsys
```

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
results/01-analyse/01-offload-on-dense-and-expert-model/
└── <model>/<bf16|mxfp8>/<run-name__参数标签>/<run-time>/
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
results/01-analyse/01-offload-on-dense-and-expert-model/offload-comparison-<run-time>.xlsx
```

包含 `Summary`（每 run 吞吐均值/中位数/波动、相对 baseline 的比例）和 `Samples`（逐 iteration step time、tokens/s、raw result 目录）两个 sheet。

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
