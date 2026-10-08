# GB200 32 卡 DLC：MLP fine-grained activation offload 实验

本文是 [`benchmark_mlp_offload.sh`](./benchmark_mlp_offload.sh) 的 32 卡运行说明。实验使用 8 个 DLC worker，每个 worker 暴露 4 张 GB200 GPU；每个 worker 必须执行同一个入口脚本。

## 实验边界

这个 benchmark 测量的是 fine-grained activation offload 对 dense MLP 和 MoE expert MLP 的显存、传输和吞吐影响。它使用仓库现有的 Qwen3.5 9B dense、27B dense、35B-A3B expert，以及 DeepSeek proxy recipe。它**不是** Qwen3.5-122B-A10B 的 32 卡训练配置；122B/A10B 需要单独的模型 recipe、并行度和 checkpoint 入口。

32 卡只改变 distributed launcher 的规模，recipe 中的模型并行度保持不变：

| 模型 | TP | PP | CP | EP | 数据并行组（32 卡时） |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen dense 9B/27B | 1 | 1 | 1 | 1 | DP=32 |
| Qwen expert 35B-A3B | 1 | 1 | 1 | 4 | DP=8，EP=4 |
| DeepSeek dense/expert proxy | 由 recipe 保持 | 由 recipe 保持 | 由 recipe 保持 | 由 recipe 保持 | 以最终 `config.yaml` 为准 |

固定条件：sequence length 4096、BF16 优先、FSDP1、mock data、关闭 activation recompute、关闭 layer-level CPU activation offload、关闭 optimizer CPU offload、关闭 CUDA graph。入口脚本会强制设置 `model.cuda_graph_impl=none`，因此 baseline 与 offload case 不会混入 graph capture 的影响。

## 测试矩阵

每个模型类型包含以下三个 case：

| case | `fine_grained_activation_offloading` | `offload_modules` |
| --- | --- | --- |
| `baseline` | `false` | `null` |
| `offload-mlp` | `true` | dense：`[mlp_norm,mlp_act]`；MoE：`[mlp_norm,expert_fc1,moe_act]` |
| `offload-attn-mlp` | `true` | 在对应 MLP 列表后加入 attention offload 模块 |

Qwen expert case 会分别运行 `alltoall` 和 `hybridep`；dense case 使用 recipe 的默认 dispatcher。默认 `benchmark_mlp_offload.sh` 矩阵在一个 dtype 下包含 36 个运行（dense 12 个、expert 24 个）。32 卡首次验证建议先跑 27 个 BF16 运行：MBS `1,2,4`；MBS=8 作为显存和吞吐都通过后的扩展项。

`GLOBAL_BATCH_SIZE` 必须同时满足所有 MBS 和数据并行组的整除关系。本文使用 256：

```text
GBS=256，MBS=1/2/4/8
dense DP=32：每卡 batch = 8
Qwen expert DP=8：每卡 batch = 32
```

Qwen expert 的 learned routing 在大 MBS 下可能显著增加 token imbalance；如果 MBS=4 或 8 OOM，应先保留 MBS=1/2 的结果，不要把失败样本当作 offload 性能结果。

## DLC 前置条件

### 1. 共享文件系统和容器

8 个 DLC worker 必须看到相同的仓库路径、`RESULTS_ROOT`、`HF_HOME`、`NEMO_HOME` 和 `UV_CACHE_DIR`。不要把结果写到 worker 私有磁盘，否则 rank 0 无法收集其他节点日志。

容器内先完成依赖同步，并确认每个 worker 有 4 张 GPU：

```bash
cd /shared/megatron-bridge
uv sync
nvidia-smi -L | wc -l                 # 应为 4
command -v numarun                    # 多机默认需要
```

可选地在每个 DLC worker 运行节点诊断：

```bash
bash examples/gb200/env-info/check_dlc_node_environment.sh
```

### 2. 统一环境变量

以下变量需要由 DLC job 的公共环境注入。`RUN_TIME` 必须在提交任务前生成一次，然后传给全部 8 个 worker；不能让每个 worker 各自执行 `date`。

```bash
export EXPERIMENT_DIR=/shared/megatron-bridge/examples/scale-down/01-analyse/01-fine-grained-offload/02-offload-on-dense-or-expert-model
export RESULTS_ROOT=/shared/megatron-bridge/results/scale-down/01-fine-grained-offload/02-offload-on-dense-or-expert-model
export RUN_TIME="20261005-32gpu-001"  # DLC 提交端生成一次，并传给全部 worker
export MASTER_ADDR="rank-0-worker-hostname-or-ip"  # 替换为实际可达地址
export MASTER_PORT=29501
export NUMARUN=numarun
```

实际提交脚本使用的 DLC rank 变量名称可能不同。启动入口前必须把它映射为节点 rank `0..7`：

```bash
export NODE_RANK="${DLC_NODE_RANK}"  # 替换为 DLC 实际提供的节点 rank 变量
export RANK="${NODE_RANK}"
export DLC_MASTER_ADDR="${MASTER_ADDR}"
```

`run_pretrain_fsdp1.sh` 会按 `--gpu 32` 计算 `NNODES=8`、`GPUS_PER_NODE=4`，并将 `RANK` 作为 `node_rank` 传给 `torchrun`。若容器没有 `numarun`，在所有 worker 上设置 `NUMARUN=`，脚本会直接调用带 `--nnodes=8` 和 `--node_rank` 的 `torch.distributed.run`。

## 运行步骤

### 1. 先做不启动训练的矩阵检查

在任意一个 worker 上执行，确认组合数量、recipe 和 GPU 规模正确：

```bash
cd /shared/megatron-bridge
bash "${EXPERIMENT_DIR}/benchmark_mlp_offload.sh" \
  --gpu 32 \
  --model qwen \
  --dense-model 9b \
  --scope all \
  --dtype bf16 \
  --micro-batch-sizes 1,2,4 \
  --global-batch-size 256 \
  --train-iters 10 \
  --results-root "${RESULTS_ROOT}" \
  --dry-run
```

`--dry-run` 不会初始化 NCCL，也不会生成训练结果；正式运行时必须让 8 个 DLC worker 同时执行同一命令。

### 2. BF16 吞吐矩阵

通过 DLC 的多 worker launcher 在每个 worker 执行：

```bash
cd /shared/megatron-bridge
bash "${EXPERIMENT_DIR}/benchmark_mlp_offload.sh" \
  --gpu 32 \
  --model qwen \
  --dense-model 9b \
  --scope all \
  --dtype bf16 \
  --micro-batch-sizes 1,2,4 \
  --global-batch-size 256 \
  --train-iters 10 \
  --profile none \
  --results-root "${RESULTS_ROOT}"
```

如需比较 27B dense，把 `--dense-model 9b` 改成 `--dense-model 27b`。如需只测 MoE HybridEP，使用更小的代表性矩阵：

```bash
bash "${EXPERIMENT_DIR}/benchmark_mlp_offload.sh" \
  --gpu 32 --model qwen --scope expert --dispatcher hybridep \
  --dtype bf16 --micro-batch-sizes 1,2,4 \
  --global-batch-size 256 --train-iters 10 --profile none \
  --results-root "${RESULTS_ROOT}"
```

### 3. MXFP8 对照矩阵

只有 BF16 的 baseline/offload 结果和显存都稳定后再运行 MXFP8，保持相同的 GBS、MBS、case、dispatcher 和结果目录规则：

```bash
bash "${EXPERIMENT_DIR}/benchmark_mlp_offload.sh" \
  --gpu 32 --model qwen --scope all --dense-model 9b \
  --dtype mxfp8 --micro-batch-sizes 1,2,4 \
  --global-batch-size 256 --train-iters 10 --profile none \
  --results-root "${RESULTS_ROOT}"
```

### 4. Nsight / PyTorch 代表性 profile

不要对完整 27/36-run 矩阵开启 profile。每个模型只选择同一 MBS、同一 dispatcher 下的三个 case：

```bash
bash "${EXPERIMENT_DIR}/benchmark_mlp_offload.sh" \
  --gpu 32 --model qwen --scope expert --dispatcher hybridep \
  --dtype bf16 --case baseline --micro-batch-size 1 \
  --global-batch-size 256 --train-iters 10 --profile nsys \
  --results-root "${RESULTS_ROOT}"

bash "${EXPERIMENT_DIR}/benchmark_mlp_offload.sh" \
  --gpu 32 --model qwen --scope expert --dispatcher hybridep \
  --dtype bf16 --case offload-mlp --micro-batch-size 1 \
  --global-batch-size 256 --train-iters 10 --profile nsys \
  --results-root "${RESULTS_ROOT}"
```

脚本在 profile 模式采集 step 7–8 的 Nsight 或 PyTorch trace，并记录 memory snapshot。Nsight 需要容器内提供 `nsys`；如果只需要显存和吞吐，保持 `--profile none`。

## 结果和汇总

结果根目录由 `RESULTS_ROOT` 控制，典型路径为：

```text
${RESULTS_ROOT}/
├── qwen35_text_9b/gpu_32-dtype_bf16-mbs_1-gbs_256-baseline/${RUN_TIME}/
├── qwen35_text_9b/gpu_32-dtype_bf16-mbs_1-gbs_256-offload-mlp/${RUN_TIME}/
└── qwen35_text_35b_a3b/
    └── gpu_32-dtype_bf16-mbs_1-gbs_256-dispatcher_hybridep-offload-mlp/${RUN_TIME}/
```

每个成功运行至少应包含 `config.json`、`config.yaml`、`command.txt`、`environment.json`、`train.log`、`summary.json`、`gpu_memory/`；profile 运行还应包含 `memory/` 和 `profile/`。非零退出的 run 保留日志，但不得参与性能汇总。

训练完成后，在共享仓库中执行：

```bash
uv run python "${EXPERIMENT_DIR}/analyse_mlp_offload_results.py" \
  --model qwen \
  --results-root "${RESULTS_ROOT}" \
  --output "${RESULTS_ROOT}/offload-throughput-qwen-32gpu.xlsx" \
  --csv-output "${RESULTS_ROOT}/offload-throughput-qwen-32gpu.csv"
```

分析脚本只读取 `profile=none` 且 `summary.json.status=0` 的运行，并要求每个运行至少有 10 个 GPU utilization 样本。`Summary` 对 baseline、offload-mlp 和 offload-attn-mlp 配对，`Samples` 保存用于平均的最后 4 个样本。

## 验收和故障定位

每组结果按以下顺序验收：

1. `summary.json.status == 0`，所有 rank 都正常退出。
2. `train.log` 没有 NaN/Inf、NCCL timeout、CUDA OOM 或 Hydra 配置错误。
3. baseline 与 offload 的 loss 走势和梯度健康；不要用失败或不完整的 profile 比较吞吐。
4. 读取 `gpu_memory/` 对比峰值 HBM，读取 `run_info.txt`、Nsight trace 和 memory snapshot 判断 D2H/H2D copy 是否重叠。
5. MoE 运行额外检查 expert token count、router imbalance 和 dispatcher；HybridEP 需要确认 `NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN` 与 GB200 拓扑一致。

常见问题：

| 现象 | 检查 | 处理 |
| --- | --- | --- |
| 其他节点卡在 rendezvous | `MASTER_ADDR` 是否可达、`RANK` 是否为 0–7、8 个 worker 是否都启动 | 修正 DLC rank 映射和端口，保证所有 worker 使用同一 `RUN_TIME` |
| `numarun` 找不到 | `command -v numarun` | 所有 worker 设置 `NUMARUN=`，或安装/挂载容器中的 NUMA launcher |
| 结果目录被覆盖 | 多节点是否共享 `RESULTS_ROOT`，`RUN_TIME` 是否相同 | 使用共享路径，并在提交端固定 `RUN_TIME` |
| MBS=4/8 OOM | `gpu_memory/`、expert token imbalance | 先降到 MBS=1/2；保留失败目录，不把它纳入汇总 |
| offload case 被拒绝 | `config.yaml` 中是否同时开启 recompute、layer CPU offload 或 CUDA graph | 本脚本只允许 fine-grained offload；不要叠加互斥的 activation offload 机制 |

最终报告至少应包含每个模型和 dispatcher 的 baseline/offload 峰值显存、step time、tokens/s 或 model TFLOPS、相对性能下降、D2H/H2D bytes 和失败原因。
