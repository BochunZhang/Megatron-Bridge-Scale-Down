# Megatron versus DeepSpeed offload experiment

This directory is the Megatron counterpart to
`../02-deepspeed-offload/pretrain_experiment.sh`. It keeps the shared
workload contract fixed and varies Megatron's activation and optimizer memory
mechanisms independently.

## Shared contract

- Qwen3.5-9B dense and Qwen3.5-35B-A3B MoE
- 4 GPUs, sequence length 4096, BF16, seed 42
- per-GPU batch budget 16, global batch size 64, micro batches 1/2/4/8
- AdamW: lr 0.001, betas 0.9/0.999, eps 1e-8, weight decay 0.01
- MoE uses 64 experts, expert parallel size 4, HybridEP by default
- CUDA graphs disabled for every case

The model contract is materialized by `experiment_manifest.py`. This is
important because the DeepSpeed runner loads `Qwen/Qwen3.5-*`, while the
Megatron recipes use the corresponding `*-Base` text configuration.

## Case mapping

| Case | Megatron configuration | Comparison meaning |
|---|---|---|
| `baseline` | no recompute/offload | Megatron baseline |
| `recompute` | selective `layernorm,mlp` | activation recompute control |
| `offload` | fine-grained offload `mlp_norm` | module-level activation offload |
| `recompute_offload` | selective recompute plus `mlp_norm` offload | fine-grained Megatron combination |
| `optimizer_cpu_025` ... `100` | above combination plus CPU optimizer fraction | memory/throughput trade-off |

DeepSpeed `param_cpu` has no direct Megatron equivalent in this runner. Its
`act_cpu` path is also not an implementation-equivalent layer checkpoint
offload: Megatron uses fine-grained module offload plus selective recompute.
The result summary must therefore compare trends, not claim kernel-level
equivalence.

## Usage

Print the full matrix without launching jobs:

```bash
./pretrain_experiment.sh --dry-run
```

Run a small smoke matrix:

```bash
./pretrain_experiment.sh \
  --models "dense" \
  --cases "baseline recompute_offload optimizer_cpu_050" \
  --micro-batch-sizes "1" \
  --train-iters 10 \
  --warmup-steps 2
```

Results are written below
`results/01-analyse/02-megatron-vs-deepspeed/`, with one `config.yaml`,
`config.json`, `summary.json`, log, memory trace, and profiler output per run.
Run `summarize.py` to regenerate `summary.csv`.
