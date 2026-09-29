# Megatron versus DeepSpeed Offload Experiment

This directory defines the Megatron side of the offload and recompute study.
The goal is to hold the model and training contract constant, then compare
memory usage, iteration time, and throughput as Megatron changes activation
recompute, fine-grained activation offload, and fractional optimizer CPU
offload.

The experiment entry point is `pretrain_experiment.sh`. It launches the
four-GPU Megatron runner, writes one result directory per case, and regenerates
the aggregate `summary.csv` at the end. `experiment_manifest.py` records the
shared HF model contract used to compare this matrix with the DeepSpeed
matrix.

## Prerequisites

- Run from the repository root or use absolute paths.
- Four visible NVIDIA GPUs are required by the default launcher.
- The repository's `uv` environment must be installed; the scripts invoke
  `uv run --no-sync`.
- The model configuration and tokenizer caches must be available locally, or
  the environment must be able to access Hugging Face when the manifest and
  recipe are first resolved.

## Shared Experiment Contract

The default matrix uses:

- Qwen3.5-9B dense and Qwen3.5-35B-A3B MoE;
- four GPUs, sequence length 4096, BF16, and seed 42;
- a fixed per-GPU batch budget of 16, giving global batch size 64;
- micro-batch sizes 1, 2, 4, and 8;
- AdamW with learning rate 0.001, betas `(0.9, 0.999)`, epsilon `1e-8`, and
  weight decay `0.01`;
- 64 MoE experts, expert parallel size 4, and HybridEP for the MoE case;
- CUDA graphs disabled.

The manifest is generated from the HF configurations and preserves the
architecture fields needed for the cross-framework comparison. The Megatron
recipes use the corresponding text configuration rather than loading the HF
weights, so the manifest is the source of truth for the intended shape.

## Cases

| Case | Megatron settings | Purpose |
| --- | --- | --- |
| `baseline` | No recompute or offload | Megatron reference |
| `recompute` | Selective recompute of `[layernorm,mlp]` | Recompute memory/compute trade-off |
| `offload` | Fine-grained offload of `[mlp_norm]` | Activation offload trade-off |
| `recompute_offload` | Selective recompute plus `[mlp_norm]` offload | Combined fine-grained strategy |
| `optimizer_cpu_025` | Combined strategy plus CPU optimizer fraction `0.25` | Partial optimizer offload |
| `optimizer_cpu_050` | Combined strategy plus CPU optimizer fraction `0.50` | Partial optimizer offload |
| `optimizer_cpu_075` | Combined strategy plus CPU optimizer fraction `0.75` | Partial optimizer offload |
| `optimizer_cpu_100` | Combined strategy plus CPU optimizer fraction `1.00` | Full optimizer offload |

For optimizer cases, the intended settings are
`optimizer.optimizer_cpu_offload=true`, the case-specific
`optimizer.optimizer_offload_fraction`, and
`optimizer.overlap_cpu_optimizer_d2h_h2d=true`.

These cases are comparable by memory and performance trends, not by claiming
identical kernels. DeepSpeed `param_cpu` has no direct equivalent in this
Megatron runner. DeepSpeed `act_cpu` is also not implementation-equivalent to
Megatron fine-grained module offload plus selective recompute.

## Running the Matrix

From the repository root, print the planned matrix without launching jobs:

```bash
./examples/scale-down/01-analyse/02-megatron-vs-deepspeed/pretrain_experiment.sh \
  --dry-run
```

Run the default dense and MoE matrix:

```bash
./examples/scale-down/01-analyse/02-megatron-vs-deepspeed/pretrain_experiment.sh
```

Run a small smoke matrix before a longer sweep:

```bash
./examples/scale-down/01-analyse/02-megatron-vs-deepspeed/pretrain_experiment.sh \
  --models "dense" \
  --cases "baseline recompute_offload optimizer_cpu_050" \
  --micro-batch-sizes "1" \
  --train-iters 10 \
  --warmup-steps 2 \
  --repeats 1
```

The smoke command checks that the selected recipes and launcher wiring work;
ten iterations are not sufficient for a statistically stable throughput
benchmark.

Useful matrix options include:

```text
--models dense\ moe
--cases baseline\ recompute\ offload\ recompute_offload\ optimizer_cpu_050
--micro-batch-sizes "1 2 4 8"
--precision bf16|fp8mx
--train-iters N
--warmup-steps N
--repeats N
--per-gpu-batch-size N
--profile none|nsys|torch
--results-root PATH
--run-time YYYYMMDD-HHMMSS
```

For example, write a short dense sweep to an explicit result directory:

```bash
RESULTS_ROOT=/tmp/megatron-vs-deepspeed \
./examples/scale-down/01-analyse/02-megatron-vs-deepspeed/pretrain_experiment.sh \
  --models dense \
  --cases "baseline recompute_offload" \
  --micro-batch-sizes "1 2" \
  --train-iters 20 \
  --warmup-steps 4 \
  --results-root /tmp/megatron-vs-deepspeed
```

### Running One Megatron Job Directly

The lower-level runner supports the memory measurement options directly. This
example enables selective recompute, fine-grained offload, and the memory
metrics after two warmup iterations:

```bash
RESULTS_ROOT=/tmp/megatron-one-run \
./examples/scale-down/01-analyse/01-fine-grained-offload/01-mlp-scope-pretrain/run_pretrain_fsdp1.sh \
  --model qwen35_text_9b \
  --recipe qwen35_text_9b_pretrain_4gpu_gb200_bf16_fsdp1_config \
  --precision bf16 \
  --run-name dense-recompute-offload \
  --recompute-granularity selective \
  --recompute-modules '[layernorm,mlp]' \
  --fine-grained-offload true \
  --offload-modules '[mlp_norm]' \
  --train-iters 10 \
  --global-batch-size 64 \
  --micro-batch-size 1 \
  --record-memory-usage true \
  --memory-usage-start-step 2
```

The lower-level script currently exposes the model, recompute, fine-grained
offload, profiling, and memory-usage options shown above. The higher-level
matrix owns the optimizer-offload case selection and passes its intended
configuration through the experiment runner.

## Memory and Profiling Outputs

The matrix enables:

```text
profiling.record_memory_usage=true
profiling.memory_usage_start_step=<warmup steps>
```

At each logging interval, the training log records raw byte values:

```text
cuda_memory_allocated_bytes
cuda_peak_memory_allocated_bytes
host_memory_allocated_bytes
host_peak_memory_allocated_bytes
```

The `host_memory_*` values are CUDA pinned-host allocator statistics, matching
the DeepSpeed measurement. They are not process RSS and should not be
interpreted as total system CPU memory. The existing `nvidia-smi` monitor
continues to provide device-level GPU memory traces under `gpu_memory/`.

When `--profile nsys` or `--profile torch` is selected, the runner also writes
the configured profiler output and CUDA allocator memory snapshots under
`profile/` and `memory/` as applicable. Peak-counter reset affects the
benchmark metrics only; it does not truncate allocator history snapshots.

## Results and Summaries

The default result root is:

```text
results/01-analyse/02-megatron-vs-deepspeed/
```

Each run has its own timestamped directory containing the launcher metadata,
`config.yaml`, `config.json`, `summary.json`, `train.log`, GPU memory traces,
and profiler or memory artifacts when enabled. Regenerate the aggregate CSV
with:

```bash
uv run --no-sync python \
  examples/scale-down/01-analyse/02-megatron-vs-deepspeed/summarize.py \
  results/01-analyse/02-megatron-vs-deepspeed
```

Interpret results using the post-warmup window: compare peak GPU allocator
usage and pinned-host usage together with steady-state iteration time and
tokens/s. Treat an OOM as a valid boundary observation, and do not compare a
single smoke run as a final throughput result.
