#!/usr/bin/env bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../.." && pwd)"
case_name="$1"
parameter_tier="$2"
activation_tier="$3"
optimizer_tier="$4"
shift 4
nvme_path="${NVME_PATH:-}"
output_dir="${OUTPUT_DIR:-${repo_root}/results/02-layer-by-layer}"
layers=8
train_iters=10
profile=nsys
cpu_optimizer=torch
overlap_cpu_optimizer=false
cpu_threads=""
dry_run=false
print_result_dir=false

fail() { printf '%s\n' "$*" >&2; exit 2; }

while (( $# )); do
    case "$1" in
        --nvme|--nvme-path|--output-dir|--train-iters|--profile|--cpu-optimizer|--cpu-threads)
            [[ -n "${2:-}" && "$2" != --* ]] || fail "Missing value for $1"
            case "$1" in
                --nvme|--nvme-path) nvme_path="$2";;
                --output-dir) output_dir="$2";;
                --cpu-optimizer)
                    [[ "$2" == grace_adam || "$2" == torch ]] || fail "--cpu-optimizer must be grace_adam or torch"
                    cpu_optimizer="$2";;
                --cpu-threads)
                    [[ "$2" =~ ^[1-9][0-9]*$ ]] || fail "--cpu-threads must be a positive integer"
                    cpu_threads="$2";;
                --profile)
                    [[ "$2" == nsys || "$2" == none ]] || fail "--profile must be nsys or none"
                    profile="$2";;
                --train-iters)
                    [[ "$2" =~ ^[1-9][0-9]*$ ]] || fail "--train-iters must be a positive integer"
                    train_iters="$2";;
            esac
            shift 2;;
        --overlap-cpu-optimizer) overlap_cpu_optimizer=true; shift;;
        --dry-run) dry_run=true; shift;;
        --print-result-dir) print_result_dir=true; shift;;
        -h|--help)
            cat <<'HELP'
Run an 8-layer Qwen3.5-27B full-attention case on one four-GPU node.

  --nvme DIR            Local scratch directory when an NVMe tier is selected
  --output-dir DIR      Result root (default: results/02-layer-by-layer)
  --train-iters N       Optimizer steps (default: 10)
  --profile nsys|none   Use Nsight Systems 2026.6.1 (default: nsys)
  --cpu-optimizer torch|grace_adam   Optimizer backend when a CPU/NVMe optimizer tier is selected
  --overlap-cpu-optimizer  Start GraceAdam updates as CPU main gradients become ready
  --cpu-threads N       CPU optimizer and PyTorch intra-op threads per rank (default: 32)
  --dry-run             Show the training command
  --print-result-dir    Print the node0 result directory
HELP
            exit 0;;
        *) fail "Unknown option $1";;
    esac
done

if [[ "${overlap_cpu_optimizer}" == true ]]; then
    [[ "${parameter_tier}" == cpu && "${optimizer_tier}" == cpu && "${cpu_optimizer}" == grace_adam ]] ||
        fail "--overlap-cpu-optimizer requires CPU weights and the CPU optimizer tier with GraceAdam"
fi

[[ "${output_dir}" == /* ]] || output_dir="${PWD}/${output_dir}"
model_name="model_27b-lbl-schedule-layers_${layers}-attn_full-cg_off-mbs_1-seq_4096-tp_1-cp_1-gbs_16-gpus_4"
model_name+="-param_${parameter_tier}-act_${activation_tier}-opt_${optimizer_tier}-iters_${train_iters}-profile_${profile}"
result_dir="${output_dir%/}/01-baseline/${case_name}/${model_name}/node0"
if [[ "${print_result_dir}" == true ]]; then
    printf '%s\n' "${result_dir}"
    exit 0
fi

args=(--single-case-layout --results-dir "${result_dir}"
    --layers "${layers}" --seq-length 4096 --micro-batch-size 1 --microbatches 4
    --steps "${train_iters}" --parameter-tier "${parameter_tier}" --activation-tier "${activation_tier}"
    --optimizer-tier "${optimizer_tier}"
    --cpu-optimizer "${cpu_optimizer}"
    --profile "${profile}")
[[ "${overlap_cpu_optimizer}" == true ]] && args+=(--overlap-cpu-optimizer)
[[ "${dry_run}" == true ]] && args+=(--dry-run)
[[ -z "${nvme_path}" ]] || args+=(--nvme-dir "${nvme_path}")
[[ -z "${cpu_threads}" ]] || args+=(--cpu-threads "${cpu_threads}")

exec bash "${script_dir}/../run_pretrain_layer_by_layer.sh" "${args[@]}"
