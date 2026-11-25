#!/usr/bin/env bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../.." && pwd)"
case_name="$1"
parameter_tier="$2"
activation_tier="$3"
shift 3
nvme_path="${NVME_PATH:-/tmp}"
output_dir="${OUTPUT_DIR:-${repo_root}/results/02-layer-by-layer}"
train_iters=10
profile=nsys
dry_run=false
print_result_dir=false

fail() { printf '%s\n' "$*" >&2; exit 2; }

while (( $# )); do
    case "$1" in
        --nvme|--nvme-path|--output-dir|--train-iters|--profile)
            [[ -n "${2:-}" && "$2" != --* ]] || fail "Missing value for $1"
            case "$1" in
                --nvme|--nvme-path) nvme_path="$2";;
                --output-dir) output_dir="$2";;
                --profile)
                    [[ "$2" == nsys || "$2" == none ]] || fail "--profile must be nsys or none"
                    profile="$2";;
                --train-iters)
                    [[ "$2" =~ ^[1-9][0-9]*$ ]] || fail "--train-iters must be a positive integer"
                    train_iters="$2";;
            esac
            shift 2;;
        --dry-run) dry_run=true; shift;;
        --print-result-dir) print_result_dir=true; shift;;
        -h|--help)
            cat <<'HELP'
Run a 32-layer Qwen3.5-27B offload case on one four-GPU node.

  --nvme DIR            Local NVMe scratch directory (default: /tmp, or set NVME_PATH)
  --output-dir DIR      Result root (default: results/02-layer-by-layer)
  --train-iters N       Optimizer steps (default: 10)
  --profile nsys|none   Use Nsight Systems 2026.6.1 (default: nsys)
  --dry-run             Show the training command
  --print-result-dir    Print the node0 result directory
HELP
            exit 0;;
        *) fail "Unknown option $1";;
    esac
done

[[ "${output_dir}" == /* ]] || output_dir="${PWD}/${output_dir}"
model_name="model_27b-layers_32-cg_off-mbs_1-seq_4096-tp_1-cp_1-gbs_16-gpus_4"
model_name+="-param_${parameter_tier}-act_${activation_tier}-iters_${train_iters}-profile_${profile}"
result_dir="${output_dir%/}/01-baseline/${case_name}/${model_name}/node0"
if [[ "${print_result_dir}" == true ]]; then
    printf '%s\n' "${result_dir}"
    exit 0
fi

args=(--single-case-layout --results-dir "${result_dir}"
    --layers 32 --seq-length 4096 --micro-batch-size 1 --microbatches 4
    --steps "${train_iters}" --parameter-tier "${parameter_tier}" --activation-tier "${activation_tier}"
    --profile "${profile}")
[[ "${dry_run}" == true ]] && args+=(--dry-run)

exec bash "${script_dir}/../run_sweep.sh" --nvme-dir "${nvme_path}" "${args[@]}"
