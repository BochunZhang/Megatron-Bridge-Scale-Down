#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
RUNNER="${SCRIPT_DIR}/../04-fsdp3-comm/run_pretrain_fsdp3.sh"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/01-analyse/05-tflops}"
RUN_DATE="${RUN_DATE:-$(date +%y%m%d-%H%M%S)}"
MODELS=(27b 2.4t)
FSDP_LEVELS=(1 2)
PRECISIONS=(bf16 mxfp8 nvfp4)
LAYOUTS=(dp4 cp4 tp4)
PROFILE=nsys
RUN_OPTIONS=()

usage() {
    cat <<'USAGE'
Usage: benchmark_tflops.sh [options]

Run the 4-GPU CUDA Graph training matrix for Qwen3.8-27B and Qwen3.8-2.4T.
Select Qwen3.5-27B with --model qwen35-27b. Use full-iteration graphs for
both 27B models and native local MoE partial graphs for 2.4T.
Each case runs 20 iterations with 4 full-attention layers, no MTP, GBS=32,
MBS=1, BF16 gradient communication and FP32 main gradients. MoE uses 64
experts and EP=4.

  --model 27b|qwen35-27b|2.4t   Defaults to 27b (Qwen3.8-27B) and 2.4t
  --fsdp 1|2                    Defaults to both FSDP levels
  --precision bf16|mxfp8|nvfp4  Defaults to all three precisions
  --parallel dp4|cp4|tp4        Defaults to all three layouts (4K/16K/16K)
  --profile nsys|none           Defaults to nsys, capturing step [7,8)
  --results-root DIR            Store this matrix under DIR/$RUN_DATE
  --unlock                      Restore automatic GPU clocks (TDP lock by default)
  --resume                      Reuse a RUN_DATE and skip completed cases
  --dry-run                     Print the case names and commands without launching
  -h, --help                    Show this help

For the four matrix selectors, pass one or more values separated by commas
or spaces, for example --precision bf16,nvfp4 or --precision bf16 nvfp4.
USAGE
}

die() { echo "$*" >&2; exit 2; }

while (( $# > 0 )); do
    case "$1" in
        --model|--fsdp|--precision|--parallel)
            option="$1"
            case "${option}" in
                --model) allowed="27b qwen35-27b 2.4t";;
                --fsdp) allowed="1 2";;
                --precision) allowed="bf16 mxfp8 nvfp4";;
                --parallel) allowed="dp4 cp4 tp4";;
            esac
            shift
            selected=()
            while (( $# > 0 )) && [[ "$1" != -* ]]; do
                [[ "$1" =~ ^[a-z0-9.-]+(,[a-z0-9.-]+)*$ ]] || die "Invalid value for ${option}: $1"
                IFS=' ' read -r -a values <<<"${1//,/ }"
                for value in "${values[@]}"; do
                    [[ " ${allowed} " == *" ${value} "* ]] || die "${option} accepts: ${allowed}"
                    if [[ " ${selected[*]:-} " != *" ${value} "* ]]; then
                        selected+=("${value}")
                    fi
                done
                shift
            done
            (( ${#selected[@]} > 0 )) || die "${option} requires a value"
            case "${option}" in
                --model) MODELS=("${selected[@]}");;
                --fsdp) FSDP_LEVELS=("${selected[@]}");;
                --precision) PRECISIONS=("${selected[@]}");;
                --parallel) LAYOUTS=("${selected[@]}");;
            esac
            ;;
        --profile|--results-root)
            [[ -n "${2:-}" && "$2" != -* ]] || die "$1 requires a value"
            if [[ "$1" == --profile ]]; then
                PROFILE="$2"
            else
                RESULTS_ROOT="$2"
            fi
            shift 2
            ;;
        --unlock|--resume|--dry-run) RUN_OPTIONS+=("$1"); shift;;
        -h|--help) usage; exit 0;;
        *) die "Unknown option: $1";;
    esac
done

[[ "${PROFILE}" == nsys || "${PROFILE}" == none ]] || die "--profile accepts nsys or none"
[[ "${RUN_DATE}" =~ ^[0-9]{6}-[0-9]{6}$ ]] || die "RUN_DATE must use YYMMDD-HHMMSS"

# Prefer the adjacent MCore checkout when it is available. The FSDP2 cases
# require its singleton gradient-bucket fix.
if [[ -z "${MEGATRON_LM_DIR:-}" && -d "${REPO_ROOT}/../megatron-lm/megatron/core" ]]; then
    export MEGATRON_LM_DIR="${REPO_ROOT}/../megatron-lm"
fi
common=(
    --gpus 4 --nproc-per-node 4 --nnodes 1 --node-rank 0
    --num-layers 4 --full --mtp-num-layers 0 --cuda-graph on --label-dtype-and-dp
    --gbs 32 --mbs 1 --train-iters 20 --profile "${PROFILE}" --profile-start 7 --profile-end 8
    --results-root "${RESULTS_ROOT}/${RUN_DATE}" --run-date "${RUN_DATE}" --stable-result-dir
    --override logger.log_throughput=true
    # Use the overlapped FSDP path for BF16 reductions in both sharding modes.
    # FSDP1 still reduces only on the final microbatch.
    --override ddp.overlap_grad_reduce=true --override comm_overlap.overlap_grad_reduce=true
    # These optional torch.dtype fields are unset in the base recipes.
    --override 'ddp.megatron_fsdp_main_grads_dtype={_target_:megatron.bridge.utils.activation_map.str_to_dtype,name:float32}'
    --override 'ddp.megatron_fsdp_grad_comm_dtype={_target_:megatron.bridge.utils.activation_map.str_to_dtype,name:bfloat16}'
)
total=$(( ${#MODELS[@]} * ${#FSDP_LEVELS[@]} * ${#PRECISIONS[@]} * ${#LAYOUTS[@]} ))
case_number=0
failed=()

for model in "${MODELS[@]}"; do
    case "${model}" in
        27b) model_args=(--model qwen38-27b --ep 1); model_label=qwen3.8-27b;;
        qwen35-27b) model_args=(--model 27b --ep 1); model_label=qwen3.5-27b;;
        2.4t) model_args=(--model 2.4t --num-experts 64 --ep 4 --dispatcher hybridep); model_label=qwen3.8-2.4t;;
    esac
    for layout in "${LAYOUTS[@]}"; do
        case "${layout}" in
            dp4) parallel_args=(--tp 1 --cp 1 --seq 4096);;
            cp4) parallel_args=(--tp 1 --cp 4 --seq 16384);;
            tp4) parallel_args=(--tp 4 --cp 1 --seq 16384);;
        esac
        for fsdp in "${FSDP_LEVELS[@]}"; do
            for precision in "${PRECISIONS[@]}"; do
                case_number=$((case_number + 1))
                label="${model_label}-fsdp${fsdp}-${precision}-${layout}"
                printf '[%d/%d] %s\n' "${case_number}" "${total}" "${label}"
                if ! bash "${RUNNER}" "${common[@]}" "${model_args[@]}" "${parallel_args[@]}" \
                    --fsdp "${fsdp}" --precision "${precision}" --run-name "${layout}-grad_bf16-main_grad_fp32" \
                    ${RUN_OPTIONS[@]+"${RUN_OPTIONS[@]}"}; then
                    failed+=("${label}")
                fi
            done
        done
    done
done

if (( ${#failed[@]} > 0 )); then
    printf 'Failed cases:\n' >&2
    printf '  %s\n' "${failed[@]}" >&2
    exit 1
fi
