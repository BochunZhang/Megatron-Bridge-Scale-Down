#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ONE="${SCRIPT_DIR}/run_pretrain_fsdp1.sh"

PRECISION="${PRECISION:-bf16}"
MODEL="qwen35_text_9b"
RECIPE_BF16="qwen35_text_9b_pretrain_4gpu_gb200_bf16_fsdp1_config"
RECIPE_FP8MX="qwen35_text_9b_pretrain_4gpu_gb200_fp8mx_fsdp1_config"
TEST_ONLY=false
PROFILE=false

# Training parameters with defaults (can be overridden via environment variables)
TRAIN_ITERS="${TRAIN_ITERS:-10}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
PROFILE_STEP_START="${PROFILE_STEP_START:-7}"
PROFILE_STEP_END="${PROFILE_STEP_END:-8}"

usage() {
    cat <<'EOF'
Usage: run_qwen35_9b.sh [OPTIONS]

Options:
    --test                Run one short baseline to validate the environment
    --profile             Enable nsys, NVTX, and memory-history recording
    --precision <bf16|fp8mx>  Precision mode (default: bf16)
    --train-iters <n>     Number of training iterations (default: 10)
    --global-batch-size <n>   Global batch size (default: 32)
    --micro-batch-size <n>  Micro batch size (default: 1)
    --profile-step-start <n>  Profile start step (default: 7)
    --profile-step-end <n>    Profile end step (default: 8)
    -h, --help            Show this help message

Without --test, run baseline, recompute-1/2, and offload-1/2 in order.
With --test, use the configured test iteration count; with --profile, record steps 0..10.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --test)
            TEST_ONLY=true
            shift
            ;;
        --profile)
            PROFILE=true
            shift
            ;;
        --precision)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            PRECISION="$2"
            shift 2
            ;;
        --train-iters)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            TRAIN_ITERS="$2"
            shift 2
            ;;
        --global-batch-size)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            GLOBAL_BATCH_SIZE="$2"
            shift 2
            ;;
        --micro-batch-size)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            MICRO_BATCH_SIZE="$2"
            shift 2
            ;;
        --profile-step-start)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            PROFILE_STEP_START="$2"
            shift 2
            ;;
        --profile-step-end)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            PROFILE_STEP_END="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

PROFILE_ARGS=()
if [[ "${PROFILE}" == true ]]; then
    PROFILE_ARGS=(--profile)
fi

if [[ "${PRECISION}" == bf16 ]]; then
    RECIPE="${RECIPE_BF16}"
elif [[ "${PRECISION}" == fp8mx ]]; then
    RECIPE="${RECIPE_FP8MX}"
else
    echo "Unsupported precision: ${PRECISION}" >&2
    exit 2
fi

run_config() {
    local run_name="$1"
    local recompute_granularity="$2"
    local recompute_modules="$3"
    local fine_grained_offload="$4"
    local offload_modules="$5"

    "${RUN_ONE}" \
        "${PROFILE_ARGS[@]}" \
        --model "${MODEL}" \
        --recipe "${RECIPE}" \
        --precision "${PRECISION}" \
        --run-name "${run_name}" \
        --recompute-granularity "${recompute_granularity}" \
        --recompute-modules "${recompute_modules}" \
        --fine-grained-offload "${fine_grained_offload}" \
        --offload-modules "${offload_modules}" \
        --train-iters "${TRAIN_ITERS}" \
        --global-batch-size "${GLOBAL_BATCH_SIZE}" \
        --micro-batch-size "${MICRO_BATCH_SIZE}" \
        --profile-step-start "${PROFILE_STEP_START}" \
        --profile-step-end "${PROFILE_STEP_END}"
}

if [[ "${TEST_ONLY}" == true ]]; then
    "${RUN_ONE}" \
        "${PROFILE_ARGS[@]}" \
        --model "${MODEL}" \
        --recipe "${RECIPE}" \
        --precision "${PRECISION}" \
        --run-name baseline \
        --recompute-granularity null \
        --recompute-modules null \
        --fine-grained-offload false \
        --offload-modules null \
        --train-iters "${TRAIN_ITERS}" \
        --global-batch-size "${GLOBAL_BATCH_SIZE}" \
        --micro-batch-size "${MICRO_BATCH_SIZE}" \
        --profile-step-start 0 \
        --profile-step-end 10
    exit 0
fi

run_config baseline null null false null
run_config recompute-1 selective "[layernorm,mlp]" false null
run_config recompute-2 selective "[layernorm,mlp]" true "[mlp_norm]"
run_config offload-1 selective "[layernorm,mlp_act]" true "[mlp_act]"
run_config offload-2 selective "[layernorm,mlp_act]" true "[mlp_norm,mlp_act]"
