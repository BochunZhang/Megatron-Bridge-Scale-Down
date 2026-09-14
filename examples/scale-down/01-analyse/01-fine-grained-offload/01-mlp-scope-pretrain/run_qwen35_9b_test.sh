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

MODEL="qwen35_text_9b"
RECIPE="qwen35_text_9b_pretrain_8gpu_gb200_bf16_config"
PRECISION="bf16"

# Training parameters with defaults (can be overridden via environment variables or command line)
TRAIN_ITERS="${TRAIN_ITERS:-10}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
PROFILE_STEP_START="${PROFILE_STEP_START:-7}"
PROFILE_STEP_END="${PROFILE_STEP_END:-8}"

usage() {
    cat <<'EOF'
Usage: run_qwen35_9b_test.sh [OPTIONS]

Options:
    --train-iters <n>     Number of training iterations (default: 10)
    --global-batch-size <n>   Global batch size (default: 8)
    --micro-batch-size <n>  Micro batch size (default: 1)
    --profile-step-start <n>  Profile start step (default: 7)
    --profile-step-end <n>    Profile end step (default: 8)
    -h, --help            Show this help message

Runs a baseline configuration for qwen35_text_9b model.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
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

run_config() {
    local run_name="$1"
    local recompute_granularity="$2"
    local recompute_modules="$3"
    local fine_grained_offload="$4"
    local offload_modules="$5"

    "${RUN_ONE}" \
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

run_config baseline null null false null
