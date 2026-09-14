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

MODEL="qwen35_text_27b"
RECIPE="qwen35_text_27b_pretrain_4gpu_gb200_bf16_fsdp1_config"
PRECISION="bf16"

# Training parameters with defaults (can be overridden via environment variables or command line)
TRAIN_ITERS="${TRAIN_ITERS:-10}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
PROFILE_STEP_START="${PROFILE_STEP_START:-7}"
PROFILE_STEP_END="${PROFILE_STEP_END:-8}"

usage() {
    cat <<'EOF'
Usage: test_qwen35_27b.sh [OPTIONS]

Options:
    --train-iters <n>     Number of training iterations (default: 10)
    --global-batch-size <n>   Global batch size (default: 32)
    --micro-batch-size <n>  Micro batch size (default: 1)
    --profile-step-start <n>  Profile start step (default: 7)
    --profile-step-end <n>    Profile end step (default: 8)
    -h, --help            Show this help message

Runs a baseline configuration for qwen35_text_27b using bf16 and FSDP1.
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

"${RUN_ONE}" \
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
    --profile-step-start "${PROFILE_STEP_START}" \
    --profile-step-end "${PROFILE_STEP_END}"
