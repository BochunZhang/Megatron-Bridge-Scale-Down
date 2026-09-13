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
MODEL="qwen38_text_27b"
RECIPE_BF16="qwen38_text_27b_pretrain_4gpu_gb200_bf16_fsdp1_config"
RECIPE_FP8MX="qwen38_text_27b_pretrain_4gpu_gb200_fp8mx_fsdp1_config"
TEST_ONLY=false

usage() {
    cat <<'EOF'
Usage: run_dense_test.sh [--test] [--precision bf16|fp8mx]

Without --test, run one baseline, recompute-1/2, and offload-1/2 in order.
With --test, run one short baseline to validate the four-GPU environment.
EOF
}

run_baseline_test() {
    TRAIN_ITERS="${TEST_TRAIN_ITERS:-2}" \
    GLOBAL_BATCH_SIZE="${TEST_GLOBAL_BATCH_SIZE:-4}" \
    PROFILE_STEP_START=0 \
    PROFILE_STEP_END=1 \
    "${RUN_ONE}" \
        --model "${MODEL}" \
        --recipe "${RECIPE}" \
        --precision "${PRECISION}" \
        --run-name baseline \
        --recompute-granularity null \
        --recompute-modules null \
        --fine-grained-offload false \
        --offload-modules null
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --test)
            TEST_ONLY=true
            shift
            ;;
        --precision)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            PRECISION="$2"
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
        --model "${MODEL}" \
        --recipe "${RECIPE}" \
        --precision "${PRECISION}" \
        --run-name "${run_name}" \
        --recompute-granularity "${recompute_granularity}" \
        --recompute-modules "${recompute_modules}" \
        --fine-grained-offload "${fine_grained_offload}" \
        --offload-modules "${offload_modules}"
}

if [[ "${TEST_ONLY}" == true ]]; then
    run_baseline_test
    exit 0
fi

run_config baseline null null false null
run_config recompute-1 selective "[layernorm,mlp]" false null
run_config recompute-2 selective "[layernorm,mlp]" true "[mlp_norm]"
run_config offload-1 selective "[layernorm,mlp_act]" true "[mlp_act]"
run_config offload-2 selective "[layernorm,mlp_act]" true "[mlp_norm,mlp_act]"
