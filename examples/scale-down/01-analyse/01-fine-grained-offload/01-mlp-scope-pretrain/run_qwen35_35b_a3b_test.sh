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

MODEL="qwen35_text_35b_a3b"
RECIPE="qwen35_text_35b_a3b_pretrain_8gpu_gb200_bf16_config"
PRECISION="bf16"

TRAIN_ITERS=10
GLOBAL_BATCH_SIZE=8
MICRO_BATCH_SIZE=1

run_config() {
    local run_name="$1"
    local recompute_granularity="$2"
    local recompute_modules="$3"
    local fine_grained_offload="$4"
    local offload_modules="$5"

    TRAIN_ITERS="${TRAIN_ITERS}" \
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE}" \
    MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE}" \
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

run_config baseline null null false null