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
#
# Compare baseline / offload-mlp / offload-attn-mlp on Qwen3.5 dense and expert
# models across two DLC nodes (8 GPUs).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

usage() {
    cat <<EOF
Usage: run_8gpu_compare_activation_offload_influence_on_dense_and_expert.sh [OPTIONS]

DLC, 2 nodes x 4 GPUs, FSDP1, PP=1:
  dense   Qwen3.5-9B        complete model             EP=1 (DP=8)
  expert  Qwen3.5-35B-A3B   complete, 128 of 256 experts  EP=8 (DP=8, EDP=1), HybridEP

The expert run is the 16-GPU expert run with half the experts, so every GPU
still holds 16 experts per MoE layer. The dense model is the same as on 16 GPUs:
Qwen3.5-9B needs ~68 GB/GPU for parameters, gradients and optimizer state at
DP=8, well below the GB200 capacity.

Every worker runs the same command; MASTER_ADDR, MASTER_PORT, WORLD_SIZE and
RANK come from DLC.

Defaults: --mbs 1,2 --model-type dense,expert
          --case baseline,offload-mlp,offload-attn-mlp --gbs 64
          --sync-with-sm true (expert only)

${COMMON_USAGE}
EOF
}

MBS_LIST=1,2
MODEL_TYPES=dense,expert
CASES=baseline,offload-mlp,offload-attn-mlp
GBS=64
SYNC_WITH_SM=true
PROFILE=none
parse_matrix_args "$@"
[[ "${MODEL_FAMILY}" == qwen ]] || die "the 8-GPU comparison uses Qwen3.5 models only"
setup_matrix 8
stage_local_hf_home Qwen/Qwen3.5-9B-Base Qwen/Qwen3.5-35B-A3B-Base

for mbs in "${MBS_VALUES[@]}"; do
    for model_type in "${MODEL_TYPE_VALUES[@]}"; do
        for case_name in "${CASE_VALUES[@]}"; do
            modules="$(offload_modules "${case_name}" "${model_type}" qwen)"
            if [[ "${model_type}" == dense ]]; then
                run_case 8 1 1 default qwen35_text_9b dense "${case_name}" "${modules}" "${mbs}" false
            else
                run_case 8 1 8 hybridep qwen35_text_35b_a3b expert "${case_name}" "${modules}" "${mbs}" \
                    "${SYNC_WITH_SM}" --num-experts 128
            fi
        done
    done
done
finish_matrix
