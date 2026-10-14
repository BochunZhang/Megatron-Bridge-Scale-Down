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
# Compare baseline / offload-mlp / offload-attn-mlp on cropped dense and expert
# models on a single 4-GPU node.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

usage() {
    cat <<EOF
Usage: run_4gpu_compare_activation_offload_influence_on_dense_and_expert.sh [OPTIONS]

Single node, 4 GPUs, FSDP1, cropped models with the same depth:

  --model-family qwen (default)
    dense   Qwen3.5-9B,      16 layers, MTP off, EP=1
    expert  Qwen3.5-35B-A3B, 16 layers, 64 experts, MTP off, EP=4, HybridEP
  --model-family deepseek
    dense   DeepSeek-V3, 4 layers, moe_layer_freq=[0,0,0,0], EP=1
    expert  DeepSeek-V3, 4 layers, 32 experts, moe_layer_freq=[1,1,1,1], EP=4, HybridEP

Defaults: --mbs 1,2 --model-type dense,expert
          --case baseline,offload-mlp,offload-attn-mlp --gbs 32
          --sync-with-sm true (expert only; removes the HybridEP .item() block)

  --model-family <qwen|deepseek>

${COMMON_USAGE}
EOF
}

MBS_LIST=1,2
MODEL_TYPES=dense,expert
CASES=baseline,offload-mlp,offload-attn-mlp
GBS=32
SYNC_WITH_SM=true
PROFILE=none
parse_matrix_args "$@"
setup_matrix 4

for mbs in "${MBS_VALUES[@]}"; do
    for model_type in "${MODEL_TYPE_VALUES[@]}"; do
        for case_name in "${CASE_VALUES[@]}"; do
            modules="$(offload_modules "${case_name}" "${model_type}" "${MODEL_FAMILY}")"
            if [[ "${MODEL_FAMILY}" == qwen && "${model_type}" == dense ]]; then
                run_case 4 1 1 default qwen35_text_9b dense "${case_name}" "${modules}" "${mbs}" false \
                    --num-layers 16 --linear-attention-freq 4 --mtp-num-layers 0
            elif [[ "${MODEL_FAMILY}" == qwen ]]; then
                run_case 4 1 4 hybridep qwen35_text_35b_a3b expert "${case_name}" "${modules}" "${mbs}" \
                    "${SYNC_WITH_SM}" \
                    --num-layers 16 --num-experts 64 --linear-attention-freq 4 --mtp-num-layers 0
            elif [[ "${model_type}" == dense ]]; then
                run_case 4 1 1 default deepseek_v3 dense "${case_name}" "${modules}" "${mbs}" false \
                    --num-layers 4 --num-experts 32 --moe-layer-freq '[0,0,0,0]'
            else
                run_case 4 1 4 hybridep deepseek_v3 expert "${case_name}" "${modules}" "${mbs}" \
                    "${SYNC_WITH_SM}" \
                    --num-layers 4 --num-experts 32 --moe-layer-freq '[1,1,1,1]'
            fi
        done
    done
done
finish_matrix
