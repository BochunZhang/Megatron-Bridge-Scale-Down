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
# Compare baseline / offload-mlp / offload-attn-mlp on complete Qwen3.5 dense
# and expert models with PP=4 across eight DLC nodes (32 GPUs).
#
# Unlike the 4/8/16-GPU matrices, this one runs with --sharding none instead of
# Megatron-FSDP. Megatron-FSDP builds its DP-TP device mesh from the global
# world size while dp_cp is a per-pipeline-stage group, so its mesh helpers only
# balance when PP=1; at PP=4 the launch dies in _get_dp_tp_mesh with
# "Shape mismatch, 32 != 8" before the first step. --sharding none keeps the
# same PP=4/EP/DP layout on the distributed optimizer.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

usage() {
    cat <<EOF
Usage: run_32gpu_compare_activation_offload_influence_on_dense_and_expert.sh [OPTIONS]

DLC, 8 nodes x 4 GPUs, DDP (--sharding none), PP=4, complete models (no cropping):
  dense   Qwen3.5-27B        PP=4, EP=1 (DP=8)
  expert  Qwen3.5-122B-A10B  PP=4, EP=8 (DP=8, EDP=1), HybridEP

Megatron-FSDP is not used here: its device-mesh helpers assume PP=1, so the
32-GPU PP=4 layout fails at setup. Megatron-FSDP remains active in the 4/8/16
GPU matrices, which are all PP=1.

Defaults: --mbs 1 --model-type dense,expert
          --case baseline,offload-mlp,offload-attn-mlp --gbs 128
          --sync-with-sm true (expert only)

MBS=2 is not in the default matrix: with 1F1B, pipeline stage 0 keeps four
microbatches of activations, which puts both models at roughly 180 GB/GPU for
the baseline. Add --mbs 1,2 to try it; an OOM there is expected.
Qwen3.5-122B-A10B only has a bf16 recipe, so --dtype mxfp8 is rejected.

Every worker runs the same command; MASTER_ADDR, MASTER_PORT, WORLD_SIZE and
RANK come from DLC.

${COMMON_USAGE}
EOF
}

MBS_LIST=1
MODEL_TYPES=dense,expert
CASES=baseline,offload-mlp,offload-attn-mlp
GBS=128
SYNC_WITH_SM=true
PROFILE=none
parse_matrix_args "$@"
[[ "${MODEL_FAMILY}" == qwen ]] || die "the 32-GPU comparison uses Qwen3.5 models only"
if [[ "${DTYPE}" != bf16 && ",${MODEL_TYPES}," == *",expert,"* ]]; then
    die "Qwen3.5-122B-A10B only has a bf16 recipe; use --dtype bf16 or --model-type dense"
fi
setup_matrix 32
stage_local_hf_home Qwen/Qwen3.5-27B Qwen/Qwen3.5-122B-A10B

for mbs in "${MBS_VALUES[@]}"; do
    for model_type in "${MODEL_TYPE_VALUES[@]}"; do
        for case_name in "${CASE_VALUES[@]}"; do
            modules="$(offload_modules "${case_name}" "${model_type}" qwen)"
            if [[ "${model_type}" == dense ]]; then
                run_case 32 4 1 default qwen35_text_27b dense "${case_name}" "${modules}" "${mbs}" false \
                    --sharding none
            else
                run_case 32 4 8 hybridep qwen35_text_122b_a10b expert "${case_name}" "${modules}" "${mbs}" \
                    "${SYNC_WITH_SM}" \
                    --sharding none
            fi
        done
    done
done
finish_matrix
