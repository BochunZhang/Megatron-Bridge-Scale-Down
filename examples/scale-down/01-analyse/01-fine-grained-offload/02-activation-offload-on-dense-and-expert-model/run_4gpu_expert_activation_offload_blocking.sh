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
# Manually verify that --sync-with-sm removes the HybridEP host block.
#
# Runs the same expert configuration twice under nsys, once with the blocking
# `.item()` in _HybridEPManager.setup_metadata and once with the SM copy, so the
# two timelines can be compared side by side.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

usage() {
    cat <<EOF
Usage: run_4gpu_expert_activation_offload_blocking.sh [OPTIONS]

Single node, 4 GPUs, FSDP1. Model: Qwen3.5-35B-A3B cropped to 16 layers and
64 experts, MTP disabled, EP=4, HybridEP. Each case runs twice:
  <case>-blocking  model.moe_hybridep_sync_with_sm=false  (.item())
  <case>-sync-sm   model.moe_hybridep_sync_with_sm=true   (SM copy)

Defaults: --case offload-attn-mlp --mbs 1 --gbs 32 --profile nsys.
--sync-with-sm and --model-type are fixed by this script.

What to check in each nsys report (step 7-8):
  - blocking: cudaMemcpyAsync D2H followed by cudaStreamSynchronize inside
    the MoEFlexTokenDispatcher.dispatch_preprocess NVTX range.
  - sync-sm: the same range shows the Triton _sm_copy_with_flag_kernel and the
    NVTX range "hybrid-ep sm-sync wait" instead of the stream sync.
  - train.log loss is identical step by step between the two runs.

${COMMON_USAGE}
EOF
}

MBS_LIST=1
MODEL_TYPES=expert
CASES=offload-attn-mlp
GBS=32
SYNC_WITH_SM=true
PROFILE=nsys
parse_matrix_args "$@"
[[ "${MODEL_FAMILY}" == qwen ]] || die "this check uses the Qwen3.5 expert model only"
setup_matrix 4

EXPERT_CROP=(--num-layers 16 --num-experts 64 --linear-attention-freq 4 --mtp-num-layers 0)
for mbs in "${MBS_VALUES[@]}"; do
    for case_name in "${CASE_VALUES[@]}"; do
        modules="$(offload_modules "${case_name}" expert qwen)"
        run_case 4 1 4 hybridep qwen35_text_35b_a3b expert "${case_name}-blocking" "${modules}" \
            "${mbs}" false "${EXPERT_CROP[@]}"
        run_case 4 1 4 hybridep qwen35_text_35b_a3b expert "${case_name}-sync-sm" "${modules}" \
            "${mbs}" true "${EXPERT_CROP[@]}"
    done
done
finish_matrix
