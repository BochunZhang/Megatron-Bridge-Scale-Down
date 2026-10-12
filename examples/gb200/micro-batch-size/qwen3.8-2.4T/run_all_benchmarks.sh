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
# Run the Qwen3.8 2.4T mbench scopes (one process each) and build the CSVs.
#
#   run_all_benchmarks.sh [--scope "GEMM GatedDeltaNet GatedAttention MLP Experts"]
#                         [--nsys] [--micro-batch-sizes 1,2,4,8] [--device N]
#
# --scope takes a space- or comma-separated list (case-insensitive); default: all scopes.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

SCOPES="GEMM GatedDeltaNet GatedAttention MLP Experts"
PASS_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --scope) SCOPES="$2"; shift 2 ;;
    --nsys) PASS_ARGS+=("--nsys"); shift ;;
    --micro-batch-sizes|--device) PASS_ARGS+=("$1" "$2"); shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

read -r -a SCOPE_LIST <<< "${SCOPES//,/ }"
for scope in "${SCOPE_LIST[@]}"; do
  "$BENCHMARK_ROOT/run_mbench.sh" --config "$CONFIG" --scope "$scope" --output-dir "$RESULTS_ROOT" \
    ${PASS_ARGS[@]+"${PASS_ARGS[@]}"}
done
"$SCRIPT_DIR/build_csv.sh"
