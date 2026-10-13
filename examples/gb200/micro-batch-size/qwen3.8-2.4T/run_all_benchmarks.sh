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
#                         [--seq 4k] [--nsys] [--micro-batch-sizes 1,2,4,8] [--device N]
#
# --scope takes a space- or comma-separated list (case-insensitive); default: all scopes.
# --seq   sequence length(s), space- or comma-separated: 4k (default), 256, 1k, 16k, 256k,
#         ... or a token count; e.g. --seq "256,1k,4k,16k" runs each length in turn.
#         Results go to $RESULTS_ROOT/seq<len>/ (e.g. seq256, seq4k, seq256k); cases that do
#         not fit in GPU memory are kept in the CSV with status=oom.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

SCOPES="GEMM GatedDeltaNet GatedAttention MLP Experts"
SEQ=4k
PASS_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --scope) SCOPES="$2"; shift 2 ;;
    --seq) SEQ="$2"; shift 2 ;;
    --nsys) PASS_ARGS+=("--nsys"); shift ;;
    --micro-batch-sizes|--device) PASS_ARGS+=("$1" "$2"); shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
read -r -a SEQ_LIST <<< "${SEQ//,/ }"
read -r -a SCOPE_LIST <<< "${SCOPES//,/ }"
SEQ_TOKENS_LIST=()
for seq in "${SEQ_LIST[@]}"; do
  SEQ_TOKENS_LIST+=("$(mbench_seq_tokens "$seq")")  # validate every length before running any
done

for seq_tokens in "${SEQ_TOKENS_LIST[@]}"; do
  OUTPUT_DIR="$RESULTS_ROOT/seq$(mbench_seq_tag "$seq_tokens")"
  for scope in "${SCOPE_LIST[@]}"; do
    "$BENCHMARK_ROOT/run_mbench.sh" --config "$CONFIG" --scope "$scope" --output-dir "$OUTPUT_DIR" \
      --seq "$seq_tokens" ${PASS_ARGS[@]+"${PASS_ARGS[@]}"}
  done
  "$SCRIPT_DIR/build_csv.sh" --seq "$seq_tokens"
done
