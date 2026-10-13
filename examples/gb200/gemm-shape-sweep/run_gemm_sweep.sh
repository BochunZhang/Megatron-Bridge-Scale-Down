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
# Run the GEMM shape sweep (mbench scope GemmSweep) and build its CSV.
#
#   run_gemm_sweep.sh [--nsys] [--cases square_m8192_k8192_n8192,...] [--device N]
#
# Shapes come from gemm_sweep.yaml (override with CONFIG=...); results go to
# $RESULTS_ROOT/gemm_sweep/ and $RESULTS_ROOT/csv/gemm_sweep.csv.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

PASS_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --nsys) PASS_ARGS+=("--nsys"); shift ;;
    --cases|--device) PASS_ARGS+=("$1" "$2"); shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

"$BENCHMARK_ROOT/run_mbench.sh" --config "$CONFIG" --scope GemmSweep --output-dir "$RESULTS_ROOT" \
  ${PASS_ARGS[@]+"${PASS_ARGS[@]}"}
"$SCRIPT_DIR/build_csv.sh"
