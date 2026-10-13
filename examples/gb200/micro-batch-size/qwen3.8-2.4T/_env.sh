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
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
BENCHMARK_ROOT="$REPO_ROOT/examples/gb200/benchmark"
CONFIG="$SCRIPT_DIR/mbench.yaml"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/results/gb200/micro-batch-size/qwen3.8_2.4T}"
PYTHONPATH="$BENCHMARK_ROOT:$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
# Use the sibling flash-linear-attention checkout for Megatron's GatedDeltaNet when present.
if [[ -d "$REPO_ROOT/../flash-linear-attention" ]]; then
  PYTHONPATH="$REPO_ROOT/../flash-linear-attention:$PYTHONPATH"
fi
export PYTHONPATH SCRIPT_DIR REPO_ROOT BENCHMARK_ROOT CONFIG RESULTS_ROOT
# shellcheck source=../../benchmark/seq.sh
source "$BENCHMARK_ROOT/seq.sh"
cd "$REPO_ROOT"
