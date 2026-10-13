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
# Build RESULTS_ROOT/seq<len>/csv/<category>.csv from the JSON records; adds nsys_*
# columns for every category that has RESULTS_ROOT/seq<len>/nsys/<category>.sqlite.
#
#   build_csv.sh [--seq 4k] [extra mbench.build_csv arguments]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

SEQ=4k
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seq) SEQ="$2"; shift 2 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done
SEQ_TOKENS="$(mbench_seq_tokens "$SEQ")"
INPUT_DIR="$RESULTS_ROOT/seq$(mbench_seq_tag "$SEQ_TOKENS")"
uv run --no-sync python -m mbench.build_csv --input-dir "$INPUT_DIR" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
