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

set -uo pipefail

usage() {
    cat <<'USAGE'
Post-process one run_nvls_collective_benchmark.sh run directory: compare.md from every
<algo>/results.jsonl.

The launcher calls this on node 0 after the last pass and only warns when it fails, so the
measurements are kept either way; run it again by hand to retry. It takes seconds. The --nsys
reports are analysed separately, after the job, by analyse_nsys.sh (hours for a 32-GPU run).

Usage:
  analyse_results.sh --output-dir DIR

Options:
  --output-dir DIR        results/gb200/nvls-collective/gpu-num-<N>/<yyyymmdd-hhmmss>
  -h, --help              Show this help
USAGE
}

die() {
    echo "ERROR: $*" >&2
    exit 2
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
output_dir=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir) [[ $# -ge 2 ]] || die "$1 requires a value"; output_dir="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1 (see --help)" ;;
    esac
done
[[ -n "$output_dir" ]] || die "--output-dir is required"
[[ -d "$output_dir" ]] || die "${output_dir} is not a directory"

compare_args=()
for algo in ring nvls symm; do
    [[ -f "${output_dir}/${algo}/results.jsonl" ]] && compare_args+=(--input "${algo}=${output_dir}/${algo}/results.jsonl")
done
(( ${#compare_args[@]} )) || die "no <algo>/results.jsonl under ${output_dir}; cannot write compare.md"
uv run --no-sync python "${script_dir}/collective_metrics.py" "${compare_args[@]}" \
    --output "${output_dir}/compare.md" \
    || { echo "ERROR: compare.md failed for ${output_dir}; see the messages above" >&2; exit 1; }
