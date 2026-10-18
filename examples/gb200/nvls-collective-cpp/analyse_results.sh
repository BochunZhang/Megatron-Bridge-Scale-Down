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
Post-process one run_nvls_collective_benchmark.sh (C++) run directory: compare.md from every
<impl>/results.jsonl (build/collective_report).

The launcher calls this on node 0 after the last pass and only warns when it fails, so the
measurements are kept either way; run it again by hand to retry. It takes seconds. The --nsys
reports are analysed separately, after the job, by ../nvls-collective/analyse_nsys.sh (hours for
a 32-GPU run).

Usage:
  analyse_results.sh --output-dir DIR [--report-bin PATH]

Options:
  --output-dir DIR        results/gb200/nvls-collective-cpp/gpu-num-<N>/<yyyymmdd-hhmmss>
  --report-bin PATH       collective_report binary (default: build/collective_report next to this script)
  -h, --help              Show this help
USAGE
}

die() {
    echo "ERROR: $*" >&2
    exit 2
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
output_dir=""
report_bin="${script_dir}/build/collective_report"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir) [[ $# -ge 2 ]] || die "$1 requires a value"; output_dir="$2"; shift 2 ;;
        --report-bin) [[ $# -ge 2 ]] || die "$1 requires a value"; report_bin="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1 (see --help)" ;;
    esac
done
[[ -n "$output_dir" ]] || die "--output-dir is required"
[[ -d "$output_dir" ]] || die "${output_dir} is not a directory"

report_args=()
for impl in ring nvls symmetric symmetric-tma copy-engine; do
    [[ -f "${output_dir}/${impl}/results.jsonl" ]] && report_args+=(--input "${impl}=${output_dir}/${impl}/results.jsonl")
done
(( ${#report_args[@]} )) || die "no <impl>/results.jsonl under ${output_dir}; cannot write compare.md"
"$report_bin" "${report_args[@]}" --output "${output_dir}/compare.md" \
    || { echo "ERROR: compare.md failed for ${output_dir}; see the messages above" >&2; exit 1; }
