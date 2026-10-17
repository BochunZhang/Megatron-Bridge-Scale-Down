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
<algo>/results.jsonl, then, if the run used --nsys, file the reports under
<algo>/cases/<algo>-<case>/ and write nsys_kernels.md (analyse_nsys.py).

The launcher calls this on node 0 after the last pass and only warns when it fails, so the
measurements are kept either way. Run it again by hand to retry or to refresh the reports;
filed reports and their SQLite exports are reused. Every step runs even if an earlier one
failed; the exit status is non-zero if any of them failed.

Usage:
  analyse_results.sh --output-dir DIR [--nsys-bin PATH] [--wait-seconds S]

Options:
  --output-dir DIR        results/gb200/nvls-collective/gpu-num-<N>/<yyyymmdd-hhmmss>
  --nsys-bin PATH         nsys used for `nsys export` (default: nsys)
  --wait-seconds S        How long to wait for other nodes' reports (default: 300)
  -h, --help              Show this help
USAGE
}

die() {
    echo "ERROR: $*" >&2
    exit 2
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
output_dir=""
nsys_bin="nsys"
wait_seconds="300"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir) [[ $# -ge 2 ]] || die "$1 requires a value"; output_dir="$2"; shift 2 ;;
        --nsys-bin) [[ $# -ge 2 ]] || die "$1 requires a value"; nsys_bin="$2"; shift 2 ;;
        --wait-seconds) [[ $# -ge 2 ]] || die "$1 requires a value"; wait_seconds="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1 (see --help)" ;;
    esac
done
[[ -n "$output_dir" ]] || die "--output-dir is required"
[[ -d "$output_dir" ]] || die "${output_dir} is not a directory"

failed=()
compare_args=()
for algo in ring nvls symm; do
    [[ -f "${output_dir}/${algo}/results.jsonl" ]] && compare_args+=(--input "${algo}=${output_dir}/${algo}/results.jsonl")
done
if (( ${#compare_args[@]} )); then
    uv run --no-sync python "${script_dir}/collective_metrics.py" "${compare_args[@]}" \
        --output "${output_dir}/compare.md" || failed+=(compare.md)
else
    echo "warning: no <algo>/results.jsonl under ${output_dir}; skipping compare.md" >&2
    failed+=(compare.md)
fi

if compgen -G "${output_dir}/*/nsys/index.tsv" >/dev/null; then
    uv run --no-sync python "${script_dir}/analyse_nsys.py" --output-dir "$output_dir" \
        --nsys-bin "$nsys_bin" --wait-seconds "$wait_seconds" || failed+=(nsys_kernels.md)
fi

if (( ${#failed[@]} )); then
    echo "ERROR: ${failed[*]} failed for ${output_dir}; see the messages above" >&2
    exit 1
fi
