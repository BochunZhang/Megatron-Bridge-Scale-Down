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
Analyse the --nsys reports of finished run_nvls_collective_benchmark.sh runs, offline.

Works on run directories of both launchers (nvls-collective and nvls-collective-cpp): files any
report still under <algo>/nsys/, exports every report to SQLite, runs nsys_collective_kernels.sql
and writes <algo>/nsys/kernels.csv and <run dir>/nsys_kernels.md (analyse_nsys.py); reports
with GPU metrics also give <algo>/nsys/gpu_metrics.csv and <run dir>/nsys_gpu_metrics.md (NVLink,
DRAM and C2C throughput over each trial's collective kernels).

The launchers do not run this step: exporting every report of a 32-GPU run takes hours on node 0,
and the other nodes would sit in the next job's rendezvous until it timed out. Run it after the
job, on any machine that sees the run directory and has nsys; it needs no GPU. Filed reports and
SQLite exports newer than their report are reused, so an interrupted analysis resumes. Every run
directory is analysed even if an earlier one failed; the exit status is non-zero if any failed.

Usage:
  analyse_nsys.sh --output-dir DIR [--output-dir DIR ...] [--nsys-bin PATH] [--wait-seconds S]

Options:
  --output-dir DIR        A launcher run directory, e.g.
                          results/gb200/nvls-collective/gpu-num-<N>/<yyyymmdd-hhmmss>; repeatable
  --nsys-bin PATH         nsys used for `nsys export` (default: nsys)
  --wait-seconds S        How long to wait for reports still being written (default: 0, the job
                          is over)
  -h, --help              Show this help
USAGE
}

die() {
    echo "ERROR: $*" >&2
    exit 2
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
output_dirs=()
nsys_bin="nsys"
wait_seconds="0"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir) [[ $# -ge 2 ]] || die "$1 requires a value"; output_dirs+=("$2"); shift 2 ;;
        --nsys-bin) [[ $# -ge 2 ]] || die "$1 requires a value"; nsys_bin="$2"; shift 2 ;;
        --wait-seconds) [[ $# -ge 2 ]] || die "$1 requires a value"; wait_seconds="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1 (see --help)" ;;
    esac
done
(( ${#output_dirs[@]} )) || die "--output-dir is required"
command -v "$nsys_bin" >/dev/null || die "${nsys_bin} not found; pass --nsys-bin"

failed=()
for output_dir in "${output_dirs[@]}"; do
    if [[ ! -d "$output_dir" ]]; then
        echo "ERROR: ${output_dir} is not a directory" >&2
        failed+=("$output_dir")
        continue
    fi
    echo "=== nsys analysis of ${output_dir}" >&2
    uv run --no-sync python "${script_dir}/analyse_nsys.py" --output-dir "$output_dir" \
        --nsys-bin "$nsys_bin" --wait-seconds "$wait_seconds" || failed+=("$output_dir")
done

if (( ${#failed[@]} )); then
    echo "ERROR: nsys analysis failed for: ${failed[*]}; see the messages above" >&2
    exit 1
fi
