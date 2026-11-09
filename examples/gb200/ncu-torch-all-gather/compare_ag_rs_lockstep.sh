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

usage() {
    cat <<'USAGE'
Compare ncu lockstep off and on for concurrent AllGather_STMC + ReduceScatter_LDMC.
Both runs use pause=0, 1024 MiB bf16, one iteration, and app-range replay over tcp.
Build the pause library with install/install-nccl-2.32.3-pause.sh before running.

Usage: bash examples/gb200/ncu-torch-all-gather/compare_ag_rs_lockstep.sh [options]

Options:
  --ncu PATH       Nsight Compute executable (default: <repo>/NVIDIA-Nsight-Compute-2026.4/ncu)
  --venv DIR|none Torch venv, or none to keep the current uv environment
                   (default: <repo>/.venv/python3.12-torch2.13-nccl2.32.3)
  --nccl-lib PATH  NCCL 2.32.3-pause library (default: <repo>/nccl/nccl-2.32.3-pause/build/lib/libnccl.so.2)
  --gpus LIST      Four GPU indices (default: 0,1,2,3)
  --timeout SEC    Seconds per ncu run (default: 300)
  --output-dir DIR New or empty directory for both runs and acceptance.tsv
  --dry-run        Print both launcher commands without running them
  -h, --help
USAGE
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
launcher="${script_dir}/run_ncu_torch_all_gather.sh"
ncu_bin="${repo_root}/NVIDIA-Nsight-Compute-2026.4/ncu"
venv="${repo_root}/.venv/python3.12-torch2.13-nccl2.32.3"
nccl_lib="${repo_root}/nccl/nccl-2.32.3-pause/build/lib/libnccl.so.2"
gpus="0,1,2,3"
timeout_sec=300
output_dir=""
dry_run=false
modes=(lockstep-off lockstep-on)
metrics=launch__count,launch__sub_launch_name,smsp__warps_active.sum,gpu__time_duration.sum
metrics+=,smsp__warps_issue_stalled_lg_throttle.sum,smsp__warps_issue_stalled_long_scoreboard.sum

die() {
    echo "ERROR: $*" >&2
    exit 1
}

while (( $# )); do
    case "$1" in
        --ncu) ncu_bin="$2"; shift 2 ;;
        --venv) venv="$2"; shift 2 ;;
        --nccl-lib) nccl_lib="$2"; shift 2 ;;
        --gpus) gpus="$2"; shift 2 ;;
        --timeout) timeout_sec="$2"; shift 2 ;;
        --output-dir) output_dir="$2"; shift 2 ;;
        --dry-run) dry_run=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

[[ "$gpus" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*)){3}$ ]] || die "--gpus requires four GPU indices"
[[ "$(tr ',' '\n' <<<"$gpus" | sort -u | wc -l | tr -d ' ')" == 4 ]] || die "--gpus requires four distinct indices"
[[ "$timeout_sec" =~ ^[1-9][0-9]*$ ]] || die "--timeout must be positive"
output_dir="${output_dir:-${repo_root}/results/gb200/ncu-torch-all-gather/ag-rs-lockstep-ab-$(date +%Y%m%d-%H%M%S)}"

case_cmd() {
    cmd=(
        env -u NCCL_SYM_CLOCK64_CSV_PROFILE NCCL_SYM_AG_STMC_PAUSE_CYCLES=0 NCCL_SYM_TMA_ENABLE=0
        bash "$launcher" --algo symm --ag-rs --gpus "$gpus" --size-mib 1024 --dtype bf16 --warmup 5 --iters 1
        --replay-mode app-range --communicator tcp --skip-baseline --metrics "$metrics" --timeout "$timeout_sec"
        --ncu "$ncu_bin" --nccl-lib "$nccl_lib" --output-dir "${output_dir}/$1"
    )
    if [[ "$1" == lockstep-off ]]; then
        cmd+=(--no-lockstep)
    fi
    cmd+=(-- --clock-control none)
}

if $dry_run; then
    for mode in "${modes[@]}"; do
        case_cmd "$mode"
        printf '%q ' "${cmd[@]}"
        echo
    done
    exit 0
fi

ncu_bin="$(command -v "$ncu_bin")" || die "ncu not found; pass --ncu"
ncu_bin="$(cd -- "$(dirname -- "$ncu_bin")" && pwd -P)/$(basename -- "$ncu_bin")"
[[ -f "$nccl_lib" ]] || die "${nccl_lib} not found; build it with install/install-nccl-2.32.3-pause.sh"
grep -aFq "NCCL version 2.32.3-pause+cuda" "$nccl_lib" || die "${nccl_lib} is not the NCCL 2.32.3-pause build"
nccl_lib="$(cd -- "$(dirname -- "$nccl_lib")" && pwd -P)/$(basename -- "$nccl_lib")"
if [[ "$venv" != none ]]; then
    [[ -x "${venv}/bin/python" ]] || die "no virtualenv at ${venv}; pass --venv DIR or --venv none"
    export UV_PROJECT_ENVIRONMENT="$(cd -- "$venv" && pwd -P)"
fi

mkdir -p "$output_dir"
output_dir="$(cd -- "$output_dir" && pwd -P)"
[[ -z "$(ls -A "$output_dir")" ]] || die "${output_dir} is not empty"
cd "$repo_root"

exit_codes=()
for mode in "${modes[@]}"; do
    case_cmd "$mode"
    echo "=== ${mode}"
    rc=0
    "${cmd[@]}" 2>&1 | tee "${output_dir}/${mode}.log" || rc=$?
    exit_codes+=("$rc")
done

uv run --no-sync python "${script_dir}/analyse_ag_rs_lockstep.py" "$output_dir" --ncu "$ncu_bin" \
    --off-exit "${exit_codes[0]}" --on-exit "${exit_codes[1]}"
