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

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
launcher="${script_dir}/run_ncu_torch_all_gather.sh"
ncu_sweep="${script_dir}/sweep_symm_ag_pause_rs_blocking_metrics.sh"
pause_cycles="0,128,256,512,768,1024,1536,2048,2560,3072,4096,6144"
nccl_root="${repo_root}/nccl"
ncu_bin="${repo_root}/NVIDIA-Nsight-Compute-2026.4/ncu"
venv="${repo_root}/.venv/python3.12-torch2.13-nccl2.32.3"
gpus="0,1,2,3"
size_mib=1024
dtype=bf16
warmup=5
iters=1
timeout_sec=1800
stride=32
group_samples=8
output_dir=""
dry_run=false
ncu_args=()

usage() {
    cat <<'USAGE'
Usage: sweep_symm_ag_rs_clock64.sh [options] [-- extra ncu arguments]

Run three complete sweeps of concurrent symmetric AG + RS on NCCL 2.32.3:
  store  pause-store-clock64, one store window (AG multicast / RS local)
  split  pause-split-clock64, load issue, dependent wait, and store issue
  ncu    pause-only build, app-range replay with the existing tcp ncu sweep

Options:
  --pause-cycles LIST  Default 0,128,256,512,768,1024,1536,2048,2560,3072,4096,6144
  --nccl-root DIR      Parent of the three builds (default: repo/nccl)
  --venv DIR|none      Torch venv; none uses the active uv environment
  --ncu PATH           Nsight Compute 2026.4 executable
  --gpus LIST          Default 0,1,2,3
  --size-mib N         AG output / RS input per rank (default 1024)
  --dtype bf16|fp32    Default bf16
  --warmup N           Warmup pairs, including serial connect (default 5)
  --iters N            Measured pairs per pause point (default 1)
  --stride N           Sample once per N tiles per warp (default 32)
  --group-samples N    Accumulate N samples per row and keep the tail (default 8)
  --timeout N          Seconds per launcher pass (default 1800)
  --output-dir DIR     New or empty experiment directory
  --dry-run            Print all three command schedules without CUDA or ncu

Outputs: clock64.csv (both instrumented variants), clock64-runs.csv (case
status), rank CSVs in store/ and split/, and the existing reports and metric
CSVs under ncu/. Each NCCL library is built with install-nccl-2.32.3-clock64.sh.
USAGE
}

die() { echo "ERROR: $*" >&2; exit 1; }

while (( $# )); do
    case "$1" in
        --pause-cycles) pause_cycles="$2"; shift 2 ;;
        --nccl-root) nccl_root="$2"; shift 2 ;;
        --venv) venv="$2"; shift 2 ;;
        --ncu) ncu_bin="$2"; shift 2 ;;
        --gpus) gpus="$2"; shift 2 ;;
        --size-mib) size_mib="$2"; shift 2 ;;
        --dtype) dtype="$2"; shift 2 ;;
        --warmup) warmup="$2"; shift 2 ;;
        --iters) iters="$2"; shift 2 ;;
        --timeout) timeout_sec="$2"; shift 2 ;;
        --stride) stride="$2"; shift 2 ;;
        --group-samples) group_samples="$2"; shift 2 ;;
        --output-dir) output_dir="$2"; shift 2 ;;
        --dry-run) dry_run=true; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; ncu_args=("$@"); break ;;
        *) die "unknown argument: $1" ;;
    esac
done

for name in size_mib warmup iters stride group_samples timeout_sec; do
    [[ "${!name}" =~ ^[1-9][0-9]{0,9}$ ]] && (( ${!name} <= 4294967295 )) || die "${name} must be a positive uint32"
done
[[ "$dtype" == bf16 || "$dtype" == fp32 ]] || die "--dtype must be bf16 or fp32"
[[ "$gpus" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))+$ ]] || die "--gpus must contain at least two GPU indices"
IFS=, read -r -a gpu_list <<< "$gpus"
[[ "$pause_cycles" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "--pause-cycles must be a comma-separated list"
IFS=, read -r -a pauses <<< "$pause_cycles"
seen=","
for pause in "${pauses[@]}"; do
    [[ "$pause" =~ ^(0|[1-9][0-9]{0,9})$ ]] && (( pause <= 4294967295 )) || die "invalid pause value ${pause}"
    [[ "$seen" != *",${pause},"* ]] || die "duplicate pause value ${pause}"
    seen+="${pause},"
done
output_dir="${output_dir:-${repo_root}/results/gb200/ncu-torch-all-gather/symm-ag-rs-clock64-$(date +%Y%m%d-%H%M%S)}"
[[ "$output_dir" = /* ]] || output_dir="$PWD/$output_dir"
[[ "$nccl_root" = /* ]] || nccl_root="$PWD/$nccl_root"
[[ "$ncu_bin" = /* ]] || ncu_bin="$PWD/$ncu_bin"
base_lib="${nccl_root}/nccl-2.32.3-pause/build/lib/libnccl.so.2"
common_args=(--gpus "$gpus" --dtype "$dtype" --warmup "$warmup" --iters "$iters" --timeout "$timeout_sec")

ncu_cmd=(bash "$ncu_sweep" --pause-cycles "$pause_cycles" --size-mib "$size_mib"
    --ncu "$ncu_bin" --venv "$venv" --nccl-lib "$base_lib" --output-dir "${output_dir}/ncu")
$dry_run && ncu_cmd+=(--dry-run)
ncu_cmd+=(-- "${common_args[@]}")
if (( ${#ncu_args[@]} )); then
    ncu_cmd+=(-- "${ncu_args[@]}")
fi

# Match the ncu sweep's CTA policy and keep inherited capture settings out
# of its pause-only processes. The two clock64 cases set these per process.
unset NCCL_MIN_CTAS NCCL_MAX_CTAS NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS NCCL_MIN_NRINGS NCCL_MAX_NRINGS \
    NCCL_NVLS_NCHANNELS NCCL_MIN_P2P_NCHANNELS NCCL_MAX_P2P_NCHANNELS NCCL_SYM_CTAS NCCL_SYM_AG_ST_PAUSE_CYCLES \
    NCCL_SYM_CLOCK64_CSV_PROFILE NCCL_SYM_CLOCK64_CSV_DIR NCCL_SYM_CLOCK64_CSV_LIB \
    NCCL_SYM_CLOCK64_STRIDE NCCL_SYM_CLOCK64_GROUP

if ! $dry_run; then
    for suffix in pause pause-store-clock64 pause-split-clock64; do
        lib="${nccl_root}/nccl-2.32.3-${suffix}/build/lib/libnccl.so.2"
        [[ -f "$lib" ]] && grep -aFq "NCCL version 2.32.3-${suffix}+cuda" "$lib" \
            || die "missing or wrong ${lib}; run install/install-nccl-2.32.3-clock64.sh"
    done
    [[ -x "$ncu_bin" ]] || die "${ncu_bin} not found; pass --ncu"
    ncu_version="$("$ncu_bin" --version | sed -n 's/.*Version \([0-9]*\.[0-9]*\).*/\1/p' | head -1)"
    [[ "$ncu_version" == 2026.4 ]] || die "expected Nsight Compute 2026.4, got ${ncu_version:-unknown}"
    if [[ "$venv" != none ]]; then
        [[ -f "${venv}/bin/activate" ]] || die "no virtualenv at ${venv}; pass --venv DIR or --venv none"
        venv="$(cd -- "$venv" && pwd -P)"
        # shellcheck source=/dev/null
        source "${venv}/bin/activate"
        export UV_PROJECT_ENVIRONMENT="$venv"
    fi
    mkdir -p -- "$output_dir"
    [[ -z "$(ls -A "$output_dir")" ]] || die "${output_dir} is not empty"
    printf 'profile,pause_cycles,status\n' >"${output_dir}/clock64-runs.csv"
fi

status=0
csv_files=()
for profile in store split; do
    lib="${nccl_root}/nccl-2.32.3-pause-${profile}-clock64/build/lib/libnccl.so.2"
    for pause in "${pauses[@]}"; do
        case_dir="${output_dir}/${profile}/pause-${pause}cyc"
        csv_dir="${case_dir}/clock64-ranks"
        cmd=(env "NCCL_SYM_AG_STMC_PAUSE_CYCLES=${pause}" NCCL_SYM_TMA_ENABLE=0
            "NCCL_SYM_CLOCK64_CSV_PROFILE=${profile}" "NCCL_SYM_CLOCK64_CSV_DIR=${csv_dir}"
            "NCCL_SYM_CLOCK64_CSV_LIB=${lib}" "NCCL_SYM_CLOCK64_STRIDE=${stride}"
            "NCCL_SYM_CLOCK64_GROUP=${group_samples}"
            bash "$launcher" --baseline-only --algo symm --ag-rs --size-mib "$size_mib"
            --nccl-lib "$lib" --output-dir "$case_dir" "${common_args[@]}")
        if $dry_run; then
            printf '%q ' "${cmd[@]}"; printf '\n'
            continue
        fi

        echo "=== ${profile} clock64, pause=${pause}"
        result=fail
        if "${cmd[@]}"; then
            result=ok
            rank_files=()
            for (( rank=0; rank < ${#gpu_list[@]}; ++rank )); do
                file="${csv_dir}/rank-${rank}.csv"
                if [[ ! -s "$file" ]]; then
                    echo "Missing clock64 rank CSV ${file}" >&2
                    result=invalid
                    break
                fi
                rank_files+=("$file")
            done
            if [[ "$result" == ok ]]; then
                csv_files+=("${rank_files[@]}")
            fi
        fi
        printf '%s,%s,%s\n' "$profile" "$pause" "$result" >>"${output_dir}/clock64-runs.csv"
        [[ "$result" == ok ]] || status=1
    done
done

if ! $dry_run && (( ${#csv_files[@]} )); then
    awk 'FNR != 1 || NR == 1' "${csv_files[@]}" >"${output_dir}/clock64.csv"
    echo "=== ${output_dir}/clock64.csv"
fi

echo "=== ncu pause-only baseline"
"${ncu_cmd[@]}" || status=1
exit "$status"
