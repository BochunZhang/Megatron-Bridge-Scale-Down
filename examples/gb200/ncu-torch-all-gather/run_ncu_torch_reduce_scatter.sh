#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
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
Profile the NCCL symmetric ReduceScatter_LDMC kernel (multimem.ld_reduce) on a single GB200 node.
Requires ncu >= 2026.4 for application replay with the shmem communicator, and a torch/NCCL
build with symmetric mempool registration. Run with the desired uv project environment.

Usage: run_ncu_torch_reduce_scatter.sh [options]
  --gpus LIST          One rank per GPU (default: CUDA_VISIBLE_DEVICES, or 0,1,2,3)
  --size-mib N         Reduce-scatter input size per rank in MiB (default: 1024)
  --dtype NAME         bf16 | fp32 (default: bf16)
  --warmup N           Reduce-scatters before the one profiled launch (default: 5)
  --group NAME         all | source | memory | nvlink (default: all, in separate ncu sessions)
  --timeout SEC        Time limit for each ncu session (default: 1800)
  --ncu PATH           ncu binary (default: ncu in PATH, NVIDIA-Nsight-Compute-2026.4 first)
  --nccl-lib PATH      Optional libnccl.so.2 to preload
  --output-dir DIR     Default: results/gb200/ncu-torch-all-gather/symm-reduce-scatter-<time>
  -h, --help

Each group writes reduce_scatter.ncu-repz (or .ncu-rep), summary.csv, ncu.log and NCCL logs.
  source  SourceCounters and source_sass.csv for LDGMC and per-instruction warp stalls.
  memory  Warp stalls, SM/XBAR pressure, load latency, L2 and HBM traffic.
  nvlink  PM-sampled NVLink request, response and total bytes (timeline in the report).
USAGE
}

die() { echo "ERROR: $*" >&2; exit 1; }

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
gpus="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
size_mib=1024 dtype=bf16 warmup=5 group=all timeout_sec=1800 ncu_bin=ncu nccl_lib="" output_dir=""

while (( $# > 0 )); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --gpus|--size-mib|--dtype|--warmup|--group|--timeout|--ncu|--nccl-lib|--output-dir)
            [[ -n "${2:-}" && "$2" != --* ]] || die "$1 needs a value"
            case "$1" in
                --gpus) gpus="$2" ;;
                --size-mib) size_mib="$2" ;;
                --dtype) dtype="$2" ;;
                --warmup) warmup="$2" ;;
                --group) group="$2" ;;
                --timeout) timeout_sec="$2" ;;
                --ncu) ncu_bin="$2" ;;
                --nccl-lib) nccl_lib="$2" ;;
                --output-dir) output_dir="$2" ;;
            esac
            shift 2 ;;
        *) die "unknown argument: $1 (see --help)" ;;
    esac
done

[[ "$gpus" =~ ^[^,[:space:]]+(,[^,[:space:]]+)+$ ]] || die "--gpus needs at least two comma-separated GPU IDs"
IFS=, read -r -a gpu_list <<<"$gpus"
nranks=${#gpu_list[@]}
[[ "$size_mib" =~ ^[1-9][0-9]*$ && "$warmup" =~ ^[1-9][0-9]*$ && "$timeout_sec" =~ ^[1-9][0-9]*$ ]] \
    || die "--size-mib, --warmup and --timeout must be positive integers"
[[ "$dtype" =~ ^(bf16|fp32)$ ]] || die "--dtype must be bf16 or fp32"
case "$group" in
    all) groups=(source memory nvlink) ;;
    source|memory|nvlink) groups=("$group") ;;
    *) die "--group must be all, source, memory or nvlink" ;;
esac

ncu_home="${repo_root}/NVIDIA-Nsight-Compute-2026.4"
[[ -d "$ncu_home" ]] && export PATH="${ncu_home}:${PATH}"
ncu_bin="$(command -v "$ncu_bin")" || die "ncu not found; pass --ncu PATH"
# Resolve paths from the caller's cwd before entering the uv project.
ncu_bin="$(cd -- "$(dirname -- "$ncu_bin")" && pwd)/$(basename -- "$ncu_bin")"
ncu_version="$("$ncu_bin" --version | sed -nE 's/.*Version ([0-9]+\.[0-9]+).*/\1/p')"
IFS=. read -r ncu_year ncu_minor <<<"$ncu_version"
[[ -n "$ncu_version" ]] && (( ncu_year > 2026 || (ncu_year == 2026 && ncu_minor >= 4) )) \
    || die "need ncu >= 2026.4 for application replay with shmem; found ${ncu_version:-unknown}"
command -v uv >/dev/null || die "uv is required"
timeout_bin="$(command -v timeout || command -v gtimeout)" || die "GNU timeout is required"

if [[ -n "$nccl_lib" ]]; then
    [[ -f "$nccl_lib" ]] || die "--nccl-lib does not exist: ${nccl_lib}"
    nccl_lib="$(cd -- "$(dirname -- "$nccl_lib")" && pwd)/$(basename -- "$nccl_lib")"
    export LD_PRELOAD="${nccl_lib}${LD_PRELOAD:+:${LD_PRELOAD}}"
fi

output_dir="${output_dir:-${repo_root}/results/gb200/ncu-torch-all-gather/symm-reduce-scatter-$(date +%Y%m%d-%H%M%S)}"
mkdir -p -- "$output_dir"
output_dir="$(cd -- "$output_dir" && pwd)"
[[ -z "$(ls -A "$output_dir")" ]] || die "${output_dir} is not empty"
cd "$repo_root"

nccl_version="$(uv run --no-sync python -c '
import sys
from torch.distributed import ProcessGroupNCCL

sys.stdout.write(".".join(map(str, ProcessGroupNCCL.get_runtime_nccl_version())))
')" || die "could not query the NCCL library loaded by torch"
IFS=. read -r nccl_major nccl_minor _ <<<"$nccl_version"

kernel=ReduceScatter_LDMC
kernel_re="^ncclSymkDevKernel_${kernel}(_[A-Za-z0-9_]+)?$"
unset NCCL_ALGO NCCL_PROTO NCCL_CTA_POLICY NCCL_SYM_NOWIN_ENABLE NCCL_NVTX_DISABLE
export NCCL_SYM_KERNEL="$kernel"
if (( nccl_major > 2 || (nccl_major == 2 && nccl_minor >= 32) )); then
    export NCCL_SYM_KERNEL="ReduceScatter:${kernel}"
fi
export NCCL_NVLS_ENABLE=1 NCCL_WIN_ENABLE=1 NCCL_SYM_CE_THRESHOLD=9223372036854775807
export CUDA_VISIBLE_DEVICES="$gpus" NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV,TUNING

worker_cmd=(
    uv run --no-sync python -m torch.distributed.run --standalone --nproc_per_node "$nranks"
    "${script_dir}/ncu_torch_reduce_scatter.py" --size-mib "$size_mib" --dtype "$dtype" --warmup "$warmup"
)
(
    set -o noclobber
    {
        echo "ncu=${ncu_bin} ${ncu_version} NCCL=${nccl_version} nccl_lib=${nccl_lib:-<from torch>}"
        echo "size_mib=${size_mib} dtype=${dtype} warmup=${warmup} group=${group}"
        env | grep -E '^(NCCL_|CUDA_VISIBLE_DEVICES=|LD_PRELOAD=)' | sort
    } >"${output_dir}/config.txt"
) || die "could not create ${output_dir}/config.txt"

common_metrics=(gpu__time_duration.sum launch__grid_size launch__block_size sm__cycles_elapsed.avg.per_second)
# From the AG blocking-metrics set, keep the load/response path and the local output store.
memory_metrics=(
    smsp__warps_active.sum
    smsp__warps_issue_stalled_long_scoreboard.sum
    smsp__warps_issue_stalled_lg_throttle.sum
    smsp__warps_issue_stalled_barrier.sum
    smsp__warps_issue_stalled_membar.sum
    sm__average_inst_latency_pipe_lsu_ops_generic_cmd_read.ratio
    sm__average_inst_latency_pipe_lsu_mem_lg_cmd_read.ratio
    l1tex__m_l1tex2xbar_req_cycles_stalled.sum
    l1tex__m_l1tex2xbar_req_cycles_active.sum
    l1tex__m_xbar2l1tex_read_bytes.sum
    l1tex__m_l1tex2xbar_write_bytes.sum
    lrc__lrc2xbar_cycles_stalled.sum
    lrc__lrc2xbar_cycles_active.sum
    syslts__t_sectors_srcnode_gpc_aperture_peer_op_read.sum
    lts__t_sectors_op_read.sum
    lts__t_sectors_op_write.sum
    lts__ltcfabric2lts_cycles_stalled.sum
    lts__ltcfabric2lts_cycles_active.sum
    lts__t_sectors_srcunit_ltcfabric_op_read.sum
    dram__sectors_read.sum
    dram__sectors_write.sum
)
# The kernel profiler's NVLink byte counters reported zero on GB200 + NVSwitch in the AG runs.
nvlink_metrics=(
    pmsampling:nvlrx__bytes.sum
    pmsampling:nvltx__bytes.sum
    pmsampling:nvlrx__bytes_packet_request_data_user.sum
    pmsampling:nvltx__bytes_packet_request_data_user.sum
    pmsampling:nvlrx__bytes_packet_response_data_user.sum
    pmsampling:nvltx__bytes_packet_response_data_user.sum
)

check_tuning() {
    local log_dir="$1" logs=("${1}"/*.log)
    # NCCL logs tuning on communicator rank 0, with a new PID for every replay pass.
    (( ${#logs[@]} > 0 )) || die "no NCCL logs in ${log_dir}"
    awk -v expected="$(( warmup + 1 ))" -v kernel="$kernel" '
        /NCCL INFO ReduceScatter( \[[^]]+\])?: [0-9]+ Bytes -> / {
            counts[FILENAME]++
            if ($0 !~ ("\\[Symmetric\\]: [0-9]+ Bytes -> Kernel " kernel " ")) {
                print FILENAME ": " $0 > "/dev/stderr"
                bad = 1
            }
        }
        END {
            for (file in counts) {
                runs++
                if (counts[file] != expected) bad = 1
            }
            exit (bad || !runs)
        }
    ' "${logs[@]}" || die "expected $(( warmup + 1 )) ${kernel} launches per replay; see ${log_dir}"
}

profile_group() {
    local name="$1" dir="${output_dir}/$1" metrics report status=0
    local collect_args=() selected=("${common_metrics[@]}")
    case "$name" in
        source) collect_args=(--section SourceCounters --import-source yes) ;;
        memory) selected+=("${memory_metrics[@]}") ;;
        nvlink) selected+=("${nvlink_metrics[@]}") ;;
    esac
    metrics="$(IFS=,; echo "${selected[*]}")"
    mkdir -p "${dir}/nccl-logs"
    local cmd=(
        "$ncu_bin" --communicator shmem --communicator-shmem-num-peers "$nranks"
        --replay-mode application --target-processes all --clock-control none
        --nvtx --nvtx-include ncu_reduce_scatter/ --kernel-name "regex:${kernel_re}" --launch-count 1
        ${collect_args[@]+"${collect_args[@]}"} --metrics "$metrics" --export "${dir}/reduce_scatter"
        "${worker_cmd[@]}"
    )
    printf '%q ' "${cmd[@]}" >"${dir}/command.txt"
    printf '\n' >>"${dir}/command.txt"
    echo "=== ${name}: ${dir}"
    NCCL_DEBUG_FILE="${dir}/nccl-logs/nccl-%h-%p.log" \
        "$timeout_bin" --signal=TERM --kill-after=30 "$timeout_sec" "${cmd[@]}" 2>&1 | tee "${dir}/ncu.log" || status=$?
    (( status == 0 )) || die "${name}: capture exited with ${status}; see ${dir}/ncu.log"
    check_tuning "${dir}/nccl-logs"

    report="${dir}/reduce_scatter.ncu-repz"
    [[ -s "$report" ]] || report="${dir}/reduce_scatter.ncu-rep"
    [[ -s "$report" ]] || die "${name}: no ncu report; see ${dir}/ncu.log"
    "$ncu_bin" --import "$report" --page raw --csv --print-units base --print-kernel-base function >"${dir}/summary.csv"
    grep -Eq "\"ncclSymkDevKernel_${kernel}(_[A-Za-z0-9_]+)?\"" "${dir}/summary.csv" \
        || die "${name}: ${kernel} missing from ${dir}/summary.csv"
    if [[ "$name" == source ]]; then
        "$ncu_bin" --import "$report" --page source --print-source sass --csv >"${dir}/source_sass.csv"
        grep -q 'LDGMC\.' "${dir}/source_sass.csv" || die "${name}: no LDGMC instruction in ${dir}/source_sass.csv"
    fi
    echo "=== ${name}: saved ${report}"
}

shopt -s nullglob
for group in "${groups[@]}"; do
    profile_group "$group"
done
