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
Profile the multicast symmetric AllGather kernel (ncclSymkDevKernel_AllGather_STMC, register
multimem.st) of stock NCCL 2.32.3 with Nsight Compute >= 2026.1, on the GPUs of one node.

nccl-tests all_gather_perf (install/install-nccl-tests-2.32.3.sh, build-mpi) runs one MPI process
per GPU. The NCCL kernels of the ranks only make progress together, so ncu's multiprocess
communicator launches and replays each profiled kernel on all ranks in lockstep:
  shmem  one ncu wraps mpirun and sees every rank in its process tree (default)
  tcp    mpirun starts one ncu per rank; they meet at 127.0.0.1:<port>

The kernel is forced with:
  -R 2                                   symmetric window registration of send/recv buffers
  NCCL_SYM_KERNEL=AllGather:AllGather_STMC
  NCCL_SYM_CE_THRESHOLD=INT64_MAX        NCCL_SYM_KERNEL does not stop the Blackwell CE AllGather
                                         route for large counts (enqueue.cc); only this does
  NCCL_NVLS_ENABLE=1 NCCL_WIN_ENABLE=1

nccl-tests runs every size out-of-place and then in-place, each -w warmup + -n timed + -c check
launches. Two passes, each checked against NCCL's TUNING log ("AllGather [Symmetric]: <n> Bytes -> Kernel
AllGather_STMC" for every launch, nothing else):
  1. baseline  no ncu, with data check: proves the kernel is selectable and the result correct
  2. ncu       the profiled run; a hang past --timeout is reported as such

Usage:
  bash examples/gb200/nccl-tests/run_ncu_all_gather_stmc.sh [options] [-- extra ncu args]

Options:
  --gpus LIST            CUDA_VISIBLE_DEVICES, one rank per GPU (default: 0,1,2,3)
  --size BYTES           AllGather output size, nccl-tests -b/-e syntax; each rank sends size/n
                         (default: 256M)
  --datatype TYPE        nccl-tests -d (default: bfloat16)
  --communicator MODE    shmem | tcp (default: shmem)
  --tcp-port N           Port of the tcp communicator (default: ncu's own)
  --set NAME             ncu --set (default: full)
  --replay-mode MODE     ncu --replay-mode: application | kernel (default: application). kernel
                         replay saves and restores the memory the kernel can reach before every
                         pass, which fails for the symmetric windows ("Failed to save memory for
                         replay"); application relaunches all ranks once per pass instead
  --launch-skip N        STMC launches per rank to skip before profiling (default: 1, the warmup)
  --launch-count N       STMC launches to profile (default: 1)
  --timeout SEC          Kill a pass after SEC seconds and report a hang (default: 1800)
  --ncu PATH             ncu binary (default: ncu in PATH)
  --mpirun PATH          mpirun binary (default: mpirun in PATH)
  --nccl-home DIR        NCCL 2.32.3 build (default: <megatron-bridge>/nccl/nccl-2.32.3/build)
  --nccl-tests DIR       nccl-tests checkout (default: <megatron-bridge>/nccl/nccl-tests)
  --output-dir DIR       (default: results/gb200/nccl-tests/ncu-all-gather-stmc/<time>)
  --skip-baseline        Only run the ncu pass
  -h, --help

Outputs in --output-dir: baseline.log, ncu.log, nccl-logs/<pass>/*.log, the ncu report
(ag_stmc.ncu-rep for shmem, ag_stmc.rank<R>.ncu-rep for tcp) and summary.csv
(one row per profiled kernel: rank/device, grid, duration).
USAGE
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"

gpus="0,1,2,3"
size="256M"
datatype="bfloat16"
communicator="shmem"
tcp_port=""
ncu_set="full"
replay_mode="application"
launch_skip=1
launch_count=1
timeout_sec=1800
ncu_bin="ncu"
mpirun_bin="mpirun"
nccl_home="${repo_root}/nccl/nccl-2.32.3/build"
nccl_tests="${repo_root}/nccl/nccl-tests"
output_dir=""
run_baseline=true
extra_ncu_args=()

die() {
    echo "ERROR: $*" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus) gpus="$2"; shift 2 ;;
        --size) size="$2"; shift 2 ;;
        --datatype) datatype="$2"; shift 2 ;;
        --communicator) communicator="$2"; shift 2 ;;
        --tcp-port) tcp_port="$2"; shift 2 ;;
        --set) ncu_set="$2"; shift 2 ;;
        --replay-mode) replay_mode="$2"; shift 2 ;;
        --launch-skip) launch_skip="$2"; shift 2 ;;
        --launch-count) launch_count="$2"; shift 2 ;;
        --timeout) timeout_sec="$2"; shift 2 ;;
        --ncu) ncu_bin="$2"; shift 2 ;;
        --mpirun) mpirun_bin="$2"; shift 2 ;;
        --nccl-home) nccl_home="$2"; shift 2 ;;
        --nccl-tests) nccl_tests="$2"; shift 2 ;;
        --output-dir) output_dir="$2"; shift 2 ;;
        --skip-baseline) run_baseline=false; shift ;;
        --) shift; extra_ncu_args=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1 (see --help)" ;;
    esac
done

[[ "$communicator" == shmem || "$communicator" == tcp ]] || die "--communicator must be shmem or tcp"
[[ "$replay_mode" == application || "$replay_mode" == kernel ]] || die "--replay-mode must be application or kernel"
IFS=, read -r -a gpu_list <<<"$gpus"
nranks=${#gpu_list[@]}
(( nranks >= 2 )) || die "--gpus needs at least two GPUs"
output_dir="${output_dir:-${repo_root}/results/gb200/nccl-tests/ncu-all-gather-stmc/$(date +%Y%m%d-%H%M%S)}"

################################################################################
# Preflight

bench="${nccl_tests}/build-mpi/all_gather_perf"
[[ -x "$bench" ]] || die "${bench} not found; run ./install/install-nccl-tests-2.32.3.sh (needs MPI)"
[[ -f "${nccl_home}/lib/libnccl.so" ]] || die "no libnccl.so in ${nccl_home}/lib; pass --nccl-home"
command -v "$mpirun_bin" >/dev/null || die "${mpirun_bin} not found; pass --mpirun"
command -v "$ncu_bin" >/dev/null || die "${ncu_bin} not found; pass --ncu"

ncu_version="$("$ncu_bin" --version | sed -n 's/.*Version \([0-9]*\.[0-9]*\).*/\1/p' | head -1)"
echo "ncu: $(command -v "$ncu_bin") (${ncu_version:-unknown version})"
ncu_year="${ncu_version%%.*}"
ncu_minor="${ncu_version#*.}"
if [[ -z "$ncu_version" ]] || (( ncu_year < 2026 || (ncu_year == 2026 && ncu_minor < 1) )); then
    die "Nsight Compute ${ncu_version:-?} has no multiprocess communicator; need >= 2026.1"
fi

if command -v nvidia-smi >/dev/null; then
    ngpu="$(nvidia-smi -L | wc -l | tr -d ' ')"
    for g in "${gpu_list[@]}"; do
        (( g < ngpu )) || die "GPU ${g} does not exist (this node has ${ngpu})"
    done
fi

# Open MPI refuses root unless told otherwise.
mpirun_args=(-np "$nranks")
if "$mpirun_bin" --version 2>&1 | grep -qi "open mpi\|open-rte\|prrte"; then
    mpirun_args+=(--bind-to none)
    (( EUID == 0 )) && mpirun_args+=(--allow-run-as-root)
fi

mkdir -p "$output_dir"
[[ -z "$(ls -A "$output_dir")" ]] || die "${output_dir} is not empty"

# Every rank inherits mpirun's environment on a single node. Drop the per-process selectors that
# would compete with NCCL_SYM_KERNEL.
unset NCCL_ALGO NCCL_PROTO NCCL_CTA_POLICY NCCL_CE_COLL_AG_MULTICAST_THRESHOLD
export CUDA_VISIBLE_DEVICES="$gpus"
export LD_LIBRARY_PATH="${nccl_home}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export NCCL_NVLS_ENABLE=1
export NCCL_WIN_ENABLE=1
export NCCL_SYM_KERNEL="AllGather:AllGather_STMC"
export NCCL_SYM_CE_THRESHOLD=9223372036854775807
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV,TUNING

# -w warmup, -n timed iterations, -c data-check iterations; every one launches one AllGather.
bench_args=(-b "$size" -e "$size" -d "$datatype" -g 1 -R 2)

{
    echo "nccl_home=${nccl_home}"
    echo "nccl_tests=${nccl_tests} ($(git -C "$nccl_tests" log -1 --format='%h %cd' --date=short 2>/dev/null || echo ?))"
    echo "ncu=$(command -v "$ncu_bin") ${ncu_version}"
    echo "gpus=${gpus} size=${size} datatype=${datatype} communicator=${communicator} set=${ncu_set} replay=${replay_mode}"
    echo "launch_skip=${launch_skip} launch_count=${launch_count} extra_ncu_args=${extra_ncu_args[*]:-}"
    env | grep -E '^(NCCL_|CUDA_VISIBLE_DEVICES=)' | sort
} >"${output_dir}/config.txt"
cat "${output_dir}/config.txt"

# Every AllGather in the TUNING log of the pass must be AllGather_STMC, and there must be
# <expected> of them per application run. NCCL prints the line on comm rank 0 only, into one log
# per process; application replay starts the ranks again for every ncu pass, so <runs> is the
# number of rank-0 logs.
check_tuning() {
    local pass="$1" expected="$2" logs ag stmc total runs
    logs=("${output_dir}/nccl-logs/${pass}"/*.log)
    ag="$(cat "${logs[@]}" | grep -E 'NCCL INFO AllGather( \[[^]]*\])?: [0-9]+ Bytes -> ' || true)"
    stmc="$(grep -c 'AllGather \[Symmetric\]: [0-9]* Bytes -> Kernel AllGather_STMC' <<<"$ag" || true)"
    total="$(grep -c . <<<"$ag" || true)"
    runs="$(grep -lE 'NCCL INFO AllGather( \[[^]]*\])?: [0-9]+ Bytes -> ' "${logs[@]}" | wc -l | tr -d ' ')"
    if (( runs == 0 || total != stmc || total != runs * expected )); then
        echo "${pass}: expected ${expected} AllGather_STMC launches per run in the TUNING log (${runs} run(s)), got:" >&2
        sort <<<"$ag" | sed 's/.*NCCL INFO //; s/ Bytes -> \(\S* \S*\).*/ Bytes -> \1/' | uniq -c >&2
        die "${pass}: NCCL did not run AllGather_STMC for every launch; logs in ${output_dir}/nccl-logs/${pass}"
    fi
    echo "${pass}: ${stmc}/${total} AllGather launches are AllGather_STMC (${runs} run(s) x ${expected})"
}

# run_pass <name> <expected AllGather count per run> <cmd...>
run_pass() {
    local pass="$1" expected="$2"; shift 2
    mkdir -p "${output_dir}/nccl-logs/${pass}"
    echo "=== ${pass}: $*"
    local status=0
    NCCL_DEBUG_FILE="${output_dir}/nccl-logs/${pass}/nccl-%h-%p.log" \
        timeout --signal=TERM --kill-after=30 "$timeout_sec" "$@" 2>&1 | tee "${output_dir}/${pass}.log" \
        || status=${PIPESTATUS[0]}
    if (( status == 124 || status == 137 )); then
        die "${pass} did not finish in ${timeout_sec}s (hang); see ${output_dir}/${pass}.log"
    fi
    if grep -q "ERR_NVGPUCTRPERM" "${output_dir}/${pass}.log"; then
        die "${pass}: no permission for GPU performance counters; run as root or load nvidia with NVreg_RestrictProfilingToAdminUsers=0"
    fi
    if grep -q "Failed to save memory for replay" "${output_dir}/${pass}.log"; then
        die "${pass}: ncu could not save the kernel's memory for kernel replay; use --replay-mode application (see the /tmp/nsight-compute-*.log named in ${output_dir}/${pass}.log)"
    fi
    (( status == 0 )) || die "${pass} exited with ${status}; see ${output_dir}/${pass}.log"
    grep -q "Out of bounds values : 0 OK" "${output_dir}/${pass}.log" \
        || die "${pass}: nccl-tests did not report 'Out of bounds values : 0 OK'"
    check_tuning "$pass" "$expected"
}

################################################################################
# 1. Baseline: no ncu, data check on

if $run_baseline; then
    # nccl-tests runs out-of-place, then in-place, each 5 warmup + 20 timed + 1 check launches.
    run_pass baseline 52 "$mpirun_bin" "${mpirun_args[@]}" "$bench" "${bench_args[@]}" -w 5 -n 20 -c 1
fi

################################################################################
# 2. ncu

# Launch only what is profiled: warmups cover --launch-skip, one timed iteration per profiled
# launch, no data check (it would add launches the counters do not need). The profiled launches
# are in the out-of-place pass; the in-place pass that follows repeats the same count unprofiled.
ncu_iters=(-w "$launch_skip" -n "$launch_count" -c 0)
ncu_args=(
    --set "$ncu_set"
    --replay-mode "$replay_mode"
    --kernel-name "regex:^ncclSymkDevKernel_AllGather_STMC"
    --launch-skip "$launch_skip"
    --launch-count "$launch_count"
    --target-processes all
    --force-overwrite
    "${extra_ncu_args[@]}"
)

if [[ "$communicator" == shmem ]]; then
    report="${output_dir}/ag_stmc"
    run_pass ncu $(( 2 * (launch_skip + launch_count) )) \
        "$ncu_bin" --communicator shmem --communicator-shmem-num-peers "$nranks" "${ncu_args[@]}" \
        --export "$report" \
        "$mpirun_bin" "${mpirun_args[@]}" "$bench" "${bench_args[@]}" "${ncu_iters[@]}"
    reports=("${report}.ncu-rep")
else
    tcp_args=(--communicator tcp --communicator-tcp-hostname 127.0.0.1 --communicator-tcp-num-peers "$nranks"
        --lockstep-kernel-launch)
    [[ -n "$tcp_port" ]] && tcp_args+=(--communicator-tcp-port "$tcp_port")
    # One ncu per rank; the report name carries the MPI rank.
    run_pass ncu $(( 2 * (launch_skip + launch_count) )) \
        "$mpirun_bin" "${mpirun_args[@]}" bash -c \
        'rank="${OMPI_COMM_WORLD_RANK:-${PMIX_RANK:-${PMI_RANK:?no MPI rank in the environment}}}"
         report="$1"; ncu="$2"; shift 2
         exec "$ncu" -o "${report}.rank${rank}" "$@"' \
        rank-wrapper "${output_dir}/ag_stmc" "$ncu_bin" "${tcp_args[@]}" "${ncu_args[@]}" \
        "$bench" "${bench_args[@]}" "${ncu_iters[@]}"
    reports=()
    for (( r = 0; r < nranks; r++ )); do reports+=("${output_dir}/ag_stmc.rank${r}.ncu-rep"); done
fi

################################################################################
# Summary: one row per profiled kernel

profiled=0
summary="${output_dir}/summary.csv"
: >"$summary"
for rep in "${reports[@]}"; do
    [[ -f "$rep" ]] || die "ncu wrote no ${rep}; see ${output_dir}/ncu.log"
    rows="$("$ncu_bin" --import "$rep" --page raw --csv \
        --metrics launch__grid_size,launch__block_size,gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed)"
    # Line 1 is the header, line 2 the units; keep one header in summary.csv.
    if [[ ! -s "$summary" ]]; then
        head -n 2 <<<"$rows" >"$summary"
    fi
    tail -n +3 <<<"$rows" >>"$summary"
    profiled=$(( profiled + $(tail -n +3 <<<"$rows" | grep -c . || true) ))
done
expected_profiled=$(( nranks * launch_count ))
echo "=== ${profiled} kernel result(s) in ${reports[*]} (expected ${expected_profiled}: ${nranks} ranks x ${launch_count})"
column -s, -t <"$summary" 2>/dev/null || cat "$summary"
(( profiled == expected_profiled )) \
    || echo "WARNING: --launch-count may count launches over all ranks in ${communicator} mode; compare with ncu.log" >&2
echo "Open in the Nsight Compute GUI: ${reports[*]}"
