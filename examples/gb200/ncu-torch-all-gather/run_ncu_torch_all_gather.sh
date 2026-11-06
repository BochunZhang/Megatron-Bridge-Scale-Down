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
Profile the NCCL kernel of a torch all-gather with Nsight Compute >= 2026.1 on one node.

One ncu wraps torchrun (the NVIDIA docs' pattern):
  ncu --communicator shmem --communicator-shmem-num-peers <n> ... torchrun --nproc_per_node=<n> ncu_torch_all_gather.py
The shmem communicator sees every rank in its process tree and launches / replays each selected
kernel on all of them together; without it the first NCCL kernel waits forever for its peers.
torchrun itself is excluded from profiling by ncu.

Algorithms (--algo), each checked against NCCL's TUNING log for every all-gather launch:
  ring   NCCL_ALGO=Ring, plain buffers      kernel ncclDevKernel_AllGather_RING_LL
  nvls   NCCL_ALGO=NVLS, plain buffers      kernel ncclDevKernel_AllGather_RING_LL (same symbol:
                                            NCCL builds one AllGather kernel and dispatches the
                                            algorithm inside it)
  symm   symmetric windows, AllGather_STMC  kernel ncclSymkDevKernel_AllGather_STMC

What is profiled: the worker runs --warmup all-gathers, then --iters all-gathers inside the NVTX
range "ncu_all_gather". ncu selects with --nvtx --nvtx-include "ncu_all_gather/" and
  kernel / application replay   -k regex:^nccl(Symk)?DevKernel_AllGather, --launch-count <iters>
  app-range replay              the NVTX range itself (one result per rank), --launch-count 1

Two passes:
  1. baseline  no ncu: the worker checks the gathered data, the TUNING log the algorithm
  2. ncu       the profiled run; a hang past --timeout is reported as such

Usage:
  bash examples/gb200/ncu-torch-all-gather/run_ncu_torch_all_gather.sh [options] [-- extra ncu args]

Options:
  --gpus LIST            CUDA_VISIBLE_DEVICES, one rank per GPU (default: 0,1,2,3)
  --algo NAME            ring | nvls | symm (default: nvls)
  --size-mib N           All-gather output size per rank in MiB (default: 256)
  --dtype NAME           bf16 | fp32 (default: bf16)
  --warmup N             All-gathers before the NVTX range (default: 5)
  --iters N              All-gathers inside the NVTX range (default: 1)
  --communicator MODE    shmem | none (default: shmem); none is the hang reproducer
  --set NAME             ncu --set (default: basic)
  --replay-mode MODE     kernel | application | app-range (default: kernel)
                           kernel       saves and restores the memory the kernel can reach before
                                        every pass; known to fail for symm windows ("Failed to
                                        save memory for replay", see ../nccl-tests/README.md)
                           application  relaunches the whole torchrun tree once per pass
                           app-range    relaunches once per pass and profiles the NVTX range
  --timeout SEC          Kill a pass after SEC seconds and report a hang (default: 1800)
  --ncu PATH             ncu binary (default: ncu in PATH)
  --nccl-lib PATH        LD_PRELOAD this libnccl.so.2 instead of the one bundled with torch
  --output-dir DIR       (default: results/gb200/ncu-torch-all-gather/<algo>-<time>)
  --skip-baseline        Only run the ncu pass
  -h, --help

Outputs in --output-dir: config.txt, baseline.log, ncu.log, nccl-logs/<pass>/*.log, the ncu
report all_gather.ncu-rep and summary.csv (one row per profiled result).
USAGE
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"

gpus="0,1,2,3"
algo="nvls"
size_mib=256
dtype="bf16"
warmup=5
iters=1
communicator="shmem"
ncu_set="basic"
replay_mode="kernel"
timeout_sec=1800
ncu_bin="ncu"
nccl_lib=""
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
        --algo) algo="$2"; shift 2 ;;
        --size-mib) size_mib="$2"; shift 2 ;;
        --dtype) dtype="$2"; shift 2 ;;
        --warmup) warmup="$2"; shift 2 ;;
        --iters) iters="$2"; shift 2 ;;
        --communicator) communicator="$2"; shift 2 ;;
        --set) ncu_set="$2"; shift 2 ;;
        --replay-mode) replay_mode="$2"; shift 2 ;;
        --timeout) timeout_sec="$2"; shift 2 ;;
        --ncu) ncu_bin="$2"; shift 2 ;;
        --nccl-lib) nccl_lib="$2"; shift 2 ;;
        --output-dir) output_dir="$2"; shift 2 ;;
        --skip-baseline) run_baseline=false; shift ;;
        --) shift; extra_ncu_args=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1 (see --help)" ;;
    esac
done

[[ "$algo" =~ ^(ring|nvls|symm)$ ]] || die "--algo must be ring, nvls or symm"
[[ "$dtype" =~ ^(bf16|fp32)$ ]] || die "--dtype must be bf16 or fp32"
[[ "$communicator" =~ ^(shmem|none)$ ]] || die "--communicator must be shmem or none"
[[ "$replay_mode" =~ ^(kernel|application|app-range)$ ]] || die "--replay-mode must be kernel, application or app-range"
(( iters >= 1 )) || die "--iters must be at least 1"
IFS=, read -r -a gpu_list <<<"$gpus"
nranks=${#gpu_list[@]}
(( nranks >= 2 )) || die "--gpus needs at least two GPUs"
output_dir="${output_dir:-${repo_root}/results/gb200/ncu-torch-all-gather/${algo}-$(date +%Y%m%d-%H%M%S)}"

################################################################################
# Preflight

command -v "$ncu_bin" >/dev/null || die "${ncu_bin} not found; pass --ncu"
ncu_version="$("$ncu_bin" --version | sed -n 's/.*Version \([0-9]*\.[0-9]*\).*/\1/p' | head -1)"
echo "ncu: $(command -v "$ncu_bin") (${ncu_version:-unknown version})"
ncu_year="${ncu_version%%.*}"
ncu_minor="${ncu_version#*.}"
if [[ -z "$ncu_version" ]] || (( ncu_year < 2026 || (ncu_year == 2026 && ncu_minor < 1) )); then
    die "Nsight Compute ${ncu_version:-?} cannot profile NCCL kernels with --communicator shmem; need >= 2026.1"
fi
[[ -z "$nccl_lib" || -f "$nccl_lib" ]] || die "--nccl-lib ${nccl_lib} does not exist"

if command -v nvidia-smi >/dev/null; then
    ngpu="$(nvidia-smi -L | wc -l | tr -d ' ')"
    for g in "${gpu_list[@]}"; do
        (( g < ngpu )) || die "GPU ${g} does not exist (this node has ${ngpu})"
    done
fi

mkdir -p "$output_dir"
[[ -z "$(ls -A "$output_dir")" ]] || die "${output_dir} is not empty"

# Drop inherited selectors that would compete with the per-algorithm settings below.
unset NCCL_ALGO NCCL_PROTO NCCL_CTA_POLICY NCCL_SYM_KERNEL NCCL_SYM_CTAS NCCL_SYM_CE_THRESHOLD \
    NCCL_CE_COLL_AG_MULTICAST_THRESHOLD NCCL_WIN_ENABLE NCCL_NVTX_DISABLE
case "$algo" in
    ring)
        export NCCL_ALGO=Ring
        tuning_re='AllGather: [0-9]+ Bytes -> Algo Ring '
        ;;
    nvls)
        export NCCL_ALGO=NVLS NCCL_NVLS_ENABLE=1
        tuning_re='AllGather: [0-9]+ Bytes -> Algo NVLS '
        ;;
    symm)
        # Window registration selects the symmetric kernels, not NCCL_ALGO. NCCL_SYM_KERNEL does
        # not stop large AllGathers going to the copy engines on Blackwell (enqueue.cc); only an
        # INT64_MAX NCCL_SYM_CE_THRESHOLD does. The "<func>:<kernel>" syntax needs NCCL >= 2.32.
        export NCCL_NVLS_ENABLE=1 NCCL_WIN_ENABLE=1
        export NCCL_SYM_KERNEL="AllGather:AllGather_STMC"
        export NCCL_SYM_CE_THRESHOLD=9223372036854775807
        tuning_re='AllGather \[Symmetric\]: [0-9]+ Bytes -> Kernel AllGather_STMC '
        ;;
esac
export CUDA_VISIBLE_DEVICES="$gpus"
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV,TUNING
[[ -n "$nccl_lib" ]] && export LD_PRELOAD="${nccl_lib}${LD_PRELOAD:+:${LD_PRELOAD}}"

# --standalone picks a free rendezvous port, so application replay can relaunch torchrun at once.
torchrun_cmd=(
    uv run --no-sync python -m torch.distributed.run --standalone --nproc_per_node "$nranks"
    "${script_dir}/ncu_torch_all_gather.py"
    --algo "$algo" --size-mib "$size_mib" --dtype "$dtype" --warmup "$warmup" --iters "$iters"
)

{
    echo "ncu=$(command -v "$ncu_bin") ${ncu_version}"
    echo "gpus=${gpus} algo=${algo} size_mib=${size_mib} dtype=${dtype} warmup=${warmup} iters=${iters}"
    echo "communicator=${communicator} set=${ncu_set} replay=${replay_mode} extra_ncu_args=${extra_ncu_args[*]:-}"
    echo "nccl_lib=${nccl_lib:-<bundled with torch>}"
    env | grep -E '^(NCCL_|CUDA_VISIBLE_DEVICES=|LD_PRELOAD=)' | sort
} >"${output_dir}/config.txt"
cat "${output_dir}/config.txt"
cd "$repo_root"

# Every AllGather in the TUNING log of the pass must match tuning_re, warmup + iters per run.
# NCCL prints the line on comm rank 0 only, one log per process; application and app-range replay
# start torchrun again for every ncu pass, so <runs> is the number of logs that contain one.
check_tuning() {
    local pass="$1" expected=$(( warmup + iters )) logs ag ok total runs
    logs=("${output_dir}/nccl-logs/${pass}"/*.log)
    ag="$(cat "${logs[@]}" | grep -E 'NCCL INFO AllGather( \[[^]]*\])?: [0-9]+ Bytes -> ' || true)"
    ok="$(grep -cE "$tuning_re" <<<"$ag" || true)"
    total="$(grep -c . <<<"$ag" || true)"
    runs="$(grep -lE 'NCCL INFO AllGather( \[[^]]*\])?: [0-9]+ Bytes -> ' "${logs[@]}" | wc -l | tr -d ' ')"
    if (( runs == 0 || total != ok || total != runs * expected )); then
        echo "${pass}: expected ${expected} launches matching /${tuning_re}/ per run (${runs} run(s)), got:" >&2
        sort <<<"$ag" | sed 's/.*NCCL INFO //; s/ Bytes -> \(\S* \S*\).*/ Bytes -> \1/' | uniq -c >&2
        die "${pass}: NCCL did not run --algo ${algo} for every all-gather; logs in ${output_dir}/nccl-logs/${pass}"
    fi
    echo "${pass}: ${ok}/${total} AllGather launches match --algo ${algo} (${runs} run(s) x ${expected})"
}

# run_pass <name> <cmd...>
run_pass() {
    local pass="$1"; shift
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
        die "${pass}: ncu could not save the kernel's memory for kernel replay; try --replay-mode application or app-range (see the /tmp/nsight-compute-*.log named in ${output_dir}/${pass}.log)"
    fi
    if grep -q "No kernels were profiled\|No ranges were profiled" "${output_dir}/${pass}.log"; then
        die "${pass}: ncu profiled nothing (communicator=${communicator}, replay=${replay_mode}); see ${output_dir}/${pass}.log"
    fi
    (( status == 0 )) || die "${pass} exited with ${status}; see ${output_dir}/${pass}.log"
    check_tuning "$pass"
}

################################################################################
# 1. Baseline: no ncu

if $run_baseline; then
    run_pass baseline "${torchrun_cmd[@]}"
fi

################################################################################
# 2. ncu

# The worker's NVTX range holds exactly the --iters all-gathers. The kernel name filter keeps the
# warmup and any non-NCCL kernel out even if a future torch launches one inside the range.
select_args=(--nvtx --nvtx-include "ncu_all_gather/")
if [[ "$replay_mode" == app-range ]]; then
    select_args+=(--launch-count 1)
    metrics="gpu__time_duration.sum"
    expected_profiled=$nranks
else
    select_args+=(--kernel-name "regex:^nccl(Symk)?DevKernel_AllGather" --launch-count "$iters")
    metrics="launch__grid_size,launch__block_size,gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed"
    expected_profiled=$(( nranks * iters ))
fi
comm_args=(--communicator none)
[[ "$communicator" == shmem ]] && comm_args=(--communicator shmem --communicator-shmem-num-peers "$nranks")
report="${output_dir}/all_gather"

run_pass ncu "$ncu_bin" "${comm_args[@]}" \
    --set "$ncu_set" \
    --replay-mode "$replay_mode" \
    "${select_args[@]}" \
    --target-processes all \
    --force-overwrite \
    --export "$report" \
    "${extra_ncu_args[@]}" \
    "${torchrun_cmd[@]}"

################################################################################
# Summary: one row per profiled kernel (or range)

[[ -f "${report}.ncu-rep" ]] || die "ncu wrote no ${report}.ncu-rep; see ${output_dir}/ncu.log"
rows="$("$ncu_bin" --import "${report}.ncu-rep" --page raw --csv --metrics "$metrics")"
summary="${output_dir}/summary.csv"
# Line 1 is the header, line 2 the units.
printf '%s\n' "$rows" >"$summary"
profiled="$(tail -n +3 "$summary" | grep -c . || true)"
echo "=== ${profiled} result(s) in ${report}.ncu-rep (expected ${expected_profiled})"
column -s, -t <"$summary" 2>/dev/null || cat "$summary"
(( profiled == expected_profiled )) \
    || echo "WARNING: --launch-count may count over all ranks in shmem mode; compare with ${output_dir}/ncu.log" >&2
echo "Open in the Nsight Compute GUI: ${report}.ncu-rep"
