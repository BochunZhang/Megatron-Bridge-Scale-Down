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
Profile the NCCL kernel of a torch all-gather with Nsight Compute >= 2026.4 on one node, or an
all-gather and a reduce-scatter running at the same time (--ag-rs, meant for a range replay).

ncu: <repo>/NVIDIA-Nsight-Compute-2026.4 is put first in PATH when it exists; --ncu overrides.
2026.4 is the first version with application replay under the shmem communicator.

--communicator shmem (default): one ncu wraps torchrun (the NVIDIA docs' pattern):
  ncu --communicator shmem --communicator-shmem-num-peers <n> ... torchrun --nproc_per_node=<n> ncu_torch_all_gather.py
The shmem communicator sees every rank in its process tree and launches / replays each selected
kernel on all of them together; without it the first NCCL kernel waits forever for its peers.
torchrun itself is excluded from profiling by ncu. The 2026.4 CLI docs say shmem "only supports
kernel and range replay modes"; the 2026.4 release notes add application replay. app-range is in
neither, and shmem + app-range fails at the range's start ("Failed to profile "range"").

--communicator tcp: one ncu per rank (ncu_tcp_ranks.sh), the docs' MPI pattern, without torchrun
(whose agent store would outlive the relaunches of application / app-range replay):
  RANK=<r> ... ncu --communicator tcp --communicator-tcp-num-peers <n> --lockstep-kernel-launch \
      --lockstep-nvtx-include <range>/ ... -o <report>-rank<r> python ncu_torch_all_gather.py
Each rank is a process tree of its own, so ncu writes one report per rank. --no-lockstep drops
the two --lockstep-* options.

Algorithms (--algo), each checked against NCCL's TUNING log for every launch of every op:
  ring   ring all-gather                   NCCL_ALGO=Ring, plain buffers
  nvls   NVLS multicast all-gather         NCCL_ALGO=NVLS, plain buffers
  symm   symmetric kernel, multicast       AllGather_STMC on registered symmetric windows
  p2p    symmetric kernel, peer-to-peer    AllGather_ST on registered symmetric windows,
                                           NCCL_NVLS_ENABLE=0 (no multicast mappings at all)
Kernels: ring and nvls launch ncclDevKernel_AllGather_RING[_<proto>] (NCCL builds one AllGather
kernel and dispatches the algorithm inside it), symm and p2p ncclSymkDevKernel_AllGather_STMC /
_ST. ring keeps an inherited NCCL_NVLS_ENABLE; run it with NCCL_NVLS_ENABLE=0 for a communicator
without multicast mappings. symm and p2p pin the kernel with NCCL_SYM_KERNEL, whose syntax
depends on the NCCL that torch loads (probed once): "AllGather_ST" before 2.32,
"AllGather:AllGather_ST" from 2.32 on.

Two cases: the all-gather alone (ncu_torch_all_gather.py, the default), or with --ag-rs an
all-gather and a reduce-scatter launched together on two CUDA streams (ncu_torch_ag_rs_overlap.py,
one NCCL communicator each). The reduce-scatter runs
  ring / nvls   the same NCCL_ALGO               kernel ncclDevKernel_ReduceScatter_*
  symm          ReduceScatter_LDMC (multicast)   kernel ncclSymkDevKernel_ReduceScatter_LDMC
  p2p           ReduceScatter_LD (unicast)       kernel ncclSymkDevKernel_ReduceScatter_LD
Before NCCL 2.32, NCCL_SYM_KERNEL holds one kernel name, so with --ag-rs on symm / p2p it stays unset
and NCCL's own choice is checked against the kernels above; from 2.32 both are pinned
("AllGather:<k>;ReduceScatter:<k>"). Only range and app-range replay keep the two kernels
concurrent: kernel and application replay profile one kernel at a time and serialize it.

Multicast memory and kernel replay: a kernel replay of more than one pass first saves all memory
the kernel can reach. With nvls (NCCL's NVLS buffers) and symm (the registered windows) that
includes multicast-mapped memory, and ncu 2026.3 fails there with "Failed to save memory for
replay" (ContextSaveFailed). Ways around it:
  --metrics LIST                 metrics that fit one pass: ncu saves no memory for a single-pass
                                 kernel replay (ncu.log shows "- 1 pass")
  --replay-mode application|app-range
                                 no memory is saved; with --communicator shmem needs ncu >= 2026.4
range replay saves and restores memory like kernel replay, and the docs list Virtual Memory
Management, peer context access and cuLaunchKernelEx as unsupported inside the range.

What is profiled: the worker runs --warmup launches, then --iters launches inside an NVTX range
("ncu_all_gather"; with --ag-rs "ncu_collectives", each launch one AG + one RS). ncu selects with
--nvtx-include and
  kernel / application replay   -k regex:^nccl(Symk)?DevKernel_AllGather (with --ag-rs
                                ..._(AllGather|ReduceScatter)), --launch-count <iters> (x 2)
  range / app-range replay      the NVTX range itself (one result per rank), --launch-count 1

Two passes:
  1. baseline  no ncu: the worker checks the gathered data, the TUNING log the algorithm
  2. ncu       the profiled run; a hang past --timeout is reported as such

Usage:
  bash examples/gb200/ncu-torch-all-gather/run_ncu_torch_all_gather.sh [options] [-- extra ncu args]

Options:
  --gpus LIST            CUDA_VISIBLE_DEVICES, one rank per GPU (default: 0,1,2,3)
  --algo NAME            ring | nvls | symm | p2p (default: nvls)
  --ag-rs                Run the all-gather and a reduce-scatter together (needs --warmup >= 1)
  --size-mib N           All-gather output (= reduce-scatter input) size per rank in MiB (default: 256)
  --dtype NAME           bf16 | fp32 (default: bf16)
  --warmup N             All-gathers before the NVTX range (default: 5)
  --iters N              All-gathers inside the NVTX range (default: 1)
  --communicator MODE    shmem | tcp | none (default: shmem); none is the hang reproducer
  --no-lockstep          With --communicator tcp: leave out --lockstep-kernel-launch and
                         --lockstep-nvtx-include
  --set NAME             ncu --set (default: basic)
  --section NAME         Collect this ncu section instead of --set; repeat for several sections.
                         gpu__time_duration.sum is added for the summary unless --metrics is given
  --metrics LIST         Collect only these comma-separated metrics instead of --set (together
                         with --section: the sections plus these metrics)
  --blocking-metrics     Collect where the data path pushes back, stage by stage: sender SM -> XBAR
                         (stalled / active, memory warp stalls, load latency), NVLink (request /
                         response / wire bytes, PM-sampled), receiver L2 (XBAR -> LTS, writes from
                         outside, LTC fabric stalls) and HBM (read / write activity). Added to
                         --metrics (replaces --set), with --clock-control none unless given after
                         "--"; writes blocking.txt (analyse_blocking_metrics.py). Needs several
                         passes, so nvls / symm need --replay-mode application or app-range
  --replay-mode MODE     kernel | application | range | app-range (default: kernel)
                           kernel       saves and restores the memory the kernel can reach when
                                        more than one pass is needed; fails for nvls and symm
                                        (see "Multicast memory and kernel replay" above)
                           application  relaunches the whole torchrun tree once per pass
                           range        captures the NVTX range and replays it in the process
                           app-range    relaunches once per pass and profiles the NVTX range
  --timeout SEC          Kill a pass after SEC seconds and report a hang (default: 1800)
  --ncu PATH             ncu binary (default: ncu in PATH, NVIDIA-Nsight-Compute-2026.4 first)
  --nccl-lib PATH        LD_PRELOAD this libnccl.so.2 instead of the one bundled with torch
  --output-dir DIR       (default: results/gb200/ncu-torch-all-gather/<algo>-all-gather-<time>,
                         <algo>-all-gather-reduce-scatter-<time> with --ag-rs)
  --skip-baseline        Only run the ncu pass
  --baseline-only        Only run the correctness/timing pass, without ncu
  -h, --help

Outputs in --output-dir: config.txt, baseline.log, ncu.log, nccl-logs/<pass>/*.log, the ncu
report <name>.ncu-rep or <name>.ncu-repz (ncu 2026.4; <name> all_gather, or ag_rs with --ag-rs;
<name>-rank<r> for each rank with --communicator tcp, whose ncu.log prefixes every line with
[rank <r>]) and summary.csv (one row per profiled result); blocking.txt with --blocking-metrics.
USAGE
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"

gpus="0,1,2,3"
algo="nvls"
ag_rs=false
size_mib=256
dtype="bf16"
warmup=5
iters=1
communicator="shmem"
lockstep=true
ncu_set="basic"
metrics=""
blocking=false
sections=()
replay_mode="kernel"
timeout_sec=1800
ncu_bin="ncu"
nccl_lib=""
output_dir=""
run_baseline=true
baseline_only=false
extra_ncu_args=()

die() {
    echo "ERROR: $*" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus) gpus="$2"; shift 2 ;;
        --algo) algo="$2"; shift 2 ;;
        --ag-rs) ag_rs=true; shift ;;
        --size-mib) size_mib="$2"; shift 2 ;;
        --dtype) dtype="$2"; shift 2 ;;
        --warmup) warmup="$2"; shift 2 ;;
        --iters) iters="$2"; shift 2 ;;
        --communicator) communicator="$2"; shift 2 ;;
        --no-lockstep) lockstep=false; shift ;;
        --set) ncu_set="$2"; shift 2 ;;
        --section) sections+=("$2"); shift 2 ;;
        --metrics) metrics="$2"; shift 2 ;;
        --blocking-metrics) blocking=true; shift ;;
        --replay-mode) replay_mode="$2"; shift 2 ;;
        --timeout) timeout_sec="$2"; shift 2 ;;
        --ncu) ncu_bin="$2"; shift 2 ;;
        --nccl-lib) nccl_lib="$2"; shift 2 ;;
        --output-dir) output_dir="$2"; shift 2 ;;
        --skip-baseline) run_baseline=false; shift ;;
        --baseline-only) baseline_only=true; shift ;;
        --) shift; extra_ncu_args=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1 (see --help)" ;;
    esac
done

[[ "$algo" =~ ^(ring|nvls|symm|p2p)$ ]] || die "--algo must be ring, nvls, symm or p2p"
[[ "$dtype" =~ ^(bf16|fp32)$ ]] || die "--dtype must be bf16 or fp32"
[[ "$communicator" =~ ^(shmem|tcp|none)$ ]] || die "--communicator must be shmem, tcp or none"
[[ "$replay_mode" =~ ^(kernel|application|range|app-range)$ ]] \
    || die "--replay-mode must be kernel, application, range or app-range"
(( iters >= 1 )) || die "--iters must be at least 1"
if $baseline_only && ! $run_baseline; then
    die "--baseline-only and --skip-baseline cannot be combined"
fi
if $ag_rs; then
    # The overlap worker connects each communicator in the first warmup launch, serially.
    (( warmup >= 1 )) || die "--ag-rs needs --warmup >= 1"
    op_funcs=(AllGather ReduceScatter) case_name=all-gather-reduce-scatter report_name=ag_rs nvtx_range=ncu_collectives
else
    op_funcs=(AllGather) case_name=all-gather report_name=all_gather nvtx_range=ncu_all_gather
fi
nops=${#op_funcs[@]}
IFS=, read -r -a gpu_list <<<"$gpus"
nranks=${#gpu_list[@]}
(( nranks >= 2 )) || die "--gpus needs at least two GPUs"

# --blocking-metrics: every name checked with `ncu --query-metrics-mode suffix --chip gb100`
# (2026.4). The interface stall counters (*_cycles_stalled) are in no ncu section and cannot be
# PM-sampled; NVLink bytes are PM-sampled because the profiler path reported 0 on GB200 + NVSwitch
# (MemoryWorkloadAnalysis_Chart), the profiler counters stay as a cross-check. GB200 specifics:
# clocks are recorded because --clock-control none lets each replay pass (one torchrun each) run
# at its own frequency under the power cap; a B200 is two dies on the LTC fabric, so local reads
# can cross it too; C2C (ctc) and sysmem show traffic to Grace, which the data path should not
# need (#13 saw ~45 MiB C2C tx during the kernel, unexplained).
blocking_profiler_metrics=(
    gpu__time_duration.sum
    sm__cycles_elapsed.avg.per_second
    lts__cycles_elapsed.avg.per_second
    dram__cycles_elapsed.avg.per_second
    nvlrx__cycles_elapsed.avg.per_second
    l1tex__m_l1tex2xbar_req_cycles_stalled.sum
    l1tex__m_l1tex2xbar_req_cycles_active.sum
    l1tex__m_l1tex2xbar_write_bytes.sum
    l1tex__m_xbar2l1tex_read_bytes.sum
    smsp__warps_active.sum
    smsp__warps_issue_stalled_lg_throttle.sum
    smsp__warps_issue_stalled_long_scoreboard.sum
    smsp__warps_issue_stalled_membar.sum
    smsp__warps_issue_stalled_barrier.sum
    smsp__warps_issue_stalled_wait.sum
    smsp__warps_issue_stalled_drain.sum
    sm__average_inst_latency_pipe_lsu_ops_generic_cmd_read.ratio
    sm__average_inst_latency_pipe_lsu_mem_lg_cmd_read.ratio
    lrc__lrc2xbar_cycles_stalled.sum
    lrc__lrc2xbar_cycles_active.sum
    syslts__t_sectors_srcnode_gpc_aperture_peer_op_write.sum
    syslts__t_sectors_srcnode_gpc_aperture_peer_op_read.sum
    syslts__t_sectors_aperture_sysmem.sum
    nvlrx__bytes.sum
    nvltx__bytes.sum
    nvlrx__cycles_active.avg.pct_of_peak_sustained_elapsed
    nvltx__cycles_active.avg.pct_of_peak_sustained_elapsed
    lts__xbar2lts_cycles_active.avg.pct_of_peak_sustained_elapsed
    lts__lts2xbar_cycles_active.avg.pct_of_peak_sustained_elapsed
    lts__t_sectors_op_write.sum
    lts__t_sectors_srcnode_gpc_op_write.sum
    lts__ltcfabric2lts_cycles_stalled.sum
    lts__ltcfabric2lts_cycles_active.sum
    lts__t_sectors_srcunit_ltcfabric_op_write.sum
    lts__t_sectors_srcunit_ltcfabric_op_read.sum
    dram__cycles_active_read.avg.pct_of_peak_sustained_elapsed
    dram__cycles_active_write.avg.pct_of_peak_sustained_elapsed
    dram__sectors_read.sum
    dram__sectors_write.sum
)
blocking_pm_metrics=(
    sm__cycles_elapsed.avg.per_second
    nvlrx__bytes_packet_request_data_user.sum
    nvlrx__bytes_packet_response_data_user.sum
    nvltx__bytes_packet_request_data_user.sum
    nvltx__bytes_packet_response_data_user.sum
    nvlrx__bytes.sum
    nvltx__bytes.sum
    nvlrx__bytes.avg.pct_of_peak_sustained_elapsed
    nvltx__bytes.avg.pct_of_peak_sustained_elapsed
    ctc__tx_bytes_data_user.sum
    ctc__rx_bytes_data_user.sum
    smsp__warps_issue_stalled_lg_throttle.avg.pct_of_peak_sustained_elapsed
    smsp__warps_issue_stalled_long_scoreboard.avg.pct_of_peak_sustained_elapsed
    l1tex__m_l1tex2xbar_req_cycles_active.avg.pct_of_peak_sustained_elapsed
    lts__xbar2lts_cycles_active.avg.pct_of_peak_sustained_elapsed
    lts__lts2xbar_cycles_active.avg.pct_of_peak_sustained_elapsed
    dram__sectors_read.sum
    dram__sectors_write.sum
)
if $blocking; then
    if [[ "$replay_mode" == kernel && "$algo" =~ ^(nvls|symm)$ ]]; then
        die "--blocking-metrics needs several passes; a multi-pass kernel replay cannot save the multicast memory of ${algo}: use --replay-mode application or app-range"
    fi
    blocking_list="$(IFS=,; echo "${blocking_profiler_metrics[*]}")"
    for m in "${blocking_pm_metrics[@]}"; do blocking_list+=",pmsampling:${m}"; done
    metrics="${metrics:+${metrics},}${blocking_list}"
    # ncu locks clocks to base by default, which slows the SMs but not NVLink and so understates
    # sender pressure.
    if [[ " ${extra_ncu_args[*]:-} " != *" --clock-control "* ]]; then
        extra_ncu_args=(--clock-control none ${extra_ncu_args[@]+"${extra_ncu_args[@]}"})
    fi
fi
output_dir="${output_dir:-${repo_root}/results/gb200/ncu-torch-all-gather/${algo}-${case_name}-$(date +%Y%m%d-%H%M%S)}"

################################################################################
# Preflight

if ! $baseline_only; then
    ncu_home="${repo_root}/NVIDIA-Nsight-Compute-2026.4"
    [[ -d "$ncu_home" ]] && export PATH="${ncu_home}:${PATH}"
    command -v "$ncu_bin" >/dev/null || die "${ncu_bin} not found; put Nsight Compute 2026.4 in ${ncu_home} or pass --ncu"
    ncu_version="$("$ncu_bin" --version | sed -n 's/.*Version \([0-9]*\.[0-9]*\).*/\1/p' | head -1)"
    echo "ncu: $(command -v "$ncu_bin") (${ncu_version:-unknown version})"
    ncu_year="${ncu_version%%.*}"
    ncu_minor="${ncu_version#*.}"
    # Nsight Compute 2026.4 added application replay with the shmem communicator.
    if [[ -z "$ncu_version" ]] || (( ncu_year < 2026 || (ncu_year == 2026 && ncu_minor < 4) )); then
        die "Nsight Compute ${ncu_version:-?} is too old; need >= 2026.4 (put it in ${ncu_home} or pass --ncu)"
    fi
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
[[ -n "$nccl_lib" ]] && export LD_PRELOAD="${nccl_lib}${LD_PRELOAD:+:${LD_PRELOAD}}"

# sym_kernel <func>:<kernel>...: NCCL_SYM_KERNEL for these kernels in the NCCL torch loads. From
# 2.32 NCCL parses "<func>:<kernel>;<func>:<kernel>" (parseList in tuning/cost_model.cc); before,
# it compares the whole value with one kernel name, so two functions cannot both be pinned and the
# output is empty (unset): pinning one kernel would leave the other function no symmetric kernel.
sym_kernel() {
    local version major minor
    version="$(cd "$repo_root" && uv run --no-sync python -c \
        'import torch.distributed as d; print(".".join(map(str, d.ProcessGroupNCCL.get_runtime_nccl_version())))')" \
        || die "could not read the runtime NCCL version from torch"
    IFS=. read -r major minor _ <<<"$version"
    echo "runtime NCCL ${version}" >&2
    if (( major > 2 || (major == 2 && minor >= 32) )); then
        (IFS=';'; echo "$*")
    elif (( $# == 1 )); then
        echo "${1#*:}"
    else
        echo "NCCL ${version} takes one NCCL_SYM_KERNEL name; leaving it unset for $*" >&2
    fi
}

# tuning_re_<func>: what every TUNING line of that function must match (no associative arrays:
# the unit tests run this under macOS's bash 3.2).
case "$algo" in
    ring|nvls)
        if [[ "$algo" == ring ]]; then
            export NCCL_ALGO=Ring
        else
            export NCCL_ALGO=NVLS NCCL_NVLS_ENABLE=1
        fi
        for f in "${op_funcs[@]}"; do printf -v "tuning_re_${f}" '%s' "${f}: [0-9]+ Bytes -> Algo ${NCCL_ALGO} "; done
        ;;
    symm|p2p)
        # Window registration selects the symmetric kernels, not NCCL_ALGO. NCCL_SYM_KERNEL does
        # not stop large AllGathers going to the copy engines on Blackwell (enqueue.cc); only an
        # INT64_MAX NCCL_SYM_CE_THRESHOLD does. p2p turns NVLS off: AllGather_ST / ReduceScatter_LD
        # go through peer pointers and the communicator then has no multicast mappings at all.
        if [[ "$algo" == symm ]]; then
            kernel_AllGather=AllGather_STMC kernel_ReduceScatter=ReduceScatter_LDMC
            export NCCL_NVLS_ENABLE=1
        else
            kernel_AllGather=AllGather_ST kernel_ReduceScatter=ReduceScatter_LD
            export NCCL_NVLS_ENABLE=0
        fi
        export NCCL_WIN_ENABLE=1 NCCL_SYM_CE_THRESHOLD=9223372036854775807
        pins=()
        for f in "${op_funcs[@]}"; do
            k="kernel_${f}"
            pins+=("${f}:${!k}")
            printf -v "tuning_re_${f}" '%s' "${f} \\[Symmetric\\]: [0-9]+ Bytes -> Kernel ${!k} "
        done
        NCCL_SYM_KERNEL="$(sym_kernel "${pins[@]}")" || exit 1
        if [[ -n "$NCCL_SYM_KERNEL" ]]; then export NCCL_SYM_KERNEL; else unset NCCL_SYM_KERNEL; fi
        ;;
esac
export CUDA_VISIBLE_DEVICES="$gpus"
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV,TUNING

# --standalone picks a free rendezvous port, so application replay can relaunch torchrun at once.
if $ag_rs; then
    worker_cmd=("${script_dir}/ncu_torch_ag_rs_overlap.py")
else
    worker_cmd=("${script_dir}/ncu_torch_all_gather.py")
fi
worker_cmd+=(--algo "$algo" --size-mib "$size_mib" --dtype "$dtype" --warmup "$warmup" --iters "$iters")
torchrun_cmd=(uv run --no-sync python -m torch.distributed.run --standalone --nproc_per_node "$nranks" "${worker_cmd[@]}")
worker_cmd=(uv run --no-sync python "${worker_cmd[@]}")

{
    if $baseline_only; then
        echo "ncu=disabled"
    else
        echo "ncu=$(command -v "$ncu_bin") ${ncu_version}"
    fi
    echo "gpus=${gpus} algo=${algo} ag_rs=${ag_rs} size_mib=${size_mib} dtype=${dtype} warmup=${warmup} iters=${iters}"
    echo "communicator=${communicator} lockstep=${lockstep} set=${ncu_set} sections=${sections[*]:-<from --set>} metrics=${metrics:-<from --set>} replay=${replay_mode} extra_ncu_args=${extra_ncu_args[*]:-}"
    echo "blocking_metrics=${blocking}"
    echo "nccl_lib=${nccl_lib:-<bundled with torch>}"
    env | grep -E '^(NCCL_|CUDA_VISIBLE_DEVICES=|LD_PRELOAD=)' | sort
} >"${output_dir}/config.txt"
cat "${output_dir}/config.txt"
cd "$repo_root"

# For every op, each of its TUNING lines in the pass must match tuning_re_<func>, warmup + iters
# per run. NCCL prints the line on comm rank 0 only, one log per process; application and app-range
# replay start the workers again for every ncu pass, so <runs> is the number of logs that contain one.
check_tuning() {
    local pass="$1" expected=$(( warmup + iters )) logs f any re lines ok total runs
    logs=("${output_dir}/nccl-logs/${pass}"/*.log)
    for f in "${op_funcs[@]}"; do
        any="NCCL INFO ${f}( \\[[^]]*\\])?: [0-9]+ Bytes -> "
        re="tuning_re_${f}"
        re="${!re}"
        lines="$(cat "${logs[@]}" | grep -E "$any" || true)"
        ok="$(grep -cE "$re" <<<"$lines" || true)"
        total="$(grep -c . <<<"$lines" || true)"
        runs="$(grep -lE "$any" "${logs[@]}" | wc -l | tr -d ' ')"
        if (( runs == 0 || total != ok || total != runs * expected )); then
            echo "${pass}: expected ${expected} ${f} launches matching /${re}/ per run (${runs} run(s)), got:" >&2
            sort <<<"$lines" | sed 's/.*NCCL INFO //; s/ Bytes -> \(\S* \S*\).*/ Bytes -> \1/' | uniq -c >&2
            die "${pass}: NCCL did not run --algo ${algo} for every ${f}; logs in ${output_dir}/nccl-logs/${pass}"
        fi
        echo "${pass}: ${ok}/${total} ${f} launches match --algo ${algo} (${runs} run(s) x ${expected})"
    done
}

free_port() {
    python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'
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
        die "${pass}: ncu could not save the kernel's memory for a multi-pass kernel replay (multicast mappings of nvls / symm);" \
            "use --metrics with a single-pass list, or ncu >= 2026.4 with --replay-mode application or app-range;" \
            "details in the /tmp/nsight-compute-*.log named in ${output_dir}/${pass}.log"
    fi
    # ncu itself stops waiting for the ranks of a replay pass ("Failed to wait for profiling") well
    # before --timeout; name the pass that never finished.
    if grep -q "Failed to wait for profiling" "${output_dir}/${pass}.log"; then
        local stuck
        stuck="$(grep -o 'replay pass [0-9]*' "${output_dir}/${pass}.log" | tail -n 1)"
        die "${pass}: ncu gave up waiting for the profiled kernel (hang in ${stuck:-the first pass}); see ${output_dir}/${pass}.log"
    fi
    if grep -q "No kernels were profiled\|No ranges were profiled" "${output_dir}/${pass}.log"; then
        die "${pass}: ncu profiled nothing (communicator=${communicator}, replay=${replay_mode}); see ${output_dir}/${pass}.log"
    fi
    if grep -q 'Failed to profile "range"' "${output_dir}/${pass}.log"; then
        die "${pass}: ncu could not profile the NVTX range (communicator=${communicator}, replay=${replay_mode});" \
            "see ${output_dir}/${pass}.log and the /tmp/nsight-compute-*.log of the failure"
    fi
    (( status == 0 )) || die "${pass} exited with ${status}; see ${output_dir}/${pass}.log"
    check_tuning "$pass"
}

################################################################################
# 1. Baseline: no ncu

if $run_baseline; then
    run_pass baseline "${torchrun_cmd[@]}"
fi
if $baseline_only; then
    exit 0
fi

################################################################################
# 2. ncu

# The worker's NVTX range holds exactly the --iters launches. The kernel name filter keeps
# the warmup and any non-NCCL kernel out even if a future torch launches one inside the range.
select_args=(--nvtx --nvtx-include "${nvtx_range}/")
if [[ "$replay_mode" == range || "$replay_mode" == app-range ]]; then
    select_args+=(--launch-count 1)
    summary_metrics="gpu__time_duration.sum"
    expected_profiled=$nranks
else
    kernel_re="$(IFS='|'; echo "${op_funcs[*]}")"
    select_args+=(--kernel-name "regex:^nccl(Symk)?DevKernel_(${kernel_re})" --launch-count $(( iters * nops )))
    summary_metrics="launch__grid_size,launch__block_size,gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed"
    expected_profiled=$(( nranks * iters * nops ))
fi
report="${output_dir}/${report_name}"
collect_args=(--set "$ncu_set")
if (( ${#sections[@]} > 0 )); then
    # ncu collects the metrics of every --section plus the --metrics list.
    collect_args=()
    for s in "${sections[@]}"; do collect_args+=(--section "$s"); done
    collect_args+=(--metrics "${metrics:-gpu__time_duration.sum}")
    summary_metrics="${metrics:-gpu__time_duration.sum}"
elif [[ -n "$metrics" ]]; then
    collect_args=(--metrics "$metrics")
    summary_metrics="$metrics"
fi
ncu_cmd=("$ncu_bin")
case "$communicator" in
    shmem) ncu_cmd+=(--communicator shmem --communicator-shmem-num-peers "$nranks") ;;
    tcp) ncu_cmd+=(--communicator tcp --communicator-tcp-num-peers "$nranks" --communicator-tcp-port "$(free_port)") ;;
    none) ncu_cmd+=(--communicator none) ;;
esac
if [[ "$communicator" == tcp ]] && $lockstep; then
    ncu_cmd+=(--lockstep-kernel-launch --lockstep-nvtx-include "${nvtx_range}/")
fi
ncu_cmd+=(
    "${collect_args[@]}"
    --replay-mode "$replay_mode"
    "${select_args[@]}"
    --target-processes all
    --force-overwrite
)

if [[ "$communicator" == tcp ]]; then
    # ncu expands %q{RANK} from each rank's environment.
    ncu_cmd+=(--export "${report}-rank%q{RANK}" ${extra_ncu_args[@]+"${extra_ncu_args[@]}"})
    report_bases=()
    for (( r = 0; r < nranks; r++ )); do report_bases+=("${report}-rank${r}"); done
    # rank 0's worker hosts the TCPStore on MASTER_PORT again in every relaunch.
    export MASTER_ADDR=127.0.0.1 MASTER_PORT
    MASTER_PORT="$(free_port)"
    run_pass ncu bash "${script_dir}/ncu_tcp_ranks.sh" "$nranks" "${ncu_cmd[@]}" -- "${worker_cmd[@]}"
else
    ncu_cmd+=(--export "$report" ${extra_ncu_args[@]+"${extra_ncu_args[@]}"})
    report_bases=("$report")
    run_pass ncu "${ncu_cmd[@]}" "${torchrun_cmd[@]}"
fi

################################################################################
# Summary: one row per profiled kernel (or range)

# ncu 2026.4 writes a compressed all_gather.ncu-repz; accept either format.
report_files=()
for base in "${report_bases[@]}"; do
    found=""
    for ext in ncu-repz ncu-rep; do
        [[ -f "${base}.${ext}" ]] && { found="${base}.${ext}"; break; }
    done
    [[ -n "$found" ]] || die "ncu ${ncu_version} wrote neither ${base}.ncu-rep nor .ncu-repz; see ${output_dir}/ncu.log"
    report_files+=("$found")
done
summary="${output_dir}/summary.csv"
: >"$summary"
for f in "${report_files[@]}"; do
    if $blocking; then
        # Every metric of the report: the raw page's names for the PM-sampled aggregates are not
        # documented, so they are not passed back as a filter.
        rows="$("$ncu_bin" --import "$f" --page raw --csv)"
    else
        rows="$("$ncu_bin" --import "$f" --page raw --csv --metrics "$summary_metrics")"
    fi
    # Line 1 is the header, line 2 the units; the reports of the tcp ranks share them.
    if [[ ! -s "$summary" ]]; then
        printf '%s\n' "$rows" >"$summary"
    else
        [[ "$(head -n 1 <<<"$rows")" == "$(head -n 1 "$summary")" ]] \
            || die "${f} has other columns than ${report_files[0]}; compare them with ncu --import --page raw --csv"
        tail -n +3 <<<"$rows" >>"$summary"
    fi
done
profiled="$(tail -n +3 "$summary" | grep -c . || true)"
echo "=== ${profiled} result(s) in ${report_files[*]} (expected ${expected_profiled})"
if $blocking; then
    python3 "${script_dir}/analyse_blocking_metrics.py" "$summary" --size-mib "$size_mib" --nranks "$nranks" \
        || die "analyse_blocking_metrics.py failed on ${summary}"
else
    column -s, -t <"$summary" 2>/dev/null || cat "$summary"
fi
(( profiled == expected_profiled )) \
    || echo "WARNING: --launch-count may count over all ranks in shmem mode; compare with ${output_dir}/ncu.log" >&2
echo "Open in the Nsight Compute GUI: ${report_files[*]}"
