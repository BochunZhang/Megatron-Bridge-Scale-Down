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
Benchmark torch all-gather / reduce-scatter / all-reduce with ring and NVLS on GB200.

Five passes (families), each its own torchrun job:
  ring            NCCL_ALGO=Ring on plain buffers; ag, rs, ar, ag+rs
  nvls            NCCL_ALGO=NVLS on plain buffers; ag, rs, ar, ag+rs
  symm            symmetric kernels on registered windows; ag, rs, ar, ag+rs
  symm-tma   AG = AllGather_TmaSTMC (TMA -> multicast), RS = ReduceScatter_LDMC; ag, ag+rs
  copy-engine     AG = copy engine writing the multicast address, RS = ReduceScatter_LDMC; ag, ag+rs
symm-tma and copy-engine skip rs and ar: NCCL has no TMA multicast or CE reduce-scatter, so
their RS is symm's ReduceScatter_LDMC. Both still run every single and split scenario (the split
cross case too: group A all-gathers with the pass's AG while group B runs ReduceScatter_LDMC).

Run the same command on every node (DLC style). For each algorithm a fresh torchrun job
is started, because NCCL_ALGO is fixed when a communicator is created. After the last pass
node 0 runs analyse_results.sh (compare.md, seconds); a failure there is only a warning, so a
finished benchmark is never lost to its post-processing. Re-run it by hand with:
  bash examples/gb200/nvls-collective/analyse_results.sh --output-dir <run dir>
The --nsys reports are not analysed here: exporting them takes node 0 hours, long enough for the
other nodes to time out in whatever the job runs next. Analyse them after the job with:
  bash examples/gb200/nvls-collective/analyse_nsys.sh --output-dir <run dir>

Output: results/gb200/nvls-collective/gpu-num-<nnodes*gpus-per-node>/<yyyymmdd-hhmmss> in this
repo, one <algo>/ per family. Each test gets <algo>/cases/<algo>-<case>/ (e.g.
nvls/cases/nvls-all_reduce-1024MiB-bf16-single-n16/) with every member rank's NCCL log
<algo>-<case>.rank_<RR>.nccl.log, case.json and, with --nsys, the per-rank reports
<algo>-<case>.rank_<RR>.nsys-rep. A test fails unless NCCL's TUNING log shows the expected
implementation for every launch. Node 0 stamps the id and hands it to the other nodes through the run_timestamp_and_logs
helper (TCPStore on MASTER_ADDR:<master-port - 1>), so the repo must be on storage every node
can see. An exported RUN_TIME pins it, which is what a re-run of one comparison needs.

Gloo/NCCL bootstrap interface: eth0 for a single 4-GPU node (DSW), bond1 for more than
4 GPUs (DLC), checked against `ip -o -4 addr show up`. An exported GLOO_SOCKET_IFNAME wins.

Usage:
  run_nvls_collective_benchmark.sh [options]

Options:
  --nnodes N              Nodes in the job (default: NNODES or 1)
  --node-rank N           This node's rank (default: NODE_RANK, RANK or 0)
  --master-addr ADDR      Rendezvous host (default: MASTER_ADDR, DLC_MASTER_ADDR or 127.0.0.1)
  --master-port PORT      First rendezvous port; algorithm i uses PORT+i (default: 29611)
  --gpus-per-node N       GPUs per node (default: 4, one GB200 compute tray)
  --algos LIST            ring,nvls,symm,symm-tma,copy-engine (default: all five)
  --group-sizes LIST      Single-group sizes (default: 4,8,16,32 capped at nnodes*gpus-per-node)
  --split-totals LIST     Ranks cut into two concurrent half-groups; T/2 must be in --group-sizes
                          (default: the largest of 16,32 that fits: 8+8 on 16 GPUs, 16+16 on 32)
  --split-layouts LIST    contiguous,interleaved (default: contiguous)
  --workloads LIST        ag,rs,ar,ag+rs (default: all); ar is an in-place all-reduce of the whole
                          tensor; with both ag and rs, split scenarios also run the cross case
                          (group A all-gathers while group B reduce-scatters)
  --sizes-mib LIST        Full tensor sizes in MiB (default: 16,32,64,128,256,512,1024,2048)
  --dtype bf16|fp32       Payload dtype (default: bf16)
  --warmup N              Untimed iterations per case (default: 5)
  --iters N               Back-to-back iterations per trial (default: 20)
  --trials N              Timed trials, median reported (default: 5)
  --symm-kernel KERNEL    For the symm pass only: default, ag, rs or ar (pins NCCL_SYM_KERNEL); needs
                          --workloads with that op only, because the pin masks the other op
  --nsys                  One Nsight Systems report per rank per case (capture range = timed trials),
                          with GPU metrics (NVLink RX/TX, DRAM, ...) sampled on the rank's own GPU;
                          node 0 files them under cases/ after each pass (renames only);
                          nsys_kernels.md and nsys_gpu_metrics.md come from analyse_nsys.sh after the job
  --nsys-gpu-metrics-set SET
                          nsys GPU metric set alias (default: nsys picks the first set for the GPU);
                          the choices are saved to nsys-gpu-metrics.node<N>.txt in the run directory
  --nsys-gpu-metrics-frequency HZ
                          GPU metric sampling rate, 10..200000 (default: 50000, one sample per 20 us)
  --ncu                   One Nsight Compute report per rank (<algo>/ncu/rank<R>.ncu-rep): every
                          NCCL collective kernel of trial0 of every case, all ranks in lockstep;
                          needs one group of every rank (--group-sizes = all GPUs, no split);
                          exclusive with --nsys
  --ncu-sections LIST     Comma-separated ncu section identifiers (default: SpeedOfLight, memory
                          workload, NVLink + tables + topology, C2CLink, launch/occupancy/scheduler,
                          WarpStateStats, SourceCounters, PmSampling)
  --ncu-port PORT         First ncu lockstep communicator port; algorithm i uses PORT+i (default: 49217)
  --dry-run               Print commands only
  -h, --help              Show this help

Examples:
  # One tray (4 GPUs): smoke test.
  bash examples/gb200/nvls-collective/run_nvls_collective_benchmark.sh --sizes-mib 64 --trials 2

  # One tray: only the TMA and copy-engine AllGather passes (plus symm as their reference).
  bash examples/gb200/nvls-collective/run_nvls_collective_benchmark.sh \
    --algos symm,symm-tma,copy-engine --sizes-mib 64,256,1024 --trials 3

  # Eight trays in one NVL72 domain (32 GPUs); run the same command on every node.
  bash examples/gb200/nvls-collective/run_nvls_collective_benchmark.sh --nnodes 8

  # One tray: nsys trace plus NVLink / DRAM throughput curves of the symmetric kernels.
  bash examples/gb200/nvls-collective/run_nvls_collective_benchmark.sh \
    --algos symm --sizes-mib 256,1024 --trials 3 --nsys

  # One tray: Nsight Compute details of the symmetric AG kernel (keep the case count small:
  # every profiled kernel is replayed once per counter pass).
  bash examples/gb200/nvls-collective/run_nvls_collective_benchmark.sh \
    --algos symm --workloads ag --sizes-mib 1024 --iters 2 --trials 1 --ncu
USAGE
}

die() {
    echo "ERROR: $*" >&2
    exit 2
}

need_value() {
    [[ $# -ge 2 && -n "$2" ]] || die "$1 requires a value"
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"

# Run id and log helpers shared by the scale-down experiment scripts. resolve_run_time is the
# single source of the run directory name; sourcing only defines functions. log/warn are defined
# here first so the helper keeps these versions and its status line goes to stderr, and die
# (above) keeps the ERROR: prefix this script already uses.
log() { printf '%s\n' "$*" >&2; }
warn() { printf 'warning: %s\n' "$*" >&2; }
# shellcheck source=../../../scripts/scale-down/common/run_timestamp_and_logs.sh
source "${repo_root}/scripts/scale-down/common/run_timestamp_and_logs.sh"

nnodes="${NNODES:-1}"
node_rank="${NODE_RANK:-${RANK:-0}}"
master_addr="${MASTER_ADDR:-${DLC_MASTER_ADDR:-127.0.0.1}}"
master_port="29611"
gpus_per_node="4"
algos="ring,nvls,symm,symm-tma,copy-engine"
group_sizes=""
split_totals=""
split_layouts="contiguous"
workloads="ag,rs,ar,ag+rs"
sizes_mib="16,32,64,128,256,512,1024,2048"
dtype="bf16"
warmup="5"
iters="20"
trials="5"
symm_kernel="default"
nsys_enabled=false
nsys_gpu_metrics_set=""
nsys_gpu_metrics_frequency="50000"
ncu_enabled=false
# Kernel-level detail for the symmetric kernels: throughput and memory-chart (HBM, L2, peer) of
# the kernel, NVLink totals and per-link tables, the Grace C2C link, where warps stall (per SASS
# instruction in SourceCounters, e.g. multimem.ld_reduce / multimem.st), and PM sampling for the
# behavior over the kernel's runtime.
ncu_sections="SpeedOfLight,MemoryWorkloadAnalysis,MemoryWorkloadAnalysis_Chart,MemoryWorkloadAnalysis_Tables"
ncu_sections+=",Nvlink,Nvlink_Tables,Nvlink_Topology,C2CLink,LaunchStats,Occupancy,SchedulerStats"
ncu_sections+=",WarpStateStats,SourceCounters,PmSampling"
ncu_port="49217"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nnodes) need_value "$@"; nnodes="$2"; shift 2 ;;
        --node-rank) need_value "$@"; node_rank="$2"; shift 2 ;;
        --master-addr) need_value "$@"; master_addr="$2"; shift 2 ;;
        --master-port) need_value "$@"; master_port="$2"; shift 2 ;;
        --gpus-per-node) need_value "$@"; gpus_per_node="$2"; shift 2 ;;
        --algos) need_value "$@"; algos="$2"; shift 2 ;;
        --group-sizes) need_value "$@"; group_sizes="$2"; shift 2 ;;
        --split-totals) need_value "$@"; split_totals="$2"; shift 2 ;;
        --split-layouts) need_value "$@"; split_layouts="$2"; shift 2 ;;
        --workloads) need_value "$@"; workloads="$2"; shift 2 ;;
        --sizes-mib) need_value "$@"; sizes_mib="$2"; shift 2 ;;
        --dtype) need_value "$@"; dtype="$2"; shift 2 ;;
        --warmup) need_value "$@"; warmup="$2"; shift 2 ;;
        --iters) need_value "$@"; iters="$2"; shift 2 ;;
        --trials) need_value "$@"; trials="$2"; shift 2 ;;
        --symm-kernel) need_value "$@"; symm_kernel="$2"; shift 2 ;;
        --nsys) nsys_enabled=true; shift ;;
        --nsys-gpu-metrics-set) need_value "$@"; nsys_gpu_metrics_set="$2"; shift 2 ;;
        --nsys-gpu-metrics-frequency) need_value "$@"; nsys_gpu_metrics_frequency="$2"; shift 2 ;;
        --ncu) ncu_enabled=true; shift ;;
        --ncu-sections) need_value "$@"; ncu_sections="$2"; shift 2 ;;
        --ncu-port) need_value "$@"; ncu_port="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1 (see --help)" ;;
    esac
done

for name in nnodes gpus_per_node master_port nsys_gpu_metrics_frequency ncu_port; do
    [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || die "--${name//_/-} must be a positive integer"
done
[[ "$node_rank" =~ ^[0-9]+$ ]] && (( node_rank < nnodes )) || die "--node-rank must be in [0, nnodes)"
world_size=$(( nnodes * gpus_per_node ))
(( nsys_gpu_metrics_frequency >= 10 && nsys_gpu_metrics_frequency <= 200000 )) \
    || die "--nsys-gpu-metrics-frequency must be in [10, 200000] Hz"
# Both profilers own the GPU performance counters; nsys GPU metrics cannot run under ncu.
if $nsys_enabled && $ncu_enabled; then
    die "--nsys and --ncu are exclusive: both need the GPU performance counters"
fi
[[ -n "$ncu_sections" ]] || die "--ncu-sections must not be empty"

if $ncu_enabled; then
    # ncu replays every profiled kernel once per counter pass, in lockstep across all ranks
    # (--lockstep-kernel-launch). Lockstep waits for every ncu instance, so a case in which some
    # rank launches no collective (a smaller group, or the idle half of a split) would hang.
    # Only one group of every rank is allowed.
    [[ -z "$group_sizes" || "$group_sizes" == "$world_size" ]] \
        || die "--ncu profiles all ranks in lockstep; --group-sizes must be ${world_size} (all GPUs)"
    [[ -z "$split_totals" || "$split_totals" == none ]] \
        || die "--ncu profiles all ranks in lockstep; split scenarios leave ranks idle, drop --split-totals"
    group_sizes="$world_size"
    split_totals="none"
fi

# Defaults follow the allocation: every 4/8/16/32 that fits, and one split of the largest of 16/32
# that fits into two halves (8+8 on 16 GPUs, 16+16 on 32). Each half's isolated reference is the
# single group of the same size, which the group-size default already includes.
if [[ -z "$group_sizes" ]]; then
    for n in 4 8 16 32; do
        (( n <= world_size )) && group_sizes+="${group_sizes:+,}$n"
    done
    [[ -n "$group_sizes" ]] || group_sizes="$world_size"
fi
if [[ -z "$split_totals" ]]; then
    for n in 16 32; do
        (( n <= world_size )) && split_totals="$n"
    done
fi
IFS=, read -r -a algo_list <<<"$algos"
for algo in "${algo_list[@]}"; do
    case "$algo" in
        ring|nvls|symm|symm-tma|copy-engine) ;;
        *) die "--algos accepts ring,nvls,symm,symm-tma,copy-engine; got $algo" ;;
    esac
done
case "$symm_kernel" in
    default) symm_kernel_env="" ;;
    # One name per kernel id; NCCL_SYM_KERNEL takes a single kernel name, so the values that
    # answer "does the multicast symmetric kernel help" are the *MC ones.
    ag) symm_kernel_env="AllGather_STMC" ;;
    rs) symm_kernel_env="ReduceScatter_LDMC" ;;
    # Reduce-scatter + all-gather through multicast: the only symmetric AR the benchmark accepts.
    ar) symm_kernel_env="AllReduce_RSxLDMC_AGxSTMC" ;;
    *) die "--symm-kernel accepts default, ag, rs or ar; got $symm_kernel" ;;
esac
# Fail before the ring/nvls passes run: the benchmark rejects this in the symm pass anyway.
if [[ "$symm_kernel" != default && "$workloads" != "$symm_kernel" ]]; then
    die "--symm-kernel ${symm_kernel} masks the other op's symmetric kernels; pass --workloads ${symm_kernel}"
fi
if (( nnodes > 1 )) && [[ "$master_addr" == 127.0.0.1 ]]; then
    die "multi-node runs require --master-addr or MASTER_ADDR"
fi
if [[ -n "${NCCL_ALGO:-}" || -n "${NCCL_NVLS_ENABLE:-}" ]]; then
    echo "WARNING: NCCL_ALGO / NCCL_NVLS_ENABLE from the environment are overridden per algorithm" >&2
fi

(( master_port > 1 )) || die "--master-port must be > 1 (master-port - 1 hosts the run-dir handshake)"

# Pick the bootstrap interface from `ip addr`: DSW (one 4-GPU node) uses eth0, DLC (more than
# 4 GPUs) uses bond1. Only interfaces that are up and carry an IPv4 address qualify.
select_ifname() {
    local preferred=eth0 other=bond1 available
    if (( world_size > 4 )); then
        preferred=bond1
        other=eth0
    fi
    if ! command -v ip >/dev/null; then
        $DRY_RUN || die "the ip command is required to choose GLOO_SOCKET_IFNAME"
        echo "WARNING: ip not found; dry-run assumes ${preferred}" >&2
        echo "$preferred"
        return
    fi
    available="$(ip -o -4 addr show up | awk '{print $2}' | sort -u | tr '\n' ' ')"
    if [[ " $available" == *" ${preferred} "* ]]; then
        echo "$preferred"
    elif [[ " $available" == *" ${other} "* ]]; then
        echo "WARNING: ${preferred} has no IPv4 address; falling back to ${other} (up: ${available})" >&2
        echo "$other"
    else
        die "neither ${preferred} nor ${other} is up with an IPv4 address (up: ${available:-none})"
    fi
}
if [[ -n "${GLOO_SOCKET_IFNAME:-}" ]]; then
    gloo_ifname="$GLOO_SOCKET_IFNAME"
    echo "GLOO_SOCKET_IFNAME=${gloo_ifname} taken from the environment" >&2
else
    gloo_ifname="$(select_ifname)"
fi
# NCCL bootstraps over the same network; keep an explicitly exported value.
nccl_socket_ifname="${NCCL_SOCKET_IFNAME:-$gloo_ifname}"

results_root="${repo_root}/results/gb200/nvls-collective"
# One batch id for the whole job: node 0 stamps it and publishes it on MASTER_PORT, the other
# nodes block until they read the same value. resolve_run_time needs GLOO_SOCKET_IFNAME (the
# store bootstraps through Gloo) and reads NNODES/NODE_RANK/MASTER_ADDR/MASTER_PORT from the
# environment, so hand it the values this script resolved. MASTER_PORT keeps master-port - 1,
# which the algorithm jobs (master-port + i) leave free.
export MASTER_ADDR="$master_addr"
export MASTER_PORT="$(( master_port - 1 ))"
export NNODES="$nnodes"
export NODE_RANK="$node_rank"
export GLOO_SOCKET_IFNAME="$gloo_ifname"
resolve_run_time "$DRY_RUN"
# <yyyymmdd-hhmmss> keeps each job's directory unique without scanning the tree, so a re-run of
# the same comparison gets a new one; export RUN_TIME to write into a specific id instead.
output_dir="${results_root}/gpu-num-${world_size}/${RUN_TIME}"
[[ -n "$RUN_TIME" ]] || die "could not resolve the run id"

bench_args=(
    --group-sizes "$group_sizes"
    --split-layouts "$split_layouts"
    --workloads "$workloads"
    --sizes-mib "$sizes_mib"
    --dtype "$dtype"
    --warmup "$warmup"
    --iters "$iters"
    --trials "$trials"
)
if [[ -n "$split_totals" ]]; then
    bench_args+=(--split-totals "$split_totals")
else
    # Fewer than 16 GPUs: no default split; only single-group cases run.
    bench_args+=(--split-totals none)
fi

# TUNING carries the per-collective "Algo ... proto ..." and "[Symmetric] ... Kernel ..." lines
# that the benchmark parses as evidence of the executed kernel; INIT carries nNodes/MNNVL.
debug_subsys="INIT,ENV,TUNING"
if $nsys_enabled; then
    bench_args+=(--nsys)
    if ! $DRY_RUN; then
        command -v nsys >/dev/null || die "nsys is required when --nsys is enabled"
        nsys --version >&2
    fi
fi
if $ncu_enabled; then
    # Same per-case capture windows and case NVTX ranges as --nsys; ncu profiles inside them only.
    bench_args+=(--ncu)
    if ! $DRY_RUN; then
        command -v ncu >/dev/null || die "ncu is required when --ncu is enabled"
        ncu --version >&2
    fi
fi

run() {
    if $DRY_RUN; then
        printf '%q ' "$@"
        printf '\n'
    else
        "$@"
    fi
}

echo "world_size=${world_size} nnodes=${nnodes} node_rank=${node_rank} group_sizes=${group_sizes}" \
    "split_totals=${split_totals:-none} ifname=${gloo_ifname} output_dir=${output_dir}" >&2
$DRY_RUN || mkdir -p "$output_dir"
if $nsys_enabled && ! $DRY_RUN; then
    # Which GPU ids nsys samples (to check against the ranks' pci_bus_id in metadata.json) and which
    # metric sets this GPU offers. Informational only: a failure here never stops the benchmark.
    {
        nsys profile --gpu-metrics-devices=help 2>&1
        nsys profile --gpu-metrics-devices=all --gpu-metrics-set=help 2>&1
    } >"${output_dir}/nsys-gpu-metrics.node${node_rank}.txt" || true
fi

# NCCL_SYM_CE_THRESHOLD is an element count; INT64_MAX keeps every AllGather off the copy engines.
int64_max=9223372036854775807
for index in "${!algo_list[@]}"; do
    algo="${algo_list[$index]}"
    algo_dir="${output_dir}/${algo}"
    $DRY_RUN || mkdir -p "$algo_dir"
    sym_ce_threshold=""
    sym_kernel=""
    ce_multicast_threshold=""
    algo_args=()
    case "$algo" in
        nvls) nccl_algo="NVLS"; nvls_enable=1; win_enable="" ;;
        ring) nccl_algo="Ring"; nvls_enable=0; win_enable="" ;;
        # The symmetric path is chosen by window registration, not by NCCL_ALGO. An empty
        # NCCL_ALGO means "unset" to both NCCL and the benchmark. NVLS_ENABLE=1 keeps the
        # multicast (MC) symmetric kernels eligible; NCCL_WIN_ENABLE turns on window registration.
        # Without the CE threshold, an AllGather whose per-rank count exceeds 8Mi elements runs on
        # the copy engines (enqueue.cc, Blackwell + NVLink + registered window) and never reaches a
        # symmetric kernel, whatever NCCL_SYM_KERNEL says.
        symm)
            nccl_algo=""; nvls_enable=1; win_enable=1; sym_ce_threshold="$int64_max"
            sym_kernel="$symm_kernel_env"
            algo_args=(--symm-kernel "$symm_kernel")
            ;;
        # Same windows; NCCL 2.32 takes "<func>:<kernel>;..." in NCCL_SYM_KERNEL, so AG and RS are
        # pinned separately and AllReduce (the start gate) is left alone.
        symm-tma)
            nccl_algo=""; nvls_enable=1; win_enable=1; sym_ce_threshold="$int64_max"
            sym_kernel="allgather:AllGather_TmaSTMC;reducescatter:ReduceScatter_LDMC"
            ;;
        # Threshold 0 sends every AllGather to the copy engines; a multicast threshold above every
        # shard (2^62 bytes, ce_coll.cc ncclCeAllGatherUseMulticast) makes it one CE copy to the
        # multicast address instead of N-1 unicast copies. Only the RS needs a kernel pin.
        copy-engine)
            nccl_algo=""; nvls_enable=1; win_enable=1; sym_ce_threshold=0
            ce_multicast_threshold=4611686018427387904
            sym_kernel="reducescatter:ReduceScatter_LDMC"
            ;;
    esac
    # Whole-process NCCL logs; the benchmark copies each test's slice into <algo>/cases/<algo>-<case>/.
    nccl_raw_dir="${algo_dir}/nccl-raw"
    $DRY_RUN || mkdir -p "$nccl_raw_dir"
    algo_env=(
        NCCL_ALGO="$nccl_algo"
        NCCL_NVLS_ENABLE="$nvls_enable"
        # Two communicators run concurrently: keep their kernel launch order identical on every
        # rank. With CUDA runtime and driver >= 12.3 this orders launches, not completions.
        NCCL_LAUNCH_ORDER_IMPLICIT=1
        NCCL_DEBUG=INFO
        NCCL_DEBUG_SUBSYS="$debug_subsys"
        NCCL_DEBUG_FILE="${nccl_raw_dir}/nccl-%h-%p.log"
        GLOO_SOCKET_IFNAME="$gloo_ifname"
        NCCL_SOCKET_IFNAME="$nccl_socket_ifname"
    )
    [[ -n "$win_enable" ]] && algo_env+=(NCCL_WIN_ENABLE="$win_enable")
    [[ -n "$sym_ce_threshold" ]] && algo_env+=(NCCL_SYM_CE_THRESHOLD="$sym_ce_threshold")
    [[ -n "$sym_kernel" ]] && algo_env+=(NCCL_SYM_KERNEL="$sym_kernel")
    [[ -n "$ce_multicast_threshold" ]] && algo_env+=(NCCL_CE_COLL_AG_MULTICAST_THRESHOLD="$ce_multicast_threshold")
    # Cross-tray groups must use the NVLink fabric; forcing MNNVL makes a fallback to IB fail.
    (( nnodes > 1 )) && algo_env+=(NCCL_MNNVL_ENABLE=1)

    worker=("${script_dir}/nvls_collective_benchmark.py")
    torchrun_extra=()
    if $nsys_enabled; then
        nsys_dir="${algo_dir}/nsys"
        $DRY_RUN || mkdir -p "$nsys_dir"
        if ! $DRY_RUN && compgen -G "${nsys_dir}/*.nsys-rep" >/dev/null; then
            die "${nsys_dir} already contains Nsight reports; ${output_dir} was not a fresh run directory"
        fi
        # Each worker gets its own nsys session. Only the timed trials of each case are captured
        # (cudaProfilerStart/Stop in the benchmark); "repeat" writes rank<R>.<K>.nsys-rep for the
        # K-th case. nsys/index.tsv maps K to the case, and the case label is also the outermost
        # NVTX range in each report, which the analysis checks against where the report was filed.
        # nsys_worker.sh adds --gpu-metrics-devices=<this rank's GPU>: GPU metrics (NVLink RX/TX,
        # DRAM, ...) are sampled per GPU, and only one session may sample a GPU, so every worker
        # samples its own. The id is only known once torchrun has set LOCAL_RANK.
        torchrun_extra=(--no-python)
        worker=(
            bash "${script_dir}/nsys_worker.sh"
            --trace=cuda,nvtx
            --sample=none
            --cpuctxsw=none
            --gpu-metrics-frequency="$nsys_gpu_metrics_frequency"
        )
        [[ -n "$nsys_gpu_metrics_set" ]] && worker+=(--gpu-metrics-set="$nsys_gpu_metrics_set")
        worker+=(
            --capture-range=cudaProfilerApi
            --capture-range-end=repeat
            --force-overwrite=false
            --output="${nsys_dir}/rank%q{RANK}"
            python "${script_dir}/nvls_collective_benchmark.py"
        )
        algo_env+=(NSYS_TMPDIR="${NSYS_TMPDIR:-${nsys_dir}/.tmp}")
        # nsys numbers GPUs like nvidia-smi (PCI bus order); make CUDA's LOCAL_RANK index agree.
        algo_env+=(CUDA_DEVICE_ORDER=PCI_BUS_ID)
        $DRY_RUN || mkdir -p "${NSYS_TMPDIR:-${nsys_dir}/.tmp}"
    fi
    if $ncu_enabled; then
        ncu_dir="${algo_dir}/ncu"
        $DRY_RUN || mkdir -p "$ncu_dir"
        if ! $DRY_RUN && compgen -G "${ncu_dir}/*.ncu-rep" >/dev/null; then
            die "${ncu_dir} already contains Nsight Compute reports; ${output_dir} was not a fresh run directory"
        fi
        # One ncu instance per worker, joined through a TCP communicator on node 0, launching and
        # replaying every profiled kernel in lockstep: a symmetric kernel waits on its peers in an
        # LSA barrier, so a rank replaying alone would hang. Profiled: NCCL collective kernels in
        # the trial0 NVTX range of each case (the start gate runs before that range opens, warmup
        # outside the capture window). Clocks and caches are left alone so the kernel runs as in
        # the timed passes: ncu would otherwise lock base clocks and flush L2 before every pass.
        IFS=, read -r -a ncu_section_list <<<"$ncu_sections"
        torchrun_extra=(--no-python)
        worker=(
            ncu
            --communicator=tcp
            --communicator-tcp-hostname="$master_addr"
            --communicator-tcp-port="$(( ncu_port + index ))"
            --communicator-tcp-num-peers="$world_size"
            --lockstep-kernel-launch
            --lockstep-nvtx-include="trial0/"
            --profile-from-start=off
            --nvtx
            --nvtx-include="trial0/"
            --kernel-name="regex:^nccl(Symk)?DevKernel_(AllGather|ReduceScatter|AllReduce)"
            --replay-mode=kernel
            --clock-control=none
            --cache-control=none
        )
        for section in "${ncu_section_list[@]}"; do
            worker+=(--section="$section")
        done
        worker+=(
            --export="${ncu_dir}/rank%q{RANK}"
            python "${script_dir}/nvls_collective_benchmark.py"
        )
    fi

    echo "=== algo=${algo} (${algo_env[*]})" >&2
    run env "${algo_env[@]}" \
        uv run --no-sync python -m torch.distributed.run \
        --nproc_per_node="$gpus_per_node" \
        --nnodes="$nnodes" \
        --node_rank="$node_rank" \
        --master_addr="$master_addr" \
        --master_port="$(( master_port + index ))" \
        ${torchrun_extra[@]+"${torchrun_extra[@]}"} \
        "${worker[@]}" \
        --algo "$algo" --output-dir "$algo_dir" "${bench_args[@]}" ${algo_args[@]+"${algo_args[@]}"} \
        2>&1 | { $DRY_RUN && cat || tee "${algo_dir}/torchrun-node${node_rank}.log"; }

    if $nsys_enabled && ! $DRY_RUN; then
        compgen -G "${nsys_dir}/rank*.nsys-rep" >/dev/null \
            || die "nsys wrote no report under ${nsys_dir}; inspect ${algo_dir}/torchrun-node${node_rank}.log"
    fi
    if $ncu_enabled && ! $DRY_RUN; then
        # Copy-engine AllGather is a memcpy, so an "ag"-only copy-engine pass has nothing to profile.
        compgen -G "${ncu_dir}/rank*.ncu-rep" >/dev/null \
            || warn "ncu wrote no report under ${ncu_dir}; inspect ${algo_dir}/torchrun-node${node_rank}.log"
    fi
    if $nsys_enabled && (( node_rank == 0 )); then
        # Rename this pass's reports into cases/<algo>-<case>/ now, so a later pass that fails
        # leaves them labelled. Only renames (no export); waits for the other nodes' reports.
        run uv run --no-sync python "${script_dir}/analyse_nsys.py" --output-dir "$output_dir" \
            --algo "$algo" --file-only \
            || warn "filing ${algo} nsys reports failed; analyse_nsys.sh retries it"
    fi
done

if (( node_rank == 0 )); then
    run bash "${script_dir}/analyse_results.sh" --output-dir "$output_dir" \
        || warn "post-processing failed; the measurements are in ${output_dir}. Retry with:" \
            "bash ${script_dir}/analyse_results.sh --output-dir ${output_dir}"
    if $nsys_enabled; then
        log "nsys reports are filed, not analysed; after the job run:" \
            "bash ${script_dir}/analyse_nsys.sh --output-dir ${output_dir}"
    fi
    if $ncu_enabled; then
        log "ncu reports: ${output_dir}/<algo>/ncu/rank<R>.ncu-rep; open them in ncu-ui, or e.g." \
            "ncu --import <report> --page details --section Nvlink_Tables"
    fi
fi
