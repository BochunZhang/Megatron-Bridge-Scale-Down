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

Run the same command on every node (DLC style). For each algorithm a fresh torchrun job
is started, because NCCL_ALGO is fixed when a communicator is created. Node 0 merges the
results into compare.md.

Output: results/gb200/nvls-collective/gpu-num-<nnodes*gpus-per-node>/<yyyymmdd-hhmmss> in this
repo, one <algo>/ per family. Each test gets <algo>/cases/<case>/ (e.g.
symm/cases/single-n16_ag_bf16_1024MiB/) with every member rank's NCCL log of that test,
case.json and, with --nsys, the per-rank reports <algo>_<case>_rank<RR>.nsys-rep. A test fails
unless NCCL's TUNING log shows the expected implementation for every launch. Node 0 stamps the id and hands it to the other nodes through the run_timestamp_and_logs
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
  --algos LIST            ring,nvls,symm (default: ring,nvls,symm)
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
  --symm-kernel KERNEL    For the symm pass: default, ag, rs or ar (pins NCCL_SYM_KERNEL); needs
                          --workloads with that op only, because the pin masks the other op
  --nsys                  One Nsight Systems report per rank per case (capture range = timed trials);
                          node 0 then exports them to SQLite and writes nsys_kernels.md
  --dry-run               Print commands only
  -h, --help              Show this help

Examples:
  # One tray (4 GPUs): smoke test.
  bash examples/gb200/nvls-collective/run_nvls_collective_benchmark.sh --sizes-mib 64 --trials 2

  # Eight trays in one NVL72 domain (32 GPUs); run the same command on every node.
  bash examples/gb200/nvls-collective/run_nvls_collective_benchmark.sh --nnodes 8
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
algos="ring,nvls,symm"
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
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1 (see --help)" ;;
    esac
done

for name in nnodes gpus_per_node master_port; do
    [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || die "--${name//_/-} must be a positive integer"
done
[[ "$node_rank" =~ ^[0-9]+$ ]] && (( node_rank < nnodes )) || die "--node-rank must be in [0, nnodes)"
world_size=$(( nnodes * gpus_per_node ))

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
    [[ "$algo" == ring || "$algo" == nvls || "$algo" == symm ]] || die "--algos accepts ring,nvls,symm; got $algo"
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
bench_args+=(--symm-kernel "$symm_kernel")

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

for index in "${!algo_list[@]}"; do
    algo="${algo_list[$index]}"
    algo_dir="${output_dir}/${algo}"
    $DRY_RUN || mkdir -p "$algo_dir"
    case "$algo" in
        nvls) nccl_algo="NVLS"; nvls_enable=1; win_enable="" ;;
        ring) nccl_algo="Ring"; nvls_enable=0; win_enable="" ;;
        # The symmetric path is chosen by window registration, not by NCCL_ALGO. An empty
        # NCCL_ALGO means "unset" to both NCCL and the benchmark. NVLS_ENABLE=1 keeps the
        # multicast (MC) symmetric kernels eligible; NCCL_WIN_ENABLE turns on window registration.
        symm) nccl_algo=""; nvls_enable=1; win_enable=1 ;;
    esac
    # Whole-process NCCL logs; the benchmark copies each test's slice into <algo>/cases/<case>/.
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
    # Without this, an AllGather whose per-rank count exceeds 8Mi elements runs on the copy
    # engines (enqueue.cc, Blackwell + NVLink + registered window) and never reaches a
    # symmetric kernel, whatever NCCL_SYM_KERNEL says. INT64_MAX disables that branch.
    [[ -n "$win_enable" ]] && algo_env+=(NCCL_SYM_CE_THRESHOLD=9223372036854775807)
    [[ -n "$win_enable" && -n "$symm_kernel_env" ]] && algo_env+=(NCCL_SYM_KERNEL="$symm_kernel_env")
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
        # (cudaProfilerStart/Stop in the benchmark); "repeat" writes one report per case. The case
        # label is the outermost NVTX range in each report and nsys/index.tsv lists every case in
        # capture order with its wall-clock window, so mapping never depends on nsys file naming.
        torchrun_extra=(--no-python)
        worker=(
            nsys profile
            --trace=cuda,nvtx
            --sample=none
            --cpuctxsw=none
            --capture-range=cudaProfilerApi
            --capture-range-end=repeat
            --force-overwrite=false
            --output="${nsys_dir}/rank%q{RANK}"
            python "${script_dir}/nvls_collective_benchmark.py"
        )
        algo_env+=(NSYS_TMPDIR="${NSYS_TMPDIR:-${nsys_dir}/.tmp}")
        $DRY_RUN || mkdir -p "${NSYS_TMPDIR:-${nsys_dir}/.tmp}"
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
        --algo "$algo" --output-dir "$algo_dir" "${bench_args[@]}" \
        2>&1 | { $DRY_RUN && cat || tee "${algo_dir}/torchrun-node${node_rank}.log"; }

    if $nsys_enabled && ! $DRY_RUN; then
        compgen -G "${nsys_dir}/rank*.nsys-rep" >/dev/null \
            || die "nsys wrote no report under ${nsys_dir}; inspect ${algo_dir}/torchrun-node${node_rank}.log"
    fi
done

if (( node_rank == 0 )); then
    compare_args=()
    for algo in "${algo_list[@]}"; do
        compare_args+=(--input "${algo}=${output_dir}/${algo}/results.jsonl")
    done
    run uv run --no-sync python "${script_dir}/collective_metrics.py" "${compare_args[@]}" \
        --output "${output_dir}/compare.md"
    if $nsys_enabled; then
        # nsys export --type sqlite on every rank's report, then nsys_collective_kernels.sql.
        # Waits for the other nodes' reports to land on the shared output directory first.
        run uv run --no-sync python "${script_dir}/analyse_nsys.py" --output-dir "$output_dir"
    fi
fi
