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
Benchmark NCCL 2.32.3 all-gather / reduce-scatter on GB200 with a C++ program (no torch, no MPI).

Run the same command on every node (DLC style). Each node starts one nvls_collective_bench
process per GPU; NCCL bootstraps through NCCL_COMM_ID=<master-addr>:<port>. One pass per
implementation, because NCCL reads NCCL_NVLS_ENABLE and the CE multicast threshold once per
process. Node 0 merges the passes into compare.md.

Implementations (the kernel is pinned per call with ncclCollConfig_t, forceAlgSelection=1):
  ring    AG / RS algSelection=RING, plain cudaMalloc buffers
  nvls    AG / RS algSelection=NVLS_SIMPLE, plain buffers
  symm    symmetric windows, NCCL picks the symmetric kernel (NCCL_SYM_CE_THRESHOLD keeps AG off the CE)
  stmc    AG = AllGather_STMC (register multimem.st), RS = ReduceScatter_LDMC (multimem.ld_reduce)
  tma     AG = AllGather_TmaSTMC (TMA cp.async.bulk -> multicast), RS = ReduceScatter_LDMC
  ce      AG = copy engine, one copy to the multicast address (CTAPolicy=ZERO), RS = ReduceScatter_LDMC
  ce-uc   AG = copy engine, N-1 unicast copies, RS = ReduceScatter_LDMC

Output: results/gb200/nvls-collective-cpp/gpu-num-<nnodes*gpus-per-node>/<yymmdd-hhmmss>, one
<impl>/ per pass, laid out like ../nvls-collective: results.jsonl, metadata.json, nccl-raw/,
comm/ and per test <impl>/cases/<case>/ (e.g. tma/cases/single-n4_ag_bf16_1024MiB/) with every
member rank's NCCL log of that test, case.json and, with --nsys, <impl>_<case>_rank<RR>.nsys-rep.
A test fails unless NCCL's TUNING log shows the pinned implementation for every launch.
Node 0 builds the binaries first, then stamps the run id and hands it to the other nodes
(run_timestamp_and_logs helper, TCPStore on MASTER_ADDR:<master-port - 1>), so the repo must
be on storage every node can see. An exported RUN_TIME pins the id.

Usage:
  run_nvls_collective_cpp.sh [options]

Options:
  --nnodes N              Nodes in the job (default: NNODES or 1)
  --node-rank N           This node's rank (default: NODE_RANK, RANK or 0)
  --master-addr ADDR      Bootstrap host (default: MASTER_ADDR, DLC_MASTER_ADDR or 127.0.0.1)
  --master-port PORT      First NCCL_COMM_ID port; implementation i uses PORT+i (default: 29711)
  --gpus-per-node N       GPUs per node (default: 4, one GB200 compute tray)
  --impls LIST            ring,nvls,symm,stmc,tma,ce,ce-uc (default: ring,nvls,symm,stmc,tma,ce)
  --group-sizes LIST      Single-group sizes (default: 4,8,16,32 capped at nnodes*gpus-per-node)
  --split-totals LIST     Ranks shared by two concurrent half-groups (default: 16 if available)
  --split-layouts LIST    contiguous,interleaved (default: contiguous)
  --workloads LIST        ag,rs,ag+rs (default: all); with both ag and rs, split scenarios also run
                          the cross case (group A all-gathers while group B reduce-scatters)
  --sizes-mib LIST        Full tensor sizes in MiB (default: 16,64,256,1024)
  --dtype bf16|fp32       Payload dtype (default: bf16)
  --warmup N              Untimed iterations per case, >= 1 (default: 5)
  --iters N               Back-to-back iterations per trial (default: 20)
  --trials N              Timed trials, median reported (default: 5)
  --nccl-home DIR         NCCL 2.32.3 build (default: NCCL_HOME or <repo>/nccl/nccl-2.32.3/build)
  --cuda-home DIR         CUDA toolkit (default: CUDA_HOME or /usr/local/cuda)
  --no-build              Use the existing build/ binaries
  --nsys                  One Nsight Systems report per rank per case (capture range = timed trials);
                          node 0 then files them under cases/ and writes nsys_kernels.md
                          (../nvls-collective/analyse_nsys.py, Python standard library only)
  --dry-run               Print commands only
  -h, --help              Show this help

Examples:
  # One tray (4 GPUs): smoke test of the TMA and CE AllGather against the register kernel.
  bash examples/gb200/nvls-collective-cpp/run_nvls_collective_cpp.sh \
    --impls stmc,tma,ce --sizes-mib 64 --trials 2

  # Eight trays in one NVL72 domain (32 GPUs); run the same command on every node.
  bash examples/gb200/nvls-collective-cpp/run_nvls_collective_cpp.sh --nnodes 8
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

# Same run-id helper as ../nvls-collective; log/warn first so the helper keeps these versions.
log() { printf '%s\n' "$*" >&2; }
warn() { printf 'warning: %s\n' "$*" >&2; }
# shellcheck source=../../../scripts/scale-down/common/run_timestamp_and_logs.sh
source "${repo_root}/scripts/scale-down/common/run_timestamp_and_logs.sh"

nnodes="${NNODES:-1}"
node_rank="${NODE_RANK:-${RANK:-0}}"
master_addr="${MASTER_ADDR:-${DLC_MASTER_ADDR:-127.0.0.1}}"
master_port="29711"
gpus_per_node="4"
impls="ring,nvls,symm,stmc,tma,ce"
group_sizes=""
split_totals=""
split_layouts="contiguous"
workloads="ag,rs,ag+rs"
sizes_mib="16,64,256,1024"
dtype="bf16"
warmup="5"
iters="20"
trials="5"
nccl_home="${NCCL_HOME:-${repo_root}/nccl/nccl-2.32.3/build}"
cuda_home="${CUDA_HOME:-/usr/local/cuda}"
build=true
nsys_enabled=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nnodes) need_value "$@"; nnodes="$2"; shift 2 ;;
        --node-rank) need_value "$@"; node_rank="$2"; shift 2 ;;
        --master-addr) need_value "$@"; master_addr="$2"; shift 2 ;;
        --master-port) need_value "$@"; master_port="$2"; shift 2 ;;
        --gpus-per-node) need_value "$@"; gpus_per_node="$2"; shift 2 ;;
        --impls) need_value "$@"; impls="$2"; shift 2 ;;
        --group-sizes) need_value "$@"; group_sizes="$2"; shift 2 ;;
        --split-totals) need_value "$@"; split_totals="$2"; shift 2 ;;
        --split-layouts) need_value "$@"; split_layouts="$2"; shift 2 ;;
        --workloads) need_value "$@"; workloads="$2"; shift 2 ;;
        --sizes-mib) need_value "$@"; sizes_mib="$2"; shift 2 ;;
        --dtype) need_value "$@"; dtype="$2"; shift 2 ;;
        --warmup) need_value "$@"; warmup="$2"; shift 2 ;;
        --iters) need_value "$@"; iters="$2"; shift 2 ;;
        --trials) need_value "$@"; trials="$2"; shift 2 ;;
        --nccl-home) need_value "$@"; nccl_home="$2"; shift 2 ;;
        --cuda-home) need_value "$@"; cuda_home="$2"; shift 2 ;;
        --no-build) build=false; shift ;;
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
(( master_port > 1 )) || die "--master-port must be > 1 (master-port - 1 hosts the run-dir handshake)"
world_size=$(( nnodes * gpus_per_node ))

if [[ -z "$group_sizes" ]]; then
    for n in 4 8 16 32; do
        (( n <= world_size )) && group_sizes+="${group_sizes:+,}$n"
    done
    [[ -n "$group_sizes" ]] || group_sizes="$world_size"
fi
if [[ -z "$split_totals" ]] && (( world_size >= 16 )); then
    split_totals="16"
fi
IFS=, read -r -a impl_list <<<"$impls"
for impl in "${impl_list[@]}"; do
    case "$impl" in
        ring|nvls|symm|stmc|tma|ce|ce-uc) ;;
        *) die "--impls accepts ring,nvls,symm,stmc,tma,ce,ce-uc; got $impl" ;;
    esac
done
if (( nnodes > 1 )) && [[ "$master_addr" == 127.0.0.1 ]]; then
    die "multi-node runs require --master-addr or MASTER_ADDR"
fi
for var in NCCL_ALGO NCCL_PROTO NCCL_SYM_KERNEL NCCL_CTA_POLICY NCCL_SYM_CE_THRESHOLD NCCL_CE_COLL_AG_MULTICAST_THRESHOLD; do
    [[ -n "${!var:-}" ]] && echo "WARNING: ${var} from the environment is removed; the kernel is pinned per call" >&2
done

# Bootstrap interface: eth0 for one 4-GPU node (DSW), bond1 for more than 4 GPUs (DLC).
select_ifname() {
    local preferred=eth0 other=bond1 available
    if (( world_size > 4 )); then
        preferred=bond1
        other=eth0
    fi
    if ! command -v ip >/dev/null; then
        $DRY_RUN || die "the ip command is required to choose the bootstrap interface"
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
else
    gloo_ifname="$(select_ifname)"
fi
nccl_socket_ifname="${NCCL_SOCKET_IFNAME:-$gloo_ifname}"

run() {
    if $DRY_RUN; then
        printf '%q ' "$@"
        printf '\n'
    else
        "$@"
    fi
}

build_dir="${script_dir}/build"
bench="${build_dir}/nvls_collective_bench"
report="${build_dir}/collective_report"
# Node 0 builds before it publishes the run id, so the other nodes (blocked in the handshake)
# find finished binaries on the shared file system.
if $build && (( node_rank == 0 )); then
    run make -C "$script_dir" NCCL_HOME="$nccl_home" CUDA_HOME="$cuda_home" BUILD="$build_dir" all
fi

results_root="${repo_root}/results/gb200/nvls-collective-cpp"
export MASTER_ADDR="$master_addr"
export MASTER_PORT="$(( master_port - 1 ))"
export NNODES="$nnodes"
export NODE_RANK="$node_rank"
export GLOO_SOCKET_IFNAME="$gloo_ifname"
resolve_run_time "$DRY_RUN"
[[ -n "$RUN_TIME" ]] || die "could not resolve the run id"
output_dir="${results_root}/gpu-num-${world_size}/${RUN_TIME}"

if ! $DRY_RUN; then
    [[ -x "$bench" && -x "$report" ]] || die "missing ${bench} or ${report}; build with make or drop --no-build"
    [[ -f "${nccl_home}/lib/libnccl.so" ]] || die "${nccl_home}/lib/libnccl.so not found (--nccl-home)"
    if $nsys_enabled; then
        command -v nsys >/dev/null || die "nsys is required when --nsys is enabled"
        nsys --version >&2
    fi
fi

bench_args=(
    --group-sizes "$group_sizes"
    --split-layouts "$split_layouts"
    --workloads "$workloads"
    --sizes-mib "$sizes_mib"
    --dtype "$dtype"
    --warmup "$warmup"
    --iters "$iters"
    --trials "$trials"
    --split-totals "${split_totals:-none}"
)
$nsys_enabled && bench_args+=(--nsys)

echo "world_size=${world_size} nnodes=${nnodes} node_rank=${node_rank} group_sizes=${group_sizes}" \
    "split_totals=${split_totals:-none} ifname=${gloo_ifname} nccl_home=${nccl_home} output_dir=${output_dir}" >&2
$DRY_RUN || mkdir -p "$output_dir"

pids=()
kill_ranks() {
    local pid
    for pid in ${pids[@]+"${pids[@]}"}; do
        kill "$pid" 2>/dev/null || true
    done
}
trap 'kill_ranks' INT TERM

for index in "${!impl_list[@]}"; do
    impl="${impl_list[$index]}"
    impl_dir="${output_dir}/${impl}"
    # Whole-process NCCL logs; the benchmark copies each test's slice into <impl>/cases/<case>/.
    nccl_raw_dir="${impl_dir}/nccl-raw"
    node_log="${impl_dir}/bench-node${node_rank}.log"
    $DRY_RUN || mkdir -p "$nccl_raw_dir"
    impl_env=(
        # Per-call configs choose the kernel; global overrides would narrow or replace them.
        -u NCCL_ALGO -u NCCL_PROTO -u NCCL_SYM_KERNEL -u NCCL_CTA_POLICY -u NCCL_SYM_CE_THRESHOLD
        -u NCCL_CE_COLL_AG_MULTICAST_THRESHOLD
        LD_LIBRARY_PATH="${nccl_home}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        NCCL_COMM_ID="${master_addr}:$(( master_port + index ))"
        WORLD_SIZE="$world_size"
        # Two communicators run concurrently: keep their launch order identical on every rank.
        NCCL_LAUNCH_ORDER_IMPLICIT=1
        NCCL_DEBUG=INFO
        # TUNING carries the per-launch kernel line; INIT carries nNodes/MNNVL/symmetric multimem.
        NCCL_DEBUG_SUBSYS=INIT,ENV,TUNING
        NCCL_DEBUG_FILE="${nccl_raw_dir}/nccl-%h-%p.log"
        NCCL_SOCKET_IFNAME="$nccl_socket_ifname"
    )
    case "$impl" in
        ring) impl_env+=(NCCL_NVLS_ENABLE=0) ;;
        nvls) impl_env+=(NCCL_NVLS_ENABLE=1) ;;
        # Without a selection, an AllGather whose per-rank count exceeds 8Mi elements runs on the
        # copy engines (enqueue.cc, Blackwell + NVLink + registered window). INT64_MAX disables
        # that branch, so symm measures the symmetric kernels like ../nvls-collective.
        symm) impl_env+=(NCCL_NVLS_ENABLE=1 NCCL_WIN_ENABLE=1 NCCL_SYM_CE_THRESHOLD=9223372036854775807) ;;
        stmc|tma) impl_env+=(NCCL_NVLS_ENABLE=1 NCCL_WIN_ENABLE=1) ;;
        # threshold >= 0 bypasses the CE cost model (ce_coll.cc): MC iff shard <= threshold.
        ce) impl_env+=(NCCL_NVLS_ENABLE=1 NCCL_WIN_ENABLE=1 NCCL_CE_COLL_AG_MULTICAST_THRESHOLD=4611686018427387904) ;;
        ce-uc) impl_env+=(NCCL_NVLS_ENABLE=1 NCCL_WIN_ENABLE=1 NCCL_CE_COLL_AG_MULTICAST_THRESHOLD=0) ;;
    esac
    # Cross-tray groups must use the NVLink fabric; forcing MNNVL makes a fallback to IB fail.
    (( nnodes > 1 )) && impl_env+=(NCCL_MNNVL_ENABLE=1)
    if $nsys_enabled; then
        nsys_dir="${impl_dir}/nsys"
        $DRY_RUN || mkdir -p "$nsys_dir"
        if ! $DRY_RUN && compgen -G "${nsys_dir}/*.nsys-rep" >/dev/null; then
            die "${nsys_dir} already contains Nsight reports; ${output_dir} was not a fresh run directory"
        fi
        impl_env+=(NSYS_TMPDIR="${NSYS_TMPDIR:-${nsys_dir}/.tmp}")
        $DRY_RUN || mkdir -p "${NSYS_TMPDIR:-${nsys_dir}/.tmp}"
    fi

    echo "=== impl=${impl} (${impl_env[*]})" >&2
    pids=()
    for (( local_rank = 0; local_rank < gpus_per_node; local_rank++ )); do
        rank=$(( node_rank * gpus_per_node + local_rank ))
        cmd=("$bench" --impl "$impl" --output-dir "$impl_dir" "${bench_args[@]}")
        if $nsys_enabled; then
            # Each process gets its own nsys session. Only the timed trials of each case are
            # captured (cudaProfilerStart/Stop in the benchmark); "repeat" writes one report per
            # case. analyse_nsys.py files each report under its case by the outer NVTX label.
            cmd=(nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none
                --capture-range=cudaProfilerApi --capture-range-end=repeat --force-overwrite=false
                --output="${nsys_dir}/rank%q{RANK}" "${cmd[@]}")
        fi
        if $DRY_RUN; then
            run env "${impl_env[@]}" RANK="$rank" LOCAL_RANK="$local_rank" "${cmd[@]}"
        else
            # All ranks of this node append to one log, like torchrun-node<k>.log next door.
            env "${impl_env[@]}" RANK="$rank" LOCAL_RANK="$local_rank" "${cmd[@]}" >>"$node_log" 2>&1 &
            pids+=("$!")
        fi
    done
    failed=0
    # Reap ranks in completion order; the first failure aborts the pass, because the remaining
    # ranks would wait for it in NCCL forever.
    for (( remaining = ${#pids[@]}; remaining > 0; remaining-- )); do
        if ! wait -n; then
            failed=1
            kill_ranks
            break
        fi
    done
    if (( failed )); then
        wait 2>/dev/null || true
        tail -n 40 "$node_log" >&2 || true
        die "impl=${impl} failed on node ${node_rank}; logs in ${impl_dir}"
    fi
    $DRY_RUN || cat "$node_log" >&2
    if $nsys_enabled && ! $DRY_RUN; then
        compgen -G "${nsys_dir}/rank*.nsys-rep" >/dev/null \
            || die "nsys wrote no report under ${nsys_dir}; inspect ${node_log}"
    fi
done

if (( node_rank == 0 )); then
    report_args=()
    for impl in "${impl_list[@]}"; do
        report_args+=(--input "${impl}=${output_dir}/${impl}/results.jsonl")
    done
    run "$report" "${report_args[@]}" --output "${output_dir}/compare.md"
    if $nsys_enabled; then
        # nsys export --type sqlite on every rank's report, file it under cases/<case>/ and run
        # ../nvls-collective/nsys_collective_kernels.sql. Waits for the other nodes' reports first.
        run uv run --no-sync python "${script_dir}/../nvls-collective/analyse_nsys.py" --output-dir "$output_dir"
    fi
fi
