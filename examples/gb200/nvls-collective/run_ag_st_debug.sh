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
Sweep the AllGather_STMC pause and time the multimem.st of one 1 GiB AllGather per pause, with
nccl-2.32.3-pause-with-debug preloaded. Every pause is one test with one output file.

Every GPU of the job is in one group. For each value of --pause-cycles, one torchrun job (NCCL reads the
variable once per process) runs ag_st_debug_benchmark.py: --warmup AllGathers that print nothing, then
one measured AllGather of --size-mib (the output; every rank stores its size / n shard). The library
(install/install-nccl-2.32.3-pause-with-debug.sh) reads clock64 right before and right after the 8
multimem.st of every 4 KiB chunk; lane 0 of each warp prints one "[nccl-ag-stmc-st]" line (sum / min /
max / last cycles) per --print-interval chunks, and one more for the chunks left at the end. Every chunk
is timed, only the lines are spaced out; a countdown, not a division, spaces them. The pause
(NCCL_SYM_AG_STMC_PAUSE_CYCLES, as in nccl-2.32.3-pause) spins every warp after each chunk, outside
the timed window.

The window is the issue of the chunk's stores: they are posted, so it covers the wait for the
chunk's ld.global data (the first store needs it) and the LSU / NVLink backpressure, not the stores
landing. Cycles are SM cycles: ns = cycles / SM clock (pin it with nvidia-smi -lgc).

How many lines: a rank stores size / n / 4 KiB chunks (1 GiB: 262144 / n), split over the
AllGather_STMC warps (CTAs x 16). At NCCL's own CTA counts for 1 GiB (AllGather_STMC 4 / 2 / 1 / 1 / 1
CTAs at 4 / 8 / 16 / 32 / 64 GPUs) every warp stores 16384 / (n x CTAs) chunks:
  GPUs  CTAs  warps/rank  chunks/rank  chunks/warp  lines/rank at --print-interval 64
     4     4          64        65536         1024                             1024
     8     2          32        32768         1024                              512
    16     1          16        16384         1024                              256
    32     1          16         8192          512                              128
    64     1          16         4096          256                               64
Lines per rank are chunks/rank / interval (plus at most one tail line per warp), whatever the CTA
count or the pause. Each line is a ~100-byte record in the device printf FIFO (1 MiB by default,
emptied at the next synchronization; records past its end are dropped), so the script refuses more
than 4096 lines per rank. 64 chunks are a ~83 us window at ~1.3 us per chunk; a device printf costs
its warp on the order of a microsecond (an estimate, about 1% of that window), and it falls outside
every st window.

Why warmup, then one printed AllGather: the first launch is cold (lazy kernel loading, first touch of
the multicast mapping, SM clock still ramping up), so its cycles are not those of steady state.
Printing every launch multiplies the lines (and the FIFO pressure) without new information. The
library counts each communicator's AllGather_STMC launches: NCCL_SYM_AG_STMC_ST_PRINT_SKIP=--warmup
and NCCL_SYM_AG_STMC_ST_PRINT_LAUNCHES=1 make the measured launch the only one that prints.

Output, one file per pause (node 0 writes it from what every rank sends over Gloo):
  results/gb200/nvls-collective/ag-st-debug/gpu-num-<N>/<id>/pause<P>cyc.log
It starts with "# key=value" lines (status, library banner, kernel, CTAs, warps per CTA, sizes, the four
NCCL_SYM_AG_STMC_* values, then per rank: time, busbw, lines, warps, chunks, multimem.st cycles per
chunk mean / min / max), followed by every rank's raw "[nccl-ag-stmc-st]" lines sorted by rank, CTA,
warp and chunk. A run that fails a check writes pause<P>cyc.failed.log instead, with "# error:" lines.
<id> is shared by the whole sweep (an exported RUN_TIME pins it; on several nodes node 0 stamps it and
hands it out on master-port - 1).

Each rank checks its own run: the AllGather output, its lines covering every chunk it stored, the
"NCCL version 2.32.3-pause-with-debug+cuda..." banner and the four values in its NCCL log, and that no
CTA variable reached NCCL; rank 0 also checks that every AllGather launch was AllGather_STMC and that
CTAs x warps per CTA warps printed. The per-rank printf / NCCL logs and the torchrun log go to
<id>/.work/pause<P>cyc.node<K>/ on each node, which is removed when the run passes and kept otherwise.
Process-wide CTA variables (GB200 images export NCCL_MIN_CTAS=4) are unset first, so NCCL picks its
own CTA count.

Run the same command on every node.

Usage:
  run_ag_st_debug.sh [options]

Options:
  --nnodes N            Nodes in the job (default: NNODES or 1)
  --gpus-per-node N     GPUs per node (default: 4)
  --node-rank N         This node's rank (default: NODE_RANK, RANK or 0)
  --master-addr ADDR    Rendezvous host (default: MASTER_ADDR, DLC_MASTER_ADDR or 127.0.0.1)
  --master-port PORT    Rendezvous port of the first pause, + i for the i-th; PORT - 1 agrees on the
                        run id (default: 29631)
  --pause-cycles LIST   SM cycles per 4 KiB chunk and warp, one test each (default: per group size, as in
                        run_symm_cta_pause_sweep.sh:
                          4 / 8 / 16 GPUs: 0,128,256,512,768,1024,1536,2048,2560,3072,4096,6144
                          32 GPUs:         0,2048,3072,3584,4096,5120,6144,7168,8192,10240,12288,14336
                          64 GPUs:         0,6144,8192,9728,11264,12288,14336,16384,18432,22528,26624,31744
                        other sizes need an explicit list)
  --size-mib N          AllGather output size (default: 1024)
  --warmup N            AllGathers before the measured one (default: 2)
  --print-interval N    Chunks per warp between two lines (default: 64)
  --venv DIR            Virtualenv to activate, or 'none' to keep the current one
                        (default: <repo>/.venv/python3.12-torch2.13-nccl2.32.3)
  --nccl-lib PATH       libnccl.so.2 to preload
                        (default: <repo>/nccl/nccl-2.32.3-pause-with-debug/build/lib/libnccl.so.2)
  --dry-run             Print the torchrun commands instead of running them
  -h, --help            Show this help

Examples:
  # One tray, 4 GPUs, the default 4-GPU pauses.
  bash examples/gb200/nvls-collective/run_ag_st_debug.sh

  # One tray, three pauses, one line per 128 chunks.
  bash examples/gb200/nvls-collective/run_ag_st_debug.sh --pause-cycles 0,1024,4096 --print-interval 128
USAGE
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
nnodes="${NNODES:-1}"
gpus_per_node=4
node_rank="${NODE_RANK:-${RANK:-0}}"
master_addr="${MASTER_ADDR:-${DLC_MASTER_ADDR:-127.0.0.1}}"
master_port=29631
pause_cycles=""
size_mib=1024
warmup=2
print_interval=64
venv="${repo_root}/.venv/python3.12-torch2.13-nccl2.32.3"
nccl_lib="${repo_root}/nccl/nccl-2.32.3-pause-with-debug/build/lib/libnccl.so.2"
dry_run=false
nccl_suffix="-pause-with-debug"
debug_envs=(NCCL_SYM_AG_STMC_PAUSE_CYCLES NCCL_SYM_AG_STMC_ST_PRINT_INTERVAL NCCL_SYM_AG_STMC_ST_PRINT_SKIP
    NCCL_SYM_AG_STMC_ST_PRINT_LAUNCHES)
# ~100-byte records in the 1 MiB device printf FIFO; half of it leaves room for the tail lines.
max_lines_per_rank=4096
chunk_bytes=4096

die() {
    echo "ERROR: $*" >&2
    exit 1
}
# The common helpers log to stdout by default; keep their messages out of the run id captured below.
log() { printf '%s\n' "$*" >&2; }

# NCCL picks the CTA count itself; GB200 images export NCCL_MIN_CTAS=4, which would raise AllGather_STMC
# to min(4, divUp(32, n)) CTAs. The worker fails a run whose NCCL log still shows one of them.
cta_env=(NCCL_MIN_CTAS NCCL_MAX_CTAS NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS NCCL_MIN_NRINGS NCCL_MAX_NRINGS
    NCCL_NVLS_NCHANNELS NCCL_MIN_P2P_NCHANNELS NCCL_MAX_P2P_NCHANNELS NCCL_SYM_CTAS)
for name in "${cta_env[@]}"; do
    [[ -n "${!name:-}" ]] && log "warning: unsetting ${name}=${!name}; the AllGather runs at NCCL's own CTA count"
    unset "$name"
done

need_value() { [[ $# -ge 2 && -n "$2" ]] || die "$1 needs a value"; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --nnodes) need_value "$@"; nnodes="$2"; shift 2 ;;
        --gpus-per-node) need_value "$@"; gpus_per_node="$2"; shift 2 ;;
        --node-rank) need_value "$@"; node_rank="$2"; shift 2 ;;
        --master-addr) need_value "$@"; master_addr="$2"; shift 2 ;;
        --master-port) need_value "$@"; master_port="$2"; shift 2 ;;
        --pause-cycles) need_value "$@"; pause_cycles="$2"; shift 2 ;;
        --size-mib) need_value "$@"; size_mib="$2"; shift 2 ;;
        --warmup) need_value "$@"; warmup="$2"; shift 2 ;;
        --print-interval) need_value "$@"; print_interval="$2"; shift 2 ;;
        --venv) need_value "$@"; venv="$2"; shift 2 ;;
        --nccl-lib) need_value "$@"; nccl_lib="$2"; shift 2 ;;
        --dry-run) dry_run=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1 (see --help)" ;;
    esac
done

for name in nnodes gpus_per_node size_mib print_interval; do
    [[ "${!name}" =~ ^[0-9]+$ ]] && (( 10#${!name} >= 1 )) || die "--${name//_/-} must be a positive integer, got ${!name}"
done
for name in node_rank master_port warmup; do
    [[ "${!name}" =~ ^[0-9]+$ ]] || die "--${name//_/-} must be a non-negative integer, got ${!name}"
done
(( node_rank < nnodes )) || die "--node-rank must be in [0, nnodes)"
(( master_port > 1 )) || die "--master-port must be > 1 (master-port - 1 agrees on the run id)"
(( print_interval <= 4294967295 )) || die "--print-interval must fit the kernel's uint32"
if (( nnodes > 1 )) && [[ "$master_addr" == 127.0.0.1 ]]; then
    die "multi-node runs require --master-addr or MASTER_ADDR"
fi

world_size=$(( nnodes * gpus_per_node ))
(( world_size >= 3 )) || die "AllGather_STMC needs multicast, i.e. a group of >= 3 GPUs; got ${world_size}"

# The same defaults as run_symm_cta_pause_sweep.sh, so the two sweeps line up pause for pause.
if [[ -z "$pause_cycles" ]]; then
    case "$world_size" in
        4|8|16) pause_cycles="0,128,256,512,768,1024,1536,2048,2560,3072,4096,6144" ;;
        32) pause_cycles="0,2048,3072,3584,4096,5120,6144,7168,8192,10240,12288,14336" ;;
        64) pause_cycles="0,6144,8192,9728,11264,12288,14336,16384,18432,22528,26624,31744" ;;
        *) die "no default --pause-cycles for a ${world_size}-GPU group; pass --pause-cycles" ;;
    esac
    log "pause cycles for the ${world_size}-GPU group: ${pause_cycles}"
fi
[[ "$pause_cycles" =~ ^[0-9]+(,[0-9]+)*$ ]] \
    || die "--pause-cycles must be a comma-separated list of cycles, got '${pause_cycles}'"
IFS=, read -r -a pause_list <<<"$pause_cycles"
for (( i = 0; i < ${#pause_list[@]}; i++ )); do
    pause_list[i]=$(( 10#${pause_list[i]} ))
    (( pause_list[i] <= 4294967295 )) || die "--pause-cycles: ${pause_list[i]} does not fit the kernel's uint32 cycles"
    for (( j = 0; j < i; j++ )); do
        (( pause_list[i] != pause_list[j] )) || die "--pause-cycles lists ${pause_list[i]} twice"
    done
done
(( master_port + ${#pause_list[@]} - 1 <= 65535 )) || die "--master-port + pauses runs past port 65535"

size_bytes=$(( size_mib * 1024 * 1024 ))
(( size_bytes % (world_size * chunk_bytes) == 0 )) \
    || die "--size-mib ${size_mib} does not split into ${world_size} shards of whole 4 KiB chunks"
chunks_per_rank=$(( size_bytes / world_size / chunk_bytes ))
lines_per_rank=$(( chunks_per_rank / print_interval ))
log "AllGather ${size_mib} MiB over ${world_size} GPUs: ${chunks_per_rank} chunks of 4 KiB per rank;" \
    "one line per ${print_interval} chunks: ${lines_per_rank} lines per rank (+ at most one tail line per warp)"
case "$world_size" in
    4) ctas=4 ;;
    8) ctas=2 ;;
    16|32|64) ctas=1 ;;
    *) ctas="" ;;
esac
if [[ -n "$ctas" ]] && (( size_mib == 1024 )); then
    log "  at NCCL's own ${ctas} AllGather_STMC CTAs (16 warps each): $(( chunks_per_rank / (ctas * 16) )) chunks" \
        "and $(( chunks_per_rank / (ctas * 16) / print_interval )) lines per warp"
fi
(( lines_per_rank <= max_lines_per_rank )) \
    || die "${lines_per_rank} lines per rank overflow the 1 MiB device printf FIFO; pass --print-interval" \
        ">= $(( (chunks_per_rank + max_lines_per_rank - 1) / max_lines_per_rank ))"

if [[ "$venv" != none ]]; then
    [[ -f "${venv}/bin/activate" ]] || die "no virtualenv at ${venv}; pass --venv DIR or --venv none"
    # shellcheck source=/dev/null
    source "${venv}/bin/activate"
fi
for name in "${debug_envs[@]}" NCCL_DEBUG_FILE; do
    [[ -n "${!name:-}" ]] && log "warning: ${name}=${!name} from the environment is replaced for every run"
done

[[ -f "$nccl_lib" ]] || die "${nccl_lib} not found; build it with install/install-nccl-2.32.3-pause-with-debug.sh or pass --nccl-lib"
nccl_lib="$(cd -- "$(dirname -- "$nccl_lib")" && pwd -P)/$(basename -- "$nccl_lib")"
grep -aq "NCCL version 2.32.3${nccl_suffix}+cuda" "$nccl_lib" \
    || die "${nccl_lib} is not the nccl-2.32.3${nccl_suffix} build (no \"NCCL version 2.32.3${nccl_suffix}+cuda\" banner)"
for name in "${debug_envs[@]}"; do
    grep -aq "$name" "$nccl_lib" || die "${nccl_lib} does not read ${name}"
done
preload="${nccl_lib}${LD_PRELOAD:+:${LD_PRELOAD}}"

cd "$repo_root"
if ! $dry_run; then
    echo "=== Checking that torch maps only ${nccl_lib}" >&2
    LD_PRELOAD="$preload" uv run --no-sync python - "$nccl_lib" <<'PY'
import os
import sys

import torch  # noqa: F401  (libtorch_cuda.so needs libnccl.so.2; the official wheel also preloads its own)


want = os.path.realpath(sys.argv[1])
with open("/proc/self/maps") as maps:
    paths = {line.split(None, 5)[5].strip() for line in maps if len(line.split(None, 5)) == 6}
loaded = sorted({os.path.realpath(p) for p in paths if os.path.basename(p).startswith("libnccl.so")})
if loaded != [want]:
    sys.exit(f"ERROR: torch {torch.__version__} maps libnccl {loaded}, expected only {want}; LD_PRELOAD needs a torch "
             "without nvidia-nccl-cu13 (install/install-torch-2.13.0-nccl-2.32.3.sh)")
print(f"  [ok]   torch {torch.__version__} maps only {want}", file=sys.stderr)
PY
fi

# shellcheck source=../../../scripts/scale-down/common/gloo_socket_ifname.sh
source "${repo_root}/scripts/scale-down/common/gloo_socket_ifname.sh"
# shellcheck source=../../../scripts/scale-down/common/run_timestamp_and_logs.sh
source "${repo_root}/scripts/scale-down/common/run_timestamp_and_logs.sh"
# Gloo (the worker's default group and the run-id handshake) and NCCL's bootstrap use this interface;
# a dry run keeps an exported one.
$dry_run || resolve_gloo_socket_ifname "$nnodes"
# One id for the whole sweep: node 0 stamps it and hands it out on master-port - 1, unless RUN_TIME is exported.
sweep_id="$(
    export NNODES="$nnodes" NODE_RANK="$node_rank" MASTER_ADDR="$master_addr" MASTER_PORT="$(( master_port - 1 ))"
    resolve_run_time "$dry_run"
)"
[[ -n "$sweep_id" ]] || die "could not resolve the sweep id"
output_dir="${repo_root}/results/gb200/nvls-collective/ag-st-debug/gpu-num-${world_size}/${sweep_id}"
work_root="${output_dir}/.work"
echo "=== output_dir=${output_dir}" >&2

int64_max=9223372036854775807
base_env=(
    NCCL_SYM_AG_STMC_ST_PRINT_INTERVAL="$print_interval"
    NCCL_SYM_AG_STMC_ST_PRINT_SKIP="$warmup"
    NCCL_SYM_AG_STMC_ST_PRINT_LAUNCHES=1
    # The symmetric multicast AllGather on registered windows, never AllGather_TmaSTMC (it stores through the
    # TMA and is not timed) and never the copy engines.
    NCCL_WIN_ENABLE=1
    NCCL_NVLS_ENABLE=1
    NCCL_SYM_KERNEL="allgather:AllGather_STMC"
    NCCL_SYM_TMA_ENABLE=0
    NCCL_SYM_CE_THRESHOLD="$int64_max"
    # The worker points NCCL_DEBUG_FILE at its own scratch file and checks it.
    NCCL_DEBUG=INFO
    NCCL_DEBUG_SUBSYS=INIT,ENV,TUNING
    LD_PRELOAD="$preload"
)
if [[ -n "${GLOO_SOCKET_IFNAME:-}" ]]; then
    base_env+=(GLOO_SOCKET_IFNAME="$GLOO_SOCKET_IFNAME" NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$GLOO_SOCKET_IFNAME}")
fi
# Cross-tray groups must use the NVLink fabric; forcing MNNVL makes a fallback to IB fail.
(( nnodes > 1 )) && base_env+=(NCCL_MNNVL_ENABLE=1)

status=0
results=()
for (( i = 0; i < ${#pause_list[@]}; i++ )); do
    pause="${pause_list[i]}"
    output_file="${output_dir}/pause${pause}cyc.log"
    scratch_dir="${work_root}/pause${pause}cyc.node${node_rank}"
    command=(
        env -u NCCL_ALGO -u NCCL_DEBUG_FILE "${base_env[@]}" NCCL_SYM_AG_STMC_PAUSE_CYCLES="$pause"
        uv run --no-sync python -m torch.distributed.run
        --nproc_per_node="$gpus_per_node"
        --nnodes="$nnodes"
        --node_rank="$node_rank"
        --master_addr="$master_addr"
        --master_port="$(( master_port + i ))"
        "${script_dir}/ag_st_debug_benchmark.py"
        --output-file "$output_file"
        --scratch-dir "$scratch_dir"
        --size-mib "$size_mib"
        --warmup "$warmup"
    )
    echo "=== NCCL_SYM_AG_STMC_PAUSE_CYCLES=${pause} ($(( i + 1 ))/${#pause_list[@]}): ${output_file}" >&2
    if $dry_run; then
        printf '%q ' "${command[@]}"
        printf '\n'
        continue
    fi
    mkdir -p "$scratch_dir"
    set +e
    "${command[@]}" 2>&1 | tee "${scratch_dir}/torchrun-node${node_rank}.log"
    run_status="${PIPESTATUS[0]}"
    set -e
    if (( run_status == 0 )); then
        rm -rf -- "$scratch_dir"
        result="ok"
    else
        result="exit ${run_status}, logs kept in ${scratch_dir}"
        status="$run_status"
    fi
    (( node_rank == 0 )) && [[ -f "${output_dir}/pause${pause}cyc.failed.log" ]] && output_file="${output_dir}/pause${pause}cyc.failed.log"
    results+=("pause ${pause} cycles -> ${output_file} (${result})")
done

if ! $dry_run; then
    rmdir "$work_root" 2>/dev/null || true
    echo "=== Sweep ${sweep_id}" >&2
    printf '  %s\n' ${results[@]+"${results[@]}"} >&2
fi
exit "$status"
