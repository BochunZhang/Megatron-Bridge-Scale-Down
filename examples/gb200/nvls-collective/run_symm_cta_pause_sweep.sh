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
Sweep the AllGather_STMC pause at NCCL's default CTA counts, with nccl-2.32.3-cta-with-pause preloaded.

For every value of --pause-cycles it runs
run_nvls_collective_benchmark.sh --algos symm --workloads ag,rs,ag+rs --sizes-mib 1024 --nsys
--symm-kernel mc --cta-configs default with NCCL_SYM_AG_STMC_PAUSE_CYCLES=<cycles>. Only the pause
changes between runs: no CTA count is set anywhere, so NCCL picks its own. The process-wide CTA
variables (NCCL_MIN_CTAS, NCCL_MIN_NCHANNELS, ..., NCCL_SYM_CTAS) are unset for the runs: GB200
images export NCCL_MIN_CTAS=4, which the nccl-2.32.3-cta patch lets raise the multicast kernels past
their divUp(32, n) cap (64 GPUs would run 4 CTAs instead of 1). After each run the script fails if
an NCCL log shows one of them, and prints the CTA counts of the AG / RS launches.

Every case runs the symmetric multicast kernels, never the unicast (P2P) ones: --symm-kernel mc pins
NCCL_SYM_KERNEL=allgather:AllGather_STMC;reducescatter:ReduceScatter_LDMC with NCCL_NVLS_ENABLE=1,
NCCL_SYM_TMA_ENABLE=0 keeps the AG off AllGather_TmaSTMC, and the benchmark fails any case whose NCCL
log shows another kernel. --algos, --symm-kernel and --cta-configs cannot be passed through.

The preloaded build (install/install-nccl-2.32.3-cta.sh --with-pause) is nccl-2.32.3-cta plus that variable: AllGather_STMC makes every warp spin <cycles> SM
cycles (clock64) after each 4 KiB chunk of multimem stores, so the AllGather gets slower at the same CTA
count and ReduceScatter_LDMC is left alone. 0 runs the kernels unchanged, which makes it the baseline
of the same library. In ns a pause is <cycles> / SM clock: read the clock from the nsys GPU metrics
(GPC Clock Frequency), or pin it with nvidia-smi -lgc before the sweep. ar is not swept: AllReduce_RSxLDMC_AGxSTMC has its own store loop and is not paused;
rs is kept as the unpaused reference of ag+rs.

NCCL reads the variable once per process, so one value per launcher run. Each run is its own
results directory, tagged with the pause:
  results/gb200/nvls-collective/gpu-num-<N>/<id>-pause<cycles>cyc
<id> is shared by the whole sweep (an exported RUN_TIME pins it; on several nodes node 0 stamps it
and hands it out like the launcher does). Node 0 writes pause.json there, and metadata.json of
every algorithm records NCCL_SYM_AG_STMC_PAUSE_CYCLES. Node 0 also tags the pause in the name of
every filed nsys report: <algo>/cases/<algo>-<case>/<algo>-<case>.pause<cycles>cyc.rank_<RR>.nsys-rep.
After each run the script checks that every
NCCL process logged the "NCCL version 2.32.3-cta-pause+cuda..." banner and
"NCCL_SYM_AG_STMC_PAUSE_CYCLES set by environment to <cycles>".

The torch requirements are those of run_symm_cta_sweep.sh: one libnccl.so.2 loaded through its
soname (install/install-torch-2.13.0-nccl-2.32.3.sh), checked before the first run.

Run the same command on every node, like run_nvls_collective_benchmark.sh.

Usage:
  run_symm_cta_pause_sweep.sh [options] [-- run_nvls_collective_benchmark.sh options]

Options:
  --nnodes N          Nodes in the job (default: 1, one 4-GPU tray)
  --group-sizes LIST  Single-group sizes (default: nnodes x --gpus-per-node of the launcher, 4 unless
                      passed through)
  --pause-cycles LIST SM cycles per 4 KiB chunk and warp, one run each (default: per group size, below;
                      a group size without a default, or several group sizes, need an explicit list)

Default --pause-cycles, for the CTA counts NCCL's own model (tuning/sym_model) picks at 1 GiB with
no CTA variable set: AllGather_STMC 4 / 2 / 1 / 1 / 1 CTAs and ReduceScatter_LDMC 6 / 4 / 2 / 1 / 1 at
4 / 8 / 16 / 32 / 64 GPUs (the printed "Kernel AllGather_STMC nchannels" counts show the real ones).
The AG writes n x CTA_AG paths into every GPU, each path 64 KiB per round (16 warps x 4 KiB) and about
1.3 us per round when the ingress is not full (50 GB/s per path), so the AG rate is about
min(711, paths x 64 KiB / (1.3 us + pause / SM clock)) GB/s and T_AG = 16384 / paths rounds at 1 GiB.
At a 2.06 GHz SM clock:
  4 / 8 / 16 GPUs (16 paths): 0,128,256,512,768,1024,1536,2048,2560,3072,4096,6144
      711 up to about 360 cycles, then about 585 / 455 / 375 / 320 / 245 GB/s at
      1024 / 2048 / 3072 / 4096 / 6144 (T_AG +0.5 ms per 1000 cycles past the knee)
  32 GPUs (32 paths):         0,2048,3072,3584,4096,5120,6144,7168,8192,10240,12288,14336
      711 up to about 3400 cycles, then about 640 / 490 / 400 / 335 / 255 GB/s at
      4096 / 6144 / 8192 / 10240 / 14336 (T_AG +0.25 ms per 1000 cycles past the knee)
  64 GPUs (64 paths):         0,6144,8192,9728,11264,12288,14336,16384,18432,22528,26624,31744
      711 up to about 9500 cycles, then about 620 / 510 / 410 / 345 / 250 GB/s at
      11264 / 14336 / 18432 / 22528 / 31744 (T_AG +0.12 ms per 1000 cycles past the knee)
The values below each knee check the plateau (AG still at 711) and bracket the knee for a 1.0-1.6 us
round; the rest walk the AG from about 690 down to about 250 GB/s, past the rate at which a fully
overlapped AG+RS (T_pair = T_AG) stops beating the default T_pair (about 340-360 GB/s at 1 GiB).
If the printed AG CTA counts differ, recompute the knee with paths = n x CTA_AG as
(paths x 64 KiB / AG rate at pause 0 - 1.3 us) x SM clock and pass --pause-cycles.
  --venv DIR          Virtualenv to activate, or 'none' to keep the current one
                      (default: <repo>/.venv/python3.12-torch2.13-nccl2.32.3)
  --nccl-lib PATH     libnccl.so.2 to preload
                      (default: <repo>/nccl/nccl-2.32.3-cta-with-pause/build/lib/libnccl.so.2)
  -h, --help          Show this help

Anything after the options (or after --) goes to run_nvls_collective_benchmark.sh and, coming
last, overrides the defaults above, e.g. --workloads ag or --dry-run (but not --algos, --symm-kernel
or --cta-configs).

Examples:
  # Eight trays, one 32-GPU group, the 32-GPU pauses (run on every node).
  bash examples/gb200/nvls-collective/run_symm_cta_pause_sweep.sh --nnodes 8

  # One tray, 4 GPUs, three pauses.
  bash examples/gb200/nvls-collective/run_symm_cta_pause_sweep.sh --pause-cycles 0,1000,4000
USAGE
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
nnodes=1
group_sizes=""
pause_cycles=""
venv="${repo_root}/.venv/python3.12-torch2.13-nccl2.32.3"
nccl_lib="${repo_root}/nccl/nccl-2.32.3-cta-with-pause/build/lib/libnccl.so.2"
nccl_suffix="-cta-pause"
pause_env="NCCL_SYM_AG_STMC_PAUSE_CYCLES"
passthrough=()

die() {
    echo "ERROR: $*" >&2
    exit 1
}
log() { printf '%s\n' "$*" >&2; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nnodes) nnodes="$2"; shift 2 ;;
        --group-sizes) group_sizes="$2"; shift 2 ;;
        --pause-cycles) pause_cycles="$2"; shift 2 ;;
        --venv) venv="$2"; shift 2 ;;
        --nccl-lib) nccl_lib="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; passthrough+=("$@"); break ;;
        *) passthrough+=("$1"); shift ;;
    esac
done

[[ "$nnodes" =~ ^[0-9]+$ ]] && (( nnodes >= 1 )) || die "--nnodes must be a positive integer, got ${nnodes}"

# The launcher's --gpus-per-node, for the default group (all GPUs of the job).
gpus_per_node=4
args=(${passthrough[@]+"${passthrough[@]}"})
for (( i = 0; i < ${#args[@]}; i++ )); do
    [[ "${args[i]}" == --gpus-per-node ]] && gpus_per_node="${args[i + 1]:-}"
done
[[ "$gpus_per_node" =~ ^[0-9]+$ ]] && (( gpus_per_node >= 1 )) \
    || die "--gpus-per-node must be a positive integer, got ${gpus_per_node}"
[[ -n "$group_sizes" ]] || group_sizes=$(( nnodes * gpus_per_node ))

# The AG rate falls off once the pause outlasts the backpressure wait of a round, which grows with
# the paths per GPU (n x CTA_AG at NCCL's own CTA count); see the usage text for the model.
if [[ -z "$pause_cycles" ]]; then
    [[ "$group_sizes" =~ ^[0-9]+$ ]] \
        || die "no default --pause-cycles for several group sizes (${group_sizes}); pass one group or a list"
    case "$group_sizes" in
        4|8|16) pause_cycles="0,128,256,512,768,1024,1536,2048,2560,3072,4096,6144" ;;
        32) pause_cycles="0,2048,3072,3584,4096,5120,6144,7168,8192,10240,12288,14336" ;;
        64) pause_cycles="0,6144,8192,9728,11264,12288,14336,16384,18432,22528,26624,31744" ;;
        *) die "no default --pause-cycles for a ${group_sizes}-GPU group; pass --pause-cycles" ;;
    esac
    log "pause cycles for the ${group_sizes}-GPU group: ${pause_cycles}"
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

# The rendezvous settings the launcher would use, so the sweep id is agreed on the same host.
dry_run=false
node_rank="${NODE_RANK:-${RANK:-0}}"
master_addr="${MASTER_ADDR:-${DLC_MASTER_ADDR:-127.0.0.1}}"
master_port=29611
args=(${passthrough[@]+"${passthrough[@]}"})
for (( i = 0; i < ${#args[@]}; i++ )); do
    case "${args[i]}" in
        --dry-run) dry_run=true ;;
        --node-rank) node_rank="${args[i + 1]:-}" ;;
        --master-addr) master_addr="${args[i + 1]:-}" ;;
        --master-port) master_port="${args[i + 1]:-}" ;;
        # Only the pause may change between runs, and only on the multicast symmetric kernels.
        --algos|--symm-kernel|--cta-configs) die "${args[i]} is fixed by this sweep (symm, mc, default)" ;;
    esac
done

if [[ "$venv" != none ]]; then
    [[ -f "${venv}/bin/activate" ]] || die "no virtualenv at ${venv}; pass --venv DIR or --venv none"
    # shellcheck source=/dev/null
    source "${venv}/bin/activate"
fi
# Nsight Compute unpacked into the repo, for --ncu passed through.
[[ -d "${repo_root}/NVIDIA-Nsight-Compute-2026.3" ]] && export PATH="${repo_root}/NVIDIA-Nsight-Compute-2026.3:${PATH}"
# NCCL picks every CTA count itself: the process-wide CTA variables override it (and with the cta patch
# NCCL_MIN_CTAS lifts the multicast kernels past their divUp(32, n) cap), so none reaches the runs.
cta_env=(NCCL_MIN_CTAS NCCL_MAX_CTAS NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS NCCL_MIN_NRINGS NCCL_MAX_NRINGS
    NCCL_NVLS_NCHANNELS NCCL_MIN_P2P_NCHANNELS NCCL_MAX_P2P_NCHANNELS NCCL_SYM_CTAS)
for name in "${cta_env[@]}"; do
    [[ -n "${!name:-}" ]] && log "warning: unsetting ${name}=${!name}; the sweep runs at NCCL's own CTA counts"
    unset "$name"
done
if [[ -n "${!pause_env:-}" ]]; then
    log "warning: ${pause_env}=${!pause_env} from the environment is replaced by each --pause-cycles value"
fi

[[ -f "$nccl_lib" ]] || die "${nccl_lib} not found; build it with install/install-nccl-2.32.3-cta.sh --with-pause or pass --nccl-lib"
nccl_lib="$(cd -- "$(dirname -- "$nccl_lib")" && pwd -P)/$(basename -- "$nccl_lib")"
grep -aq "NCCL version 2.32.3${nccl_suffix}+cuda" "$nccl_lib" \
    || die "${nccl_lib} is not the nccl-2.32.3${nccl_suffix} build (no \"NCCL version 2.32.3${nccl_suffix}+cuda\" banner)"
grep -aq "$pause_env" "$nccl_lib" || die "${nccl_lib} does not read ${pause_env}"
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

# One id for the whole sweep, tagged per pause below. The launcher's own handshake port
# (master-port - 1) is free: with RUN_TIME exported the launcher skips its handshake.
# The subshell keeps the handshake's interface and rendezvous settings out of the launcher's environment.
sweep_id="$(
    # shellcheck source=../../../scripts/scale-down/common/gloo_socket_ifname.sh
    source "${repo_root}/scripts/scale-down/common/gloo_socket_ifname.sh"
    # shellcheck source=../../../scripts/scale-down/common/run_timestamp_and_logs.sh
    source "${repo_root}/scripts/scale-down/common/run_timestamp_and_logs.sh"
    export NNODES="$nnodes" NODE_RANK="$node_rank" MASTER_ADDR="$master_addr" MASTER_PORT="$(( master_port - 1 ))"
    if [[ -z "${RUN_TIME:-}" ]] && (( nnodes > 1 )) && ! $dry_run; then
        resolve_gloo_socket_ifname "$nnodes"
    fi
    resolve_run_time "$dry_run"
)"
[[ -n "$sweep_id" ]] || die "could not resolve the sweep id"

# Every NCCL process logs its banner once at NCCL_DEBUG=INFO (init.cc showVersion), and the pause
# when it first builds a symmetric launch (NCCL_DEBUG_SUBSYS includes ENV); the launcher writes
# those logs to <run>/<algo>/nccl-raw/, one file per process.
check_logs() {
    local output_dir="$1" pause="$2" banners file missing=0 files=0
    banners="$(cat "${output_dir}"/*/nccl-raw/*.log 2>/dev/null | grep -o "NCCL version [^ ]*" | sort | uniq -c || true)"
    echo "=== libnccl banners in ${output_dir}/*/nccl-raw" >&2
    echo "${banners:-  none}" >&2
    if [[ -z "$banners" ]] || grep -qv "NCCL version 2.32.3${nccl_suffix}+" <<<"$banners"; then
        echo "ERROR: not every NCCL process ran nccl-2.32.3${nccl_suffix}; the pause=${pause}cyc results are not valid" >&2
        return 1
    fi
    for file in "${output_dir}"/*/nccl-raw/*.log; do
        grep -q "NCCL version 2.32.3${nccl_suffix}+" "$file" || continue
        files=$(( files + 1 ))
        grep -q "${pause_env} set by environment to ${pause}\." "$file" || missing=$(( missing + 1 ))
    done
    if (( missing )); then
        echo "ERROR: ${missing} of ${files} NCCL processes did not log ${pause_env}=${pause}; the results are not valid" >&2
        return 1
    fi
    echo "  [ok]   ${files} NCCL processes ran with ${pause_env}=${pause}" >&2
    # NCCL_DEBUG_SUBSYS includes ENV, so a CTA variable that still reached NCCL is in the logs.
    local cta_re overrides
    cta_re="$(IFS='|'; echo "${cta_env[*]}")"
    overrides="$(cat "${output_dir}"/*/nccl-raw/*.log 2>/dev/null | grep -oE "(${cta_re}) set by environment to [0-9-]+" | sort -u || true)"
    if [[ -n "$overrides" ]]; then
        echo "ERROR: NCCL read a CTA variable, so the pause=${pause}cyc run is not at NCCL's own CTA counts:" >&2
        echo "$overrides" >&2
        return 1
    fi
    echo "=== CTAs per launch (TUNING nchannels) in ${output_dir}/*/nccl-raw" >&2
    cat "${output_dir}"/*/nccl-raw/*.log 2>/dev/null \
        | grep -oE "Kernel (AllGather_STMC|ReduceScatter_LDMC) nchannels [0-9]+" | sort | uniq -c >&2 || true
}

# The launcher files each report as cases/<algo>-<case>/<algo>-<case>.rank_<RR>.nsys-rep (node 0,
# after each pass). Tag the pause before .rank_ so a report copied out of the run still says which
# pause it ran with; analyse_nsys.py reads the rank from the .rank_<RR> tail and the case from the
# directory, so the tagged reports are analysed as before.
tag_nsys_reports() {
    local output_dir="$1" pause="$2" report base tagged count=0
    for report in "${output_dir}"/*/cases/*/*.rank_*.nsys-rep; do
        [[ -f "$report" ]] || continue
        base="${report%.rank_*}"
        [[ "$base" == *".pause${pause}cyc" ]] && continue
        tagged="${base}.pause${pause}cyc.rank_${report##*.rank_}"
        [[ -e "$tagged" ]] && { echo "WARNING: ${tagged} exists; left ${report} untagged" >&2; continue; }
        mv -- "$report" "$tagged"
        count=$(( count + 1 ))
    done
    echo "=== Tagged ${count} nsys reports in ${output_dir}/*/cases with .pause${pause}cyc" >&2
    if compgen -G "${output_dir}/*/nsys/rank*.nsys-rep" >/dev/null; then
        echo "WARNING: unfiled reports left in ${output_dir}/*/nsys; analyse_nsys.sh files them without the pause tag" >&2
    fi
}

status=0
runs=()
for pause in "${pause_list[@]}"; do
    export RUN_TIME="${sweep_id}-pause${pause}cyc"
    echo "=== ${pause_env}=${pause}: run id ${RUN_TIME}" >&2
    log_file="$(mktemp -t symm-cta-pause-sweep.XXXXXX)"
    set +e
    # NCCL_SYM_TMA_ENABLE=0 and --symm-kernel mc keep every config on AllGather_STMC /
    # ReduceScatter_LDMC, as in run_symm_cta_sweep.sh; AllGather_TmaSTMC would be paused too.
    env "${pause_env}=${pause}" NCCL_SYM_TMA_ENABLE=0 LD_PRELOAD="$preload" \
        bash "${script_dir}/run_nvls_collective_benchmark.sh" \
        --nnodes "$nnodes" \
        --group-sizes "$group_sizes" \
        --sizes-mib 1024 \
        --algos symm \
        --workloads ag,rs,ag+rs \
        --nsys \
        --cta-configs default \
        --symm-kernel mc \
        ${passthrough[@]+"${passthrough[@]}"} 2>&1 | tee "$log_file"
    run_status="${PIPESTATUS[0]}"
    set -e
    output_dir="$(sed -n 's/.* output_dir=\([^ ]*\).*/\1/p' "$log_file" | tail -1)"
    rm -f "$log_file"
    if $dry_run; then
        (( run_status == 0 )) || exit "$run_status"
        continue
    fi
    if [[ -z "$output_dir" ]]; then
        echo "WARNING: no output_dir in the launcher output; skipped the pause=${pause}cyc log checks" >&2
    else
        if (( node_rank == 0 )); then
            printf '{\n  "%s": %d,\n  "sweep_id": "%s",\n  "nccl_lib": "%s",\n  "paused_kernels": ["AllGather_STMC", "AllGather_TmaSTMC"],\n  "pause_unit": "SM cycles (clock64) per warp after each 4 KiB chunk of multimem stores; ns = cycles / SM clock"\n}\n' \
                "$pause_env" "$pause" "$sweep_id" "nccl-2.32.3${nccl_suffix}" >"${output_dir}/pause.json"
            tag_nsys_reports "$output_dir" "$pause"
        fi
        check_logs "$output_dir" "$pause" || run_status=1
        runs+=("pause ${pause} cycles -> ${output_dir} (exit ${run_status})")
    fi
    (( run_status == 0 )) || status="$run_status"
done
unset RUN_TIME

if ! $dry_run; then
    echo "=== Sweep ${sweep_id}" >&2
    printf '  %s\n' ${runs[@]+"${runs[@]}"} >&2
fi
exit "$status"
