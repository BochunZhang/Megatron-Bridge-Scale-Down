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
Sweep the symmetric AG / RS CTA counts once per AllGather_STMC pause, with nccl-2.32.3-cta-with-pause preloaded.

run_symm_cta_sweep.sh with a paused AllGather: for every value of --pause-cycles it runs
run_nvls_collective_benchmark.sh --algos symm --workloads ag,rs,ag+rs --sizes-mib 1024 --nsys
--symm-kernel mc --cta-configs 'default;ag=A,rs=R;...' (1 <= A <= R <= --max-ctas) with
NCCL_SYM_AG_STMC_PAUSE_CYCLES=<cycles>. The preloaded build (install/install-nccl-2.32.3-cta.sh
--with-pause) is nccl-2.32.3-cta plus that variable: AllGather_STMC makes every warp spin <cycles> SM
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
every algorithm records NCCL_SYM_AG_STMC_PAUSE_CYCLES. After each run the script checks that every
NCCL process logged the "NCCL version 2.32.3-cta-pause+cuda..." banner and
"NCCL_SYM_AG_STMC_PAUSE_CYCLES set by environment to <cycles>".

The torch requirements are those of run_symm_cta_sweep.sh: one libnccl.so.2 loaded through its
soname (install/install-torch-2.13.0-nccl-2.32.3.sh), checked before the first run.

Run the same command on every node, like run_nvls_collective_benchmark.sh.

Usage:
  run_symm_cta_pause_sweep.sh [options] [-- run_nvls_collective_benchmark.sh options]

Options:
  --nnodes N          Nodes in the job (default: 8)
  --group-sizes LIST  Single-group sizes (default: 32)
  --max-ctas N        Largest ag / rs CTA count of the sweep (default: 8)
  --pause-cycles LIST SM cycles per 4 KiB chunk and warp, one run each
                      (default: 0,500,1000,2000,4000, about 0-2 us at a 2 GHz SM clock)
  --venv DIR          Virtualenv to activate, or 'none' to keep the current one
                      (default: <repo>/.venv/python3.12-torch2.13-nccl2.32.3)
  --nccl-lib PATH     libnccl.so.2 to preload
                      (default: <repo>/nccl/nccl-2.32.3-cta-with-pause/build/lib/libnccl.so.2)
  -h, --help          Show this help

Anything after the options (or after --) goes to run_nvls_collective_benchmark.sh and, coming
last, overrides the defaults above, e.g. --workloads ag or --dry-run.

Examples:
  # Eight trays, one 32-GPU group (run on every node).
  bash examples/gb200/nvls-collective/run_symm_cta_pause_sweep.sh

  # One tray, 4 GPUs, AG / RS up to 4 CTAs, three pauses.
  bash examples/gb200/nvls-collective/run_symm_cta_pause_sweep.sh --nnodes 1 --group-sizes 4 \
    --max-ctas 4 --pause-cycles 0,1000,4000
USAGE
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
nnodes=8
group_sizes=32
max_ctas=8
pause_cycles="0,500,1000,2000,4000"
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
        --max-ctas) max_ctas="$2"; shift 2 ;;
        --pause-cycles) pause_cycles="$2"; shift 2 ;;
        --venv) venv="$2"; shift 2 ;;
        --nccl-lib) nccl_lib="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; passthrough+=("$@"); break ;;
        *) passthrough+=("$1"); shift ;;
    esac
done

[[ "$max_ctas" =~ ^[0-9]+$ ]] && (( max_ctas >= 1 && max_ctas <= 64 )) || die "--max-ctas must be in 1..64, got ${max_ctas}"
[[ "$nnodes" =~ ^[0-9]+$ ]] && (( nnodes >= 1 )) || die "--nnodes must be a positive integer, got ${nnodes}"
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
    esac
done

cta_configs="default"
for (( ag = 1; ag <= max_ctas; ag++ )); do
    for (( rs = ag; rs <= max_ctas; rs++ )); do
        cta_configs+=";ag=${ag},rs=${rs}"
    done
done

if [[ "$venv" != none ]]; then
    [[ -f "${venv}/bin/activate" ]] || die "no virtualenv at ${venv}; pass --venv DIR or --venv none"
    # shellcheck source=/dev/null
    source "${venv}/bin/activate"
fi
# Nsight Compute unpacked into the repo, for --ncu passed through.
[[ -d "${repo_root}/NVIDIA-Nsight-Compute-2026.3" ]] && export PATH="${repo_root}/NVIDIA-Nsight-Compute-2026.3:${PATH}"
# GB200 images export these; they would override every communicator's minCTAs (the launcher unsets them too).
unset NCCL_MIN_CTAS NCCL_MIN_NCHANNELS
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
        --cta-configs "$cta_configs" \
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
