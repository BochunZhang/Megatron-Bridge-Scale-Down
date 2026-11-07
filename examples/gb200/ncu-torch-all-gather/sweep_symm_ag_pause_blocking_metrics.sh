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
Sweep the AllGather_STMC pause of nccl-2.32.3-pause and collect the --blocking-metrics of the
symmetric multicast all-gather at every pause: one run_ncu_torch_all_gather.sh run per pause.

nccl-2.32.3-pause (install/install-nccl-2.32.3-pause.sh) is stock 2.32.3 plus
NCCL_SYM_AG_STMC_PAUSE_CYCLES: AllGather_STMC makes every warp spin <cycles> SM cycles (clock64)
after each 4 KiB chunk of multimem stores, so the all-gather gets slower at the same CTA count. 0
runs the kernel unchanged, the baseline of the same library. In ns a pause is <cycles> / SM clock;
--blocking-metrics records the SM clock of the profiled passes (--clock-control none), and
summary.txt converts the pause with it. NCCL reads the variable once per process, so every pause
is its own launcher run:

  NCCL_SYM_AG_STMC_PAUSE_CYCLES=<cycles> NCCL_SYM_TMA_ENABLE=0 \
  run_ncu_torch_all_gather.sh --algo symm --blocking-metrics --replay-mode application \
      --communicator shmem --size-mib <size> --nccl-lib <nccl-lib> \
      --output-dir <dir>/pause-<cycles>cyc <launcher args>

application + shmem is the setup that profiled AllGather_STMC section by section (#13);
--blocking-metrics needs several passes, which kernel replay cannot do on multicast memory.
NCCL_SYM_TMA_ENABLE=0 keeps the all-gather off AllGather_TmaSTMC, and the launcher pins
NCCL_SYM_KERNEL=AllGather:AllGather_STMC and fails a run whose TUNING log shows another kernel.

The CTA count stays NCCL's own: the process-wide CTA variables (NCCL_MIN_CTAS, ..., NCCL_SYM_CTAS)
are unset before the first run (GB200 images export NCCL_MIN_CTAS=4), and summary.txt lists the
CTAs (TUNING nchannels) of every pause.

The virtualenv is activated and exported as UV_PROJECT_ENVIRONMENT, so the launcher's
`uv run --no-sync python` runs it (uv ignores a VIRTUAL_ENV that is not the project's). Before the
first run the script checks that this python is the virtualenv's and that its torch maps only
--nccl-lib (install/install-torch-2.13.0-nccl-2.32.3.sh: no nvidia-nccl-cu13 wheel). After each run
it checks that every NCCL process of the run logged the "NCCL version 2.32.3-pause+cuda..." banner
and "NCCL_SYM_AG_STMC_PAUSE_CYCLES set by environment to <cycles>.".

Usage:
  bash examples/gb200/ncu-torch-all-gather/sweep_symm_ag_pause_blocking_metrics.sh [options] [-- launcher args]

Options:
  --pause-cycles LIST  SM cycles per 4 KiB chunk and warp, one run each (default:
                       0,128,256,512,768,1024,1536,2048,2560,3072,4096,6144, the 4-GPU list of
                       ../nvls-collective/run_symm_cta_pause_sweep.sh: the AG rate stays near 711 GB/s
                       up to about 360 cycles and falls to about 245 GB/s at 6144)
  --size-mib N         All-gather output size per rank in MiB (default: 1024)
  --venv DIR           Virtualenv to activate, or 'none' to keep the current one
                       (default: <repo>/.venv/python3.12-torch2.13-nccl2.32.3)
  --nccl-lib PATH      libnccl.so.2 to preload
                       (default: <repo>/nccl/nccl-2.32.3-pause/build/lib/libnccl.so.2)
  --output-dir DIR     (default: results/gb200/ncu-torch-all-gather/symm-all-gather-blocking-metric-pause-<time>)
  --dry-run            Print the launcher commands without running them
  -h, --help

Launcher args go to run_ncu_torch_all_gather.sh after the defaults above, so they can override
them, e.g. -- --gpus 0,1 --replay-mode app-range --communicator tcp --timeout 900, or ncu args after
a second "--". --algo, --ag-rs, --nccl-lib and --output-dir are fixed by the sweep.

Outputs in --output-dir:
  sweep.txt          the sweep's settings
  pause-<C>cyc/      one launcher run: config.txt, baseline.log, ncu.log, nccl-logs/, the report,
                     summary.csv, blocking.txt, plus launcher.log, its console output
  summary.txt        one line per pause: <cycles> <status> <ctas> <sm_ghz> <pause_ns> <ag_us>
                     <algbw_gbps> <detail>; status ok | hang | fail | invalid (the run finished, but
                     not every NCCL process ran nccl-2.32.3-pause with that pause); sm_ghz and ag_us
                     are the means over the results in summary.csv, algbw = size / ag_us
USAGE
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
launcher="${script_dir}/run_ncu_torch_all_gather.sh"

pause_cycles="0,128,256,512,768,1024,1536,2048,2560,3072,4096,6144"
size_mib=1024
venv="${repo_root}/.venv/python3.12-torch2.13-nccl2.32.3"
nccl_lib="${repo_root}/nccl/nccl-2.32.3-pause/build/lib/libnccl.so.2"
nccl_suffix="-pause"
pause_env="NCCL_SYM_AG_STMC_PAUSE_CYCLES"
output_dir=""
dry_run=false
launcher_args=()

die() {
    echo "ERROR: $*" >&2
    exit 1
}
log() { printf '%s\n' "$*" >&2; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pause-cycles) pause_cycles="$2"; shift 2 ;;
        --size-mib) size_mib="$2"; shift 2 ;;
        --venv) venv="$2"; shift 2 ;;
        --nccl-lib) nccl_lib="$2"; shift 2 ;;
        --output-dir) output_dir="$2"; shift 2 ;;
        --dry-run) dry_run=true; shift ;;
        --) shift; launcher_args=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1 (see --help)" ;;
    esac
done

[[ "$size_mib" =~ ^[0-9]+$ ]] && (( size_mib >= 1 )) || die "--size-mib must be a positive integer, got ${size_mib}"
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
# Only the launcher's own options (before its "--") can clash with the sweep.
for arg in ${launcher_args[@]+"${launcher_args[@]}"}; do
    [[ "$arg" == -- ]] && break
    case "$arg" in
        --algo|--ag-rs|--nccl-lib|--output-dir) die "${arg} is fixed by this sweep (symm all-gather, ${nccl_lib##*/}, one folder per pause)" ;;
    esac
done
output_dir="${output_dir:-${repo_root}/results/gb200/ncu-torch-all-gather/symm-all-gather-blocking-metric-pause-$(date +%Y%m%d-%H%M%S)}"

# pause_cmd <cycles>: the launcher command for one pause, in the array "cmd"; its folder in "dir".
pause_cmd() {
    dir="${output_dir}/pause-${1}cyc"
    cmd=(
        bash "$launcher" --algo symm --blocking-metrics --replay-mode application --communicator shmem
        --size-mib "$size_mib" --nccl-lib "$nccl_lib" --output-dir "$dir"
        ${launcher_args[@]+"${launcher_args[@]}"}
    )
}

if $dry_run; then
    for pause in "${pause_list[@]}"; do
        pause_cmd "$pause"
        printf '%s=%s NCCL_SYM_TMA_ENABLE=0 ' "$pause_env" "$pause"
        printf '%q ' "${cmd[@]}"
        echo
    done
    exit 0
fi

################################################################################
# Environment

# NCCL picks every CTA count itself (GB200 images export NCCL_MIN_CTAS=4, which raises the multicast
# kernels to min(4, divUp(32, n)) CTAs); summary.txt shows the counts NCCL used.
cta_env=(NCCL_MIN_CTAS NCCL_MAX_CTAS NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS NCCL_MIN_NRINGS NCCL_MAX_NRINGS
    NCCL_NVLS_NCHANNELS NCCL_MIN_P2P_NCHANNELS NCCL_MAX_P2P_NCHANNELS NCCL_SYM_CTAS)
# The unicast pause would only slow AllGather_ST, but it must not leak into a run either.
for name in "${cta_env[@]}" "$pause_env" NCCL_SYM_AG_ST_PAUSE_CYCLES; do
    [[ -n "${!name:-}" ]] && log "warning: unsetting ${name}=${!name}"
    unset "$name"
done

if [[ "$venv" != none ]]; then
    [[ -f "${venv}/bin/activate" ]] || die "no virtualenv at ${venv}; pass --venv DIR or --venv none"
    venv="$(cd -- "$venv" && pwd -P)"
    # shellcheck source=/dev/null
    source "${venv}/bin/activate"
    export UV_PROJECT_ENVIRONMENT="$venv"
fi

[[ -f "$nccl_lib" ]] || die "${nccl_lib} not found; build it with install/install-nccl-2.32.3-pause.sh or pass --nccl-lib"
nccl_lib="$(cd -- "$(dirname -- "$nccl_lib")" && pwd -P)/$(basename -- "$nccl_lib")"
grep -aq "NCCL version 2.32.3${nccl_suffix}+cuda" "$nccl_lib" \
    || die "${nccl_lib} is not the nccl-2.32.3${nccl_suffix} build (no \"NCCL version 2.32.3${nccl_suffix}+cuda\" banner)"
grep -aq "$pause_env" "$nccl_lib" || die "${nccl_lib} does not read ${pause_env}; rebuild it with install/install-nccl-2.32.3-pause.sh"

cd "$repo_root"
echo "=== Checking the python of uv run and the libnccl its torch maps" >&2
LD_PRELOAD="${nccl_lib}${LD_PRELOAD:+:${LD_PRELOAD}}" uv run --no-sync python - "$nccl_lib" "$venv" <<'PY'
import os
import sys

import torch  # noqa: F401  (libtorch_cuda.so needs libnccl.so.2; the official wheel also preloads its own)


want, venv = os.path.realpath(sys.argv[1]), sys.argv[2]
if venv != "none" and os.path.realpath(sys.prefix) != os.path.realpath(venv):
    sys.exit(f"ERROR: uv run --no-sync python runs {sys.prefix}, not {venv}")
with open("/proc/self/maps") as maps:
    paths = {line.split(None, 5)[5].strip() for line in maps if len(line.split(None, 5)) == 6}
loaded = sorted({os.path.realpath(p) for p in paths if os.path.basename(p).startswith("libnccl.so")})
if loaded != [want]:
    sys.exit(f"ERROR: torch {torch.__version__} maps libnccl {loaded}, expected only {want}; LD_PRELOAD needs a torch "
             "without nvidia-nccl-cu13 (install/install-torch-2.13.0-nccl-2.32.3.sh)")
print(f"  [ok]   {sys.prefix}: torch {torch.__version__} maps only {want}", file=sys.stderr)
PY

mkdir -p "$output_dir"
[[ -z "$(ls -A "$output_dir")" ]] || die "${output_dir} is not empty"
summary="${output_dir}/summary.txt"
{
    echo "pause_env=${pause_env} pause_cycles=${pause_list[*]}"
    echo "size_mib=${size_mib} venv=${venv} nccl_lib=${nccl_lib}"
    echo "launcher_args=${launcher_args[*]:-}"
} >"${output_dir}/sweep.txt"
cat "${output_dir}/sweep.txt"

################################################################################
# Per-pause checks

# A hung NCCL kernel can outlive the launcher; the next pause must start on idle GPUs.
kill_leftovers() {
    local pattern="ncu_torch_all_gather\\.py|${1}/"
    pgrep -f "$pattern" >/dev/null || return 0
    echo "killing leftover workers / ncu of ${1}"
    pkill -KILL -f "$pattern" || true
    for _ in $(seq 60); do
        pgrep -f "$pattern" >/dev/null || return 0
        sleep 1
    done
    die "processes matching ${pattern} still alive 60s after SIGKILL; check the GPUs before going on"
}

# check_pause <dir> <cycles>: every NCCL process of the run (one log each, NCCL_DEBUG_SUBSYS includes
# INIT and ENV) printed the pause build's banner and read the pause.
check_pause() {
    local logs=("${1}"/nccl-logs/*/*.log) file files=0 missing=0
    [[ -f "${logs[0]}" ]] || { echo "no NCCL logs in ${1}/nccl-logs"; return 1; }
    for file in "${logs[@]}"; do
        grep -q "NCCL version " "$file" || continue
        files=$(( files + 1 ))
        if ! grep -q "NCCL version 2.32.3${nccl_suffix}+" "$file"; then
            echo "${file} ran $(grep -o 'NCCL version [^ ]*' "$file" | head -n 1), not nccl-2.32.3${nccl_suffix}"
            return 1
        fi
        grep -q "${pause_env} set by environment to ${2}\." "$file" || missing=$(( missing + 1 ))
    done
    (( files > 0 )) || { echo "no NCCL banner in ${1}/nccl-logs"; return 1; }
    (( missing == 0 )) || { echo "${missing} of ${files} NCCL processes did not log ${pause_env}=${2}"; return 1; }
}

# ctas <dir>: the CTA counts of the AllGather_STMC launches of the ncu pass, e.g. "4"; "-" without any.
ctas() {
    local logs=("${1}/nccl-logs/ncu/"*.log) counts
    [[ -f "${logs[0]}" ]] || { echo -; return 0; }
    counts="$(cat "${logs[@]}" | sed -nE 's/.*Kernel AllGather_STMC nchannels ([0-9]+).*/\1/p' | sort -un | paste -sd/ -)"
    echo "${counts:--}"
}

# result_stats <summary.csv> <cycles>: "<sm_ghz> <pause_ns> <ag_us> <algbw_gbps>", the means over the
# results (ncu --print-units base: Hz and ns); "-" for what the report does not have.
result_stats() {
    python3 -I - "$1" "$2" "$size_mib" <<'PY'
import csv
import sys


path, pause, size_mib = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
try:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))[1:]  # the first row after the header holds the units
except OSError:
    rows = []


def mean(name):
    values = []
    for row in rows:
        try:
            values.append(float((row.get(name) or "").replace(",", "")))
        except ValueError:
            pass
    return sum(values) / len(values) if values else None


hz, ns = mean("sm__cycles_elapsed.avg.per_second"), mean("gpu__time_duration.sum")
out = [
    f"{hz / 1e9:.3f}" if hz else "-",
    f"{pause / hz * 1e9:.1f}" if hz else "-",
    f"{ns / 1e3:.1f}" if ns else "-",
    f"{size_mib * 2**20 / ns:.1f}" if ns else "-",
]
print(" ".join(out))
PY
}

################################################################################
# Sweep

printf '%-8s %-8s %-6s %-7s %-9s %-9s %-10s %s\n' \
    cycles status ctas sm_ghz pause_ns ag_us algbw_gbps detail | tee "$summary"
status=0
for pause in "${pause_list[@]}"; do
    pause_cmd "$pause"
    log_file="${output_dir}/pause-${pause}cyc.log"
    echo "=== ${pause_env}=${pause}: ${cmd[*]}"
    rc=0
    env "${pause_env}=${pause}" NCCL_SYM_TMA_ENABLE=0 "${cmd[@]}" 2>&1 | tee "$log_file" || rc=${PIPESTATUS[0]}
    if (( rc == 0 )); then
        result=ok
    elif grep -qE "did not finish in .* \(hang\)|ncu gave up waiting" "$log_file"; then
        result=hang
    else
        result=fail
    fi
    # The launcher's die message, else the first ncu error.
    detail="$(grep -m1 "^ERROR: " "$log_file" || grep -m1 "==ERROR==" "$log_file" || true)"
    # The launcher needs an empty --output-dir, so its console log is moved in afterwards.
    if [[ -d "$dir" ]]; then
        mv "$log_file" "${dir}/launcher.log"
        log_file="${dir}/launcher.log"
    fi
    if [[ "$result" == ok ]]; then
        if bad="$(check_pause "$dir" "$pause")"; then
            detail="${dir}/blocking.txt"
        else
            result=invalid
            detail="${bad}; log ${log_file}"
        fi
    else
        detail="${detail:-exit ${rc}}; log ${log_file}"
        kill_leftovers "$dir"
    fi
    [[ "$result" == ok ]] || status=1
    printf '%-8s %-8s %-6s %-7s %-9s %-9s %-10s %s\n' "$pause" "$result" "$(ctas "$dir")" \
        $(result_stats "${dir}/summary.csv" "$pause" || echo - - - -) "$detail" | tee -a "$summary"
    # A wrong algorithm or library fails every pause the same way.
    if [[ "$detail" == *"NCCL did not run --algo"* || "$detail" == *"not nccl-2.32.3${nccl_suffix}"* ]]; then
        die "pause ${pause}: ${detail}; fix the NCCL environment before the other pauses"
    fi
done

echo "=== summary: ${summary}"
cat "$summary"
exit "$status"
