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
GEMM time on GPUs 0-3 of one GB200 tray, alone and while a unicast symmetric AllGather between the
four GPUs keeps NVLink / HBM busy, with nccl-2.32.3-pause preloaded.

Cases (one torchrun job each, every worker under nsys with NVLink / DRAM GPU metrics):
  baseline        no AllGather; --gemm-rounds back-to-back GEMMs [M, K] x [K, N]
  pause<C>cyc     for every C of --pause-cycles: --ag-rounds AllGathers of --size-mib on a
                  high-priority stream; rounds [0, warmup) are warmup, the next --ag-isolated run
                  alone, and an event after the last isolated round releases the GEMM stream,
                  which runs the same GEMMs during the remaining (background) rounds

The AllGather is AllGather_ST, the unicast symmetric kernel: the buffers are a registered
NCCL_WIN_COLL_SYMMETRIC window, NCCL_SYM_KERNEL=allgather:AllGather_ST pins it, NCCL_NVLS_ENABLE=0
keeps multicast off, NCCL_SYM_TMA_ENABLE=0 keeps it off AllGather_TmaST and NCCL_SYM_CE_THRESHOLD
at INT64_MAX off the copy engines. Its traffic is set by NCCL_SYM_AG_ST_PAUSE_CYCLES (one value per
process, so one job per pause): every AllGather warp spins C SM cycles (clock64) after each tile it
has stored to all peers (2 KiB per peer), so a larger C means less NVLink / HBM traffic on the same
CTAs. 0 runs the stock kernel. In ns a pause is C / SM clock (GPC Clock Frequency in the nsys
metrics, or pin it with nvidia-smi -lgc). The CTA count is NCCL's own unless --ctas is given: the
process-wide CTA variables (GB200 images export NCCL_MIN_CTAS=4) are unset first.

After each job the script checks that every NCCL process loaded nccl-2.32.3-pause, that every
sweep process logged NCCL_SYM_AG_ST_PAUSE_CYCLES=<C> and that every AllGather launch was
AllGather_ST. At the end analyse_hbm_conflict.py exports the reports (nsys export --type sqlite),
queries them with hbm_conflict.sql and writes report.md next to the cases; rerun it with
  uv run --no-sync python examples/gb200/hbm-conflict/analyse_hbm_conflict.py --output-dir <run dir>

Output: results/gb200/hbm-conflict/gpu-num-<gpus>/<yyyymmdd-hhmmss> (export RUN_TIME=<id> to pick
the id):
  metadata.json  report.md  summary.json  ag_rounds.csv  gemms.csv  gpu_metrics.csv
  <case>/case.json  torchrun.log  events.rank_<RR>.json  nccl-raw/  nsys/<case>.rank_<R>.nsys-rep

Usage:
  run_hbm_conflict_benchmark.sh [options]

Options:
  --pause-cycles LIST   SM cycles per tile and warp, one sweep case each
                        (default: 0,128,256,512,1024,2048,4096,8192,16384,32768)
  --size-mib N          AllGather output (gathered tensor) in MiB (default: 1024)
  --gemm-mkn M,K,N      GEMM [M, K] x [K, N] (default: 4096,7168,2048)
  --dtype bf16|fp16     GEMM and AllGather dtype (default: bf16)
  --gemm-rounds N       Measured GEMMs per case (default: 10)
  --gemm-warmup N       GEMMs before the capture window, for cuBLAS (default: 5)
  --gemm-delay-us US    GPU sleep on the GEMM stream after its release (default: 50)
  --ag-rounds N         AllGathers per sweep case (default: 25)
  --ag-warmup N         Of which warmup (default: 5)
  --ag-isolated N       Of which alone after the warmup (default: 5)
  --ctas N              Pin the AllGather CTA count (NCCL_MIN_CTAS = NCCL_MAX_CTAS = N; default: NCCL's)
  --gpus N              GPUs (default: 4, GPUs 0-3)
  --skip-baseline       Do not run the baseline case
  --no-analyse          Do not run analyse_hbm_conflict.py at the end
  --nsys-freq HZ        GPU metric sampling rate (default: 50000, one sample per 20 us; 10..200000)
  --nsys-gpu-metrics-set SET
                        nsys GPU metric set (default: nsys's first set for the GPU; the choices are
                        saved to nsys-gpu-metrics.txt in the run directory)
  --master-port PORT    torchrun port (default: 29631)
  --venv DIR            Virtualenv to activate, or 'none' (default:
                        <repo>/.venv/python3.12-torch2.13-nccl2.32.3)
  --nccl-lib PATH       libnccl.so.2 to preload
                        (default: <repo>/nccl/nccl-2.32.3-pause/build/lib/libnccl.so.2)
  --dry-run             Print the commands only
  -h, --help            Show this help

Example:
  bash examples/gb200/hbm-conflict/run_hbm_conflict_benchmark.sh --pause-cycles 0,1024,8192
USAGE
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
nsys_worker="${repo_root}/examples/gb200/nvls-collective/nsys_worker.sh"
pause_cycles="0,128,256,512,1024,2048,4096,8192,16384,32768"
size_mib=1024
gemm_mkn="4096,7168,2048"
dtype=bf16
gemm_rounds=10
gemm_warmup=5
gemm_delay_us=50
ag_rounds=25
ag_warmup=5
ag_isolated=5
ctas=""
gpus=4
skip_baseline=false
analyse=true
nsys_freq=50000
nsys_gpu_metrics_set=""
master_port=29631
venv="${repo_root}/.venv/python3.12-torch2.13-nccl2.32.3"
nccl_lib="${repo_root}/nccl/nccl-2.32.3-pause/build/lib/libnccl.so.2"
dry_run=false
pause_env="NCCL_SYM_AG_ST_PAUSE_CYCLES"
nccl_suffix="-pause"
int64_max=9223372036854775807

die() {
    echo "ERROR: $*" >&2
    exit 1
}
log() { printf '%s\n' "$*" >&2; }
warn() { log "WARNING: $*"; }
run() {
    if $dry_run; then
        printf '+'; printf ' %q' "$@"; printf '\n'
    else
        "$@"
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pause-cycles) pause_cycles="$2"; shift 2 ;;
        --size-mib) size_mib="$2"; shift 2 ;;
        --gemm-mkn) gemm_mkn="$2"; shift 2 ;;
        --dtype) dtype="$2"; shift 2 ;;
        --gemm-rounds) gemm_rounds="$2"; shift 2 ;;
        --gemm-warmup) gemm_warmup="$2"; shift 2 ;;
        --gemm-delay-us) gemm_delay_us="$2"; shift 2 ;;
        --ag-rounds) ag_rounds="$2"; shift 2 ;;
        --ag-warmup) ag_warmup="$2"; shift 2 ;;
        --ag-isolated) ag_isolated="$2"; shift 2 ;;
        --ctas) ctas="$2"; shift 2 ;;
        --gpus) gpus="$2"; shift 2 ;;
        --skip-baseline) skip_baseline=true; shift ;;
        --no-analyse) analyse=false; shift ;;
        --nsys-freq) nsys_freq="$2"; shift 2 ;;
        --nsys-gpu-metrics-set) nsys_gpu_metrics_set="$2"; shift 2 ;;
        --master-port) master_port="$2"; shift 2 ;;
        --venv) venv="$2"; shift 2 ;;
        --nccl-lib) nccl_lib="$2"; shift 2 ;;
        --dry-run) dry_run=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1 (see --help)" ;;
    esac
done

for name in size_mib gemm_rounds ag_rounds gpus master_port nsys_freq; do
    [[ "${!name}" =~ ^[0-9]+$ ]] && (( ${!name} > 0 )) || die "--${name//_/-} must be a positive integer, got ${!name}"
done
for name in gemm_warmup ag_warmup ag_isolated; do
    [[ "${!name}" =~ ^[0-9]+$ ]] || die "--${name//_/-} must be a non-negative integer, got ${!name}"
done
(( ag_warmup + ag_isolated < ag_rounds )) || die "--ag-rounds must leave background rounds after warmup + isolated"
(( nsys_freq >= 10 && nsys_freq <= 200000 )) || die "--nsys-freq must be in 10..200000 Hz, got ${nsys_freq}"
[[ "$gemm_delay_us" =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "--gemm-delay-us must be a non-negative number, got ${gemm_delay_us}"
[[ "$dtype" == bf16 || "$dtype" == fp16 ]] || die "--dtype must be bf16 or fp16, got ${dtype}"
[[ "$gemm_mkn" =~ ^[0-9]+,[0-9]+,[0-9]+$ ]] || die "--gemm-mkn must be M,K,N, got ${gemm_mkn}"
IFS=, read -r gemm_m gemm_k gemm_n <<<"$gemm_mkn"
[[ -z "$ctas" || "$ctas" =~ ^[1-9][0-9]*$ ]] || die "--ctas must be a positive integer, got ${ctas}"
[[ "$pause_cycles" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "--pause-cycles must be a comma-separated list, got '${pause_cycles}'"
IFS=, read -r -a pause_list <<<"$pause_cycles"
for (( i = 0; i < ${#pause_list[@]}; i++ )); do
    pause_list[i]=$(( 10#${pause_list[i]} ))
    (( pause_list[i] <= 4294967295 )) || die "--pause-cycles: ${pause_list[i]} does not fit the kernel's uint32 cycles"
    for (( j = 0; j < i; j++ )); do
        (( pause_list[i] != pause_list[j] )) || die "--pause-cycles lists ${pause_list[i]} twice"
    done
done
[[ -f "$nsys_worker" ]] || die "${nsys_worker} not found"
visible_devices=0
for (( i = 1; i < gpus; i++ )); do visible_devices+=",${i}"; done

# NCCL picks the CTA count unless --ctas pins it: no process-wide CTA variable may reach it.
cta_env=(NCCL_MIN_CTAS NCCL_MAX_CTAS NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS NCCL_MIN_NRINGS NCCL_MAX_NRINGS
    NCCL_NVLS_NCHANNELS NCCL_MIN_P2P_NCHANNELS NCCL_MAX_P2P_NCHANNELS NCCL_SYM_CTAS)
for name in "${cta_env[@]}"; do
    [[ -n "${!name:-}" ]] && warn "unsetting ${name}=${!name}"
    unset "$name"
done
for name in "$pause_env" NCCL_SYM_AG_STMC_PAUSE_CYCLES NCCL_SYM_KERNEL NCCL_ALGO NCCL_PROTO; do
    [[ -n "${!name:-}" ]] && warn "unsetting ${name}=${!name}; the launcher sets what each case needs"
    unset "$name"
done

if [[ "$venv" != none ]]; then
    [[ -f "${venv}/bin/activate" ]] || die "no virtualenv at ${venv}; pass --venv DIR or --venv none"
    # shellcheck source=/dev/null
    source "${venv}/bin/activate"
fi
if ! $dry_run; then
    [[ -f "$nccl_lib" ]] || die "${nccl_lib} not found; build it with install/install-nccl-2.32.3-pause.sh"
    nccl_lib="$(cd -- "$(dirname -- "$nccl_lib")" && pwd -P)/$(basename -- "$nccl_lib")"
    grep -aq "NCCL version 2.32.3${nccl_suffix}+cuda" "$nccl_lib" \
        || die "${nccl_lib} is not the nccl-2.32.3${nccl_suffix} build"
    grep -aq "$pause_env" "$nccl_lib" \
        || die "${nccl_lib} does not read ${pause_env}; rebuild it with install/install-nccl-2.32.3-pause.sh"
    command -v nsys >/dev/null || die "nsys is required"
fi
preload="${nccl_lib}${LD_PRELOAD:+:${LD_PRELOAD}}"

cd "$repo_root"
if ! $dry_run; then
    nsys --version >&2
    echo "=== Checking that torch maps only ${nccl_lib}" >&2
    LD_PRELOAD="$preload" uv run --no-sync python - "$nccl_lib" <<'PY'
import os
import sys

import torch  # noqa: F401  (libtorch_cuda.so needs libnccl.so.2)


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

run_time="${RUN_TIME:-$(date +%Y%m%d-%H%M%S)}"
output_dir="${repo_root}/results/gb200/hbm-conflict/gpu-num-${gpus}/${run_time}"
if ! $dry_run; then
    [[ -e "$output_dir" ]] && die "${output_dir} exists; export another RUN_TIME"
    mkdir -p "$output_dir"
    {
        nsys profile --gpu-metrics-devices=help 2>&1
        nsys profile --gpu-metrics-devices=all --gpu-metrics-set=help 2>&1
    } >"${output_dir}/nsys-gpu-metrics.txt" || true
    pause_json="$(printf '%s,' "${pause_list[@]}")"
    cat >"${output_dir}/metadata.json" <<JSON
{
  "run_time": "${run_time}",
  "gpus": ${gpus},
  "nccl_lib": "nccl-2.32.3${nccl_suffix}",
  "ag_kernel": "AllGather_ST",
  "pause_env": "${pause_env}",
  "pause_unit": "SM cycles (clock64) per warp after each tile stored to every peer (2 KiB per peer); ns = cycles / SM clock",
  "pause_cycles": [${pause_json%,}],
  "size_mib": ${size_mib},
  "dtype": "${dtype}",
  "gemm_mkn": [${gemm_m}, ${gemm_k}, ${gemm_n}],
  "gemm_rounds": ${gemm_rounds},
  "gemm_warmup": ${gemm_warmup},
  "gemm_delay_us": ${gemm_delay_us},
  "ag_rounds": ${ag_rounds},
  "ag_warmup": ${ag_warmup},
  "ag_isolated": ${ag_isolated},
  "ctas": ${ctas:-null},
  "nsys_freq": ${nsys_freq},
  "nsys_gpu_metrics_set": "${nsys_gpu_metrics_set}"
}
JSON
fi
log "output_dir=${output_dir}"

# Every NCCL process logs its banner (NCCL_DEBUG=INFO), the pause when it first builds a symmetric
# launch (NCCL_DEBUG_SUBSYS has ENV) and, on comm rank 0, one TUNING line per launch.
check_logs() {
    local case_dir="$1" mode="$2" pause="$3" logs file files=0 missing=0 kernels overrides cta_re
    logs=("${case_dir}"/nccl-raw/*.log)
    [[ -f "${logs[0]}" ]] || { log "ERROR: no NCCL logs in ${case_dir}/nccl-raw"; return 1; }
    for file in "${logs[@]}"; do
        files=$(( files + 1 ))
        grep -q "NCCL version 2.32.3${nccl_suffix}+" "$file" || { log "ERROR: ${file} did not run nccl-2.32.3${nccl_suffix}"; return 1; }
        if [[ "$mode" == sweep ]]; then
            grep -q "${pause_env} set by environment to ${pause}\." "$file" || missing=$(( missing + 1 ))
        fi
    done
    if (( missing )); then
        log "ERROR: ${missing} of ${files} NCCL processes did not log ${pause_env}=${pause}"
        return 1
    fi
    log "  [ok]   ${files} NCCL processes ran nccl-2.32.3${nccl_suffix}${pause:+ with ${pause_env}=${pause}}"
    if [[ -z "$ctas" ]]; then
        cta_re="$(IFS='|'; echo "${cta_env[*]}")"
        overrides="$(cat "${logs[@]}" | grep -oE "(${cta_re}) set by environment to [0-9-]+" | sort -u || true)"
        if [[ -n "$overrides" ]]; then
            log "ERROR: NCCL read a CTA variable, so ${case_dir} is not at NCCL's own CTA count:"
            log "$overrides"
            return 1
        fi
    fi
    if [[ "$mode" == sweep ]]; then
        kernels="$(cat "${logs[@]}" | grep -oE "Kernel AllGather_[A-Za-z]+ nchannels [0-9]+" | sort | uniq -c || true)"
        log "=== AllGather launches (TUNING) in ${case_dir}/nccl-raw"
        log "${kernels:-  none}"
        if [[ -z "$kernels" ]] || grep -qv "Kernel AllGather_ST nchannels" <<<"$kernels"; then
            log "ERROR: not every AllGather launch in ${case_dir} was AllGather_ST"
            return 1
        fi
    fi
}

run_case() {
    local case="$1" mode="$2" pause="${3:-}" case_dir status
    case_dir="${output_dir}/${case}"
    local case_env=(
        LD_PRELOAD="$preload"
        NCCL_NVLS_ENABLE=0
        NCCL_WIN_ENABLE=1
        NCCL_SYM_TMA_ENABLE=0
        NCCL_SYM_CE_THRESHOLD="$int64_max"
        NCCL_SYM_KERNEL="allgather:AllGather_ST"
        NCCL_DEBUG=INFO
        NCCL_DEBUG_SUBSYS=INIT,ENV,TUNING
        NCCL_DEBUG_FILE="${case_dir}/nccl-raw/nccl-%h-%p.log"
        CUDA_VISIBLE_DEVICES="$visible_devices"
        # nsys numbers GPUs like nvidia-smi (PCI bus order); make the CUDA index agree.
        CUDA_DEVICE_ORDER=PCI_BUS_ID
        NSYS_TMPDIR="${NSYS_TMPDIR:-${case_dir}/nsys/.tmp}"
    )
    [[ "$mode" == sweep ]] && case_env+=("${pause_env}=${pause}")
    [[ -n "$ctas" ]] && case_env+=(NCCL_MIN_CTAS="$ctas" NCCL_MAX_CTAS="$ctas")
    local nsys_args=(
        --trace=cuda,nvtx
        --sample=none
        --cpuctxsw=none
        --gpu-metrics-frequency="$nsys_freq"
    )
    [[ -n "$nsys_gpu_metrics_set" ]] && nsys_args+=(--gpu-metrics-set="$nsys_gpu_metrics_set")
    nsys_args+=(
        --capture-range=cudaProfilerApi
        --capture-range-end=stop
        --force-overwrite=false
        --output="${case_dir}/nsys/${case}.rank_%q{RANK}"
    )
    local worker_args=(
        --mode "$mode" --case "$case" --output-dir "$case_dir"
        --size-mib "$size_mib" --dtype "$dtype"
        --gemm-m "$gemm_m" --gemm-k "$gemm_k" --gemm-n "$gemm_n"
        --gemm-rounds "$gemm_rounds" --gemm-warmup "$gemm_warmup" --gemm-delay-us "$gemm_delay_us"
        --ag-rounds "$ag_rounds" --ag-warmup "$ag_warmup" --ag-isolated "$ag_isolated"
    )
    if ! $dry_run; then
        mkdir -p "${case_dir}/nccl-raw" "${case_dir}/nsys" "${NSYS_TMPDIR:-${case_dir}/nsys/.tmp}"
        printf '{\n  "case": "%s",\n  "mode": "%s",\n  "%s": %s,\n  "env": [%s]\n}\n' \
            "$case" "$mode" "$pause_env" "${pause:-null}" \
            "$(printf '"%s",' "${case_env[@]}" | sed 's/,$//')" >"${case_dir}/case.json"
    fi
    echo "=== case=${case} (${case_env[*]})" >&2
    set +e
    run env "${case_env[@]}" \
        uv run --no-sync python -m torch.distributed.run \
        --nproc_per_node="$gpus" --nnodes=1 --master_addr=127.0.0.1 --master_port="$master_port" \
        --no-python bash "$nsys_worker" "${nsys_args[@]}" \
        python "${script_dir}/hbm_conflict_benchmark.py" "${worker_args[@]}" \
        2>&1 | { $dry_run && cat || tee "${case_dir}/torchrun.log"; }
    status="${PIPESTATUS[0]}"
    set -e
    $dry_run && return 0
    (( status == 0 )) || { log "ERROR: ${case} exited with ${status}; see ${case_dir}/torchrun.log"; return "$status"; }
    compgen -G "${case_dir}/nsys/*.rank_*.nsys-rep" >/dev/null || { log "ERROR: nsys wrote no report for ${case}"; return 1; }
    check_logs "$case_dir" "$mode" "$pause"
}

status=0
if ! $skip_baseline; then
    run_case baseline baseline || status=1
fi
for pause in "${pause_list[@]}"; do
    run_case "pause${pause}cyc" sweep "$pause" || status=1
done

if $analyse; then
    run uv run --no-sync python "${script_dir}/analyse_hbm_conflict.py" --output-dir "$output_dir" \
        || warn "analysis failed; the reports are in ${output_dir}. Retry with:" \
            "uv run --no-sync python ${script_dir}/analyse_hbm_conflict.py --output-dir ${output_dir}"
fi
$dry_run || log "=== ${output_dir}"
exit "$status"
