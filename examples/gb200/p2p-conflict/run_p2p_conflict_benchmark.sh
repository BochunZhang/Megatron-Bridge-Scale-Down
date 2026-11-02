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
Does NVLink traffic into G0 slow down a P2P flow out of G0? NCCL P2P on the four GPUs of one GB200 tray.

Four ranks play G0..G3 (--gpus picks their GPUs). The test flow is G0 -> G3; each scenario adds
background flows into G0, every flow sending --size-mib per trial, all starting together:
  alone  G0->G3
  bg1    G0->G3 + G1->G0
  bg2    G0->G3 + G1->G0 + G2->G0
  bg3    G0->G3 + G1->G0 + G2->G0 + G3->G0
Every flow has its own NCCL communicator, stream and SendRecv kernels. The job fails unless every
flow's sender logs NVLink P2P channels in the requested direction (write for push, /read for pull).

With nsys (the default) ONE nsys session wraps torchrun: a single report holds the CUDA/NVTX
trace of all four processes and the GPU metrics (NVLink TX/RX requests/responses, ...) of all four
GPUs. With --nsys-capture range (default) it covers one cudaProfilerStart/Stop window around all
timed trials; --nsys-capture all records the whole job. analyse_nsys.py then exports it to SQLite
and writes nsys_summary.md (per-flow kernels incl. CTA count, NVLink counters per GPU).

Output: results/gb200/p2p-conflict/<yyyymmdd-hhmmss> in this repo (an exported RUN_TIME pins the
id) with summary.md, results.jsonl, metadata.json, torchrun.log, nccl-raw/,
nsys/p2p-conflict.nsys-rep and the analysis files.

Usage:
  run_p2p_conflict_benchmark.sh [options]

Options:
  --scenarios LIST        alone,bg1,bg2,bg3 (default: all four)
  --gpus G0,G1,G2,G3      nvidia-smi ids playing G0..G3, four distinct ids in 0..3 (default: 0,1,2,3)
  --size-mib N            Bytes each flow sends per trial, in MiB (default: 1024)
  --dtype bf16|fp32       Payload dtype (default: bf16)
  --p2p-mode push|pull    NCCL_P2P_READ_ENABLE=0 (write, the GB200 default) or 1 (read) (default: push)
  --warmup N              Untimed trials per scenario (default: 3)
  --trials N              Timed trials per scenario (default: 10)
  --launch-delay-ms MS    GPU sleep in front of each start gate (default: 20)
  --no-nsys               Skip Nsight Systems (timing only)
  --nsys-capture range|all
                          Profile the timed trials only, or the whole job (default: range)
  --nsys-gpu-metrics-set SET
                          nsys GPU metric set alias (default: nsys picks the first set for the GPU)
  --nsys-freq HZ|high     GPU metric sampling rate (default: 50000, one sample per 20 us); a number
                          is clamped into nsys's range 10..200000, high takes 200000
  --skip-analysis         Do not run analyse_nsys.py after the job
  --master-port PORT      Rendezvous port (default: 29721)
  --output-dir DIR        Run directory (default: see Output)
  --dry-run               Print commands only (skips the node checks)
  -h, --help              Show this help

Examples:
  # All four scenarios, 1 GiB per flow, one nsys report.
  bash examples/gb200/p2p-conflict/run_p2p_conflict_benchmark.sh

  # P2P read, 20 trials, timing only.
  bash examples/gb200/p2p-conflict/run_p2p_conflict_benchmark.sh --p2p-mode pull --trials 20 --no-nsys
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

# Same helpers as the other gb200 launchers; log/warn first so the sourced file keeps them.
log() { printf '%s\n' "$*" >&2; }
warn() { printf 'warning: %s\n' "$*" >&2; }
# shellcheck source=../../../scripts/scale-down/common/run_timestamp_and_logs.sh
source "${repo_root}/scripts/scale-down/common/run_timestamp_and_logs.sh"

# The node this runs on: one GB200 compute tray.
node_gpus=4
scenarios="alone,bg1,bg2,bg3"
gpus="0,1,2,3"
size_mib="1024"
dtype="bf16"
p2p_mode="push"
warmup="3"
trials="10"
launch_delay_ms="20"
nsys_enabled=true
nsys_capture="range"
nsys_gpu_metrics_set=""
nsys_freq="50000"
run_analysis=true
master_port="29721"
output_dir=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scenarios) need_value "$@"; scenarios="$2"; shift 2 ;;
        --gpus) need_value "$@"; gpus="$2"; shift 2 ;;
        --size-mib) need_value "$@"; size_mib="$2"; shift 2 ;;
        --dtype) need_value "$@"; dtype="$2"; shift 2 ;;
        --p2p-mode) need_value "$@"; p2p_mode="$2"; shift 2 ;;
        --warmup) need_value "$@"; warmup="$2"; shift 2 ;;
        --trials) need_value "$@"; trials="$2"; shift 2 ;;
        --launch-delay-ms) need_value "$@"; launch_delay_ms="$2"; shift 2 ;;
        --no-nsys) nsys_enabled=false; shift ;;
        --nsys-capture) need_value "$@"; nsys_capture="$2"; shift 2 ;;
        --nsys-gpu-metrics-set) need_value "$@"; nsys_gpu_metrics_set="$2"; shift 2 ;;
        --nsys-freq) need_value "$@"; nsys_freq="$2"; shift 2 ;;
        --skip-analysis) run_analysis=false; shift ;;
        --master-port) need_value "$@"; master_port="$2"; shift 2 ;;
        --output-dir) need_value "$@"; output_dir="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1 (see --help)" ;;
    esac
done

IFS=, read -r -a scenario_list <<<"$scenarios"
(( ${#scenario_list[@]} > 0 )) || die "--scenarios is empty"
for scenario in "${scenario_list[@]}"; do
    case "$scenario" in
        alone|bg1|bg2|bg3) ;;
        *) die "--scenarios: unknown scenario '${scenario}' (alone, bg1, bg2, bg3)" ;;
    esac
done
IFS=, read -r -a gpu_list <<<"$gpus"
(( ${#gpu_list[@]} == node_gpus )) || die "--gpus takes four ids (G0,G1,G2,G3), got '${gpus}'"
for gpu in "${gpu_list[@]}"; do
    [[ "$gpu" =~ ^[0-9]+$ ]] && (( gpu < node_gpus )) || die "--gpus: '${gpu}' is not a GPU id in 0..$(( node_gpus - 1 ))"
done
[[ "$(printf '%s\n' "${gpu_list[@]}" | sort -u | wc -l)" -eq node_gpus ]] || die "--gpus: the four ids must be distinct"
[[ "$size_mib" =~ ^[1-9][0-9]*$ ]] || die "--size-mib takes a positive integer"
[[ "$dtype" == bf16 || "$dtype" == fp32 ]] || die "--dtype must be bf16 or fp32"
[[ "$p2p_mode" == push || "$p2p_mode" == pull ]] || die "--p2p-mode must be push or pull"
for value in "$warmup" "$trials" "$master_port"; do
    [[ "$value" =~ ^[0-9]+$ ]] || die "--warmup/--trials/--master-port take non-negative integers, got '${value}'"
done
(( trials > 0 )) || die "--trials must be positive"
[[ "$launch_delay_ms" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "--launch-delay-ms takes a number of ms"
[[ "$nsys_capture" == range || "$nsys_capture" == all ]] || die "--nsys-capture must be range or all"
if [[ "$nsys_freq" == high ]]; then
    nsys_freq=200000
elif [[ "$nsys_freq" =~ ^[0-9]+$ ]]; then
    (( nsys_freq < 10 )) && nsys_freq=10
    (( nsys_freq > 200000 )) && nsys_freq=200000
else
    die "--nsys-freq takes a rate in Hz or high, got '${nsys_freq}'"
fi

# One 4-GPU tray only: all four GPUs take part.
if $DRY_RUN; then
    warn "dry run: not checking that this node has ${node_gpus} GPUs"
else
    command -v nvidia-smi >/dev/null || die "nvidia-smi is required to check the GPU count"
    found="$(nvidia-smi -L | grep -c '^GPU ' || true)"
    (( found == node_gpus )) || die "this benchmark runs on a ${node_gpus}-GPU node only; nvidia-smi -L lists ${found}"
    if $nsys_enabled; then
        command -v nsys >/dev/null || die "nsys is required (or pass --no-nsys)"
        nsys --version >&2
    fi
fi
[[ -n "${NCCL_P2P_READ_ENABLE:-}" ]] && warn "NCCL_P2P_READ_ENABLE=${NCCL_P2P_READ_ENABLE} from the environment is overridden by --p2p-mode"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && warn "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} is dropped; --gpus picks the GPUs"

# Gloo/NCCL bootstrap stays on this node; eth0 as in the other gb200 launchers, lo otherwise.
if [[ -n "${GLOO_SOCKET_IFNAME:-}" ]]; then
    gloo_ifname="$GLOO_SOCKET_IFNAME"
elif command -v ip >/dev/null && ip -o -4 addr show up | awk '{print $2}' | grep -qx eth0; then
    gloo_ifname=eth0
else
    gloo_ifname=lo
fi
nccl_socket_ifname="${NCCL_SOCKET_IFNAME:-$gloo_ifname}"

if [[ -z "$output_dir" ]]; then
    resolve_run_time "$DRY_RUN" >/dev/null
    [[ -n "${RUN_TIME:-}" ]] || die "could not resolve the run id"
    output_dir="${repo_root}/results/gb200/p2p-conflict/${RUN_TIME}"
fi
if ! $DRY_RUN; then
    [[ -e "${output_dir}/results.jsonl" ]] && die "${output_dir} already holds results; use a fresh --output-dir"
    mkdir -p "${output_dir}/nccl-raw"
fi

read_enable=0
[[ "$p2p_mode" == pull ]] && read_enable=1
run_env=(
    # Every process sees all GPUs in PCI order, so CUDA, nvidia-smi and nsys number them alike.
    -u CUDA_VISIBLE_DEVICES
    CUDA_DEVICE_ORDER=PCI_BUS_ID
    NCCL_P2P_READ_ENABLE="$read_enable"
    NCCL_DEBUG=INFO
    # INIT carries the "Channel ... via P2P/...[/read]" connect lines the worker checks.
    NCCL_DEBUG_SUBSYS=INIT,P2P,ENV
    NCCL_DEBUG_FILE="${output_dir}/nccl-raw/nccl-%h-%p.log"
    GLOO_SOCKET_IFNAME="$gloo_ifname"
    NCCL_SOCKET_IFNAME="$nccl_socket_ifname"
)

bench_args=(
    --output-dir "$output_dir"
    --scenarios "$scenarios"
    --gpus "$gpus"
    --size-mib "$size_mib"
    --dtype "$dtype"
    --p2p-mode "$p2p_mode"
    --warmup "$warmup"
    --trials "$trials"
    --launch-delay-ms "$launch_delay_ms"
)

profiler=()
if $nsys_enabled; then
    bench_args+=(--nsys)
    nsys_dir="${output_dir}/nsys"
    $DRY_RUN || mkdir -p "${nsys_dir}/.tmp"
    run_env+=(NSYS_TMPDIR="${NSYS_TMPDIR:-${nsys_dir}/.tmp}")
    # One session for the whole process tree: torchrun and its four workers end up in one report,
    # and --gpu-metrics-devices=all samples every GPU from this single session.
    profiler=(
        nsys profile
        --trace=cuda,nvtx
        --sample=none
        --cpuctxsw=none
        --gpu-metrics-devices=all
        --gpu-metrics-frequency="$nsys_freq"
    )
    [[ -n "$nsys_gpu_metrics_set" ]] && profiler+=(--gpu-metrics-set="$nsys_gpu_metrics_set")
    if [[ "$nsys_capture" == range ]]; then
        profiler+=(--capture-range=cudaProfilerApi --capture-range-end=stop)
    fi
    profiler+=(--force-overwrite=false --output="${nsys_dir}/p2p-conflict")
fi

run() {
    if $DRY_RUN; then
        printf '%q ' "$@"
        printf '\n'
    else
        "$@"
    fi
}

echo "=== scenarios ${scenarios}: G0..G3 = GPU ${gpus}, p2p ${p2p_mode}, ${size_mib} MiB per flow" >&2
run env "${run_env[@]}" \
    uv run --no-sync \
    ${profiler[@]+"${profiler[@]}"} \
    python -m torch.distributed.run \
    --nproc_per_node="$node_gpus" \
    --nnodes=1 \
    --master_addr=127.0.0.1 \
    --master_port="$master_port" \
    "${script_dir}/p2p_conflict_benchmark.py" \
    "${bench_args[@]}" \
    2>&1 | { $DRY_RUN && cat || tee "${output_dir}/torchrun.log"; }

if $nsys_enabled && $run_analysis; then
    run uv run --no-sync python "${script_dir}/analyse_nsys.py" --output-dir "$output_dir"
fi
$DRY_RUN || log "summary: ${output_dir}/summary.md"
log "results: ${output_dir}"
