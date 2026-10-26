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
Profile one-way NCCL P2P between two GPUs of one GB200 tray: push (P2P write) vs pull (P2P read).

Runs on a single 4-GPU node only, and starts two ranks on two of its GPUs: rank 0 sends, rank 1
receives. Each workload is its own torchrun job, because NCCL fixes read vs write when the P2P
channels connect:
  push   NCCL_P2P_READ_ENABLE=0: the sender's SMs store into the receiver's staging buffer
         over NVLink; the sender is the NVLink initiator (data on its NVLink TX)
  pull   NCCL_P2P_READ_ENABLE=1: the receiver's SMs load the sender's staging buffer over
         NVLink; the receiver is the NVLink initiator (data on its NVLink RX)
The job fails unless the sender's NCCL INIT log shows the pair connected via P2P, with /read
exactly for pull.

With nsys (the default) every GPU gets one report per size, covering the timed trials, with GPU
metrics (NVLink RX/TX, DRAM, ...) sampled on that GPU. The reports are renamed to say which GPU
did what:
  <workload>/nsys/<workload>.<size>MiB.gpu<G>.<sender|receiver>[.initiator].nsys-rep
e.g. push.1024MiB.gpu0.sender.initiator.nsys-rep (push P2P sender) and
pull.1024MiB.gpu1.receiver.initiator.nsys-rep (pull P2P receiver); nsys-reports.md in the run
directory lists them all. The outermost NVTX range of each report repeats the name and what the
GPU's SMs do.

Output: results/gb200/nvls-collective/p2p-push-pull/<yyyymmdd-hhmmss> in this repo (an exported
RUN_TIME pins the id), one <workload>/ per workload with results.jsonl, metadata.json,
torchrun.log, the NCCL logs under nccl-raw/ and the reports under nsys/.

Usage:
  run_p2p_push_pull_benchmark.sh [options]

Options:
  --workloads LIST        push,pull (default: push,pull)
  --gpus S,R              nvidia-smi ids of the sender and the receiver, two distinct ids in 0..3
                          (default: 0,1)
  --sizes-mib LIST        Message sizes in MiB (default: 1024)
  --dtype bf16|fp32       Payload dtype (default: bf16)
  --warmup N              Untimed sends per size (default: 5)
  --iters N               Back-to-back sends per trial (default: 20)
  --trials N              Timed trials per size, median reported (default: 5)
  --no-nsys               Skip Nsight Systems (timing only)
  --nsys-gpu-metrics-set SET
                          nsys GPU metric set alias (default: nsys picks the first set for the GPU)
  --nsys-freq HZ|high     GPU metric sampling rate (default: 50000, one sample per 20 us); a number
                          is clamped into nsys's range 10..200000, high takes 200000
  --master-port PORT      Rendezvous port; workload i uses PORT+i (default: 29711)
  --output-dir DIR        Run directory (default: see Output)
  --dry-run               Print commands only (skips the 4-GPU check)
  -h, --help              Show this help

Examples:
  # GPU0 -> GPU1, 1 GiB, both workloads with nsys.
  bash examples/gb200/nvls-collective/run_p2p_push_pull_benchmark.sh

  # Only pull, GPU2 -> GPU3, two sizes, three trials.
  bash examples/gb200/nvls-collective/run_p2p_push_pull_benchmark.sh \
    --workloads pull --gpus 2,3 --sizes-mib 256,1024 --trials 3
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

# Same helpers as run_nvls_collective_benchmark.sh; log/warn first so the sourced file keeps them.
log() { printf '%s\n' "$*" >&2; }
warn() { printf 'warning: %s\n' "$*" >&2; }
# shellcheck source=../../../scripts/scale-down/common/run_timestamp_and_logs.sh
source "${repo_root}/scripts/scale-down/common/run_timestamp_and_logs.sh"

# The node this runs on: one GB200 compute tray.
node_gpus=4
workloads="push,pull"
gpus="0,1"
sizes_mib="1024"
dtype="bf16"
warmup="5"
iters="20"
trials="5"
nsys_enabled=true
nsys_gpu_metrics_set=""
nsys_freq="50000"
master_port="29711"
output_dir=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workloads) need_value "$@"; workloads="$2"; shift 2 ;;
        --gpus) need_value "$@"; gpus="$2"; shift 2 ;;
        --sizes-mib) need_value "$@"; sizes_mib="$2"; shift 2 ;;
        --dtype) need_value "$@"; dtype="$2"; shift 2 ;;
        --warmup) need_value "$@"; warmup="$2"; shift 2 ;;
        --iters) need_value "$@"; iters="$2"; shift 2 ;;
        --trials) need_value "$@"; trials="$2"; shift 2 ;;
        --no-nsys) nsys_enabled=false; shift ;;
        --nsys-gpu-metrics-set) need_value "$@"; nsys_gpu_metrics_set="$2"; shift 2 ;;
        --nsys-freq) need_value "$@"; nsys_freq="$2"; shift 2 ;;
        --master-port) need_value "$@"; master_port="$2"; shift 2 ;;
        --output-dir) need_value "$@"; output_dir="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1 (see --help)" ;;
    esac
done

IFS=, read -r -a workload_list <<<"$workloads"
(( ${#workload_list[@]} > 0 )) || die "--workloads is empty"
for workload in "${workload_list[@]}"; do
    [[ "$workload" == push || "$workload" == pull ]] || die "--workloads: unknown workload '${workload}' (push, pull)"
done
IFS=, read -r -a gpu_list <<<"$gpus"
(( ${#gpu_list[@]} == 2 )) || die "--gpus takes exactly two ids (sender,receiver), got '${gpus}'"
for gpu in "${gpu_list[@]}"; do
    [[ "$gpu" =~ ^[0-9]+$ ]] && (( gpu < node_gpus )) || die "--gpus: '${gpu}' is not a GPU id in 0..$(( node_gpus - 1 ))"
done
(( gpu_list[0] != gpu_list[1] )) || die "--gpus: the sender and the receiver must be different GPUs"
[[ "$sizes_mib" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "--sizes-mib must be a comma-separated list of MiB"
[[ "$dtype" == bf16 || "$dtype" == fp32 ]] || die "--dtype must be bf16 or fp32"
for value in "$warmup" "$iters" "$trials" "$master_port"; do
    [[ "$value" =~ ^[0-9]+$ ]] || die "--warmup/--iters/--trials/--master-port take non-negative integers, got '${value}'"
done
(( iters > 0 && trials > 0 )) || die "--iters and --trials must be positive"
if [[ "$nsys_freq" == high ]]; then
    nsys_freq=200000
elif [[ "$nsys_freq" =~ ^[0-9]+$ ]]; then
    (( nsys_freq < 10 )) && nsys_freq=10
    (( nsys_freq > 200000 )) && nsys_freq=200000
else
    die "--nsys-freq takes a rate in Hz or high, got '${nsys_freq}'"
fi

# One 4-GPU tray only: the pair is two GPUs of the same tray, the other two stay idle.
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
if [[ -n "${NCCL_P2P_READ_ENABLE:-}" ]]; then
    warn "NCCL_P2P_READ_ENABLE=${NCCL_P2P_READ_ENABLE} from the environment is overridden per workload"
fi

# Gloo/NCCL bootstrap stays on this node; eth0 as in run_nvls_collective_benchmark.sh, lo otherwise.
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
    output_dir="${repo_root}/results/gb200/nvls-collective/p2p-push-pull/${RUN_TIME}"
fi

run() {
    if $DRY_RUN; then
        printf '%q ' "$@"
        printf '\n'
    else
        "$@"
    fi
}

bench_args=(
    --sizes-mib "$sizes_mib"
    --dtype "$dtype"
    --warmup "$warmup"
    --iters "$iters"
    --trials "$trials"
)
$nsys_enabled && bench_args+=(--nsys)

# Renames rank<R>.<K>.nsys-rep to the name rank 0 wrote for it in nsys/index.tsv.
file_reports() {
    local nsys_dir="$1" tsv="$1/index.tsv" k size rank gpu role initiator report src
    [[ -f "$tsv" ]] || die "${tsv} is missing; the benchmark did not finish"
    while IFS=$'\t' read -r k size rank gpu role initiator report; do
        [[ "$k" == capture_index ]] && continue
        src="${nsys_dir}/rank${rank}.${k}.nsys-rep"
        if [[ -f "$src" ]]; then
            mv -n -- "$src" "${nsys_dir}/${report}.nsys-rep"
        else
            warn "no ${src} for ${report}"
        fi
    done <"$tsv"
}

$DRY_RUN || mkdir -p "$output_dir"
index=0
for workload in "${workload_list[@]}"; do
    workload_dir="${output_dir}/${workload}"
    if ! $DRY_RUN; then
        [[ -e "${workload_dir}/results.jsonl" ]] && die "${workload_dir} already holds results; use a fresh --output-dir"
        mkdir -p "${workload_dir}/nccl-raw"
    fi
    read_enable=0
    [[ "$workload" == pull ]] && read_enable=1
    run_env=(
        CUDA_VISIBLE_DEVICES="$gpus"
        NCCL_P2P_READ_ENABLE="$read_enable"
        NCCL_DEBUG=INFO
        # INIT carries the "Channel ... via P2P/...[/read]" connect lines the worker checks.
        NCCL_DEBUG_SUBSYS=INIT,P2P,ENV
        NCCL_DEBUG_FILE="${workload_dir}/nccl-raw/nccl-%h-%p.log"
        GLOO_SOCKET_IFNAME="$gloo_ifname"
        NCCL_SOCKET_IFNAME="$nccl_socket_ifname"
    )
    worker=(python "${script_dir}/p2p_push_pull_benchmark.py")
    torchrun_extra=()
    if $nsys_enabled; then
        nsys_dir="${workload_dir}/nsys"
        $DRY_RUN || mkdir -p "${nsys_dir}/.tmp"
        # One nsys session per GPU, GPU metrics on its own GPU (nsys_worker.sh); "repeat" writes
        # rank<R>.<K>.nsys-rep for size K, renamed below.
        torchrun_extra=(--no-python)
        worker=(
            bash "${script_dir}/nsys_worker.sh"
            --trace=cuda,nvtx
            --sample=none
            --cpuctxsw=none
            --gpu-metrics-frequency="$nsys_freq"
        )
        [[ -n "$nsys_gpu_metrics_set" ]] && worker+=(--gpu-metrics-set="$nsys_gpu_metrics_set")
        worker+=(
            --capture-range=cudaProfilerApi
            --capture-range-end=repeat
            --force-overwrite=false
            --output="${nsys_dir}/rank%q{RANK}"
            python "${script_dir}/p2p_push_pull_benchmark.py"
        )
        run_env+=(NSYS_TMPDIR="${NSYS_TMPDIR:-${nsys_dir}/.tmp}")
        # nsys numbers GPUs like nvidia-smi (PCI bus order); make CUDA's ordering agree.
        run_env+=(CUDA_DEVICE_ORDER=PCI_BUS_ID)
    fi

    echo "=== workload=${workload}: sender GPU${gpu_list[0]} -> receiver GPU${gpu_list[1]} (${run_env[*]})" >&2
    run env "${run_env[@]}" \
        uv run --no-sync python -m torch.distributed.run \
        --nproc_per_node=2 \
        --nnodes=1 \
        --master_addr=127.0.0.1 \
        --master_port="$(( master_port + index ))" \
        ${torchrun_extra[@]+"${torchrun_extra[@]}"} \
        "${worker[@]}" \
        --workload "$workload" --output-dir "$workload_dir" "${bench_args[@]}" \
        2>&1 | { $DRY_RUN && cat || tee "${workload_dir}/torchrun.log"; }
    if $nsys_enabled && ! $DRY_RUN; then
        file_reports "$nsys_dir"
    fi
    index=$(( index + 1 ))
done

if $nsys_enabled && ! $DRY_RUN; then
    summary="${output_dir}/nsys-reports.md"
    {
        echo "# P2P push / pull Nsight Systems reports"
        echo
        echo "Sender GPU${gpu_list[0]}, receiver GPU${gpu_list[1]}. The initiator is the GPU whose SMs move the data over NVLink."
        echo
        echo "| workload | size (MiB) | GPU | role | initiator | report |"
        echo "| --- | --- | --- | --- | --- | --- |"
        for workload in "${workload_list[@]}"; do
            while IFS=$'\t' read -r k size rank gpu role initiator report; do
                [[ "$k" == capture_index ]] && continue
                echo "| ${workload} | ${size} | GPU${gpu} | ${workload} P2P ${role} | ${initiator} | ${workload}/nsys/${report}.nsys-rep |"
            done <"${output_dir}/${workload}/nsys/index.tsv"
        done
    } >"$summary"
    log "reports: ${summary}"
fi
log "results: ${output_dir}"
