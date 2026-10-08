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
Run a single-node GB200 C2C benchmark with CPU Gloo/ibverbs background traffic.

The wrapper launches four torchrun workers through numarun. CPU P2P
send/recv uses CPU tensors and Gloo's ibverbs transport; only the measured
H2D/D2H copies use the GPUs. The PyTorch build must include Gloo ibverbs
support.

Usage:
  run_gb200_c2c_with_cpu_rdma_gloo_p2p_benchmark.sh [options]

Options:
  --gpus LIST               Physical GPUs exposed to torchrun (default: 0,1,2,3)
  --hca HCA                 Gloo ibverbs device name (default: automatic selection)
  --c2c-buffer-mib MIB      Pinned-host/GPU copy buffer (default: 512)
  --background-buffer-mib MIB
                            CPU background tensor size (default: 256)
  --warmup-iterations N     C2C warmup copies (default: 5)
  --copy-iterations N       Timed C2C copies per direction (default: 20)
  --background-warmup-seconds S
                            CPU background warmup before C2C starts (default: 3)
  --background-ready-timeout-seconds S
                            Timeout for the first CPU background operation (default: 120)
  --background-alone-iterations N
                            Standalone CPU background operations after warmup (default: 20)
  --nsys                    Profile torchrun and its workers with Nsight Systems
  --output-dir DIR          Logs and rank-0 JSON directory
  -h, --help                Show this help

Examples:
  bash examples/gb200/rdma-background/run_gb200_c2c_with_cpu_rdma_gloo_p2p_benchmark.sh \
    --gpus 0,1,2,3 --hca mlx5_bond_0
USAGE
}

die() {
    echo "ERROR: $*" >&2
    exit 2
}

require_value() {
    local option="$1"
    local remaining="$2"
    if [[ "$remaining" -lt 2 ]]; then
        die "$option requires a value"
    fi
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
benchmark_script="${script_dir}/gb200_c2c_with_cpu_rdma_gloo_p2p_benchmark.py"

gpu_list="${GPU_LIST:-0,1,2,3}"
hca=""
c2c_buffer_mib="512"
background_buffer_mib="256"
warmup_iterations="5"
copy_iterations="20"
background_warmup_seconds="3"
background_ready_timeout_seconds="120"
background_alone_iterations="20"
nsys_enabled=false
output_dir="${CPU_C2C_GLOO_P2P_INTERNAL_OUTPUT_DIR:-}"

original_args=("$@")

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            require_value "$1" "$#"
            gpu_list="$2"
            shift 2
            ;;
        --hca)
            require_value "$1" "$#"
            hca="$2"
            shift 2
            ;;
        --c2c-buffer-mib)
            require_value "$1" "$#"
            c2c_buffer_mib="$2"
            shift 2
            ;;
        --background-buffer-mib)
            require_value "$1" "$#"
            background_buffer_mib="$2"
            shift 2
            ;;
        --warmup-iterations)
            require_value "$1" "$#"
            warmup_iterations="$2"
            shift 2
            ;;
        --copy-iterations)
            require_value "$1" "$#"
            copy_iterations="$2"
            shift 2
            ;;
        --background-warmup-seconds)
            require_value "$1" "$#"
            background_warmup_seconds="$2"
            shift 2
            ;;
        --background-ready-timeout-seconds)
            require_value "$1" "$#"
            background_ready_timeout_seconds="$2"
            shift 2
            ;;
        --background-alone-iterations)
            require_value "$1" "$#"
            background_alone_iterations="$2"
            shift 2
            ;;
        --nsys)
            nsys_enabled=true
            shift
            ;;
        --output-dir)
            require_value "$1" "$#"
            output_dir="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

if [[ -z "$output_dir" ]]; then
    output_dir="${repo_root}/results/gb200/rdma-backward/c2c-cpu-rdma-gloo-p2p-$(date +%s)"
fi

if [[ "${CPU_C2C_GLOO_P2P_LOG_CAPTURED:-0}" != 1 ]]; then
    mkdir -p "$output_dir"
    run_log="${output_dir}/run.log"
    [[ ! -e "$run_log" ]] || die "output directory already contains run.log; use a fresh directory: $output_dir"
    [[ ! -e "${output_dir}/torchrun.log" ]] \
        || die "output directory already contains torchrun.log; use a fresh directory: $output_dir"
    export CPU_C2C_GLOO_P2P_LOG_CAPTURED=1
    export CPU_C2C_GLOO_P2P_INTERNAL_OUTPUT_DIR="$output_dir"
    bash "$0" "${original_args[@]}" 2>&1 | tee "$run_log"
    exit "${PIPESTATUS[0]}"
fi

[[ -f "$benchmark_script" ]] || die "benchmark not found: $benchmark_script"
command -v nvidia-smi >/dev/null || die "nvidia-smi is required"
command -v numactl >/dev/null || die "numactl is required by numarun"
command -v torchrun >/dev/null || die "torchrun is required"
command -v python >/dev/null || die "python is required"
if [[ "$nsys_enabled" == true ]]; then
    command -v nsys >/dev/null || die "nsys is required when --nsys is enabled"
    nsys --version
fi

if command -v numarun >/dev/null; then
    numarun_command=(numarun)
elif [[ -r "${repo_root}/.cache/numarun" ]]; then
    numarun_command=(bash "${repo_root}/.cache/numarun")
else
    die "numarun is required (also checked ${repo_root}/.cache/numarun)"
fi

IFS=',' read -r -a gpu_indices <<< "$gpu_list"
[[ ${#gpu_indices[@]} -eq 4 ]] || die "--gpus must contain exactly four GPU indices"

declare -A seen_gpus=()
gpu_pci_bus_ids=()
for gpu in "${gpu_indices[@]}"; do
    [[ "$gpu" =~ ^[0-9]+$ ]] || die "GPU indices must be non-negative integers: $gpu"
    [[ -z "${seen_gpus[$gpu]:-}" ]] || die "duplicate GPU index: $gpu"
    seen_gpus["$gpu"]=1
    gpu_pci_bus_id="$(
        nvidia-smi -i "$gpu" --query-gpu=pci.bus_id --format=csv,noheader \
            | tr '[:upper:]' '[:lower:]' \
            | tr -d '[:space:]' \
            | sed -E 's/^0000([[:xdigit:]]{4}):/\1:/'
    )"
    [[ -n "$gpu_pci_bus_id" ]] || die "could not resolve PCI bus ID for GPU $gpu"
    gpu_pci_bus_ids+=("$gpu_pci_bus_id")
done
gpu_pci_bus_ids_csv="$(IFS=,; echo "${gpu_pci_bus_ids[*]}")"

mkdir -p "$output_dir"
run_log="${output_dir}/run.log"
torchrun_log="${output_dir}/torchrun.log"
if [[ "$nsys_enabled" == true ]]; then
    if [[ -z "${NSYS_TMPDIR:-}" ]]; then
        export NSYS_TMPDIR="${output_dir}/.nsys-tmp"
    fi
    mkdir -p "$NSYS_TMPDIR"
    [[ -w "$NSYS_TMPDIR" ]] || die "Nsight temporary directory is not writable: $NSYS_TMPDIR"
fi
shopt -s nullglob
existing_nsys_reports=(
    "${output_dir}"/nsys*.nsys-rep
    "${output_dir}"/nsys*.qdrep
    "${output_dir}"/nsys*.qdstrm
)
[[ ${#existing_nsys_reports[@]} -eq 0 ]] \
    || die "output directory already contains Nsight reports; use a fresh directory: $output_dir"

export CUDA_VISIBLE_DEVICES="$gpu_list"
export BENCHMARK_GPU_PCI_BUS_IDS="$gpu_pci_bus_ids_csv"
export NUMARUN_MEMBIND="${NUMARUN_MEMBIND:-1}"
export GLOO_DEVICE_TRANSPORT=IBVERBS
if [[ -n "$hca" ]]; then
    export TORCH_GLOO_IBV_NAME="$hca"
fi
unset GLOO_SOCKET_IFNAME
unset MASTER_ADDR MASTER_PORT

ib_device_for_log="${TORCH_GLOO_IBV_NAME:-auto}"
echo "gpus=$gpu_list pci_bus_ids=$gpu_pci_bus_ids_csv background=p2p ib_device=$ib_device_for_log numarun_membind=$NUMARUN_MEMBIND" >&2
echo "background_transport=cpu/gloo-ibverbs control_transport=ibverbs rdzv=127.0.0.1:0" >&2
echo "output_dir=$output_dir" >&2

torchrun_command=(
    torchrun
    --rdzv-backend=c10d
    --rdzv-endpoint=127.0.0.1:0
    --rdzv_conf="timeout=60"
    --local_addr=127.0.0.1
    --nnodes=1
    --nproc-per-node=4
    --no-python
    "${numarun_command[@]}"
    python
    "$benchmark_script"
    --c2c-buffer-mib "$c2c_buffer_mib"
    --background-buffer-mib "$background_buffer_mib"
    --warmup-iterations "$warmup_iterations"
    --copy-iterations "$copy_iterations"
    --background-warmup-seconds "$background_warmup_seconds"
    --background-ready-timeout-seconds "$background_ready_timeout_seconds"
    --background-alone-iterations "$background_alone_iterations"
    --output "${output_dir}/result.json"
)

launch_command=("${torchrun_command[@]}")
if [[ "$nsys_enabled" == true ]]; then
    launch_command=(
        nsys profile
        --trace=cuda,nvtx,osrt
        --sample=none
        --cpuctxsw=none
        --output "${output_dir}/nsys"
        "${torchrun_command[@]}"
    )
fi

printf 'launch_command:' >&2
printf ' %q' "${launch_command[@]}" >&2
printf '\n' >&2
"${launch_command[@]}" 2>&1 | tee "$torchrun_log"

echo "Result: ${output_dir}/result.json" >&2
