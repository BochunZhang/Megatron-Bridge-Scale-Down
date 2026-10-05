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
Run a single-node GB200 NCCL P2P send/recv benchmark with C2C contention.

The wrapper launches four torchrun workers. Each worker is started through
numarun so LOCAL_RANK selects its CPU and NUMA binding. P2P/NVLink and SHM
NCCL transports are disabled; P2P send/recv must use the configured IB HCA or
NCCL's automatic HCA selection when --hca is omitted.

Usage:
  run_gb200_c2c_p2p_benchmark.sh [options]

Options:
  --gpus LIST               Physical GPUs exposed to torchrun (default: 0,1,2,3)
  --hca HCA                 NCCL HCA name/prefix or comma-separated list (default: NCCL selects)
  --c2c-buffer-mib MIB      Pinned-host/GPU copy buffer (default: 512)
  --p2p-buffer-mib MIB      P2P send/recv GPU buffer (default: 256)
  --warmup-iterations N     C2C warmup copies per direction (default: 5)
  --copy-iterations N       Baseline C2C copies per direction (default: 20)
  --p2p-iterations N        Timed P2P operations per phase (default: 20)
  --nsys                    Profile torchrun and its workers with Nsight Systems
  --output-dir DIR          Logs and rank-0 JSON directory (default: results/gb200/rdma-backward/c2c-p2p-rdma-<timestamp>)
  -h, --help                Show this help

Example:
  bash examples/gb200/rdma-backgrad/run_gb200_c2c_p2p_benchmark.sh \
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
benchmark_script="${script_dir}/gb200_c2c_p2p_benchmark.py"
default_output_dir="${repo_root}/results/gb200/rdma-backward/c2c-p2p-rdma-$(date +%s)"

gpu_list="${GPU_LIST:-0,1,2,3}"
hca=""
c2c_buffer_mib="512"
p2p_buffer_mib="256"
warmup_iterations="5"
copy_iterations="20"
p2p_iterations="20"
nsys_enabled=false
output_dir="${RDMA_C2C_P2P_INTERNAL_OUTPUT_DIR:-$default_output_dir}"

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
        --p2p-buffer-mib)
            require_value "$1" "$#"
            p2p_buffer_mib="$2"
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
        --p2p-iterations)
            require_value "$1" "$#"
            p2p_iterations="$2"
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

if [[ "${RDMA_C2C_P2P_LOG_CAPTURED:-0}" != 1 ]]; then
    mkdir -p "$output_dir"
    run_log="${output_dir}/run.log"
    [[ ! -e "$run_log" ]] || die "output directory already contains run.log; use a fresh directory: $output_dir"
    [[ ! -e "${output_dir}/torchrun.log" ]] \
        || die "output directory already contains torchrun.log; use a fresh directory: $output_dir"
    export RDMA_C2C_P2P_LOG_CAPTURED=1
    export RDMA_C2C_P2P_INTERNAL_OUTPUT_DIR="$output_dir"
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
existing_nccl_logs=("${output_dir}"/nccl-*)
[[ ${#existing_nccl_logs[@]} -eq 0 ]] \
    || die "output directory already contains NCCL logs; use a fresh directory: $output_dir"
if [[ "$nsys_enabled" == true ]]; then
    existing_nsys_reports=(
        "${output_dir}"/nsys*.nsys-rep
        "${output_dir}"/nsys*.qdrep
        "${output_dir}"/nsys*.qdstrm
    )
    [[ ${#existing_nsys_reports[@]} -eq 0 ]] \
        || die "output directory already contains Nsight reports; use a fresh directory: $output_dir"
fi

export CUDA_VISIBLE_DEVICES="$gpu_list"
export BENCHMARK_GPU_PCI_BUS_IDS="$gpu_pci_bus_ids_csv"
export NUMARUN_MEMBIND="${NUMARUN_MEMBIND:-1}"
if [[ -n "$hca" ]]; then
    export NCCL_IB_HCA="$hca"
fi
export NCCL_IB_DISABLE=0
export NCCL_MNNVL_ENABLE=0
export NCCL_NET=IB
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_NET_GDR_C2C=1
export NCCL_P2P_DISABLE=1
export NCCL_NVB_DISABLE=1
export NCCL_PXN_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH
export NCCL_DEBUG_FILE="${output_dir}/nccl-%h-%p.log"
export GLOO_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=lo
export NCCL_SOCKET_FAMILY=AF_INET
unset MASTER_ADDR MASTER_PORT

hca_for_log="${hca:-${NCCL_IB_HCA:-auto}}"
echo "gpus=$gpu_list pci_bus_ids=$gpu_pci_bus_ids_csv hca=$hca_for_log" >&2
echo "transport=IB p2p=network-only nvb=disabled pxn=disabled shm=disabled numarun_membind=$NUMARUN_MEMBIND" >&2
echo "control_transport=gloo/lo nccl_bootstrap=lo rdzv=127.0.0.1:0" >&2
echo "output_dir=$output_dir" >&2

if command -v ip >/dev/null; then
    ip addr
else
    echo "WARNING: ip command is unavailable; skipping network interface dump" >&2
fi

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
    --p2p-buffer-mib "$p2p_buffer_mib"
    --warmup-iterations "$warmup_iterations"
    --copy-iterations "$copy_iterations"
    --p2p-iterations "$p2p_iterations"
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

if [[ "$nsys_enabled" == true ]]; then
    nsys_reports=("${output_dir}"/nsys*.nsys-rep "${output_dir}"/nsys*.qdrep)
    if [[ ${#nsys_reports[@]} -eq 0 ]]; then
        nsys_intermediate=("${output_dir}"/nsys*.qdstrm)
        if [[ ${#nsys_intermediate[@]} -gt 0 ]]; then
            die "nsys left intermediate files without a report: ${nsys_intermediate[*]}; inspect ${torchrun_log} and run nsys import"
        fi
        die "nsys did not create a report in ${output_dir}; inspect ${torchrun_log}"
    fi
fi

nccl_logs=("${output_dir}"/nccl-*)
[[ ${#nccl_logs[@]} -gt 0 ]] || die "NCCL did not create transport logs in ${output_dir}"
grep -Eq "NET/IB|Using network IB|IBext" "${nccl_logs[@]}" \
    || die "NCCL logs do not confirm an IB transport"
grep -Eq "GDRDMA" "${nccl_logs[@]}" \
    || die "NCCL logs do not confirm GPUDirect RDMA; refusing to report this run as an RDMA experiment"
if grep -Eq "NET/Socket|Using network Socket" "${nccl_logs[@]}"; then
    die "NCCL logs show a Socket fallback; refusing to report this run as an RDMA experiment"
fi

echo "Result: ${output_dir}/result.json" >&2
