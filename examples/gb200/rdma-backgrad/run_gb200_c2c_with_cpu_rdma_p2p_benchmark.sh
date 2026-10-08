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
Run the GB200 C2C benchmark with persistent CPU RDMA Verbs traffic.

The four workers use Gloo/TCP only for control. Each pair starts one
ib_write_bw server and one client on host memory, with the HCA and source IP
selected from nvidia-smi topo -m and ip addr output. The C2C data path is
measured in four direction/role groups: d2h_send, d2h_recv, h2d_send,
h2d_recv.

Usage:
  run_gb200_c2c_with_cpu_rdma_p2p_benchmark.sh [options]

Options:
  --gpus LIST               Physical GPUs (default: 0,1,2,3)
  --c2c-buffer-mib MIB      Pinned-host/GPU copy buffer (default: 512)
  --warmup-iterations N     C2C warmup copies (default: 5)
  --copy-iterations N       Timed C2C copies (default: 20)
  --ib-write-bw PATH        ib_write_bw executable (default: ib_write_bw)
  --rdma-size-mib MIB       Fixed ib_write_bw message size (default: 256)
  --rdma-qp N               ib_write_bw queue pairs (default: 1)
  --rdma-tx-depth N         ib_write_bw TX depth (default: 128)
  --rdma-report-interval S  Perftest report interval (default: 5)
  --rdma-ready-timeout-seconds S
                            Startup timeout (default: 30)
  --rdma-startup-delay-seconds S
                            Time to keep perftest alive before C2C (default: 2)
  --rdma-port PORT          First perftest TCP port (default: 18515)
  --topology-file PATH      nvidia-smi topo -m capture
  --ip-addr-file PATH       ip addr capture (default: .cache/ip_addr.txt)
  --nsys                    Profile with Nsight Systems
  --output-dir DIR          Logs and rank-0 JSON directory
  -h, --help                Show this help

Example:
  bash examples/gb200/rdma-backgrad/run_gb200_c2c_with_cpu_rdma_p2p_benchmark.sh \
    --gpus 0,1,2,3 --rdma-size-mib 256
USAGE
}

die() {
    echo "ERROR: $*" >&2
    exit 2
}

require_value() {
    [[ $# -ge 2 ]] || die "$1 requires a value"
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
benchmark_script="${script_dir}/gb200_c2c_with_cpu_rdma_p2p_benchmark.py"

gpu_list="${GPU_LIST:-0,1,2,3}"
c2c_buffer_mib="512"
warmup_iterations="5"
copy_iterations="20"
ib_write_bw="ib_write_bw"
rdma_size_mib="256"
rdma_qp="1"
rdma_tx_depth="128"
rdma_report_interval="5"
rdma_ready_timeout_seconds="30"
rdma_startup_delay_seconds="2"
rdma_port="18515"
topology_file="${repo_root}/.cache/nvidia-smi_topo.txt"
ip_addr_file=""
nsys_enabled=false
output_dir="${CPU_RDMA_P2P_INTERNAL_OUTPUT_DIR:-}"
original_args=("$@")

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus) require_value "$@"; gpu_list="$2"; shift 2 ;;
        --c2c-buffer-mib) require_value "$@"; c2c_buffer_mib="$2"; shift 2 ;;
        --warmup-iterations) require_value "$@"; warmup_iterations="$2"; shift 2 ;;
        --copy-iterations) require_value "$@"; copy_iterations="$2"; shift 2 ;;
        --ib-write-bw) require_value "$@"; ib_write_bw="$2"; shift 2 ;;
        --rdma-size-mib) require_value "$@"; rdma_size_mib="$2"; shift 2 ;;
        --rdma-qp) require_value "$@"; rdma_qp="$2"; shift 2 ;;
        --rdma-tx-depth) require_value "$@"; rdma_tx_depth="$2"; shift 2 ;;
        --rdma-report-interval) require_value "$@"; rdma_report_interval="$2"; shift 2 ;;
        --rdma-ready-timeout-seconds) require_value "$@"; rdma_ready_timeout_seconds="$2"; shift 2 ;;
        --rdma-startup-delay-seconds) require_value "$@"; rdma_startup_delay_seconds="$2"; shift 2 ;;
        --rdma-port) require_value "$@"; rdma_port="$2"; shift 2 ;;
        --topology-file) require_value "$@"; topology_file="$2"; shift 2 ;;
        --ip-addr-file) require_value "$@"; ip_addr_file="$2"; shift 2 ;;
        --nsys) nsys_enabled=true; shift ;;
        --output-dir) require_value "$@"; output_dir="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

if [[ -z "$output_dir" ]]; then
    output_dir="${repo_root}/results/gb200/rdma-backward/c2c-cpu-rdma-p2p-$(date +%s)"
fi

if [[ "${CPU_RDMA_P2P_LOG_CAPTURED:-0}" != 1 ]]; then
    mkdir -p "$output_dir"
    [[ ! -e "${output_dir}/run.log" ]] || die "output directory already contains run.log: $output_dir"
    export CPU_RDMA_P2P_LOG_CAPTURED=1
    export CPU_RDMA_P2P_INTERNAL_OUTPUT_DIR="$output_dir"
    bash "$0" "${original_args[@]}" 2>&1 | tee "${output_dir}/run.log"
    exit "${PIPESTATUS[0]}"
fi

[[ -f "$benchmark_script" ]] || die "benchmark not found: $benchmark_script"
command -v nvidia-smi >/dev/null || die "nvidia-smi is required"
command -v ip >/dev/null || die "ip is required to capture RDMA addresses"
command -v numactl >/dev/null || die "numactl is required by numarun"
command -v torchrun >/dev/null || die "torchrun is required"
command -v python >/dev/null || die "python is required"
command -v "$ib_write_bw" >/dev/null || die "ib_write_bw is required: $ib_write_bw"
if [[ "$nsys_enabled" == true ]]; then
    command -v nsys >/dev/null || die "nsys is required when --nsys is enabled"
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
    bus_id="$(nvidia-smi -i "$gpu" --query-gpu=pci.bus_id --format=csv,noheader | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]' | sed -E 's/^0000([[:xdigit:]]{4}):/\1:/')"
    [[ -n "$bus_id" ]] || die "could not resolve PCI bus ID for GPU $gpu"
    gpu_pci_bus_ids+=("$bus_id")
done
gpu_pci_bus_ids_csv="$(IFS=,; echo "${gpu_pci_bus_ids[*]}")"

gpu_indices_csv="$gpu_list"
mkdir -p "$output_dir"
if [[ -z "$ip_addr_file" ]]; then
    command -v rdma >/dev/null || die "rdma is required to map HCA names to netdevs"
    ip_addr_file="${output_dir}/ip_addr.txt"
    {
        echo "===== Network addresses (ip addr) ====="
        ip addr
        echo "===== RDMA links (rdma link show) ====="
        rdma link show
    } > "$ip_addr_file"
fi
[[ -f "$topology_file" ]] || die "topology file not found: $topology_file"
[[ -f "$ip_addr_file" ]] || die "IP address file not found: $ip_addr_file"

export CUDA_VISIBLE_DEVICES="$gpu_list"
export BENCHMARK_GPU_INDICES="$gpu_indices_csv"
export BENCHMARK_GPU_PCI_BUS_IDS="$gpu_pci_bus_ids_csv"
export NUMARUN_MEMBIND="${NUMARUN_MEMBIND:-1}"
export GLOO_DEVICE_TRANSPORT=TCP
export GLOO_SOCKET_IFNAME=lo
unset MASTER_ADDR MASTER_PORT

echo "gpus=$gpu_list pci_bus_ids=$gpu_pci_bus_ids_csv background=ib_write_bw fixed_size_mib=$rdma_size_mib" >&2
echo "background_transport=host-rdma-verbs control_transport=gloo-tcp topology_file=$topology_file ip_addr_file=$ip_addr_file" >&2
echo "no NCCL/GPU-to-GPU operation; output_dir=$output_dir" >&2

torchrun_command=(
    torchrun --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:0 --rdzv_conf="timeout=60"
    --local_addr=127.0.0.1 --nnodes=1 --nproc-per-node=4 --no-python
    "${numarun_command[@]}" python "$benchmark_script"
    --c2c-buffer-mib "$c2c_buffer_mib" --warmup-iterations "$warmup_iterations"
    --copy-iterations "$copy_iterations" --ib-write-bw "$ib_write_bw"
    --rdma-size-mib "$rdma_size_mib" --rdma-qp "$rdma_qp" --rdma-tx-depth "$rdma_tx_depth"
    --rdma-report-interval "$rdma_report_interval"
    --rdma-ready-timeout-seconds "$rdma_ready_timeout_seconds"
    --rdma-startup-delay-seconds "$rdma_startup_delay_seconds" --rdma-port "$rdma_port"
    --topology-file "$topology_file" --ip-addr-file "$ip_addr_file"
    --output "${output_dir}/result.json"
)

launch_command=("${torchrun_command[@]}")
if [[ "$nsys_enabled" == true ]]; then
    launch_command=(nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --output "${output_dir}/nsys" "${torchrun_command[@]}")
fi

printf 'launch_command:' >&2
printf ' %q' "${launch_command[@]}" >&2
printf '\n' >&2
"${launch_command[@]}" 2>&1 | tee "${output_dir}/torchrun.log"
echo "Result: ${output_dir}/result.json" >&2
