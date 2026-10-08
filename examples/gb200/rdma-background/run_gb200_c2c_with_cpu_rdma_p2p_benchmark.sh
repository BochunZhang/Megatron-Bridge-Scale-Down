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
Measure GPU0 C2C bandwidth during persistent CPU RDMA traffic from GPU3's HCA.

Bash selects mlx5_bond_0 (receiver) and mlx5_bond_3 (sender), resolves their
current IPv4 addresses from rdma link show and ip addr, and binds host memory
to each GPU's live NUMA node. Ten seconds after starting the sender, one Python
process measures D2H/H2D on GPU0. Both perftest processes stop when Python exits.

Usage:
  run_gb200_c2c_with_cpu_rdma_p2p_benchmark.sh [options]

Options:
  --c2c-buffer-mib MIB      Pinned-host/GPU copy buffer (default: 512)
  --warmup-iterations N     C2C warmup copies (default: 5)
  --copy-iterations N       Timed C2C copies (default: 20)
  --ib-write-bw PATH        ib_write_bw executable (default: ib_write_bw)
  --rdma-size-mib MIB       Fixed ib_write_bw message size (default: 256)
  --rdma-qp N               ib_write_bw queue pairs (default: 8)
  --rdma-tx-depth N         ib_write_bw TX depth (default: 128)
  --rdma-report-interval S  Perftest report interval (default: 5)
  --rdma-port PORT          Perftest connection port (default: 18515)
  --nsys                    Profile the GPU0 Python process with Nsight Systems
  --output-dir DIR          Logs and result.json directory
  -h, --help                Show this help

Example:
  bash examples/gb200/rdma-background/run_gb200_c2c_with_cpu_rdma_p2p_benchmark.sh \
    --rdma-size-mib 256
USAGE
}

die() {
    echo "ERROR: $*" >&2
    exit 2
}

require_value() {
    [[ $# -ge 2 ]] || die "$1 requires a value"
}

positive_int() {
    [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "$1 must be a positive integer: $2"
}

gpu_pci_bus_id() {
    nvidia-smi -i "$1" --query-gpu=pci.bus_id --format=csv,noheader \
        | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]' \
        | sed -E 's/^0000([[:xdigit:]]{4}):/\1:/'
}

gpu_numa_node() {
    local node
    local path="/sys/bus/pci/devices/$1/numa_node"
    [[ -r "$path" ]] || die "GPU NUMA node is unavailable: $path"
    read -r node < "$path"
    [[ "$node" =~ ^[0-9]+$ ]] || die "GPU PCI device $1 has no usable NUMA node: $node"
    echo "$node"
}

resolve_rdma_address() {
    local hca="$1"
    local netdev
    netdev="$(awk -v hca="$hca" '
        $1 == "link" && $2 == hca "/1" {
            for (i = 3; i < NF; i++) if ($i == "netdev") print $(i + 1)
        }
    ' <<< "$rdma_links")"
    [[ -n "$netdev" && "$netdev" != *$'\n'* ]] || die "cannot resolve a netdev for $hca port 1"
    # A bond slave may own the RDMA link while its master owns the IPv4 address.
    awk -v netdev="$netdev" '
        /^[0-9]+:/ {
            interface = $2
            sub(/:$/, "", interface)
            sub(/@.*/, "", interface)
            for (i = 3; i < NF; i++) if ($i == "master") master[interface] = $(i + 1)
        }
        $1 == "inet" && !address[interface] {
            split($2, parts, "/")
            address[interface] = parts[1]
        }
        END {
            selected = master[netdev] ? master[netdev] : netdev
            if (!address[selected]) selected = netdev
            if (!address[selected]) exit 1
            print selected, address[selected]
        }
    ' <<< "$ip_addresses" || die "cannot find a live IPv4 address for $hca ($netdev)"
}

log_command() {
    printf '%s:' "$1"
    shift
    printf ' %q' "$@"
    printf '\n'
}

cleanup() {
    local status=$?
    local pid attempt alive
    trap - EXIT INT TERM HUP
    # Each command starts in its own session, so this also stops profiler children.
    for pid in "${process_pids[@]:-}"; do
        [[ -n "$pid" ]] || continue
        kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    done
    for ((attempt = 0; attempt < 50; attempt++)); do
        alive=false
        for pid in "${process_pids[@]:-}"; do
            [[ -n "$pid" ]] || continue
            if kill -0 -- "-$pid" 2>/dev/null || kill -0 "$pid" 2>/dev/null; then alive=true; fi
        done
        [[ "$alive" == true ]] || break
        sleep 0.1
    done
    for pid in "${process_pids[@]:-}"; do
        [[ -n "$pid" ]] || continue
        kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    exit "$status"
}

check_rdma_processes() {
    kill -0 "$receiver_pid" 2>/dev/null || die "RDMA receiver exited; see $receiver_log"
    if [[ -n "${sender_pid:-}" ]]; then
        kill -0 "$sender_pid" 2>/dev/null || die "RDMA sender exited; see $sender_log"
    fi
}

main() {
    local script_dir repo_root benchmark_script
    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
    benchmark_script="${script_dir}/gb200_c2c_with_cpu_rdma_p2p_benchmark.py"
    local c2c_buffer_mib=512 warmup_iterations=5 copy_iterations=20
    local ib_write_bw=ib_write_bw rdma_size_mib=256 rdma_qp=8 rdma_tx_depth=128
    local rdma_report_interval=5 rdma_port=18515 nsys_enabled=false output_dir=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --c2c-buffer-mib) require_value "$@"; c2c_buffer_mib="$2"; shift 2 ;;
            --warmup-iterations) require_value "$@"; warmup_iterations="$2"; shift 2 ;;
            --copy-iterations) require_value "$@"; copy_iterations="$2"; shift 2 ;;
            --ib-write-bw) require_value "$@"; ib_write_bw="$2"; shift 2 ;;
            --rdma-size-mib) require_value "$@"; rdma_size_mib="$2"; shift 2 ;;
            --rdma-qp) require_value "$@"; rdma_qp="$2"; shift 2 ;;
            --rdma-tx-depth) require_value "$@"; rdma_tx_depth="$2"; shift 2 ;;
            --rdma-report-interval) require_value "$@"; rdma_report_interval="$2"; shift 2 ;;
            --rdma-port) require_value "$@"; rdma_port="$2"; shift 2 ;;
            --nsys) nsys_enabled=true; shift ;;
            --output-dir) require_value "$@"; output_dir="$2"; shift 2 ;;
            -h|--help) usage; return 0 ;;
            *) die "unknown option: $1" ;;
        esac
    done
    positive_int --c2c-buffer-mib "$c2c_buffer_mib"
    positive_int --warmup-iterations "$warmup_iterations"
    positive_int --copy-iterations "$copy_iterations"
    positive_int --rdma-size-mib "$rdma_size_mib"
    positive_int --rdma-qp "$rdma_qp"
    positive_int --rdma-tx-depth "$rdma_tx_depth"
    positive_int --rdma-report-interval "$rdma_report_interval"
    positive_int --rdma-port "$rdma_port"
    [[ "$rdma_port" -le 65535 ]] || die "--rdma-port must be at most 65535"

    [[ -f "$benchmark_script" ]] || die "benchmark not found: $benchmark_script"
    local tool
    for tool in nvidia-smi ip rdma numactl setsid python "$ib_write_bw"; do
        command -v "$tool" >/dev/null || die "required command not found: $tool"
    done
    if [[ "$nsys_enabled" == true ]]; then
        command -v nsys >/dev/null || die "nsys is required when --nsys is enabled"
    fi
    if [[ -z "$output_dir" ]]; then
        output_dir="${repo_root}/results/gb200/rdma-background/c2c-cpu-rdma-p2p-$(date +%s)"
    fi
    mkdir -p "${output_dir}/rdma_logs"
    [[ ! -e "${output_dir}/run.log" ]] || die "output directory already contains run.log: $output_dir"
    exec > >(tee "${output_dir}/run.log") 2>&1

    local rdma_links ip_addresses receiver_address sender_address
    rdma_links="$(rdma link show)"
    ip_addresses="$(ip addr)"
    receiver_address="$(resolve_rdma_address mlx5_bond_0)"
    sender_address="$(resolve_rdma_address mlx5_bond_3)"
    local receiver_netdev receiver_ip sender_netdev sender_ip
    read -r receiver_netdev receiver_ip <<< "$receiver_address"
    read -r sender_netdev sender_ip <<< "$sender_address"
    [[ "$receiver_ip" != "$sender_ip" ]] || die "RDMA endpoints must have distinct IPv4 addresses"
    local receiver_bus_id sender_bus_id receiver_numa sender_numa
    receiver_bus_id="$(gpu_pci_bus_id 0)"
    sender_bus_id="$(gpu_pci_bus_id 3)"
    [[ -n "$receiver_bus_id" && -n "$sender_bus_id" ]] || die "could not resolve GPU PCI bus IDs"
    receiver_numa="$(gpu_numa_node "$receiver_bus_id")"
    sender_numa="$(gpu_numa_node "$sender_bus_id")"
    echo "receiver: gpu=0 hca=mlx5_bond_0 netdev=$receiver_netdev ip=$receiver_ip numa=$receiver_numa"
    echo "sender: gpu=3 hca=mlx5_bond_3 netdev=$sender_netdev ip=$sender_ip numa=$sender_numa"
    echo "background_transport=host-rdma-verbs; C2C on GPU0; output_dir=$output_dir"

    local -a rdma_args receiver_command sender_command launch_command
    rdma_args=(-R -i 1 -s "$((rdma_size_mib * 1024 * 1024))" -q "$rdma_qp" -t "$rdma_tx_depth"
        -D "$rdma_report_interval" --run_infinitely --report_gbits -p "$rdma_port")
    receiver_command=(numactl --cpunodebind="$receiver_numa" --membind="$receiver_numa"
        "$ib_write_bw" "${rdma_args[@]}" -d mlx5_bond_0 --bind_source_ip "$receiver_ip")
    sender_command=(numactl --cpunodebind="$sender_numa" --membind="$sender_numa"
        "$ib_write_bw" "${rdma_args[@]}" -d mlx5_bond_3 --bind_source_ip "$sender_ip" "$receiver_ip")
    launch_command=(numactl --cpunodebind="$receiver_numa" --membind="$receiver_numa"
        python "$benchmark_script" --c2c-buffer-mib "$c2c_buffer_mib"
        --warmup-iterations "$warmup_iterations" --copy-iterations "$copy_iterations"
        --output "${output_dir}/result.json")
    if [[ "$nsys_enabled" == true ]]; then
        launch_command=(nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none
            --output "${output_dir}/nsys" "${launch_command[@]}")
    fi

    # Disable job control so setsid can exec without forking; $! is the group ID.
    set +m
    process_pids=()
    receiver_log="${output_dir}/rdma_logs/gpu0.recv.log"
    sender_log="${output_dir}/rdma_logs/gpu3.send.log"
    sender_pid=""
    trap cleanup EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    trap 'exit 129' HUP
    log_command receiver_command "${receiver_command[@]}"
    CUDA_VISIBLE_DEVICES=0 setsid "${receiver_command[@]}" > "$receiver_log" 2>&1 &
    receiver_pid=$!
    process_pids+=("$receiver_pid")
    # Give the receiver time to open its endpoint before the client connects.
    sleep 1
    check_rdma_processes
    log_command sender_command "${sender_command[@]}"
    CUDA_VISIBLE_DEVICES=3 setsid "${sender_command[@]}" > "$sender_log" 2>&1 &
    sender_pid=$!
    process_pids+=("$sender_pid")
    echo "RDMA processes started; waiting 10 seconds before GPU0 C2C"
    local second
    for ((second = 0; second < 10; second++)); do
        sleep 1
        check_rdma_processes
    done

    log_command launch_command "${launch_command[@]}"
    CUDA_VISIBLE_DEVICES=0 BENCHMARK_GPU_PCI_BUS_ID="$receiver_bus_id" \
        setsid "${launch_command[@]}" > "${output_dir}/c2c.log" 2>&1 &
    local c2c_pid=$!
    process_pids+=("$c2c_pid")
    while kill -0 "$c2c_pid" 2>/dev/null; do
        check_rdma_processes
        sleep 1
    done
    local c2c_status=0
    wait "$c2c_pid" || c2c_status=$?
    [[ "$c2c_status" -eq 0 ]] || { echo "ERROR: C2C failed; see ${output_dir}/c2c.log" >&2; exit "$c2c_status"; }
    check_rdma_processes
    echo "Result: ${output_dir}/result.json"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
