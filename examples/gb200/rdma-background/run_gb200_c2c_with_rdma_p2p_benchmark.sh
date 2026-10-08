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
Measure GPU0 C2C bandwidth in three sequential rounds:
  1. Baseline with no background traffic.
  2. RDMA traffic from mlx5_bond_3 to mlx5_bond_0.
  3. RDMA traffic from mlx5_bond_0 to mlx5_bond_3.

RDMA buffers use host memory by default; --use_cuda uses device memory on
GPU0 and GPU3. Current IPv4 addresses come from rdma link show and ip addr;
CPU and host memory are bound to each GPU's live NUMA node. Each RDMA round waits ten
seconds after starting the sender, then measures D2H/H2D on GPU0. Both
perftest processes stop after Python exits, before the next round starts.
Results are stored in baseline/, gpu3_to_gpu0/, and gpu0_to_gpu3/.

Usage:
  run_gb200_c2c_with_rdma_p2p_benchmark.sh [options]

Options:
  --c2c-buffer-mib MIB      Pinned-host/GPU copy buffer (default: 512)
  --warmup-iterations N     C2C warmup copies (default: 5)
  --copy-iterations N       Timed C2C copies (default: 20)
  --ib-write-bw PATH        ib_write_bw executable (default: ib_write_bw)
  --use_cuda                Use CUDA device buffers for both RDMA endpoints
                            (default: host buffers)
  --rdma-size-mib MIB       Fixed ib_write_bw message size (default: 256)
  --rdma-qp N               ib_write_bw queue pairs (default: 8)
  --rdma-tx-depth N         ib_write_bw TX depth (default: 128)
  --rdma-report-interval S  Perftest report interval (default: 5)
  --rdma-port PORT          Perftest connection port (default: 18515)
  --nsys                    Profile the GPU0 Python process with Nsight Systems
  --output-dir DIR          Root directory for logs and per-round results
                            (default: results/gb200/rdma-background/
                             c2c-with-{host|cuda}-rdma-p2p-YYMMDD-HHMMSS, local time)
  -h, --help                Show this help

Example:
  bash examples/gb200/rdma-background/run_gb200_c2c_with_rdma_p2p_benchmark.sh \
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

stop_processes() {
    local pid attempt alive
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
    process_pids=()
}

cleanup() {
    local status=$?
    trap - EXIT INT TERM HUP
    stop_processes
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
    benchmark_script="${script_dir}/gb200_c2c_with_rdma_p2p_benchmark.py"
    local c2c_buffer_mib=512 warmup_iterations=5 copy_iterations=20
    local ib_write_bw=ib_write_bw rdma_size_mib=256 rdma_qp=8 rdma_tx_depth=128
    local rdma_report_interval=5 rdma_port=18515 nsys_enabled=false output_dir=""
    local rdma_memory=host
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --c2c-buffer-mib) require_value "$@"; c2c_buffer_mib="$2"; shift 2 ;;
            --warmup-iterations) require_value "$@"; warmup_iterations="$2"; shift 2 ;;
            --copy-iterations) require_value "$@"; copy_iterations="$2"; shift 2 ;;
            --ib-write-bw) require_value "$@"; ib_write_bw="$2"; shift 2 ;;
            --use_cuda) rdma_memory=cuda; shift ;;
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
        output_dir="${repo_root}/results/gb200/rdma-background/c2c-with-${rdma_memory}-rdma-p2p-$(date +%y%m%d-%H%M%S)"
    fi
    mkdir -p "$output_dir"
    [[ ! -e "${output_dir}/run.log" ]] || die "output directory already contains run.log: $output_dir"
    exec > >(tee "${output_dir}/run.log") 2>&1

    local rdma_links ip_addresses gpu0_address gpu3_address
    rdma_links="$(rdma link show)"
    ip_addresses="$(ip addr)"
    gpu0_address="$(resolve_rdma_address mlx5_bond_0)"
    gpu3_address="$(resolve_rdma_address mlx5_bond_3)"
    local gpu0_netdev gpu0_ip gpu3_netdev gpu3_ip
    read -r gpu0_netdev gpu0_ip <<< "$gpu0_address"
    read -r gpu3_netdev gpu3_ip <<< "$gpu3_address"
    [[ "$gpu0_ip" != "$gpu3_ip" ]] || die "RDMA endpoints must have distinct IPv4 addresses"
    local gpu0_bus_id gpu3_bus_id gpu0_numa gpu3_numa
    gpu0_bus_id="$(gpu_pci_bus_id 0)"
    gpu3_bus_id="$(gpu_pci_bus_id 3)"
    [[ -n "$gpu0_bus_id" && -n "$gpu3_bus_id" ]] || die "could not resolve GPU PCI bus IDs"
    gpu0_numa="$(gpu_numa_node "$gpu0_bus_id")"
    gpu3_numa="$(gpu_numa_node "$gpu3_bus_id")"
    echo "gpu=0 hca=mlx5_bond_0 netdev=$gpu0_netdev ip=$gpu0_ip numa=$gpu0_numa"
    echo "gpu=3 hca=mlx5_bond_3 netdev=$gpu3_netdev ip=$gpu3_ip numa=$gpu3_numa"
    echo "background_transport=rdma-verbs rdma_memory=$rdma_memory; C2C on GPU0; output_dir=$output_dir"

    local -a rdma_args receiver_command sender_command launch_command
    rdma_args=(-R -i 1 -s "$((rdma_size_mib * 1024 * 1024))" -q "$rdma_qp" -t "$rdma_tx_depth"
        -D "$rdma_report_interval" --run_infinitely --report_gbits -p "$rdma_port")
    if [[ "$rdma_memory" == cuda ]]; then
        # Each endpoint exposes just its physical GPU via CUDA_VISIBLE_DEVICES;
        # perftest therefore selects device 0 in that process's CUDA namespace.
        rdma_args+=(--use_cuda=0)
    fi
    # Disable job control so setsid can exec without forking; $! is the group ID.
    set +m
    process_pids=()
    trap cleanup EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    trap 'exit 129' HUP
    local gpu0_role round_dir receiver_gpu sender_gpu receiver_ip sender_ip receiver_numa sender_numa
    local second c2c_pid c2c_status
    for gpu0_role in none recv send; do
        if [[ "$gpu0_role" == none ]]; then
            round_dir="${output_dir}/baseline"
            mkdir -p "$round_dir"
        else
            if [[ "$gpu0_role" == recv ]]; then
                receiver_gpu=0
                receiver_ip="$gpu0_ip"
                receiver_numa="$gpu0_numa"
                sender_gpu=3
                sender_ip="$gpu3_ip"
                sender_numa="$gpu3_numa"
            else
                receiver_gpu=3
                receiver_ip="$gpu3_ip"
                receiver_numa="$gpu3_numa"
                sender_gpu=0
                sender_ip="$gpu0_ip"
                sender_numa="$gpu0_numa"
            fi
            round_dir="${output_dir}/gpu${sender_gpu}_to_gpu${receiver_gpu}"
            mkdir -p "${round_dir}/rdma_logs"
            receiver_log="${round_dir}/rdma_logs/gpu${receiver_gpu}.recv.log"
            sender_log="${round_dir}/rdma_logs/gpu${sender_gpu}.send.log"
            sender_pid=""
            receiver_command=(numactl --cpunodebind="$receiver_numa" --membind="$receiver_numa"
                "$ib_write_bw" "${rdma_args[@]}" -d "mlx5_bond_${receiver_gpu}" --bind_source_ip "$receiver_ip")
            sender_command=(numactl --cpunodebind="$sender_numa" --membind="$sender_numa"
                "$ib_write_bw" "${rdma_args[@]}" -d "mlx5_bond_${sender_gpu}" --bind_source_ip "$sender_ip" "$receiver_ip")
        fi
        launch_command=(numactl --cpunodebind="$gpu0_numa" --membind="$gpu0_numa"
            python "$benchmark_script" --c2c-buffer-mib "$c2c_buffer_mib"
            --warmup-iterations "$warmup_iterations" --copy-iterations "$copy_iterations"
            --rdma-role "$gpu0_role" --rdma-memory "$rdma_memory" --output "${round_dir}/result.json")
        if [[ "$nsys_enabled" == true ]]; then
            launch_command=(nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none
                --output "${round_dir}/nsys" "${launch_command[@]}")
        fi

        if [[ "$gpu0_role" == none ]]; then
            echo "Round: baseline; GPU0 C2C without background traffic"
        else
            echo "Round: GPU${sender_gpu} -> GPU${receiver_gpu}; GPU0 RDMA role=$gpu0_role"
            log_command receiver_command "${receiver_command[@]}"
            CUDA_VISIBLE_DEVICES="$receiver_gpu" setsid "${receiver_command[@]}" > "$receiver_log" 2>&1 &
            receiver_pid=$!
            process_pids+=("$receiver_pid")
            # Give the receiver time to open its endpoint before the client connects.
            sleep 1
            check_rdma_processes
            log_command sender_command "${sender_command[@]}"
            CUDA_VISIBLE_DEVICES="$sender_gpu" setsid "${sender_command[@]}" > "$sender_log" 2>&1 &
            sender_pid=$!
            process_pids+=("$sender_pid")
            echo "RDMA processes started; waiting 10 seconds before GPU0 C2C"
            for ((second = 0; second < 10; second++)); do
                sleep 1
                check_rdma_processes
            done
        fi

        log_command launch_command "${launch_command[@]}"
        CUDA_VISIBLE_DEVICES=0 BENCHMARK_GPU_PCI_BUS_ID="$gpu0_bus_id" \
            setsid "${launch_command[@]}" > "${round_dir}/c2c.log" 2>&1 &
        c2c_pid=$!
        process_pids+=("$c2c_pid")
        while kill -0 "$c2c_pid" 2>/dev/null; do
            if [[ "$gpu0_role" != none ]]; then
                check_rdma_processes
            fi
            sleep 1
        done
        c2c_status=0
        wait "$c2c_pid" || c2c_status=$?
        [[ "$c2c_status" -eq 0 ]] || { echo "ERROR: C2C failed; see ${round_dir}/c2c.log" >&2; exit "$c2c_status"; }
        if [[ "$gpu0_role" != none ]]; then
            check_rdma_processes
        fi
        stop_processes
        echo "Result: ${round_dir}/result.json"
    done
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
