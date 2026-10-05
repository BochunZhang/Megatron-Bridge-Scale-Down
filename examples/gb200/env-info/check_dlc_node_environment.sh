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

set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
if [[ "${DLC_NODE_DIAGNOSTICS_CAPTURED:-0}" != 1 ]]; then
    output_dir="${repo_root}/results/gb200/rdma-backward/dlc-node-$(date +%s)"
    mkdir -p "$output_dir"
    export DLC_NODE_DIAGNOSTICS_CAPTURED=1
    export DLC_NODE_DIAGNOSTICS_OUTPUT_DIR="$output_dir"
    bash "$0" "$@" 2>&1 | tee "${output_dir}/node-environment.log"
    exit "${PIPESTATUS[0]}"
fi
output_dir="${DLC_NODE_DIAGNOSTICS_OUTPUT_DIR}"

section() {
    printf '\n===== %s =====\n' "$1"
}

command_available() {
    command -v "$1" >/dev/null 2>&1
}

print_command_status() {
    if command_available "$1"; then
        printf '%s: %s\n' "$1" "$(command -v "$1")"
    else
        printf '%s: unavailable\n' "$1"
    fi
}

print_path_entries() {
    local path="$1"
    local label="$2"
    local entries

    if [[ ! -e "$path" ]]; then
        printf '%s: %s is absent\n' "$label" "$path"
        return 0
    fi

    entries="$(find "$path" -mindepth 1 -maxdepth 1 -exec basename {} \; 2>/dev/null | sort)"
    if [[ -n "$entries" ]]; then
        printf '%s (%s):\n%s\n' "$label" "$path" "$entries"
    else
        printf '%s (%s): empty\n' "$label" "$path"
    fi
}

print_matching_entries() {
    local path="$1"
    local label="$2"
    local pattern="$3"
    local entries

    if [[ ! -e "$path" ]]; then
        printf '%s: %s is absent\n' "$label" "$path"
        return 0
    fi

    entries="$(find "$path" -mindepth 1 -maxdepth 1 -name "$pattern" -exec basename {} \; 2>/dev/null | sort)"
    if [[ -n "$entries" ]]; then
        printf '%s (%s):\n%s\n' "$label" "$path" "$entries"
    else
        printf '%s (%s): empty\n' "$label" "$path"
    fi
}

print_port_details() {
    local hca_path="$1"
    local hca_name
    local port_path
    local state_path
    local state
    local net_path
    local net_name

    hca_name="${hca_path##*/}"
    while IFS= read -r port_path; do
        [[ -d "$port_path" ]] || continue
        state_path="${port_path}/state"
        state="unknown"
        [[ -r "$state_path" ]] && state="$(<"$state_path")"
        printf 'HCA=%s %s state=%s' "$hca_name" "${port_path##*/}" "$state"

        net_path="${port_path}/gid_attrs/ndevs"
        if [[ -d "$net_path" ]]; then
            net_name="$(find "$net_path" -mindepth 1 -maxdepth 1 -type f -print -quit 2>/dev/null)"
            if [[ -n "$net_name" ]]; then
                printf ' netdev=%s' "$(<"$net_name")"
            fi
        fi
        printf '\n'
    done < <(find "$hca_path/ports" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort)
}

printf 'DLC node diagnostics\n'
printf 'output_dir=%s\n' "$output_dir"
printf 'timestamp=%s\n' "$(date --iso-8601=seconds 2>/dev/null || date)"
printf 'hostname=%s\n' "$(hostname 2>/dev/null || printf 'unknown')"

section "DLC environment (printenv)"
printenv | sort

section "Command availability"
for command_name in df ip find lsblk lspci ibdev2netdev ibv_devinfo rdma numactl nvidia-smi; do
    print_command_status "$command_name"
done

section "Filesystems (df -h)"
if command_available df; then
    df -h
    printf '\nMounted NVMe filesystems:\n'
    nvme_mounts="$(df -hP | awk 'NR == 1 || $1 ~ /^\/dev\/nvme[0-9]+n[0-9]+/')"
    if [[ -n "$nvme_mounts" ]]; then
        printf '%s\n' "$nvme_mounts"
        nvme_mount_count="$(df -hP | awk 'NR > 1 && $1 ~ /^\/dev\/nvme[0-9]+n[0-9]+/ {count++} END {print count + 0}')"
        printf 'mounted_nvme_filesystem_count=%s\n' "$nvme_mount_count"
    else
        printf 'none\nmounted_nvme_filesystem_count=0\n'
    fi
else
    printf 'df is unavailable\n'
fi

if command_available lsblk; then
    section "Block devices (lsblk NVMe summary)"
    lsblk -e 7 -o NAME,PATH,MODEL,SIZE,TYPE,TRAN,FSTYPE,MOUNTPOINTS 2>&1 \
        | awk 'NR == 1 || $1 ~ /^nvme/ || $2 ~ /\/nvme/'
fi

section "Network addresses (ip addr)"
if command_available ip; then
    ip addr show
    printf '\nRDMA-related interfaces:\n'
    ip -o link show | awk -F': ' 'tolower($2) ~ /(^|[^[:alnum:]])(ib|mlx|rdma|bond)([^[:alnum:]]|$)/'
    if command_available ibdev2netdev; then
        printf '\nRDMA device-to-netdev mapping (ibdev2netdev):\n'
        ibdev2netdev || true
    fi
    if command_available rdma; then
        printf '\nRDMA link state (rdma link show):\n'
        rdma link show || true
    fi
else
    printf 'ip is unavailable\n'
fi

section "/dev NVMe and RDMA entries"
printf '/dev top-level entries matching NVMe/RDMA:\n'
dev_top_entries="$(find /dev -mindepth 1 -maxdepth 1 \( -iname 'nvme*' -o -iname '*rdma*' -o -iname 'infiniband' \) -print 2>/dev/null | sort)"
if [[ -n "$dev_top_entries" ]]; then
    printf '%s\n' "$dev_top_entries"
else
    printf 'none\n'
fi
dev_entries="$(find /dev -maxdepth 2 \( -iname 'nvme*' -o -iname '*rdma*' -o -path '/dev/infiniband/*' \) -print 2>/dev/null | sort)"
if [[ -n "$dev_entries" ]]; then
    printf '%s\n' "$dev_entries"
else
    printf 'No NVMe or RDMA entries found below /dev\n'
fi
dev_nvme_count="$(find /dev -maxdepth 1 -type b -name 'nvme*' -print 2>/dev/null | wc -l | tr -d ' ')"
dev_rdma_count="$(find /dev/infiniband -maxdepth 1 -mindepth 1 -print 2>/dev/null | wc -l | tr -d ' ')"
printf 'dev_nvme_block_device_count=%s\n' "$dev_nvme_count"
printf 'dev_infiniband_entry_count=%s\n' "$dev_rdma_count"

section "/sys NVMe information"
print_path_entries "/sys/class/nvme" "NVMe controllers"
if [[ -d /sys/class/nvme ]]; then
    while IFS= read -r controller_path; do
        [[ -d "$controller_path" ]] || continue
        controller_name="${controller_path##*/}"
        model="unknown"
        serial="unknown"
        address="unknown"
        [[ -r "$controller_path/model" ]] && model="$(<"$controller_path/model")"
        [[ -r "$controller_path/serial" ]] && serial="$(<"$controller_path/serial")"
        [[ -L "$controller_path/device" ]] && address="$(readlink -f "$controller_path/device" 2>/dev/null || printf 'unknown')"
        printf 'controller=%s model=%s serial=%s device=%s\n' "$controller_name" "$model" "$serial" "$address"
    done < <(
        find /sys/class/nvme -mindepth 1 -maxdepth 1 \( -type l -o -type d \) 2>/dev/null | sort
    )
fi
sys_nvme_namespace_count="$(find /sys/class/block -mindepth 1 -maxdepth 1 -type l -name 'nvme*n*' -print 2>/dev/null | wc -l | tr -d ' ')"
printf 'sys_nvme_namespace_count=%s\n' "$sys_nvme_namespace_count"

section "/sys RDMA information"
print_path_entries "/sys/class/infiniband" "RDMA HCAs"
print_path_entries "/sys/class/rdma_cm" "RDMA CM devices"
print_path_entries "/sys/class/rdma" "RDMA class devices"
if [[ -d /sys/class/infiniband ]]; then
    while IFS= read -r hca_path; do
        [[ -d "$hca_path" ]] || continue
        hca_name="${hca_path##*/}"
        numa_node="unknown"
        device_path="${hca_path}/device"
        [[ -r "${device_path}/numa_node" ]] && numa_node="$(<"${device_path}/numa_node")"
        printf 'hca=%s numa_node=%s device=%s\n' "$hca_name" "$numa_node" "$(readlink -f "$device_path" 2>/dev/null || printf 'unknown')"
        print_port_details "$hca_path"
    done < <(
        find /sys/class/infiniband -mindepth 1 -maxdepth 1 \( -type l -o -type d \) 2>/dev/null | sort
    )
fi

print_matching_entries "/sys/block" "NVMe block devices" "nvme*"
print_path_entries "/sys/bus/pci/drivers/nvme" "NVMe PCI driver devices"
print_path_entries "/sys/bus/pci/drivers/mlx5_core" "mlx5 PCI driver devices"

print_path_entries "/sys/class/net" "Network interfaces"
if [[ -d /sys/class/net ]]; then
    printf '\nNetwork interfaces with RDMA-like names:\n'
    while IFS= read -r net_path; do
        [[ -e "$net_path" ]] || continue
        printf '%s -> %s\n' "$(basename "$net_path")" "$(readlink -f "$net_path" 2>/dev/null || printf 'unknown')"
    done < <(
        find /sys/class/net -mindepth 1 -maxdepth 1 \
            \( -iname '*ib*' -o -iname '*mlx*' -o -iname '*rdma*' -o -iname '*bond*' \) \
            -print 2>/dev/null | sort
    )
fi

section "PCI NVMe/RDMA devices"
if command_available lspci; then
    lspci -Dnn 2>&1 | grep -Ei 'Non-Volatile memory|NVMe|Ethernet controller|InfiniBand|Mellanox|NVIDIA' || true
else
    printf 'lspci is unavailable\n'
fi

section "Summary"
printf 'nvme_controllers=%s\n' "$(find /sys/class/nvme -mindepth 1 -maxdepth 1 2>/dev/null | wc -l | tr -d ' ')"
printf 'nvme_namespaces=%s\n' "$sys_nvme_namespace_count"
printf 'rdma_hcas=%s\n' "$(find /sys/class/infiniband -mindepth 1 -maxdepth 1 2>/dev/null | wc -l | tr -d ' ')"
printf 'rdma_device_entries=%s\n' "$dev_rdma_count"
