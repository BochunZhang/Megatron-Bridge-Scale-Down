#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

set -euo pipefail
export LC_ALL=C

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

show_link() {
    local device=$1 field value
    for field in class max_link_speed max_link_width current_link_speed current_link_width; do
        value=$(cat -- "$device/$field" 2>/dev/null) || value=unavailable
        printf '  %s = %s\n' "$field" "$value"
    done
}

if [[ ${1-} == --help || ${1-} == -h ]]; then
    printf 'Usage: %s [PATH]\nReport the PCIe link for the NVMe mount containing PATH (default: /tmp).\n' "${0##*/}"
    exit 0
fi
(( $# <= 1 )) || die 'Expected at most one path. See --help.'
[[ $(uname -s) == Linux ]] || die 'This diagnostic requires Linux sysfs.'
command -v findmnt >/dev/null || die 'findmnt from util-linux is required.'

target=${1-/tmp}
mount=$(findmnt --target "$target" --noheadings --raw --output MAJ:MIN,SOURCE,TARGET,FSTYPE) ||
    die "Cannot find the mount containing $target."
read -r device_id source mountpoint filesystem <<< "$mount"
printf 'Host: %s\n' "$(hostname)"
printf 'Mount: %s (%s) on %s [%s]\n' "$source" "$device_id" "$mountpoint" "$filesystem"

# The mount's major/minor identifies the block device even for a bind mount in a container.
namespace=$(readlink -e -- "/sys/dev/block/$device_id") ||
    die "Block device $device_id is unavailable in sysfs."
[[ ${namespace##*/} =~ ^nvme[0-9]+n[0-9]+(p[0-9]+)?$ ]] ||
    die "The mount uses ${namespace##*/}; expected a direct NVMe namespace or partition."
if [[ -n ${BASH_REMATCH[1]} ]]; then
    namespace=${namespace%/*}
fi
controller=${namespace%/*}
transport=$(cat -- "$controller/transport" 2>/dev/null) || die "Cannot resolve a direct NVMe controller for $source."
[[ $transport == pcie ]] || die "The NVMe transport for $source is $transport; expected pcie."
model=$(cat -- "$controller/model") || die "Cannot read the SSD model for $source."
pci=$(readlink -e -- "$controller/device") || die "Cannot resolve the PCI device for ${controller##*/}."
upstream=${pci%/*}

printf 'Namespace: %s\nController: %s\nSSD model: %s\nTransport: %s\n' \
    "${namespace##*/}" "${controller##*/}" "$model" "$transport"
printf 'PCI endpoint: %s\n' "${pci##*/}"
show_link "$pci"
if [[ -e $upstream/class ]]; then
    printf 'PCI upstream port: %s\n' "${upstream##*/}"
    show_link "$upstream"
else
    printf 'PCI upstream port: unavailable\n'
fi
