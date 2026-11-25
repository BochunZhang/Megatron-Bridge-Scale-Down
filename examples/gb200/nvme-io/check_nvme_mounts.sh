#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

set -u
export LC_ALL=C

if [[ ${1-} == --help || ${1-} == -h ]]; then
    printf 'Usage: %s [PATH]\n' "${0##*/}"
    printf 'Show NVMe mounts and probe signatures read-only. PATH defaults to /tmp.\n'
    exit 0
fi

if (( $# > 1 )); then
    printf 'Expected at most one path. See --help.\n' >&2
    exit 2
fi
if [[ $(uname -s) != Linux ]]; then
    printf 'This diagnostic requires Linux.\n' >&2
    exit 1
fi
for command in lsblk findmnt wipefs; do
    if ! command -v "$command" >/dev/null; then
        printf 'Missing util-linux command: %s\n' "$command" >&2
        exit 1
    fi
done

printf 'Effective UID: %s\n' "$EUID"
if command -v systemd-detect-virt >/dev/null; then
    printf 'Virtualization: '
    systemd-detect-virt || :
fi

printf '\n=== Block device tree ===\n'
lsblk -e 7 -o NAME,PATH,MAJ:MIN,TYPE,SIZE,RO,PTTYPE,FSTYPE,MOUNTPOINT || exit 1
device_rows=$(lsblk --list --noheadings --output KNAME,PATH,TYPE) || exit 1

status=0
target=${1-/tmp}
printf '\n=== Mount containing %s ===\n' "$target"
if ! findmnt --target "$target" --output TARGET,SOURCE,MAJ:MIN,FSTYPE,OPTIONS; then
    printf 'Unable to resolve the mount for this path.\n'
    status=1
fi

printf '\n=== NVMe signatures (type and offset) ===\n'
found=0
while read -r kernel_name device device_type; do
    [[ $kernel_name =~ ^nvme[0-9]+n[0-9]+(p[0-9]+)?$ ]] || continue
    [[ $device_type == disk || $device_type == part ]] || continue
    found=1
    printf '%s (%s)\n' "$device" "$device_type"

    if [[ ! -b $device ]]; then
        printf '  Block device node unavailable in this environment.\n'
        status=1
    elif ! signatures=$(wipefs --no-act --noheadings --output TYPE,OFFSET -- "$device" 2>&1); then
        printf '  Probe failed: %s\n' "$signatures"
        status=1
    elif [[ -n $signatures ]]; then
        printf '%s\n' "$signatures"
    else
        printf '  No recognized signatures detected.\n'
    fi
done <<< "$device_rows"

if (( ! found )); then
    printf 'No NVMe disks or partitions listed by lsblk.\n'
    status=1
fi
exit "$status"
