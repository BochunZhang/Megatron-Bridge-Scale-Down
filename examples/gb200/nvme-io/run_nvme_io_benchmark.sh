#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

set -euo pipefail
export LC_ALL=C

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

(( $# == 0 )) || die "Run this benchmark without arguments."
[[ $(uname -s) == Linux ]] || die "Run this benchmark on the GB200 Linux node."
command -v findmnt >/dev/null || die "findmnt from util-linux is required."
command -v fio >/dev/null || die "The fio storage benchmark is required."
fio_version=$(fio --version) && [[ $fio_version == fio-* ]] || die "The fio storage benchmark is required."
fio --enghelp=libaio >/dev/null || die "fio must support the Linux libaio engine."

# Resolve the mounted device through sysfs, which also works for bind mounts
# and in containers where the corresponding /dev node is hidden.
nvme_mount() {
    local device_id filesystem namespace controller model
    read -r device_id filesystem < <(
        findmnt --target "$1" --noheadings --output MAJ:MIN,FSTYPE
    ) || return 1
    [[ $filesystem == ext4 || $filesystem == xfs ]] || return 1
    namespace=$(readlink -e -- "/sys/dev/block/$device_id") || return 1
    [[ ${namespace##*/} =~ ^nvme[0-9]+n[0-9]+(p[0-9]+)?$ ]] || return 1
    if [[ -n ${BASH_REMATCH[1]} ]]; then
        namespace=${namespace%/*}
    fi
    controller=${namespace%/*}
    [[ $(cat -- "$controller/transport" 2>/dev/null) == pcie ]] || return 1
    model=$(cat -- "$controller/model" 2>/dev/null) || return 1
    case "$model" in
        *"Elastic Block Storage"*|*"Amazon Elastic Block Store"*|*"PersistentDisk"*) return 1 ;;
    esac
    printf '%s\n' "$controller"
}

size_gib=16
root_controller=$(nvme_mount / || true)
scratch_dir=''
while IFS= read -r candidate; do
    # DLC mounts job-local NVMe at /tmp. Other systems may expose a data disk
    # at a different mount point. Skip system mounts and file bind mounts.
    case "$candidate" in
        /|/boot|/boot/*|/dev|/dev/*|/etc|/etc/*|/proc|/proc/*|/run|/run/*|/sys|/sys/*|/usr|/usr/*) continue ;;
    esac
    [[ -d $candidate && -w $candidate ]] || continue
    controller=$(nvme_mount "$candidate") || continue
    [[ $controller != "$root_controller" ]] || continue
    free_kib=$(df -k --output=avail -- "$candidate" | awk 'END {print $1}') || continue
    [[ $free_kib =~ ^[0-9]+$ ]] && (( free_kib >= size_gib * 1024 * 1024 )) || continue
    scratch_dir=$(mktemp -d -- "${candidate%/}/.gb200-nvme-io.XXXXXX") || continue
    break
done < <(printf '/tmp\n'; findmnt --list --noheadings --types ext4,xfs --output TARGET)
[[ -n $scratch_dir ]] || die "No writable single-disk PCIe NVMe mount (ext4/XFS) with $size_gib GiB free."

cleanup() {
    rm -f -- "$scratch_dir/payload.bin"
    rmdir -- "$scratch_dir"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)
results_root="$repo_root/results/gb200/nvme-io"
mkdir -p -- "$results_root"
results_dir=$(mktemp -d -- "$results_root/$(date +%Y%m%d-%H%M%S).XXXXXX")
printf 'Results directory: %s\n' "$results_dir"
{
    printf 'Host: %s\nfio: %s\n' "$(hostname)" "$fio_version"
    printf 'SSD controller: %s\nSSD model: %s\n' "${controller##*/}" "$(cat -- "$controller/model")"
    findmnt --target "$scratch_dir" --output SOURCE,TARGET,FSTYPE,MAJ:MIN
    printf 'Working set: %s GiB, block: 128 KiB, queue depth: 32, direct I/O\n' "$size_gib"
} | tee "$results_dir/host_info.txt"

cd -- "$scratch_dir"
fio_options=(--eta=never --filename=payload.bin --ioengine=libaio --direct=1 --bs=128K --iodepth=32 --size="${size_gib}G")
timed_options=(--time_based=1 --runtime=15s --ramp_time=3s --allow_file_create=0)

printf '\nPreparing the full %s GiB test file.\n' "$size_gib"
if ! fio --name=prepare "${fio_options[@]}" --rw=write --fallocate=none --end_fsync=1 >"$results_dir/prepare.log" 2>&1; then
    cat -- "$results_dir/prepare.log" >&2
    die "fio could not prepare the test file."
fi

printf '\nSequential write (3 s warmup, 15 s measured)\n'
fio --name=seq-write "${fio_options[@]}" "${timed_options[@]}" --rw=write --overwrite=1 --end_fsync=1 \
    2>&1 | tee "$results_dir/write.log"

printf '\nSequential read (3 s warmup, 15 s measured)\n'
fio --readonly --name=seq-read "${fio_options[@]}" "${timed_options[@]}" --rw=read \
    2>&1 | tee "$results_dir/read.log"
