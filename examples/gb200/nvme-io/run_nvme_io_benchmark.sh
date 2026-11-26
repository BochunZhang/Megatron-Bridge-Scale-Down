#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

set -euo pipefail
export LC_ALL=C

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

(( $# <= 1 )) || die "Expected at most one option. See --help."
profile=${1:-}
case "$profile" in
    '')
        size_mib=16384
        size_label='16 GiB'
        results_prefix=''
        ;;
    --40mib)
        size_mib=40
        size_label='40 MiB'
        results_prefix='fio-40mib-'
        ;;
    -h|--help)
        cat <<'EOF'
Usage: run_nvme_io_benchmark.sh [--40mib]

Run a 16 GiB sequential read/write bandwidth baseline on a local NVMe SSD.
With --40mib, measure single-pass sequential reads and overwrites of a
40 MiB file at 64, 128, and 256 KiB blocks and queue depths 1 to 64 by
powers of two. The sweep writes summary.csv and matrix.log under
results/gb200/nvme-io/fio-40mib-*/.
EOF
        exit 0
        ;;
    *) die "Unknown option: $profile. See --help." ;;
esac
size_kib=$((size_mib * 1024))

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
    [[ $free_kib =~ ^[0-9]+$ ]] && (( free_kib >= size_kib )) || continue
    scratch_dir=$(mktemp -d -- "${candidate%/}/.gb200-nvme-io.XXXXXX") || continue
    break
done < <(printf '/tmp\n'; findmnt --list --noheadings --types ext4,xfs --output TARGET)
[[ -n $scratch_dir ]] || die "No writable single-disk PCIe NVMe mount (ext4/XFS) with $size_label free."

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
results_dir=$(mktemp -d -- "$results_root/$results_prefix$(date +%Y%m%d-%H%M%S).XXXXXX")
printf 'Results directory: %s\n' "$results_dir"
{
    printf 'Host: %s\nfio: %s\n' "$(hostname)" "$fio_version"
    printf 'SSD controller: %s\nSSD model: %s\n' "${controller##*/}" "$(cat -- "$controller/model")"
    findmnt --target "$scratch_dir" --output SOURCE,TARGET,FSTYPE,MAJ:MIN
    if [[ $profile == --40mib ]]; then
        printf 'Working set: %s, direct I/O, one pass per job, writes without fsync\n' "$size_label"
    else
        printf 'Working set: %s, block: 128 KiB, queue depth: 32, direct I/O\n' "$size_label"
    fi
} | tee "$results_dir/host_info.txt"

cd -- "$scratch_dir"
fio_options=(--eta=never --filename=payload.bin --ioengine=libaio --direct=1 --size="$((size_kib * 1024))")
baseline_options=(--bs=128K --iodepth=32)

printf '\nPreparing the full %s test file.\n' "$size_label"
if ! fio --name=prepare "${fio_options[@]}" "${baseline_options[@]}" --rw=write --fallocate=none --end_fsync=1 \
    >"$results_dir/prepare.log" 2>&1; then
    cat -- "$results_dir/prepare.log" >&2
    die "fio could not prepare the test file."
fi

if [[ $profile == --40mib ]]; then
    blocks_kib=(64 128 256)
    queue_depths=(1 2 4 8 16 32 64)
    matrix_log="$results_dir/matrix.log"
    for block_kib in "${blocks_kib[@]}"; do
        printf '\nRunning single-pass sequential read and write with %s KiB blocks.\n' "$block_kib"
        for qp in "${queue_depths[@]}"; do
            matrix_options=(
                --bs="${block_kib}K" --iodepth="$qp" --allow_file_create=0
                --output-format=normal,terse --terse-version=3
            )
            fio --readonly --name="sequential_read-$block_kib-$qp" "${fio_options[@]}" "${matrix_options[@]}" \
                --rw=read >>"$matrix_log" 2>&1
            fio --name="sequential_write-$block_kib-$qp" "${fio_options[@]}" "${matrix_options[@]}" \
                --rw=write --overwrite=1 >>"$matrix_log" 2>&1
        done
    done

    expected_rows=$((${#blocks_kib[@]} * ${#queue_depths[@]} * 2))
    # fio terse v3 fields 6/7 and 47/48 are total KiB and KiB/s for reads and writes.
    if ! awk -F ';' -v size_kib="$size_kib" -v expected_rows="$expected_rows" '
        BEGIN { print "file_size,block_kib,qp,mode,read_gib_per_s,write_gib_per_s,total_gib_per_s" }
        $1 == 3 && $2 ~ /^fio-/ {
            split($3, job, "-")
            if ($5 != 0) exit 1
            if (job[1] == "sequential_read" && $6 == size_kib && $7 > 0) {
                bw = $7 / (1024 * 1024)
                printf "40MiB,%s,%s,%s,%.6f,,%.6f\n", job[2], job[3], job[1], bw, bw
            } else if (job[1] == "sequential_write" && $47 == size_kib && $48 > 0) {
                bw = $48 / (1024 * 1024)
                printf "40MiB,%s,%s,%s,,%.6f,%.6f\n", job[2], job[3], job[1], bw, bw
            } else {
                exit 1
            }
            rows++
        }
        END { if (rows != expected_rows) exit 1 }
    ' "$matrix_log" >"$results_dir/partial.csv"; then
        die "Incomplete 40 MiB sweep. Inspect $matrix_log."
    fi
    mv -- "$results_dir/partial.csv" "$results_dir/summary.csv"
    cat -- "$results_dir/summary.csv"
    exit 0
fi

timed_options=(--time_based=1 --runtime=15s --ramp_time=3s --allow_file_create=0)

printf '\nSequential write (3 s warmup, 15 s measured)\n'
fio --name=seq-write "${fio_options[@]}" "${baseline_options[@]}" "${timed_options[@]}" --rw=write --overwrite=1 --end_fsync=1 \
    2>&1 | tee "$results_dir/write.log"

printf '\nSequential read (3 s warmup, 15 s measured)\n'
fio --readonly --name=seq-read "${fio_options[@]}" "${baseline_options[@]}" "${timed_options[@]}" --rw=read \
    2>&1 | tee "$results_dir/read.log"
