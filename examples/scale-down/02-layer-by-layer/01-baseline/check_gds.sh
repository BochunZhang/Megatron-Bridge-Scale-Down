#!/usr/bin/env bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
set -euo pipefail
export LC_ALL=C

nvme_dir="${NVME_PATH:-}"
gpu_list=0
tools_dir="${GDS_TOOLS_DIR:-${CUDA_HOME:-/usr/local/cuda}/gds/tools}"
size_mib=256
seconds=5

usage() {
    cat <<'HELP'
Usage: check_gds.sh --nvme DIR [--gpus 0,1,2,3]

Test cuFile GDS writes, then reads, on one mount. Each CUDA device uses its own
256 MiB temporary file. All selected devices run concurrently in each phase.

  --nvme DIR     Existing scratch directory (or set NVME_PATH)
  --gpus LIST    Comma-separated CUDA ordinals after device masking (default: 0)
  GDS_TOOLS_DIR  Directory containing gdscheck.py and gdsio
                 (default: ${CUDA_HOME:-/usr/local/cuda}/gds/tools)

The script prints the mount and device topology and retains the tool logs.
HELP
}

fail() { printf 'error: %s\n' "$*" >&2; exit 2; }

while (( $# )); do
    case "$1" in
        --nvme|--gpus)
            [[ -n "${2:-}" && "$2" != --* ]] || fail "Missing value for $1"
            case "$1" in
                --nvme) nvme_dir="$2";;
                --gpus) gpu_list="$2";;
            esac
            shift 2;;
        -h|--help) usage; exit 0;;
        *) fail "Unknown option $1";;
    esac
done

[[ -d "$nvme_dir" && -w "$nvme_dir" ]] || fail "Set --nvme to an existing writable directory"
nvme_dir="$(cd -- "$nvme_dir" && pwd -P)"
[[ "$gpu_list" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ ]] || fail "Invalid CUDA ordinal list: $gpu_list"
IFS=, read -r -a gpus <<< "$gpu_list"
seen=,
for gpu in "${gpus[@]}"; do
    [[ "$seen" != *",$gpu,"* ]] || fail "Repeated CUDA ordinal: $gpu"
    seen+="$gpu,"
done

for tool in gdscheck.py gdsio; do
    [[ -x "$tools_dir/$tool" ]] || fail "Missing $tools_dir/$tool. Set GDS_TOOLS_DIR to your CUDA GDS tools directory."
done
command -v findmnt >/dev/null || fail "findmnt is required on the Linux GPU worker"

findmnt -T "$nvme_dir" -o TARGET,SOURCE,FSTYPE,MAJ:MIN
source_device="$(findmnt -n --nofsroot -T "$nvme_dir" -o SOURCE)"
if [[ -b "$source_device" ]] && command -v lsblk >/dev/null; then
    lsblk -s -o NAME,TYPE,PKNAME,TRAN,MODEL,SERIAL -- "$source_device" || true
elif [[ "$source_device" == /dev/* ]]; then
    printf 'Look up %s on the host if the block device is hidden in this container.\n' "$source_device"
fi
if command -v nvme >/dev/null; then
    nvme list -v || true
fi
printf 'CUDA ordinals: %s (CUDA_VISIBLE_DEVICES=%s)\n' "$gpu_list" "${CUDA_VISIBLE_DEVICES-<unset>}"
if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=index,uuid,pci.bus_id --format=csv || true
    nvidia-smi topo -nvme || true
fi

log_dir="$(mktemp -d "${TMPDIR:-/tmp}/gds-check-logs.XXXXXX")"
log_dir="$(cd -- "$log_dir" && pwd -P)"
printf 'Logs: %s\n' "$log_dir"
scratch_dir="$(mktemp -d "$nvme_dir/gds-check.XXXXXX")"
pids=()
cleanup() {
    for index in "${!pids[@]}"; do
        kill "${pids[index]}" 2>/dev/null || true
        wait "${pids[index]}" 2>/dev/null || true
    done
    rm -rf -- "$scratch_dir"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

export CUFILE_ALLOW_COMPAT_MODE=false CUFILE_FORCE_COMPAT_MODE=false CUFILE_STATS_LEVEL=3
# The per-process log records the final counters, including I/O after any live snapshot.
export CUFILE_LOGGING_LEVEL=INFO
if ! CUFILE_LOGFILE_PATH="$log_dir/platform.cufile.log" "$tools_dir/gdscheck.py" -p \
    > "$log_dir/platform.txt" 2>&1; then
    printf 'Platform check returned an error. Continuing with the cuFile I/O test.\n' >&2
fi
cat "$log_dir/platform.txt"

check_cufile_stats() {
    local direction="$1" logfile="$2"
    [[ -s "$logfile" ]] || return 1
    awk -v direction="$direction:" '
        function check(row, section, fields, n, i, requests, posix, errors) {
            if (row !~ / Read:.* Write:/) return
            print row
            section = row
            sub("^.* " direction, "", section)
            sub(/[[:space:]]+(Write:|BufRegister:).*/, "", section)
            n = split(section, fields, /[[:space:]]+/)
            requests = posix = errors = -1
            for (i = 1; i <= n; i++) {
                if (fields[i] ~ /^n=[0-9]+$/) requests = substr(fields[i], 3) + 0
                if (fields[i] ~ /^posix=[0-9]+$/) posix = substr(fields[i], 7) + 0
                if (fields[i] ~ /^err=[0-9]+$/) errors = substr(fields[i], 5) + 0
            }
            if (posix > 0 || errors > 0 || (requests > 0 && (posix < 0 || errors < 0))) failed = 1
            if (requests > 0 && posix == 0 && errors == 0) confirmed = 1
        }
        /(^|[[:space:]])GPU[[:space:]]+[0-9]+([[:space:](]|$)/ {
            check(row)
            row = $0
            next
        }
        row && /^[[:space:]]*(Read:|Write:|BufRegister:|[[:alnum:]_]+=[0-9])/ {
            row = row " " $0
            next
        }
        { check(row); row = "" }
        END { check(row); exit !(confirmed && !failed) }
    ' "$logfile"
}

run_phase() {
    local direction="$1" io_type="$2" index gpu prefix status failed=0
    printf '\n%s phase (%s seconds, %s MiB per CUDA device)\n' "$direction" "$seconds" "$size_mib"
    for index in "${!gpus[@]}"; do
        gpu="${gpus[index]}"
        prefix="$log_dir/$direction-gpu$gpu"
        CUFILE_LOGFILE_PATH="$prefix.cufile.log" "$tools_dir/gdsio" \
            -f "$scratch_dir/gpu$gpu.bin" -d "$gpu" -w 1 -s "${size_mib}M" \
            -i 1M -x 0 -I "$io_type" -T "$seconds" > "$prefix.gdsio.txt" 2>&1 &
        pids[index]=$!
    done

    for index in "${!gpus[@]}"; do
        gpu="${gpus[index]}"
        prefix="$log_dir/$direction-gpu$gpu"
        status=0
        wait "${pids[index]}" || status=$?
        unset 'pids[index]'
        printf '\n%s CUDA device %s (gdsio exit %s)\n' "$direction" "$gpu" "$status"
        cat "$prefix.gdsio.txt"
        # With 1 MiB sequential operations, at least one full pass must complete.
        if ! awk -v minimum="$size_mib" '/(^|[[:space:]])ops:[[:space:]]/ {
            for (i = 1; i < NF; i++) if ($i == "ops:" && $(i+1) ~ /^[0-9]+$/ && $(i+1) >= minimum) complete = 1
        } END { exit !complete }' "$prefix.gdsio.txt"; then
            printf 'CUDA device %s did not report a complete %s pass.\n' "$gpu" "$direction" >&2
            failed=1
        fi
        if ! check_cufile_stats "$direction" "$prefix.cufile.log"; then
            printf 'Could not confirm %s without POSIX fallback or errors on CUDA device %s.\n' \
                "$direction" "$gpu" >&2
            failed=1
        fi
        if (( status != 0 )); then
            failed=1
        fi
    done
    return "$failed"
}

run_phase Write 1 || fail "GDS write check failed. See $log_dir"
run_phase Read 0 || fail "GDS read check failed. See $log_dir"
printf '\nPASS: cuFile completed writes and reads with no POSIX fallback on CUDA devices %s.\n' "$gpu_list"
