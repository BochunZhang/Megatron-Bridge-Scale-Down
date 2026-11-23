#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
    cat <<'EOF'
Run three copy-engine multicast AllGather cases on one GB200 host.
Lock the selected GPU clocks to TDP before launching the cases.

Usage: bash run_cpu_allgather.sh [options]
  --gpus IDS          Comma-separated GPU IDs in one multicast clique (default: 0,1,2,3)
  --cases IDS         Comma-separated case1,case2,case3 (default: all three)
  --shard-mib N       Input shard size per rank in MiB (default: 256)
  --warmup N          Warmup iterations (default: 3)
  --iters N           Measured iterations (default: 20)
  --profile           Capture one report per case (default, NSYS_BIN defaults to nsys)
  --no-profile        Run with CUDA event timing only
  --nccl-lib PATH     Preload a specific libnccl.so.2 for the benchmark
  --output-dir DIR    Output root (default: results/gb200/cpu-allgather/<timestamp>)
  --dry-run           Show commands without locking clocks or starting jobs
  -h, --help          Show this help

case1: GPU -> multicast GPU
case2: pinned CPU -> local GPU -> multicast GPU
case3: pinned CPU -> multicast GPU
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 2; }
show_command() { printf '%q ' "$@"; printf '\n'; }

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../../.." && pwd)"
gpu_ids=0,1,2,3
case_ids=case1,case2,case3
shard_mib=256 warmup=3 iters=20
output_dir="" nccl_lib="" profile=true dry_run=false
nsys_bin="${NSYS_BIN:-nsys}"

while (( $# )); do
    case "$1" in
        --gpus|--cases|--shard-mib|--warmup|--iters|--nccl-lib|--output-dir)
            (( $# >= 2 )) && [[ -n "$2" ]] || die "$1 requires a value"
            case "$1" in
                --gpus) gpu_ids="$2" ;;
                --cases) case_ids="$2" ;;
                --shard-mib) shard_mib="$2" ;;
                --warmup) warmup="$2" ;;
                --iters) iters="$2" ;;
                --nccl-lib) nccl_lib="$2" ;;
                --output-dir) output_dir="$2" ;;
            esac
            shift 2 ;;
        --profile) profile=true; shift ;;
        --no-profile) profile=false; shift ;;
        --dry-run) dry_run=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument $1" ;;
    esac
done

[[ $shard_mib =~ ^[1-9][0-9]*$ ]] || die "--shard-mib must be a positive integer"
[[ $iters =~ ^[1-9][0-9]*$ ]] || die "--iters must be a positive integer"
[[ $warmup =~ ^(0|[1-9][0-9]*)$ ]] || die "--warmup must be a nonnegative integer"
gpu_pattern='([0-9]+|GPU-[[:xdigit:]-]+)'
[[ $gpu_ids =~ ^${gpu_pattern}(,${gpu_pattern})*$ ]] || die "--gpus requires comma-separated GPU indices or UUIDs"
[[ $case_ids =~ ^case[123](,case[123])*$ ]] || die "--cases accepts case1,case2,case3"
IFS=, read -r -a gpus <<< "$gpu_ids"
IFS=, read -r -a cases <<< "$case_ids"
ranks="${#gpus[@]}"
(( ranks >= 3 && ranks <= 255 )) || die "multicast AllGather needs 3–255 ranks in one clique"
for list in "$gpu_ids" "$case_ids"; do
    duplicates="$(tr ',' '\n' <<< "$list" | LC_ALL=C sort | uniq -d)"
    [[ -z $duplicates ]] || die "duplicate entries in $list"
done

output_dir="${output_dir:-$repo_root/results/gb200/cpu-allgather/$(date +%Y%m%d-%H%M%S)}"
[[ $output_dir == /* ]] || output_dir="$PWD/$output_dir"
if [[ -n $nccl_lib ]]; then
    [[ $nccl_lib == /* ]] || nccl_lib="$PWD/$nccl_lib"
    $dry_run || [[ -f $nccl_lib ]] || die "NCCL library not found at $nccl_lib"
fi
if $profile && ! $dry_run; then
    command -v "$nsys_bin" >/dev/null || die "Nsight Systems executable not found at $nsys_bin"
fi

ce_multicast_threshold=4611686018427387904
gloo_socket="${GLOO_SOCKET_IFNAME:-${GLOO_SOCKET_NAME:-lo}}"
nccl_socket="${NCCL_SOCKET_IFNAME:-=$gloo_socket}"

validate_nccl_path() {
    local case_dir="$1"
    local expected="NCCL INFO AllGather [Copy Engine]: $((shard_mib * 1048576)) Bytes -> cudaMemcpy; CE synchronization with NVLS"
    local threshold_log="NCCL_CE_COLL_AG_MULTICAST_THRESHOLD set by environment to $ce_multicast_threshold."
    local -a logs=("$case_dir"/nccl-raw/nccl-*.log)
    if [[ ! -f "${logs[0]}" ]]; then
        printf 'error: no NCCL logs in %s/nccl-raw\n' "$case_dir" >&2
        return 1
    fi
    # NCCL emits tuning decisions on rank 0. The NVLS sync flag and effective
    # threshold together identify the multicast data branch on this NCCL path.
    awk -v expected="$expected" -v threshold="$threshold_log" '
        index($0, threshold) { configured = 1 }
        /NCCL INFO AllGather( \[[^]]+\])?: [0-9]+ Bytes -> / {
            if (substr($0, index($0, "NCCL INFO AllGather")) == expected) {
                correct = 1
            } else {
                print "Unexpected NCCL AllGather path: " $0
                fallback = 1
            }
        }
        END {
            if (!configured) print "Missing NCCL environment evidence: " threshold
            if (!correct) print "Missing NCCL tuning evidence: " expected
            exit (!configured || !correct || fallback)
        }' "${logs[@]}" >&2
}

run_case() {
    local case_id="$1" case_dir="$output_dir/$1"
    local nccl_log_dir="$case_dir/nccl-raw"
    local -a worker=(uv run --active --no-sync python -m torch.distributed.run
        --standalone --nnodes=1 --nproc_per_node="$ranks" "$script_dir/cpu_allgather.py"
        --case "$case_id" --output-dir "$case_dir" --shard-mib "$shard_mib" --warmup "$warmup" --iters "$iters")
    if $profile; then
        worker=("$nsys_bin" profile --trace=cuda,nvtx
            --cuda-trace-all-apis=true --sample=none --cpuctxsw=none
            --capture-range=cudaProfilerApi --capture-range-end=stop --output="$case_dir/$case_id"
            "${worker[@]}" --profile)
    fi
    local -a command=(env "CUDA_VISIBLE_DEVICES=$gpu_ids" "CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
        "GLOO_SOCKET_IFNAME=$gloo_socket" "NCCL_SOCKET_IFNAME=$nccl_socket"
        NCCL_COMM_BLOCKING=1 NCCL_CUMEM_ENABLE=1 NCCL_CTA_POLICY=ZERO NCCL_WIN_ENABLE=1 NCCL_NVLS_ENABLE=1
        "NCCL_CE_COLL_AG_MULTICAST_THRESHOLD=$ce_multicast_threshold" NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING,ENV
        "NCCL_DEBUG_FILE=${nccl_log_dir//%/%%}/nccl-%h-%p.log")
    if [[ -n $nccl_lib ]]; then
        command+=("LD_PRELOAD=$nccl_lib${LD_PRELOAD:+:$LD_PRELOAD}" "CE_HBM_EXPECTED_NCCL_LIB=$nccl_lib")
    fi
    if [[ -n ${NCCL_CE_MC_SELF_IMPORT:-} ]]; then
        command+=("NCCL_CE_MC_SELF_IMPORT=$NCCL_CE_MC_SELF_IMPORT")
    fi
    command+=("${worker[@]}")
    if $dry_run; then
        show_command "${command[@]}"
        return
    fi

    mkdir -- "$case_dir" && mkdir -- "$nccl_log_dir" || return 1
    show_command "${command[@]}" >"$case_dir/command.txt" || return 1
    printf 'Running %s on GPUs %s\n' "$case_id" "$gpu_ids"
    "${command[@]}" 2>&1 | tee "$case_dir/torchrun.log" || return 1
    if $profile && [[ ! -s "$case_dir/$case_id.nsys-rep" ]]; then
        printf 'error: Nsight report missing at %s/%s.nsys-rep\n' "$case_dir" "$case_id" >&2
        return 1
    fi
    validate_nccl_path "$case_dir" || return 1
    printf 'Verified NCCL multicast path for %s\n' "$case_id"
}

lock_command=(nvidia-smi -i "$gpu_ids" --lock-gpu-clocks=tdp,tdp)
if $dry_run; then
    show_command "${lock_command[@]}"
else
    command -v nvidia-smi >/dev/null || die "nvidia-smi is required to lock GPU clocks"
    command -v uv >/dev/null || die "uv is required to launch the benchmark"
    mkdir -p -- "$output_dir"
    printf 'Locking GPU clocks to TDP on GPUs %s\n' "$gpu_ids"
    "${lock_command[@]}" 2>&1 | tee "$output_dir/gpu_clock_lock.log" \
        || die "TDP clock lock failed, see $output_dir/gpu_clock_lock.log"
fi
status=0
for case_id in "${cases[@]}"; do
    if ! run_case "$case_id"; then
        printf 'error: %s failed, see %s/%s\n' "$case_id" "$output_dir" "$case_id" >&2
        status=1
    fi
done
exit "$status"
