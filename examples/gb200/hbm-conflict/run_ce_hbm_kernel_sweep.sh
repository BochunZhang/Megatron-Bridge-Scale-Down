#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
    cat <<'USAGE'
Profile 8 BF16 GEMMs alone or overlapped with stock NCCL copy-engine AllGather.
GPU metrics are sampled at 200 kHz on physical GPU 2 for every case.

Usage: run_ce_hbm_kernel_sweep.sh [options]
  --test                Run M=8 only
  --m-values LIST       Subset of powers of two from 1 through 8192
  --gpus LIST           Four ordered AllGather GPU IDs including 2 (default: 0,1,2,3)
  --cases LIST          task1,task4,task3,task7 (default: all; always includes task1)
  --warmup N            GEMM warmup rounds (default: 5)
  --nsys PATH           Nsight Systems 2026.6.1 executable (also NSYS_BIN)
  --nccl-lib PATH       Stock NCCL 2.32.3 or newer (default: PyTorch's loaded library)
  --output-dir DIR      Results directory
  --dry-run             Show the archive-derived plan and capture commands
  -h, --help

Each AllGather uses separate send and receive buffers. One completed lead-in
runs before collection. The measured AllGather size and rounds come from
archived timing estimates. Results contain raw reports, worker records, logs,
plan.tsv and status.tsv. Analyse the downloaded sweep with analyse_ce_hbm_overlap.py.
USAGE
}

die() { printf 'error: %s\n' "$*" >&2; exit 2; }

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../../.." && pwd)"
nsys_bin="${NSYS_BIN:-NsightSystems-2026.6.1/bin/nsys}"
gpus=0,1,2,3
metrics_gpu=2
m_values=1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192
case_ids=task1,task4,task3,task7
gemm_rounds=8 warmup=5
output_dir="" nccl_lib="" dry_run=false test_only=false

while (( $# )); do
    case "$1" in
        --m-values|--gpus|--cases|--warmup|--nsys|--nccl-lib|--output-dir)
            (( $# >= 2 )) && [[ -n "$2" && "$2" != --* ]] || die "$1 requires a value"
            case "$1" in
                --m-values) m_values="$2" ;;
                --gpus) gpus="$2" ;;
                --cases) case_ids="$2" ;;
                --warmup) warmup="$2" ;;
                --nsys) nsys_bin="$2" ;;
                --nccl-lib) nccl_lib="$2" ;;
                --output-dir) output_dir="$2" ;;
            esac
            shift 2 ;;
        --test) test_only=true; shift ;;
        --dry-run) dry_run=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument $1" ;;
    esac
done
if $test_only; then m_values=8; fi

[[ "$m_values" =~ ^[1-9][0-9]{0,3}(,[1-9][0-9]{0,3})*$ ]] || die "use distinct powers of two from 1 to 8192"
IFS=, read -r -a requested_m <<< "$m_values"
selected_m=,
for m in "${requested_m[@]}"; do
    (( m <= 8192 && (m & (m - 1)) == 0 )) && [[ "$selected_m" != *",$m,"* ]] \
        || die "use distinct powers of two from 1 to 8192"
    selected_m+="$m,"
done
[[ "$case_ids" =~ ^task[1347](,task[1347])*$ ]] || die "use distinct case IDs from task1,task4,task3,task7"
IFS=, read -r -a requested_cases <<< "$case_ids"
selected_cases=,
for case_id in "${requested_cases[@]}"; do
    [[ "$selected_cases" != *",$case_id,"* ]] || die "duplicate case $case_id"
    selected_cases+="$case_id,"
done
[[ "$warmup" =~ ^[1-9][0-9]*$ ]] || die "--warmup needs a positive integer"
[[ "$gpus" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*)){3}$ ]] || die "--gpus needs four physical GPU IDs"
IFS=, read -r -a requested_gpus <<< "$gpus"
selected_gpus=,
for gpu in "${requested_gpus[@]}"; do
    [[ "$selected_gpus" != *",$gpu,"* ]] || die "duplicate GPU ID $gpu"
    selected_gpus+="$gpu,"
done
[[ "$selected_gpus" == *",$metrics_gpu,"* ]] || die "--gpus must include physical GPU $metrics_gpu"

nsys_bin="$(command -v "$nsys_bin" || printf '%s' "$nsys_bin")"
[[ "$nsys_bin" = /* ]] || nsys_bin="$PWD/$nsys_bin"
output_dir="${output_dir:-$repo_root/results/gb200/hbm-conflict/$(date +%Y%m%d-%H%M%S)-ce-overlap-sweep}"
[[ "$output_dir" = /* ]] || output_dir="$PWD/$output_dir"
if ! $dry_run; then
    [[ -x "$nsys_bin" ]] || die "cannot execute Nsight Systems at $nsys_bin"
    command -v uv >/dev/null || die "uv is required"
    mkdir -p -- "$(dirname -- "$output_dir")"
    mkdir -- "$output_dir" || die "use a new results directory"
    printf '%s\n' 'Source: ce-kernel-sweep-20261123-222531/summary.json, 1024 MiB attempts, GPU 0' \
        'Planning uses p90 kernel, p90 within-round gap (solo p90 fallback), and p10 payload duration.' \
        "200 kHz metrics on physical GPU $metrics_gpu for every case." \
        'Analyse the downloaded sweep to verify coverage.' >"$output_dir/run_info.txt"
    printf 'm\tcase\tmetrics_gpus\tsize_mib\ttimed_ag_rounds\tgemm_p90_us\tgap_p90_us\tpayload_1gib_p10_us\n' \
        >"$output_dir/plan.tsv"
    printf 'm\tcase\tmetrics_gpus\tsize_mib\tstatus\tattempt_dir\n' >"$output_dir/status.tsv"
fi

archived_timing() {
    local m="$1" case_id="$2" shape cpu_multicast gpu_multicast gpu_unicast
    # ce-kernel-sweep-20261123-222531/summary.json, 1 GiB attempts, GPU 0.
    # Entries per path are p90 kernel duration / p90 gap between contained GEMMs
    # in the same round (or solo p90 gap) / p10 longest continuous payload.
    while IFS=' ' read -r shape cpu_multicast gpu_multicast gpu_unicast; do
        [[ "$shape" == "$m" ]] || continue
        case "$case_id" in
            task4) printf '%s\n' "$cpu_multicast" ;;
            task3) printf '%s\n' "$gpu_multicast" ;;
            task7) printf '%s\n' "$gpu_unicast" ;;
        esac
        return
    done <<'ESTIMATES'
1     30/50/1521  245/9/1486   30/26/1046
2     29/48/1526  251/9/1487   30/25/1043
4     29/46/1527  257/9/1486   31/52/1044
8     26/49/1533  200/8/1486   27/25/1042
16    26/46/1526  211/10/1486  28/48/1043
32    24/9/1527   248/9/1486   29/47/1042
64    29/45/1534  249/9/1485   29/28/1044
128   31/45/1530  257/9/1486   31/29/1044
256   40/46/1540  260/9/1486   40/58/1047
512   68/46/1532  304/9/1486   71/38/1043
1024  124/50/1525 358/9/1487   118/62/1044
2048  224/38/2321 471/10/1485  223/52/1044
4096  382/49/1526 617/9/1485   390/54/1047
8192  735/47/1523 980/9/1485   743/9/1047
ESTIMATES
    return 1
}

run_case() {
    local case_id="$1" m="$2" metrics="$3" devices="$4"
    local size_mib=1024 rounds=0 gemm_us=0 gap_us=0 payload_us=0 fits budget
    local guard_us=150 attempt_path case_path launch_exit=0 status=captured timing
    if [[ "$case_id" == task1 ]]; then
        case_path="$case_id/gpu$metrics"
    else
        case_path="$case_id"
        timing=$(archived_timing "$m" "$case_id") || return
        IFS=/ read -r gemm_us gap_us payload_us <<< "$timing"
        # Reserve the 100-us worker settle interval plus 50 us for the tail.
        # If 1 GiB leaves space for fewer than two GEMMs, try 2 GiB.
        if (( 2 * gemm_us + gap_us + guard_us > payload_us )); then size_mib=2048; fi
        budget=$((payload_us * (size_mib / 1024) - guard_us))
        fits=$(((budget + gap_us) / (gemm_us + gap_us)))
        if (( fits < 1 )); then fits=1; fi
        rounds=$(((gemm_rounds + fits - 1) / fits))
    fi

    printf -v attempt_path 'm%04d/%s/size-%dmib' "$m" "$case_path" "$size_mib"
    # The standalone worker requires a positive --comm-rounds argument and never enqueues CE.
    local -a command=(bash "$script_dir/run_ce_hbm_benchmark.sh" --overlap-profile
        --gpus "$devices" --metrics-gpus "$metrics" --nsys "$nsys_bin" --nsys-freq 200000
        --cases "$case_id" --gemm "[$m,8192,7168]" --size-mib "$size_mib" --gemm-rounds "$gemm_rounds"
        --comm-rounds "$((rounds == 0 ? 1 : rounds))" --warmup "$warmup" --output-dir "$output_dir/$attempt_path")
    if [[ "$case_id" != task1 && -n "$nccl_lib" ]]; then command+=(--nccl-lib "$nccl_lib"); fi
    printf 'M=%d %s metrics GPUs=%s size=%d MiB timed AG=%d\n' "$m" "$case_id" "$metrics" "$size_mib" "$rounds"
    if $dry_run; then
        "${command[@]}" --dry-run
        return
    fi
    printf '%d\t%s\t%s\t%d\t%d\t%d\t%d\t%d\n' \
        "$m" "$case_id" "$metrics" "$size_mib" "$rounds" "$gemm_us" "$gap_us" "$payload_us" \
        >>"$output_dir/plan.tsv" || return
    "${command[@]}" || launch_exit=$?
    if (( launch_exit )); then status=launch_error; fi
    printf '%d\t%s\t%s\t%d\t%s\t%s\n' "$m" "$case_id" "$metrics" "$size_mib" "$status" "$attempt_path" \
        >>"$output_dir/status.tsv" || return
    return "$launch_exit"
}

failed=0
for (( m=1; m<=8192; m*=2 )); do
    [[ "$selected_m" == *",$m,"* ]] || continue
    run_case task1 "$m" "$metrics_gpu" "$metrics_gpu" || failed=1
    for case_id in task4 task3 task7; do
        [[ "$selected_cases" == *",$case_id,"* ]] || continue
        run_case "$case_id" "$m" "$metrics_gpu" "$gpus" || failed=1
    done
done
if ! $dry_run; then printf 'Results directory: %s\n' "$output_dir"; fi
exit "$failed"
