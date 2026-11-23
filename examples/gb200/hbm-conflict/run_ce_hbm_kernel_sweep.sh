#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
    cat <<'USAGE'
Measure GEMM-only and CE AllGather overlap from CUDA and NVTX traces.

Usage: run_ce_hbm_kernel_sweep.sh [options]
  --m-values LIST       Subset of powers of two from 1 through 8192
  --gpus LIST           Ordered GPU IDs (default: first four CUDA_VISIBLE_DEVICES or 0,1,2,3)
  --cases LIST          task1,task3,task4,task7,task8 (default: all; always includes task1)
  --gemm-rounds N       Measured GEMMs (default: 16)
  --comm-rounds N       Minimum measured AG rounds per attempt (default: 25)
  --warmup N            Warmup rounds (default: 5)
  --max-size-mib N      Maximum AG output MiB per GPU when retrying (default: 8192)
  --nsys PATH           Nsight Systems executable (also NSYS_BIN)
  --nccl-lib PATH       NCCL shared library for AG cases
  --mc-self-import 0|1  Self-import setting for multicast cases
  --output-dir DIR      Results directory
  --dry-run             Show the initial 1 GiB commands
  -h, --help

Each AG case starts at 1 GiB of output per GPU. The sweep increases the size
when the trace finds fewer than two GEMMs fully inside the same AG round.
Results include summary.json, report.md, raw traces, and per-attempt metadata.
USAGE
}

die() { printf 'error: %s\n' "$*" >&2; exit 2; }

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../../.." && pwd)"
nsys_bin="${NSYS_BIN:-NsightSystems-2026.6.1/bin/nsys}"
IFS=, read -r -a default_gpus <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
gpus="$(IFS=,; printf '%s' "${default_gpus[*]:0:4}")"
m_values=1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192
case_ids=task1,task3,task4,task7,task8
gemm_rounds=16 comm_rounds=25 warmup=5 max_size_mib=8192
output_dir="" nccl_lib="" mc_self_import="" dry_run=false

while (( $# )); do
    case "$1" in
        --m-values|--gpus|--cases|--gemm-rounds|--comm-rounds|--warmup|--max-size-mib|--nsys|--nccl-lib|--mc-self-import|--output-dir)
            (( $# >= 2 )) && [[ -n "$2" && "$2" != --* ]] || die "$1 requires a value"
            case "$1" in
                --m-values) m_values="$2" ;;
                --gpus) gpus="$2" ;;
                --cases) case_ids="$2" ;;
                --gemm-rounds) gemm_rounds="$2" ;;
                --comm-rounds) comm_rounds="$2" ;;
                --warmup) warmup="$2" ;;
                --max-size-mib) max_size_mib="$2" ;;
                --nsys) nsys_bin="$2" ;;
                --nccl-lib) nccl_lib="$2" ;;
                --mc-self-import) mc_self_import="$2" ;;
                --output-dir) output_dir="$2" ;;
            esac
            shift 2 ;;
        --dry-run) dry_run=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument $1" ;;
    esac
done

[[ "$m_values" =~ ^[1-9][0-9]{0,3}(,[1-9][0-9]{0,3})*$ ]] || die "use distinct powers of two from 1 to 8192"
IFS=, read -r -a requested_m <<< "$m_values"
selected_m=,
for m in "${requested_m[@]}"; do
    (( m <= 8192 && (m & (m - 1)) == 0 )) && [[ "$selected_m" != *",$m,"* ]] \
        || die "use distinct powers of two from 1 to 8192"
    selected_m+="$m,"
done
[[ "$case_ids" =~ ^task[13478](,task[13478])*$ ]] || die "use distinct case IDs from task1,task3,task4,task7,task8"
IFS=, read -r -a requested_cases <<< "$case_ids"
selected_cases=,
for case_id in "${requested_cases[@]}"; do
    [[ "$selected_cases" != *",$case_id,"* ]] || die "duplicate case $case_id"
    selected_cases+="$case_id,"
done
for value in "$gemm_rounds" "$comm_rounds" "$warmup" "$max_size_mib"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "expected a positive integer, got $value"
done
(( gemm_rounds >= 2 && max_size_mib >= 1024 )) || die "use at least 2 GEMM rounds and a size limit of 1024 MiB or more"
[[ -z "$mc_self_import" || "$mc_self_import" == 0 || "$mc_self_import" == 1 ]] || die "--mc-self-import must be 0 or 1"

nsys_bin="$(command -v "$nsys_bin" || printf '%s' "$nsys_bin")"
[[ "$nsys_bin" = /* ]] || nsys_bin="$PWD/$nsys_bin"
output_dir="${output_dir:-$repo_root/results/gb200/hbm-conflict/$(date +%Y%m%d-%H%M%S)-ce-kernel-sweep}"
[[ "$output_dir" = /* ]] || output_dir="$PWD/$output_dir"
if ! $dry_run; then
    [[ -x "$nsys_bin" ]] || die "cannot execute Nsight Systems at $nsys_bin"
    command -v uv >/dev/null || die "uv is required"
    mkdir -p -- "$(dirname -- "$output_dir")"
    mkdir -- "$output_dir" || die "use a new results directory"
fi
analysis=(uv run --active --no-sync python "$script_dir/analyse_ce_hbm_kernel_trace.py")

run_case() {
    local case_id="$1" m="$2" size_mib=1024
    local attempt_dir launch_exit analysis_exit
    while true; do
        printf -v attempt_dir '%s/m%04d/%s/size-%dmib' "$output_dir" "$m" "$case_id" "$size_mib"
        local -a command=(bash "$script_dir/run_ce_hbm_benchmark.sh" --kernel-trace-only
            --gpus "$gpus" --nsys "$nsys_bin" --cases "$case_id" --gemm "[$m,8192,7168]"
            --size-mib "$size_mib" --gemm-rounds "$gemm_rounds" --comm-rounds "$comm_rounds"
            --warmup "$warmup" --output-dir "$attempt_dir")
        if [[ "$case_id" != task1 && -n "$nccl_lib" ]]; then
            command+=(--nccl-lib "$nccl_lib")
        fi
        if [[ "$case_id" == task[34] && -n "$mc_self_import" ]]; then
            command+=(--mc-self-import "$mc_self_import")
        fi
        if $dry_run; then
            "${command[@]}" --dry-run
            return
        fi

        launch_exit=0
        # The shared launcher starts nsys profile and the torch.distributed worker.
        "${command[@]}" || launch_exit=$?
        analysis_exit=0
        "${analysis[@]}" attempt --output-dir "$attempt_dir" --case "$case_id" --m "$m" \
            --size-mib "$size_mib" --gemm-rounds "$gemm_rounds" --comm-rounds "$comm_rounds" \
            --nsys "$nsys_bin" --launch-exit-code "$launch_exit" || analysis_exit=$?
        if (( analysis_exit != 3 )); then
            return "$analysis_exit"
        fi
        if (( size_mib == max_size_mib )); then
            printf 'error: %s M=%s reached %s MiB without two fully covered GEMMs in one AG round\n' \
                "$case_id" "$m" "$size_mib" >&2
            return 1
        fi
        (( size_mib *= 2 ))
        if (( size_mib > max_size_mib )); then size_mib=$max_size_mib; fi
        printf '%s M=%s has insufficient overlap. Retrying with %s MiB per GPU\n' "$case_id" "$m" "$size_mib" >&2
    done
}

failed=0
for (( m=1; m<=8192; m*=2 )); do
    [[ "$selected_m" == *",$m,"* ]] || continue
    for case_id in task1 task3 task4 task7 task8; do
        [[ "$case_id" == task1 || "$selected_cases" == *",$case_id,"* ]] || continue
        run_case "$case_id" "$m" || failed=1
        if ! $dry_run; then "${analysis[@]}" report --output-dir "$output_dir"; fi
    done
done
if ! $dry_run; then printf 'Results directory: %s\n' "$output_dir"; fi
exit "$failed"
