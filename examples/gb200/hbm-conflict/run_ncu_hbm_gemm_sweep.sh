#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
    cat <<'USAGE'
Profile standalone BF16 GEMM with Nsight Compute on one GB200 GPU.

Usage: bash run_ncu_hbm_gemm_sweep.sh [options]
  --gpu N          Physical GPU ID in PCI bus order (default: 0)
  --m-values LIST   Comma-separated M values (default: 1,2,...,8192 by powers of two)
  --gemm-rounds N   Measured GEMMs per shape (default: 16)
  --warmup N        Warmup GEMMs per shape (default: 5)
  --ncu PATH        Nsight Compute executable (default: NCU_BIN or <repo>/NVIDIA-Nsight-Compute-2026.4/ncu)
  --output-dir DIR  Results directory (default: results/gb200/hbm-conflict/<time>-ncu-gemm)
  --dry-run         Show commands without profiling
  -h, --help

Writes a .ncu-rep and raw CSV per shape and summarizes traffic in summary.json.
USAGE
}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

show_command() {
    printf '%q ' "$@"
    printf '\n'
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../../.." && pwd)"
analyser="$script_dir/analyse_ncu_hbm_gemm.py"
gpu=0 m_values=1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192
gemm_rounds=16 warmup=5 ncu_bin="${NCU_BIN:-$repo_root/NVIDIA-Nsight-Compute-2026.4/ncu}" output_dir="" dry_run=false
pm_interval_ns=1000

while (( $# )); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --dry-run) dry_run=true; shift ;;
        --gpu|--m-values|--gemm-rounds|--warmup|--ncu|--output-dir)
            [[ -n "${2:-}" && "$2" != --* ]] || die "$1 needs a value"
            case "$1" in
                --gpu) gpu="$2" ;;
                --m-values) m_values="$2" ;;
                --gemm-rounds) gemm_rounds="$2" ;;
                --warmup) warmup="$2" ;;
                --ncu) ncu_bin="$2" ;;
                --output-dir) output_dir="$2" ;;
            esac
            shift 2 ;;
        *) die "unknown argument $1 (see --help)" ;;
    esac
done

[[ "$gpu" =~ ^(0|[1-9][0-9]*)$ && "$gemm_rounds" =~ ^[1-9][0-9]*$ && "$warmup" =~ ^[1-9][0-9]*$ ]] \
    || die "--gpu must be nonnegative and --gemm-rounds/--warmup must be positive"
[[ "$m_values" =~ ^[1-9][0-9]{0,3}(,[1-9][0-9]{0,3})*$ ]] || die "--m-values requires distinct integers in 1..8192"
IFS=, read -r -a shapes <<< "$m_values"
seen=','
for m in "${shapes[@]}"; do
    (( m <= 8192 )) && [[ "$seen" != *",$m,"* ]] || die "--m-values requires distinct integers in 1..8192"
    seen+="$m,"
done

output_dir="${output_dir:-$repo_root/results/gb200/hbm-conflict/$(date +%Y%m%d-%H%M%S)-ncu-gemm}"
[[ "$output_dir" = /* ]] || output_dir="$PWD/$output_dir"
[[ "$ncu_bin" != */* || "$ncu_bin" = /* ]] || ncu_bin="$PWD/$ncu_bin"
cd -- "$repo_root"

python=(uv run --active --no-sync python)
gpu_env=(env "CUDA_VISIBLE_DEVICES=$gpu" CUDA_DEVICE_ORDER=PCI_BUS_ID LC_ALL=C)
lock=(nvidia-smi -i "$gpu" --lock-gpu-clocks=tdp,tdp)
if $dry_run; then
    show_command "${lock[@]}"
    # Show the GB200 byte-counter candidates without starting a tool or Python environment.
    dry_metrics=(gpu__time_duration.sum dram__bytes_read.sum dram__bytes_write.sum
        pmsampling:nvlrx__bytes.sum pmsampling:nvltx__bytes.sum
        pmsampling:ctc__rx_bytes_data_user.sum pmsampling:ctc__tx_bytes_data_user.sum)
    metrics="$(IFS=,; printf '%s' "${dry_metrics[*]}")"
else
    command -v "$ncu_bin" >/dev/null || die "ncu not found at $ncu_bin; pass --ncu PATH"
    command -v uv >/dev/null || die "uv is required"
    mkdir -p -- "$(dirname -- "$output_dir")"
    mkdir -- "$output_dir" || die "use a new results directory"
    "${gpu_env[@]}" "$ncu_bin" --version >"$output_dir/ncu-version.txt"
    profiler_catalog="$output_dir/metrics-profiling.txt"
    pm_catalog="$output_dir/metrics-pmsampling.txt"
    query=("${gpu_env[@]}" "$ncu_bin" --query-metrics-mode all --devices 0 --query-metrics-collection)
    "${query[@]}" profiling >"$profiler_catalog" 2>&1 || die "NCU metric query failed; see $profiler_catalog"
    metric_args=(metrics --profiler-catalog "$profiler_catalog")
    if "${query[@]}" pmsampling >"$pm_catalog" 2>&1; then
        metric_args+=(--pm-catalog "$pm_catalog")
    else
        printf 'Warning: NCU PM sampling metric query failed. See %s\n' "$pm_catalog" >&2
    fi
    metrics="$("${python[@]}" "$analyser" "${metric_args[@]}")"
    "${lock[@]}" >"$output_dir/gpu_clock_lock.log" 2>&1 || die "GPU clock lock failed; see $output_dir/gpu_clock_lock.log"
fi

ncu_args=(--replay-mode application --cache-control none --clock-control none --target-processes all --nvtx)
for (( round = 0; round < gemm_rounds; round++ )); do
    printf -v nvtx_range 'gemm%02d/' "$round"
    ncu_args+=(--nvtx-include "$nvtx_range")
done
if [[ "$metrics" == *pmsampling:* ]]; then
    ncu_args+=(--pm-sampling-interval "$pm_interval_ns")
fi
ncu_args+=(--metrics "$metrics")

failed=0
for m in "${shapes[@]}"; do
    case_name="gemm-m$m-n8192-k7168"
    case_dir="$output_dir/$case_name"
    command=("${gpu_env[@]}" "$ncu_bin" "${ncu_args[@]}" --export "$case_dir/gemm" --
        "${python[@]}" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1
        "$script_dir/ce_hbm_benchmark.py" --mode gemm --ncu-gemm --case "$case_name"
        --gemm "[$m,8192,7168]" --warmup "$warmup" --gemm-rounds "$gemm_rounds" --comm-rounds 1
        --size-mib 1024 --output-dir "$case_dir")
    show_command "${command[@]}"
    if $dry_run; then continue; fi

    mkdir -- "$case_dir"
    show_command "${command[@]}" >"$case_dir/command.txt"
    summary=("${python[@]}" "$analyser" summarize --case-dir "$case_dir" --metrics "$metrics"
        --m "$m" --gpu "$gpu" --warmup "$warmup" --gemm-rounds "$gemm_rounds")
    if "${command[@]}" >"$case_dir/ncu.log" 2>&1; then
        "${gpu_env[@]}" "$ncu_bin" --import "$case_dir/gemm.ncu-rep" --page raw --csv --print-units base \
            --print-nvtx-rename kernel >"$case_dir/gemm-raw.csv" 2>>"$case_dir/ncu.log" \
            || summary+=(--error "NCU report import exited with status $?. See $case_dir/ncu.log")
    else
        summary+=(--error "NCU profiling exited with status $?. See $case_dir/ncu.log")
    fi
    "${summary[@]}" || failed=1
done

exit "$failed"
