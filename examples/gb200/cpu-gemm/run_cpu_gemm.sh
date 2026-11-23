#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash run_cpu_gemm.sh [options]
  --m N --n N --k N   GEMM dimensions (default: 4096, 2048, 8192)
  --batches LIST     Comma-separated batch sizes (default: 1,2,4,8,16)
  --gpu N            Physical GPU index from nvidia-smi (default: 0)
  --numa-node N      Bind to a Grace CPU NUMA node (default: auto; none uses current policy)
  --warmup N         Warmup GEMMs per case (default: 16)
  --iters N          Profiled GEMMs per case (default: 100)
  --slots N          Rotate this many sets of buffers (default: 8)
  --frequency HZ     Nsight Systems GPU Metrics frequency (default: 50000)
  --metrics-set SET  Choose a metric set explicitly (default: bundled, then gb10x-ct if rejected)
  --output-dir DIR   Result directory (default: results/gb200/cpu-gemm/<timestamp>)
  --dry-run          Show commands without running them
  -h, --help         Show this help

A is [batch,M,K], B is [K,N], and C is [batch,M,N]. All matrices use BF16.
case1 keeps A/B/C on GPU. case2 reads B directly from CPU memory. case3
writes C directly to CPU memory. The selected GPU is locked to tdp,tdp.
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 2; }
show_command() { printf '%q ' "$@"; printf '\n'; }
run() {
    if $dry_run; then
        show_command "$@"
    else
        "$@"
    fi
}
is_uint() { [[ $1 =~ ^(0|[1-9][0-9]{0,9})$ ]] && (( $1 <= 2147483647 )); }

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../../.." && pwd)"
m=4096 n=2048 k=8192 batches=1,2,4,8,16 gpu=0 numa_node=auto
warmup=16 iters=100 slots=8 frequency=50000 output_dir="" dry_run=false
metrics_set="file:$script_dir/gb200-gemm-metrics.config"
retry_with_stock=true
nvcc_bin="${NVCC:-nvcc}" nsys_bin="${NSYS_BIN:-nsys}" python_bin="${PYTHON:-python3}"

while (( $# )); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --dry-run) dry_run=true; shift ;;
        --m|--n|--k|--batches|--gpu|--numa-node|--warmup|--iters|--slots|--frequency|--metrics-set|--output-dir)
            [[ -n ${2:-} && $2 != --* ]] || die "$1 requires a value"
            case "$1" in
                --m) m="$2" ;;
                --n) n="$2" ;;
                --k) k="$2" ;;
                --batches) batches="$2" ;;
                --gpu) gpu="$2" ;;
                --numa-node) numa_node="$2" ;;
                --warmup) warmup="$2" ;;
                --iters) iters="$2" ;;
                --slots) slots="$2" ;;
                --frequency) frequency="$2" ;;
                --metrics-set) metrics_set="$2"; retry_with_stock=false ;;
                --output-dir) output_dir="$2" ;;
            esac
            shift 2 ;;
        *) die "unknown argument: $1" ;;
    esac
done

for field in m n k iters slots; do
    is_uint "${!field}" && (( ${!field} > 0 )) || die "--$field must be a positive 32-bit integer"
done
is_uint "$warmup" || die '--warmup must be a nonnegative 32-bit integer'
is_uint "$gpu" || die '--gpu must be a physical GPU index'
is_uint "$frequency" && (( frequency >= 10 && frequency <= 200000 )) || die '--frequency must be 10–200000 Hz'
[[ $numa_node == auto || $numa_node == none ]] || is_uint "$numa_node" || die 'invalid --numa-node'
[[ $batches =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]] || die '--batches requires comma-separated positive integers'
IFS=, read -r -a batch_list <<< "$batches"
seen=,
for batch in "${batch_list[@]}"; do
    is_uint "$batch" && (( m * batch <= 2147483647 )) || die "M*batch exceeds cuBLAS GEMM range at batch $batch"
    [[ $seen != *",$batch,"* ]] || die "duplicate batch size: $batch"
    seen+="$batch,"
done

output_dir="${output_dir:-$repo_root/results/gb200/cpu-gemm/$(date +%Y%m%d-%H%M%S)}"
[[ $output_dir = /* ]] || output_dir="$PWD/$output_dir"
binary="$output_dir/cpu_gemm"
build=("$nvcc_bin" -O3 -std=c++17 -arch=sm_100 "$script_dir/cpu_gemm.cu" -lcublas -o "$binary")
lock=(nvidia-smi -i "$gpu" --lock-gpu-clocks=tdp,tdp)
gpu_uuid='<GPU UUID>'

if $dry_run; then
    show_command "${build[@]}"
    show_command "${lock[@]}"
    [[ $numa_node != auto ]] || numa_node='<local Grace NUMA node>'
else
    for tool in "$nvcc_bin" "$nsys_bin" "$python_bin" nvidia-smi; do
        command -v "$tool" >/dev/null || die "required tool not found: $tool"
    done
    if [[ $metrics_set == file:* ]]; then
        [[ -r ${metrics_set#file:} ]] || die "metric config is unreadable: ${metrics_set#file:}"
    fi
    gpu_uuid="$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)"
    [[ $gpu_uuid =~ ^GPU-[[:xdigit:]-]+$ ]] || die "could not resolve GPU UUID for index $gpu"
    if [[ $numa_node == auto ]]; then
        numa_node="$(nvidia-smi topo -C -i "$gpu" | awk '$NF ~ /^[0-9]+$/ {print $NF}')"
        is_uint "$numa_node" || die 'could not resolve local Grace NUMA node; use --numa-node N'
    fi
    if [[ $numa_node != none ]]; then
        command -v numactl >/dev/null || die 'numactl is required for NUMA binding'
        numactl --cpunodebind="$numa_node" --membind="$numa_node" true
    fi
    mkdir -p -- "$(dirname -- "$output_dir")"
    mkdir -- "$output_dir" || die "use a new output directory: $output_dir"
    {
        "$nsys_bin" --version
        "$nvcc_bin" --version
        printf 'm=%s n=%s k=%s batches=%s gpu=%s uuid=%s numa_node=%s warmup=%s iters=%s slots=%s frequency_hz=%s metrics_set=%s\n' \
            "$m" "$n" "$k" "$batches" "$gpu" "$gpu_uuid" "$numa_node" "$warmup" "$iters" "$slots" "$frequency" "$metrics_set"
        show_command "${build[@]}"
    } > "$output_dir/config.txt"
    "${build[@]}" 2>&1 | tee "$output_dir/build.log"
    "${lock[@]}" 2>&1 | tee "$output_dir/gpu_clock_lock.log"
    nvidia-smi -i "$gpu" --query-gpu=index,name,uuid,clocks.current.sm,clocks.current.memory,power.limit \
        --format=csv > "$output_dir/gpu.csv"
fi

binding=()
if [[ $numa_node != none ]]; then
    binding=(numactl "--cpunodebind=$numa_node" "--membind=$numa_node")
fi

profile_case() {
    local case_dir="$1" case_id="$2" batch="$3"
    local -a cmd=("$nsys_bin" profile --trace=cuda --sample=none --cpuctxsw=none
        --capture-range=cudaProfilerApi --capture-range-end=stop
        "--gpu-metrics-devices=$gpu" "--gpu-metrics-set=$metrics_set" "--gpu-metrics-frequency=$frequency"
        "--output=$case_dir/profile" env "CUDA_VISIBLE_DEVICES=$gpu_uuid"
        ${binding[@]+"${binding[@]}"} "$binary" "$case_id" "$m" "$n" "$k" "$batch" "$warmup" "$iters" "$slots")
    if $dry_run; then
        show_command "${cmd[@]}"
    else
        show_command "${cmd[@]}" > "$case_dir/command.txt" || return
        "${cmd[@]}" 2>&1 | tee "$case_dir/profile.log"
    fi
}

metric_set_rejected() {
    grep -Fq -e 'Illegal --gpu-metrics-set argument' -e "Metric set alias doesn't match the config file name" "$1"
}

status=0
for batch in "${batch_list[@]}"; do
    for case_id in 1 2 3; do
        case_dir="$output_dir/case$case_id-b$batch"
        if ! $dry_run; then mkdir -- "$case_dir"; fi
        profiled=false
        if profile_case "$case_dir" "$case_id" "$batch"; then
            profiled=true
            retry_with_stock=false
        elif $retry_with_stock && [[ ! -e $case_dir/profile.nsys-rep ]] && metric_set_rejected "$case_dir/profile.log"; then
            mv -- "$case_dir/profile.log" "$case_dir/profile-custom.log"
            mv -- "$case_dir/command.txt" "$case_dir/command-custom.txt"
            metrics_set=gb10x-ct
            retry_with_stock=false
            printf 'metrics_set_fallback=%s\n' "$metrics_set" >> "$output_dir/config.txt"
            printf 'nsys rejected the bundled metric set. Retrying with gb10x-ct (DRAM active cycles %%, C2C GB/s).\n' >&2
            if profile_case "$case_dir" "$case_id" "$batch"; then profiled=true; fi
        fi
        if ! $dry_run; then
            if ! $profiled && metric_set_rejected "$case_dir/profile.log"; then
                "$nsys_bin" profile "--gpu-metrics-devices=$gpu" --gpu-metrics-set=help \
                    > "$output_dir/nsys-gpu-metrics.txt" 2>&1 || true
                die "nsys rejected $metrics_set. See $case_dir/profile.log and $output_dir/nsys-gpu-metrics.txt"
            fi
            if ! $profiled ||
                ! grep -Fq 'validation=passed' "$case_dir/profile.log" ||
                [[ ! -s $case_dir/profile.nsys-rep ]]; then
                printf 'error: capture or validation failed, see %s/profile.log\n' "$case_dir" >&2
                status=1
                continue
            fi
        fi
        if ! run "$nsys_bin" export --type=sqlite "--output=$case_dir/profile.sqlite" "$case_dir/profile.nsys-rep"; then
            status=1
        fi
    done
done

run "$python_bin" "$script_dir/analyse_nsys.py" "$output_dir" --gpu "$gpu" --frequency "$frequency" --iters "$iters" || status=1
printf 'Results directory: %s\n' "$output_dir"
exit "$status"
