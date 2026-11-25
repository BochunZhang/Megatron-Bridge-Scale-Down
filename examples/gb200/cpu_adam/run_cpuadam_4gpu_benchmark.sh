#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
benchmark="${script_dir}/cpuadam_4gpu_benchmark.py"
threads=1,2,4,8,16,32
modes=thread,process
param_mib=320
warmup=5
steps=20
results_dir=""
dry_run=false

usage() {
    cat <<EOF
Usage: bash $0 [options]
  --threads LIST     Comma-separated CPUAdam thread counts per rank (default: 1,2,4,8,16,32)
  --modes LIST       Comma-separated CPUAdam caller modes (default: thread,process)
  --param-mib COUNT  FP32 parameter storage in MiB per rank (default: 320)
  --warmup COUNT     Warmup updates (default: 5)
  --steps COUNT      Timed updates (default: 20)
  --results-dir DIR  New output directory (alias: --output-dir)
  --dry-run          Show the commands without launching or creating results
  -h, --help         Show this help

PYTHON_BIN can select the same Python executable used for training.
EOF
}

die() {
    printf '%s\n' "$*" >&2
    exit 1
}

while (($#)); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --dry-run) dry_run=true; shift; continue ;;
        --threads|--modes|--param-mib|--warmup|--steps|--results-dir|--output-dir)
            (($# >= 2)) && [[ -n "$2" && "$2" != --* ]] || die "Missing value for $1"
            ;;
        *) die "Unknown option $1. Run with --help for usage." ;;
    esac
    case "$1" in
        --threads) threads="$2" ;;
        --modes) modes="$2" ;;
        --param-mib) param_mib="$2" ;;
        --warmup) warmup="$2" ;;
        --steps) steps="$2" ;;
        --results-dir|--output-dir) results_dir="$2" ;;
    esac
    shift 2
done

[[ "${threads}" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]] || die "--threads must be a comma-separated list of positive integers."
[[ "${modes}" =~ ^(thread|process)(,(thread|process))*$ ]] || die "--modes accepts thread and process."
[[ "${param_mib}" =~ ^[1-9][0-9]*$ ]] || die "--param-mib must be a positive integer."
[[ "${warmup}" =~ ^[1-9][0-9]*$ ]] || die "--warmup must be a positive integer."
[[ "${steps}" =~ ^[1-9][0-9]*$ ]] || die "--steps must be a positive integer."

IFS=, read -r -a thread_counts <<< "${threads}"
IFS=, read -r -a optimizer_modes <<< "${modes}"
for option in "${threads}" "${modes}"; do
    IFS=, read -r -a values <<< "${option}"
    seen=,
    for value in "${values[@]}"; do
        [[ "${seen}" != *",${value},"* ]] || die "Duplicate sweep value: ${value}."
        seen+="${value},"
    done
done

python_bin="${PYTHON_BIN:-${PYTHON:-}}"
python=(uv run --project "${repo_root}" --no-sync python)
[[ -z "${python_bin}" ]] || python=("${python_bin}")
numa=(numarun)
if ! command -v numarun >/dev/null; then
    if command -v numactl >/dev/null && [[ -r "${repo_root}/.cache/numarun" ]]; then
        numa=(bash "${repo_root}/.cache/numarun")
    elif [[ "${dry_run}" == false ]]; then
        die "numarun is required to match the training rank CPU affinity."
    fi
fi

results_dir="${results_dir:-${repo_root}/results/gb200/cpu-adam-4gpu/cpuadam-$(date +%Y%m%d-%H%M%S).$$}"
[[ "${results_dir}" == /* ]] || results_dir="${PWD}/${results_dir}"
if [[ "${dry_run}" == false ]]; then
    [[ -r "${benchmark}" ]] || die "Benchmark driver is missing: ${benchmark}"
    command -v "${python[0]}" >/dev/null || die "Executable not found: ${python[0]}"
    mkdir -p -- "$(dirname -- "${results_dir}")"
    mkdir -- "${results_dir}" || die "Use a new results directory."
fi

printf 'Results directory %s\n' "${results_dir}"
for count in "${thread_counts[@]}"; do
    for mode in "${optimizer_modes[@]}"; do
        output="${results_dir}/${mode}-t${count}"
        command=("${python[@]}" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4
            --no-python "${numa[@]}" env "OMP_NUM_THREADS=${count}" OMP_DYNAMIC=FALSE
            OMP_PROC_BIND=FALSE "OMP_THREAD_LIMIT=${count}" "${python[@]}" "${benchmark}"
            --mode "${mode}" --threads "${count}" --param-mib "${param_mib}" --warmup "${warmup}"
            --steps "${steps}" --out "${output}.json")
        if [[ "${dry_run}" == true ]]; then
            printf '%q ' "${command[@]}"
            printf '\n'
            continue
        fi
        printf 'Running %s mode with %s CPU threads per rank\n' "${mode}" "${count}"
        if ! "${command[@]}" 2>&1 | tee "${output}.log"; then
            die "CPUAdam benchmark failed. See ${output}.log"
        fi
        [[ -s "${output}.json" ]] || die "Missing benchmark result: ${output}.json"
    done
done

summary=("${python[@]}" "${benchmark}" --summarize "${results_dir}")
if [[ "${dry_run}" == true ]]; then
    printf '%q ' "${summary[@]}"
    printf '\n'
else
    "${summary[@]}"
fi
