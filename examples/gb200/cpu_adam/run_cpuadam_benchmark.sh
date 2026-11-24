#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="${SCRIPT_DIR}/cpuadam_benchmark.py"
PYTHON_BIN="${PYTHON_BIN:-python}"
CPU_NODE=0
THREADS=1,2,4,8,16,32
PARAMS=268435456
DURATION=30
WARMUP=5
OUTPUT_ROOT="${PWD}"

usage() {
    cat <<EOF
Usage: bash $0 [options]
  --cpu-node NODE     NUMA node for CPU and memory binding (default: 0)
  --threads LIST      Comma-separated thread counts (default: 1,2,4,8,16,32)
  --params COUNT      FP32 elements in one tensor (default: 268435456, or 1 GiB of weights)
  --duration SEC      Time window including warmup (default: 30)
  --warmup SEC        Leading time to exclude from samples (default: 5)
  --output-root DIR   Parent for a new results directory (default: current directory)
  -h, --help          Show this help

Requires numactl, cpuadam_benchmark.py alongside this script, and a Python environment with PyTorch and DeepSpeed.
PYTHON_BIN can override the Python executable. Set thread counts to fit the allocated cores on the node.
With the default parameter count, weights, gradients and both Adam states occupy 4 GiB in total.
EOF
}

die() {
    printf '%s\n' "$*" >&2
    exit 1
}

while (($#)); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --cpu-node|--threads|--params|--duration|--warmup|--output-root)
            (($# >= 2)) && [[ -n "$2" && "$2" != --* ]] || die "Missing value for $1"
            ;;
        *) die "Unknown option $1. Run with --help for usage." ;;
    esac
    case "$1" in
        --cpu-node) CPU_NODE="$2" ;;
        --threads) THREADS="$2" ;;
        --params) PARAMS="$2" ;;
        --duration) DURATION="$2" ;;
        --warmup) WARMUP="$2" ;;
        --output-root) OUTPUT_ROOT="$2" ;;
    esac
    shift 2
done

[[ "${CPU_NODE}" =~ ^[0-9]+$ ]] || die "--cpu-node must be a nonnegative integer."
[[ "${PARAMS}" =~ ^[1-9][0-9]*$ ]] || die "--params must be a positive integer."
[[ "${THREADS}" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]] || die "--threads must be a comma-separated list of positive integers."
IFS=, read -r -a thread_counts <<< "${THREADS}"
seen=,
for threads in "${thread_counts[@]}"; do
    [[ "${seen}" != *",${threads},"* ]] || die "Duplicate thread count ${threads}."
    seen+="${threads},"
done

[[ -r "${BENCH}" ]] || die "Keep cpuadam_benchmark.py alongside this script (${SCRIPT_DIR})."
command -v numactl >/dev/null || die "numactl is required for CPU and memory binding."
command -v "${PYTHON_BIN}" >/dev/null || die "Python executable not found (${PYTHON_BIN})."

mkdir -p -- "${OUTPUT_ROOT}"
run_dir="$(mktemp -d "${OUTPUT_ROOT}/cpuadam-$(date +%Y%m%d-%H%M%S).XXXXXX")"
printf 'Results directory %s\n' "${run_dir}"
for threads in "${thread_counts[@]}"; do
    printf 'Running %s threads on NUMA node %s\n' "${threads}" "${CPU_NODE}"
    log="${run_dir}/t${threads}.log"
    if ! OMP_NUM_THREADS="${threads}" OMP_DYNAMIC=FALSE OMP_PLACES=cores OMP_PROC_BIND=close \
        numactl --cpunodebind="${CPU_NODE}" --membind="${CPU_NODE}" \
        "${PYTHON_BIN}" "${BENCH}" --threads "${threads}" \
        --params "${PARAMS}" --duration "${DURATION}" --warmup "${WARMUP}" \
        --out "${run_dir}/t${threads}.json" >"${log}" 2>&1; then
        cat -- "${log}" >&2
        die "CPUAdam benchmark failed at ${threads} threads. Log at ${log}"
    fi
done

"${PYTHON_BIN}" - "${run_dir}" "${thread_counts[@]}" <<'PY' | tee "${run_dir}/summary.tsv"
import csv
import json
import sys
from pathlib import Path


run_dir = Path(sys.argv[1])
records = [json.loads((run_dir / f"t{threads}.json").read_text()) for threads in sys.argv[2:]]
baseline = next((record["step_s_median"] for record in records if record["threads"] == 1), None)
writer = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
writer.writerow(["threads", "iters", "median_ms", "params_per_s", "speedup_vs_1t", "effective_GB_per_s"])
for record in records:
    seconds = record["step_s_median"]
    threads = record["threads"]
    if seconds is None:
        writer.writerow([threads, 0, "NA", "NA", "NA", "NA"])
        continue
    writer.writerow([
        threads, record["iters"], f"{1000 * seconds:.3f}", f"{record['params_per_s']:.6g}",
        f"{baseline / seconds:.3f}" if baseline is not None else "NA", f"{record['gbps_median']:.3f}",
    ])

if any(record["iters"] == 0 for record in records):
    sys.exit("At least one thread count produced no timed steps. Increase --duration.")
PY
