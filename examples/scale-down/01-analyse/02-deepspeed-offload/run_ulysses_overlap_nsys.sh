#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

default_output=results/scale-down/01-analyse/02-deepspeed-offload/ulysses_overlap/ulysses-overlap

if [[ ${1:-} == -h || ${1:-} == --help ]]; then
    cat <<EOF
Usage: bash run_ulysses_overlap_nsys.sh [benchmark arguments]

Capture CUDA and NVTX in the active DeepSpeed CUDA environment. The benchmark
defaults to four GPUs, two warmup iterations and three measured iterations per
mode for this trace. Extra arguments go to benchmark_ulysses_overlap.py.

Set NSYS_BIN to select the Nsight Systems executable (default nsys).
Set NSYS_OUTPUT to override the report prefix. The default report path under
the repository root is ${default_output}.nsys-rep.
EOF
    exit 0
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../../.." && pwd)
output_prefix=${NSYS_OUTPUT:-"$repo_root/$default_output"}
mkdir -p -- "$(dirname -- "$output_prefix")"

exec uv run --no-project --active "${NSYS_BIN:-nsys}" profile \
    --trace=cuda,nvtx --sample=none --cpuctxsw=none -o "$output_prefix" \
    python "$script_dir/benchmark_ulysses_overlap.py" --warmup 2 --iters 3 --nvtx "$@"
