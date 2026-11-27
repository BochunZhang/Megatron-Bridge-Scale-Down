#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

default_output=results/scale-down/01-analyse/02-deepspeed-offload/ulysses_overlap/ulysses-overlap

if [[ ${1:-} == -h || ${1:-} == --help ]]; then
    cat <<EOF
Usage: bash run_ulysses_overlap_nsys.sh [benchmark arguments]

Trace the four-GPU DeepSpeed training CUDA Graphs at node granularity. By
default this profiles 4K and 256K at both KV head counts. Extra arguments go
to train_ulysses_cuda_graph.py.

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
    --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none -o "$output_prefix" \
    python "$script_dir/train_ulysses_cuda_graph.py" --seq-lens 4096 262144 --iters 3 --nvtx "$@"
