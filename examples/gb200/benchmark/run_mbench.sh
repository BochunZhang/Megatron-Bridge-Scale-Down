#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Run one mbench scope in its own process, optionally under Nsight Systems.
#
#   run_mbench.sh --config CONFIG --scope {GEMM|GatedDeltaNet|GatedAttention|MLP|Experts|GemmSweep} \
#                 --output-dir DIR [--nsys] [--micro-batch-sizes 1,2] [--seq 16k] [--cases a,b] [--device N]
#
# --seq overrides benchmark.sequence_length (4k, 16k, 64k, 256k, ... or a token count);
# GemmSweep ignores --seq and --micro-batch-sizes.
# Results go to DIR/<category>/ (gemm, gated_delta_net, gated_attention, mlp, experts, gemm_sweep).
# With --nsys the trace is written to DIR/nsys/<category>.nsys-rep and exported
# to DIR/nsys/<category>.sqlite for mbench.build_csv.
set -euo pipefail

BENCHMARK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG=""
SCOPE=""
OUTPUT_DIR=""
USE_NSYS=0
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --scope) SCOPE="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --nsys) USE_NSYS=1; shift ;;
    --micro-batch-sizes|--seq|--cases|--device) EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ -z "$CONFIG" || -z "$SCOPE" || -z "$OUTPUT_DIR" ]]; then
  echo "usage: $0 --config CONFIG --scope SCOPE --output-dir DIR [--nsys] [--micro-batch-sizes LIST] [--seq LEN] [--cases LIST] [--device N]" >&2
  exit 2
fi
# Must match mbench.run.SCOPES.
case "$(echo "$SCOPE" | tr '[:upper:]' '[:lower:]')" in
  gemm) CATEGORY=gemm ;;
  gateddeltanet|gated_delta_net) CATEGORY=gated_delta_net ;;
  gatedattention|gated_attention) CATEGORY=gated_attention ;;
  mlp) CATEGORY=mlp ;;
  experts) CATEGORY=experts ;;
  gemmsweep|gemm_sweep) CATEGORY=gemm_sweep ;;
  *) echo "unknown scope: $SCOPE (expected GEMM, GatedDeltaNet, GatedAttention, MLP, Experts or GemmSweep)" >&2; exit 2 ;;
esac

export PYTHONPATH="$BENCHMARK_ROOT${PYTHONPATH:+:$PYTHONPATH}"
COMMAND=(uv run --no-sync python -m mbench.run --config "$CONFIG" --scope "$SCOPE" --output-dir "$OUTPUT_DIR" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})
mkdir -p "$OUTPUT_DIR"

if [[ "$USE_NSYS" -eq 0 ]]; then
  exec "${COMMAND[@]}"
fi

NSYS_DIR="$OUTPUT_DIR/nsys"
mkdir -p "$NSYS_DIR"
nsys profile \
  --trace=cuda,nvtx \
  --sample=none \
  --cpuctxsw=none \
  --cuda-memory-usage=false \
  --force-overwrite=true \
  --output="$NSYS_DIR/$CATEGORY" \
  "${COMMAND[@]}"
nsys export --type=sqlite --force-overwrite=true --output="$NSYS_DIR/$CATEGORY.sqlite" "$NSYS_DIR/$CATEGORY.nsys-rep"
