#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Runs gb_benchmark on one GB200 compute tray (2 Grace CPUs + 4 B200 GPUs).
#   --suite relay (default): CPU-relayed GPU->GPU copies (GPU0->GPU1, GPU0->GPU2), their read/write legs,
#                            and the direct GPU->GPU baselines.
#   --suite full:            relay + the gh_benchmark-style matrices (host, device, memcpy, latency,
#                            ping-pong, thread/block scaling). Takes much longer.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
BIN="${GB_BENCHMARK_BIN:-$SCRIPT_DIR/build/gb_benchmark}"
SUITE=relay
BYTES=4G
ITERS=10
WARMUP=2
SRC_GPU=0
DST_GPUS=1,2
OUT_ROOT="$REPO_ROOT/results/gb200/intra-super-node"
EXTRA=()

usage() {
  cat <<EOF
usage: $0 [--suite relay|full] [--bytes 4G] [--iters 10] [--warmup 2] [--src-gpu 0] [--dst-gpus 1,2]
          [--out-root DIR] [-- extra gb_benchmark options, e.g. --hbm-backend managed --hbm-numa 0=2,1=10]
Binary: \$GB_BENCHMARK_BIN or $SCRIPT_DIR/build/gb_benchmark
EOF
}

while (($#)); do
  case "$1" in
    --suite) SUITE=$2; shift 2 ;;
    --bytes) BYTES=$2; shift 2 ;;
    --iters) ITERS=$2; shift 2 ;;
    --warmup) WARMUP=$2; shift 2 ;;
    --src-gpu) SRC_GPU=$2; shift 2 ;;
    --dst-gpus) DST_GPUS=$2; shift 2 ;;
    --out-root) OUT_ROOT=$2; shift 2 ;;
    --) shift; EXTRA=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ "$SUITE" == relay || "$SUITE" == full ]] || { echo "--suite must be relay or full" >&2; exit 2; }
[[ -x "$BIN" ]] || { echo "missing $BIN: run 'make' on the GB200 node or build_cross_in_docker.sh" >&2; exit 1; }
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "[warn] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES: gpu<g> indices are relative to it" >&2
fi

RUN_DIR="$OUT_ROOT/$(date +%y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"
COMMON=(--bytes "$BYTES" --iters "$ITERS" --warmup "$WARMUP")
FAILED=()

# run <case dir> <gb_benchmark args...>: result.json + benchmark.log per case; failures do not stop the suite.
run() {
  local dir="$RUN_DIR/$1"
  shift
  mkdir -p "$dir"
  echo "=== ${dir#"$RUN_DIR"/}: gb_benchmark $*"
  set +e
  "$BIN" "$@" ${EXTRA[@]+"${EXTRA[@]}"} --out "$dir/result.json" 2>&1 | tee "$dir/benchmark.log" | grep -v '^\[topo\]'
  local rc=${PIPESTATUS[0]}
  set -e
  if ((rc != 0)); then
    FAILED+=("${dir#"$RUN_DIR"/}")
  fi
}

{
  echo "# $(date -Iseconds) $(hostname)"
  nvidia-smi -L
  nvidia-smi topo -m
  numactl -H
  lscpu
} >"$RUN_DIR/env.txt" 2>&1 || true
"$BIN" topo ${EXTRA[@]+"${EXTRA[@]}"} --out "$RUN_DIR/topology.json" | tee "$RUN_DIR/topology.log"

KERNELS=ldp_stp,ldnp_stnp,memcpy
S=$SRC_GPU
run "legs/read_hbm$S" host --op read --src "hbm$S" --cpu-nodes all --kernels "$KERNELS" "${COMMON[@]}"
for D in ${DST_GPUS//,/ }; do
  P="gpu${S}_to_gpu${D}"
  # The test itself: GPU S HBM -> CPU registers -> GPU D HBM, copy threads on the CPU of either GPU.
  run "$P/relay" relay --src-gpu "$S" --dst-gpus "$D" --cpu-nodes src,dst --threads all --kernels "$KERNELS" "${COMMON[@]}"
  run "legs/write_hbm$D" host --op write --dst "hbm$D" --cpu-nodes all --kernels "$KERNELS" "${COMMON[@]}"
  # Direct GPU->GPU references (NVLink): copy engine and SM copy.
  run "$P/baseline_memcpy" memcpy --gpu "$S" --src "cuda$S" --dst "cuda$D" "${COMMON[@]}"
  run "$P/baseline_sm_copy" device --gpu "$S" --op copy --src "cuda$S" --dst "cuda$D" "${COMMON[@]}"
done

if [[ "$SUITE" == full ]]; then
  run scaling/relay_threads relay --src-gpu "$S" --dst-gpus "$DST_GPUS" --cpu-nodes src,dst \
    --threads 1,2,4,8,16,24,32,48,64,all "${COMMON[@]}"
  run scaling/device_blocks device --gpu "$S" --op read --src "hbm$S,cpu0,cuda$S" \
    --blocks 1,2,4,8,16,32,64,128,auto "${COMMON[@]}"
  run host/read host --op read --src 'cpu*,hbm*' --cpu-nodes all --threads 1,all "${COMMON[@]}"
  run host/write host --op write --dst 'cpu*,hbm*' --cpu-nodes all --threads 1,all "${COMMON[@]}"
  run host/copy host --op copy --src 'cpu*,hbm*' --dst 'cpu*,hbm*' --cpu-nodes all "${COMMON[@]}"
  run device/read device --gpu "$S" --op read --src 'cpu*,hbm*,cuda*' "${COMMON[@]}"
  run device/write device --gpu "$S" --op write --dst 'cpu*,hbm*,cuda*' "${COMMON[@]}"
  run device/copy device --gpu "$S" --op copy --src "cpu0,hbm$S,cuda$S" --dst 'cpu*,hbm*,cuda*' "${COMMON[@]}"
  run memcpy/heatmap memcpy --gpu "$S" --src 'cuda*,pinned*,cpu*,hbm*' --dst 'cuda*,pinned*,cpu*,hbm*' \
    "${COMMON[@]}" --bytes 1G
  run latency latency --agents "cpu0,cpu1,gpu$S,gpu2" --targets 'cpu*,hbm*,cuda*'
  run pingpong pingpong --flags 'cpu*,hbm*'
fi

grep -h -E '^\[(relay|host|device|memcpy|latency|pingpong|skip|warn|error)\]' \
  "$RUN_DIR"/topology.log $(find "$RUN_DIR" -name benchmark.log | sort) >"$RUN_DIR/summary.txt" || true
echo "results: $RUN_DIR (summary.txt, */result.json)"
if ((${#FAILED[@]})); then
  echo "failed cases: ${FAILED[*]}" >&2
  exit 1
fi
