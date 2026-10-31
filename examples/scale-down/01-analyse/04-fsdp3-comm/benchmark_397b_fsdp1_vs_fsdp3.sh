#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

# Compare Megatron FSDP3 with FSDP1 on Qwen3.5-397B-A17B cropped to a few routed
# experts, at short sequences where FSDP and EP communication is not hidden.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ONE="${RUN_ONE:-${SCRIPT_DIR}/run_pretrain_fsdp3.sh}"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/01-analyse/04-fsdp3-comm}"
RUN_DATE="${RUN_DATE:-$(date +%y%m%d-%H%M%S)}"
RECIPE=qwen35_text_397b_a17b_pretrain_4gpu_gb200_bf16_fsdp1_config

EXPERTS_LIST=(64 32)
SEQUENCES=(256 1024 4096)
FSDP_LEVELS=(3 1)
EXPERT_PARALLEL=4
NUM_LAYERS="${NUM_LAYERS:-4}"
MTP_NUM_LAYERS="${MTP_NUM_LAYERS:-1}"
# Microbatches per iteration; GBS = nmb * MBS * DP with MBS=1, DP=GPUS (CP=1).
NUM_MICROBATCHES="${NUM_MICROBATCHES:-4}"
GPUS="${GPUS:-4}"
TRAIN_ITERS="${TRAIN_ITERS:-10}"
DISPATCHER="${DISPATCHER:-hybridep}"
PROFILE="${PROFILE:-nsys}"
PROFILE_START="${PROFILE_START:-7}"
PROFILE_END="${PROFILE_END:-8}"
MASTER_PORT="${MASTER_PORT:-29501}"
DRY_RUN=false
RESUME=false

usage() {
    cat <<'USAGE'
Usage: benchmark_397b_fsdp1_vs_fsdp3.sh [options]

Qwen3.5-397B-A17B MoE, 4 GPUs on one node, DP=4 and EP=4 (TP=PP=CP=1), BF16,
CUDA graph off, 4 layers (3 GatedDeltaNet + 1 gated attention, every layer MoE)
+ 1 MTP layer, MBS=1, 4 microbatches per iteration (GBS=16), Nsight Systems on
step 7. Default matrix: experts 64,32 x seq 256,1024,4096 x FSDP 3,1 = 12 runs.

Options:
  --experts N[,N...]           Routed experts, multiples of EP=4 and >= top-k (default: 64,32)
  --seq N[,N...]               Sequence lengths (default: 256,1024,4096)
  --fsdp 3|1[,...]             FSDP levels (default: 3,1)
  --num-layers N               Decoder layers (default: 4)
  --mtp-num-layers N           MTP layers (default: 1)
  --num-microbatches N         Microbatches per iteration (default: 4)
  --dispatcher hybridep|alltoall  MoE token dispatcher (default: hybridep)
  --train-iters N              Training iterations (default: 10)
  --profile none|nsys          Profiling mode (default: nsys)
  --profile-start N            First profiled iteration (default: 7)
  --profile-end N              Stop iteration, exclusive (default: 8)
  --master-port N              Rendezvous port (default: 29501)
  --run-date YYMMDD-HHMMSS     Shared output timestamp
  --output-dir PATH            Result root
  --resume                     Reuse --run-date: skip completed cases, rerun the others
  --dry-run                    Print cases and commands without running
USAGE
}

need_value() { [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; usage >&2; exit 2; }; }
parse_csv() { local value="$1"; IFS=',' read -r -a "$2" <<<"${value}"; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --experts|--num-experts) need_value "$@"; parse_csv "$2" EXPERTS_LIST; shift 2;;
        --seq|--seq-length|--seq-lengths) need_value "$@"; parse_csv "$2" SEQUENCES; shift 2;;
        --fsdp) need_value "$@"; parse_csv "$2" FSDP_LEVELS; shift 2;;
        --num-layers) need_value "$@"; NUM_LAYERS="$2"; shift 2;;
        --mtp-num-layers) need_value "$@"; MTP_NUM_LAYERS="$2"; shift 2;;
        --num-microbatches) need_value "$@"; NUM_MICROBATCHES="$2"; shift 2;;
        --dispatcher) need_value "$@"; DISPATCHER="$2"; shift 2;;
        --train-iters) need_value "$@"; TRAIN_ITERS="$2"; shift 2;;
        --profile) need_value "$@"; PROFILE="$2"; shift 2;;
        --profile-start) need_value "$@"; PROFILE_START="$2"; shift 2;;
        --profile-end) need_value "$@"; PROFILE_END="$2"; shift 2;;
        --master-port) need_value "$@"; MASTER_PORT="$2"; shift 2;;
        --run-date) need_value "$@"; RUN_DATE="$2"; shift 2;;
        --output-dir) need_value "$@"; OUTPUT_DIR="$2"; shift 2;;
        --resume) RESUME=true; shift;;
        --dry-run) DRY_RUN=true; shift;;
        -h|--help) usage; exit 0;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2;;
    esac
done

[[ "${RUN_DATE}" =~ ^[0-9]{6}-[0-9]{6}$ ]] || { echo "--run-date must use yymmdd-hhmmss" >&2; exit 2; }
case "${PROFILE}" in none|nsys) ;; *) echo "--profile must be none or nsys" >&2; exit 2;; esac
case "${DISPATCHER}" in hybridep|alltoall) ;; *) echo "--dispatcher must be hybridep or alltoall" >&2; exit 2;; esac
for name in NUM_LAYERS NUM_MICROBATCHES TRAIN_ITERS; do
    [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || { echo "${name} must be a positive integer" >&2; exit 2; }
done
[[ "${MTP_NUM_LAYERS}" =~ ^[0-9]+$ ]] || { echo "mtp-num-layers must be a non-negative integer" >&2; exit 2; }
# Qwen3.5-397B-A17B routes each token to 10 experts.
for experts in "${EXPERTS_LIST[@]}"; do
    [[ "${experts}" =~ ^[1-9][0-9]*$ ]] && (( experts % EXPERT_PARALLEL == 0 && experts >= 10 )) || {
        echo "experts must be multiples of EP=${EXPERT_PARALLEL} and at least top-k=10: ${experts}" >&2; exit 2
    }
done
for seq in "${SEQUENCES[@]}"; do
    [[ "${seq}" =~ ^[1-9][0-9]*$ ]] && (( seq % 2 == 0 )) || { echo "seq must be a positive even integer: ${seq}" >&2; exit 2; }
done
for fsdp in "${FSDP_LEVELS[@]}"; do
    case "${fsdp}" in 1|3) ;; *) echo "--fsdp values must be 1 or 3" >&2; exit 2;; esac
done
GBS=$((NUM_MICROBATCHES * GPUS))
MOE_LAYER_FREQ="[1"
for (( i = 1; i < NUM_LAYERS; i++ )); do MOE_LAYER_FREQ+=",1"; done
MOE_LAYER_FREQ+="]"

# A failed case is recorded and the sweep continues; an interrupt stops it.
trap 'echo "benchmark interrupted" >&2; exit 130' INT TERM
FAILED=()
total=0
for experts in "${EXPERTS_LIST[@]}"; do
    for seq in "${SEQUENCES[@]}"; do
        for fsdp in "${FSDP_LEVELS[@]}"; do
            total=$((total + 1))
            ARGS=(
                --model 397b --recipe "${RECIPE}" --fsdp "${fsdp}" --seq-length "${seq}" --cp 1
                --micro-batch-size 1 --global-batch-size "${GBS}" --train-iters "${TRAIN_ITERS}"
                --num-layers "${NUM_LAYERS}" --linear-attention-freq 4 --mtp-num-layers "${MTP_NUM_LAYERS}"
                --num-experts "${experts}" --ep "${EXPERT_PARALLEL}" --dispatcher "${DISPATCHER}"
                --moe-layer-freq "${MOE_LAYER_FREQ}"
                --cuda-graph off --dtype bf16 --gpu "${GPUS}" --nproc-per-node "${GPUS}" --nnodes 1
                --master-port "${MASTER_PORT}" --profile "${PROFILE}"
                --profile-start "${PROFILE_START}" --profile-end "${PROFILE_END}"
                --run-date "${RUN_DATE}" --results-root "${OUTPUT_DIR}"
            )
            [[ "${RESUME}" == true ]] && ARGS+=(--resume)
            [[ "${DRY_RUN}" == true ]] && ARGS+=(--dry-run)
            printf 'launch model=397b experts=%s fsdp=%s seq=%s gbs=%s\n' "${experts}" "${fsdp}" "${seq}" "${GBS}" >&2
            if ! "${RUN_ONE}" "${ARGS[@]}"; then
                echo "case failed: experts=${experts} fsdp=${fsdp} seq=${seq}" >&2
                FAILED+=("experts=${experts} fsdp=${fsdp} seq=${seq}")
            fi
        done
    done
done

printf 'finished %d cases, %d failed\n' "${total}" "${#FAILED[@]}" >&2
if (( ${#FAILED[@]} > 0 )); then
    printf '  failed: %s\n' "${FAILED[@]}" >&2
    exit 1
fi
