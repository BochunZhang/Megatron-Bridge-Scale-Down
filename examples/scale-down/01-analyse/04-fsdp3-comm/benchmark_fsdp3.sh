#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

# Run the Qwen3.5 FSDP3 communication matrix one case at a time.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ONE="${SCRIPT_DIR}/run_pretrain_fsdp3.sh"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/01-analyse/04-fsdp3-comm}"
RUN_DATE="${RUN_DATE:-$(date +%y%m%d-%H%M%S)}"

MODELS=(397b 122b 27b)
SEQUENCES=(4096 16384 32768)
MICRO_BATCH_SIZES=(1 2 4)
CONTEXT_PARALLEL_SIZES=(1 4)
DTYPE="${DTYPE:-bf16}"
PROFILE="${PROFILE:-nsys}"
TRAIN_ITERS="${TRAIN_ITERS:-10}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
GPUS="${GPUS:-4}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NNODES="${NNODES:-}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-${DLC_MASTER_ADDR:-}}"
MASTER_PORT="${MASTER_PORT:-29501}"
PROFILE_START="${PROFILE_START:-5}"
PROFILE_END="${PROFILE_END:-8}"
DRY_RUN=false

usage() {
    cat <<'USAGE'
Usage: benchmark_fsdp3.sh [options]

The default matrix is models=397b,122b,27b; CP=1,4; sequence=4096,16384,32768;
MBS=1,2,4; GBS=32; BF16. Each selected model is cropped to 8 layers and MoE
models to 64 routed experts by the per-case launcher.

Options:
  --model NAME[,NAME...]        Select models (or --models)
  --seq N[,N...]                Select sequence lengths (or --seq-lengths)
  --cp N[,N...]                 Select context parallel sizes
  --mbs N[,N...]                Select micro batch sizes
  --dtype|--precision bf16|mxfp8
  --profile none|nsys            Profiling mode (default: nsys)
  --nsys                         Alias for --profile nsys
  --train-iters N               Training iterations
  --global-batch-size N         Global batch size
  --gpus N                      Global GPUs (default: 4)
  --nproc-per-node N            GPUs per node
  --nnodes N                    Number of nodes
  --node-rank N                 Node rank
  --master-addr HOST            Rendezvous address
  --master-port N               Rendezvous port
  --profile-start N             First profiled iteration
  --profile-end N               Last profiled iteration
  --run-date YYMMDD-HHMMSS      Shared output timestamp
  --output-dir PATH             Result root
  --dry-run                     Print cases and commands without running
USAGE
}

need_value() { [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; usage >&2; exit 2; }; }
parse_csv() { local value="$1"; IFS=',' read -r -a "$2" <<<"${value}"; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model|--models) need_value "$@"; parse_csv "$2" MODELS; shift 2;;
        --seq|--seq-length|--seq-lengths) need_value "$@"; parse_csv "$2" SEQUENCES; shift 2;;
        --cp) need_value "$@"; parse_csv "$2" CONTEXT_PARALLEL_SIZES; shift 2;;
        --mbs|--micro-batch-size|--micro-batch-sizes) need_value "$@"; parse_csv "$2" MICRO_BATCH_SIZES; shift 2;;
        --dtype|--precision) need_value "$@"; DTYPE="$2"; [[ "${DTYPE}" == fp8mx ]] && DTYPE=mxfp8; shift 2;;
        --profile) need_value "$@"; PROFILE="$2"; shift 2;;
        --nsys) PROFILE=nsys; shift;;
        --train-iters) need_value "$@"; TRAIN_ITERS="$2"; shift 2;;
        --global-batch-size) need_value "$@"; GLOBAL_BATCH_SIZE="$2"; shift 2;;
        --gpus|--gpu) need_value "$@"; GPUS="$2"; shift 2;;
        --nproc-per-node) need_value "$@"; NPROC_PER_NODE="$2"; shift 2;;
        --nnodes) need_value "$@"; NNODES="$2"; shift 2;;
        --node-rank) need_value "$@"; NODE_RANK="$2"; shift 2;;
        --master-addr) need_value "$@"; MASTER_ADDR="$2"; shift 2;;
        --master-port) need_value "$@"; MASTER_PORT="$2"; shift 2;;
        --profile-start) need_value "$@"; PROFILE_START="$2"; shift 2;;
        --profile-end) need_value "$@"; PROFILE_END="$2"; shift 2;;
        --run-date) need_value "$@"; RUN_DATE="$2"; shift 2;;
        --output-dir) need_value "$@"; OUTPUT_DIR="$2"; shift 2;;
        --dry-run) DRY_RUN=true; shift;;
        -h|--help) usage; exit 0;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2;;
    esac
done

[[ "${DTYPE}" == fp8mx ]] && DTYPE=mxfp8
case "${DTYPE}" in bf16|mxfp8) ;; *) echo "--dtype must be bf16 or mxfp8" >&2; exit 2;; esac
case "${PROFILE}" in none|nsys) ;; *) echo "--profile must be none or nsys" >&2; exit 2;; esac
[[ "${RUN_DATE}" =~ ^[0-9]{6}-[0-9]{6}$ ]] || { echo "--run-date must use yymmdd-hhmmss" >&2; exit 2; }
[[ "${GPUS}" =~ ^[1-9][0-9]*$ && "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]] || { echo "GPU counts must be positive" >&2; exit 2; }
[[ "${TRAIN_ITERS}" =~ ^[1-9][0-9]*$ && "${GLOBAL_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || { echo "training sizes must be positive" >&2; exit 2; }

if (( ${#MODELS[@]} == 1 )) && [[ "${MODELS[0]}" == all ]]; then
    MODELS=(397b 122b 27b)
fi
(( GPUS % NPROC_PER_NODE == 0 )) || { echo "gpus must be divisible by nproc-per-node" >&2; exit 2; }
NNODES="${NNODES:-$((GPUS / NPROC_PER_NODE))}"
[[ "${NNODES}" =~ ^[1-9][0-9]*$ ]] || { echo "nnodes must be positive" >&2; exit 2; }
(( NNODES * NPROC_PER_NODE == GPUS )) || { echo "gpus must equal nnodes*nproc-per-node" >&2; exit 2; }
if (( NNODES > 1 )) && [[ -z "${MASTER_ADDR}" ]]; then echo "multi-node runs require --master-addr or MASTER_ADDR" >&2; exit 2; fi

recipe_for_model() {
    case "$1" in
        397b) echo qwen35_text_397b_a17b_pretrain_4gpu_gb200_bf16_fsdp1_config;;
        122b) echo qwen35_text_122b_a10b_pretrain_4gpu_gb200_bf16_fsdp1_config;;
        27b) echo qwen35_text_27b_pretrain_4gpu_gb200_bf16_fsdp1_config;;
        *) echo "Unsupported model: $1" >&2; return 2;;
    esac
}

for model in "${MODELS[@]}"; do
    recipe="$(recipe_for_model "${model}")"
    for cp in "${CONTEXT_PARALLEL_SIZES[@]}"; do
        for seq in "${SEQUENCES[@]}"; do
            for mbs in "${MICRO_BATCH_SIZES[@]}"; do
                [[ "${cp}" =~ ^[1-9][0-9]*$ && "${seq}" =~ ^[1-9][0-9]*$ && "${mbs}" =~ ^[1-9][0-9]*$ ]] || { echo "cp/seq/mbs must be positive integers" >&2; exit 2; }
                (( GPUS % cp == 0 )) || { echo "gpus=${GPUS} is not divisible by cp=${cp}" >&2; exit 2; }
                (( GLOBAL_BATCH_SIZE % (mbs * (GPUS / cp)) == 0 )) || { echo "gbs=${GLOBAL_BATCH_SIZE} is not divisible by mbs=${mbs}*dp=$((GPUS / cp))" >&2; exit 2; }
                args=(
                    --model "${model}" --recipe "${recipe}" --seq-length "${seq}" --cp "${cp}"
                    --micro-batch-size "${mbs}" --global-batch-size "${GLOBAL_BATCH_SIZE}" --train-iters "${TRAIN_ITERS}"
                    --num-layers 8 --gpu "${GPUS}" --nproc-per-node "${NPROC_PER_NODE}" --nnodes "${NNODES}"
                    --node-rank "${NODE_RANK}" --master-port "${MASTER_PORT}" --dtype "${DTYPE}" --profile "${PROFILE}"
                    --linear-attention-freq 4 --mtp-num-layers 0
                    --profile-start "${PROFILE_START}" --profile-end "${PROFILE_END}" --run-date "${RUN_DATE}" --results-root "${OUTPUT_DIR}"
                )
                (( NNODES > 1 )) && args+=(--master-addr "${MASTER_ADDR}")
                if [[ "${model}" != 27b ]]; then
                    args+=(--num-experts 64 --moe-layer-freq '[1,1,1,1,1,1,1,1]')
                fi
                [[ "${DRY_RUN}" == true ]] && args+=(--dry-run)
                printf 'launch model=%s cp=%s seq=%s mbs=%s\n' "${model}" "${cp}" "${seq}" "${mbs}" >&2
                "${RUN_ONE}" "${args[@]}"
            done
        done
    done
done
