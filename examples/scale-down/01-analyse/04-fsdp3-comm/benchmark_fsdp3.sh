#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

# Run the Qwen3.5 FSDP3 communication cases one at a time.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ONE="${SCRIPT_DIR}/run_pretrain_fsdp3.sh"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/01-analyse/04-fsdp3-comm}"
RUN_DATE="${RUN_DATE:-$(date +%y%m%d-%H%M%S)}"

# Representative cases: "model cp seq mbs". Dense vs MoE (EP=4) crossed with
# short sequence, long sequence without CP, and long sequence with CP.
REPRESENTATIVE_CASES=(
    "27b 1 4096 1" "27b 1 32768 1" "27b 4 32768 1"
    "397b 1 4096 1" "397b 1 32768 1" "397b 4 32768 1"
)
MODELS=(397b 122b 27b)
SEQUENCES=(4096 16384 32768)
MICRO_BATCH_SIZES=(1 2 4)
CONTEXT_PARALLEL_SIZES=(1 4)
CUSTOM_MATRIX=false
DTYPE="${DTYPE:-bf16}"
PROFILE="${PROFILE:-nsys}"
TRAIN_ITERS="${TRAIN_ITERS:-10}"
# Microbatches per iteration; GBS is derived per case as nmb * MBS * DP.
NUM_MICROBATCHES_LIST=(1 4)
# A fixed GBS (--global-batch-size) replaces the per-case derivation.
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-}"
GPUS="${GPUS:-4}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NNODES="${NNODES:-}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-${DLC_MASTER_ADDR:-}}"
MASTER_PORT="${MASTER_PORT:-29501}"
PROFILE_START="${PROFILE_START:-5}"
PROFILE_END="${PROFILE_END:-8}"
DRY_RUN=false
RESUME=false

usage() {
    cat <<'USAGE'
Usage: benchmark_fsdp3.sh [options]

By default six representative cases run, each with 1 and 4 microbatches per
iteration (MBS=1, BF16, 12 runs; GBS = microbatches * MBS * DP):
  27b  (dense)      seq=4096 CP=1 | seq=32768 CP=1 | seq=32768 CP=4
  397b (MoE, EP=4)  seq=4096 CP=1 | seq=32768 CP=1 | seq=32768 CP=4
Passing any of --model/--seq/--cp/--mbs switches to the Cartesian matrix of
models=397b,122b,27b; CP=1,4; seq=4096,16384,32768; MBS=1,2,4, restricted by the
given options. Models are cropped to 8 layers and MoE models to 64 routed
experts with EP=4. All cases are validated before the first launch; a failed
case does not stop the sweep.

Options:
  --model NAME[,NAME...]        Select models (or --models)
  --seq N[,N...]                Select sequence lengths (or --seq-lengths)
  --cp N[,N...]                 Select context parallel sizes
  --mbs N[,N...]                Select micro batch sizes
  --dtype|--precision bf16|mxfp8
  --profile none|nsys            Profiling mode (default: nsys)
  --nsys                         Alias for --profile nsys
  --train-iters N               Training iterations
  --num-microbatches N[,N...]   Microbatches per iteration (default: 1,4)
  --global-batch-size N         Fixed global batch size; overrides
                                --num-microbatches
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
  --resume                      Reuse --run-date: skip completed cases and
                                rerun failed or incomplete ones
  --dry-run                     Print cases and commands without running
USAGE
}

need_value() { [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; usage >&2; exit 2; }; }
parse_csv() { local value="$1"; IFS=',' read -r -a "$2" <<<"${value}"; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model|--models) need_value "$@"; parse_csv "$2" MODELS; CUSTOM_MATRIX=true; shift 2;;
        --seq|--seq-length|--seq-lengths) need_value "$@"; parse_csv "$2" SEQUENCES; CUSTOM_MATRIX=true; shift 2;;
        --cp) need_value "$@"; parse_csv "$2" CONTEXT_PARALLEL_SIZES; CUSTOM_MATRIX=true; shift 2;;
        --mbs|--micro-batch-size|--micro-batch-sizes) need_value "$@"; parse_csv "$2" MICRO_BATCH_SIZES; CUSTOM_MATRIX=true; shift 2;;
        --dtype|--precision) need_value "$@"; DTYPE="$2"; [[ "${DTYPE}" == fp8mx ]] && DTYPE=mxfp8; shift 2;;
        --profile) need_value "$@"; PROFILE="$2"; shift 2;;
        --nsys) PROFILE=nsys; shift;;
        --train-iters) need_value "$@"; TRAIN_ITERS="$2"; shift 2;;
        --num-microbatches) need_value "$@"; parse_csv "$2" NUM_MICROBATCHES_LIST; shift 2;;
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
        --resume) RESUME=true; shift;;
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
[[ "${TRAIN_ITERS}" =~ ^[1-9][0-9]*$ ]] || { echo "train-iters must be positive" >&2; exit 2; }
if [[ -n "${GLOBAL_BATCH_SIZE}" ]]; then
    [[ "${GLOBAL_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || { echo "global-batch-size must be positive" >&2; exit 2; }
else
    for nmb in "${NUM_MICROBATCHES_LIST[@]}"; do
        [[ "${nmb}" =~ ^[1-9][0-9]*$ ]] || { echo "num-microbatches must be positive integers" >&2; exit 2; }
    done
fi

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

# Validate the full matrix before launching, so an invalid combination does
# not stop a long sweep halfway through.
CANDIDATES=()
if [[ "${CUSTOM_MATRIX}" == true ]]; then
    for model in "${MODELS[@]}"; do
        for cp in "${CONTEXT_PARALLEL_SIZES[@]}"; do
            for seq in "${SEQUENCES[@]}"; do
                for mbs in "${MICRO_BATCH_SIZES[@]}"; do
                    CANDIDATES+=("${model} ${cp} ${seq} ${mbs}")
                done
            done
        done
    done
else
    CANDIDATES=("${REPRESENTATIVE_CASES[@]}")
fi

CASES=()
for case_spec in "${CANDIDATES[@]}"; do
    read -r model cp seq mbs <<<"${case_spec}"
    recipe_for_model "${model}" >/dev/null || exit 2
    [[ "${cp}" =~ ^[1-9][0-9]*$ && "${seq}" =~ ^[1-9][0-9]*$ && "${mbs}" =~ ^[1-9][0-9]*$ ]] || { echo "cp/seq/mbs must be positive integers" >&2; exit 2; }
    (( GPUS % cp == 0 )) || { echo "gpus=${GPUS} is not divisible by cp=${cp}" >&2; exit 2; }
    (( seq % (2 * cp) == 0 )) || { echo "seq=${seq} is not divisible by 2*cp=$((2 * cp))" >&2; exit 2; }
    if [[ "${model}" != 27b ]]; then
        (( GPUS % 4 == 0 )) || { echo "MoE cases use EP=4; gpus=${GPUS} must be divisible by 4" >&2; exit 2; }
    fi
    if [[ -n "${GLOBAL_BATCH_SIZE}" ]]; then
        (( GLOBAL_BATCH_SIZE % (mbs * (GPUS / cp)) == 0 )) || {
            echo "gbs=${GLOBAL_BATCH_SIZE} is not divisible by mbs=${mbs}*dp=$((GPUS / cp)); use a --global-batch-size multiple of $((mbs * GPUS / cp))" >&2
            exit 2
        }
        CASES+=("${case_spec} ${GLOBAL_BATCH_SIZE}")
    else
        for nmb in "${NUM_MICROBATCHES_LIST[@]}"; do
            CASES+=("${case_spec} $((nmb * mbs * (GPUS / cp)))")
        done
    fi
done

# A failed case is recorded and the sweep continues; an interrupt stops it.
trap 'echo "benchmark interrupted" >&2; exit 130' INT TERM
FAILED=()
for case_spec in "${CASES[@]}"; do
    read -r model cp seq mbs gbs <<<"${case_spec}"
    recipe="$(recipe_for_model "${model}")"
    args=(
        --model "${model}" --recipe "${recipe}" --seq-length "${seq}" --cp "${cp}"
        --micro-batch-size "${mbs}" --global-batch-size "${gbs}" --train-iters "${TRAIN_ITERS}"
        --num-layers 8 --gpu "${GPUS}" --nproc-per-node "${NPROC_PER_NODE}" --nnodes "${NNODES}"
        --node-rank "${NODE_RANK}" --master-port "${MASTER_PORT}" --dtype "${DTYPE}" --profile "${PROFILE}"
        --linear-attention-freq 4 --mtp-num-layers 0
        --profile-start "${PROFILE_START}" --profile-end "${PROFILE_END}" --run-date "${RUN_DATE}" --results-root "${OUTPUT_DIR}"
    )
    (( NNODES > 1 )) && args+=(--master-addr "${MASTER_ADDR}")
    if [[ "${model}" != 27b ]]; then
        args+=(--num-experts 64 --ep 4 --moe-layer-freq '[1,1,1,1,1,1,1,1]')
    fi
    [[ "${RESUME}" == true ]] && args+=(--resume)
    [[ "${DRY_RUN}" == true ]] && args+=(--dry-run)
    printf 'launch model=%s cp=%s seq=%s mbs=%s gbs=%s\n' "${model}" "${cp}" "${seq}" "${mbs}" "${gbs}" >&2
    if ! "${RUN_ONE}" "${args[@]}"; then
        echo "case failed: model=${model} cp=${cp} seq=${seq} mbs=${mbs} gbs=${gbs}" >&2
        FAILED+=("model=${model} cp=${cp} seq=${seq} mbs=${mbs} gbs=${gbs}")
    fi
done

printf 'finished %d cases, %d failed\n' "${#CASES[@]}" "${#FAILED[@]}" >&2
if (( ${#FAILED[@]} > 0 )); then
    printf '  failed: %s\n' "${FAILED[@]}" >&2
    exit 1
fi
