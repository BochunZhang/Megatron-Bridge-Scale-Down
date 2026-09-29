#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)
RUNNER="${REPO_ROOT}/examples/scale-down/01-analyse/01-fine-grained-offload/01-mlp-scope-pretrain/run_pretrain_fsdp1.sh"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/01-analyse/02-megatron-vs-deepspeed}"
MODELS="dense moe"; CASES="baseline recompute offload recompute_offload optimizer_cpu_025 optimizer_cpu_050 optimizer_cpu_075 optimizer_cpu_100"
MICRO_BATCH_SIZES="1 2 4 8"; PRECISION=bf16; TRAIN_ITERS="${TRAIN_ITERS:-10}"; WARMUP_STEPS="${WARMUP_STEPS:-2}"
REPEATS="${REPEATS:-1}"; PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-16}"; PROFILE="${PROFILE:-none}"
RUN_TIME="${RUN_TIME:-$(date +%Y%m%d-%H%M%S)}"; DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models) MODELS="$2"; shift 2;; --cases) CASES="$2"; shift 2;;
        --micro-batch-sizes) MICRO_BATCH_SIZES="$2"; shift 2;; --precision) PRECISION="$2"; shift 2;;
        --train-iters) TRAIN_ITERS="$2"; shift 2;; --warmup-steps) WARMUP_STEPS="$2"; shift 2;;
        --repeats) REPEATS="$2"; shift 2;; --per-gpu-batch-size) PER_GPU_BATCH_SIZE="$2"; shift 2;;
        --profile) PROFILE="$2"; shift 2;; --results-root) RESULTS_ROOT="$2"; shift 2;;
        --run-time) RUN_TIME="$2"; shift 2;; --dry-run) DRY_RUN=true; shift;;
        -h|--help) echo "See README.md for options."; exit 0;;
        *) echo "Unknown argument: $1" >&2; exit 2;;
    esac
done
case "${PRECISION}" in bf16|fp8mx) ;; *) echo "Invalid precision" >&2; exit 2;; esac
case "${PROFILE}" in none|nsys|torch) ;; *) echo "Invalid profile" >&2; exit 2;; esac
[[ "${TRAIN_ITERS}" =~ ^[1-9][0-9]*$ && "${WARMUP_STEPS}" =~ ^[0-9]+$ && "${REPEATS}" =~ ^[1-9][0-9]*$ && "${PER_GPU_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || exit 2
(( WARMUP_STEPS <= TRAIN_ITERS )) || exit 2

mkdir -p "${RESULTS_ROOT}"
MANIFEST="${RESULTS_ROOT}/model_manifest.json"
if [[ "${DRY_RUN}" == false ]]; then
    uv run --no-sync python "${SCRIPT_DIR}/experiment_manifest.py" --output "${MANIFEST}" --num-gpus 4 --num-experts 64
fi

set_case() {
    RECOMPUTE_GRANULARITY=null; RECOMPUTE_MODULES=null; FINE_GRAINED_OFFLOAD=false; OFFLOAD_MODULES=null
    OPTIMIZER_CPU_OFFLOAD=false; OPTIMIZER_OFFLOAD_FRACTION=0.0; OVERLAP_CPU_OPTIMIZER_D2H_H2D=false
    case "$1" in
        baseline) ;;
        recompute) RECOMPUTE_GRANULARITY=selective; RECOMPUTE_MODULES='[layernorm,mlp]' ;;
        offload) FINE_GRAINED_OFFLOAD=true; OFFLOAD_MODULES='[mlp_norm]' ;;
        recompute_offload) RECOMPUTE_GRANULARITY=selective; RECOMPUTE_MODULES='[layernorm,mlp]'; FINE_GRAINED_OFFLOAD=true; OFFLOAD_MODULES='[mlp_norm]' ;;
        optimizer_cpu_025|optimizer_cpu_050|optimizer_cpu_075|optimizer_cpu_100)
            RECOMPUTE_GRANULARITY=selective; RECOMPUTE_MODULES='[layernorm,mlp]'; FINE_GRAINED_OFFLOAD=true; OFFLOAD_MODULES='[mlp_norm]'
            OPTIMIZER_CPU_OFFLOAD=true; OVERLAP_CPU_OPTIMIZER_D2H_H2D=true
            case "$1" in
                optimizer_cpu_025) OPTIMIZER_OFFLOAD_FRACTION=0.25 ;;
                optimizer_cpu_050) OPTIMIZER_OFFLOAD_FRACTION=0.50 ;;
                optimizer_cpu_075) OPTIMIZER_OFFLOAD_FRACTION=0.75 ;;
                optimizer_cpu_100) OPTIMIZER_OFFLOAD_FRACTION=1.0 ;;
            esac
            ;;
        *) echo "Unknown case: $1" >&2; exit 2;;
    esac
}
set_model() {
    case "$1" in
        dense) MODEL=qwen35_text_9b; HF_MODEL=Qwen/Qwen3.5-9B; RECIPE="qwen35_text_9b_pretrain_4gpu_gb200_${PRECISION}_fsdp1_config"; DISPATCHER=default; NUM_EXPERTS="";;
        moe) MODEL=qwen35_text_35b_a3b; HF_MODEL=Qwen/Qwen3.5-35B-A3B; RECIPE="qwen35_text_35b_a3b_pretrain_4gpu_gb200_${PRECISION}_fsdp1_config"; DISPATCHER=hybridep; NUM_EXPERTS=64;;
        *) echo "Unknown model: $1" >&2; exit 2;;
    esac
}

SUMMARY_FILE="${RESULTS_ROOT}/experiment_summary_${RUN_TIME}.txt"; : > "${SUMMARY_FILE}"; set +e
for MODEL_KIND in ${MODELS}; do
    set_model "${MODEL_KIND}"
    for MBS in ${MICRO_BATCH_SIZES}; do
        (( PER_GPU_BATCH_SIZE % MBS == 0 )) || exit 2
        GLOBAL_BATCH_SIZE=$((PER_GPU_BATCH_SIZE * 4))
        for CASE_NAME in ${CASES}; do
            set_case "${CASE_NAME}"
            for ((REPEAT=1; REPEAT<=REPEATS; REPEAT++)); do
                RUN_NAME="${MODEL_KIND}-${CASE_NAME}-mbs${MBS}-r${REPEAT}"
                echo "model=${MODEL_KIND} hf_model=${HF_MODEL} case=${CASE_NAME} mbs=${MBS} repeat=${REPEAT}"
                if [[ "${DRY_RUN}" == true ]]; then echo "${RUN_NAME} DRY_RUN" | tee -a "${SUMMARY_FILE}"; continue; fi
                RUN_ARGS=(--model "${MODEL}" --recipe "${RECIPE}" --precision "${PRECISION}" --run-name "${RUN_NAME}" --recompute-granularity "${RECOMPUTE_GRANULARITY}" --recompute-modules "${RECOMPUTE_MODULES}" --fine-grained-offload "${FINE_GRAINED_OFFLOAD}" --offload-modules "${OFFLOAD_MODULES}" --optimizer-cpu-offload "${OPTIMIZER_CPU_OFFLOAD}" --optimizer-offload-fraction "${OPTIMIZER_OFFLOAD_FRACTION}" --overlap-cpu-optimizer-d2h-h2d "${OVERLAP_CPU_OPTIMIZER_D2H_H2D}" --seed 42 --train-iters "${TRAIN_ITERS}" --warmup-steps "${WARMUP_STEPS}" --global-batch-size "${GLOBAL_BATCH_SIZE}" --micro-batch-size "${MBS}" --dispatcher "${DISPATCHER}" --profile "${PROFILE}" --record-memory-usage true --memory-usage-start-step "${WARMUP_STEPS}")
                [[ -n "${NUM_EXPERTS}" ]] && RUN_ARGS+=(--num-experts "${NUM_EXPERTS}")
                RESULTS_ROOT="${RESULTS_ROOT}" RUN_TIME="${RUN_TIME}" "${RUNNER}" "${RUN_ARGS[@]}"
                RUN_STATUS=$?; [[ ${RUN_STATUS} -eq 0 ]] && STATUS=OK || STATUS="FAILED(rc=${RUN_STATUS})"; echo "${RUN_NAME} ${STATUS}" | tee -a "${SUMMARY_FILE}"
            done
        done
    done
done
set -e
if [[ "${DRY_RUN}" == false ]]; then uv run --no-sync python "${SCRIPT_DIR}/summarize.py" "${RESULTS_ROOT}"; fi
printf 'Summary: %s\nResults: %s\nManifest: %s\n' "${SUMMARY_FILE}" "${RESULTS_ROOT}" "${MANIFEST}"
