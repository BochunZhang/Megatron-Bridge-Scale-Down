#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
RUNNER="${SCRIPT_DIR}/run_pretrain_fsdp1.sh"

RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/01-analyse/03-megatron-vs-deepspeed}"
MODELS="dense expert"
ACTIVATION_STRATEGIES="baseline recompute recompute_offload"
# CPU optimizer offload is disabled while it cannot be combined with Megatron FSDP.
# OPTIMIZER_STRATEGIES="optimizer_none optimizer_cpu_090 optimizer_cpu_075 optimizer_cpu_100"
OPTIMIZER_STRATEGIES="optimizer_none"
MICRO_BATCH_SIZES="1 2 4"
PRECISION="${PRECISION:-bf16}"
TRAIN_ITERS="${TRAIN_ITERS:-10}"
WARMUP_STEPS="${WARMUP_STEPS:-3}"
REPEATS="${REPEATS:-1}"
PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-16}"
RUN_TIME="${RUN_TIME:-$(date +%Y%m%d-%H%M%S)}"
DRY_RUN=false
TEST_MODE=false

usage() {
    cat <<'EOF'
Usage: pretrain_experiment.sh [options]

Options:
  --models "dense expert"
  --activation-strategies "baseline recompute recompute_offload"
  --micro-batch-sizes "1 2 4"
  --precision <bf16|fp8mx>
  --train-iters <n>
  --warmup-steps <n>
  --repeats <n>
  --per-gpu-batch-size <n>
  --results-root <path>
  --test                    Run the smoke matrix with MBS=1
  --dry-run
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models) MODELS="$2"; shift 2 ;;
        --activation-strategies) ACTIVATION_STRATEGIES="$2"; shift 2 ;;
        --micro-batch-sizes) MICRO_BATCH_SIZES="$2"; shift 2 ;;
        --precision) PRECISION="$2"; shift 2 ;;
        --train-iters) TRAIN_ITERS="$2"; shift 2 ;;
        --warmup-steps) WARMUP_STEPS="$2"; shift 2 ;;
        --repeats) REPEATS="$2"; shift 2 ;;
        --per-gpu-batch-size) PER_GPU_BATCH_SIZE="$2"; shift 2 ;;
        --results-root) RESULTS_ROOT="$2"; shift 2 ;;
        --test) TEST_MODE=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ "${TEST_MODE}" == true ]]; then
    MICRO_BATCH_SIZES="1"
fi

case "${PRECISION}" in
    bf16|fp8mx) ;;
    *) echo "Unsupported precision: ${PRECISION}" >&2; exit 2 ;;
esac
DENSE_RECIPE="qwen35_text_9b_pretrain_4gpu_gb200_${PRECISION}_fsdp1_config"
EXPERT_RECIPE="qwen35_text_35b_a3b_pretrain_4gpu_gb200_${PRECISION}_fsdp1_config"
if ! [[ "${TRAIN_ITERS}" =~ ^[1-9][0-9]*$ && "${WARMUP_STEPS}" =~ ^[0-9]+$ && \
    "${REPEATS}" =~ ^[1-9][0-9]*$ && "${PER_GPU_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Iteration, repeat, and batch-size arguments must be integers" >&2
    exit 2
fi
if (( WARMUP_STEPS > TRAIN_ITERS )); then
    echo "warmup-steps must not exceed train-iters" >&2
    exit 2
fi

set_model() {
    case "$1" in
        dense)
            MODEL="qwen35_text_9b"
            HF_MODEL="Qwen/Qwen3.5-9B-Base"
            RECIPE="${DENSE_RECIPE}"
            DISPATCHER="default"
            NUM_EXPERTS=""
            ;;
        expert)
            MODEL="qwen35_text_35b_a3b"
            HF_MODEL="Qwen/Qwen3.5-35B-A3B-Base"
            RECIPE="${EXPERT_RECIPE}"
            DISPATCHER="hybridep"
            NUM_EXPERTS=64
            ;;
        *) echo "Unknown model: $1" >&2; exit 2 ;;
    esac
}

set_activation_strategy() {
    RECOMPUTE_GRANULARITY="null"
    RECOMPUTE_METHOD="null"
    RECOMPUTE_NUM_LAYERS="null"
    RECOMPUTE_MODULES="null"
    FINE_GRAINED_OFFLOAD=false
    OFFLOAD_MODULES="null"

    case "$1" in
        baseline)
            ;;
        recompute)
            RECOMPUTE_GRANULARITY="full"
            RECOMPUTE_METHOD="uniform"
            RECOMPUTE_NUM_LAYERS=1
            ;;
        recompute_offload)
            RECOMPUTE_GRANULARITY="selective"
            FINE_GRAINED_OFFLOAD=true
            if [[ "${MODEL_KIND}" == "dense" ]]; then
                RECOMPUTE_MODULES="[layernorm,mlp_act]"
                OFFLOAD_MODULES="[mlp_norm,mlp_act]"
            else
                RECOMPUTE_MODULES="[layernorm,moe_act]"
                OFFLOAD_MODULES="[mlp_norm,expert_fc1,moe_act]"
            fi
            ;;
        *) echo "Unknown activation strategy: $1" >&2; exit 2 ;;
    esac
}

set_optimizer_strategy() {
    OPTIMIZER_CPU_OFFLOAD=false
    OPTIMIZER_OFFLOAD_FRACTION=0.0
    USE_TORCH_OPTIMIZER_FOR_CPU_OFFLOAD=false
    OVERLAP_CPU_OPTIMIZER_D2H_H2D=false

    case "$1" in
        optimizer_none)
            ;;
        optimizer_cpu_090)
            OPTIMIZER_CPU_OFFLOAD=true
            OPTIMIZER_OFFLOAD_FRACTION=0.9
            OVERLAP_CPU_OPTIMIZER_D2H_H2D=true
            ;;
        optimizer_cpu_075)
            OPTIMIZER_CPU_OFFLOAD=true
            OPTIMIZER_OFFLOAD_FRACTION=0.75
            OVERLAP_CPU_OPTIMIZER_D2H_H2D=true
            ;;
        optimizer_cpu_100)
            OPTIMIZER_CPU_OFFLOAD=true
            OPTIMIZER_OFFLOAD_FRACTION=1.0
            OVERLAP_CPU_OPTIMIZER_D2H_H2D=true
            ;;
        *) echo "Unknown optimizer strategy: $1" >&2; exit 2 ;;
    esac
}

mkdir -p "${RESULTS_ROOT}"
export RESULTS_ROOT RUN_TIME
MANIFEST="${RESULTS_ROOT}/model_manifest.json"
if [[ "${DRY_RUN}" == false ]]; then
    uv run --no-sync python "${SCRIPT_DIR}/experiment_manifest.py" \
        --output "${MANIFEST}" \
        --dense-recipe "${DENSE_RECIPE}" \
        --moe-recipe "${EXPERT_RECIPE}" \
        --num-gpus 4 \
        --num-experts 64
fi

SUMMARY_FILE="${RESULTS_ROOT}/experiment_summary_${RUN_TIME}.txt"
: > "${SUMMARY_FILE}"
set +e
for MODEL_KIND in ${MODELS}; do
    set_model "${MODEL_KIND}"
    for MBS in ${MICRO_BATCH_SIZES}; do
        if (( PER_GPU_BATCH_SIZE % MBS != 0 )); then
            echo "micro-batch-size ${MBS} must divide per-gpu-batch-size ${PER_GPU_BATCH_SIZE}" >&2
            exit 2
        fi
        GLOBAL_BATCH_SIZE=$((PER_GPU_BATCH_SIZE * 4))
        for ACTIVATION_STRATEGY in ${ACTIVATION_STRATEGIES}; do
            set_activation_strategy "${ACTIVATION_STRATEGY}"
            for OPTIMIZER_STRATEGY in ${OPTIMIZER_STRATEGIES}; do
                set_optimizer_strategy "${OPTIMIZER_STRATEGY}"
                for ((REPEAT = 1; REPEAT <= REPEATS; REPEAT++)); do
                    TEST_NAME="${ACTIVATION_STRATEGY//_/-}-${OPTIMIZER_STRATEGY//_/-}-r${REPEAT}"
                    RUN_NAME="${MODEL_KIND}-${TEST_NAME}"
                    RESULT_PATH_NAME="dtype_${PRECISION}-mbs_${MBS}-gbs_${GLOBAL_BATCH_SIZE}"
                    if [[ "${MODEL_KIND}" == expert ]]; then
                        RESULT_PATH_NAME+="-dispatcher_${DISPATCHER}"
                    fi
                    RESULT_PATH_NAME+="-${TEST_NAME}"
                    echo "matrix run=${RUN_NAME} result_path_name=${RESULT_PATH_NAME} model=${MODEL_KIND} hf_model=${HF_MODEL} activation=${ACTIVATION_STRATEGY} optimizer=${OPTIMIZER_STRATEGY} sharding=optim_grads_params recompute_granularity=${RECOMPUTE_GRANULARITY} recompute_method=${RECOMPUTE_METHOD} recompute_num_layers=${RECOMPUTE_NUM_LAYERS} recompute_modules=${RECOMPUTE_MODULES} fine_grained_activation_offloading=${FINE_GRAINED_OFFLOAD} offload_modules=${OFFLOAD_MODULES} optimizer_cpu_offload=${OPTIMIZER_CPU_OFFLOAD} optimizer_offload_fraction=${OPTIMIZER_OFFLOAD_FRACTION} overlap_cpu_optimizer_d2h_h2d=${OVERLAP_CPU_OPTIMIZER_D2H_H2D} record_memory_usage=true"
                    if [[ "${DRY_RUN}" == true ]]; then
                        echo "${RUN_NAME} DRY_RUN" >> "${SUMMARY_FILE}"
                        continue
                    fi

                    RUN_ARGS=(
                        --model "${MODEL}"
                        --recipe "${RECIPE}"
                        --precision "${PRECISION}"
                        --run-name "${RUN_NAME}"
                        --activation-strategy "${ACTIVATION_STRATEGY}"
                        --optimizer-strategy "${OPTIMIZER_STRATEGY}"
                        --recompute-granularity "${RECOMPUTE_GRANULARITY}"
                        --recompute-method "${RECOMPUTE_METHOD}"
                        --recompute-num-layers "${RECOMPUTE_NUM_LAYERS}"
                        --recompute-modules "${RECOMPUTE_MODULES}"
                        --fine-grained-offload "${FINE_GRAINED_OFFLOAD}"
                        --offload-modules "${OFFLOAD_MODULES}"
                        --optimizer-cpu-offload "${OPTIMIZER_CPU_OFFLOAD}"
                        --optimizer-offload-fraction "${OPTIMIZER_OFFLOAD_FRACTION}"
                        --use-torch-optimizer-for-cpu-offload "${USE_TORCH_OPTIMIZER_FOR_CPU_OFFLOAD}"
                        --overlap-cpu-optimizer-d2h-h2d "${OVERLAP_CPU_OPTIMIZER_D2H_H2D}"
                        --seed 42
                        --train-iters "${TRAIN_ITERS}"
                        --warmup-steps "${WARMUP_STEPS}"
                        --global-batch-size "${GLOBAL_BATCH_SIZE}"
                        --micro-batch-size "${MBS}"
                        --dispatcher "${DISPATCHER}"
                        --profile none
                        --record-memory-usage true
                        --memory-usage-start-step "${WARMUP_STEPS}"
                    )
                    if [[ -n "${NUM_EXPERTS}" ]]; then
                        RUN_ARGS+=(--num-experts "${NUM_EXPERTS}")
                    fi
                    "${RUNNER}" "${RUN_ARGS[@]}"
                    RUN_STATUS=$?
                    if [[ ${RUN_STATUS} -eq 0 ]]; then STATUS=OK; else STATUS="FAILED(rc=${RUN_STATUS})"; fi
                    echo "${RUN_NAME} ${STATUS}" | tee -a "${SUMMARY_FILE}"
                done
            done
        done
    done
done
set -e

if [[ "${DRY_RUN}" == false ]]; then
    uv run --no-sync python "${SCRIPT_DIR}/summarize.py" "${RESULTS_ROOT}"
    uv run --no-sync python "${SCRIPT_DIR}/collect_results.py" \
        --results-root "${RESULTS_ROOT}" \
        --run-time "${RUN_TIME}"
fi
printf 'Summary: %s\nResults: %s\nManifest: %s\n' "${SUMMARY_FILE}" "${RESULTS_ROOT}" "${MANIFEST}"
