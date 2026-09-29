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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
RUN_ONE="${SCRIPT_DIR}/run_pretrain_fsdp1.sh"

SCOPE="all"
MODEL_FAMILY="qwen"
DENSE_MODEL="9b"
DTYPE="${DTYPE:-bf16}"
PROFILE="none"
CASE_FILTER="all"
DISPATCHER_FILTER="all"
TRAIN_ITERS="${TRAIN_ITERS:-10}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
MICRO_BATCH_SIZES="${MICRO_BATCH_SIZES:-1,2,4,8}"
HYBRIDEP_NUM_SMS="${HYBRIDEP_NUM_SMS:-32}"
QWEN_EXPERT_NUM_LAYERS=16
DEEPSEEK_NUM_LAYERS=4
EXPERT_NUM_EXPERTS=64
DEEPSEEK_NUM_EXPERTS=32
EXPERT_LINEAR_ATTENTION_FREQ=4
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/result/01-analyse/01-fine-grained-offload/02-offload-on-dense-or-expert-model}"
RUN_TIME="${RUN_TIME:-$(date +%Y%m%d-%H%M%S)}"
DRY_RUN=false
FAILURES=0

usage() {
    cat <<'EOF'
Usage: benchmark_mlp_offload.sh [OPTIONS]

Run a controlled MLP activation-memory benchmark on four local GB200 GPUs.
Qwen runs dense and expert cases. DeepSeek-V3 runs both dense and expert
proxies by overriding its four-layer MoE layout. The Qwen expert uses 16
layers and 64 experts; DeepSeek-V3 uses 4 layers and 32 experts. Activation
recompute is disabled in every case.

Options:
    --scope <all|dense|expert>    Matrix subset (default: all)
    --model <qwen|deepseek>       Model family (default: qwen)
    --dense-model <9b|27b>       Dense model size (default: 9b)
    --dtype <bf16|mxfp8>         Training dtype (default: bf16)
    --profile <none|nsys|torch>  Profiling backend (default: none)
    --case <name|all>            Run baseline or offload (default: all)
    --dispatcher <all|alltoall|hybridep>
                                 MoE dispatcher filter (default: all)
    --train-iters <n>            Total steps per run (default: 10)
    --global-batch-size <n>      Global batch size (default: 32)
    --micro-batch-sizes <list>   Comma-separated MBS sweep (default: 1,2,4,8)
    --micro-batch-size <n>       Run a single MBS value
    --hybridep-num-sms <n>       HybridEP communication SMs (default: 32)
    --results-root <path>        Result tree root
    --dry-run                    Print the matrix without launching training
    -h, --help                   Show this help

Use --profile none for timing runs. Rerun representative cases with nsys or
torch profiling to attribute copy and synchronization overhead.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scope) SCOPE="$2"; shift 2 ;;
        --model) MODEL_FAMILY="$2"; shift 2 ;;
        --dense-model) DENSE_MODEL="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --precision)
            case "$2" in
                bf16) DTYPE=bf16 ;;
                fp8mx|mxfp8) DTYPE=mxfp8 ;;
                *) DTYPE="$2" ;;
            esac
            shift 2
            ;;
        --profile) PROFILE="$2"; shift 2 ;;
        --case) CASE_FILTER="$2"; shift 2 ;;
        --dispatcher) DISPATCHER_FILTER="$2"; shift 2 ;;
        --train-iters) TRAIN_ITERS="$2"; shift 2 ;;
        --global-batch-size) GLOBAL_BATCH_SIZE="$2"; shift 2 ;;
        --micro-batch-sizes) MICRO_BATCH_SIZES="$2"; shift 2 ;;
        --micro-batch-size) MICRO_BATCH_SIZES="$2"; shift 2 ;;
        --hybridep-num-sms) HYBRIDEP_NUM_SMS="$2"; shift 2 ;;
        --results-root) RESULTS_ROOT="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "${SCOPE}" in all|dense|expert) ;; *) echo "Invalid scope: ${SCOPE}" >&2; exit 2 ;; esac
case "${MODEL_FAMILY}" in qwen|deepseek) ;; *) echo "Invalid model: ${MODEL_FAMILY}; expected qwen or deepseek" >&2; exit 2 ;; esac
case "${DENSE_MODEL}" in 9b|27b) ;; *) echo "Invalid dense model: ${DENSE_MODEL}" >&2; exit 2 ;; esac
case "${DTYPE}" in
    bf16) RECIPE_DTYPE=bf16 ;;
    mxfp8) RECIPE_DTYPE=fp8mx ;;
    *) echo "Invalid dtype: ${DTYPE}; expected bf16 or mxfp8" >&2; exit 2 ;;
esac
case "${PROFILE}" in none|nsys|torch) ;; *) echo "Invalid profile: ${PROFILE}" >&2; exit 2 ;; esac
case "${CASE_FILTER}" in
    all|baseline|offload) ;;
    *) echo "Invalid case: ${CASE_FILTER}" >&2; exit 2 ;;
esac
case "${DISPATCHER_FILTER}" in
    all|alltoall|hybridep) ;;
    *) echo "Invalid dispatcher: ${DISPATCHER_FILTER}" >&2; exit 2 ;;
esac

for value_name in TRAIN_ITERS GLOBAL_BATCH_SIZE HYBRIDEP_NUM_SMS; do
    if ! [[ "${!value_name}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${value_name} must be a positive integer" >&2
        exit 2
    fi
done

IFS=',' read -r -a MICRO_BATCH_SIZE_VALUES <<< "${MICRO_BATCH_SIZES}"
if (( ${#MICRO_BATCH_SIZE_VALUES[@]} == 0 )); then
    echo "MICRO_BATCH_SIZES must contain at least one value" >&2
    exit 2
fi
for micro_batch_size in "${MICRO_BATCH_SIZE_VALUES[@]}"; do
    if ! [[ "${micro_batch_size}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Each micro batch size must be a positive integer: ${micro_batch_size}" >&2
        exit 2
    fi
    if (( GLOBAL_BATCH_SIZE % micro_batch_size != 0 )); then
        echo "GLOBAL_BATCH_SIZE must be divisible by every micro batch size: ${micro_batch_size}" >&2
        exit 2
    fi
done

if [[ "${MODEL_FAMILY}" == deepseek ]]; then
    DENSE_MODEL_NAME="deepseek_v3"
    DENSE_RECIPE_PREFIX="deepseek_v3_pretrain_4gpu_gb200"
    EXPERT_NUM_LAYERS="${DEEPSEEK_NUM_LAYERS}"
    EXPERT_NUM_EXPERTS="${DEEPSEEK_NUM_EXPERTS}"
    EXPERT_MODEL_NAME="deepseek_v3"
    EXPERT_RECIPE_PREFIX="deepseek_v3_pretrain_4gpu_gb200"
else
    EXPERT_NUM_LAYERS="${QWEN_EXPERT_NUM_LAYERS}"
    if [[ "${DENSE_MODEL}" == 9b ]]; then
        DENSE_MODEL_NAME="qwen35_text_9b"
        DENSE_RECIPE_PREFIX="qwen35_text_9b_pretrain_4gpu_gb200"
    else
        DENSE_MODEL_NAME="qwen35_text_27b"
        DENSE_RECIPE_PREFIX="qwen35_text_27b_pretrain_4gpu_gb200"
    fi
    EXPERT_MODEL_NAME="qwen35_text_35b_a3b"
    EXPERT_RECIPE_PREFIX="qwen35_text_35b_a3b_pretrain_4gpu_gb200"
fi
PROFILE_STEP_START=7
PROFILE_STEP_END=8

run_case() {
    local model_kind="$1"
    local model="$2"
    local recipe_prefix="$3"
    local dispatcher="$4"
    local case_name="$5"
    local recompute_granularity="$6"
    local recompute_modules="$7"
    local fine_grained_offload="$8"
    local offload_modules="$9"
    local micro_batch_size="${10}"
    local run_name
    local recipe="${recipe_prefix}_${RECIPE_DTYPE}_fsdp1_config"
    local num_layers="default"
    local num_experts="default"
    local moe_layer_freq="default"
    local num_nextn_predict_layers="default"
    local -a profile_args=()
    local -a model_override_args=()

    if [[ "${CASE_FILTER}" != all && "${CASE_FILTER}" != "${case_name}" ]]; then
        return
    fi

    printf -v run_name '%s-%s-%s-mbs%s-r01' \
        "${model_kind}" "${dispatcher}" "${case_name}" "${micro_batch_size}"
    if [[ "${PROFILE}" != none ]]; then
        profile_args=(--profile "${PROFILE}")
    fi
    if [[ "${MODEL_FAMILY}" == deepseek ]]; then
        # Keep the list in one argv element so Hydra parses it as list[int].
        num_layers="${DEEPSEEK_NUM_LAYERS}"
        num_experts="${DEEPSEEK_NUM_EXPERTS}"
        num_nextn_predict_layers=0
        if [[ "${model_kind}" == expert ]]; then
            moe_layer_freq='[1,1,1,1]'
        else
            moe_layer_freq='[0,0,0,0]'
        fi
        model_override_args=(
            --num-layers "${DEEPSEEK_NUM_LAYERS}"
            --num-experts "${DEEPSEEK_NUM_EXPERTS}"
            --moe-layer-freq "${moe_layer_freq}"
            --num-nextn-predict-layers "${num_nextn_predict_layers}"
        )
    elif [[ "${model_kind}" == expert ]]; then
        num_layers="${EXPERT_NUM_LAYERS}"
        num_experts="${EXPERT_NUM_EXPERTS}"
        model_override_args=(
            --num-layers "${EXPERT_NUM_LAYERS}"
            --num-experts "${EXPERT_NUM_EXPERTS}"
        )
        if [[ "${MODEL_FAMILY}" == qwen ]]; then
            model_override_args+=(--linear-attention-freq "${EXPERT_LINEAR_ATTENTION_FREQ}")
        fi
    fi

    printf 'matrix model=%s dtype=%s recipe=%s case=%s dispatcher=%s mbs=%s repeat=1 layers=%s experts=%s moe_layer_freq=%s num_nextn_predict_layers=%s offload=%s recompute=%s\n' \
        "${model}" "${DTYPE}" "${recipe}" "${case_name}" "${dispatcher}" \
        "${micro_batch_size}" "${num_layers}" "${num_experts}" \
        "${moe_layer_freq}" "${num_nextn_predict_layers}" \
        "${offload_modules}" "${recompute_modules}"
    if [[ "${DRY_RUN}" == true ]]; then
        return
    fi

    set +e
    "${RUN_ONE}" \
        "${profile_args[@]}" \
        "${model_override_args[@]}" \
        --model "${model}" \
        --recipe "${recipe}" \
        --dtype "${DTYPE}" \
        --run-name "${run_name}" \
        --dispatcher "${dispatcher}" \
        --hybridep-num-sms "${HYBRIDEP_NUM_SMS}" \
        --recompute-granularity "${recompute_granularity}" \
        --recompute-modules "${recompute_modules}" \
        --fine-grained-offload "${fine_grained_offload}" \
        --offload-modules "${offload_modules}" \
        --train-iters "${TRAIN_ITERS}" \
        --global-batch-size "${GLOBAL_BATCH_SIZE}" \
        --micro-batch-size "${micro_batch_size}" \
        --profile-step-start "${PROFILE_STEP_START}" \
        --profile-step-end "${PROFILE_STEP_END}"
    local status=$?
    set -e
    if (( status != 0 )); then
        printf 'Run failed with status %d: %s\n' "${status}" "${run_name}" >&2
        FAILURES=$((FAILURES + 1))
    fi
}

run_dense_matrix() {
    local micro_batch_size="$1"
    run_case dense "${DENSE_MODEL_NAME}" "${DENSE_RECIPE_PREFIX}" default baseline null null false null "${micro_batch_size}"
    run_case dense "${DENSE_MODEL_NAME}" "${DENSE_RECIPE_PREFIX}" default offload null null true '[mlp_norm,mlp_act]' "${micro_batch_size}"
}

run_expert_matrix() {
    local micro_batch_size="$1"
    local dispatcher
    for dispatcher in alltoall hybridep; do
        if [[ "${DISPATCHER_FILTER}" != all && "${DISPATCHER_FILTER}" != "${dispatcher}" ]]; then
            continue
        fi
        run_case expert "${EXPERT_MODEL_NAME}" "${EXPERT_RECIPE_PREFIX}" "${dispatcher}" baseline null null false null "${micro_batch_size}"
        run_case expert "${EXPERT_MODEL_NAME}" "${EXPERT_RECIPE_PREFIX}" "${dispatcher}" offload null null true '[mlp_norm,expert_fc1,moe_act]' "${micro_batch_size}"
    done
}

export RESULTS_ROOT RUN_TIME
printf 'benchmark_id=%s model=%s dtype=%s dispatcher=%s results_root=%s train_iters=%s runs_per_case=1 micro_batch_sizes=%s\n' \
    "${RUN_TIME}" "${MODEL_FAMILY}" "${DTYPE}" "${DISPATCHER_FILTER}" "${RESULTS_ROOT}" "${TRAIN_ITERS}" "${MICRO_BATCH_SIZES}"

for micro_batch_size in "${MICRO_BATCH_SIZE_VALUES[@]}"; do
    case "${SCOPE}" in
        all)
            run_dense_matrix "${micro_batch_size}"
            run_expert_matrix "${micro_batch_size}"
            ;;
        dense) run_dense_matrix "${micro_batch_size}" ;;
        expert) run_expert_matrix "${micro_batch_size}" ;;
    esac
done

if [[ "${DRY_RUN}" == true ]]; then
    exit 0
fi

if (( FAILURES > 0 )); then
    printf '%d training run(s) failed; inspect the raw result directories above.\n' "${FAILURES}" >&2
    exit 1
fi

printf 'Raw results: %s (run_time=%s)\n' "${RESULTS_ROOT}" "${RUN_TIME}"
printf 'Collect XLSX: %s/collect_mlp_offload_results.mjs\n' "${SCRIPT_DIR}"
