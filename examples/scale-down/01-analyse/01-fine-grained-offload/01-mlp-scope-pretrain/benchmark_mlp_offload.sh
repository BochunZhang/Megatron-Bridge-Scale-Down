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
SUMMARIZER="${SCRIPT_DIR}/summarize_mlp_offload.py"

SCOPE="all"
DENSE_MODEL="9b"
PRECISION="bf16"
PROFILE="none"
CASE_FILTER="all"
TRAIN_ITERS="${TRAIN_ITERS:-10}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
HYBRIDEP_NUM_SMS="${HYBRIDEP_NUM_SMS:-32}"
EXPERT_NUM_LAYERS=20
EXPERT_NUM_EXPERTS=64
EXPERT_LINEAR_ATTENTION_FREQ=4
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/01-analyse/01-fine-grained-offload}"
RUN_TIME="${RUN_TIME:-$(date +%Y%m%d-%H%M%S)}"
DRY_RUN=false
FAILURES=0

usage() {
    cat <<'EOF'
Usage: benchmark_mlp_offload.sh [OPTIONS]

Run a controlled MLP activation-memory benchmark on four local GB200 GPUs.
Dense models run baseline and MLP activation offload. The expert model runs
baseline and expert activation offload with both all-to-all and HybridEP
dispatchers. The expert model is reduced to 20 layers and 64 experts to lower
peak memory. Activation recompute is disabled in every case.

Options:
    --scope <all|dense|expert>    Matrix subset (default: all)
    --dense-model <9b|27b>       Dense model size (default: 9b)
    --precision <bf16|fp8mx>     Precision (default: bf16)
    --profile <none|nsys|torch>  Profiling backend (default: none)
    --case <name|all>            Run baseline or offload (default: all)
    --train-iters <n>            Total steps per run (default: 10)
    --global-batch-size <n>      Global batch size (default: 32)
    --micro-batch-size <n>       Micro batch size (default: 1)
    --hybridep-num-sms <n>       HybridEP communication SMs (default: 32)
    --results-root <path>        Result tree root
    --run-time <id>              Stable batch id used to group/resume results
    --dry-run                    Print the matrix without launching training
    -h, --help                   Show this help

Use --profile none for timing runs. Rerun representative cases with nsys or
torch profiling to attribute copy and synchronization overhead.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scope) SCOPE="$2"; shift 2 ;;
        --dense-model) DENSE_MODEL="$2"; shift 2 ;;
        --precision) PRECISION="$2"; shift 2 ;;
        --profile) PROFILE="$2"; shift 2 ;;
        --case) CASE_FILTER="$2"; shift 2 ;;
        --train-iters) TRAIN_ITERS="$2"; shift 2 ;;
        --global-batch-size) GLOBAL_BATCH_SIZE="$2"; shift 2 ;;
        --micro-batch-size) MICRO_BATCH_SIZE="$2"; shift 2 ;;
        --hybridep-num-sms) HYBRIDEP_NUM_SMS="$2"; shift 2 ;;
        --results-root) RESULTS_ROOT="$2"; shift 2 ;;
        --run-time) RUN_TIME="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "${SCOPE}" in all|dense|expert) ;; *) echo "Invalid scope: ${SCOPE}" >&2; exit 2 ;; esac
case "${DENSE_MODEL}" in 9b|27b) ;; *) echo "Invalid dense model: ${DENSE_MODEL}" >&2; exit 2 ;; esac
case "${PRECISION}" in bf16|fp8mx) ;; *) echo "Invalid precision: ${PRECISION}" >&2; exit 2 ;; esac
case "${PROFILE}" in none|nsys|torch) ;; *) echo "Invalid profile: ${PROFILE}" >&2; exit 2 ;; esac
case "${CASE_FILTER}" in
    all|baseline|offload) ;;
    *) echo "Invalid case: ${CASE_FILTER}" >&2; exit 2 ;;
esac

for value_name in TRAIN_ITERS GLOBAL_BATCH_SIZE MICRO_BATCH_SIZE HYBRIDEP_NUM_SMS; do
    if ! [[ "${!value_name}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${value_name} must be a positive integer" >&2
        exit 2
    fi
done

if [[ "${DENSE_MODEL}" == 9b ]]; then
    DENSE_MODEL_NAME="qwen35_text_9b"
    DENSE_RECIPE_PREFIX="qwen35_text_9b_pretrain_4gpu_gb200"
else
    DENSE_MODEL_NAME="qwen35_text_27b"
    DENSE_RECIPE_PREFIX="qwen35_text_27b_pretrain_4gpu_gb200"
fi
EXPERT_MODEL_NAME="qwen35_text_35b_a3b"
EXPERT_RECIPE_PREFIX="qwen35_text_35b_a3b_pretrain_4gpu_gb200"
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
    local run_name
    local recipe="${recipe_prefix}_${PRECISION}_fsdp1_config"
    local num_layers="default"
    local num_experts="default"
    local -a profile_args=()
    local -a model_override_args=()

    if [[ "${CASE_FILTER}" != all && "${CASE_FILTER}" != "${case_name}" ]]; then
        return
    fi

    printf -v run_name '%s-%s-%s-r01' "${model_kind}" "${dispatcher}" "${case_name}"
    if [[ "${PROFILE}" != none ]]; then
        profile_args=(--profile "${PROFILE}")
    fi
    if [[ "${model_kind}" == expert ]]; then
        num_layers="${EXPERT_NUM_LAYERS}"
        num_experts="${EXPERT_NUM_EXPERTS}"
        model_override_args=(
            --num-layers "${EXPERT_NUM_LAYERS}"
            --num-experts "${EXPERT_NUM_EXPERTS}"
            --linear-attention-freq "${EXPERT_LINEAR_ATTENTION_FREQ}"
        )
    fi

    printf 'matrix model=%s case=%s dispatcher=%s repeat=1 layers=%s experts=%s offload=%s recompute=%s\n' \
        "${model}" "${case_name}" "${dispatcher}" \
        "${num_layers}" "${num_experts}" "${offload_modules}" "${recompute_modules}"
    if [[ "${DRY_RUN}" == true ]]; then
        return
    fi

    set +e
    "${RUN_ONE}" \
        "${profile_args[@]}" \
        "${model_override_args[@]}" \
        --model "${model}" \
        --recipe "${recipe}" \
        --precision "${PRECISION}" \
        --run-name "${run_name}" \
        --dispatcher "${dispatcher}" \
        --hybridep-num-sms "${HYBRIDEP_NUM_SMS}" \
        --recompute-granularity "${recompute_granularity}" \
        --recompute-modules "${recompute_modules}" \
        --fine-grained-offload "${fine_grained_offload}" \
        --offload-modules "${offload_modules}" \
        --train-iters "${TRAIN_ITERS}" \
        --global-batch-size "${GLOBAL_BATCH_SIZE}" \
        --micro-batch-size "${MICRO_BATCH_SIZE}" \
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
    run_case dense "${DENSE_MODEL_NAME}" "${DENSE_RECIPE_PREFIX}" default baseline null null false null
    run_case dense "${DENSE_MODEL_NAME}" "${DENSE_RECIPE_PREFIX}" default offload null null true '[mlp_norm,mlp_act]'
}

run_expert_matrix() {
    local dispatcher
    for dispatcher in alltoall hybridep; do
        run_case expert "${EXPERT_MODEL_NAME}" "${EXPERT_RECIPE_PREFIX}" "${dispatcher}" baseline null null false null
        run_case expert "${EXPERT_MODEL_NAME}" "${EXPERT_RECIPE_PREFIX}" "${dispatcher}" offload null null true '[mlp_norm,expert_fc1,moe_act]'
    done
}

export RESULTS_ROOT RUN_TIME
printf 'benchmark_id=%s results_root=%s train_iters=%s runs_per_case=1\n' \
    "${RUN_TIME}" "${RESULTS_ROOT}" "${TRAIN_ITERS}"

case "${SCOPE}" in
    all) run_dense_matrix; run_expert_matrix ;;
    dense) run_dense_matrix ;;
    expert) run_expert_matrix ;;
esac

if [[ "${DRY_RUN}" == true ]]; then
    exit 0
fi

REPORT_DIR="${RESULTS_ROOT}/benchmarks/${RUN_TIME}"
uv run --no-sync python "${SUMMARIZER}" \
    --results-root "${RESULTS_ROOT}" \
    --run-time "${RUN_TIME}" \
    --output-dir "${REPORT_DIR}"

printf 'Benchmark report: %s/summary.md\n' "${REPORT_DIR}"
if (( FAILURES > 0 )); then
    printf '%d benchmark run(s) failed; see runs.csv for status.\n' "${FAILURES}" >&2
    exit 1
fi
