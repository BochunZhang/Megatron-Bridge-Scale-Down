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
# Matrix entry point for the selective activation-offload experiments.
#
# Three groups answer three different questions:
#   a  DSW, 4 GPUs   -> where does the offload cost come from, dense vs expert,
#                       and how much of it is the HybridEP host sync.
#   b  DLC, 16 GPUs  -> the full Qwen3.5 MoE with FSDP1 and selective offload.
#   c  DLC, 32 GPUs  -> the same model under pipeline parallelism, as the
#                       reference for what offload buys at a larger scale.
#
# Every group runs matched baseline/offload pairs on an identical stack.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

RUN_ONE="${SCRIPT_DIR}/run_pretrain.sh"

GROUP="all"
FAMILY="qwen"
DTYPE="bf16"
MBS_LIST="1,2"
GBS=""
TRAIN_ITERS=10
PROFILE="none"
SYNC_SM=true
RESULTS_ROOT=""
RUN_TIME=""
DRY_RUN=false
FAILURES=0

usage() {
    cat <<'EOF'
Usage: benchmark_offload.sh [OPTIONS]

Groups:
    a   4 GPUs, dense and expert proxies, includes the HybridEP sync A/B.
    b   16 GPUs, full Qwen3.5 MoE, FSDP1 with and without selective offload.
    c   32 GPUs, the same model under PP, with and without selective offload.

Options:
    --group <a|b|c|all>       Which group to run (default: all).
    --model-family <qwen|deepseek|all>
                              Model family (default: qwen).
    --dtype <bf16|mxfp8>      Training precision (default: bf16).
    --mbs <list>              Comma-separated micro batch sizes. Defaults to
                              1,2 for group a and 1,2,4 for groups b and c.
    --gbs <n>                 Global batch size. Defaults to 32 for group a,
                              128 for group b, and 32 for group c.
    --train-iters <n>         Iterations per run (default: 10).
    --profile <none|nsys|torch>
                              Profiling backend (default: none).
    --sync-sm <true|false>    Include the HybridEP SM-sync A/B cases in group a
                              (default: true).
    --results-root <path>     Result tree root.
    --run-time <id>           Batch id; generate once and share across DLC nodes.
    --dry-run                 Print the resolved matrix without launching.
    -h, --help                Show this help.

Group c compares a 16-GPU FSDP1 run with selective offload against a 32-GPU
pipeline-parallel run of the same model, with and without offload. Use
--group a for the attribution runs and --group c for the scale comparison.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --group) GROUP="$2"; shift 2 ;;
        --model-family) FAMILY="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --mbs|--micro-batch-sizes) MBS_LIST="$2"; shift 2 ;;
        --gbs|--global-batch-size) GBS="$2"; shift 2 ;;
        --train-iters) TRAIN_ITERS="$2"; shift 2 ;;
        --profile) PROFILE="$2"; shift 2 ;;
        --sync-sm) SYNC_SM="$2"; shift 2 ;;
        --results-root) RESULTS_ROOT="$2"; shift 2 ;;
        --run-time) RUN_TIME="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

case "${GROUP}" in a|b|c|all) ;; *) die "--group must be a, b, c or all" ;; esac
case "${FAMILY}" in qwen|deepseek|all) ;; *) die "--model-family must be qwen, deepseek or all" ;; esac
case "${DTYPE}" in bf16|mxfp8) ;; *) die "--dtype must be bf16 or mxfp8" ;; esac
case "${SYNC_SM}" in true|false) ;; *) die "--sync-sm must be true or false" ;; esac
case "${PROFILE}" in none|nsys|torch) ;; *) die "--profile must be none, nsys or torch" ;; esac

IFS=',' read -r -a MBS_VALUES <<< "${MBS_LIST}"
(( ${#MBS_VALUES[@]} > 0 )) || die "--mbs needs at least one value"
for mbs in "${MBS_VALUES[@]}"; do
    [[ "${mbs}" =~ ^[1-9][0-9]*$ ]] || die "invalid micro batch size: ${mbs}"
done

RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/01-analyse/01-fine-grained-offload/02-offload-on-dense-or-expert-model}"
RUN_TIME="${RUN_TIME:-$(date +%Y%m%d-%H%M%S)}"
export RESULTS_ROOT RUN_TIME

# run_case <gpu> <pp> <ep> <dispatcher> <sharding> <run-name> <offload> <modules> <mbs> <gbs>
#          <model> <sync-sm> [extra run_pretrain.sh args...]
run_case() {
    local gpu="$1" pp="$2" ep="$3" dispatcher="$4" sharding="$5"
    local run_name="$6" offload="$7" modules="$8" mbs="$9" gbs="${10}"
    local model="${11}" sync_sm="${12}"
    shift 12
    local -a extra=("$@")
    local -a profile_args=()
    [[ "${PROFILE}" != none ]] && profile_args=(--profile "${PROFILE}")

    log "matrix group=${GROUP} model=${model} gpu=${gpu} pp=${pp} ep=${ep} dp=$(( gpu / (pp * ep) )) \
mbs=${mbs} gbs=${gbs} dispatcher=${dispatcher} sharding=${sharding} run=${run_name} \
offload=${offload} modules=${modules} sync_sm=${sync_sm}"
    if [[ "${DRY_RUN}" == true ]]; then
        return
    fi
    set +e
    "${RUN_ONE}" \
        "${profile_args[@]}" \
        --model "${model}" \
        --dtype "${DTYPE}" \
        --run-name "${run_name}" \
        --offload "${offload}" \
        --offload-modules "${modules}" \
        --gpu "${gpu}" --pp "${pp}" --ep "${ep}" \
        --dispatcher "${dispatcher}" \
        --sharding "${sharding}" \
        --sync-with-sm "${sync_sm}" \
        --mbs "${mbs}" --gbs "${gbs}" \
        --train-iters "${TRAIN_ITERS}" \
        --results-root "${RESULTS_ROOT}" --run-time "${RUN_TIME}" \
        "${extra[@]}"
    local status=$?
    set -e
    if (( status != 0 )); then
        log "run failed with status ${status}: group=${GROUP} model=${model} run=${run_name} mbs=${mbs}"
        FAILURES=$((FAILURES + 1))
    fi
}

# Group a: 4-GPU attribution. Dense and expert proxies share a 16-layer cut so the
# two columns are comparable, and HybridEP runs both with and without the host sync.
run_group_a() {
    local mbs gbs="${GBS:-32}"
    for mbs in "${MBS_VALUES[@]}"; do
        if [[ "${FAMILY}" == qwen || "${FAMILY}" == all ]]; then
            # Dense proxy: 16 layers keeps the same depth as the expert proxy.
            for run_name in baseline offload-mlp offload-attn-mlp; do
                local offload=false modules=null
                [[ "${run_name}" != baseline ]] && offload=true \
                    && modules="$(offload_modules "${run_name}" dense qwen false)"
                run_case 4 1 1 default fsdp1 "${run_name}" "${offload}" "${modules}" \
                    "${mbs}" "${gbs}" qwen35_text_9b false \
                    --num-layers 16 --mtp-num-layers 0
            done
            # Expert proxy: 16 layers, 64 experts, both dispatchers, plus the sync A/B.
            local dispatcher
            for dispatcher in alltoall hybridep; do
                for run_name in baseline offload-mlp offload-attn-mlp; do
                    local offload=false modules=null
                    [[ "${run_name}" != baseline ]] && offload=true \
                        && modules="$(offload_modules "${run_name}" expert qwen false)"
                    run_case 4 1 4 "${dispatcher}" fsdp1 "${run_name}" "${offload}" "${modules}" \
                        "${mbs}" "${gbs}" qwen35_text_35b_a3b false \
                        --num-layers 16 --num-experts 64 --linear-attention-freq 4 --mtp-num-layers 0
                done
                if [[ "${SYNC_SM}" == true && "${dispatcher}" == hybridep ]]; then
                    # Isolates the HybridEP host sync: same stack, blocking vs SM copy.
                    # Only meaningful for HybridEP; alltoall has no such sync.
                    run_case 4 1 4 "${dispatcher}" fsdp1 baseline-sync-sm false null \
                        "${mbs}" "${gbs}" qwen35_text_35b_a3b true \
                        --num-layers 16 --num-experts 64 --linear-attention-freq 4 --mtp-num-layers 0
                    run_case 4 1 4 "${dispatcher}" fsdp1 offload-mlp-sync-sm true \
                        "$(offload_modules offload-mlp expert qwen false)" \
                        "${mbs}" "${gbs}" qwen35_text_35b_a3b true \
                        --num-layers 16 --num-experts 64 --linear-attention-freq 4 --mtp-num-layers 0
                fi
            done
        fi
        if [[ "${FAMILY}" == deepseek || "${FAMILY}" == all ]]; then
            # DeepSeek-V3 proxies: four layers, 32 experts, MoE on every layer or none.
            for run_name in baseline offload-mlp offload-attn-mlp; do
                local offload=false modules=null
                [[ "${run_name}" != baseline ]] && offload=true \
                    && modules="$(offload_modules "${run_name}" dense deepseek false)"
                run_case 4 1 4 default fsdp1 "${run_name}" "${offload}" "${modules}" \
                    "${mbs}" "${gbs}" deepseek_v3 false \
                    --num-layers 4 --num-experts 32 --moe-layer-freq '[0,0,0,0]' --mtp-num-layers 0
            done
            local dispatcher
            for dispatcher in alltoall hybridep; do
                for run_name in baseline offload-mlp offload-attn-mlp; do
                    local offload=false modules=null
                    [[ "${run_name}" != baseline ]] && offload=true \
                        && modules="$(offload_modules "${run_name}" expert deepseek false)"
                    run_case 4 1 4 "${dispatcher}" fsdp1 "${run_name}" "${offload}" "${modules}" \
                        "${mbs}" "${gbs}" deepseek_v3 false \
                        --num-layers 4 --num-experts 32 --moe-layer-freq '[1,1,1,1]' --mtp-num-layers 0
                done
            done
        fi
    done
}

# Group b: 16 GPUs, the complete Qwen3.5 MoE, FSDP1, one DP group per EP group.
run_group_b() {
    [[ "${FAMILY}" == all || "${FAMILY}" == qwen ]] || return 0
    local mbs gbs="${GBS:-128}"
    for mbs in "${MBS_VALUES[@]}"; do
        local run_name offload modules
        for run_name in baseline offload-mlp offload-attn-mlp; do
            offload=false; modules=null
            [[ "${run_name}" != baseline ]] && offload=true \
                && modules="$(offload_modules "${run_name}" expert qwen false)"
            run_case 16 1 16 hybridep fsdp1 "${run_name}" "${offload}" "${modules}" \
                "${mbs}" "${gbs}" qwen35_text_35b_a3b false
        done
    done
}

# Group c: 32 GPUs. The model under PP is the reference for what offload buys at
# a larger scale, so it runs with the same model, dtype, MBS and GBS as group b.
run_group_c() {
    [[ "${FAMILY}" == all || "${FAMILY}" == qwen ]] || return 0
    local mbs gbs="${GBS:-32}"
    for mbs in "${MBS_VALUES[@]}"; do
        local run_name offload modules
        for run_name in baseline offload-mlp offload-attn-mlp; do
            offload=false; modules=null
            [[ "${run_name}" != baseline ]] && offload=true \
                && modules="$(offload_modules "${run_name}" expert qwen false)"
            # PP=2, EP=16: the expert layers split across PP stages, so each GPU
            # holds half the layers of the group-b run.
            run_case 32 2 16 hybridep none "${run_name}" "${offload}" "${modules}" \
                "${mbs}" "${gbs}" qwen35_text_35b_a3b false
        done
    done
}

log "benchmark_id=${RUN_TIME} group=${GROUP} family=${FAMILY} dtype=${DTYPE} mbs=${MBS_LIST} profile=${PROFILE}"
log "results_root=${RESULTS_ROOT}"

case "${GROUP}" in
    a) run_group_a ;;
    b) run_group_b ;;
    c) run_group_c ;;
    all) run_group_a; run_group_b; run_group_c ;;
esac

if [[ "${DRY_RUN}" == true ]]; then
    exit 0
fi
if (( FAILURES > 0 )); then
    log "${FAILURES} run(s) failed; inspect the result directories before analysing."
    exit 1
fi
log "raw results: ${RESULTS_ROOT} (run_time=${RUN_TIME})"
log "analyse with: analyse_offload_results.py --results-root ${RESULTS_ROOT}"
