#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

# Run the Qwen3.5 / Qwen3.8 FSDP3 communication cases one at a time.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ONE="${RUN_ONE:-${SCRIPT_DIR}/run_pretrain_fsdp3.sh}"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/01-analyse/04-fsdp3-comm}"
RUN_DATE="${RUN_DATE:-$(date +%y%m%d-%H%M%S)}"

# Representative cases: "model cp seq mbs". Dense, MoE (EP=4) and the expert-cropped
# Qwen3.8 2.4T crossed with short sequence, long sequence without CP, and long
# sequence with CP.
REPRESENTATIVE_CASES=(
    "27b 1 4096 1" "27b 1 32768 1" "27b 4 16384 1"
    "397b 1 4096 1" "397b 1 32768 1" "397b 4 16384 1"
    "2p4t 1 4096 1" "2p4t 1 32768 1" "2p4t 4 16384 1"
)
MODELS=(397b 27b 2p4t)
SEQUENCES=(4096 16384 32768)
MICRO_BATCH_SIZES=(1 2 4)
CONTEXT_PARALLEL_SIZES=(1 4)
CUDA_GRAPHS=(off on)
CUSTOM_MATRIX=false
DTYPE="${DTYPE:-bf16}"
PROFILE="${PROFILE:-nsys}"
TRAIN_ITERS="${TRAIN_ITERS:-10}"
NUM_LAYERS="${NUM_LAYERS:-4}"
MTP_NUM_LAYERS="${MTP_NUM_LAYERS:-1}"
# Qwen3.8 2.4T routed experts: empty = halve from MAX_EXPERTS_2P4T on OOM.
EXPERTS_2P4T="${EXPERTS_2P4T:-}"
MAX_EXPERTS_2P4T="${MAX_EXPERTS_2P4T:-512}"
MOE_EP=4
MOE_TOPK_2P4T=10
# Microbatches per iteration; GBS is derived per case as nmb * MBS * DP.
NUM_MICROBATCHES_LIST=(4)
# A fixed GBS (--global-batch-size) replaces the per-case derivation.
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-}"
GPUS="${GPUS:-4}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NNODES="${NNODES:-}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-${DLC_MASTER_ADDR:-}}"
MASTER_PORT="${MASTER_PORT:-29501}"
DISPATCHER="${DISPATCHER:-hybridep}"
PROFILE_START="${PROFILE_START:-7}"
PROFILE_END="${PROFILE_END:-8}"
DRY_RUN=false
RESUME=false

usage() {
    cat <<'USAGE'
Usage: benchmark_fsdp3.sh [options]

By default nine representative cases run with CUDA graph off and on (18 runs,
plus the expert-halving retries of 2p4t), 4 microbatches per iteration, MBS=1,
BF16 (GBS = microbatches * MBS * DP, i.e. 16 at DP=4):
  27b  (Qwen3.5-27B, dense)              seq=4096 CP=1 | seq=32768 CP=1 | seq=16384 CP=4
  397b (Qwen3.5-397B-A17B, MoE, EP=4)    same three
  2p4t (Qwen3.8-2.4T-A95B, MoE, EP=4)    same three
Every model is cropped to 4 layers (3 GatedDeltaNet + 1 gated attention) plus one
MTP layer. 397b keeps all 512 routed experts. 2p4t starts at 512 routed experts
and halves the count after an out-of-memory failure until the case trains; both
CUDA graph modes of a case use the same expert count.
Passing any of --model/--seq/--cp/--mbs switches to the Cartesian matrix of
models=397b,27b,2p4t; CP=1,4; seq=4096,16384,32768; MBS=1,2,4, restricted by the
given options. All cases are validated before the first launch; a failed case
does not stop the sweep.

Options:
  --model NAME[,NAME...]        Select models: 397b,122b,27b,2p4t or all (or --models)
  --seq N[,N...]                Select sequence lengths (or --seq-lengths)
  --cp N[,N...]                 Select context parallel sizes
  --mbs N[,N...]                Select micro batch sizes
  --cuda-graph off|on[,...]     CUDA graph modes per case (default: off,on)
  --num-layers N                Decoder layers (default: 4)
  --mtp-num-layers N            MTP layers (default: 1)
  --experts-2p4t N              Fix the Qwen3.8 2.4T routed-expert count (no search)
  --max-experts-2p4t N          First count of the halving search (default: 512)
  --dtype|--precision bf16|mxfp8
  --dispatcher hybridep|alltoall MoE token dispatcher (default: hybridep)
  --profile none|nsys            Profiling mode (default: nsys)
  --nsys                         Alias for --profile nsys
  --train-iters N               Training iterations
  --num-microbatches N[,N...]   Microbatches per iteration (default: 4)
  --global-batch-size N         Fixed global batch size; overrides
                                --num-microbatches
  --gpus N                      Global GPUs (default: 4)
  --nproc-per-node N            GPUs per node
  --nnodes N                    Number of nodes
  --node-rank N                 Node rank
  --master-addr HOST            Rendezvous address
  --master-port N               Rendezvous port
  --profile-start N             First profiled iteration (default: 7)
  --profile-end N               Stop iteration, exclusive (default: 8)
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
        --cuda-graph|--cuda-graphs) need_value "$@"; parse_csv "$2" CUDA_GRAPHS; shift 2;;
        --num-layers) need_value "$@"; NUM_LAYERS="$2"; shift 2;;
        --mtp-num-layers) need_value "$@"; MTP_NUM_LAYERS="$2"; shift 2;;
        --experts-2p4t) need_value "$@"; EXPERTS_2P4T="$2"; shift 2;;
        --max-experts-2p4t) need_value "$@"; MAX_EXPERTS_2P4T="$2"; shift 2;;
        --dtype|--precision) need_value "$@"; DTYPE="$2"; [[ "${DTYPE}" == fp8mx ]] && DTYPE=mxfp8; shift 2;;
        --profile) need_value "$@"; PROFILE="$2"; shift 2;;
        --dispatcher) need_value "$@"; DISPATCHER="$2"; shift 2;;
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
[[ "${NUM_LAYERS}" =~ ^[1-9][0-9]*$ ]] || { echo "num-layers must be positive" >&2; exit 2; }
[[ "${MTP_NUM_LAYERS}" =~ ^[0-9]+$ ]] || { echo "mtp-num-layers must be a non-negative integer" >&2; exit 2; }
for cg in "${CUDA_GRAPHS[@]}"; do
    case "${cg}" in off|on) ;; *) echo "--cuda-graph values must be off or on" >&2; exit 2;; esac
done
for experts in "${EXPERTS_2P4T}" "${MAX_EXPERTS_2P4T}"; do
    [[ -z "${experts}" ]] && continue
    [[ "${experts}" =~ ^[1-9][0-9]*$ ]] && (( experts % MOE_EP == 0 && experts >= MOE_TOPK_2P4T )) || {
        echo "2p4t expert counts must be multiples of EP=${MOE_EP} and at least top-k=${MOE_TOPK_2P4T}" >&2; exit 2
    }
done
if [[ -n "${GLOBAL_BATCH_SIZE}" ]]; then
    [[ "${GLOBAL_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || { echo "global-batch-size must be positive" >&2; exit 2; }
else
    for nmb in "${NUM_MICROBATCHES_LIST[@]}"; do
        [[ "${nmb}" =~ ^[1-9][0-9]*$ ]] || { echo "num-microbatches must be positive integers" >&2; exit 2; }
    done
fi

if (( ${#MODELS[@]} == 1 )) && [[ "${MODELS[0]}" == all ]]; then
    MODELS=(397b 122b 27b 2p4t)
fi
for i in "${!MODELS[@]}"; do [[ "${MODELS[i]}" == 2.4t ]] && MODELS[i]=2p4t; done
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
        2p4t) echo qwen38_text_2p4t_a95b_pretrain_4gpu_gb200_bf16_fsdp1_config;;
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
        (( GPUS % MOE_EP == 0 )) || { echo "MoE cases use EP=${MOE_EP}; gpus=${GPUS} must be divisible by ${MOE_EP}" >&2; exit 2; }
    fi
    # Each node decides the next expert count from its own train.log, so an
    # automatic search is only consistent on a single node.
    if [[ "${model}" == 2p4t && -z "${EXPERTS_2P4T}" ]] && (( NNODES > 1 )); then
        echo "multi-node 2p4t runs require --experts-2p4t (the OOM halving search is single-node only)" >&2; exit 2
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

MOE_LAYER_FREQ="[1"
for (( i = 1; i < NUM_LAYERS; i++ )); do MOE_LAYER_FREQ+=",1"; done
MOE_LAYER_FREQ+="]"
SEARCH_LOG="${OUTPUT_DIR}/expert_search-date_${RUN_DATE}-node${NODE_RANK}.tsv"

# case_args MODEL CP SEQ MBS GBS CUDA_GRAPH [EXPERTS] -> fills ARGS for run_pretrain_fsdp3.sh
case_args() {
    local model="$1" cp="$2" seq="$3" mbs="$4" gbs="$5" cg="$6" experts="${7:-}"
    ARGS=(
        --model "${model}" --recipe "$(recipe_for_model "${model}")" --seq-length "${seq}" --cp "${cp}"
        --micro-batch-size "${mbs}" --global-batch-size "${gbs}" --train-iters "${TRAIN_ITERS}"
        --num-layers "${NUM_LAYERS}" --gpu "${GPUS}" --nproc-per-node "${NPROC_PER_NODE}" --nnodes "${NNODES}"
        --node-rank "${NODE_RANK}" --master-port "${MASTER_PORT}" --dtype "${DTYPE}" --profile "${PROFILE}"
        --linear-attention-freq 4 --mtp-num-layers "${MTP_NUM_LAYERS}" --cuda-graph "${cg}"
        --profile-start "${PROFILE_START}" --profile-end "${PROFILE_END}" --run-date "${RUN_DATE}" --results-root "${OUTPUT_DIR}"
    )
    (( NNODES > 1 )) && ARGS+=(--master-addr "${MASTER_ADDR}")
    if [[ "${model}" != 27b ]]; then
        ARGS+=(--ep "${MOE_EP}" --dispatcher "${DISPATCHER}" --moe-layer-freq "${MOE_LAYER_FREQ}")
        [[ -n "${experts}" ]] && ARGS+=(--num-experts "${experts}")
    fi
    [[ "${RESUME}" == true ]] && ARGS+=(--resume)
    [[ "${DRY_RUN}" == true ]] && ARGS+=(--dry-run)
    return 0
}

# run_case ARGS... -> 0 trained, 3 out of memory, otherwise the launcher status.
run_case() {
    local result_dir status=0
    result_dir="$("${RUN_ONE}" "$@" --print-result-dir)"
    # --resume: a previous out-of-memory attempt is reused instead of rerun.
    if [[ "${RESUME}" == true && "${DRY_RUN}" == false ]] && grep -q '^status=[1-9]' "${result_dir}/run_info.txt" 2>/dev/null \
        && grep -qiE 'out of memory|OutOfMemoryError' "${result_dir}/train.log" 2>/dev/null; then
        echo "reuse out-of-memory attempt: ${result_dir}" >&2
        return 3
    fi
    "${RUN_ONE}" "$@" || status=$?
    if (( status != 0 )) && grep -qiE 'out of memory|OutOfMemoryError' "${result_dir}/train.log" 2>/dev/null; then
        return 3
    fi
    return "${status}"
}

# A failed case is recorded and the sweep continues; an interrupt stops it.
trap 'echo "benchmark interrupted" >&2; exit 130' INT TERM
FAILED=()
for case_spec in "${CASES[@]}"; do
    read -r model cp seq mbs gbs <<<"${case_spec}"
    if [[ "${model}" != 2p4t ]]; then
        for cg in "${CUDA_GRAPHS[@]}"; do
            case_args "${model}" "${cp}" "${seq}" "${mbs}" "${gbs}" "${cg}"
            printf 'launch model=%s cp=%s seq=%s mbs=%s gbs=%s cuda_graph=%s\n' "${model}" "${cp}" "${seq}" "${mbs}" "${gbs}" "${cg}" >&2
            if ! run_case "${ARGS[@]}"; then
                echo "case failed: model=${model} cp=${cp} seq=${seq} mbs=${mbs} gbs=${gbs} cuda_graph=${cg}" >&2
                FAILED+=("model=${model} cp=${cp} seq=${seq} mbs=${mbs} gbs=${gbs} cuda_graph=${cg}")
            fi
        done
        continue
    fi

    # Qwen3.8 2.4T: run every CUDA graph mode at one expert count, CUDA graph on
    # first (more memory). Any out-of-memory failure halves the count and reruns
    # all modes, so the modes of a case stay comparable.
    MODES=()
    for cg in on off; do [[ " ${CUDA_GRAPHS[*]} " == *" ${cg} "* ]] && MODES+=("${cg}"); done
    experts="${EXPERTS_2P4T:-${MAX_EXPERTS_2P4T}}"
    tried=()
    while :; do
        outcome=ok
        for cg in "${MODES[@]}"; do
            case_args "${model}" "${cp}" "${seq}" "${mbs}" "${gbs}" "${cg}" "${experts}"
            printf 'launch model=%s experts=%s cp=%s seq=%s mbs=%s gbs=%s cuda_graph=%s\n' "${model}" "${experts}" "${cp}" "${seq}" "${mbs}" "${gbs}" "${cg}" >&2
            status=0
            run_case "${ARGS[@]}" || status=$?
            if (( status == 3 )); then outcome=oom; break; fi
            (( status == 0 )) || outcome=failed
        done
        tried+=("${experts}:${outcome}")
        next=$((experts / 2))
        if [[ "${outcome}" == oom && -z "${EXPERTS_2P4T}" ]] && (( next >= MOE_TOPK_2P4T && next % MOE_EP == 0 )); then
            echo "out of memory with ${experts} experts; retry with ${next}" >&2
            experts="${next}"
            continue
        fi
        break
    done
    if [[ "${DRY_RUN}" == false ]]; then
        mkdir -p "${OUTPUT_DIR}"
        printf 'model=%s\tcp=%s\tseq=%s\tmbs=%s\tgbs=%s\tcuda_graph=%s\texperts=%s\toutcome=%s\ttried=%s\n' \
            "${model}" "${cp}" "${seq}" "${mbs}" "${gbs}" "$(IFS=,; echo "${MODES[*]}")" "${experts}" "${outcome}" "$(IFS=,; echo "${tried[*]}")" >>"${SEARCH_LOG}"
    fi
    if [[ "${outcome}" != ok ]]; then
        echo "case failed: model=${model} experts=${experts} cp=${cp} seq=${seq} mbs=${mbs} gbs=${gbs} (${outcome})" >&2
        FAILED+=("model=${model} experts=${experts} cp=${cp} seq=${seq} mbs=${mbs} gbs=${gbs} ${outcome}")
    fi
done

printf 'finished %d cases, %d failed\n' "${#CASES[@]}" "${#FAILED[@]}" >&2
if (( ${#FAILED[@]} > 0 )); then
    printf '  failed: %s\n' "${FAILED[@]}" >&2
    exit 1
fi
