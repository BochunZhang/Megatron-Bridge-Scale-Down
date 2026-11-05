#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

# Sweep cropped Qwen3.5 / Qwen3.8 cases over data parallel mode (DDP / FSDP1 / FSDP2 /
# FSDP3), TP+SP, CP, CUDA graph, sequence length, batch size and routed experts, one
# run_pretrain_fsdp3.sh launch per case.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ONE="${RUN_ONE:-${SCRIPT_DIR}/run_pretrain_fsdp3.sh}"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/01-analyse/04-fsdp3-comm}"
RUN_DATE="${RUN_DATE:-$(date +%y%m%d-%H%M%S)}"

MODELS=()
FSDP_LEVELS=(3 1)
SEQUENCES=(256 1024 4096)
TP_SIZES=(1)
CP_SIZES=(1)
CUDA_GRAPHS=(off)
MICRO_BATCH_SIZES=(1)
GLOBAL_BATCH_SIZES=()
# Microbatches per iteration; without --gbs, GBS = nmb * MBS * DP with DP = GPUS / (TP*CP).
NUM_MICROBATCHES_LIST=(4)
# Routed experts of the MoE models, multiples of EP=4 and >= top-k=10; "full" keeps all.
EXPERTS_LIST=(64 32)
# Empty: 8 layers for dense 27b, 4 for the MoE models.
NUM_LAYERS="${NUM_LAYERS:-}"
ATTENTION_FLAG=""
MTP_NUM_LAYERS="${MTP_NUM_LAYERS:-0}"
DISPATCHER="${DISPATCHER:-hybridep}"
DTYPE="${DTYPE:-bf16}"
PROFILE="${PROFILE:-nsys}"
TRAIN_ITERS="${TRAIN_ITERS:-10}"
PROFILE_START="${PROFILE_START:-7}"
PROFILE_END="${PROFILE_END:-8}"
GPUS="${GPUS:-4}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NNODES="${NNODES:-}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-${DLC_MASTER_ADDR:-}}"
MASTER_PORT="${MASTER_PORT:-29501}"
FORCE_BALANCE=false
MERGE_GRAPHS=false
DRY_RUN=false
RESUME=false

usage() {
    cat <<'USAGE'
Usage: benchmark_fsdp3.sh --model 27b|397b|2.4t[,...] [options]

Runs the Cartesian product of the list options below, one case at a time on 4 GPUs
(one GB200 tray) by default: BF16, MBS=1, 4 microbatches per iteration, no MTP,
27b cropped to 8 layers and the MoE models (EP=4, HybridEP) to 4 layers, every
4th layer gated attention and the rest GatedDeltaNet, Nsight Systems on step 7
(one report per case). Default matrix per model: FSDP 3,1 x seq 256,1024,4096,
times experts 64,32 for the MoE models. All cases are validated before the first
launch; a failed case does not stop the sweep.

Options:
  --model NAME[,NAME...]        27b, 397b, 2.4t (alias 2p4t), 122b (required)
  --fsdp ddp|1|2|3[,...]        Data parallel modes (default: 3,1); see
                                run_pretrain_fsdp3.sh --help
  --tp N[,N...]                 Tensor parallel sizes, N > 1 with sequence
                                parallelism (default: 1)
  --cp N[,N...]                 Context parallel sizes (default: 1)
  --cuda-graph off|on[,...]     CUDA graph modes (default: off); on = full-iteration
                                graph for FSDP (MoE needs hybridep), TE per-layer
                                graphs for DDP
  --seq N[,N...]                Sequence lengths (default: 256,1024,4096)
  --mbs N[,N...]                Micro batch sizes (default: 1)
  --gbs N[,N...]                Fixed global batch sizes, multiples of MBS*DP;
                                replaces --num-microbatches
  --num-microbatches N[,N...]   Microbatches per iteration (default: 4)
  --experts N|full[,...]        Routed experts for MoE models (default: 64,32)
  --num-layers N                Decoder layers (default: 8 for 27b, 4 for MoE)
  --full                        Every decoder layer gated (full) attention
  --linear                      Every decoder layer GatedDeltaNet
  --mtp-num-layers N            MTP layers (default: 0)
  --dispatcher hybridep|alltoall MoE token dispatcher (default: hybridep)
  --force-balance               Uniform random routing for MoE cases (benchmark
                                only; 27b is unaffected); names get "-fb"
  --dtype bf16|mxfp8            Compute mode (default: bf16); --mxfp8 is an alias
  --profile none|nsys           Profiling mode (default: nsys); --nsys is an alias
  --merge                       nsys draws each CUDA graph launch as one block
                                instead of its kernels (cg=on cases get "-merge")
  --profile-start N             First profiled iteration (default: 7)
  --profile-end N               Stop iteration, exclusive (default: 8)
  --train-iters N               Training iterations (default: 10)
  --gpus N                      Global GPUs (default: 4)
  --nproc-per-node N            GPUs per node (default: 4)
  --nnodes N                    Number of nodes (default: gpus / nproc-per-node)
  --node-rank N                 Node rank
  --master-addr HOST            Rendezvous address (multi-node)
  --master-port N               Rendezvous port (default: 29501)
  --run-date YYMMDD-HHMMSS      Shared output timestamp
  --output-dir PATH             Result root
  --resume                      Reuse --run-date: skip completed cases and rerun
                                failed or incomplete ones
  --dry-run                     Print cases and commands without running
USAGE
}

need_value() { [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; usage >&2; exit 2; }; }
parse_csv() { local value="$1"; IFS=',' read -r -a "$2" <<<"${value}"; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model|--models) need_value "$@"; parse_csv "$2" MODELS; shift 2;;
        --fsdp) need_value "$@"; parse_csv "$2" FSDP_LEVELS; shift 2;;
        --tp) need_value "$@"; parse_csv "$2" TP_SIZES; shift 2;;
        --cp) need_value "$@"; parse_csv "$2" CP_SIZES; shift 2;;
        --cuda-graph|--cuda-graphs) need_value "$@"; parse_csv "$2" CUDA_GRAPHS; shift 2;;
        --seq|--seq-length|--seq-lengths) need_value "$@"; parse_csv "$2" SEQUENCES; shift 2;;
        --mbs|--micro-batch-size) need_value "$@"; parse_csv "$2" MICRO_BATCH_SIZES; shift 2;;
        --gbs|--global-batch-size) need_value "$@"; parse_csv "$2" GLOBAL_BATCH_SIZES; shift 2;;
        --num-microbatches) need_value "$@"; parse_csv "$2" NUM_MICROBATCHES_LIST; shift 2;;
        --experts|--num-experts) need_value "$@"; parse_csv "$2" EXPERTS_LIST; shift 2;;
        --num-layers) need_value "$@"; NUM_LAYERS="$2"; shift 2;;
        --full) [[ "${ATTENTION_FLAG}" != --linear ]] || { echo "--full and --linear are exclusive" >&2; exit 2; }; ATTENTION_FLAG=--full; shift;;
        --linear) [[ "${ATTENTION_FLAG}" != --full ]] || { echo "--full and --linear are exclusive" >&2; exit 2; }; ATTENTION_FLAG=--linear; shift;;
        --mtp-num-layers) need_value "$@"; MTP_NUM_LAYERS="$2"; shift 2;;
        --dispatcher) need_value "$@"; DISPATCHER="$2"; shift 2;;
        --force-balance) FORCE_BALANCE=true; shift;;
        --dtype|--precision) need_value "$@"; DTYPE="$2"; shift 2;;
        --mxfp8) DTYPE=mxfp8; shift;;
        --profile) need_value "$@"; PROFILE="$2"; shift 2;;
        --nsys) PROFILE=nsys; shift;;
        --merge) MERGE_GRAPHS=true; shift;;
        --profile-start) need_value "$@"; PROFILE_START="$2"; shift 2;;
        --profile-end) need_value "$@"; PROFILE_END="$2"; shift 2;;
        --train-iters) need_value "$@"; TRAIN_ITERS="$2"; shift 2;;
        --gpus|--gpu) need_value "$@"; GPUS="$2"; shift 2;;
        --nproc-per-node) need_value "$@"; NPROC_PER_NODE="$2"; shift 2;;
        --nnodes) need_value "$@"; NNODES="$2"; shift 2;;
        --node-rank) need_value "$@"; NODE_RANK="$2"; shift 2;;
        --master-addr) need_value "$@"; MASTER_ADDR="$2"; shift 2;;
        --master-port) need_value "$@"; MASTER_PORT="$2"; shift 2;;
        --run-date) need_value "$@"; RUN_DATE="$2"; shift 2;;
        --output-dir) need_value "$@"; OUTPUT_DIR="$2"; shift 2;;
        --resume) RESUME=true; shift;;
        --dry-run) DRY_RUN=true; shift;;
        -h|--help) usage; exit 0;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2;;
    esac
done

(( ${#MODELS[@]} > 0 )) || { echo "--model is required (27b, 397b, 2.4t)" >&2; usage >&2; exit 2; }
for i in "${!MODELS[@]}"; do [[ "${MODELS[i]}" == 2p4t ]] && MODELS[i]=2.4t; done
is_positive() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }
# A fixed GBS replaces the per-case derivation from the microbatch count. The empty
# GBS list is never expanded: bash 3.2 treats that as unbound under set -u.
if (( ${#GLOBAL_BATCH_SIZES[@]} > 0 )); then BATCH_SPECS=("${GLOBAL_BATCH_SIZES[@]/#/gbs=}"); else BATCH_SPECS=("${NUM_MICROBATCHES_LIST[@]/#/nmb=}"); fi
for value in "${TP_SIZES[@]}" "${CP_SIZES[@]}" "${SEQUENCES[@]}" "${MICRO_BATCH_SIZES[@]}" "${BATCH_SPECS[@]#*=}"; do
    is_positive "${value}" || { echo "tp/cp/seq/mbs/gbs/num-microbatches values must be positive integers: ${value}" >&2; exit 2; }
done
for value in "${EXPERTS_LIST[@]}"; do
    [[ "${value}" == full ]] || is_positive "${value}" || { echo "experts must be positive integers or full: ${value}" >&2; exit 2; }
done
is_positive "${GPUS}" || { echo "gpus must be positive" >&2; exit 2; }
[[ -z "${NUM_LAYERS}" ]] || is_positive "${NUM_LAYERS}" || { echo "num-layers must be positive" >&2; exit 2; }

# case_args MODEL EXPERTS SEQ TP CP MBS GBS CUDA_GRAPH FSDP -> fills ARGS for run_pretrain_fsdp3.sh
case_args() {
    local model="$1" experts="$2" seq="$3" tp="$4" cp="$5" mbs="$6" gbs="$7" cg="$8" fsdp="$9" layers
    layers="${NUM_LAYERS:-$([[ "${model}" == 27b ]] && echo 8 || echo 4)}"
    ARGS=(
        --model "${model}" --fsdp "${fsdp}" --seq-length "${seq}" --tp "${tp}" --cp "${cp}"
        --mbs "${mbs}" --gbs "${gbs}" --train-iters "${TRAIN_ITERS}" --num-layers "${layers}"
        --mtp-num-layers "${MTP_NUM_LAYERS}" --cuda-graph "${cg}" --dtype "${DTYPE}"
        --gpu "${GPUS}" --nproc-per-node "${NPROC_PER_NODE}" --node-rank "${NODE_RANK}" --master-port "${MASTER_PORT}"
        --profile "${PROFILE}" --profile-start "${PROFILE_START}" --profile-end "${PROFILE_END}"
        --run-date "${RUN_DATE}" --results-root "${OUTPUT_DIR}"
    )
    [[ -n "${NNODES}" ]] && ARGS+=(--nnodes "${NNODES}")
    [[ -n "${MASTER_ADDR}" ]] && ARGS+=(--master-addr "${MASTER_ADDR}")
    [[ -n "${ATTENTION_FLAG}" ]] && ARGS+=("${ATTENTION_FLAG}")
    if [[ "${model}" != 27b ]]; then
        ARGS+=(--ep 4 --dispatcher "${DISPATCHER}")
        [[ "${experts}" != full ]] && ARGS+=(--num-experts "${experts}")
        [[ "${FORCE_BALANCE}" == true ]] && ARGS+=(--force-balance)
    fi
    [[ "${MERGE_GRAPHS}" == true ]] && ARGS+=(--merge)
    [[ "${RESUME}" == true ]] && ARGS+=(--resume)
    [[ "${DRY_RUN}" == true ]] && ARGS+=(--dry-run)
    return 0
}

# Expand and validate the full matrix before launching, so an invalid combination
# does not stop a long sweep halfway through. run_pretrain_fsdp3.sh --print-result-dir
# runs every check of a real launch.
CASES=()
for model in "${MODELS[@]}"; do
    if [[ "${model}" == 27b ]]; then experts_values=(0); else experts_values=("${EXPERTS_LIST[@]}"); fi
    for experts in "${experts_values[@]}"; do
        if [[ "${model}" != 27b && "${experts}" != full ]] && ! (( experts % 4 == 0 && experts >= 10 )); then
            echo "experts must be multiples of EP=4 and at least top-k=10: ${experts}" >&2; exit 2
        fi
        for seq in "${SEQUENCES[@]}"; do
            for tp in "${TP_SIZES[@]}"; do
                for cp in "${CP_SIZES[@]}"; do
                    (( GPUS % (tp * cp) == 0 )) || { echo "gpus=${GPUS} is not divisible by tp*cp=${tp}*${cp}" >&2; exit 2; }
                    dp=$((GPUS / (tp * cp)))
                    for mbs in "${MICRO_BATCH_SIZES[@]}"; do
                        for batch in "${BATCH_SPECS[@]}"; do
                            if [[ "${batch}" == gbs=* ]]; then gbs="${batch#gbs=}"; else gbs=$((${batch#nmb=} * mbs * dp)); fi
                            for cg in "${CUDA_GRAPHS[@]}"; do
                                for fsdp in "${FSDP_LEVELS[@]}"; do
                                    case_args "${model}" "${experts}" "${seq}" "${tp}" "${cp}" "${mbs}" "${gbs}" "${cg}" "${fsdp}"
                                    "${RUN_ONE}" "${ARGS[@]}" --print-result-dir >/dev/null || {
                                        echo "invalid case: model=${model} experts=${experts} seq=${seq} tp=${tp} cp=${cp} mbs=${mbs} gbs=${gbs} cuda_graph=${cg} fsdp=${fsdp}" >&2
                                        exit 2
                                    }
                                    CASES+=("${model} ${experts} ${seq} ${tp} ${cp} ${mbs} ${gbs} ${cg} ${fsdp}")
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

# A failed case is recorded and the sweep continues; an interrupt stops it.
trap 'echo "benchmark interrupted" >&2; exit 130' INT TERM
FAILED=()
for case_spec in "${CASES[@]}"; do
    read -r model experts seq tp cp mbs gbs cg fsdp <<<"${case_spec}"
    case_args "${model}" "${experts}" "${seq}" "${tp}" "${cp}" "${mbs}" "${gbs}" "${cg}" "${fsdp}"
    label="model=${model} dtype=${DTYPE} experts=${experts} fsdp=${fsdp} cuda_graph=${cg} seq=${seq} tp=${tp} cp=${cp} mbs=${mbs} gbs=${gbs}"
    printf 'launch %s\n' "${label}" >&2
    if ! "${RUN_ONE}" "${ARGS[@]}"; then
        echo "case failed: ${label}" >&2
        FAILED+=("${label}")
    fi
done

printf 'finished %d cases, %d failed\n' "${#CASES[@]}" "${#FAILED[@]}" >&2
if (( ${#FAILED[@]} > 0 )); then
    printf '  failed: %s\n' "${FAILED[@]}" >&2
    exit 1
fi
