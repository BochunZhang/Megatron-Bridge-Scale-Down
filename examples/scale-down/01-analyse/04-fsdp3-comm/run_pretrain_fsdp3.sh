#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

# Run one cropped Qwen3.5 case through scripts/training/run_recipe.py.
# The selected GB200 recipe is FSDP1 by default; the overrides below select
# Megatron FSDP3 for this communication experiment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/01-analyse/04-fsdp3-comm}"

MODEL=""
RECIPE=""
SEQ_LENGTH=""
DTYPE="${DTYPE:-bf16}"
PROFILE="${PROFILE:-none}"
CONTEXT_PARALLEL="1"
MICRO_BATCH_SIZE="1"
GLOBAL_BATCH_SIZE="32"
TRAIN_ITERS="10"
PROFILE_START="5"
PROFILE_END="8"
NUM_LAYERS=""
NUM_EXPERTS=""
MOE_LAYER_FREQ=""
LINEAR_ATTENTION_FREQ=""
MTP_NUM_LAYERS=""
GPU_COUNT="${GPU_COUNT:-4}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NNODES="${NNODES:-}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-${DLC_MASTER_ADDR:-}}"
MASTER_PORT="${MASTER_PORT:-29501}"
RUN_DATE="${RUN_DATE:-$(date +%y%m%d-%H%M%S)}"
RUN_NAME=""
DRY_RUN=false
EXTRA_OVERRIDES=()

usage() {
    cat <<'USAGE'
Usage: run_pretrain_fsdp3.sh --model 397b|122b|27b --recipe NAME --seq-length N [options]

Options:
  --cp N                         Context parallel size (default: 1)
  --micro-batch-size N           Micro batch size (default: 1)
  --global-batch-size N          Global batch size (default: 32)
  --train-iters N                Training iterations (default: 10)
  --num-layers N                 Crop transformer layers for this case
  --num-experts N                Crop routed experts for a MoE case
  --moe-layer-freq VALUE         Override model.moe_layer_freq
  --linear-attention-freq VALUE  Override model.linear_attention_freq
  --mtp-num-layers N             Override model.mtp_num_layers
  --dtype|--precision bf16|mxfp8 Compute mode (default: bf16)
  --gpu|--gpus N                 Global GPU count (default: 4)
  --nproc-per-node N              GPUs per node (default: 4)
  --nnodes N                     Number of nodes (default: gpu / nproc-per-node)
  --node-rank N                  Node rank (default: NODE_RANK or 0)
  --master-addr HOST             Multi-node rendezvous address
  --master-port N                Multi-node rendezvous port (default: 29501)
  --profile none|nsys            Profiling mode (default: none)
  --nsys                         Alias for --profile nsys
  --profile-start N              First profiler iteration (default: 5)
  --profile-end N                Last profiler iteration (default: 8)
  --run-date YYMMDD-HHMMSS       Shared matrix timestamp
  --run-name NAME                Optional human-readable case name
  --results-root PATH            Result root
  --override KEY=VALUE           Additional run_recipe.py override (repeatable)
  --dry-run                      Print the resolved command and exit
USAGE
}

need_value() { [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; usage >&2; exit 2; }; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) need_value "$@"; MODEL="$2"; shift 2;;
        --recipe) need_value "$@"; RECIPE="$2"; shift 2;;
        --seq-length) need_value "$@"; SEQ_LENGTH="$2"; shift 2;;
        --dtype|--precision) need_value "$@"; DTYPE="$2"; [[ "${DTYPE}" == fp8mx ]] && DTYPE=mxfp8; shift 2;;
        --cp) need_value "$@"; CONTEXT_PARALLEL="$2"; shift 2;;
        --micro-batch-size) need_value "$@"; MICRO_BATCH_SIZE="$2"; shift 2;;
        --global-batch-size) need_value "$@"; GLOBAL_BATCH_SIZE="$2"; shift 2;;
        --train-iters) need_value "$@"; TRAIN_ITERS="$2"; shift 2;;
        --num-layers) need_value "$@"; NUM_LAYERS="$2"; shift 2;;
        --num-experts) need_value "$@"; NUM_EXPERTS="$2"; shift 2;;
        --moe-layer-freq) need_value "$@"; MOE_LAYER_FREQ="$2"; shift 2;;
        --linear-attention-freq) need_value "$@"; LINEAR_ATTENTION_FREQ="$2"; shift 2;;
        --mtp-num-layers) need_value "$@"; MTP_NUM_LAYERS="$2"; shift 2;;
        --gpu|--gpus) need_value "$@"; GPU_COUNT="$2"; shift 2;;
        --nproc-per-node) need_value "$@"; GPUS_PER_NODE="$2"; shift 2;;
        --nnodes) need_value "$@"; NNODES="$2"; shift 2;;
        --node-rank) need_value "$@"; NODE_RANK="$2"; shift 2;;
        --master-addr) need_value "$@"; MASTER_ADDR="$2"; shift 2;;
        --master-port) need_value "$@"; MASTER_PORT="$2"; shift 2;;
        --profile) need_value "$@"; PROFILE="$2"; shift 2;;
        --nsys) PROFILE=nsys; shift;;
        --profile-start|--profile-step-start) need_value "$@"; PROFILE_START="$2"; shift 2;;
        --profile-end|--profile-step-end) need_value "$@"; PROFILE_END="$2"; shift 2;;
        --run-date) need_value "$@"; RUN_DATE="$2"; shift 2;;
        --run-name) need_value "$@"; RUN_NAME="$2"; shift 2;;
        --results-root) need_value "$@"; RESULTS_ROOT="$2"; shift 2;;
        --override) need_value "$@"; EXTRA_OVERRIDES+=("$2"); shift 2;;
        --dry-run) DRY_RUN=true; shift;;
        -h|--help) usage; exit 0;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2;;
    esac
done

[[ -n "${MODEL}" && -n "${RECIPE}" && -n "${SEQ_LENGTH}" ]] || { echo "--model, --recipe, and --seq-length are required" >&2; exit 2; }
[[ "${DTYPE}" == fp8mx ]] && DTYPE=mxfp8
case "${MODEL}" in 397b|122b|27b) ;; *) echo "Unsupported model: ${MODEL}" >&2; exit 2;; esac
case "${DTYPE}" in bf16|mxfp8) ;; *) echo "--dtype must be bf16 or mxfp8" >&2; exit 2;; esac
case "${PROFILE}" in none|nsys) ;; *) echo "--profile must be none or nsys" >&2; exit 2;; esac
for name in SEQ_LENGTH CONTEXT_PARALLEL MICRO_BATCH_SIZE GLOBAL_BATCH_SIZE TRAIN_ITERS GPU_COUNT GPUS_PER_NODE MASTER_PORT; do
    [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || { echo "${name} must be a positive integer" >&2; exit 2; }
done
for name in NODE_RANK PROFILE_START PROFILE_END; do
    [[ "${!name}" =~ ^[0-9]+$ ]] || { echo "${name} must be a non-negative integer" >&2; exit 2; }
done
[[ "${RUN_DATE}" =~ ^[0-9]{6}-[0-9]{6}$ ]] || { echo "--run-date must use yymmdd-hhmmss" >&2; exit 2; }
(( GPU_COUNT >= 2 && GPUS_PER_NODE > 0 && GPU_COUNT % GPUS_PER_NODE == 0 )) || { echo "invalid GPU count" >&2; exit 2; }
NNODES="${NNODES:-$((GPU_COUNT / GPUS_PER_NODE))}"
[[ "${NNODES}" =~ ^[1-9][0-9]*$ ]] || { echo "nnodes must be positive" >&2; exit 2; }
(( NNODES * GPUS_PER_NODE == GPU_COUNT && NODE_RANK < NNODES )) || { echo "invalid node topology" >&2; exit 2; }
(( GPU_COUNT % CONTEXT_PARALLEL == 0 )) || { echo "world size must be divisible by context parallel size" >&2; exit 2; }
(( GLOBAL_BATCH_SIZE % (MICRO_BATCH_SIZE * (GPU_COUNT / CONTEXT_PARALLEL)) == 0 )) || { echo "global batch size is not divisible by MBS*DP" >&2; exit 2; }
(( TRAIN_ITERS > 0 )) || { echo "train-iters must be positive" >&2; exit 2; }
if [[ "${PROFILE}" == nsys ]] && ! (( PROFILE_START < PROFILE_END && PROFILE_END < TRAIN_ITERS )); then
    echo "nsys requires profile-start < profile-end < train-iters" >&2; exit 2
fi
if (( NNODES > 1 )); then
    [[ -n "${MASTER_ADDR}" ]] || { echo "multi-node runs require --master-addr or MASTER_ADDR" >&2; exit 2; }
else
    MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
fi
if [[ -n "${NUM_LAYERS}" ]] && ! [[ "${NUM_LAYERS}" =~ ^[1-9][0-9]*$ ]]; then echo "num-layers must be positive" >&2; exit 2; fi
if [[ -n "${NUM_EXPERTS}" ]] && ! [[ "${NUM_EXPERTS}" =~ ^[1-9][0-9]*$ ]]; then echo "num-experts must be positive" >&2; exit 2; fi

case "${MODEL}" in
    397b) MODEL_ID=Qwen/Qwen3.5-397B-A17B;;
    122b) MODEL_ID=Qwen/Qwen3.5-122B-A10B;;
    27b) MODEL_ID=Qwen/Qwen3.5-27B;;
esac
CASE_NAME="model_${MODEL}-fsdp_3-mbs_${MICRO_BATCH_SIZE}-seq_${SEQ_LENGTH}-cp_${CONTEXT_PARALLEL}-gbs_${GLOBAL_BATCH_SIZE}-gpus_${GPU_COUNT}-profile_${PROFILE}-date_${RUN_DATE}"
RESULT_DIR="${RESULTS_ROOT}/${CASE_NAME}/node${NODE_RANK}"
NUM_MICROBATCHES=$((GLOBAL_BATCH_SIZE / (MICRO_BATCH_SIZE * (GPU_COUNT / CONTEXT_PARALLEL))))

# torchrun creates LOCAL_RANK before this worker wrapper is invoked. Prefer a
# target-provided numarun function/command, then the repository wrapper, then
# a simple numactl binding; finally run without binding when neither exists.
NUMA_WRAPPER=()
NUMA_MODE=unbound
if command -v numarun >/dev/null 2>&1; then
    NUMA_WRAPPER=(numarun); NUMA_MODE=numarun
elif [[ -r "${REPO_ROOT}/.cache/numarun" ]] && command -v numactl >/dev/null 2>&1; then
    NUMA_WRAPPER=(bash "${REPO_ROOT}/.cache/numarun"); NUMA_MODE=repo-numarun
elif command -v numactl >/dev/null 2>&1; then
    NUMA_WRAPPER=(numactl --localalloc); NUMA_MODE=numactl
fi

# Single-node four-GPU Gloo uses the GB200 host interface. DLC multi-node
# launchers own their network environment and do not receive this override.
if (( NNODES == 1 && GPU_COUNT == 4 )); then export GLOO_SOCKET_IFNAME=eth0; else unset GLOO_SOCKET_IFNAME || true; fi

PROFILE_RANKS=0
for (( rank = 1; rank < GPU_COUNT; rank++ )); do PROFILE_RANKS+=",${rank}"; done
OVERRIDES=(
    "model.seq_length=${SEQ_LENGTH}" "dataset.seq_length=${SEQ_LENGTH}" "dataset.blend=null"
    "train.train_iters=${TRAIN_ITERS}" "train.global_batch_size=${GLOBAL_BATCH_SIZE}" "train.micro_batch_size=${MICRO_BATCH_SIZE}"
    "scheduler.lr_warmup_iters=1" "scheduler.lr_decay_iters=${TRAIN_ITERS}"
    "validation.eval_iters=0" "validation.eval_interval=0"
    "model.context_parallel_size=${CONTEXT_PARALLEL}" "model.tensor_model_parallel_size=1" "model.pipeline_model_parallel_size=1"
    "model.expert_tensor_parallel_size=1" "model.sequence_parallel=false"
    "model.cp_comm_type=p2p" "model.mtp_num_layers=0" "model.linear_attention_freq=4"
    "model.recompute_granularity=null" "model.recompute_method=null" "model.recompute_num_layers=null"
    "model.recompute_modules=[]" "model.fine_grained_activation_offloading=false" "model.offload_modules=[]"
    "model.cuda_graph_impl=none" "model.cuda_graph_scope=null" "model.cuda_graph_modules=[]"
    "dist.use_megatron_fsdp=true" "ddp.use_megatron_fsdp=true" "ddp.use_distributed_optimizer=true"
    "ddp.data_parallel_sharding_strategy=optim_grads_params" "ddp.average_in_collective=false"
    "ddp.overlap_grad_reduce=true" "ddp.overlap_param_gather=true" "ddp.grad_reduce_in_fp32=true"
    "checkpoint.ckpt_format=fsdp_dtensor" "checkpoint.save=null" "checkpoint.load=null"
    "logger.log_interval=1" "logger.tensorboard_dir=null" "logger.save_config_filepath=${RESULT_DIR}/config.yaml"
    "profiling.use_nsys_profiler=$([[ "${PROFILE}" == nsys ]] && echo true || echo false)"
    "profiling.use_pytorch_profiler=false" "profiling.profile_step_start=${PROFILE_START}" "profiling.profile_step_end=${PROFILE_END}"
    "profiling.profile_ranks=[${PROFILE_RANKS}]"
    "profiling.nvtx_ranges=$([[ "${PROFILE}" == nsys ]] && echo true || echo false)"
)
[[ -n "${NUM_LAYERS}" ]] && OVERRIDES+=("model.num_layers=${NUM_LAYERS}")
[[ -n "${NUM_EXPERTS}" ]] && OVERRIDES+=("model.num_moe_experts=${NUM_EXPERTS}")
[[ -n "${MOE_LAYER_FREQ}" ]] && OVERRIDES+=("model.moe_layer_freq=${MOE_LAYER_FREQ}")
[[ -n "${LINEAR_ATTENTION_FREQ}" ]] && OVERRIDES+=("model.linear_attention_freq=${LINEAR_ATTENTION_FREQ}")
[[ -n "${MTP_NUM_LAYERS}" ]] && OVERRIDES+=("model.mtp_num_layers=${MTP_NUM_LAYERS}")
if [[ "${MODEL}" != 27b ]]; then
    OVERRIDES+=("model.moe_token_dispatcher_type=alltoall" "model.moe_flex_dispatcher_backend=null")
fi
if [[ "${DTYPE}" == mxfp8 ]]; then
    OVERRIDES+=("mixed_precision.fp8=e4m3" "mixed_precision.fp8_recipe=mxfp8" "mixed_precision.fp8_param_gather=false" "mixed_precision.reuse_grad_buf_for_mxfp8_param_ag=false")
    [[ "${MODEL}" != 27b ]] && OVERRIDES+=("model.moe_router_padding_for_fp8=true")
fi
if (( ${#EXTRA_OVERRIDES[@]} > 0 )); then
    OVERRIDES+=("${EXTRA_OVERRIDES[@]}")
fi

TRAIN_ARGS=("${REPO_ROOT}/scripts/training/run_recipe.py" --recipe "${RECIPE}" --mode pretrain --dataset mock --step-func llm_step "${OVERRIDES[@]}")
if (( NNODES == 1 )); then
    DIST_ARGS=(--standalone --nproc_per_node="${GPUS_PER_NODE}" --master_port="${MASTER_PORT}")
else
    DIST_ARGS=(--nproc_per_node="${GPUS_PER_NODE}" --nnodes="${NNODES}" --node_rank="${NODE_RANK}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}")
fi
WORKER=(uv run --no-sync python)
if [[ "${PROFILE}" == nsys ]]; then
    WORKER=(nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true --output="${RESULT_DIR}/nsys-rank%q{LOCAL_RANK}" uv run --no-sync python)
fi
COMMAND=(uv run --no-sync python -m torch.distributed.run "${DIST_ARGS[@]}" --no-python)
if (( ${#NUMA_WRAPPER[@]} > 0 )); then
    COMMAND+=("${NUMA_WRAPPER[@]}")
fi
COMMAND+=("${WORKER[@]}" "${TRAIN_ARGS[@]}")

if [[ "${DRY_RUN}" == true ]]; then
    printf 'model=%s recipe=%s dtype=%s seq=%s cp=%s mbs=%s gbs=%s gpus=%s num_microbatches=%s numa=%s\n' "${MODEL}" "${RECIPE}" "${DTYPE}" "${SEQ_LENGTH}" "${CONTEXT_PARALLEL}" "${MICRO_BATCH_SIZE}" "${GLOBAL_BATCH_SIZE}" "${GPU_COUNT}" "${NUM_MICROBATCHES}" "${NUMA_MODE}" >&2
    printf 'GLOO_SOCKET_IFNAME=%s\ncommand:' "${GLOO_SOCKET_IFNAME:-<unset>}" >&2
    printf ' %q' "${COMMAND[@]}" >&2; printf '\n' >&2
    exit 0
fi
mkdir -p "${RESULT_DIR}"
[[ ! -e "${RESULT_DIR}/launch.json" ]] || { echo "result already exists: ${RESULT_DIR}" >&2; exit 2; }
printf 'command:' >"${RESULT_DIR}/command.txt"
printf ' %q' "${COMMAND[@]}" >>"${RESULT_DIR}/command.txt"
printf '\n' >>"${RESULT_DIR}/command.txt"
cat >"${RESULT_DIR}/launch.json" <<JSON
{"model":"${MODEL_ID}","recipe":"${RECIPE}","run_name":"${RUN_NAME}","dtype":"${DTYPE}","fsdp":3,"seq_length":${SEQ_LENGTH},"context_parallel_size":${CONTEXT_PARALLEL},"micro_batch_size":${MICRO_BATCH_SIZE},"global_batch_size":${GLOBAL_BATCH_SIZE},"num_microbatches":${NUM_MICROBATCHES},"gpu_count":${GPU_COUNT},"nnodes":${NNODES},"node_rank":${NODE_RANK},"numa_mode":"${NUMA_MODE}","run_date":"${RUN_DATE}"}
JSON
printf 'model=%s dtype=%s seq=%s cp=%s mbs=%s gbs=%s gpus=%s num_microbatches=%s numa=%s\n' "${MODEL}" "${DTYPE}" "${SEQ_LENGTH}" "${CONTEXT_PARALLEL}" "${MICRO_BATCH_SIZE}" "${GLOBAL_BATCH_SIZE}" "${GPU_COUNT}" "${NUM_MICROBATCHES}" "${NUMA_MODE}" | tee "${RESULT_DIR}/run_info.txt"
set +e
"${COMMAND[@]}" 2>&1 | tee "${RESULT_DIR}/train.log"
STATUS=${PIPESTATUS[0]}
set -e
printf 'status=%s\n' "${STATUS}" >>"${RESULT_DIR}/run_info.txt"
exit "${STATUS}"
