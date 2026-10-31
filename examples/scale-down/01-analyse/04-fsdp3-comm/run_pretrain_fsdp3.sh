#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

# Run one cropped Qwen3.5 / Qwen3.8 case through scripts/training/run_recipe.py.
# The selected GB200 recipe is FSDP1 by default; the overrides below select
# Megatron FSDP3 (default) or keep FSDP1 with --fsdp 1.
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
GLOBAL_BATCH_SIZE="16"
TRAIN_ITERS="10"
PROFILE_START="7"
PROFILE_END="8"
NUM_LAYERS=""
NUM_EXPERTS=""
EXPERT_PARALLEL=""
MOE_LAYER_FREQ=""
LINEAR_ATTENTION_FREQ=""
MTP_NUM_LAYERS=""
DISPATCHER="${DISPATCHER:-hybridep}"
CUDA_GRAPH="${CUDA_GRAPH:-off}"
FSDP_LEVEL="${FSDP_LEVEL:-3}"
HYBRIDEP_NUM_SMS="${HYBRIDEP_NUM_SMS:-32}"
GPU_COUNT="${GPU_COUNT:-4}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NNODES="${NNODES:-}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-${DLC_MASTER_ADDR:-}}"
MASTER_PORT="${MASTER_PORT:-29501}"
RUN_DATE="${RUN_DATE:-$(date +%y%m%d-%H%M%S)}"
RUN_NAME=""
DRY_RUN=false
PRINT_RESULT_DIR=false
RESUME=false
EXTRA_OVERRIDES=()

usage() {
    cat <<'USAGE'
Usage: run_pretrain_fsdp3.sh --model 397b|122b|27b|2p4t --recipe NAME --seq-length N [options]

Models: 397b/122b/27b are Qwen3.5-397B-A17B / 122B-A10B / 27B; 2p4t (alias 2.4t)
is Qwen3.8-2.4T-A95B.

Options:
  --cp N                         Context parallel size (default: 1)
  --micro-batch-size N           Micro batch size (default: 1)
  --global-batch-size N          Global batch size (default: 16)
  --train-iters N                Training iterations (default: 10)
  --num-layers N                 Crop transformer layers for this case
  --num-experts N                Crop routed experts for a MoE case
  --ep N                         Expert parallel size (default: 4 for MoE,
                                 1 for dense; dense models only accept 1)
  --moe-layer-freq VALUE         Override model.moe_layer_freq
  --linear-attention-freq VALUE  Override model.linear_attention_freq
  --mtp-num-layers N             Override model.mtp_num_layers
  --dispatcher hybridep|alltoall MoE token dispatcher (default: hybridep)
  --hybridep-num-sms N           SMs for HybridEP dispatch (default: 32)
  --cuda-graph off|on            off: no CUDA graph. on: Transformer Engine
                                 per-layer graphs, attn+mlp for dense models and
                                 the partial attn+moe_router+moe_preprocess scope
                                 for MoE models (default: off)
  --fsdp 3|1                     3: Megatron FSDP optim_grads_params with prefetch and
                                 overlapped reduce-scatter. 1: the recipe's optim
                                 strategy (only optimizer state sharded; gradient
                                 reduce-scatter and parameter all-gather once per
                                 iteration, not overlapped) (default: 3)
  --dtype|--precision bf16|mxfp8 Compute mode (default: bf16)
  --gpu|--gpus N                 Global GPU count (default: 4)
  --nproc-per-node N              GPUs per node (default: 4)
  --nnodes N                     Number of nodes (default: gpu / nproc-per-node)
  --node-rank N                  Node rank (default: NODE_RANK or 0)
  --master-addr HOST             Multi-node rendezvous address
  --master-port N                Multi-node rendezvous port (default: 29501)
  --profile none|nsys            Profiling mode (default: none)
  --nsys                         Alias for --profile nsys
  --profile-start N              First profiled iteration (default: 7)
  --profile-end N                Stop iteration, exclusive (default: 8, i.e. one step)
  --run-date YYMMDD-HHMMSS       Shared matrix timestamp
  --run-name NAME                Optional human-readable case name
  --results-root PATH            Result root
  --override KEY=VALUE           Additional run_recipe.py override (repeatable)
  --resume                       Skip a completed case; archive a failed or
                                 incomplete result and rerun it
  --dry-run                      Print the resolved command and exit
  --print-result-dir             Print this node's result directory and exit
USAGE
}

need_value() { [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; usage >&2; exit 2; }; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) need_value "$@"; MODEL="$2"; [[ "${MODEL}" == 2.4t ]] && MODEL=2p4t; shift 2;;
        --recipe) need_value "$@"; RECIPE="$2"; shift 2;;
        --seq-length) need_value "$@"; SEQ_LENGTH="$2"; shift 2;;
        --dtype|--precision) need_value "$@"; DTYPE="$2"; [[ "${DTYPE}" == fp8mx ]] && DTYPE=mxfp8; shift 2;;
        --cp) need_value "$@"; CONTEXT_PARALLEL="$2"; shift 2;;
        --micro-batch-size) need_value "$@"; MICRO_BATCH_SIZE="$2"; shift 2;;
        --global-batch-size) need_value "$@"; GLOBAL_BATCH_SIZE="$2"; shift 2;;
        --train-iters) need_value "$@"; TRAIN_ITERS="$2"; shift 2;;
        --num-layers) need_value "$@"; NUM_LAYERS="$2"; shift 2;;
        --num-experts) need_value "$@"; NUM_EXPERTS="$2"; shift 2;;
        --ep|--expert-parallel) need_value "$@"; EXPERT_PARALLEL="$2"; shift 2;;
        --moe-layer-freq) need_value "$@"; MOE_LAYER_FREQ="$2"; shift 2;;
        --linear-attention-freq) need_value "$@"; LINEAR_ATTENTION_FREQ="$2"; shift 2;;
        --mtp-num-layers) need_value "$@"; MTP_NUM_LAYERS="$2"; shift 2;;
        --dispatcher) need_value "$@"; DISPATCHER="$2"; shift 2;;
        --hybridep-num-sms) need_value "$@"; HYBRIDEP_NUM_SMS="$2"; shift 2;;
        --cuda-graph) need_value "$@"; CUDA_GRAPH="$2"; shift 2;;
        --fsdp) need_value "$@"; FSDP_LEVEL="$2"; shift 2;;
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
        --resume) RESUME=true; shift;;
        --dry-run) DRY_RUN=true; shift;;
        --print-result-dir) PRINT_RESULT_DIR=true; shift;;
        -h|--help) usage; exit 0;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2;;
    esac
done

[[ -n "${MODEL}" && -n "${RECIPE}" && -n "${SEQ_LENGTH}" ]] || { echo "--model, --recipe, and --seq-length are required" >&2; exit 2; }
[[ "${DTYPE}" == fp8mx ]] && DTYPE=mxfp8
case "${MODEL}" in 397b|122b|27b|2p4t) ;; *) echo "Unsupported model: ${MODEL}" >&2; exit 2;; esac
case "${DTYPE}" in bf16|mxfp8) ;; *) echo "--dtype must be bf16 or mxfp8" >&2; exit 2;; esac
case "${PROFILE}" in none|nsys) ;; *) echo "--profile must be none or nsys" >&2; exit 2;; esac
case "${DISPATCHER}" in hybridep|alltoall) ;; *) echo "--dispatcher must be hybridep or alltoall" >&2; exit 2;; esac
case "${CUDA_GRAPH}" in off|on) ;; *) echo "--cuda-graph must be off or on" >&2; exit 2;; esac
# FSDP1 keeps the recipe's settings: with the optim strategy Megatron-FSDP delays the
# gradient reduction to the end of the iteration, so there is nothing to overlap.
case "${FSDP_LEVEL}" in
    3) SHARDING_STRATEGY=optim_grads_params; FSDP_OVERLAP=true;;
    1) SHARDING_STRATEGY=optim; FSDP_OVERLAP=false;;
    *) echo "--fsdp must be 1 or 3" >&2; exit 2;;
esac
for name in SEQ_LENGTH CONTEXT_PARALLEL MICRO_BATCH_SIZE GLOBAL_BATCH_SIZE TRAIN_ITERS GPU_COUNT GPUS_PER_NODE MASTER_PORT HYBRIDEP_NUM_SMS; do
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
(( SEQ_LENGTH % (2 * CONTEXT_PARALLEL) == 0 )) || { echo "seq-length must be divisible by 2*cp" >&2; exit 2; }
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
if [[ -n "${MTP_NUM_LAYERS}" ]] && ! [[ "${MTP_NUM_LAYERS}" =~ ^[0-9]+$ ]]; then echo "mtp-num-layers must be a non-negative integer" >&2; exit 2; fi
# MoE cases always use expert parallelism; the EP group folds onto the same
# GPUs as the DP x CP mesh, so EP only needs to divide the world size.
if [[ "${MODEL}" == 27b ]]; then
    EXPERT_PARALLEL="${EXPERT_PARALLEL:-1}"
    [[ "${EXPERT_PARALLEL}" == 1 ]] || { echo "dense model 27b only supports --ep 1" >&2; exit 2; }
    DISPATCHER=none
else
    EXPERT_PARALLEL="${EXPERT_PARALLEL:-4}"
    [[ "${EXPERT_PARALLEL}" =~ ^[1-9][0-9]*$ ]] || { echo "ep must be a positive integer" >&2; exit 2; }
    (( GPU_COUNT % EXPERT_PARALLEL == 0 )) || { echo "world size must be divisible by ep" >&2; exit 2; }
    if [[ "${DISPATCHER}" == hybridep ]]; then
        (( EXPERT_PARALLEL <= 72 )) || { echo "HybridEP ep must fit in the 72-rank NVLink domain" >&2; exit 2; }
    fi
    if [[ -n "${NUM_EXPERTS}" ]]; then
        (( NUM_EXPERTS % EXPERT_PARALLEL == 0 )) || { echo "num-experts must be divisible by ep" >&2; exit 2; }
    fi
fi

case "${MODEL}" in
    397b) MODEL_ID=Qwen/Qwen3.5-397B-A17B; MODEL_TAG=qwen35_397b;;
    122b) MODEL_ID=Qwen/Qwen3.5-122B-A10B; MODEL_TAG=qwen35_122b;;
    27b) MODEL_ID=Qwen/Qwen3.5-27B; MODEL_TAG=qwen35_27b;;
    2p4t) MODEL_ID=Qwen/Qwen3.8-2.4T-A95B; MODEL_TAG=qwen38_2p4t;;
esac
# Shape tags in the result path: an unset crop keeps the recipe value ("full");
# the launcher disables MTP unless --mtp-num-layers is given.
LAYERS_TAG="${NUM_LAYERS:-full}"
MTP_TAG="${MTP_NUM_LAYERS:-0}"
if [[ "${MODEL}" == 27b ]]; then EXPERTS_TAG=0; else EXPERTS_TAG="${NUM_EXPERTS:-full}"; fi
CASE_NAME="model_${MODEL}-fsdp_${FSDP_LEVEL}-layers_${LAYERS_TAG}-mtp_${MTP_TAG}-experts_${EXPERTS_TAG}-cg_${CUDA_GRAPH}-mbs_${MICRO_BATCH_SIZE}-seq_${SEQ_LENGTH}-cp_${CONTEXT_PARALLEL}-ep_${EXPERT_PARALLEL}-disp_${DISPATCHER}-gbs_${GLOBAL_BATCH_SIZE}-gpus_${GPU_COUNT}-profile_${PROFILE}-date_${RUN_DATE}"
RESULT_DIR="${RESULTS_ROOT}/${CASE_NAME}/node${NODE_RANK}"
if [[ "${PRINT_RESULT_DIR}" == true ]]; then
    printf '%s\n' "${RESULT_DIR}"
    exit 0
fi
# nsys report name, e.g. qwen35_27b-fsdp_3-experts_0-cg_off-cp_4-ep_1-mbs_1-gbs_4-seq_16k-rank_0;
# %q{RANK} expands to the torchrun global rank in each worker.
if (( SEQ_LENGTH % 1024 == 0 )); then SEQ_TAG="$((SEQ_LENGTH / 1024))k"; else SEQ_TAG="${SEQ_LENGTH}"; fi
NSYS_NAME="${MODEL_TAG}-fsdp_${FSDP_LEVEL}-experts_${EXPERTS_TAG}-cg_${CUDA_GRAPH}-cp_${CONTEXT_PARALLEL}-ep_${EXPERT_PARALLEL}-mbs_${MICRO_BATCH_SIZE}-gbs_${GLOBAL_BATCH_SIZE}-seq_${SEQ_TAG}-rank_%q{RANK}"
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
# HybridEP topology on GB200 NVL72: the whole EP group sits in one MNNVL domain.
if [[ "${DISPATCHER}" == hybridep ]]; then
    export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN="${EXPERT_PARALLEL}" NVLINK_DOMAIN_SIZE=72 USE_MNNVL=1
fi
# Keep NCCL to warnings even when the shell exports NCCL_DEBUG=INFO; INFO floods train.log
# with per-communicator channel/topology lines.
export NCCL_DEBUG=WARN
unset NCCL_DEBUG_SUBSYS NCCL_DEBUG_FILE || true

PROFILE_RANKS=0
for (( rank = 1; rank < GPU_COUNT; rank++ )); do PROFILE_RANKS+=",${rank}"; done
OVERRIDES=(
    "model.seq_length=${SEQ_LENGTH}" "dataset.seq_length=${SEQ_LENGTH}" "dataset.blend=null"
    "train.train_iters=${TRAIN_ITERS}" "train.global_batch_size=${GLOBAL_BATCH_SIZE}" "train.micro_batch_size=${MICRO_BATCH_SIZE}"
    "scheduler.lr_warmup_iters=1" "scheduler.lr_decay_iters=${TRAIN_ITERS}"
    "validation.eval_iters=0" "validation.eval_interval=0"
    "model.context_parallel_size=${CONTEXT_PARALLEL}" "model.tensor_model_parallel_size=1" "model.pipeline_model_parallel_size=1"
    "model.expert_model_parallel_size=${EXPERT_PARALLEL}" "model.expert_tensor_parallel_size=1" "model.sequence_parallel=false"
    "model.cp_comm_type=p2p" "model.mtp_num_layers=0" "model.linear_attention_freq=4"
    "model.recompute_granularity=null" "model.recompute_method=null" "model.recompute_num_layers=null"
    "model.recompute_modules=[]" "model.fine_grained_activation_offloading=false" "model.offload_modules=[]"
    "dist.use_megatron_fsdp=true" "ddp.use_megatron_fsdp=true" "ddp.use_distributed_optimizer=true"
    "ddp.data_parallel_sharding_strategy=${SHARDING_STRATEGY}" "ddp.average_in_collective=false"
    "ddp.overlap_grad_reduce=${FSDP_OVERLAP}" "ddp.overlap_param_gather=${FSDP_OVERLAP}" "ddp.grad_reduce_in_fp32=true"
    # The recipe pins comm_overlap.overlap_*=false, and CommOverlapConfig.setup()
    # copies those values back into ddp at runtime. Override both layers so
    # FSDP3 keeps parameter prefetch and asynchronous reduce-scatter enabled.
    "comm_overlap.overlap_grad_reduce=${FSDP_OVERLAP}" "comm_overlap.overlap_param_gather=${FSDP_OVERLAP}"
    "checkpoint.ckpt_format=fsdp_dtensor" "checkpoint.save=null" "checkpoint.load=null"
    "logger.log_interval=1" "logger.tensorboard_dir=null" "logger.save_config_filepath=${RESULT_DIR}/config.yaml"
    "profiling.use_nsys_profiler=$([[ "${PROFILE}" == nsys ]] && echo true || echo false)"
    "profiling.use_pytorch_profiler=false" "profiling.profile_step_start=${PROFILE_START}" "profiling.profile_step_end=${PROFILE_END}"
    "profiling.profile_ranks=[${PROFILE_RANKS}]"
    # nvtx_ranges also turns on Megatron-FSDP's per-layer all-gather / reduce-scatter
    # ranges ("FSDP AG fwd prefetch L2 ...", "FSDP RS L1 ...").
    "profiling.nvtx_ranges=$([[ "${PROFILE}" == nsys ]] && echo true || echo false)"
    # emit_nvtx(record_shapes=True): every op range, including c10d
    # record_param_comms for FSDP all-gather / reduce-scatter, carries tensor shapes.
    "profiling.record_shapes=$([[ "${PROFILE}" == nsys ]] && echo true || echo false)"
)
if [[ "${CUDA_GRAPH}" == on ]]; then
    # TE per-layer graphs around the module compute only; FSDP hooks, all-gather
    # and reduce-scatter stay eager. Dense layers graph attention and MLP; MoE
    # layers graph attention and the router/preprocess part, leaving dispatch,
    # experts and combine eager. megatron_fsdp_cuda_graph_mode keeps gradient
    # references alive for replay (Megatron-LM sets it automatically, Bridge
    # does not); the recipe's max-pool double buffer keeps bucket addresses fixed.
    if [[ "${MODEL}" == 27b ]]; then GRAPH_MODULES="[attn,mlp]"; else GRAPH_MODULES="[attn,moe_router,moe_preprocess]"; fi
    OVERRIDES+=(
        "model.cuda_graph_impl=transformer_engine" "model.cuda_graph_scope=null" "model.cuda_graph_modules=${GRAPH_MODULES}"
        "model.cuda_graph_warmup_steps=3" "ddp.megatron_fsdp_cuda_graph_mode=true"
    )
else
    OVERRIDES+=("model.cuda_graph_impl=none" "model.cuda_graph_scope=null" "model.cuda_graph_modules=[]")
fi
[[ -n "${NUM_LAYERS}" ]] && OVERRIDES+=("model.num_layers=${NUM_LAYERS}")
[[ -n "${NUM_EXPERTS}" ]] && OVERRIDES+=("model.num_moe_experts=${NUM_EXPERTS}")
[[ -n "${MOE_LAYER_FREQ}" ]] && OVERRIDES+=("model.moe_layer_freq=${MOE_LAYER_FREQ}")
[[ -n "${LINEAR_ATTENTION_FREQ}" ]] && OVERRIDES+=("model.linear_attention_freq=${LINEAR_ATTENTION_FREQ}")
[[ -n "${MTP_NUM_LAYERS}" ]] && OVERRIDES+=("model.mtp_num_layers=${MTP_NUM_LAYERS}")
case "${DISPATCHER}" in
    alltoall) OVERRIDES+=("model.moe_token_dispatcher_type=alltoall" "model.moe_flex_dispatcher_backend=null" "model.moe_flex_dispatcher_num_sms=null");;
    hybridep) OVERRIDES+=("model.moe_token_dispatcher_type=flex" "model.moe_flex_dispatcher_backend=hybridep" "model.moe_flex_dispatcher_num_sms=${HYBRIDEP_NUM_SMS}");;
esac
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
    # NCCL's own NVTX ranges (domain NCCL) carry message bytes and comm hash per
    # collective; nsys NCCL recipes need them to report sizes and bandwidth.
    export NCCL_NVTX_DISABLE=0
    WORKER=(nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true --output="${RESULT_DIR}/${NSYS_NAME}" uv run --no-sync python)
fi
COMMAND=(uv run --no-sync python -m torch.distributed.run "${DIST_ARGS[@]}" --no-python)
if (( ${#NUMA_WRAPPER[@]} > 0 )); then
    COMMAND+=("${NUMA_WRAPPER[@]}")
fi
COMMAND+=("${WORKER[@]}" "${TRAIN_ARGS[@]}")

if [[ "${DRY_RUN}" == true ]]; then
    printf 'model=%s recipe=%s fsdp=%s dtype=%s layers=%s mtp=%s experts=%s cuda_graph=%s seq=%s cp=%s ep=%s dispatcher=%s mbs=%s gbs=%s gpus=%s num_microbatches=%s numa=%s\n' "${MODEL}" "${RECIPE}" "${FSDP_LEVEL}" "${DTYPE}" "${LAYERS_TAG}" "${MTP_TAG}" "${EXPERTS_TAG}" "${CUDA_GRAPH}" "${SEQ_LENGTH}" "${CONTEXT_PARALLEL}" "${EXPERT_PARALLEL}" "${DISPATCHER}" "${MICRO_BATCH_SIZE}" "${GLOBAL_BATCH_SIZE}" "${GPU_COUNT}" "${NUM_MICROBATCHES}" "${NUMA_MODE}" >&2
    printf 'GLOO_SOCKET_IFNAME=%s\ncommand:' "${GLOO_SOCKET_IFNAME:-<unset>}" >&2
    printf ' %q' "${COMMAND[@]}" >&2; printf '\n' >&2
    exit 0
fi
if [[ -e "${RESULT_DIR}/launch.json" ]]; then
    [[ "${RESUME}" == true ]] || { echo "result already exists: ${RESULT_DIR} (use --resume)" >&2; exit 2; }
    if grep -qx 'status=0' "${RESULT_DIR}/run_info.txt" 2>/dev/null; then
        echo "skip completed case: ${RESULT_DIR}" >&2
        exit 0
    fi
    ARCHIVE_DIR="${RESULT_DIR}.attempt-$(date +%y%m%d-%H%M%S)"
    mv "${RESULT_DIR}" "${ARCHIVE_DIR}"
    echo "archived incomplete or failed attempt: ${ARCHIVE_DIR}" >&2
fi
mkdir -p "${RESULT_DIR}"
printf 'command:' >"${RESULT_DIR}/command.txt"
printf ' %q' "${COMMAND[@]}" >>"${RESULT_DIR}/command.txt"
printf '\n' >>"${RESULT_DIR}/command.txt"
cat >"${RESULT_DIR}/launch.json" <<JSON
{"model":"${MODEL_ID}","recipe":"${RECIPE}","run_name":"${RUN_NAME}","dtype":"${DTYPE}","fsdp":${FSDP_LEVEL},"num_layers":"${LAYERS_TAG}","mtp_num_layers":${MTP_TAG},"num_moe_experts":"${EXPERTS_TAG}","cuda_graph":"${CUDA_GRAPH}","seq_length":${SEQ_LENGTH},"context_parallel_size":${CONTEXT_PARALLEL},"expert_model_parallel_size":${EXPERT_PARALLEL},"dispatcher":"${DISPATCHER}","micro_batch_size":${MICRO_BATCH_SIZE},"global_batch_size":${GLOBAL_BATCH_SIZE},"num_microbatches":${NUM_MICROBATCHES},"gpu_count":${GPU_COUNT},"nnodes":${NNODES},"node_rank":${NODE_RANK},"numa_mode":"${NUMA_MODE}","run_date":"${RUN_DATE}"}
JSON
printf 'model=%s fsdp=%s dtype=%s layers=%s mtp=%s experts=%s cuda_graph=%s seq=%s cp=%s ep=%s dispatcher=%s mbs=%s gbs=%s gpus=%s num_microbatches=%s numa=%s\n' "${MODEL}" "${FSDP_LEVEL}" "${DTYPE}" "${LAYERS_TAG}" "${MTP_TAG}" "${EXPERTS_TAG}" "${CUDA_GRAPH}" "${SEQ_LENGTH}" "${CONTEXT_PARALLEL}" "${EXPERT_PARALLEL}" "${DISPATCHER}" "${MICRO_BATCH_SIZE}" "${GLOBAL_BATCH_SIZE}" "${GPU_COUNT}" "${NUM_MICROBATCHES}" "${NUMA_MODE}" | tee "${RESULT_DIR}/run_info.txt"
set +e
"${COMMAND[@]}" 2>&1 | tee "${RESULT_DIR}/train.log"
STATUS=${PIPESTATUS[0]}
set -e
printf 'status=%s\n' "${STATUS}" >>"${RESULT_DIR}/run_info.txt"
exit "${STATUS}"
