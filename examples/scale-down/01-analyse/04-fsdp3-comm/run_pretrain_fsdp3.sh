#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

# Run one cropped Qwen3.5 / Qwen3.8 case through scripts/training/run_recipe.py.
# The selected GB200 recipe is FSDP1 by default; the overrides below select
# Megatron FSDP3 (default), FSDP2, keep FSDP1, or switch to Megatron DDP.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/01-analyse/04-fsdp3-comm}"

MODEL=""
RECIPE=""
SEQ_LENGTH=""
DTYPE="${DTYPE:-bf16}"
PROFILE="${PROFILE:-none}"
TENSOR_PARALLEL="1"
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
ATTENTION=hybrid
MTP_NUM_LAYERS=""
DISPATCHER="${DISPATCHER:-hybridep}"
CUDA_GRAPH="${CUDA_GRAPH:-off}"
MERGE_GRAPHS=false
FORCE_BALANCE=false
FSDP_LEVEL="${FSDP_LEVEL:-3}"
OVERLAP=false
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
Usage: run_pretrain_fsdp3.sh --model 27b|397b|2.4t|122b --seq-length N [options]

Models: 27b / 397b / 122b are Qwen3.5-27B / 397B-A17B / 122B-A10B; 2.4t (alias 2p4t)
is Qwen3.8-2.4T-A95B. The model selects its 4-GPU GB200 FSDP1 recipe.

Options:
  --recipe NAME                  Override the recipe selected by --model
  --seq|--seq-length N           Sequence length (required)
  --fsdp ddp|1|2|3               3: Megatron FSDP optim_grads_params with prefetch and
                                 overlapped reduce-scatter. 2: optim_grads; gradients
                                 sharded and reduce-scattered every micro-batch
                                 (overlapped), parameters replicated and all-gathered
                                 once after the optimizer step. 1: the recipe's optim
                                 strategy (only optimizer state sharded; gradient
                                 reduce-scatter and parameter all-gather once per
                                 iteration, not overlapped). ddp: Megatron DDP with
                                 the distributed optimizer; bucketed gradient
                                 reduce-scatter overlapped with backward, parameter
                                 all-gather with the next forward (default: 3)
  --overlap                      Megatron-FSDP only: give parameter all-gathers their
                                 own DP communicators (dist.create_all_gather_group,
                                 Megatron-LM --create-all-gather-group), so they no
                                 longer queue behind gradient reduce-scatters on one
                                 NCCL communicator; only FSDP3 overlaps the two. Names get
                                 "-overlap"
  --tp N                         Tensor parallel size; N > 1 turns on sequence
                                 parallelism (default: 1)
  --cp N                         Context parallel size (default: 1)
  --cuda-graph off|on            off: no CUDA graph. on with FSDP: full-iteration graph
                                 of forward+backward, FSDP all-gather / reduce-scatter
                                 included; MoE models add a per-rank token budget
                                 (capacity factor 1.5) and paged stash, hybridep
                                 only. on with DDP: TE per-layer graphs, dense
                                 attn+mlp, MoE attn+moe_router+moe_preprocess with
                                 dispatch / experts / combine eager (default: off)
  --mbs|--micro-batch-size N     Micro batch size (default: 1)
  --gbs|--global-batch-size N    Global batch size, a multiple of MBS*DP with
                                 DP = gpus / (TP*CP) (default: 16)
  --train-iters N                Training iterations (default: 10)
  --num-layers N                 Crop decoder layers for this case
  --full                         Every decoder layer uses gated (full) attention
  --linear                       Every decoder layer uses GatedDeltaNet (needs --num-layers)
                                 Default: 3 GatedDeltaNet + 1 gated attention per 4 layers
  --num-experts N                Crop routed experts for a MoE case
  --ep N                         Expert parallel size (default: 4 for MoE,
                                 1 for dense; dense models only accept 1)
  --moe-layer-freq VALUE         Override model.moe_layer_freq (default with
                                 --num-layers on MoE: every layer MoE)
  --mtp-num-layers N             Override model.mtp_num_layers (default: 0)
  --dispatcher hybridep|alltoall MoE token dispatcher (default: hybridep)
  --hybridep-num-sms N           SMs for HybridEP dispatch (default: 32)
  --force-balance                MoE only: replace router logits with random values
                                 (moe_router_force_load_balancing), so tokens spread
                                 uniformly over experts; benchmark only, the loss is
                                 not meaningful. Names get "-fb"
  --dtype|--precision bf16|mxfp8 Compute mode (default: bf16)
  --mxfp8                        Same as --dtype mxfp8
  --gpu|--gpus N                 Global GPU count (default: 4)
  --nproc-per-node N             GPUs per node (default: 4)
  --nnodes N                     Number of nodes (default: gpu / nproc-per-node)
  --node-rank N                  Node rank (default: NODE_RANK or 0)
  --master-addr HOST             Multi-node rendezvous address
  --master-port N                Multi-node rendezvous port (default: 29501)
  --profile none|nsys            Profiling mode (default: none). nsys records the GPUs
                                 of node 0 into one report
  --nsys                         Alias for --profile nsys
  --merge                        With nsys and --cuda-graph on, draw each graph launch
                                 as one block (--cuda-graph-trace=graph) instead of
                                 its kernels; names get "-merge"
  --profile-start N              First profiled iteration (default: 7)
  --profile-end N                Stop iteration, exclusive (default: 8, i.e. one step)
  --run-date YYMMDD-HHMMSS       Shared matrix timestamp
  --run-name NAME                Optional human-readable case name
  --results-root PATH            Result root
  --override KEY=VALUE           Additional run_recipe.py override (repeatable)
  --resume                       Skip a completed case; archive a failed or
                                 incomplete result and rerun it
  --dry-run                      Print the resolved command and exit
  --print-result-dir             Validate the case, print this node's result
                                 directory and exit
USAGE
}

need_value() { [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; usage >&2; exit 2; }; }
# ones_list N -> "[1,1,...,1]" with N entries (a per-layer pattern for N layers)
ones_list() { local list="[1" i; for (( i = 1; i < $1; i++ )); do list+=",1"; done; printf '%s]' "${list}"; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) need_value "$@"; MODEL="$2"; [[ "${MODEL}" == 2p4t ]] && MODEL=2.4t; shift 2;;
        --recipe) need_value "$@"; RECIPE="$2"; shift 2;;
        --seq|--seq-length) need_value "$@"; SEQ_LENGTH="$2"; shift 2;;
        --dtype|--precision) need_value "$@"; DTYPE="$2"; shift 2;;
        --mxfp8) DTYPE=mxfp8; shift;;
        --tp|--tensor-parallel) need_value "$@"; TENSOR_PARALLEL="$2"; shift 2;;
        --cp|--context-parallel) need_value "$@"; CONTEXT_PARALLEL="$2"; shift 2;;
        --mbs|--micro-batch-size) need_value "$@"; MICRO_BATCH_SIZE="$2"; shift 2;;
        --gbs|--global-batch-size) need_value "$@"; GLOBAL_BATCH_SIZE="$2"; shift 2;;
        --train-iters) need_value "$@"; TRAIN_ITERS="$2"; shift 2;;
        --num-layers) need_value "$@"; NUM_LAYERS="$2"; shift 2;;
        --full) [[ "${ATTENTION}" != linear ]] || { echo "--full and --linear are exclusive" >&2; exit 2; }; ATTENTION=full; shift;;
        --linear) [[ "${ATTENTION}" != full ]] || { echo "--full and --linear are exclusive" >&2; exit 2; }; ATTENTION=linear; shift;;
        --num-experts) need_value "$@"; NUM_EXPERTS="$2"; shift 2;;
        --ep|--expert-parallel) need_value "$@"; EXPERT_PARALLEL="$2"; shift 2;;
        --moe-layer-freq) need_value "$@"; MOE_LAYER_FREQ="$2"; shift 2;;
        --mtp-num-layers) need_value "$@"; MTP_NUM_LAYERS="$2"; shift 2;;
        --dispatcher) need_value "$@"; DISPATCHER="$2"; shift 2;;
        --hybridep-num-sms) need_value "$@"; HYBRIDEP_NUM_SMS="$2"; shift 2;;
        --cuda-graph) need_value "$@"; CUDA_GRAPH="$2"; shift 2;;
        --force-balance) FORCE_BALANCE=true; shift;;
        --fsdp) need_value "$@"; FSDP_LEVEL="$2"; shift 2;;
        --overlap) OVERLAP=true; shift;;
        --gpu|--gpus) need_value "$@"; GPU_COUNT="$2"; shift 2;;
        --nproc-per-node) need_value "$@"; GPUS_PER_NODE="$2"; shift 2;;
        --nnodes) need_value "$@"; NNODES="$2"; shift 2;;
        --node-rank) need_value "$@"; NODE_RANK="$2"; shift 2;;
        --master-addr) need_value "$@"; MASTER_ADDR="$2"; shift 2;;
        --master-port) need_value "$@"; MASTER_PORT="$2"; shift 2;;
        --profile) need_value "$@"; PROFILE="$2"; shift 2;;
        --nsys) PROFILE=nsys; shift;;
        --merge) MERGE_GRAPHS=true; shift;;
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

[[ -n "${MODEL}" && -n "${SEQ_LENGTH}" ]] || { echo "--model and --seq-length are required" >&2; exit 2; }
[[ "${DTYPE}" == fp8mx ]] && DTYPE=mxfp8
case "${MODEL}" in
    397b) MODEL_ID=Qwen/Qwen3.5-397B-A17B; MODEL_TAG=qwen35_397b; DEFAULT_RECIPE=qwen35_text_397b_a17b_pretrain_4gpu_gb200_bf16_fsdp1_config;;
    122b) MODEL_ID=Qwen/Qwen3.5-122B-A10B; MODEL_TAG=qwen35_122b; DEFAULT_RECIPE=qwen35_text_122b_a10b_pretrain_4gpu_gb200_bf16_fsdp1_config;;
    27b) MODEL_ID=Qwen/Qwen3.5-27B; MODEL_TAG=qwen35_27b; DEFAULT_RECIPE=qwen35_text_27b_pretrain_4gpu_gb200_bf16_fsdp1_config;;
    2.4t) MODEL_ID=Qwen/Qwen3.8-2.4T-A95B; MODEL_TAG=qwen38_2.4t; DEFAULT_RECIPE=qwen38_text_2p4t_a95b_pretrain_4gpu_gb200_bf16_fsdp1_config;;
    *) echo "Unsupported model: ${MODEL} (use 27b, 397b, 2.4t or 122b)" >&2; exit 2;;
esac
RECIPE="${RECIPE:-${DEFAULT_RECIPE}}"
case "${DTYPE}" in bf16|mxfp8) ;; *) echo "--dtype must be bf16 or mxfp8" >&2; exit 2;; esac
case "${PROFILE}" in none|nsys) ;; *) echo "--profile must be none or nsys" >&2; exit 2;; esac
case "${DISPATCHER}" in hybridep|alltoall) ;; *) echo "--dispatcher must be hybridep or alltoall" >&2; exit 2;; esac
case "${CUDA_GRAPH}" in off|on) ;; *) echo "--cuda-graph must be off or on" >&2; exit 2;; esac
# nsys --cuda-graph-trace mode; only a profiled graph run has one. --merge without a
# graph or without nsys changes nothing, so a sweep can pass it to every case.
GRAPH_TRACE=none
if [[ "${PROFILE}" == nsys && "${CUDA_GRAPH}" == on ]]; then
    GRAPH_TRACE=$([[ "${MERGE_GRAPHS}" == true ]] && echo graph || echo node)
fi
# FSDP1 keeps the recipe's settings: with the optim strategy Megatron-FSDP delays the
# gradient reduction to the end of the iteration, so there is nothing to overlap.
# FSDP2 reduce-scatters gradients on every backward, which Megatron-FSDP requires to be
# overlapped; its parameters stay replicated, so the parameter all-gather after the
# optimizer step stays synchronous as in FSDP1. DDP replaces Megatron-FSDP with MCore
# DistributedDataParallel; the sharding strategy is FSDP-only, so it goes back to the
# no_shard default.
case "${FSDP_LEVEL}" in
    3) SHARDING_STRATEGY=optim_grads_params; GRAD_OVERLAP=true; PARAM_OVERLAP=true; DP_TAG=fsdp_3; FSDP_JSON=3;;
    2) SHARDING_STRATEGY=optim_grads; GRAD_OVERLAP=true; PARAM_OVERLAP=false; DP_TAG=fsdp_2; FSDP_JSON=2;;
    1) SHARDING_STRATEGY=optim; GRAD_OVERLAP=false; PARAM_OVERLAP=false; DP_TAG=fsdp_1; FSDP_JSON=1;;
    ddp) SHARDING_STRATEGY=no_shard; GRAD_OVERLAP=true; PARAM_OVERLAP=true; DP_TAG=ddp; FSDP_JSON=null;;
    *) echo "--fsdp must be ddp, 1, 2, or 3" >&2; exit 2;;
esac
# Only Megatron-FSDP builds its parameter all-gathers on the separate communicators.
[[ "${OVERLAP}" == false || "${FSDP_LEVEL}" != ddp ]] || { echo "--overlap requires --fsdp 1, 2 or 3" >&2; exit 2; }
for name in SEQ_LENGTH TENSOR_PARALLEL CONTEXT_PARALLEL MICRO_BATCH_SIZE GLOBAL_BATCH_SIZE TRAIN_ITERS GPU_COUNT GPUS_PER_NODE MASTER_PORT HYBRIDEP_NUM_SMS; do
    [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || { echo "${name} must be a positive integer" >&2; exit 2; }
done
for name in NODE_RANK PROFILE_START PROFILE_END; do
    [[ "${!name}" =~ ^[0-9]+$ ]] || { echo "${name} must be a non-negative integer" >&2; exit 2; }
done
[[ "${RUN_DATE}" =~ ^[0-9]{6}-[0-9]{6}$ ]] || { echo "--run-date must use yymmdd-hhmmss" >&2; exit 2; }
(( GPU_COUNT >= 2 && GPU_COUNT % GPUS_PER_NODE == 0 )) || { echo "invalid GPU count" >&2; exit 2; }
NNODES="${NNODES:-$((GPU_COUNT / GPUS_PER_NODE))}"
[[ "${NNODES}" =~ ^[1-9][0-9]*$ ]] || { echo "nnodes must be positive" >&2; exit 2; }
(( NNODES * GPUS_PER_NODE == GPU_COUNT && NODE_RANK < NNODES )) || { echo "invalid node topology" >&2; exit 2; }
# TP groups are consecutive ranks; keep each one inside a node's NVLink-connected GPUs.
(( GPUS_PER_NODE % TENSOR_PARALLEL == 0 )) || { echo "tp=${TENSOR_PARALLEL} must divide nproc-per-node=${GPUS_PER_NODE}" >&2; exit 2; }
(( GPU_COUNT % (TENSOR_PARALLEL * CONTEXT_PARALLEL) == 0 )) || { echo "world size must be divisible by tp*cp" >&2; exit 2; }
DATA_PARALLEL=$((GPU_COUNT / (TENSOR_PARALLEL * CONTEXT_PARALLEL)))
(( SEQ_LENGTH % (2 * CONTEXT_PARALLEL) == 0 )) || { echo "seq-length must be divisible by 2*cp" >&2; exit 2; }
# Sequence parallelism splits each CP rank's sequence chunk over the TP group.
(( (SEQ_LENGTH / CONTEXT_PARALLEL) % TENSOR_PARALLEL == 0 )) || { echo "seq-length/cp must be divisible by tp" >&2; exit 2; }
(( GLOBAL_BATCH_SIZE % (MICRO_BATCH_SIZE * DATA_PARALLEL) == 0 )) || {
    echo "global batch size ${GLOBAL_BATCH_SIZE} is not divisible by MBS*DP=${MICRO_BATCH_SIZE}*${DATA_PARALLEL}" >&2; exit 2
}
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
# The all-GatedDeltaNet pattern is a per-layer list, so it needs the layer count.
[[ "${ATTENTION}" != linear || -n "${NUM_LAYERS}" ]] || { echo "--linear requires --num-layers" >&2; exit 2; }
# MoE cases always use expert parallelism; the EP group folds onto the same
# GPUs as the TP x DP x CP mesh, so EP only needs to divide the world size.
if [[ "${MODEL}" == 27b ]]; then
    [[ "${FORCE_BALANCE}" == false ]] || { echo "--force-balance only applies to MoE models" >&2; exit 2; }
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
    # The FSDP full-iteration graph needs static MoE shapes, which only the HybridEP
    # per-rank token budget (moe_expert_rank_capacity_factor) provides. DDP per-layer
    # graphs stop before the dispatch and keep dropless MoE eager.
    if [[ "${CUDA_GRAPH}" == on && "${FSDP_LEVEL}" != ddp && "${DISPATCHER}" != hybridep ]]; then
        echo "--cuda-graph on for MoE models with FSDP requires --dispatcher hybridep" >&2; exit 2
    fi
    if [[ -n "${NUM_EXPERTS}" ]]; then
        (( NUM_EXPERTS % EXPERT_PARALLEL == 0 )) || { echo "num-experts must be divisible by ep" >&2; exit 2; }
    fi
    # A cropped MoE model keeps every remaining layer MoE.
    if [[ -n "${NUM_LAYERS}" && -z "${MOE_LAYER_FREQ}" ]]; then
        MOE_LAYER_FREQ="$(ones_list "${NUM_LAYERS}")"
    fi
fi
# linear_attention_freq: N puts one gated attention layer after every N-1 GatedDeltaNet
# layers, so 1 is all gated attention; a list gives the per-layer pattern (1 = GDN).
case "${ATTENTION}" in
    hybrid) LINEAR_ATTENTION_FREQ=4;;
    full) LINEAR_ATTENTION_FREQ=1;;
    linear) LINEAR_ATTENTION_FREQ="$(ones_list "${NUM_LAYERS}")";;
esac
SEQUENCE_PARALLEL=false
(( TENSOR_PARALLEL > 1 )) && SEQUENCE_PARALLEL=true

# Shape tags in the result path: an unset crop keeps the recipe value ("full");
# the launcher disables MTP unless --mtp-num-layers is given.
LAYERS_TAG="${NUM_LAYERS:-full}"
MTP_TAG="${MTP_NUM_LAYERS:-0}"
if [[ "${MODEL}" == 27b ]]; then EXPERTS_TAG=0; else EXPERTS_TAG="${NUM_EXPERTS:-full}"; fi
# Default settings add no tag, so earlier result names stay resumable: --overlap adds
# "-overlap" after the data parallel mode, MXFP8 "-mxfp8", forced balanced routing "-fb",
# --full / --linear "-attn_<full|linear>" after the layer count, and TP > 1 "-tp_<N>"
# before the CP size.
OVERLAP_TAG=""
[[ "${OVERLAP}" == true ]] && OVERLAP_TAG="-overlap"
DTYPE_TAG=""
[[ "${DTYPE}" == mxfp8 ]] && DTYPE_TAG="-mxfp8"
[[ "${FORCE_BALANCE}" == true ]] && DTYPE_TAG+="-fb"
ATTN_TAG=""
[[ "${ATTENTION}" != hybrid ]] && ATTN_TAG="-attn_${ATTENTION}"
TP_TAG=""
(( TENSOR_PARALLEL > 1 )) && TP_TAG="-tp_${TENSOR_PARALLEL}"
CG_TAG="${CUDA_GRAPH}"
[[ "${GRAPH_TRACE}" == graph ]] && CG_TAG+="-merge"
CASE_NAME="model_${MODEL}-${DP_TAG}${OVERLAP_TAG}${DTYPE_TAG}-layers_${LAYERS_TAG}${ATTN_TAG}-mtp_${MTP_TAG}-experts_${EXPERTS_TAG}-cg_${CG_TAG}-mbs_${MICRO_BATCH_SIZE}-seq_${SEQ_LENGTH}${TP_TAG}-cp_${CONTEXT_PARALLEL}-ep_${EXPERT_PARALLEL}-disp_${DISPATCHER}-gbs_${GLOBAL_BATCH_SIZE}-gpus_${GPU_COUNT}-profile_${PROFILE}-date_${RUN_DATE}"
RESULT_DIR="${RESULTS_ROOT}/${CASE_NAME}/node${NODE_RANK}"
if [[ "${PRINT_RESULT_DIR}" == true ]]; then
    printf '%s\n' "${RESULT_DIR}"
    exit 0
fi
# nsys report name, e.g. qwen35_27b-fsdp_3-mxfp8-experts_0-cg_off-tp_2-cp_1-ep_1-mbs_1-gbs_8-seq_4k.
if (( SEQ_LENGTH % 1024 == 0 )); then SEQ_TAG="$((SEQ_LENGTH / 1024))k"; else SEQ_TAG="${SEQ_LENGTH}"; fi
NSYS_NAME="${MODEL_TAG}-${DP_TAG}${OVERLAP_TAG}${DTYPE_TAG}${ATTN_TAG}-experts_${EXPERTS_TAG}-cg_${CG_TAG}${TP_TAG}-cp_${CONTEXT_PARALLEL}-ep_${EXPERT_PARALLEL}-mbs_${MICRO_BATCH_SIZE}-gbs_${GLOBAL_BATCH_SIZE}-seq_${SEQ_TAG}"
NUM_MICROBATCHES=$((GLOBAL_BATCH_SIZE / (MICRO_BATCH_SIZE * DATA_PARALLEL)))

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

# nsys records node 0 only: its ranks 0..nproc-per-node-1 (the 4 GPUs of one GB200
# tray) call cudaProfilerStart / Stop, and only node 0 runs under nsys, so a run
# leaves exactly one report.
PROFILE_RANKS=0
for (( rank = 1; rank < GPUS_PER_NODE; rank++ )); do PROFILE_RANKS+=",${rank}"; done
# DDP replicates parameters and gradients (the distributed optimizer still shards
# optimizer state); fsdp_dtensor checkpoints only exist for Megatron-FSDP.
if [[ "${FSDP_LEVEL}" == ddp ]]; then
    DP_OVERRIDES=("dist.use_megatron_fsdp=false" "ddp.use_megatron_fsdp=false" "checkpoint.ckpt_format=torch_dist")
else
    DP_OVERRIDES=("dist.use_megatron_fsdp=true" "ddp.use_megatron_fsdp=true" "checkpoint.ckpt_format=fsdp_dtensor")
fi
OVERRIDES=(
    "model.seq_length=${SEQ_LENGTH}" "dataset.seq_length=${SEQ_LENGTH}" "dataset.blend=null"
    "train.train_iters=${TRAIN_ITERS}" "train.global_batch_size=${GLOBAL_BATCH_SIZE}" "train.micro_batch_size=${MICRO_BATCH_SIZE}"
    "scheduler.lr_warmup_iters=1" "scheduler.lr_decay_iters=${TRAIN_ITERS}"
    "validation.eval_iters=0" "validation.eval_interval=0"
    # TP > 1 always runs with sequence parallelism; the recipe keeps
    # comm_overlap.tp_comm_overlap=false, so TP / SP collectives are plain NCCL.
    "model.tensor_model_parallel_size=${TENSOR_PARALLEL}" "model.sequence_parallel=${SEQUENCE_PARALLEL}"
    "model.context_parallel_size=${CONTEXT_PARALLEL}" "model.pipeline_model_parallel_size=1"
    "model.expert_model_parallel_size=${EXPERT_PARALLEL}" "model.expert_tensor_parallel_size=1"
    "model.cp_comm_type=p2p" "model.mtp_num_layers=0" "model.linear_attention_freq=${LINEAR_ATTENTION_FREQ}"
    "model.recompute_granularity=null" "model.recompute_method=null" "model.recompute_num_layers=null"
    "model.recompute_modules=[]" "model.fine_grained_activation_offloading=false" "model.offload_modules=[]"
    "${DP_OVERRIDES[@]}" "dist.create_all_gather_group=${OVERLAP}" "ddp.use_distributed_optimizer=true"
    "ddp.data_parallel_sharding_strategy=${SHARDING_STRATEGY}" "ddp.average_in_collective=false"
    "ddp.overlap_grad_reduce=${GRAD_OVERLAP}" "ddp.overlap_param_gather=${PARAM_OVERLAP}" "ddp.grad_reduce_in_fp32=true"
    # The recipe pins comm_overlap.overlap_*=false, and CommOverlapConfig.setup()
    # copies those values back into ddp at runtime. Override both layers so
    # FSDP3 / FSDP2 and DDP keep parameter prefetch and asynchronous gradient reduction.
    "comm_overlap.overlap_grad_reduce=${GRAD_OVERLAP}" "comm_overlap.overlap_param_gather=${PARAM_OVERLAP}"
    "checkpoint.save=null" "checkpoint.load=null"
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
GRAPH_IMPL=none
GRAPH_MODULES="[]"
if [[ "${CUDA_GRAPH}" == on && "${FSDP_LEVEL}" == ddp ]]; then
    # TE per-layer graphs: one make_graphed_callables() graph per layer and micro-batch
    # for forward and one for backward. Bridge removes DDP's forward pre-hook
    # (parameter all-gather) for the capture and re-installs it as the layers'
    # manual hooks. Dense layers are captured whole. MoE layers stop after the router
    # and preprocess, so the dropless HybridEP / alltoall dispatch, routed experts and
    # combine stay eager and keep the unfused TEGroupedMLP of the off mode.
    GRAPH_IMPL=transformer_engine
    if [[ "${MODEL}" == 27b ]]; then
        GRAPH_MODULES="[attn,mlp]"
    elif [[ "${DTYPE}" == mxfp8 && "${DISPATCHER}" == alltoall ]]; then
        # moe_router_padding_for_fp8 makes the alltoall preprocess copy token counts
        # to the host, which MCore refuses to graph.
        GRAPH_MODULES="[attn,moe_router]"
    else
        GRAPH_MODULES="[attn,moe_router,moe_preprocess]"
    fi
    OVERRIDES+=(
        "model.cuda_graph_impl=${GRAPH_IMPL}" "model.cuda_graph_scope=null" "model.cuda_graph_modules=${GRAPH_MODULES}"
        "model.cuda_graph_warmup_steps=3" "model.use_te_rng_tracker=true" "rng.te_rng_tracker=true"
    )
elif [[ "${CUDA_GRAPH}" == on ]]; then
    # One full-iteration graph of forward+backward (optimizer step stays eager), with
    # the FSDP all-gather / reduce-scatter, TP / SP and CP collectives captured inside.
    # Per-layer graphs (TE or local) are not an option: both refuse modules carrying
    # Megatron-FSDP's forward / backward hooks. megatron_fsdp_cuda_graph_mode keeps
    # gradient references alive for replay, and fsdp_all_gather_in_start_param_sync=false
    # keeps the FSDP3 optimizer step from issuing an all-gather the graph would wait on
    # (Megatron-LM sets both automatically, Bridge does not); FSDP1 / FSDP2 all-gather
    # synchronously in the eager optimizer step. The recipe's max-pool double buffer
    # keeps bucket addresses fixed. The graph cannot check NaN loss or gradients.
    GRAPH_IMPL=full_iteration
    OVERRIDES+=(
        "model.cuda_graph_impl=${GRAPH_IMPL}" "model.cuda_graph_scope=null" "model.cuda_graph_modules=${GRAPH_MODULES}"
        "model.cuda_graph_warmup_steps=3" "ddp.megatron_fsdp_cuda_graph_mode=true"
        "ddp.fsdp_all_gather_in_start_param_sync=false"
        "rerun_state_machine.check_for_nan_in_loss=false" "ddp.check_for_nan_in_grad=false"
    )
    # Dropless MoE copies token counts to the host and changes shapes every step,
    # which a graph cannot capture. A per-rank token budget of 1.5x the balanced
    # load fixes the HybridEP buffers (it needs the TE op-fuser grouped MLP), and
    # paged stash keeps the routed-expert activations in fixed pages. An overflowing
    # step is re-run eagerly by PagedStashRunner.
    if [[ "${MODEL}" != 27b ]]; then
        OVERRIDES+=(
            "model.moe_expert_rank_capacity_factor=1.5" "model.moe_paged_stash=true"
            "model.use_transformer_engine_op_fuser=true"
        )
        # MCore only builds the fused SwiGLU grouped MLP (TE CuTe DSL kernel, sm100+)
        # when this is set; otherwise TEGroupedMLP asserts "Fused GroupedMLP is not
        # supported for this configuration".
        export NVTE_CUTEDSL_FUSED_GROUPED_MLP=1
    fi
fi
if [[ "${CUDA_GRAPH}" == on ]]; then
    # Keep NCCL graph registration off. The full-iteration graph holds NCCL
    # collectives, where registering expandable segments (often inherited through
    # PYTORCH_CUDA_ALLOC_CONF) can hit illegal memory access, and TECudaGraphHelper
    # asserts it under expandable segments. nccl_ub stays off either way.
    export NCCL_GRAPH_REGISTER=0
else
    OVERRIDES+=("model.cuda_graph_impl=none" "model.cuda_graph_scope=null" "model.cuda_graph_modules=[]")
fi
[[ -n "${NUM_LAYERS}" ]] && OVERRIDES+=("model.num_layers=${NUM_LAYERS}")
[[ -n "${NUM_EXPERTS}" ]] && OVERRIDES+=("model.num_moe_experts=${NUM_EXPERTS}")
[[ -n "${MOE_LAYER_FREQ}" ]] && OVERRIDES+=("model.moe_layer_freq=${MOE_LAYER_FREQ}")
[[ -n "${MTP_NUM_LAYERS}" ]] && OVERRIDES+=("model.mtp_num_layers=${MTP_NUM_LAYERS}")
case "${DISPATCHER}" in
    alltoall) OVERRIDES+=("model.moe_token_dispatcher_type=alltoall" "model.moe_flex_dispatcher_backend=null" "model.moe_flex_dispatcher_num_sms=null");;
    hybridep) OVERRIDES+=("model.moe_token_dispatcher_type=flex" "model.moe_flex_dispatcher_backend=hybridep" "model.moe_flex_dispatcher_num_sms=${HYBRIDEP_NUM_SMS}");;
esac
if [[ "${DTYPE}" == mxfp8 ]]; then
    OVERRIDES+=("mixed_precision.fp8=e4m3" "mixed_precision.fp8_recipe=mxfp8" "mixed_precision.fp8_param_gather=false" "mixed_precision.reuse_grad_buf_for_mxfp8_param_ag=false")
    [[ "${MODEL}" != 27b ]] && OVERRIDES+=("model.moe_router_padding_for_fp8=true")
fi
# MCore draws N(0,1) router logits per rank and keeps the router GEMM, top-k, dispatch
# and expert compute; only the expert choice of each token becomes uniform random.
[[ "${FORCE_BALANCE}" == true ]] && OVERRIDES+=("model.moe_router_force_load_balancing=true")
if (( ${#EXTRA_OVERRIDES[@]} > 0 )); then
    OVERRIDES+=("${EXTRA_OVERRIDES[@]}")
fi

TRAIN_ARGS=("${REPO_ROOT}/scripts/training/run_recipe.py" --recipe "${RECIPE}" --mode pretrain --dataset mock --step-func llm_step "${OVERRIDES[@]}")
if (( NNODES == 1 )); then
    DIST_ARGS=(--standalone --nproc_per_node="${GPUS_PER_NODE}" --master_port="${MASTER_PORT}")
else
    DIST_ARGS=(--nproc_per_node="${GPUS_PER_NODE}" --nnodes="${NNODES}" --node_rank="${NODE_RANK}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}")
fi
COMMAND=()
if [[ "${PROFILE}" == nsys ]] && (( NODE_RANK == 0 )); then
    # NCCL's own NVTX ranges (domain NCCL) carry message bytes and comm hash per
    # collective; nsys NCCL recipes need them to report sizes and bandwidth.
    export NCCL_NVTX_DISABLE=0
    # nsys outside torchrun follows every worker process into one report, so all
    # ranks of the node share one timeline. Every rank of the node calls
    # cudaProfilerStart / Stop, bounding the capture to the profiled steps.
    COMMAND=(nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true --output="${RESULT_DIR}/${NSYS_NAME}")
    # nsys draws each graph launch as one opaque block by default (graph, kept by
    # --merge); node tracing records every kernel inside the graph (GEMMs, NCCL,
    # HybridEP) by name.
    if [[ "${GRAPH_TRACE}" != none ]]; then
        COMMAND+=(--cuda-graph-trace="${GRAPH_TRACE}")
    fi
fi
COMMAND+=(uv run --no-sync python -m torch.distributed.run "${DIST_ARGS[@]}" --no-python)
if (( ${#NUMA_WRAPPER[@]} > 0 )); then
    COMMAND+=("${NUMA_WRAPPER[@]}")
fi
COMMAND+=(uv run --no-sync python "${TRAIN_ARGS[@]}")

INFO=(
    "model=${MODEL}" "recipe=${RECIPE}" "fsdp=${FSDP_LEVEL}" "overlap=${OVERLAP}" "dtype=${DTYPE}" "layers=${LAYERS_TAG}" "attention=${ATTENTION}"
    "mtp=${MTP_TAG}" "experts=${EXPERTS_TAG}" "force_balance=${FORCE_BALANCE}" "cuda_graph=${CUDA_GRAPH}" "cuda_graph_trace=${GRAPH_TRACE}" "seq=${SEQ_LENGTH}"
    "tp=${TENSOR_PARALLEL}" "sp=${SEQUENCE_PARALLEL}" "cp=${CONTEXT_PARALLEL}" "dp=${DATA_PARALLEL}" "ep=${EXPERT_PARALLEL}"
    "dispatcher=${DISPATCHER}" "mbs=${MICRO_BATCH_SIZE}" "gbs=${GLOBAL_BATCH_SIZE}" "gpus=${GPU_COUNT}"
    "num_microbatches=${NUM_MICROBATCHES}" "numa=${NUMA_MODE}"
)
if [[ "${DRY_RUN}" == true ]]; then
    printf '%s\n' "${INFO[*]}" >&2
    printf 'GLOO_SOCKET_IFNAME=%s\nPYTORCH_CUDA_ALLOC_CONF=%s\nNCCL_GRAPH_REGISTER=%s\nNVTE_CUTEDSL_FUSED_GROUPED_MLP=%s\ncommand:' "${GLOO_SOCKET_IFNAME:-<unset>}" "${PYTORCH_CUDA_ALLOC_CONF:-<unset>}" "${NCCL_GRAPH_REGISTER:-<unset>}" "${NVTE_CUTEDSL_FUSED_GROUPED_MLP:-<unset>}" >&2
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
{"model":"${MODEL_ID}","recipe":"${RECIPE}","run_name":"${RUN_NAME}","dtype":"${DTYPE}","fsdp":${FSDP_JSON},"data_parallel":"${DP_TAG}","create_all_gather_group":${OVERLAP},"num_layers":"${LAYERS_TAG}","attention":"${ATTENTION}","linear_attention_freq":"${LINEAR_ATTENTION_FREQ}","mtp_num_layers":${MTP_TAG},"num_moe_experts":"${EXPERTS_TAG}","force_balance":${FORCE_BALANCE},"cuda_graph":"${CUDA_GRAPH}","cuda_graph_impl":"${GRAPH_IMPL}","cuda_graph_modules":"${GRAPH_MODULES}","cuda_graph_trace":"${GRAPH_TRACE}","seq_length":${SEQ_LENGTH},"tensor_model_parallel_size":${TENSOR_PARALLEL},"sequence_parallel":${SEQUENCE_PARALLEL},"context_parallel_size":${CONTEXT_PARALLEL},"data_parallel_size":${DATA_PARALLEL},"expert_model_parallel_size":${EXPERT_PARALLEL},"dispatcher":"${DISPATCHER}","micro_batch_size":${MICRO_BATCH_SIZE},"global_batch_size":${GLOBAL_BATCH_SIZE},"num_microbatches":${NUM_MICROBATCHES},"gpu_count":${GPU_COUNT},"nnodes":${NNODES},"node_rank":${NODE_RANK},"numa_mode":"${NUMA_MODE}","pytorch_cuda_alloc_conf":"${PYTORCH_CUDA_ALLOC_CONF:-}","nccl_graph_register":"${NCCL_GRAPH_REGISTER:-}","nvte_cutedsl_fused_grouped_mlp":"${NVTE_CUTEDSL_FUSED_GROUPED_MLP:-}","run_date":"${RUN_DATE}"}
JSON
printf '%s\n' "${INFO[*]}" | tee "${RESULT_DIR}/run_info.txt"
set +e
"${COMMAND[@]}" 2>&1 | tee "${RESULT_DIR}/train.log"
STATUS=${PIPESTATUS[0]}
set -e
printf 'status=%s\n' "${STATUS}" >>"${RESULT_DIR}/run_info.txt"
exit "${STATUS}"
