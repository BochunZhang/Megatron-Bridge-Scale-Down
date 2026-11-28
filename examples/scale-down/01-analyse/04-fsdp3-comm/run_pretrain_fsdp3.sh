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
# shellcheck source=../gpu_clocks.sh
source "${SCRIPT_DIR}/../gpu_clocks.sh"

MODEL=""
RECIPE=""
SEQ_LENGTH=""
DTYPE="${DTYPE:-bf16}"
GRAD_COMM_DTYPE=""
PROFILE="${PROFILE:-none}"
METRIC_FREQUENCY=""
NSYS_REPORT_ONLY=false
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
LOCAL_ATTENTION_GRAPH=false
LABEL_DTYPE_AND_DP=false
MERGE_GRAPHS=false
FORCE_BALANCE=false
FSDP_LEVEL="${FSDP_LEVEL:-3}"
OVERLAP=false
SEPARATE_RS=false
TP_COMM_OVERLAP=false
SYMMETRIC=false
MULTIMEM=false
OFFLOAD=""
CPU_GRAD_ACCUMULATION=false
CPU_OPTIMIZER=false
TORCH_THREAD=""
NVME_PATH=""
DDP_OVERLAP=false
HYBRIDEP_NUM_SMS="${HYBRIDEP_NUM_SMS:-32}"
GPU_COUNT="${GPU_COUNT:-4}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NNODES="${NNODES:-}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-${DLC_MASTER_ADDR:-}}"
MASTER_PORT="${MASTER_PORT:-29501}"
RUN_DATE="${RUN_DATE:-$(date +%y%m%d-%H%M%S)}"
RUN_NAME=""
STABLE_RESULT_DIR=false
UNLOCK_GPU_CLOCKS=false
DRY_RUN=false
PRINT_RESULT_DIR=false
RESUME=false
EXTRA_OVERRIDES=()

usage() {
    cat <<'USAGE'
Usage: run_pretrain_fsdp3.sh --model 27b|qwen38-27b|397b|2.4t|122b --seq-length N [options]

Models: 27b / 397b / 122b are Qwen3.5-27B / 397B-A17B / 122B-A10B; 2.4t (alias 2p4t)
is Qwen3.8-2.4T-A95B. qwen38-27b is Qwen3.8-27B.
The model selects its 4-GPU GB200 FSDP1 recipe.

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
                                 reduce-scatter overlapped with backward and parameter
                                 all-gather in the optimizer step (default: 3)
  --ddp-overlap                  DDP only: prefetch parameter buckets during the first
                                 micro-batch forward of the next iteration
  --overlap                      Megatron-FSDP only: give parameter all-gathers their
                                 own DP communicators (dist.create_all_gather_group,
                                 Megatron-LM --create-all-gather-group), so they no
                                 longer queue behind gradient reduce-scatters on one
                                 NCCL communicator; only FSDP3 overlaps the two. Names get
                                 "-overlap"
  --separate-rs                  FSDP3 only: route gradient reduce-scatter through
                                 a dedicated DP communicator. Names get "-separate-rs"
  --symmetric                    Megatron-FSDP only: allocate the FSDP buffers (sharded
                                 weights and gradients, max-pool double buffers holding
                                 the all-gather output and reduce-scatter input) from an
                                 ncclMemAlloc pool and register it once, after the first
                                 step, as NCCL_WIN_COLL_SYMMETRIC windows on every FSDP
                                 communicator (ddp.nccl_ub), so NCCL runs its symmetric
                                 kernels. Exports NCCL_NVLS_ENABLE=1 and refuses
                                 NCCL_ALGO. Names get "-symm"
  --multimem                     With --symmetric and NCCL 2.32.3, select STMC
                                 AllGather and LDMC ReduceScatter, with Ring for
                                 buffers without symmetric windows
  --tp N                         Tensor parallel size; N > 1 turns on sequence
                                 parallelism (default: 1)
  --tp-comm-overlap              Enable TE TP communication overlap (requires TP > 1);
                                 names get "-tp-overlap"
  --cp N                         Context parallel size (default: 1)
  --cuda-graph off|on            off: no CUDA graph (default). on with FSDP: dense uses
                                 full_iteration; MoE uses native local graphs for attention,
                                 router and preprocess. Extended FSDP1/2 support for HybridEP,
                                 TP and quantized compute requires the corresponding MCore code.
                                 FSDP3 also graphs eligible parameter prefetches.
                                 on with DDP: TE per-layer graphs, dense attn+mlp,
                                 MoE attn+moe_router+moe_preprocess where supported.
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
  --dtype|--precision bf16|mxfp8|nvfp4
                                 Compute mode (default: bf16)
  --mxfp8                        Same as --dtype mxfp8
  --grad-comm-dtype bf16|fp32    Gradient reduce-scatter dtype for FSDP2/3
                                 (default: follow the recipe)
  --offload MODULE[,MODULE...]   CPU offload for optimizer, weight, and/or gradient.
                                 CUDA graph testing requires FSDP3.
  --gpu|--gpus N                 Global GPU count (default: 4)
  --nproc-per-node N             GPUs per node (default: 4)
  --nnodes N                     Number of nodes (default: gpu / nproc-per-node)
  --node-rank N                  Node rank (default: NODE_RANK or 0)
  --master-addr HOST             Multi-node rendezvous address
  --master-port N                Multi-node rendezvous port (default: 29501)
  --cpu-grad-accumulation        Stage reduced shards in pinned MaxPool double buffers and accumulate on CPU
  --cpu-optimizer                Run AdamW on CPU; implies optimizer and gradient offload
  --torch-thread N               PyTorch CPU intra-op threads per rank, including CPU
                                 gradient accumulation and AdamW (default: inherit)
  --local-attention-graph        Use native attention-only graphs with --cuda-graph on,
                                 BF16 FSDP1 or FSDP3 and TP=1 (including on dense models)
  --label-dtype-and-dp            Label BF16 explicitly, add DP size to the report name
                                 and label full-iteration graphs explicitly
  --nvme-path DIR                Use the native NVMe AIO extension for weights, gradients and AdamW state;
                                 implies --cpu-optimizer. DIR is local to each node.
  MEGATRON_LM_DIR                Optional environment variable pointing to the MCore checkout
  --profile none|nsys|pytorch    Profiling mode (default: none). nsys records node 0
                                 into one report. pytorch exports per-rank Chrome traces
                                 under each node's torch_profile directory.
  --nsys                         Alias for --profile nsys
  --nsys-report-only             Capture the native Nsight Systems report without SQLite export
  --metric [HZ]                  Enable nsys GPU metrics on all GPUs on node 0
                                 (default frequency: 20000 Hz)
  --merge                        With nsys and --cuda-graph on, draw each graph launch
                                 as one block (--cuda-graph-trace=graph) instead of
                                 its kernels; names get "-merge"
  --profile-start N              First profiled iteration (default: 7)
  --profile-end N                Stop iteration, exclusive (default: 8, i.e. one step)
  --run-date YYMMDD-HHMMSS       Shared matrix timestamp
  --run-name NAME                Optional human-readable case name
  --stable-result-dir            Omit the timestamp and append --run-name if provided;
                                 require a nonempty node 0 report for successful nsys runs
  --results-root PATH            Result root
  --override KEY=VALUE           Additional run_recipe.py override (repeatable)
  --unlock                       Restore automatic GPU clocks (TDP lock by default)
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
        --grad-comm-dtype)
            need_value "$@"
            [[ "$2" == bf16 || "$2" == fp32 ]] || { echo "--grad-comm-dtype must be bf16 or fp32" >&2; exit 2; }
            GRAD_COMM_DTYPE="$2"
            shift 2;;
        --tp|--tensor-parallel) need_value "$@"; TENSOR_PARALLEL="$2"; shift 2;;
        --tp-comm-overlap) TP_COMM_OVERLAP=true; shift;;
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
        --ddp-overlap) DDP_OVERLAP=true; shift;;
        --overlap) OVERLAP=true; shift;;
        --separate-rs) SEPARATE_RS=true; shift;;
        --symmetric) SYMMETRIC=true; shift;;
        --multimem) MULTIMEM=true; shift;;
        --cpu-grad-accumulation) CPU_GRAD_ACCUMULATION=true; shift;;
        --cpu-optimizer) CPU_OPTIMIZER=true; shift;;
        --torch-thread)
            need_value "$@"
            [[ "$2" =~ ^[1-9][0-9]*$ ]] || { echo "--torch-thread must be a positive integer" >&2; exit 2; }
            TORCH_THREAD="$2"
            shift 2;;
        --nvme-path) need_value "$@"; NVME_PATH="$2"; [[ -n "${NVME_PATH}" ]] || { echo "--nvme-path requires a directory" >&2; exit 2; }; shift 2;;
        --offload) need_value "$@"; OFFLOAD="$2"; [[ -n "${OFFLOAD}" ]] || { echo "--offload requires a module list" >&2; exit 2; }; shift 2;;
        --local-attention-graph) LOCAL_ATTENTION_GRAPH=true; shift;;
        --label-dtype-and-dp) LABEL_DTYPE_AND_DP=true; shift;;
        --gpu|--gpus) need_value "$@"; GPU_COUNT="$2"; shift 2;;
        --nproc-per-node) need_value "$@"; GPUS_PER_NODE="$2"; shift 2;;
        --nnodes) need_value "$@"; NNODES="$2"; shift 2;;
        --node-rank) need_value "$@"; NODE_RANK="$2"; shift 2;;
        --master-addr) need_value "$@"; MASTER_ADDR="$2"; shift 2;;
        --master-port) need_value "$@"; MASTER_PORT="$2"; shift 2;;
        --profile) need_value "$@"; PROFILE="$2"; shift 2;;
        --nsys) PROFILE=nsys; shift;;
        --nsys-report-only) NSYS_REPORT_ONLY=true; shift;;
        --metric|--metric=*)
            METRIC_FREQUENCY=20000
            if [[ "$1" == --metric=* ]]; then
                METRIC_FREQUENCY="${1#*=}"
            elif (( $# > 1 )) && [[ "$2" != -* ]]; then
                METRIC_FREQUENCY="$2"
                shift
            fi
            if ! [[ "${METRIC_FREQUENCY}" =~ ^[1-9][0-9]{1,5}$ ]] || ! (( METRIC_FREQUENCY <= 200000 )); then
                echo "--metric frequency must be an integer from 10 to 200000 Hz" >&2
                exit 2
            fi
            shift
            ;;
        --merge) MERGE_GRAPHS=true; shift;;
        --profile-start|--profile-step-start) need_value "$@"; PROFILE_START="$2"; shift 2;;
        --profile-end|--profile-step-end) need_value "$@"; PROFILE_END="$2"; shift 2;;
        --run-date) need_value "$@"; RUN_DATE="$2"; shift 2;;
        --run-name) need_value "$@"; RUN_NAME="$2"; shift 2;;
        --results-root) need_value "$@"; RESULTS_ROOT="$2"; shift 2;;
        --stable-result-dir) STABLE_RESULT_DIR=true; shift;;
        --override) need_value "$@"; EXTRA_OVERRIDES+=("$2"); shift 2;;
        --unlock) UNLOCK_GPU_CLOCKS=true; shift;;
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
    qwen38-27b) MODEL_ID=Qwen/Qwen3.8-27B; MODEL_TAG=qwen38_27b; DEFAULT_RECIPE=qwen38_text_27b_pretrain_4gpu_gb200_bf16_fsdp1_config;;
    2.4t) MODEL_ID=Qwen/Qwen3.8-2.4T-A95B; MODEL_TAG=qwen38_2.4t; DEFAULT_RECIPE=qwen38_text_2p4t_a95b_pretrain_4gpu_gb200_bf16_fsdp1_config;;
    *) echo "Unsupported model: ${MODEL} (use 27b, qwen38-27b, 397b, 2.4t or 122b)" >&2; exit 2;;
esac
DENSE=false
[[ "${MODEL}" == 27b || "${MODEL}" == qwen38-27b ]] && DENSE=true
RECIPE="${RECIPE:-${DEFAULT_RECIPE}}"
case "${DTYPE}" in bf16|mxfp8|nvfp4) ;; *) echo "--dtype must be bf16, mxfp8 or nvfp4" >&2; exit 2;; esac
case "${PROFILE}" in none|nsys|pytorch) ;; *) echo "--profile must be none, nsys or pytorch" >&2; exit 2;; esac
if [[ "${PROFILE}" == pytorch && -n "${METRIC_FREQUENCY}" ]]; then
    echo "--metric requires Nsight Systems and cannot be used with --profile pytorch" >&2; exit 2
fi
[[ -n "${METRIC_FREQUENCY}" ]] && PROFILE=nsys
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
    ddp) SHARDING_STRATEGY=no_shard; GRAD_OVERLAP=true; PARAM_OVERLAP="${DDP_OVERLAP}"; DP_TAG=ddp; FSDP_JSON=null;;
    *) echo "--fsdp must be ddp, 1, 2, or 3" >&2; exit 2;;
esac
[[ "${DDP_OVERLAP}" == false || "${FSDP_LEVEL}" == ddp ]] || { echo "--ddp-overlap requires --fsdp ddp" >&2; exit 2; }
if [[ -n "${GRAD_COMM_DTYPE}" && "${FSDP_LEVEL}" != 2 && "${FSDP_LEVEL}" != 3 ]]; then
    echo "--grad-comm-dtype requires --fsdp 2 or 3" >&2
    exit 2
fi
if [[ -n "${NVME_PATH}" ]]; then
    CPU_OPTIMIZER=true
    OFFLOAD="${OFFLOAD:+${OFFLOAD},}weight"
fi
if [[ "${CPU_OPTIMIZER}" == true ]]; then
    [[ "${FSDP_LEVEL}" == 3 && "${DTYPE}" == bf16 ]] || {
        echo "--cpu-optimizer and --nvme-path require --fsdp 3 and BF16" >&2; exit 2
    }
    CPU_GRAD_ACCUMULATION=true
    OFFLOAD="${OFFLOAD:+${OFFLOAD},}optimizer"
fi
if [[ "${CPU_GRAD_ACCUMULATION}" == true ]]; then
    OFFLOAD="${OFFLOAD:+${OFFLOAD},}gradient"
fi
OFFLOAD_TAG=""
OFFLOAD_JSON='[]'
if [[ -n "${OFFLOAD}" ]]; then
    [[ "${OFFLOAD}" =~ ^(optimizer|weight|gradient)(,(optimizer|weight|gradient))*$ ]] || {
        echo "--offload expects a comma-separated list of optimizer, weight, and gradient" >&2
        exit 2
    }
    [[ "${FSDP_LEVEL}" != ddp && "${SYMMETRIC}" == false ]] || {
        echo "--offload requires Megatron-FSDP and no --symmetric" >&2
        exit 2
    }
    [[ "${CUDA_GRAPH}" == off || "${FSDP_LEVEL}" == 3 ]] || {
        echo "--offload with --cuda-graph on requires --fsdp 3" >&2
        exit 2
    }
    requested_offload=",${OFFLOAD},"
    OFFLOAD=""
    for component in optimizer weight gradient; do
        if [[ "${requested_offload}" == *",${component},"* ]]; then
            OFFLOAD="${OFFLOAD:+${OFFLOAD},}${component}"
        fi
    done
    OFFLOAD_TAG="-offload_${OFFLOAD//,/_}"
    OFFLOAD_JSON="[\"${OFFLOAD//,/\",\"}\"]"
    if [[ "${FSDP_LEVEL}" == 1 && "${requested_offload}" == *,gradient,* ]]; then
        GRAD_OVERLAP=true
    fi
fi
[[ "${CPU_GRAD_ACCUMULATION}" == true ]] && OFFLOAD_TAG+="-cpu_grad_accum"
[[ "${CPU_OPTIMIZER}" == true ]] && OFFLOAD_TAG+="-cpu_adamw"
[[ -n "${NVME_PATH}" ]] && OFFLOAD_TAG+="-nvme"
# Only Megatron-FSDP builds its parameter all-gathers on the separate communicators.
[[ "${OVERLAP}" == false || "${FSDP_LEVEL}" != ddp ]] || { echo "--overlap requires --fsdp 1, 2 or 3" >&2; exit 2; }
[[ "${SEPARATE_RS}" == false || "${FSDP_LEVEL}" == 3 ]] || { echo "--separate-rs requires --fsdp 3" >&2; exit 2; }
# The max-pool double buffers are Megatron-FSDP's. A forced NCCL_ALGO keeps every
# collective on the classic ring / NVLS kernels even on registered windows.
[[ "${SYMMETRIC}" == false || "${FSDP_LEVEL}" != ddp ]] || { echo "--symmetric requires --fsdp 1, 2 or 3" >&2; exit 2; }
[[ "${SYMMETRIC}" == false || -z "${NCCL_ALGO:-}" ]] || { echo "--symmetric requires NCCL_ALGO to be unset (got ${NCCL_ALGO})" >&2; exit 2; }
if [[ "${MULTIMEM}" == true ]]; then
    [[ "${SYMMETRIC}" == true ]] || { echo "--multimem requires --symmetric" >&2; exit 2; }
    # NCCL 2.32.3 tries symmetric kernels before the general-kernel fallback.
    # Bridge registers FSDP windows after step 1, so keep Ring available for warmup.
    export NCCL_ALGO='AllGather:Ring;ReduceScatter:Ring'
    export NCCL_SYM_KERNEL='AllGather:AllGather_STMC;ReduceScatter:ReduceScatter_LDMC'
    export NCCL_WIN_ENABLE=1 NCCL_SYM_CE_THRESHOLD=9223372036854775807
    unset NCCL_CTA_POLICY
fi
for name in SEQ_LENGTH TENSOR_PARALLEL CONTEXT_PARALLEL MICRO_BATCH_SIZE GLOBAL_BATCH_SIZE TRAIN_ITERS GPU_COUNT GPUS_PER_NODE MASTER_PORT HYBRIDEP_NUM_SMS; do
    [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || { echo "${name} must be a positive integer" >&2; exit 2; }
done
[[ "${TP_COMM_OVERLAP}" == false ]] || (( TENSOR_PARALLEL > 1 )) || { echo "--tp-comm-overlap requires --tp > 1" >&2; exit 2; }
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
if [[ "${PROFILE}" != none ]] && ! (( PROFILE_START < PROFILE_END && PROFILE_END < TRAIN_ITERS )); then
    echo "${PROFILE} requires profile-start < profile-end < train-iters" >&2; exit 2
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
if [[ "${DENSE}" == true ]]; then
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
    if [[ -n "${NUM_EXPERTS}" ]]; then
        (( NUM_EXPERTS % EXPERT_PARALLEL == 0 )) || { echo "num-experts must be divisible by ep" >&2; exit 2; }
    fi
    # A cropped MoE model keeps every remaining layer MoE.
    if [[ -n "${NUM_LAYERS}" && -z "${MOE_LAYER_FREQ}" ]]; then
        MOE_LAYER_FREQ="$(ones_list "${NUM_LAYERS}")"
    fi
fi

GRAPH_IMPL=none
GRAPH_MODULES="[]"
if [[ "${LOCAL_ATTENTION_GRAPH}" == true && ( "${CUDA_GRAPH}" != on || ( "${FSDP_LEVEL}" != 1 && "${FSDP_LEVEL}" != 3 ) ) ]]; then
    echo "--local-attention-graph requires --cuda-graph on and --fsdp 1 or 3" >&2; exit 2
fi
if [[ "${CUDA_GRAPH}" == on ]]; then
    if [[ "${FSDP_LEVEL}" == ddp ]]; then
        GRAPH_IMPL=transformer_engine
        if [[ "${DENSE}" == true ]]; then
            GRAPH_MODULES="[attn,mlp]"
        elif [[ "${DTYPE}" != bf16 && "${DISPATCHER}" == alltoall ]]; then
            # Quantized router padding makes all-to-all preprocess synchronize token counts to host.
            GRAPH_MODULES="[attn,moe_router]"
        else
            GRAPH_MODULES="[attn,moe_router,moe_preprocess]"
        fi
    elif [[ "${DENSE}" == true && "${LOCAL_ATTENTION_GRAPH}" == false ]]; then
        GRAPH_IMPL=full_iteration
    elif [[ "${DENSE}" == false && "${LOCAL_ATTENTION_GRAPH}" == false &&
        "${DISPATCHER}" == hybridep && "${SYMMETRIC}" == false && -z "${OFFLOAD}" &&
        ( "${FSDP_LEVEL}" == 1 || "${FSDP_LEVEL}" == 2 ) ]]; then
        GRAPH_IMPL=local
        GRAPH_MODULES="[attn,moe_router,moe_preprocess]"
    else
        [[ "${FSDP_LEVEL}" == 1 || "${FSDP_LEVEL}" == 3 ]] || {
            echo "FSDP attention CUDA graphs require --fsdp 1 or 3" >&2; exit 2;
        }
        [[ "${DTYPE}" == bf16 ]] || { echo "FSDP attention CUDA graphs require --dtype bf16" >&2; exit 2; }
        (( TENSOR_PARALLEL == 1 )) || {
            echo "FSDP attention CUDA graphs require --tp 1" >&2; exit 2;
        }
        [[ "${SYMMETRIC}" == false ]] || { echo "FSDP attention CUDA graphs require --symmetric to be off" >&2; exit 2; }
        GRAPH_IMPL=local
        GRAPH_MODULES="[attn]"
        if [[ "${DENSE}" == false && "${LOCAL_ATTENTION_GRAPH}" == false ]]; then
            GRAPH_MODULES="[attn,moe_router,moe_preprocess]"
        fi
    fi
fi
if [[ "${CPU_GRAD_ACCUMULATION}" == true ]]; then
    if [[ ( "${FSDP_LEVEL}" != 2 && "${FSDP_LEVEL}" != 3 ) || ( "${GRAPH_IMPL}" != none && "${GRAPH_IMPL}" != local ) ]]; then
        echo "--cpu-grad-accumulation requires eager FSDP2/3 or FSDP3 with local attention CUDA graphs" >&2
        exit 2
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
if [[ "${DENSE}" == true ]]; then EXPERTS_TAG=0; else EXPERTS_TAG="${NUM_EXPERTS:-full}"; fi
# --overlap adds "-overlap" and --symmetric adds "-symm". Synchronous DDP uses
# "-sync-ag" while --ddp-overlap uses the existing DDP name for forward prefetch.
# Quantized modes include the dtype, forced balanced routing "-fb", --full / --linear
# "-attn_<full|linear>" after the layer count, and TP > 1 "-tp_<N>" before CP.
OVERLAP_TAG=""
[[ "${OVERLAP}" == true ]] && OVERLAP_TAG="-overlap"
[[ "${SEPARATE_RS}" == true ]] && OVERLAP_TAG+="-separate-rs"
[[ "${SYMMETRIC}" == true ]] && OVERLAP_TAG+="-symm"
[[ "${FSDP_LEVEL}" == ddp && "${DDP_OVERLAP}" == false ]] && OVERLAP_TAG+="-sync-ag"
DTYPE_TAG=""
[[ "${DTYPE}" != bf16 || "${LABEL_DTYPE_AND_DP}" == true ]] && DTYPE_TAG="-${DTYPE}"
[[ "${FORCE_BALANCE}" == true ]] && DTYPE_TAG+="-fb"
ATTN_TAG=""
[[ "${ATTENTION}" != hybrid ]] && ATTN_TAG="-attn_${ATTENTION}"
TP_TAG=""
(( TENSOR_PARALLEL > 1 )) && TP_TAG="-tp_${TENSOR_PARALLEL}"
[[ "${TP_COMM_OVERLAP}" == true ]] && TP_TAG+="-tp-overlap"
CG_TAG="${CUDA_GRAPH}"
[[ "${LABEL_DTYPE_AND_DP}" == true && "${GRAPH_IMPL}" == full_iteration ]] && CG_TAG+="-full-iteration"
[[ "${GRAPH_IMPL}" == local ]] && CG_TAG+="-partial"
[[ "${GRAPH_IMPL}" == local && "${GRAPH_MODULES}" == *moe_router* ]] && CG_TAG+="-moe"
[[ "${GRAPH_TRACE}" == graph ]] && CG_TAG+="-merge"
METRIC_TAG="${METRIC_FREQUENCY:+-metric_${METRIC_FREQUENCY}}"
TORCH_THREAD_TAG="${TORCH_THREAD:+-torch_thread_${TORCH_THREAD}}"
GRAD_COMM_TAG="${GRAD_COMM_DTYPE:+-grad_comm_${GRAD_COMM_DTYPE}}"
CASE_TAG="-date_${RUN_DATE}"
if [[ "${STABLE_RESULT_DIR}" == true ]]; then
    [[ -z "${RUN_NAME}" || "${RUN_NAME}" =~ ^[a-zA-Z0-9_-]+$ ]] || {
        echo "--run-name must be filesystem-safe with --stable-result-dir" >&2
        exit 2
    }
    CASE_TAG="${RUN_NAME:+-${RUN_NAME}}"
fi
CASE_NAME="model_${MODEL}-${DP_TAG}${OVERLAP_TAG}${DTYPE_TAG}-layers_${LAYERS_TAG}${ATTN_TAG}-mtp_${MTP_TAG}-experts_${EXPERTS_TAG}-cg_${CG_TAG}-mbs_${MICRO_BATCH_SIZE}-seq_${SEQ_LENGTH}${TP_TAG}-cp_${CONTEXT_PARALLEL}-ep_${EXPERT_PARALLEL}-disp_${DISPATCHER}-gbs_${GLOBAL_BATCH_SIZE}-gpus_${GPU_COUNT}-profile_${PROFILE}${CASE_TAG}${OFFLOAD_TAG}${TORCH_THREAD_TAG}${GRAD_COMM_TAG}${METRIC_TAG}"
RESULT_DIR="${RESULTS_ROOT}/${CASE_NAME}/node${NODE_RANK}"
if [[ "${PRINT_RESULT_DIR}" == true ]]; then
    printf '%s\n' "${RESULT_DIR}"
    exit 0
fi
# nsys report name, e.g. qwen35_27b-fsdp_3-mxfp8-experts_0-cg_off-tp_2-cp_1-ep_1-mbs_1-gbs_8-seq_4k.
if (( SEQ_LENGTH % 1024 == 0 )); then SEQ_TAG="$((SEQ_LENGTH / 1024))k"; else SEQ_TAG="${SEQ_LENGTH}"; fi
NSYS_NAME="${MODEL_TAG}-${DP_TAG}${OVERLAP_TAG}${DTYPE_TAG}${ATTN_TAG}-experts_${EXPERTS_TAG}-cg_${CG_TAG}${TP_TAG}-cp_${CONTEXT_PARALLEL}-ep_${EXPERT_PARALLEL}-mbs_${MICRO_BATCH_SIZE}-gbs_${GLOBAL_BATCH_SIZE}-seq_${SEQ_TAG}${OFFLOAD_TAG}${TORCH_THREAD_TAG}${GRAD_COMM_TAG}${METRIC_TAG}"
if [[ "${LABEL_DTYPE_AND_DP}" == true ]]; then
    NSYS_NAME+="-dp_${DATA_PARALLEL}"
fi
has_required_profile() {
    if [[ "${PROFILE}" == pytorch ]]; then
        local rank
        for (( rank = FIRST_PROFILE_RANK; rank < FIRST_PROFILE_RANK + GPUS_PER_NODE; rank++ )); do
            [[ -s "${RESULT_DIR}/torch_profile/rank-${rank}.json.gz" ]] || return 1
        done
        return 0
    fi
    [[ "${STABLE_RESULT_DIR}" == false || "${PROFILE}" != nsys || "${NODE_RANK}" != 0 ||
        -s "${RESULT_DIR}/${NSYS_NAME}.nsys-rep" ]]
}
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

# HybridEP topology on GB200 NVL72: the whole EP group sits in one MNNVL domain.
if [[ "${DISPATCHER}" == hybridep ]]; then
    export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN="${EXPERT_PARALLEL}" NVLINK_DOMAIN_SIZE=72 USE_MNNVL=1
fi
# Keep NCCL to warnings even when the shell exports NCCL_DEBUG=INFO; INFO floods train.log
# with per-communicator channel/topology lines.
export NCCL_DEBUG=WARN
unset NCCL_DEBUG_SUBSYS NCCL_DEBUG_FILE || true

# Nsight Systems records node 0. PyTorch exports one trace for each local worker
# using its global rank, under that node's result directory.
FIRST_PROFILE_RANK=0
PROFILER_START="${PROFILE_START}"
PROFILER_END="${PROFILE_END}"
TENSORBOARD_DIR=null
if [[ "${PROFILE}" == pytorch ]]; then
    FIRST_PROFILE_RANK=$((NODE_RANK * GPUS_PER_NODE))
    # Bridge advances the PyTorch schedule before executing training iteration 0.
    # Shift the schedule boundaries to capture the same training steps as Nsight.
    PROFILER_START=$((PROFILE_START + 1))
    PROFILER_END=$((PROFILE_END + 1))
    TENSORBOARD_DIR="${RESULT_DIR}/tensorboard"
fi
PROFILE_RANKS="${FIRST_PROFILE_RANK}"
for (( rank = FIRST_PROFILE_RANK + 1; rank < FIRST_PROFILE_RANK + GPUS_PER_NODE; rank++ )); do
    PROFILE_RANKS+=",${rank}"
done
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
    # TP > 1 always runs with sequence parallelism.
    "model.tensor_model_parallel_size=${TENSOR_PARALLEL}" "model.sequence_parallel=${SEQUENCE_PARALLEL}"
    "model.context_parallel_size=${CONTEXT_PARALLEL}" "model.pipeline_model_parallel_size=1"
    "model.expert_model_parallel_size=${EXPERT_PARALLEL}" "model.expert_tensor_parallel_size=1"
    "model.cp_comm_type=p2p" "model.mtp_num_layers=0" "model.linear_attention_freq=${LINEAR_ATTENTION_FREQ}"
    "model.recompute_granularity=null" "model.recompute_method=null" "model.recompute_num_layers=null"
    "model.recompute_modules=[]" "model.fine_grained_activation_offloading=false" "model.offload_modules=[]"
    "${DP_OVERRIDES[@]}" "dist.create_all_gather_group=${OVERLAP}"
    "dist.create_reduce_scatter_group=${SEPARATE_RS}" "ddp.use_distributed_optimizer=true"
    "ddp.nccl_ub=${SYMMETRIC}"
    # grad_reduce_in_fp32 selects the MCore DDP buffer dtype. Megatron-FSDP
    # uses megatron_fsdp_grad_comm_dtype or falls back to the main gradient dtype.
    "ddp.data_parallel_sharding_strategy=${SHARDING_STRATEGY}" "ddp.average_in_collective=false"
    "ddp.overlap_grad_reduce=${GRAD_OVERLAP}" "ddp.overlap_param_gather=${PARAM_OVERLAP}" "ddp.grad_reduce_in_fp32=true"
    # CommOverlapConfig.setup() applies these settings to DDP at runtime.
    "comm_overlap.overlap_grad_reduce=${GRAD_OVERLAP}" "comm_overlap.overlap_param_gather=${PARAM_OVERLAP}"
    "checkpoint.save=null" "checkpoint.load=null"
    "logger.log_interval=1" "logger.tensorboard_dir=${TENSORBOARD_DIR}" "logger.save_config_filepath=${RESULT_DIR}/config.yaml"
    "profiling.use_nsys_profiler=$([[ "${PROFILE}" == nsys ]] && echo true || echo false)"
    "profiling.use_pytorch_profiler=$([[ "${PROFILE}" == pytorch ]] && echo true || echo false)"
    "profiling.profile_step_start=${PROFILER_START}" "profiling.profile_step_end=${PROFILER_END}"
    "profiling.profile_ranks=[${PROFILE_RANKS}]"
    # nvtx_ranges also turns on Megatron-FSDP's per-layer all-gather / reduce-scatter
    # ranges ("FSDP AG fwd prefetch L2 ...", "FSDP RS L1 ...").
    "profiling.nvtx_ranges=$([[ "${PROFILE}" == nsys ]] && echo true || echo false)"
    # emit_nvtx(record_shapes=True): every op range, including c10d
    # record_param_comms for FSDP all-gather / reduce-scatter, carries tensor shapes.
    "profiling.record_shapes=$([[ "${PROFILE}" == nsys ]] && echo true || echo false)"
)
if [[ "${PARAM_OVERLAP}" == false ]]; then
    # Keep synchronous post-update gathers even if a recipe requests optimizer overlap.
    OVERRIDES+=("optimizer.overlap_param_gather=false"
        "optimizer.overlap_param_gather_with_optimizer_step=false"
        "comm_overlap.overlap_param_gather_with_optimizer_step=false")
fi
if [[ "${SYMMETRIC}" == true ]]; then
    # nccl_ub allocates every Megatron-FSDP buffer created under its allocation context
    # (sharded weights, gradients, the max-pool double buffers that hold the all-gather
    # output and reduce-scatter input) from one ncclMemAlloc MemPool. Bridge turns on
    # fsdp_manual_registration, so train.py registers the whole pool once after the
    # first step on every FSDP communicator (including the --overlap all-gather ones)
    # as NCCL_WIN_COLL_SYMMETRIC windows. The double-buffer settings are pinned here
    # rather than inherited from the recipe: the max pool gives the full- and
    # linear-attention layers one fixed set of slots, and units outside it (embedding,
    # LM head) get persistent buffers inside the pool instead of dynamic, unregistered
    # memory. The multicast symmetric kernels (AllGather_STMC, ReduceScatter_LDMC)
    # need NVLS; run_recipe.py only defaults NCCL_NVLS_ENABLE, so force it.
    OVERRIDES+=(
        "ddp.disable_symmetric_registration=false" "ddp.fsdp_double_buffer=true"
        "ddp.megatron_fsdp_max_pool_double_buffer=true" "ddp.fsdp_db_use_persist_buf_on_alloc_fail=true"
    )
    export NCCL_NVLS_ENABLE=1
fi
OVERRIDES+=("model.cuda_graph_impl=${GRAPH_IMPL}" "model.cuda_graph_scope=null" "model.cuda_graph_modules=${GRAPH_MODULES}")
case "${GRAPH_IMPL}" in
    transformer_engine|local)
        OVERRIDES+=(
            "model.cuda_graph_warmup_steps=3" "model.use_te_rng_tracker=true" "rng.te_rng_tracker=true"
        )
        if [[ "${GRAPH_IMPL}" == local ]]; then
            OVERRIDES+=(
                "model.gradient_accumulation_fusion=false"
                "ddp.megatron_fsdp_cuda_graph_mode=false"
            )
            if [[ "${FSDP_LEVEL}" == 3 ]]; then
                # Prefetch into the fixed weight slots consumed by the attention graph.
                OVERRIDES+=(
                    "ddp.fsdp_double_buffer=true" "ddp.megatron_fsdp_max_pool_double_buffer=true"
                    "ddp.fsdp_all_gather_in_start_param_sync=true"
                )
            fi
        fi
        ;;
    full_iteration)
        # Keep gradient references and dispatch FSDP3 parameter gathers inside the graph.
        OVERRIDES+=(
            "model.cuda_graph_warmup_steps=3" "ddp.megatron_fsdp_cuda_graph_mode=true"
            "ddp.fsdp_all_gather_in_start_param_sync=false"
            "rerun_state_machine.check_for_nan_in_loss=false" "ddp.check_for_nan_in_grad=false"
        )
        ;;
esac
if [[ "${CUDA_GRAPH}" == on ]]; then
    # Keep NCCL graph registration off. FSDP graphs hold NCCL collectives,
    # where registering expandable segments (often inherited through
    # PYTORCH_CUDA_ALLOC_CONF) can hit illegal memory access, and TECudaGraphHelper
    # asserts it under expandable segments. This only stops NCCL from registering
    # buffers during capture; the --symmetric windows are registered explicitly after
    # the first step, before capture.
    export NCCL_GRAPH_REGISTER=0
fi
[[ -n "${NUM_LAYERS}" ]] && OVERRIDES+=("model.num_layers=${NUM_LAYERS}")
[[ -n "${NUM_EXPERTS}" ]] && OVERRIDES+=("model.num_moe_experts=${NUM_EXPERTS}")
[[ -n "${MOE_LAYER_FREQ}" ]] && OVERRIDES+=("model.moe_layer_freq=${MOE_LAYER_FREQ}")
[[ -n "${MTP_NUM_LAYERS}" ]] && OVERRIDES+=("model.mtp_num_layers=${MTP_NUM_LAYERS}")
case "${DISPATCHER}" in
    alltoall) OVERRIDES+=("model.moe_token_dispatcher_type=alltoall" "model.moe_flex_dispatcher_backend=null" "model.moe_flex_dispatcher_num_sms=null");;
    hybridep) OVERRIDES+=("model.moe_token_dispatcher_type=flex" "model.moe_flex_dispatcher_backend=hybridep" "model.moe_flex_dispatcher_num_sms=${HYBRIDEP_NUM_SMS}");;
esac
case "${DTYPE}" in
    mxfp8)
        OVERRIDES+=("mixed_precision.fp8=e4m3" "mixed_precision.fp8_recipe=mxfp8" "mixed_precision.fp8_param_gather=false" "mixed_precision.reuse_grad_buf_for_mxfp8_param_ag=false")
        [[ "${DENSE}" == false ]] && OVERRIDES+=("model.moe_router_padding_for_fp8=true")
        ;;
    nvfp4)
        OVERRIDES+=("mixed_precision.fp4=e2m1" "mixed_precision.fp4_recipe=nvfp4" "mixed_precision.fp4_param_gather=false")
        [[ "${DENSE}" == false ]] && OVERRIDES+=("model.moe_router_padding_for_quantization=true")
        ;;
esac
# MCore draws N(0,1) router logits per rank and keeps the router GEMM, top-k, dispatch
# and expert compute; only the expert choice of each token becomes uniform random.
[[ "${FORCE_BALANCE}" == true ]] && OVERRIDES+=("model.moe_router_force_load_balancing=true")
if (( ${#EXTRA_OVERRIDES[@]} > 0 )); then
    OVERRIDES+=("${EXTRA_OVERRIDES[@]}")
fi
GRAD_COMM_JSON=null
if [[ -n "${GRAD_COMM_DTYPE}" ]]; then
    OVERRIDES+=("ddp.megatron_fsdp_grad_comm_dtype={_target_:megatron.bridge.utils.activation_map.str_to_dtype,name:${GRAD_COMM_DTYPE}}")
    GRAD_COMM_JSON="\"${GRAD_COMM_DTYPE}\""
fi
if [[ -n "${OFFLOAD}" ]]; then
    OVERRIDES+=("ddp.fsdp_cpu_offload=[${OFFLOAD}]" "optimizer.optimizer_cpu_offload=false")
    if [[ "${requested_offload}" == *,optimizer,* ]]; then
        OVERRIDES+=("optimizer.use_precision_aware_optimizer=false" "optimizer.optimizer_cuda_graph=false")
    fi
fi
if [[ "${CPU_GRAD_ACCUMULATION}" == true ]]; then
    OVERRIDES+=("ddp.fsdp_cpu_grad_accumulation=true")
fi
if [[ "${CPU_OPTIMIZER}" == true ]]; then
    OVERRIDES+=("ddp.fsdp_cpu_optimizer=true" "ddp.megatron_fsdp_use_decoupled_grad=false"
        "optimizer.decoupled_weight_decay=true")
fi
if [[ -n "${NVME_PATH}" ]]; then
    # Quote the directory for OmegaConf, including spaces and override punctuation.
    nvme_value="${NVME_PATH//\\/\\\\}"
    nvme_value="${nvme_value//\"/\\\"}"
    OVERRIDES+=("ddp.fsdp_nvme_path=\"${nvme_value}\"")
fi
[[ "${TP_COMM_OVERLAP}" == true ]] && OVERRIDES+=("comm_overlap.tp_comm_overlap=true")

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
    if [[ -n "${METRIC_FREQUENCY}" ]]; then
        COMMAND+=(--gpu-metrics-devices=all --gpu-metrics-frequency="${METRIC_FREQUENCY}")
    fi
    if [[ "${NSYS_REPORT_ONLY}" == true ]]; then
        COMMAND+=(--export=none --stats=false)
    fi
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
WORKER_ENV=()
if [[ -n "${MEGATRON_LM_DIR:-}" ]]; then
    [[ -d "${MEGATRON_LM_DIR}/megatron/core" ]] || {
        echo "MEGATRON_LM_DIR does not contain megatron/core" >&2; exit 2
    }
    MLM_SOURCE="$(cd "${MEGATRON_LM_DIR}" && pwd)"
    WORKER_ENV+=("PYTHONPATH=${MLM_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}")
fi
if [[ -n "${TORCH_THREAD}" ]]; then
    # numarun may set its own thread defaults. Apply the override inside each bound rank.
    WORKER_ENV+=("OMP_NUM_THREADS=${TORCH_THREAD}" "MKL_NUM_THREADS=${TORCH_THREAD}")
fi
if (( ${#WORKER_ENV[@]} > 0 )); then
    COMMAND+=(env "${WORKER_ENV[@]}")
fi
COMMAND+=(uv run --no-sync python "${TRAIN_ARGS[@]}")
GPU_CLOCK_MODE=$(gpu_clock_mode "${UNLOCK_GPU_CLOCKS}")

INFO=(
    "model=${MODEL}" "recipe=${RECIPE}" "fsdp=${FSDP_LEVEL}" "overlap=${OVERLAP}" "symmetric=${SYMMETRIC}" "dtype=${DTYPE}" "layers=${LAYERS_TAG}" "attention=${ATTENTION}"
    "ddp_overlap=${DDP_OVERLAP}" "separate_rs=${SEPARATE_RS}"
    "mtp=${MTP_TAG}" "experts=${EXPERTS_TAG}" "force_balance=${FORCE_BALANCE}" "cuda_graph=${CUDA_GRAPH}" "cuda_graph_trace=${GRAPH_TRACE}" "seq=${SEQ_LENGTH}"
    "tp=${TENSOR_PARALLEL}" "sp=${SEQUENCE_PARALLEL}" "cp=${CONTEXT_PARALLEL}" "dp=${DATA_PARALLEL}" "ep=${EXPERT_PARALLEL}"
    "dispatcher=${DISPATCHER}" "mbs=${MICRO_BATCH_SIZE}" "gbs=${GLOBAL_BATCH_SIZE}" "gpus=${GPU_COUNT}"
    "profile=${PROFILE}" "profile_start=${PROFILE_START}" "profile_end=${PROFILE_END}"
    "num_microbatches=${NUM_MICROBATCHES}" "numa=${NUMA_MODE}" "offload=${OFFLOAD:-none}"
    "cpu_grad_accumulation=${CPU_GRAD_ACCUMULATION}" "cpu_optimizer=${CPU_OPTIMIZER}" "nvme=$([[ -n ${NVME_PATH} ]] && echo true || echo false)"
)
[[ -n "${TORCH_THREAD}" ]] && INFO+=("torch_thread=${TORCH_THREAD}")
[[ -n "${GRAD_COMM_DTYPE}" ]] && INFO+=("fsdp_grad_comm_dtype=${GRAD_COMM_DTYPE}")
if [[ "${MULTIMEM}" == true ]]; then
    INFO+=("nccl_algo=${NCCL_ALGO}" "nccl_sym_kernel=${NCCL_SYM_KERNEL}")
fi
if [[ "${DRY_RUN}" == true ]]; then
    printf '%s\n' "${INFO[*]}" >&2
    printf 'gpu_clocks=%s\n' "${GPU_CLOCK_MODE}" >&2
    printf 'GLOO_SOCKET_IFNAME=%s\nPYTORCH_CUDA_ALLOC_CONF=%s\nNCCL_GRAPH_REGISTER=%s\nNCCL_NVLS_ENABLE=%s\nNVTE_CUTEDSL_FUSED_GROUPED_MLP=%s\ncommand:' "${GLOO_SOCKET_IFNAME:-<auto>}" "${PYTORCH_CUDA_ALLOC_CONF:-<unset>}" "${NCCL_GRAPH_REGISTER:-<unset>}" "${NCCL_NVLS_ENABLE:-<unset>}" "${NVTE_CUTEDSL_FUSED_GROUPED_MLP:-<unset>}" >&2
    printf ' %q' "${COMMAND[@]}" >&2; printf '\n' >&2
    exit 0
fi
ARCHIVE_DIR=""
if [[ -e "${RESULT_DIR}/launch.json" || ( "${STABLE_RESULT_DIR}" == true && -d "${RESULT_DIR}" ) ]]; then
    [[ "${RESUME}" == true ]] || { echo "result already exists: ${RESULT_DIR} (use --resume)" >&2; exit 2; }
    if grep -qx 'status=0' "${RESULT_DIR}/run_info.txt" 2>/dev/null && has_required_profile; then
        echo "skip completed case: ${RESULT_DIR}" >&2
        exit 0
    fi
    ARCHIVE_DIR="${RESULT_DIR}.attempt-$(date +%y%m%d-%H%M%S)-$$"
fi
# Resolve Gloo only when a new training process needs a local interface.
# shellcheck source=../../../../scripts/scale-down/common/gloo_socket_ifname.sh
source "${REPO_ROOT}/scripts/scale-down/common/gloo_socket_ifname.sh"
resolve_gloo_socket_ifname "${NNODES}" >&2
INFO+=("gloo_socket_ifname=${GLOO_SOCKET_IFNAME}")
configure_gpu_clocks "${UNLOCK_GPU_CLOCKS}"
if [[ -n "${ARCHIVE_DIR}" ]]; then
    mv "${RESULT_DIR}" "${ARCHIVE_DIR}"
    echo "archived incomplete or failed attempt: ${ARCHIVE_DIR}" >&2
fi
mkdir -p "${RESULT_DIR}"
if [[ "${PROFILE}" == pytorch ]]; then
    # The profiler writes to ../torch_profile relative to this directory.
    mkdir -p "${TENSORBOARD_DIR}"
fi
printf 'command:' >"${RESULT_DIR}/command.txt"
printf ' %q' "${COMMAND[@]}" >>"${RESULT_DIR}/command.txt"
printf '\n' >>"${RESULT_DIR}/command.txt"
cat >"${RESULT_DIR}/launch.json" <<JSON
{"torch_thread":${TORCH_THREAD:-null},"model":"${MODEL_ID}","recipe":"${RECIPE}","run_name":"${RUN_NAME}","dtype":"${DTYPE}","fsdp":${FSDP_JSON},"fsdp_cpu_offload":${OFFLOAD_JSON},"fsdp_cpu_grad_accumulation":${CPU_GRAD_ACCUMULATION},"fsdp_cpu_optimizer":${CPU_OPTIMIZER},"fsdp_nvme":$([[ -n ${NVME_PATH} ]] && echo true || echo false),"data_parallel":"${DP_TAG}","create_all_gather_group":${OVERLAP},"create_reduce_scatter_group":${SEPARATE_RS},"nccl_ub":${SYMMETRIC},"num_layers":"${LAYERS_TAG}","attention":"${ATTENTION}","linear_attention_freq":"${LINEAR_ATTENTION_FREQ}","mtp_num_layers":${MTP_TAG},"num_moe_experts":"${EXPERTS_TAG}","force_balance":${FORCE_BALANCE},"cuda_graph":"${CUDA_GRAPH}","cuda_graph_impl":"${GRAPH_IMPL}","cuda_graph_modules":"${GRAPH_MODULES}","cuda_graph_trace":"${GRAPH_TRACE}","seq_length":${SEQ_LENGTH},"tensor_model_parallel_size":${TENSOR_PARALLEL},"sequence_parallel":${SEQUENCE_PARALLEL},"context_parallel_size":${CONTEXT_PARALLEL},"data_parallel_size":${DATA_PARALLEL},"expert_model_parallel_size":${EXPERT_PARALLEL},"dispatcher":"${DISPATCHER}","micro_batch_size":${MICRO_BATCH_SIZE},"global_batch_size":${GLOBAL_BATCH_SIZE},"num_microbatches":${NUM_MICROBATCHES},"gpu_count":${GPU_COUNT},"gpu_clocks":"${GPU_CLOCK_MODE}","nnodes":${NNODES},"node_rank":${NODE_RANK},"numa_mode":"${NUMA_MODE}","pytorch_cuda_alloc_conf":"${PYTORCH_CUDA_ALLOC_CONF:-}","nccl_graph_register":"${NCCL_GRAPH_REGISTER:-}","nccl_nvls_enable":"${NCCL_NVLS_ENABLE:-}","nvte_cutedsl_fused_grouped_mlp":"${NVTE_CUTEDSL_FUSED_GROUPED_MLP:-}","run_date":"${RUN_DATE}",
"fsdp_grad_comm_dtype":${GRAD_COMM_JSON}}
JSON
printf '%s\n' "${INFO[*]}" | tee "${RESULT_DIR}/run_info.txt"
printf 'gpu_clocks=%s\n' "${GPU_CLOCK_MODE}" >>"${RESULT_DIR}/run_info.txt"
set +e
"${COMMAND[@]}" 2>&1 | tee "${RESULT_DIR}/train.log"
PIPE_STATUSES=("${PIPESTATUS[@]}")
set -e
STATUS=${PIPE_STATUSES[0]}
if (( STATUS == 0 )); then STATUS=${PIPE_STATUSES[1]}; fi
if (( STATUS == 0 )) && ! has_required_profile; then
    if [[ "${PROFILE}" == pytorch ]]; then
        echo "missing or empty PyTorch rank trace in ${RESULT_DIR}/torch_profile" >&2
    else
        echo "missing or empty nsys report: ${RESULT_DIR}/${NSYS_NAME}.nsys-rep" >&2
    fi
    STATUS=1
fi
printf 'status=%s\n' "${STATUS}" >>"${RESULT_DIR}/run_info.txt"
exit "${STATUS}"
