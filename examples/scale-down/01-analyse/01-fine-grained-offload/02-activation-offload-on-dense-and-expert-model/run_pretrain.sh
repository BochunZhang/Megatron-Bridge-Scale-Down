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
# Run one selective activation-offload training configuration and record every
# artifact needed to explain its step time: the resolve
# config, the launch command, per-step GPU utilization and a GPU memory trace.
#
# Every run is a mock-data pretraining step. Only the offload settings, the
# HybridEP synchronization mode and the parallelism layout are varied.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

usage() {
    cat <<'EOF'
Usage: run_pretrain.sh [OPTIONS]

Trains one configuration and writes results under
<results-root>/<experiment>/<model>-<model-type>/<test-name>/<run-time>/.

Required:
    --model <id>                 Megatron model id, e.g. qwen35_text_35b_a3b
                                 or deepseek_v3.
    --experiment <name>          Comparison this run belongs to; first directory
                                 level under --results-root, e.g.
                                 4gpu_expert_activation_offload_blocking.
    --model-type <dense|expert>  Which side of the dense/expert comparison this
                                 run belongs to; part of the result path.
    --run-name <name>            Run name, must start with "baseline" or
                                 "offload" so the analyser can pair cases.
    --offload <true|false>       Enable fine-grained activation offloading.

Offload options:
    --offload-modules <list>     Hydra list such as [mlp_norm,moe_act]. Required
                                 when --offload is true.
    --dtype <bf16|mxfp8>         Training precision (default: bf16).

Scale and parallelism:
    --gpu <4|16|32>              Total GPUs. GB200 uses four per node, so 16 and
                                 32 mean four and eight nodes (default: 4).
                                 Multi-node runs read MASTER_ADDR, MASTER_PORT,
                                 WORLD_SIZE (nodes) and RANK (node rank) from
                                 the DLC environment.
    --master-port <port>         Rendezvous port for this run (default:
                                 MASTER_PORT + 1; MASTER_PORT itself serves the
                                 batch-id exchange).
    --pp <n>                     Pipeline parallel size (default: 1).
    --ep <n>                     Expert parallel size (default: 1).
    --dispatcher <name>          default | alltoall | hybridep.
    --hybridep-num-sms <n>       Communication SMs for HybridEP (default: 32).
    --sync-with-sm <true|false>  Read the HybridEP padded token count with an SM
                                 copy and a host spin-wait instead of .item()
                                 (default: false). Only valid with hybridep.
    --sharding <fsdp1|none>      fsdp1 keeps the recipe's Megatron-FSDP optimizer
                                 sharding; none disables FSDP for a plain
                                 PP+EP+DP layout (default: fsdp1).

Model overrides (scale-down proxies):
    --num-layers <n>             Override model.num_layers.
    --num-experts <n>            Override model.num_moe_experts.
    --moe-layer-freq <list>      Override model.moe_layer_freq.
    --linear-attention-freq <n>  Override model.linear_attention_freq.
    --mtp-num-layers <n>         Override model.mtp_num_layers.

Training shape:
    --mbs <n>                    Micro batch size (default: 1).
    --gbs <n>                    Global batch size (default: 32).
    --seq-length <n>             Sequence length (default: 4096).
    --train-iters <n>            Iterations (default: 10).
    --warmup-steps <n>           LR warmup iterations (default: 0).

Profiling and output:
    --profile <none|nsys|torch>  Profiling backend (default: none).
    --profile-step-start <n>     First profiled step (default: 7).
    --profile-step-end <n>       Last profiled step (default: 8).
    --results-root <path>        Result tree root.
    --run-time <id>              Batch id. When omitted on a multi-node run, node 0
                                 generates it and shares it with every node so all
                                 nodes write under the master's result directory.
    --dry-run                    Print the resolved run without launching.
    -h, --help                   Show this help.
EOF
}

MODEL=""
MODEL_TYPE=""
EXPERIMENT=""
DTYPE="bf16"
RUN_NAME=""
OFFLOAD=""
OFFLOAD_MODULES=""
GPU_COUNT=4
PP=1
EP=1
DISPATCHER="default"
HYBRIDEP_NUM_SMS=32
SYNC_WITH_SM=false
SHARDING="fsdp1"
NUM_LAYERS=""
NUM_EXPERTS=""
MOE_LAYER_FREQ=""
LINEAR_ATTENTION_FREQ=""
MTP_NUM_LAYERS=""
MBS=1
GBS=32
SEQ_LENGTH=4096
TRAIN_ITERS=10
WARMUP_STEPS=0
PROFILE="none"
PROFILE_STEP_START=7
PROFILE_STEP_END=8
RESULTS_ROOT=""
RUN_TIME=""
RUN_PORT=""
DRY_RUN=false
GPU_MEMORY_TRACE_INTERVAL="${GPU_MEMORY_TRACE_INTERVAL:-1.0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --model-type) MODEL_TYPE="$2"; shift 2 ;;
        --experiment) EXPERIMENT="$2"; shift 2 ;;
        --master-port) RUN_PORT="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --run-name) RUN_NAME="$2"; shift 2 ;;
        --offload) OFFLOAD="$2"; shift 2 ;;
        --offload-modules) OFFLOAD_MODULES="$2"; shift 2 ;;
        --gpu) GPU_COUNT="$2"; shift 2 ;;
        --pp) PP="$2"; shift 2 ;;
        --ep) EP="$2"; shift 2 ;;
        --dispatcher) DISPATCHER="$2"; shift 2 ;;
        --hybridep-num-sms) HYBRIDEP_NUM_SMS="$2"; shift 2 ;;
        --sync-with-sm) SYNC_WITH_SM="$2"; shift 2 ;;
        --sharding) SHARDING="$2"; shift 2 ;;
        --num-layers) NUM_LAYERS="$2"; shift 2 ;;
        --num-experts) NUM_EXPERTS="$2"; shift 2 ;;
        --moe-layer-freq) MOE_LAYER_FREQ="$2"; shift 2 ;;
        --linear-attention-freq) LINEAR_ATTENTION_FREQ="$2"; shift 2 ;;
        --mtp-num-layers) MTP_NUM_LAYERS="$2"; shift 2 ;;
        --mbs|--micro-batch-size) MBS="$2"; shift 2 ;;
        --gbs|--global-batch-size) GBS="$2"; shift 2 ;;
        --seq-length) SEQ_LENGTH="$2"; shift 2 ;;
        --train-iters) TRAIN_ITERS="$2"; shift 2 ;;
        --warmup-steps) WARMUP_STEPS="$2"; shift 2 ;;
        --profile) PROFILE="$2"; shift 2 ;;
        --profile-step-start) PROFILE_STEP_START="$2"; shift 2 ;;
        --profile-step-end) PROFILE_STEP_END="$2"; shift 2 ;;
        --results-root) RESULTS_ROOT="$2"; shift 2 ;;
        --run-time) RUN_TIME="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

[[ -n "${MODEL}" ]] || die "--model is required"
[[ "${EXPERIMENT}" =~ ^[A-Za-z0-9._-]+$ ]] || die "--experiment is required and must be a single path segment: ${EXPERIMENT}"
case "${MODEL_TYPE}" in dense|expert) ;; *) die "--model-type must be dense or expert" ;; esac
[[ -n "${RUN_NAME}" ]] || die "--run-name is required"
[[ -n "${OFFLOAD}" ]] || die "--offload is required"
case "${DTYPE}" in bf16|mxfp8) ;; *) die "--dtype must be bf16 or mxfp8" ;; esac
case "${OFFLOAD}" in true|false) ;; *) die "--offload must be true or false" ;; esac
case "${SYNC_WITH_SM}" in true|false) ;; *) die "--sync-with-sm must be true or false" ;; esac
case "${SHARDING}" in fsdp1|none) ;; *) die "--sharding must be fsdp1 or none" ;; esac
case "${PROFILE}" in none|nsys|torch) ;; *) die "--profile must be none, nsys or torch" ;; esac
case "${GPU_COUNT}" in 4|16|32) ;; *) die "--gpu must be 4, 16 or 32" ;; esac
case "${DISPATCHER}" in default|alltoall|hybridep) ;; *) die "--dispatcher must be default, alltoall or hybridep" ;; esac
[[ "${SYNC_WITH_SM}" == false || "${DISPATCHER}" == hybridep ]] \
    || die "--sync-with-sm true requires --dispatcher hybridep"
[[ "${RUN_NAME}" == baseline* || "${RUN_NAME}" == offload* ]] \
    || die "--run-name must start with baseline or offload: ${RUN_NAME}"
if [[ "${OFFLOAD}" == true ]]; then
    [[ -n "${OFFLOAD_MODULES}" ]] || die "--offload-modules is required when --offload is true"
else
    # Keep the case pair symmetric: the baseline must not carry an offload list.
    OFFLOAD_MODULES="null"
fi
for name in MBS GBS SEQ_LENGTH TRAIN_ITERS PP EP GPU_COUNT; do
    [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer: ${!name}"
done
[[ "${WARMUP_STEPS}" =~ ^[0-9]+$ ]] || die "--warmup-steps must be a non-negative integer"
(( TRAIN_ITERS >= 1 )) || die "--train-iters must be at least 1"
(( GPU_COUNT % PP == 0 )) || die "--gpu ${GPU_COUNT} must be divisible by --pp ${PP}"
# TP=CP=1, so DP = GPUs / PP. Expert parallelism folds into the DP group.
DP=$(( GPU_COUNT / PP ))
(( DP % EP == 0 )) || die "--ep ${EP} must divide DP ${DP} (= gpu / pp)"
if [[ "${MODEL_TYPE}" == dense ]]; then
    (( EP == 1 )) || die "dense models have no experts; use --ep 1"
fi
(( GBS % (MBS * DP) == 0 )) \
    || die "GBS ${GBS} must be divisible by MBS*DP = ${MBS}*${DP} = $((MBS * DP))"

if [[ "${PROFILE}" != none ]]; then
    (( PROFILE_STEP_END > PROFILE_STEP_START )) || die "--profile-step-end must exceed --profile-step-start"
    (( PROFILE_STEP_END <= TRAIN_ITERS )) || die "--profile-step-end must be <= --train-iters"
fi

case "${DTYPE}" in
    bf16) RECIPE_SUFFIX=bf16 ;;
    mxfp8) RECIPE_SUFFIX=fp8mx ;;
esac
RECIPE="${MODEL}_pretrain_4gpu_gb200_${RECIPE_SUFFIX}_fsdp1_config"
RESULTS_ROOT="${RESULTS_ROOT:-${DEFAULT_RESULTS_ROOT}}"
resolve_dist_env "${GPU_COUNT}" "${DRY_RUN}"
sync_run_time "${DRY_RUN}"
RUN_PORT="${RUN_PORT:-$(( MASTER_PORT + 1 ))}"

# Result paths are shared by every node in a DLC job; the run name is part of the
# test name so baseline and offload never overwrite each other.
TEST_NAME="gpu_${GPU_COUNT}-${DTYPE}-pp_${PP}-ep_${EP}-mbs_${MBS}-gbs_${GBS}-${DISPATCHER}-${RUN_NAME}"
MODEL_DIR_NAME="${MODEL}-${MODEL_TYPE}"
RESULT_DIR="${RESULTS_ROOT}/${EXPERIMENT}/${MODEL_DIR_NAME}/${TEST_NAME}/${RUN_TIME}"
# Non-zero nodes keep node-local logs and traces in a subdirectory so the shared
# result directory is not clobbered by concurrent writers.
RUN_DIR="${RESULT_DIR}"
if (( NNODES > 1 && NODE_RANK != 0 )); then
    RUN_DIR="${RESULT_DIR}/rank${NODE_RANK}"
fi

case "${PROFILE}" in
    nsys)
        USE_NSYS_PROFILER=true
        USE_PYTORCH_PROFILER=false
        RECORD_MEMORY_HISTORY=true
        RECORD_SHAPES=true
        NVTX_RANGES=true
        TENSORBOARD_DIR=null
        ;;
    torch)
        USE_NSYS_PROFILER=false
        USE_PYTORCH_PROFILER=true
        RECORD_MEMORY_HISTORY=true
        RECORD_SHAPES=true
        NVTX_RANGES=true
        TENSORBOARD_DIR="${RUN_DIR}/profile/tensorboard"
        ;;
    *)
        USE_NSYS_PROFILER=false
        USE_PYTORCH_PROFILER=false
        RECORD_MEMORY_HISTORY=false
        RECORD_SHAPES=false
        NVTX_RANGES=false
        TENSORBOARD_DIR=null
        ;;
esac

if [[ "${SHARDING}" == fsdp1 ]]; then
    FSDP_OVERRIDES=(
        "ddp.use_megatron_fsdp=true"
        "dist.use_megatron_fsdp=true"
        "ddp.data_parallel_sharding_strategy=optim"
    )
else
    FSDP_OVERRIDES=(
        "ddp.use_megatron_fsdp=false"
        "dist.use_megatron_fsdp=false"
        "ddp.use_distributed_optimizer=true"
        "optimizer.use_distributed_optimizer=true"
    )
fi

# Kept identical between the baseline and the offload case so the only moving
# parts are the offload flags, the HybridEP sync mode and the parallelism.
OVERRIDES=(
    "model.seq_length=${SEQ_LENGTH}"
    "dataset.seq_length=${SEQ_LENGTH}"
    "dataset.blend=null"
    "train.train_iters=${TRAIN_ITERS}"
    "train.global_batch_size=${GBS}"
    "train.micro_batch_size=${MBS}"
    "scheduler.lr_warmup_iters=${WARMUP_STEPS}"
    "validation.eval_iters=0"
    "validation.eval_interval=0"
    "model.tensor_model_parallel_size=1"
    "model.pipeline_model_parallel_size=${PP}"
    "model.expert_model_parallel_size=${EP}"
    "model.virtual_pipeline_model_parallel_size=null"
    "model.context_parallel_size=1"
    "model.recompute_granularity=null"
    "model.recompute_method=null"
    "model.recompute_num_layers=null"
    "model.recompute_modules=null"
    "model.cpu_offloading=false"
    "model.fine_grained_activation_offloading=${OFFLOAD}"
    "model.offload_modules=${OFFLOAD_MODULES}"
    "model.cuda_graph_impl=none"
    "model.cuda_graph_scope=null"
    "model.cuda_graph_modules=[]"
    "model.moe_shared_expert_overlap=false"
    "model.moe_permute_fusion_into_hybridep=true"
    "model.moe_hybridep_sync_with_sm=${SYNC_WITH_SM}"
    "optimizer.optimizer_cpu_offload=false"
    "optimizer.optimizer_offload_fraction=0.0"
    "optimizer.overlap_cpu_optimizer_d2h_h2d=false"
    "checkpoint.save=null"
    "checkpoint.load=null"
    "logger.log_interval=1"
    "logger.tensorboard_dir=${TENSORBOARD_DIR}"
    "logger.save_config_filepath=${RUN_DIR}/config.yaml"
    "profiling.use_pytorch_profiler=${USE_PYTORCH_PROFILER}"
    "profiling.use_nsys_profiler=${USE_NSYS_PROFILER}"
    "profiling.profile_step_start=${PROFILE_STEP_START}"
    "profiling.profile_step_end=${PROFILE_STEP_END}"
    "profiling.profile_ranks=[0,1,2,3]"
    "profiling.record_memory_history=${RECORD_MEMORY_HISTORY}"
    "profiling.record_shapes=${RECORD_SHAPES}"
    "profiling.memory_snapshot_path=${RUN_DIR}/memory/snapshot.pickle"
    "profiling.nvtx_ranges=${NVTX_RANGES}"
    "${FSDP_OVERRIDES[@]}"
)
if [[ -n "${NUM_LAYERS}" ]]; then OVERRIDES+=("model.num_layers=${NUM_LAYERS}"); fi
if [[ -n "${NUM_EXPERTS}" ]]; then OVERRIDES+=("model.num_moe_experts=${NUM_EXPERTS}"); fi
if [[ -n "${MOE_LAYER_FREQ}" ]]; then OVERRIDES+=("model.moe_layer_freq=${MOE_LAYER_FREQ}"); fi
if [[ -n "${LINEAR_ATTENTION_FREQ}" ]]; then OVERRIDES+=("model.linear_attention_freq=${LINEAR_ATTENTION_FREQ}"); fi
if [[ -n "${MTP_NUM_LAYERS}" ]]; then OVERRIDES+=("model.mtp_num_layers=${MTP_NUM_LAYERS}"); fi
case "${DISPATCHER}" in
    alltoall)
        OVERRIDES+=(
            "model.moe_token_dispatcher_type=alltoall"
            "model.moe_flex_dispatcher_backend=null"
            "model.moe_flex_dispatcher_num_sms=null"
        )
        ;;
    hybridep)
        OVERRIDES+=(
            "model.moe_token_dispatcher_type=flex"
            "model.moe_flex_dispatcher_backend=hybridep"
            "model.moe_flex_dispatcher_num_sms=${HYBRIDEP_NUM_SMS}"
        )
        ;;
esac

TRAIN_ARGS=(
    scripts/training/run_recipe.py
    --recipe "${RECIPE}"
    --step-func llm_step
    "${OVERRIDES[@]}"
)

if (( NNODES == 1 )); then
    COMMAND=(
        uv run --no-sync python -m torch.distributed.run
        --standalone
        --nproc_per_node="${GPUS_PER_NODE}"
        --master_port="${RUN_PORT}"
        --log_dir="${RUN_DIR}/rank_logs"
        --redirects=3 --tee=3
        "${TRAIN_ARGS[@]}"
    )
else
    DISTRIBUTED_ARGS=(
        --nproc_per_node="${GPUS_PER_NODE}"
        --nnodes="${NNODES}"
        --node_rank="${NODE_RANK}"
        --master_addr="${MASTER_ADDR}"
        --master_port="${RUN_PORT}"
        --log_dir="${RUN_DIR}/rank_logs"
        --redirects=3 --tee=3
    )
    COMMAND=(uv run --no-sync python -m torch.distributed.run "${DISTRIBUTED_ARGS[@]}" "${TRAIN_ARGS[@]}")
fi
if [[ "${PROFILE}" == nsys ]]; then
    NSYS_NAME="${EXPERIMENT}/${MODEL_DIR_NAME}/${TEST_NAME}/${RUN_TIME}"
    NSYS_NAME="${NSYS_NAME//\//-}"
    COMMAND=(
        nsys profile -s none -t cuda,nvtx
        --capture-range=cudaProfilerApi --capture-range-end=stop
        -o "${RUN_DIR}/profile/nsys-${NSYS_NAME}_%p_%h"
        --force-overwrite=true
        "${COMMAND[@]}"
    )
fi

log "resolved experiment=${EXPERIMENT} model=${MODEL} model_type=${MODEL_TYPE} dtype=${DTYPE} recipe=${RECIPE}"
log "resolved gpu=${GPU_COUNT} nnodes=${NNODES} node_rank=${NODE_RANK} master=${MASTER_ADDR:-unset}:${RUN_PORT} pp=${PP} ep=${EP} dp=${DP}"
log "resolved mbs=${MBS} gbs=${GBS} seq=${SEQ_LENGTH} iters=${TRAIN_ITERS} dispatcher=${DISPATCHER}"
log "resolved offload=${OFFLOAD} offload_modules=${OFFLOAD_MODULES} sync_with_sm=${SYNC_WITH_SM}"
log "resolved sharding=${SHARDING} profile=${PROFILE} result=${RESULT_DIR}"

if [[ "${DRY_RUN}" == true ]]; then
    log "command=${COMMAND[*]}"
    exit 0
fi

mkdir -p "${RUN_DIR}/memory" "${RUN_DIR}/profile" "${RUN_DIR}/gpu_memory" "${RUN_DIR}/rank_logs"
export HF_HOME="${REPO_ROOT}/.cache/huggingface"
export UV_CACHE_DIR="${REPO_ROOT}/.cache/uv"
export NEMO_HOME="${REPO_ROOT}/.cache/nemo"
export NVTE_CPU_OFFLOAD_V1="1"
export TORCH_NCCL_AVOID_RECORD_STREAMS="1"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG="WARN"
export NCCL_GRAPH_REGISTER="0"
export TOKENIZERS_PARALLELISM="false"
# HybridEP detects how many EP ranks share one NVLink domain. Leave
# NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN unset unless the detection is wrong;
# forcing 4 would route EP=8/16 traffic inside one NVL72 rack through RDMA.

COMMAND_TEXT="${COMMAND[*]}"
export COMMAND_TEXT
if (( NODE_RANK == 0 )); then
    printf '%s\n' "${COMMAND_TEXT}" > "${RUN_DIR}/command.txt"
    RUN_DIR="${RUN_DIR}" EXPERIMENT="${EXPERIMENT}" MODEL="${MODEL}" MODEL_TYPE="${MODEL_TYPE}" DTYPE="${DTYPE}" RECIPE="${RECIPE}" \
    TEST_NAME="${TEST_NAME}" RUN_NAME="${RUN_NAME}" RUN_TIME="${RUN_TIME}" \
    GPU_COUNT="${GPU_COUNT}" NNODES="${NNODES}" GPUS_PER_NODE="${GPUS_PER_NODE}" \
    PP="${PP}" EP="${EP}" DP="${DP}" MBS="${MBS}" GBS="${GBS}" SEQ_LENGTH="${SEQ_LENGTH}" \
    TRAIN_ITERS="${TRAIN_ITERS}" WARMUP_STEPS="${WARMUP_STEPS}" \
    OFFLOAD="${OFFLOAD}" OFFLOAD_MODULES="${OFFLOAD_MODULES}" \
    DISPATCHER="${DISPATCHER}" HYBRIDEP_NUM_SMS="${HYBRIDEP_NUM_SMS}" \
    SYNC_WITH_SM="${SYNC_WITH_SM}" SHARDING="${SHARDING}" PROFILE="${PROFILE}" \
    RESULTS_ROOT="${RESULTS_ROOT}" \
    uv run --no-sync python -c '
import json, os
root = os.environ["RUN_DIR"]
config = {
    "experiment": os.environ["EXPERIMENT"],
    "model": os.environ["MODEL"],
    "model_type": os.environ["MODEL_TYPE"],
    "dtype": os.environ["DTYPE"],
    "recipe": os.environ["RECIPE"],
    "run_name": os.environ["RUN_NAME"],
    "test_name": os.environ["TEST_NAME"],
    "run_time": os.environ["RUN_TIME"],
    "gpu_count": int(os.environ["GPU_COUNT"]),
    "nnodes": int(os.environ["NNODES"]),
    "gpus_per_node": int(os.environ["GPUS_PER_NODE"]),
    "pp": int(os.environ["PP"]),
    "ep": int(os.environ["EP"]),
    "dp": int(os.environ["DP"]),
    "mbs": int(os.environ["MBS"]),
    "gbs": int(os.environ["GBS"]),
    "seq_length": int(os.environ["SEQ_LENGTH"]),
    "train_iters": int(os.environ["TRAIN_ITERS"]),
    "warmup_steps": int(os.environ["WARMUP_STEPS"]),
    "offload": os.environ["OFFLOAD"] == "true",
    "offload_modules": os.environ["OFFLOAD_MODULES"],
    "dispatcher": os.environ["DISPATCHER"],
    "hybridep_num_sms": int(os.environ["HYBRIDEP_NUM_SMS"]),
    "hybridep_sync_with_sm": os.environ["SYNC_WITH_SM"] == "true",
    "sharding": os.environ["SHARDING"],
    "profile": os.environ["PROFILE"],
    "results_root": os.environ["RESULTS_ROOT"],
    "result_dir": root,
}
with open(os.path.join(root, "config.json"), "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2, sort_keys=True)
'
fi

mkdir -p "${RUN_DIR}"
GPU_MONITOR_PID=""
GPU_MONITOR_READY_FILE="${RUN_DIR}/.gpu-monitor-ready"
GPU_MONITOR_ERROR=""
start_gpu_monitor "${RUN_DIR}/gpu_memory" "${GPU_MONITOR_READY_FILE}" "${GPU_MEMORY_TRACE_INTERVAL}" \
    || log "warning: ${GPU_MONITOR_ERROR}"
trap 'stop_gpu_monitor || true' EXIT

set +e
"${COMMAND[@]}" 2>&1 | tee "${RUN_DIR}/train.log"
RUN_STATUS=${PIPESTATUS[0]}
set -e

stop_gpu_monitor || true
trap - EXIT

if [[ -f "${RUN_DIR}/train.log" ]]; then
    export_gpu_utilization "${RUN_DIR}/train.log" "${RUN_DIR}/gpu_utilization.json" \
        || log "warning: could not export GPU utilization from ${RUN_DIR}/train.log"
fi

if (( NODE_RANK == 0 )); then
    RUN_DIR="${RUN_DIR}" RUN_STATUS="${RUN_STATUS}" TEST_NAME="${TEST_NAME}" \
    RUN_NAME="${RUN_NAME}" RUN_TIME="${RUN_TIME}" \
    uv run --no-sync python -c '
import json, os
root = os.environ["RUN_DIR"]
summary = {
    "status": int(os.environ["RUN_STATUS"]),
    "test_name": os.environ["TEST_NAME"],
    "run_name": os.environ["RUN_NAME"],
    "run_time": os.environ["RUN_TIME"],
    "result_dir": root,
}
with open(os.path.join(root, "summary.json"), "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2, sort_keys=True)
raise SystemExit(summary["status"])
'
else
    exit "${RUN_STATUS}"
fi
