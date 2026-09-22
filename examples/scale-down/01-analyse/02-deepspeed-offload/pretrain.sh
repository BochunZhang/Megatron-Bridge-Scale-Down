#!/bin/bash
# Launch ONE train.py run for the offload/recompute experiment matrix.
#
# Uses train.py arguments ONLY (not finetune_zero3.py arguments).
# Every tunable is a variable (env-overridable) so the sweep can be extended
# without editing the command line below.
#
# Usage:
#   pretrain.sh <test_name> <model> <micro_batch_size> <ds_config> <recompute> <metrics_out>
#
#   test_name        experiment ID from README (B0/R1/R2/O1..O4/P1..P4/E1), used for logging only
#   model            train.py --model preset: qwen3_5 (dense) | qwen3_5_moe | mixtral | llama4
#   micro_batch_size train.py --micro_batch_size (sweep axis: 1/2/4/8)
#   ds_config        path to the generated DeepSpeed JSON (built by pretrain_experiment.sh)
#   recompute        none | act | act_cpu  -> train.py checkpointing flags
#   metrics_out      path of the per-run metrics CSV
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Positional arguments
# ---------------------------------------------------------------------------
if [ "$#" -ne 6 ]; then
    echo "Usage: $0 <test_name> <model> <micro_batch_size> <ds_config> <recompute:none|act|act_cpu> <metrics_out>" >&2
    exit 2
fi

TEST_NAME=$1
MODEL=$2
MICRO_BATCH_SIZE=$3
DS_CONFIG=$4
RECOMPUTE=$5
METRICS_OUT=$6

if [ ! -f "$DS_CONFIG" ]; then
    echo "ds_config not found: $DS_CONFIG" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Launcher / hardware knobs
# ---------------------------------------------------------------------------
NUM_GPUS=${NUM_GPUS:-4}
# NUMA binding wrapper prefixed to the deepspeed launcher (set NUMARUN= to
# disable on machines without numarun; note the dash form so an explicitly
# empty value is honored instead of falling back to the default).
NUMARUN=${NUMARUN-numarun}
# README §2: SuperOffload runs launch with --bind_cores_to_rank. Applied
# automatically when the ds_config enables super_offload; force via env var.
BIND_CORES_TO_RANK=${BIND_CORES_TO_RANK:-auto}   # auto | true | false

# train.py refuses to start without expandable_segments (ZeRO-3 + act+cpu
# offload/restore cycles fragment the default allocator).
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# README §3.1: keep the default torch pinned-memory backend; native (mlock,
# no cudaHostRegister) stalls side-stream DMA in act+cpu.
export DS_PIN_MEMORY_BACKEND=${DS_PIN_MEMORY_BACKEND:-torch}

# Repo-local HuggingFace cache. REPO_ROOT defaults to the megatron-bridge
# checkout root (4 levels up from this script); override via env if needed.
REPO_ROOT=${REPO_ROOT:-"$(cd "$SCRIPT_DIR/../../../.." && pwd)"}
HF_CACHE="${REPO_ROOT}/.cache/huggingface"
mkdir -p "$HF_CACHE"
export HF_HOME="${HF_CACHE}"
export HF_HUB_CACHE="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}"

# ---------------------------------------------------------------------------
# Data knobs
# ---------------------------------------------------------------------------
DATASET_NAME=${DATASET_NAME:-wikitext}
DATASET_PERCENTAGE=${DATASET_PERCENTAGE:-10.0}
SEQ_LEN=${SEQ_LEN:-4096}
HF_NUM_DATALOADER_WORKERS=${HF_NUM_DATALOADER_WORKERS:-0}
# Empty -> train.py falls back to the preset default tokenizer.
TOKENIZER_NAME=${TOKENIZER_NAME:-}

# ---------------------------------------------------------------------------
# Training loop knobs
# ---------------------------------------------------------------------------
STEPS=${STEPS:-10}
WARMUP_STEPS=${WARMUP_STEPS:-2}
LOG_INTERVAL=${LOG_INTERVAL:-1}
SEED=${SEED:-42}
# Empty -> train.py does not load a shared init artifact.
LOAD_INIT_WEIGHTS=${LOAD_INIT_WEIGHTS:-}

# ---------------------------------------------------------------------------
# Batch shape (README §2 + sweep requirement)
#   global batch per step  = 64 samples
#   per-GPU batch per step = 16 samples = micro_batch_size * grad_accum
#   grad_accum is derived so the global batch stays constant across the
#   micro-batch sweep {1, 2, 4, 8}.
# ---------------------------------------------------------------------------
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
PER_GPU_BATCH_SIZE=${PER_GPU_BATCH_SIZE:-16}

if [ $((PER_GPU_BATCH_SIZE * NUM_GPUS)) -ne "$GLOBAL_BATCH_SIZE" ]; then
    echo "Inconsistent batch shape: PER_GPU_BATCH_SIZE($PER_GPU_BATCH_SIZE) * NUM_GPUS($NUM_GPUS) != GLOBAL_BATCH_SIZE($GLOBAL_BATCH_SIZE)" >&2
    exit 2
fi
if [ $((PER_GPU_BATCH_SIZE % MICRO_BATCH_SIZE)) -ne 0 ]; then
    echo "PER_GPU_BATCH_SIZE($PER_GPU_BATCH_SIZE) must be divisible by MICRO_BATCH_SIZE($MICRO_BATCH_SIZE)" >&2
    exit 2
fi
GRAD_ACCUM=$((PER_GPU_BATCH_SIZE / MICRO_BATCH_SIZE))

# ---------------------------------------------------------------------------
# Model shape: shrink every model to 8 layers, linear attention every 4th
# layer (data_utils.build_model_config expands linear_attention_freq into
# layer_types; num_hidden_layers is mandatory when it is set).
# ---------------------------------------------------------------------------
NUM_LAYERS=${NUM_LAYERS:-8}
LINEAR_ATTENTION_FREQ=${LINEAR_ATTENTION_FREQ:-4}

# ---------------------------------------------------------------------------
# Parallelism: MoE presets run AutoEP with autoep_size=4; dense presets run
# plain ZeRO-3 (--mode dense). Switch MOE_TRAIN_MODE=zero3_leaf to use the
# ZeRO-3 leaf-module path instead of AutoEP (required if a DeepSpeed version
# rejects expert_parallel + offload combinations, see README §6.4).
# ---------------------------------------------------------------------------
AUTOEP_SIZE=${AUTOEP_SIZE:-4}
MOE_TRAIN_MODE=${MOE_TRAIN_MODE:-autoep}         # autoep | zero3_leaf
DENSE_TRAIN_MODE=${DENSE_TRAIN_MODE:-dense}      # dense | zero3_leaf

case "$MODEL" in
    qwen3_5_moe|mixtral|llama4)
        TRAIN_MODE=$MOE_TRAIN_MODE
        MODE_ARGS="--autoep_size $AUTOEP_SIZE"
        ;;
    qwen3_5)
        TRAIN_MODE=$DENSE_TRAIN_MODE
        MODE_ARGS=""
        ;;
    *)
        echo "Unknown model preset: $MODEL (expected one of: qwen3_5 qwen3_5_moe mixtral llama4)" >&2
        exit 2
        ;;
esac
if [ "$TRAIN_MODE" != "autoep" ]; then
    MODE_ARGS=""
fi

# ---------------------------------------------------------------------------
# Recompute axis (README §3.1): none | act | act+cpu
# ---------------------------------------------------------------------------
case "$RECOMPUTE" in
    none)    RECOMPUTE_ARGS="" ;;
    act)     RECOMPUTE_ARGS="--activation_checkpointing" ;;
    act_cpu) RECOMPUTE_ARGS="--activation_checkpointing --cpu_checkpointing" ;;
    *)
        echo "Unknown recompute mode: $RECOMPUTE (expected none|act|act_cpu)" >&2
        exit 2
        ;;
esac

# ---------------------------------------------------------------------------
# Optional arguments (only appended when the corresponding variable is set)
# ---------------------------------------------------------------------------
TOKENIZER_ARGS=""
if [ -n "$TOKENIZER_NAME" ]; then
    TOKENIZER_ARGS="--tokenizer_name $TOKENIZER_NAME"
fi

INIT_WEIGHTS_ARGS=""
if [ -n "$LOAD_INIT_WEIGHTS" ]; then
    INIT_WEIGHTS_ARGS="--load_init_weights $LOAD_INIT_WEIGHTS"
fi

# ---------------------------------------------------------------------------
# deepspeed launcher flags
# ---------------------------------------------------------------------------
DS_LAUNCHER_ARGS=""
bind_cores="$BIND_CORES_TO_RANK"
if [ "$bind_cores" = "auto" ]; then
    if grep -q '"super_offload"' "$DS_CONFIG"; then
        bind_cores="true"
    else
        bind_cores="false"
    fi
fi
if [ "$bind_cores" = "true" ]; then
    DS_LAUNCHER_ARGS="--bind_cores_to_rank"
fi

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
echo "================================================"
echo "Test:            $TEST_NAME"
echo "Model:           $MODEL (mode=$TRAIN_MODE)"
echo "DeepSpeed cfg:   $DS_CONFIG"
echo "Recompute:       $RECOMPUTE"
echo "GPUs:            $NUM_GPUS"
echo "Seq len:         $SEQ_LEN"
echo "Micro batch:     $MICRO_BATCH_SIZE  grad_accum: $GRAD_ACCUM  global batch: $GLOBAL_BATCH_SIZE"
echo "Steps:           $STEPS (warmup=$WARMUP_STEPS)"
echo "Layers:          $NUM_LAYERS (linear_attention_freq=$LINEAR_ATTENTION_FREQ)"
echo "Metrics out:     $METRICS_OUT"
echo "================================================"

# NOTE: no profiler arguments on purpose (--use_pytorch_profiler /
# --record_memory_history / --profile_* are intentionally omitted).
# NUMARUN prefixes deepspeed to bind the process to the correct NUMA node;
# expand to empty (NUMARUN=) to launch deepspeed directly.
CMD="$NUMARUN deepspeed --num_gpus=$NUM_GPUS $DS_LAUNCHER_ARGS train.py \
    --deepspeed_config $DS_CONFIG \
    --model $MODEL \
    --mode $TRAIN_MODE \
    $MODE_ARGS \
    --override num_hidden_layers=$NUM_LAYERS \
    --override linear_attention_freq=$LINEAR_ATTENTION_FREQ \
    --dataset_name $DATASET_NAME \
    --dataset_percentage $DATASET_PERCENTAGE \
    --seq_len $SEQ_LEN \
    --micro_batch_size $MICRO_BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --steps $STEPS \
    --warmup_steps $WARMUP_STEPS \
    --log_interval $LOG_INTERVAL \
    --seed $SEED \
    $TOKENIZER_ARGS \
    $INIT_WEIGHTS_ARGS \
    --metrics_out $METRICS_OUT \
    $RECOMPUTE_ARGS"

eval "$CMD"
