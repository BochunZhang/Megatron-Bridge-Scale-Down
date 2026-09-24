#!/bin/bash
# Launch ONE train.py run for the offload/recompute experiment matrix.
#
# Uses train.py arguments ONLY (not finetune_zero3.py arguments).
# Runtime knobs are grouped below; batch settings come from the DeepSpeed JSON.
#
# Usage:
#   pretrain.sh <test_name> <model> <ds_config> <recompute> <metrics_out>
#
#   test_name        self-describing test ID (<strategy>__<recompute>__mbs<N>,
#                    e.g. super_offload_0.9__recompute_act__mbs4), used for logging
#   model            HF model name or path, e.g. Qwen/Qwen3.5-9B (dense) or
#                    Qwen/Qwen3.5-35B-A3B (MoE); MoE is detected via the
#                    config's int num_experts field
#   ds_config        path to the generated DeepSpeed JSON (working copy under
#                    <repo_root>/.tmp; archived into dirname(metrics_out) before launch)
#   recompute        none | act | act_cpu  -> train.py checkpointing flags
#   metrics_out      path of the per-run metrics CSV (inside the timestamped run dir)
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Positional arguments
# ---------------------------------------------------------------------------
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <test_name> <model> <ds_config> <recompute:none|act|act_cpu> <metrics_out>" >&2
    exit 2
fi

TEST_NAME=$1
MODEL=$2
DS_CONFIG=$3
RECOMPUTE=$4
METRICS_OUT=$5

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
# NUMARUN=${NUMARUN-numarun}
# README §2: SuperOffload runs launch with --bind_cores_to_rank. Applied
# automatically when the ds_config enables super_offload; force via env var.
BIND_CORES_TO_RANK=${BIND_CORES_TO_RANK:-auto}   # auto | true | false

# train.py refuses to start without expandable_segments (ZeRO-3 + act+cpu
# offload/restore cycles fragment the default allocator).

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

export TORCH_NCCL_AVOID_RECORD_STREAMS="1"
export NVLINK_DOMAIN_SIZE="72"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_NVLS_ENABLE="0"
export NCCL_DEBUG="WARN"
export NCCL_GRAPH_REGISTER="0"

# ---------------------------------------------------------------------------
# Data knobs
# ---------------------------------------------------------------------------
DATASET_NAME=${DATASET_NAME:-wikitext}
DATASET_PERCENTAGE=${DATASET_PERCENTAGE:-1.0}
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
# Model shape: optionally shrink every model to $NUM_LAYERS layers with linear
# attention every $LINEAR_ATTENTION_FREQ-th layer (data_utils.build_model_config
# expands linear_attention_freq into layer_types; num_hidden_layers is mandatory
# when it is set).
# DISABLED by default: run with the model's native layer count / attention
# pattern. Set APPLY_MODEL_SHAPE_OVERRIDES=true to re-enable shrinking;
# pretrain_experiment.sh tags result dirs with _<N>layer only when the flag
# is true, so folder names always follow the actual overrides.
# ---------------------------------------------------------------------------
NUM_LAYERS=${NUM_LAYERS:-8}
LINEAR_ATTENTION_FREQ=${LINEAR_ATTENTION_FREQ:-4}
APPLY_MODEL_SHAPE_OVERRIDES=${APPLY_MODEL_SHAPE_OVERRIDES:-false}
if [ "$APPLY_MODEL_SHAPE_OVERRIDES" = "true" ]; then
    MODEL_SHAPE_OVERRIDE_ARGS="--override num_hidden_layers=$NUM_LAYERS --override linear_attention_freq=$LINEAR_ATTENTION_FREQ"
    MODEL_SHAPE_DESC="${NUM_LAYERS} layers (linear_attention_freq=$LINEAR_ATTENTION_FREQ)"
else
    MODEL_SHAPE_OVERRIDE_ARGS=""
    MODEL_SHAPE_DESC="native (no num_hidden_layers/linear_attention_freq overrides)"
fi

# ---------------------------------------------------------------------------
# Parallelism: MoE models run AutoEP with autoep_size=4; dense models run
# plain ZeRO-3 (--mode dense). Switch MOE_TRAIN_MODE=zero3_leaf to use the
# ZeRO-3 leaf-module path instead of AutoEP (required if a DeepSpeed version
# rejects expert_parallel + offload combinations, see README §6.4).
#
# MoE detection mirrors train.py: load the HF config (text_config backbone)
# and check for an int num_experts field. MODEL is an HF name/path, e.g.
# "Qwen/Qwen3.5-9B" (dense) or "Qwen/Qwen3.5-35B-A3B" (MoE).
# ---------------------------------------------------------------------------
AUTOEP_SIZE=${AUTOEP_SIZE:-4}
MOE_TRAIN_MODE=${MOE_TRAIN_MODE:-autoep}         # autoep | zero3_leaf
DENSE_TRAIN_MODE=${DENSE_TRAIN_MODE:-dense}      # dense | zero3_leaf
PYTHON_BIN=${PYTHON_BIN:-python}

# Batch settings have one source of truth: the archived DeepSpeed JSON.
read -r MICRO_BATCH_SIZE GRAD_ACCUM < <("$PYTHON_BIN" - "$DS_CONFIG" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as config_file:
    config = json.load(config_file)

print(config["train_micro_batch_size_per_gpu"], config["gradient_accumulation_steps"])
PY
)
GLOBAL_BATCH_SIZE=$((MICRO_BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))

IS_MOE=$("$PYTHON_BIN" -c "
from transformers import AutoConfig
config = AutoConfig.from_pretrained('$MODEL', trust_remote_code=True)
config = getattr(config, 'text_config', config)
num_experts = getattr(config, 'num_experts', None)
print('1' if isinstance(num_experts, int) and not isinstance(num_experts, bool) else '0')
")

if [ "$IS_MOE" = "1" ]; then
    TRAIN_MODE=$MOE_TRAIN_MODE
    MODE_ARGS="--autoep_size $AUTOEP_SIZE"
else
    TRAIN_MODE=$DENSE_TRAIN_MODE
    MODE_ARGS=""
fi
if [ "$TRAIN_MODE" != "autoep" ]; then
    MODE_ARGS=""
fi

# ---------------------------------------------------------------------------
# Extra HF config overrides (space-separated KEY=VALUE pairs), appended as
# --override arguments; e.g. the experiment driver sets
# EXTRA_OVERRIDES="num_experts=64" for Qwen3.5-35B-A3B. (The fixed
# num_hidden_layers / linear_attention_freq overrides are commented out above.)
# ---------------------------------------------------------------------------
EXTRA_OVERRIDES=${EXTRA_OVERRIDES:-}
EXTRA_OVERRIDE_ARGS=""
for kv in $EXTRA_OVERRIDES; do
    EXTRA_OVERRIDE_ARGS="$EXTRA_OVERRIDE_ARGS --override $kv"
done

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
# Archive the exact ds_config used by this run next to its results. The
# working copy lives under <repo_root>/.tmp (generated by
# pretrain_experiment.sh); RUN_DIR is the timestamped result folder that also
# receives run.log and metrics.csv.
RUN_DIR=$(dirname "$METRICS_OUT")
mkdir -p "$RUN_DIR"
cp "$DS_CONFIG" "${RUN_DIR}/ds_config.json"

echo "================================================"
echo "Test:            $TEST_NAME"
echo "Model:           $MODEL (mode=$TRAIN_MODE)"
echo "DeepSpeed cfg:   $DS_CONFIG (archived to ${RUN_DIR}/ds_config.json)"
echo "Run dir:         $RUN_DIR"
echo "Recompute:       $RECOMPUTE"
echo "GPUs:            $NUM_GPUS"
echo "Seq len:         $SEQ_LEN"
echo "Global batch:    $GLOBAL_BATCH_SIZE"
echo "Steps:           $STEPS (warmup=$WARMUP_STEPS)"
echo "Model shape:     $MODEL_SHAPE_DESC"
echo "Metrics out:     $METRICS_OUT"
echo "================================================"

# NOTE: no profiler arguments on purpose (--use_pytorch_profiler /
# --record_memory_history / --profile_* are intentionally omitted).
# NUMARUN prefixes deepspeed to bind the process to the correct NUMA node;
# expand to empty (NUMARUN=) to launch deepspeed directly.
CMD="deepspeed --num_gpus=$NUM_GPUS $DS_LAUNCHER_ARGS train.py \
    --deepspeed_config $DS_CONFIG \
    --model $MODEL \
    --mode $TRAIN_MODE \
    $MODE_ARGS \
    $MODEL_SHAPE_OVERRIDE_ARGS \
    $EXTRA_OVERRIDE_ARGS \
    --dataset_name $DATASET_NAME \
    --dataset_percentage $DATASET_PERCENTAGE \
    --seq_len $SEQ_LEN \
    --steps $STEPS \
    --warmup_steps $WARMUP_STEPS \
    --log_interval $LOG_INTERVAL \
    --seed $SEED \
    $TOKENIZER_ARGS \
    $INIT_WEIGHTS_ARGS \
    --metrics_out $METRICS_OUT \
    $RECOMPUTE_ARGS"

eval "$CMD"
