#!/bin/bash
# Launch ONE train.py run for the offload/recompute experiment matrix.
#
# Uses train.py arguments ONLY (not finetune_zero3.py arguments).
# Runtime knobs are grouped below; batch settings come from the DeepSpeed JSON.
#
# Usage:
#   pretrain.sh --test_name <name> --model <model> --ds_config <path> \
#       --recompute <none|act|act_cpu> --metrics_out <path> \
#       --mode <autoep|zero3_leaf|dense> [--autoep_size <N>]
#
#   test_name        self-describing test ID (<strategy>__<recompute>__mbs<N>,
#                    e.g. super_offload_0.9__recompute_act__mbs4), used for logging
#   model            HF model name or path, e.g. Qwen/Qwen3.5-9B (dense) or
#                    Qwen/Qwen3.5-35B-A3B (MoE)
#   ds_config        path to the generated DeepSpeed JSON (working copy under
#                    <repo_root>/.tmp; archived into dirname(metrics_out) before launch)
#   recompute        none | act | act_cpu  -> train.py checkpointing flags
#   metrics_out      path of the per-run metrics CSV (inside the timestamped run dir)
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Named arguments
# ---------------------------------------------------------------------------
TEST_NAME=""
MODEL=""
DS_CONFIG=""
RECOMPUTE=""
METRICS_OUT=""
TRAIN_MODE=""
AUTOEP_SIZE=""
EXTRA_OVERRIDES=""

usage() {
    sed -n '2,20p' "${SCRIPT_DIR}/$(basename "$0")"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --test_name|--model|--ds_config|--recompute|--metrics_out|--mode|--autoep_size|\
        --num_gpus|--num_layers|--per_gpu_batch_size|--linear_attention_freq|\
        --apply_model_shape_overrides|--tokenizer_name|--load_init_weights|\
        --dataset_name|--dataset_percentage|--seq_len|--steps|--warmup_steps|\
        --log_interval|--seed)
            if [ "$#" -lt 2 ]; then
                echo "Missing value for $1" >&2
                exit 2
            fi
            case "$1" in
                --test_name) TEST_NAME=$2 ;;
                --model) MODEL=$2 ;;
                --ds_config) DS_CONFIG=$2 ;;
                --recompute) RECOMPUTE=$2 ;;
                --metrics_out) METRICS_OUT=$2 ;;
                --mode) TRAIN_MODE=$2 ;;
                --autoep_size) AUTOEP_SIZE=$2 ;;
                --num_gpus) NUM_GPUS=$2 ;;
                --num_layers) NUM_LAYERS=$2 ;;
                --per_gpu_batch_size) PER_GPU_BATCH_SIZE=$2 ;;
                --linear_attention_freq) LINEAR_ATTENTION_FREQ=$2 ;;
                --apply_model_shape_overrides) APPLY_MODEL_SHAPE_OVERRIDES=$2 ;;
                --tokenizer_name) TOKENIZER_NAME=$2 ;;
                --load_init_weights) LOAD_INIT_WEIGHTS=$2 ;;
                --dataset_name) DATASET_NAME=$2 ;;
                --dataset_percentage) DATASET_PERCENTAGE=$2 ;;
                --seq_len) SEQ_LEN=$2 ;;
                --steps) STEPS=$2 ;;
                --warmup_steps) WARMUP_STEPS=$2 ;;
                --log_interval) LOG_INTERVAL=$2 ;;
                --seed) SEED=$2 ;;
            esac
            shift 2
            ;;
        --override)
            if [ "$#" -lt 2 ]; then
                echo "Missing value for --override" >&2
                exit 2
            fi
            EXTRA_OVERRIDES="$EXTRA_OVERRIDES $2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [ -z "$TEST_NAME" ] || [ -z "$MODEL" ] || [ -z "$DS_CONFIG" ] || [ -z "$RECOMPUTE" ] || \
    [ -z "$METRICS_OUT" ] || [ -z "$TRAIN_MODE" ]; then
    echo "Missing required argument; see --help" >&2
    exit 2
fi

case "$TRAIN_MODE" in
    autoep|zero3_leaf|dense) ;;
    *)
        echo "Unknown --mode: $TRAIN_MODE (expected autoep, zero3_leaf, or dense)" >&2
        exit 2
        ;;
esac

if [ "$TRAIN_MODE" = "autoep" ] && [ -z "$AUTOEP_SIZE" ]; then
    echo "--autoep_size is required when --mode autoep" >&2
    exit 2
fi

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
    MODEL_SHAPE_DESC="${NUM_LAYERS} layers (linear_attention_freq=$LINEAR_ATTENTION_FREQ)"
else
    MODEL_SHAPE_DESC="native (no num_hidden_layers/linear_attention_freq overrides)"
fi

# ---------------------------------------------------------------------------
# Parallelism is selected by the caller. In particular, the experiment driver
# identifies MoE models and passes --mode autoep --autoep_size <N> explicitly.
# This keeps this launcher independent of Hugging Face config loading.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Extra HF config overrides are supplied directly by the caller as repeated
# --override KEY=VALUE options; e.g. --override num_experts=64 for
# Qwen3.5-35B-A3B.
# ---------------------------------------------------------------------------
# Recompute axis (README §3.1): none | act | act+cpu
# ---------------------------------------------------------------------------
case "$RECOMPUTE" in
    none)    ;;
    act)     ;;
    act_cpu) ;;
    *)
        echo "Unknown recompute mode: $RECOMPUTE (expected none|act|act_cpu)" >&2
        exit 2
        ;;
esac

# ---------------------------------------------------------------------------
# Optional arguments (only appended when the corresponding variable is set)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# deepspeed launcher flags
# ---------------------------------------------------------------------------
bind_cores="$BIND_CORES_TO_RANK"
if [ "$bind_cores" = "auto" ]; then
    if grep -q '"super_offload"' "$DS_CONFIG"; then
        bind_cores="true"
    else
        bind_cores="false"
    fi
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
CMD=(deepspeed "--num_gpus=$NUM_GPUS")
if [ "$bind_cores" = "true" ]; then
    CMD+=(--bind_cores_to_rank)
fi
CMD+=(train.py
    --deepspeed_config "$DS_CONFIG"
    --model "$MODEL"
    --mode "$TRAIN_MODE"
    --dataset_name "$DATASET_NAME"
    --dataset_percentage "$DATASET_PERCENTAGE"
    --seq_len "$SEQ_LEN"
    --steps "$STEPS"
    --warmup_steps "$WARMUP_STEPS"
    --log_interval "$LOG_INTERVAL"
    --seed "$SEED"
    --metrics_out "$METRICS_OUT")
if [ "$TRAIN_MODE" = "autoep" ]; then
    CMD+=(--autoep_size "$AUTOEP_SIZE")
fi
if [ "$APPLY_MODEL_SHAPE_OVERRIDES" = "true" ]; then
    CMD+=(--override "num_hidden_layers=$NUM_LAYERS")
    CMD+=(--override "linear_attention_freq=$LINEAR_ATTENTION_FREQ")
fi
for kv in $EXTRA_OVERRIDES; do
    CMD+=(--override "$kv")
done
if [ -n "$TOKENIZER_NAME" ]; then
    CMD+=(--tokenizer_name "$TOKENIZER_NAME")
fi
if [ -n "$LOAD_INIT_WEIGHTS" ]; then
    CMD+=(--load_init_weights "$LOAD_INIT_WEIGHTS")
fi
case "$RECOMPUTE" in
    act) CMD+=(--activation_checkpointing) ;;
    act_cpu) CMD+=(--activation_checkpointing --cpu_checkpointing) ;;
esac

"${CMD[@]}"
