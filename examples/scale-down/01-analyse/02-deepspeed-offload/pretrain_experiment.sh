#!/bin/bash
# Experiment driver for the DeepSpeed offload + recompute matrix.
#
# Loop structure (outer -> inner):
#   outermost: HF model                   {Qwen/Qwen3.5-9B (dense),
#                                          Qwen/Qwen3.5-35B-A3B (MoE, 64 experts)}
#   outer : micro-batch size sweep        {1, 2, 4, 8}
#   middle: recompute x cpu_checkpoint    {recompute_none, recompute_act,
#                                          recompute_act_cpu}
#           (cpu_checkpoint requires recompute=act, so the (off, on)
#           combination is invalid and not generated)
#   inner : optimizer strategy × param placement
#           optimizer_strategy ∈ {zero_3, zero_offload,
#                                 super_offload_1.0, super_offload_0.9,
#                                 super_offload_0.75, super_offload_0.1}
#           param_placement    ∈ {param_gpu, param_cpu}
#                                (param_cpu adds an offload_param cpu block
#                                 to zero_optimization; orthogonal axis)
#
# Every test carries a long, self-describing name:
#   <strategy>__<recompute_combo>__mbs<N>
#   e.g. super_offload_0.9-param_cpu__recompute_act__mbs4
# (NVMe offload shows up in the name via the zero_offload_nvme strategy.)
#
# Output layout — one independent folder per test, timestamped per run:
#   <repo_root>/results/01-analyse/02-deepspeed/<model>[_<N>layer]/<TEST_NAME>/<timestamp>/
#       ├── run.log        full stdout/stderr
#       ├── metrics.csv    per-step metrics (train.py MetricsLogger)
#       └── ds_config.json exact config used (copied by pretrain.sh)
#   <repo_root>/results/01-analyse/02-deepspeed/experiment_summary_<ts>.txt
#
# The working ds_config JSONs are generated under <repo_root>/.tmp/ (built by
# an explicit per-strategy heredoc block in build_ds_config() so each file can
# be verified by hand); pretrain.sh copies the one it used into the run dir.
#
# Usage:
#   ./pretrain_experiment.sh                          # full matrix, dense + MoE
#   ./pretrain_experiment.sh --models Qwen/Qwen3.5-9B         # dense only
#   ./pretrain_experiment.sh --models Qwen/Qwen3.5-35B-A3B    # MoE only
#   ./pretrain_experiment.sh --optimizer_strategies "zero_3 super_offload_0.9" \
#       --param_positions "param_cpu param_gpu" --recompute_combos recompute_act
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
# Repo root (megatron-bridge): 4 levels up from this script.
REPO_ROOT=${REPO_ROOT:-"$(cd "$SCRIPT_DIR/../../../.." && pwd)"}

# ---------------------------------------------------------------------------
# Experiment matrix knobs
# ---------------------------------------------------------------------------
# Outermost loop: HF model names — Qwen3.5 dense + Qwen3.5 MoE. MoE is
# selected by the caller below from the model name.
MODELS="Qwen/Qwen3.5-9B Qwen/Qwen3.5-35B-A3B"
# The 35B-A3B MoE model is resized to 64 experts for the sweep (must be
# divisible by AUTOEP_SIZE in pretrain.sh). Applied as an extra
# --override num_experts=$MOE_NUM_EXPERTS for model names matching *A3B*.
MOE_NUM_EXPERTS=64

# Outer loop: micro-batch size sweep axis.
MICRO_BATCH_SIZES="1 2 4 8"

# Middle loop: recompute x cpu_checkpoint combinations (train.py flags:
#   recompute_none    -> (no flags)
#   recompute_act     -> --activation_checkpointing
#   recompute_act_cpu -> --activation_checkpointing --cpu_checkpointing
# )
RECOMPUTE_COMBOS="recompute_none recompute_act recompute_act_cpu"

# Inner loops: optimizer configuration and parameter placement are independent
# axes. Keep these lists separate so callers can select either axis directly.
OPTIMIZER_STRATEGIES="zero_3 zero_offload super_offload_1.0 super_offload_0.9 super_offload_0.75 super_offload_0.1"
PARAM_POSITIONS="param_cpu param_gpu"

# Per-GPU samples per optimizer step; grad_accum = PER_GPU_BATCH_SIZE / mbs so
# the global batch stays constant across the micro-batch sweep (must match
# PER_GPU_BATCH_SIZE in pretrain.sh).
PER_GPU_BATCH_SIZE=16
NUM_GPUS=4
# Shrunk layer count; only applied — and only tagged onto result folder names
# as _<N>layer — when APPLY_MODEL_SHAPE_OVERRIDES=true (see pretrain.sh model
# shape section). Passed explicitly to pretrain.sh.
NUM_LAYERS=8
APPLY_MODEL_SHAPE_OVERRIDES=false
AUTOEP_SIZE=4

usage() {
    cat <<'EOF'
Usage: pretrain_experiment.sh [options]

Options use space-separated values where noted:
  --models VALUE
  --micro_batch_sizes VALUE
  --recompute_combos VALUE
  --optimizer_strategies VALUE
  --param_positions VALUE
  --per_gpu_batch_size VALUE
  --num_gpus VALUE
  --num_layers VALUE
  --apply_model_shape_overrides true|false
  --moe_num_experts VALUE
  --autoep_size VALUE
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --models|--micro_batch_sizes|--recompute_combos|--optimizer_strategies|--param_positions|\
        --per_gpu_batch_size|--num_gpus|--num_layers|--apply_model_shape_overrides|--moe_num_experts|--autoep_size)
            if [ "$#" -lt 2 ]; then
                echo "Missing value for $1" >&2
                exit 2
            fi
            case "$1" in
                --models) MODELS=$2 ;;
                --micro_batch_sizes) MICRO_BATCH_SIZES=$2 ;;
                --recompute_combos) RECOMPUTE_COMBOS=$2 ;;
                --optimizer_strategies) OPTIMIZER_STRATEGIES=$2 ;;
                --param_positions) PARAM_POSITIONS=$2 ;;
                --per_gpu_batch_size) PER_GPU_BATCH_SIZE=$2 ;;
                --num_gpus) NUM_GPUS=$2 ;;
                --num_layers) NUM_LAYERS=$2 ;;
                --apply_model_shape_overrides) APPLY_MODEL_SHAPE_OVERRIDES=$2 ;;
                --moe_num_experts) MOE_NUM_EXPERTS=$2 ;;
                --autoep_size) AUTOEP_SIZE=$2 ;;
            esac
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

# ---------------------------------------------------------------------------
# Output layout
#   working ds_configs : <repo_root>/.tmp/
#   results            : <repo_root>/results/01-analyse/02-deepspeed/
#                        └── <model>[_<N>layer]/<TEST_NAME>/<timestamp>/{run.log,
#                            metrics.csv, ds_config.json}
# ---------------------------------------------------------------------------
TMP_CONFIG_DIR=${TMP_CONFIG_DIR:-"${REPO_ROOT}/.tmp"}
RESULTS_ROOT=${RESULTS_ROOT:-"${REPO_ROOT}/results/01-analyse/02-deepspeed"}
mkdir -p "$TMP_CONFIG_DIR" "$RESULTS_ROOT"

# DeepSpeed communication overlap is configurable for performance experiments.
OVERLAP_COMM=${OVERLAP_COMM:-true}

# ---------------------------------------------------------------------------
# build_ds_config <optimizer_strategy> <param_position> <micro_batch_size>
#   <grad_accum> <output_path>
#   Batch settings are the only values that vary inside the common JSON
#   template.
# ---------------------------------------------------------------------------
build_ds_config() {
    local optimizer_strategy=$1
    local param_position=$2
    local mbs=$3
    local grad_accum=$4
    local ds_config_json=$5

    local param_block=""
    local optimizer_block=""
    case "$param_position" in
        param_cpu)
            param_block=',
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        }'
            ;;
        param_gpu) ;;
        *)
            echo "Unknown parameter position: $param_position (expected param_cpu or param_gpu)" >&2
            exit 2
            ;;
    esac

    case "$optimizer_strategy" in
        zero_3) ;;
        zero_offload|zero_offload_cpu)
            optimizer_block=',
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        }'
            ;;
        super_offload_1.0|super_offload_0.9|super_offload_0.75|super_offload_0.1)
            local ratio="${optimizer_strategy#super_offload_}"
            optimizer_block=',
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true,
            "ratio": '"$ratio"',
            "super_offload": true,
            "cpuadam_cores_perc": 0.90
        }'
            ;;
        *)
            echo "Unknown optimizer strategy: $optimizer_strategy (expected zero_3|zero_offload|super_offload_1.0|super_offload_0.9|super_offload_0.75|super_offload_0.1)" >&2
            exit 2
            ;;
    esac

    cat > "$ds_config_json" << EOF
{
    "train_micro_batch_size_per_gpu": $mbs,
    "gradient_accumulation_steps": $grad_accum,
    "bf16": { "enabled": true },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 0.001,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": $OVERLAP_COMM,
        "reduce_bucket_size": 4e8,
        "sub_group_size": 4e8${optimizer_block}${param_block}
    },
    "wall_clock_breakdown": true
}
EOF
}

# ---------------------------------------------------------------------------
# recompute_token <recompute_combo>
#   Maps the middle-loop name to the pretrain.sh recompute token.
# ---------------------------------------------------------------------------
recompute_token() {
    case "$1" in
        recompute_none)     echo "none" ;;
        recompute_act)      echo "act" ;;
        recompute_act_cpu)  echo "act_cpu" ;;
        *)
            echo "Unknown recompute combo: $1 (expected recompute_none|recompute_act|recompute_act_cpu)" >&2
            exit 2
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Main sweep: model -> batch size -> recompute combo -> offload strategy
# ---------------------------------------------------------------------------
INVOKE_TS=$(date +%Y%m%d_%H%M%S)
SUMMARY_FILE="${RESULTS_ROOT}/experiment_summary_${INVOKE_TS}.txt"
: > "$SUMMARY_FILE"

set +e  # keep sweeping after a failing/OOM run; status is recorded per run
for MODEL in $MODELS; do
    # Strip the HF org prefix for filesystem use: Qwen/Qwen3.5-9B -> Qwen3.5-9B.
    # The _<N>layer tag follows the override configuration: it is only present
    # when the layer-shrink overrides are actually applied to the model.
    if [ "$APPLY_MODEL_SHAPE_OVERRIDES" = "true" ]; then
        MODEL_SHAPE_TAG="_${NUM_LAYERS}layer"
    else
        MODEL_SHAPE_TAG=""
    fi
    MODEL_DIR="${MODEL##*/}${MODEL_SHAPE_TAG}"

    # The caller owns model-mode selection. The 35B-A3B model is the MoE
    # entry in this matrix and receives the explicit AutoEP arguments below.
    case "$MODEL" in
        *A3B*)
            TRAIN_MODE=autoep
            ;;
        *)
            TRAIN_MODE=dense
            ;;
    esac

    for MBS in $MICRO_BATCH_SIZES; do
        if [ $((PER_GPU_BATCH_SIZE % MBS)) -ne 0 ]; then
            echo "PER_GPU_BATCH_SIZE($PER_GPU_BATCH_SIZE) must be divisible by micro_batch_size($MBS)" >&2
            exit 2
        fi
        GRAD_ACCUM=$((PER_GPU_BATCH_SIZE / MBS))

        for RECOMPUTE_COMBO in $RECOMPUTE_COMBOS; do
            RECOMPUTE=$(recompute_token "$RECOMPUTE_COMBO")

            for OPTIMIZER_STRATEGY in $OPTIMIZER_STRATEGIES; do
                for PARAM_POSITION in $PARAM_POSITIONS; do
                    STRATEGY="${OPTIMIZER_STRATEGY}-${PARAM_POSITION}"
                    TEST_NAME="${STRATEGY}__${RECOMPUTE_COMBO}__mbs${MBS}"
                    RUN_TS=$(date +%Y%m%d_%H%M%S)
                    RUN_DIR="${RESULTS_ROOT}/${MODEL_DIR}/${TEST_NAME}/${RUN_TS}"
                    mkdir -p "$RUN_DIR"

                    # Working copy under <repo_root>/.tmp; pretrain.sh archives it
                    # into $RUN_DIR/ds_config.json before launching. The config
                    # content is model-independent, so one file per TEST_NAME is
                    # enough (rebuilt for each model to keep .tmp self-contained).
                    DS_CONFIG="${TMP_CONFIG_DIR}/ds_config_${TEST_NAME}.json"
                    METRICS_OUT="${RUN_DIR}/metrics.csv"
                    LOG_FILE="${RUN_DIR}/run.log"

                    build_ds_config "$OPTIMIZER_STRATEGY" "$PARAM_POSITION" "$MBS" "$GRAD_ACCUM" "$DS_CONFIG"

                    echo ""
                    echo "################ RUN ${MODEL_DIR}/${TEST_NAME}/${RUN_TS} ################"
                    PRETRAIN_ARGS=(
                        --test_name "$TEST_NAME"
                        --model "$MODEL"
                        --ds_config "$DS_CONFIG"
                        --recompute "$RECOMPUTE"
                        --metrics_out "$METRICS_OUT"
                        --mode "$TRAIN_MODE"
                        --num_gpus "$NUM_GPUS"
                        --num_layers "$NUM_LAYERS"
                        --per_gpu_batch_size "$PER_GPU_BATCH_SIZE"
                        --apply_model_shape_overrides "$APPLY_MODEL_SHAPE_OVERRIDES"
                    )
                    if [ "$TRAIN_MODE" = "autoep" ]; then
                        PRETRAIN_ARGS+=(--autoep_size "$AUTOEP_SIZE")
                        PRETRAIN_ARGS+=(--override "num_experts=$MOE_NUM_EXPERTS")
                    fi
                    bash "${SCRIPT_DIR}/pretrain.sh" "${PRETRAIN_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
                    RUN_RC=${PIPESTATUS[0]}

                    STATUS="OK"
                    if [ "$RUN_RC" -ne 0 ]; then
                        if grep -qi "out of memory\|CUDA out of memory\|OutOfMemoryError" "$LOG_FILE"; then
                            STATUS="OOM"
                        else
                            STATUS="FAILED(rc=$RUN_RC)"
                        fi
                    fi
                    echo "${MODEL_DIR}/${TEST_NAME}/${RUN_TS} ${STATUS}" | tee -a "$SUMMARY_FILE"
                done
            done
        done
    done
done
set -e

echo ""
echo "================ SUMMARY ================"
cat "$SUMMARY_FILE"
echo "Tmp configs: $TMP_CONFIG_DIR"
echo "Results:     $RESULTS_ROOT"
