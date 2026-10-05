#!/bin/bash
# Experiment driver for the DeepSpeed offload + recompute matrix.
#
# Loop structure (outer -> inner):
#   outermost: HF model                   {Qwen/Qwen3.5-9B-Base (dense),
#                                          Qwen/Qwen3.5-35B-A3B-Base (MoE, 64 experts)}
#   outer : micro-batch size sweep        {1, 2, 4, 8}
#   middle: recompute x cpu_checkpoint    {recompute_none, recompute_act,
#                                          recompute_act_cpu}
#           (cpu_checkpoint requires recompute=act, so the (off, on)
#           combination is invalid and not generated)
#   inner : optimizer strategy × param placement
#           optimizer_strategy ∈ {zero_3, zero_offload,
#                                 super_offload_1.0, super_offload_0.9,
#                                 super_offload_0.75, super_offload_0.1}
#           param_placement    ∈ {param_gpu, param_cpu, param_nvme}
#                                (param_cpu adds an offload_param cpu block
#                                 and param_nvme adds a ZeRO-Infinity NVMe
#                                 block; orthogonal axis)
#
# Every test carries a long, self-describing name:
#   <strategy>__<recompute_combo>__mbs<N>
#   e.g. super_offload_0.9-param_cpu__recompute_act__mbs4
# NVMe parameter offload is named with the param_nvme placement.
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
#   ./pretrain_experiment.sh --models Qwen/Qwen3.5-9B-Base         # dense only
#   ./pretrain_experiment.sh --models Qwen/Qwen3.5-35B-A3B-Base    # MoE only
#   ./pretrain_experiment.sh --optimizer_strategies "zero_3 super_offload_0.9" \
#       --param_positions "param_cpu param_gpu" --recompute_combos recompute_act
#   ./pretrain_experiment.sh --models Qwen/Qwen3.5-9B-Base \
#       --optimizer_strategies zero_3 --param_positions param_nvme \
#       --recompute_combos recompute_act --micro_batch_sizes 1 \
#       --nvme_path /tmp/deepspeed_nvme_offload \
#       --nvme_device /dev/nvme2n1
#   # NVMe-only sweep using the default model, batch, recompute, and optimizer axes:
#   ./pretrain_experiment.sh --param_positions param_nvme \
#       --nvme_path /tmp/deepspeed_nvme_offload \
#       --nvme_device /dev/nvme2n1
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
# Repo root (megatron-bridge): 4 levels up from this script.
REPO_ROOT=${REPO_ROOT:-"$(cd "$SCRIPT_DIR/../../../.." && pwd)"}

# ---------------------------------------------------------------------------
# Experiment matrix knobs
# ---------------------------------------------------------------------------
# Outermost loop: HF model names — Qwen3.5 dense + Qwen3.5 MoE. MoE is
# selected by the caller below from the model name.
MODELS="Qwen/Qwen3.5-9B-Base Qwen/Qwen3.5-35B-A3B-Base"
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
PARAM_POSITIONS="param_cpu param_gpu param_nvme"

# Per-GPU samples per optimizer step; grad_accum = PER_GPU_BATCH_SIZE / mbs so
# the global batch stays constant across the micro-batch sweep (must match
# PER_GPU_BATCH_SIZE in pretrain.sh).
PER_GPU_BATCH_SIZE=16
NUM_GPUS=4
# Conservative AdamW learning rate for the short random-initialization
# stability/throughput comparison. 1e-3 is too large for this model scale and
# can turn a valid first update into NaN/Inf before the offload path is tested.
LEARNING_RATE=${LEARNING_RATE:-1e-4}
# Shrunk layer count; only applied — and only tagged onto result folder names
# as _<N>layer — when APPLY_MODEL_SHAPE_OVERRIDES=true (see pretrain.sh model
# shape section). Passed explicitly to pretrain.sh.
NUM_LAYERS=8
APPLY_MODEL_SHAPE_OVERRIDES=false
AUTOEP_SIZE=4

# ZeRO-Infinity parameter offload. /tmp is mounted from /dev/nvme2n1 on the
# target machine. Each run gets a unique child directory which is removed
# after the run by default so a full matrix cannot accumulate stale swap data.
NVME_PATH=${NVME_PATH:-/tmp/deepspeed_nvme_offload}
NVME_DEVICE=${NVME_DEVICE:-/dev/nvme2n1}
NVME_DEVICE_CHECK=${NVME_DEVICE_CHECK:-true}
NVME_BUFFER_COUNT=${NVME_BUFFER_COUNT:-16}
# Qwen3.5 has a large embedding parameter. 4e8 elements covers its per-rank
# partition in the default four-GPU matrix; override when changing world size.
NVME_BUFFER_SIZE=${NVME_BUFFER_SIZE:-400000000}
# Keep no parameter elements permanently resident in CPU memory so this axis
# actually exercises NVMe residency (small non-swappable tensors are exempt).
NVME_MAX_IN_CPU=${NVME_MAX_IN_CPU:-0}
KEEP_NVME_DATA=${KEEP_NVME_DATA:-false}
PYTHON_BIN=${PYTHON_BIN:-python}
DRY_RUN=false

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
  --learning_rate VALUE
  --num_gpus VALUE
  --num_layers VALUE
  --apply_model_shape_overrides true|false
  --moe_num_experts VALUE
  --autoep_size VALUE
  --nvme_path VALUE
  --nvme_device VALUE
  --nvme_device_check true|false
  --nvme_buffer_count VALUE
  --nvme_buffer_size VALUE
  --nvme_max_in_cpu VALUE
  --keep_nvme_data true|false
  --dry_run
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --models|--micro_batch_sizes|--recompute_combos|--optimizer_strategies|--param_positions|\
        --per_gpu_batch_size|--learning_rate|--num_gpus|--num_layers|--apply_model_shape_overrides|--moe_num_experts|--autoep_size)
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
                --learning_rate) LEARNING_RATE=$2 ;;
                --num_gpus) NUM_GPUS=$2 ;;
                --num_layers) NUM_LAYERS=$2 ;;
                --apply_model_shape_overrides) APPLY_MODEL_SHAPE_OVERRIDES=$2 ;;
                --moe_num_experts) MOE_NUM_EXPERTS=$2 ;;
                --autoep_size) AUTOEP_SIZE=$2 ;;
            esac
            shift 2
            ;;
        --nvme_path|--nvme_device|--nvme_device_check|--nvme_buffer_count|--nvme_buffer_size|\
        --nvme_max_in_cpu|--keep_nvme_data)
            if [ "$#" -lt 2 ]; then
                echo "Missing value for $1" >&2
                exit 2
            fi
            case "$1" in
                --nvme_path) NVME_PATH=$2 ;;
                --nvme_device) NVME_DEVICE=$2 ;;
                --nvme_device_check) NVME_DEVICE_CHECK=$2 ;;
                --nvme_buffer_count) NVME_BUFFER_COUNT=$2 ;;
                --nvme_buffer_size) NVME_BUFFER_SIZE=$2 ;;
                --nvme_max_in_cpu) NVME_MAX_IN_CPU=$2 ;;
                --keep_nvme_data) KEEP_NVME_DATA=$2 ;;
            esac
            shift 2
            ;;
        --dry_run)
            DRY_RUN=true
            shift
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

for MODEL in $MODELS; do
    case "$MODEL" in
        Qwen/Qwen3.5-9B-Base|Qwen/Qwen3.5-35B-A3B-Base) ;;
        *)
            echo "Unsupported comparison model: $MODEL; use the aligned Qwen3.5 Base model IDs" >&2
            exit 2
            ;;
    esac
done
if [ "$APPLY_MODEL_SHAPE_OVERRIDES" != "false" ]; then
    echo "Model shape overrides are disabled for the Megatron comparison; use native layer counts" >&2
    exit 2
fi
if ! [[ "$LEARNING_RATE" =~ ^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]] || \
    ! awk -v lr="$LEARNING_RATE" 'BEGIN { exit !(lr + 0 > 0) }'; then
    echo "--learning_rate must be a positive finite decimal or scientific-notation value" >&2
    exit 2
fi
if [ "$MOE_NUM_EXPERTS" -ne 64 ]; then
    echo "The Megatron comparison fixes the MoE model to 64 experts" >&2
    exit 2
fi

USES_NVME=false
for PARAM_POSITION in $PARAM_POSITIONS; do
    case "$PARAM_POSITION" in
        param_gpu|param_cpu) ;;
        param_nvme) USES_NVME=true ;;
        *)
            echo "Unknown parameter position: $PARAM_POSITION (expected param_gpu, param_cpu, or param_nvme)" >&2
            exit 2
            ;;
    esac
done

if [ "$USES_NVME" = "true" ]; then
    case "$NVME_DEVICE_CHECK" in
        true|false) ;;
        *) echo "--nvme_device_check must be true or false" >&2; exit 2 ;;
    esac
    case "$KEEP_NVME_DATA" in
        true|false) ;;
        *) echo "--keep_nvme_data must be true or false" >&2; exit 2 ;;
    esac
    if [ -z "$NVME_PATH" ] || [ "$NVME_PATH" = "/" ] || [[ "$NVME_PATH" != /* ]]; then
        echo "--nvme_path must be an absolute directory other than /" >&2
        exit 2
    fi
    NVME_PATH=${NVME_PATH%/}
    if ! [[ "$NVME_BUFFER_COUNT" =~ ^[0-9]+$ ]] || [ "$NVME_BUFFER_COUNT" -lt 1 ]; then
        echo "--nvme_buffer_count must be a positive integer" >&2
        exit 2
    fi
    if ! [[ "$NVME_BUFFER_SIZE" =~ ^[0-9]+$ ]] || [ "$NVME_BUFFER_SIZE" -lt 1 ]; then
        echo "--nvme_buffer_size must be a positive integer" >&2
        exit 2
    fi
    if ! [[ "$NVME_MAX_IN_CPU" =~ ^[0-9]+$ ]]; then
        echo "--nvme_max_in_cpu must be a non-negative integer" >&2
        exit 2
    fi
fi

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

# Validate the target mount before model construction. The device check catches
# the easy-to-miss case where /tmp exists but the NVMe mount is absent, which
# would otherwise fill the root filesystem. Containers that expose the host
# mount under a virtual source can opt out with --nvme_device_check false.
validate_nvme_environment() {
    mkdir -p "$NVME_PATH"
    if [ ! -d "$NVME_PATH" ] || [ ! -w "$NVME_PATH" ]; then
        echo "NVMe path is not a writable directory: $NVME_PATH" >&2
        exit 2
    fi

    local mount_source
    local available_kb
    local mount_point
    read -r mount_source available_kb mount_point < <(df -Pk "$NVME_PATH" | awk 'END {print $1, $4, $6}')
    if [ "$NVME_DEVICE_CHECK" = "true" ] && \
        [ "$mount_source" != "$NVME_DEVICE" ] && [[ "$mount_source" != "${NVME_DEVICE}"p* ]]; then
        echo "NVMe path $NVME_PATH is on $mount_source, expected $NVME_DEVICE" >&2
        echo "Mount $NVME_DEVICE under /tmp, or use --nvme_device_check false when the device is hidden by a container." >&2
        exit 2
    fi

    echo "NVMe offload: path=$NVME_PATH source=$mount_source mount=$mount_point available=$((available_kb / 1024 / 1024)) GiB"

    if ! "$PYTHON_BIN" - <<'PY'
from deepspeed.ops.op_builder import AsyncIOBuilder

AsyncIOBuilder().load(verbose=False)
PY
    then
        echo "DeepSpeed async_io preflight failed; NVMe parameter offload cannot run" >&2
        exit 2
    fi
}

# ---------------------------------------------------------------------------
# build_ds_config <optimizer_strategy> <param_position> <micro_batch_size>
#   <grad_accum> <output_path> <run_nvme_path>
#   Batch settings are the only values that vary inside the common JSON
#   template.
# ---------------------------------------------------------------------------
build_ds_config() {
    local optimizer_strategy=$1
    local param_position=$2
    local mbs=$3
    local grad_accum=$4
    local ds_config_json=$5
    local run_nvme_path=$6

    local param_block=""
    local aio_block=""
    local optimizer_block=""
    local optimizer_params_block=""
    case "$param_position" in
        param_cpu)
            param_block=',
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        }'
            ;;
        param_gpu) ;;
        param_nvme)
            param_block=',
        "offload_param": {
            "device": "nvme",
            "nvme_path": "'"$run_nvme_path"'",
            "pin_memory": true,
            "buffer_count": '"$NVME_BUFFER_COUNT"',
            "buffer_size": '"$NVME_BUFFER_SIZE"',
            "max_in_cpu": '"$NVME_MAX_IN_CPU"'
        }'
            aio_block=',
    "aio": {
        "block_size": 1048576,
        "queue_depth": 8,
        "intra_op_parallelism": 1,
        "single_submit": false,
        "overlap_events": true,
        "use_gds": false
    }'
            ;;
        *)
            echo "Unknown parameter position: $param_position (expected param_cpu, param_gpu, or param_nvme)" >&2
            exit 2
            ;;
    esac

    case "$optimizer_strategy" in
        zero_3)
            optimizer_params_block=',
            "torch_adam": true'
            ;;
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
            "lr": $LEARNING_RATE,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01${optimizer_params_block}
        }
    },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": $OVERLAP_COMM,
        "reduce_bucket_size": 4e8,
        "sub_group_size": 4e8${optimizer_block}${param_block}
    },
    "wall_clock_breakdown": true${aio_block}
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

if [ "$USES_NVME" = "true" ]; then
    if [ "$DRY_RUN" = "true" ]; then
        echo "DRY RUN: skipping NVMe mount and DeepSpeed async_io preflight"
    else
        validate_nvme_environment
    fi
fi

ACTIVE_NVME_PATH=""
cleanup_nvme_run_path() {
    local run_nvme_path=$1
    if [ -z "$run_nvme_path" ] || [ "$KEEP_NVME_DATA" = "true" ]; then
        return 0
    fi
    case "$run_nvme_path" in
        "${NVME_PATH}/${INVOKE_TS}/"*) ;;
        *)
            echo "Refusing to remove unexpected NVMe path: $run_nvme_path" >&2
            return 1
            ;;
    esac
    rm -rf -- "$run_nvme_path"
}

cleanup_active_nvme_path() {
    if [ -n "$ACTIVE_NVME_PATH" ]; then
        cleanup_nvme_run_path "$ACTIVE_NVME_PATH"
    fi
}
trap cleanup_active_nvme_path EXIT

for MODEL in $MODELS; do
    # Strip the HF org prefix for filesystem use: Qwen/Qwen3.5-9B-Base -> Qwen3.5-9B-Base.
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
                    RUN_NVME_PATH=""
                    if [ "$PARAM_POSITION" = "param_nvme" ]; then
                        RUN_NVME_PATH="${NVME_PATH}/${INVOKE_TS}/${MODEL_DIR}/${TEST_NAME}/${RUN_TS}"
                        if [ "$DRY_RUN" != "true" ] && ! mkdir -p "$RUN_NVME_PATH"; then
                            echo "Failed to create NVMe run directory: $RUN_NVME_PATH" >&2
                            exit 2
                        fi
                        ACTIVE_NVME_PATH=$RUN_NVME_PATH
                    fi

                    build_ds_config \
                        "$OPTIMIZER_STRATEGY" "$PARAM_POSITION" "$MBS" "$GRAD_ACCUM" \
                        "$DS_CONFIG" "$RUN_NVME_PATH"

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
                    if [ "$DRY_RUN" = "true" ]; then
                        cp "$DS_CONFIG" "${RUN_DIR}/ds_config.json"
                        echo "DRY RUN: generated $DS_CONFIG"
                        RUN_RC=0
                    else
                        set +e  # record a failing/OOM run and continue the sweep
                        bash "${SCRIPT_DIR}/pretrain.sh" "${PRETRAIN_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
                        RUN_RC=${PIPESTATUS[0]}
                        set -e
                    fi

                    if [ "$DRY_RUN" = "true" ]; then
                        STATUS="DRY_RUN"
                    elif [ "$RUN_RC" -eq 0 ]; then
                        STATUS="OK"
                    else
                        if grep -qi "out of memory\|CUDA out of memory\|OutOfMemoryError" "$LOG_FILE"; then
                            STATUS="OOM"
                        else
                            STATUS="FAILED(rc=$RUN_RC)"
                        fi
                    fi
                    echo "${MODEL_DIR}/${TEST_NAME}/${RUN_TS} ${STATUS}" | tee -a "$SUMMARY_FILE"

                    if [ "$PARAM_POSITION" = "param_nvme" ]; then
                        if [ "$KEEP_NVME_DATA" = "true" ] && [ "$DRY_RUN" != "true" ]; then
                            echo "Retained NVMe swap data: $RUN_NVME_PATH"
                        elif ! cleanup_nvme_run_path "$RUN_NVME_PATH"; then
                            echo "Failed to clean NVMe swap data; stopping before the next run" >&2
                            exit 1
                        fi
                        ACTIVE_NVME_PATH=""
                    fi
                done
            done
        done
    done
done

echo ""
echo "================ SUMMARY ================"
cat "$SUMMARY_FILE"
echo "Tmp configs: $TMP_CONFIG_DIR"
echo "Results:     $RESULTS_ROOT"
if [ "$USES_NVME" = "true" ]; then
    echo "NVMe root:   $NVME_PATH (keep_data=$KEEP_NVME_DATA)"
fi
