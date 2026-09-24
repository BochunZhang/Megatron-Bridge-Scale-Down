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
#   inner : offload strategy              <optimizer_strategy>-<param_placement>
#           optimizer_strategy ∈ {zero_3, zero_offload_cpu,
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
#   MODELS="Qwen/Qwen3.5-9B" ./pretrain_experiment.sh         # dense only
#   MODELS="Qwen/Qwen3.5-35B-A3B" ./pretrain_experiment.sh    # MoE only
#   OFFLOAD_STRATEGIES="zero_3-param_gpu zero_offload_cpu-param_cpu super_offload_1.0-param_cpu" ./pretrain_experiment.sh
#   RECOMPUTE_COMBOS="recompute_act" MICRO_BATCH_SIZES="1 8" ./pretrain_experiment.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
# Repo root (megatron-bridge): 4 levels up from this script.
REPO_ROOT=${REPO_ROOT:-"$(cd "$SCRIPT_DIR/../../../.." && pwd)"}

# ---------------------------------------------------------------------------
# Experiment matrix knobs
# ---------------------------------------------------------------------------
# Outermost loop: HF model names — Qwen3.5 dense + Qwen3.5 MoE. MoE is
# detected by train.py/pretrain.sh via the config's int num_experts field.
MODELS=${MODELS:-"Qwen/Qwen3.5-9B Qwen/Qwen3.5-35B-A3B"}
# The 35B-A3B MoE model is resized to 64 experts for the sweep (must be
# divisible by AUTOEP_SIZE in pretrain.sh). Applied as an extra
# --override num_experts=$MOE_NUM_EXPERTS for model names matching *A3B*.
MOE_NUM_EXPERTS=${MOE_NUM_EXPERTS:-64}

# Outer loop: micro-batch size sweep axis.
MICRO_BATCH_SIZES=${MICRO_BATCH_SIZES:-"1 2 4 8"}

# Middle loop: recompute x cpu_checkpoint combinations (train.py flags:
#   recompute_none    -> (no flags)
#   recompute_act     -> --activation_checkpointing
#   recompute_act_cpu -> --activation_checkpointing --cpu_checkpointing
# )
RECOMPUTE_COMBOS=${RECOMPUTE_COMBOS:-"recompute_none recompute_act recompute_act_cpu"}

# Inner loop: offload strategies. Each name is <optimizer_strategy>-<param_placement>:
#   optimizer_strategy: zero_3 | zero_offload_cpu | super_offload_<ratio>
#                       (each maps to one ds_config heredoc below)
#   param_placement   : param_gpu (params stay on GPU) |
#                       param_cpu (adds "offload_param": cpu to zero_optimization)
# Bare names without a -param_* suffix are also accepted (== param_gpu).
OFFLOAD_STRATEGIES=${OFFLOAD_STRATEGIES:-"\
zero_3-param_gpu zero_3-param_cpu \
zero_offload_cpu-param_gpu zero_offload_cpu-param_cpu \
super_offload_1.0-param_gpu super_offload_1.0-param_cpu \
super_offload_0.9-param_gpu super_offload_0.9-param_cpu \
super_offload_0.75-param_gpu super_offload_0.75-param_cpu \
super_offload_0.1-param_gpu super_offload_0.1-param_cpu"}

# Per-GPU samples per optimizer step; grad_accum = PER_GPU_BATCH_SIZE / mbs so
# the global batch stays constant across the micro-batch sweep (must match
# PER_GPU_BATCH_SIZE in pretrain.sh).
PER_GPU_BATCH_SIZE=${PER_GPU_BATCH_SIZE:-16}
# Shrunk layer count; only applied — and only tagged onto result folder names
# as _<N>layer — when APPLY_MODEL_SHAPE_OVERRIDES=true (see pretrain.sh model
# shape section). Exported so pretrain.sh sees the same values.
NUM_LAYERS=${NUM_LAYERS:-8}
APPLY_MODEL_SHAPE_OVERRIDES=${APPLY_MODEL_SHAPE_OVERRIDES:-false}
export NUM_LAYERS PER_GPU_BATCH_SIZE APPLY_MODEL_SHAPE_OVERRIDES

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

# ---------------------------------------------------------------------------
# Common DeepSpeed knobs, fixed for all runs (README §2)
# ---------------------------------------------------------------------------
BF16_ENABLED=${BF16_ENABLED:-true}
ZERO_STAGE=${ZERO_STAGE:-3}
OVERLAP_COMM=${OVERLAP_COMM:-false}
REDUCE_BUCKET_SIZE=${REDUCE_BUCKET_SIZE:-4e8}
SUB_GROUP_SIZE=${SUB_GROUP_SIZE:-4e8}
PIN_MEMORY=${PIN_MEMORY:-true}
WALL_CLOCK_BREAKDOWN=${WALL_CLOCK_BREAKDOWN:-true}
# SuperOffload knobs (README §2: cpuadam_cores_perc = 0.90 for all runs).
CPUADAM_CORES_PERC=${CPUADAM_CORES_PERC:-0.90}
# ZeRO-Infinity NVMe offload path (zero_offload_nvme only).
NVME_PATH=${NVME_PATH:-/dev/nvme2n1}
# Optimizer section for every generated ds_config. train.py does NOT create a
# client optimizer: DeepSpeed builds DeepSpeedCPUAdam from this section when
# zero-offload / super-offload is enabled, and GPU FusedAdam when it is not.
OPTIMIZER_TYPE=${OPTIMIZER_TYPE:-AdamW}
OPTIMIZER_LR=${OPTIMIZER_LR:-0.001}
OPTIMIZER_BETA1=${OPTIMIZER_BETA1:-0.9}
OPTIMIZER_BETA2=${OPTIMIZER_BETA2:-0.999}
OPTIMIZER_EPS=${OPTIMIZER_EPS:-1e-8}
OPTIMIZER_WEIGHT_DECAY=${OPTIMIZER_WEIGHT_DECAY:-0.0}

# ---------------------------------------------------------------------------
# build_ds_config <strategy> <micro_batch_size> <grad_accum> <output_path>
#   <strategy> is <optimizer_strategy>-<param_placement> (a bare
#   <optimizer_strategy> is accepted and treated as param_gpu). One explicit
#   heredoc block per optimizer strategy (finetune_qwen35_7b.sh style) so
#   every generated JSON can be diffed/verified by hand; the param axis is
#   injected via $param_block appended to zero_optimization.
#
#   NOTE: the "optimizer" section IS emitted (AdamW by default) and no
#   "scheduler" section is: train.py passes optimizer=None to
#   deepspeed.initialize(), so DeepSpeed builds the optimizer from this
#   section — DeepSpeedCPUAdam when offload is enabled, GPU FusedAdam
#   otherwise.
# ---------------------------------------------------------------------------
build_ds_config() {
    local strategy=$1
    local mbs=$2
    local grad_accum=$3
    local ds_config_json=$4

    # Split off the -param_cpu / -param_gpu suffix; bare names keep params on
    # GPU (== param_gpu). param_cpu appends an offload_param cpu block to
    # zero_optimization (ZeRO-Infinity parameter offload; requires stage 3).
    local base=$strategy
    local param_block=""
    case "$strategy" in
        *-param_cpu)
            base="${strategy%-param_cpu}"
            param_block=$(printf ',\n        "offload_param": {\n            "device": "cpu",\n            "pin_memory": %s\n        }' "$PIN_MEMORY")
            ;;
        *-param_gpu)
            base="${strategy%-param_gpu}"
            ;;
    esac

    if [ "$base" = "zero_3" ]; then
cat > "$ds_config_json" << EOF
{
    "train_micro_batch_size_per_gpu": $mbs,
    "gradient_accumulation_steps": $grad_accum,
    "bf16": { "enabled": $BF16_ENABLED },
    "optimizer": {
        "type": "$OPTIMIZER_TYPE",
        "params": {
            "lr": $OPTIMIZER_LR,
            "betas": [$OPTIMIZER_BETA1, $OPTIMIZER_BETA2],
            "eps": $OPTIMIZER_EPS,
            "weight_decay": $OPTIMIZER_WEIGHT_DECAY
        }
    },
    "zero_optimization": {
        "stage": $ZERO_STAGE,
        "overlap_comm": $OVERLAP_COMM,
        "reduce_bucket_size": $REDUCE_BUCKET_SIZE,
        "sub_group_size": $SUB_GROUP_SIZE$param_block
    },
    "wall_clock_breakdown": $WALL_CLOCK_BREAKDOWN
}
EOF

    elif [ "$base" = "zero_offload_cpu" ]; then
cat > "$ds_config_json" << EOF
{
    "train_micro_batch_size_per_gpu": $mbs,
    "gradient_accumulation_steps": $grad_accum,
    "bf16": { "enabled": $BF16_ENABLED },
    "optimizer": {
        "type": "$OPTIMIZER_TYPE",
        "params": {
            "lr": $OPTIMIZER_LR,
            "betas": [$OPTIMIZER_BETA1, $OPTIMIZER_BETA2],
            "eps": $OPTIMIZER_EPS,
            "weight_decay": $OPTIMIZER_WEIGHT_DECAY
        }
    },
    "zero_optimization": {
        "stage": $ZERO_STAGE,
        "overlap_comm": $OVERLAP_COMM,
        "reduce_bucket_size": $REDUCE_BUCKET_SIZE,
        "sub_group_size": $SUB_GROUP_SIZE,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": $PIN_MEMORY
        }$param_block
    },
    "wall_clock_breakdown": $WALL_CLOCK_BREAKDOWN
}
EOF

    elif [ "$base" = "super_offload_1.0" ]; then
cat > "$ds_config_json" << EOF
{
    "train_micro_batch_size_per_gpu": $mbs,
    "gradient_accumulation_steps": $grad_accum,
    "bf16": { "enabled": $BF16_ENABLED },
    "optimizer": {
        "type": "$OPTIMIZER_TYPE",
        "params": {
            "lr": $OPTIMIZER_LR,
            "betas": [$OPTIMIZER_BETA1, $OPTIMIZER_BETA2],
            "eps": $OPTIMIZER_EPS,
            "weight_decay": $OPTIMIZER_WEIGHT_DECAY
        }
    },
    "zero_optimization": {
        "stage": $ZERO_STAGE,
        "overlap_comm": $OVERLAP_COMM,
        "reduce_bucket_size": $REDUCE_BUCKET_SIZE,
        "sub_group_size": $SUB_GROUP_SIZE,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": $PIN_MEMORY,
            "ratio": 1.0,
            "super_offload": true,
            "cpuadam_cores_perc": $CPUADAM_CORES_PERC
        }$param_block
    },
    "wall_clock_breakdown": $WALL_CLOCK_BREAKDOWN
}
EOF

    elif [ "$base" = "super_offload_0.9" ]; then
cat > "$ds_config_json" << EOF
{
    "train_micro_batch_size_per_gpu": $mbs,
    "gradient_accumulation_steps": $grad_accum,
    "bf16": { "enabled": $BF16_ENABLED },
    "optimizer": {
        "type": "$OPTIMIZER_TYPE",
        "params": {
            "lr": $OPTIMIZER_LR,
            "betas": [$OPTIMIZER_BETA1, $OPTIMIZER_BETA2],
            "eps": $OPTIMIZER_EPS,
            "weight_decay": $OPTIMIZER_WEIGHT_DECAY
        }
    },
    "zero_optimization": {
        "stage": $ZERO_STAGE,
        "overlap_comm": $OVERLAP_COMM,
        "reduce_bucket_size": $REDUCE_BUCKET_SIZE,
        "sub_group_size": $SUB_GROUP_SIZE,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": $PIN_MEMORY,
            "ratio": 0.9,
            "super_offload": true,
            "cpuadam_cores_perc": $CPUADAM_CORES_PERC
        }$param_block
    },
    "wall_clock_breakdown": $WALL_CLOCK_BREAKDOWN
}
EOF

    elif [ "$base" = "super_offload_0.75" ]; then
cat > "$ds_config_json" << EOF
{
    "train_micro_batch_size_per_gpu": $mbs,
    "gradient_accumulation_steps": $grad_accum,
    "bf16": { "enabled": $BF16_ENABLED },
    "optimizer": {
        "type": "$OPTIMIZER_TYPE",
        "params": {
            "lr": $OPTIMIZER_LR,
            "betas": [$OPTIMIZER_BETA1, $OPTIMIZER_BETA2],
            "eps": $OPTIMIZER_EPS,
            "weight_decay": $OPTIMIZER_WEIGHT_DECAY
        }
    },
    "zero_optimization": {
        "stage": $ZERO_STAGE,
        "overlap_comm": $OVERLAP_COMM,
        "reduce_bucket_size": $REDUCE_BUCKET_SIZE,
        "sub_group_size": $SUB_GROUP_SIZE,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": $PIN_MEMORY,
            "ratio": 0.75,
            "super_offload": true,
            "cpuadam_cores_perc": $CPUADAM_CORES_PERC
        }$param_block
    },
    "wall_clock_breakdown": $WALL_CLOCK_BREAKDOWN
}
EOF

    elif [ "$base" = "super_offload_0.1" ]; then
cat > "$ds_config_json" << EOF
{
    "train_micro_batch_size_per_gpu": $mbs,
    "gradient_accumulation_steps": $grad_accum,
    "bf16": { "enabled": $BF16_ENABLED },
    "optimizer": {
        "type": "$OPTIMIZER_TYPE",
        "params": {
            "lr": $OPTIMIZER_LR,
            "betas": [$OPTIMIZER_BETA1, $OPTIMIZER_BETA2],
            "eps": $OPTIMIZER_EPS,
            "weight_decay": $OPTIMIZER_WEIGHT_DECAY
        }
    },
    "zero_optimization": {
        "stage": $ZERO_STAGE,
        "overlap_comm": $OVERLAP_COMM,
        "reduce_bucket_size": $REDUCE_BUCKET_SIZE,
        "sub_group_size": $SUB_GROUP_SIZE,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": $PIN_MEMORY,
            "ratio": 0.1,
            "super_offload": true,
            "cpuadam_cores_perc": $CPUADAM_CORES_PERC
        }$param_block
    },
    "wall_clock_breakdown": $WALL_CLOCK_BREAKDOWN
}
EOF

    else
        echo "Unknown offload strategy: $strategy (base: $base; expected zero_3|zero_offload_cpu|super_offload_1.0|super_offload_0.9|super_offload_0.75|super_offload_0.1 with optional -param_gpu|-param_cpu suffix)" >&2
        exit 2
    fi
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

    # Per-model extra HF config overrides consumed by pretrain.sh: the
    # 35B-A3B MoE model is resized to $MOE_NUM_EXPERTS experts.
    case "$MODEL" in
        *A3B*) EXTRA_OVERRIDES="num_experts=$MOE_NUM_EXPERTS" ;;
        *)     EXTRA_OVERRIDES="" ;;
    esac
    export EXTRA_OVERRIDES

    for MBS in $MICRO_BATCH_SIZES; do
        if [ $((PER_GPU_BATCH_SIZE % MBS)) -ne 0 ]; then
            echo "PER_GPU_BATCH_SIZE($PER_GPU_BATCH_SIZE) must be divisible by micro_batch_size($MBS)" >&2
            exit 2
        fi
        GRAD_ACCUM=$((PER_GPU_BATCH_SIZE / MBS))

        for RECOMPUTE_COMBO in $RECOMPUTE_COMBOS; do
            RECOMPUTE=$(recompute_token "$RECOMPUTE_COMBO")

            for STRATEGY in $OFFLOAD_STRATEGIES; do
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

                build_ds_config "$STRATEGY" "$MBS" "$GRAD_ACCUM" "$DS_CONFIG"

                echo ""
                echo "################ RUN ${MODEL_DIR}/${TEST_NAME}/${RUN_TS} ################"
                bash "${SCRIPT_DIR}/pretrain.sh" \
                    "$TEST_NAME" "$MODEL" "$MBS" "$DS_CONFIG" "$RECOMPUTE" "$METRICS_OUT" \
                    2>&1 | tee "$LOG_FILE"
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
set -e

echo ""
echo "================ SUMMARY ================"
cat "$SUMMARY_FILE"
echo "Tmp configs: $TMP_CONFIG_DIR"
echo "Results:     $RESULTS_ROOT"
