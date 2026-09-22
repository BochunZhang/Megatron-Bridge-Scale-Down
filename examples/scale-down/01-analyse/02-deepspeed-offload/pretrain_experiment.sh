#!/bin/bash
# Experiment driver for the DeepSpeed offload + recompute matrix.
#
# Loop structure (outer -> inner):
#   outer : micro-batch size sweep        {1, 2, 4, 8}
#   middle: recompute x cpu_checkpoint    {recompute_none, recompute_act,
#                                          recompute_act_cpu}
#           (cpu_checkpoint requires recompute=act, so the (off, on)
#           combination is invalid and not generated)
#   inner : offload strategy              {zero_3, zero_offload_cpu,
#                                          zero_offload_nvme,
#                                          super_offload_1.0,
#                                          super_offload_0.9,
#                                          super_offload_0.75}
#
# Every test carries a long, self-describing name:
#   <model>__<strategy>__<recompute_combo>__mbs<N>
#   e.g. qwen3_5__super_offload_0.9__recompute_act__mbs4
# and gets its own ds_config JSON, built by an explicit per-strategy
# heredoc block in build_ds_config() so each file can be verified by hand.
#
# Usage:
#   ./pretrain_experiment.sh                          # full matrix, dense model
#   MODEL=qwen3_5_moe ./pretrain_experiment.sh        # MoE model
#   OFFLOAD_STRATEGIES="zero_3 super_offload_1.0" ./pretrain_experiment.sh
#   RECOMPUTE_COMBOS="recompute_act" MICRO_BATCH_SIZES="1 8" ./pretrain_experiment.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# ---------------------------------------------------------------------------
# Experiment matrix knobs
# ---------------------------------------------------------------------------
# train.py model preset: qwen3_5 (dense) | qwen3_5_moe | mixtral | llama4
MODEL=${MODEL:-qwen3_5}

# Outer loop: micro-batch size sweep axis.
MICRO_BATCH_SIZES=${MICRO_BATCH_SIZES:-"1 2 4 8"}

# Middle loop: recompute x cpu_checkpoint combinations (train.py flags:
#   recompute_none    -> (no flags)
#   recompute_act     -> --activation_checkpointing
#   recompute_act_cpu -> --activation_checkpointing --cpu_checkpointing
# )
RECOMPUTE_COMBOS=${RECOMPUTE_COMBOS:-"recompute_none recompute_act recompute_act_cpu"}

# Inner loop: offload strategies (each maps to one ds_config heredoc below).
OFFLOAD_STRATEGIES=${OFFLOAD_STRATEGIES:-"zero_3 zero_offload_cpu zero_offload_nvme super_offload_1.0 super_offload_0.9 super_offload_0.75"}

# Per-GPU samples per optimizer step; grad_accum = PER_GPU_BATCH_SIZE / mbs so
# the global batch stays constant across the micro-batch sweep (must match
# PER_GPU_BATCH_SIZE in pretrain.sh).
PER_GPU_BATCH_SIZE=${PER_GPU_BATCH_SIZE:-16}

# ---------------------------------------------------------------------------
# Output layout
# ---------------------------------------------------------------------------
CONFIG_DIR=${CONFIG_DIR:-"${SCRIPT_DIR}/configs"}
RESULTS_DIR=${RESULTS_DIR:-"${SCRIPT_DIR}/results"}
mkdir -p "$CONFIG_DIR" "$RESULTS_DIR"

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
NVME_PATH=${NVME_PATH:-/local_nvme}

# ---------------------------------------------------------------------------
# build_ds_config <strategy> <micro_batch_size> <grad_accum> <output_path>
#   One explicit heredoc block per offload strategy (finetune_qwen35_7b.sh
#   style) so every generated JSON can be diffed/verified by hand.
#
#   NOTE: no "optimizer"/"scheduler" section is emitted on purpose —
#   train.py passes a client optimizer to deepspeed.initialize(), and
#   DeepSpeed rejects configs that specify an optimizer twice.
# ---------------------------------------------------------------------------
build_ds_config() {
    local strategy=$1
    local mbs=$2
    local grad_accum=$3
    local ds_config_json=$4

    if [ "$strategy" = "zero_3" ]; then
cat > "$ds_config_json" << EOF
{
    "train_micro_batch_size_per_gpu": $mbs,
    "gradient_accumulation_steps": $grad_accum,
    "bf16": { "enabled": $BF16_ENABLED },
    "zero_optimization": {
        "stage": $ZERO_STAGE,
        "overlap_comm": $OVERLAP_COMM,
        "reduce_bucket_size": $REDUCE_BUCKET_SIZE,
        "sub_group_size": $SUB_GROUP_SIZE
    },
    "wall_clock_breakdown": $WALL_CLOCK_BREAKDOWN
}
EOF

    elif [ "$strategy" = "zero_offload_cpu" ]; then
cat > "$ds_config_json" << EOF
{
    "train_micro_batch_size_per_gpu": $mbs,
    "gradient_accumulation_steps": $grad_accum,
    "bf16": { "enabled": $BF16_ENABLED },
    "zero_optimization": {
        "stage": $ZERO_STAGE,
        "overlap_comm": $OVERLAP_COMM,
        "reduce_bucket_size": $REDUCE_BUCKET_SIZE,
        "sub_group_size": $SUB_GROUP_SIZE,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": $PIN_MEMORY
        }
    },
    "wall_clock_breakdown": $WALL_CLOCK_BREAKDOWN
}
EOF

    elif [ "$strategy" = "zero_offload_nvme" ]; then
cat > "$ds_config_json" << EOF
{
    "train_micro_batch_size_per_gpu": $mbs,
    "gradient_accumulation_steps": $grad_accum,
    "bf16": { "enabled": $BF16_ENABLED },
    "zero_optimization": {
        "stage": $ZERO_STAGE,
        "overlap_comm": $OVERLAP_COMM,
        "reduce_bucket_size": $REDUCE_BUCKET_SIZE,
        "sub_group_size": $SUB_GROUP_SIZE,
        "offload_optimizer": {
            "device": "nvme",
            "nvme_path": "$NVME_PATH",
            "pin_memory": $PIN_MEMORY
        }
    },
    "wall_clock_breakdown": $WALL_CLOCK_BREAKDOWN
}
EOF

    elif [ "$strategy" = "super_offload_1.0" ]; then
cat > "$ds_config_json" << EOF
{
    "train_micro_batch_size_per_gpu": $mbs,
    "gradient_accumulation_steps": $grad_accum,
    "bf16": { "enabled": $BF16_ENABLED },
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
        }
    },
    "wall_clock_breakdown": $WALL_CLOCK_BREAKDOWN
}
EOF

    elif [ "$strategy" = "super_offload_0.9" ]; then
cat > "$ds_config_json" << EOF
{
    "train_micro_batch_size_per_gpu": $mbs,
    "gradient_accumulation_steps": $grad_accum,
    "bf16": { "enabled": $BF16_ENABLED },
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
        }
    },
    "wall_clock_breakdown": $WALL_CLOCK_BREAKDOWN
}
EOF

    elif [ "$strategy" = "super_offload_0.75" ]; then
cat > "$ds_config_json" << EOF
{
    "train_micro_batch_size_per_gpu": $mbs,
    "gradient_accumulation_steps": $grad_accum,
    "bf16": { "enabled": $BF16_ENABLED },
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
        }
    },
    "wall_clock_breakdown": $WALL_CLOCK_BREAKDOWN
}
EOF

    else
        echo "Unknown offload strategy: $strategy" >&2
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
# Main sweep: outer batch size -> middle recompute combo -> inner strategy
# ---------------------------------------------------------------------------
SUMMARY_FILE="${RESULTS_DIR}/experiment_summary.txt"
: > "$SUMMARY_FILE"

set +e  # keep sweeping after a failing/OOM run; status is recorded per run
for MBS in $MICRO_BATCH_SIZES; do
    if [ $((PER_GPU_BATCH_SIZE % MBS)) -ne 0 ]; then
        echo "PER_GPU_BATCH_SIZE($PER_GPU_BATCH_SIZE) must be divisible by micro_batch_size($MBS)" >&2
        exit 2
    fi
    GRAD_ACCUM=$((PER_GPU_BATCH_SIZE / MBS))

    for RECOMPUTE_COMBO in $RECOMPUTE_COMBOS; do
        RECOMPUTE=$(recompute_token "$RECOMPUTE_COMBO")

        for STRATEGY in $OFFLOAD_STRATEGIES; do
            TEST_NAME="${MODEL}__${STRATEGY}__${RECOMPUTE_COMBO}__mbs${MBS}"
            DS_CONFIG="${CONFIG_DIR}/ds_config_${TEST_NAME}.json"
            METRICS_OUT="${RESULTS_DIR}/${TEST_NAME}_metrics.csv"
            LOG_FILE="${RESULTS_DIR}/${TEST_NAME}.log"

            build_ds_config "$STRATEGY" "$MBS" "$GRAD_ACCUM" "$DS_CONFIG"

            echo ""
            echo "################ RUN ${TEST_NAME} ################"
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
            echo "${TEST_NAME} ${STATUS}" | tee -a "$SUMMARY_FILE"
        done
    done
done
set -e

echo ""
echo "================ SUMMARY ================"
cat "$SUMMARY_FILE"
echo "Configs:  $CONFIG_DIR"
echo "Results:  $RESULTS_DIR"
