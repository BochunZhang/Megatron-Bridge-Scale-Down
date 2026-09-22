#!/bin/bash
# Experiment driver for the DeepSpeed offload + recompute matrix (README §4).
#
# - Iterates over the test combinations; each test has an independent name
#   (B0/R1/R2/O1..O4/P1..P4/E1) and its own generated ds_config JSON.
# - build_ds_config() renders the DeepSpeed JSON from the test name.
# - For every (test, model) pair, sweeps the micro-batch size {1, 2, 4, 8}
#   and calls pretrain.sh once per combination.
#
# Usage:
#   ./pretrain_experiment.sh                     # full matrix
#   ./pretrain_experiment.sh B0 O2 E1            # only selected tests
#   TESTS="O1 O2" MODELS="qwen3_5" ./pretrain_experiment.sh
#   MICRO_BATCH_SIZES="1 8" ./pretrain_experiment.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# ---------------------------------------------------------------------------
# Experiment matrix knobs (README §2 / §4)
# ---------------------------------------------------------------------------
# All test IDs from README §4. Override via env or positional arguments.
ALL_TESTS="B0 R1 R2 O1 O2 O3 O4 P1 P2 P3 P4 E1"
TESTS=${TESTS:-$ALL_TESTS}
if [ "$#" -gt 0 ]; then
    TESTS="$*"
fi

# Dense = qwen3_5, MoE = qwen3_5_moe (train.py presets, shrunk to 8 layers).
MODELS=${MODELS:-"qwen3_5 qwen3_5_moe"}

# Phase 5 micro-batch sweep axis.
MICRO_BATCH_SIZES=${MICRO_BATCH_SIZES:-"1 2 4 8"}

# Recompute tier carried over from Phase 1 into Phase 2/3 runs (README §4:
# "阶段之间用上一阶段选出的最优配置传递"; expected winner is act).
DEFAULT_RECOMPUTE=${DEFAULT_RECOMPUTE:-act}

# ---------------------------------------------------------------------------
# Output layout
# ---------------------------------------------------------------------------
CONFIG_DIR=${CONFIG_DIR:-"${SCRIPT_DIR}/configs"}
RESULTS_DIR=${RESULTS_DIR:-"${SCRIPT_DIR}/results"}
mkdir -p "$CONFIG_DIR" "$RESULTS_DIR"

# ---------------------------------------------------------------------------
# Common DeepSpeed knobs, fixed for all runs (README §2)
# ---------------------------------------------------------------------------
ZERO_STAGE=${ZERO_STAGE:-3}
OVERLAP_COMM=${OVERLAP_COMM:-false}
REDUCE_BUCKET_SIZE=${REDUCE_BUCKET_SIZE:-4e8}
SUB_GROUP_SIZE=${SUB_GROUP_SIZE:-4e8}
PIN_MEMORY=${PIN_MEMORY:-true}
CPUADAM_CORES_PERC=${CPUADAM_CORES_PERC:-0.90}
WALL_CLOCK_BREAKDOWN=${WALL_CLOCK_BREAKDOWN:-true}
BF16_ENABLED=${BF16_ENABLED:-true}

# ---------------------------------------------------------------------------
# resolve_test <test_name>
#   Sets TEST_PARAM_OFFLOAD (none|cpu), TEST_OPT_OFFLOAD
#   (gpu|zerooffload|superoffload), TEST_RATIO, TEST_RECOMPUTE
#   (none|act|act_cpu) according to the README §4 phase tables.
# ---------------------------------------------------------------------------
resolve_test() {
    local test_name=$1
    TEST_PARAM_OFFLOAD="none"
    TEST_OPT_OFFLOAD="gpu"
    TEST_RATIO=""
    TEST_RECOMPUTE="$DEFAULT_RECOMPUTE"

    case "$test_name" in
        # Phase 0 — baseline: pure ZeRO-3, everything on GPU, no recompute
        B0) TEST_RECOMPUTE="none" ;;
        # Phase 1 — recompute axis (param=GPU, optimizer=GPU)
        R1) TEST_RECOMPUTE="act" ;;
        R2) TEST_RECOMPUTE="act_cpu" ;;
        # Phase 2 — optimizer axis (param=GPU)
        O1) TEST_OPT_OFFLOAD="zerooffload" ;;
        O2) TEST_OPT_OFFLOAD="superoffload"; TEST_RATIO="1.0" ;;
        O3) TEST_OPT_OFFLOAD="superoffload"; TEST_RATIO="0.9" ;;
        O4) TEST_OPT_OFFLOAD="superoffload"; TEST_RATIO="0.75" ;;
        # Phase 3 — parameter axis (ZeRO-Infinity)
        P1) TEST_PARAM_OFFLOAD="cpu"; TEST_OPT_OFFLOAD="gpu" ;;
        P2) TEST_PARAM_OFFLOAD="cpu"; TEST_OPT_OFFLOAD="zerooffload" ;;
        P3) TEST_PARAM_OFFLOAD="cpu"; TEST_OPT_OFFLOAD="superoffload"; TEST_RATIO="1.0" ;;
        P4) TEST_PARAM_OFFLOAD="cpu"; TEST_OPT_OFFLOAD="superoffload"; TEST_RATIO="0.75" ;;
        # Phase 4 — lowest-memory combination
        E1)
            TEST_PARAM_OFFLOAD="cpu"
            TEST_OPT_OFFLOAD="superoffload"
            TEST_RATIO="1.0"
            TEST_RECOMPUTE="act_cpu"
            ;;
        *)
            echo "Unknown test name: $test_name (expected: $ALL_TESTS)" >&2
            exit 2
            ;;
    esac
}

# ---------------------------------------------------------------------------
# build_ds_config <test_name> <output_path>
#   Renders the DeepSpeed JSON for one test.
#
#   NOTE: no "optimizer"/"scheduler" section is emitted on purpose —
#   train.py passes a client optimizer to deepspeed.initialize(), and
#   DeepSpeed rejects configs that specify an optimizer twice. Offload
#   placement is fully expressed via zero_optimization.offload_optimizer /
#   offload_param; batch keys are injected by train.py from its CLI.
# ---------------------------------------------------------------------------
build_ds_config() {
    local test_name=$1
    local out_path=$2
    resolve_test "$test_name"

    local zero_extra=""
    case "$TEST_OPT_OFFLOAD" in
        gpu)
            ;;
        zerooffload)
            zero_extra="\"offload_optimizer\": { \"device\": \"cpu\", \"pin_memory\": $PIN_MEMORY },"
            ;;
        superoffload)
            zero_extra="\"offload_optimizer\": { \"device\": \"cpu\", \"pin_memory\": $PIN_MEMORY, \"super_offload\": true, \"ratio\": $TEST_RATIO, \"cpuadam_cores_perc\": $CPUADAM_CORES_PERC },"
            ;;
        *)
            echo "Unknown optimizer placement: $TEST_OPT_OFFLOAD" >&2
            exit 2
            ;;
    esac
    if [ "$TEST_PARAM_OFFLOAD" = "cpu" ]; then
        zero_extra="${zero_extra}
        \"offload_param\": { \"device\": \"cpu\", \"pin_memory\": $PIN_MEMORY },"
    fi

    cat > "$out_path" << EOF
{
    "bf16": { "enabled": $BF16_ENABLED },
    "wall_clock_breakdown": $WALL_CLOCK_BREAKDOWN,
    "zero_optimization": {
        "stage": $ZERO_STAGE,
        $zero_extra
        "overlap_comm": $OVERLAP_COMM,
        "reduce_bucket_size": $REDUCE_BUCKET_SIZE,
        "sub_group_size": $SUB_GROUP_SIZE
    }
}
EOF
    echo "Built ds_config for $test_name -> $out_path"
    cat "$out_path"
}

# ---------------------------------------------------------------------------
# Main sweep: test x model x micro_batch_size
# ---------------------------------------------------------------------------
SUMMARY_FILE="${RESULTS_DIR}/experiment_summary.txt"
: > "$SUMMARY_FILE"

set +e  # keep sweeping after a failing/OOM run; status is recorded per run
for TEST_NAME in $TESTS; do
    DS_CONFIG="${CONFIG_DIR}/ds_config_${TEST_NAME}.json"
    build_ds_config "$TEST_NAME" "$DS_CONFIG"
    resolve_test "$TEST_NAME"   # re-resolve to read TEST_RECOMPUTE for this test

    for MODEL in $MODELS; do
        for MBS in $MICRO_BATCH_SIZES; do
            RUN_TAG="${TEST_NAME}_${MODEL}_mbs${MBS}"
            METRICS_OUT="${RESULTS_DIR}/${RUN_TAG}_metrics.csv"
            LOG_FILE="${RESULTS_DIR}/${RUN_TAG}.log"

            echo ""
            echo "################ RUN ${RUN_TAG} (recompute=${TEST_RECOMPUTE}) ################"
            bash "${SCRIPT_DIR}/pretrain.sh" \
                "$TEST_NAME" "$MODEL" "$MBS" "$DS_CONFIG" "$TEST_RECOMPUTE" "$METRICS_OUT" \
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
            echo "${RUN_TAG} ${STATUS}" | tee -a "$SUMMARY_FILE"
        done
    done
done
set -e

echo ""
echo "================ SUMMARY ================"
cat "$SUMMARY_FILE"
echo "Configs:  $CONFIG_DIR"
echo "Results:  $RESULTS_DIR"
