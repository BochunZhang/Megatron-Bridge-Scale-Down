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

# Run one FSDP1 experiment on four local GPUs.
#
# Usage:
#   run_pretrain_fsdp1.sh --model <model> --recipe <recipe> --precision <precision> \
#       --run-name <run-name> --recompute-granularity <value> \
#       --recompute-modules <value> --fine-grained-offload <true|false> \
#       --offload-modules <value> \
#       [--train-iters <iters>] [--global-batch-size <gbs>] [--micro-batch-size <mbs>] \
#       [--profile-step-start <start>] [--profile-step-end <end>]

set -euo pipefail

export GLOO_SOCKET_IFNAME=eth0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

usage() {
    cat <<'EOF'
Usage: run_pretrain_fsdp1.sh \
    --model <model> \
    --recipe <recipe> \
    --precision <bf16|fp8mx> \
    --run-name <run-name> \
    --recompute-granularity <null|selective> \
    --recompute-modules <value> \
    --fine-grained-offload <true|false> \
    --offload-modules <value> \
    [--train-iters <iters>] \
    [--global-batch-size <gbs>] \
    [--micro-batch-size <mbs>] \
    [--profile-step-start <start>] \
    [--profile-step-end <end>]

The model, recipe, and precision are selected by the caller and are passed
through without model/precision combination logic. Hydra values such as null,
selective, [layernorm,mlp], or [expert_fc1,moe_act] are accepted.

Optional training parameters (with defaults):
    --train-iters         Number of training iterations (default: 10)
    --global-batch-size   Global batch size (default: 8)
    --micro-batch-size    Micro batch size (default: 1)
    --profile-step-start  Profile start step (default: 7)
    --profile-step-end    Profile end step (default: 8)
EOF
}

MODEL=""
RECIPE=""
PRECISION=""
RUN_NAME=""
RECOMPUTE_GRANULARITY=""
RECOMPUTE_MODULES=""
FINE_GRAINED_OFFLOAD=""
OFFLOAD_MODULES=""

# Training parameters with defaults
TRAIN_ITERS=""
GLOBAL_BATCH_SIZE=""
MICRO_BATCH_SIZE=""
PROFILE_STEP_START=""
PROFILE_STEP_END=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            MODEL="$2"
            shift 2
            ;;
        --recipe)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            RECIPE="$2"
            shift 2
            ;;
        --precision)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            PRECISION="$2"
            shift 2
            ;;
        --run-name)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            RUN_NAME="$2"
            shift 2
            ;;
        --recompute-granularity)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            RECOMPUTE_GRANULARITY="$2"
            shift 2
            ;;
        --recompute-modules)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            RECOMPUTE_MODULES="$2"
            shift 2
            ;;
        --fine-grained-offload)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            FINE_GRAINED_OFFLOAD="$2"
            shift 2
            ;;
        --offload-modules)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            OFFLOAD_MODULES="$2"
            shift 2
            ;;
        --train-iters)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            TRAIN_ITERS="$2"
            shift 2
            ;;
        --global-batch-size)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            GLOBAL_BATCH_SIZE="$2"
            shift 2
            ;;
        --micro-batch-size)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            MICRO_BATCH_SIZE="$2"
            shift 2
            ;;
        --profile-step-start)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            PROFILE_STEP_START="$2"
            shift 2
            ;;
        --profile-step-end)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            PROFILE_STEP_END="$2"
            shift 2
            ;;
        -h|--help)
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

for required in MODEL RECIPE PRECISION RUN_NAME RECOMPUTE_GRANULARITY RECOMPUTE_MODULES FINE_GRAINED_OFFLOAD OFFLOAD_MODULES; do
    if [[ -z "${!required}" ]]; then
        echo "Missing required option for ${required}" >&2
        usage >&2
        exit 2
    fi
done

# Apply defaults for optional training parameters
TRAIN_ITERS="${TRAIN_ITERS:-10}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
PROFILE_STEP_START="${PROFILE_STEP_START:-7}"
PROFILE_STEP_END="${PROFILE_STEP_END:-8}"

MODEL_ID="${MODEL}"
RESULT_MODEL_NAME="${MODEL}"

case "${PRECISION}" in
    bf16|fp8mx) ;;
    *) echo "Unsupported precision: ${PRECISION}" >&2; exit 2 ;;
esac
case "${RECOMPUTE_GRANULARITY}" in
    null|selective) ;;
    *) echo "Unsupported recompute granularity: ${RECOMPUTE_GRANULARITY}" >&2; exit 2 ;;
esac
case "${FINE_GRAINED_OFFLOAD}" in
    true|false) ;;
    *) echo "fine-grained offload must be true or false: ${FINE_GRAINED_OFFLOAD}" >&2; exit 2 ;;
esac

RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/01-analyse/01-fine-grained-offload}"
RUN_TIME="${RUN_TIME:-$(date +%Y%m%d-%H%M%S)}"
RESULT_DIR="${RESULTS_ROOT}/${RESULT_MODEL_NAME}/${PRECISION}/${RUN_NAME}/${RUN_TIME}"
HF_CACHE="${REPO_ROOT}/.cache/huggingface"
NEMO_CACHE="${REPO_ROOT}/.cache/nemo"
UV_CACHE="${REPO_ROOT}/.cache/uv"
MASTER_PORT="${MASTER_PORT:-29501}"

if ! [[ "${TRAIN_ITERS}" =~ ^[0-9]+$ && "${GLOBAL_BATCH_SIZE}" =~ ^[0-9]+$ && "${MICRO_BATCH_SIZE}" =~ ^[0-9]+$ ]]; then
    echo "Training sizes must be non-negative integers" >&2
    exit 2
fi
if (( TRAIN_ITERS < 1 || GLOBAL_BATCH_SIZE < 1 || MICRO_BATCH_SIZE < 1 )); then
    echo "Training sizes must be positive" >&2
    exit 2
fi
if ! [[ "${PROFILE_STEP_START}" =~ ^[0-9]+$ && "${PROFILE_STEP_END}" =~ ^[0-9]+$ ]] || (( PROFILE_STEP_END <= PROFILE_STEP_START || PROFILE_STEP_END > TRAIN_ITERS )); then
    echo "Require PROFILE_STEP_START < PROFILE_STEP_END <= TRAIN_ITERS" >&2
    exit 2
fi

mkdir -p "${RESULT_DIR}/memory" "${RESULT_DIR}/rank_logs" \
    "${HF_CACHE}" "${NEMO_CACHE}/datasets" "${NEMO_CACHE}/models" "${UV_CACHE}"

export HF_HOME="${HF_CACHE}"
export HF_HUB_CACHE="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}"
export NEMO_HOME="${NEMO_CACHE}"
export NEMO_DATASETS_CACHE="${NEMO_CACHE}/datasets"
export NEMO_MODELS_CACHE="${NEMO_CACHE}/models"
export UV_CACHE_DIR="${UV_CACHE}"
export NVTE_CPU_OFFLOAD_V1="1"
export TORCH_NCCL_AVOID_RECORD_STREAMS="1"
export NCCL_NVLS_ENABLE="0"
export RESULT_DIR MODEL MODEL_ID RESULT_MODEL_NAME PRECISION RUN_NAME RUN_TIME RECIPE

OVERRIDES=(
    "model.seq_length=4096"
    "dataset.seq_length=4096"
    "dataset.blend=null"
    "train.train_iters=${TRAIN_ITERS}"
    "train.global_batch_size=${GLOBAL_BATCH_SIZE}"
    "train.micro_batch_size=${MICRO_BATCH_SIZE}"
    "scheduler.lr_warmup_iters=0"
    "model.recompute_granularity=${RECOMPUTE_GRANULARITY}"
    "model.recompute_method=null"
    "model.recompute_num_layers=null"
    "model.recompute_modules=${RECOMPUTE_MODULES}"
    "model.fine_grained_activation_offloading=${FINE_GRAINED_OFFLOAD}"
    "model.offload_modules=${OFFLOAD_MODULES}"
    "model.cuda_graph_impl=none"
    "model.cuda_graph_scope=null"
    "model.cuda_graph_modules=[]"
    "model.moe_shared_expert_overlap=false"
    "checkpoint.save=null"
    "checkpoint.load=null"
    "logger.log_interval=1"
    "logger.tensorboard_dir=null"
    "logger.save_config_filepath=${RESULT_DIR}/config.yaml"
    "profiling.use_nsys_profiler=false"
    "profiling.profile_step_start=${PROFILE_STEP_START}"
    "profiling.profile_step_end=${PROFILE_STEP_END}"
    "profiling.profile_ranks=[0,1,2,3]"
    "profiling.record_memory_history=true"
    "profiling.memory_snapshot_path=${RESULT_DIR}/memory/snapshot.pickle"
    "profiling.nvtx_ranges=true"
)

COMMAND=(
    uv run --no-sync python -m torch.distributed.run
    --standalone
    --nproc_per_node=4
    --master_port="${MASTER_PORT}"
    --log_dir="${RESULT_DIR}/rank_logs"
    --redirects=3
    --tee=3
    scripts/training/run_recipe.py
    --recipe "${RECIPE}"
    --step-func llm_step
    "${OVERRIDES[@]}"
)
COMMAND_TEXT="${COMMAND[*]}"
export COMMAND_TEXT

uv run --no-sync python -c 'import json, os, re; pattern = re.compile(r"(^|_)(TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|ACCESS_KEY|SECRET_KEY|PRIVATE_KEY|AUTHORIZATION)(_|$)", re.I); root = os.environ["RESULT_DIR"]; env = {k: ("[REDACTED]" if pattern.search(k) else v) for k, v in sorted(os.environ.items())}; json.dump(env, open(os.path.join(root, "environment.json"), "w"), indent=2, sort_keys=True); open(os.path.join(root, "command.txt"), "w").write(os.environ["COMMAND_TEXT"] + "\n"); config = {"model": os.environ["MODEL"], "model_id": os.environ["MODEL_ID"], "precision": os.environ["PRECISION"], "run_name": os.environ["RUN_NAME"], "run_time": os.environ["RUN_TIME"], "recipe": os.environ["RECIPE"], "result_dir": root, "profile_ranks": [0, 1, 2, 3], "cache_paths": {"hf": env["HF_HOME"], "nemo": env["NEMO_HOME"]}, "cli": os.environ["COMMAND_TEXT"]}; json.dump(config, open(os.path.join(root, "config.json"), "w"), indent=2, sort_keys=True)'

printf 'model=%s precision=%s run_name=%s run_time=%s\n' "${MODEL}" "${PRECISION}" "${RUN_NAME}" "${RUN_TIME}" | tee "${RESULT_DIR}/run_info.txt"
printf 'train_iters=%s global_batch_size=%s micro_batch_size=%s\n' "${TRAIN_ITERS}" "${GLOBAL_BATCH_SIZE}" "${MICRO_BATCH_SIZE}" | tee -a "${RESULT_DIR}/run_info.txt"
printf 'profile_step_start=%s profile_step_end=%s\n' "${PROFILE_STEP_START}" "${PROFILE_STEP_END}" | tee -a "${RESULT_DIR}/run_info.txt"
printf 'result_dir=%s\nprofile_ranks=0,1,2,3\ncommand=%s\n' "${RESULT_DIR}" "${COMMAND_TEXT}" | tee -a "${RESULT_DIR}/run_info.txt"

set +e
"${COMMAND[@]}" 2>&1 | tee "${RESULT_DIR}/train.log"
RUN_STATUS=${PIPESTATUS[0]}
set -e

export RUN_STATUS
uv run --no-sync python -c 'import json, os; root = os.environ["RESULT_DIR"]; result = {"status": int(os.environ["RUN_STATUS"]), "model": os.environ["MODEL"], "precision": os.environ["PRECISION"], "run_name": os.environ["RUN_NAME"], "run_time": os.environ["RUN_TIME"], "recipe": os.environ["RECIPE"], "result_dir": root, "profile_ranks": [0, 1, 2, 3]}; json.dump(result, open(os.path.join(root, "summary.json"), "w"), indent=2, sort_keys=True); raise SystemExit(result["status"])'
