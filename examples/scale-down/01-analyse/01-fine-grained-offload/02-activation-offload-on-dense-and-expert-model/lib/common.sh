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
# Shared helpers for the selective activation-offload experiments.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../.." && pwd)"
LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "${LIB_DIR}/.." && pwd)"
GPU_MONITOR="${REPO_ROOT}/scripts/scale-down/nvidia-smi/check_gpu.py"
GPU_UTIL_EXPORTER="${REPO_ROOT}/scripts/scale-down/analyse/export_gpu_utilization.py"
DEFAULT_RESULTS_ROOT="${REPO_ROOT}/results/01-analyse/01-fine-grained-offload/02-activation-offload-on-dense-and-expert-model"
GPUS_PER_NODE=4

log() {
    printf '%s\n' "$*"
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

# ---------------------------------------------------------------------------
# Distributed environment
# ---------------------------------------------------------------------------

# Resolve the node layout for a total GPU count. DLC injects MASTER_ADDR,
# MASTER_PORT, WORLD_SIZE (number of nodes) and RANK (node rank) into every
# worker, so nothing has to be exported by hand.
#
# Sets: NNODES NODE_RANK MASTER_ADDR MASTER_PORT
resolve_dist_env() {
    local gpu_count="$1"
    local dry_run="$2"
    (( gpu_count % GPUS_PER_NODE == 0 )) || die "--gpu must be a multiple of ${GPUS_PER_NODE}: ${gpu_count}"
    NNODES=$(( gpu_count / GPUS_PER_NODE ))
    MASTER_PORT="${MASTER_PORT:-29500}"
    if (( NNODES == 1 )); then
        NODE_RANK=0
        MASTER_ADDR="${MASTER_ADDR:-localhost}"
        return
    fi
    NODE_RANK="${NODE_RANK:-${RANK:-0}}"
    MASTER_ADDR="${MASTER_ADDR:-}"
    [[ "${NODE_RANK}" =~ ^[0-9]+$ ]] || die "RANK must be the DLC node rank: ${NODE_RANK}"
    (( NODE_RANK < NNODES )) || die "RANK ${NODE_RANK} must be < ${NNODES} nodes for --gpu ${gpu_count}"
    if [[ -n "${WORLD_SIZE:-}" && "${WORLD_SIZE}" != "${NNODES}" ]]; then
        die "DLC WORLD_SIZE=${WORLD_SIZE} nodes, but --gpu ${gpu_count} needs ${NNODES} nodes"
    fi
    if [[ "${dry_run}" != true && -z "${MASTER_ADDR}" ]]; then
        die "MASTER_ADDR is empty; DLC sets it for multi-node jobs"
    fi
}

# Fix Gloo's bootstrap interface. A stale GLOO_SOCKET_IFNAME inherited from the
# environment (for example bond1 on a single-node DSW) aborts process-group
# creation with "Unable to find address for: <ifname>", because Gloo resolves
# the name against the local interfaces. Select from `ip addr` instead: a single
# 4-GPU node uses eth0, multi-node DLC uses bond1. An exported GLOO_SOCKET_IFNAME
# is honored only when the interface actually carries an IPv4 address here; an
# unusable value is replaced instead of being passed through to Gloo.
#
# Requires NNODES. Exports GLOO_SOCKET_IFNAME.
resolve_gloo_ifname() {
    local preferred=eth0 fallback=bond1 available
    if (( NNODES > 1 )); then
        preferred=bond1
        fallback=eth0
    fi
    if ! command -v ip >/dev/null; then
        die "the ip command is required to select GLOO_SOCKET_IFNAME; export it explicitly"
    fi
    available="$(ip -o -4 addr show up | awk '{print $2}' | sort -u | tr '\n' ' ')"
    if [[ -n "${GLOO_SOCKET_IFNAME:-}" ]]; then
        if [[ " ${available} " == *" ${GLOO_SOCKET_IFNAME} "* ]]; then
            log "gloo interface: ${GLOO_SOCKET_IFNAME} (from the environment)"
            return
        fi
        log "warning: GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME} has no IPv4 address; overriding it"
    fi
    if [[ " ${available} " == *" ${preferred} "* ]]; then
        export GLOO_SOCKET_IFNAME="${preferred}"
    elif [[ " ${available} " == *" ${fallback} "* ]]; then
        log "warning: ${preferred} has no IPv4 address; using ${fallback} (up: ${available})"
        export GLOO_SOCKET_IFNAME="${fallback}"
    else
        die "neither ${preferred} nor ${fallback} is up with an IPv4 address (up: ${available:-none})"
    fi
    log "gloo interface: ${GLOO_SOCKET_IFNAME}"
}

# Agree on one batch id (RUN_TIME) across all nodes so every node writes into
# the master's result directory. Node 0 generates the id and serves it from a
# TCPStore on MASTER_ADDR:MASTER_PORT; the other nodes block until they read
# it, and node 0 waits until every node has acknowledged before returning.
#
# Requires resolve_dist_env. Sets RUN_TIME when it is empty.
sync_run_time() {
    local dry_run="$1"
    if [[ -n "${RUN_TIME:-}" ]]; then
        return
    fi
    if (( NNODES == 1 )) || [[ "${dry_run}" == true ]]; then
        RUN_TIME="$(date +%Y%m%d-%H%M%S)"
        return
    fi
    RUN_TIME="$(
        SYNC_HOST="${MASTER_ADDR}" SYNC_PORT="${MASTER_PORT}" \
        SYNC_NNODES="${NNODES}" SYNC_RANK="${NODE_RANK}" \
        uv run --no-sync python -c '
import datetime
import os
import sys
import time

import torch.distributed as dist

nnodes = int(os.environ["SYNC_NNODES"])
rank = int(os.environ["SYNC_RANK"])
timeout = datetime.timedelta(minutes=30)
store = dist.TCPStore(
    os.environ["SYNC_HOST"],
    int(os.environ["SYNC_PORT"]),
    world_size=nnodes,
    is_master=rank == 0,
    timeout=timeout,
    wait_for_workers=False,
)
if rank == 0:
    run_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    store.set("run_time", run_time)
    deadline = time.monotonic() + timeout.total_seconds()
    while store.add("acks", 0) < nnodes - 1:
        if time.monotonic() > deadline:
            sys.exit("timed out waiting for every node to read the run id")
        time.sleep(0.2)
else:
    run_time = store.get("run_time").decode()
    store.add("acks", 1)
print(run_time)
'
    )" || die "could not agree on RUN_TIME with the master node"
    [[ -n "${RUN_TIME}" ]] || die "received an empty RUN_TIME from the master node"
}

# Stage the HF config and tokenizer files on node-local disk. With every rank
# reading tokenizer.json from the shared CPFS HF_HOME, a concurrent rewrite of
# the cache blob fails json.load with "[Errno 116] Stale file handle". Each node
# downloads the repos once (weights excluded) before the matrix starts, and
# run_pretrain.sh then trains with HF_HOME=LOCAL_HF_HOME and HF_HUB_OFFLINE=1.
#
# Requires DRY_RUN. Exports LOCAL_HF_HOME (default /tmp/megatron-bridge-hf-home).
stage_local_hf_home() {
    export LOCAL_HF_HOME="${LOCAL_HF_HOME:-/tmp/megatron-bridge-hf-home}"
    log "local hf home: ${LOCAL_HF_HOME} repos=[$*]"
    if [[ "${DRY_RUN}" == true ]]; then
        return
    fi
    mkdir -p "${LOCAL_HF_HOME}"
    HF_HOME="${LOCAL_HF_HOME}" HF_HUB_OFFLINE=0 uv run --no-sync python -c '
import sys
import time

from huggingface_hub import snapshot_download

for repo in sys.argv[1:]:
    for attempt in range(1, 4):
        try:
            snapshot_download(repo, ignore_patterns=["*.safetensors", "*.bin", "*.pt", "*.pth", "*.gguf"])
            break
        except Exception as exc:
            if attempt == 3:
                sys.exit(f"could not stage {repo}: {exc}")
            print(f"retrying {repo} after attempt {attempt} failed: {exc}", flush=True)
            time.sleep(10 * attempt)
' "$@" || die "could not stage the HF config and tokenizer files in ${LOCAL_HF_HOME}"
}

# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

# Select the offload module list for one case and model shape.
#
# `offload-mlp`      offloads only the MLP/MoE activation path.
# `offload-attn-mlp` adds the attention path; GatedDeltaNet reuses the
#                    attention group names (qkv_linear, core_attn, attn_proj).
offload_modules() {
    local case_name="$1"
    local scope="$2"   # dense | expert
    local family="$3"  # qwen | deepseek
    local modules
    if [[ "${case_name}" == baseline ]]; then
        printf 'null\n'
        return
    fi
    if [[ "${scope}" == expert ]]; then
        modules="mlp_norm,expert_fc1,moe_act"
    else
        modules="mlp_norm,mlp_act"
    fi
    if [[ "${case_name}" == offload-attn-mlp ]]; then
        # DeepSeek uses the standard attention projection; Qwen3.5 does not add
        # qkv_linear so the GatedDeltaNet in_proj input stays on the GPU.
        if [[ "${family}" == deepseek ]]; then
            modules="${modules},attn_norm,qkv_linear,core_attn,attn_proj"
        else
            modules="${modules},attn_norm,core_attn,attn_proj"
        fi
    fi
    printf '[%s]\n' "${modules}"
}

# ---------------------------------------------------------------------------
# Matrix wrappers
# ---------------------------------------------------------------------------

COMMON_USAGE='Common options:
    --mbs <list>              Comma-separated micro batch sizes, e.g. "1" or "1,2".
    --model-type <list>       Comma-separated subset of dense,expert.
    --case <list>             Comma-separated subset of baseline,offload-mlp,offload-attn-mlp.
    --dtype <bf16|mxfp8>      Training precision (default: mxfp8).
    --gbs <n>                 Global batch size; must divide by MBS * (GPUs / PP).
    --train-iters <n>         Iterations per run (default: 10).
    --sync-with-sm <bool>     Replace the HybridEP .item() with an SM copy on expert runs.
    --profile <none|nsys|torch>
    --results-root <path>     Result tree root (must be shared storage on DLC).
    --run-time <id>           Batch id; by default node 0 generates and shares it.
    --dry-run                 Print the matrix and resolved runs without training.
    -h, --help                Show this help.

Use --mbs, --model-type and --case to split one matrix across several DLC jobs.'

# Parse the options shared by every wrapper script. The caller sets defaults
# for MBS_LIST, MODEL_TYPES, CASES, GBS, SYNC_WITH_SM and PROFILE first, and may
# preset DTYPE to change the mxfp8 default.
parse_matrix_args() {
    DTYPE="${DTYPE:-mxfp8}"
    MODEL_FAMILY="${MODEL_FAMILY:-qwen}"
    TRAIN_ITERS="${TRAIN_ITERS:-10}"
    RESULTS_ROOT="${RESULTS_ROOT:-${DEFAULT_RESULTS_ROOT}}"
    RUN_TIME="${RUN_TIME:-}"
    DRY_RUN=false
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --mbs) MBS_LIST="$2"; shift 2 ;;
            --model-type) MODEL_TYPES="$2"; shift 2 ;;
            --model-family) MODEL_FAMILY="$2"; shift 2 ;;
            --case) CASES="$2"; shift 2 ;;
            --dtype) DTYPE="$2"; shift 2 ;;
            --gbs) GBS="$2"; shift 2 ;;
            --train-iters) TRAIN_ITERS="$2"; shift 2 ;;
            --sync-with-sm) SYNC_WITH_SM="$2"; shift 2 ;;
            --profile) PROFILE="$2"; shift 2 ;;
            --results-root) RESULTS_ROOT="$2"; shift 2 ;;
            --run-time) RUN_TIME="$2"; shift 2 ;;
            --dry-run) DRY_RUN=true; shift ;;
            -h|--help) usage; exit 0 ;;
            *) die "unknown argument: $1" ;;
        esac
    done
    case "${DTYPE}" in bf16|mxfp8) ;; *) die "--dtype must be bf16 or mxfp8" ;; esac
    case "${MODEL_FAMILY}" in qwen|deepseek) ;; *) die "--model-family must be qwen or deepseek" ;; esac
    case "${SYNC_WITH_SM}" in true|false) ;; *) die "--sync-with-sm must be true or false" ;; esac
    case "${PROFILE}" in none|nsys|torch) ;; *) die "--profile must be none, nsys or torch" ;; esac
    IFS=',' read -r -a MBS_VALUES <<< "${MBS_LIST}"
    IFS=',' read -r -a MODEL_TYPE_VALUES <<< "${MODEL_TYPES}"
    IFS=',' read -r -a CASE_VALUES <<< "${CASES}"
    (( ${#MBS_VALUES[@]} > 0 )) || die "--mbs needs at least one value"
    local value
    for value in "${MBS_VALUES[@]}"; do
        [[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "invalid micro batch size: ${value}"
    done
    for value in "${MODEL_TYPE_VALUES[@]}"; do
        case "${value}" in dense|expert) ;; *) die "--model-type values must be dense or expert: ${value}" ;; esac
    done
    for value in "${CASE_VALUES[@]}"; do
        case "${value}" in
            baseline|offload-mlp|offload-attn-mlp) ;;
            *) die "--case values must be baseline, offload-mlp or offload-attn-mlp: ${value}" ;;
        esac
    done
}

# Resolve the node layout and the shared batch id once per wrapper run.
# The experiment name is the calling script's name, e.g.
# 4gpu_expert_activation_offload_blocking; results land under
# <results-root>/<experiment>/.
setup_matrix() {
    local gpu_count="$1"
    EXPERIMENT="$(basename "${0%.sh}")"
    EXPERIMENT="${EXPERIMENT#run_}"
    resolve_dist_env "${gpu_count}" "${DRY_RUN}"
    sync_run_time "${DRY_RUN}"
    # Each case gets its own rendezvous port; the batch-id store keeps MASTER_PORT.
    CASE_PORT="${MASTER_PORT}"
    FAILURES=0
    log "benchmark_id=${RUN_TIME} gpu=${gpu_count} nnodes=${NNODES} node_rank=${NODE_RANK} family=${MODEL_FAMILY} dtype=${DTYPE}"
    log "experiment=${EXPERIMENT} results=${RESULTS_ROOT}/${EXPERIMENT}"
}

# run_case <gpu> <pp> <ep> <dispatcher> <model> <model-type> <run-name> <offload-modules>
#          <mbs> <sync-with-sm> [extra run_pretrain.sh args...]
run_case() {
    local gpu="$1" pp="$2" ep="$3" dispatcher="$4" model="$5" model_type="$6" run_name="$7"
    local modules="$8" mbs="$9" sync_with_sm="${10}"
    shift 10
    local -a extra=("$@")
    local offload=true
    [[ "${modules}" == null ]] && offload=false
    local -a profile_args=()
    [[ "${PROFILE}" != none ]] && profile_args=(--profile "${PROFILE}")
    CASE_PORT=$(( CASE_PORT + 1 ))

    log "matrix model=${model} type=${model_type} gpu=${gpu} pp=${pp} ep=${ep} dp=$(( gpu / pp )) mbs=${mbs} gbs=${GBS} \
dispatcher=${dispatcher} run=${run_name} offload=${offload} modules=${modules} sync_sm=${sync_with_sm} \
profile=${PROFILE} extra=[${extra[*]:-}]"
    local -a dry_args=()
    [[ "${DRY_RUN}" == true ]] && dry_args=(--dry-run)
    set +e
    "${EXPERIMENT_DIR}/run_pretrain.sh" \
        ${profile_args[@]+"${profile_args[@]}"} ${dry_args[@]+"${dry_args[@]}"} \
        --experiment "${EXPERIMENT}" \
        --model "${model}" --model-type "${model_type}" --dtype "${DTYPE}" \
        --run-name "${run_name}" --offload "${offload}" --offload-modules "${modules}" \
        --gpu "${gpu}" --pp "${pp}" --ep "${ep}" \
        --dispatcher "${dispatcher}" --sync-with-sm "${sync_with_sm}" \
        --mbs "${mbs}" --gbs "${GBS}" --train-iters "${TRAIN_ITERS}" \
        --results-root "${RESULTS_ROOT}" --run-time "${RUN_TIME}" \
        --master-port "${CASE_PORT}" \
        ${extra[@]+"${extra[@]}"}
    local status=$?
    set -e
    if (( status != 0 )); then
        log "run failed with status ${status}: model=${model} run=${run_name} mbs=${mbs}"
        FAILURES=$(( FAILURES + 1 ))
    fi
}

finish_matrix() {
    if [[ "${DRY_RUN}" == true ]]; then
        return 0
    fi
    if (( FAILURES > 0 )); then
        log "${FAILURES} run(s) failed; inspect the result directories before analysing."
        return 1
    fi
    log "raw results: ${RESULTS_ROOT}/${EXPERIMENT} (run_time=${RUN_TIME})"
    log "analyse with: uv run --no-sync python ${EXPERIMENT_DIR}/analyse_offload_results.py --experiment ${EXPERIMENT}"
}

# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------

start_gpu_monitor() {
    local output_dir="$1"
    local ready_file="$2"
    local interval="$3"
    rm -f "${ready_file}"
    uv run --no-sync python "${GPU_MONITOR}" \
        --interval "${interval}" \
        --ready-file "${ready_file}" \
        --output-dir "${output_dir}" &
    GPU_MONITOR_PID=$!
    local waited=0
    while [[ ! -e "${ready_file}" ]]; do
        if ! kill -0 "${GPU_MONITOR_PID}" 2>/dev/null; then
            wait "${GPU_MONITOR_PID}" || true
            GPU_MONITOR_PID=""
            GPU_MONITOR_ERROR="gpu monitor exited before its first sample"
            return 1
        fi
        sleep 0.05
        waited=$((waited + 1))
        if (( waited > 600 )); then
            GPU_MONITOR_ERROR="gpu monitor did not produce a sample within 30s"
            return 1
        fi
    done
    return 0
}

stop_gpu_monitor() {
    local status=0
    if [[ -n "${GPU_MONITOR_PID:-}" ]]; then
        kill -TERM "${GPU_MONITOR_PID}" 2>/dev/null || true
        wait "${GPU_MONITOR_PID}" || status=$?
        GPU_MONITOR_PID=""
    fi
    rm -f "${GPU_MONITOR_READY_FILE:-}"
    return "${status}"
}

export_gpu_utilization() {
    local log_file="$1"
    local output="$2"
    uv run --no-sync python "${GPU_UTIL_EXPORTER}" --log-file "${log_file}" --output "${output}"
}
