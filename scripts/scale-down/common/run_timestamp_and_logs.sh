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
# One batch id and one log location for every node of a DLC job.
#
# Sourcing this file only defines functions; run it to see what this node
# resolves.
#
#   source scripts/scale-down/common/gloo_socket_ifname.sh
#   source scripts/scale-down/common/run_timestamp_and_logs.sh
#   resolve_gloo_socket_ifname "${NNODES}"   # before the TCPStore, it needs Gloo
#   setup_log_dir logs/01-analyse/01-fine-grained-offload
#   exec > >(tee -a "${RUN_LOG}") 2>&1
#
# Why a master timestamp at all: each node formats its own `date`, so a job that
# takes two minutes to fan out would otherwise scatter one logical run across
# several timestamps and several result directories. Node 0 stamps the id once
# and serves it from a TCPStore on MASTER_ADDR:MASTER_PORT; every other node
# blocks until it reads that value, and node 0 does not return until all of them
# have acknowledged.
#
# Requires: MASTER_ADDR, MASTER_PORT and NNODES for a multi-node run; those come
# from the DLC environment.
#
# Node RANK number and interface notes:
#   - The clock sync is CPU only. It must run before any CUDA context is created
#     (before the training launch), and torch here is not the training process.
#   - RANK comes from DLC. It is ambiguous on Knative because DLC also sets RANK
#     to the worker rank, so if a job mixes both, RANK must be translated before
#     this is called.

# log/warn/die are shared with the caller; define only what is missing.
declare -F log >/dev/null || log() { printf '%s\n' "$*"; }
declare -F warn >/dev/null || warn() { printf 'warning: %s\n' "$*" >&2; }
declare -F die >/dev/null || die() { printf 'error: %s\n' "$*" >&2; exit 2; }

# Repo root, derived from this file's location (scripts/scale-down/common).
SCALE_DOWN_COMMON_DIR="${SCALE_DOWN_COMMON_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCALE_DOWN_COMMON_DIR}/../../.." && pwd)}"
export REPO_ROOT

# Node-local log root for the ranks that must not write into the shared tree.
# Override with LOG_DIR to place per-node logs somewhere else.
LOG_DIR="${LOG_DIR:-/tmp/megatron-bridge/logs}"
RUN_TIME_SYNC_TIMEOUT_MINUTES="${RUN_TIME_SYNC_TIMEOUT_MINUTES:-30}"

# resolve_run_time [DRY_RUN]
#
# Prints the batch id and sets RUN_TIME. An explicit RUN_TIME (or one exported by
# the caller) always wins, which is what a re-run or a pinned comparison needs.
#
# Single node: the local clock is the only clock, so stamp it directly.
# Multi-node: node 0 stamps and publishes, the other nodes read back the same
# value. Nothing to agree on in a dry run either, so it stamps locally.
resolve_run_time() {
    local dry_run="${1:-false}"
    if [[ -n "${RUN_TIME:-}" ]]; then
        printf '%s\n' "${RUN_TIME}"
        return 0
    fi
    local nodes="${NNODES:-1}"
    local rank="${NODE_RANK:-0}"
    if (( nodes == 1 )) || [[ "${dry_run}" == true ]]; then
        RUN_TIME="$(date +%Y%m%d-%H%M%S)"
        export RUN_TIME
        log "run time: ${RUN_TIME} (local clock, nodes=${nodes}${dry_run:+ dry_run=${dry_run}})"
        printf '%s\n' "${RUN_TIME}"
        return 0
    fi
    [[ -n "${MASTER_ADDR:-}" ]] || die "MASTER_ADDR is required to agree on a run id across ${nodes} nodes"
    [[ -n "${MASTER_PORT:-}" ]] || die "MASTER_PORT is required to agree on a run id across ${nodes} nodes"
    # The TCPStore bootstraps through Gloo, so an interface it cannot resolve
    # would hang here instead of failing with a clear message.
    [[ -n "${GLOO_SOCKET_IFNAME:-}" ]] \
        || die "GLOO_SOCKET_IFNAME is required for the run-id handshake; \
call resolve_gloo_socket_ifname ${nodes} first"
    RUN_TIME="$(
        SYNC_HOST="${MASTER_ADDR}" SYNC_PORT="${MASTER_PORT}" \
        SYNC_NNODES="${nodes}" SYNC_RANK="${rank}" \
        SYNC_TIMEOUT_MIN="${RUN_TIME_SYNC_TIMEOUT_MINUTES}" \
        uv run --no-sync python -c '
import datetime
import os
import sys
import time

import torch.distributed as dist

nnodes = int(os.environ["SYNC_NNODES"])
rank = int(os.environ["SYNC_RANK"])
timeout = datetime.timedelta(minutes=int(os.environ["SYNC_TIMEOUT_MIN"]))
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
    )" || die "could not agree on RUN_TIME with node 0 (${MASTER_ADDR}:${MASTER_PORT})"
    [[ -n "${RUN_TIME}" ]] || die "node 0 published an empty RUN_TIME"
    export RUN_TIME
    log "run time: ${RUN_TIME} (from node 0 via ${MASTER_ADDR}:${MASTER_PORT})"
    printf '%s\n' "${RUN_TIME}"
}

# setup_log_dir <relative-path> [DRY_RUN]
#
# Creates this node's log directory under the shared result tree and exports
# RUN_LOG_DIR and RUN_LOG:
#
#   <RESULTS_ROOT>/<relative-path>/<RUN_TIME>/           node 0
#   <RESULTS_ROOT>/<relative-path>/<RUN_TIME>/rank<N>/   every other node
#
# Non-zero ranks get their own subdirectory because a shared job log written by
# several nodes at once is unreadable. When the shared tree is not reachable
# from this node — or LOG_LOCAL=1 is set — non-zero ranks fall back to
# <LOG_DIR>/<relative-path>/<RUN_TIME>/rank<N>/ on node-local disk, and the path
# is printed so it can be collected once the job is done.
#
# Requires: resolve_run_time first. Exports RUN_TIME, RUN_LOG_DIR, RUN_LOG,
# RESULTS_ROOT.
setup_log_dir() {
    local relative_path="${1:-}"
    local dry_run="${2:-false}"
    [[ -n "${relative_path}" ]] || die "setup_log_dir needs a path relative to RESULTS_ROOT"
    case "${relative_path}" in
        /*) die "setup_log_dir path must be relative to RESULTS_ROOT: ${relative_path}" ;;
    esac
    [[ -n "${RUN_TIME:-}" ]] || die "call resolve_run_time before setup_log_dir"
    [[ "${relative_path}" != *".."* ]] || die "setup_log_dir path must not contain '..': ${relative_path}"

    export RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results}"
    local nodes="${NNODES:-1}"
    local rank="${NODE_RANK:-0}"
    [[ "${rank}" =~ ^[0-9]+$ ]] || die "RANK/NODE_RANK must be the DLC node rank: ${rank}"
    (( rank < nodes )) || die "node rank ${rank} must be < ${nodes} nodes"

    local local_base="${LOG_DIR}/${relative_path}/${RUN_TIME}"
    local base="${RESULTS_ROOT}/${relative_path}/${RUN_TIME}"
    local dir="${base}"
    (( rank != 0 )) && dir="${base}/rank${rank}"
    if (( rank != 0 )) && { [[ "${LOG_LOCAL:-0}" == "1" ]] || ! mkdir -p "${dir}" 2>/dev/null; }; then
        dir="${local_base}/rank${rank}"
        warn "shared log dir is not writable from node ${rank}; using node-local ${dir}"
    fi
    mkdir -p "${dir}" || die "could not create the log directory ${dir}"

    if [[ "${dry_run}" != true ]]; then
        local node_json="${dir}/node.json"
        if ! printf '{"run_time": "%s", "node_rank": %s, "nnodes": %s, "host": "%s", "master_addr": "%s", "gloo_ifname": "%s", "log_dir": "%s"}\n' \
            "${RUN_TIME}" "${rank}" "${nodes}" "$(hostname 2>/dev/null || echo unknown)" \
            "${MASTER_ADDR:-localhost}" "${GLOO_SOCKET_IFNAME:-unset}" "${dir}" \
            > "${node_json}" 2>/dev/null; then
            warn "could not write ${node_json}"
        fi
    fi
    export RUN_TIME
    export RUN_LOG_DIR="${dir}"
    export RUN_LOG="${RUN_LOG_DIR}/job.log"
    log "run id: ${RUN_TIME} node_rank=${rank}/${nodes} log=${RUN_LOG}"
}

# shellcheck disable=SC2317  # reachable when this file is executed, not sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -euo pipefail
    resolve_run_time "${DRY_RUN:-false}"
    setup_log_dir "${RUN_LOG_RELATIVE_PATH:-logs/scale-down}" "${DRY_RUN:-false}" >/dev/null
    printf 'export RUN_TIME=%s\nexport RUN_LOG_DIR=%s\nexport RUN_LOG=%s\n' \
        "${RUN_TIME}" "${RUN_LOG_DIR}" "${RUN_LOG}"
fi
