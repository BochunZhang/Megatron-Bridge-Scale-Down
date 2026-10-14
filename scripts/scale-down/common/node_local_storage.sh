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
# Runtime storage for a scale-down run: shared result tree for artifacts,
# node-local disk for the caches rank 0 does not need to publish.
#
# Sourcing this file only defines functions. Run it to print the resolved paths.
#
#   source scripts/scale-down/common/node_local_storage.sh
#   setup_runtime_storage                       # caches under /tmp/megatron-bridge
#   stage_hf_assets Qwen/Qwen3.5-9B-Base        # configs + tokenizer on this node
#   cd "${REPO_ROOT}" && uv run --no-sync python train.py
#
# The Dockerfile installs the venv at a fixed path and the editable install of
# 3rdparty/Megatron-LM resolves through the shared repo path, so the uv cache
# and VIRTUAL_ENV stay on shared storage — only the caches below can be moved
# without breaking the launch.
#
# Why node-local for the HF library: with every rank reading tokenizer.json
# from the shared HF_HOME, a concurrent rewrite of the cache blob fails
# json.load with "[Errno 116] Stale file handle". Each node stages the repo
# once (weights excluded) and then trains offline against its own copy.
#
# Set HF_CACHE_ROOT to a shared path to opt out and keep the historical
# shared-cache behavior, at the cost of that failure mode returning.

# log/die are shared with the caller; define only what is missing.
declare -F log >/dev/null || log() { printf '%s\n' "$*"; }
declare -F die >/dev/null || die() { printf 'error: %s\n' "$*" >&2; exit 2; }

# Repo root, derived from this file's location (scripts/scale-down/common).
SCALE_DOWN_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCALE_DOWN_COMMON_DIR}/../../.." && pwd)}"
export REPO_ROOT

# Node-local cache root. /tmp is mounted from a local NVMe device on GB200 DLC
# workers, so this does not traverse the fabric.
NODE_LOCAL_CACHE_ROOT="${NODE_LOCAL_CACHE_ROOT:-/tmp/megatron-bridge}"

# setup_runtime_storage
#
# Creates and exports every cache path a training run needs, and refuses to
# launch against a broken node-local mount before training starts rather than
# failing 200 lines into a NCCL timeout.
#
# Node-local (free to delete, invisible to the other nodes):
#   HF_HOME, HF_HUB_CACHE, TRANSFORMERS_CACHE   configs and tokenizer
#   NEMO_HOME, NEMO_DATASETS_CACHE, NEMO_MODELS_CACHE
#   TRITON_CACHE_DIR                            kernel build output, write-heavy
#
# Shared (must be identical on every node):
#   RESULTS_ROOT    default ${REPO_ROOT}/results
#   UV_CACHE_DIR    default ${REPO_ROOT}/.cache/uv
setup_runtime_storage() {
    local hf_cache="${HF_CACHE_ROOT:-${HF_HOME:-${NODE_LOCAL_CACHE_ROOT}/huggingface}}"
    local nemo_cache="${NEMO_HOME:-${NODE_LOCAL_CACHE_ROOT}/nemo}"
    local triton_cache="${TRITON_CACHE_DIR:-${NODE_LOCAL_CACHE_ROOT}/triton}"
    local uv_cache="${UV_CACHE_DIR:-${REPO_ROOT}/.cache/uv}"

    # /tmp exists but the NVMe mount behind it may not; the easy-to-miss case is
    # a /tmp that is quietly backed by the shared filesystem, which reintroduces
    # exactly the contention this function is here to avoid.
    [[ -d "${NODE_LOCAL_CACHE_ROOT}" || -d "$(dirname "${NODE_LOCAL_CACHE_ROOT}")" ]] \
        || die "node-local cache root ${NODE_LOCAL_CACHE_ROOT} is not mounted"
    local probe="${NODE_LOCAL_CACHE_ROOT}/.writable-$$"
    if ! mkdir -p "${probe}" 2>/dev/null; then
        die "node-local cache root ${NODE_LOCAL_CACHE_ROOT} is not writable"
    fi
    rmdir "${probe}"

    mkdir -p "${hf_cache}" "${nemo_cache}/datasets" "${nemo_cache}/models" \
        "${triton_cache}" "${uv_cache}" || die "could not create the cache directories"

    export HF_HOME="${hf_cache}"
    export HF_HUB_CACHE="${hf_cache}"
    export TRANSFORMERS_CACHE="${hf_cache}"
    export NEMO_HOME="${nemo_cache}"
    export NEMO_DATASETS_CACHE="${nemo_cache}/datasets"
    export NEMO_MODELS_CACHE="${nemo_cache}/models"
    export TRITON_CACHE_DIR="${triton_cache}"
    export UV_CACHE_DIR="${uv_cache}"
    export RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results}"

    log "runtime storage: hf=${HF_HOME} nemo=${NEMO_HOME} triton=${TRITON_CACHE_DIR} \
uv=${UV_CACHE_DIR} results=${RESULTS_ROOT}"
}

# stage_hf_assets <repo> [repo...]
#
# Downloads each repo's config and tokenizer files into this node's HF_HOME and
# leaves the weights alone: only the library files are read on the training
# path, and the checkpoint is loaded from its own path. The staging is
# idempotent — a repo whose tokenizer is already present is skipped — and
# retried, since a transient hub error must not fail an eight-node job.
#
# After this call the run should set HF_HUB_OFFLINE=1 so a missing file fails
# loudly on the node instead of reaching the network mid-step.
#
# Requires setup_runtime_storage. Exports HF_HUB_OFFLINE=1.
stage_hf_assets() {
    (( $# > 0 )) || die "stage_hf_assets needs at least one Hugging Face repo id"
    [[ -n "${HF_HOME:-}" ]] || die "call setup_runtime_storage before stage_hf_assets"
    log "staging hf assets into ${HF_HOME}: $*"
    HF_HOME="${HF_HOME}" HF_HUB_OFFLINE=0 uv run --no-sync python - "$@" <<'PY' \
        || die "could not stage the Hugging Face config and tokenizer files in ${HF_HOME}"
import sys
import time

from huggingface_hub import snapshot_download

# Only the library files: weights load from their own checkpoint path.
IGNORE = ["*.safetensors", "*.bin", "*.pt", "*.pth", "*.gguf", "*.h5", "*.msgpack"]

for repo in sys.argv[1:]:
    for attempt in range(1, 4):
        try:
            path = snapshot_download(repo, ignore_patterns=IGNORE)
            print(f"staged {repo} -> {path}", flush=True)
            break
        except Exception as exc:  # noqa: BLE001 - the hub raises several types
            if attempt == 3:
                sys.exit(f"could not stage {repo}: {exc}")
            print(f"retrying {repo} after attempt {attempt} failed: {exc}", flush=True)
            time.sleep(10 * attempt)
PY
    export HF_HUB_OFFLINE=1
}

# shellcheck disable=SC2317  # reachable when this file is executed, not sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -euo pipefail
    setup_runtime_storage
    for name in HF_HOME NEMO_HOME TRITON_CACHE_DIR UV_CACHE_DIR RESULTS_ROOT; do
        printf 'export %s=%s\n' "${name}" "${!name}"
    done
fi
