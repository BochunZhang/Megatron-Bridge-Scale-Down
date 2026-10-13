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
GPU_MONITOR="${REPO_ROOT}/scripts/scale-down/nvidia-smi/check_gpu.py"
GPU_UTIL_EXPORTER="${REPO_ROOT}/scripts/scale-down/analyse/export_gpu_utilization.py"
GPUS_PER_NODE=4

log() {
    printf '%s\n' "$*"
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

# Fill in the default model ladder for a family. Qwen3.5 uses three rungs so one
# model spans the dense proxy, the 16-GPU full run and the 32-GPU comparison;
# DeepSeek-V3 reuses its single proxy recipe at several layer/expert cuts.
model_defaults() {
    case "$1" in
        qwen)
            case "$2" in
                dense-proxy) MODEL="qwen35_text_9b" ;;
                small) MODEL="qwen35_text_35b_a3b" ;;
                large) MODEL="qwen35_text_122b_a10b" ;;
                *) die "unknown qwen rung: $2 (expected dense-proxy|small|large)" ;;
            esac
            ;;
        deepseek)
            MODEL="deepseek_v3"
            ;;
        *)
            die "unknown model family: $1 (expected qwen|deepseek)"
            ;;
    esac
}

# Recipe names follow "<model>_pretrain_4gpu_gb200_<bf16|fp8mx>_fsdp1_config".
recipe_name() {
    case "$2" in
        bf16) printf '%s_pretrain_4gpu_gb200_bf16_fsdp1_config\n' "$1" ;;
        mxfp8) printf '%s_pretrain_4gpu_gb200_fp8mx_fsdp1_config\n' "$1" ;;
        *) die "unknown dtype: $2 (expected bf16|mxfp8)" ;;
    esac
}

# Trace the mxfp8 provider name used by the DeepSeek recipes.
dtype_recipe_suffix() {
    case "$1" in
        bf16) printf 'bf16' ;;
        mxfp8) printf 'fp8mx' ;;
        *) die "unknown dtype: $1" ;;
    esac
}

# Select the offload module list for one case and model shape.
#
# `mlp`      offloads only the MLP/MoE activation path.
# `attn-mlp` adds the attention path, where GatedDeltaNet reuses the attention
#            group names ("qkv_linear", "core_attn", "attn_proj").
# `attn-mlp-fused` uses the fused grouped-MLP group, which requires the TE op
#            fuser and is therefore the only MoE variant compatible with the
#            sync-free HybridEP path.
offload_modules() {
    local case_name="$1"
    local scope="$2"   # dense | expert
    local family="$3"  # qwen | deepseek
    local fused="$4"   # true | false
    local attn
    if [[ "${scope}" == expert ]]; then
        if [[ "${fused}" == true ]]; then
            attn="mlp_norm,fused_group_mlp"
        else
            attn="mlp_norm,expert_fc1,moe_act"
        fi
    else
        attn="mlp_norm,mlp_act"
    fi
    if [[ "${case_name}" == offload-mlp ]]; then
        printf '[%s]\n' "${attn}"
        return
    fi
    # DeepSeek uses the standard attention projection; Qwen3.5 GatedDeltaNet does
    # not have a qkv_linear group.
    if [[ "${family}" == deepseek ]]; then
        attn="${attn},attn_norm,qkv_linear,core_attn,attn_proj"
    else
        attn="${attn},attn_norm,core_attn,attn_proj"
    fi
    printf '[%s]\n' "${attn}"
}

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
