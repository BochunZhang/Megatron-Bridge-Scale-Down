#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

# Source from a numbered experiment script.
set -euo pipefail

BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${BENCHMARK_DIR}/../../../.." && pwd)"
GROUP="$(basename "$(dirname "${BASH_SOURCE[1]}")")"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/01-analyse/04-fsdp3-comm}"
MODELS=(27b 2.4t)
if [[ "${GROUP}" == 05-nccl ]]; then MODELS=(27b); fi
LAUNCHER=("${BENCHMARK_DIR}/run_pretrain_fsdp3.sh")
DRY_RUN=false
FAILED=0
NVME_PATH="${NVME_PATH:-}"
NCCL_LIB="${NCCL_LIB:-}"

fail() { echo "$*" >&2; exit 2; }

while (( $# )); do
    case "$1" in
        --model|--output-dir|--nvme-path|--nccl-lib)
            [[ -n "${2:-}" && "$2" != --* ]] || fail "Missing value for $1"
            case "$1" in
                --model)
                    [[ "$2" == 27b || "$2" == 2.4t ]] || fail "--model must be 27b or 2.4t"
                    [[ "${GROUP}" != 05-nccl || "$2" == 27b ]] || fail "05-nccl uses only the 27b model"
                    MODELS=("$2");;
                --output-dir) OUTPUT_DIR="$2";;
                --nvme-path)
                    [[ "${GROUP}" == 04-nvme ]] || fail "--nvme-path applies to 04-nvme"
                    NVME_PATH="$2";;
                --nccl-lib)
                    [[ "${GROUP}" == 05-nccl ]] || fail "--nccl-lib applies to 05-nccl"
                    NCCL_LIB="$2";;
            esac
            shift 2;;
        --dry-run) DRY_RUN=true; shift;;
        -h|--help)
            printf 'Usage: %s [--model 27b|2.4t] [--output-dir DIR] [--dry-run]\n' "$0"
            case "${GROUP}" in
                04-nvme) echo '  --nvme-path DIR    Directory on local NVMe (default: /tmp/megatron-fsdp3-nvme, or set NVME_PATH)';;
                05-nccl)
                    echo '  --nccl-lib FILE    Override the NCCL library (or set NCCL_LIB); 27b only'
                    echo '                     Default: nccl/nccl-2.32.3-cta-with-pause/build/lib/libnccl.so.2';;
            esac
            exit 0;;
        *) fail "Unknown option: $1";;
    esac
done

# run_case MODEL NAME FSDP GRAPH TP CP MBS [launcher flags...]
# An empty NAME uses a subdirectory named after the calling script.
run_case() {
    local model="$1" name="$2" fsdp="$3" graph="$4" tp="$5" cp="$6" mbs="$7"
    shift 7
    local seq=4096
    if (( tp == 4 || cp == 4 )); then seq=16384; fi
    local model_args=(--num-layers 8)
    if [[ "${model}" == 2.4t ]]; then
        model_args=(--num-layers 4 --num-experts 64 --ep 4 --dispatcher hybridep)
    fi
    local result_group="${GROUP}"
    local result_args=(--stable-result-dir)
    if [[ -n "${name}" ]]; then
        result_args+=(--run-name "${name}")
    else
        local script_name="${BASH_SOURCE[1]##*/}"
        script_name="${script_name%.sh}"
        result_group+="/${script_name//_/-}"
    fi
    local args=("${model_args[@]}"
        --model "${model}" --full --dtype bf16 --gpu 4 --nproc-per-node 4 --nnodes 1 --node-rank 0
        --gbs 16 --mbs "${mbs}" --seq "${seq}" --tp "${tp}" --cp "${cp}" --fsdp "${fsdp}" --cuda-graph "${graph}"
        --nsys --train-iters 10 --profile-start 7 --profile-end 8
        --results-root "${OUTPUT_DIR}/${result_group}" "${result_args[@]}" "$@")
    local label="${result_group}${name:+/${name}} model=${model} fsdp=${fsdp} graph=${graph} tp=${tp} cp=${cp} mbs=${mbs} seq=${seq}"
    local case_dir=""
    if [[ "${DRY_RUN}" == true ]]; then
        args+=(--dry-run)
    else
        local node_dir
        if ! node_dir=$("${LAUNCHER[@]}" "${args[@]}" --print-result-dir); then
            printf 'failed to resolve result directory for %s\n' "${label}" >&2
            FAILED=$(( FAILED + 1 ))
            return 0
        fi
        case_dir="${node_dir%/*}"
        if [[ -d "${case_dir}" ]]; then
            printf 'skip existing case: %s\n' "${case_dir}" >&2
            return 0
        fi
        if declare -F prepare_case >/dev/null; then
            prepare_case "${name}"
        fi
        # Claim the case directory before launching so parallel invocations also skip it.
        mkdir -p -- "${case_dir%/*}"
        if ! mkdir -- "${case_dir}" 2>/dev/null; then
            [[ -d "${case_dir}" ]] || fail "Cannot create case directory: ${case_dir}"
            printf 'skip existing case: %s\n' "${case_dir}" >&2
            return 0
        fi
    fi
    printf 'case %s\n' "${label}" >&2
    if ! "${LAUNCHER[@]}" "${args[@]}"; then
        if [[ -n "${case_dir}" ]]; then
            rmdir -- "${case_dir}" 2>/dev/null || true
        fi
        printf 'failed %s\n' "${label}" >&2
        FAILED=$((FAILED + 1))
    fi
}

trap 'exit 130' INT
trap 'exit 143' TERM
