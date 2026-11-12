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
MODE_ARGS=(--resume)
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
        --dry-run) DRY_RUN=true; MODE_ARGS+=(--dry-run); shift;;
        -h|--help)
            printf 'Usage: %s [--model 27b|2.4t] [--output-dir DIR] [--dry-run]\n' "$0"
            case "${GROUP}" in
                04-nvme) echo '  --nvme-path DIR    Directory on local NVMe (default: /tmp/megatron-fsdp3-nvme, or set NVME_PATH)';;
                05-nccl) echo '  --nccl-lib FILE    NCCL 2.32.3-pause library (or set NCCL_LIB); 27b only';;
            esac
            exit 0;;
        *) fail "Unknown option: $1";;
    esac
done

# run_case MODEL NAME FSDP GRAPH TP CP MBS [launcher flags...]
run_case() {
    local model="$1" name="$2" fsdp="$3" graph="$4" tp="$5" cp="$6" mbs="$7"
    shift 7
    local seq=4096
    if (( tp == 4 || cp == 4 )); then seq=16384; fi
    local model_args=(--num-layers 8)
    if [[ "${model}" == 2.4t ]]; then
        model_args=(--num-layers 4 --num-experts 64 --ep 4 --dispatcher hybridep)
    fi
    local label="${GROUP}/${name} model=${model} fsdp=${fsdp} graph=${graph} tp=${tp} cp=${cp} mbs=${mbs} seq=${seq}"
    printf 'case %s\n' "${label}" >&2
    if ! "${LAUNCHER[@]}" "${MODE_ARGS[@]}" "${model_args[@]}" \
        --model "${model}" --full --dtype bf16 --gpu 4 --nproc-per-node 4 --nnodes 1 --node-rank 0 \
        --gbs 16 --mbs "${mbs}" --seq "${seq}" --tp "${tp}" --cp "${cp}" --fsdp "${fsdp}" --cuda-graph "${graph}" \
        --nsys --train-iters 10 --profile-start 7 --profile-end 8 \
        --results-root "${OUTPUT_DIR}/${GROUP}" --stable-result-dir --run-name "${name}" "$@"; then
        printf 'failed %s\n' "${label}" >&2
        FAILED=$((FAILED + 1))
    fi
}

trap 'exit 130' INT
trap 'exit 143' TERM
