#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

# Source from a numbered experiment script.
set -euo pipefail

BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${BENCHMARK_DIR}/../../../.." && pwd)"
GROUP="$(basename "$(dirname "${BASH_SOURCE[1]}")")"
STUDY_SCRIPT="${BASH_SOURCE[1]##*/}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/01-analyse/04-fsdp3-comm}"
MODELS=(27b 2.4t)
if [[ "${GROUP}" == 05-nccl ]]; then MODELS=(27b); fi
LAUNCHER=("${BENCHMARK_DIR}/run_pretrain_fsdp3.sh")
DRY_RUN=false
PRINT_CASE_DIRS=false
RESUME_CASES=false
ACTIVE_CASE_LOCK=""
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
        --print-case-dirs) PRINT_CASE_DIRS=true; shift;;
        -h|--help)
            printf 'Usage: %s [--model 27b|2.4t] [--output-dir DIR] [--dry-run | --print-case-dirs]\n' "$0"
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
# An empty NAME omits the case suffix.
run_case() {
    local model="$1" name="$2" fsdp="$3" graph="$4" tp="$5" cp="$6" mbs="$7"
    shift 7
    local seq=4096
    if (( tp == 4 || cp == 4 )); then seq=16384; fi
    local model_args=(--num-layers 8)
    if [[ "${model}" == 2.4t ]]; then
        model_args=(--num-layers 4 --num-experts 64 --ep 4 --dispatcher hybridep)
    fi
    local script_name="${STUDY_SCRIPT%.sh}"
    local result_group="${GROUP}/${script_name//_/-}"
    local result_args=(--stable-result-dir)
    if [[ "${RESUME_CASES}" == true ]]; then result_args+=(--resume); fi
    if [[ -n "${name}" ]]; then
        result_args+=(--run-name "${name}")
    fi
    local args=("${model_args[@]}"
        --model "${model}" --full --dtype bf16 --gpu 4 --nproc-per-node 4 --nnodes 1 --node-rank 0
        --gbs 16 --mbs "${mbs}" --seq "${seq}" --tp "${tp}" --cp "${cp}" --fsdp "${fsdp}" --cuda-graph "${graph}"
        --nsys --train-iters 10 --profile-start 7 --profile-end 8
        --results-root "${OUTPUT_DIR}/${result_group}" "${result_args[@]}" "$@")
    local label="${result_group}${name:+/${name}} model=${model} fsdp=${fsdp} graph=${graph} tp=${tp} cp=${cp} mbs=${mbs} seq=${seq}"
    local case_dir=""
    local claim_dir=""
    if [[ "${DRY_RUN}" == true && "${PRINT_CASE_DIRS}" == false ]]; then
        args+=(--dry-run)
    else
        local node_dir
        if ! node_dir=$("${LAUNCHER[@]}" "${args[@]}" --print-result-dir); then
            printf 'failed to resolve result directory for %s\n' "${label}" >&2
            FAILED=$(( FAILED + 1 ))
            return 0
        fi
        case_dir="${node_dir%/*}"
        if [[ "${PRINT_CASE_DIRS}" == true ]]; then
            printf '%s\n' "${case_dir}"
            return 0
        fi
        # Resumable cases need a separate claim so an incomplete result can be retried.
        claim_dir="${case_dir}"
        local skip_reason=existing
        if [[ "${RESUME_CASES}" == true ]]; then
            claim_dir="${case_dir}.lock"
            skip_reason=active
        fi
        if [[ -d "${claim_dir}" ]]; then
            printf 'skip %s case: %s\n' "${skip_reason}" "${case_dir}" >&2
            return 0
        fi
        if declare -F prepare_case >/dev/null; then
            prepare_case "${name}"
        fi
        mkdir -p -- "${claim_dir%/*}"
        if ! mkdir -- "${claim_dir}" 2>/dev/null; then
            [[ -d "${claim_dir}" ]] || fail "Cannot create case claim: ${claim_dir}"
            printf 'skip %s case: %s\n' "${skip_reason}" "${case_dir}" >&2
            return 0
        fi
        if [[ "${RESUME_CASES}" == true ]]; then ACTIVE_CASE_LOCK="${claim_dir}"; fi
    fi
    printf 'case %s\n' "${label}" >&2
    if ! "${LAUNCHER[@]}" "${args[@]}"; then
        if [[ -n "${case_dir}" && "${RESUME_CASES}" == false ]]; then
            rmdir -- "${case_dir}" 2>/dev/null || true
        fi
        printf 'failed %s\n' "${label}" >&2
        FAILED=$((FAILED + 1))
    fi
    release_case_lock
}

release_case_lock() {
    if [[ -n "${ACTIVE_CASE_LOCK}" ]]; then
        rmdir -- "${ACTIVE_CASE_LOCK}"
        ACTIVE_CASE_LOCK=""
    fi
}

trap release_case_lock EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
