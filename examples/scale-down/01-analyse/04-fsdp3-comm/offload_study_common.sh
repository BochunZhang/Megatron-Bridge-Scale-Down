#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

# Source from a numbered experiment script.
set -euo pipefail

BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${BENCHMARK_DIR}/../../../.." && pwd)"
STUDY_DIR="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
GROUP="${STUDY_DIR##*/}"
STUDY_SCRIPT="${BASH_SOURCE[1]##*/}"
if [[ "${STUDY_SCRIPT}" == nccl_common.sh ]]; then
    STUDY_SCRIPT="${BASH_SOURCE[2]##*/}"
fi
STUDY_GPUS=4
STUDY_GPUS_PER_NODE=4
STUDY_NNODES=1
STUDY_NODE_RANK=0
STUDY_GBS=16
STUDY_SEQ_LENGTH=""
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/01-analyse/04-fsdp3-comm}"
MODELS=(27b 2.4t)
if [[ "${GROUP}" == 05-nccl ]]; then MODELS=(27b); fi
LAUNCHER=("${BENCHMARK_DIR}/run_pretrain_fsdp3.sh")
UNLOCK_GPU_CLOCKS=false
NSYS_REPORT_ONLY=false
TORCH_THREADS=""
DRY_RUN=false
PRINT_CASE_DIRS=false
RESUME_CASES=false
STUDY_PROFILE=nsys
ACTIVE_CASE_LOCK=""
FAILED=0
NVME_PATH="${NVME_PATH:-}"
NCCL_LIB="${NCCL_LIB:-}"

fail() { echo "$*" >&2; exit 2; }

while (( $# )); do
    case "$1" in
        --model|--output-dir|--nvme-path|--nccl-lib|--fsdp|--profile|--torch-thread|--gpu|--gpus)
            [[ -n "${2:-}" && "$2" != --* ]] || fail "Missing value for $1"
            case "$1" in
                --model)
                    [[ "$2" == 27b || "$2" == 2.4t ]] || fail "--model must be 27b or 2.4t"
                    [[ "${GROUP}" != 05-nccl || "$2" == 27b ]] || fail "05-nccl uses only the 27b model"
                    MODELS=("$2");;
                --output-dir) OUTPUT_DIR="$2";;
                --gpu|--gpus)
                    [[ "${STUDY_SCRIPT}" == 04_adaptive_pause.sh ]] || fail "$1 applies to 04_adaptive_pause.sh"
                    [[ "$2" =~ ^[1-9][0-9]*$ ]] && (( $2 >= 3 )) ||
                        fail "$1 requires at least 3 GPUs for symmetric multicast"
                    STUDY_GPUS="$2";;
                --torch-thread)
                    [[ "$2" =~ ^[1-9][0-9]*$ ]] || fail "--torch-thread must be a positive integer"
                    TORCH_THREADS="$2";;
                --profile)
                    [[ "${GROUP}" == 02-comm-overlap ]] || fail "--profile applies to 02-comm-overlap"
                    [[ "$2" == nsys || "$2" == pytorch ]] || fail "--profile must be nsys or pytorch"
                    STUDY_PROFILE="$2"
                    if [[ "$2" == pytorch ]]; then RESUME_CASES=true; fi;;
                --fsdp)
                    [[ "${STUDY_SCRIPT}" == 02_mbs.sh ]] || fail "--fsdp applies to 02_mbs.sh"
                    [[ "$2" == 1 || "$2" == 3 ]] || fail "--fsdp must be 1 or 3"
                    MBS_FSDP="$2";;
                --nvme-path)
                    [[ "${GROUP}" == 04-nvme ]] || fail "--nvme-path applies to 04-nvme"
                    NVME_PATH="$2";;
                --nccl-lib)
                    [[ "${GROUP}" == 05-nccl ]] || fail "--nccl-lib applies to 05-nccl"
                    NCCL_LIB="$2";;
            esac
            shift 2;;
        --unlock) UNLOCK_GPU_CLOCKS=true; shift;;
        --dry-run) DRY_RUN=true; shift;;
        --print-case-dirs) PRINT_CASE_DIRS=true; shift;;
        -h|--help)
            model_choices='27b|2.4t'
            [[ "${GROUP}" == 05-nccl ]] && model_choices=27b
            printf 'Usage: %s [--model %s] [--output-dir DIR] [--torch-thread N] [--unlock] [--dry-run | --print-case-dirs]\n' "$0" "${model_choices}"
            echo '  --unlock    Restore automatic GPU clocks'
            echo '  --torch-thread N    Set PyTorch intra-op threads per rank for every case'
            if [[ "${STUDY_SCRIPT}" == 02_mbs.sh ]]; then
                echo '  --fsdp 1|3    MBS sweep sharding level (default: 1, synchronous parameter AG in optimizer step)'
            fi
            if [[ "${STUDY_SCRIPT}" == 04_adaptive_pause.sh ]]; then
                echo '  --gpu N      Total GPUs (default 4). GBS is four times the GPU count.'
                echo '               For multiple nodes, set NODE_RANK (or RANK) and MASTER_ADDR.'
                echo '               GPUS_PER_NODE defaults to 4, capped at the requested GPU count.'
            fi
            case "${GROUP}" in
                02-comm-overlap) echo '  --profile nsys|pytorch    Capture Nsight Systems (default) or per-rank PyTorch traces for HTA';;
                04-nvme) echo '  --nvme-path DIR    Directory on local NVMe (default: /tmp/megatron-fsdp3-nvme, or set NVME_PATH)';;
                05-nccl)
                    echo '  --nccl-lib FILE    Override the patched NCCL library for this 27b study'
                    echo "                     Defaults to ${NCCL_DEFAULT_LIB}";;
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
    seq="${STUDY_SEQ_LENGTH:-${seq}}"
    local model_args=(--num-layers 8)
    if [[ "${model}" == 2.4t ]]; then
        model_args=(--num-layers 4 --num-experts 64 --ep 4 --dispatcher hybridep)
    fi
    local script_name="${STUDY_SCRIPT%.sh}"
    local result_group="${GROUP}/${script_name//_/-}"
    local result_args=(--stable-result-dir)
    if (( STUDY_NNODES == 1 )) && [[ "${RESUME_CASES}" == true ]]; then result_args+=(--resume); fi
    if [[ -n "${name}" ]]; then
        result_args+=(--run-name "${name}")
    fi
    local args=("${model_args[@]}"
        --model "${model}" --full --dtype bf16 --gpu "${STUDY_GPUS}" --nproc-per-node "${STUDY_GPUS_PER_NODE}"
        --nnodes "${STUDY_NNODES}" --node-rank "${STUDY_NODE_RANK}" --gbs "${STUDY_GBS}"
        --mbs "${mbs}" --seq "${seq}" --tp "${tp}" --cp "${cp}" --fsdp "${fsdp}" --cuda-graph "${graph}"
        --profile "${STUDY_PROFILE}" --train-iters 10 --profile-start 7 --profile-end 8
        --results-root "${OUTPUT_DIR}/${result_group}" "${result_args[@]}" "$@")
    [[ "${UNLOCK_GPU_CLOCKS}" == true ]] && args+=(--unlock)
    [[ "${NSYS_REPORT_ONLY}" == true ]] && args+=(--nsys-report-only)
    [[ -n "${TORCH_THREADS}" ]] && args+=(--torch-thread "${TORCH_THREADS}")
    local label="${result_group}${name:+/${name}} model=${model} fsdp=${fsdp} graph=${graph} tp=${tp} cp=${cp} mbs=${mbs} seq=${seq} profile=${STUDY_PROFILE}"
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
        # Multi-node cases share a result root and each node claims its own directory.
        claim_dir="${case_dir}"
        local skip_reason=existing
        if (( STUDY_NNODES > 1 )); then
            [[ ! -e "${node_dir}" ]] || fail "Result already exists at ${node_dir}; use a fresh --output-dir on all nodes"
            claim_dir="${node_dir}.lock"
        elif [[ "${RESUME_CASES}" == true ]]; then
            claim_dir="${case_dir}.lock"
            skip_reason=active
        fi
        if [[ -d "${claim_dir}" ]]; then
            (( STUDY_NNODES == 1 )) || fail "Another process is using ${node_dir}"
            printf 'skip %s case: %s\n' "${skip_reason}" "${case_dir}" >&2
            return 0
        fi
        if declare -F prepare_case >/dev/null; then
            prepare_case "${name}"
        fi
        mkdir -p -- "${claim_dir%/*}"
        if ! mkdir -- "${claim_dir}" 2>/dev/null; then
            [[ -d "${claim_dir}" ]] || fail "Cannot create case claim: ${claim_dir}"
            (( STUDY_NNODES == 1 )) || fail "Another process is using ${node_dir}"
            printf 'skip %s case: %s\n' "${skip_reason}" "${case_dir}" >&2
            return 0
        fi
        if (( STUDY_NNODES > 1 )) || [[ "${RESUME_CASES}" == true ]]; then ACTIVE_CASE_LOCK="${claim_dir}"; fi
    fi
    printf 'case %s\n' "${label}" >&2
    local run_status=0
    "${LAUNCHER[@]}" "${args[@]}" || run_status=$?
    if [[ "${DRY_RUN}" == false ]] && declare -F finish_case >/dev/null; then
        finish_case "${node_dir}" "${run_status}" || run_status=$?
    fi
    if (( run_status != 0 )); then
        if (( STUDY_NNODES == 1 )) && [[ -n "${case_dir}" && "${RESUME_CASES}" == false ]]; then
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
