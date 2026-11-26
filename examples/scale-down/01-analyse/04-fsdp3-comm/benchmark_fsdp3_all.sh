#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
CLOCK_MODE=tdp
DRY_RUN=false
TORCH_THREADS=""

usage() {
    cat <<'USAGE'
Usage: benchmark_fsdp3_all.sh [--output-dir DIR] [--torch-thread N] [--unlock] [--dry-run]

Run the numbered single-node studies in order on one 4-GPU GB200 node.
Cases normally lock GPU clocks to TDP. The adaptive pause study always resets GPU clocks.
Run 05-nccl/05_cp_adaptive_pause.sh directly for the CP sweep.
Use --unlock to reset GPU clocks before every case.
Defaults to results/01-analyse/04-fsdp3-comm-tdp (-unlocked with --unlock).
Use NVME_PATH for the NVMe scratch path, NCCL_LIB for the static pause build and
NCCL_ADAPTIVE_PAUSE_LIB for the adaptive pause build.
Pass --torch-thread N to set PyTorch intra-op threads per rank for every case.
Use --dry-run to preview every case without launching training or changing clocks.
USAGE
}

while (( $# )); do
    case "$1" in
        --output-dir|--torch-thread)
            [[ -n "${2:-}" && "$2" != --* ]] || { echo "Missing value for $1" >&2; exit 2; }
            case "$1" in
                --output-dir) OUTPUT_DIR="$2";;
                --torch-thread)
                    [[ "$2" =~ ^[1-9][0-9]*$ ]] || { echo "--torch-thread must be a positive integer" >&2; exit 2; }
                    TORCH_THREADS="$2";;
            esac
            shift 2;;
        --unlock) CLOCK_MODE=unlocked; shift;;
        --dry-run) DRY_RUN=true; shift;;
        -h|--help) usage; exit 0;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2;;
    esac
done

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/01-analyse/04-fsdp3-comm-${CLOCK_MODE}}"
ARGS=(--output-dir "${OUTPUT_DIR}")
if [[ -n "${TORCH_THREADS}" ]]; then ARGS+=(--torch-thread "${TORCH_THREADS}"); fi
if [[ "${CLOCK_MODE}" == unlocked ]]; then ARGS+=(--unlock); fi
if [[ "${DRY_RUN}" == true ]]; then ARGS+=(--dry-run); fi

export LC_ALL=C
shopt -s nullglob
STUDIES=()
for study in "${SCRIPT_DIR}"/[0-9][0-9]-*/[0-9][0-9]_*.sh; do
    [[ "${study##*/}" == 05_cp_adaptive_pause.sh ]] || STUDIES+=("${study}")
done
[[ -n "${STUDIES[0]:-}" ]] || { echo 'No numbered FSDP3 studies found' >&2; exit 2; }

trap 'exit 130' INT
trap 'exit 143' TERM
FAILED=0
for study in "${STUDIES[@]}"; do
    name="${study#"${SCRIPT_DIR}/"}"
    printf '\nStudy %s\n' "${name}" >&2
    if bash "${study}" "${ARGS[@]}"; then
        continue
    else
        printf 'Failed study %s (exit %s)\n' "${name}" "$?" >&2
        FAILED=$((FAILED + 1))
    fi
done

printf '\nChecked %s studies, %s failed\n' "${#STUDIES[@]}" "${FAILED}" >&2
(( FAILED == 0 ))
