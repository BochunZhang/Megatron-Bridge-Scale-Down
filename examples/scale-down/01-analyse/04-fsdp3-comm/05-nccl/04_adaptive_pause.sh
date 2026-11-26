#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

NCCL_DEFAULT_LIB=nccl/nccl-2.32.3-adaptive-pause/build/lib/libnccl.so.2
NCCL_LIB="${NCCL_ADAPTIVE_PAUSE_LIB:-}"
# shellcheck source=nccl_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/nccl_common.sh" --unlock "$@"

STUDY_GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
[[ "${STUDY_GPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]] || fail "GPUS_PER_NODE must be a positive integer"
if (( STUDY_GPUS_PER_NODE > STUDY_GPUS )); then STUDY_GPUS_PER_NODE="${STUDY_GPUS}"; fi
(( STUDY_GPUS % STUDY_GPUS_PER_NODE == 0 )) || fail "--gpu must be divisible by GPUS_PER_NODE"
STUDY_NNODES=$((STUDY_GPUS / STUDY_GPUS_PER_NODE))
STUDY_GBS=$((4 * STUDY_GPUS))

# Separate rendezvous ports keep early arrivals out of the previous case's store.
stock_port="${MASTER_PORT:-29501}"
adaptive_port="${stock_port}"
if (( STUDY_NNODES > 1 )); then
    STUDY_NODE_RANK="${NODE_RANK:-${RANK:-}}"
    [[ -n "${STUDY_NODE_RANK}" ]] || fail "Set NODE_RANK or RANK for a multi-node launch"
    [[ "${stock_port}" =~ ^[1-9][0-9]{0,4}$ ]] && (( stock_port < 65535 )) ||
        fail "MASTER_PORT must be between 1 and 65534 for a multi-node launch"
    adaptive_port=$((stock_port + 1))
fi

NSYS_REPORT_ONLY=true
case_args=(--overlap --separate-rs --symmetric --metric)
stock_lib="${NCCL_STOCK_LIB:-${REPO_ROOT}/nccl/nccl-2.32.3/build/lib/libnccl.so.2}"

NCCL_BUILD=stock NCCL_LIB="${stock_lib}" run_nccl_case stock high 0 "${case_args[@]}" --master-port "${stock_port}"
if (( STUDY_NNODES > 1 && FAILED != 0 )); then exit 1; fi
NCCL_BUILD=adaptive-pause run_nccl_case adaptive-pause high -1 "${case_args[@]}" --master-port "${adaptive_port}"

(( FAILED == 0 ))
