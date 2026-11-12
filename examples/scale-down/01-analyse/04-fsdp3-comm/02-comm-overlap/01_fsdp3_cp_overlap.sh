#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

# shellcheck source=../offload_study_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/../offload_study_common.sh" "$@"

export NCCL_COMM_SPLIT_SHARE_RESOURCES=0
for model in "${MODELS[@]}"; do
    graph=on
    # The 2.4t attention graph requires CP=1. Its graph baseline is in 01-baseline.
    if [[ "${model}" == 2.4t ]]; then graph=off; fi
    for cp in 1 4; do
        run_case "${model}" shared-ag-rs 3 "${graph}" 1 "${cp}" 1
        run_case "${model}" split-ag-rs 3 "${graph}" 1 "${cp}" 1 --overlap
    done
done

(( FAILED == 0 ))
