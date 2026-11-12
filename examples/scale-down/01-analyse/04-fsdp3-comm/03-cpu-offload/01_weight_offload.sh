#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

# shellcheck source=../offload_study_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/../offload_study_common.sh" "$@"

export NCCL_COMM_SPLIT_SHARE_RESOURCES=0
for model in "${MODELS[@]}"; do
    cp_sizes=(1 4)
    if [[ "${model}" == 2.4t ]]; then cp_sizes=(1); fi
    for cp in "${cp_sizes[@]}"; do
        run_case "${model}" shared-ag-rs 3 on 1 "${cp}" 1
        run_case "${model}" split-ag-rs 3 on 1 "${cp}" 1 --overlap
        run_case "${model}" shared-ag-rs 3 on 1 "${cp}" 1 --offload weight
        run_case "${model}" split-ag-rs 3 on 1 "${cp}" 1 --overlap --offload weight
    done
done

(( FAILED == 0 ))
