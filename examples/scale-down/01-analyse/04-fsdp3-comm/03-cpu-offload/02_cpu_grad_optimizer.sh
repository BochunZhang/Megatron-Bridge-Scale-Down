#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

# shellcheck source=../offload_study_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/../offload_study_common.sh" "$@"

# Match the offloaded components and graph scope while moving accumulation, then AdamW.
for model in "${MODELS[@]}"; do
    graph_args=()
    if [[ "${model}" == 27b ]]; then graph_args=(--local-attention-graph); fi
    run_case "${model}" gpu-grad-gpu-adamw 3 on 1 1 1 ${graph_args[@]+"${graph_args[@]}"} --offload optimizer,weight,gradient
    run_case "${model}" cpu-grad-gpu-adamw 3 on 1 1 1 ${graph_args[@]+"${graph_args[@]}"} --offload optimizer,weight,gradient \
        --cpu-grad-accumulation
    run_case "${model}" cpu-grad-cpu-adamw 3 on 1 1 1 ${graph_args[@]+"${graph_args[@]}"} --offload weight --cpu-optimizer
done

(( FAILED == 0 ))
