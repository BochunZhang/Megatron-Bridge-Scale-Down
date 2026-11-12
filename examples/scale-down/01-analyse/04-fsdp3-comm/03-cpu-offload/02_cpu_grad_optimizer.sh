#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

# shellcheck source=../offload_study_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/../offload_study_common.sh" "$@"

# Match the offloaded components and graph scope while moving accumulation, then AdamW.
for model in "${MODELS[@]}"; do
    run_case "${model}" gpu-grad-gpu-adamw 3 on 1 1 1 --local-attention-graph --offload optimizer,weight,gradient
    run_case "${model}" cpu-grad-gpu-adamw 3 on 1 1 1 --local-attention-graph --offload optimizer,weight,gradient \
        --cpu-grad-accumulation
    run_case "${model}" cpu-grad-cpu-adamw 3 on 1 1 1 --local-attention-graph --offload weight --cpu-optimizer
done

(( FAILED == 0 ))
