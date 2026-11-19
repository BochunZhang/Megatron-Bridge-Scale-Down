#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

# shellcheck source=../offload_study_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/../offload_study_common.sh" "$@"

for model in "${MODELS[@]}"; do
    name=""
    gather_in_start_param_sync=true
    if [[ "${model}" == 27b ]]; then
        # Match full-iteration replay: gather the first bucket in forward on both paths.
        name=forward-ag
        gather_in_start_param_sync=false
    fi
    run_case "${model}" "${name}" 3 off 1 1 1 \
        --override "ddp.fsdp_all_gather_in_start_param_sync=${gather_in_start_param_sync}"
    run_case "${model}" "${name}" 3 on 1 1 1
done

(( FAILED == 0 ))
