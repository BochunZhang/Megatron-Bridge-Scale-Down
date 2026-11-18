#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

# shellcheck source=../offload_study_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/../offload_study_common.sh" "$@"

RESUME_CASES=true
for STUDY_PROFILE in none nsys pytorch; do
    for model in "${MODELS[@]}"; do
        for fsdp in 1 2 3; do
            run_case "${model}" "" "${fsdp}" on 1 1 1
        done
    done
done

(( FAILED == 0 ))
