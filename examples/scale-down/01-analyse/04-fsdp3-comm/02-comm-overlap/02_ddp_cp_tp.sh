#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

# shellcheck source=../offload_study_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/../offload_study_common.sh" "$@"

RESUME_CASES=true
# On four GPUs TP=4 has a one-rank dense FSDP group. Use DDP to trace CP/TP.
for model in "${MODELS[@]}"; do
    run_case "${model}" cp4 ddp on 1 4 1
    run_case "${model}" tp4 ddp on 4 1 1
    run_case "${model}" tp4-overlap ddp on 4 1 1 --tp-comm-overlap
done

(( FAILED == 0 ))
