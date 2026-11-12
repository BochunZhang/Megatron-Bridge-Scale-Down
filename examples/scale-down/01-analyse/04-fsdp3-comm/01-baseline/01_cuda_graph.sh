#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

# shellcheck source=../offload_study_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/../offload_study_common.sh" "$@"

for model in "${MODELS[@]}"; do
    run_case "${model}" "" 3 off 1 1 1
    run_case "${model}" "" 3 on 1 1 1
done

(( FAILED == 0 ))
