#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

# shellcheck source=../offload_study_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/../offload_study_common.sh" "$@"

for model in "${MODELS[@]}"; do
    for mbs in 1 2 4; do
        run_case "${model}" baseline 3 on 1 1 "${mbs}"
    done
done

(( FAILED == 0 ))
