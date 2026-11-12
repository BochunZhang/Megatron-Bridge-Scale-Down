#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

# shellcheck source=../offload_study_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/../offload_study_common.sh" "$@"

for model in "${MODELS[@]}"; do
    graph=on
    if [[ "${model}" == 2.4t ]]; then graph=off; fi
    run_case "${model}" baseline 1 "${graph}" 1 1 1
    run_case "${model}" baseline 3 "${graph}" 1 1 1
done

(( FAILED == 0 ))
