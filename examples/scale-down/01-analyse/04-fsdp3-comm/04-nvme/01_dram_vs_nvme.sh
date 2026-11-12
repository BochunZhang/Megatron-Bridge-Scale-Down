#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

# shellcheck source=../offload_study_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/../offload_study_common.sh" "$@"

if [[ -z "${NVME_PATH}" ]]; then
    [[ "${DRY_RUN}" == true ]] || fail "Supply --nvme-path DIR on local NVMe or set NVME_PATH"
    NVME_PATH=/path/to/local-nvme
fi

for model in "${MODELS[@]}"; do
    run_case "${model}" dram 3 off 1 1 1 --offload weight --cpu-optimizer
    run_case "${model}" nvme 3 off 1 1 1 --nvme-path "${NVME_PATH}"
done

(( FAILED == 0 ))
