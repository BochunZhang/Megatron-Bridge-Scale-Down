#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

# shellcheck source=../offload_study_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/../offload_study_common.sh" "$@"

if [[ -z "${NVME_PATH}" ]]; then
    # The target nodes use /tmp for local NVMe scratch.
    NVME_PATH=/tmp/megatron-fsdp3-nvme
    if [[ "${DRY_RUN}" == false ]]; then
        probe_path=/tmp
        if [[ -e "${NVME_PATH}" || -L "${NVME_PATH}" ]]; then probe_path="${NVME_PATH}"; fi
        mount_source="$(df -P "${probe_path}" | awk 'NR == 2 { print $1 }')" ||
            fail "Cannot inspect the filesystem for ${NVME_PATH}. Set --nvme-path DIR or NVME_PATH."
        [[ "${mount_source}" =~ ^/dev/nvme[0-9]+n[0-9]+(p[0-9]+)?$ ]] ||
            fail "Default NVMe path ${NVME_PATH} is on ${mount_source:-an unknown device}. Set --nvme-path DIR or NVME_PATH to local NVMe scratch."
    fi
fi

for model in "${MODELS[@]}"; do
    run_case "${model}" dram 3 off 1 1 1 --offload weight --cpu-optimizer
    run_case "${model}" nvme 3 off 1 1 1 --nvme-path "${NVME_PATH}"
done

(( FAILED == 0 ))
