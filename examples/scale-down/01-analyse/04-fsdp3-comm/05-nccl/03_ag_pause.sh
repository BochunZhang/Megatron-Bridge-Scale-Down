#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

# shellcheck source=nccl_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/nccl_common.sh" "$@"

for pause in 1536 3072 4608; do
    run_nccl_case "split-high-symm-pause${pause}" high "${pause}" --overlap --symmetric
done

(( FAILED == 0 ))
