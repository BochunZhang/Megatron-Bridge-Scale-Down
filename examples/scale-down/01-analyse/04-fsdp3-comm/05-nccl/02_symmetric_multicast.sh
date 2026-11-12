#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

# shellcheck source=nccl_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/nccl_common.sh" "$@"

run_nccl_case shared-high high 0
run_nccl_case split-high high 0 --overlap --separate-rs
run_nccl_case shared-high-symm-pause0 high 0 --symmetric
run_nccl_case split-high-symm-pause0 high 0 --overlap --separate-rs --symmetric

(( FAILED == 0 ))
