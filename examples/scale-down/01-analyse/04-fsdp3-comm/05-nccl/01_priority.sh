#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

# shellcheck source=nccl_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/nccl_common.sh" "$@"

run_nccl_case split-default default 0 --overlap --separate-rs
run_nccl_case split-high high 0 --overlap --separate-rs

(( FAILED == 0 ))
