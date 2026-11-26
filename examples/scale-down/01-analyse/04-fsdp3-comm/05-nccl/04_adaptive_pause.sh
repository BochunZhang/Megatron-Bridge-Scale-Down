#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

NCCL_DEFAULT_LIB=nccl/nccl-2.32.3-adaptive-pause/build/lib/libnccl.so.2
NCCL_LIB="${NCCL_ADAPTIVE_PAUSE_LIB:-}"
# shellcheck source=nccl_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/nccl_common.sh" --unlock "$@"

NSYS_REPORT_ONLY=true
case_args=(--overlap --separate-rs --symmetric --metric)
stock_lib="${NCCL_STOCK_LIB:-${REPO_ROOT}/nccl/nccl-2.32.3/build/lib/libnccl.so.2}"

NCCL_BUILD=stock NCCL_LIB="${stock_lib}" run_nccl_case stock high 0 "${case_args[@]}"
NCCL_BUILD=adaptive-pause run_nccl_case adaptive-pause high -1 "${case_args[@]}"

(( FAILED == 0 ))
