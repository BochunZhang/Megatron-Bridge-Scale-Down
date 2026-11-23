#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
gemms='[2,8192,7168],[4,8192,7168],[8,8192,7168],[16,8192,7168],[32,8192,7168],[64,8192,7168]'

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    printf 'Usage: bash run_ce_unicast_hbm_sweep.sh [benchmark options]\n\n'
    printf 'Default cases: task1,baseline5,baseline6,task7,task8\nDefault GEMM shapes: %s\n\n' "$gemms"
fi

exec bash "$script_dir/run_ce_hbm_benchmark.sh" \
    --cases task1,baseline5,baseline6,task7,task8 --gemm "$gemms" "$@"
