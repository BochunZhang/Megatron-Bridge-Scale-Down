#!/usr/bin/env bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${script_dir}/offload_common.sh" 05-param-cpu-act-cpu-opt-cpu-overlap cpu cpu cpu \
    --cpu-optimizer grace_adam --overlap-cpu-optimizer "$@"
