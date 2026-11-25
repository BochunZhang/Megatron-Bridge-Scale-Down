#!/usr/bin/env bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${script_dir}/offload_common.sh" 06-param-nvme-act-cpu-opt-cpu-overlap nvme cpu cpu \
    --nvme "${NVME_PATH:-/tmp}" --cpu-optimizer grace_adam --overlap-cpu-optimizer "$@"
