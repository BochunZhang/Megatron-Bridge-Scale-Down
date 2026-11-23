#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../../.." && pwd)"
launcher=(uv run --no-sync --active --project "$repo_root" python "$script_dir/cpu_tensor_add_benchmark.py" "$@")

if [[ -n ${CPU_ADD_NUMA_NODE:-} ]]; then
    exec numactl --cpunodebind="$CPU_ADD_NUMA_NODE" --membind="$CPU_ADD_NUMA_NODE" "${launcher[@]}"
fi
exec "${launcher[@]}"
