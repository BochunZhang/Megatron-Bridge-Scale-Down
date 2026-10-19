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

# torchrun --no-python worker for run_nvls_collective_benchmark.sh --nsys:
#   bash nsys_worker.sh <nsys profile options...> <command...>
# runs `nsys profile --gpu-metrics-devices=<GPU> <options...> <command...>`, where <GPU> is the
# GPU this worker uses (cuda:LOCAL_RANK). GPU metrics are sampled per device and only one nsys
# session may sample a device, so each worker samples its own GPU. nsys numbers devices like
# nvidia-smi (PCI bus order); the launcher sets CUDA_DEVICE_ORDER=PCI_BUS_ID so the CUDA index
# (and numeric CUDA_VISIBLE_DEVICES entries) use the same order.

set -euo pipefail

die() {
    echo "ERROR: nsys_worker.sh: $*" >&2
    exit 2
}

[[ "${LOCAL_RANK:-}" =~ ^[0-9]+$ ]] || die "LOCAL_RANK is not set; run under torchrun --no-python"
gpu="$LOCAL_RANK"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=, read -r -a visible <<<"$CUDA_VISIBLE_DEVICES"
    gpu="${visible[$LOCAL_RANK]:-}"
fi
[[ "$gpu" =~ ^[0-9]+$ ]] \
    || die "cannot map LOCAL_RANK=${LOCAL_RANK} to an nsys GPU id (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}); use numeric ids"

exec nsys profile --gpu-metrics-devices="$gpu" "$@"
