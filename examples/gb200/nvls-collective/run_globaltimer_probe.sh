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

usage() {
    cat <<'USAGE'
Measure %globaltimer and the clocks of one GB200 GPU (globaltimer_probe.cu).

Builds build/globaltimer_probe (nvcc -arch=sm_100), records the GPU's clocks from nvidia-smi
(current, max, application and supported clocks, clock event reasons), then runs the probe plainly
and, unless --no-nsys, once more under nsys with the GPU metrics the NVLS launcher collects:
an attached profiler may change how often %globaltimer advances and the SM clock that turns the
pause sweep's clock64 cycles into ns, and the sweep runs under nsys.
Everything goes to results/gb200/nvls-collective/globaltimer-probe/<host>-<yyyymmdd-hhmmss>/.

Usage:
  run_globaltimer_probe.sh [options] [-- globaltimer_probe options]

Options:
  --gpu N           GPU to probe (default: 0)
  --cuda-home DIR   CUDA toolkit (default: CUDA_HOME or /usr/local/cuda)
  --no-nsys         Skip the run under nsys
  -h, --help        Show this help

globaltimer_probe options: --window-ms MS, --reps N, --pause-cycles LIST (see globaltimer_probe --help).
USAGE
}

die() {
    echo "ERROR: $*" >&2
    exit 2
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
gpu=0
cuda_home="${CUDA_HOME:-/usr/local/cuda}"
use_nsys=true
probe_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) [[ $# -ge 2 ]] || die "$1 requires a value"; gpu="$2"; shift 2 ;;
        --cuda-home) [[ $# -ge 2 ]] || die "$1 requires a value"; cuda_home="$2"; shift 2 ;;
        --no-nsys) use_nsys=false; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; probe_args+=("$@"); break ;;
        *) die "unknown option: $1 (see --help)" ;;
    esac
done
[[ "$gpu" =~ ^[0-9]+$ ]] || die "--gpu must be a GPU id, got ${gpu}"
[[ -x "${cuda_home}/bin/nvcc" ]] || die "${cuda_home}/bin/nvcc not found (--cuda-home)"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found"

binary="${script_dir}/build/globaltimer_probe"
mkdir -p "${script_dir}/build"
if [[ ! -x "$binary" || "${script_dir}/globaltimer_probe.cu" -nt "$binary" ]]; then
    echo "=== Building ${binary}" >&2
    "${cuda_home}/bin/nvcc" -O2 -std=c++17 -arch=sm_100 -o "$binary" "${script_dir}/globaltimer_probe.cu"
fi

output_dir="${repo_root}/results/gb200/nvls-collective/globaltimer-probe/$(hostname -s)-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$output_dir"
echo "=== Writing to ${output_dir}" >&2

# The clocks the driver reports for this GPU. "Max Clocks" is the top of the range nvidia-smi -lgc
# accepts; the supported (memory, graphics) pairs are the steps it can lock to.
{
    nvidia-smi -i "$gpu" -q -d CLOCK,PERFORMANCE
    echo "=== supported clocks (memory MHz, graphics MHz): count, lowest and highest graphics clock"
    nvidia-smi -i "$gpu" --query-supported-clocks=memory,graphics --format=csv,noheader,nounits \
        | sort -t, -k2 -n | awk -F', *' 'NR == 1 { lo = $0 } { hi = $0; n++ } END { print n " pairs; lowest " lo "; highest " hi }'
} >"${output_dir}/nvidia-smi-clocks.txt" 2>&1 || true
nvidia-smi -i "$gpu" --query-gpu=name,clocks.sm,clocks.max.sm,clocks.applications.graphics,clocks.default_applications.graphics,clocks.mem,clocks.max.mem \
    --format=csv | tee -a "${output_dir}/nvidia-smi-clocks.txt"

echo "=== globaltimer_probe on GPU ${gpu}" >&2
CUDA_VISIBLE_DEVICES="$gpu" "$binary" ${probe_args[@]+"${probe_args[@]}"} | tee "${output_dir}/probe.txt"

if $use_nsys; then
    command -v nsys >/dev/null || die "nsys not found; pass --no-nsys"
    echo "=== globaltimer_probe on GPU ${gpu} under nsys (GPU metrics at 50 kHz)" >&2
    CUDA_VISIBLE_DEVICES="$gpu" nsys profile --trace=cuda --sample=none --cpuctxsw=none \
        --gpu-metrics-devices=all --gpu-metrics-frequency=50000 \
        --output="${output_dir}/probe" --force-overwrite=true \
        "$binary" ${probe_args[@]+"${probe_args[@]}"} | tee "${output_dir}/probe.nsys.txt"
fi
echo "=== Done: ${output_dir}" >&2
