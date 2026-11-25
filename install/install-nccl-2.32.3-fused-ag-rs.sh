#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

tag=v2.32.3-1
kernel_base=ncclDevKernel_FusedAllGatherReduceScatter
symbol=ncclDevFusedAllGatherReduceScatterWithConfig
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
patch="${script_dir}/patches/nccl-2.32.3-fused-ag-rs.patch"
dest="${repo_root}/nccl/nccl-2.32.3-fused-ag-rs"
cuda_home="${CUDA_HOME:-/usr/local/cuda}"
jobs="$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc 2>/dev/null || echo 16)"
check_only=false
smoke_ranks=

die() {
    echo "ERROR: $*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Build the experimental NCCL v2.32.3-1 fused AG/RS library for GB200 (sm_100).

  --dest DIR        NCCL checkout (default: nccl/nccl-2.32.3-fused-ag-rs under Bridge)
  --cuda-home DIR   CUDA toolkit >= 12.8 (default: CUDA_HOME or /usr/local/cuda)
  --jobs N          Parallel make jobs (default: CPU affinity count)
  --check-only      Reuse the existing patched checkout and library
  --smoke-test N    Build and run the correctness test on 2 or 4 NVLink GPUs

The build exports ncclDevFusedAllGatherReduceScatterWithConfig via nccl_device.h.
The installer reports sm_100 resources for all four variants. The smoke test
checks each variant at 4 and 8 CTAs.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dest|--cuda-home|--jobs|--smoke-test)
            [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || die "$1 needs a value"
            case "$1" in
                --dest) dest="$2" ;;
                --cuda-home) cuda_home="$2" ;;
                --jobs) jobs="$2" ;;
                --smoke-test) smoke_ranks="$2" ;;
            esac
            shift 2
            ;;
        --check-only) check_only=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || die "--jobs must be a positive integer"
[[ -z "$smoke_ranks" || "$smoke_ranks" == 2 || "$smoke_ranks" == 4 ]] || die "--smoke-test accepts 2 or 4"
[[ -f "$patch" ]] || die "patch not found: ${patch}"
[[ -x "${cuda_home}/bin/cuobjdump" && -x "${cuda_home}/bin/nvcc" ]] || die "set --cuda-home to a CUDA toolkit"
cuda_home="$(cd -- "$cuda_home" && pwd)"

if [[ ! -e "$dest" ]]; then
    $check_only && die "checkout not found: ${dest}"
    mkdir -p "$(dirname -- "$dest")"
    git clone --depth 1 --branch "$tag" https://github.com/NVIDIA/nccl.git "$dest"
fi
dest="$(cd -- "$dest" && pwd)"
[[ -e "${dest}/.git" ]] || die "expected an NCCL checkout at ${dest}"
head="$(git -C "$dest" rev-parse --verify HEAD)"
tag_commit="$(git -C "$dest" rev-parse --verify "refs/tags/${tag}^{commit}")"
[[ "$head" == "$tag_commit" ]] || die "${dest} must be at ${tag}"

if git -C "$dest" apply --reverse --check "$patch" 2>/dev/null; then
    echo "=== ${patch##*/} is already applied"
elif $check_only; then
    die "${dest} does not carry ${patch##*/}"
else
    git -C "$dest" apply --check "$patch" || die "patch does not apply to ${dest}; use --dest for a fresh checkout"
    git -C "$dest" apply "$patch"
fi

build_dir="${dest}/build"
lib="${build_dir}/lib/libnccl.so"
build_log="${dest}/build-fused-ag-rs.log"
if ! $check_only; then
    echo "=== Building NCCL ${tag} for sm_100"
    if ! make -C "$dest" -j"$jobs" src.build CUDA_HOME="$cuda_home" BUILDDIR="$build_dir" \
        NVCC_GENCODE="-gencode=arch=compute_100,code=sm_100" >"$build_log" 2>&1; then
        tail -n 40 "$build_log" >&2
        die "build failed; full log in ${build_log}"
    fi
fi
[[ -f "$lib" ]] || die "library not found: ${lib}"
grep -aqF 'NCCL version 2.32.3-fused-ag-rs+cuda' "$lib" || die "library has the wrong NCCL version suffix"
exports="$(nm -D --defined-only "$lib")"
grep -qw "$symbol" <<<"$exports" || die "${symbol} is missing from ${lib}"

for variant in AgFirst RsFirst WarpSplit WarpSplitPause; do
    kernel="${kernel_base}${variant}"
    if ! resources="$("${cuda_home}/bin/cuobjdump" --dump-resource-usage --gpu-architecture sm_100 \
        --function "$kernel" "$lib" 2>&1)"; then
        echo "$resources" >&2
        die "cuobjdump failed for ${kernel}"
    fi
    resource_line="$(awk -v kernel="$kernel" '$1 == "Function" { want = $2 == kernel ":"; next }
        want && /REG:/ { print; exit }' <<<"$resources")"
    [[ -n "$resource_line" ]] || die "could not read sm_100 resources for ${kernel}"
    echo "=== ${kernel} (256 threads/CTA)"
    echo "$resource_line"
    if [[ -f "$build_log" ]]; then
        awk -v kernel="$kernel" '$0 ~ ("Function properties for " kernel "$") {
            if (getline) print
            exit
        }' "$build_log"
    fi
done
echo "Library: ${lib}"

if [[ -n "$smoke_ranks" ]]; then
    smoke_src="${dest}/docs/examples/06_device_api/07_fused_ag_rs/c/main.cu"
    smoke_bin="${build_dir}/fused-ag-rs-smoke"
    "${cuda_home}/bin/nvcc" -std=c++17 -arch=sm_100 -I"${build_dir}/include" "$smoke_src" \
        -L"${build_dir}/lib" -Xlinker -rpath -Xlinker "${build_dir}/lib" -lnccl -o "$smoke_bin"
    LD_LIBRARY_PATH="${build_dir}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" "$smoke_bin" "$smoke_ranks"
fi
