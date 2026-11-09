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

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
tag=v2.32.3-1
upstream=https://github.com/NVIDIA/nccl.git
nccl_root="${repo_root}/nccl"
cuda_home="${CUDA_HOME:-/usr/local/cuda}"
jobs="$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc 2>/dev/null || echo 16)"
variant=all

usage() {
    cat <<'USAGE'
Usage: install-nccl-2.32.3-clock64.sh [--variant all|store|split] [--nccl-root DIR]
                                      [--cuda-home DIR] [--jobs N]

Clone v2.32.3-1 afresh and build the selected pause-store-clock64 or
pause-split-clock64 libraries for sm_100. Replace a previous build only after
its replacement has passed the build checks. The default all also builds
the existing pause-only ncu baseline if its library is missing.
USAGE
}

die() { echo "ERROR: $*" >&2; exit 1; }

while (( $# )); do
    case "$1" in
        --variant) variant="$2"; shift 2 ;;
        --nccl-root) nccl_root="$2"; shift 2 ;;
        --cuda-home) cuda_home="$2"; shift 2 ;;
        --jobs) jobs="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || die "--jobs must be positive"
[[ -x "${cuda_home}/bin/nvcc" ]] || die "nvcc not found under ${cuda_home}"
cuda_home="$(cd -- "$cuda_home" && pwd -P)"
case "$variant" in
    all) variants=(store split) ;;
    store|split) variants=("$variant") ;;
    *) die "--variant must be all, store or split" ;;
esac

mkdir -p -- "$nccl_root"
nccl_root="$(cd -- "$nccl_root" && pwd -P)"
command -v flock >/dev/null || die "flock is required to lock ${nccl_root} during rebuilds"
exec 9>"${nccl_root}/.clock64-build.lock"
flock -n 9 || die "another clock64 build is running in ${nccl_root}"
stage="$(mktemp -d "${nccl_root}/.clock64-build.XXXXXX")"
cleanup() {
    local name previous dest
    for name in "${variants[@]}"; do
        previous="${stage}/previous-${name}"
        dest="${nccl_root}/nccl-2.32.3-pause-${name}-clock64"
        if [[ -d "$previous" && ! -e "$dest" ]]; then
            mv -T -- "$previous" "$dest" || {
                echo "Previous build is preserved at ${previous}" >&2
                return 1
            }
        fi
    done
    rm -rf -- "$stage"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for name in "${variants[@]}"; do
    build_name="nccl-2.32.3-pause-${name}-clock64"
    dest="${nccl_root}/${build_name}"
    # Only replace directories created by this installer.
    if [[ -e "$dest" || -L "$dest" ]]; then
        [[ ! -L "$dest" && -f "${dest}/.clock64-build" && "$(cat "${dest}/.clock64-build")" == "$build_name" ]] \
            || die "refusing to replace ${dest}; it is not a managed ${build_name} build"
    fi
    fresh="${stage}/${build_name}"
    git clone --depth 1 --branch "$tag" "$upstream" "$fresh"
    for patch in pause clock64-csv "pause-${name}-clock64"; do
        git -C "$fresh" apply --whitespace=error "${script_dir}/patches/nccl-2.32.3-${patch}.patch"
    done
    "${script_dir}/install-nccl-2.32.3.sh" --dest "$fresh" --cuda-home "$cuda_home" --jobs "$jobs"
    lib="${fresh}/build/lib/libnccl.so.2"
    grep -aFq "NCCL version 2.32.3-pause-${name}-clock64+cuda" "$lib" || die "wrong version banner in ${lib}"
    exports="$(nm -D --defined-only "$lib")"
    grep -qw ncclSymClock64Register <<<"$exports" || die "missing clock64 capture entry point in ${lib}"
    printf '%s\n' "$build_name" > "${fresh}/.clock64-build"
    previous="${stage}/previous-${name}"
    if [[ -d "$dest" ]]; then
        mv -T -- "$dest" "$previous"
    fi
    mv -T -- "$fresh" "$dest" || die "could not install ${dest}"
    rm -rf -- "$previous"
    echo "=== Installed ${dest}/build/lib/libnccl.so.2"
done

pause_dest="${nccl_root}/nccl-2.32.3-pause"
if [[ "$variant" == all && ! -f "${pause_dest}/build/lib/libnccl.so.2" ]]; then
    "${script_dir}/install-nccl-2.32.3-pause.sh" --dest "$pause_dest" --cuda-home "$cuda_home" --jobs "$jobs"
fi
