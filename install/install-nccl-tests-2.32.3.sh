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

# Latest nccl-tests built against stock NCCL 2.32.3, for examples/gb200/nccl-tests.
#
#   1. Check that --nccl-home is the build of an unmodified v2.32.3-1 tree (the one
#      install-nccl-2.32.3.sh makes in <megatron-bridge>/nccl/nccl-2.32.3, not the -cta / -pause trees).
#   2. Clone the default branch of nccl-tests into <megatron-bridge>/nccl/nccl-tests (reused if it
#      exists; --update fetches the latest commit again).
#   3. Build it for sm_100 twice:
#        <nccl-tests>/build      MPI=0: one process drives all GPUs (-g / -t)
#        <nccl-tests>/build-mpi  MPI=1: one process per GPU (mpirun -np N ... -g 1); needed by
#                                 run_ncu_all_gather_stmc.sh. Skipped with a warning without MPI.
#   nccl-tests only has an $ORIGIN rpath, so NCCL is found through LD_LIBRARY_PATH=<nccl-home>/lib.
#
# Usage:
#   ./install/install-nccl-tests-2.32.3.sh [--nccl-home DIR] [--dest DIR] [--mpi-home DIR]
#                                          [--cuda-home DIR] [--jobs N] [--update]
#
#   --nccl-home DIR NCCL 2.32.3 build (default: <megatron-bridge>/nccl/nccl-2.32.3/build)
#   --dest DIR      nccl-tests checkout (default: <megatron-bridge>/nccl/nccl-tests)
#   --mpi-home DIR  MPI install with include/mpi.h and lib/libmpi.so (default: from mpicc in PATH)
#   --cuda-home DIR CUDA toolkit, >= 12.8 for sm_100 (default: CUDA_HOME or /usr/local/cuda)
#   --jobs N        make -j (default: nproc)
#   --update        Fetch the latest nccl-tests commit into an existing checkout before building

set -euo pipefail

NCCL_TESTS_REPO="https://github.com/NVIDIA/nccl-tests.git"
NCCL_TAG="v2.32.3-1"
GENCODE="-gencode=arch=compute_100,code=sm_100"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
nccl_home="${repo_root}/nccl/nccl-2.32.3/build"
dest="${repo_root}/nccl/nccl-tests"
mpi_home="${MPI_HOME:-}"
cuda_home="${CUDA_HOME:-/usr/local/cuda}"
# GNU nproc honours OMP_NUM_THREADS, which training environments usually export as 1.
jobs="$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc 2>/dev/null || echo 16)"
update=false

die() {
    echo "ERROR: $*" >&2
    exit 1
}
note() { echo "  [warn] $*"; }
ok() { echo "  [ok]   $*"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nccl-home) nccl_home="$2"; shift 2 ;;
        --dest) dest="$2"; shift 2 ;;
        --mpi-home) mpi_home="$2"; shift 2 ;;
        --cuda-home) cuda_home="$2"; shift 2 ;;
        --jobs) jobs="$2"; shift 2 ;;
        --update) update=true; shift ;;
        -h|--help) sed -n '16,38p' "$0"; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

[[ -x "${cuda_home}/bin/nvcc" ]] || die "nvcc not found under ${cuda_home}; pass --cuda-home"
echo "CUDA toolkit: ${cuda_home} ($("${cuda_home}/bin/nvcc" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p'))"

################################################################################
# 1. Stock NCCL 2.32.3

echo "=== NCCL: ${nccl_home}"
[[ -f "${nccl_home}/include/nccl.h" && -f "${nccl_home}/lib/libnccl.so" ]] \
    || die "no NCCL build in ${nccl_home}; run ./install/install-nccl-2.32.3.sh first or pass --nccl-home"
code="$(awk '/^#define[ \t]+NCCL_VERSION_CODE[ \t]+[0-9]+/ {print $3}' "${nccl_home}/include/nccl.h")"
[[ "$code" == 23203 ]] || die "${nccl_home}/include/nccl.h is NCCL_VERSION_CODE ${code}, expected 23203 (2.32.3)"
ok "headers are NCCL 2.32.3"
nccl_src="$(cd -- "${nccl_home}/.." && pwd)"
if git -C "$nccl_src" rev-parse --git-dir >/dev/null 2>&1; then
    tag="$(git -C "$nccl_src" describe --tags --exact-match 2>/dev/null || git -C "$nccl_src" describe --tags --always)"
    modified="$(git -C "$nccl_src" status --porcelain --untracked-files=no | wc -l | tr -d ' ')"
    [[ "$tag" == "$NCCL_TAG" ]] || die "${nccl_src} is at ${tag}, expected ${NCCL_TAG}"
    (( modified == 0 )) || die "${nccl_src} has ${modified} modified file(s); this needs the unpatched tree"
    ok "source ${nccl_src} is ${tag} with no modified files"
else
    note "${nccl_src} is not a git checkout; cannot prove it is unpatched"
fi

################################################################################
# 2. Source

if [[ -d "${dest}/.git" ]]; then
    if $update; then
        echo "=== Updating ${dest}"
        git -C "$dest" fetch --depth 1 origin HEAD
        git -C "$dest" checkout --detach FETCH_HEAD
    else
        echo "=== Reusing ${dest} (pass --update for the latest commit)"
    fi
elif [[ -e "$dest" ]]; then
    die "${dest} exists but is not a git checkout; move it away or pass --dest"
else
    echo "=== Cloning nccl-tests into ${dest}"
    mkdir -p "$(dirname -- "$dest")"
    git clone --depth 1 "$NCCL_TESTS_REPO" "$dest"
fi
echo "nccl-tests: $(git -C "$dest" log -1 --format='%h %cd' --date=short)"

################################################################################
# 3. Build

build() {
    local build_dir="$1"; shift
    echo "=== Building ${build_dir} ($*, make -j${jobs})"
    # The build directory is derived from its sources only, so rebuild from scratch: a reused
    # checkout may have been built against another NCCL.
    rm -rf "$build_dir"
    if ! make -C "$dest" -j"$jobs" BUILDDIR="$build_dir" CUDA_HOME="$cuda_home" NCCL_HOME="$nccl_home" \
        NVCC_GENCODE="$GENCODE" "$@" >"${build_dir}.log" 2>&1; then
        tail -n 40 "${build_dir}.log" >&2
        die "build failed; full log in ${build_dir}.log"
    fi
    [[ -x "${build_dir}/all_gather_perf" ]] || die "${build_dir}/all_gather_perf was not built"
    local resolved
    resolved="$(LD_LIBRARY_PATH="${nccl_home}/lib" ldd "${build_dir}/all_gather_perf" | awk '/libnccl/ {print $3}')"
    [[ "$(readlink -f "$resolved")" == "$(readlink -f "${nccl_home}/lib/libnccl.so")" ]] \
        || die "${build_dir}/all_gather_perf loads ${resolved:-no libnccl}, not ${nccl_home}/lib"
    ok "${build_dir}/all_gather_perf loads ${nccl_home}/lib/libnccl.so"
}

build "${dest}/build" MPI=0

if [[ -z "$mpi_home" ]] && command -v mpicc >/dev/null; then
    mpi_home="$(cd -- "$(dirname -- "$(readlink -f "$(command -v mpicc)")")/.." && pwd)"
fi
if [[ -n "$mpi_home" && -f "${mpi_home}/include/mpi.h" ]]; then
    echo "MPI: ${mpi_home}"
    build "${dest}/build-mpi" MPI=1 MPI_HOME="$mpi_home"
else
    note "no MPI found (mpicc not in PATH, no --mpi-home); skipped build-mpi"
    note "run_ncu_all_gather_stmc.sh needs it: one process per GPU is only possible through MPI"
fi

echo "=== nccl-tests ready: ${dest}"
echo "  LD_LIBRARY_PATH=${nccl_home}/lib ${dest}/build/all_gather_perf -b 256M -e 256M -g 4"
[[ -x "${dest}/build-mpi/all_gather_perf" ]] \
    && echo "  LD_LIBRARY_PATH=${nccl_home}/lib mpirun -np 4 ${dest}/build-mpi/all_gather_perf -b 256M -e 256M -g 1"
echo "  bash examples/gb200/nccl-tests/run_ncu_all_gather_stmc.sh"
