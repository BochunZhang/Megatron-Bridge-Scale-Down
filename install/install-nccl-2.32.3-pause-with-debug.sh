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

# NCCL 2.32.3 with the multicast AllGather pause and a clock64 timing of its multimem.st
# (nccl-2.32.3-pause-with-debug).
#
# install/patches/nccl-2.32.3-pause-with-debug.patch on stock v2.32.3-1 (not on the pause or cta patch; the CTA
# selection stays NCCL's own) carries the pause of nccl-2.32.3-pause and adds the timing:
#   NCCL_SYM_AG_STMC_PAUSE_CYCLES       (default 0) SM cycles every AllGather_STMC / AllGather_TmaSTMC warp spins
#                                       after each 4 KiB chunk of multimem stores, as in nccl-2.32.3-pause.
#   NCCL_SYM_AG_STMC_ST_PRINT_INTERVAL  (default 0 = off) AllGather_STMC reads clock64 right before and right
#                                       after the 8 multimem.st of each chunk; lane 0 of every warp prints one
#                                       "[nccl-ag-stmc-st] ..." line (sum / min / max / last cycles) per this many
#                                       chunks, plus one for the rest. A countdown, not a division, picks them.
#   NCCL_SYM_AG_STMC_ST_PRINT_SKIP      (default 0) AllGather_STMC launches of each communicator that print
#                                       nothing first (the warmup).
#   NCCL_SYM_AG_STMC_ST_PRINT_LAUNCHES  (default 1) launches printed after those; 0 prints every later launch.
# All four are read at run time (once per process), so one build serves every value. The patch sets
# NCCL_SUFFIX := -pause-with-debug in makefiles/version.mk: NCCL_DEBUG=VERSION prints
# "NCCL version 2.32.3-pause-with-debug+cudaX.Y"; NCCL_VERSION_CODE (23203), ncclGetVersion() and the soname
# libnccl.so.2 stay those of 2.32.3. With every variable at 0 the kernels still carry the printf call (more
# registers), so pause-only sweeps keep using nccl-2.32.3-pause.
#
#   1. Clone tag v2.32.3-1 into <megatron-bridge>/nccl/nccl-2.32.3-pause-with-debug. An existing checkout is
#      reused only if it is at that tag.
#   2. Apply the patch (skipped when it is already applied).
#   3. Build and check with install-nccl-2.32.3.sh --dest <that tree>: make -j<jobs> src.build for
#      sm_100 into <tree>/build, then the same kernel / symbol / environment checks.
#
# The AllGather test preloads it into the torch of install-torch-2.13.0-nccl-2.32.3.sh
# (examples/gb200/nvls-collective/run_ag_st_debug.sh).
#
# Usage:
#   ./install/install-nccl-2.32.3-pause-with-debug.sh [--dest DIR] [--cuda-home DIR] [--jobs N] [--check-only]
#
#   --dest DIR      Source tree (default: <megatron-bridge>/nccl/nccl-2.32.3-pause-with-debug)
#   --cuda-home DIR CUDA toolkit, >= 12.8 for sm_100 (default: CUDA_HOME or /usr/local/cuda)
#   --jobs N        make -j (default: every CPU in the affinity mask)
#   --check-only    Skip clone, patch and build; only check the patch and the existing build

set -euo pipefail

NCCL_TAG="v2.32.3-1"
NCCL_REPO="https://github.com/NVIDIA/nccl.git"
NCCL_SUFFIX="-pause-with-debug"  # set by the patch in makefiles/version.mk
DEBUG_ENVS=(NCCL_SYM_AG_STMC_PAUSE_CYCLES NCCL_SYM_AG_STMC_ST_PRINT_INTERVAL NCCL_SYM_AG_STMC_ST_PRINT_SKIP
    NCCL_SYM_AG_STMC_ST_PRINT_LAUNCHES)

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
patch="${script_dir}/patches/nccl-2.32.3-pause-with-debug.patch"
dest="${repo_root}/nccl/nccl-2.32.3-pause-with-debug"
cuda_home="${CUDA_HOME:-/usr/local/cuda}"
# GNU nproc returns OMP_NUM_THREADS when it is set, and training environments usually export
# OMP_NUM_THREADS=1; drop it (and OMP_THREAD_LIMIT) so only the CPU affinity mask counts.
jobs="$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc 2>/dev/null || echo 16)"
check_only=false

die() {
    echo "ERROR: $*" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dest) dest="$2"; shift 2 ;;
        --cuda-home) cuda_home="$2"; shift 2 ;;
        --jobs) jobs="$2"; shift 2 ;;
        --check-only) check_only=true; shift ;;
        -h|--help) sed -n '16,52p' "$0"; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

[[ -f "$patch" ]] || die "${patch} not found"
(( jobs >= 1 )) || die "--jobs must be >= 1, got ${jobs}"

################################################################################
# 1. Source

if ! $check_only; then
    if [[ -d "${dest}/.git" ]]; then
        [[ "$(git -C "$dest" describe --tags --exact-match 2>/dev/null)" == "$NCCL_TAG" ]] \
            || die "${dest} is not at ${NCCL_TAG}; move it away or pass --dest"
        echo "=== Reusing ${dest}"
    elif [[ -e "$dest" ]]; then
        die "${dest} exists but is not a git checkout; move it away or pass --dest"
    else
        echo "=== Cloning NCCL ${NCCL_TAG} into ${dest}"
        mkdir -p "$(dirname -- "$dest")"
        git clone --depth 1 --branch "$NCCL_TAG" "$NCCL_REPO" "$dest"
    fi
fi

################################################################################
# 2. Patch

if git -C "$dest" apply --reverse --check "$patch" 2>/dev/null; then
    echo "=== ${patch##*/} is applied"
elif $check_only; then
    die "${dest} does not carry ${patch##*/}; run without --check-only"
else
    # A tree that carries nccl-2.32.3-pause.patch (or the cta patch) does not take this one: it replaces both hunks.
    git -C "$dest" apply --check "$patch" \
        || die "${patch##*/} does not apply to ${dest}; reset it with git -C ${dest} checkout -- ."
    git -C "$dest" apply "$patch"
    echo "=== Applied ${patch##*/}"
fi
git -C "$dest" diff --stat | sed 's/^/  /'
# The CTA selection must stay NCCL's own: the cta patch's lsa.cc / gin.cc hunks would let minCTAs lift it.
if git -C "$dest" diff --name-only | grep -q "^src/tuning/"; then
    die "${dest} changes src/tuning (the cta patch?); reset it with git -C ${dest} checkout -- . and rerun"
fi

################################################################################
# 3. Build (make -j${jobs}) and checks, shared with the stock 2.32.3 build

args=(--dest "$dest" --cuda-home "$cuda_home" --jobs "$jobs")
$check_only && args+=(--check-only)
"${script_dir}/install-nccl-2.32.3.sh" "${args[@]}"

echo "=== Version suffix ${NCCL_SUFFIX}"
build_dir="${dest}/build"
grep -q "^#define NCCL_SUFFIX \"${NCCL_SUFFIX}\"" "${build_dir}/include/nccl.h" \
    || die "${build_dir}/include/nccl.h does not define NCCL_SUFFIX \"${NCCL_SUFFIX}\"; rebuild without --check-only"
# The banner is a string literal (init.cc VERSION_STRING), so the library itself tells the builds apart.
grep -aq "NCCL version 2.32.3${NCCL_SUFFIX}+cuda" "${build_dir}/lib/libnccl.so" \
    || die "${build_dir}/lib/libnccl.so has no \"NCCL version 2.32.3${NCCL_SUFFIX}+cuda\" banner; rebuild without --check-only"
echo "  [ok]   nccl.h NCCL_SUFFIX \"${NCCL_SUFFIX}\", libnccl.so banner \"NCCL version 2.32.3${NCCL_SUFFIX}+cuda...\""
# NCCL_PARAM keeps every variable name as a string literal for ncclGetEnv and the INFO line.
for name in "${DEBUG_ENVS[@]}"; do
    grep -aq "$name" "${build_dir}/lib/libnccl.so" \
        || die "${build_dir}/lib/libnccl.so does not read ${name}; rebuild without --check-only"
done
echo "  [ok]   libnccl.so reads ${DEBUG_ENVS[*]}"
# The format string of the device printf sits in the kernels' fatbin, which nvcc may compress; only warn.
if grep -aq "\[nccl-ag-stmc-st\]" "${build_dir}/lib/libnccl.so"; then
    echo "  [ok]   libnccl.so carries the \"[nccl-ag-stmc-st]\" printf format"
else
    echo "  [warn] no \"[nccl-ag-stmc-st]\" string in libnccl.so (compressed fatbin?); run_ag_st_debug.sh checks the output"
fi

echo "=== nccl-2.32.3-pause-with-debug: stock 2.32.3, plus the AllGather_STMC pause and multimem.st timing"
echo "  test: bash examples/gb200/nvls-collective/run_ag_st_debug.sh"
