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

# NCCL 2.32.3 with a pause in the multicast and unicast AllGathers (nccl-2.32.3-pause).
#
# install/patches/nccl-2.32.3-pause.patch on stock v2.32.3-1 (not on the cta patch: the CTA selection
# stays NCCL's own) adds NCCL_SYM_AG_STMC_PAUSE_CYCLES (default 0). AllGather_STMC / AllGather_TmaSTMC
# then make every warp spin that many SM cycles (clock64) after each 4 KiB chunk of multimem stores,
# which slows the multicast AllGather without changing its CTA count or its result. NCCL_SYM_AG_ST_PAUSE_CYCLES
# (default 0) does the same for the unicast AllGather_ST / AllGather_TmaST: every warp spins after each tile it
# has stored to every peer (bcastDeep, 2 KiB per peer on the 16 B path). In ns a pause is
# cycles / SM clock (nvidia-smi -lgc pins it). The value is read at run time (once per process), so one
# build serves every pause. The patch also sets NCCL_SUFFIX := -pause in makefiles/version.mk, so
# NCCL_DEBUG=VERSION prints "NCCL version 2.32.3-pause+cudaX.Y" and nccl.h defines NCCL_SUFFIX "-pause";
# NCCL_VERSION_CODE (23203), ncclGetVersion() and the soname libnccl.so.2 stay those of 2.32.3.
#
#   1. Clone tag v2.32.3-1 into <megatron-bridge>/nccl/nccl-2.32.3-pause. An existing checkout is reused
#      only if it is at that tag.
#   2. Apply the patch (skipped when it is already applied).
#   3. Build and check with install-nccl-2.32.3.sh --dest <that tree>: make -j<jobs> src.build for
#      sm_100 into <tree>/build, then the same kernel / symbol / environment checks.
#
# The sweep preloads it into the torch of install-torch-2.13.0-nccl-2.32.3.sh
# (examples/gb200/nvls-collective/run_symm_cta_pause_sweep.sh, multicast) and the GEMM / AllGather HBM
# contention sweep (examples/gb200/hbm-conflict/run_hbm_conflict_benchmark.sh, unicast).
#
# A tree patched with an older version of the patch (multicast pause only) does not take this one;
# reset it with git -C <tree> checkout -- . and rerun.
#
# Usage:
#   ./install/install-nccl-2.32.3-pause.sh [--fsdp-multicast] [--dest DIR] [--cuda-home DIR] [--jobs N] [--check-only]
#
#   --fsdp-multicast  Add the CTA patch and FSDP-scoped kernel selection and pause.
#                     Defaults to <megatron-bridge>/nccl/nccl-2.32.3-fsdp-multicast.
#   --dest DIR      Source tree (default: <megatron-bridge>/nccl/nccl-2.32.3-pause)
#   --cuda-home DIR CUDA toolkit, >= 12.8 for sm_100 (default: CUDA_HOME or /usr/local/cuda)
#   --jobs N        make -j (default: every CPU in the affinity mask)
#   --check-only    Skip clone, patch and build; only check the patch and the existing build

set -euo pipefail

NCCL_TAG="v2.32.3-1"
NCCL_REPO="https://github.com/NVIDIA/nccl.git"
NCCL_SUFFIX="-pause"  # set by the patch in makefiles/version.mk
PAUSE_ENV="NCCL_SYM_AG_STMC_PAUSE_CYCLES"
ST_PAUSE_ENV="NCCL_SYM_AG_ST_PAUSE_CYCLES"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
patch="${script_dir}/patches/nccl-2.32.3-pause.patch"
dest=""
fsdp_multicast=false
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
        --fsdp-multicast) fsdp_multicast=true; shift ;;
        --cuda-home) cuda_home="$2"; shift 2 ;;
        --jobs) jobs="$2"; shift 2 ;;
        --check-only) check_only=true; shift ;;
        -h|--help) sed -n '16,48p' "$0"; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

if $fsdp_multicast; then
    NCCL_SUFFIX="-cta-pause-fsdp"
    dest="${dest:-${repo_root}/nccl/nccl-2.32.3-fsdp-multicast}"
else
    dest="${dest:-${repo_root}/nccl/nccl-2.32.3-pause}"
fi

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

apply_patch_file() {
    local file="$1"
    shift
    if git -C "$dest" apply "$@" --reverse --check "$file" 2>/dev/null; then
        echo "=== ${file##*/} is applied"
    elif $check_only; then
        die "${dest} does not carry ${file##*/}; run without --check-only"
    else
        git -C "$dest" apply "$@" --check "$file" \
            || die "${file##*/} does not apply to ${dest}; use a fresh --dest"
        git -C "$dest" apply "$@" "$file"
        echo "=== Applied ${file##*/}"
    fi
}

if $fsdp_multicast; then
    scoped_patch="${script_dir}/patches/nccl-2.32.3-fsdp-multicast.patch"
    # The scoped patch changes overlapping hunks in the pause patch, including its version suffix.
    if ! git -C "$dest" apply --reverse --check "$scoped_patch" 2>/dev/null; then
        apply_patch_file "$patch"
    fi
    apply_patch_file "${script_dir}/patches/nccl-2.32.3-cta.patch" --exclude=makefiles/version.mk
    apply_patch_file "$scoped_patch"
else
    apply_patch_file "$patch"
fi
git -C "$dest" diff --stat | sed 's/^/  /'
# The CTA selection must stay NCCL's own: the cta patch's lsa.cc / gin.cc hunks would let minCTAs lift it.
if ! $fsdp_multicast && git -C "$dest" diff --name-only | grep -q "^src/tuning/"; then
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
# NCCL_PARAM keeps the variable name as a string literal for ncclGetEnv and the INFO line.
grep -aq "$PAUSE_ENV" "${build_dir}/lib/libnccl.so" \
    || die "${build_dir}/lib/libnccl.so does not read ${PAUSE_ENV}; rebuild without --check-only"
echo "  [ok]   libnccl.so reads ${PAUSE_ENV}"
grep -aq "$ST_PAUSE_ENV" "${build_dir}/lib/libnccl.so" \
    || die "${build_dir}/lib/libnccl.so does not read ${ST_PAUSE_ENV} (built from an older patch?); rebuild without --check-only"
echo "  [ok]   libnccl.so reads ${ST_PAUSE_ENV}"
if $fsdp_multicast; then
    grep -aq 'ncclSetFsdpMulticast' "${build_dir}/lib/libnccl.so" \
        || die "${build_dir}/lib/libnccl.so lacks FSDP-scoped kernel selection; rebuild without --check-only"
    echo "=== nccl-2.32.3-fsdp-multicast: FSDP-scoped kernel selection and AG pause"
    exit 0
fi

echo "=== nccl-2.32.3-pause: stock 2.32.3, plus ${PAUSE_ENV}=<cycles> slowing AllGather_STMC"
echo "    and ${ST_PAUSE_ENV}=<cycles> slowing AllGather_ST"
echo "  sweeps: bash examples/gb200/nvls-collective/run_symm_cta_pause_sweep.sh"
echo "          bash examples/gb200/hbm-conflict/run_hbm_conflict_benchmark.sh"
