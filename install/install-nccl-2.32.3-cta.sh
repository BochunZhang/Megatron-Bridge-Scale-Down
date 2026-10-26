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

# NCCL 2.32.3 with the multicast symmetric kernels' CTA cap lifted (nccl-2.32.3-cta).
#
# Stock NCCL caps AllGather_STMC / AllGather_TmaSTMC / ReduceScatter_LDMC / AllReduce_RSxLDMC_AGxSTMC
# at divUp(32, nRanks) CTAs on sm_100 (src/tuning/sym_model/lsa.cc maxBlocksLsa(), and the same cap for
# AllGather_RailRing_LsaSTMC in gin.cc). minCTAs/maxCTAs from ncclConfig_t cannot go above it; only the
# process-wide NCCL_SYM_CTAS can. install/patches/nccl-2.32.3-cta.patch keeps the cap as the bound of
# NCCL's own choice, but lets a minCTAs above it (ncclConfig_t, per-call config or NCCL_MIN_CTAS) raise
# the bound up to maxCTAs (at most ncclSymkMaxBlocks = 64, which the barriers and the device code are
# sized for). With the default minCTAs = 1 the selection is unchanged. The patch also sets
# NCCL_SUFFIX := -cta in makefiles/version.mk, so NCCL_DEBUG=VERSION prints "NCCL version 2.32.3-cta+cudaX.Y"
# and nccl.h defines NCCL_SUFFIX "-cta"; NCCL_VERSION_CODE (23203), ncclGetVersion() and the soname
# libnccl.so.2 stay those of 2.32.3.
#
# --with-pause builds nccl-2.32.3-cta-with-pause instead: install/patches/nccl-2.32.3-cta-pause.patch on
# top of the cta patch adds NCCL_SYM_AG_STMC_PAUSE_CYCLES (default 0). AllGather_STMC / AllGather_TmaSTMC then
# make every warp spin that many SM cycles (clock64) after each 4 KiB chunk of multimem stores, which slows
# the multicast AllGather without changing its CTA count or its result. In ns a pause is cycles / SM clock
# (nvidia-smi -lgc pins it). The value is read at run time (once per process), so one build serves every
# pause; the suffix is -cta-pause.
#
#   1. Clone tag v2.32.3-1 into <megatron-bridge>/nccl/nccl-2.32.3-cta (with --with-pause:
#      nccl-2.32.3-cta-with-pause). An existing checkout is reused only if it is at that tag.
#   2. Apply the patch(es) in order (skipped when they are already applied).
#   3. Build and check with install-nccl-2.32.3.sh --dest <that tree>: make -j<jobs> src.build for
#      sm_100 into <tree>/build, then the same kernel / symbol / environment checks.
#
# torch built against this NCCL: install-torch-2.13.0-nccl-2.32.3-cta.sh. The sweeps preload it into the
# torch of install-torch-2.13.0-nccl-2.32.3.sh instead (run_symm_cta_sweep.sh, run_symm_cta_pause_sweep.sh).
#
# Usage:
#   ./install/install-nccl-2.32.3-cta.sh [--with-pause] [--dest DIR] [--cuda-home DIR] [--jobs N] [--check-only]
#
#   --with-pause    Also apply the AllGather_STMC pause patch (default tree: nccl-2.32.3-cta-with-pause)
#   --dest DIR      Source tree (default: <megatron-bridge>/nccl/nccl-2.32.3-cta)
#   --cuda-home DIR CUDA toolkit, >= 12.8 for sm_100 (default: CUDA_HOME or /usr/local/cuda)
#   --jobs N        make -j (default: every CPU in the affinity mask)
#   --check-only    Skip clone, patch and build; only check the patch and the existing build

set -euo pipefail

NCCL_TAG="v2.32.3-1"
NCCL_REPO="https://github.com/NVIDIA/nccl.git"
NCCL_SUFFIX="-cta"  # set by the patch in makefiles/version.mk

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
patches=("${script_dir}/patches/nccl-2.32.3-cta.patch")
dest=""
with_pause=false
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
        --with-pause) with_pause=true; shift ;;
        --dest) dest="$2"; shift 2 ;;
        --cuda-home) cuda_home="$2"; shift 2 ;;
        --jobs) jobs="$2"; shift 2 ;;
        --check-only) check_only=true; shift ;;
        -h|--help) sed -n '16,52p' "$0"; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

if $with_pause; then
    # The pause patch's version.mk hunk sits on the cta patch's, so it only applies on top of it.
    patches+=("${script_dir}/patches/nccl-2.32.3-cta-pause.patch")
    NCCL_SUFFIX="-cta-pause"
    [[ -n "$dest" ]] || dest="${repo_root}/nccl/nccl-2.32.3-cta-with-pause"
fi
[[ -n "$dest" ]] || dest="${repo_root}/nccl/nccl-2.32.3-cta"
for patch in "${patches[@]}"; do
    [[ -f "$patch" ]] || die "${patch} not found"
done
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

patched() { git -C "$dest" apply --reverse --check "$1" 2>/dev/null; }
# A later patch reverses only on top of the earlier ones, and once it is applied the earlier ones no
# longer reverse on their own (the pause patch rewrites the cta patch's NCCL_SUFFIX line). So the last
# patch tells whether the tree is done; otherwise apply, in order, those that are not in yet.
last="${patches[${#patches[@]} - 1]}"
if patched "$last"; then
    for patch in "${patches[@]}"; do
        echo "=== ${patch##*/} is applied"
    done
elif $check_only; then
    die "${dest} does not carry ${last##*/}; run without --check-only"
else
    for patch in "${patches[@]}"; do
        if [[ "$patch" != "$last" ]] && patched "$patch"; then
            echo "=== ${patch##*/} is applied"
            continue
        fi
        git -C "$dest" apply --check "$patch" \
            || die "${patch##*/} does not apply to ${dest}; reset it with git -C ${dest} checkout -- ."
        git -C "$dest" apply "$patch"
        echo "=== Applied ${patch##*/}"
    done
fi
git -C "$dest" diff --stat | sed 's/^/  /'

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

if $with_pause; then
    # NCCL_PARAM keeps the variable name as a string literal for ncclGetEnv and the INFO line.
    grep -aq "NCCL_SYM_AG_STMC_PAUSE_CYCLES" "${build_dir}/lib/libnccl.so" \
        || die "${build_dir}/lib/libnccl.so does not read NCCL_SYM_AG_STMC_PAUSE_CYCLES; rebuild without --check-only"
    echo "  [ok]   libnccl.so reads NCCL_SYM_AG_STMC_PAUSE_CYCLES"
    echo "=== nccl-2.32.3-cta-with-pause: the cta build, plus NCCL_SYM_AG_STMC_PAUSE_CYCLES=<cycles> slowing AllGather_STMC"
    echo "  sweep: bash examples/gb200/nvls-collective/run_symm_cta_pause_sweep.sh"
else
    echo "=== nccl-2.32.3-cta: minCTAs above divUp(32, nRanks) now reaches the multicast symmetric kernels"
    echo "  torch against it: ./install/install-torch-2.13.0-nccl-2.32.3-cta.sh --python <env>/bin/python"
fi
