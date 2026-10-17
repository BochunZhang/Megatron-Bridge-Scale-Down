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

# NCCL 2.32.3 source + build for examples/gb200/nvls-collective-cpp.
#
#   1. Clone tag v2.32.3-1 into <megatron-bridge>/nccl/nccl-2.32.3. An existing checkout is reused
#      only if it is that version.
#   2. Build libnccl.so for sm_100 (GB200) into <nccl-2.32.3>/build. This does not touch the
#      libnccl.so that torch loads; the benchmark links this build through an rpath.
#   3. Check that the features the benchmark measures are compiled in and not disabled:
#      AllGather_TmaSTMC (TMA + multimem), AllGather_STMC, ReduceScatter_LDMC (multimem.ld_reduce),
#      the copy-engine AllGather and the per-call ncclCollConfig_t API.
#
# The torch side (which NCCL torch was built with / loads) is probe-torch-nccl-version.sh.
#
# Usage:
#   ./install/install-nccl-2.32.3.sh [--dest DIR] [--cuda-home DIR] [--jobs N] [--check-only]
#
#   --dest DIR      Source tree (default: <megatron-bridge>/nccl/nccl-2.32.3)
#   --cuda-home DIR CUDA toolkit, >= 12.8 for sm_100 (default: CUDA_HOME or /usr/local/cuda)
#   --jobs N        make -j (default: nproc)
#   --check-only    Skip clone and build; only run the checks on an existing build

set -euo pipefail

NCCL_TAG="v2.32.3-1"
NCCL_REPO="https://github.com/NVIDIA/nccl.git"
GENCODE="-gencode=arch=compute_100,code=sm_100"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
dest="${repo_root}/nccl/nccl-2.32.3"
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
        -h|--help) sed -n '16,35p' "$0"; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

build_dir="${dest}/build"
lib="${build_dir}/lib/libnccl.so"
failures=0
ok() { echo "  [ok]   $*"; }
bad() { echo "  [FAIL] $*"; failures=$(( failures + 1 )); }
note() { echo "  [warn] $*"; }

[[ -x "${cuda_home}/bin/nvcc" ]] || die "nvcc not found under ${cuda_home}; pass --cuda-home"
cuda_version="$("${cuda_home}/bin/nvcc" --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')"
echo "CUDA toolkit: ${cuda_home} (${cuda_version})"
cuda_major="${cuda_version%%.*}"
cuda_minor="${cuda_version#*.}"
if (( cuda_major < 12 || (cuda_major == 12 && cuda_minor < 8) )); then
    die "CUDA ${cuda_version} cannot build sm_100; NCCL's makefiles/common.mk adds Blackwell from CUDA 12.8"
fi

################################################################################
# 1. Source

if ! $check_only; then
    if [[ -d "${dest}/.git" ]]; then
        echo "=== Reusing ${dest}"
    elif [[ -e "$dest" ]]; then
        die "${dest} exists but is not a git checkout; move it away or pass --dest"
    else
        echo "=== Cloning NCCL ${NCCL_TAG} into ${dest}"
        mkdir -p "$(dirname -- "$dest")"
        git clone --depth 1 --branch "$NCCL_TAG" "$NCCL_REPO" "$dest"
    fi
fi
[[ -f "${dest}/makefiles/version.mk" ]] || die "${dest} is not an NCCL source tree"
version="$(awk -F':= *' '/^NCCL_MAJOR/{a=$2} /^NCCL_MINOR/{b=$2} /^NCCL_PATCH/{c=$2} END{print a"."b"."c}' "${dest}/makefiles/version.mk")"
[[ "$version" == "2.32.3" ]] || die "${dest} is NCCL ${version}, expected 2.32.3; pass another --dest"
if git -C "$dest" rev-parse --git-dir >/dev/null 2>&1; then
    echo "source: $(git -C "$dest" describe --tags --always 2>/dev/null) $(git -C "$dest" status --porcelain | wc -l | tr -d ' ') modified file(s)"
fi

################################################################################
# 2. Build (src.build only: the library and headers, no packaging)

if ! $check_only; then
    echo "=== Building libnccl.so for sm_100 (make -j${jobs})"
    if ! make -C "$dest" -j"$jobs" src.build \
        CUDA_HOME="$cuda_home" BUILDDIR="$build_dir" NVCC_GENCODE="$GENCODE" >"${dest}/build.log" 2>&1; then
        tail -n 40 "${dest}/build.log" >&2
        die "build failed; full log in ${dest}/build.log"
    fi
    grep -m1 "NVCC_GENCODE is" "${dest}/build.log" || true
fi
[[ -f "$lib" ]] || die "${lib} not found; build first (drop --check-only)"

################################################################################
# 3. Checks

echo "=== Source: the kernels and switches the benchmark relies on"
src="${dest}/src"
for kernel in AllGather_TmaSTMC AllGather_STMC ReduceScatter_LDMC; do
    grep -q "\"${kernel}\"" "${src}/include/sym_kernels.h" && ok "symmetric kernel ${kernel} is in ncclSymKernelStr" \
        || bad "symmetric kernel ${kernel} missing from src/include/sym_kernels.h"
done
grep -q 'NCCL_PARAM(SymTmaEnable, "SYM_TMA_ENABLE", 1)' "${src}/sym_kernels.cc" \
    && ok "NCCL_SYM_TMA_ENABLE defaults to 1 (sym_kernels.cc)" || bad "NCCL_SYM_TMA_ENABLE default changed"
grep -q 'NCCL_PARAM(WinEnable, "WIN_ENABLE", 1)' "${src}/init.cc" \
    && ok "NCCL_WIN_ENABLE defaults to 1 (init.cc)" || bad "NCCL_WIN_ENABLE default changed"
grep -q 'ncclResult_t ncclCeAllGather(' "${src}/ce_coll.cc" && grep -q 'CE_COLL_AG_MULTICAST_THRESHOLD' "${src}/ce_coll.cc" \
    && ok "copy-engine AllGather with the multicast/unicast switch (ce_coll.cc)" || bad "CE AllGather not found in ce_coll.cc"
grep -q '"SYMK_TmaSTMC"' "${src}/config/algorithm_registry.cc" \
    && ok "algSelection knows SYMK_TmaSTMC / SYMK_STMC / SYMK_LDMC" || bad "algorithm_registry.cc has no SYMK_TmaSTMC"

echo "=== Library: ${lib}"
for symbol in ncclAllGatherConfig ncclReduceScatterConfig ncclCommWindowRegister ncclMemAlloc ncclCommQueryProperties; do
    nm -D --defined-only "$lib" 2>/dev/null | grep -qw "$symbol" && ok "exports ${symbol}" || bad "does not export ${symbol}"
done
cuobjdump="${cuda_home}/bin/cuobjdump"
if [[ -x "$cuobjdump" ]]; then
    # The TMA/multimem kernel bodies are compiled under __CUDA_ARCH__ >= 900
    # (src/device/symmetric/generate.py); without sm_100 SASS the kernels exist but do nothing.
    "$cuobjdump" --list-elf "$lib" 2>/dev/null | grep -q "sm_100" && ok "fatbin holds sm_100 SASS" \
        || bad "no sm_100 SASS in ${lib}; rebuild with NVCC_GENCODE=${GENCODE}"
    symbols="$("$cuobjdump" --dump-elf-symbols "$lib" 2>/dev/null || true)"
    for kernel in AllGather_TmaSTMC AllGather_STMC ReduceScatter_LDMC_sum_bf16 ReduceScatter_LDMC_sum_f32; do
        grep -q "ncclSymkDevKernel_${kernel}" <<<"$symbols" && ok "device kernel ncclSymkDevKernel_${kernel}" \
            || bad "device kernel ncclSymkDevKernel_${kernel} not in the fatbin"
    done
else
    note "${cuobjdump} not found; skipped the SASS checks"
fi

echo "=== Runtime switches that would turn the measured paths off"
# NCCL also reads NCCL_CONF_FILE, ~/.nccl.conf and /etc/nccl.conf (src/misc/param.cc).
conf_files=("${NCCL_CONF_FILE:-}" "${HOME}/.nccl.conf" /etc/nccl.conf)
check_switch() {
    local name="$1" off="$2" why="$3" file
    if [[ "${!name:-}" == "$off" ]]; then
        bad "${name}=${off} in the environment: ${why}"
    fi
    for file in "${conf_files[@]}"; do
        [[ -n "$file" && -f "$file" ]] || continue
        if grep -Eq "^[[:space:]]*${name}[[:space:]]*=[[:space:]]*${off}[[:space:]]*$" "$file"; then
            bad "${name}=${off} in ${file}: ${why}"
        fi
    done
}
check_switch NCCL_SYM_TMA_ENABLE 0 "removes AllGather_TmaSTMC / ReduceScatter_TmaLD"
check_switch NCCL_WIN_ENABLE 0 "no symmetric windows, so no symmetric kernels and no CE collectives"
check_switch NCCL_CUMEM_ENABLE 0 "symmetricSupport requires cuMem (init.cc)"
check_switch NCCL_NVLS_ENABLE 0 "no multicast: STMC / TmaSTMC / LDMC and the CE multicast path are removed"
for name in NCCL_ALGO NCCL_SYM_KERNEL NCCL_CTA_POLICY; do
    [[ -n "${!name:-}" ]] && note "${name}=${!name} is set here; run_nvls_collective_cpp.sh unsets it per pass"
done

if command -v nvidia-smi >/dev/null; then
    echo "=== GPU / driver"
    driver_cuda="$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1)"
    caps="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sort -u | tr '\n' ' ')"
    echo "  driver CUDA version ${driver_cuda:-?}, compute capability ${caps:-?}"
    # ncclSymkTmaAvailable: minCompCap >= 100; ncclCeImplemented: driver >= 12.5.
    for cap in $caps; do
        [[ "${cap%%.*}" -ge 10 ]] && ok "compute capability ${cap} >= 10.0 (TMA kernels)" || bad "compute capability ${cap} < 10.0: no TMA kernels"
    done
    if [[ -n "$driver_cuda" ]]; then
        dmaj="${driver_cuda%%.*}" dmin="${driver_cuda#*.}"
        (( dmaj > 12 || (dmaj == 12 && ${dmin%%.*} >= 5) )) && ok "driver CUDA ${driver_cuda} >= 12.5 (CE collectives)" \
            || bad "driver CUDA ${driver_cuda} < 12.5: no CE collectives"
    fi
    nvidia-smi -q 2>/dev/null | grep -iE "^ *(Fabric|State|Status|Clique ID|Cluster UUID)" | sort | uniq -c | sed 's/^/  /' || true
else
    note "nvidia-smi not found; GPU / driver checks skipped (run this on the GB200 node)"
fi

echo
echo "Multicast support, the LSA team size and TMA shared-memory limits are only known once a"
echo "communicator exists; nvls_collective_bench checks them per group and stops with the reason."
if (( failures )); then
    die "${failures} check(s) failed"
fi
echo "=== NCCL 2.32.3 ready: NCCL_HOME=${build_dir}"
echo "  make -C examples/gb200/nvls-collective-cpp NCCL_HOME=${build_dir} CUDA_HOME=${cuda_home}"
