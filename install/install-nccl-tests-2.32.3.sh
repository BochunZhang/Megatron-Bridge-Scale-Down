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
#   3. Build it for sm_100 twice (each is skipped when an earlier build used the same nccl-tests
#      commit, NCCL, CUDA and make vars, recorded in <build dir>/.build-stamp):
#        <nccl-tests>/build      MPI=0: one process drives all GPUs (-g / -t)
#        <nccl-tests>/build-mpi  MPI=1: one process per GPU (mpirun -np N ... -g 1); needed by
#                                 run_ncu_all_gather_stmc.sh. Skipped with a warning without MPI.
#   nccl-tests only has an $ORIGIN rpath, so NCCL is found through LD_LIBRARY_PATH=<nccl-home>/lib.
#
# Usage:
#   ./install/install-nccl-tests-2.32.3.sh [--nccl-home DIR] [--dest DIR] [--mpi-home DIR]
#                                          [--mpi-include DIR] [--mpi-lib DIR]
#                                          [--cuda-home DIR] [--jobs N] [--update] [--rebuild]
#
#   --nccl-home DIR NCCL 2.32.3 build (default: <megatron-bridge>/nccl/nccl-2.32.3/build)
#   --dest DIR      nccl-tests checkout (default: <megatron-bridge>/nccl/nccl-tests)
#   --mpi-home DIR  MPI prefix with include/mpi.h and lib/libmpi.so (default: ask mpicc in PATH for
#                   its include / library directories, then try the libopenmpi-dev layout)
#   --mpi-include DIR  Directory with mpi.h, when it is not under the prefix
#                   (e.g. /usr/lib/aarch64-linux-gnu/openmpi/include)
#   --mpi-lib DIR   Directory with libmpi.so (e.g. /usr/lib/aarch64-linux-gnu)
#   --cuda-home DIR CUDA toolkit, >= 12.8 for sm_100 (default: CUDA_HOME or /usr/local/cuda)
#   --jobs N        make -j (default: nproc)
#   --update        Fetch the latest nccl-tests commit into an existing checkout before building
#   --rebuild       Rebuild even if a build matches its stamp (commit, NCCL, CUDA, make vars)

set -euo pipefail

NCCL_TESTS_REPO="https://github.com/NVIDIA/nccl-tests.git"
NCCL_TAG="v2.32.3-1"
GENCODE="-gencode=arch=compute_100,code=sm_100"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
nccl_home="${repo_root}/nccl/nccl-2.32.3/build"
dest="${repo_root}/nccl/nccl-tests"
mpi_home="${MPI_HOME:-}"
mpi_include_arg=""
mpi_lib_arg=""
cuda_home="${CUDA_HOME:-/usr/local/cuda}"
# GNU nproc honours OMP_NUM_THREADS, which training environments usually export as 1.
jobs="$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc 2>/dev/null || echo 16)"
update=false
rebuild=false

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
        --mpi-include) mpi_include_arg="$2"; shift 2 ;;
        --mpi-lib) mpi_lib_arg="$2"; shift 2 ;;
        --cuda-home) cuda_home="$2"; shift 2 ;;
        --jobs) jobs="$2"; shift 2 ;;
        --update) update=true; shift ;;
        --rebuild) rebuild=true; shift ;;
        -h|--help) sed -n '16,45p' "$0"; exit 0 ;;
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

check_nccl_link() {
    local bin="$1" resolved
    resolved="$(LD_LIBRARY_PATH="${nccl_home}/lib" ldd "$bin" | awk '/libnccl/ {print $3}')"
    [[ -n "$resolved" && "$(readlink -f "$resolved")" == "$(readlink -f "${nccl_home}/lib/libnccl.so")" ]]
}

# build <dir> <make vars...>: a build is reused when its stamp (nccl-tests commit, NCCL, CUDA,
# gencode, make vars) matches and all_gather_perf still loads this NCCL; --rebuild forces it.
build() {
    local build_dir="$1"; shift
    local stamp="${build_dir}/.build-stamp" want
    want="$(printf '%s\n' "nccl-tests $(git -C "$dest" rev-parse HEAD)" \
        "nccl ${nccl_home} $(stat -c %Y "${nccl_home}/lib/libnccl.so" 2>/dev/null || echo ?)" \
        "cuda ${cuda_home}" "gencode ${GENCODE}" "make $*")"
    if ! $rebuild && [[ -x "${build_dir}/all_gather_perf" && -f "$stamp" && "$(cat "$stamp")" == "$want" ]] \
        && check_nccl_link "${build_dir}/all_gather_perf"; then
        ok "reusing ${build_dir} (same nccl-tests commit, NCCL and make vars; --rebuild to force)"
        return
    fi
    # Builds from before the stamp existed: adopt one that loads this NCCL, unless --update may
    # have moved the checkout past the commit it was built from.
    if ! $rebuild && ! $update && [[ -x "${build_dir}/all_gather_perf" && ! -f "$stamp" ]] \
        && check_nccl_link "${build_dir}/all_gather_perf"; then
        echo "$want" >"$stamp"
        ok "reusing ${build_dir} (built before stamps; it loads ${nccl_home}/lib; --rebuild to force)"
        return
    fi
    echo "=== Building ${build_dir} ($*, make -j${jobs})"
    # Rebuild from scratch: make only tracks the nccl-tests sources, not which NCCL or MPI they
    # were built against.
    rm -rf "$build_dir"
    if ! make -C "$dest" -j"$jobs" BUILDDIR="$build_dir" CUDA_HOME="$cuda_home" NCCL_HOME="$nccl_home" \
        NVCC_GENCODE="$GENCODE" "$@" >"${build_dir}.log" 2>&1; then
        tail -n 40 "${build_dir}.log" >&2
        die "build failed; full log in ${build_dir}.log"
    fi
    [[ -x "${build_dir}/all_gather_perf" ]] || die "${build_dir}/all_gather_perf was not built"
    check_nccl_link "${build_dir}/all_gather_perf" \
        || die "${build_dir}/all_gather_perf does not load ${nccl_home}/lib/libnccl.so"
    ok "${build_dir}/all_gather_perf loads ${nccl_home}/lib/libnccl.so"
    echo "$want" >"$stamp"
}

build "${dest}/build" MPI=0

# The MPI include and library directories, first match wins:
#   --mpi-include / --mpi-lib, then --mpi-home, then what mpicc in PATH reports, then the
#   Debian/Ubuntu libopenmpi-dev layout (/usr/lib/<arch>-linux-gnu/openmpi/include, libmpi.so in
#   /usr/lib/<arch>-linux-gnu). <mpicc>/.. is not always the prefix, so the wrapper is asked.
mpi_incdirs=() mpi_libdirs=() mpi_sources=()
[[ -n "$mpi_include_arg" ]] && { mpi_incdirs+=("$mpi_include_arg"); mpi_sources+=("--mpi-include"); }
[[ -n "$mpi_lib_arg" ]] && { mpi_libdirs+=("$mpi_lib_arg"); mpi_sources+=("--mpi-lib"); }
if [[ -n "$mpi_home" ]]; then
    mpi_incdirs+=("${mpi_home}/include") mpi_libdirs+=("${mpi_home}/lib" "${mpi_home}/lib64")
    mpi_sources+=("--mpi-home / MPI_HOME=${mpi_home}")
fi
if command -v mpicc >/dev/null; then
    mpicc_bin="$(command -v mpicc)"
    if incs="$(mpicc --showme:incdirs 2>/dev/null)"; then
        # Open MPI / HPC-X
        read -r -a dirs <<<"$incs"; mpi_incdirs+=("${dirs[@]}")
        read -r -a dirs <<<"$(mpicc --showme:libdirs 2>/dev/null)"; mpi_libdirs+=("${dirs[@]}")
    else
        # MPICH / Intel MPI: "-show" prints the compiler command line with -I / -L flags.
        for flag in $(mpicc -show 2>/dev/null); do
            case "$flag" in
                -I*) mpi_incdirs+=("${flag#-I}") ;;
                -L*) mpi_libdirs+=("${flag#-L}") ;;
            esac
        done
    fi
    prefix="$(cd -- "$(dirname -- "$(readlink -f "$mpicc_bin")")/.." && pwd)"
    mpi_incdirs+=("${prefix}/include") mpi_libdirs+=("${prefix}/lib" "${prefix}/lib64")
    mpi_sources+=("$mpicc_bin")
fi
multiarch="$(uname -m)-linux-gnu"
mpi_incdirs+=("/usr/lib/${multiarch}/openmpi/include" /usr/include/openmpi /usr/include/mpich /usr/include)
mpi_libdirs+=("/usr/lib/${multiarch}" "/usr/lib/${multiarch}/openmpi/lib")
mpi_include="" mpi_lib=""
for dir in "${mpi_incdirs[@]}"; do
    [[ -n "$dir" && -f "${dir}/mpi.h" ]] && { mpi_include="$dir"; break; }
done
for dir in "${mpi_libdirs[@]}"; do
    [[ -n "$dir" && -e "${dir}/libmpi.so" ]] && { mpi_lib="$dir"; break; }
done
if [[ -z "$mpi_include" && -z "$mpi_lib" && ${#mpi_sources[@]} -eq 0 ]]; then
    note "no MPI found (no mpicc in PATH, no --mpi-home / --mpi-include, no libopenmpi-dev); skipped build-mpi"
    note "run_ncu_all_gather_stmc.sh needs it: one process per GPU is only possible through MPI"
elif [[ -z "$mpi_include" ]]; then
    die "no mpi.h in: ${mpi_incdirs[*]}; pass --mpi-include <dir with mpi.h>"
elif [[ -z "$mpi_lib" ]]; then
    die "no libmpi.so in: ${mpi_libdirs[*]}; pass --mpi-lib <dir with libmpi.so>"
else
    # nccl-tests compiles with -I$(MPI_INCLUDE) and links -L$(MPI_HOME)/lib{,64} -lmpi. MPI_HOME is
    # set to the parent of the library directory, and the directory itself also reaches the linker
    # through LIBRARY_PATH: overriding NVLDFLAGS would drop the Makefile's own flags.
    echo "MPI: mpi.h in ${mpi_include}, libmpi.so in ${mpi_lib}"
    LIBRARY_PATH="${mpi_lib}${LIBRARY_PATH:+:${LIBRARY_PATH}}" \
        build "${dest}/build-mpi" MPI=1 MPI_HOME="$(dirname -- "$mpi_lib")" MPI_INCLUDE="$mpi_include"
    LD_LIBRARY_PATH="${nccl_home}/lib" ldd "${dest}/build-mpi/all_gather_perf" | grep -q "libmpi.*=> /" \
        && ok "${dest}/build-mpi/all_gather_perf links $(LD_LIBRARY_PATH="${nccl_home}/lib" ldd "${dest}/build-mpi/all_gather_perf" | awk '/libmpi\./ {print $3; exit}')" \
        || note "libmpi does not resolve at run time; add ${mpi_lib} to LD_LIBRARY_PATH"
    # all_gather_perf must be started by the mpirun of the MPI it links.
    if command -v mpirun >/dev/null; then
        echo "  mpirun in PATH: $(command -v mpirun) ($(mpirun --version 2>&1 | head -1))"
    fi
fi

echo "=== nccl-tests ready: ${dest}"
echo "  LD_LIBRARY_PATH=${nccl_home}/lib ${dest}/build/all_gather_perf -b 256M -e 256M -g 4"
[[ -x "${dest}/build-mpi/all_gather_perf" ]] \
    && echo "  LD_LIBRARY_PATH=${nccl_home}/lib mpirun -np 4 ${dest}/build-mpi/all_gather_perf -b 256M -e 256M -g 1"
echo "  bash examples/gb200/nccl-tests/run_ncu_all_gather_stmc.sh"
