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

usage() {
    cat <<'USAGE'
Build NCCL 2.32.3-rank0-only from the stock v2.32.3-1 tag for GB200.
Set NCCL_SYM_MULTIMEM_RANK0_ONLY=1 on every rank to limit AllGather_STMC
and ReduceScatter_LDMC payload operations to communicator rank 0. All ranks
still participate in both barriers. The default 0 runs the full collectives.

Usage: install-nccl-2.32.3-rank0-only.sh [options]
  --dest DIR      Source tree (default: nccl/nccl-2.32.3-rank0-only under the repo)
  --cuda-home DIR CUDA toolkit >= 12.8 (default: CUDA_HOME or /usr/local/cuda)
  --jobs N        Parallel build jobs (default: CPU affinity count)
  --check-only    Verify the patch and an existing build
  -h, --help      Show this help
USAGE
}

die() { echo "ERROR: $*" >&2; exit 1; }

nccl_tag=v2.32.3-1
nccl_repo=https://github.com/NVIDIA/nccl.git
suffix=-rank0-only
rank0_env=NCCL_SYM_MULTIMEM_RANK0_ONLY

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
patch="${script_dir}/patches/nccl-2.32.3-rank0-only.patch"
dest="${repo_root}/nccl/nccl-2.32.3-rank0-only"
cuda_home="${CUDA_HOME:-/usr/local/cuda}"
jobs="$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc 2>/dev/null || echo 16)"
check_only=false

while (( $# )); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --check-only) check_only=true; shift ;;
        --dest|--cuda-home|--jobs)
            [[ -n "${2:-}" && "$2" != --* ]] || die "$1 needs a value"
            case "$1" in
                --dest) dest="$2" ;;
                --cuda-home) cuda_home="$2" ;;
                --jobs) jobs="$2" ;;
            esac
            shift 2 ;;
        *) die "unknown argument: $1 (see --help)" ;;
    esac
done

[[ -f "$patch" ]] || die "${patch} not found"
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || die "--jobs must be a positive integer"
[[ "$dest" = /* ]] || dest="${PWD}/${dest}"
[[ "$cuda_home" = /* ]] || cuda_home="${PWD}/${cuda_home}"

if [[ ! -e "$dest" ]] && ! $check_only; then
    echo "=== Cloning NCCL ${nccl_tag} into ${dest}"
    mkdir -p -- "$(dirname -- "$dest")"
    git clone --depth 1 --branch "$nccl_tag" "$nccl_repo" "$dest"
fi
[[ -e "${dest}/.git" ]] || die "${dest} is not an NCCL git checkout"
expected="$(git -C "$dest" rev-parse --verify "refs/tags/${nccl_tag}^{commit}" 2>/dev/null)" \
    || die "${dest} does not have the ${nccl_tag} tag"
[[ "$(git -C "$dest" rev-parse HEAD)" == "$expected" ]] || die "${dest} must be at ${nccl_tag}"

if git -C "$dest" apply --reverse --check "$patch" 2>/dev/null; then
    echo "=== ${patch##*/} is applied"
elif $check_only; then
    die "${dest} does not carry ${patch##*/}; run without --check-only"
else
    git -C "$dest" apply --check "$patch" \
        || die "${patch##*/} does not apply to ${dest}; use a stock ${nccl_tag} checkout"
    git -C "$dest" apply "$patch"
    echo "=== Applied ${patch##*/}"
fi
git -C "$dest" diff --stat | sed 's/^/  /'

args=(--dest "$dest" --cuda-home "$cuda_home" --jobs "$jobs")
if $check_only; then
    args+=(--check-only)
fi
"${script_dir}/install-nccl-2.32.3.sh" "${args[@]}"

build_dir="${dest}/build"
lib="${build_dir}/lib/libnccl.so"
grep -Fqx "#define NCCL_SUFFIX \"${suffix}\"" "${build_dir}/include/nccl.h" \
    || die "nccl.h does not define NCCL_SUFFIX \"${suffix}\"; rebuild without --check-only"
grep -aqF "NCCL version 2.32.3${suffix}+cuda" "$lib" \
    || die "${lib} lacks the rank0-only version banner; rebuild without --check-only"
grep -aqF "$rank0_env" "$lib" || die "${lib} does not read ${rank0_env}; rebuild without --check-only"
echo "=== ${lib} is NCCL 2.32.3${suffix} with ${rank0_env}"
