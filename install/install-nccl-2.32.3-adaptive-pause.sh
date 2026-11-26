#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

tag=v2.32.3-1
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
patch="${script_dir}/patches/nccl-2.32.3-adaptive-pause.patch"
dest="${repo_root}/nccl/nccl-2.32.3-adaptive-pause"
cuda_home="${CUDA_HOME:-/usr/local/cuda}"
jobs="$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc 2>/dev/null || echo 16)"
check_only=false

die() {
    echo "ERROR: $*" >&2
    exit 1
}

usage() {
    cat <<'USAGE'
Build NCCL 2.32.3-adaptive-pause for GB200 (sm_100).

  --dest DIR        NCCL checkout (default: nccl/nccl-2.32.3-adaptive-pause)
  --cuda-home DIR   CUDA toolkit >= 12.8 (default: CUDA_HOME or /usr/local/cuda)
  --jobs N          Parallel make jobs (default: CPU affinity count)
  --check-only      Check the patched checkout and existing library
  -h, --help        Show this help

AllGather_STMC and AllGather_TmaSTMC use a higher pause while a local
ReduceScatter_LDMC CTA is active. See install/README-nccl-2.32.3-adaptive-pause.md.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dest|--cuda-home|--jobs)
            [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || die "$1 needs a value"
            case "$1" in
                --dest) dest="$2" ;;
                --cuda-home) cuda_home="$2" ;;
                --jobs) jobs="$2" ;;
            esac
            shift 2
            ;;
        --check-only) check_only=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || die "--jobs must be a positive integer"
[[ -f "$patch" ]] || die "patch not found: ${patch}"
[[ -x "${cuda_home}/bin/nvcc" ]] || die "nvcc not found under ${cuda_home}; pass --cuda-home"

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

args=(--dest "$dest" --cuda-home "$cuda_home" --jobs "$jobs")
$check_only && args+=(--check-only)
"${script_dir}/install-nccl-2.32.3.sh" "${args[@]}"

lib="${dest}/build/lib/libnccl.so.2"
grep -qF '#define NCCL_SUFFIX "-adaptive-pause"' "${dest}/build/include/nccl.h" || die "nccl.h has the wrong suffix"
grep -aqF 'NCCL version 2.32.3-adaptive-pause+cuda' "$lib" || die "library has the wrong NCCL version banner"
for knob in PAUSE_CYCLES RS_PAUSE_CYCLES PAUSE_POLL_CHUNKS; do
    grep -aqF "NCCL_SYM_AG_STMC_${knob}" "$lib" || die "NCCL_SYM_AG_STMC_${knob} is missing from ${lib}"
done
echo "Library: ${lib}"
