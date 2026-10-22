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

# torch 2.13.0 + torchvision 0.28.0 linked against nccl-2.32.3-cta (install-nccl-2.32.3-cta.sh).
#
# The build, install and checks are install-torch-2.13.0-nccl-2.32.3.sh; only two defaults differ:
#   --nccl-home  <megatron-bridge>/nccl/nccl-2.32.3-cta/build, whose source tree must carry
#                install/patches/nccl-2.32.3-cta.patch. It replaces the stock 2.32.3 lib dir in the RPATH.
#   --work-dir   <megatron-bridge>/.tmp/torch-nccl-2.32.3-cta: its own pytorch checkout, build dir and
#                wheel dir. A shared build/CMakeCache.txt would keep the stock NCCL paths, and the shared
#                wheel dir would mix the two torch wheels.
# The NCCL headers define NCCL_SUFFIX "-cta", so torch.cuda.nccl.version() is (2, 32, 3, '-cta'), and the
# checks expect that and the banner "NCCL version 2.32.3-cta+cuda..." from the 2-GPU run.
# Every other option is passed through unchanged. The wheel is installed into --python, replacing the
# torch there, so point --python at a separate env to keep the stock build.
#
# Usage:
#   ./install/install-torch-2.13.0-nccl-2.32.3-cta.sh [--python PY] [--nccl-home DIR] [--work-dir DIR]
#                                                     [--jobs N] [other install-torch-2.13.0-nccl-2.32.3.sh options]
#
#   --python PY      Env to install into (default: $VIRTUAL_ENV/bin/python)
#   --nccl-home DIR  nccl-2.32.3-cta build (default: <megatron-bridge>/nccl/nccl-2.32.3-cta/build)
#   --work-dir DIR   Sources, build venv, NVPL / ACL and wheels (default: <megatron-bridge>/.tmp/torch-nccl-2.32.3-cta)
#   --jobs N         MAX_JOBS (default: the official min(nproc - 2, 12); more risks OOM in flash-attention)

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
patch="${script_dir}/patches/nccl-2.32.3-cta.patch"
nccl_home="${repo_root}/nccl/nccl-2.32.3-cta/build"
work="${repo_root}/.tmp/torch-nccl-2.32.3-cta"
passthrough=()

die() {
    echo "ERROR: $*" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nccl-home) nccl_home="$2"; shift 2 ;;
        --work-dir) work="$2"; shift 2 ;;
        -h|--help) sed -n '16,36p' "$0"; exit 0 ;;
        *) passthrough+=("$1"); shift ;;
    esac
done

# The build dir alone does not tell the patched NCCL from the stock one; its source tree does.
nccl_src="$(dirname -- "$nccl_home")"
git -C "$nccl_src" apply --reverse --check "$patch" 2>/dev/null \
    || die "${nccl_src} does not carry ${patch##*/}; run install/install-nccl-2.32.3-cta.sh or pass --nccl-home"
echo "NCCL: ${nccl_home} (2.32.3 + ${patch##*/})"

exec "${script_dir}/install-torch-2.13.0-nccl-2.32.3.sh" --nccl-home "$nccl_home" --work-dir "$work" \
    ${passthrough[@]+"${passthrough[@]}"}
