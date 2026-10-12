#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

# Usage: apply_superoffload_nvme_patch.sh [--recovery]
# Apply (or with --recovery, reverse) the NVMe patches (SuperOffload + ZeRO-3
# NVMe prefetch) to the DeepSpeed installed in $PYTHON_BIN. See README §3.1 / §6.2 / §6.3.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PATCHES=(
    "$SCRIPT_DIR"/patches/superoffload_nvme{,_fp32_device}_override.patch
    "$SCRIPT_DIR"/patches/zero3_nvme_prefetch_dedup_override.patch
)

case "${1:-}" in
    "") wanted=(-N) done_state=(-R) ;;
    --recovery) wanted=(-R) done_state=(-N) ;;
    *) echo "usage: $0 [--recovery]" >&2; exit 1 ;;
esac

# find_spec locates the package without importing (and initializing) DeepSpeed.
DS_DIR=$("${PYTHON_BIN:-python}" -c \
    'import importlib.util as u, pathlib; print(pathlib.Path(u.find_spec("deepspeed").origin).parent.parent)')

run_patch() { patch --force --silent -p1 -d "$DS_DIR" "$@"; }

# Check every patch first so a failure leaves DeepSpeed untouched.
pending=()
for p in "${PATCHES[@]}"; do
    if run_patch --dry-run "${wanted[@]}" < "$p" >/dev/null; then
        pending+=("$p")
    elif run_patch --dry-run "${done_state[@]}" < "$p" >/dev/null; then
        echo "skip: $(basename "$p") already in target state"
    else
        echo "error: $(basename "$p") does not match $DS_DIR" >&2
        exit 1
    fi
done

for p in ${pending[@]+"${pending[@]}"}; do
    run_patch "${wanted[@]}" < "$p"
    echo "done: $(basename "$p")"
done
