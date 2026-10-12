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

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
PATCH_MODE=override

# Both patches are required for param_nvme + super_offload_*:
# - superoffload_nvme_override.patch: CPU subgroups whose parameters live on
#   NVMe have no fp16 flat buffer; reuse the ZeRO-3 parent swap-out path.
# - superoffload_nvme_fp32_device_override.patch: with ratio < 1, GPU subgroups
#   whose parameters live on NVMe must create their FP32 master on the GPU.
# Recovery reverse-applies the same patches, restoring the original DeepSpeed
# source byte for byte. Recovery runs in the opposite order.
PATCH_FILES=(
    "$SCRIPT_DIR/patches/superoffload_nvme_override.patch"
    "$SCRIPT_DIR/patches/superoffload_nvme_fp32_device_override.patch"
)

usage() {
    cat <<'EOF'
Usage: apply_superoffload_nvme_patch.sh [--recovery]

Apply both NVMe-safe SuperOffload patches by default:
  patches/superoffload_nvme_override.patch
  patches/superoffload_nvme_fp32_device_override.patch
Use --recovery to reverse-apply them and restore the original DeepSpeed code.

The DeepSpeed location is resolved from the Python interpreter selected by
PYTHON_BIN. Patches already in the requested state are skipped; if any patch
cannot be applied or reversed cleanly, nothing is modified.
EOF
}

die() {
    echo "error: $*" >&2
    exit 1
}

while (($# > 0)); do
    case "$1" in
        --recovery)
            PATCH_MODE=recovery
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            die "unknown argument: $1"
            ;;
    esac
    shift
done

command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python interpreter not found: $PYTHON_BIN"
if ! DEEPSPEED_DIR=$("$PYTHON_BIN" - <<'PY'
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("deepspeed")
if spec is None:
    raise SystemExit("DeepSpeed is not importable from this Python environment")

if spec.submodule_search_locations:
    package_dir = Path(next(iter(spec.submodule_search_locations))).resolve()
elif spec.origin:
    package_dir = Path(spec.origin).resolve().parent
else:
    raise SystemExit("DeepSpeed package location is unavailable")

print(package_dir.parent)
PY
); then
    die "cannot locate DeepSpeed with $PYTHON_BIN"
fi

for patch_file in "${PATCH_FILES[@]}"; do
    [[ -f "$patch_file" ]] || die "patch file not found: $patch_file"
done
for target in deepspeed/runtime/superoffload/superoffload_stage3.py deepspeed/runtime/zero/stage3.py; do
    [[ -f "$DEEPSPEED_DIR/$target" ]] || die "DeepSpeed source file not found: $DEEPSPEED_DIR/$target"
done

if git -C "$DEEPSPEED_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    USE_GIT=true
else
    USE_GIT=false
    command -v patch >/dev/null 2>&1 || die "git or patch is required"
fi

# run_patch <forward|reverse> <check|apply> <patch_file>
run_patch() {
    local direction=$1
    local action=$2
    local patch_file=$3
    local reverse_args=()
    if [[ "$direction" == reverse ]]; then
        reverse_args=(-R)
    else
        # patch(1): refuse to treat an already-applied patch as a reversed one.
        [[ "$USE_GIT" == true ]] || reverse_args=(--forward)
    fi

    if [[ "$USE_GIT" == true ]]; then
        local check_args=()
        [[ "$action" == check ]] && check_args=(--check)
        git -C "$DEEPSPEED_DIR" apply ${check_args[@]+"${check_args[@]}"} ${reverse_args[@]+"${reverse_args[@]}"} "$patch_file"
    else
        local dry_run_args=()
        [[ "$action" == check ]] && dry_run_args=(--dry-run)
        # --force: never auto-reverse a patch that "looks reversed".
        patch ${dry_run_args[@]+"${dry_run_args[@]}"} --force --silent ${reverse_args[@]+"${reverse_args[@]}"} -p1 -d "$DEEPSPEED_DIR" < "$patch_file"
    fi
}

if [[ "$PATCH_MODE" == override ]]; then
    wanted=forward
    done_state=reverse
    ordered_patches=("${PATCH_FILES[@]}")
else
    wanted=reverse
    done_state=forward
    ordered_patches=()
    for ((i = ${#PATCH_FILES[@]} - 1; i >= 0; i--)); do
        ordered_patches+=("${PATCH_FILES[i]}")
    done
fi

# Phase 1: check every patch before touching any file.
pending=()
for patch_file in "${ordered_patches[@]}"; do
    name=$(basename "$patch_file")
    if run_patch "$wanted" check "$patch_file" >/dev/null 2>&1; then
        pending+=("$patch_file")
    elif run_patch "$done_state" check "$patch_file" >/dev/null 2>&1; then
        echo "skip: $name is already in ${PATCH_MODE} state"
    else
        die "$name cannot be applied in ${PATCH_MODE} mode; inspect local changes and DeepSpeed version ($DEEPSPEED_DIR)"
    fi
done

# Phase 2: apply; undo already-applied steps if a later one fails.
applied=()
for patch_file in ${pending[@]+"${pending[@]}"}; do
    if ! run_patch "$wanted" apply "$patch_file"; then
        for ((i = ${#applied[@]} - 1; i >= 0; i--)); do
            run_patch "$done_state" apply "${applied[i]}" || true
        done
        die "failed to apply $(basename "$patch_file") in ${PATCH_MODE} mode; earlier steps were rolled back"
    fi
    applied+=("$patch_file")
    echo "${PATCH_MODE}: $(basename "$patch_file")"
done

echo "DeepSpeed SuperOffload NVMe patches are in ${PATCH_MODE} state: $DEEPSPEED_DIR"
