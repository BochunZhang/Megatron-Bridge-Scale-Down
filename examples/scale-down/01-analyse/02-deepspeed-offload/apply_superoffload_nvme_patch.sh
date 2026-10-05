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

usage() {
    cat <<'EOF'
Usage: apply_superoffload_nvme_patch.sh [--recovery]

Apply the NVMe-safe SuperOffload Stage-3 patch by default. Use --recovery to
restore the original SuperOffload implementation. The DeepSpeed location is
resolved from the Python interpreter selected by PYTHON_BIN.
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

TARGET_FILE="$DEEPSPEED_DIR/deepspeed/runtime/superoffload/superoffload_stage3.py"
[[ -f "$TARGET_FILE" ]] || die "DeepSpeed source file not found: $TARGET_FILE"

case "$PATCH_MODE" in
    override)
        PATCH_FILE="$SCRIPT_DIR/patches/superoffload_nvme_override.patch"
        ;;
    recovery)
        PATCH_FILE="$SCRIPT_DIR/patches/superoffload_nvme_recovery.patch"
        ;;
    *)
        die "unsupported patch mode: $PATCH_MODE"
        ;;
esac
[[ -f "$PATCH_FILE" ]] || die "patch file not found: $PATCH_FILE"

if git -C "$DEEPSPEED_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    if ! git -C "$DEEPSPEED_DIR" apply --check "$PATCH_FILE"; then
        die "patch does not apply cleanly; inspect local changes and DeepSpeed version"
    fi
    git -C "$DEEPSPEED_DIR" apply "$PATCH_FILE"
else
    command -v patch >/dev/null 2>&1 || die "git or patch is required"
    patch --dry-run --batch --forward -p1 -d "$DEEPSPEED_DIR" < "$PATCH_FILE" >/dev/null \
        || die "patch does not apply cleanly; inspect local changes and DeepSpeed version"
    patch --batch --forward -p1 -d "$DEEPSPEED_DIR" < "$PATCH_FILE"
fi

echo "DeepSpeed SuperOffload ${PATCH_MODE} patch applied: $TARGET_FILE"
