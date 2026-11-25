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

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
patch_file="$script_dir/patches/v0.20.1/deepspeed-cpu-adam-release-gil.patch"
python_bin="${PYTHON_BIN:-${PYTHON:-python}}"

usage() {
    printf 'Usage: %s [--recovery]\n' "${0##*/}"
    printf '%s\n' 'Patch the installed DeepSpeed v0.20.1 CPUAdam sources and verify the JIT build.'
    printf '%s\n' 'Set PYTHON_BIN to the training interpreter. Use --recovery to restore the original sources.'
}

case "$#:${1:-}" in
    0:) wanted=(-N); opposite=(-R); expected=true ;;
    1:--recovery) wanted=(-R); opposite=(-N); expected=false ;;
    1:-h|1:--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

ds_ops=$("$python_bin" - <<'PY'
import sys
from importlib.util import find_spec
from pathlib import Path

spec = find_spec("deepspeed")
if spec is None or spec.origin is None:
    sys.exit("DeepSpeed is missing from " + sys.executable)
ops = Path(spec.origin).resolve().parent / "ops"
for name in ("cpu_adam.cpp", "cpu_adam_impl.cpp"):
    if not (ops / "csrc" / "adam" / name).is_file():
        sys.exit(f"CPUAdam JIT source {name} is missing under {ops}")
sys.stdout.write(str(ops))
PY
)

# DeepSpeed prefers an installed binary when its accelerator matches this process.
"$python_bin" - <<'PY'
import sys

from deepspeed.accelerator import get_accelerator
from deepspeed.git_version_info import accelerator_name, installed_ops

if installed_ops.get("cpu_adam") and accelerator_name == get_accelerator()._name:
    sys.exit("CPUAdam is precompiled in this environment. Use a JIT-enabled DeepSpeed installation.")
PY

run_patch() { patch --force --silent --fuzz=0 -p1 -d "$ds_ops" "$@" < "$patch_file"; }

changed=false
if run_patch --dry-run "${wanted[@]}" >/dev/null 2>&1; then
    adam_dir="$ds_ops/csrc/adam"
    for target in "$adam_dir" "$adam_dir/cpu_adam.cpp" "$adam_dir/cpu_adam_impl.cpp"; do
        if [[ ! -w "$target" ]]; then
            printf 'CPUAdam patch cannot write to %s\n' "$target" >&2
            exit 1
        fi
    done
    run_patch "${wanted[@]}"
    changed=true
elif ! run_patch --dry-run "${opposite[@]}" >/dev/null 2>&1; then
    printf 'CPUAdam GIL patch does not match the installed sources under %s\n' "$ds_ops" >&2
    exit 1
fi

# Use the same cache as the training launcher. Ninja rebuilds modified sources.
if "$python_bin" - "$expected" <<'PY'
import sys

from deepspeed.ops.op_builder import CPUAdamBuilder

expected = sys.argv[1] == "true"
op = CPUAdamBuilder().load()
actual = bool(getattr(op, "adam_update_releases_gil", False))
if actual != expected:
    sys.exit(f"CPUAdam GIL release is {actual}, expected {expected}, loaded from {op.__file__}")
sys.stdout.write(f"CPUAdam GIL release is {actual}, verified at {op.__file__}\n")
PY
then
    exit 0
fi

if [[ "$changed" == true ]]; then
    run_patch "${opposite[@]}"
    printf '%s\n' 'CPUAdam JIT verification failed. Restored the previous sources.' >&2
fi
exit 1
