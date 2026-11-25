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
patch_dir="$script_dir/patches/v0.20.1"
gil_patch="$patch_dir/deepspeed-cpu-adam-release-gil.patch"
omp_patch="$patch_dir/deepspeed-cpu-adam-graceadam-omp.patch"
python_bin="${PYTHON_BIN:-${PYTHON:-python}}"

usage() {
    printf 'Usage: %s [--recovery]\n' "${0##*/}"
    printf '%s\n' 'Patch DeepSpeed v0.20.1 CPUAdam for GIL release and GraceAdam OpenMP threads.'
    printf '%s\n' 'Set PYTHON_BIN to the training interpreter. Use --recovery to restore the original sources.'
}

case "$#:${1:-}" in
    0:) wanted=-N; opposite=-R; expected=true; patches=("$gil_patch" "$omp_patch") ;;
    1:--recovery) wanted=-R; opposite=-N; expected=false; patches=("$omp_patch" "$gil_patch") ;;
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
for name in ("adam/cpu_adam.cpp", "adam/cpu_adam_impl.cpp", "includes/sve.h"):
    if not (ops / "csrc" / name).is_file():
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

run_patch() {
    local input="$1"
    shift
    patch --force --silent --fuzz=0 --no-backup-if-mismatch -p1 -d "$ds_ops" "$@" < "$input"
}

changes=()
for patch_file in "${patches[@]}"; do
    if run_patch "$patch_file" --dry-run "$wanted" >/dev/null 2>&1; then
        changes+=("$patch_file")
    elif ! run_patch "$patch_file" --dry-run "$opposite" >/dev/null 2>&1; then
        printf 'CPUAdam patch %s does not match the installed sources under %s\n' "${patch_file##*/}" "$ds_ops" >&2
        exit 1
    fi
done

if ((${#changes[@]})); then
    for source in adam/cpu_adam.cpp adam/cpu_adam_impl.cpp includes/sve.h; do
        target="$ds_ops/csrc/$source"
        if [[ ! -w "$target" || ! -w "${target%/*}" ]]; then
            printf 'CPUAdam patch cannot write to %s\n' "$target" >&2
            exit 1
        fi
    done
fi

changed=()
restore_sources() {
    local status=$? index
    if ((status == 0 || ${#changed[@]} == 0)); then
        return
    fi
    for ((index = ${#changed[@]} - 1; index >= 0; index--)); do
        if ! run_patch "${changed[index]}" "$opposite"; then
            printf 'Could not restore CPUAdam sources for %s\n' "${changed[index]}" >&2
            return
        fi
    done
    printf 'CPUAdam patch verification failed. Restored the previous sources.\n' >&2
}
trap restore_sources EXIT
for patch_file in "${changes[@]+${changes[@]}}"; do
    run_patch "$patch_file" "$wanted"
    changed+=("$patch_file")
done

# Use the same cache as the training launcher. Ninja rebuilds modified sources.
"$python_bin" - "$expected" <<'PY'
import sys

from deepspeed.ops.op_builder import CPUAdamBuilder

expected = sys.argv[1] == "true"
op = CPUAdamBuilder().load()
actual = bool(getattr(op, "adam_update_releases_gil", False))
if actual != expected:
    sys.exit(f"CPUAdam GIL release is {actual}, expected {expected}, loaded from {op.__file__}")
omp_patch = bool(getattr(op, "graceadam_omp_threads_patch", False))
if omp_patch != expected:
    sys.exit(f"CPUAdam GraceAdam OpenMP patch is {omp_patch}, expected {expected}, loaded from {op.__file__}")
sys.stdout.write(f"CPUAdam GIL release={actual}, GraceAdam OpenMP patch={omp_patch}, verified at {op.__file__}\n")
PY
