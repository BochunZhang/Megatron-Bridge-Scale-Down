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

# Report which NCCL the current torch environment was built against and which one it loads.
# Needs no GPU and creates no communicator, so it is safe on a login node.
#
# Usage:
#   ./install/probe-torch-nccl-version.sh [--python CMD]
#
#   --python CMD   Interpreter to probe (default: "uv run --no-sync python" from the repo root)
#
# Output: one "key: value" line per fact, grouped under "## <section>" headers. Keys are stable;
# a value of "n/a" means the fact is not available in this torch build.
#
#   build.*    compile time: NCCL_MAJOR/MINOR/PATCH/SUFFIX baked into libtorch_cuda
#              (torch.cuda.nccl.version(), ProcessGroupNCCL.get_build_nccl_version())
#   runtime.*  ncclGetVersion() of the libnccl.so actually loaded, and its path
#   wheel.*    the nvidia-nccl-cu* wheel that pip/uv installed, if any
#   features.* torch-side switches relevant to symmetric memory / CE collectives

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
python_cmd=(uv run --no-sync python)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python) read -r -a python_cmd <<<"$2"; shift 2 ;;
        -h|--help) sed -n '16,30p' "$0"; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; exit 1 ;;
    esac
done

cd "$repo_root"
echo "## host"
echo "hostname: $(hostname)"
echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "python_cmd: ${python_cmd[*]}"
echo "env.LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-}"
echo "env.LD_PRELOAD: ${LD_PRELOAD:-}"
echo "env.NCCL_*: $(env | grep '^NCCL_' | sort | tr '\n' ' ')"

"${python_cmd[@]}" - <<'PY'
import importlib.metadata as md
import os
import re
import sys


def show(key, value):
    print(f"{key}: {'n/a' if value is None or value == '' else value}")


def guard(fn):
    try:
        return fn()
    except Exception as error:  # report, never abort: every line is useful on its own
        return f"error ({type(error).__name__}: {error})"


print("## python")
show("python.executable", sys.executable)
show("python.version", sys.version.split()[0])

import torch  # noqa: E402

print("## torch")
show("torch.version", torch.__version__)
show("torch.git_version", getattr(torch.version, "git_version", None))
show("torch.cuda", torch.version.cuda)
show("torch.file", torch.__file__)
show("torch.cuda_available", guard(torch.cuda.is_available))
# torch.__config__.show() lists the build settings as "KEY=VALUE, KEY=VALUE, ..." on one line.
for item in re.split(r",\s*|\n", str(guard(torch.__config__.show))):
    item = item.strip().lstrip("- ")
    if "NCCL" in item.upper() or item.startswith(("CUDA Runtime", "CUDA_VERSION", "TORCH_CUDA_ARCH_LIST")):
        show("torch.config", item)

print("## build")
# torch/csrc/cuda/nccl.cpp version(): (NCCL_MAJOR << 32) | (NCCL_MINOR << 16) | NCCL_PATCH from nccl.h at compile time.
show("build.torch_cuda_nccl_version", guard(lambda: ".".join(map(str, torch.cuda.nccl.version()))))
show("build.nccl_suffix", guard(lambda: torch._C._nccl_version_suffix().decode() or None))
pg = guard(lambda: torch.distributed.ProcessGroupNCCL)
show("build.process_group_nccl", "available" if not isinstance(pg, str) else pg)
if not isinstance(pg, str):
    show("build.get_build_nccl_version", guard(lambda: ".".join(map(str, pg.get_build_nccl_version()))))

print("## runtime")
if not isinstance(pg, str):
    # c10d::getNcclVersionTuple(): ncclGetVersion() of the library the process resolved.
    show("runtime.get_runtime_nccl_version", guard(lambda: ".".join(map(str, pg.get_runtime_nccl_version()))))
libs = []
try:
    with open("/proc/self/maps", encoding="utf-8") as maps:
        libs = sorted({line.split()[-1] for line in maps if re.search(r"libnccl[^/]*\.so", line)})
except OSError:
    pass
show("runtime.libnccl_loaded", ";".join(libs) if libs else "none mapped (static NCCL in libtorch_cuda, or not loaded)")
for path in libs:
    show("runtime.libnccl_realpath", os.path.realpath(path))
torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
show("runtime.torch_lib_libnccl", ";".join(sorted(f for f in os.listdir(torch_lib) if "nccl" in f)) or None)

print("## wheel")
found = False
for dist in md.distributions():
    name = (dist.metadata["Name"] or "").lower()
    if name.startswith("nvidia-nccl"):
        found = True
        show(f"wheel.{name}", dist.version)
        files = [str(f) for f in (dist.files or []) if "libnccl" in str(f)]
        for f in files:
            show(f"wheel.{name}.file", os.path.join(str(dist.locate_file("")), f))
if not found:
    show("wheel.nvidia-nccl", None)
for req in guard(lambda: md.requires("torch")) or []:
    if "nccl" in str(req).lower():
        show("wheel.torch_requires", req)

print("## features")
if not isinstance(pg, str):
    # Present only when torch was compiled against an NCCL with CTA policies (NCCL_HAS_CTA_POLICY).
    show("features.NCCL_CTA_POLICY_ZERO", getattr(pg, "NCCL_CTA_POLICY_ZERO", None))
    show("features.register_mem_pool", hasattr(pg, "register_mem_pool"))
    show("features.mem_allocator", hasattr(pg, "mem_allocator"))
    options = guard(lambda: pg.Options())
    config = getattr(options, "config", None)
    show("features.options.config", type(config).__name__ if config is not None else None)
    if config is not None:
        show("features.options.config.cta_policy", hasattr(config, "cta_policy"))
PY
