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

# torch 2.13.0 + torchvision 0.28.0 from source, linked dynamically against the NCCL 2.32.3 that
# install-nccl-2.32.3.sh builds, with the build options of the official aarch64 cu13 wheels.
#
#   torch        v2.13.0 replaced .ci/manywheel/build_cuda.sh with build_env_setup.py, build_wheel.py
#                and repair_wheel.py. They are imported from the cloned tree, so the build flags
#                (cuda_build_env: TORCH_CUDA_ARCH_LIST, TORCH_NVCC_FLAGS, USE_SYSTEM_NCCL=1,
#                USE_STATIC_NCCL=0, ...), the pinned runtime wheels (PYTORCH_EXTRA_INSTALL_REQUIREMENTS)
#                and the RPATH list are the official ones. What differs, and why:
#                  - nccl.h / libnccl.so come from --nccl-home, and its lib dir takes the place of
#                    $ORIGIN/../../nvidia/nccl/lib in the RPATH. nvidia-nccl-cu13 is dropped from the
#                    requirements: torch/__init__.py preloads it by path and would win over the RPATH.
#                  - cuDNN / cuSPARSELt headers come from the pinned pip wheels (the manywheel image has
#                    the same versions under /usr/local/cuda).
#                  - NVPL and ACL are installed under the work dir with the official install_nvpl.sh /
#                    install_acl.sh unless /usr/local/lib and /acl already have them.
#                  - TORCH_CUDA_ARCH_LIST is only CUDA_ARCH_LIST (10.0, GB200) instead of 8.0;9.0;10.0;11.0;12.0.
#                  - version 2.13.0+cu130 like the official wheel, but tag linux_aarch64, since the wheel is
#                    built against the host glibc rather than manylinux_2_28.
#                torch itself is compiled in a build venv under --work-dir with the official build deps
#                (requirements-build.txt, numpy pin), so they do not change --python.
#   torchvision  v0.28.0 with the env pytorch/test-infra release/2.13 sets for CUDA wheels, also only for
#                CUDA_ARCH_LIST. The torch wheel is installed into --python first, and torchvision is compiled
#                with --python (its setuptools and that torch; the 0.28.0 wheel on PyPI was built against the
#                official torch), then installed there too.
#
# The checks confirm torch was compiled against and loads NCCL 2.32.3 (run
# install/probe-torch-nccl-version.sh for the full report).
#
# Usage:
#   ./install/install-torch-2.13.0-nccl-2.32.3.sh [--python PY] [--nccl-home DIR] [--cuda-home DIR]
#                                                 [--work-dir DIR] [--jobs N] [--no-install] [--check-only]
#
#   --python PY      Env to install into, build torchvision with and check (default: $VIRTUAL_ENV/bin/python)
#   --nccl-home DIR  NCCL 2.32.3 build (default: <megatron-bridge>/nccl/nccl-2.32.3/build)
#   --cuda-home DIR  CUDA 13.0 or 13.2 toolkit (default: CUDA_HOME or /usr/local/cuda)
#   --work-dir DIR   Sources, build venv, NVPL / ACL and wheels (default: <megatron-bridge>/.tmp)
#   --jobs N         MAX_JOBS (default: the official min(nproc - 2, 12))
#   --skip-torch-build  Reuse the torch wheel already in the wheel dir; rebuild only torchvision
#   --no-install     Stop after the torch wheel (torchvision needs that torch installed in --python)
#   --check-only     Skip clone, build and install; only run the checks against --python

set -euo pipefail

# uv pip reads [tool.uv] from the pyproject.toml in the working directory, and megatron-bridge's
# override-dependencies turn torch / torchvision into "sys_platform == 'never'": every install of
# those wheels would silently resolve to nothing.
export UV_NO_CONFIG=1

TORCH_TAG="v2.13.0"
VISION_TAG="v0.28.0"
TORCH_REPO="https://github.com/pytorch/pytorch.git"
VISION_REPO="https://github.com/pytorch/vision.git"
NCCL_VERSION="2.32.3"
NCCL_VERSION_CODE=23203
# GB200 only; the official cu13 aarch64 wheels build 8.0;9.0;10.0;11.0;12.0.
CUDA_ARCH_LIST="10.0"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
python_arg="${VIRTUAL_ENV:+${VIRTUAL_ENV}/bin/python}"
nccl_home="${repo_root}/nccl/nccl-${NCCL_VERSION}/build"
cuda_home="${CUDA_HOME:-/usr/local/cuda}"
work="${repo_root}/.tmp"
jobs=""
install=true
check_only=false
skip_torch_build=false

die() {
    echo "ERROR: $*" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python) python_arg="$2"; shift 2 ;;
        --nccl-home) nccl_home="$2"; shift 2 ;;
        --cuda-home) cuda_home="$2"; shift 2 ;;
        --work-dir) work="$2"; shift 2 ;;
        --jobs) jobs="$2"; shift 2 ;;
        --skip-torch-build) skip_torch_build=true; shift ;;
        --no-install) install=false; shift ;;
        --check-only) check_only=true; shift ;;
        -h|--help) sed -n '16,55p' "$0"; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

failures=0
ok() { echo "  [ok]   $*"; }
bad() { echo "  [FAIL] $*"; failures=$(( failures + 1 )); }
note() { echo "  [warn] $*"; }

# torch's CMake forwards every USE_* / BUILD_* / CMAKE_* env var (cmake/EnvVarForwarding.cmake), and
# install-gb200.sh / the installation notes export CFLAGS, TORCH_CUDA_ARCH_LIST and friends. Clear them
# in the build subshells so only the official options (plus the ones set below) reach the build.
clean_build_env() {
    local var
    for var in $(compgen -e); do
        case "$var" in
            USE_*|BUILD_*|CMAKE_*|TORCH_*|TORCHVISION_*|PYTORCH_*|NVCC_*|NCCL_*|CUDNN_*|CUSPARSELT_*|NVSHMEM_*) unset "$var" ;;
        esac
    done
    unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS CUDACXX CUDAHOSTCXX BLAS ACL_ROOT_DIR MAX_JOBS FORCE_CUDA
}

[[ "$(uname -s)" == Linux && "$(uname -m)" == aarch64 ]] || die "only the GB200 (Linux aarch64) build is scripted"
[[ -n "$python_arg" ]] || die "no --python given and no virtualenv active"
target_py="$(command -v -- "$python_arg")" || die "${python_arg} not found"
command -v uv >/dev/null || die "uv not found"
[[ -f "${nccl_home}/include/nccl.h" && -e "${nccl_home}/lib/libnccl.so.2" ]] \
    || die "no NCCL build in ${nccl_home}; run install/install-nccl-${NCCL_VERSION}.sh or pass --nccl-home"
nccl_code="$(sed -n 's/^#define NCCL_VERSION_CODE \([0-9]*\).*/\1/p' "${nccl_home}/include/nccl.h")"
[[ "$nccl_code" == "$NCCL_VERSION_CODE" ]] || die "${nccl_home}/include/nccl.h is NCCL_VERSION_CODE ${nccl_code}, expected ${NCCL_VERSION_CODE}"
# Empty for stock NCCL, "-cta" for install-nccl-2.32.3-cta.sh. torch reports it next to the version
# (torch.cuda.nccl.version() appends it as a 4th element) and NCCL prints it in its banner.
nccl_suffix="$(sed -n 's/^#define NCCL_SUFFIX "\(.*\)".*/\1/p' "${nccl_home}/include/nccl.h")"
nccl_home="$(cd -- "$nccl_home" && pwd -P)"
nccl_lib="${nccl_home}/lib"
mkdir -p "$work"
work="$(cd -- "$work" && pwd -P)"
py_ver="$("$target_py" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"

torch_src="${work}/pytorch-${TORCH_TAG}"
vision_src="${work}/vision-${VISION_TAG}"
build_venv="${work}/torch-${TORCH_TAG#v}-build-py${py_ver}"
bpy="${build_venv}/bin/python"
deps="${work}/torch-${TORCH_TAG#v}-build-deps"
wheel_dir="${work}/wheels/torch-${TORCH_TAG#v}-nccl-${NCCL_VERSION}"

echo "target python: ${target_py} (${py_ver})"
echo "NCCL: ${nccl_home} (${NCCL_VERSION}${nccl_suffix})"

if ! $check_only; then
    [[ -x "${cuda_home}/bin/nvcc" ]] || die "nvcc not found under ${cuda_home}; pass --cuda-home"
    cuda_short="$("${cuda_home}/bin/nvcc" --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')"
    desired_cuda="cu${cuda_short//./}"
    if [[ -z "$jobs" ]]; then
        # .ci/pytorch/binary_populate_env.sh: MAX_JOBS = min(nproc - 2, 12); the cap avoids OOM while
        # compiling flash-attention. GNU nproc honours OMP_NUM_THREADS, which training environments
        # often set to 1.
        cpus="$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)"
        jobs=$(( cpus - 2 < 12 ? cpus - 2 : 12 ))
        (( jobs >= 1 )) || jobs=1
    fi
    echo "CUDA toolkit: ${cuda_home} (${cuda_short}, DESIRED_CUDA=${desired_cuda}), MAX_JOBS=${jobs}"
    # .ci/docker/common/install_cuda.sh installs 13.0.2 / 13.2.1; Dockerfile_cuda_aarch64 uses gcc-toolset-13.
    cuda_full="$("$target_py" -c 'import json, sys; print(json.load(open(sys.argv[1]))["cuda"]["version"])' \
        "${cuda_home}/version.json" 2>/dev/null || true)"
    case "${cuda_short}:${cuda_full}" in
        13.0:13.0.2|13.2:13.2.1) ;;
        13.0:*|13.2:*) note "CUDA ${cuda_full:-${cuda_short}.?}; the official ${cuda_short} wheels are built with $([[ $cuda_short == 13.0 ]] && echo 13.0.2 || echo 13.2.1)" ;;
        *) die "CUDA ${cuda_short}: the official build matrix only has 13.0 and 13.2 for torch ${TORCH_TAG#v}" ;;
    esac
    gcc_major="$(c++ -dumpversion | cut -d. -f1)"
    [[ "$gcc_major" == 13 ]] || note "host compiler is $(c++ --version | head -1); the official image uses gcc-toolset-13"

    ############################################################################
    # 1. Sources

    clone() {
        local repo="$1" tag="$2" dest="$3"
        if [[ -d "${dest}/.git" ]]; then
            [[ "$(git -C "$dest" describe --tags --exact-match 2>/dev/null)" == "$tag" ]] \
                || die "${dest} is not at ${tag}; move it away or pass --work-dir"
            echo "=== Reusing ${dest}"
            git -C "$dest" submodule update --init --recursive
        elif [[ -e "$dest" ]]; then
            die "${dest} exists but is not a git checkout; move it away or pass --work-dir"
        else
            echo "=== Cloning ${repo} ${tag} into ${dest}"
            git clone --depth 1 --branch "$tag" --recurse-submodules "$repo" "$dest"
        fi
    }
    clone "$TORCH_REPO" "$TORCH_TAG" "$torch_src"
    clone "$VISION_REPO" "$VISION_TAG" "$vision_src"

    ############################################################################
    # 2. Build venv: the official build deps (.ci/manywheel/build_install_deps.py), kept out of the
    #    target env because they pin numpy and setuptools.

    echo "=== Build venv ${build_venv}"
    [[ -x "$bpy" ]] || "$target_py" -m venv "$build_venv"
    uv pip install --python "$bpy" -U -r "${torch_src}/requirements-build.txt"
    uv pip install --python "$bpy" -r "${torch_src}/requirements.txt"
    numpy_pin="$("$bpy" - "$torch_src" <<'PY'
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("deps", f"{sys.argv[1]}/.ci/manywheel/build_install_deps.py")
deps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deps)
print(deps.numpy_pin())
PY
)"
    uv pip install --python "$bpy" "numpy==${numpy_pin}" build wheel patchelf

    ############################################################################
    # 3. Official build env, read from the v2.13.0 tree

    echo "=== Official cu13 aarch64 build env"
    # binary_populate_env.sh: a tag build is <tag>+<DESIRED_CUDA>; build_env_setup.py reads it to drop +PTX.
    torch_version="${TORCH_TAG#v}+${desired_cuda}"
    env_file="${work}/torch-${TORCH_TAG#v}-build.env"
    runtime_reqs="${work}/torch-${TORCH_TAG#v}-runtime-requirements.txt"
    PYTORCH_BUILD_VERSION="$torch_version" "$bpy" - "$torch_src" "$cuda_short" "$runtime_reqs" >"$env_file" <<'PY'
import importlib.util
import shlex
import sys
from pathlib import Path

src, cuda, runtime_reqs = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


env = load(src / ".ci/manywheel/build_env_setup.py").cuda_build_env(cuda, "aarch64")
matrix = load(src / ".github/scripts/generate_binary_build_matrix.py")
reqs = [req.strip() for req in matrix.PYTORCH_EXTRA_INSTALL_REQUIREMENTS[cuda].split("|")]
dropped = [req for req in reqs if req.startswith("nvidia-nccl-")]
if not dropped:
    sys.exit(f"no nvidia-nccl-* in PYTORCH_EXTRA_INSTALL_REQUIREMENTS[{cuda}]; re-check the RPATH replacement")
reqs = [req for req in reqs if req not in dropped]
runtime_reqs.write_text("\n".join(reqs) + "\n")
# binary_populate_env.sh appends triton for wheels with PYTORCH_EXTRA_INSTALL_REQUIREMENTS.
triton = (src / ".ci/docker/triton_version.txt").read_text().strip()
reqs.append(f"triton=={triton}; platform_system == 'Linux' and python_version < '3.15'")
env["PYTORCH_EXTRA_INSTALL_REQUIREMENTS"] = " | ".join(reqs)
for key, value in env.items():
    print(f"export {key}={shlex.quote(value)}")
print(f"dropped from the wheel requirements: {', '.join(dropped)}", file=sys.stderr)
PY
    sed 's/^/  /' "$env_file"
    echo "  (the build overrides TORCH_CUDA_ARCH_LIST=${CUDA_ARCH_LIST} and the NCCL / cuDNN / cuSPARSELt paths)"

    ############################################################################
    # 4. Libraries the manywheel image provides: the pinned CUDA wheels, NVPL, ACL

    echo "=== Pinned runtime wheels (also the cuDNN / cuSPARSELt / NVSHMEM the build compiles against)"
    uv pip install --python "$bpy" -r "$runtime_reqs"
    site="$("$bpy" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
    # FindCUDNN / FindCUSPARSELT look for the unversioned libcudnn.so / libcusparseLt.so, which the
    # wheels do not ship (build_env_setup.py creates the same links in /usr/local/cuda/lib64).
    stage_wheel() {
        local name="$1" header="$2" src="${site}/nvidia/$1" dst="${deps}/$1" so
        [[ -f "${src}/include/${header}" ]] || die "${src}/include/${header} not found"
        rm -rf "$dst"
        mkdir -p "${dst}/lib"
        ln -s "${src}/include" "${dst}/include"
        for so in "${src}"/lib/lib*.so.*; do
            [[ -e "$so" ]] || die "no lib*.so.* in ${src}/lib"
            ln -sf "$so" "${dst}/lib/$(basename "$so")"
            ln -sf "$so" "${dst}/lib/$(basename "${so%%.so.*}").so"
        done
    }
    stage_wheel cudnn cudnn.h
    stage_wheel cusparselt cusparseLt.h

    if [[ -f /usr/local/lib/libnvpl_blas_lp64_gomp.so.0 ]]; then
        nvpl=/usr/local
    else
        nvpl="${deps}/nvpl"
        if [[ ! -f "${nvpl}/lib/libnvpl_blas_lp64_gomp.so.0" ]]; then
            echo "=== NVPL into ${nvpl} (.ci/docker/common/install_nvpl.sh)"
            command -v wget >/dev/null || die "install_nvpl.sh needs wget"
            mkdir -p "${deps}/nvpl-src"
            sed "s#/opt/nvpl#${nvpl}#g" "${torch_src}/.ci/docker/common/install_nvpl.sh" >"${deps}/nvpl-src/install_nvpl.sh"
            (cd "${deps}/nvpl-src" && bash install_nvpl.sh)
        fi
    fi
    if [[ -f /acl/build/libarm_compute.so ]]; then
        acl=/acl
    else
        acl="${deps}/acl"
        if [[ ! -f "${acl}/build/libarm_compute.so" ]]; then
            echo "=== ACL into ${acl} (.ci/docker/common/install_acl.sh)"
            uv pip install --python "$bpy" scons==4.7.0  # Dockerfile_cuda_aarch64
            rm -rf "${deps}/acl-src"
            mkdir -p "${deps}/acl-src"
            sed -e "s#^ACL_INSTALL_DIR=.*#ACL_INSTALL_DIR=\"${acl}\"#" -e 's/sudo //g' \
                "${torch_src}/.ci/docker/common/install_acl.sh" >"${deps}/acl-src/install_acl.sh"
            (cd "${deps}/acl-src" && PATH="${build_venv}/bin:${PATH}" bash install_acl.sh)
        fi
    fi
    echo "NVPL: ${nvpl}, ACL: ${acl}"

    ############################################################################
    # 5. torch wheel (build_wheel.py), then repair_wheel.py with our NCCL in the RPATH

    raw_dir="${work}/wheels/raw-torch-${TORCH_TAG#v}"
    mkdir -p "$wheel_dir"
    rm -f "${wheel_dir}"/torchvision-*.whl
    if $skip_torch_build; then
        torch_whl="$(ls "${wheel_dir}"/torch-*.whl 2>/dev/null)" \
            || die "--skip-torch-build: no torch wheel in ${wheel_dir}; drop the flag to build it"
        echo "=== --skip-torch-build: reusing ${torch_whl}"
    else
    rm -rf "$raw_dir"
    mkdir -p "$raw_dir"
    rm -f "${wheel_dir}"/torch-*.whl
    echo "=== Building torch ${torch_version} (log: ${work}/torch-build.log)"
    (
        clean_build_env
        # shellcheck source=/dev/null
        source "$env_file"
        export NCCL_ROOT="$nccl_home" NCCL_ROOT_DIR="$nccl_home"
        export NCCL_INCLUDE_DIR="${nccl_home}/include" NCCL_LIB_DIR="$nccl_lib"
        export CUDNN_ROOT="${deps}/cudnn" CUDNN_ROOT_DIR="${deps}/cudnn"
        export CUDNN_INCLUDE_DIR="${deps}/cudnn/include" CUDNN_LIB_DIR="${deps}/cudnn/lib"
        export CUSPARSELT_ROOT="${deps}/cusparselt" CUSPARSELT_INCLUDE_DIR="${deps}/cusparselt/include"
        export CUSPARSELT_LIBRARY="${deps}/cusparselt/lib"
        export TORCH_CUDA_ARCH_LIST="$CUDA_ARCH_LIST"
        # binary_populate_env.sh
        export PYTORCH_BUILD_VERSION="$torch_version" PYTORCH_BUILD_NUMBER=1 TORCH_PACKAGE_NAME=torch
        export USE_FBGEMM=1 USE_GOLD_LINKER=OFF MAX_JOBS="$jobs"
        export GPU_ARCH_TYPE=cuda-aarch64 GPU_ARCH_VERSION="$cuda_short" DESIRED_CUDA="$desired_cuda"
        # build_wheel.py configure_blas_env() for cuda-aarch64
        export USE_MKLDNN=1 USE_MKLDNN_ACL=1 ACL_ROOT_DIR="$acl" BLAS=NVPL
        export CMAKE_PREFIX_PATH="${nvpl}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
        export LD_LIBRARY_PATH="${acl}/build${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        export CUDA_HOME="$cuda_home" PATH="${cuda_home}/bin:${build_venv}/bin:${PATH}"
        cd "$torch_src"
        "$bpy" -m build --wheel --no-isolation --outdir "$raw_dir"
    ) 2>&1 | tee "${work}/torch-build.log"

    echo "=== Repairing the torch wheel (repair_wheel.py)"
    PATH="${build_venv}/bin:${PATH}" "$bpy" - "$torch_src" "$cuda_short" "$nccl_lib" "$raw_dir" "$wheel_dir" "$acl" "$nvpl" <<'PY'
import importlib.util
import shutil
import sys
from pathlib import Path

src, cuda, nccl_lib, raw_dir, wheel_dir, acl, nvpl = sys.argv[1:]
spec = importlib.util.spec_from_file_location("repair_wheel", f"{src}/.ci/manywheel/repair_wheel.py")
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)
repair.PATCHELF = shutil.which("patchelf") or sys.exit("patchelf not found")

official = "$ORIGIN/../../nvidia/nccl/lib"
rpaths = repair.cuda_rpaths(cuda)
if official not in rpaths.split(":"):
    sys.exit(f"{official} not in the official RPATH {rpaths}; re-check before replacing it")
rpaths = rpaths.replace(official, nccl_lib)

# aarch64_extra_deps(use_cuda=True), with /usr/lib64, /acl and /usr/local/lib swapped for where they are here.
gfortran = [p for p in (Path("/usr/lib64/libgfortran.so.5"), Path("/usr/lib/aarch64-linux-gnu/libgfortran.so.5")) if p.is_file()]
required = [Path(acl, "build", name) for name in ("libarm_compute.so", "libarm_compute_graph.so")]
required += [
    Path(nvpl, "lib", name)
    for name in (
        "libnvpl_blas_lp64_gomp.so.0",
        "libnvpl_lapack_lp64_gomp.so.0",
        "libnvpl_blas_core.so.0",
        "libnvpl_lapack_core.so.0",
    )
]
missing = [str(p) for p in required if not p.is_file()]
if missing:
    sys.exit(f"missing libraries to bundle: {missing}")

wheels = sorted(Path(raw_dir).glob("torch-*.whl"))
if not wheels:
    sys.exit(f"no torch wheel in {raw_dir}")
for wheel in wheels:
    repair.repair_wheel(
        wheel,
        Path(wheel_dir),
        repair.detect_libgomp(),
        gfortran[:1] + required,
        [],
        [],
        f"{rpaths}:$ORIGIN:$ORIGIN/lib",
        f"{rpaths}:$ORIGIN",
        True,
    )
print(f"RPATH: {rpaths}:$ORIGIN")
PY
    torch_whl="$(ls "${wheel_dir}"/torch-*.whl)"
    fi

    ############################################################################
    # 6. Install torch into --python (torchvision is compiled against that torch)

    if ! $install; then
        echo "=== --no-install: torch wheel ${torch_whl}"
        echo "  torchvision is built against the torch in --python, so it is not built here. Install with"
        echo "  UV_NO_CONFIG=1 uv pip uninstall --python ${target_py} torch torchvision nvidia-nccl-cu13"
        echo "  UV_NO_CONFIG=1 uv pip install --python ${target_py} ${torch_whl}"
        echo "  then: $0 --python ${target_py} --skip-torch-build"
        exit 0
    fi
    echo "=== Installing torch into ${target_py}"
    nvshmem_version() {
        "$target_py" -c 'import importlib.metadata as m; print(m.version("nvidia-nvshmem-cu13"))' 2>/dev/null || echo none
    }
    nvshmem_before="$(nvshmem_version)"
    uv pip uninstall --python "$target_py" torch torchvision nvidia-nccl-cu13 || true  # any may be absent
    uv pip install --python "$target_py" "$torch_whl"
    nvshmem_after="$(nvshmem_version)"
    if [[ "$nvshmem_before" != none && "$nvshmem_before" != "$nvshmem_after" ]]; then
        note "nvidia-nvshmem-cu13 ${nvshmem_before} -> ${nvshmem_after} (torch pins it); reinstall ${nvshmem_before} if DeepEP needs it"
    fi
    "$target_py" -c 'import torch; print(f"torch {torch.__version__} at {torch.__file__}")' \
        || die "torch is not importable in ${target_py} after installing ${torch_whl}"

    ############################################################################
    # 7. torchvision: built with --python (its setuptools and the torch just installed), then installed

    echo "=== Building torchvision ${VISION_TAG#v} with ${target_py} (log: ${work}/torchvision-build.log)"
    # Build frontends and tools only; anything already installed keeps its version (no --upgrade).
    uv pip install --python "$target_py" build wheel packaging ninja pyyaml
    if ! "$target_py" -c 'import setuptools' 2>/dev/null; then
        # A fresh 3.12 venv has no setuptools; < 80 like megatron-bridge's own build-system.
        uv pip install --python "$target_py" "setuptools<80"
    fi
    "$target_py" -c 'import setuptools; print(f"setuptools {setuptools.__version__}")'
    # torchvision setup.py: PNG via libpng-config; JPEG / WebP headers via TORCHVISION_INCLUDE, CONDA_PREFIX,
    # /usr/include, /usr/local/include.
    command -v libpng-config >/dev/null \
        || note "libpng-config not found: torchvision builds without PNG (see README: PNG / JPEG / WebP)"
    for header in jpeglib.h webp/decode.h; do
        found=false
        IFS=: read -r -a dirs <<<"${TORCHVISION_INCLUDE:-}"
        for dir in ${dirs[@]+"${dirs[@]}"} "${CONDA_PREFIX:+${CONDA_PREFIX}/include}" /usr/include /usr/local/include; do
            [[ -n "$dir" && -f "${dir}/${header}" ]] && found=true
        done
        $found || note "${header} not found: torchvision builds without that codec (see README: PNG / JPEG / WebP)"
    done
    (
        # Keep the codec search paths across clean_build_env.
        tv_include="${TORCHVISION_INCLUDE:-}" tv_library="${TORCHVISION_LIBRARY:-}"
        clean_build_env
        [[ -n "$tv_include" ]] && export TORCHVISION_INCLUDE="$tv_include"
        [[ -n "$tv_library" ]] && export TORCHVISION_LIBRARY="$tv_library"
        # pytorch/test-infra release/2.13 tools/pkg-helpers/pytorch_pkg_helpers/{cuda,version}.py
        export CUDA_HOME="$cuda_home" CUDA_PATH="$cuda_home" PATH="${cuda_home}/bin:$(dirname -- "$target_py"):${PATH}"
        export TORCH_CUDA_ARCH_LIST="$CUDA_ARCH_LIST" FORCE_CUDA=1  # official: 8.0;9.0;10.0;11.0;12.0+PTX
        export BUILD_VERSION="${VISION_TAG#v}+${desired_cuda}" PYTORCH_VERSION="${TORCH_TAG#v}" MAX_JOBS="$jobs"
        cd "$vision_src"
        rm -rf build
        "$target_py" -m build --wheel --no-isolation --outdir "$wheel_dir"
    ) 2>&1 | tee "${work}/torchvision-build.log"
    vision_whl="$(ls "${wheel_dir}"/torchvision-*.whl)"
    echo "wheels:"
    ls -l "$wheel_dir"
    echo "=== Installing torchvision into ${target_py}"
    uv pip install --python "$target_py" "$vision_whl"
fi

################################################################################
# 8. Checks

echo "=== Build: CMake picked the NCCL in ${nccl_home}"
cache="${torch_src}/build/CMakeCache.txt"
if [[ -f "$cache" ]]; then
    for var in NCCL_INCLUDE_DIRS NCCL_LIBRARIES; do
        value="$(sed -n "s/^${var}:[A-Z]*=//p" "$cache")"
        [[ -n "$value" && "$(realpath -m "$value")" == "${nccl_home}"/* ]] && ok "${var}=${value}" \
            || bad "${var}=${value:-<unset>} is not under ${nccl_home}"
    done
    for var in CUDNN_INCLUDE_PATH CUSPARSELT_INCLUDE_PATH NVSHMEM_INCLUDE_DIR ACL_INCLUDE_DIR; do
        echo "  ${var}=$(sed -n "s/^${var}:[A-Z]*=//p" "$cache")"
    done
else
    note "${cache} not found; skipped (the probe below still reports the compile-time NCCL)"
fi

echo "=== Installed torch: ${target_py}"
probe="$("${script_dir}/probe-torch-nccl-version.sh" --python "$target_py" 2>&1)" || die "probe-torch-nccl-version.sh failed:"$'\n'"$probe"
field() { sed -n "s/^$1: //p" <<<"$probe"; }
version="$(field torch.version)"
# The version matches the official cu13 wheel; the NCCL checks below tell the two apart.
[[ "$version" == "${TORCH_TAG#v}+cu13"* ]] && ok "torch ${version}" || bad "torch ${version}, expected ${TORCH_TAG#v}+cu13x"
value="$(field build.torch_cuda_nccl_version)"
# probe-torch-nccl-version.sh joins torch.cuda.nccl.version() with ".", suffix included: 2.32.3.-cta
expected="${NCCL_VERSION}${nccl_suffix:+.${nccl_suffix}}"
[[ "$value" == "$expected" ]] && ok "compiled against NCCL ${value}" || bad "compiled against NCCL ${value}, expected ${expected}"
value="$(field runtime.get_runtime_nccl_version)"
[[ "$value" == "$NCCL_VERSION" ]] && ok "loads NCCL ${value} (ncclGetVersion)" || bad "loads NCCL ${value}"
mapfile -t loaded < <(field runtime.libnccl_realpath)
if (( ${#loaded[@]} == 1 )) && [[ "${loaded[0]}" == "${nccl_lib}"/* ]]; then
    ok "the only libnccl mapped is ${loaded[0]}"
else
    bad "libnccl mapped: ${loaded[*]:-none}; expected exactly one under ${nccl_lib}"
fi
grep -q '^wheel\.nvidia-nccl-' <<<"$probe" \
    && bad "an nvidia-nccl-* wheel is installed; torch/__init__.py preloads it, uninstall it" \
    || ok "no nvidia-nccl-* wheel installed"
if command -v readelf >/dev/null; then
    torch_lib="$("$target_py" -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
    dynamic="$(readelf -d "${torch_lib}/libtorch_cuda.so")"
    grep -q 'NEEDED.*\[libnccl\.so\.2\]' <<<"$dynamic" && ok "libtorch_cuda.so: NEEDED libnccl.so.2 (dynamic)" \
        || bad "libtorch_cuda.so does not link libnccl.so.2 dynamically"
    rpath="$(sed -n 's/.*(RPATH).*\[\(.*\)\]/\1/p' <<<"$dynamic")"
    [[ ":${rpath}:" == *":${nccl_lib}:"* ]] && ok "libtorch_cuda.so RPATH has ${nccl_lib}" \
        || bad "libtorch_cuda.so RPATH is ${rpath:-<none>}"
else
    note "readelf not found; skipped the NEEDED / RPATH checks"
fi

echo "=== torch features and torchvision"
# One "status|message" line per check; anything else (a traceback) counts as a failure.
while IFS='|' read -r status message; do
    case "$status" in
        ok) ok "$message" ;;
        warn) note "$message" ;;
        *) bad "${status}${message:+|${message}}" ;;
    esac
done < <("$target_py" - 2>&1 <<'PY'
import torch
import torch.distributed as dist


def emit(cond, message, otherwise="bad"):
    print(f"{'ok' if cond else otherwise}|{message}")


config = dist.ProcessGroupNCCL.Options().config
emit(hasattr(config, "max_p2p_peers"), "NCCLConfig.max_p2p_peers is bound (NCCL_HAS_MAX_P2P_PEERS: NCCL >= 2.30 at compile time)")
try:
    import torchvision
    import torchvision.io as tvio
except Exception as error:
    emit(False, f"import torchvision: {type(error).__name__}: {error}")
    raise SystemExit
emit(torchvision.__version__.startswith("0.28.0"), f"torchvision {torchvision.__version__}")
emit(torchvision.version.cuda == torch.version.cuda, f"torchvision CUDA {torchvision.version.cuda}, torch CUDA {torch.version.cuda}")
image = torch.randint(0, 256, (3, 16, 16), dtype=torch.uint8)
for codec in ("png", "jpeg"):
    try:
        getattr(tvio, f"decode_{codec}")(getattr(tvio, f"encode_{codec}")(image))
        emit(True, f"torchvision {codec} round trip")
    except Exception as error:
        emit(False, f"torchvision built without {codec}: {type(error).__name__}", otherwise="warn")
if torch.cuda.is_available():
    arch = torch.cuda.get_arch_list()
    emit("sm_100" in arch, f"torch arch list {arch}")
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [50.0, 50.0, 60.0, 60.0]], device="cuda")
    keep = torchvision.ops.nms(boxes, torch.tensor([0.9, 0.8, 0.7], device="cuda"), 0.5)
    emit(keep.tolist() == [0, 2], f"torchvision.ops.nms on {torch.cuda.get_device_name()}")
else:
    emit(False, "no GPU visible; skipped the arch list and the CUDA nms run", otherwise="warn")
PY
)

gpus="$("$target_py" -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 0)"
if (( gpus >= 2 )); then
    echo "=== NCCL collectives on 2 GPUs"
    smoke="$(mktemp --suffix=.py)"
    trap 'rm -f "$smoke"' EXIT
    cat >"$smoke" <<'PY'
import os

import torch
import torch.distributed as dist


torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
n = 1 << 20
x = torch.full((n,), float(rank + 1), device="cuda")
dist.all_reduce(x)
assert torch.all(x == world * (world + 1) / 2), "all_reduce"
gathered = torch.empty(world * n, device="cuda")
dist.all_gather_into_tensor(gathered, torch.full((n,), float(rank), device="cuda"))
assert torch.equal(gathered, torch.arange(world, device="cuda", dtype=torch.float).repeat_interleave(n)), "all_gather"
scattered = torch.empty(n, device="cuda")
dist.reduce_scatter_tensor(scattered, torch.ones(world * n, device="cuda"))
assert torch.all(scattered == world), "reduce_scatter"
torch.cuda.synchronize()
dist.destroy_process_group()
if rank == 0:
    print("nccl smoke: all_reduce / all_gather / reduce_scatter correct")
PY
    if output="$(NCCL_DEBUG=VERSION "$target_py" -m torch.distributed.run --standalone --nproc_per_node=2 "$smoke" 2>&1)"; then
        grep -q "nccl smoke: .* correct" <<<"$output" && ok "all_reduce / all_gather_into_tensor / reduce_scatter_tensor correct" \
            || bad "smoke run printed no result"
    else
        bad "smoke run failed"
    fi
    banner="$(grep -m1 -o "NCCL version [^ ]*" <<<"$output" || true)"
    # "+cuda" ends the version in the banner, so a stock 2.32.3 does not pass for 2.32.3-cta or the reverse.
    [[ "$banner" == "NCCL version ${NCCL_VERSION}${nccl_suffix}+"* ]] && ok "${banner} (NCCL_DEBUG=VERSION)" \
        || { bad "NCCL banner: ${banner:-none}"; tail -n 20 <<<"$output" | sed 's/^/    /'; }
else
    note "${gpus} GPU(s) visible; skipped the 2-GPU collective run (run this on the GB200 node)"
fi

echo
if (( failures )); then
    die "${failures} check(s) failed"
fi
echo "=== torch ${version} + torchvision ready in ${target_py}, NCCL ${NCCL_VERSION}${nccl_suffix} from ${nccl_lib}"
echo "  Transformer Engine has to be rebuilt against the same NCCL: NCCL_HOME=${nccl_home}"
