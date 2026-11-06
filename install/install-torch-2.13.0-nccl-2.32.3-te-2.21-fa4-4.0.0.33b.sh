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

# GB200 environment: torch 2.13.0 built from source against NCCL 2.32.3 + Transformer Engine 2.21.0
# + FlashAttention-4 4.0.0b33 + Apex + HybridEP, installed WITHOUT uv.lock (no `uv sync`).
#
# The steps are those of install-gb200-te2.21-fa4.sh, except:
#   torch    install-torch-2.13.0-nccl-2.32.3.sh builds and installs torch / torchvision against the stock
#            NCCL build (--nccl-home), whose lib dir is in the RPATH of libtorch_cuda.so. A torch wheel already
#            in .tmp/wheels/torch-2.13.0-nccl-2.32.3 is reused (--rebuild-torch to build it again); the checks
#            of that script reject a wheel built against another NCCL.
#   TE       compiled against the same nccl.h (TE's CMake would otherwise pick a system or wheel nccl.h), with
#            NCCL EP on: it needs NCCL >= 2.30.4, which the PyPI torch's 2.29.7 did not meet.
#   Apex     _apex_nccl_allocator (APEX_ALL_CONTRIB_EXT) links -lnccl from the same build.
#   overrides nvidia-nccl-cu12 / -cu13 are dropped: torch/__init__.py would preload the wheel's libnccl.
#
# Switching NCCL. Every build of install-nccl-2.32.3*.sh is 2.32.3 (NCCL_VERSION_CODE 23203) with the soname
# libnccl.so.2, and libtorch_cuda.so, TE's NCCL EP, Apex and DeepEP all ask for libnccl.so.2 by soname. A
# libnccl.so.2 in LD_PRELOAD is loaded first and serves all of them, so one environment runs either build:
#   source .venv/<env>/bin/activate
#   nccl_use pause    # LD_PRELOAD=<megatron-bridge>/nccl/nccl-2.32.3-pause/build/lib/libnccl.so.2
#   nccl_use stock    # drop it: back to the RPATH build
#   nccl_use /path/to/nccl-2.32.3-<variant>/build/lib/libnccl.so.2
# torchrun / srun / uv run inherit LD_PRELOAD. torch.cuda.nccl.version() reports the headers torch was compiled
# with (2.32.3, no suffix); NCCL_DEBUG=VERSION prints the loaded build ("NCCL version 2.32.3-pause+cuda...").
# The sweeps (examples/gb200/nvls-collective/run_symm_cta_pause_sweep.sh,
# examples/gb200/hbm-conflict/run_hbm_conflict_benchmark.sh) preload the pause build themselves.
# deactivate does not reset LD_PRELOAD; run nccl_use stock first.
#
# Prerequisites (both are checked before anything is installed):
#   bash install/install-nccl-2.32.3.sh          # stock build: nccl/nccl-2.32.3/build
#   bash install/install-nccl-2.32.3-pause.sh    # pause build: nccl/nccl-2.32.3-pause/build
#
# Run from the repository root on a GB200 node:
#   bash install/install-torch-2.13.0-nccl-2.32.3-te-2.21-fa4-4.0.0.33b.sh [options]
#
#   --nccl-home DIR        Stock NCCL 2.32.3 build torch / TE / Apex link (default: nccl/nccl-2.32.3/build)
#   --nccl-pause-home DIR  nccl-2.32.3-pause build for nccl_use pause (default: nccl/nccl-2.32.3-pause/build)
#   --torch-jobs N         MAX_JOBS of the torch build (default: that script's min(nproc - 2, 12))
#   --rebuild-torch        Build the torch wheel even if one is already in the wheel dir
#   --no-nccl-ep           Build TE with NVTE_WITH_NCCL_EP=0

set -eo pipefail

TE_VERSION="2.21.0"
FA4_VERSION="4.0.0b33"
APEX_TAG="25.09"
NCCL_VERSION="2.32.3"
NCCL_PAUSE_SUFFIX="-pause"  # set by install/patches/nccl-2.32.3-pause.patch

REPO_ROOT="$(pwd)"
BASE_PYTHON_ENV="${REPO_ROOT}/.python/python3.12.12"
VENV_DIR="${REPO_ROOT}/.venv/python3.12-torch2.13-nccl${NCCL_VERSION}-engine${TE_VERSION%.*}-hybrid-ep"
OVERRIDES="${REPO_ROOT}/install/overrides-te2.21-fa4.txt"
TMP_DIR="${REPO_ROOT}/.tmp"
# Separate from install-gb200-te2.21-fa4.sh: its CMake cache holds the nvidia-nccl-cu13 wheel's nccl.h.
TE_SRC_DIR="${TMP_DIR}/TransformerEngine-v${TE_VERSION}-nccl-${NCCL_VERSION}"
APEX_SRC_DIR="${TMP_DIR}/apex"
# install-torch-2.13.0-nccl-2.32.3.sh with its default --work-dir (.tmp)
TORCH_WHEEL_DIR="${TMP_DIR}/wheels/torch-2.13.0-nccl-${NCCL_VERSION}"

NCCL_STOCK_HOME="${REPO_ROOT}/nccl/nccl-${NCCL_VERSION}/build"
NCCL_PAUSE_HOME="${REPO_ROOT}/nccl/nccl-${NCCL_VERSION}-pause/build"
TORCH_JOBS=""
REBUILD_TORCH=false
WITH_NCCL_EP=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nccl-home) NCCL_STOCK_HOME="$2"; shift 2 ;;
        --nccl-pause-home) NCCL_PAUSE_HOME="$2"; shift 2 ;;
        --torch-jobs) TORCH_JOBS="$2"; shift 2 ;;
        --rebuild-torch) REBUILD_TORCH=true; shift ;;
        --no-nccl-ep) WITH_NCCL_EP=0; shift ;;
        -h|--help) sed -n '16,53p' "$0"; exit 0 ;;
        *) echo "Error: unknown argument: $1"; exit 1 ;;
    esac
done

if [[ ! -f "${REPO_ROOT}/pyproject.toml" || ! -f "${OVERRIDES}" ]]; then
    echo "Error: run this script from the Megatron-Bridge repository root"
    exit 1
fi
if [[ ! -f "${REPO_ROOT}/3rdparty/Megatron-LM/pyproject.toml" ]]; then
    echo "Error: 3rdparty/Megatron-LM is not initialized (see documents/installation/submodule.md)"
    exit 1
fi
if [[ "$(uname -s)" != Linux || "$(uname -m)" != aarch64 ]]; then
    echo "Error: only the GB200 (Linux aarch64) build is scripted"
    exit 1
fi
# The torch checks expect the RPATH build to be the only libnccl in the process.
if [[ "${LD_PRELOAD:-}" == *libnccl* ]]; then
    echo "Error: LD_PRELOAD has a libnccl (${LD_PRELOAD}); unset it (nccl_use stock) and rerun"
    exit 1
fi
command -v readelf >/dev/null || { echo "Error: readelf (binutils) not found"; exit 1; }

##############################################################################
## 0. The two NCCL builds: same version and soname, so either one serves torch
##############################################################################

echo "=== NCCL ${NCCL_VERSION} builds"
if [[ ! -f "${NCCL_STOCK_HOME}/include/nccl.h" || ! -e "${NCCL_STOCK_HOME}/lib/libnccl.so.2" ]]; then
    echo "Error: no NCCL build in ${NCCL_STOCK_HOME}; run install/install-nccl-${NCCL_VERSION}.sh or pass --nccl-home"
    exit 1
fi
if [[ ! -e "${NCCL_PAUSE_HOME}/lib/libnccl.so.2" ]]; then
    echo "Error: no NCCL build in ${NCCL_PAUSE_HOME}; run install/install-nccl-${NCCL_VERSION}-pause.sh or pass --nccl-pause-home"
    exit 1
fi
# The patch, the banner "NCCL version 2.32.3-pause+cuda" and both pause variables.
bash install/install-nccl-${NCCL_VERSION}-pause.sh --check-only --dest "$(dirname -- "${NCCL_PAUSE_HOME}")"

NCCL_STOCK_HOME="$(cd -- "${NCCL_STOCK_HOME}" && pwd -P)"
NCCL_PAUSE_HOME="$(cd -- "${NCCL_PAUSE_HOME}" && pwd -P)"
NCCL_STOCK_LIB="${NCCL_STOCK_HOME}/lib/libnccl.so.2"
NCCL_PAUSE_LIB="${NCCL_PAUSE_HOME}/lib/libnccl.so.2"
nccl_code() { sed -n 's/^#define NCCL_VERSION_CODE \([0-9]*\).*/\1/p' "$1/include/nccl.h"; }
nccl_soname() { readelf -d "$1" | sed -n 's/.*(SONAME).*\[\(.*\)\]/\1/p'; }
for home in "${NCCL_STOCK_HOME}" "${NCCL_PAUSE_HOME}"; do
    echo "  ${home}: NCCL_VERSION_CODE $(nccl_code "${home}"), SONAME $(nccl_soname "${home}/lib/libnccl.so.2")"
done
if [[ "$(nccl_code "${NCCL_STOCK_HOME}")" != "$(nccl_code "${NCCL_PAUSE_HOME}")" ]]; then
    echo "Error: the stock and pause builds have different NCCL_VERSION_CODEs; LD_PRELOAD cannot swap them"
    exit 1
fi
if [[ "$(nccl_soname "${NCCL_STOCK_LIB}")" != libnccl.so.2 || "$(nccl_soname "${NCCL_PAUSE_LIB}")" != libnccl.so.2 ]]; then
    echo "Error: both builds need the soname libnccl.so.2 to stand in for each other"
    exit 1
fi

# Build environment for GB200 (same as documents/installation/installation.md)
export NVTE_BUILD_NUM_PHILOX_ROUNDS=3
export TORCH_CUDA_ARCH_LIST="10.0"
export NVCC_THREADS=16
export FLASH_MLA_DISABLE_SM90=1
export UV_HTTP_TIMEOUT=120
export UV_LINK_MODE=copy
export CFLAGS="-I/usr/local/cuda/include/cccl -DNDEBUG"
export CXXFLAGS="-I/usr/local/cuda/include/cccl -DNDEBUG"
export MAMBA_FORCE_BUILD=TRUE
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
export FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE

mkdir -p "${TMP_DIR}"

# An nvidia-nccl-cu* wheel pulled in by any dependency would be preloaded by torch/__init__.py ahead of the
# RPATH (and of LD_PRELOAD's soname match), so drop the requirement everywhere.
OVERRIDES_NCCL="${TMP_DIR}/overrides-te2.21-fa4-nccl-${NCCL_VERSION}.txt"
{
    cat "${OVERRIDES}"
    echo
    echo "# torch loads libnccl.so.2 from ${NCCL_STOCK_HOME}/lib (RPATH) or LD_PRELOAD."
    echo "nvidia-nccl-cu12; sys_platform == 'never'"
    echo "nvidia-nccl-cu13; sys_platform == 'never'"
} > "${OVERRIDES_NCCL}"

# --no-config: uv pip otherwise merges pyproject.toml [tool.uv] override-dependencies
# (nvidia-cudnn-frontend==1.26.0, ...) with OVERRIDES and the resolution becomes unsatisfiable.
UV_PIP_BASE=(
    uv pip install
    --no-config
    --extra-index-url https://pypi.nvidia.com
    --index-strategy unsafe-best-match
)
UV_PIP_INSTALL=("${UV_PIP_BASE[@]}" --override "${OVERRIDES_NCCL}")

##############################################################################
## 1. Virtual environment
##############################################################################

echo "=== Creating virtual environment at ${VENV_DIR} ==="
if [[ -d "${VENV_DIR}" ]]; then
    echo "Error: ${VENV_DIR} already exists; remove it first to reinstall"
    exit 1
fi
source "${BASE_PYTHON_ENV}/bin/activate"
virtualenv -p "${BASE_PYTHON_ENV}/bin/python" "${VENV_DIR}"
deactivate
# UV_NO_SYNC keeps `uv run` from re-syncing this environment back to uv.lock.
echo "export UV_PROJECT_ENVIRONMENT=\"\$VIRTUAL_ENV\"" >> "${VENV_DIR}/bin/activate"
echo "export UV_NO_SYNC=1" >> "${VENV_DIR}/bin/activate"
# TE loads cuDNN from CUDNN_HOME / CUDA_HOME / /usr/local/cuda / LD_LIBRARY_PATH before the pip
# package, RTLD_GLOBAL; a system cuDNN found there then also serves torch (same libcudnn.so.9
# soname) and torch rejects it as a version mismatch. Point both at torch's bundled cuDNN.
cat >> "${VENV_DIR}/bin/activate" <<'EOF'
export CUDNN_HOME="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cudnn"
export LD_LIBRARY_PATH="$CUDNN_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
EOF
cat >> "${VENV_DIR}/bin/activate" <<EOF
# libtorch_cuda.so finds libnccl.so.2 in ${NCCL_STOCK_HOME}/lib (RPATH); a libnccl.so.2 in
# LD_PRELOAD is loaded first and replaces it in torch, TE, Apex and DeepEP. nccl_use switches it.
export NCCL_STOCK_LIB="${NCCL_STOCK_LIB}"
export NCCL_PAUSE_LIB="${NCCL_PAUSE_LIB}"
EOF
cat >> "${VENV_DIR}/bin/activate" <<'EOF'
nccl_use() {
    local lib rest
    case "${1:-}" in
        stock) lib="" ;;
        pause) lib="$NCCL_PAUSE_LIB" ;;
        /*) lib="$1" ;;
        *) echo "usage: nccl_use stock | pause | /path/to/libnccl.so.2" >&2; return 1 ;;
    esac
    if [ -n "$lib" ] && [ ! -f "$lib" ]; then
        echo "nccl_use: ${lib} not found" >&2
        return 1
    fi
    # Keep whatever else is preloaded, minus any libnccl.
    rest="$(printf '%s' "${LD_PRELOAD:-}" | tr ':' '\n' | { grep -v '/libnccl\.so' || true; } | paste -sd: -)"
    if [ -n "$lib" ]; then
        export LD_PRELOAD="${lib}${rest:+:${rest}}"
    elif [ -n "$rest" ]; then
        export LD_PRELOAD="$rest"
    else
        unset LD_PRELOAD
    fi
    echo "NCCL: ${lib:-$NCCL_STOCK_LIB}" >&2
}
EOF
source "${VENV_DIR}/bin/activate"
echo "VIRTUAL_ENV: ${VIRTUAL_ENV}"

##############################################################################
## 2. PyTorch from source against NCCL 2.32.3 + patch
##############################################################################

"${UV_PIP_INSTALL[@]}" setuptools wheel

TORCH_ARGS=(--python "${VIRTUAL_ENV}/bin/python" --nccl-home "${NCCL_STOCK_HOME}")
[[ -n "${TORCH_JOBS}" ]] && TORCH_ARGS+=(--jobs "${TORCH_JOBS}")
if ! ${REBUILD_TORCH} && ls "${TORCH_WHEEL_DIR}"/torch-*.whl >/dev/null 2>&1; then
    echo "=== Reusing the torch wheel in ${TORCH_WHEEL_DIR} (--rebuild-torch to build it again) ==="
    TORCH_ARGS+=(--skip-torch-build)
else
    echo "=== Building PyTorch 2.13.0 against ${NCCL_STOCK_HOME} ==="
fi
bash install/install-torch-2.13.0-nccl-${NCCL_VERSION}.sh "${TORCH_ARGS[@]}"
bash install/torch-patch.sh

##############################################################################
## 3. Build tools, cuDNN frontend and CUTLASS DSL (TE 2.21 reads cudnn-frontend headers at build time)
##############################################################################

echo "=== Installing build tools, nvidia-cudnn-frontend and nvidia-cutlass-dsl ==="
"${UV_PIP_INSTALL[@]}" pybind11 "Cython>=3.0.0" numpy ninja nvidia-mathdx \
    nvidia-cudnn-frontend "nvidia-cutlass-dsl[cu13]"

##############################################################################
## 4. Transformer Engine from source, against the stock NCCL headers
##############################################################################

echo "=== Building Transformer Engine v${TE_VERSION} (NVTE_WITH_NCCL_EP=${WITH_NCCL_EP}) ==="
# NCCL_HOME: NCCL EP's build (setup.py _discover_nccl_home). CPLUS_INCLUDE_PATH: the PyTorch extension,
# whose ProcessGroupNCCL.hpp includes nccl.h. LIBRARY_PATH: -lnccl of NCCL EP and of Apex below.
export NCCL_HOME="${NCCL_STOCK_HOME}"
export CPLUS_INCLUDE_PATH="${NCCL_HOME}/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export LIBRARY_PATH="${NCCL_HOME}/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="${NCCL_HOME}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [[ ! -d "${TE_SRC_DIR}" ]]; then
    git clone --branch "v${TE_VERSION}" --depth 1 --recurse-submodules --shallow-submodules \
        https://github.com/NVIDIA/TransformerEngine.git "${TE_SRC_DIR}"
fi

# An override replaces every requirement on its package, including this local source path,
# so the transformer-engine pin would send uv to the registry, which has no 2.21.0. Build with
# the remaining overrides; the pin still guards the later steps against swapping TE out.
TE_BUILD_OVERRIDES="${TMP_DIR}/overrides-te2.21-fa4-nccl-${NCCL_VERSION}-te-build.txt"
grep -vE '^transformer-engine([[:space:]=<>!~@;\[]|$)' "${OVERRIDES_NCCL}" > "${TE_BUILD_OVERRIDES}"

# -DNCCL_INCLUDE_DIR: common/CMakeLists.txt find_path() only hints at the nvidia-nccl wheel, /opt/nvidia/nccl
# and /usr/local/nccl, and would settle on a system nccl.h. --refresh-package: uv caches the wheel built from
# this directory and would not notice a change of NVTE_WITH_NCCL_EP or NCCL.
NVTE_FRAMEWORK=pytorch \
    NVTE_CUDA_ARCHS="100" \
    NVTE_WITH_NCCL_EP="${WITH_NCCL_EP}" \
    NVTE_NO_LOCAL_VERSION=1 \
    NVTE_CMAKE_EXTRA_ARGS="-DNCCL_INCLUDE_DIR=${NCCL_HOME}/include" \
    MAX_JOBS="${MAX_JOBS:-8}" \
    "${UV_PIP_BASE[@]}" --override "${TE_BUILD_OVERRIDES}" --refresh-package transformer-engine \
    --no-build-isolation -v "${TE_SRC_DIR}"

##############################################################################
## 5. FlashAttention-4
##############################################################################

echo "=== Installing flash-attn-4 ${FA4_VERSION} ==="
"${UV_PIP_INSTALL[@]}" "flash-attn-4[cu13]"

##############################################################################
## 6. Megatron-Core + Megatron-Bridge (editable, resolved without uv.lock)
##############################################################################

# --no-sources: ignore [tool.uv.sources] so Megatron-LM's transformer-engine git pin is not
# rebuilt; megatron-core is satisfied by the editable checkout passed in the same command.
echo "=== Installing Megatron-Core and Megatron-Bridge ==="
"${UV_PIP_INSTALL[@]}" --no-sources \
    --no-build-isolation-package fast-hadamard-transform \
    --no-build-isolation-package flash-linear-attention \
    -e "./3rdparty/Megatron-LM[dev,mlm]" \
    -e "."

# Create symlink for tilelang's libcudart_stub.so if tilelang is installed
TILELANG_STUB="${VENV_DIR}/lib/python3.12/site-packages/tilelang/lib/libcudart_stub.so"
if [[ -f "${TILELANG_STUB}" ]]; then
    LIBCUDART_PATH=$(ldconfig -p 2>/dev/null | awk '/libcudart\.so\.[0-9]+ /{print $NF; exit}' || echo "")
    if [[ -n "${LIBCUDART_PATH}" ]]; then
        ln -sf "${LIBCUDART_PATH}" "${TILELANG_STUB}"
    fi
fi

##############################################################################
## 7. Apex (fused_weight_gradient_mlp_cuda for GPTModel's ColumnParallelLinear output_layer)
##############################################################################

# Without it, gradient_accumulation_fusion=True (the default when TE is importable) fails in
# ColumnParallelLinear. The source torch has the version of the PyPI one (2.13.0+cu130), so the wheel
# dir also names the NCCL: the contrib NCCL extensions are compiled against the headers found here.
echo "=== Installing NVIDIA Apex ${APEX_TAG} ==="
TORCH_VERSION=$(python -c "import torch; print(torch.__version__.replace('+', '-'))")
APEX_WHEEL_DIR="${TMP_DIR}/wheels/apex-${APEX_TAG}-torch${TORCH_VERSION}-nccl-${NCCL_VERSION}"
APEX_WHEEL=$(ls "${APEX_WHEEL_DIR}"/apex-*.whl 2>/dev/null | head -n 1 || true)
if [[ -z "${APEX_WHEEL}" ]]; then
    if [[ ! -d "${APEX_SRC_DIR}" ]]; then
        git clone https://github.com/NVIDIA/apex.git "${APEX_SRC_DIR}"
    fi
    git -C "${APEX_SRC_DIR}" checkout "${APEX_TAG}"
    # Objects left by a build against another torch / NCCL would be linked as they are.
    rm -rf "${APEX_SRC_DIR}/build"

    NVCC_APPEND_FLAGS="--threads 32" \
        APEX_PARALLEL_BUILD=32 \
        APEX_CPP_EXT=1 \
        APEX_CUDA_EXT=1 \
        APEX_ALL_CONTRIB_EXT=1 \
        uv build --no-config --wheel --no-build-isolation --python "${VIRTUAL_ENV}/bin/python" \
        --out-dir "${APEX_WHEEL_DIR}" "${APEX_SRC_DIR}"
    APEX_WHEEL=$(ls "${APEX_WHEEL_DIR}"/apex-*.whl | head -n 1)
fi
echo "Using ${APEX_WHEEL}"
"${UV_PIP_INSTALL[@]}" --no-deps "${APEX_WHEEL}"

##############################################################################
## 8. HybridEP
##############################################################################

bash install/install-hybrid-ep.sh

##############################################################################
## 9. Verification
##############################################################################

echo "=== Installed versions ==="
python - <<'PY'
import importlib.metadata as md

for dist in (
    "torch",
    "transformer_engine",
    "flash-attn-4",
    "nvidia-cutlass-dsl",
    "nvidia-cutlass-dsl-libs-cu13",
    "nvidia-cudnn-frontend",
    "megatron-core",
    "megatron-bridge",
    "apex",
    "deep_ep",
    "nvidia-nccl-cu13",
):
    try:
        print(f"{dist:30s} {md.version(dist)}")
    except md.PackageNotFoundError:
        print(f"{dist:30s} NOT INSTALLED")

import fused_weight_gradient_mlp_cuda  # noqa: F401
import torch
from transformer_engine.pytorch.attention.dot_product_attention import backends  # noqa: F401
from transformer_engine.pytorch.attention.dot_product_attention.utils import FlashAttentionUtils
from transformer_engine.pytorch.utils import get_cudnn_version

print(f"{'cuDNN (TE runtime)':30s} {get_cudnn_version()}")
print(f"{'cuDNN (torch)':30s} {torch.backends.cudnn.version()}")
print(f"{'libcudnn loaded':30s} {sorted({l.split()[-1] for l in open('/proc/self/maps') if 'libcudnn.so' in l})}")
print(f"{'FA4 detected by TE':30s} {FlashAttentionUtils.v4_is_installed} ({FlashAttentionUtils.fa4_version})")
PY

# The later steps must not have swapped torch for the PyPI wheel or brought an nvidia-nccl wheel back:
# RPATH, a single libnccl from the stock build, and the 2-GPU collectives with its banner.
echo "=== torch NCCL checks (install-torch-2.13.0-nccl-${NCCL_VERSION}.sh --check-only) ==="
bash install/install-torch-2.13.0-nccl-${NCCL_VERSION}.sh --python "${VIRTUAL_ENV}/bin/python" \
    --nccl-home "${NCCL_STOCK_HOME}" --check-only

##############################################################################
## 10. NCCL switch: stock (RPATH) and pause (nccl_use pause) in the full stack
##############################################################################

echo "=== NCCL switch: stock and ${NCCL_PAUSE_SUFFIX#-} builds ==="
SWITCH_CHECK="${TMP_DIR}/nccl-switch-check.py"
cat > "${SWITCH_CHECK}" <<'PY'
"""Load the stack that uses NCCL and check that only argv[1] serves libnccl.so.2."""

import ctypes
import glob
import importlib
import os
import sys

import torch
import torch.distributed as dist
import transformer_engine
import transformer_engine.pytorch  # noqa: F401


for module in ("deep_ep", "_apex_nccl_allocator", "nccl_p2p_cuda"):
    try:
        importlib.import_module(module)
    except ImportError:
        pass
# TE loads NCCL EP on first use; load it here so it resolves libnccl.so.2 as well.
for path in glob.glob(os.path.join(os.path.dirname(transformer_engine.__file__), "**", "libnccl_ep*.so*"), recursive=True):
    ctypes.CDLL(path)

want = os.path.realpath(sys.argv[1])
with open("/proc/self/maps") as maps:
    paths = {line.split(None, 5)[5].strip() for line in maps if len(line.split(None, 5)) == 6}
loaded = sorted({os.path.realpath(p) for p in paths if os.path.basename(p).startswith("libnccl.so")})
if loaded != [want]:
    sys.exit(f"libnccl mapped: {loaded}, expected only {want}")
runtime = ".".join(map(str, dist.ProcessGroupNCCL.get_runtime_nccl_version()))

if "LOCAL_RANK" in os.environ:
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
    torch.cuda.synchronize()
    dist.destroy_process_group()
    if rank != 0:
        sys.exit(0)
print(f"nccl switch ok: {want} (ncclGetVersion {runtime})")
PY

GPUS="$(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 0)"
for build in stock pause; do
    if [[ "${build}" == stock ]]; then lib="${NCCL_STOCK_LIB}" suffix=""; else lib="${NCCL_PAUSE_LIB}" suffix="${NCCL_PAUSE_SUFFIX}"; fi
    (
        nccl_use "${build}"
        echo "  LD_PRELOAD=${LD_PRELOAD:-}"
        python "${SWITCH_CHECK}" "${lib}"
        if (( GPUS >= 2 )); then
            output="$(NCCL_DEBUG=VERSION python -m torch.distributed.run --standalone --nproc_per_node=2 \
                "${SWITCH_CHECK}" "${lib}" 2>&1)" || { tail -n 30 <<<"${output}"; exit 1; }
            grep "nccl switch ok" <<<"${output}"
            banner="$(grep -m1 -o "NCCL version [^ ]*" <<<"${output}" || true)"
            # "+cuda" ends the version in the banner, so the stock build does not pass for -pause or the reverse.
            if [[ "${banner}" != "NCCL version ${NCCL_VERSION}${suffix}+"* ]]; then
                echo "Error: NCCL banner '${banner:-none}', expected NCCL version ${NCCL_VERSION}${suffix}+cuda..."
                exit 1
            fi
            echo "  ${banner} on 2 GPUs (all_reduce / all_gather_into_tensor correct)"
        else
            echo "  ${GPUS} GPU(s) visible; skipped the 2-GPU run with the ${build} build"
        fi
    )
done

echo "=== Installation complete ==="
echo "Activate with: source ${VENV_DIR#${REPO_ROOT}/}/bin/activate"
echo "Switch NCCL:   nccl_use pause | nccl_use stock   (torch links ${NCCL_STOCK_LIB})"
echo "Do NOT run 'uv sync' in this environment: it would restore the uv.lock versions."
