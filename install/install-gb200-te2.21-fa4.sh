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

# GB200 environment with Transformer Engine 2.21.0 + FlashAttention-4 (context parallel
# capable) + HybridEP, installed WITHOUT uv.lock (no `uv sync`).
#
# Version constraints live in install/overrides-te2.21-fa4.txt.
# Run from the repository root on a GB200 node:
#   bash install/install-gb200-te2.21-fa4.sh

set -eo pipefail

TE_VERSION="2.21.0"
FA4_VERSION="4.0.0b33"

REPO_ROOT="$(pwd)"
BASE_PYTHON_ENV="${REPO_ROOT}/.python/python3.12.12"
VENV_DIR="${REPO_ROOT}/.venv/python3.12-torch2.13-engine${TE_VERSION%.*}-hybrid-ep"
OVERRIDES="${REPO_ROOT}/install/overrides-te2.21-fa4.txt"
TMP_DIR="${REPO_ROOT}/.tmp"
TE_SRC_DIR="${TMP_DIR}/TransformerEngine-v${TE_VERSION}"

if [[ ! -f "${REPO_ROOT}/pyproject.toml" || ! -f "${OVERRIDES}" ]]; then
    echo "Error: run this script from the Megatron-Bridge repository root"
    exit 1
fi
if [[ ! -f "${REPO_ROOT}/3rdparty/Megatron-LM/pyproject.toml" ]]; then
    echo "Error: 3rdparty/Megatron-LM is not initialized (see documents/installation/submodule.md)"
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

# --no-config: uv pip otherwise merges pyproject.toml [tool.uv] override-dependencies
# (nvidia-cudnn-frontend==1.26.0, ...) with OVERRIDES and the resolution becomes unsatisfiable.
UV_PIP_BASE=(
    uv pip install
    --no-config
    --extra-index-url https://pypi.nvidia.com
    --index-strategy unsafe-best-match
)
UV_PIP_INSTALL=("${UV_PIP_BASE[@]}" --override "${OVERRIDES}")

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
source "${VENV_DIR}/bin/activate"
echo "VIRTUAL_ENV: ${VIRTUAL_ENV}"

mkdir -p "${TMP_DIR}"

##############################################################################
## 2. PyTorch + patch
##############################################################################

echo "=== Installing PyTorch 2.13.0 ==="
"${UV_PIP_INSTALL[@]}" setuptools wheel torch==2.13.0 torchvision==0.28.0 triton==3.7.1
bash install/torch-patch.sh

##############################################################################
## 3. Build tools, cuDNN frontend and CUTLASS DSL (TE 2.21 reads cudnn-frontend headers at build time)
##############################################################################

echo "=== Installing build tools, nvidia-cudnn-frontend and nvidia-cutlass-dsl ==="
"${UV_PIP_INSTALL[@]}" pybind11 "Cython>=3.0.0" numpy ninja nvidia-mathdx \
    nvidia-cudnn-frontend "nvidia-cutlass-dsl[cu13]"

##############################################################################
## 4. Transformer Engine from source
##############################################################################

echo "=== Building Transformer Engine v${TE_VERSION} ==="
NCCL_DIR=$(python -c "import nvidia.nccl; print(nvidia.nccl.__path__[0])" 2>/dev/null || true)
if [[ -n "${NCCL_DIR}" ]]; then
    export NCCL_HOME="${NCCL_DIR}"
    export CPLUS_INCLUDE_PATH="${NCCL_HOME}/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
    export LD_LIBRARY_PATH="${NCCL_HOME}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

if [[ ! -d "${TE_SRC_DIR}" ]]; then
    git clone --branch "v${TE_VERSION}" --depth 1 --recurse-submodules --shallow-submodules \
        https://github.com/NVIDIA/TransformerEngine.git "${TE_SRC_DIR}"
fi

# An override replaces every requirement on its package, including this local source path,
# so the transformer-engine pin would send uv to the registry, which has no 2.21.0. Build with
# the remaining overrides; the pin still guards the later steps against swapping TE out.
TE_BUILD_OVERRIDES="${TMP_DIR}/overrides-te2.21-fa4-te-build.txt"
grep -vE '^transformer-engine([[:space:]=<>!~@;\[]|$)' "${OVERRIDES}" > "${TE_BUILD_OVERRIDES}"

# NVTE_WITH_NCCL_EP=0: the NCCL-EP extension needs a newer NCCL than torch 2.13 ships.
NVTE_FRAMEWORK=pytorch \
    NVTE_CUDA_ARCHS="100" \
    NVTE_WITH_NCCL_EP=0 \
    NVTE_NO_LOCAL_VERSION=1 \
    MAX_JOBS="${MAX_JOBS:-8}" \
    "${UV_PIP_BASE[@]}" --override "${TE_BUILD_OVERRIDES}" --no-build-isolation -v "${TE_SRC_DIR}"

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
## 7. HybridEP
##############################################################################

bash install/install-hybrid-ep.sh

##############################################################################
## 8. Verification
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
    "deep_ep",
):
    try:
        print(f"{dist:30s} {md.version(dist)}")
    except md.PackageNotFoundError:
        print(f"{dist:30s} NOT INSTALLED")

import torch
from transformer_engine.pytorch.attention.dot_product_attention import backends  # noqa: F401
from transformer_engine.pytorch.attention.dot_product_attention.utils import FlashAttentionUtils
from transformer_engine.pytorch.utils import get_cudnn_version

print(f"{'cuDNN (TE runtime)':30s} {get_cudnn_version()}")
print(f"{'cuDNN (torch)':30s} {torch.backends.cudnn.version()}")
print(f"{'libcudnn loaded':30s} {sorted({l.split()[-1] for l in open('/proc/self/maps') if 'libcudnn.so' in l})}")
print(f"{'FA4 detected by TE':30s} {FlashAttentionUtils.v4_is_installed} ({FlashAttentionUtils.fa4_version})")
PY

echo "=== Installation complete ==="
echo "Activate with: source ${VENV_DIR#${REPO_ROOT}/}/bin/activate"
echo "Do NOT run 'uv sync' in this environment: it would restore the uv.lock versions."
