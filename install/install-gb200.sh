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

# Installation script for GB200 (Blackwell) architecture
# Usage: ./install-gb200.sh [--ep hybrid|deep]

set -e

# Default values (hybrid-ep)
EP_TYPE="hybrid"
DEEPEP_COMMIT="34152ae28f80bcc3ee38d7a12cb2ad87cfd4ea72"
INSTALL_DEEPEP="True"
REINSTALL_NVSHMEM="True"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --ep)
            EP_TYPE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--ep hybrid|deep]"
            echo "  --ep: Expert parallelism type (hybrid or deep), default: hybrid"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate EP_TYPE
if [[ "$EP_TYPE" != "hybrid" && "$EP_TYPE" != "deep" ]]; then
    echo "Error: --ep must be 'hybrid' or 'deep'"
    exit 1
fi

# Set DEEPEP_COMMIT based on EP_TYPE
if [[ "$EP_TYPE" == "deep" ]]; then
    DEEPEP_COMMIT="018f21ea62c6c2888d9b46806aad6d90b399b23d"
fi

echo "=== Installing Megatron-Bridge for GB200 (Blackwell) ==="
echo "Expert Parallelism type: $EP_TYPE"

# Determine virtual environment suffix
if [[ "$EP_TYPE" == "hybrid" ]]; then
    VENV_SUFFIX="hybrid-ep"
else
    VENV_SUFFIX="deep-ep"
fi

VENV_DIR=".venv/python3.12-torch2.13-${VENV_SUFFIX}"
VENV_PATH="$(pwd)/${VENV_DIR}"

# Set environment variables for GB200
export NVTE_BUILD_NUM_PHILOX_ROUNDS=3
export HYBRID_EP_MULTINODE=1
export TORCH_CUDA_ARCH_LIST="10.0"  # Blackwell architecture
export NVCC_THREADS=16
export FLASH_MLA_DISABLE_SM90=1

# Set UV environment
export UV_HTTP_TIMEOUT=120
export UV_LINK_MODE=copy

echo "=== Creating virtual environment at ${VENV_DIR} ==="

# Create virtual environment with Python 3.12 using virtualenv
virtualenv -p python3.12 "${VENV_PATH}"

# Append UV_PROJECT_ENVIRONMENT to activate script
echo "export UV_PROJECT_ENVIRONMENT=\"\$VIRTUAL_ENV\"" >> "${VENV_PATH}/bin/activate"

# Activate the virtual environment
source "${VENV_PATH}/bin/activate"

echo "=== Virtual environment activated ==="
echo "VIRTUAL_ENV: $VIRTUAL_ENV"
echo "UV_PROJECT_ENVIRONMENT: $UV_PROJECT_ENVIRONMENT"

# The following system packages are typically installed via apt in Docker.
# For non-Docker environments, ensure these are installed manually:
#
# apt-get update
# apt install -y --only-upgrade gnupg openssl libssl3t64
# apt-get install -y --no-install-recommends libnvidia-ml-dev
#
# For DeepEP (deep expert parallelism), you may also need:
# apt-get install -y libmlx5-dev (or create symlink manually)
#
# Note: Clean up after installation:
# apt-get clean && rm -rf /var/lib/apt/lists/*

# Create temporary directory for builds
TMP_DIR="$(pwd)/.tmp"
mkdir -p "${TMP_DIR}"

echo "=== Installing setuptools ==="
uv pip install setuptools

echo "=== Installing PyTorch 2.13.0 with Blackwell (12.0) support ==="
uv pip install torch==2.13.0 torchvision==0.28.0

##############################################################################
##
## Patch PyTorch library finalizer to fix shutdown crashes
##
##############################################################################

echo "=== Patching PyTorch library.py ==="

# Find torch site-packages location
TORCH_SITE_PACKAGES=$(python -c 'from pathlib import Path; import torch.library; print(Path(torch.library.__file__).parent.parent)')
LIBRARY_PY="${TORCH_SITE_PACKAGES}/torch/library.py"

if [[ -f "${LIBRARY_PY}" ]]; then
    # Create backup
    cp "${LIBRARY_PY}" "${LIBRARY_PY}.bak"

    # Replace the problematic line
    sed -i 's/if not hasattr(namespace, name):/if name not in vars(namespace):/' "${LIBRARY_PY}"

    # Verify the change
    if grep -q "if name not in vars(namespace):" "${LIBRARY_PY}"; then
        echo "PyTorch library.py patched successfully"
    else
        echo "Warning: Failed to patch library.py, restoring backup"
        mv "${LIBRARY_PY}.bak" "${LIBRARY_PY}"
    fi
else
    echo "Warning: torch/library.py not found at ${LIBRARY_PY}"
fi

##############################################################################
##
## Install NVIDIA Apex
##
##############################################################################

echo "=== Installing NVIDIA Apex ==="

Apex_TAG="25.09"

pushd "${TMP_DIR}"
    if [[ -d "apex" ]]; then
        rm -rf apex
    fi
    git clone https://github.com/NVIDIA/apex.git
    cd apex
    git checkout ${Apex_TAG}

    # Install Apex with parallel build and GB200 (Blackwell) support
    NVCC_APPEND_FLAGS="--threads 32" \
        APEX_PARALLEL_BUILD=32 \
        APEX_CPP_EXT=1 \
        APEX_CUDA_EXT=1 \
        APEX_ALL_CONTRIB_EXT=1 \
        TORCH_CUDA_ARCH_LIST="10.0" \
        uv pip install -v --no-build-isolation .
popd

echo "=== Apex installation complete ==="

##############################################################################
##
## Install DeepEP and nvshmem
##
##############################################################################

if [[ "$INSTALL_DEEPEP" == "True" ]]; then
    echo "=== Installing DeepEP ==="
    echo "Using DeepEP commit: ${DEEPEP_COMMIT}"

    # Clone and build DeepEP to ep-specific directory
    DEEPEP_DIR="${EP_TYPE}-ep"
    pushd "${TMP_DIR}"
        if [[ -d "${DEEPEP_DIR}" ]]; then
            rm -rf "${DEEPEP_DIR}"
        fi
        git clone https://github.com/deepseek-ai/DeepEP.git "${DEEPEP_DIR}"
        cd "${DEEPEP_DIR}"
        git fetch origin ${DEEPEP_COMMIT}
        git checkout FETCH_HEAD

        # Modify setup.py to add CCCL include path
        echo "Modifying setup.py for CCCL includes..."
        sed -i "s/include_dirs = \['csrc\/'\]/include_dirs = ['csrc\/', '\/usr\/local\/cuda\/include\/cccl\/']/g" setup.py

        # Install nvshmem if requested
        if [[ "$REINSTALL_NVSHMEM" == "True" ]]; then
            uv pip install nvidia-nvshmem-cu13==3.6.5
            # Create symlink for libnvshmem_host.so
            NVSHMEM_LIB_PATH=$(uv pip show nvidia-nvshmem-cu13 | grep "Location:" | cut -d' ' -f2)/nvidia/nvshmem/lib
            if [[ -f "${NVSHMEM_LIB_PATH}/libnvshmem_host.so.3" ]]; then
                ln -sf "${NVSHMEM_LIB_PATH}/libnvshmem_host.so.3" "${NVSHMEM_LIB_PATH}/libnvshmem_host.so"
            fi
        fi

        # Install DeepEP
        uv pip install --no-build-isolation -v .
    popd

    echo "=== DeepEP installation complete ==="
fi

echo "=== Installing Megatron-Bridge dependencies ==="

# Reinstall nvidia-cutlass-dsl (keep compatible with cuDNN Frontend)
uv pip uninstall nvidia-cutlass-dsl nvidia-cutlass-dsl-libs-base nvidia-cutlass-dsl-libs-cu13 || true
uv pip install "nvidia-cutlass-dsl[cu13]==4.5.0"

# Set build flags for flash-mla and other CUDA extensions
export CFLAGS="-I/usr/local/cuda/include/cccl -DNDEBUG"
export CXXFLAGS="-I/usr/local/cuda/include/cccl -DNDEBUG"
export MAMBA_FORCE_BUILD=TRUE
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
export FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE

echo "=== Running uv sync ==="
uv sync --only-group build --no-clean && \
    uv sync --link-mode copy --all-extras --all-groups --no-group diffusion --no-clean

# Create symlink for tilelang's libcudart_stub.so if tilelang is installed
if [[ -f "${VENV_PATH}/lib/python3.12/site-packages/tilelang/lib/libcudart_stub.so" ]]; then
    echo "=== Creating symlink for tilelang libcudart_stub.so ==="
    LIBCUDART_PATH=$(ldconfig -p 2>/dev/null | awk '/libcudart\.so\.[0-9]+ /{print $NF; exit}' || echo "")
    if [[ -n "$LIBCUDART_PATH" ]]; then
        ln -sf "$LIBCUDART_PATH" "${VENV_PATH}/lib/python3.12/site-packages/tilelang/lib/libcudart_stub.so"
    else
        echo "Warning: Could not find libcudart.so for tilelang symlink"
    fi
fi

echo "=== Installation complete ==="
echo ""
echo "To activate the environment, run:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "Virtual environment location: ${VENV_PATH}"
