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

# DeepEP Installation Script for Deep EP mode
# Usage: ./install-deepep.sh

set -e

# DeepEP commit for deep EP mode
DEEPEP_COMMIT="018f21ea62c6c2888d9b46806aad6d90b399b23d"
EP_TYPE="deep"

# Create temporary directory for builds
TMP_DIR="$(pwd)/.tmp"
mkdir -p "${TMP_DIR}"

echo "=== Installing DeepEP (Deep EP mode) ==="
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
    sed -i "s/include_dirs = \['csrc\/ '\]/include_dirs = ['csrc\/ ', '\/usr\/local\/cuda\/include\/cccl\/']/g" setup.py

    # Install nvshmem
    echo "Installing nvshmem..."
    uv pip install nvidia-nvshmem-cu13==3.6.5
    # Create symlink for libnvshmem_host.so
    NVSHMEM_LIB_PATH=$(uv pip show nvidia-nvshmem-cu13 | grep "Location:" | cut -d' ' -f2)/nvidia/nvshmem/lib
    if [[ -f "${NVSHMEM_LIB_PATH}/libnvshmem_host.so.3" ]]; then
        ln -sf "${NVSHMEM_LIB_PATH}/libnvshmem_host.so.3" "${NVSHMEM_LIB_PATH}/libnvshmem_host.so"
    fi

    apt-get update
    apt-get install -y --no-install-recommends libnvidia-ml-dev
    TORCH_CUDA_ARCH_LIST="10.0" uv pip install --no-cache-dir --no-build-isolation -v . 
    
popd

echo "=== DeepEP (Deep EP) installation complete ==="
