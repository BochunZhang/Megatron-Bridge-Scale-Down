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

# DeepEP Installation Script for Hybrid EP mode
# Usage: ./install-hybrid-ep.sh

set -e

TMP_DIR="$(pwd)/.tmp"
mkdir -p "${TMP_DIR}"

export HYBRID_EP_MULTINODE=1
export RDMA_CORE_HOME='${TMP_DIR}/hybrid/build'


# DeepEP commit for hybrid EP mode
DEEPEP_COMMIT="34152ae28f80bcc3ee38d7a12cb2ad87cfd4ea72"
EP_TYPE="hybrid"

# Create temporary directory for builds

echo "=== Installing DeepEP (Hybrid EP mode) ==="
echo "Using DeepEP commit: ${DEEPEP_COMMIT}"


mkdir -p ${RDMA_CORE_HOME} && \
ln -sfn /usr/include ${RDMA_CORE_HOME}/include && \
ln -sfn /usr/lib/aarch64-linux-gnu ${RDMA_CORE_HOME}/lib && \

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

echo "=== DeepEP (Hybrid EP) installation complete ==="
