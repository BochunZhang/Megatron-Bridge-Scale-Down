#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Cross-compiles gb_benchmark for GB200 (aarch64 Grace + sm_100) on an x86_64 Docker host, using
# nvidia/cuda:13.0.0-devel-ubuntu22.04 plus NVIDIA's cross-linux-sbsa packages. The binary is written to
# build/gb_benchmark (aarch64). Set NATIVE_X86=1 to also build build-x86/gb_benchmark, an x86_64 binary for
# checking the host-side logic (topology, thread pool, host copies) on a machine without GB200.
set -euo pipefail

IMAGE="${IMAGE:-nvidia/cuda:13.0.0-devel-ubuntu22.04}"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker run --rm -v "$SRC_DIR:/work" -w /work -e NATIVE_X86="${NATIVE_X86:-0}" "$IMAGE" bash -euo pipefail -c '
export DEBIAN_FRONTEND=noninteractive
# Keep the amd64 sources amd64-only, then add the arm64 Ubuntu ports and the CUDA sbsa cross repo.
sed -i "s|^deb http|deb [arch=amd64] http|" /etc/apt/sources.list
dpkg --add-architecture arm64
cat > /etc/apt/sources.list.d/ports-arm64.list <<EOF
deb [arch=arm64] http://ports.ubuntu.com/ubuntu-ports jammy main universe
deb [arch=arm64] http://ports.ubuntu.com/ubuntu-ports jammy-updates main universe
EOF
echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/cross-linux-sbsa/ /" \
  > /etc/apt/sources.list.d/cuda-cross-sbsa.list
apt-get update -qq
# Cross packages pinned to the toolkit version of the image (13.0.0 = nvcc 13.0.48).
apt-get install -y -qq --no-install-recommends make g++-aarch64-linux-gnu libnuma-dev libnuma-dev:arm64 \
  cuda-cudart-cross-sbsa-13-0=13.0.48-1 cuda-crt-cross-sbsa-13-0=13.0.48-1 \
  cuda-cccl-cross-sbsa-13-0=13.0.50-1 cuda-nvcc-cross-sbsa-13-0=13.0.48-1 \
  cuda-driver-cross-sbsa-13-0=13.0.48-1 >/dev/null
make CROSS=1 BUILD_DIR=build
if [[ "$NATIVE_X86" == 1 ]]; then make CUDA_ARCH=90 BUILD_DIR=build-x86; fi
'
file "$SRC_DIR/build/gb_benchmark" 2>/dev/null || true
