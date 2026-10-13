# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Native build on a GB200 node (aarch64 + CUDA 13):   make
# Cross build on an x86_64 host with sbsa cross pkgs: make CROSS=1   (see build_cross_in_docker.sh)
# Host-logic build for x86_64 without GB200 asm:      make CUDA_ARCH=90 BUILD_DIR=build-x86 (for local checks only)

NVCC ?= nvcc
CUDA_ARCH ?= 100
BUILD_DIR ?= build
CROSS ?= 0

NVCC_FLAGS := -std=c++17 -O2 -lineinfo \
	-gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) \
	-gencode arch=compute_$(CUDA_ARCH),code=compute_$(CUDA_ARCH) \
	-Xcompiler -Wall,-Wextra,-pthread
LDLIBS := -lnuma

ifeq ($(CROSS),1)
NVCC_FLAGS += -ccbin aarch64-linux-gnu-g++ --target-directory sbsa-linux
LDLIBS := -L/usr/lib/aarch64-linux-gnu $(LDLIBS)
endif

SRCS := $(wildcard src/*.cuh src/*.hpp)

.PHONY: all probe
all: $(BUILD_DIR)/gb_benchmark $(BUILD_DIR)/hbm_probe
probe: $(BUILD_DIR)/hbm_probe

$(BUILD_DIR)/gb_benchmark: src/main.cu $(SRCS)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ src/main.cu $(LDLIBS)

# hbm_probe does not need libnuma (raw mbind/get_mempolicy syscalls, sysfs topology).
$(BUILD_DIR)/hbm_probe: src/hbm_probe.cu $(SRCS)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ src/hbm_probe.cu

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
