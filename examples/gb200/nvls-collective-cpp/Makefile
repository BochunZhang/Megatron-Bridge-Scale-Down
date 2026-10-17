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

# make NCCL_HOME=<nccl-2.32.3>/build CUDA_HOME=/usr/local/cuda
#   build/nvls_collective_bench   CUDA + NCCL benchmark (sm_100, rpath to NCCL_HOME/lib)
#   build/collective_report       merges results.tsv files into compare.md (host only)
# make selftest                   CPU-only checks of the log parser and report helpers

CUDA_HOME ?= /usr/local/cuda
NCCL_HOME ?= $(abspath ../../../nccl/nccl-2.32.3/build)
NVCC ?= $(CUDA_HOME)/bin/nvcc
GENCODE ?= -gencode=arch=compute_100,code=sm_100
BUILD ?= $(CURDIR)/build

CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra
NVCCFLAGS ?= -O2 -std=c++17 $(GENCODE) -Xcompiler -Wall

HEADERS := bench_core.h

all: $(BUILD)/nvls_collective_bench $(BUILD)/collective_report

$(BUILD):
	mkdir -p $@

$(BUILD)/nvls_collective_bench: nvls_collective_bench.cu $(HEADERS) | $(BUILD)
	@test -f $(NCCL_HOME)/include/nccl.h || { echo "NCCL_HOME=$(NCCL_HOME) has no include/nccl.h; run install/install-nccl-2.32.3.sh" >&2; exit 1; }
	$(NVCC) $(NVCCFLAGS) -I$(NCCL_HOME)/include -o $@ $< \
		-L$(NCCL_HOME)/lib -lnccl -ldl -Xlinker -rpath -Xlinker $(NCCL_HOME)/lib

$(BUILD)/collective_report: collective_report.cc $(HEADERS) | $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(BUILD)/selftest: selftest.cc $(HEADERS) | $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ $<

selftest: $(BUILD)/selftest
	$(BUILD)/selftest

report: $(BUILD)/collective_report

clean:
	rm -rf $(BUILD)

.PHONY: all selftest report clean
