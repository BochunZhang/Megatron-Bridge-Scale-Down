// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cuda/atomic>

#include "common.hpp"

namespace gb {

__device__ __forceinline__ uint64_t globaltimer_ns() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ void xor_into(uint4& acc, const uint4& v) {
  acc.x ^= v.x;
  acc.y ^= v.y;
  acc.z ^= v.z;
  acc.w ^= v.w;
}

// Grid-stride kernels over 16 B elements, unrolled 4x to keep more loads in flight.
__global__ void dev_read_kernel(const uint4* __restrict__ src, size_t n, uint4* sink) {
  const size_t stride = size_t(gridDim.x) * blockDim.x;
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  uint4 acc = make_uint4(0, 0, 0, 0);
  for (; i + 3 * stride < n; i += 4 * stride) {
    const uint4 a = src[i];
    const uint4 b = src[i + stride];
    const uint4 c = src[i + 2 * stride];
    const uint4 d = src[i + 3 * stride];
    xor_into(acc, a);
    xor_into(acc, b);
    xor_into(acc, c);
    xor_into(acc, d);
  }
  for (; i < n; i += stride) xor_into(acc, src[i]);
  if ((acc.x & acc.y & acc.z & acc.w) == 0xFFFFFFFFu) *sink = acc;  // keeps the loads alive
}

__global__ void dev_write_kernel(uint4* __restrict__ dst, size_t n, uint4 value) {
  const size_t stride = size_t(gridDim.x) * blockDim.x;
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  for (; i + 3 * stride < n; i += 4 * stride) {
    dst[i] = value;
    dst[i + stride] = value;
    dst[i + 2 * stride] = value;
    dst[i + 3 * stride] = value;
  }
  for (; i < n; i += stride) dst[i] = value;
}

__global__ void dev_copy_kernel(const uint4* __restrict__ src, uint4* __restrict__ dst, size_t n) {
  const size_t stride = size_t(gridDim.x) * blockDim.x;
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  for (; i + 3 * stride < n; i += 4 * stride) {
    const uint4 a = src[i];
    const uint4 b = src[i + stride];
    const uint4 c = src[i + 2 * stride];
    const uint4 d = src[i + 3 * stride];
    dst[i] = a;
    dst[i + stride] = b;
    dst[i + 2 * stride] = c;
    dst[i + 3 * stride] = d;
  }
  for (; i < n; i += stride) dst[i] = src[i];
}

__global__ void dev_fill_pattern_kernel(uint64_t* dst, size_t n_words, uint64_t seed) {
  const size_t stride = size_t(gridDim.x) * blockDim.x;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n_words; i += stride) {
    dst[i] = pattern_word(i, seed);
  }
}

__global__ void dev_count_mismatch_kernel(const uint64_t* src, size_t n_words, uint64_t seed,
                                          unsigned long long* count) {
  const size_t stride = size_t(gridDim.x) * blockDim.x;
  unsigned long long bad = 0;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n_words; i += stride) {
    bad += src[i] != pattern_word(i, seed);
  }
  if (bad) atomicAdd(count, bad);
}

// Single-thread dependent-load chain. Each slot holds the address of the next slot.
__global__ void dev_chase_kernel(const uint64_t* start, size_t steps, unsigned long long* elapsed_ns,
                                 unsigned long long* sink) {
  const uint64_t* p = start;
  const uint64_t t0 = globaltimer_ns();
  for (size_t i = 0; i < steps; ++i) p = reinterpret_cast<const uint64_t*>(*p);
  const uint64_t t1 = globaltimer_ns();
  *elapsed_ns = t1 - t0;
  *sink = reinterpret_cast<unsigned long long>(p);
}

// Flag ping-pong. Round i: ping stores 2i+1 and waits for 2i+2; pong waits for 2i+1 and stores 2i+2.
// The ping side records the time of the measured rounds; both sides give up after `timeout_ns`.
__global__ void dev_pingpong_kernel(uint32_t* flag, int is_ping, unsigned long long warmup, unsigned long long rounds,
                                    unsigned long long timeout_ns, unsigned long long* elapsed_ns, int* timed_out) {
  cuda::atomic_ref<uint32_t, cuda::thread_scope_system> f(*flag);
  const uint64_t deadline = globaltimer_ns() + timeout_ns;
  uint64_t t0 = globaltimer_ns();
  for (unsigned long long i = 0; i < warmup + rounds; ++i) {
    if (i == warmup) t0 = globaltimer_ns();
    const uint32_t base = static_cast<uint32_t>(2 * i);
    if (is_ping) {
      f.store(base + 1, cuda::std::memory_order_release);
      while (f.load(cuda::std::memory_order_acquire) != base + 2) {
        if (globaltimer_ns() > deadline) {
          *timed_out = 1;
          return;
        }
      }
    } else {
      while (f.load(cuda::std::memory_order_acquire) != base + 1) {
        if (globaltimer_ns() > deadline) {
          *timed_out = 1;
          return;
        }
      }
      f.store(base + 2, cuda::std::memory_order_release);
    }
  }
  if (is_ping) *elapsed_ns = globaltimer_ns() - t0;
}

}  // namespace gb
