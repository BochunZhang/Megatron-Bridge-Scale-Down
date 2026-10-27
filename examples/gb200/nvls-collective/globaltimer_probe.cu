// Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Measure %globaltimer and clock64 on one GPU: the timer's tick, the SM clock, and how long the
// NCCL_SYM_AG_STMC_PAUSE_CYCLES spin (ncclSymkPauseCycles in nccl-2.32.3-pause.patch) really pauses.
//
//   tick   one thread reads %globaltimer back to back and records the first kTicks increments.
//          PTX only promises a 64-bit ns counter; how often it advances is up to the GPU and the
//          driver (and may change while a profiler is attached), so run_globaltimer_probe.sh runs
//          this both plainly and under nsys.
//          A tick T is why the patch spins on clock64: a %globaltimer spin of p ns starts somewhere
//          inside a tick and ends on the first edge >= p, so it lasts ceil(p / T) * T - T .. that,
//          and every p below T looks the same.
//   clock  one block per SM spins --window-ms of %globaltimer and divides its clock64 delta by it:
//          the SM clock each SM ran at, which is what turns clock64 cycles into ns.
//   pause  one thread runs the patch's spin --reps times per --pause-cycles value and times each with
//          clock64; expect target plus the few cycles of the last clock64 read. In ns it is cycles
//          divided by the clock measured over the whole pause kernel.
//
// Build and run: run_globaltimer_probe.sh (nvcc -arch=sm_100).

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#define CHECK(call)                                                                                  \
  do {                                                                                               \
    cudaError_t err_ = (call);                                                                       \
    if (err_ != cudaSuccess) {                                                                       \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(err_));        \
      exit(1);                                                                                       \
    }                                                                                                \
  } while (0)

constexpr int kTicks = 4096;

__device__ __forceinline__ uint64_t globaltimer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)::"memory");
  return t;
}

// The spin of ncclSymkPauseCycles in install/patches/nccl-2.32.3-pause.patch.
__device__ __forceinline__ void pauseCycles(uint32_t cycles) {
  long long start = clock64();
  while (clock64() - start < (long long)cycles) {
  }
}

// The first kTicks increments of %globaltimer after an edge, and how many reads each took.
__global__ void tickKernel(uint64_t* deltas, uint32_t* reads) {
  uint64_t prev = globaltimer(), t;
  do {
    t = globaltimer();
  } while (t == prev);
  prev = t;
  for (int i = 0; i < kTicks; i++) {
    uint32_t n = 0;
    do {
      t = globaltimer();
      n++;
    } while (t == prev);
    deltas[i] = t - prev;
    reads[i] = n;
    prev = t;
  }
}

// One block per SM: clock64 cycles over windowNs of %globaltimer.
__global__ void clockKernel(uint64_t windowNs, uint64_t* cycles, uint64_t* ns, uint32_t* smid) {
  if (threadIdx.x != 0) return;
  uint64_t g0 = globaltimer(), g;
  long long c0 = clock64();
  do {
    g = globaltimer();
  } while (g - g0 < windowNs);
  long long c1 = clock64();
  uint32_t id;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(id));
  cycles[blockIdx.x] = (uint64_t)(c1 - c0);
  ns[blockIdx.x] = g - g0;
  smid[blockIdx.x] = id;
}

// clock64 cycles of each pause; span[0..1] = total cycles and %globaltimer ns of the kernel.
__global__ void pauseKernel(const uint32_t* targets, int nTargets, int reps, uint64_t* cycles, uint64_t* span) {
  uint64_t g0 = globaltimer();
  long long k0 = clock64();
  for (int k = 0; k < nTargets; k++) {
    for (int r = 0; r < reps; r++) {
      long long c0 = clock64();
      pauseCycles(targets[k]);
      long long c1 = clock64();
      cycles[k * reps + r] = (uint64_t)(c1 - c0);
    }
  }
  span[0] = (uint64_t)(clock64() - k0);
  span[1] = globaltimer() - g0;
}

template <typename T>
static T quantile(std::vector<T> v, double q) {
  std::sort(v.begin(), v.end());
  return v[std::min(v.size() - 1, (size_t)(q * (v.size() - 1) + 0.5))];
}

static void usage(const char* argv0) {
  fprintf(stderr,
          "usage: %s [--device N] [--window-ms MS] [--reps N] [--pause-cycles LIST]\n"
          "  --device        CUDA device (default 0)\n"
          "  --window-ms     %%globaltimer window of the SM clock measurement (default 10)\n"
          "  --reps          pauses per --pause-cycles value (default 2000)\n"
          "  --pause-cycles  comma-separated pause targets in SM cycles (default 0,500,1000,2000,4000)\n",
          argv0);
}

int main(int argc, char** argv) {
  int device = 0, reps = 2000;
  double windowMs = 10.0;
  std::vector<uint32_t> targets = {0, 500, 1000, 2000, 4000};
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "-h" || a == "--help") {
      usage(argv[0]);
      return 0;
    }
    if (i + 1 >= argc) {
      usage(argv[0]);
      return 2;
    }
    const char* v = argv[++i];
    if (a == "--device") {
      device = atoi(v);
    } else if (a == "--window-ms") {
      windowMs = atof(v);
    } else if (a == "--reps") {
      reps = atoi(v);
    } else if (a == "--pause-cycles") {
      targets.clear();
      for (const char* p = v; *p;) {
        targets.push_back((uint32_t)strtoul(p, (char**)&p, 10));
        if (*p == ',') p++;
      }
    } else {
      usage(argv[0]);
      return 2;
    }
  }
  if (reps <= 0 || windowMs <= 0 || targets.empty()) {
    usage(argv[0]);
    return 2;
  }

  CHECK(cudaSetDevice(device));
  cudaDeviceProp prop;
  CHECK(cudaGetDeviceProperties(&prop, device));
  int nSm = prop.multiProcessorCount;
  printf("device %d: %s, %d SMs, sm_%d%d\n", device, prop.name, nSm, prop.major, prop.minor);

  // Warm up: the first launch pays for module loading and may run before the clocks ramp up.
  uint64_t *dDeltas, *dCycles, *dNs, *dSpan;
  uint32_t *dReads, *dSmid, *dTargets;
  CHECK(cudaMalloc(&dDeltas, kTicks * sizeof(uint64_t)));
  CHECK(cudaMalloc(&dReads, kTicks * sizeof(uint32_t)));
  CHECK(cudaMalloc(&dCycles, std::max<size_t>(nSm, targets.size() * reps) * sizeof(uint64_t)));
  CHECK(cudaMalloc(&dNs, nSm * sizeof(uint64_t)));
  CHECK(cudaMalloc(&dSmid, nSm * sizeof(uint32_t)));
  CHECK(cudaMalloc(&dTargets, targets.size() * sizeof(uint32_t)));
  CHECK(cudaMalloc(&dSpan, 2 * sizeof(uint64_t)));
  clockKernel<<<nSm, 32>>>((uint64_t)(windowMs * 1e6), dCycles, dNs, dSmid);
  CHECK(cudaDeviceSynchronize());

  // tick
  tickKernel<<<1, 1>>>(dDeltas, dReads);
  CHECK(cudaDeviceSynchronize());
  std::vector<uint64_t> deltas(kTicks);
  std::vector<uint32_t> reads(kTicks);
  CHECK(cudaMemcpy(deltas.data(), dDeltas, kTicks * sizeof(uint64_t), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(reads.data(), dReads, kTicks * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  std::map<uint64_t, int> hist;
  for (uint64_t d : deltas) hist[d]++;
  std::vector<std::pair<int, uint64_t>> top;
  for (auto& [d, n] : hist) top.push_back({n, d});
  std::sort(top.rbegin(), top.rend());
  printf("\n%%globaltimer tick over %d increments: min %llu ns, median %llu ns, p99 %llu ns, max %llu ns\n", kTicks,
         (unsigned long long)quantile(deltas, 0.0), (unsigned long long)quantile(deltas, 0.5),
         (unsigned long long)quantile(deltas, 0.99), (unsigned long long)quantile(deltas, 1.0));
  printf("  reads per tick: median %u, max %u\n", quantile(reads, 0.5), quantile(reads, 1.0));
  printf("  most common increments:");
  for (size_t i = 0; i < std::min<size_t>(top.size(), 6); i++)
    printf(" %llu ns x%d", (unsigned long long)top[i].second, top[i].first);
  printf("\n");

  // clock
  clockKernel<<<nSm, 32>>>((uint64_t)(windowMs * 1e6), dCycles, dNs, dSmid);
  CHECK(cudaDeviceSynchronize());
  std::vector<uint64_t> cycles(nSm), ns(nSm);
  std::vector<uint32_t> smid(nSm);
  CHECK(cudaMemcpy(cycles.data(), dCycles, nSm * sizeof(uint64_t), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(ns.data(), dNs, nSm * sizeof(uint64_t), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(smid.data(), dSmid, nSm * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  std::vector<double> ghz(nSm);
  for (int i = 0; i < nSm; i++) ghz[i] = (double)cycles[i] / (double)ns[i];
  std::vector<uint32_t> distinct = smid;
  std::sort(distinct.begin(), distinct.end());
  distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
  printf("\nSM clock (clock64 / %%globaltimer over %.1f ms, %d blocks on %zu SMs): min %.3f GHz, median %.3f GHz, "
         "max %.3f GHz\n",
         windowMs, nSm, distinct.size(), quantile(ghz, 0.0), quantile(ghz, 0.5), quantile(ghz, 1.0));

  // pause
  CHECK(cudaMemcpy(dTargets, targets.data(), targets.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
  pauseKernel<<<1, 1>>>(dTargets, (int)targets.size(), reps, dCycles, dSpan);
  CHECK(cudaDeviceSynchronize());
  std::vector<uint64_t> pc(targets.size() * reps);
  uint64_t span[2];
  CHECK(cudaMemcpy(pc.data(), dCycles, pc.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(span, dSpan, sizeof(span), cudaMemcpyDeviceToHost));
  double pauseGhz = (double)span[0] / (double)span[1];
  printf("\npause spin (%d reps each, clock64 at %.3f GHz measured over the %.2f ms pause kernel):\n", reps, pauseGhz,
         span[1] / 1e6);
  printf("  %13s %10s %10s %10s %10s %10s %10s\n", "target_cycles", "min_cyc", "median_cyc", "p99_cyc", "max_cyc",
         "target_ns", "median_ns");
  for (size_t k = 0; k < targets.size(); k++) {
    std::vector<uint64_t> c(pc.begin() + k * reps, pc.begin() + (k + 1) * reps);
    uint64_t median = quantile(c, 0.5);
    printf("  %13u %10llu %10llu %10llu %10llu %10.0f %10.0f\n", targets[k], (unsigned long long)quantile(c, 0.0),
           (unsigned long long)median, (unsigned long long)quantile(c, 0.99), (unsigned long long)quantile(c, 1.0),
           targets[k] / pauseGhz, median / pauseGhz);
  }

  CHECK(cudaFree(dDeltas));
  CHECK(cudaFree(dReads));
  CHECK(cudaFree(dCycles));
  CHECK(cudaFree(dNs));
  CHECK(cudaFree(dSmid));
  CHECK(cudaFree(dTargets));
  CHECK(cudaFree(dSpan));
  return 0;
}
