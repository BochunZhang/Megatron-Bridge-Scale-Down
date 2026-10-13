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

// hbm_probe: for each way of putting a buffer in GPU HBM, check whether CPU threads can load/store it in place
// (over C2C) or whether CPU access migrates the pages out of HBM. Residency is inferred from GPU read bandwidth
// before and after the CPU touches the buffer: HBM reads run at several TB/s, reads of pages that were moved to
// CPU memory (or migrated back on the GPU fault) are an order of magnitude slower on the first pass.
// Needs no libnuma: NUMA binding and placement queries use raw syscalls, topology comes from sysfs.

#include <cuda.h>
#include <linux/mempolicy.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>

#include "device_kernels.cuh"
#include "host_pool.hpp"

namespace {

using namespace gb;

int device_attr(cudaDeviceAttr attr, int device) {
  int value = -1;
  if (cudaDeviceGetAttribute(&value, attr, device) != cudaSuccess) {
    cudaGetLastError();
    return -1;
  }
  return value;
}

const char* kUsage = R"(usage: hbm_probe [--gpu G] [--bytes 1G] [--iters 5] [--methods numa,managed,managed_noadvise,cudamalloc,vmm]
                 [--out result.json]

For GPU G (default 2 if the node has >2 GPUs, else 0) and each allocation method:
  1. fill the buffer from the GPU and measure GPU read bandwidth (pages in HBM)
  2. probe one CPU load/store (SIGSEGV/SIGBUS are caught and reported)
  3. CPU read with all cores of the CPU attached to G, then GPU read again
  4. CPU read with all cores of the other CPU, CPU write from the attached CPU, GPU read again
Verdict IN PLACE = CPU numbers are C2C load/store into HBM; MIGRATES = CPU access moved the pages.

Methods:
  numa              mmap + mbind(MPOL_BIND) to the GPU's HBM NUMA node (needs HBM onlined as NUMA node)
  managed           cudaMallocManaged + PreferredLocation=GPU + AccessedBy=CPU (what gb_benchmark uses)
  managed_noadvise  cudaMallocManaged, prefetched to the GPU, no advice
  cudamalloc        cudaMalloc on the GPU, CPU dereferences the device pointer
  vmm               cuMemCreate on the GPU + cuMemSetAccess for HOST / HOST_NUMA
)";

std::string read_text(const std::string& path) {
  std::ifstream f(path);
  std::stringstream ss;
  ss << f.rdbuf();
  return trim(ss.str());
}

std::vector<int> parse_cpulist(const std::string& text) {
  std::vector<int> out;
  for (const std::string& range : split(text, ',')) {
    const size_t dash = range.find('-');
    const int lo = static_cast<int>(parse_int(range.substr(0, dash), "cpulist"));
    const int hi = dash == std::string::npos ? lo : static_cast<int>(parse_int(range.substr(dash + 1), "cpulist"));
    for (int c = lo; c <= hi; ++c) out.push_back(c);
  }
  return out;
}

std::vector<int> node_cpus(int node) {
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  sched_getaffinity(0, sizeof(allowed), &allowed);
  std::vector<int> out;
  for (int c : parse_cpulist(read_text("/sys/devices/system/node/node" + std::to_string(node) + "/cpulist"))) {
    if (c < CPU_SETSIZE && CPU_ISSET(c, &allowed)) out.push_back(c);
  }
  return out;
}

// cpu_touch (CPU access probe) is in host_pool.hpp.

// ---------------------------------------------------------------- driver API (no -lcuda needed)

struct Driver {
  decltype(&::cuGetErrorName) getErrorName = nullptr;
  decltype(&::cuMemGetAllocationGranularity) getGranularity = nullptr;
  decltype(&::cuMemCreate) create = nullptr;
  decltype(&::cuMemAddressReserve) reserve = nullptr;
  decltype(&::cuMemMap) map = nullptr;
  decltype(&::cuMemSetAccess) setAccess = nullptr;
  decltype(&::cuMemUnmap) unmap = nullptr;
  decltype(&::cuMemRelease) release = nullptr;
  decltype(&::cuMemAddressFree) addressFree = nullptr;

  template <typename F>
  static void load(const char* name, F& fn) {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult status;
    if (cudaGetDriverEntryPointByVersion(name, &p, 12000, cudaEnableDefault, &status) != cudaSuccess ||
        status != cudaDriverEntryPointSuccess || !p) {
      fail(std::string("cannot load driver symbol ") + name);
    }
    fn = reinterpret_cast<F>(p);
  }

  Driver() {
    load("cuGetErrorName", getErrorName);
    load("cuMemGetAllocationGranularity", getGranularity);
    load("cuMemCreate", create);
    load("cuMemAddressReserve", reserve);
    load("cuMemMap", map);
    load("cuMemSetAccess", setAccess);
    load("cuMemUnmap", unmap);
    load("cuMemRelease", release);
    load("cuMemAddressFree", addressFree);
  }

  std::string err(CUresult r) const {
    const char* s = nullptr;
    getErrorName(r, &s);
    return s ? s : std::to_string(static_cast<int>(r));
  }
};

// ---------------------------------------------------------------- context

struct Ctx {
  int gpu = 0;
  int sm_count = 0;
  int numa_config = -1;
  int hbm_numa = -1;
  int host_numa = -1;
  int other_numa = -1;
  int pageable_access = 0;
  int direct_managed = 0;
  int concurrent_managed = 0;
  size_t bytes = 0;
  int iters = 5;
  std::unique_ptr<HostThreadPool> attached;
  std::unique_ptr<HostThreadPool> other;
};

struct Alloc {
  std::string method;
  std::string api;
  std::string unavailable;
  std::vector<std::string> notes;
  uint8_t* ptr = nullptr;
  bool host_numa_checkable = false;
  std::function<void()> free_fn;
};

Alloc make_alloc(const std::string& method, const std::string& api) {
  Alloc a;
  a.method = method;
  a.api = api;
  return a;
}

Alloc setup_numa(const Ctx& c) {
  Alloc a = make_alloc("numa", "mmap + mbind(MPOL_BIND, MPOL_MF_STRICT) to the GPU HBM NUMA node");
  if (c.hbm_numa < 0 || c.hbm_numa >= 1024) {
    a.unavailable = "HBM is not an online NUMA node (cudaDevAttrNumaConfig=" + std::to_string(c.numa_config) +
                    ", cudaDevAttrNumaId=" + std::to_string(c.hbm_numa) + ")";
    return a;
  }
  void* m = mmap(nullptr, c.bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (m == MAP_FAILED) {
    a.unavailable = std::string("mmap failed: ") + std::strerror(errno);
    return a;
  }
  unsigned long mask[1024 / (8 * sizeof(unsigned long))] = {};
  mask[c.hbm_numa / (8 * sizeof(unsigned long))] |= 1ul << (c.hbm_numa % (8 * sizeof(unsigned long)));
  if (syscall(SYS_mbind, m, c.bytes, MPOL_BIND, mask, 1025ul, MPOL_MF_STRICT) != 0) {
    a.unavailable = std::string("mbind failed: ") + std::strerror(errno) + " (container seccomp/cpuset.mems?)";
    munmap(m, c.bytes);
    return a;
  }
  a.ptr = static_cast<uint8_t*>(m);
  a.host_numa_checkable = true;
  const size_t bytes = c.bytes;
  a.free_fn = [m, bytes] { munmap(m, bytes); };
  return a;
}

Alloc setup_managed(const Ctx& c, bool advise) {
  Alloc a = make_alloc(advise ? "managed" : "managed_noadvise",
                       advise ? "cudaMallocManaged + SetPreferredLocation(GPU) + SetAccessedBy(CPU)"
                              : "cudaMallocManaged, prefetched to the GPU");
  void* p = nullptr;
  cudaError_t e = cudaMallocManaged(&p, c.bytes, cudaMemAttachGlobal);
  if (e != cudaSuccess) {
    a.unavailable = std::string("cudaMallocManaged: ") + cudaGetErrorName(e);
    cudaGetLastError();
    return a;
  }
  if (advise) {
    cudaMemLocation dev{};
    dev.type = cudaMemLocationTypeDevice;
    dev.id = c.gpu;
    cudaMemLocation host{};
    host.type = cudaMemLocationTypeHost;
    host.id = 0;
    e = cudaMemAdvise(p, c.bytes, cudaMemAdviseSetPreferredLocation, dev);
    a.notes.push_back(std::string("SetPreferredLocation(GPU): ") + cudaGetErrorName(e));
    e = cudaMemAdvise(p, c.bytes, cudaMemAdviseSetAccessedBy, host);
    a.notes.push_back(std::string("SetAccessedBy(CPU): ") + cudaGetErrorName(e));
    cudaGetLastError();
  }
  a.ptr = static_cast<uint8_t*>(p);
  a.host_numa_checkable = true;
  a.free_fn = [p] { cudaFree(p); };
  return a;
}

Alloc setup_cudamalloc(const Ctx& c) {
  Alloc a = make_alloc("cudamalloc", "cudaMalloc");
  void* p = nullptr;
  const cudaError_t e = cudaMalloc(&p, c.bytes);
  if (e != cudaSuccess) {
    a.unavailable = std::string("cudaMalloc: ") + cudaGetErrorName(e);
    cudaGetLastError();
    return a;
  }
  a.ptr = static_cast<uint8_t*>(p);
  a.free_fn = [p] { cudaFree(p); };
  return a;
}

Alloc setup_vmm(const Ctx& c, const Driver& d) {
  Alloc a = make_alloc("vmm", "cuMemCreate(location=DEVICE) + cuMemMap + cuMemSetAccess(DEVICE, then HOST / HOST_NUMA)");
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = c.gpu;
  size_t gran = 0;
  CUresult r = d.getGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  if (r != CUDA_SUCCESS) {
    a.unavailable = "cuMemGetAllocationGranularity: " + d.err(r);
    return a;
  }
  const size_t size = (c.bytes + gran - 1) / gran * gran;
  CUmemGenericAllocationHandle handle = 0;
  if ((r = d.create(&handle, size, &prop, 0)) != CUDA_SUCCESS) {
    a.unavailable = "cuMemCreate: " + d.err(r);
    return a;
  }
  CUdeviceptr va = 0;
  if ((r = d.reserve(&va, size, gran, 0, 0)) != CUDA_SUCCESS || (r = d.map(va, size, 0, handle, 0)) != CUDA_SUCCESS) {
    a.unavailable = "cuMemAddressReserve/cuMemMap: " + d.err(r);
    d.release(handle);
    return a;
  }
  CUmemAccessDesc desc = {};
  desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  desc.location.id = c.gpu;
  desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  if ((r = d.setAccess(va, size, &desc, 1)) != CUDA_SUCCESS) {
    a.unavailable = "cuMemSetAccess(DEVICE): " + d.err(r);
    return a;
  }
  desc.location.type = CU_MEM_LOCATION_TYPE_HOST;
  desc.location.id = 0;
  a.notes.push_back("cuMemSetAccess(HOST): " + d.err(d.setAccess(va, size, &desc, 1)));
  desc.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  desc.location.id = std::max(c.host_numa, 0);
  a.notes.push_back("cuMemSetAccess(HOST_NUMA " + std::to_string(desc.location.id) +
                    "): " + d.err(d.setAccess(va, size, &desc, 1)));
  a.ptr = reinterpret_cast<uint8_t*>(va);
  a.free_fn = [&d, va, size, handle] {
    d.unmap(va, size);
    d.addressFree(va, size);
    d.release(handle);
  };
  return a;
}

// ---------------------------------------------------------------- measurements

std::vector<double> gpu_read(const Ctx& c, const uint8_t* p, uint4* sink) {
  GB_CHECK_CUDA(cudaSetDevice(c.gpu));
  cudaEvent_t start, stop;
  GB_CHECK_CUDA(cudaEventCreate(&start));
  GB_CHECK_CUDA(cudaEventCreate(&stop));
  std::vector<double> gbps;
  for (int i = 0; i < c.iters; ++i) {
    GB_CHECK_CUDA(cudaEventRecord(start));
    dev_read_kernel<<<c.sm_count * 4, 512>>>(reinterpret_cast<const uint4*>(p), c.bytes / sizeof(uint4), sink);
    GB_CHECK_CUDA(cudaGetLastError());
    GB_CHECK_CUDA(cudaEventRecord(stop));
    GB_CHECK_CUDA(cudaEventSynchronize(stop));
    float ms = 0;
    GB_CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    gbps.push_back(static_cast<double>(c.bytes) / (ms * 1e6));
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return gbps;
}

std::vector<double> cpu_op(const Ctx& c, HostThreadPool& pool, HostOp op, uint8_t* p) {
  std::vector<double> gbps;
  for (int i = 0; i < c.iters; ++i) {
    const HostTiming t = timed_host_op(pool, pool.size(), op, HostKernel::LdpStp, p, p, c.bytes);
    gbps.push_back(static_cast<double>(c.bytes) / (t.ms * 1e6));
  }
  return gbps;
}

size_t gpu_mismatches(const Ctx& c, const uint8_t* p, uint64_t seed) {
  GB_CHECK_CUDA(cudaSetDevice(c.gpu));
  unsigned long long* count = nullptr;
  GB_CHECK_CUDA(cudaHostAlloc(&count, sizeof(*count), cudaHostAllocPortable | cudaHostAllocMapped));
  *count = 0;
  dev_count_mismatch_kernel<<<c.sm_count * 4, 512>>>(reinterpret_cast<const uint64_t*>(p), c.bytes / 8, seed, count);
  GB_CHECK_CUDA(cudaGetLastError());
  GB_CHECK_CUDA(cudaDeviceSynchronize());
  const size_t bad = *count;
  cudaFreeHost(count);
  return bad;
}

// NUMA node of up to 256 sampled pages, via get_mempolicy(MPOL_F_NODE | MPOL_F_ADDR). Negative key = -errno.
std::map<int, size_t> page_nodes(const uint8_t* p, size_t bytes) {
  const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  const size_t pages = bytes / page;
  const size_t n = std::min<size_t>(pages, 256);
  std::map<int, size_t> hist;
  for (size_t i = 0; i < n; ++i) {
    int node = -1;
    const void* addr = p + (pages * i / n) * page;
    if (syscall(SYS_get_mempolicy, &node, nullptr, 0ul, addr, MPOL_F_NODE | MPOL_F_ADDR) != 0) node = -errno;
    ++hist[node];
  }
  return hist;
}

double median_of(const std::vector<double>& v) { return v.empty() ? NAN : summarize(v).median; }

std::string fmt(double v) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.1f", v);
  return buf;
}

constexpr double kHbmFloorGbps = 1500;  // well above C2C/LPDDR, well below B200 HBM read bandwidth

std::string run_method(Ctx& c, Alloc a, const uint64_t seed) {
  JsonObject j;
  j.str("method", a.method).str("api", a.api);
  std::vector<std::string> notes;
  for (const std::string& n : a.notes) notes.push_back(json_quote(n));
  std::printf("\n=== %s: %s\n", a.method.c_str(), a.api.c_str());
  for (const std::string& n : a.notes) std::printf("    %s\n", n.c_str());
  if (!a.unavailable.empty()) {
    std::printf("    unavailable: %s\n", a.unavailable.c_str());
    return j.raw("notes", json_array(notes)).str("verdict", "UNAVAILABLE: " + a.unavailable).dump();
  }

  uint4* sink = nullptr;
  GB_CHECK_CUDA(cudaSetDevice(c.gpu));
  GB_CHECK_CUDA(cudaMalloc(&sink, sizeof(uint4)));
  std::string verdict;

  // 1. GPU writes the pattern (first touch from the GPU), then GPU read bandwidth with pages in HBM.
  dev_fill_pattern_kernel<<<c.sm_count * 4, 512>>>(reinterpret_cast<uint64_t*>(a.ptr), c.bytes / 8, seed);
  cudaError_t fill_err = cudaGetLastError();
  if (fill_err == cudaSuccess) fill_err = cudaDeviceSynchronize();
  if (fill_err != cudaSuccess) {
    verdict = std::string("UNAVAILABLE: GPU cannot write this memory (") + cudaGetErrorName(fill_err) + ")";
    std::printf("    %s\n", verdict.c_str());
    cudaFree(sink);
    if (a.free_fn) a.free_fn();
    return j.raw("notes", json_array(notes)).str("verdict", verdict).dump();
  }
  if (starts_with(a.method, "managed")) {
    cudaMemLocation dev{};
    dev.type = cudaMemLocationTypeDevice;
    dev.id = c.gpu;
    GB_CHECK_CUDA(cudaMemPrefetchAsync(a.ptr, c.bytes, dev, 0, 0));
    GB_CHECK_CUDA(cudaDeviceSynchronize());
  }
  const auto gpu_before = gpu_read(c, a.ptr, sink);
  std::printf("    gpu%d read, before CPU access:   %s GB/s (median; per iter %s)\n", c.gpu,
              fmt(median_of(gpu_before)).c_str(), json_numbers(gpu_before).c_str());
  j.raw("gpu_read_before_gbps", json_numbers(gpu_before));
  if (a.method == "numa") {  // get_mempolicy on managed memory would itself be a CPU access
    std::vector<std::string> h;
    for (const auto& kv : page_nodes(a.ptr, c.bytes)) {
      h.push_back(JsonObject().integer("node", kv.first).integer("pages", static_cast<long long>(kv.second)).dump());
    }
    j.raw("page_nodes_before", json_array(h));
  }

  // 2. One CPU load/store.
  uint64_t loaded = 0;
  const std::string fault = cpu_touch(reinterpret_cast<uint64_t*>(a.ptr), &loaded);
  j.str("cpu_access", fault.empty() ? "ok" : fault);
  if (!fault.empty()) {
    verdict = "CPU CANNOT ACCESS: dereferencing the pointer raised " + fault;
    std::printf("    cpu load/store: %s\n    verdict: %s\n", fault.c_str(), verdict.c_str());
  } else {
    std::printf("    cpu load/store: ok (word 0 = %s)\n",
                loaded == pattern_word(0, seed) ? "expected pattern" : "UNEXPECTED VALUE");
    j.boolean("cpu_word0_matches", loaded == pattern_word(0, seed));

    // 3. Attached CPU reads, GPU reads again.
    const auto cpu_a = cpu_op(c, *c.attached, HostOp::Read, a.ptr);
    const auto gpu_after_a = gpu_read(c, a.ptr, sink);
    const size_t bad = gpu_mismatches(c, a.ptr, seed);
    // 4. Other CPU reads, attached CPU writes, GPU reads again.
    std::vector<double> cpu_b;
    if (c.other) cpu_b = cpu_op(c, *c.other, HostOp::Read, a.ptr);
    const auto cpu_w = cpu_op(c, *c.attached, HostOp::Write, a.ptr);
    const auto gpu_after_w = gpu_read(c, a.ptr, sink);

    std::printf("    cpu read  by numa%d (attached): %s GB/s (1st pass %s)\n", c.host_numa,
                fmt(median_of(cpu_a)).c_str(), fmt(cpu_a.front()).c_str());
    std::printf("    gpu%d read, after CPU read:     %s GB/s (1st pass %s), data mismatches %zu\n", c.gpu,
                fmt(median_of(gpu_after_a)).c_str(), fmt(gpu_after_a.front()).c_str(), bad);
    if (c.other) {
      std::printf("    cpu read  by numa%d (other):    %s GB/s (1st pass %s)\n", c.other_numa,
                  fmt(median_of(cpu_b)).c_str(), fmt(cpu_b.front()).c_str());
    }
    std::printf("    cpu write by numa%d (attached): %s GB/s (1st pass %s)\n", c.host_numa,
                fmt(median_of(cpu_w)).c_str(), fmt(cpu_w.front()).c_str());
    std::printf("    gpu%d read, after CPU write:    %s GB/s (1st pass %s)\n", c.gpu,
                fmt(median_of(gpu_after_w)).c_str(), fmt(gpu_after_w.front()).c_str());

    const double before = median_of(gpu_before);
    const bool gpu_dropped = gpu_after_a.front() < 0.5 * before || gpu_after_w.front() < 0.5 * before;
    const bool cpu_first_slow = cpu_a.front() < 0.5 * median_of(cpu_a);
    if (before < kHbmFloorGbps) {
      verdict = "NOT IN HBM: GPU read before any CPU access is only " + fmt(before) + " GB/s";
    } else if (gpu_dropped || cpu_first_slow) {
      verdict = "MIGRATES: CPU access moves pages out of HBM (GPU 1st-pass read after CPU access " +
                fmt(std::min(gpu_after_a.front(), gpu_after_w.front())) + " GB/s vs " + fmt(before) +
                " before; CPU 1st pass " + fmt(cpu_a.front()) + " vs " + fmt(median_of(cpu_a)) + " GB/s)";
    } else {
      verdict = "IN PLACE: pages stay in HBM, CPU numbers are C2C load/store bandwidth";
    }
    std::printf("    verdict: %s\n", verdict.c_str());
    j.raw("cpu_read_attached_gbps", json_numbers(cpu_a))
        .raw("gpu_read_after_cpu_read_gbps", json_numbers(gpu_after_a))
        .integer("mismatched_words_after_cpu_read", static_cast<long long>(bad))
        .raw("cpu_read_other_gbps", json_numbers(cpu_b))
        .raw("cpu_write_attached_gbps", json_numbers(cpu_w))
        .raw("gpu_read_after_cpu_write_gbps", json_numbers(gpu_after_w));
    if (a.host_numa_checkable) {
      std::vector<std::string> h;
      for (const auto& kv : page_nodes(a.ptr, c.bytes)) {
        h.push_back(JsonObject().integer("node", kv.first).integer("pages", static_cast<long long>(kv.second)).dump());
      }
      j.raw("page_nodes_after", json_array(h));
    }
  }
  GB_CHECK_CUDA(cudaSetDevice(c.gpu));
  cudaFree(sink);
  if (a.free_fn) a.free_fn();
  GB_CHECK_CUDA(cudaDeviceSynchronize());
  return j.raw("notes", json_array(notes)).str("verdict", verdict).dump();
}

}  // namespace

int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "-h" || std::string(argv[i]) == "--help") {
      std::printf("%s", kUsage);
      return 0;
    }
  }
  try {
    Args args(argc, argv, 1);
    int n_gpus = 0;
    const cudaError_t e = cudaGetDeviceCount(&n_gpus);
    if (e != cudaSuccess || n_gpus == 0) fail(std::string("no CUDA device: ") + cudaGetErrorString(e));

    Ctx c;
    c.gpu = args.get_int("gpu", n_gpus > 2 ? 2 : 0);
    if (c.gpu < 0 || c.gpu >= n_gpus) fail("--gpu out of range");
    c.bytes = parse_size(args.get("bytes", "1G"));
    if (c.bytes % 4096 != 0) fail("--bytes must be a multiple of 4096");
    c.iters = args.get_int("iters", 5);
    if (c.iters < 2) fail("--iters must be >= 2");
    const std::vector<std::string> methods = split(args.get("methods", "numa,managed,managed_noadvise,cudamalloc,vmm"), ',');
    const std::string out = args.get("out", "");
    args.finish();

    GB_CHECK_CUDA(cudaSetDevice(c.gpu));
    GB_CHECK_CUDA(cudaFree(nullptr));  // create the primary context for the driver-API calls
    c.sm_count = device_attr(cudaDevAttrMultiProcessorCount, c.gpu);
    c.numa_config = device_attr(cudaDevAttrNumaConfig, c.gpu);
    c.hbm_numa = c.numa_config == cudaDeviceNumaConfigNumaNode ? device_attr(cudaDevAttrNumaId, c.gpu) : -1;
    c.host_numa = device_attr(cudaDevAttrHostNumaId, c.gpu);
    c.pageable_access = device_attr(cudaDevAttrPageableMemoryAccess, c.gpu);
    c.direct_managed = device_attr(cudaDevAttrDirectManagedMemAccessFromHost, c.gpu);
    c.concurrent_managed = device_attr(cudaDevAttrConcurrentManagedAccess, c.gpu);

    std::vector<int> cpu_numas;
    for (int n : parse_cpulist(read_text("/sys/devices/system/node/has_cpu"))) {
      if (!node_cpus(n).empty()) cpu_numas.push_back(n);
    }
    if (cpu_numas.empty()) fail("no CPU NUMA node with allowed CPUs");
    if (std::find(cpu_numas.begin(), cpu_numas.end(), c.host_numa) == cpu_numas.end()) {
      std::printf("[warn] cudaDevAttrHostNumaId=%d is not a CPU node; using numa%d as the attached CPU\n", c.host_numa,
                  cpu_numas.front());
      c.host_numa = cpu_numas.front();
    }
    for (int n : cpu_numas) {
      if (n != c.host_numa) {
        c.other_numa = n;
        break;
      }
    }
    c.attached = std::make_unique<HostThreadPool>(node_cpus(c.host_numa));
    if (c.other_numa >= 0) c.other = std::make_unique<HostThreadPool>(node_cpus(c.other_numa));

    char name[256] = {0};
    cudaDeviceProp prop;
    GB_CHECK_CUDA(cudaGetDeviceProperties(&prop, c.gpu));
    std::snprintf(name, sizeof(name), "%s", prop.name);
    int driver = 0;
    cudaDriverGetVersion(&driver);
    std::printf("[probe] gpu%d %s, driver CUDA %d, %s per buffer, %d iters\n", c.gpu, name, driver,
                format_bytes(c.bytes).c_str(), c.iters);
    std::printf("[probe] numa_config=%d (1 = HBM is a NUMA node) hbm_numa=%d host_numa=%d other_cpu_numa=%d\n",
                c.numa_config, c.hbm_numa, c.host_numa, c.other_numa);
    std::printf("[probe] pageableMemoryAccess=%d directManagedMemAccessFromHost=%d concurrentManagedAccess=%d\n",
                c.pageable_access, c.direct_managed, c.concurrent_managed);
    std::printf("[probe] CPU threads: numa%d x%zu, numa%d x%zu\n", c.host_numa, c.attached->size(), c.other_numa,
                c.other ? c.other->size() : size_t(0));

    std::unique_ptr<Driver> driver_api;
    std::vector<std::string> results;
    uint64_t seed = 0x1234567890ABCDEFull;
    for (const std::string& m : methods) {
      Alloc a;
      if (m == "numa") {
        a = setup_numa(c);
      } else if (m == "managed") {
        a = setup_managed(c, true);
      } else if (m == "managed_noadvise") {
        a = setup_managed(c, false);
      } else if (m == "cudamalloc") {
        a = setup_cudamalloc(c);
      } else if (m == "vmm") {
        if (!driver_api) driver_api = std::make_unique<Driver>();
        a = setup_vmm(c, *driver_api);
      } else {
        fail("unknown method '" + m + "'");
      }
      results.push_back(run_method(c, std::move(a), seed++));
      std::fflush(stdout);
    }

    if (!out.empty()) {
      const std::filesystem::path parent = std::filesystem::path(out).parent_path();
      if (!parent.empty()) std::filesystem::create_directories(parent);
      std::ofstream f(out);
      if (!f) fail("cannot write " + out);
      f << JsonObject()
               .str("tool", "hbm_probe")
               .str("timestamp", timestamp_iso())
               .str("gpu_name", name)
               .integer("gpu", c.gpu)
               .integer("cuda_driver_version", driver)
               .integer("numa_config", c.numa_config)
               .integer("hbm_numa", c.hbm_numa)
               .integer("host_numa", c.host_numa)
               .integer("other_cpu_numa", c.other_numa)
               .integer("pageable_memory_access", c.pageable_access)
               .integer("direct_managed_mem_access_from_host", c.direct_managed)
               .integer("concurrent_managed_access", c.concurrent_managed)
               .integer("bytes", static_cast<long long>(c.bytes))
               .integer("iters", c.iters)
               .raw("results", json_array(results))
               .dump()
        << "\n";
      std::printf("\n[out] wrote %s\n", out.c_str());
    }
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "[error] %s\n", ex.what());
    return 1;
  }
  return 0;
}
