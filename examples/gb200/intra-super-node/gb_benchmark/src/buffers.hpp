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

#include <numa.h>
#include <numaif.h>
#include <sys/mman.h>

#include <memory>

#include "device_kernels.cuh"
#include "host_pool.hpp"
#include "topology.hpp"

namespace gb {

// Memory targets:
//   cpu<i>    LPDDR of CPU NUMA node i (numa_alloc_onnode, pageable)
//   pinned<i> same as cpu<i>, additionally registered with cudaHostRegister
//   hbm<g>    HBM of GPU g mapped as system memory: numa_alloc_onnode on the GPU's NUMA node
//             (or cudaMallocManaged with preferred location gpu g, see --hbm-backend)
//   cuda<g>   cudaMalloc on GPU g (not CPU accessible)
enum class TargetKind { Cpu, Pinned, Hbm, Cuda };

struct Target {
  TargetKind kind;
  int index;
  std::string name;
};

enum class HbmBackend { Auto, Numa, Managed };

inline HbmBackend parse_hbm_backend(const std::string& text) {
  if (text == "auto") return HbmBackend::Auto;
  if (text == "numa") return HbmBackend::Numa;
  if (text == "managed") return HbmBackend::Managed;
  fail("--hbm-backend must be auto, numa or managed, got '" + text + "'");
}

inline Target parse_target(const std::string& text, const Topology& topo) {
  static const std::pair<const char*, TargetKind> kinds[] = {
      {"pinned", TargetKind::Pinned}, {"cpu", TargetKind::Cpu}, {"hbm", TargetKind::Hbm}, {"cuda", TargetKind::Cuda}};
  for (const auto& kv : kinds) {
    const std::string prefix = kv.first;
    if (!starts_with(text, prefix)) continue;
    const std::string digits = text.substr(prefix.size());
    if (digits.empty() || digits.find_first_not_of("0123456789") != std::string::npos) break;
    Target t{kv.second, static_cast<int>(parse_int(digits, "target index")), text};
    if (t.kind == TargetKind::Cpu || t.kind == TargetKind::Pinned) {
      topo.cpu(t.index);
    } else {
      topo.gpu(t.index);
    }
    return t;
  }
  fail("unknown memory target '" + text + "' (expected cpu<i>, pinned<i>, hbm<g> or cuda<g>)");
}

// Comma list; "hbm*" expands to hbm0..hbmN-1 (likewise cpu*, pinned*, cuda*).
inline std::vector<Target> expand_targets(const std::string& spec, const Topology& topo) {
  std::vector<Target> out;
  for (const std::string& item : split(spec, ',')) {
    if (item.back() == '*') {
      const std::string prefix = item.substr(0, item.size() - 1);
      const size_t n = (prefix == "cpu" || prefix == "pinned") ? topo.cpus.size() : topo.gpus.size();
      for (size_t i = 0; i < n; ++i) out.push_back(parse_target(prefix + std::to_string(i), topo));
    } else {
      out.push_back(parse_target(item, topo));
    }
  }
  if (out.empty()) fail("empty memory target list '" + spec + "'");
  return out;
}

inline bool hbm_uses_managed(const Topology& topo, int gpu, HbmBackend backend) {
  if (backend == HbmBackend::Managed) return true;
  if (backend == HbmBackend::Numa) return false;
  return topo.gpu(gpu).numa_id < 0;
}

inline bool host_accessible(const Target& t) { return t.kind != TargetKind::Cuda; }

// Empty string if CPU threads can use this target, otherwise the reason.
inline std::string host_access_problem(const Target& t) {
  return host_accessible(t) ? "" : t.name + " is cudaMalloc memory, which CPU threads cannot load/store";
}

// Empty string if GPU `gpu` can load/store this target from a kernel, otherwise the reason.
inline std::string device_access_problem(const Topology& topo, int gpu, const Target& t, HbmBackend backend) {
  const GpuInfo& g = topo.gpu(gpu);
  switch (t.kind) {
    case TargetKind::Pinned:
      return "";
    case TargetKind::Cuda:
      if (t.index == gpu || topo.can_access_peer[gpu][t.index]) return "";
      return "gpu" + std::to_string(gpu) + " has no peer access to gpu" + std::to_string(t.index);
    case TargetKind::Hbm:
      if (hbm_uses_managed(topo, t.index, backend)) return "";
      [[fallthrough]];
    case TargetKind::Cpu:
      if (g.pageable_access == 1) return "";
      return "gpu" + std::to_string(gpu) + " cannot access pageable system memory (" + t.name + ")";
  }
  return "";
}

struct Placement {
  std::string status = "not checked";
  int expected = -1;
  size_t sampled = 0;
  size_t on_expected = 0;
  std::map<int, size_t> histogram;  // node -> sampled pages

  std::string to_json() const {
    std::vector<std::string> hist;
    for (const auto& kv : histogram) {
      hist.push_back(JsonObject().integer("node", kv.first).integer("pages", static_cast<long long>(kv.second)).dump());
    }
    return JsonObject()
        .str("status", status)
        .integer("expected_node", expected)
        .integer("sampled_pages", static_cast<long long>(sampled))
        .integer("pages_on_expected_node", static_cast<long long>(on_expected))
        .raw("histogram", json_array(hist))
        .dump();
  }
};

class Buffer {
 public:
  Buffer(const Topology& topo, const Target& target, size_t bytes, HbmBackend backend)
      : target_(target), bytes_(bytes), page_size_(topo.page_size), numa_available_(topo.numa_available) {
    switch (target.kind) {
      case TargetKind::Cpu:
      case TargetKind::Pinned:
        numa_node_ = topo.cpu(target.index).numa;
        alloc_numa();
        backend_ = "numa_alloc_onnode";
        if (target.kind == TargetKind::Pinned) {
          GB_CHECK_CUDA(cudaHostRegister(data_, bytes_, cudaHostRegisterPortable | cudaHostRegisterMapped));
          registered_ = true;
          backend_ += "+cudaHostRegister";
        }
        break;
      case TargetKind::Hbm: {
        const GpuInfo& g = topo.gpu(target.index);
        device_ = g.ordinal;
        if (hbm_uses_managed(topo, g.ordinal, backend)) {
          alloc_managed(g);
        } else {
          if (g.numa_id < 0) {
            fail("gpu" + std::to_string(g.ordinal) +
                 " HBM is not exposed as a NUMA node (driver in CDMM mode?). Use --hbm-backend managed or "
                 "--hbm-numa " + std::to_string(g.ordinal) + "=<node>");
          }
          numa_node_ = g.numa_id;
          alloc_numa();
          backend_ = "numa_alloc_onnode";
        }
        break;
      }
      case TargetKind::Cuda:
        device_ = target.index;
        GB_CHECK_CUDA(cudaSetDevice(device_));
        GB_CHECK_CUDA(cudaMalloc(&data_, bytes_));
        backend_ = "cudaMalloc";
        break;
    }
  }

  ~Buffer() {
    if (!data_) return;
    if (registered_) cudaHostUnregister(data_);
    if (backend_ == "cudaMalloc" || backend_ == "cudaMallocManaged") {
      cudaFree(data_);
    } else if (numa_alloc_) {
      numa_free(data_, bytes_);
    } else {
      munmap(data_, bytes_);
    }
  }

  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  uint8_t* data() const { return static_cast<uint8_t*>(data_); }
  size_t bytes() const { return bytes_; }
  const Target& target() const { return target_; }
  bool host_ok() const { return host_accessible(target_); }
  int device() const { return device_; }
  int numa_node() const { return numa_node_; }
  const std::string& backend() const { return backend_; }

  // Samples up to 1024 pages with get_mempolicy(MPOL_F_NODE | MPOL_F_ADDR) to check which NUMA node backs them
  // (move_pages is often blocked by container seccomp profiles). Call after first touch.
  void check_placement() {
    placement_ = Placement();
    placement_.expected = numa_node_;
    if (numa_node_ < 0) {
      placement_.status = backend_ == "cudaMallocManaged" ? "managed (not checked)" : "device memory";
      return;
    }
    if (!numa_available_) {
      placement_.status = "libnuma unavailable";
      return;
    }
    const size_t pages = (bytes_ + page_size_ - 1) / page_size_;
    const size_t n = std::min<size_t>(pages, 1024);
    for (size_t i = 0; i < n; ++i) {
      int node = -1;
      if (get_mempolicy(&node, nullptr, 0, data() + (pages * i / n) * page_size_, MPOL_F_NODE | MPOL_F_ADDR) != 0) {
        placement_ = Placement();
        placement_.expected = numa_node_;
        placement_.status = std::string("get_mempolicy failed: ") + std::strerror(errno);
        return;
      }
      ++placement_.histogram[node];
      placement_.on_expected += node == numa_node_;
    }
    placement_.sampled = n;
    placement_.status = placement_.on_expected == n ? "ok" : "MISPLACED";
    if (placement_.on_expected != n) {
      std::fprintf(stderr,
                   "[warn] %s: only %zu/%zu sampled pages are on NUMA node %d; the measurement does not exercise the "
                   "intended memory (mbind blocked by cgroup/seccomp? node full?)\n",
                   target_.name.c_str(), placement_.on_expected, n, numa_node_);
    }
  }

  const Placement& placement() const { return placement_; }

  std::string to_json() const {
    return JsonObject()
        .str("target", target_.name)
        .str("backend", backend_)
        .integer("numa_node", numa_node_)
        .integer("device", device_)
        .integer("bytes", static_cast<long long>(bytes_))
        .raw("placement", placement_.to_json())
        .dump();
  }

 private:
  void alloc_numa() {
    if (numa_available_) {
      data_ = numa_alloc_onnode(bytes_, numa_node_);
      numa_alloc_ = true;
    } else {
      data_ = mmap(nullptr, bytes_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      if (data_ == MAP_FAILED) data_ = nullptr;
    }
    if (!data_) fail("failed to allocate " + format_bytes(bytes_) + " for " + target_.name);
  }

  void alloc_managed(const GpuInfo& g) {
    GB_CHECK_CUDA(cudaSetDevice(g.ordinal));
    GB_CHECK_CUDA(cudaMallocManaged(&data_, bytes_, cudaMemAttachGlobal));
    backend_ = "cudaMallocManaged";
    cudaMemLocation dev_loc{};
    dev_loc.type = cudaMemLocationTypeDevice;
    dev_loc.id = g.ordinal;
    cudaMemLocation host_loc{};
    host_loc.type = cudaMemLocationTypeHost;
    host_loc.id = 0;
    GB_CHECK_CUDA(cudaMemAdvise(data_, bytes_, cudaMemAdviseSetPreferredLocation, dev_loc));
    GB_CHECK_CUDA(cudaMemAdvise(data_, bytes_, cudaMemAdviseSetAccessedBy, host_loc));
    if (!g.direct_managed_host_access) {
      std::fprintf(stderr,
                   "[warn] gpu%d reports directManagedMemAccessFromHost=0: CPU access to %s may migrate pages to the "
                   "CPU instead of crossing C2C\n",
                   g.ordinal, target_.name.c_str());
    }
  }

  Target target_;
  size_t bytes_;
  long page_size_;
  bool numa_available_;
  void* data_ = nullptr;
  bool numa_alloc_ = false;
  bool registered_ = false;
  int numa_node_ = -1;
  int device_ = -1;
  std::string backend_;
  Placement placement_;
};

// Lazily created thread pools, one per CPU NUMA node, covering every allowed CPU of that node.
class PoolSet {
 public:
  explicit PoolSet(const Topology& topo) : topo_(topo) {}

  HostThreadPool& get(int cpu_index) {
    auto& slot = pools_[cpu_index];
    if (!slot) slot = std::make_unique<HostThreadPool>(topo_.cpu(cpu_index).cpus);
    return *slot;
  }

  // Pool on the CPU closest to the buffer (used for initialization and verification).
  HostThreadPool& nearest(const Buffer& b) {
    const Target& t = b.target();
    if (t.kind == TargetKind::Cpu || t.kind == TargetKind::Pinned) return get(t.index);
    const int cpu = t.kind == TargetKind::Hbm ? topo_.gpu(t.index).cpu_index : -1;
    return get(cpu >= 0 ? cpu : 0);
  }

 private:
  const Topology& topo_;
  std::map<int, std::unique_ptr<HostThreadPool>> pools_;
};

inline void prefetch_managed(Buffer& b) {
  if (b.backend() != "cudaMallocManaged") return;
  cudaMemLocation loc{};
  loc.type = cudaMemLocationTypeDevice;
  loc.id = b.device();
  GB_CHECK_CUDA(cudaSetDevice(b.device()));
  GB_CHECK_CUDA(cudaMemPrefetchAsync(b.data(), b.bytes(), loc, 0, 0));
  GB_CHECK_CUDA(cudaDeviceSynchronize());
}

// Writes the verification pattern. CPU-accessible memory is first-touched by CPU threads so the NUMA binding
// decides placement; cudaMalloc memory is filled by its GPU.
inline void fill_buffer(Buffer& b, PoolSet& pools, uint64_t seed) {
  if (b.host_ok()) {
    parallel_fill(pools.nearest(b), b.data(), b.bytes(), seed);
  } else {
    GB_CHECK_CUDA(cudaSetDevice(b.device()));
    dev_fill_pattern_kernel<<<1024, 256>>>(reinterpret_cast<uint64_t*>(b.data()), b.bytes() / 8, seed);
    GB_CHECK_CUDA(cudaGetLastError());
    GB_CHECK_CUDA(cudaDeviceSynchronize());
  }
  prefetch_managed(b);
}

inline size_t count_mismatch(Buffer& b, PoolSet& pools, size_t bytes, uint64_t seed) {
  if (b.host_ok()) return parallel_count_mismatch(pools.nearest(b), b.data(), bytes, seed);
  GB_CHECK_CUDA(cudaSetDevice(b.device()));
  unsigned long long* count = nullptr;
  GB_CHECK_CUDA(cudaHostAlloc(&count, sizeof(*count), cudaHostAllocPortable | cudaHostAllocMapped));
  *count = 0;
  dev_count_mismatch_kernel<<<1024, 256>>>(reinterpret_cast<const uint64_t*>(b.data()), bytes / 8, seed, count);
  GB_CHECK_CUDA(cudaGetLastError());
  GB_CHECK_CUDA(cudaDeviceSynchronize());
  const size_t bad = *count;
  GB_CHECK_CUDA(cudaFreeHost(count));
  return bad;
}

// Allocated-and-initialized buffers keyed by target name, so matrices reuse them.
class BufferCache {
 public:
  BufferCache(const Topology& topo, PoolSet& pools, size_t bytes, HbmBackend backend, uint64_t seed)
      : topo_(topo), pools_(pools), bytes_(bytes), backend_(backend), seed_(seed) {}

  Buffer& get(const Target& t) {
    auto& slot = buffers_[t.name];
    if (!slot) {
      slot = std::make_unique<Buffer>(topo_, t, bytes_, backend_);
      fill_buffer(*slot, pools_, seed_);
      slot->check_placement();
    }
    return *slot;
  }

  uint64_t seed() const { return seed_; }

 private:
  const Topology& topo_;
  PoolSet& pools_;
  size_t bytes_;
  HbmBackend backend_;
  uint64_t seed_;
  std::map<std::string, std::unique_ptr<Buffer>> buffers_;
};

}  // namespace gb
