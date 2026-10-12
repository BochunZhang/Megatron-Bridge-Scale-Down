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
#include <sched.h>
#include <unistd.h>

#include "common.hpp"

namespace gb {

// A NUMA node that has CPUs (one Grace socket on GB200). `index` is the logical cpu<i> name.
struct CpuNode {
  int index = -1;
  int numa = -1;
  std::vector<int> cpus;  // CPUs of this node that the process may run on
  int total_cpus = 0;
  long long mem_total = -1;
  long long mem_free = -1;
};

struct GpuInfo {
  int ordinal = -1;
  std::string name;
  std::string pci_bus_id;
  std::string uuid;
  int cc_major = 0;
  int cc_minor = 0;
  int sm_count = 0;
  size_t total_mem = 0;
  int numa_config = -1;  // cudaDeviceNumaConfig
  int numa_id = -1;      // NUMA node of this GPU's HBM, -1 if HBM is not a NUMA node
  bool numa_overridden = false;
  int host_numa = -1;    // closest host NUMA node
  int cpu_index = -1;    // logical cpu<i> of host_numa
  int pageable_access = 0;
  int pageable_uses_host_page_tables = 0;
  int direct_managed_host_access = 0;
  int concurrent_managed_access = 0;
  int host_native_atomic = 0;
  long long hbm_node_total = -1;
  long long hbm_node_free = -1;
};

struct Topology {
  std::string hostname;
  long page_size = 0;
  bool numa_available = false;
  std::string cuda_status = "ok";
  int cuda_runtime_version = 0;
  int cuda_driver_version = 0;
  std::vector<CpuNode> cpus;
  std::vector<GpuInfo> gpus;
  std::vector<std::vector<int>> can_access_peer;
  std::vector<std::vector<int>> peer_rank;
  std::vector<std::vector<int>> peer_native_atomic;

  const CpuNode& cpu(int index) const {
    if (index < 0 || index >= static_cast<int>(cpus.size())) {
      fail("cpu" + std::to_string(index) + " does not exist (" + std::to_string(cpus.size()) + " CPU NUMA nodes)");
    }
    return cpus[index];
  }

  const GpuInfo& gpu(int index) const {
    if (index < 0 || index >= static_cast<int>(gpus.size())) {
      fail("gpu" + std::to_string(index) + " does not exist (" + std::to_string(gpus.size()) +
           " CUDA devices visible, CUDA status: " + cuda_status + ")");
    }
    return gpus[index];
  }

  int cpu_index_of_numa(int numa) const {
    for (const CpuNode& c : cpus) {
      if (c.numa == numa) return c.index;
    }
    return -1;
  }

  std::string to_json() const;
  void print_summary(FILE* out) const;
};

// "0=2,1=10" -> {0: 2, 1: 10}
inline std::map<int, int> parse_index_map(const std::string& text) {
  std::map<int, int> out;
  for (const std::string& item : split(text, ',')) {
    const size_t eq = item.find('=');
    if (eq == std::string::npos) fail("expected <index>=<numa node>, got '" + item + "'");
    out[static_cast<int>(parse_int(item.substr(0, eq), "index"))] =
        static_cast<int>(parse_int(item.substr(eq + 1), "numa node"));
  }
  return out;
}

inline int device_attr(cudaDeviceAttr attr, int device) {
  int value = -1;
  if (cudaDeviceGetAttribute(&value, attr, device) != cudaSuccess) {
    cudaGetLastError();
    return -1;
  }
  return value;
}

inline std::string uuid_string(const cudaUUID_t& uuid) {
  const auto* b = reinterpret_cast<const unsigned char*>(uuid.bytes);
  char buf[64];
  std::snprintf(buf, sizeof(buf), "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", b[0], b[1],
                b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]);
  return buf;
}

// `--hbm-numa g=node,...` overrides the HBM NUMA node reported by CUDA (for drivers that do not report it).
inline Topology discover_topology(const Args& args) {
  Topology t;
  char host[256] = {0};
  gethostname(host, sizeof(host) - 1);
  t.hostname = host;
  t.page_size = sysconf(_SC_PAGESIZE);
  t.numa_available = numa_available() >= 0;

  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  sched_getaffinity(0, sizeof(allowed), &allowed);

  if (t.numa_available) {
    struct bitmask* mask = numa_allocate_cpumask();
    for (int node = 0; node <= numa_max_node(); ++node) {
      if (numa_node_to_cpus(node, mask) != 0 || numa_bitmask_weight(mask) == 0) continue;
      CpuNode c;
      c.index = static_cast<int>(t.cpus.size());
      c.numa = node;
      for (unsigned cpu = 0; cpu < mask->size; ++cpu) {
        if (!numa_bitmask_isbitset(mask, cpu)) continue;
        ++c.total_cpus;
        if (cpu < CPU_SETSIZE && CPU_ISSET(cpu, &allowed)) c.cpus.push_back(static_cast<int>(cpu));
      }
      c.mem_total = numa_node_size64(node, &c.mem_free);
      t.cpus.push_back(c);
    }
    numa_free_cpumask(mask);
  } else {
    CpuNode c;
    c.index = 0;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
      if (CPU_ISSET(cpu, &allowed)) c.cpus.push_back(cpu);
    }
    c.total_cpus = static_cast<int>(c.cpus.size());
    t.cpus.push_back(c);
  }

  int n_gpus = 0;
  cudaError_t err = cudaGetDeviceCount(&n_gpus);
  if (err != cudaSuccess) {
    t.cuda_status = cudaGetErrorString(err);
    cudaGetLastError();
    n_gpus = 0;
  }
  cudaRuntimeGetVersion(&t.cuda_runtime_version);
  if (cudaDriverGetVersion(&t.cuda_driver_version) != cudaSuccess) cudaGetLastError();

  const std::map<int, int> hbm_override = parse_index_map(args.get("hbm-numa", ""));
  for (int d = 0; d < n_gpus; ++d) {
    cudaDeviceProp prop;
    GB_CHECK_CUDA(cudaGetDeviceProperties(&prop, d));
    GpuInfo g;
    g.ordinal = d;
    g.name = prop.name;
    char pci[32] = {0};
    if (cudaDeviceGetPCIBusId(pci, sizeof(pci), d) == cudaSuccess) g.pci_bus_id = pci;
    g.uuid = uuid_string(prop.uuid);
    g.cc_major = prop.major;
    g.cc_minor = prop.minor;
    g.sm_count = prop.multiProcessorCount;
    g.total_mem = prop.totalGlobalMem;
    g.numa_config = device_attr(cudaDevAttrNumaConfig, d);
    g.numa_id = g.numa_config == cudaDeviceNumaConfigNumaNode ? device_attr(cudaDevAttrNumaId, d) : -1;
    g.host_numa = device_attr(cudaDevAttrHostNumaId, d);
    g.pageable_access = device_attr(cudaDevAttrPageableMemoryAccess, d);
    g.pageable_uses_host_page_tables = device_attr(cudaDevAttrPageableMemoryAccessUsesHostPageTables, d);
    g.direct_managed_host_access = device_attr(cudaDevAttrDirectManagedMemAccessFromHost, d);
    g.concurrent_managed_access = device_attr(cudaDevAttrConcurrentManagedAccess, d);
    g.host_native_atomic = device_attr(cudaDevAttrHostNativeAtomicSupported, d);
    auto it = hbm_override.find(d);
    if (it != hbm_override.end()) {
      g.numa_id = it->second;
      g.numa_overridden = true;
    }
    g.cpu_index = t.cpu_index_of_numa(g.host_numa);
    if (t.numa_available && g.numa_id >= 0) g.hbm_node_total = numa_node_size64(g.numa_id, &g.hbm_node_free);
    t.gpus.push_back(g);
  }
  for (const auto& kv : hbm_override) {
    if (kv.first >= n_gpus) fail("--hbm-numa refers to gpu" + std::to_string(kv.first) + " which does not exist");
  }

  t.can_access_peer.assign(n_gpus, std::vector<int>(n_gpus, 0));
  t.peer_rank.assign(n_gpus, std::vector<int>(n_gpus, -1));
  t.peer_native_atomic.assign(n_gpus, std::vector<int>(n_gpus, 0));
  for (int i = 0; i < n_gpus; ++i) {
    for (int j = 0; j < n_gpus; ++j) {
      if (i == j) continue;
      int can = 0;
      if (cudaDeviceCanAccessPeer(&can, i, j) != cudaSuccess) cudaGetLastError();
      t.can_access_peer[i][j] = can;
      if (cudaDeviceGetP2PAttribute(&t.peer_rank[i][j], cudaDevP2PAttrPerformanceRank, i, j) != cudaSuccess) {
        cudaGetLastError();
      }
      if (cudaDeviceGetP2PAttribute(&t.peer_native_atomic[i][j], cudaDevP2PAttrNativeAtomicSupported, i, j) !=
          cudaSuccess) {
        cudaGetLastError();
      }
    }
  }
  // Peer mappings are needed for cudaMalloc buffers that are touched by another GPU (device / memcpy tests).
  for (int i = 0; i < n_gpus; ++i) {
    GB_CHECK_CUDA(cudaSetDevice(i));
    for (int j = 0; j < n_gpus; ++j) {
      if (i == j || !t.can_access_peer[i][j]) continue;
      cudaError_t e = cudaDeviceEnablePeerAccess(j, 0);
      if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {
        t.can_access_peer[i][j] = 0;
      }
      cudaGetLastError();
    }
  }
  if (n_gpus > 0) GB_CHECK_CUDA(cudaSetDevice(0));
  return t;
}

inline std::string Topology::to_json() const {
  std::vector<std::string> cpu_items;
  for (const CpuNode& c : cpus) {
    cpu_items.push_back(JsonObject()
                            .str("name", "cpu" + std::to_string(c.index))
                            .integer("numa", c.numa)
                            .str("allowed_cpus", compress_list(c.cpus))
                            .integer("allowed_cpu_count", static_cast<long long>(c.cpus.size()))
                            .integer("total_cpu_count", c.total_cpus)
                            .integer("mem_total_bytes", c.mem_total)
                            .integer("mem_free_bytes", c.mem_free)
                            .dump());
  }
  std::vector<std::string> gpu_items;
  for (const GpuInfo& g : gpus) {
    gpu_items.push_back(JsonObject()
                            .str("name", "gpu" + std::to_string(g.ordinal))
                            .str("model", g.name)
                            .str("pci_bus_id", g.pci_bus_id)
                            .str("uuid", g.uuid)
                            .str("compute_capability", std::to_string(g.cc_major) + "." + std::to_string(g.cc_minor))
                            .integer("sm_count", g.sm_count)
                            .integer("total_mem_bytes", static_cast<long long>(g.total_mem))
                            .integer("numa_config", g.numa_config)
                            .integer("hbm_numa", g.numa_id)
                            .boolean("hbm_numa_overridden", g.numa_overridden)
                            .integer("hbm_numa_total_bytes", g.hbm_node_total)
                            .integer("hbm_numa_free_bytes", g.hbm_node_free)
                            .integer("host_numa", g.host_numa)
                            .str("attached_cpu", g.cpu_index >= 0 ? "cpu" + std::to_string(g.cpu_index) : "")
                            .integer("pageable_memory_access", g.pageable_access)
                            .integer("pageable_memory_access_uses_host_page_tables", g.pageable_uses_host_page_tables)
                            .integer("direct_managed_mem_access_from_host", g.direct_managed_host_access)
                            .integer("concurrent_managed_access", g.concurrent_managed_access)
                            .integer("host_native_atomic_supported", g.host_native_atomic)
                            .dump());
  }
  auto matrix = [](const std::vector<std::vector<int>>& m) {
    std::vector<std::string> rows;
    for (const auto& row : m) rows.push_back(json_ints(row));
    return json_array(rows);
  };
  // NUMA distance over CPU nodes followed by HBM nodes.
  std::vector<int> nodes;
  for (const CpuNode& c : cpus) {
    if (c.numa >= 0) nodes.push_back(c.numa);
  }
  for (const GpuInfo& g : gpus) {
    if (g.numa_id >= 0 && std::find(nodes.begin(), nodes.end(), g.numa_id) == nodes.end()) nodes.push_back(g.numa_id);
  }
  std::vector<std::vector<int>> distance;
  if (numa_available) {
    for (int a : nodes) {
      std::vector<int> row;
      for (int b : nodes) row.push_back(numa_distance(a, b));
      distance.push_back(row);
    }
  }
  return JsonObject()
      .str("hostname", hostname)
      .integer("page_size", page_size)
      .boolean("numa_available", numa_available)
      .str("cuda_status", cuda_status)
      .integer("cuda_runtime_version", cuda_runtime_version)
      .integer("cuda_driver_version", cuda_driver_version)
      .raw("cpus", json_array(cpu_items))
      .raw("gpus", json_array(gpu_items))
      .raw("can_access_peer", matrix(can_access_peer))
      .raw("p2p_performance_rank", matrix(peer_rank))
      .raw("p2p_native_atomic", matrix(peer_native_atomic))
      .raw("numa_distance_nodes", json_ints(nodes))
      .raw("numa_distance", matrix(distance))
      .dump();
}

inline void Topology::print_summary(FILE* out) const {
  std::fprintf(out, "[topo] host %s, page size %ld, libnuma %s, CUDA runtime %d driver %d (%s)\n", hostname.c_str(),
               page_size, numa_available ? "yes" : "no", cuda_runtime_version, cuda_driver_version,
               cuda_status.c_str());
  for (const CpuNode& c : cpus) {
    std::fprintf(out, "[topo] cpu%d: numa %d, allowed cpus %s (%zu/%d), mem %s\n", c.index, c.numa,
                 compress_list(c.cpus).c_str(), c.cpus.size(), c.total_cpus,
                 c.mem_total >= 0 ? format_bytes(static_cast<size_t>(c.mem_total)).c_str() : "?");
  }
  for (const GpuInfo& g : gpus) {
    std::fprintf(out,
                 "[topo] gpu%d: %s sm_%d%d, %d SMs, pci %s, hbm numa %d%s, attached to cpu%d (host numa %d), "
                 "pageable access %d, direct managed host access %d\n",
                 g.ordinal, g.name.c_str(), g.cc_major, g.cc_minor, g.sm_count, g.pci_bus_id.c_str(), g.numa_id,
                 g.numa_overridden ? " (override)" : "", g.cpu_index, g.host_numa, g.pageable_access,
                 g.direct_managed_host_access);
  }
  for (size_t i = 0; i < gpus.size(); ++i) {
    std::string row;
    for (size_t j = 0; j < gpus.size(); ++j) row += i == j ? " -" : " " + std::to_string(can_access_peer[i][j]);
    std::fprintf(out, "[topo] peer access gpu%zu ->%s\n", i, row.c_str());
  }
}

}  // namespace gb
