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

#include <random>

#include "buffers.hpp"

namespace gb {

constexpr uint64_t kSrcSeed = 0x0123456789ABCDEFull;
constexpr uint64_t kDstSeed = 0xFEDCBA9876543210ull;
constexpr uint64_t kPoisonSeed = 0x5A5A5A5A00000000ull;

struct Common {
  size_t bytes = 0;
  int iters = 0;
  int warmup = 0;
  HbmBackend backend = HbmBackend::Auto;
  bool verify = true;
};

inline Common parse_common(const Args& a, const std::string& default_bytes, int iters, int warmup, bool with_verify) {
  Common c;
  c.bytes = parse_size(a.get("bytes", default_bytes));
  if (c.bytes % kLine != 0) fail("--bytes must be a multiple of 64");
  c.iters = a.get_int("iters", iters);
  c.warmup = a.get_int("warmup", warmup);
  if (c.iters < 1 || c.warmup < 0) fail("--iters must be >= 1 and --warmup >= 0");
  c.backend = parse_hbm_backend(a.get("hbm-backend", "auto"));
  if (with_verify) {
    const std::string v = a.get("verify", "full");
    if (v != "full" && v != "none") fail("--verify must be full or none");
    c.verify = v == "full";
  }
  return c;
}

inline JsonObject common_json(const Common& c) {
  return JsonObject().integer("bytes", static_cast<long long>(c.bytes)).integer("iters", c.iters).integer("warmup", c.warmup);
}

inline std::string skipped(const std::string& benchmark, const std::string& label, const std::string& reason) {
  std::printf("[skip] %s %s: %s\n", benchmark.c_str(), label.c_str(), reason.c_str());
  std::fflush(stdout);
  return JsonObject().str("benchmark", benchmark).str("label", label).str("skipped", reason).dump();
}

inline std::vector<double> to_gbps(const std::vector<double>& ms, size_t bytes, double factor) {
  std::vector<double> out;
  for (double m : ms) out.push_back(factor * static_cast<double>(bytes) / (m * 1e6));
  return out;
}

inline std::string verify_json(bool enabled, long long mismatches) {
  return JsonObject().str("mode", enabled ? "full" : "none").integer("mismatched_words", mismatches).dump();
}

inline void print_bandwidth(const char* tag, const std::string& label, const Stats& s, bool copy, bool verify,
                            long long mismatches) {
  std::printf("[%s] %s: median %.2f GB/s (min %.2f, max %.2f)", tag, label.c_str(), s.median, s.min, s.max);
  if (copy) std::printf(", read+write %.2f GB/s", 2 * s.median);
  if (copy && verify) std::printf(", verify %s", mismatches == 0 ? "ok" : "FAILED");
  std::printf("\n");
  std::fflush(stdout);
  if (copy && verify && mismatches != 0) {
    std::fprintf(stderr, "[error] %s: %lld mismatched 64-bit words after the copy\n", label.c_str(), mismatches);
  }
}

// Pinned host scalar that every GPU can read and write.
template <typename T>
class HostScalar {
 public:
  HostScalar() {
    GB_CHECK_CUDA(cudaHostAlloc(&ptr_, sizeof(T), cudaHostAllocPortable | cudaHostAllocMapped));
    *ptr_ = T();
  }
  ~HostScalar() { cudaFreeHost(ptr_); }
  HostScalar(const HostScalar&) = delete;
  HostScalar& operator=(const HostScalar&) = delete;
  T* get() const { return ptr_; }

 private:
  T* ptr_ = nullptr;
};

// ================================================================ CPU-thread bandwidth (host / relay)

inline std::string host_case(const Topology& topo, PoolSet& pools, int cpu_index, size_t n_threads, HostOp op,
                             HostKernel kernel, Buffer* src, Buffer* dst, const Common& c, const std::string& benchmark,
                             const std::string& label, const JsonObject& extra) {
  HostThreadPool& pool = pools.get(cpu_index);
  if (op == HostOp::Copy && c.verify) fill_buffer(*dst, pools, kPoisonSeed);  // the check below covers this case only
  std::vector<double> ms;
  double late_us = 0;
  for (int i = 0; i < c.warmup + c.iters; ++i) {
    const HostTiming t = timed_host_op(pool, n_threads, op, kernel, src ? src->data() : nullptr,
                                       dst ? dst->data() : nullptr, c.bytes);
    if (i >= c.warmup) {
      ms.push_back(t.ms);
      late_us = std::max(late_us, t.max_late_us);
    }
  }
  long long mismatches = -1;
  if (op == HostOp::Copy && c.verify) mismatches = static_cast<long long>(count_mismatch(*dst, pools, c.bytes, kSrcSeed));
  const Stats bw = summarize(to_gbps(ms, c.bytes, 1));
  const std::vector<int> used(pool.cpus().begin(), pool.cpus().begin() + n_threads);
  JsonObject j;
  j.str("benchmark", benchmark)
      .str("label", label)
      .str("op", op_name(op))
      .str("kernel", kernel_name(kernel))
      .str("cpu", "cpu" + std::to_string(cpu_index))
      .integer("cpu_numa", topo.cpu(cpu_index).numa)
      .integer("threads", static_cast<long long>(n_threads))
      .str("thread_cpus", compress_list(used))
      .integer("pin_failures", static_cast<long long>(pool.pin_failures()))
      .raw("src", src ? src->to_json() : "null")
      .raw("dst", dst ? dst->to_json() : "null")
      .merge(common_json(c))
      .raw("times_ms", json_numbers(ms))
      .raw("payload_gb_s", json_stats(bw));
  if (op == HostOp::Copy) j.raw("aggregate_read_write_gb_s", json_stats(summarize(to_gbps(ms, c.bytes, 2))));
  j.num("max_late_start_us", late_us).raw("verify", verify_json(op == HostOp::Copy && c.verify, mismatches)).merge(extra);
  print_bandwidth(benchmark.c_str(), label + " [" + kernel_name(kernel) + ", " + std::to_string(n_threads) + " threads]",
                  bw, op == HostOp::Copy, c.verify, mismatches);
  if (late_us > 1000) {
    std::fprintf(stderr, "[warn] %s: a worker started %.0f us late; bandwidth is underestimated\n", label.c_str(),
                 late_us);
  }
  if (pool.pin_failures()) std::fprintf(stderr, "[warn] %zu pool threads could not be pinned\n", pool.pin_failures());
  return j.dump();
}

// "all" or a positive count; returns 0 if the count exceeds the pool.
inline size_t resolve_threads(const std::string& spec, size_t pool_size) {
  if (spec == "all") return pool_size;
  const long long n = parse_int(spec, "--threads");
  if (n < 1) fail("--threads entries must be >= 1 or 'all'");
  return static_cast<size_t>(n) <= pool_size ? static_cast<size_t>(n) : 0;
}

inline std::vector<HostKernel> parse_kernels(const std::string& spec) {
  std::vector<HostKernel> out;
  for (const std::string& k : split(spec, ',')) out.push_back(parse_kernel(k));
  if (out.empty()) fail("--kernels is empty");
  return out;
}

// ---------------------------------------------------------------- relay

struct RelayOptions {
  int src_gpu = 0;
  std::vector<int> dst_gpus;
  std::vector<std::string> cpu_specs;
  std::vector<std::string> threads;
  std::vector<HostKernel> kernels;
  Common c;
};

inline RelayOptions parse_relay(const Args& a) {
  RelayOptions o;
  o.src_gpu = a.get_int("src-gpu", 0);
  for (const std::string& d : split(a.get("dst-gpus", "1,2"), ',')) o.dst_gpus.push_back(static_cast<int>(parse_int(d, "--dst-gpus")));
  o.cpu_specs = split(a.get("cpu-nodes", "src,dst"), ',');
  o.threads = split(a.get("threads", "all"), ',');
  o.kernels = parse_kernels(a.get("kernels", "ldp_stp"));
  o.c = parse_common(a, "4GiB", 10, 2, true);
  return o;
}

// GPU src HBM --(CPU threads load)--> registers --(CPU threads store)--> GPU dst HBM. Nothing is staged in CPU
// memory: each worker moves 64 B per iteration directly between the two HBM NUMA mappings.
inline std::vector<std::string> run_relay(const Topology& topo, const RelayOptions& o) {
  std::vector<std::string> results;
  const GpuInfo& sg = topo.gpu(o.src_gpu);
  PoolSet pools(topo);
  BufferCache srcs(topo, pools, o.c.bytes, o.c.backend, kSrcSeed);
  BufferCache dsts(topo, pools, o.c.bytes, o.c.backend, kDstSeed);
  Buffer& src = srcs.get(parse_target("hbm" + std::to_string(o.src_gpu), topo));
  for (int dst_gpu : o.dst_gpus) {
    if (dst_gpu == o.src_gpu) fail("--dst-gpus must not contain the source GPU");
    const GpuInfo& dg = topo.gpu(dst_gpu);
    Buffer& dst = dsts.get(parse_target("hbm" + std::to_string(dst_gpu), topo));
    const bool same_cpu = sg.cpu_index >= 0 && sg.cpu_index == dg.cpu_index;
    const std::string s = "gpu" + std::to_string(o.src_gpu);
    const std::string d = "gpu" + std::to_string(dst_gpu);
    const std::string sc = "cpu" + std::to_string(sg.cpu_index);
    const std::string dc = "cpu" + std::to_string(dg.cpu_index);
    const std::string path = same_cpu ? s + " -C2C-> " + sc + " -C2C-> " + d
                                      : s + " -C2C-> " + sc + " -> " + dc + " -C2C-> " + d;
    std::printf("[relay] %s -> %s: %s (%s)\n", s.c_str(), d.c_str(), path.c_str(),
                same_cpu ? "same CPU" : "crosses CPUs");
    if (sg.cpu_index < 0 || dg.cpu_index < 0) {
      std::fprintf(stderr, "[warn] could not map %s/%s to a CPU NUMA node (host numa %d/%d)\n", s.c_str(), d.c_str(),
                   sg.host_numa, dg.host_numa);
    }

    std::vector<int> cpu_nodes;
    for (const std::string& spec : o.cpu_specs) {
      int cpu = -1;
      if (spec == "src") {
        cpu = sg.cpu_index;
      } else if (spec == "dst") {
        cpu = dg.cpu_index;
      } else {
        cpu = static_cast<int>(parse_int(spec, "--cpu-nodes"));
      }
      topo.cpu(cpu);
      if (std::find(cpu_nodes.begin(), cpu_nodes.end(), cpu) == cpu_nodes.end()) cpu_nodes.push_back(cpu);
    }

    for (int cpu : cpu_nodes) {
      const std::string role = cpu == sg.cpu_index ? "src_attached" : cpu == dg.cpu_index ? "dst_attached" : "other";
      for (HostKernel kernel : o.kernels) {
        for (const std::string& tspec : o.threads) {
          const std::string label = s + "->" + d + " via cpu" + std::to_string(cpu);
          const size_t n = resolve_threads(tspec, pools.get(cpu).size());
          if (n == 0) {
            results.push_back(skipped("relay", label, "cpu" + std::to_string(cpu) + " has only " +
                                                          std::to_string(pools.get(cpu).size()) + " allowed CPUs"));
            continue;
          }
          JsonObject extra;
          extra.integer("src_gpu", o.src_gpu)
              .integer("dst_gpu", dst_gpu)
              .str("path", path)
              .str("path_class", same_cpu ? "same_cpu" : "cross_cpu")
              .str("copy_cpu_role", role);
          results.push_back(host_case(topo, pools, cpu, n, HostOp::Copy, kernel, &src, &dst, o.c, "relay", label, extra));
        }
      }
    }
  }
  return results;
}

// ---------------------------------------------------------------- host

struct HostOptions {
  HostOp op = HostOp::Copy;
  std::vector<Target> srcs;
  std::vector<Target> dsts;
  std::vector<int> cpu_nodes;
  std::vector<std::string> threads;
  std::vector<HostKernel> kernels;
  Common c;
};

inline std::vector<int> parse_cpu_nodes(const std::string& spec, const Topology& topo) {
  std::vector<int> out;
  if (spec == "all") {
    for (const CpuNode& c : topo.cpus) out.push_back(c.index);
    return out;
  }
  for (const std::string& item : split(spec, ',')) {
    out.push_back(static_cast<int>(parse_int(item, "--cpu-nodes")));
    topo.cpu(out.back());
  }
  return out;
}

inline HostOptions parse_host(const Args& a, const Topology& topo) {
  HostOptions o;
  o.op = parse_op(a.get("op", "copy"));
  if (o.op != HostOp::Write) o.srcs = expand_targets(a.get("src", ""), topo);
  if (o.op != HostOp::Read) o.dsts = expand_targets(a.get("dst", ""), topo);
  o.cpu_nodes = parse_cpu_nodes(a.get("cpu-nodes", "all"), topo);
  o.threads = split(a.get("threads", "all"), ',');
  o.kernels = parse_kernels(a.get("kernels", "ldp_stp"));
  o.c = parse_common(a, "4GiB", 10, 2, true);
  return o;
}

inline std::vector<std::pair<const Target*, const Target*>> target_pairs(HostOp op, const std::vector<Target>& srcs,
                                                                         const std::vector<Target>& dsts) {
  std::vector<std::pair<const Target*, const Target*>> pairs;
  if (op == HostOp::Read) {
    for (const Target& s : srcs) pairs.emplace_back(&s, nullptr);
  } else if (op == HostOp::Write) {
    for (const Target& d : dsts) pairs.emplace_back(nullptr, &d);
  } else {
    for (const Target& s : srcs) {
      for (const Target& d : dsts) pairs.emplace_back(&s, &d);
    }
  }
  return pairs;
}

inline std::string pair_label(HostOp op, const Target* s, const Target* d) {
  if (op == HostOp::Read) return "read " + s->name;
  if (op == HostOp::Write) return "write " + d->name;
  return s->name + "->" + d->name;
}

inline std::vector<std::string> run_host(const Topology& topo, const HostOptions& o) {
  std::vector<std::string> results;
  PoolSet pools(topo);
  BufferCache srcs(topo, pools, o.c.bytes, o.c.backend, kSrcSeed);
  BufferCache dsts(topo, pools, o.c.bytes, o.c.backend, kDstSeed);
  for (const auto& p : target_pairs(o.op, o.srcs, o.dsts)) {
    const std::string base = pair_label(o.op, p.first, p.second);
    std::string problem = p.first ? host_access_problem(*p.first) : "";
    if (problem.empty() && p.second) problem = host_access_problem(*p.second);
    if (!problem.empty()) {
      results.push_back(skipped("host", base, problem));
      continue;
    }
    Buffer* src = p.first ? &srcs.get(*p.first) : nullptr;
    Buffer* dst = p.second ? &dsts.get(*p.second) : nullptr;
    for (int cpu : o.cpu_nodes) {
      for (HostKernel kernel : o.kernels) {
        for (const std::string& tspec : o.threads) {
          const std::string label = base + " by cpu" + std::to_string(cpu);
          const size_t n = resolve_threads(tspec, pools.get(cpu).size());
          if (n == 0) {
            results.push_back(skipped("host", label, "not enough allowed CPUs on cpu" + std::to_string(cpu)));
            continue;
          }
          results.push_back(host_case(topo, pools, cpu, n, o.op, kernel, src, dst, o.c, "host", label, JsonObject()));
        }
      }
    }
  }
  return results;
}

// ================================================================ GPU-side bandwidth (device / memcpy)

class CudaTimer {
 public:
  explicit CudaTimer(int device) {
    GB_CHECK_CUDA(cudaSetDevice(device));
    GB_CHECK_CUDA(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    GB_CHECK_CUDA(cudaEventCreate(&start_));
    GB_CHECK_CUDA(cudaEventCreate(&stop_));
  }
  ~CudaTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
    cudaStreamDestroy(stream_);
  }
  CudaTimer(const CudaTimer&) = delete;
  CudaTimer& operator=(const CudaTimer&) = delete;

  cudaStream_t stream() const { return stream_; }

  template <typename F>
  double time_ms(F&& launch) {
    GB_CHECK_CUDA(cudaEventRecord(start_, stream_));
    launch();
    GB_CHECK_CUDA(cudaGetLastError());
    GB_CHECK_CUDA(cudaEventRecord(stop_, stream_));
    GB_CHECK_CUDA(cudaEventSynchronize(stop_));
    float ms = 0;
    GB_CHECK_CUDA(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
  }

 private:
  cudaStream_t stream_ = nullptr;
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

// Shared tail of the device and memcpy cases: times `launch`, verifies copies, builds the JSON record.
template <typename F>
std::string gpu_case(int gpu, HostOp op, Buffer* src, Buffer* dst, PoolSet& pools, const Common& c,
                     const std::string& benchmark, const std::string& label, JsonObject extra, F&& launch) {
  if (op == HostOp::Copy && c.verify) fill_buffer(*dst, pools, kPoisonSeed);
  CudaTimer timer(gpu);
  std::vector<double> ms;
  for (int i = 0; i < c.warmup + c.iters; ++i) {
    const double t = timer.time_ms([&] { launch(timer.stream()); });
    if (i >= c.warmup) ms.push_back(t);
  }
  long long mismatches = -1;
  if (op == HostOp::Copy && c.verify) mismatches = static_cast<long long>(count_mismatch(*dst, pools, c.bytes, kSrcSeed));
  const Stats bw = summarize(to_gbps(ms, c.bytes, 1));
  JsonObject j;
  j.str("benchmark", benchmark)
      .str("label", label)
      .str("op", op_name(op))
      .integer("gpu", gpu)
      .raw("src", src ? src->to_json() : "null")
      .raw("dst", dst ? dst->to_json() : "null")
      .merge(common_json(c))
      .merge(extra)
      .raw("times_ms", json_numbers(ms))
      .raw("payload_gb_s", json_stats(bw));
  if (op == HostOp::Copy) j.raw("aggregate_read_write_gb_s", json_stats(summarize(to_gbps(ms, c.bytes, 2))));
  j.raw("verify", verify_json(op == HostOp::Copy && c.verify, mismatches));
  print_bandwidth(benchmark.c_str(), label, bw, op == HostOp::Copy, c.verify, mismatches);
  return j.dump();
}

// ---------------------------------------------------------------- device

struct DeviceOptions {
  int gpu = 0;
  HostOp op = HostOp::Copy;
  std::vector<Target> srcs;
  std::vector<Target> dsts;
  std::vector<std::string> blocks;
  int threads_per_block = 512;
  Common c;
};

inline DeviceOptions parse_device(const Args& a, const Topology& topo) {
  DeviceOptions o;
  o.gpu = a.get_int("gpu", 0);
  topo.gpu(o.gpu);
  o.op = parse_op(a.get("op", "copy"));
  if (o.op != HostOp::Write) o.srcs = expand_targets(a.get("src", ""), topo);
  if (o.op != HostOp::Read) o.dsts = expand_targets(a.get("dst", ""), topo);
  o.blocks = split(a.get("blocks", "auto"), ',');
  o.threads_per_block = a.get_int("threads-per-block", 512);
  if (o.threads_per_block < 32 || o.threads_per_block > 1024) fail("--threads-per-block must be in [32, 1024]");
  o.c = parse_common(a, "4GiB", 10, 2, true);
  return o;
}

inline std::vector<std::string> run_device(const Topology& topo, const DeviceOptions& o) {
  std::vector<std::string> results;
  PoolSet pools(topo);
  BufferCache srcs(topo, pools, o.c.bytes, o.c.backend, kSrcSeed);
  BufferCache dsts(topo, pools, o.c.bytes, o.c.backend, kDstSeed);
  const GpuInfo& g = topo.gpu(o.gpu);
  GB_CHECK_CUDA(cudaSetDevice(o.gpu));
  uint4* sink = nullptr;
  GB_CHECK_CUDA(cudaMalloc(&sink, sizeof(uint4)));
  const size_t n = o.c.bytes / sizeof(uint4);
  for (const auto& p : target_pairs(o.op, o.srcs, o.dsts)) {
    const std::string base = "gpu" + std::to_string(o.gpu) + " " + pair_label(o.op, p.first, p.second);
    std::string problem = p.first ? device_access_problem(topo, o.gpu, *p.first, o.c.backend) : "";
    if (problem.empty() && p.second) problem = device_access_problem(topo, o.gpu, *p.second, o.c.backend);
    if (!problem.empty()) {
      results.push_back(skipped("device", base, problem));
      continue;
    }
    Buffer* src = p.first ? &srcs.get(*p.first) : nullptr;
    Buffer* dst = p.second ? &dsts.get(*p.second) : nullptr;
    for (const std::string& bspec : o.blocks) {
      const int blocks = bspec == "auto" ? g.sm_count * (2048 / o.threads_per_block)
                                         : static_cast<int>(parse_int(bspec, "--blocks"));
      if (blocks < 1) fail("--blocks entries must be >= 1 or 'auto'");
      const std::string label = base + " [" + std::to_string(blocks) + "x" + std::to_string(o.threads_per_block) + "]";
      JsonObject extra;
      extra.integer("blocks", blocks).integer("threads_per_block", o.threads_per_block);
      const int tpb = o.threads_per_block;
      results.push_back(gpu_case(o.gpu, o.op, src, dst, pools, o.c, "device", label, extra, [&](cudaStream_t s) {
        switch (o.op) {
          case HostOp::Read:
            dev_read_kernel<<<blocks, tpb, 0, s>>>(reinterpret_cast<const uint4*>(src->data()), n, sink);
            break;
          case HostOp::Write:
            dev_write_kernel<<<blocks, tpb, 0, s>>>(reinterpret_cast<uint4*>(dst->data()), n,
                                                    make_uint4(0xA5A5A5A5u, 0xA5A5A5A5u, 0xA5A5A5A5u, 0xA5A5A5A5u));
            break;
          case HostOp::Copy:
            dev_copy_kernel<<<blocks, tpb, 0, s>>>(reinterpret_cast<const uint4*>(src->data()),
                                                   reinterpret_cast<uint4*>(dst->data()), n);
            break;
        }
      }));
    }
  }
  GB_CHECK_CUDA(cudaSetDevice(o.gpu));
  GB_CHECK_CUDA(cudaFree(sink));
  return results;
}

// ---------------------------------------------------------------- memcpy

struct MemcpyOptions {
  int gpu = 0;
  std::vector<Target> srcs;
  std::vector<Target> dsts;
  Common c;
};

inline MemcpyOptions parse_memcpy(const Args& a, const Topology& topo) {
  MemcpyOptions o;
  o.gpu = a.get_int("gpu", 0);
  topo.gpu(o.gpu);
  o.srcs = expand_targets(a.get("src", ""), topo);
  o.dsts = expand_targets(a.get("dst", ""), topo);
  o.c = parse_common(a, "4GiB", 10, 2, true);
  return o;
}

// cudaMemcpyAsync(cudaMemcpyDefault) on a stream of --gpu (gh_benchmark's cudaMemcpy heatmap).
inline std::vector<std::string> run_memcpy(const Topology& topo, const MemcpyOptions& o) {
  std::vector<std::string> results;
  PoolSet pools(topo);
  BufferCache srcs(topo, pools, o.c.bytes, o.c.backend, kSrcSeed);
  BufferCache dsts(topo, pools, o.c.bytes, o.c.backend, kDstSeed);
  for (const auto& p : target_pairs(HostOp::Copy, o.srcs, o.dsts)) {
    Buffer* src = &srcs.get(*p.first);
    Buffer* dst = &dsts.get(*p.second);
    const std::string label = "stream on gpu" + std::to_string(o.gpu) + " " + pair_label(HostOp::Copy, p.first, p.second);
    results.push_back(gpu_case(o.gpu, HostOp::Copy, src, dst, pools, o.c, "memcpy", label, JsonObject(),
                               [&](cudaStream_t s) {
                                 GB_CHECK_CUDA(cudaMemcpyAsync(dst->data(), src->data(), o.c.bytes, cudaMemcpyDefault, s));
                               }));
  }
  return results;
}

// ---------------------------------------------------------------- staged

struct StagedOptions {
  int src_gpu = 0;
  std::vector<int> dst_gpus;
  std::vector<std::string> cpu_specs;
  std::vector<std::string> modes;
  std::vector<size_t> chunks;
  int slots = 2;
  std::vector<std::string> threads;
  std::vector<HostKernel> kernels;
  Common c;
};

inline StagedOptions parse_staged(const Args& a) {
  StagedOptions o;
  o.src_gpu = a.get_int("src-gpu", 0);
  for (const std::string& d : split(a.get("dst-gpus", "1,3"), ',')) o.dst_gpus.push_back(static_cast<int>(parse_int(d, "--dst-gpus")));
  o.cpu_specs = split(a.get("cpu-nodes", "src,dst"), ',');
  o.modes = split(a.get("modes", "pinned,direct"), ',');
  for (const std::string& m : o.modes) {
    if (m != "pinned" && m != "direct") fail("--modes entries must be pinned or direct, got '" + m + "'");
  }
  if (o.modes.empty()) fail("--modes is empty");
  for (const std::string& s : split(a.get("chunk", "64M"), ',')) {
    const size_t n = parse_size(s);
    if (n % kLine != 0) fail("--chunk entries must be multiples of 64");
    o.chunks.push_back(n);
  }
  if (o.chunks.empty()) fail("--chunk is empty");
  o.slots = a.get_int("slots", 2);
  if (o.slots < 1) fail("--slots must be >= 1");
  o.threads = split(a.get("threads", "all"), ',');
  o.kernels = parse_kernels(a.get("kernels", "ldp_stp"));
  o.c = parse_common(a, "4GiB", 10, 2, true);
  return o;
}

// Chunk k: cudaMemcpyAsync src -> staging slot k % slots on a stream of the source GPU (D2H), then
// staging slot -> dst on a stream of the destination GPU (H2D). A slot is reused only after the H2D that last read
// it has finished, so up to `slots` chunks are in flight and D2H of one chunk overlaps H2D of the previous one.
class StagedPipeline {
 public:
  StagedPipeline(int src_gpu, int dst_gpu, int slots)
      : src_gpu_(src_gpu), dst_gpu_(dst_gpu), d2h_done_(slots), h2d_done_(slots) {
    GB_CHECK_CUDA(cudaSetDevice(src_gpu_));
    GB_CHECK_CUDA(cudaStreamCreateWithFlags(&src_stream_, cudaStreamNonBlocking));
    for (cudaEvent_t& e : d2h_done_) GB_CHECK_CUDA(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    GB_CHECK_CUDA(cudaSetDevice(dst_gpu_));
    GB_CHECK_CUDA(cudaStreamCreateWithFlags(&dst_stream_, cudaStreamNonBlocking));
    for (cudaEvent_t& e : h2d_done_) GB_CHECK_CUDA(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
  }
  ~StagedPipeline() {
    for (cudaEvent_t e : d2h_done_) cudaEventDestroy(e);
    for (cudaEvent_t e : h2d_done_) cudaEventDestroy(e);
    cudaStreamDestroy(src_stream_);
    cudaStreamDestroy(dst_stream_);
  }
  StagedPipeline(const StagedPipeline&) = delete;
  StagedPipeline& operator=(const StagedPipeline&) = delete;

  struct Timing {
    double ms = 0;          // wall time of one full transfer
    double enqueue_ms = 0;  // time spent issuing the copies; close to `ms` means the host, not the link, limits
  };

  // The streams belong to different GPUs, so CUDA events cannot time the transfer; host wall time is used instead.
  Timing run(const uint8_t* src, uint8_t* staging, uint8_t* dst, size_t bytes, size_t chunk) {
    const size_t slots = d2h_done_.size();
    const uint64_t t0 = now_ns();
    for (size_t k = 0, off = 0; off < bytes; ++k, off += chunk) {
      const size_t n = std::min(chunk, bytes - off);
      const size_t slot = k % slots;
      uint8_t* stage = staging + slot * chunk;
      GB_CHECK_CUDA(cudaSetDevice(src_gpu_));
      if (k >= slots) GB_CHECK_CUDA(cudaStreamWaitEvent(src_stream_, h2d_done_[slot], 0));
      GB_CHECK_CUDA(cudaMemcpyAsync(stage, src + off, n, cudaMemcpyDefault, src_stream_));
      GB_CHECK_CUDA(cudaEventRecord(d2h_done_[slot], src_stream_));
      GB_CHECK_CUDA(cudaSetDevice(dst_gpu_));
      GB_CHECK_CUDA(cudaStreamWaitEvent(dst_stream_, d2h_done_[slot], 0));
      GB_CHECK_CUDA(cudaMemcpyAsync(dst + off, stage, n, cudaMemcpyDefault, dst_stream_));
      GB_CHECK_CUDA(cudaEventRecord(h2d_done_[slot], dst_stream_));
    }
    const uint64_t t_enqueued = now_ns();
    GB_CHECK_CUDA(cudaStreamSynchronize(src_stream_));
    GB_CHECK_CUDA(cudaStreamSynchronize(dst_stream_));
    const uint64_t t1 = now_ns();
    return {static_cast<double>(t1 - t0) / 1e6, static_cast<double>(t_enqueued - t0) / 1e6};
  }

 private:
  int src_gpu_;
  int dst_gpu_;
  cudaStream_t src_stream_ = nullptr;
  cudaStream_t dst_stream_ = nullptr;
  std::vector<cudaEvent_t> d2h_done_;
  std::vector<cudaEvent_t> h2d_done_;
};

// cudaMalloc on GPU src -> cudaMalloc on GPU dst through a CPU, two modes:
//   pinned: copy engines move the data through pinned LPDDR of the CPU (chunked D2H then H2D; staged in CPU memory)
//   direct: CPU threads load from src and store to dst through registers (nothing staged in CPU memory); needs CPU
//           load/store access to cudaMalloc memory, otherwise the case is skipped with the fault that was caught
inline std::vector<std::string> run_staged(const Topology& topo, const StagedOptions& o) {
  std::vector<std::string> results;
  const GpuInfo& sg = topo.gpu(o.src_gpu);
  PoolSet pools(topo);
  BufferCache srcs(topo, pools, o.c.bytes, o.c.backend, kSrcSeed);
  BufferCache dsts(topo, pools, o.c.bytes, o.c.backend, kDstSeed);
  const bool run_pinned = std::find(o.modes.begin(), o.modes.end(), "pinned") != o.modes.end();
  const bool run_direct = std::find(o.modes.begin(), o.modes.end(), "direct") != o.modes.end();
  Buffer& src = srcs.get(parse_target("cuda" + std::to_string(o.src_gpu), topo));
  for (int dst_gpu : o.dst_gpus) {
    if (dst_gpu == o.src_gpu) fail("--dst-gpus must not contain the source GPU");
    const GpuInfo& dg = topo.gpu(dst_gpu);
    Buffer& dst = dsts.get(parse_target("cuda" + std::to_string(dst_gpu), topo));
    const bool same_cpu = sg.cpu_index >= 0 && sg.cpu_index == dg.cpu_index;
    const std::string s = "gpu" + std::to_string(o.src_gpu);
    const std::string d = "gpu" + std::to_string(dst_gpu);

    std::vector<int> cpu_nodes;
    for (const std::string& spec : o.cpu_specs) {
      int cpu = -1;
      if (spec == "src") {
        cpu = sg.cpu_index;
      } else if (spec == "dst") {
        cpu = dg.cpu_index;
      } else {
        cpu = static_cast<int>(parse_int(spec, "--cpu-nodes"));
      }
      topo.cpu(cpu);
      if (std::find(cpu_nodes.begin(), cpu_nodes.end(), cpu) == cpu_nodes.end()) cpu_nodes.push_back(cpu);
    }

    // direct mode needs CPU load/store on cudaMalloc pointers; probe one word of each buffer (faults are caught).
    std::string direct_problem;
    if (run_direct) {
      uint64_t word = 0;
      std::string fault = cpu_touch(reinterpret_cast<uint64_t*>(src.data()), &word);
      if (!fault.empty()) {
        direct_problem = "CPU cannot load/store cudaMalloc memory of " + s + " (" + fault + ")";
      } else if (word != pattern_word(0, kSrcSeed)) {
        direct_problem = "CPU load of " + s + " cudaMalloc memory returned an unexpected value";
      } else if (!(fault = cpu_touch(reinterpret_cast<uint64_t*>(dst.data()), &word)).empty()) {
        direct_problem = "CPU cannot load/store cudaMalloc memory of " + d + " (" + fault + ")";
      }
    }

    for (int cpu : cpu_nodes) {
      const std::string role = cpu == sg.cpu_index ? (same_cpu ? "shared" : "src_attached")
                               : cpu == dg.cpu_index ? "dst_attached"
                                                     : "other";
      const std::string path = s + " cuda -D2H-> pinned" + std::to_string(cpu) + " -H2D-> " + d + " cuda";
      if (run_pinned) {
        for (size_t chunk : o.chunks) {
          const size_t stage_bytes = std::min(chunk, o.c.bytes) * static_cast<size_t>(o.slots);
          Buffer staging(topo, parse_target("pinned" + std::to_string(cpu), topo), stage_bytes, o.c.backend);
          fill_buffer(staging, pools, kPoisonSeed);  // pages are bound by numa_alloc_onnode; checked below
          staging.check_placement();
          StagedPipeline pipe(o.src_gpu, dst_gpu, o.slots);
          if (o.c.verify) fill_buffer(dst, pools, kPoisonSeed);
          std::vector<double> ms;
          std::vector<double> enqueue_ms;
          for (int i = 0; i < o.c.warmup + o.c.iters; ++i) {
            const StagedPipeline::Timing t = pipe.run(src.data(), staging.data(), dst.data(), o.c.bytes, chunk);
            if (i >= o.c.warmup) {
              ms.push_back(t.ms);
              enqueue_ms.push_back(t.enqueue_ms);
            }
          }
          long long mismatches = -1;
          if (o.c.verify) mismatches = static_cast<long long>(count_mismatch(dst, pools, o.c.bytes, kSrcSeed));
          const Stats bw = summarize(to_gbps(ms, o.c.bytes, 1));
          const std::string label = s + "->" + d + " via pinned" + std::to_string(cpu) + " [chunk " +
                                    format_bytes(chunk) + ", " + std::to_string(o.slots) + " slots]";
          JsonObject j;
          j.str("benchmark", "staged")
              .str("label", label)
              .str("mode", "pinned")
              .boolean("staged_in_cpu_memory", true)
              .integer("src_gpu", o.src_gpu)
              .integer("dst_gpu", dst_gpu)
              .str("path", path)
              .str("path_class", same_cpu ? "same_cpu" : "cross_cpu")
              .str("staging_cpu", "cpu" + std::to_string(cpu))
              .str("staging_cpu_role", role)
              .boolean("d2h_crosses_cpu", cpu != sg.cpu_index)
              .boolean("h2d_crosses_cpu", cpu != dg.cpu_index)
              .integer("chunk_bytes", static_cast<long long>(chunk))
              .integer("slots", o.slots)
              .raw("src", src.to_json())
              .raw("staging", staging.to_json())
              .raw("dst", dst.to_json())
              .merge(common_json(o.c))
              .raw("times_ms", json_numbers(ms))
              .raw("enqueue_ms", json_numbers(enqueue_ms))
              .raw("payload_gb_s", json_stats(bw))
              .raw("verify", verify_json(o.c.verify, mismatches));
          // Single line like print_bandwidth, without its read+write column (each byte crosses C2C twice here).
          std::printf("[staged] %s: median %.2f GB/s (min %.2f, max %.2f)", label.c_str(), bw.median, bw.min, bw.max);
          if (o.c.verify) std::printf(", verify %s", mismatches == 0 ? "ok" : "FAILED");
          std::printf("\n");
          std::fflush(stdout);
          if (o.c.verify && mismatches != 0) {
            std::fprintf(stderr, "[error] %s: %lld mismatched 64-bit words after the copy\n", label.c_str(), mismatches);
          }
          const double enqueue_share = summarize(enqueue_ms).median / summarize(ms).median;
          if (enqueue_share > 0.9) {
            std::fprintf(stderr,
                         "[warn] %s: issuing the copies took %.0f%% of the transfer time; bandwidth is host-bound, use a "
                         "larger --chunk\n",
                         label.c_str(), 100 * enqueue_share);
          }
          results.push_back(j.dump());
        }
      }
      if (run_direct) {
        // CPU threads load 64 B from GPU src cudaMalloc memory into registers and store them straight into GPU dst
        // cudaMalloc memory; no CPU-memory buffer. cudaMalloc pages cannot migrate, so the data never leaves HBM.
        const std::string label = s + "->" + d + " via cpu" + std::to_string(cpu) + " (direct)";
        if (!direct_problem.empty()) {
          results.push_back(skipped("staged", label, direct_problem));
        } else {
          for (HostKernel kernel : o.kernels) {
            for (const std::string& tspec : o.threads) {
              const size_t n = resolve_threads(tspec, pools.get(cpu).size());
              if (n == 0) {
                results.push_back(skipped("staged", label, "not enough allowed CPUs on cpu" + std::to_string(cpu)));
                continue;
              }
              JsonObject extra;
              extra.str("mode", "direct")
                  .boolean("staged_in_cpu_memory", false)
                  .integer("src_gpu", o.src_gpu)
                  .integer("dst_gpu", dst_gpu)
                  .str("path", s + " cuda -C2C-> cpu" + std::to_string(cpu) + " registers -C2C-> " + d + " cuda")
                  .str("path_class", same_cpu ? "same_cpu" : "cross_cpu")
                  .str("copy_cpu_role", role);
              results.push_back(
                  host_case(topo, pools, cpu, n, HostOp::Copy, kernel, &src, &dst, o.c, "staged", label, extra));
            }
          }
        }
      }
    }
  }
  return results;
}

// ================================================================ latency and ping-pong

// cpu<i>[.k] = k-th allowed CPU of CPU node i (default 0); gpu<g> = a single GPU thread.
struct Agent {
  bool gpu = false;
  int index = -1;
  int core = -1;
  std::string name;
};

inline Agent parse_agent(const std::string& text, const Topology& topo, int default_slot) {
  Agent a;
  a.name = text;
  if (starts_with(text, "gpu")) {
    a.gpu = true;
    a.index = static_cast<int>(parse_int(text.substr(3), "agent " + text));
    topo.gpu(a.index);
    return a;
  }
  if (!starts_with(text, "cpu")) fail("agent must be cpu<i>[.k] or gpu<g>, got '" + text + "'");
  const std::string rest = text.substr(3);
  const size_t dot = rest.find('.');
  a.index = static_cast<int>(parse_int(rest.substr(0, dot), "agent " + text));
  const int slot = dot == std::string::npos ? default_slot : static_cast<int>(parse_int(rest.substr(dot + 1), "agent " + text));
  const CpuNode& node = topo.cpu(a.index);
  if (slot < 0 || slot >= static_cast<int>(node.cpus.size())) {
    fail("agent " + text + ": cpu" + std::to_string(a.index) + " has " + std::to_string(node.cpus.size()) +
         " allowed CPUs");
  }
  a.core = node.cpus[slot];
  a.name = "cpu" + std::to_string(a.index) + "." + std::to_string(slot) + " (core " + std::to_string(a.core) + ")";
  return a;
}

inline void pin_current_thread(int core) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(core, &set);
  if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0) fail("cannot pin thread to core " + std::to_string(core));
}

// ---------------------------------------------------------------- latency

struct LatencyOptions {
  std::vector<std::string> agents;
  std::vector<Target> targets;
  size_t bytes = 0;
  size_t slot = 128;
  size_t steps = 0;  // 0 = one full cycle over the chain, so the working set is the whole buffer
  int iters = 3;
  int warmup = 1;
  HbmBackend backend = HbmBackend::Auto;
};

inline LatencyOptions parse_latency(const Args& a, const Topology& topo) {
  LatencyOptions o;
  o.agents = split(a.get("agents", topo.gpus.empty() ? "cpu0" : "cpu0,gpu0"), ',');
  o.targets = expand_targets(a.get("targets", topo.gpus.empty() ? "cpu*" : "cpu*,hbm*"), topo);
  const Common c = parse_common(a, "512MiB", 3, 1, false);
  o.bytes = c.bytes;
  o.iters = c.iters;
  o.warmup = c.warmup;
  o.backend = c.backend;
  o.slot = parse_size(a.get("slot", "128"));
  if (o.slot < 8 || o.slot % 8 != 0) fail("--slot must be a multiple of 8 bytes");
  const std::string steps = a.get("steps", "all");
  o.steps = steps == "all" ? 0 : parse_size(steps);
  return o;
}

// Random cyclic chain over `slot`-sized slots: pages in random order, slots within a page in random order
// (gh_benchmark's pointer-chase layout, which keeps TLB misses to one per page). Returns the first slot.
inline const uint64_t* build_chain(Buffer& b, size_t slot, size_t page, size_t* chain_length) {
  page = std::max(page, slot);
  const size_t pages = b.bytes() / page;
  const size_t per_page = page / slot;
  if (pages == 0) fail("latency buffer is smaller than one page");
  if (pages * per_page > (size_t(1) << 31)) fail("latency buffer has too many slots; use a larger --slot");
  std::mt19937_64 rng(42);
  std::vector<uint32_t> page_order(pages);
  for (size_t i = 0; i < pages; ++i) page_order[i] = static_cast<uint32_t>(i);
  std::shuffle(page_order.begin(), page_order.end(), rng);
  std::vector<uint32_t> in_page(per_page);
  std::vector<uint32_t> order;
  order.reserve(pages * per_page);
  for (uint32_t p : page_order) {
    for (size_t s = 0; s < per_page; ++s) in_page[s] = static_cast<uint32_t>(s);
    std::shuffle(in_page.begin(), in_page.end(), rng);
    for (uint32_t s : in_page) order.push_back(static_cast<uint32_t>(p * per_page + s));
  }
  const uintptr_t base = reinterpret_cast<uintptr_t>(b.data());
  std::vector<uint8_t> image;
  uint8_t* out = b.data();
  if (!b.host_ok()) {
    image.assign(b.bytes(), 0);
    out = image.data();
  }
  for (size_t k = 0; k < order.size(); ++k) {
    const uint64_t next = base + uint64_t(order[(k + 1) % order.size()]) * slot;
    *reinterpret_cast<uint64_t*>(out + size_t(order[k]) * slot) = next;
  }
  if (!b.host_ok()) GB_CHECK_CUDA(cudaMemcpy(b.data(), image.data(), b.bytes(), cudaMemcpyHostToDevice));
  prefetch_managed(b);
  *chain_length = order.size();
  return reinterpret_cast<const uint64_t*>(base + uint64_t(order[0]) * slot);
}

inline std::vector<std::string> run_latency(const Topology& topo, const LatencyOptions& o) {
  std::vector<std::string> results;
  std::vector<Agent> agents;
  for (const std::string& a : o.agents) agents.push_back(parse_agent(a, topo, 0));
  for (const Target& t : o.targets) {
    std::unique_ptr<Buffer> buf;
    const uint64_t* start = nullptr;
    size_t chain_length = 0;
    for (const Agent& agent : agents) {
      const std::string label = agent.name + " -> " + t.name;
      const std::string problem =
          agent.gpu ? device_access_problem(topo, agent.index, t, o.backend) : host_access_problem(t);
      if (!problem.empty()) {
        results.push_back(skipped("latency", label, problem));
        continue;
      }
      if (!buf) {
        buf = std::make_unique<Buffer>(topo, t, o.bytes, o.backend);
        start = build_chain(*buf, o.slot, static_cast<size_t>(topo.page_size), &chain_length);
        buf->check_placement();
      }
      const size_t steps = o.steps ? o.steps : chain_length;
      std::vector<double> ns_per_load;
      if (agent.gpu) {
        GB_CHECK_CUDA(cudaSetDevice(agent.index));
        HostScalar<unsigned long long> elapsed;
        HostScalar<unsigned long long> sink;
        for (int i = 0; i < o.warmup + o.iters; ++i) {
          dev_chase_kernel<<<1, 1>>>(start, steps, elapsed.get(), sink.get());
          GB_CHECK_CUDA(cudaGetLastError());
          GB_CHECK_CUDA(cudaDeviceSynchronize());
          if (i >= o.warmup) ns_per_load.push_back(static_cast<double>(*elapsed.get()) / steps);
        }
      } else {
        std::thread worker([&] {
          pin_current_thread(agent.core);
          for (int i = 0; i < o.warmup + o.iters; ++i) {
            const uint64_t* p = start;
            const uint64_t t0 = now_ns();
            for (size_t s = 0; s < steps; ++s) p = reinterpret_cast<const uint64_t*>(*reinterpret_cast<const volatile uint64_t*>(p));
            const uint64_t t1 = now_ns();
            g_sink = reinterpret_cast<uintptr_t>(p);
            if (i >= o.warmup) ns_per_load.push_back(static_cast<double>(t1 - t0) / steps);
          }
        });
        worker.join();
      }
      const Stats s = summarize(ns_per_load);
      std::printf("[latency] %s: median %.1f ns/load (min %.1f, max %.1f)\n", label.c_str(), s.median, s.min, s.max);
      std::fflush(stdout);
      results.push_back(JsonObject()
                            .str("benchmark", "latency")
                            .str("label", label)
                            .str("agent", agent.name)
                            .raw("target", buf->to_json())
                            .integer("slot_bytes", static_cast<long long>(o.slot))
                            .integer("steps", static_cast<long long>(steps))
                            .integer("chain_slots", static_cast<long long>(chain_length))
                            .integer("iters", o.iters)
                            .integer("warmup", o.warmup)
                            .raw("ns_per_load_samples", json_numbers(ns_per_load))
                            .raw("ns_per_load", json_stats(s))
                            .dump());
    }
  }
  return results;
}

// ---------------------------------------------------------------- ping-pong

struct PingPongOptions {
  std::vector<std::pair<std::string, std::string>> pairs;
  std::vector<Target> flags;
  size_t rounds = 10000;
  size_t warmup_rounds = 1000;
  int iters = 5;
  double timeout_s = 30;
  HbmBackend backend = HbmBackend::Auto;
};

inline PingPongOptions parse_pingpong(const Args& a, const Topology& topo) {
  PingPongOptions o;
  std::string default_pairs = "cpu0:cpu0.1";
  if (topo.cpus.size() > 1) default_pairs += ",cpu0:cpu1";
  if (!topo.gpus.empty()) default_pairs += ",cpu0:gpu0,gpu0:cpu0";
  if (topo.gpus.size() > 1) default_pairs += ",gpu0:gpu1";
  if (topo.gpus.size() > 2) default_pairs += ",gpu0:gpu2";
  for (const std::string& item : split(a.get("pairs", default_pairs), ',')) {
    const size_t colon = item.find(':');
    if (colon == std::string::npos) fail("--pairs items must be <ping>:<pong>, got '" + item + "'");
    o.pairs.emplace_back(item.substr(0, colon), item.substr(colon + 1));
  }
  o.flags = expand_targets(a.get("flags", topo.gpus.empty() ? "cpu0" : "cpu0,hbm0"), topo);
  o.rounds = parse_size(a.get("rounds", "10000"));
  o.warmup_rounds = static_cast<size_t>(parse_int(a.get("warmup-rounds", "1000"), "--warmup-rounds"));
  o.iters = a.get_int("iters", 5);
  if (o.iters < 1) fail("--iters must be >= 1");
  o.timeout_s = std::stod(a.get("timeout-s", "30"));
  o.backend = parse_hbm_backend(a.get("hbm-backend", "auto"));
  return o;
}

inline bool wait_flag(uint32_t* flag, uint32_t value, uint64_t deadline) {
  for (uint64_t spins = 0;; ++spins) {
    if (__atomic_load_n(flag, __ATOMIC_ACQUIRE) == value) return true;
    if ((spins & 4095) == 0 && now_ns() > deadline) return false;
    cpu_relax();
  }
}

// Same protocol as dev_pingpong_kernel. Returns elapsed ns of the measured rounds (ping side only).
inline uint64_t cpu_pingpong(uint32_t* flag, bool ping, size_t warmup, size_t rounds, uint64_t deadline, bool* timed_out) {
  uint64_t t0 = now_ns();
  for (size_t i = 0; i < warmup + rounds; ++i) {
    if (i == warmup) t0 = now_ns();
    const uint32_t base = static_cast<uint32_t>(2 * i);
    if (ping) {
      __atomic_store_n(flag, base + 1, __ATOMIC_RELEASE);
      if (!wait_flag(flag, base + 2, deadline)) {
        *timed_out = true;
        return 0;
      }
    } else {
      if (!wait_flag(flag, base + 1, deadline)) {
        *timed_out = true;
        return 0;
      }
      __atomic_store_n(flag, base + 2, __ATOMIC_RELEASE);
    }
  }
  return ping ? now_ns() - t0 : 0;
}

inline std::vector<std::string> run_pingpong(const Topology& topo, const PingPongOptions& o) {
  std::vector<std::string> results;
  if (2 * (o.rounds + o.warmup_rounds) >= (size_t(1) << 32)) fail("too many rounds for a 32-bit flag");
  for (const Target& ft : o.flags) {
    Buffer flag_buf(topo, ft, static_cast<size_t>(topo.page_size), o.backend);
    auto* flag = reinterpret_cast<uint32_t*>(flag_buf.data());
    auto reset_flag = [&] {
      if (flag_buf.host_ok()) {
        __atomic_store_n(flag, 0u, __ATOMIC_SEQ_CST);
      } else {
        GB_CHECK_CUDA(cudaSetDevice(flag_buf.device()));
        GB_CHECK_CUDA(cudaMemset(flag, 0, sizeof(uint32_t)));
        GB_CHECK_CUDA(cudaDeviceSynchronize());
      }
    };
    if (flag_buf.host_ok()) std::memset(flag_buf.data(), 0, flag_buf.bytes());  // first touch
    reset_flag();
    prefetch_managed(flag_buf);
    flag_buf.check_placement();

    for (const auto& pr : o.pairs) {
      const Agent ping = parse_agent(pr.first, topo, 0);
      const bool same_node = !ping.gpu && starts_with(pr.second, "cpu") && pr.second.find('.') == std::string::npos &&
                             parse_agent(pr.second, topo, 0).index == ping.index;
      const Agent pong = parse_agent(pr.second, topo, same_node ? 1 : 0);
      const std::string label = ping.name + " <-> " + pong.name + ", flag in " + ft.name;
      std::string problem;
      for (const Agent* a : {&ping, &pong}) {
        if (problem.empty()) problem = a->gpu ? device_access_problem(topo, a->index, ft, o.backend) : host_access_problem(ft);
      }
      if (problem.empty() && !ping.gpu && !pong.gpu && ping.core == pong.core) problem = "ping and pong use the same core";
      if (problem.empty() && ping.gpu && pong.gpu && ping.index == pong.index) problem = "ping and pong use the same GPU";
      if (!problem.empty()) {
        results.push_back(skipped("pingpong", label, problem));
        continue;
      }

      std::vector<double> rtt_ns;
      bool timed_out = false;
      for (int it = 0; it < o.iters && !timed_out; ++it) {
        reset_flag();
        const uint64_t timeout_ns = static_cast<uint64_t>(o.timeout_s * 1e9);
        const uint64_t deadline = now_ns() + timeout_ns;
        // Result slots for GPU agents; only allocated when a GPU takes part (CPU-only pairs need no CUDA).
        std::unique_ptr<HostScalar<unsigned long long>> gpu_elapsed;
        std::unique_ptr<HostScalar<int>> gpu_timeout[2];
        uint64_t cpu_elapsed = 0;
        bool cpu_timeout[2] = {false, false};
        std::vector<int> launched;
        std::vector<std::thread> threads;
        const Agent* agents[2] = {&ping, &pong};
        for (int r = 0; r < 2; ++r) {  // GPU agents first, so the CPU side does not burn its timeout on launch
          if (!agents[r]->gpu) continue;
          GB_CHECK_CUDA(cudaSetDevice(agents[r]->index));
          if (!gpu_elapsed) gpu_elapsed = std::make_unique<HostScalar<unsigned long long>>();
          gpu_timeout[r] = std::make_unique<HostScalar<int>>();
          dev_pingpong_kernel<<<1, 1>>>(flag, r == 0, o.warmup_rounds, o.rounds, timeout_ns, gpu_elapsed->get(),
                                        gpu_timeout[r]->get());
          GB_CHECK_CUDA(cudaGetLastError());
          launched.push_back(agents[r]->index);
        }
        for (int r = 0; r < 2; ++r) {
          if (agents[r]->gpu) continue;
          threads.emplace_back([&, r] {
            pin_current_thread(agents[r]->core);
            const uint64_t e = cpu_pingpong(flag, r == 0, o.warmup_rounds, o.rounds, deadline, &cpu_timeout[r]);
            if (r == 0) cpu_elapsed = e;
          });
        }
        for (std::thread& t : threads) t.join();
        for (int d : launched) {
          GB_CHECK_CUDA(cudaSetDevice(d));
          GB_CHECK_CUDA(cudaDeviceSynchronize());
        }
        timed_out = cpu_timeout[0] || cpu_timeout[1];
        for (const auto& t : gpu_timeout) timed_out = timed_out || (t && *t->get());
        if (!timed_out) {
          const double elapsed = ping.gpu ? static_cast<double>(*gpu_elapsed->get()) : static_cast<double>(cpu_elapsed);
          rtt_ns.push_back(elapsed / o.rounds);
        }
      }
      const Stats s = summarize(rtt_ns);
      if (timed_out) {
        std::printf("[pingpong] %s: TIMEOUT after %.0f s (flag not visible to the peer?)\n", label.c_str(), o.timeout_s);
      } else {
        std::printf("[pingpong] %s: median round trip %.1f ns (one way ~%.1f ns)\n", label.c_str(), s.median,
                    s.median / 2);
      }
      std::fflush(stdout);
      results.push_back(JsonObject()
                            .str("benchmark", "pingpong")
                            .str("label", label)
                            .str("ping", ping.name)
                            .str("pong", pong.name)
                            .raw("flag", flag_buf.to_json())
                            .integer("rounds", static_cast<long long>(o.rounds))
                            .integer("warmup_rounds", static_cast<long long>(o.warmup_rounds))
                            .integer("iters", o.iters)
                            .boolean("timed_out", timed_out)
                            .raw("round_trip_ns_samples", json_numbers(rtt_ns))
                            .raw("round_trip_ns", json_stats(s))
                            .dump());
    }
  }
  return results;
}

}  // namespace gb
