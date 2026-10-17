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

// All-gather / reduce-scatter throughput on GB200 with NCCL 2.32.3, without torch or MPI.
//
// One process per GPU, started by run_nvls_collective_benchmark.sh with RANK / WORLD_SIZE / LOCAL_RANK
// and NCCL_COMM_ID=<master>:<port>: every rank calls ncclGetUniqueId (which returns the id built
// from NCCL_COMM_ID) and world rank 0's ncclCommInitRank starts the bootstrap root there. All
// scenario communicators are ncclCommSplit children of that world communicator; the world
// communicator itself only carries barriers (8-byte AllReduce) and result gathering (AllGather).
//
// One launch measures one --impl (see bench_core.h impls()). Each collective call carries an
// ncclCollConfig_t: algSelection pins the kernel (RING, NVLS_SIMPLE, SYMK_STMC, SYMK_TmaSTMC,
// SYMK_LDMC) and CTAPolicy=NCCL_CTA_POLICY_ZERO sends AllGather to the copy engines. With
// forceAlgSelection=1 an unavailable kernel is an NCCL error, and every launch is also checked
// against NCCL's own TUNING log line, so a result row always names the kernel that produced it.
//
// Scenarios (same as ../nvls-collective; all inside one launch):
//   single: ranks [0, n) for every n in --group-sizes run AG alone, RS alone and AG+RS at the
//           same time on two streams and two communicators.
//   split:  ranks [0, T) for every T in --split-totals are cut into two halves; both halves run
//           the same workload at the same time, and (cross) group A all-gathers while group B
//           reduce-scatters. Neither half runs alone: its reference is single-n<T/2>, so T/2
//           must be one of --group-sizes.
// Every timed trial opens with a start gate: a one-element all-reduce over all ranks of the
// scenario on its own communicator, so every group's clock starts at the same moment, which
// concurrency = min(T_A, T_B) / max(T_A, T_B) relies on.
//
// Output (same layout as ../nvls-collective): <output-dir>/results.jsonl and metadata.json from
// world rank 0, comm/<impl>-<scenario>.rank_<RR>.nccl.log, and per test cases/<impl>-<case>/ with
// case.json and every member rank's <impl>-<case>.rank_<RR>.nccl.log; with --nsys also
// nsys/index.tsv.

#include <cuda_bf16.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device/host.h>  // ncclCommQueryProperties
#include <nvtx3/nvToolsExt.h>
#include <unistd.h>

#include <chrono>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "bench_core.h"

using namespace nvls_bench;
namespace fs = std::filesystem;

extern char** environ;

static int gRank = -1;

#define CUDA_CHECK(cmd)                                                                                         \
  do {                                                                                                          \
    cudaError_t e_ = (cmd);                                                                                     \
    if (e_ != cudaSuccess) {                                                                                    \
      std::fprintf(stderr, "[rank %d] %s:%d %s: %s\n", gRank, __FILE__, __LINE__, #cmd, cudaGetErrorString(e_)); \
      std::exit(1);                                                                                             \
    }                                                                                                           \
  } while (0)

#define NCCL_CHECK(cmd)                                                                                   \
  do {                                                                                                    \
    ncclResult_t r_ = (cmd);                                                                              \
    if (r_ != ncclSuccess) {                                                                              \
      std::fprintf(stderr, "[rank %d] %s:%d %s: %s (%s)\n", gRank, __FILE__, __LINE__, #cmd,               \
                   ncclGetErrorString(r_), ncclGetLastError(nullptr));                                    \
      std::exit(1);                                                                                       \
    }                                                                                                     \
  } while (0)

[[noreturn]] static void die(const std::string& message) {
  std::fprintf(stderr, "[rank %d] ERROR: %s\n", gRank, message.c_str());
  std::exit(2);
}

static std::string envOr(const char* name, const std::string& fallback = "") {
  const char* value = std::getenv(name);
  return value ? value : fallback;
}

static double unixNow() {
  return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

////////////////////////////////////////////////////////////////////////////////
// Device helpers.

// Spin on the global timer so the gate and every launch of a trial are queued before they run.
__global__ void spinKernel(unsigned long long ns) {
  unsigned long long start, now;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
  do {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
  } while (now - start < ns);
}

template <typename T>
__device__ __forceinline__ float toFloat(T v);
template <>
__device__ __forceinline__ float toFloat<float>(float v) { return v; }
template <>
__device__ __forceinline__ float toFloat<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T>
__device__ __forceinline__ T fromFloat(float v);
template <>
__device__ __forceinline__ float fromFloat<float>(float v) { return v; }
template <>
__device__ __forceinline__ __nv_bfloat16 fromFloat<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

template <typename T>
__global__ void fillKernel(T* data, size_t n, float value) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    data[i] = fromFloat<T>(value);
  }
}

// shard > 0: AG output, element i must be i / shard + 1. shard == 0: every element is `expected`.
template <typename T>
__global__ void mismatchKernel(const T* data, size_t n, size_t shard, float expected, unsigned long long* bad) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    float want = shard ? float(i / shard + 1) : expected;
    if (toFloat<T>(data[i]) != want) atomicAdd(bad, 1ULL);
  }
}

struct Dtype {
  std::string name;
  ncclDataType_t nccl;
  size_t size;
};

static Dtype makeDtype(const std::string& name) {
  if (name == "bf16") return {name, ncclBfloat16, 2};
  if (name == "fp32") return {name, ncclFloat32, 4};
  die("--dtype accepts bf16 or fp32, got " + name);
}

static void fill(const Dtype& dt, void* ptr, size_t n, float value, cudaStream_t stream) {
  if (dt.nccl == ncclBfloat16) {
    fillKernel<<<1024, 256, 0, stream>>>(static_cast<__nv_bfloat16*>(ptr), n, value);
  } else {
    fillKernel<<<1024, 256, 0, stream>>>(static_cast<float*>(ptr), n, value);
  }
  CUDA_CHECK(cudaGetLastError());
}

static unsigned long long countMismatches(const Dtype& dt, const void* ptr, size_t n, size_t shard, float expected,
                                          cudaStream_t stream) {
  unsigned long long* bad;
  unsigned long long host = 0;
  CUDA_CHECK(cudaMalloc(&bad, sizeof(*bad)));
  CUDA_CHECK(cudaMemsetAsync(bad, 0, sizeof(*bad), stream));
  if (dt.nccl == ncclBfloat16) {
    mismatchKernel<<<1024, 256, 0, stream>>>(static_cast<const __nv_bfloat16*>(ptr), n, shard, expected, bad);
  } else {
    mismatchKernel<<<1024, 256, 0, stream>>>(static_cast<const float*>(ptr), n, shard, expected, bad);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpyAsync(&host, bad, sizeof(host), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaFree(bad));
  return host;
}

////////////////////////////////////////////////////////////////////////////////
// Options.

struct Options {
  const Impl* impl = nullptr;
  std::string outputDir;
  std::vector<int> groupSizes = {4, 8, 16, 32};
  std::vector<int> splitTotals = {16};
  std::vector<std::string> splitLayouts = {"contiguous"};
  std::vector<std::string> workloads = nvls_bench::workloads();
  std::vector<int> sizesMiB = {16, 32, 64, 128, 256, 512, 1024, 2048};
  std::string dtype = "bf16";
  int warmup = 5;
  int iters = 20;
  int trials = 5;
  double launchDelayMs = 5.0;
  bool nsys = false;
};

constexpr int kMaxTrials = 64;

static void usage() {
  std::printf(
    "Usage: nvls_collective_bench --impl IMPL --output-dir DIR [options]\n"
    "  --impl ring|nvls|symm|stmc|tma|ce|ce-uc\n"
    "  --group-sizes LIST      single-group sizes (default 4,8,16,32)\n"
    "  --split-totals LIST     ranks shared by two concurrent halves, or none (default 16);\n"
    "                          half (T/2) must be in --group-sizes\n"
    "  --split-layouts LIST    contiguous,interleaved (default contiguous)\n"
    "  --workloads LIST        ag,rs,ag+rs (default all); with ag and rs, split also runs the cross case\n"
    "  --sizes-mib LIST        full tensor = AG output / RS input (default 16,32,64,128,256,512,1024,2048)\n"
    "  --dtype bf16|fp32       (default bf16)\n"
    "  --warmup N --iters N --trials N --launch-delay-ms MS   (default 5 20 5 5)\n"
    "  --nsys                  one cudaProfilerStart/Stop window + NVTX range per case\n"
    "Environment: RANK, WORLD_SIZE, LOCAL_RANK, NCCL_COMM_ID (set by run_nvls_collective_benchmark.sh).\n");
}

static Options parseOptions(int argc, char** argv) {
  Options o;
  auto value = [&](int& i) -> std::string {
    if (i + 1 >= argc) die(std::string(argv[i]) + " requires a value");
    return argv[++i];
  };
  try {
    for (int i = 1; i < argc; i++) {
      std::string arg = argv[i];
      if (arg == "--impl") {
        std::string name = value(i);
        o.impl = findImpl(name);
        if (!o.impl) die("unknown --impl " + name);
      } else if (arg == "--output-dir") {
        o.outputDir = value(i);
      } else if (arg == "--group-sizes") {
        o.groupSizes = parsePositiveInts(value(i));
      } else if (arg == "--split-totals") {
        std::string v = value(i);
        o.splitTotals = v == "none" ? std::vector<int>{} : parsePositiveInts(v);
      } else if (arg == "--split-layouts") {
        o.splitLayouts = parseCsv(value(i));
      } else if (arg == "--workloads") {
        o.workloads = parseCsv(value(i));
      } else if (arg == "--sizes-mib") {
        o.sizesMiB = parsePositiveInts(value(i));
      } else if (arg == "--dtype") {
        o.dtype = value(i);
      } else if (arg == "--warmup") {
        o.warmup = std::stoi(value(i));
      } else if (arg == "--iters") {
        o.iters = std::stoi(value(i));
      } else if (arg == "--trials") {
        o.trials = std::stoi(value(i));
      } else if (arg == "--launch-delay-ms") {
        o.launchDelayMs = std::stod(value(i));
      } else if (arg == "--nsys") {
        o.nsys = true;
      } else if (arg == "-h" || arg == "--help") {
        usage();
        std::exit(0);
      } else {
        die("unknown option " + arg);
      }
    }
    for (const std::string& w : o.workloads) workloadOps(w);
    for (const std::string& layout : o.splitLayouts) splitGroups(4, layout);
  } catch (const std::exception& e) {
    die(e.what());
  }
  if (!o.impl || o.outputDir.empty()) die("--impl and --output-dir are required (see --help)");
  // The first warmup launch of every op runs alone so lazy NCCL setup never overlaps another
  // communicator's kernel; hence at least one warmup.
  if (o.warmup < 1 || o.iters < 1 || o.trials < 1) die("--warmup, --iters and --trials must be >= 1");
  if (o.trials > kMaxTrials) die("--trials must be <= " + std::to_string(kMaxTrials));
  if (o.launchDelayMs < 0) die("--launch-delay-ms must be >= 0");
  return o;
}

static bool hasCross(const Options& o) {
  for (const std::string& w : crossGroupWorkloads()) {
    if (std::find(o.workloads.begin(), o.workloads.end(), w) == o.workloads.end()) return false;
  }
  return true;
}

////////////////////////////////////////////////////////////////////////////////
// Environment checks: the per-call config only means what it says if nothing global overrides it.

static void checkEnvironment(const Options& o, int worldSize) {
  const Impl& impl = *o.impl;
  std::string name = impl.name;
  // NCCL_ALGO / NCCL_SYM_KERNEL narrow the kernel set below algSelection, and an NCCL_CTA_POLICY
  // override makes the comm policy authoritative over the per-call CTAPolicy (collconfig.cc).
  for (const char* var : {"NCCL_ALGO", "NCCL_PROTO", "NCCL_SYM_KERNEL", "NCCL_CTA_POLICY"}) {
    if (!envOr(var).empty()) die(std::string(var) + " must be unset: the kernel is chosen per call");
  }
  std::string nvls = envOr("NCCL_NVLS_ENABLE");
  if (name == "ring" ? nvls != "0" : nvls != "1") {
    die("--impl " + name + " requires NCCL_NVLS_ENABLE=" + (name == "ring" ? "0" : "1") + ", got '" + nvls + "'");
  }
  if (impl.windows) {
    if (envOr("NCCL_WIN_ENABLE", "1") == "0") die("NCCL_WIN_ENABLE=0 disables symmetric windows");
    if (envOr("NCCL_CUMEM_ENABLE") == "0") die("NCCL_CUMEM_ENABLE=0 disables symmetric support (init.cc)");
    if (envOr("NCCL_SYM_TMA_ENABLE", "1") == "0") die("NCCL_SYM_TMA_ENABLE=0 removes the TMA kernels");
  }
  size_t largest = 0, largestShard = 0;
  for (int mib : o.sizesMiB) {
    size_t bytes = size_t(mib) * kBytesPerMiB;
    largest = std::max(largest, bytes);
    for (int n : o.groupSizes) largestShard = std::max(largestShard, bytes / n);
    for (int t : o.splitTotals) largestShard = std::max(largestShard, bytes / (t / 2));
  }
  auto digits = [](const std::string& s) { return !s.empty() && s.find_first_not_of("0123456789") == std::string::npos; };
  if (name == "symm") {
    // Without a selection, enqueue.cc sends an AllGather with count > NCCL_SYM_CE_THRESHOLD
    // (elements) to the copy engines; a byte bound at least as large as the biggest payload
    // keeps every AllGather on the symmetric kernels.  Mirror run_nvls_collective_benchmark.sh:
    // set INT64_MAX automatically so the caller never has to remember this variable.  An
    // explicit value is kept if it is already large enough; a missing or insufficient one is
    // replaced with a warning.
    std::string threshold = envOr("NCCL_SYM_CE_THRESHOLD");
    if (!digits(threshold) || std::stoull(threshold) < largest) {
      std::fprintf(stderr, "[rank %d] WARNING: NCCL_SYM_CE_THRESHOLD='%s' is not >= %llu; setting it"
                          " to INT64_MAX so every AllGather stays on the symmetric kernels\n",
                   gRank, threshold.c_str(), (unsigned long long)largest);
      setenv("NCCL_SYM_CE_THRESHOLD", "18446744073709551615", 1);
    }
  }
  if (impl.ceMulticast) {
    // ncclCeAllGatherUseMulticast: threshold >= 0 bypasses the cost model; MC iff shard <= threshold.
    std::string threshold = envOr("NCCL_CE_COLL_AG_MULTICAST_THRESHOLD");
    bool mc = std::string(impl.ceMulticast) == "mc";
    bool ok = digits(threshold) && (mc ? std::stoull(threshold) >= largestShard : threshold == "0");
    if (!ok) {
      die("--impl " + name + " requires NCCL_CE_COLL_AG_MULTICAST_THRESHOLD " +
          (mc ? ">= " + std::to_string(largestShard) : std::string("= 0")) + ", got '" + threshold + "'");
    }
  }
  std::string level = envOr("NCCL_DEBUG");
  std::string subsys = envOr("NCCL_DEBUG_SUBSYS");
  if (level != "INFO" && level != "TRACE") die("NCCL_DEBUG=INFO is required: the executed kernel is read from NCCL's log");
  if (subsys.find("TUNING") == std::string::npos && subsys.find("ALL") == std::string::npos) {
    die("NCCL_DEBUG_SUBSYS must include TUNING: it carries the per-launch algorithm/kernel line");
  }
  if (envOr("NCCL_DEBUG_FILE").find("%p") == std::string::npos) {
    die("NCCL_DEBUG_FILE must be set and contain %p so every process reads only its own log");
  }
  if (envOr("NCCL_COMM_ID").empty()) die("NCCL_COMM_ID=<host>:<port> is required for the bootstrap");
  std::vector<int> all = o.groupSizes;
  all.insert(all.end(), o.splitTotals.begin(), o.splitTotals.end());
  for (int n : all) {
    if (n > worldSize) die("group size " + std::to_string(n) + " exceeds WORLD_SIZE=" + std::to_string(worldSize));
  }
  for (int n : o.groupSizes) {
    if (n < 2) die("--group-sizes entries must be >= 2");
  }
  for (int t : o.splitTotals) {
    if (t % 2 || t < 4) die("--split-totals must be even and >= 4");
    // A split half is compared with the single group of the same size, so that size must be
    // measured; this is the same rule as ../nvls-collective.
    if (std::find(o.groupSizes.begin(), o.groupSizes.end(), t / 2) == o.groupSizes.end()) {
      die("--split-totals " + std::to_string(t) + " needs its half (" + std::to_string(t / 2) +
          ") in --group-sizes, which provides the single-group reference");
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// NCCL log reader: this process's NCCL_DEBUG_FILE, read incrementally so every case sees only
// the lines written since the previous call. NCCL line-buffers the file (debug.cc setlinebuf).

class NcclLog {
 public:
  explicit NcclLog(std::string path) : path_(std::move(path)) {}

  std::vector<std::string> consume() {
    std::ifstream in(path_, std::ios::binary);
    if (!in) return {};
    in.seekg(0, std::ios::end);
    std::streamoff end = in.tellg();
    if (end <= offset_) return {};
    in.seekg(offset_);
    std::string data(size_t(end - offset_), '\0');
    in.read(&data[0], data.size());
    size_t cut = data.rfind('\n');
    if (cut == std::string::npos) return {};
    offset_ += std::streamoff(cut + 1);
    return splitString(data.substr(0, cut), '\n');
  }

 private:
  std::string path_;
  std::streamoff offset_ = 0;
};

static void appendLines(const fs::path& path, const std::vector<std::string>& lines) {
  fs::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::app);
  for (const std::string& line : lines) out << line << '\n';
}

////////////////////////////////////////////////////////////////////////////////
// World communicator: barrier and gather.

struct World {
  ncclComm_t comm = nullptr;
  int rank = 0;
  int size = 0;
  cudaStream_t stream = nullptr;
  int* barrierBuf = nullptr;

  // kWorldBarrierCount int32 = 8 Bytes, so its TUNING line never looks like the 4-Byte gate.
  void barrier() {
    NCCL_CHECK(ncclAllReduce(barrierBuf, barrierBuf, kWorldBarrierCount, ncclInt32, ncclSum, comm, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  // Every rank receives every rank's `bytes` bytes, in rank order.
  std::vector<char> allGather(const void* local, size_t bytes) {
    void* dev;
    std::vector<char> out(bytes * size);
    CUDA_CHECK(cudaMalloc(&dev, bytes * (size + 1)));
    char* send = static_cast<char*>(dev) + bytes * size;
    CUDA_CHECK(cudaMemcpyAsync(send, local, bytes, cudaMemcpyHostToDevice, stream));
    NCCL_CHECK(ncclAllGather(send, dev, bytes, ncclChar, comm, stream));
    CUDA_CHECK(cudaMemcpyAsync(out.data(), dev, bytes * size, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(dev));
    return out;
  }

  // Minimum of one flag over all ranks (kWorldBarrierCount wide, like the barrier).
  int allMin(int value) {
    int* dev;
    int host[kWorldBarrierCount] = {value, value};
    CUDA_CHECK(cudaMalloc(&dev, sizeof(host)));
    CUDA_CHECK(cudaMemcpyAsync(dev, host, sizeof(host), cudaMemcpyHostToDevice, stream));
    NCCL_CHECK(ncclAllReduce(dev, dev, kWorldBarrierCount, ncclInt32, ncclMin, comm, stream));
    CUDA_CHECK(cudaMemcpyAsync(host, dev, sizeof(host), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(dev));
    return host[0];
  }
};

////////////////////////////////////////////////////////////////////////////////
// Scenarios: groups with a dedicated communicator per collective so AG and RS can overlap,
// plus the start gate over every rank of the scenario.

struct Group {
  std::string name;
  std::vector<int> ranks;
  bool member = false;
  int commRank = -1;
  ncclComm_t ag = nullptr;
  ncclComm_t rs = nullptr;
  ncclComm_t ar = nullptr;  // AllReduce communicator; shared with ag/rs for classic impls
  CommLogInfo logInfo;
  int multimemSupport = -1;
  int nLsaTeams = -1;
  int deviceApiSupport = -1;

  int size() const { return int(ranks.size()); }
};

struct Scenario {
  std::string kind;   // single / split
  std::string name;   // single-n8 / split-n16-contiguous
  int totalRanks = 0;
  std::string layout;
  std::vector<Group> groups;
  std::vector<int> gateRanks;  // sorted union of the groups
  ncclComm_t gate = nullptr;   // null on ranks outside the scenario
  float* gateBuf = nullptr;
  fs::path commLog;            // comm/<impl>-<scenario>.rank_<RR>.nccl.log on member ranks

  bool gateRank0(int rank) const { return !gateRanks.empty() && gateRanks.front() == rank; }
};

// Every world rank calls ncclCommSplit for the gate, AG and RS communicators in the same order.
// key = world rank keeps the order, so a group's first rank is comm rank 0 of both of its
// communicators and the scenario's first rank is comm rank 0 of the gate: the ranks NCCL writes
// TUNING lines on.
static Scenario createScenario(World& world, const Options& o, NcclLog& log, cudaStream_t stream,
                               const std::string& kind, int totalRanks, const std::string& layout,
                               const std::vector<std::pair<std::string, std::vector<int>>>& sets) {
  Scenario s;
  s.kind = kind;
  s.name = scenarioName(kind, totalRanks, layout);
  s.totalRanks = totalRanks;
  s.layout = layout;
  int color = NCCL_SPLIT_NOCOLOR;
  for (size_t g = 0; g < sets.size(); g++) {
    Group group;
    group.name = sets[g].first;
    group.ranks = sets[g].second;
    auto it = std::find(group.ranks.begin(), group.ranks.end(), world.rank);
    if (it != group.ranks.end()) {
      group.member = true;
      group.commRank = int(it - group.ranks.begin());
      color = int(g);
    }
    s.gateRanks.insert(s.gateRanks.end(), group.ranks.begin(), group.ranks.end());
    s.groups.push_back(group);
  }
  std::sort(s.gateRanks.begin(), s.gateRanks.end());
  bool inScenario = color != NCCL_SPLIT_NOCOLOR;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  NCCL_CHECK(ncclCommSplit(world.comm, inScenario ? 0 : NCCL_SPLIT_NOCOLOR, world.rank, &s.gate, &config));
  ncclComm_t comms[3];
  for (ncclComm_t& comm : comms) {
    ncclConfig_t groupConfig = NCCL_CONFIG_INITIALIZER;
    NCCL_CHECK(ncclCommSplit(world.comm, color, world.rank, &comm, &groupConfig));
  }
  for (Group& group : s.groups) {
    if (group.member) {
      group.ag = comms[0];
      group.rs = comms[1];
      group.ar = comms[2];
    }
  }

  // Create every channel serially before any concurrent launch: NCCL connects ring/NVLS channels
  // on the first collective, and doing that while another communicator's kernel waits for peers
  // can deadlock. The gate goes first. Plain buffers and automatic selection, so nothing here
  // depends on the impl.
  if (inScenario) {
    CUDA_CHECK(cudaMalloc(&s.gateBuf, sizeof(float)));
    CUDA_CHECK(cudaMemset(s.gateBuf, 0, sizeof(float)));
    NCCL_CHECK(ncclAllReduce(s.gateBuf, s.gateBuf, 1, ncclFloat32, ncclSum, s.gate, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  world.barrier();
  for (Group& group : s.groups) {
    if (group.member) {
      size_t n = group.ranks.size(), count = 1024;
      float *shard, *full, *buf;
      CUDA_CHECK(cudaMalloc(&shard, count * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&full, n * count * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&buf, count * sizeof(float)));
      NCCL_CHECK(ncclAllGather(shard, full, count, ncclFloat32, group.ag, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      NCCL_CHECK(ncclReduceScatter(full, shard, count, ncclFloat32, ncclSum, group.rs, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      // Warm up the AR communicator with a small in-place all-reduce.
      NCCL_CHECK(ncclAllReduce(buf, buf, count, ncclFloat32, ncclSum, group.ar, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CUDA_CHECK(cudaFree(shard));
      CUDA_CHECK(cudaFree(full));
      CUDA_CHECK(cudaFree(buf));
      ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
      NCCL_CHECK(ncclCommQueryProperties(group.ag, &props));
      group.multimemSupport = props.multimemSupport;
      group.nLsaTeams = props.nLsaTeams;
      group.deviceApiSupport = props.deviceApiSupport;
    }
    world.barrier();
  }
  // Split and connect lines: nNodes / MNNVL / symmetric multimem of this rank's communicators.
  std::vector<std::string> lines = log.consume();
  for (const std::string& line : lines) {
    for (Group& group : s.groups) {
      if (group.member) parseCommLine(line, &group.logInfo);
    }
  }
  if (inScenario) {
    s.commLog = fs::path(o.outputDir) / "comm" / rankFile(o.impl->name, s.name, world.rank, world.size, ".nccl.log");
    appendLines(s.commLog, lines);
  }

  // Symmetric impls only run on a communicator where NCCL's LSA kernels exist: one LSA team
  // (symmetric kernels and CE need every rank load/store reachable) and multimem.
  int ok = 1;
  for (const Group& group : s.groups) {
    if (!group.member) continue;
    std::string why;
    if (o.impl->windows && !group.deviceApiSupport) why = "no symmetric (device API) support";
    if (o.impl->needsMultimem && group.nLsaTeams != 1) why = "nLsaTeams=" + std::to_string(group.nLsaTeams) + " (needs 1)";
    if (o.impl->needsMultimem && !group.multimemSupport) why = "no multimem support (NVLS off or cross-clique P2P)";
    if (!why.empty()) {
      std::fprintf(stderr, "[rank %d] %s cannot run --impl %s: %s\n", world.rank, group.name.c_str(), o.impl->name,
                   why.c_str());
      ok = 0;
    }
  }
  if (!world.allMin(ok)) die("--impl " + std::string(o.impl->name) + " is not available on every group of " + s.name);
  return s;
}

// Release communicators (and NVLS multicast memory) before the next scenario.
static void destroyScenario(World& world, Scenario& s, NcclLog& log) {
  world.barrier();
  for (Group& group : s.groups) {
    if (group.member) {
      NCCL_CHECK(ncclCommDestroy(group.ag));
      NCCL_CHECK(ncclCommDestroy(group.rs));
      NCCL_CHECK(ncclCommDestroy(group.ar));
    }
  }
  if (s.gate) {
    NCCL_CHECK(ncclCommDestroy(s.gate));
    CUDA_CHECK(cudaFree(s.gateBuf));
  }
  world.barrier();
  std::vector<std::string> lines = log.consume();
  if (!s.commLog.empty()) appendLines(s.commLog, lines);
}

////////////////////////////////////////////////////////////////////////////////
// One case.

struct Buffers {
  void* agIn = nullptr;
  void* agOut = nullptr;
  void* rsIn = nullptr;
  void* rsOut = nullptr;
  void* arIn = nullptr;   // AllReduce in-place buffer (in == out for AR)
  ncclWindow_t wins[5] = {};
};

struct Streams {
  cudaStream_t main = nullptr;
  cudaStream_t ag = nullptr;
  cudaStream_t rs = nullptr;
  cudaStream_t ar = nullptr;
  cudaEvent_t start = nullptr, end = nullptr, agEnd = nullptr, rsEnd = nullptr, arEnd = nullptr;
};

// Fixed-size record every rank contributes to the world gather after each case.
struct RankResult {
  int32_t rank;
  int32_t groupIndex;   // -1: this rank sat the case out
  int32_t passed;
  int32_t implChecked;  // 1 on comm rank 0 of the group, which owns the TUNING lines
  int32_t nNodes, mnnvl, symmetricMultimem, multimemSupport, nLsaTeams;
  float makespan[kMaxTrials], agMs[kMaxTrials], rsMs[kMaxTrials], arMs[kMaxTrials], enqueueMs[kMaxTrials];
  char observed[256];
  char errors[1024];
};

static void* allocPayload(const Impl& impl, size_t bytes) {
  void* ptr;
  if (impl.windows) {
    // ncclMemAlloc gives one cuMem allocation per buffer, which is what window registration
    // shares with peers and binds to multicast.
    NCCL_CHECK(ncclMemAlloc(&ptr, bytes));
  } else {
    CUDA_CHECK(cudaMalloc(&ptr, bytes));
  }
  return ptr;
}

static void freePayload(const Impl& impl, void* ptr) {
  if (!ptr) return;
  if (impl.windows) {
    NCCL_CHECK(ncclMemFree(ptr));
  } else {
    CUDA_CHECK(cudaFree(ptr));
  }
}

static bool hasOp(const std::vector<std::string>& ops, const char* op) {
  return std::find(ops.begin(), ops.end(), op) != ops.end();
}

// AG chunk k equals k + 1; RS input is 1 on even and 2 on odd group ranks, so every partial sum
// is a small integer that bf16 holds exactly in any reduction order.  AR is in-place: input is
// commRank+1 on every element, so the sum is 1+2+...+n = n*(n+1)/2.
static Buffers makeBuffers(const Impl& impl, const Group& group, const std::vector<std::string>& ops, size_t sizeBytes,
                           const Dtype& dt, cudaStream_t stream) {
  Buffers b;
  size_t numel = sizeBytes / dt.size, shard = numel / group.ranks.size();
  if (hasOp(ops, "ag")) {
    b.agIn = allocPayload(impl, shard * dt.size);
    b.agOut = allocPayload(impl, sizeBytes);
    fill(dt, b.agIn, shard, float(group.commRank + 1), stream);
    fill(dt, b.agOut, numel, 0.f, stream);
  }
  if (hasOp(ops, "rs")) {
    b.rsIn = allocPayload(impl, sizeBytes);
    b.rsOut = allocPayload(impl, shard * dt.size);
    fill(dt, b.rsIn, numel, float(group.commRank % 2 + 1), stream);
    fill(dt, b.rsOut, shard, 0.f, stream);
  }
  if (hasOp(ops, "ar")) {
    b.arIn = allocPayload(impl, sizeBytes);
    fill(dt, b.arIn, numel, float(group.commRank + 1), stream);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  if (impl.windows) {
    // Registration is collective over the communicator (bootstrap barrier in dev_runtime.cc):
    // every member registers the same buffers in the same order.
    if (b.agIn) {
      NCCL_CHECK(ncclCommWindowRegister(group.ag, b.agIn, shard * dt.size, &b.wins[0], NCCL_WIN_COLL_SYMMETRIC));
      NCCL_CHECK(ncclCommWindowRegister(group.ag, b.agOut, sizeBytes, &b.wins[1], NCCL_WIN_COLL_SYMMETRIC));
    }
    if (b.rsIn) {
      NCCL_CHECK(ncclCommWindowRegister(group.rs, b.rsIn, sizeBytes, &b.wins[2], NCCL_WIN_COLL_SYMMETRIC));
      NCCL_CHECK(ncclCommWindowRegister(group.rs, b.rsOut, shard * dt.size, &b.wins[3], NCCL_WIN_COLL_SYMMETRIC));
    }
    if (b.arIn) {
      NCCL_CHECK(ncclCommWindowRegister(group.ar, b.arIn, sizeBytes, &b.wins[4], NCCL_WIN_COLL_SYMMETRIC));
    }
  }
  return b;
}

// Release the window registration before the memory goes away.
static void freeBuffers(const Impl& impl, const Group& group, Buffers& b) {
  CUDA_CHECK(cudaDeviceSynchronize());
  for (int i = 0; i < 5; i++) {
    if (b.wins[i]) {
      ncclComm_t comm = (i < 2) ? group.ag : (i < 4) ? group.rs : group.ar;
      NCCL_CHECK(ncclCommWindowDeregister(comm, b.wins[i]));
    }
  }
  for (void* ptr : {b.agIn, b.agOut, b.rsIn, b.rsOut, b.arIn}) freePayload(impl, ptr);
  b = Buffers();
}

static ncclCollConfig_t collConfig(const char* selection, bool copyEngine) {
  ncclCollConfig_t config = NCCL_COLLCONFIG_INITIALIZER;
  if (selection) config.algSelection = selection;  // forceAlgSelection stays 1: no silent fallback
  if (copyEngine) config.CTAPolicy = NCCL_CTA_POLICY_ZERO;
  return config;
}

struct CaseContext {
  const Options& o;
  const Dtype& dt;
  World& world;
  NcclLog& log;
  Streams& streams;
  unsigned long long delayNs;
  ncclCollConfig_t agConfig;
  ncclCollConfig_t rsConfig;
  ncclCollConfig_t arConfig;
  int captureIndex = 0;
};

static void launchOps(const CaseContext& ctx, const Group& group, const Buffers& b, const std::vector<std::string>& ops,
                      size_t shardCount) {
  for (const std::string& op : ops) {
    if (op == "ag") {
      NCCL_CHECK(ncclAllGatherConfig(b.agIn, b.agOut, shardCount, ctx.dt.nccl, group.ag, ctx.streams.ag, &ctx.agConfig));
    } else if (op == "rs") {
      NCCL_CHECK(ncclReduceScatterConfig(b.rsIn, b.rsOut, shardCount, ctx.dt.nccl, ncclSum, group.rs, ctx.streams.rs,
                                         &ctx.rsConfig));
    } else {  // "ar": in-place all-reduce; shardCount here is the full element count per rank
      NCCL_CHECK(ncclAllReduceConfig(b.arIn, b.arIn, shardCount, ctx.dt.nccl, ncclSum, group.ar, ctx.streams.ar,
                                     &ctx.arConfig));
    }
  }
}

struct TrialTimes {
  float makespan = 0, ag = 0, rs = 0, ar = 0, enqueue = 0;
};

// Pass the start gate, run `iters` back-to-back launches; per-iteration ms. The spin kernel keeps
// the gate and every launch queued before any of them runs, so host enqueue time is not measured.
// Every rank of the scenario passes the gate, including ranks outside it (which return false), so
// the gate has the same members in every case. The start event is recorded after the gate: all
// groups' clocks start within a few microseconds of each other.
static bool timedTrial(const CaseContext& ctx, const Scenario& s, const Group* group, const Buffers& b,
                       const std::vector<std::string>& ops, size_t shardCount, TrialTimes* t) {
  const Streams& st = ctx.streams;
  bool hasAg = hasOp(ops, "ag"), hasRs = hasOp(ops, "rs"), hasAr = hasOp(ops, "ar");
  if (ctx.delayNs) spinKernel<<<1, 1, 0, st.main>>>(ctx.delayNs);
  auto hostStart = std::chrono::steady_clock::now();
  NCCL_CHECK(ncclAllReduce(s.gateBuf, s.gateBuf, 1, ncclFloat32, ncclSum, s.gate, st.main));
  if (!group) {
    CUDA_CHECK(cudaStreamSynchronize(st.main));
    return false;
  }
  CUDA_CHECK(cudaEventRecord(st.start, st.main));
  CUDA_CHECK(cudaStreamWaitEvent(st.ag, st.start));
  CUDA_CHECK(cudaStreamWaitEvent(st.rs, st.start));
  CUDA_CHECK(cudaStreamWaitEvent(st.ar, st.start));
  for (int i = 0; i < ctx.o.iters; i++) launchOps(ctx, *group, b, ops, shardCount);
  t->enqueue = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - hostStart).count();
  if (hasAg) CUDA_CHECK(cudaEventRecord(st.agEnd, st.ag));
  if (hasRs) CUDA_CHECK(cudaEventRecord(st.rsEnd, st.rs));
  if (hasAr) CUDA_CHECK(cudaEventRecord(st.arEnd, st.ar));
  if (hasAg) CUDA_CHECK(cudaStreamWaitEvent(st.main, st.agEnd));
  if (hasRs) CUDA_CHECK(cudaStreamWaitEvent(st.main, st.rsEnd));
  if (hasAr) CUDA_CHECK(cudaStreamWaitEvent(st.main, st.arEnd));
  CUDA_CHECK(cudaEventRecord(st.end, st.main));
  CUDA_CHECK(cudaEventSynchronize(st.end));
  float ms;
  CUDA_CHECK(cudaEventElapsedTime(&ms, st.start, st.end));
  t->makespan = ms / ctx.o.iters;
  if (hasAg) { CUDA_CHECK(cudaEventElapsedTime(&ms, st.start, st.agEnd)); t->ag = ms / ctx.o.iters; }
  if (hasRs) { CUDA_CHECK(cudaEventElapsedTime(&ms, st.start, st.rsEnd)); t->rs = ms / ctx.o.iters; }
  if (hasAr) { CUDA_CHECK(cudaEventElapsedTime(&ms, st.start, st.arEnd)); t->ar = ms / ctx.o.iters; }
  return true;
}

static void copyString(char* dst, size_t cap, const std::string& src) { std::snprintf(dst, cap, "%s", src.c_str()); }

// What one case runs: group g of the scenario runs workloads[g] when active[g].
struct CaseSpec {
  std::string phase;         // single: isolated; split: concurrent / cross
  std::string part;          // "" for single, else one of splitParts()
  std::string caseWorkload;  // the workload in the case name (kCrossWorkload for cross)
  std::vector<bool> active;
  std::vector<std::string> workloads;
  size_t sizeBytes = 0;
};

// Run the active groups of `s` at the same time, all starting at the scenario's start gate.
// Returns one results.jsonl record per active group on world rank 0, nothing elsewhere.
// Everything about the case lands in cases/<impl>-<case>/: each member rank's slice of its NCCL
// log and rank 0's case.json. Fails on every rank together if any rank computed a wrong result or comm
// rank 0 of any group logged a launch that is not the implementation this pass measures.
static std::vector<JsonObject> runCase(CaseContext& ctx, const Scenario& s, const CaseSpec& spec) {
  const Options& o = ctx.o;
  const Impl& impl = *o.impl;
  World& world = ctx.world;
  std::string name = caseName(s.name, spec.part, spec.caseWorkload, o.dtype, spec.sizeBytes / kBytesPerMiB);
  std::string label = std::string(impl.name) + "/" + name;
  fs::path caseDir = fs::path(o.outputDir) / "cases" / caseDirName(impl.name, name);

  const Group* mine = nullptr;
  int groupIndex = -1;
  for (size_t g = 0; g < s.groups.size(); g++) {
    if (s.groups[g].member && spec.active[g]) {
      mine = &s.groups[g];
      groupIndex = int(g);
    }
  }
  std::vector<std::string> ops = mine ? workloadOps(spec.workloads[groupIndex]) : std::vector<std::string>{};
  // AllReduce is in-place: the shard is the whole buffer, so shardCount = numel, not numel/n.
  size_t shardCount = 0;
  if (mine) {
    size_t numel = spec.sizeBytes / ctx.dt.size;
    shardCount = hasOp(ops, "ar") ? numel : numel / mine->ranks.size();
  }
  Buffers buffers;
  if (mine) buffers = makeBuffers(impl, *mine, ops, spec.sizeBytes, ctx.dt, ctx.streams.main);
  world.barrier();
  // Allocation, window registration and anything since the previous case belong to this case's log.
  std::vector<std::string> caseLines = ctx.log.consume();
  size_t checkedFrom = caseLines.size();

  if (mine) {
    // First launch of each op alone, then the rest together.
    for (const std::string& op : ops) {
      launchOps(ctx, *mine, buffers, {op}, shardCount);
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    for (int i = 1; i < o.warmup; i++) launchOps(ctx, *mine, buffers, ops, shardCount);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // Warmup and setup stay outside the capture window: the report holds exactly the timed trials.
  // Every rank opens the window for every case, so report k of any rank is case k.
  std::pair<double, double> window = {0, 0};
  if (o.nsys) {
    ctx.captureIndex++;
    CUDA_CHECK(cudaDeviceSynchronize());
    window.first = unixNow();
    CUDA_CHECK(cudaProfilerStart());
    nvtxRangePushA(label.c_str());
  }
  std::vector<TrialTimes> trials;
  for (int t = 0; t < o.trials; t++) {
    world.barrier();
    if (s.gate) {
      nvtxRangePushA(("trial" + std::to_string(t)).c_str());
      TrialTimes times;
      if (timedTrial(ctx, s, mine, buffers, ops, shardCount, &times)) trials.push_back(times);
      nvtxRangePop();
    }
  }
  if (o.nsys) {
    CUDA_CHECK(cudaDeviceSynchronize());
    nvtxRangePop();
    CUDA_CHECK(cudaProfilerStop());
    window.second = unixNow();
  }
  std::vector<std::string> fresh = ctx.log.consume();
  caseLines.insert(caseLines.end(), fresh.begin(), fresh.end());

  RankResult local;
  std::memset(&local, 0, sizeof(local));
  local.rank = world.rank;
  local.groupIndex = groupIndex;
  local.passed = 1;
  if (mine) {
    size_t n = mine->ranks.size();
    if (hasOp(ops, "ag")) {
      local.passed &= countMismatches(ctx.dt, buffers.agOut, shardCount * n, shardCount, 0.f, ctx.streams.main) == 0;
    }
    if (hasOp(ops, "rs")) {
      local.passed &= countMismatches(ctx.dt, buffers.rsOut, shardCount, 0, float(n + n / 2), ctx.streams.main) == 0;
    }
    if (hasOp(ops, "ar")) {
      // AR is in-place: every element should equal 1+2+...+n = n*(n+1)/2.
      local.passed &= countMismatches(ctx.dt, buffers.arIn, shardCount, 0, float(n * (n + 1) / 2), ctx.streams.main) == 0;
    }
    for (int t = 0; t < o.trials; t++) {
      local.makespan[t] = trials[t].makespan;
      local.agMs[t] = trials[t].ag;
      local.rsMs[t] = trials[t].rs;
      local.arMs[t] = trials[t].ar;
      local.enqueueMs[t] = trials[t].enqueue;
    }
    local.nNodes = mine->logInfo.nNodes;
    local.mnnvl = mine->logInfo.mnnvl;
    local.symmetricMultimem = mine->logInfo.symmetricMultimem;
    local.multimemSupport = mine->multimemSupport;
    local.nLsaTeams = mine->nLsaTeams;
    if (mine->commRank == 0) {
      std::vector<std::string> errors;
      std::vector<std::string> checked(caseLines.begin() + checkedFrom, caseLines.end());
      std::string observed =
        checkLaunches(checked, impl, ops, size_t(o.warmup) + size_t(o.trials) * o.iters, shardCount * ctx.dt.size,
                      s.gateRank0(world.rank) ? size_t(o.trials) : 0, &errors);
      std::string joined;
      for (const std::string& e : errors) joined += (joined.empty() ? "" : "; ") + e;
      if (observed.empty()) joined += (joined.empty() ? "" : "; ") + std::string("no TUNING line seen");
      local.implChecked = 1;
      copyString(local.observed, sizeof(local.observed), observed);
      copyString(local.errors, sizeof(local.errors), joined);
    }
  }
  std::vector<char> raw = world.allGather(&local, sizeof(local));
  std::vector<RankResult> all(world.size);
  std::memcpy(all.data(), raw.data(), raw.size());

  if (mine) {
    freeBuffers(impl, *mine, buffers);
    fresh = ctx.log.consume();
    caseLines.insert(caseLines.end(), fresh.begin(), fresh.end());
    appendLines(caseDir / rankFile(impl.name, name, world.rank, world.size, ".nccl.log"), caseLines);
  }

  // Every rank fails together so no rank is left waiting at the next barrier.
  std::string wrong, implFailures;
  for (const RankResult& r : all) {
    if (r.groupIndex < 0) continue;
    if (!r.passed) wrong += (wrong.empty() ? "" : ", ") + std::to_string(r.rank);
    if (r.implChecked && r.errors[0]) {
      implFailures += (implFailures.empty() ? "" : "\n  ") + s.groups[r.groupIndex].name + ": " + r.errors;
    }
  }
  if (!wrong.empty()) die(label + ": wrong result on ranks [" + wrong + "]; NCCL logs in " + caseDir.string());
  if (!implFailures.empty()) {
    if (world.rank == 0) {
      fs::create_directories(caseDir);
      JsonObject report;
      report.set("label", jStr(label));
      std::string list = "[";
      for (const std::string& e : splitString(implFailures, '\n')) {
        list += (list.size() > 1 ? ", " : "") + jStr(e.substr(e.find_first_not_of(' ')));
      }
      report.set("implementation_errors", list + "]");
      std::ofstream(caseDir / "case.json") << report.dump("  ") << "\n";
    }
    die(label + ": NCCL did not run the " + impl.name + " implementation\n  " + implFailures + "\nNCCL logs in " +
        caseDir.string());
  }
  if (world.rank != 0) return {};

  // The slowest rank defines each trial of each group.
  struct GroupTrials {
    std::vector<double> makespan, ag, rs, ar, enqueue;
    const RankResult* owner = nullptr;
  };
  std::vector<GroupTrials> perGroup(s.groups.size());
  for (size_t g = 0; g < s.groups.size(); g++) {
    if (!spec.active[g]) continue;
    for (int t = 0; t < o.trials; t++) {
      float m = 0, a = 0, r = 0, ar = 0, e = 0;
      for (const RankResult& item : all) {
        if (item.groupIndex != int(g)) continue;
        m = std::max(m, item.makespan[t]);
        a = std::max(a, item.agMs[t]);
        r = std::max(r, item.rsMs[t]);
        ar = std::max(ar, item.arMs[t]);
        e = std::max(e, item.enqueueMs[t]);
      }
      perGroup[g].makespan.push_back(m);
      perGroup[g].ag.push_back(a);
      perGroup[g].rs.push_back(r);
      perGroup[g].ar.push_back(ar);
      perGroup[g].enqueue.push_back(e);
    }
    for (const RankResult& item : all) {
      if (item.groupIndex == int(g) && item.implChecked) perGroup[g].owner = &item;
    }
  }
  // All groups start at the gate, so their makespans share an origin: min/max is the share of
  // the longer group's time during which the other group was also moving data.
  std::vector<std::vector<double>> spans;
  for (size_t g = 0; g < s.groups.size(); g++) {
    if (spec.active[g]) spans.push_back(perGroup[g].makespan);
  }
  std::vector<double> concurrency = spans.size() > 1 ? trialConcurrency(spans) : std::vector<double>{};

  std::vector<JsonObject> records;
  for (size_t g = 0; g < s.groups.size(); g++) {
    if (!spec.active[g]) continue;
    const Group& group = s.groups[g];
    const GroupTrials& gt = perGroup[g];
    std::vector<std::string> groupOps = workloadOps(spec.workloads[g]);
    int n = group.size();
    double mk = median(gt.makespan);
    auto owner = [&](int RankResult::*field) { return gt.owner ? jInt(gt.owner->*field) : kNull; };
    JsonObject r;
    r.set("algo", jStr(impl.name));
    r.set("scenario", jStr(s.kind));
    r.set("phase", jStr(spec.phase));
    r.set("total_ranks", jInt(s.totalRanks));
    r.set("layout", s.layout.empty() ? kNull : jStr(s.layout));
    r.set("group_index", jInt(long(g)));
    r.set("case", jStr(name));
    r.set("case_dir", jStr("cases/" + caseDirName(impl.name, name)));
    r.set("group_name", jStr(group.name));
    r.set("group_size", jInt(n));
    r.set("ranks", jInts(group.ranks));
    r.set("size_bytes", jInt((long long)spec.sizeBytes));
    r.set("size_mib", jInt((long long)(spec.sizeBytes / kBytesPerMiB)));
    r.set("dtype", jStr(o.dtype));
    r.set("workload", jStr(spec.workloads[g]));
    r.set("iters", jInt(o.iters));
    r.set("trials", jInt(o.trials));
    r.set("trial_makespan_ms", jNums(gt.makespan));
    r.set("host_bound", jBool(*std::max_element(gt.enqueue.begin(), gt.enqueue.end()) > o.launchDelayMs));
    r.set("check_passed", jBool(true));
    // Profiled numbers include Nsight overhead; the report keeps them apart from timing runs.
    r.set("profiled", jBool(o.nsys));
    r.set("nsys_capture_index", o.nsys ? jInt(ctx.captureIndex) : kNull);
    r.set("nsys_label", o.nsys ? jStr(label) : kNull);
    r.set("nsys_window_unix", o.nsys ? jNums({window.first, window.second}) : kNull);
    r.set("trial_concurrency", concurrency.empty() ? kNull : jNums(concurrency));
    r.set("concurrency", concurrency.empty() ? kNull : jNum(median(concurrency)));
    r.set("nccl_impl", gt.owner && gt.owner->observed[0] ? jStr(gt.owner->observed) : kNull);
    r.set("n_nodes", owner(&RankResult::nNodes));
    r.set("mnnvl", gt.owner ? jBool(gt.owner->mnnvl == 1) : kNull);
    r.set("symmetric_multimem", owner(&RankResult::symmetricMultimem));
    r.set("multimem_support", owner(&RankResult::multimemSupport));
    r.set("n_lsa_teams", owner(&RankResult::nLsaTeams));
    r.set("makespan_ms", jNum(mk));
    r.set("enqueue_ms", jNum(median(gt.enqueue)));
    r.set("mean_makespan_ms", jNum(mean(gt.makespan)));
    r.set("mean_enqueue_ms", jNum(mean(gt.enqueue)));
    double aggregate = 0;
    for (const std::string& op : groupOps) {
      const std::vector<double>& times = op == "ag" ? gt.ag : op == "rs" ? gt.rs : gt.ar;
      double ms = median(times);
      r.set(op + "_ms", jNum(ms));
      r.set("mean_" + op + "_ms", jNum(mean(times)));
      r.set(op + "_algbw_gb_s", jNum(algBandwidthGBs(double(spec.sizeBytes), ms / 1e3)));
      // An all-reduce counts both directions, so its busbw already carries the factor 2.
      r.set(op + "_busbw_gb_s", jNum(opBusBandwidthGBs(op, double(spec.sizeBytes), n, ms / 1e3)));
      aggregate += opBusBandwidthGBs(op, double(spec.sizeBytes), n, mk / 1e3);
    }
    r.set("aggregate_busbw_gb_s", jNum(aggregate));
    records.push_back(r);
  }
  fs::create_directories(caseDir);
  std::ofstream out(caseDir / "case.json");
  out << "{\n  \"label\": " << jStr(label) << ",\n  \"records\": [";
  for (size_t i = 0; i < records.size(); i++) out << (i ? ",\n    " : "\n    ") << records[i].dump();
  out << "\n  ]\n}\n";
  if (o.nsys) {
    std::ofstream tsv(fs::path(o.outputDir) / "nsys" / "index.tsv", std::ios::app);
    char line[512];
    std::snprintf(line, sizeof(line), "%d\t%s\t%s\t%s\t%.3f\t%.3f\n", ctx.captureIndex, label.c_str(), s.kind.c_str(),
                  spec.phase.c_str(), window.first, window.second);
    tsv << line;
  }
  return records;
}

////////////////////////////////////////////////////////////////////////////////
// Metadata.

struct RankInfo {
  int32_t rank, localRank, device;
  char hostname[128];
  char gpu[128];
  char pciBusId[32];
  char uuid[40];
  char cudaVisibleDevices[128];
};

static void writeMetadata(World& world, const Options& o, int localRank, int argc, char** argv) {
  RankInfo info;
  std::memset(&info, 0, sizeof(info));
  info.rank = world.rank;
  info.localRank = localRank;
  CUDA_CHECK(cudaGetDevice(&info.device));
  gethostname(info.hostname, sizeof(info.hostname) - 1);
  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, info.device));
  copyString(info.gpu, sizeof(info.gpu), props.name);
  CUDA_CHECK(cudaDeviceGetPCIBusId(info.pciBusId, sizeof(info.pciBusId), info.device));
  char* u = info.uuid;
  for (int i = 0; i < 16; i++) u += std::snprintf(u, 3, "%02x", (unsigned char)props.uuid.bytes[i]);
  copyString(info.cudaVisibleDevices, sizeof(info.cudaVisibleDevices), envOr("CUDA_VISIBLE_DEVICES"));
  std::vector<char> raw = world.allGather(&info, sizeof(info));
  if (world.rank != 0) return;

  int version, runtimeVersion, driverVersion;
  NCCL_CHECK(ncclGetVersion(&version));
  CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVersion));
  CUDA_CHECK(cudaDriverGetVersion(&driverVersion));
  std::string argList = "[";
  for (int i = 0; i < argc; i++) argList += (i ? ", " : "") + jStr(argv[i]);
  JsonObject env;
  for (char** e = environ; *e; e++) {
    std::string kv = *e;
    if (kv.rfind("NCCL_", 0) && kv.rfind("CUDA_", 0) && kv.rfind("LD_LIBRARY_PATH=", 0)) continue;
    size_t eq = kv.find('=');
    env.set(kv.substr(0, eq), jStr(kv.substr(eq + 1)));
  }
  std::sort(env.items.begin(), env.items.end());
  std::string ranks = "[";
  for (int r = 0; r < world.size; r++) {
    const RankInfo* ri = reinterpret_cast<const RankInfo*>(raw.data()) + r;
    JsonObject item;
    item.set("rank", jInt(ri->rank));
    item.set("local_rank", jInt(ri->localRank));
    item.set("device", jInt(ri->device));
    item.set("hostname", jStr(ri->hostname));
    item.set("gpu", jStr(ri->gpu));
    item.set("gpu_uuid", jStr(ri->uuid));
    item.set("pci_bus_id", jStr(ri->pciBusId));
    item.set("cuda_visible_devices", ri->cudaVisibleDevices[0] ? jStr(ri->cudaVisibleDevices) : kNull);
    ranks += (r ? ",\n    " : "\n    ") + item.dump();
  }
  JsonObject meta;
  meta.set("impl", jStr(o.impl->name));
  meta.set("argv", argList + "]");
  meta.set("world_size", jInt(world.size));
  meta.set("nccl", jStr(std::to_string(version / 10000) + "." + std::to_string(version % 10000 / 100) + "." +
                        std::to_string(version % 100)));
  meta.set("nccl_version_code", jInt(version));
  meta.set("cuda_runtime", jInt(runtimeVersion));
  meta.set("cuda_driver", jInt(driverVersion));
  meta.set("env", env.dump("    "));
  meta.set("ranks", ranks + "\n  ]");
  std::ofstream(fs::path(o.outputDir) / "metadata.json") << meta.dump("  ") << "\n";
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  Options o = parseOptions(argc, argv);
  if (envOr("RANK").empty() || envOr("WORLD_SIZE").empty() || envOr("LOCAL_RANK").empty()) {
    die("RANK, WORLD_SIZE and LOCAL_RANK must be set (run_nvls_collective_benchmark.sh does)");
  }
  World world;
  world.rank = gRank = std::stoi(envOr("RANK"));
  world.size = std::stoi(envOr("WORLD_SIZE"));
  int localRank = std::stoi(envOr("LOCAL_RANK"));
  checkEnvironment(o, world.size);
  Dtype dt = makeDtype(o.dtype);
  for (int mib : o.sizesMiB) {
    std::vector<int> divisors = o.groupSizes;
    for (int t : o.splitTotals) divisors.push_back(t / 2);
    for (int n : divisors) {
      if ((size_t(mib) * kBytesPerMiB) % (size_t(n) * dt.size)) {
        die(std::to_string(mib) + " MiB cannot be split into " + std::to_string(n) + " " + dt.name + " shards");
      }
    }
  }

  // The benchmark must load the NCCL it was built against: per-call configs are new in 2.32.
  int version;
  NCCL_CHECK(ncclGetVersion(&version));
  if (version != NCCL_VERSION_CODE || version < 23200) {
    die("loaded NCCL " + std::to_string(version) + " but built against " + std::to_string(NCCL_VERSION_CODE) +
        " (need >= 23200); check LD_LIBRARY_PATH");
  }

  CUDA_CHECK(cudaSetDevice(localRank));
  fs::path results = fs::path(o.outputDir) / "results.jsonl";
  if (world.rank == 0) {
    fs::create_directories(o.outputDir);
    if (fs::exists(results)) die(results.string() + " already exists; use a fresh --output-dir");
  }

  char hostname[256] = {};
  gethostname(hostname, sizeof(hostname) - 1);
  NcclLog log(ncclDebugPath(envOr("NCCL_DEBUG_FILE"), hostname, int(getpid())));
  ncclUniqueId id;
  NCCL_CHECK(ncclGetUniqueId(&id));  // NCCL_COMM_ID: every rank gets the same id
  ncclConfig_t worldConfig = NCCL_CONFIG_INITIALIZER;
  NCCL_CHECK(ncclCommInitRankConfig(&world.comm, world.size, id, world.rank, &worldConfig));
  CUDA_CHECK(cudaStreamCreateWithFlags(&world.stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaMalloc(&world.barrierBuf, kWorldBarrierCount * sizeof(int)));
  CUDA_CHECK(cudaMemset(world.barrierBuf, 0, kWorldBarrierCount * sizeof(int)));
  world.barrier();
  writeMetadata(world, o, localRank, argc, argv);

  Streams streams;
  CUDA_CHECK(cudaStreamCreateWithFlags(&streams.main, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&streams.ag, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&streams.rs, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&streams.ar, cudaStreamNonBlocking));
  for (cudaEvent_t* e : {&streams.start, &streams.end, &streams.agEnd, &streams.rsEnd, &streams.arEnd}) CUDA_CHECK(cudaEventCreate(e));

  CaseContext ctx{o,
                  dt,
                  world,
                  log,
                  streams,
                  (unsigned long long)(o.launchDelayMs * 1e6),
                  collConfig(o.impl->agSelection, o.impl->agCopyEngine),
                  collConfig(o.impl->rsSelection, false),
                  collConfig(o.impl->arSelection, false)};
  if (world.rank == 0 && o.nsys) {
    fs::create_directories(fs::path(o.outputDir) / "nsys");
    std::ofstream(fs::path(o.outputDir) / "nsys" / "index.tsv")
      << "capture_index\tlabel\tscenario\tphase\tstart_unix\tend_unix\n";
  }
  log.consume();  // world init, metadata gather

  auto emit = [&](const std::vector<JsonObject>& records) {
    if (world.rank != 0) return;
    std::ofstream out(results, std::ios::app);
    for (const JsonObject& record : records) {
      out << record.dump() << "\n";
      Row r = parseJsonObject(record.dump());
      std::printf("%-6s %-6s %-10s n=%-2s %5s MiB %-5s %-22s makespan %8.3f ms  busbw %7.1f GB/s  %s%s\n",
                  o.impl->name, r["scenario"].c_str(), r["phase"].c_str(), r["group_size"].c_str(),
                  r["size_mib"].c_str(), r["workload"].c_str(), r["group_name"].c_str(), std::stod(r["makespan_ms"]),
                  std::stod(r["aggregate_busbw_gb_s"]), r["nccl_impl"].empty() ? "kernel=?" : r["nccl_impl"].c_str(),
                  r["host_bound"] == "true" ? "  [HOST-BOUND]" : "");
    }
  };

  for (int n : o.groupSizes) {
    std::vector<int> ranks(n);
    for (int r = 0; r < n; r++) ranks[r] = r;
    Scenario s = createScenario(world, o, log, streams.main, "single", n, "", {{"single" + std::to_string(n), ranks}});
    for (int mib : o.sizesMiB) {
      for (const std::string& workload : measuredWorkloads(impl, o.workloads)) {
        emit(runCase(ctx, s, {"isolated", "", workload, {true}, {workload}, size_t(mib) * kBytesPerMiB}));
      }
    }
    destroyScenario(world, s, log);
  }

  for (int total : o.splitTotals) {
    for (const std::string& layout : o.splitLayouts) {
      auto halves = splitGroups(total, layout);
      std::string prefix = layout + std::to_string(total);
      Scenario s = createScenario(world, o, log, streams.main, "split", total, layout,
                                  {{prefix + "-A", halves.first}, {prefix + "-B", halves.second}});
      for (int mib : o.sizesMiB) {
        size_t sizeBytes = size_t(mib) * kBytesPerMiB;
        for (const std::string& workload : measuredWorkloads(impl, o.workloads)) {
          // Both halves run together; the reference of each half is single-n<T/2>, like
          // ../nvls-collective. No half runs alone.
          emit(runCase(ctx, s, {"concurrent", "AB-concurrent", workload, {true, true}, {workload, workload}, sizeBytes}));
        }
        if (hasCross(o)) {
          emit(runCase(ctx, s, {"cross", kCrossPart, kCrossWorkload, {true, true}, crossGroupWorkloads(), sizeBytes}));
        }
      }
      destroyScenario(world, s, log);
    }
  }

  world.barrier();
  NCCL_CHECK(ncclCommDestroy(world.comm));
  if (world.rank == 0) std::printf("wrote %s\n", results.string().c_str());
  return 0;
}
