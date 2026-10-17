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

// Host-only pieces shared by nvls_collective_bench.cu, collective_report.cc and selftest.cc:
// the implementation table, the names of scenarios / cases / per-rank files (same scheme as
// ../nvls-collective/collective_metrics.py), the NCCL TUNING-log parser that proves which kernel
// ran, the bandwidth formulas and the results.jsonl reader/writer. Nothing here touches CUDA or
// NCCL, so `make selftest` checks it on a machine without a GPU.

#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nvls_bench {

constexpr size_t kBytesPerMiB = size_t(1) << 20;
constexpr double kBytesPerGB = 1e9;
// The start gate is a one-element fp32 all-reduce: its TUNING line is "AllReduce: 4 Bytes".
// The benchmark's own world barrier all-reduces two int32 (8 Bytes), so the two never mix.
constexpr size_t kGateBytes = 4;
constexpr size_t kWorldBarrierCount = 2;

// One implementation under test. Every launch passes its own ncclCollConfig_t (NCCL >= 2.32),
// so the kernel is pinned per call instead of through NCCL_ALGO / NCCL_SYM_KERNEL, and
// forceAlgSelection=1 turns an unavailable kernel into an error instead of a silent fallback.
struct Impl {
  const char* name;
  bool windows;             // payloads are ncclMemAlloc + NCCL_WIN_COLL_SYMMETRIC windows
  bool needsMultimem;       // only valid on a communicator with one LSA team and multimem
  const char* agSelection;  // ncclCollConfig_t::algSelection for AllGather; nullptr = automatic
  bool agCopyEngine;        // AllGather sets CTAPolicy = NCCL_CTA_POLICY_ZERO (copy-engine path)
  const char* rsSelection;  // ncclCollConfig_t::algSelection for ReduceScatter
  const char* arSelection;  // ncclCollConfig_t::algSelection for AllReduce; nullptr = automatic
  const char* agExpect;     // what every AllGather launch must report, see implementationMatches
  const char* rsExpect;
  const char* arExpect;     // what every AllReduce launch must report, see implementationMatches
  unsigned workloadMask;    // which workloads() entries this impl measures, see workloadMaskOf
};

// Bit i of the returned mask is set when workloads()[i] is in `names` (defined below).
inline unsigned workloadMaskOf(const std::vector<std::string>& names);

// algSelection tags match registry names by prefix (src/config/algorithm_registry.cc), so only
// tags that are not a prefix of another name are used: "SYMK_STMC" never matches TmaSTMC, but
// "SYMK_ST" would also match SYMK_STMC.
//
// The five implementations and what each one measures:
//   ring            classic kernels on plain buffers; AG, RS, AG+RS, AR
//   nvls            NVLS_SIMPLE on plain buffers; AG, RS, AG+RS, AR
//   symmetric       register multimem: AllGather_STMC (multimem.st) and ReduceScatter_LDMC
//                   (multimem.ld_reduce), plus AllReduce_RSxLDMC_AGxSTMC; all four workloads
//   symmetric-tma   TMA AllGather (AllGather_TmaSTMC) with the same LDMC ReduceScatter, so only
//                   AG and AG+RS are measured; RS and AR are covered by `symmetric`
//   copy-engine     CE AllGather writing the multicast address with the same LDMC ReduceScatter,
//                   so only AG and AG+RS are measured; RS and AR are covered by `symmetric`
inline const std::vector<Impl>& impls() {
  static const std::vector<Impl> table = {
    {"ring", false, false, "RING", false, "RING", "RING", "Ring/*", "Ring/*", "Ring/*",
     workloadMaskOf({"ag", "rs", "ag+rs", "ar"})},
    {"nvls", false, false, "NVLS_SIMPLE", false, "NVLS_SIMPLE", "NVLS_SIMPLE", "NVLS/*", "NVLS/*", "NVLS/*",
     workloadMaskOf({"ag", "rs", "ag+rs", "ar"})},
    // Register multimem.st AllGather and multimem.ld_reduce ReduceScatter: the non-TMA baseline.
    // AllReduce_RSxLDMC_AGxSTMC is the AllReduce built from the same two primitives.
    {"symmetric", true, true, "SYMK_STMC", false, "SYMK_LDMC", "SYMK_RSxLDMC_AGxSTMC",
     "Symmetric/AllGather_STMC", "Symmetric/ReduceScatter_LDMC", "Symmetric/AllReduce_RSxLDMC_AGxSTMC",
     workloadMaskOf({"ag", "rs", "ag+rs", "ar"})},
    // TMA AllGather: cp.async.bulk global->smem, then cp.async.bulk smem->multicast address.
    {"symmetric-tma", true, true, "SYMK_TmaSTMC", false, "SYMK_LDMC", nullptr, "Symmetric/AllGather_TmaSTMC",
     "Symmetric/ReduceScatter_LDMC", nullptr, workloadMaskOf({"ag", "ag+rs"})},
    // Copy-engine AllGather writing the multicast address (one CE copy per rank), LDMC RS.
    {"copy-engine", true, true, nullptr, true, "SYMK_LDMC", nullptr, "CopyEngine+NVLSSync",
     "Symmetric/ReduceScatter_LDMC", nullptr, workloadMaskOf({"ag", "ag+rs"})},
  };
  return table;
}

inline const Impl* findImpl(const std::string& name) {
  for (const Impl& impl : impls()) {
    if (name == impl.name) return &impl;
  }
  return nullptr;
}

inline int implRank(const std::string& name) {
  for (size_t i = 0; i < impls().size(); i++) {
    if (name == impls()[i].name) return int(i);
  }
  return int(impls().size());
}

inline std::vector<std::string> splitString(const std::string& text, char sep) {
  std::vector<std::string> out;
  std::string item;
  std::istringstream stream(text);
  while (std::getline(stream, item, sep)) out.push_back(item);
  if (!text.empty() && text.back() == sep) out.emplace_back();
  return out;
}

// Comma-separated, de-duplicated, order kept; empty items are rejected.
inline std::vector<std::string> parseCsv(const std::string& text) {
  std::vector<std::string> out;
  for (const std::string& item : splitString(text, ',')) {
    if (item.empty()) throw std::invalid_argument("empty item in list '" + text + "'");
    if (std::find(out.begin(), out.end(), item) == out.end()) out.push_back(item);
  }
  if (out.empty()) throw std::invalid_argument("empty list");
  return out;
}

inline std::vector<int> parsePositiveInts(const std::string& text) {
  std::vector<int> out;
  for (const std::string& item : parseCsv(text)) {
    size_t used = 0;
    int value = std::stoi(item, &used);
    if (used != item.size() || value < 1) throw std::invalid_argument("expected positive integers, got '" + text + "'");
    out.push_back(value);
  }
  return out;
}

////////////////////////////////////////////////////////////////////////////////
// Workloads, scenarios and names (collective_metrics.py: WORKLOADS, SPLIT_PARTS, scenario_name,
// case_name, case_dir_name, rank_file).

inline const std::vector<std::string>& workloads() {
  // "ar" is always standalone: AllReduce is never mixed with AG or RS traffic.
  static const std::vector<std::string> list = {"ag", "rs", "ag+rs", "ar"};
  return list;
}

inline int workloadRank(const std::string& name) {
  const auto& list = workloads();
  return int(std::find(list.begin(), list.end(), name) - list.begin());
}

// Bit i of the returned mask is set when workloads()[i] is in `names`.
inline unsigned workloadMaskOf(const std::vector<std::string>& names) {
  unsigned mask = 0;
  for (const std::string& name : names) {
    if (workloadRank(name) == int(workloads().size())) throw std::invalid_argument("unknown workload '" + name + "'");
    mask |= 1u << workloadRank(name);
  }
  return mask;
}

// The requested workloads this impl measures, in the requested order.
inline std::vector<std::string> measuredWorkloads(const Impl& impl, const std::vector<std::string>& requested) {
  std::vector<std::string> out;
  for (const std::string& name : requested) {
    if (impl.workloadMask & (1u << workloadRank(name))) out.push_back(name);
  }
  return out;
}

inline const std::vector<std::string>& splitLayouts() {
  static const std::vector<std::string> list = {"contiguous", "interleaved"};
  return list;
}

// Split cases: both half-groups run the same workload at the same time, and "cross": group A
// all-gathers while group B reduce-scatters, at the same time. Neither half runs alone: the
// reference for a half-group of T/2 ranks is the single group of T/2 ranks (single-n<T/2>).
inline const std::vector<std::string>& splitParts() {
  static const std::vector<std::string> list = {"AB-concurrent", "AB-cross"};
  return list;
}
const std::string kCrossPart = "AB-cross";
const std::string kCrossWorkload = "agA-rsB";
inline const std::vector<std::string>& crossGroupWorkloads() {  // per split group, A then B
  static const std::vector<std::string> list = {"ag", "rs"};
  return list;
}

inline std::vector<std::string> workloadOps(const std::string& workload) {
  if (workload == "ag+rs") return {"ag", "rs"};
  if (workload == "ag" || workload == "rs" || workload == "ar") return {workload};
  throw std::invalid_argument("unknown workload '" + workload + "'");
}

inline std::string opFunc(const std::string& op) {
  if (op == "ag") return "AllGather";
  if (op == "rs") return "ReduceScatter";
  if (op == "ar") return "AllReduce";
  throw std::invalid_argument("unknown op '" + op + "'");
}

// Two equal, disjoint halves of [0, total): first/second half or even/odd ranks.
inline std::pair<std::vector<int>, std::vector<int>> splitGroups(int total, const std::string& layout) {
  if (total < 2 || total % 2) throw std::invalid_argument("split needs an even rank count >= 2");
  if (layout != "contiguous" && layout != "interleaved") throw std::invalid_argument("unknown split layout '" + layout + "'");
  std::vector<int> a, b;
  for (int r = 0; r < total; r++) {
    bool first = layout == "contiguous" ? r < total / 2 : r % 2 == 0;
    (first ? a : b).push_back(r);
  }
  return {a, b};
}

// One set of communicators: single-n8 or split-n16-contiguous.
inline std::string scenarioName(const std::string& scenario, int totalRanks, const std::string& layout = "") {
  if (scenario == "single" && layout.empty()) return "single-n" + std::to_string(totalRanks);
  if (scenario == "split" && std::find(splitLayouts().begin(), splitLayouts().end(), layout) != splitLayouts().end()) {
    return "split-n" + std::to_string(totalRanks) + "-" + layout;
  }
  throw std::invalid_argument("unknown scenario '" + scenario + "' with layout '" + layout + "'");
}

// Spelled-out workloads for case directory and report names (collective_metrics.py: WORKLOAD_NAMES).
inline std::string workloadName(const std::string& workload) {
  if (workload == "ag") return "all_gather";
  if (workload == "rs") return "reduce_scatter";
  if (workload == "ar") return "all_reduce";
  if (workload == "ag+rs") return "all_gather+reduce_scatter";
  if (workload == kCrossWorkload) return "all_gather_A+reduce_scatter_B";
  throw std::invalid_argument("unknown workload '" + workload + "'");
}

// One test, e.g. all_gather+reduce_scatter-1024MiB-bf16-split-n16-contiguous-AB-concurrent.
// Prefixed with the impl (caseDirName) it is the case directory and the stem of every per-rank
// file in it; <impl>/<case> is the outer NVTX range of the case's nsys report. The cross part
// takes kCrossWorkload, every other case one of workloads(). sizeMiB is the full tensor.
inline std::string caseName(const std::string& scenario, const std::string& part, const std::string& workload,
                            const std::string& dtype, size_t sizeMiB) {
  if (!part.empty() && std::find(splitParts().begin(), splitParts().end(), part) == splitParts().end()) {
    throw std::invalid_argument("unknown split part '" + part + "'");
  }
  if ((part == kCrossPart) != (workload == kCrossWorkload)) {
    throw std::invalid_argument("workload " + kCrossWorkload + " belongs to part " + kCrossPart + " only");
  }
  if (workload != kCrossWorkload) workloadOps(workload);
  std::string suffix = part.empty() ? scenario : scenario + "-" + part;
  return workloadName(workload) + "-" + std::to_string(sizeMiB) + "MiB-" + dtype + "-" + suffix;
}

// The case directory, e.g. tma-all_gather-1024MiB-bf16-single-n16: impl first, so a directory or
// report copied out of the run still says what it measured.
inline std::string caseDirName(const std::string& impl, const std::string& name) { return impl + "-" + name; }

// Per-rank file name <impl>-<name>.rank_<RR><suffix>, zero-padded so files sort by rank.
inline std::string rankFile(const std::string& impl, const std::string& name, int rank, int worldSize,
                            const std::string& suffix) {
  int width = std::max<int>(2, int(std::to_string(std::max(worldSize - 1, 0)).size()));
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%0*d", width, rank);
  return caseDirName(impl, name) + ".rank_" + buf + suffix;
}

// Full-tensor payload rate: AG output bytes, RS input bytes or AR input bytes per second.
inline double algBandwidthGBs(double totalBytes, double seconds) { return totalBytes / seconds / kBytesPerGB; }

// nccl-tests bus bandwidth for AG/RS: total_bytes * (n - 1) / n / t.
inline double busBandwidthGBs(double totalBytes, int groupSize, double seconds) {
  return algBandwidthGBs(totalBytes, seconds) * (groupSize - 1) / groupSize;
}

// Bus bandwidth of one op of a workload (collective_metrics.py: op_bus_bandwidth_gb_s). An
// all-reduce of total_bytes per rank sends (n - 1) / n of them and receives the same amount back,
// so it counts twice: the nccl-tests all-reduce factor 2 * (n - 1) / n.
inline double opBusBandwidthGBs(const std::string& op, double totalBytes, int groupSize, double seconds) {
  if (op != "ag" && op != "rs" && op != "ar") throw std::invalid_argument("unknown op '" + op + "'");
  double bus = busBandwidthGBs(totalBytes, groupSize, seconds);
  return op == "ar" ? 2 * bus : bus;
}

inline double median(std::vector<double> values) {
  if (values.empty()) throw std::invalid_argument("median of an empty list");
  std::sort(values.begin(), values.end());
  size_t n = values.size();
  return n % 2 ? values[n / 2] : 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

inline double mean(const std::vector<double>& values) {
  if (values.empty()) throw std::invalid_argument("mean of an empty list");
  double sum = 0;
  for (double v : values) sum += v;
  return sum / values.size();
}

// Per trial, min / max of the groups' times measured from the shared start gate. spans[g][t] is
// group g's slowest-rank time in trial t. Every group starts at the gate, so the shorter group
// was running during exactly that share of the longer group's time.
inline std::vector<double> trialConcurrency(const std::vector<std::vector<double>>& spans) {
  if (spans.size() < 2) throw std::invalid_argument("concurrency needs at least two groups");
  std::vector<double> out;
  for (size_t t = 0; t < spans[0].size(); t++) {
    double lo = spans[0][t], hi = spans[0][t];
    for (const auto& group : spans) {
      if (group.size() != spans[0].size()) throw std::invalid_argument("groups have different trial counts");
      lo = std::min(lo, group[t]);
      hi = std::max(hi, group[t]);
    }
    out.push_back(lo / hi);
  }
  return out;
}

////////////////////////////////////////////////////////////////////////////////
// NCCL log evidence.

// NCCL_DEBUG_FILE expansion as in NCCL's debug.cc: %h is the host name cut at the first '.',
// %p the pid, %% a literal '%'; anything else after '%' is echoed unchanged.
inline std::string ncclDebugPath(const std::string& tmpl, const std::string& hostname, int pid) {
  std::string out;
  for (size_t i = 0; i < tmpl.size(); i++) {
    if (tmpl[i] != '%' || i + 1 >= tmpl.size()) {
      out += tmpl[i];
      continue;
    }
    char code = tmpl[++i];
    if (code == 'h') {
      out += hostname.substr(0, hostname.find('.'));
    } else if (code == 'p') {
      out += std::to_string(pid);
    } else if (code == '%') {
      out += '%';
    } else {
      out += '%';
      out += code;
    }
  }
  return out;
}

// One INFO(NCCL_TUNING, ...) line that NCCL 2.32 prints on comm rank 0 for every launch:
//   "<Func>: <n> Bytes -> Algo <algo> proto <proto> ..."                    enqueue.cc
//   "<Func> [Symmetric]: <n> Bytes -> Kernel <name> ..."                    scheduler/symmetric_sched.cc
//   "<Func> [Copy Engine]: <n> Bytes -> cudaMemcpy[; CE synchronization with NVLS]"   ce_coll.cc
//   "<Func> [Hierarchical CE]: <n> Bytes -> ..."                             ce_coll.cc, multi-LSA
// <n> is the per-rank count times the element size: AG sendcount, RS recvcount.
struct TuningLaunch {
  std::string func;
  std::string impl;  // Ring/Simple, Symmetric/AllGather_TmaSTMC, CopyEngine, CopyEngine+NVLSSync, HierarchicalCE
  size_t bytes = 0;
};

inline bool parseTuningLine(const std::string& line, TuningLaunch* out) {
  static const std::regex pattern(
    R"(NCCL INFO (\w+)(?: \[(Symmetric|Copy Engine|Hierarchical CE)\])?: (\d+) Bytes -> )"
    R"((?:Algo (\S+) proto (\S+)|Kernel (\S+)|(.*)))");
  std::smatch m;
  if (!std::regex_search(line, m, pattern)) return false;
  out->func = m[1];
  out->bytes = std::stoull(m[3]);
  std::string path = m[2];
  if (path == "Symmetric") {
    if (!m[6].matched) return false;
    out->impl = "Symmetric/" + m[6].str();
  } else if (path == "Copy Engine") {
    out->impl = m[7].str().find("CE synchronization with NVLS") != std::string::npos ? "CopyEngine+NVLSSync" : "CopyEngine";
  } else if (path == "Hierarchical CE") {
    out->impl = "HierarchicalCE";
  } else {
    if (!m[4].matched) return false;
    out->impl = m[4].str() + "/" + m[5].str();
  }
  return true;
}

// expect is '|'-separated alternatives; "Ring/*" matches the family before '/', a trailing '*'
// is a prefix match, anything else must match exactly.  Both the prefix and exact comparisons are
// case-insensitive so the check works across NCCL releases: older releases log ncclAlgoStr()
// ("Ring", "NVLS") while 2.29+ logs ncclAlgoToString() ("RING", "NVLS"); kernel names like
// "Simple" vs "SIMPLE" vary similarly.
inline bool implementationMatches(const std::string& expect, const std::string& impl) {
  for (const std::string& alt : splitString(expect, '|')) {
    if (!alt.empty() && alt.back() == '*') {
      size_t len = alt.size() - 1;
      if (impl.size() >= len && strncasecmp(impl.c_str(), alt.c_str(), len) == 0) return true;
    } else if (strcasecmp(impl.c_str(), alt.c_str()) == 0) {
      return true;
    }
  }
  return false;
}

// Communicator facts NCCL logs at init: "comm <p> rank <r> nRanks <n> nNodes <n> localRanks <n>
// localRank <r> MNNVL <0|1>" (init.cc) and "... symmetric multimem enabled|disabled" (nvls.cc).
struct CommLogInfo {
  int nNodes = -1;
  int mnnvl = -1;
  int symmetricMultimem = -1;
};

inline void parseCommLine(const std::string& line, CommLogInfo* info) {
  static const std::regex comm(R"( nRanks \d+ nNodes (\d+) localRanks \d+ localRank \d+ MNNVL (\d+))");
  static const std::regex multimem(R"(symmetric multimem (enabled|disabled))");
  std::smatch m;
  if (std::regex_search(line, m, comm)) {
    info->nNodes = std::stoi(m[1]);
    info->mnnvl = std::stoi(m[2]);
  } else if (std::regex_search(line, m, multimem)) {
    info->symmetricMultimem = m[1] == "enabled";
  }
}

// Check every collective launch comm rank 0 of one group logged during one case (warmup + timed
// trials): each op must appear exactly `launches` times with `shardBytes` per rank on the
// implementation the impl pins, and the start gate exactly `gateLaunches` times (only the gate's
// comm rank 0 logs it). AllReduce with <= kWorldBarrierBytes (kGateBytes or world-barrier size) is
// either the start gate or the world barrier and is never counted as payload.  Returns the
// observed implementations per function.
constexpr size_t kWorldBarrierBytes = kWorldBarrierCount * sizeof(int32_t);  // 8 Bytes
inline std::string checkLaunches(const std::vector<std::string>& lines, const Impl& impl,
                                 const std::vector<std::string>& ops, size_t launches, size_t shardBytes,
                                 size_t gateLaunches, std::vector<std::string>* errors) {
  std::map<std::string, std::map<std::string, size_t>> seen;
  std::map<std::string, std::map<size_t, size_t>> sizes;
  std::map<std::string, size_t> foreign;
  std::set<std::string> funcs;
  size_t gates = 0;
  bool hasAr = std::find(ops.begin(), ops.end(), "ar") != ops.end();
  for (const std::string& op : ops) funcs.insert(opFunc(op));
  for (const std::string& line : lines) {
    TuningLaunch launch;
    if (!parseTuningLine(line, &launch)) continue;
    if (launch.func == "AllReduce") {
      if (launch.bytes == kGateBytes) {
        // Start gate: 4-Byte fp32 all-reduce on the gate communicator.
        gates++;
        continue;
      }
      if (launch.bytes <= kWorldBarrierBytes && !hasAr) {
        // World barrier (8-Byte int32*2): ignore it when AR is not the payload.
        continue;
      }
      if (hasAr) {
        // When AR is the payload, large AllReduces are the tested collective; small ones are
        // the gate or world barrier and are handled above / fall through to foreign below.
        if (launch.bytes <= kWorldBarrierBytes) {
          continue;  // world barrier
        }
        seen[launch.func][launch.impl]++;
        sizes[launch.func][launch.bytes]++;
        continue;
      }
      // AR is not in ops but we saw a large AllReduce: foreign.
      foreign[launch.func + "=" + launch.impl]++;
      continue;
    }
    if (!funcs.count(launch.func)) {
      foreign[launch.func + "=" + launch.impl]++;
      continue;
    }
    seen[launch.func][launch.impl]++;
    sizes[launch.func][launch.bytes]++;
  }
  for (const auto& item : foreign) errors->push_back("unexpected collective " + item.first + " x" + std::to_string(item.second));
  if (gates != gateLaunches) {
    errors->push_back("start gate: " + std::to_string(gates) + " AllReduce launches in the NCCL log, expected " +
                      std::to_string(gateLaunches));
  }
  std::string observed;
  for (const std::string& op : ops) {
    std::string func = opFunc(op);
    std::string expect;
    if (op == "ag") expect = impl.agExpect ? impl.agExpect : "";
    else if (op == "rs") expect = impl.rsExpect ? impl.rsExpect : "";
    else if (op == "ar") expect = impl.arExpect ? impl.arExpect : "";
    size_t total = 0;
    std::string wrong, names;
    for (const auto& item : seen[func]) {
      total += item.second;
      names += (names.empty() ? "" : "|") + item.first;
      if (!implementationMatches(expect, item.first)) {
        wrong += (wrong.empty() ? "" : ", ") + item.first + " x" + std::to_string(item.second);
      }
    }
    if (total != launches) {
      errors->push_back(func + ": " + std::to_string(total) + " launches in the NCCL log, expected " + std::to_string(launches));
    }
    if (!wrong.empty()) errors->push_back(func + ": expected " + expect + ", got " + wrong);
    for (const auto& item : sizes[func]) {
      if (item.first != shardBytes) {
        errors->push_back(func + ": expected " + std::to_string(shardBytes) + " B per rank, got " +
                          std::to_string(item.first) + " B x" + std::to_string(item.second));
      }
    }
    if (!names.empty()) observed += (observed.empty() ? "" : ";") + func + "=" + names;
  }
  return observed;
}

// The CUDA function that did the work for one implementation read from the TUNING log
// (collective_metrics.py: kernel_name). Names follow NCCL 2.32.3's generators. The symmetric path
// launches one kernel per variant, ncclSymkDevKernel_<Coll>_<algo>[_sum_<ty>]
// (src/device/symmetric/generate.py), so the launched kernel names the variant. The classic path
// launches one specialized kernel per collective (ncclDevKernel_<Coll>[_Sum_<ty>]_RING_LL) that
// dispatches on its funcId to ncclDevFunc_<Coll>[_Sum_<ty>]_<ALGO>_<PROTO> (src/device/generate.py);
// that device function is what ran, so it is the name reported here. The copy-engine path has no
// kernel. dtype is the benchmark dtype, bf16 or fp32.
inline std::string kernelName(const std::string& func, const std::string& implementation, const std::string& dtype) {
  std::string ty = dtype == "fp32" ? "f32" : dtype;
  bool reduces = func != "AllGather";
  size_t slash = implementation.find('/');
  std::string family = implementation.substr(0, slash);
  std::string detail = slash == std::string::npos ? "" : implementation.substr(slash + 1);
  if (family == "Symmetric") {
    if (detail.rfind(func + "_", 0) == 0) detail = detail.substr(func.size() + 1);
    return "ncclSymkDevKernel_" + func + "_" + detail + (reduces ? "_sum_" + ty : "");
  }
  if (family.rfind("CopyEngine", 0) == 0 || family == "HierarchicalCE") return family + "（cudaMemcpy，无 kernel）";
  auto upper = [](std::string s) {
    for (char& c : s) c = char(std::toupper((unsigned char)c));
    return s;
  };
  return "ncclDevFunc_" + func + (reduces ? "_Sum_" + ty : "") + "_" + upper(family) + "_" + upper(detail);
}

// Map each op of a record to the kernel(s) its checked nccl_impl evidence names, e.g.
// "AllGather=NVLS/SIMPLE;ReduceScatter=NVLS/SIMPLE" (collective_metrics.py: record_kernels).
// Several implementations of one op ('|') are all listed; no evidence maps to nothing.
inline std::map<std::string, std::string> recordKernels(const std::string& ncclImpl, const std::string& dtype) {
  std::map<std::string, std::string> kernels;
  for (const std::string& entry : splitString(ncclImpl, ';')) {
    size_t eq = entry.find('=');
    if (entry.empty() || eq == std::string::npos) continue;
    std::string func = entry.substr(0, eq), op;
    if (func == "AllGather") op = "ag";
    else if (func == "ReduceScatter") op = "rs";
    else if (func == "AllReduce") op = "ar";
    else continue;
    std::string names;
    for (const std::string& impl : splitString(entry.substr(eq + 1), '|')) {
      if (!impl.empty()) names += (names.empty() ? "" : " \\| ") + kernelName(func, impl, dtype);
    }
    kernels[op] = names;
  }
  return kernels;
}

////////////////////////////////////////////////////////////////////////////////
// JSON: results.jsonl / case.json / metadata.json use the schema of ../nvls-collective, so
// ../nvls-collective/analyse_nsys.py reads them unchanged. Only flat objects whose values are
// strings, numbers, booleans, null or arrays of numbers are written and read.

inline std::string jsonEscape(const std::string& text) {
  std::string out;
  for (char c : text) {
    switch (c) {
    case '"': out += "\\\""; break;
    case '\\': out += "\\\\"; break;
    case '\n': out += "\\n"; break;
    case '\t': out += "\\t"; break;
    default:
      if ((unsigned char)c < 0x20) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "\\u%04x", c);
        out += buf;
      } else {
        out += c;
      }
    }
  }
  return out;
}

inline std::string jStr(const std::string& s) { return "\"" + jsonEscape(s) + "\""; }
inline std::string jNum(double v) {
  char buf[40];
  std::snprintf(buf, sizeof(buf), "%.10g", v);
  return buf;
}
inline std::string jInt(long long v) { return std::to_string(v); }
inline std::string jBool(bool v) { return v ? "true" : "false"; }
inline const std::string kNull = "null";
inline std::string jNums(const std::vector<double>& values) {
  std::string out = "[";
  for (size_t i = 0; i < values.size(); i++) out += (i ? ", " : "") + jNum(values[i]);
  return out + "]";
}
inline std::string jInts(const std::vector<int>& values) {
  std::string out = "[";
  for (size_t i = 0; i < values.size(); i++) out += (i ? ", " : "") + std::to_string(values[i]);
  return out + "]";
}

// Insertion-ordered object of key -> already-encoded JSON value.
struct JsonObject {
  std::vector<std::pair<std::string, std::string>> items;

  void set(const std::string& key, const std::string& raw) {
    for (auto& item : items) {
      if (item.first == key) {
        item.second = raw;
        return;
      }
    }
    items.emplace_back(key, raw);
  }

  std::string dump(const std::string& indent = "") const {
    std::string sep = indent.empty() ? ", " : ",\n" + indent;
    std::string out = indent.empty() ? "{" : "{\n" + indent;
    for (size_t i = 0; i < items.size(); i++) out += (i ? sep : "") + jStr(items[i].first) + ": " + items[i].second;
    return out + (indent.empty() ? "}" : "\n}");
  }
};

// Decoded row: strings unquoted, numbers / booleans as their text, null as "", arrays raw ("[1, 2]").
using Row = std::map<std::string, std::string>;

inline Row parseJsonObject(const std::string& text) {
  Row row;
  size_t i = 0;
  auto ws = [&]() {
    while (i < text.size() && std::isspace((unsigned char)text[i])) i++;
  };
  auto fail = [&](const std::string& why) -> std::runtime_error {
    return std::runtime_error("JSON: " + why + " at offset " + std::to_string(i) + ": " + text.substr(0, 80));
  };
  auto str = [&]() {
    if (text[i] != '"') throw fail("expected a string");
    std::string out;
    for (i++; i < text.size() && text[i] != '"'; i++) {
      if (text[i] != '\\') {
        out += text[i];
        continue;
      }
      char c = text[++i];
      if (c == 'n') out += '\n';
      else if (c == 't') out += '\t';
      else if (c == 'u') {
        out += char(std::stoi(text.substr(i + 1, 4), nullptr, 16));
        i += 4;
      } else out += c;
    }
    if (i >= text.size()) throw fail("unterminated string");
    i++;
    return out;
  };
  ws();
  if (i >= text.size() || text[i] != '{') throw fail("expected '{'");
  i++;
  ws();
  if (text[i] == '}') return row;
  while (true) {
    ws();
    std::string key = str();
    ws();
    if (text[i++] != ':') throw fail("expected ':'");
    ws();
    std::string value;
    if (text[i] == '"') {
      value = str();
    } else if (text[i] == '[') {
      size_t end = text.find(']', i);
      if (end == std::string::npos) throw fail("unterminated array");
      value = text.substr(i, end + 1 - i);
      i = end + 1;
    } else {
      size_t end = text.find_first_of(",}", i);
      if (end == std::string::npos) throw fail("unterminated value");
      value = text.substr(i, end - i);
      while (!value.empty() && std::isspace((unsigned char)value.back())) value.pop_back();
      if (value == "null") value.clear();
      i = end;
    }
    row[key] = value;
    ws();
    if (text[i] == ',') {
      i++;
      continue;
    }
    if (text[i] == '}') break;
    throw fail("expected ',' or '}'");
  }
  return row;
}

inline std::vector<double> parseNumberArray(const std::string& raw) {
  std::vector<double> out;
  std::string body = raw.size() >= 2 ? raw.substr(1, raw.size() - 2) : "";
  for (const std::string& item : splitString(body, ',')) {
    if (item.find_first_not_of(" ") != std::string::npos) out.push_back(std::stod(item));
  }
  return out;
}

inline std::vector<Row> readJsonl(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot read " + path);
  std::vector<Row> rows;
  std::string line;
  while (std::getline(in, line)) {
    if (line.find_first_not_of(" \t\r") != std::string::npos) rows.push_back(parseJsonObject(line));
  }
  return rows;
}

inline std::string fmt(double value, int digits) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.*f", digits, value);
  return buf;
}

}  // namespace nvls_bench
