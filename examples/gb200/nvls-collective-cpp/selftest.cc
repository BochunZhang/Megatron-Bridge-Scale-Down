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

// CPU-only checks of bench_core.h (`make selftest`). The log lines are the exact format strings
// of NCCL 2.32.3 (enqueue.cc, scheduler/symmetric_sched.cc, ce_coll.cc, init.cc, nvls.cc)
// with values filled in; they do not prove anything about a real GPU run.

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <strings.h>

#include "bench_core.h"

using namespace nvls_bench;

static int failures = 0;

#define EXPECT(cond)                                                \
  do {                                                              \
    if (!(cond)) {                                                  \
      std::fprintf(stderr, "%s:%d: FAILED %s\n", __FILE__, __LINE__, #cond); \
      failures++;                                                   \
    }                                                               \
  } while (0)

static const char* kPrefix = "gb200-0:1234:1300 [0] NCCL INFO ";

static std::string line(const std::string& body) { return kPrefix + body; }

static void testParseTuning() {
  TuningLaunch t;
  EXPECT(parseTuningLine(line("AllGather: 8388608 Bytes -> Algo NVLS proto Simple channel{Lo..Hi}={0..15} cgaClusterSize 4"), &t));
  EXPECT(t.func == "AllGather" && t.impl == "NVLS/Simple" && t.bytes == 8388608);
  EXPECT(parseTuningLine(line("ReduceScatter: 4096 Bytes -> Algo Ring proto LL channel{Lo..Hi}={0..1} cgaClusterSize 1"), &t));
  EXPECT(t.func == "ReduceScatter" && t.impl == "Ring/LL" && t.bytes == 4096);
  EXPECT(parseTuningLine(line("AllGather [Symmetric]: 67108864 Bytes -> Kernel AllGather_TmaSTMC nchannels 16 nthreads 512 nWorks 1"), &t));
  EXPECT(t.impl == "Symmetric/AllGather_TmaSTMC" && t.bytes == 67108864);
  EXPECT(parseTuningLine(line("ReduceScatter [Symmetric]: 1024 Bytes -> Kernel ReduceScatter_LDMC nchannels 8 nthreads 512 nWorks 1"), &t));
  EXPECT(t.func == "ReduceScatter" && t.impl == "Symmetric/ReduceScatter_LDMC");
  EXPECT(parseTuningLine(line("AllGather [Copy Engine]: 268435456 Bytes -> cudaMemcpy; CE synchronization with NVLS"), &t));
  EXPECT(t.impl == "CopyEngine+NVLSSync" && t.bytes == 268435456);
  EXPECT(parseTuningLine(line("AllGather [Copy Engine]: 16 Bytes -> cudaMemcpy"), &t));
  EXPECT(t.impl == "CopyEngine");
  EXPECT(parseTuningLine(line("AlltoAll [Hierarchical CE]: 64 Bytes -> RMA proxy + CE"), &t));
  EXPECT(t.impl == "HierarchicalCE");
  EXPECT(!parseTuningLine(line("Init CE, rank 0 baseUCSymReadyPtr 0x1, baseUCSymComplPtr 0x2, seq num 1"), &t));
  EXPECT(!parseTuningLine(line("algSelection: SYMK_TmaSTMC picked within the selected set"), &t));
  EXPECT(!parseTuningLine(line("ReduceScatter_RailA2A_LsaLD : satblocks=8 bufsize=1 effbw=2"), &t));
}

static void testMatches() {
  EXPECT(implementationMatches("Ring/*", "Ring/Simple"));
  EXPECT(!implementationMatches("Ring/*", "NVLS/Simple"));
  EXPECT(!implementationMatches("NVLS/*", "NVLSTree/Simple"));
  EXPECT(implementationMatches("Symmetric/*|CopyEngine*", "CopyEngine+NVLSSync"));
  EXPECT(implementationMatches("Symmetric/*|CopyEngine*", "Symmetric/AllGather_STMC"));
  EXPECT(!implementationMatches("Symmetric/*|CopyEngine*", "Ring/Simple"));
  EXPECT(implementationMatches("Symmetric/AllGather_TmaSTMC", "Symmetric/AllGather_TmaSTMC"));
  EXPECT(!implementationMatches("Symmetric/AllGather_STMC", "Symmetric/AllGather_TmaSTMC"));
  EXPECT(!implementationMatches("CopyEngine+NVLSSync", "CopyEngine"));
}

static void testCheckLaunches() {
  const Impl& tma = *findImpl("tma");
  std::vector<std::string> lines;
  for (int i = 0; i < 3; i++) {
    lines.push_back(line("AllGather [Symmetric]: 1024 Bytes -> Kernel AllGather_TmaSTMC nchannels 2 nthreads 512 nWorks 1"));
    lines.push_back(line("ReduceScatter [Symmetric]: 1024 Bytes -> Kernel ReduceScatter_LDMC nchannels 2 nthreads 512 nWorks 1"));
    // World barrier (8 Bytes) is ignored; the start gate (4 Bytes) is counted.
    lines.push_back(line("AllReduce: 8 Bytes -> Algo Ring proto LL channel{Lo..Hi}={0..0} cgaClusterSize 1"));
    lines.push_back(line("AllReduce: 4 Bytes -> Algo Ring proto LL channel{Lo..Hi}={0..0} cgaClusterSize 1"));
  }
  std::vector<std::string> errors;
  std::string observed = checkLaunches(lines, tma, {"ag", "rs"}, 3, 1024, 3, &errors);
  EXPECT(errors.empty());
  EXPECT(observed == "AllGather=Symmetric/AllGather_TmaSTMC;ReduceScatter=Symmetric/ReduceScatter_LDMC");
  errors.clear();
  checkLaunches(lines, tma, {"ag", "rs"}, 3, 1024, 0, &errors);
  EXPECT(errors.size() == 1);  // gate lines on a rank that is not the gate's comm rank 0

  // A silent fallback to the register kernel, a wrong count and a wrong size all fail.
  lines.push_back(line("AllGather [Symmetric]: 2048 Bytes -> Kernel AllGather_STMC nchannels 2 nthreads 512 nWorks 1"));
  errors.clear();
  checkLaunches(lines, tma, {"ag", "rs"}, 3, 1024, 3, &errors);
  EXPECT(errors.size() == 3);

  const Impl& ce = *findImpl("ce");
  errors.clear();
  checkLaunches({line("AllGather [Copy Engine]: 1024 Bytes -> cudaMemcpy")}, ce, {"ag"}, 1, 1024, 0, &errors);
  EXPECT(errors.size() == 1);  // CE without NVLS sync means the comm has no LSA multimem
  errors.clear();
  checkLaunches({line("AllGather: 1024 Bytes -> Algo Ring proto Simple channel{Lo..Hi}={0..1} cgaClusterSize 1")}, ce,
                {"rs"}, 0, 1024, 0, &errors);
  EXPECT(errors.size() == 1);  // foreign collective
  const Impl& symm = *findImpl("symm");
  errors.clear();
  checkLaunches({line("AllGather [Copy Engine]: 1024 Bytes -> cudaMemcpy; CE synchronization with NVLS")}, symm, {"ag"},
                1, 1024, 0, &errors);
  EXPECT(errors.size() == 1);  // symm must stay on symmetric kernels (NCCL_SYM_CE_THRESHOLD)
}

static void testJson() {
  JsonObject r;
  r.set("algo", jStr("tma"));
  r.set("layout", kNull);
  r.set("ranks", jInts({0, 1, 2}));
  r.set("trial_makespan_ms", jNums({1.5, 2.25}));
  r.set("host_bound", jBool(false));
  r.set("nccl_impl", jStr("AllGather=Symmetric/AllGather_TmaSTMC;x\"y"));
  r.set("makespan_ms", jNum(1.875));
  r.set("algo", jStr("ce"));  // set() replaces in place
  Row row = parseJsonObject(r.dump());
  EXPECT(row["algo"] == "ce" && row["layout"].empty() && row["host_bound"] == "false");
  EXPECT(parseNumberArray(row["trial_makespan_ms"]) == std::vector<double>({1.5, 2.25}));
  EXPECT(row["nccl_impl"] == "AllGather=Symmetric/AllGather_TmaSTMC;x\"y");
  EXPECT(std::stod(row["makespan_ms"]) == 1.875);
  EXPECT(r.dump().rfind("{\"algo\": \"ce\", \"layout\": null, \"ranks\": [0, 1, 2]", 0) == 0);
  EXPECT(parseJsonObject(r.dump("  "))["ranks"] == "[0, 1, 2]");
}

static void testCommLine() {
  CommLogInfo info;
  parseCommLine(line("comm 0x5555 rank 0 nRanks 8 nNodes 1 localRanks 4 localRank 0 MNNVL 1"), &info);
  parseCommLine(line("NVLS multicast support is available on dev 0, transport MNNVL, symmetric multimem enabled"), &info);
  EXPECT(info.nNodes == 1 && info.mnnvl == 1 && info.symmetricMultimem == 1);
}

static void testHelpers() {
  auto halves = splitGroups(16, "interleaved");
  EXPECT(halves.first.size() == 8 && halves.first[1] == 2 && halves.second[0] == 1);
  halves = splitGroups(16, "contiguous");
  EXPECT(halves.first.back() == 7 && halves.second.front() == 8);
  EXPECT(caseName("single-n4", "", "ag+rs", "bf16", 64) == "single-n4_ag+rs_bf16_64MiB");
  EXPECT(caseName(scenarioName("split", 16, "contiguous"), kCrossPart, kCrossWorkload, "bf16", 1024) ==
         "split-n16-contiguous-AB-cross_agA-rsB_bf16_1024MiB");
  bool threw = false;
  try {
    caseName("split-n16-contiguous", "AB-concurrent", kCrossWorkload, "bf16", 64);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  EXPECT(threw);
  EXPECT(rankFile("symm", "single-n16_ag_bf16_1024MiB", 3, 16, ".nccl.log") ==
         "symm_single-n16_ag_bf16_1024MiB_rank03.nccl.log");
  EXPECT(rankFile("tma", "x", 7, 128, ".log") == "tma_x_rank007.log");
  EXPECT(ncclDebugPath("/r/nccl-%h-%p.log", "gb200-07.cluster", 42) == "/r/nccl-gb200-07-42.log");
  EXPECT(ncclDebugPath("a%%b%x", "h", 1) == "a%b%x");
  auto conc = trialConcurrency({{2.0, 4.0}, {1.0, 4.0}});
  EXPECT(conc.size() == 2 && conc[0] == 0.5 && conc[1] == 1.0);
  EXPECT(mean({1, 2, 3}) == 2);
  EXPECT(std::fabs(busBandwidthGBs(1e9, 4, 1.0) - 0.75) < 1e-12);
  EXPECT(median({3, 1, 2}) == 2 && median({4, 1, 2, 3}) == 2.5);
  EXPECT(parsePositiveInts("16,64,16") == std::vector<int>({16, 64}));
  EXPECT(workloadOps("ag+rs").size() == 2);
  for (const Impl& impl : impls()) {
    EXPECT(impl.agCopyEngine == (impl.ceMulticast != nullptr));
    EXPECT(!impl.needsMultimem || impl.windows);
  }
}

// ncclAlgTagMask (src/config/algorithm_registry.cc) matches a tag against the start of every
// registry name, case-insensitively. Each selection must hit only the kernel(s) the impl names.
static std::vector<std::string> tagMatches(const std::string& tag, const std::vector<std::string>& names) {
  std::vector<std::string> out;
  for (const std::string& name : names) {
    if (name.size() >= tag.size() && strncasecmp(name.c_str(), tag.c_str(), tag.size()) == 0) out.push_back(name);
  }
  return out;
}

static void testSelectionTags() {
  // The rows of algRegistry[] valid for AllGather and for ReduceScatter in NCCL 2.32.3.
  const std::vector<std::string> ag = {"RING_LL", "RING_LL128", "RING_SIMPLE", "COLLNET_DIRECT_SIMPLE", "NVLS_SIMPLE",
                                       "PAT_SIMPLE", "SYMK_LL", "SYMK_LLMC", "SYMK_TmaST", "SYMK_ST", "SYMK_TmaSTMC",
                                       "SYMK_STMC", "SYMK_RailRing_LsaSTMC"};
  const std::vector<std::string> rs = {"RING_LL", "RING_LL128", "RING_SIMPLE", "COLLNET_DIRECT_SIMPLE", "NVLS_SIMPLE",
                                       "PAT_SIMPLE", "SYMK_LL", "SYMK_TmaLD", "SYMK_LD", "SYMK_LDMC",
                                       "SYMK_RailA2A_LsaLD", "SYMK_RailA2A_LsaLDMC"};
  EXPECT(tagMatches("SYMK_ST", ag).size() == 2);  // why "SYMK_ST" alone is never used
  for (const Impl& impl : impls()) {
    if (impl.agSelection) {
      auto hits = tagMatches(impl.agSelection, ag);
      bool ring = std::string(impl.agSelection) == "RING";
      EXPECT(ring ? hits.size() == 3 : hits.size() == 1);
    }
    if (impl.rsSelection) {
      auto hits = tagMatches(impl.rsSelection, rs);
      bool ring = std::string(impl.rsSelection) == "RING";
      EXPECT(ring ? hits.size() == 3 : hits.size() == 1);
    }
  }
  EXPECT(tagMatches("SYMK_TmaSTMC", ag) == std::vector<std::string>({"SYMK_TmaSTMC"}));
  EXPECT(tagMatches("SYMK_LDMC", rs) == std::vector<std::string>({"SYMK_LDMC"}));

}

int main() {
  testParseTuning();
  testMatches();
  testCheckLaunches();
  testCommLine();
  testJson();
  testHelpers();
  testSelectionTags();
  if (failures) {
    std::fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  std::printf("selftest: all checks passed\n");
  return 0;
}
