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

// Merge the per-impl results.jsonl files of one run into compare.md (the tables of
// ../nvls-collective/collective_metrics.py, plus 2a: TMA / CE AllGather against symmetric).
//
//   collective_report --input ring=<dir>/ring/results.jsonl --input tma=<dir>/tma/results.jsonl ... \
//       --output <dir>/compare.md
//
// Every cell except table 0 is a trial median from the benchmark; a missing combination prints "-".

#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "bench_core.h"

using namespace nvls_bench;

namespace {

using Opt = std::optional<double>;

std::string implName(int rank) { return rank < int(impls().size()) ? impls()[rank].name : "?"; }

double num(const Row& row, const std::string& key) { return std::stod(row.at(key)); }

Opt optNum(const Row* row, const std::string& key) {
  if (!row) return std::nullopt;
  auto it = row->find(key);
  if (it == row->end() || it->second.empty()) return std::nullopt;
  return std::stod(it->second);
}

std::string cell(Opt value, int digits = 1) { return value ? fmt(*value, digits) : "-"; }

void table(std::ostream& out, const std::vector<std::string>& header, const std::vector<std::vector<std::string>>& rows) {
  out << "|";
  for (const std::string& h : header) out << " " << h << " |";
  out << "\n|";
  for (size_t i = 0; i < header.size(); i++) out << "---:|";
  out << "\n";
  for (const auto& row : rows) {
    out << "|";
    for (const std::string& c : row) out << " " << c << " |";
    out << "\n";
  }
}

// (impl, group_size, size_bytes, workload) -> single-scenario row.
using SingleKey = std::tuple<std::string, int, size_t, std::string>;

std::map<SingleKey, const Row*> singleIndex(const std::vector<Row>& rows) {
  std::map<SingleKey, const Row*> index;
  for (const Row& r : rows) {
    if (r.at("scenario") != "single") continue;
    index[{r.at("algo"), std::stoi(r.at("group_size")), std::stoull(r.at("size_bytes")), r.at("workload")}] = &r;
  }
  return index;
}

const Row* lookup(const std::map<SingleKey, const Row*>& index, const SingleKey& key) {
  auto it = index.find(key);
  return it == index.end() ? nullptr : it->second;
}

// Shapes (group_size, size_bytes, workload) in report order.
std::vector<std::tuple<int, size_t, std::string>> shapes(const std::map<SingleKey, const Row*>& index) {
  std::set<std::tuple<int, size_t, int>> keys;
  for (const auto& item : index) {
    keys.insert({std::get<1>(item.first), std::get<2>(item.first), workloadRank(std::get<3>(item.first))});
  }
  std::vector<std::tuple<int, size_t, std::string>> out;
  for (const auto& k : keys) out.push_back({std::get<0>(k), std::get<1>(k), workloads()[std::get<2>(k)]});
  return out;
}

std::vector<std::string> presentImpls(const std::vector<Row>& rows) {
  std::set<int> seen;
  for (const Row& r : rows) seen.insert(implRank(r.at("algo")));
  std::vector<std::string> out;
  for (int s : seen) out.push_back(implName(s));
  return out;
}

Opt aggregate(const Row* row) { return optNum(row, "aggregate_busbw_gb_s"); }

std::map<std::string, std::string> kernelsOf(const Row* row) {
  return row ? recordKernels(row->at("nccl_impl"), row->at("dtype")) : std::map<std::string, std::string>{};
}

std::string opLabel(const std::string& op) {
  std::string label = op;
  for (char& c : label) c = char(std::toupper((unsigned char)c));
  return label;
}

// All kernels of a record: the single op's kernel, or "AG <k>; RS <k>" for the ag+rs pair.
std::string kernelSummary(const Row& row) {
  auto kernels = kernelsOf(&row);
  if (kernels.empty()) return "-";
  if (kernels.size() == 1) return kernels.begin()->second;
  std::string out;
  for (const std::string& op : {"ag", "rs", "ar"}) {
    if (kernels.count(op)) out += (out.empty() ? "" : "; ") + opLabel(op) + " " + kernels.at(op);
  }
  return out;
}

// Kernel of op in its isolated case; the AG+RS pair is added only if it ran another one.
std::string kernelCell(const std::string& op, const Row* isolated, const Row* pair = nullptr) {
  auto alone = kernelsOf(isolated), together = kernelsOf(pair);
  std::string a = alone.count(op) ? alone.at(op) : "", t = together.count(op) ? together.at(op) : "";
  if (!t.empty() && t != a) return (a.empty() ? "-" : a) + "（pair：" + t + "）";
  return a.empty() ? "-" : a;
}

// Table 0: every test and group with its completion time per iteration over the trials. The
// bandwidth columns use the mean, every op of the workload contributing its own bus traffic:
// size * (n-1)/n for AG and RS, twice that for AR.
void completionTable(std::ostream& out, const std::vector<Row>& rows) {
  std::vector<const Row*> ordered;
  for (const Row& r : rows) ordered.push_back(&r);
  std::sort(ordered.begin(), ordered.end(), [](const Row* a, const Row* b) {
    return std::make_tuple(implRank(a->at("algo")), a->at("case"), a->at("group_name")) <
           std::make_tuple(implRank(b->at("algo")), b->at("case"), b->at("group_name"));
  });
  std::vector<std::vector<std::string>> body;
  for (const Row* r : ordered) {
    std::vector<double> trials = parseNumberArray(r->at("trial_makespan_ms"));
    double meanMs = optNum(r, "mean_makespan_ms").value_or(mean(trials));
    size_t bytes = std::stoull(r->at("size_bytes"));
    int n = std::stoi(r->at("group_size"));
    std::vector<std::string> ops = workloadOps(r->at("workload"));
    double algbw = ops.size() * algBandwidthGBs(double(bytes), meanMs / 1e3);
    double busbw = 0;
    for (const std::string& op : ops) busbw += opBusBandwidthGBs(op, double(bytes), n, meanMs / 1e3);
    body.push_back({r->at("algo"), r->at("case"), r->at("group_name"), r->at("phase"), std::to_string(n),
                    r->at("workload"), std::to_string(bytes / kBytesPerMiB), std::to_string(trials.size()) + "x" + r->at("iters"),
                    fmt(meanMs, 3), fmt(num(*r, "makespan_ms"), 3), fmt(*std::min_element(trials.begin(), trials.end()), 3),
                    fmt(*std::max_element(trials.begin(), trials.end()), 3), fmt(algbw, 1), fmt(busbw, 1),
                    cell(optNum(r, "concurrency"), 3), kernelSummary(*r)});
  }
  table(out, {"algo", "case", "group", "phase", "ranks", "workload", "MiB", "trials x iters", "mean ms", "median ms",
              "min ms", "max ms", "algbw", "busbw", "concurrency", "kernel"},
        body);
}

// Table 1: isolated AG / RS / AR versus AG and RS issued together on one group.
void singleTable(std::ostream& out, const std::vector<Row>& rows) {
  auto index = singleIndex(rows);
  std::set<std::tuple<int, int, size_t>> keys;
  for (const auto& item : index) {
    keys.insert({implRank(std::get<0>(item.first)), std::get<1>(item.first), std::get<2>(item.first)});
  }
  std::vector<std::vector<std::string>> body;
  for (const auto& k : keys) {
    std::string impl = implName(std::get<0>(k));
    int n = std::get<1>(k);
    size_t bytes = std::get<2>(k);
    const Row* ag = lookup(index, {impl, n, bytes, "ag"});
    const Row* rs = lookup(index, {impl, n, bytes, "rs"});
    const Row* ar = lookup(index, {impl, n, bytes, "ar"});
    const Row* both = lookup(index, {impl, n, bytes, "ag+rs"});
    Opt speedup, overlap;
    if (ag && rs && both) {
      double a = num(*ag, "makespan_ms"), r = num(*rs, "makespan_ms"), b = num(*both, "makespan_ms");
      speedup = (a + r) / b;
      overlap = (a + r - b) / std::min(a, r);
    }
    body.push_back({impl, std::to_string(n), std::to_string(bytes / kBytesPerMiB), cell(optNum(ag, "ag_busbw_gb_s")),
                    cell(optNum(rs, "rs_busbw_gb_s")), cell(optNum(ar, "ar_busbw_gb_s")),
                    cell(optNum(both, "ag_busbw_gb_s")), cell(optNum(both, "rs_busbw_gb_s")), cell(aggregate(both)),
                    cell(speedup, 2), cell(overlap, 2), kernelCell("ag", ag, both), kernelCell("rs", rs, both),
                    kernelCell("ar", ar)});
  }
  table(out, {"algo", "ranks", "MiB", "AG busbw", "RS busbw", "AR busbw", "AG in pair", "RS in pair", "pair aggregate",
              "speedup", "overlap_fraction", "AG kernel", "RS kernel", "AR kernel"},
        body);
}

void bestTable(std::ostream& out, const std::vector<Row>& rows, const std::vector<std::string>& present) {
  auto index = singleIndex(rows);
  std::vector<std::vector<std::string>> body;
  for (const auto& shape : shapes(index)) {
    std::vector<std::string> line = {std::to_string(std::get<0>(shape)), std::to_string(std::get<1>(shape) / kBytesPerMiB),
                                     std::get<2>(shape)};
    std::optional<std::pair<double, std::string>> best;
    for (const std::string& impl : present) {
      Opt bw = aggregate(lookup(index, {impl, std::get<0>(shape), std::get<1>(shape), std::get<2>(shape)}));
      line.push_back(cell(bw));
      if (bw && (!best || *bw > best->first)) best = std::make_pair(*bw, impl);
    }
    line.push_back(best ? best->second : "-");
    body.push_back(line);
  }
  std::vector<std::string> header = {"ranks", "MiB", "workload"};
  header.insert(header.end(), present.begin(), present.end());
  header.push_back("best");
  table(out, header, body);
}

// One column pair per impl in `others`, as aggregate busbw and its ratio against `base`.
void ratioTable(std::ostream& out, const std::vector<Row>& rows, const std::vector<std::string>& present,
                const std::string& base, const std::vector<std::string>& others) {
  auto index = singleIndex(rows);
  std::vector<std::string> header = {"ranks", "MiB", "workload", base};
  std::vector<std::string> cols;
  for (const std::string& impl : others) {
    if (impl == base || std::find(present.begin(), present.end(), impl) == present.end()) continue;
    cols.push_back(impl);
    header.push_back(impl);
    header.push_back(impl + "/" + base);
  }
  std::vector<std::vector<std::string>> body;
  for (const auto& shape : shapes(index)) {
    Opt baseBw = aggregate(lookup(index, {base, std::get<0>(shape), std::get<1>(shape), std::get<2>(shape)}));
    std::vector<std::string> line = {std::to_string(std::get<0>(shape)), std::to_string(std::get<1>(shape) / kBytesPerMiB),
                                     std::get<2>(shape), cell(baseBw)};
    for (const std::string& impl : cols) {
      Opt bw = aggregate(lookup(index, {impl, std::get<0>(shape), std::get<1>(shape), std::get<2>(shape)}));
      line.push_back(cell(bw));
      line.push_back(bw && baseBw ? fmt(*bw / *baseBw, 2) : "-");
    }
    body.push_back(line);
  }
  table(out, header, body);
}

void kernelTable(std::ostream& out, const std::vector<Row>& rows) {
  std::map<std::pair<int, std::string>, std::vector<const Row*>> groups;
  for (const Row& r : rows) groups[{implRank(r.at("algo")), r.at("group_name")}].push_back(&r);
  auto join = [](const std::set<std::string>& s, const char* sep) {
    std::string o;
    for (const std::string& v : s) o += (o.empty() ? "" : sep) + v;
    return o.empty() ? std::string("-") : o;
  };
  std::vector<std::vector<std::string>> body;
  for (const auto& item : groups) {
    std::set<std::string> evidence, nodes, lsa, multimem;
    for (const Row* r : item.second) {
      if (!r->at("nccl_impl").empty()) evidence.insert(r->at("nccl_impl"));
      if (!r->at("n_nodes").empty()) nodes.insert(r->at("n_nodes"));
      if (!r->at("n_lsa_teams").empty()) lsa.insert(r->at("n_lsa_teams"));
      if (!r->at("multimem_support").empty()) multimem.insert(r->at("multimem_support"));
    }
    body.push_back({item.second.front()->at("algo"), item.first.second, join(nodes, ","), join(lsa, ","),
                    join(multimem, ","), join(evidence, "; ")});
  }
  table(out, {"algo", "group", "nNodes", "nLsaTeams", "multimem", "kernel evidence"}, body);
}

// (impl, total_ranks, layout, size_bytes)
using SplitShape = std::tuple<int, int, std::string, size_t>;

SplitShape splitShape(const Row& r) {
  return {implRank(r.at("algo")), std::stoi(r.at("total_ranks")), r.at("layout"), std::stoull(r.at("size_bytes"))};
}

// Table 3: both half-groups at the same time versus one group of the same size alone.
// The isolated reference of either half of a T-rank split is the single group of T/2 ranks with
// the same size and workload; slowdown = T_concurrent / T_single.
void splitTable(std::ostream& out, const std::vector<Row>& rows) {
  auto single = singleIndex(rows);
  using Key = std::tuple<SplitShape, int, int>;  // shape, workload, group_index
  std::map<Key, const Row*> index;
  for (const Row& r : rows) {
    if (r.at("scenario") != "split" || r.at("phase") != "concurrent") continue;
    index[{splitShape(r), workloadRank(r.at("workload")), std::stoi(r.at("group_index"))}] = &r;
  }
  std::vector<std::vector<std::string>> body;
  for (const auto& item : index) {
    const SplitShape& s = std::get<0>(item.first);
    const std::string& workload = workloads()[std::get<1>(item.first)];
    const Row* conc = item.second;
    const Row* iso = lookup(single, {implName(std::get<0>(s)), std::get<1>(s) / 2, std::get<3>(s), workload});
    Opt slowdown;
    if (iso && conc) slowdown = num(*conc, "makespan_ms") / num(*iso, "makespan_ms");
    body.push_back({implName(std::get<0>(s)), std::to_string(std::get<1>(s)), std::get<2>(s),
                    std::to_string(std::get<3>(s) / kBytesPerMiB), workload,
                    std::string(1, "AB"[std::get<2>(item.first)]), cell(aggregate(iso)), cell(aggregate(conc)),
                    cell(slowdown, 3), cell(optNum(conc, "concurrency"), 3)});
  }
  table(out, {"algo", "total", "layout", "MiB", "workload", "group", "isolated busbw", "concurrent busbw", "slowdown",
              "concurrency"},
        body);
}

// Table 4: group A all-gathers while group B reduce-scatters, each against a single group of
// T/2 ranks running the same op alone (single-n<T/2> ag for A, rs for B).
void crossTable(std::ostream& out, const std::vector<Row>& rows) {
  auto single = singleIndex(rows);
  using Key = std::tuple<SplitShape, int, std::string>;  // shape, group_index, workload
  std::map<Key, const Row*> index;
  std::set<SplitShape> crossShapes;
  for (const Row& r : rows) {
    if (r.at("scenario") != "split" || r.at("phase") != "cross") continue;
    index[{splitShape(r), std::stoi(r.at("group_index")), r.at("workload")}] = &r;
    crossShapes.insert(splitShape(r));
  }
  std::vector<std::vector<std::string>> body;
  for (const SplitShape& s : crossShapes) {
    std::vector<std::string> line = {implName(std::get<0>(s)), std::to_string(std::get<1>(s)), std::get<2>(s),
                                     std::to_string(std::get<3>(s) / kBytesPerMiB)};
    double crossBw = 0;
    Opt concurrency;
    for (int g = 0; g < 2; g++) {
      const std::string& w = crossGroupWorkloads()[g];
      auto it = index.find({s, g, w});
      const Row* cross = it == index.end() ? nullptr : it->second;
      const Row* iso = lookup(single, {implName(std::get<0>(s)), std::get<1>(s) / 2, std::get<3>(s), w});
      Opt slowdown;
      if (cross && iso) slowdown = num(*cross, "makespan_ms") / num(*iso, "makespan_ms");
      if (cross) {
        crossBw += num(*cross, "aggregate_busbw_gb_s");
        if (!concurrency) concurrency = optNum(cross, "concurrency");
      }
      line.push_back(cell(aggregate(iso)));
      line.push_back(cell(aggregate(cross)));
      line.push_back(cell(optNum(cross, "makespan_ms"), 3));
      line.push_back(cell(slowdown, 3));
    }
    line.push_back(fmt(crossBw, 1));
    line.push_back(cell(concurrency, 3));
    body.push_back(line);
  }
  table(out, {"algo", "total", "layout", "MiB", "A AG isolated busbw", "A AG cross busbw", "A AG cross ms", "A slowdown",
              "B RS isolated busbw", "B RS cross busbw", "B RS cross ms", "B slowdown", "A+B cross busbw", "concurrency"},
        body);
}

int usage() {
  std::fprintf(stderr, "usage: collective_report --input IMPL=results.jsonl [...] --output compare.md\n");
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::pair<std::string, std::string>> inputs;
  std::string output;
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--input" && i + 1 < argc) {
      std::string v = argv[++i];
      size_t eq = v.find('=');
      if (eq == std::string::npos || !findImpl(v.substr(0, eq))) {
        std::fprintf(stderr, "--input expects IMPL=PATH, got %s\n", v.c_str());
        return 2;
      }
      inputs.push_back({v.substr(0, eq), v.substr(eq + 1)});
    } else if (arg == "--output" && i + 1 < argc) {
      output = argv[++i];
    } else {
      return usage();
    }
  }
  if (inputs.empty() || output.empty()) return usage();

  std::vector<Row> rows;
  try {
    for (const auto& input : inputs) {
      for (Row& r : readJsonl(input.second)) {
        if (r.at("algo") != input.first) {
          std::fprintf(stderr, "%s holds records for %s, expected only %s\n", input.second.c_str(),
                       r.at("algo").c_str(), input.first.c_str());
          return 2;
        }
        rows.push_back(r);
      }
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 2;
  }

  std::vector<std::string> present = presentImpls(rows);
  std::set<std::string> profiled;
  for (const Row& r : rows) {
    if (r.at("profiled") == "true") profiled.insert(r.at("algo"));
  }
  std::ofstream out(output);
  out << "# AG / RS / AR on GB200 with NCCL 2.32.3: ring vs NVLS vs symmetric kernels (multimem / TMA / CE)\n\n"
      << "计时：每个 trial 先过一次全 scenario 的 GPU 同步起点（1 元素 all-reduce），再背靠背发 iters 次；\n"
      << "t = (该 rank 所有 op 完成 − 起点) / iters，trial 内取组内最慢 rank。除表 0 外各表取 trials 的中位数。\n\n"
      << "换算：size = 完整张量（AG 每卡接收总量 / RS 每卡输入总量 / AR 每卡归约总量），十进制 GB/s：\n"
      << "AG、RS 每卡 shard = size/n，busbw = size·(n−1)/n / t（nccl-tests 定义，= 每卡实际收/发字节数 / t）；\n"
      << "AR 没有分片，每卡归约整块 size，busbw = 2·size·(n−1)/n / t（收、发各算一次，nccl-tests 的 all-reduce 口径）；\n"
      << "ag+rs 的 aggregate busbw = 两个 op 的 busbw 之和 / makespan；不同组的带宽不相加，需要时自己把各组行相加。\n"
      << "concurrency = min(T_A, T_B) / max(T_A, T_B)：两组从同一起点出发，1 表示两组全程同时在跑。\n\n";
  if (!profiled.empty()) {
    std::string list;
    for (const std::string& p : profiled) list += (list.empty() ? "" : ", ") + p;
    out << "> 注意：" << list << " 是 `--nsys` 采集下的结果，含 Nsight 开销，只用于看 trace，"
        << "不要和非 profile 的数字放进同一张结论表。\n\n";
  }
  out << "## 0. 每个测试的完成时间\n\n"
      << "mean/median/min/max：各 trial（每个 trial 取组内最慢 rank）的每次迭代完成时间，单位 ms；\n"
      << "本表的 algbw/busbw 用 mean 计算。\n\n"
      << "kernel：该组 comm rank 0 的 NCCL TUNING 日志里，本 case 每一次 launch（含 warmup）记录的实现，\n"
      << "换算成 NCCL 2.32.3 实际运行的函数名，不是由环境变量推断：\n"
      << "对称路径是对称内核 `ncclSymkDevKernel_<Coll>_<algo>`（launch 的就是它）；ring/nvls 是经典路径真正执行的\n"
      << "设备函数 `ncclDevFunc_<Coll>_<ALGO>_<PROTO>`——经典路径每个 collective 只 launch 一个特化 kernel\n"
      << "（如 `ncclDevKernel_AllGather_RING_LL`，nsys 里看到的就是这个名字），再按 funcId 分发到该设备函数，\n"
      << "所以 nvls 的 AG 在 nsys 里也叫 `*_RING_LL`。copy-engine 的 AllGather 是 cudaMemcpy，没有 kernel。\n"
      << "一个 op 出现多种实现时用 `\\|` 全部列出；`-` 表示没有日志证据。\n\n";
  completionTable(out, rows);
  out << "\n## 1. AG、RS、AR 单独与 AG+RS 同时发起\n\n"
      << "speedup = (T_AG + T_RS) / T_AG+RS；1 表示完全串行，overlap_fraction = 1 表示短的操作被完全隐藏。\n"
      << "AR busbw 已按 all-reduce 口径乘 2，可直接和 AG/RS 的 busbw 比。\n"
      << "AG/RS/AR kernel：单独跑该 op 那个 case 的实际函数（来源同表 0）；AG+RS 同时发起时若换了实现，括号里注明。\n\n";
  singleTable(out, rows);
  out << "\n## 1b. 各实现的绝对吞吐（aggregate busbw）\n\n";
  bestTable(out, rows, present);
  out << "\n## 2. 各实现相对 ring（aggregate busbw）\n\n"
      << "symmetric / symmetric-tma / copy-engine 是 window 注册的对称内存路径，与 ring/NVLS 走不同内核家族；"
      << "它们的绝对列见 1b，上面的比值只是和 ring 放在同一口径下的对照。\n\n";
  std::vector<std::string> all;
  for (const Impl& impl : impls()) all.push_back(impl.name);
  ratioTable(out, rows, present, "ring", all);
  out << "\n## 2a. TMA / CE AllGather 相对寄存器 multimem.st（symmetric）\n\n"
      << "三个对称实现的 ReduceScatter 都固定为 ReduceScatter_LDMC（multimem.ld_reduce），symmetric-tma 与 copy-engine "
      << "只测 ag 和 ag+rs，差异只来自 AllGather 的数据搬运方式：symmetric = SM 寄存器 multimem.st，"
      << "symmetric-tma = TMA cp.async.bulk 写 multicast 地址，copy-engine = copy engine 写 multicast 地址。\n\n";
  ratioTable(out, rows, present, "symmetric", {"symmetric-tma", "copy-engine"});
  out << "\n### 2b. 实际执行的内核（来自 NCCL TUNING 日志，非环境变量推断）\n\n"
      << "每个 case 的每次 launch 都已核对：ring 只能是 Ring/*，nvls 只能是 NVLS/*，symmetric-tma 只能是 "
      << "AllGather_TmaSTMC，copy-engine 只能是带 NVLS 同步的 Copy Engine 等（`bench_core.h` `impls()`），"
      << "任何回退到其他实现的 launch 都会让该 case 失败，不会出现在这里。\n\n";
  kernelTable(out, rows);
  out << "\n## 3. 两个半组同时运行\n\n"
      << "T 卡切成两个 T/2 组同时跑同一负载；isolated 取 single 里同样 T/2 卡、同大小、同负载的结果（rank 0..T/2-1）。\n"
      << "slowdown = T_concurrent / T_isolated，1 表示没有相互干扰。\n\n";
  splitTable(out, rows);
  out << "\n## 4. 两组同时：A 组 all-gather，B 组 reduce-scatter\n\n"
      << "isolated 取 single 里 T/2 卡单独跑同一个 op（A 对 ag、B 对 rs）；"
      << "slowdown = T_cross / T_isolated；A+B busbw 是两组 busbw 之和。\n\n";
  crossTable(out, rows);
  std::printf("wrote %s (%zu records)\n", output.c_str(), rows.size());
  return 0;
}
