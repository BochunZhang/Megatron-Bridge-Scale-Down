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
// ../nvls-collective/collective_metrics.py, plus 2a: TMA / CE AllGather against stmc).
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

// Table 0: every test and group with its completion time per iteration over the trials. The
// bandwidth columns use the mean, every op of the workload contributing size (algbw).
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
    double algbw = workloadOps(r->at("workload")).size() * algBandwidthGBs(double(bytes), meanMs / 1e3);
    body.push_back({r->at("algo"), r->at("case"), r->at("group_name"), r->at("phase"), std::to_string(n),
                    r->at("workload"), std::to_string(bytes / kBytesPerMiB), std::to_string(trials.size()) + "x" + r->at("iters"),
                    fmt(meanMs, 3), fmt(num(*r, "makespan_ms"), 3), fmt(*std::min_element(trials.begin(), trials.end()), 3),
                    fmt(*std::max_element(trials.begin(), trials.end()), 3), fmt(algbw, 1),
                    fmt(algbw * (n - 1) / n, 1), cell(optNum(r, "concurrency"), 3)});
  }
  table(out, {"algo", "case", "group", "phase", "ranks", "workload", "MiB", "trials x iters", "mean ms", "median ms",
              "min ms", "max ms", "algbw", "busbw", "concurrency"},
        body);
}

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
    const Row* both = lookup(index, {impl, n, bytes, "ag+rs"});
    Opt speedup, overlap;
    if (ag && rs && both) {
      double a = num(*ag, "makespan_ms"), r = num(*rs, "makespan_ms"), b = num(*both, "makespan_ms");
      speedup = (a + r) / b;
      overlap = (a + r - b) / std::min(a, r);
    }
    body.push_back({impl, std::to_string(n), std::to_string(bytes / kBytesPerMiB), cell(optNum(ag, "ag_busbw_gb_s")),
                    cell(optNum(rs, "rs_busbw_gb_s")), cell(optNum(both, "ag_busbw_gb_s")),
                    cell(optNum(both, "rs_busbw_gb_s")), cell(aggregate(both)), cell(speedup, 2), cell(overlap, 2)});
  }
  table(out, {"algo", "ranks", "MiB", "AG busbw", "RS busbw", "AG in pair", "RS in pair", "pair aggregate", "speedup",
              "overlap_fraction"},
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

void splitTable(std::ostream& out, const std::vector<Row>& rows) {
  using Key = std::tuple<SplitShape, int, int>;  // shape, workload, group_index
  std::map<Key, std::map<std::string, const Row*>> index;
  for (const Row& r : rows) {
    if (r.at("scenario") != "split" || (r.at("phase") != "isolated" && r.at("phase") != "concurrent")) continue;
    index[{splitShape(r), workloadRank(r.at("workload")), std::stoi(r.at("group_index"))}][r.at("phase")] = &r;
  }
  std::vector<std::vector<std::string>> body;
  for (const auto& item : index) {
    const Row* iso = item.second.count("isolated") ? item.second.at("isolated") : nullptr;
    const Row* conc = item.second.count("concurrent") ? item.second.at("concurrent") : nullptr;
    Opt slowdown;
    if (iso && conc) slowdown = num(*conc, "makespan_ms") / num(*iso, "makespan_ms");
    const SplitShape& s = std::get<0>(item.first);
    body.push_back({implName(std::get<0>(s)), std::to_string(std::get<1>(s)), std::get<2>(s),
                    std::to_string(std::get<3>(s) / kBytesPerMiB), workloads()[std::get<1>(item.first)],
                    std::string(1, "AB"[std::get<2>(item.first)]), cell(aggregate(iso)), cell(aggregate(conc)),
                    cell(slowdown, 3), cell(optNum(conc, "concurrency"), 3)});
  }
  table(out, {"algo", "total", "layout", "MiB", "workload", "group", "isolated busbw", "concurrent busbw", "slowdown",
              "concurrency"},
        body);
}

// Table 4: group A all-gathers while group B reduce-scatters, each against the same group running
// the same op alone (split A-isolated ag, B-isolated rs).
void crossTable(std::ostream& out, const std::vector<Row>& rows) {
  using Key = std::tuple<SplitShape, std::string, int, std::string>;  // shape, phase, group_index, workload
  std::map<Key, const Row*> index;
  std::set<SplitShape> crossShapes;
  for (const Row& r : rows) {
    if (r.at("scenario") != "split") continue;
    index[{splitShape(r), r.at("phase"), std::stoi(r.at("group_index")), r.at("workload")}] = &r;
    if (r.at("phase") == "cross") crossShapes.insert(splitShape(r));
  }
  auto get = [&](const Key& k) -> const Row* {
    auto it = index.find(k);
    return it == index.end() ? nullptr : it->second;
  };
  std::vector<std::vector<std::string>> body;
  for (const SplitShape& s : crossShapes) {
    std::vector<std::string> line = {implName(std::get<0>(s)), std::to_string(std::get<1>(s)), std::get<2>(s),
                                     std::to_string(std::get<3>(s) / kBytesPerMiB)};
    double crossBw = 0;
    Opt concurrency;
    for (int g = 0; g < 2; g++) {
      const std::string& w = crossGroupWorkloads()[g];
      const Row* cross = get({s, "cross", g, w});
      const Row* iso = get({s, "isolated", g, w});
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
  out << "# AG / RS on GB200 with NCCL 2.32.3: ring vs NVLS vs symmetric kernels (STMC / TMA / CE)\n\n"
      << "计时：每个 trial 先过一次全 scenario 的 GPU 同步起点（1 元素 all-reduce），再背靠背发 iters 次；\n"
      << "t = (该 rank 所有 op 完成 − 起点) / iters，trial 内取组内最慢 rank。除表 0 外各表取 trials 的中位数。\n\n"
      << "换算：size = 完整张量（AG 每卡接收总量 / RS 每卡输入总量），每卡 shard = size/n，十进制 GB/s：\n"
      << "algbw = size / t；busbw = size·(n−1)/n / t（nccl-tests 定义，= 每卡实际收/发字节数 / t）；\n"
      << "ag+rs 的 aggregate busbw = 两个 op 的 size·(n−1)/n 之和 / makespan；不同组的带宽不相加，需要时自己把各组行相加。\n"
      << "concurrency = min(T_A, T_B) / max(T_A, T_B)：两组从同一起点出发，1 表示两组全程同时在跑。\n\n";
  if (!profiled.empty()) {
    std::string list;
    for (const std::string& p : profiled) list += (list.empty() ? "" : ", ") + p;
    out << "> 注意：" << list << " 是 `--nsys` 采集下的结果，含 Nsight 开销，只用于看 trace，"
        << "不要和非 profile 的数字放进同一张结论表。\n\n";
  }
  out << "## 0. 每个测试的完成时间\n\n"
      << "mean/median/min/max：各 trial（每个 trial 取组内最慢 rank）的每次迭代完成时间，单位 ms；\n"
      << "本表的 algbw/busbw 用 mean 计算。\n\n";
  completionTable(out, rows);
  out << "\n## 1. AG、RS 单独与同时发起\n\n"
      << "speedup = (T_AG + T_RS) / T_AG+RS；1 表示完全串行，overlap_fraction = 1 表示短的操作被完全隐藏。\n\n";
  singleTable(out, rows);
  out << "\n## 1b. 各实现的绝对吞吐（aggregate busbw）\n\n";
  bestTable(out, rows, present);
  out << "\n## 2. 各实现相对 ring（aggregate busbw）\n\n";
  std::vector<std::string> all;
  for (const Impl& impl : impls()) all.push_back(impl.name);
  ratioTable(out, rows, present, "ring", all);
  out << "\n## 2a. TMA / CE AllGather 相对寄存器 multimem.st（stmc）\n\n"
      << "四个实现的 ReduceScatter 都固定为 ReduceScatter_LDMC（multimem.ld_reduce），所以 rs 行应接近 1，"
      << "ag 与 ag+rs 行的差异只来自 AllGather 的数据搬运方式：stmc = SM 寄存器 multimem.st，"
      << "tma = TMA cp.async.bulk 写 multicast 地址，ce = copy engine 写 multicast 地址，ce-uc = copy engine 写 N-1 份 unicast。\n\n";
  ratioTable(out, rows, present, "stmc", {"tma", "ce", "ce-uc"});
  out << "\n### 2b. 实际执行的内核（来自 NCCL TUNING 日志，非环境变量推断）\n\n"
      << "每个 case 的每次 launch 都已核对：每个实现只接受它固定的内核（tma 只能是 AllGather_TmaSTMC，"
      << "ce 只能是带 NVLS 同步的 Copy Engine 等），不符合的 case 会让整次运行失败，不会出现在这里。\n\n";
  kernelTable(out, rows);
  out << "\n## 3. 两个半组同时运行同一负载\n\n"
      << "slowdown = T_concurrent / T_isolated（同一组自身的 makespan），1 表示没有相互干扰。\n\n";
  splitTable(out, rows);
  out << "\n## 4. 两组同时：A 组 all-gather，B 组 reduce-scatter\n\n"
      << "isolated 是同一组单独跑同一个 op（split 的 A-isolated ag、B-isolated rs）；"
      << "slowdown = T_cross / T_isolated；A+B busbw 是两组 busbw 之和。\n\n";
  crossTable(out, rows);
  std::printf("wrote %s (%zu records)\n", output.c_str(), rows.size());
  return 0;
}
