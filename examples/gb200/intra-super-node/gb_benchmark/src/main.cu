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

// gb_benchmark: GB200 compute-tray memory/interconnect microbenchmarks, ported from gh_benchmark (GH200).
// See ../README.md for the test matrix and how to read the results.

#include <filesystem>
#include <fstream>

#include "benchmarks.hpp"

namespace {

const char* kUsage = R"(usage: gb_benchmark <command> [--option value ...]

Memory targets: cpu<i> (LPDDR of CPU NUMA node i), pinned<i> (cpu<i> + cudaHostRegister),
                hbm<g> (GPU g HBM as system memory), cuda<g> (cudaMalloc on GPU g).
                Lists are comma separated; cpu*, pinned*, hbm*, cuda* expand to all indices.
Sizes accept binary units: 64K, 256M, 4G (= 4GiB).

Commands:
  topo      Print CPU/GPU/NUMA topology.
  relay     GPU src HBM -> CPU threads (registers only) -> GPU dst HBM.
              --src-gpu 0  --dst-gpus 1,2  --cpu-nodes src,dst (src|dst|<i>)
              --threads all (all|N, list)  --kernels ldp_stp (ldp_stp,ldnp_stnp,memcpy)
  host      CPU-thread read/write/copy.
              --op read|write|copy  --src <targets>  --dst <targets>  --cpu-nodes all|<i,...>
              --threads all  --kernels ldp_stp
  device    GPU-kernel read/write/copy.
              --gpu 0  --op read|write|copy  --src <targets>  --dst <targets>
              --blocks auto (list)  --threads-per-block 512
  memcpy    cudaMemcpyAsync(cudaMemcpyDefault) matrix on a stream of --gpu.
              --gpu 0  --src <targets>  --dst <targets>
  latency   Pointer-chase latency.
              --agents cpu0,gpu0 (cpu<i>[.k], gpu<g>)  --targets cpu*,hbm*  --bytes 512M
              --slot 128  --steps all (one full cycle over the buffer)  --iters 3  --warmup 1
  pingpong  Flag ping-pong round trip between two agents.
              --pairs cpu0:cpu0.1,cpu0:cpu1,cpu0:gpu0,gpu0:cpu0,gpu0:gpu1,gpu0:gpu2
              --flags cpu0,hbm0  --rounds 10000  --warmup-rounds 1000  --iters 5  --timeout-s 30

Common options:
  --bytes 4G  --iters 10  --warmup 2  --verify full|none (copies are checked word by word)
  --hbm-backend auto|numa|managed   auto = numa_alloc_onnode on the GPU NUMA node if the driver exposes one,
                                    otherwise cudaMallocManaged preferred on the GPU (CDMM mode)
  --hbm-numa <g>=<node>,...         override the HBM NUMA node reported by CUDA
  --out <file.json>|-               write the JSON report (topology + one record per case)
)";

std::string join_argv(int argc, char** argv) {
  std::string out;
  for (int i = 0; i < argc; ++i) {
    if (i) out += " ";
    out += argv[i];
  }
  return out;
}

void write_report(const std::string& path, const std::string& json) {
  if (path.empty()) return;
  if (path == "-") {
    std::printf("%s\n", json.c_str());
    return;
  }
  const std::filesystem::path parent = std::filesystem::path(path).parent_path();
  if (!parent.empty()) std::filesystem::create_directories(parent);
  std::ofstream f(path);
  if (!f) gb::fail("cannot write " + path);
  f << json << "\n";
  std::printf("[out] wrote %s\n", path.c_str());
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help" || std::string(argv[1]) == "help") {
    std::printf("%s", kUsage);
    return argc < 2 ? 2 : 0;
  }
  const std::string command = argv[1];
  try {
    gb::Args args(argc, argv, 2);
    const std::string out = args.get("out", "");
    const gb::Topology topo = gb::discover_topology(args);
    topo.print_summary(stdout);

    std::vector<std::string> results;
    if (command == "topo") {
      args.finish();
    } else if (command == "relay") {
      const auto o = gb::parse_relay(args);
      args.finish();
      results = gb::run_relay(topo, o);
    } else if (command == "host") {
      const auto o = gb::parse_host(args, topo);
      args.finish();
      results = gb::run_host(topo, o);
    } else if (command == "device") {
      const auto o = gb::parse_device(args, topo);
      args.finish();
      results = gb::run_device(topo, o);
    } else if (command == "memcpy") {
      const auto o = gb::parse_memcpy(args, topo);
      args.finish();
      results = gb::run_memcpy(topo, o);
    } else if (command == "latency") {
      const auto o = gb::parse_latency(args, topo);
      args.finish();
      results = gb::run_latency(topo, o);
    } else if (command == "pingpong") {
      const auto o = gb::parse_pingpong(args, topo);
      args.finish();
      results = gb::run_pingpong(topo, o);
    } else {
      gb::fail("unknown command '" + command + "' (see --help)");
    }

    write_report(out, gb::JsonObject()
                          .str("tool", "gb_benchmark")
                          .str("command", command)
                          .str("argv", join_argv(argc, argv))
                          .str("timestamp", gb::timestamp_iso())
                          .raw("topology", topo.to_json())
                          .raw("results", gb::json_array(results))
                          .dump());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[error] %s\n", e.what());
    return 1;
  }
  return 0;
}
