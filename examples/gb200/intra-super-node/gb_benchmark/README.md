# gb_benchmark：GB200 compute tray 内存与互连微基准

日期：2026-10-12。本目录在 `gh_benchmark`（GH200，原始代码的本地参考副本未纳入 git，
[arXiv 2408.11556](https://arxiv.org/abs/2408.11556)）的基础上，针对 GB200 重写：拓扑在运行时发现，
并新增 CPU 中转的 GPU→GPU 拷贝测试。目前已用 CUDA 13.0 交叉编译出 `sm_100` / aarch64 版本，
**尚无 GB200 实测结果**；验证范围见文末。

## 拓扑与核心测试

一个 GB200 compute tray 有 2 个 Grace CPU 和 4 个 B200 GPU，每个 Grace 通过 NVLink-C2C
连接两个 GPU：

```text
GPU0 ─C2C─┐            ┌─C2C─ GPU2
          ├─ CPU0 ── CPU1 ─┤
GPU1 ─C2C─┘            └─C2C─ GPU3
```

图中的 GPU 编号是 CUDA ordinal。哪个 GPU 挂在哪个 CPU 上，由 `cudaDevAttrHostNumaId` 在运行时确定，
结果写在 `topology.json` 和每条 relay 记录里，不依赖上图的假设。

`relay` 测 CPU 中转的 GPU→GPU 拷贝，默认测两条路径：

| 路径 | 数据流 | `path_class` | 拷贝线程所在 CPU |
| --- | --- | --- | --- |
| GPU0 → GPU1 | GPU0 HBM ─C2C→ CPU0 ─C2C→ GPU1 HBM | `same_cpu` | CPU0 |
| GPU0 → GPU2 | GPU0 HBM ─C2C→ CPU0 → CPU1 ─C2C→ GPU2 HBM | `cross_cpu` | 分别测 CPU0（`src_attached`）和 CPU1（`dst_attached`） |

数据不经过 CPU 内存，只在 CPU 中转：

- 源和目的都是 HBM 页。分配方式是对 GPU 的 NUMA 节点调用 `numa_alloc_onnode`，并抽样检查页确实落在该节点。
- 每个 CPU 线程每次从源 HBM 读 64 B 到寄存器，再直接写到目的 HBM，没有 LPDDR 中间缓冲。
- CPU cache 仍按一致性协议参与。`ldnp_stnp` 内核用 non-temporal 指令，提示不在 cache 中保留这些行。
- 测试不调用 `cudaMemcpyPeer`、NCCL 或任何 GPU 直连传输。直连 NVLink 的性能单独作为 baseline 测。

跨 CPU 路径上，无论线程在哪个 CPU，数据都要过一次 CPU0↔CPU1：

- 线程在 CPU0：读是本地 C2C，写跨 CPU。
- 线程在 CPU1：读跨 CPU，写是本地 C2C。

两种都测，用 `copy_cpu_role` 区分。

## 测试点及与 gh_benchmark 的对应

| 命令 | 测什么 | 来自 gh_benchmark |
| --- | --- | --- |
| `topo` | CPU/GPU/NUMA 拓扑、HBM NUMA 节点、GPU 所属 CPU、P2P 能力、NUMA distance | `print_device_props`、`constants.hpp`/`memory.hpp` 里写死的 NUMA 编号 |
| `relay` | GPU HBM → CPU 线程 → GPU HBM | 新增，基于 `copy_test_host_template` / `thread_copy_function` |
| `host` | CPU 线程对 `cpu*`/`hbm*` 的读、写、拷贝，可指定线程数 | `run_{read,write,copy}_tests_host`、HOST SCALABILITY |
| `device` | GPU kernel 对任意目标的读、写、拷贝，可指定 block 数 | `run_{read,write,copy}_tests_device`、DEVICE SCALABILITY |
| `memcpy` | `cudaMemcpyAsync` 在各类分配之间的矩阵 | `run_cuda_memcpy_heatmap_tests` |
| `latency` | CPU 或 GPU 单线程指针追逐延迟 | `latency.hpp` |
| `pingpong` | CPU/GPU 之间用原子 flag 往返 | `atomic_benchmarks.cuh` |

内存目标统一按下表命名，所有命令共用：

| 名称 | 含义 |
| --- | --- |
| `cpu<i>` | 第 i 个 CPU NUMA 节点上的 LPDDR，用 `numa_alloc_onnode` 分配的可分页内存 |
| `pinned<i>` | 同 `cpu<i>`，再加 `cudaHostRegister` |
| `hbm<g>` | GPU g 的 HBM，以系统内存方式访问（NUMA 节点分配；CDMM 模式下用 managed 内存） |
| `cuda<g>` | GPU g 上的 `cudaMalloc`，CPU 线程不能访问 |

可以写逗号列表，`hbm*` 这样的通配会展开为全部编号。

**未移植的部分**：

- cuBLAS GEMM、thrust reduction、LLM 应用（`apps.hpp`、`run_llama*.py`、`alloc.cpp`）
- 多节点 NCCL/MPI 和 OSU（`nccl.cpp`、`run_osu.py`）
- Cray `pm_counters` 功耗
- `cache.hpp`、HMM/页迁移（`hmm_benchmarks.cu`、`page_migration.hpp`、`backforth.cpp`）
- multichase

这些要么不属于节点内内存路径，要么依赖原集群的环境。

**相对 gh_benchmark 的修改**：

- **拓扑和绑核**：NUMA 编号和 CPU 列表在运行时发现，不再写死 0/1/4/12 或绑 287 号核；线程只绑到进程允许的 CPU 上。
- **计时**：CPU 用 `CLOCK_MONOTONIC`；GPU 用 `%globaltimer`，因为 CUDA 13 删除了 `cudaDeviceProp::clockRate`。
- **结果校验**：每次拷贝后逐字比对目的数据；每个 NUMA 分配都抽样检查页所在节点。
- **原代码的问题**：去掉了未定义的 `run_memcpy_tests_host`；`hbm` 标签实际分配到 `REMOTE_DEVICE_MEM` 的问题不再存在；MGH/MGD 两列重复的问题不再存在。
- **指针追逐的工作集**：默认走完整条链，工作集等于整个 buffer，不会落进 cache。

## 构建

需要 CUDA 13.0+ 和 libnuma。代码用到 CUDA 13 的 `cudaMemPrefetchAsync` / `cudaMemAdvise` 的 `cudaMemLocation` 签名。

```bash
# 在 GB200 节点上本地编译（aarch64）
make                      # -> build/gb_benchmark，sm_100 SASS + compute_100 PTX

# 在 x86_64 Docker 主机上交叉编译（nvidia/cuda:13.0.0-devel-ubuntu22.04 + cross-linux-sbsa 包）
./build_cross_in_docker.sh               # -> build/gb_benchmark（aarch64）
NATIVE_X86=1 ./build_cross_in_docker.sh  # 另外生成 build-x86/gb_benchmark，仅用于在无 GB200 的机器上检查 host 侧逻辑
```

## 运行

```bash
unset CUDA_VISIBLE_DEVICES
./run_gb200.sh                 # relay 套件：两条中转路径 + 读/写分段 + NVLink baseline
./run_gb200.sh --suite full    # 另外跑 host/device/memcpy/latency/pingpong 矩阵和线程/block 扩展
./run_gb200.sh -- --hbm-backend managed   # 驱动为 CDMM 模式、HBM 不是 NUMA 节点时使用

# 单独运行某项
./build/gb_benchmark topo
./build/gb_benchmark relay --src-gpu 0 --dst-gpus 1,2 --cpu-nodes src,dst \
  --kernels ldp_stp,ldnp_stnp,memcpy --bytes 4G \
  --out ../../../../results/gb200/intra-super-node/relay-$(date +%y%m%d-%H%M%S).json
./build/gb_benchmark --help
```

运行前提：

- 在裸机上运行，或在带 `CAP_SYS_NICE` 的容器里运行。Docker 默认的 seccomp 会拦截 `mbind`，页就不会落到指定 NUMA 节点；这时记录中的 `placement.status` 为 `MISPLACED`，程序会打印警告。
- 若 cgroup 的 `cpuset.mems` 不包含 GPU 的 NUMA 节点，同样会被 placement 检查发现。
- 驱动没有报告 HBM NUMA 节点时，用 `--hbm-numa g=node` 手动指定，或改用 `--hbm-backend managed`。

## 结果

`run_gb200.sh` 把结果写到 `results/gb200/intra-super-node/YYMMDD-HHMMSS/`：

```text
env.txt  topology.json  topology.log  summary.txt
gpu0_to_gpu1/{relay,baseline_memcpy,baseline_sm_copy}/{result.json,benchmark.log}
gpu0_to_gpu2/...
legs/{read_hbm0,write_hbm1,write_hbm2}/...
```

每个 `result.json` 包含完整拓扑和 `results` 数组，每个测试 case 一条记录。主要字段：

| 字段 | 含义 |
| --- | --- |
| `payload_gb_s` | 有效载荷带宽 `bytes / time`（GB/s，10^9），给出 min/median/mean/max/stddev |
| `aggregate_read_write_gb_s` | copy 同时计入读和写的流量，等于 payload 的 2 倍 |
| `verify.mismatched_words` | 拷贝后目的端与源 pattern 不一致的 64-bit 字数，必须为 0 |
| `src.placement` / `dst.placement` | 抽样页所在 NUMA 节点的直方图；`status` 必须为 `ok` |
| `max_late_start_us` | 最慢的线程比统一起跑时刻晚了多少；明显偏大说明带宽被低估 |
| `path` / `path_class` / `copy_cpu_role` | relay 的路径，以及拷贝线程在哪个 CPU |

解读 relay 的建议：

- relay 带宽受两段中较慢的一段限制：读源 HBM（`legs/read_hbm0`）和写目的 HBM（`legs/write_hbm<d>`）。分段测试在每个 CPU 上都跑了一遍，可以直接对比。
- 跨 CPU 时，比较 `src_attached` 和 `dst_attached` 两个结果，可以看出 CPU0↔CPU1 链路分别承载读和写时的差异。
- `baseline_memcpy`（copy engine）和 `baseline_sm_copy`（SM 拷贝）是同一对 GPU 之间直接走 NVLink 的参考值。

## 验证状态

| 项目 | 状态 |
| --- | --- |
| 交叉编译 | 通过。CUDA 13.0.48，产物是 aarch64 ELF，含 `sm_100` cubin 和 `compute_100` PTX，动态依赖 `libnuma.so.1` |
| x86_64 双路主机（无 GPU）上的 host 逻辑 | 通过。拓扑发现，跨 NUMA 的 host copy/read/write，latency，CPU pingpong；校验 0 mismatch，placement `ok`，错误路径都检查过 |
| aarch64 二进制在 qemu-user 下 | 通过。`ldp_stp`、`ldnp_stnp`、`memcpy` 三种拷贝的校验都是 0 mismatch |
| GPU 相关路径 | **未验证**：HBM 分配、relay、device、memcpy、GPU latency/pingpong，以及所有性能数值，需要在 GB200 上运行 |
