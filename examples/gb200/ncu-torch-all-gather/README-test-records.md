# ncu profile NCCL all-gather：测试记录

记录用 Nsight Compute（ncu）在 GB200 上 profile NCCL all-gather kernel 时，哪些组合成功、哪些失败、失败的现象和
推断的原因。torch 的测试用本目录的 `run_ncu_torch_all_gather.sh` / `bisect_ncu_sections.sh`（用法见 [README.md](README.md)），
nccl-tests 的测试用 `../nccl-tests/run_ncu_all_gather_stmc.sh`（见 [../nccl-tests/README.md](../nccl-tests/README.md)）。
“证据”一列的 `ncu*.log` 是当时从集群取回的 ncu 输出，结果目录在 `results/gb200/ncu-torch-all-gather/` 下。

最后更新：2026-11-07（#15）。

## 环境

| 项目 | 值 |
| --- | --- |
| 机器 | GB200 单节点 4 卡（GPU 0/1 在 NUMA 0，GPU 2/3 在 NUMA 1），NVLink 经 NVSwitch |
| torch | 2.13.0+cu130，运行时 NCCL 2.29.7（`ncu (3).log` 那次是 2.32.3） |
| nccl-tests | 2.21.1，NCCL 2.32.3 |
| ncu | 2026.3（2026-10-31、2026-11-06），2026.4.0.0 build 39012641（2026-11-07 起） |
| communicator | 除特别注明外都是 `--communicator shmem`：一个 ncu 包住整个 torchrun / mpirun |

## 四种 replay 模式的区别

一个 kernel 要采的 metric 通常放不进一次执行（硬件计数器有限），ncu 要把它执行多遍（pass），每遍采一部分。
四种模式的区别在于**怎么让 kernel 再执行一遍**，以及**profile 的单位是单个 kernel 还是一段 range**。

| | kernel | application | range | app-range |
| --- | --- | --- | --- | --- |
| profile 单位 | 单个 kernel | 单个 kernel | 一段 range（NVTX 或 cuProfilerStart/Stop 圈出） | 一段 range |
| 怎么重复执行 | 在同一个进程里，把这个 kernel 原地重放 N 遍 | 整个程序重新跑 N 遍，每遍采这个 kernel 的一部分 metric | 抓下 range 内的 CUDA API 调用和 launch，在同一进程里重放 N 遍 | 整个程序重新跑 N 遍，每遍采这段 range |
| 每遍之间保存/恢复显存 | **要**（多于 1 遍时），保存 kernel 能访问的全部显存 | 不要 | **要**（保存/恢复 range 访问的显存） | 不要 |
| kernel 是否被串行化 | 是：被 profile 的 kernel 独占 GPU | 是 | **否**：range 内的 kernel 按原样并发 | **否** |
| metric 归属 | 每个 kernel 一份 | 每个 kernel 一份 | 整个 range 一份（不分 kernel） | 整个 range 一份 |
| 对程序的要求 | 无 | 每次运行的 kernel 序列必须一致 | range 内只能用支持的 API；文档列出 VMM、peer 访问等**不支持** | 每次运行一致；range 开始时 CUDA context 必须是当前线程的 |
| 开销 | 每遍只重放一个 kernel，快 | 每遍重启整个程序（这里约 14 s/遍） | 快 | 每遍重启整个程序 |
| shmem communicator | 支持 | **ncu ≥ 2026.4** | 支持（文档） | **ncu ≥ 2026.4** |
| 适合 | 普通单卡 kernel | 显存保存不了（multicast / VMM），或 kernel 有副作用 | 必须并发的 kernel，且 range 内 API 都受支持 | 必须并发的 kernel，且显存保存不了 |

对 NCCL kernel 的影响：

- **保存显存**：NVLS 和 symmetric window 的 buffer 映射了 multicast 内存，kernel replay 保存显存时失败（见下表）。
  application / app-range 不保存显存，所以能绕开。
- **串行化**：kernel 和 application replay 只 profile 单个 kernel，并且会把它和其他 kernel 串行化。所以看不到
  两个集合通信（比如 AG 和 RS）重叠时的干扰；要测重叠，只能用 range 或 app-range，而 range 又要保存显存，
  所以对 multicast 来说只剩 app-range。
- **所有 rank 必须同时进入同一个 kernel**：NCCL kernel 要等所有对端到齐才能推进，所以必须用 communicator
  （shmem / tcp）让各 rank 同步 launch 和重放；不加就卡死。

## 测试记录

图例：✅ 成功出报告，❌ 失败，⚠️ 能跑完但结果不可信。

| # | 日期 | ncu | comm | replay | 程序 / 算法 / kernel | 大小 | 采集 | 结果 | 现象 | 证据 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 10-31 | 2026.3 | shmem | kernel | nccl-tests, AllGather_STMC | 256 MiB | 多遍 | ❌ | `Backing up device memory in system memory` 后 `Failed to save memory for replay`，退出码 9 | `ncu.log` |
| 2 | 10-31 | 2026.3 | shmem | application | nccl-tests, STMC | 256 MiB | — | ❌ | 程序跑完、数据正确，但 `No kernels were profiled`（2026.3 不支持 shmem 下的 application replay） | `ncu (1).log` |
| 3 | 10-31 | 2026.3 | tcp | kernel | nccl-tests, STMC | 256 MiB | 多遍 | ❌ | communicator 正常（4 个 rank 都 `Initialized Communicator`），仍然 `Failed to save memory for replay` | `ncu (2).log` |
| 4 | 11-06 | 2026.3 | shmem | kernel | torch, nvls（NCCL 2.32.3，`ncclDevKernel_AllGather_RING`） | 256 MiB | `--set basic` | ❌ | `ContextSaveFailed`；/tmp 日志：`Failed to copy memory` → `Failed to save context state` | `ncu (3).log` |
| 5 | 11-07 | 2026.4 | shmem | application | torch, nvls（2.29.7，`..._RING_LL`） | 256 MiB | `--set full` | ❌ | pass 1–38 正常（约 14 s/遍），**pass 39** 4 个 rank 都进入 kernel 但跑不完；ncu 不到约 2.5 min 自己放弃：`Failed to wait for profiling` → `Unexpected number of profiled kernels` | `ncu (4).log` |
| 6 | 11-07 | 2026.4 | shmem | application | torch, nvls | — | 单遍 | ✅ | 1 个 pass，写出 `all_gather.ncu-repz`（当时脚本只认 `.ncu-rep`，误报 `ncu wrote no`，已修复） | `ncu (5).log` |
| 7 | 11-07 | 2026.4 | shmem | application | torch, nvls | 256 MiB | `--set full` | ❌ | 同 #5，又卡在 pass 39 | `ncu (6).log` |
| 8 | 11-07 | 2026.4 | shmem | application | torch, nvls | 1 MiB | `--set full` | ❌ | 同 #5，卡在 pass 39：和数据量无关，是卡死不是慢 | 用户口述 |
| 9 | 11-07 | 2026.4 | shmem | application | torch, ring（`NCCL_NVLS_ENABLE=0`） | 1 MiB | `--set full` | ❌ | 同 #5，卡在 pass 39：和 multicast / NVLS 无关 | 用户口述 |
| 10 | 11-07 | 2026.4 | shmem | application | torch, ring | 1 MiB | — | ❌ | 脚本 TUNING 检查报 `NCCL did not run --algo ring`（AllGather 的条数或算法对不上，未查明） | 用户口述 |
| 11 | 11-07 | 2026.4 | shmem | application | torch, nvls | — | 8 个 section，`--metrics` + `-- --section` | ⚠️ | 都 ok，但每个约 1 个 pass，不到 2 min：怀疑 `--section` 没生效，之后改成启动器的 `--section` | 用户口述 |
| 12 | 11-07 | 2026.4 | shmem | application | torch, nvls | 1 MiB | full 的每个 section 单独跑 | ✅ | 全部 ok，没有卡住的 | 用户口述 |
| 13 | 11-07 | 2026.4 | shmem | application | torch, **symm**（`ncclSymkDevKernel_AllGather_STMC`，2.29.7） | **1 GiB** | full 的 24 个 section 单独跑 | ✅ | 全部 ok，每个 1–32 个 pass，kernel 1.78–1.90 ms；每份报告只有 1 个结果（primary 进程，即 rank 0） | `symm-all-gather-20261107-140128/`（见下文「#13 的数据」） |
| 14 | 11-07 | 2026.4 | shmem | **app-range** | torch `--ag-rs`, symm（2.29.7） | 1 GiB | `--section SpeedOfLight` | ❌ | 4 个 rank 跑完 warmup，一进 NVTX range `ncu_collectives` 就 `Failed to profile "range" in client process <rank 0>`，第 1 遍都没开始，退出码 9。2026.4 CLI 文档：shmem "only supports kernel and range replay modes"，release notes 只加了 application | `ncu (7).log` |
| 15 | 11-07 | 2026.4 | 见右 | 见右 | torch `--ag-rs`, symm（2.29.7，STMC + LDMC） | 1 GiB | `--section SpeedOfLight` | 见右 | `sweep_ncu_replay_modes.sh`：**app-range + tcp** ✅ 8 pass（有无 `--lockstep-*` 都行）；application + shmem ✅ 9 pass（串行化）；range + shmem / tcp ❌ 保存显存失败；app-range + shmem ❌（同 #14）；app-range + none ❌ 卡死 900 s | `results/ncu-replay-mode.txt`（`symm-replay-sweep-20261107-195000/`） |

### 由这些记录得出的结论

- **kernel replay + 多遍 + multicast buffer 一定失败**（#1 #3 #4），和 communicator 是 shmem 还是 tcp 无关（#3）。
  2026.4 下有没有修复，没有测过。
- **shmem + application replay 需要 ncu 2026.4**（#2 vs #5 起）。
- **`--set full` 一次采完会卡在 pass 39**，和算法（nvls / ring）、数据量（256 MiB / 1 MiB）都无关（#5 #7 #8 #9）；
  而**每个 section 单独都能跑完**（#12 #13）。所以要么是若干 section 组合在同一个 pass 里才出问题，要么是
  同一个 ncu 会话里程序重启到第 39 次时碰到某个上限，尚未区分。可行的做法是按 section 拆开采（#13）。
- **ncu 在 pass 卡住时会自己放弃**，不会等到脚本的 `--timeout`；没有找到能调整这个等待时间的选项。
- 2026.4 默认写**压缩的 `.ncu-repz`**（#6）。
- **shmem + app-range 不可用**（#14），和文档一致：shmem 只写了 kernel / range，2026.4 又加了 application。
  AG + RS 重叠剩下的路是 range（shmem）或 tcp communicator，用 `sweep_ncu_replay_modes.sh` 逐个试。
- **profile 并发的 AG + RS（symmetric multicast kernel）用 app-range + tcp communicator**（#15）：每个 rank 一个 ncu，
  不保存显存、不串行化。range replay 和 kernel replay 一样会因为 multicast 内存保存失败。

## 还没有测过的组合

| 组合 | 预期（都未验证） | 用途 |
| --- | --- | --- |
| 2026.4 + kernel replay，多遍 | nvls / symm 可能仍在保存显存时失败；ring（关 NVLS）、p2p 没有 multicast 内存，可能可以 | 比 application replay 快得多 |
| app-range + tcp，AG 单独（`--no-ag-rs`）、逐个 section、`--blocking-metrics` | 预计和 #15 一样可以 | 以后 profile symmetric kernel 的主路径；SourceCounters 能否按 kernel 函数分开 AG 和 RS 的 stall |
| **p2p**（`ncclSymkDevKernel_AllGather_ST`） | 任何 replay 模式都没跑过 | 和 symm 对照：不经 multicast，每卡只收 3 份 |
| 4 个 rank 的结果 | 现在每份报告只有 primary 进程（rank 0）的结果 | 看各 rank 是否一致 |
| `PmSampling+SourceCounters` 同一次运行 | 预计约 9 个 pass，应该可以 | 时间序列和指令级数据来自同一次执行 |

## #13 的数据

结果目录 `results/gb200/ncu-torch-all-gather/symm-all-gather-20261107-140128/`（下表省略这个前缀）：

| 路径 | 内容 |
| --- | --- |
| `<section>/` | #13：每个 section 一个目录，`all_gather.ncu-repz`、`config.txt`、`ncu.log`、`nccl-logs/`、`launcher.log` |
| `<section>/export/` | 用 Docker 镜像 `nsight-tools:cuda13.0-ncu2026.4-nsys2026.6` （ncu 2026.4）导出：`details.csv`、`details.txt`、`raw.csv`。`export/` 不是脚本生成的，是事后导出的 |
| `SourceCounters/export/source_sass.csv` | 每条 SASS 指令的 stall 采样和执行次数 |
| `PmSampling/export/pm_timeseries.csv` | 164 个 PM sampling metric 的时间序列（约 1 µs 一个点，含 NVLink rx/tx） |
| `summary.txt` | 每个 section 的状态和 pass 数 |

#13 的分析要点：NVLink rx 稳定在约 895 GB/s（接收端饱和），tx 在 330–500 GB/s 之间波动；multicast 把本卡分片也
经 NVLink 送回本卡，占接收带宽的 1/4；指令级时间几乎都在主循环的 8 条 `LD.E.128` + 8 条 `STG.E.128`，stall 为
long_scoreboard 57%、lg_throttle 30%、barrier 10%。
