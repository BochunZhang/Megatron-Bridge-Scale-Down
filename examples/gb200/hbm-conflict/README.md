# GB200：AllGather 背景流量下的 GEMM 完成时间（HBM 争用）

## 回答的问题

GPU 0–3 做 GEMM 时，如果分出几个 SM 在这 4 张卡之间搬数据（AllGather），对端写进本卡 HBM 的流量和
GEMM 自己的 HBM 读写争用带宽，GEMM 会慢多少？AllGather 的流量越大，GEMM 变慢多少？

第一轮配置：

| 项 | 值 |
| --- | --- |
| GEMM | `[4096, 7168] x [7168, 2048]`，bf16，每次 120.3 GFLOP，最小 HBM 流量 100 MiB（A、B 读一次，C 写一次） |
| AllGather | 4 GPU，gathered 1 GiB（每卡 shard 256 MiB），**单播**对称内核 `AllGather_ST`（非 multicast） |
| 流量控制 | nccl-2.32.3-pause 的 `NCCL_SYM_AG_ST_PAUSE_CYCLES`：AG 的每个 warp 每把一个 tile 写完所有 peer 后自旋 C 个 SM cycle |
| 采集 | 每个 rank 一个 nsys 会话，CUDA + NVTX + GPU metrics（NVLink RX/TX、DRAM 读写，默认 50 kHz） |

## 实验流程

每个 case 是一个独立的 torchrun 作业（NCCL 每个进程只读一次 pause 变量，所以一个 pause 值一个作业），
整段测量是一个 `cudaProfilerStart/Stop` 窗口（`--capture-range-end=stop`），每个 rank 一份报告。

**baseline**（不跑 AllGather）：

```text
GEMM stream:  gemm00 gemm01 ... gemm09            （背靠背，10 轮）
```

**sweep**（每个 pause 值一个 case，`pause<C>cyc`）：

```text
comm stream (最高优先级):
  ag00..ag04 (warmup) | ag05..ag09 (独立执行) | ag10 .................................. ag24 (背景)
                                              ^ event
GEMM stream:                                  wait(event) → sleep 50 us → gemm00 ... gemm09
```

- 25 轮 AllGather 背靠背发在 comm stream 上：前 5 轮 warmup，接下来 5 轮单独执行，用来量 AG 自己的
  完成时间和带宽。
- 第 10 轮（`ag09`）之后记录一个 event，GEMM stream `wait_event` 它，所以 GEMM 在 `ag10` 开始时才被放行；
  后 15 轮期间 GEMM stream 跑 10 轮 GEMM。
- **comm stream 用最高优先级。** AllGather kernel 只占几个 CTA，GEMM 的 CTA 会占满其余 SM。如果两边同时
  有 CTA 在等，高优先级 stream 的 CTA 先被调度，所以每轮 AG 结束、下一轮开始时，AG 能先拿回自己的 SM，
  不会被排在 GEMM 的 CTA 后面。
- **GEMM 放行后先在 GPU 上 sleep `--gemm-delay-us`（默认 50 µs）**：`ag09` 结束的同一时刻 `ag10` 和 GEMM
  都变成可执行，sleep 保证 `ag10` 先驻留，第一个 GEMM 一开始就有背景流量。sleep 是单线程的
  `torch.cuda._sleep`，在 `gemm-delay` NVTX range 里，不计入 GEMM。
- 窗口之外：cuBLAS warmup（`--gemm-warmup` 轮，默认 5）、对称 window 注册、NCCL 初始化。
- 跑完后检查 AllGather 输出（第 k 块必须等于 k + 1），错了该 rank 以非零退出。

按经验估一下量级：这个 GEMM 在 GB200 上约 50–60 µs，10 轮约 0.6 ms；1 GiB 单播 AllGather 每卡收
768 MiB，pause=0 时一轮约 1 ms 量级。所以 10 个 GEMM 通常落在前 1–2 轮背景 AG 里，后面的背景轮只有 AG；
报告第 3 节逐轮标出哪些 AG 和 GEMM 重叠过。

## 为什么用 `AllGather_ST`，以及补丁的改动

**选中单播内核。** 和 `../nvls-collective` 的 symm-p2p 一样：buffer 在 `ncclMemAlloc` 支撑的
`torch.cuda.MemPool` 里，并注册成 `NCCL_WIN_COLL_SYMMETRIC` window；launcher 另外固定内核：

| 变量 | 值 | 作用 |
| --- | --- | --- |
| `NCCL_NVLS_ENABLE` | `0` | `nvlsSupport = 0`，`hasLsaMultimem` 为假，所有 `*MC` 内核被 `ncclSymkMask` 去掉 |
| `NCCL_WIN_ENABLE` | `1` | 打开 window 注册 |
| `NCCL_SYM_KERNEL` | `allgather:AllGather_ST` | AllGather 只能用 `AllGather_ST`（2.32 的 `<func>:<kernel>` 语法），AllReduce（barrier）不受影响 |
| `NCCL_SYM_TMA_ENABLE` | `0` | 不走 `AllGather_TmaST` |
| `NCCL_SYM_CE_THRESHOLD` | `INT64_MAX` | 大 AllGather 不走 copy engine（这个分支在挑内核之前，固定内核也拦不住） |

`AllGather_ST` 的数据路径：每卡把自己的 shard 直接 store 进其余 3 张卡的 window（`all_gather.cuh` 的
`bcastDeep`），即每张卡的 HBM 要接收 3 × 256 MiB 的远端写入，外加本地那一份。这正是要和 GEMM 争用的流量。

**pause 补丁要扩展。** 原来的 `install/patches/nccl-2.32.3-pause.patch` 只在 multicast 的 `bcastMultimem`
（`AllGather_STMC` / `AllGather_TmaSTMC`）里插了自旋，`NCCL_SYM_AG_STMC_PAUSE_CYCLES` 对 `AllGather_ST`
不起作用。现在补丁新增：

| 位置 | 改动 |
| --- | --- |
| `src/scheduler/symmetric_sched.cc` | `NCCL_PARAM(SymAgStPauseCycles, "SYM_AG_ST_PAUSE_CYCLES", 0)`，钳到 0..UINT32_MAX 后写进 `argsBuf->agStPauseCycles` |
| `src/include/sym_kernels.h` | `ncclSymkDevWorkArgs` 加 `uint32_t agStPauseCycles` |
| `src/device/symmetric/all_gather.cuh` | `bcastDeep` 主循环里，每个 tile 写完所有 peer 后、读下一个 tile 前调用 `ncclSymkPauseCycles(pauseCycles)`；`bcast` 把它传给三条 `bcastDeep` 路径，`AllGather_ST` / `AllGather_TmaST` 传 `args->agStPauseCycles` |

16 B 对齐路径上一个 tile 是 `UnrollPacks(4) × 32 lanes × 16 B = 2 KiB`，写给每个 peer（含自己）各一份。
默认 0 时内核与原版完全相同，multicast 的 `NCCL_SYM_AG_STMC_PAUSE_CYCLES` 也不变，所以
`../nvls-collective/run_symm_cta_pause_sweep.sh` 不受影响。

**需要重新编译。** 旧补丁打过的源码树不能直接套新补丁，先还原再编：

```bash
git -C nccl/nccl-2.32.3-pause checkout -- .
bash install/install-nccl-2.32.3-pause.sh
```

安装脚本最后会检查 `libnccl.so` 里同时有 `NCCL_SYM_AG_STMC_PAUSE_CYCLES` 和
`NCCL_SYM_AG_ST_PAUSE_CYCLES`；launcher 启动前也会检查后者，旧库直接报错。

## 运行

```bash
# 默认：baseline + 10 个 pause 值，结束时自动生成 report.md
bash examples/gb200/hbm-conflict/run_hbm_conflict_benchmark.sh

# 只跑三个 pause 值
bash examples/gb200/hbm-conflict/run_hbm_conflict_benchmark.sh --pause-cycles 0,1024,8192

# 固定 AllGather 的 CTA 数（NCCL_MIN_CTAS = NCCL_MAX_CTAS = 8），看“分出几个 SM”的影响
bash examples/gb200/hbm-conflict/run_hbm_conflict_benchmark.sh --ctas 8

# 只打印命令
bash examples/gb200/hbm-conflict/run_hbm_conflict_benchmark.sh --dry-run --venv none
```

环境要求和 `../nvls-collective/run_symm_cta_pause_sweep.sh` 一样：torch 只能加载一个 `libnccl.so.2`
（`install/install-torch-2.13.0-nccl-2.32.3.sh`），launcher 用 `LD_PRELOAD` 预加载 pause 库并先检查
`/proc/self/maps`；nsys GPU metrics 需要 root 或 `NVreg_RestrictProfilingToAdminUsers=0`，节点上有 DCGM
profiling 时先 `dcgmi profile --pause`。

主要选项（全部见 `--help`）：

| 选项 | 默认 | 说明 |
| --- | --- | --- |
| `--pause-cycles LIST` | `0,128,256,512,1024,2048,4096,8192,16384,32768` | 每个值一个 sweep case；0 是同一个库的未减速版本 |
| `--size-mib N` | 1024 | AllGather gathered 大小 |
| `--gemm-mkn M,K,N` | `4096,7168,2048` | GEMM 形状 |
| `--gemm-rounds` / `--gemm-warmup` | 10 / 5 | 测量 GEMM 数 / 窗口外 warmup 数 |
| `--ag-rounds` / `--ag-warmup` / `--ag-isolated` | 25 / 5 / 5 | AllGather 总轮数 / warmup / 独立执行 |
| `--gemm-delay-us` | 50 | GEMM 放行后的 GPU sleep |
| `--ctas N` | 不设（NCCL 自己选） | 固定 AG 的 CTA 数 |
| `--nsys-freq HZ` | 50000 | GPU metrics 采样频率（10..200000）；10 个 GEMM 只有约 0.6 ms，50 kHz 下约 30 个采样，要更细可以加到 200000 |
| `--nsys-gpu-metrics-set SET` | nsys 默认 | 可选值存在 `nsys-gpu-metrics.txt` |

默认的 pause 列表只是按 2 倍递增铺开：单播 AG 的减速拐点还没有测过。看第一轮报告里 “AG 独立 busbw”
随 pause 的变化，再把点加密到拐点附近。pause 的 ns = C / SM clock（报告里没有，nsys 时间线上的
`GPC Clock Frequency` 行可以读到，或先 `nvidia-smi -lgc` 锁频）。

所有 case 跑完后 launcher 校验：每个 NCCL 进程的 banner 是 `NCCL version 2.32.3-pause+...`；sweep 的每个
进程都打印了 `NCCL_SYM_AG_ST_PAUSE_CYCLES set by environment to <C>.`；TUNING 日志里每次 AllGather 都是
`Kernel AllGather_ST`；不传 `--ctas` 时没有任何 CTA 变量传进 NCCL。任何一项失败，该 case 判失败，最终退出码非零。

## 输出

```text
results/gb200/hbm-conflict/gpu-num-4/<yyyymmdd-hhmmss>/
  metadata.json                 # 本次运行的所有参数
  nsys-gpu-metrics.txt          # nsys 的 GPU 编号和可选 metric set
  report.md                     # 分析报告（见下）
  summary.json                  # 每个 case 的汇总（report.md 第 1、2 节的数）
  ag_rounds.csv                 # case × AG 轮：phase、是否与 GEMM 重叠、最慢 rank ms、algbw / busbw、CTAs
  gemms.csv                     # case × rank × GEMM：ns、kernel 名、被 AG 覆盖的 ns 和比例、重叠的 AG 轮
  gpu_metrics.csv               # case × rank × 窗口 × metric：采样数、均值、峰值（set 里所有 metric）
  baseline/
    case.json                   # case 名、模式、传给 torchrun 的环境变量
    torchrun.log
    events.rank_<RR>.json       # 每个 rank 的 CUDA event 时间（每轮 [start, end] ms）和设备信息
    nccl-raw/nccl-<host>-<pid>.log
    nsys/baseline.rank_<R>.nsys-rep / .sqlite
  pause<C>cyc/ ...              # 同上
```

## 分析（`analyse_hbm_conflict.py` + `hbm_conflict.sql`）

launcher 最后会自动跑；也可以单独重跑（已有且比报告新的 `.sqlite` 会复用）：

```bash
uv run --no-sync python examples/gb200/hbm-conflict/analyse_hbm_conflict.py \
  --output-dir results/gb200/hbm-conflict/gpu-num-4/<run id>
# 或对单份报告直接查：
nsys export --type=sqlite --output=r.sqlite results/.../pause1024cyc/nsys/pause1024cyc.rank_0.nsys-rep
sqlite3 -header -csv r.sqlite < examples/gb200/hbm-conflict/hbm_conflict.sql
```

| 步骤 | 规则 |
| --- | --- |
| 核对归档 | 报告里外层 NVTX range 必须是 `hbm-conflict/<所在目录名>` |
| kernel → 轮次 | 发射它的 CUDA API 调用（`correlationId`、同进程）开始于哪个 `ag<II>` / `gemm<JJ>` range，就属于哪一轮；与 `../nvls-collective/nsys_collective_kernels.sql` 相同 |
| AG 每轮时间 | 该轮 NCCL kernel 的 GPU 起止；4 个 rank 取最慢；algbw = 1 GiB / t，busbw = algbw × 3/4 |
| GEMM 时间 | 该 `gemm<JJ>` 里所有 kernel 的最早开始到最晚结束（split-K 等多 kernel 也算一个 GEMM）；TFLOPS = 2MKN / t |
| 是否在背景流量下 | 把同一张卡所有 AG kernel 的区间求并，GEMM 区间被覆盖的比例 = `coverage`；`完全` = 覆盖 100% 的 GEMM 数 |
| GPU metrics 窗口 | `ag_isolated` = 第一个独立轮开始到最后一个独立轮结束；`gemm` = 第一个 GEMM 开始到最后一个 GEMM 结束；窗口内采样求均值，再对 rank 求平均 |

`report.md` 各节：

1. **汇总**：每个 case 一行——AG kernel 名和 CTA 数（gridX）、AG 独立执行的 ms / algbw / busbw、与 GEMM
   重叠那几轮 AG 的 ms（看 GEMM 反过来拖慢了 AG 多少）、GEMM mean / min / max ms、TFLOPS、
   “最小流量” GB/s（100 MiB / t）、`vs baseline`（GEMM 平均时间之比）、覆盖（`10/10` 加最小覆盖率），
   以及 CUDA event 量出的 GEMM / AG 时间作为交叉核对。
2. **NVLink / DRAM 吞吐**：每个 case × 窗口，列出名字含 NVLink / DRAM / C2C / SMs Active 的 metric。
   NVLink 是 18 条链路合计，含协议开销，GB/s ≈ % × 9；DRAM 是 HBM 接口忙碌周期占比。sweep 的
   `gemm` 窗口 DRAM 与 baseline 的 `gemm` 窗口对比，就是 AG 额外带来的 HBM 占用。
3. **每轮 AllGather**：25 轮逐行，标出 phase 和是否与 GEMM 重叠。
4. **每个 GEMM**：10 个 GEMM 逐行，rank 平均 / 最大 ms、TFLOPS、最小覆盖、重叠的 AG 轮。

AG 被换成其他内核、有 GEMM 没有完全落在 AG 里、AG 输出错误，都会在报告末尾的“警告”里列出。

## 注意

- **nsys 有开销。** 结论里的 GEMM 时间都来自 profile 运行，baseline 和 sweep 同样带 nsys 和 GPU metrics，
  二者可比；不要和不带 nsys 的其他实验的绝对值混着比。
- **AG kernel 时间含等待。** 对称内核开头和结尾各有一次跨卡 barrier，kernel 时长包含等对端的时间；
  背靠背时这部分很小，但 4 个 rank 的时长会有差异，所以每轮取最慢 rank。
- **SM 争用和 HBM 争用叠在一起。** AG 占的 CTA 不能给 GEMM 用，所以 GEMM 变慢 = SM 变少 + HBM / L2 争用。
  pause 足够大时 AG 的流量趋近 0 但 CTA 仍驻留（在自旋），这时的变慢主要是 SM 被占；
  可以用大 pause 的 case 减掉这一部分，或者用 `--ctas` 改变 AG 的 SM 数单独看。
- **GPU metrics 是整卡计数器。** DRAM 包含别的 GPU 经 NVSwitch 写进本卡 HBM 的流量，正是要观察的争用来源。

## 本地检查（无 GPU）

```bash
uv run python -m pytest tests/unit_tests/examples/test_gb200_hbm_conflict.py
```

测试用一份 nsys SQLite schema 的合成报告跑完整分析（轮次归属、覆盖率、窗口 metric、报告内容），并用
`--dry-run` 检查 launcher 的命令和参数校验。
