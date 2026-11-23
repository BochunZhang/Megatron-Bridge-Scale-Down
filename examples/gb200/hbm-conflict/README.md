# GB200：AllGather 背景流量下的 GEMM 完成时间（HBM 争用）

## Copy Engine 和 HBM 争用测试

在 GB200 节点的工作目录准备 `NsightSystems-2026.6.1/bin/nsys`，并启用已安装 CUDA PyTorch 的虚拟环境。也可用
`NSYS_BIN` 指定同版本的安装路径。multicast 测试使用同一台主机、同一 NVLink multicast clique 内至少三张 GPU。
AG 使用当前环境中 PyTorch 加载的 NCCL，最低版本为 2.32.2。NCCL 构建和 CUDA 驱动须支持目标为 multicast
地址的批量拷贝，CPU 源 case 还需要支持从 pinned host memory 向该地址拷贝。

```bash
bash examples/gb200/hbm-conflict/run_ce_hbm_benchmark.sh \
  --gpus 0,1,2,3 --gemm '[8192,8192,7168],[16384,8192,8192]'
```

`--gemm` 顺序为 `[M,N,K]`。默认只测 `[8192,8192,7168]`，GEMM 使用 BF16 和预分配输出。
`--size-mib` 默认 1024，表示单次 H2D 或每个 rank 的完整 AG 输出大小，AG 输出字节数须按 rank 平分。
例如三卡时可以用 `--size-mib 1023`。每个 GEMM case 按 shape 各跑一次，通信 baseline 与 shape 无关。
常规运行省略 `--cases` 时，默认运行 `task1` 至 `task4`，加 `--baseline` 会追加 `baseline1` 至 `baseline3`。
显式指定 `--cases` 时按给出的列表运行，可以直接列出 baseline。用 `--cases task4,baseline3` 可只测 CPU 源 multicast AG。
`--dry-run` 展开所有命令。

| Case | GPU 数 | 计算 | 搬运 |
| --- | --- | --- | --- |
| `task1` | 1 | GEMM | — |
| `task2` | 1 | GEMM | CPU pinned → HBM |
| `task3` | ≥3 | GEMM | HBM → multicast HBM |
| `task4` | ≥3 | GEMM | CPU pinned → multicast HBM |
| `task5` | ≥2 | GEMM | HBM → 各 rank 的 pinned Host 接收区 |
| `baseline1` | 1 | — | CPU pinned → HBM |
| `baseline2` | ≥3 | — | HBM → multicast HBM |
| `baseline3` | ≥3 | — | CPU pinned → multicast HBM |
| `baseline4` | ≥2 | — | HBM → 各 rank 的 pinned Host 接收区 |

在 multicast case 中，每个 rank 使用 PyTorch ProcessGroupNCCL 的 `_comm_ptr()` 调用当前 NCCL 库的 `ncclAllGather`。接收缓冲区
由 NCCL allocator 分配在 HBM 并注册为 symmetric window。CPU 源 case 直接传入 CUDA pinned tensor 的地址，
发送端使用未注册的 host buffer。NCCL copy engine 将本地 shard 写到 multicast 接收地址的对应槽位。
单轮 CUDA event 覆盖整个本地 NCCL 调用，包括 NCCL 在拷贝前后的 NVLS 同步。

multicast AG case 独立设置 `NCCL_CTA_POLICY=ZERO`、`NCCL_WIN_ENABLE=1`、`NCCL_NVLS_ENABLE=1` 和
`NCCL_CE_COLL_AG_MULTICAST_THRESHOLD=4611686018427387904`，命令与有效设置保存在 `command.txt` 和
`nccl-raw/nccl-*.log`。启动器检查 rank 0 的实际 AllGather 调优日志，要求该分片大小对应
`AllGather [Copy Engine]: ... Bytes -> cudaMemcpy; CE synchronization with NVLS` 且 multicast 阈值已生效。
出现其他 AllGather 路径或证据缺失时，该 case 报错。worker 验证每个 rank 的接收槽位，手动分析用 nsys
核对实际 memcpy 字节和 GPU 活动。multicast case 同时启用 Nsight Systems 的 NCCL trace，便于对照 NCCL API、CUDA 拷贝和 GEMM。

启动器用 `nvidia-smi --lock-gpu-clocks=tdp,tdp` 设置参与测试的 GPU，需要锁频和 GPU metrics 权限。实际
时钟轨道会随报告保留。Gloo 启动使用 `GLOO_SOCKET_IFNAME`，优先选有效的已有设置，其次 `eth0`、`bond1`，
单机环境下可用 `lo`。也接受 `GLOO_SOCKET_NAME` 作为输入别名。

每个活动 stream 先独立 warmup，初始化、校准和输出验证都在计时窗口外。并发 case 根据预热时间延长背景
copy 队列，用同一个 CUDA 起点发出预提交的 GEMM 和 copy。只测通信的 case 执行指定的 `--comm-rounds`。
手动生成的 `report.md` 单独报告实际重叠覆盖率。

每个 case 用一个 `nsys profile` 采集整个 `torch.distributed.run` 进程树。rank 0 在所有 rank 完成预热后打开
采集窗口，并在各 rank 的计时 CUDA 工作完成后关闭。多卡 case 把各 rank 的活动和指定 GPU 的 metrics 放到
同一条时间轴上，默认采集四张卡。单卡 case 只采集选定的第一张 GPU。GPU metrics 每张卡默认以 10 kHz 采样，
可用 `--nsys-freq` 调整。

结果保存在 `results/gb200/hbm-conflict/gpu-num-N/<timestamp>-ce/`。每个 case 保留 `torchrun.log`、CUDA event
记录和一份 `<case>/nsys/<case>.nsys-rep`。启动器默认停在采集阶段。

用下面的命令提取 `.nsys` 和 `.nsys-rep`。它保留相对路径，把 `path/aaa/bbb/ccc.nsys` 复制到
`path-nsys/aaa/bbb/ccc.nsys`。

```bash
uv run --active --no-sync python examples/gb200/hbm-conflict/extract_nsys_reports.py "$path"
```

需要汇总时，对原结果目录手动运行分析器。它会导出 SQLite，并写入 `report.md`、`summary.json` 和
`gpu_metrics.csv`，包含 GEMM TFLOPS、payload 吞吐、重叠覆盖率及窗口化 DRAM、NVLink、C2C 指标。

```bash
uv run --active --no-sync python examples/gb200/hbm-conflict/analyse_ce_hbm.py \
  --output-dir "$path" --nsys "${NSYS_BIN:-NsightSystems-2026.6.1/bin/nsys}"
```

HBM 和链路指标是设备级采样，可对照独跑与并跑窗口。采样值保留 nsys 的计数器含义，payload GB/s 则由
传输大小和时间计算。
[Nsight Systems GPU Metrics](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#gpu-metrics)

### AllGather 直接写入 Host

`task5` 和 `baseline4` 按需运行。每张 GPU 用 Copy Engine 将自己的 HBM shard 直接写到每个 rank 的
共享 pinned Host 接收缓冲区，包括自己的缓冲区。四卡、`--size-mib 1024` 时，每 rank 每轮发出四次
256 MiB D2H，最终每个 Host 接收区有 1 GiB 数据。worker 在计时前清空各自的接收区，采集结束后在
CPU 上逐 shard 校验。该测试走 D2H 单播，原 NCCL multicast case 走 NVLink multicast，两者的发送流量
也不同。单个 CUDA event 覆盖本 rank 四次发出的拷贝，整轮输出在所有发送 rank 完成后可用。
接收区通过 Linux memfd 共享，每个 rank 按自身的 Host NUMA 内存策略分配接收页。

复用上一批的六个 GEMM 尺寸，并在同一批采集 GPU 0 独跑、HBM 目标和 Host 目标的数据。

```bash
bash examples/gb200/hbm-conflict/run_ce_hbm_benchmark.sh \
  --gpus 0,1,2,3 --cases task1,task3,task5,baseline4 --mc-self-import 1 \
  --gemm '[2,8192,7168],[4,8192,7168],[8,8192,7168],[16,8192,7168],[32,8192,7168],[64,8192,7168]'
```

对照 GPU 0 的 GEMM GPU kernel 时间和与 payload 的实际重叠。分析器检查每轮 D2H 字节数、pinned
目标和 C2C TX 活动。`baseline4` 记录 Host 目标 AllGather 独跑时的搬运速率。

### Copy Engine 单播 AllGather sweep

使用同一 NVLink 域内的 GPU 和 NCCL 2.32.3。以下命令复用此前的六个 BF16 GEMM 形状，
扫描 `M=2,4,8,16,32,64`，固定 `N=8192`、`K=7168`。

```bash
bash examples/gb200/hbm-conflict/run_ce_unicast_hbm_sweep.sh \
  --gpus 0,1,2,3 --output-dir "$path"
```

| Case | 计算 | 搬运 |
| --- | --- | --- |
| `task1` | 第一张 GPU 上的 GEMM | — |
| `baseline5` / `task7` | 纯通信 / 各 GPU 上的 GEMM | GPU HBM → 每个远端 GPU 的 HBM |
| `baseline6` / `task8` | 纯通信 / 各 GPU 上的 GEMM | Pinned CPU → 每个 GPU 的 HBM |

默认每张 GPU 的完整输出为 1 GiB。四卡时，每个源 rank 持有一个 256 MiB shard。GPU 源采取 in-place，
计时前预放本地槽，计时内每个 rank 发出三笔远端拷贝。CPU 源从 pinned host shard 直接向四张 GPU
各发出一笔拷贝，目标地址是各 GPU 接收缓冲区的最终槽位。CPU 源的每张 GPU 每轮接收 1 GiB payload，
对应 HBM 写入。GPU 源每张 GPU 每轮发出和接收的 payload 各为 768 MiB。多卡作业的 GEMM 耗时
按 rank 保留，和 `task1` 比较时使用第一张 GPU。

单播 case 设置 `NCCL_CTA_POLICY=ZERO`、`NCCL_WIN_ENABLE=1`、`NCCL_NVLS_ENABLE=0`、
`NCCL_CE_COLL_AG_MULTICAST_THRESHOLD=0`。worker 在计时结束后校验各 GPU 的全部接收槽位。
可先用 `--cases baseline5,baseline6 --comm-rounds 3` 短测纯通信，用 `--gemm` 覆盖默认形状。

单播 case 采集完整进程树，保留各 rank 的 NVTX 标签。分析只使用计时标签内提交的工作。
采集后，在有 Nsight Systems 2026.6.1 和 `DumpTimeline` 的分析机上运行以下命令。

```bash
uv run --active --no-sync python examples/gb200/hbm-conflict/analyse_ce_hbm.py \
  --output-dir "$path" --nsys "$NSYS_BIN" --dump-timeline "$DUMP_TIMELINE"
```

分析器逐 rank、逐轮核对目标 GPU 和 payload 字节数。CPU 源必须是 `Pinned → Device`，并逐笔确认
硬件通道为 `Async Memcpy`。GPU 源同样检查硬件通道。Nsight 的 `HtoD` / `PtoP` 标签用于定位，
P2P 的 source/destination 轨道只计一次。`Compute`、缺少通道证据或缺少预期拷贝时，分析返回非零。
GEMM 覆盖率按实际写入该 GPU 的 payload 时间线计算。

### 全范围 GEMM 争用 sweep

两条 sweep 都使用 BF16 `[M,8192,7168]`，M 从 1 倍增到 8192，每组默认计时 16 次 GEMM。
用以下命令在同一张 GPU 上采独跑基线，并在四条 CE AllGather 路径下对比。选项 `--mc-self-import 1`
复用前述 R580 multicast 库。`--m-values 2,4,8` 和 `--cases task7,task8` 可缩小试跑范围。

```bash
uv run --active --no-sync python examples/gb200/hbm-conflict/run_ce_hbm_kernel_sweep.py \
  --gpus 0,1,2,3 --mc-self-import 1
```

Nsight Systems 采集 CUDA kernel、GPU memcpy 和 NVTX 时间线。每条 AG 路径从每张 GPU **1 GiB 输出**开始，
默认至少运行 25 轮 AG。在同一轮 payload 中完整覆盖至少两个首卡 GEMM 后接受该样本。有效 trace 的覆盖不足时，
只对当前 M 和 AG 路径将输出大小翻倍重跑，默认上限为 8192 MiB。每次尝试的 capture 都保留在结果目录。
每个并跑均值使用所有完整落在单轮 payload 区间内的 GEMM。若一次 GEMM 发起多个 kernel，覆盖判定用该次 GEMM
的首尾区间，耗时用 kernel 活跃时间。`report.md` 列出 AG 大小、实际轮数、GEMM 入选数量、均值和相对同 M
独跑的变化。`summary.json` 记录每轮 AG 和每次 GEMM 的覆盖关系。

用以下命令采独跑 GEMM 的流量计数器。在两次采集中选择同一张物理 GPU，保持硬件和锁频一致。

```bash
uv run --active --no-sync python examples/gb200/hbm-conflict/run_ncu_hbm_gemm_sweep.py \
  --gpu 0
```

NCU 在 application replay 中沿用预热和外部锁频。脚本在目标 GPU 查询指标，DRAM 采逐 kernel 的读写字节数，
也支持将 sector 计数按每 sector 32 B 换算。NVLink RX/TX 采 wire bytes，C2C RX/TX 采 user bytes，使用独跑
时的整卡 PM 采样，间隔为 1 µs。每个 M 保留 `.ncu-rep` 和 raw CSV。`summary.json` 汇总每次 GEMM 的字节数、各组均值、
总量及基于 NCU kernel 时长的十进制 GB/s。目标 GPU 或 profiler 未提供的计数器保留为 `null`。
比较争用耗时时使用 Nsight Systems 的同批 GEMM 独跑和并跑值。两条 sweep 都支持 `--dry-run`。

### R580 上的 multicast handle self-import 对照

在同一台 R580 GB200 节点，用 NCCL `v2.32.3-1` 构建带 self-import 开关的库。启用时，LSA team
的 rank 0 导入自己导出的 FABRIC multicast handle，用导入的 handle 建立映射。两组共用同一个
NCCL 二进制，task 3 仍由每张卡独立分配 256 MiB HBM 输入和 1 GiB HBM 输出。这个实验使用
PyTorch PR #192530 的 handle 生命周期处理。 [PyTorch PR #192530](https://github.com/pytorch/pytorch/pull/192530)

```bash
bash install/install-nccl-2.32.3-ce-multicast-handler.sh \
  --cuda-home "${CUDA_HOME:-/usr/local/cuda-13.0}"

bash examples/gb200/hbm-conflict/run_ce_hbm_benchmark.sh \
  --gpus 0,1,2,3 --mc-self-import ab --baseline
```

`--mc-self-import ab` 会自动为 AG 作业预加载独立安装脚本默认生成的 `libnccl.so.2`，依次运行关闭组和启用组，
结果存入同一目录下的 `self-import-0` 和 `self-import-1`。省略 `--cases` 时，两组默认各运行 `task3`，
加 `--baseline` 后各追加 `baseline2`。单跑本例的一组用 `--mc-self-import 0` 或 `1`，并指定
`--cases task3,baseline2`。自定义安装路径可用 `--nccl-lib PATH` 覆盖。
PyTorch 需要动态链接到单一 `libnccl.so.2`，worker 会检查实际加载路径。每组记录库的 SHA256，
启动器检查开关、NCCL 构建标识和 multicast 路径，并要求启用组的 rank 0 日志出现
`CE multicast rank 0 self-imported FABRIC handle`。

用 Nsight Systems 打开两组 task 3 和 baseline 2 的 `.nsys-rep`，对照同一物理 GPU 上的
256 MiB payload 拷贝。判断依据是 GPU 0 的 `Channel type` 是否从 `Compute` 变成
`Async Memcpy`，并检查所有 rank 的 payload 拷贝数量与各自 `comm_rounds_measured` 一致，
且 `output_correct` 为 true。P2P 的 source 和 destination 两条轨道显示同一笔拷贝，只数 source。
对照 baseline 2 搬运窗口内的 `SMs Active`，可以检查无 GEMM 时 GPU 0 的 SM 活动是否随之消失。
每轮 4 B 的同步写与 payload 分开核对。

有 Nsight Systems `DumpTimeline` 的分析主机也可以对每份报告统计「次数、CUDA deviceId、硬件通道」。
将 `DUMP_TIMELINE` 设为该二进制路径，`report` 设为待查的 `.nsys-rep`。

```bash
(
  set -o pipefail
  "$DUMP_TIMELINE" --input "$report" --full --format=n,d --show-all |
    jq -er '
      .GenericHierarchyRows[]
      | select(.Path | test("/CudaDevices/[0-9]+/.*/Streams/[0-9]+/Memory/Memcpy/"))
      | . as $row | .Rows[].ViewDataElements[]
      | select(.Text | endswith(" (destination)") | not)
      | select(.Tooltip | contains("268435456&nbsp;bytes"))
      | [($row.Path | capture("/CudaDevices/(?<gpu>[0-9]+)/").gpu),
         (.Tooltip | capture("Channel&nbsp;type:&nbsp;(?<type>[^<]+)").type | gsub("&nbsp;"; " "))]
      | @tsv' | sort | uniq -c
)
```

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
