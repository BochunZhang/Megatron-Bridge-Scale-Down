# GB200：流入 G0 的 NVLink 流量会不会拖慢 G0 发出的 P2P 流

日期：2026-11-02。用 NCCL P2P（torch `isend`/`irecv`）在一台 GB200 compute tray 的 4 张 GPU 上测：
G0 在发送测试流的同时被其他 GPU 写入数据，测试流的带宽会不会下降。所有流量都经过 NVSwitch。

## 场景

4 个 rank 分别扮演 G0..G3，`--gpus` 指定每个角色用哪张物理 GPU，默认 `0,1,2,3`。
每条流每个 trial 发送 `--size-mib`，默认 1 GiB：

| 场景 | 测试流 | 背景流 |
| --- | --- | --- |
| `alone` | G0→G3 | 无（基线） |
| `bg1` | G0→G3 | G1→G0 |
| `bg2` | G0→G3 | G1→G0、G2→G0 |
| `bg3` | G0→G3 | G1→G0、G2→G0、G3→G0（G0↔G3 变成双向） |

push 模式（P2P write，GB200 上 NCCL 的默认）下，测试流的数据走 **G0 的 NVLink TX**，
背景流的数据走 **G0 的 NVLink RX**。所以这组场景测的是：G0 入方向的 incast 是否影响 G0 出方向的写，
包括反方向上的写响应和 credit 回传。`bg3` 中 G3 同时接收测试流并向 G0 发送，G0↔G3 是双向满载。

"incast 占满某个出端口，credit 反压阻塞发往其他端口的流"（例如 G0、G1→G2 时测 G0→G1）
是另一种流量形状。worker 的 `SCENARIOS` 表是通用的，加一行就能扩展。

## 实现要点

- **每条流有独立的 2-rank NCCL communicator**，因此有独立的 NCCL stream 和独立的
  `ncclDevKernel_SendRecv` kernel。如果把 G0 的 send 和 recv 放进同一个 group 调用，NCCL 会把它们
  合成一个 kernel，同一个 CTA 里的 warp 一半做 send、一半做 recv（`device/sendrecv.h`）。
  这会额外引入一种 SM 层面的耦合，所以刻意避开。
- **同时起跑**：每个 trial 先在 GPU 上 sleep（`--launch-delay-ms`，默认 20 ms），再做一次 4-rank
  all-reduce 作为起跑门（gate），各条流排在 gate 之后，分别放在自己的 stream 上。host 端全部入队
  完成后 sleep 才结束，所以所有 GPU 上的流在 gate 之后几微秒内同时开始。如果 sleep 在入队完成前
  就结束了，该 trial 会被记为 `late launches`。
- **计时**：每条流从 gate 结束到接收端最后一个字节落地的时间（CUDA event），发送端时间也一并记录。
  所有流同时起跑，所以"背景流结束得不早于测试流"就等价于"背景流覆盖了测试流的全过程"，
  记为 `bg covered test`。
- **校验**：每条流的发送端在 NCCL INIT 日志里必须出现 `via P2P/...`。push 模式不能带 `/read`，
  pull 模式必须带 `/read`，否则作业直接失败。接收缓冲区在 warmup 后和每个场景结束后都会检查
  payload，payload 的值由源 GPU 决定，可以区分是哪条流写进来的。

## nsys：单个文件

launcher 用**一个** `nsys profile` 包住 `uv run ... torch.distributed.run`。nsys 会跟踪整个进程树，
因此一个报告 `nsys/p2p-conflict.nsys-rep` 里同时包含：

- 4 个 rank 的 CUDA 和 NVTX trace；
- 4 张 GPU 的 GPU metrics（`--gpu-metrics-devices=all`，默认 50 kHz）。GB100 的默认 metric set 包含
  `NVLink {TX,RX} {Requests,Responses} {User,Protocol} Data`（占峰值的百分比，所有 link 合计）。

`--nsys-capture range`（默认）只采集计时 trial：每个 rank 在 barrier 后调用
`cudaProfilerStart`/`Stop`，warmup 不在采集范围内。如果所用的 nsys 版本在多进程下的
capture range 行为不符合预期，分析脚本会报告 `no kernel for flow ...`，此时改用
`--nsys-capture all` 记录整个作业。

NVTX range 的命名：

- `scenario <name>: ...`：一个场景的所有 trial；
- `<scenario>/trial<k>/gate`：起跑门的 launch；
- `<scenario>/trial<k>/flow <src>><dst> <send|recv>`：每条流两端各自的 launch。

## 运行

```bash
# 4 个场景，每条流 1 GiB，push，nsys 输出单个报告，跑完后自动分析
bash examples/gb200/p2p-conflict/run_p2p_conflict_benchmark.sh

# P2P read、20 个 trial、只计时
bash examples/gb200/p2p-conflict/run_p2p_conflict_benchmark.sh --p2p-mode pull --trials 20 --no-nsys

# 单独重跑分析（需要 nsys 在 PATH 上，用来做 sqlite 导出）
uv run --no-sync python examples/gb200/p2p-conflict/analyse_nsys.py --output-dir <run-dir>
```

只支持单节点、4 卡。`CUDA_VISIBLE_DEVICES` 会被清掉，`CUDA_DEVICE_ORDER=PCI_BUS_ID`，
这样 CUDA、nvidia-smi 和 nsys 对 GPU 的编号一致。完整参数见 `--help`。

## 输出

默认目录 `results/gb200/p2p-conflict/<yyyymmdd-hhmmss>/`：

| 文件 | 内容 |
| --- | --- |
| `summary.md` | 每个场景测试流的 GB/s、相对 `alone` 的比值、各背景流 GB/s、覆盖率和 late launch 数（CUDA event 口径） |
| `results.jsonl` | 每个场景一条记录，包含每条流每个 trial 的收发两端时间 |
| `metadata.json` | GPU 映射、NCCL 版本、各条流的 transport、NCCL/CUDA 环境变量 |
| `nsys/p2p-conflict.nsys-rep` | 唯一的 nsys 报告 |
| `nsys_summary.md` | 每条流两端的 kernel：GPU、**CTA 数（gridX）**、线程数、相对 gate 的起止时间、kernel 口径 GB/s；每个场景每张 GPU 的 8 个 NVLink 计数器均值 |
| `nsys_flow_kernels.csv` | 每个 SendRecv kernel 一行 |
| `nvlink_windows.csv` | 每个 trial、每张 GPU、每个 NVLink metric 在 trial 窗口内的均值 |
| `nvlink_samples.csv` | trial 窗口内的原始采样点（用于画时间序列） |

## 怎么看结果

1. `summary.md` 里先确认 `bg covered test` 等于 trial 数、`late launches` 为 0，再看
   `vs alone` 随 bg1→bg3 的变化。
2. 再看 `nsys_summary.md` 中 G0（测试流发送端）的计数器：
   - TX Requests User 随背景流增多而下降，同时 RX Requests User 接近饱和：G0 入方向的负载拖慢了出方向的写。
   - TX Requests User 不变，但测试流 kernel 变长：瓶颈不在 G0 的链路上。可以看 G3 一侧的计数器，
     以及 kernel 的 `start us` 是否推迟（推迟说明 SM 不够、kernel 没能按时开始）。
   - TX 方向的 Responses Protocol 随 RX 写入量增长：这是写响应占用了反方向带宽。
3. 每条流两端 kernel 的 `start us` 都应接近 0。G0 在 `bg3` 中同时驻留 4 个 SendRecv kernel，
   如果某个 kernel 的 start 明显偏后，说明 SM 资源受限，那一行的数据不能归因于 NVLink。

## NCCL P2P 用几个 CTA

按 NCCL 2.29.7 推导，单节点（一个 tray）的 2-rank communicator 发送 1 GiB：
**32 个 CTA，每个 CTA 640 线程（20 个 warp）**，接收端同样是 32 个 CTA。推导过程：

1. `graph/paths.cc`：`p2pnChannels` 取 pow2Up(32)，等于 32；`p2pnChannelsPerPeer` 取
   min(32, 32)，等于 32。
2. `enqueue.cc` 中 `nChannelsMin` 从 32 开始减半，直到 `nChannelsMin × nRanks(2) ≤ 32`，得到 16。
3. 每个 channel 的 part = 1 GiB / 16 = 64 MiB，超过 `maxPartSize = 32 × 512 KiB = 16 MiB`，
   于是倍增到 32，part = 32 MiB。此时已达到 `nChannelsMax/2` 的上限，停止。
4. P2P kernel 的 `threadPerBlock = NCCL_MAX_NTHREADS = 640`（`enqueue.cc:1101`）。

NCCL 不会为 P2P 打印 channel 数，所以以 nsys 中 SendRecv kernel 的 `gridX` 为准，
`nsys_summary.md` 的 `CTAs (gridX)` 列就是实测值。实际使用的 NCCL 版本是 torch 自带的版本，
见 `metadata.json` 中的 `nccl` 字段。
