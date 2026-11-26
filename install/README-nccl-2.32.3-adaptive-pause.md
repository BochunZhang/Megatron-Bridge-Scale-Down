# NCCL 2.32.3 adaptive pause

安装脚本基于 NCCL `v2.32.3-1`（`12df1a11afad322be5a204a2db890161cbf8131d`），面向 GB200 的
`sm_100` 编译。产物的版本横幅为 `NCCL version 2.32.3-adaptive-pause+cuda…`，版本号为 23203，soname 为
`libnccl.so.2`。见[安装脚本](install-nccl-2.32.3-adaptive-pause.sh)和[补丁](patches/nccl-2.32.3-adaptive-pause.patch)。

CUDA 模块在每个 context 中有一份初始为 0 的 32 位显存计数器。同一 context 的不同 communicator 共用该地址。
`ReduceScatter_LDMC` 每个 CTA 的 lane 0 在 opening barrier 后递增计数，等 CTA 通过 closing barrier 后递减。
`AllGather_STMC` 和 `AllGather_TmaSTMC` 的每个 warp 在第一个对齐的 4 KiB tile 之前读取计数，
随后每隔若干 tile 再由 lane 0 读取并广播。每个 tile 发出后用 `clock64()` 等待选定的周期。
读到非零就切到平衡档，读到零就恢复 AG 单独运行的档位。

| 环境变量 | 默认值 | 含义 |
| --- | ---: | --- |
| `NCCL_SYM_AG_STMC_PAUSE_CYCLES` | -1 | 无 RS 时，每个 warp 每个 4 KiB tile 的等待周期。负数选下表默认值，非负数指定实际周期 |
| `NCCL_SYM_AG_STMC_RS_PAUSE_CYCLES` | -1 | 有 RS 时的等待周期。负数选默认平衡档，0 禁用计数和轮询。正数覆盖默认值，须高于 AG 单独运行的档位才会启用轮询 |
| `NCCL_SYM_AG_STMC_PAUSE_POLL_CHUNKS` | 4 | 两次读取之间相隔的每 warp tile 数，最小为 1 |

## GB200 上的候选档位

下表取自 1 GiB BF16、约 2.05 GHz、NCCL 默认 CTA 数、`AllGather_STMC + ReduceScatter_LDMC` 的
静态 pause 扫参。单位为每 warp 每 4 KiB 的 SM cycles。AG 单跑默认档取带宽平台上沿，相对 0 cycles 的带宽变化
小于 0.4%。见[实验记录 §6.4](../documents/gb200/03-nccl-symmetric-nvls-kernel.md)。

| GPUs | AG / RS CTA | AG 单跑备选 | AG 单跑默认 | 有 RS 时轻 / 默认平衡 / RS 优先 | 平衡档静态 AG+RS (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 4 / 6 | 512 | 1024 | 2560 / **3072** / 4096 | 2.436 |
| 8 | 2 / 4 | 512 | 1024 | 2560 / **3072** / 4096 | 2.281 |
| 16 | 1 / 2 | 512 | 1024 | 2560 / **3072** / 4096 | 2.285 |
| 32 | 1 / 1 | 3072 | 4096 | 6144 / **7168** / 10240 | 2.179 |
| 64 | 1 / 1 | 8192 | 9728 | 14336 / **16384** / 18432 | 2.256 |

两档按 AG communicator 的 rank 数选择，每 GPU 一个 rank。其他规模在 AG 单跑时用 0 cycles，有 RS 时按发送 warp 数线性估算。
动态轮询开销和 TMA 档位需用新库实测。见[静态扫参 CSV](../documents/gb200/figures/03-pause-cycle/pause_sweep_summary.csv)。

## 编译与对照

在 GB200 Linux 节点的仓库根目录运行。设置 `UV_PROJECT_ENVIRONMENT`，让 benchmark 使用动态链接 NCCL 2.32.3
的 PyTorch 环境。见[PyTorch 编译说明](README-torch-2.13.0-nccl-2.32.3.md)。

```bash
bash install/install-nccl-2.32.3-adaptive-pause.sh
export UV_PROJECT_ENVIRONMENT="$PWD/.venv/python3.12-torch2.13-nccl2.32.3"
lib="$PWD/nccl/nccl-2.32.3-adaptive-pause/build/lib/libnccl.so.2"

# RS pause 0 是同库对照，-1 选择自动档。
for rs_pause in 0 -1; do
    LD_PRELOAD="$lib" NCCL_SYM_TMA_ENABLE=0 NCCL_SYM_AG_STMC_PAUSE_CYCLES=-1 \
        NCCL_SYM_AG_STMC_RS_PAUSE_CYCLES="$rs_pause" NCCL_SYM_AG_STMC_PAUSE_POLL_CHUNKS=4 \
        bash examples/gb200/nvls-collective/run_nvls_collective_benchmark.sh \
        --algos symm --symm-kernel mc --group-sizes 4 --sizes-mib 1024 --workloads ag,rs,ag+rs --nsys
done
```

核对 NCCL 日志里的版本横幅、kernel 和 CTA 数。对比 AG 单独带宽、AG+RS makespan 及两者在 Nsight Systems
中的完成顺序。用 RS 先结束且 AG 仍有剩余 tile 的档位观察恢复快档。用 `--late-join` 单独验证 AG 在 RS 已经开始后入场。
跨 tray 时每个节点都运行对应命令，并传入相同的 `--nnodes` 和 `--group-sizes`。

## 4 GiB AG 先跑，1 GiB RS 中途加入

在同一台 GB200 节点选 4 张 GPU。PyTorch 用两个独立的 NCCL communicator 和对称窗口运行 BF16。
每卡的 AG 输入为 1 GiB，输出为 4 GiB。RS 输入为 1 GiB，输出为 256 MiB。
AG 与 RS 各用一条 CUDA stream，RS 默认在共同 GPU 起跑点后延迟约 1 ms。预热在采集前完成。

```bash
bash examples/gb200/ncu-torch-all-gather/run_nsys_torch_adaptive_pause.sh --gpus 0,1,2,3
```

两轮都用同一份 adaptive-pause 库和同样的 AG 单跑档 `1024`。第一轮用 RS 平衡档 `3072`，
第二轮把 RS 档设为 0 作为对照。`--rs-delay-ms` 可以改变加入时间，
`NCCL_SYM_AG_STMC_PAUSE_POLL_CHUNKS` 可覆盖默认的 4 tile。用 `--nccl-lib` 和 `--nsys` 指定安装路径。

Nsight Systems 需要 GPU 性能计数器权限。每轮用一个会话对四张卡采样，设置为 **200 kHz（标称 5 µs）**，
采集 NVLink RX/TX 的 request/response user/protocol 字节数。采样分析会检查间隔，发现缺样会报错并保留原始报告。

每轮在 `adaptive/` 或 `control/` 下保存 `profile.nsys-rep`、`profile.sqlite`、`kernel_timing.csv` 和
`nvlink_by_phase.csv`。分析使用 CUDA kernel 时间戳核对 AG 先开始以及四张卡确实同时执行两个 kernel。
Adaptive 轮还要求最后一个 RS 比第一个 AG 先结束，顺序不满足时测试返回非零。对照轮也保留相应尾段。
打开 `adaptive/profile.nsys-rep`，对着 RS 的起止看 NVLink request user data 的三段速率，并对比
`control/profile.nsys-rep`。Response user data 可用来观察 RS 期间的读取应答。
