# GB200：用 Nsight Compute 2026 profile torch all-gather 的 NCCL kernel

日期：2026-11-06。按 Nsight Compute CLI 文档（5.3.2 Multi-Process Support）的写法，用**一个 ncu 包住整个
torchrun**，通过 shmem communicator 让各 rank 的 NCCL kernel 同时 launch、同时重放：

```bash
ncu --communicator shmem --communicator-shmem-num-peers 2 -f -o report -k regex:nccl.* \
    torchrun --nnodes=1 --nproc_per_node=2 pytorch_app.py
```

本目录把文档里的 `pytorch_app.py` 换成 `ncu_torch_all_gather.py`，并加上算法固定、TUNING 校验和卡死检测。

> 状态：2026-11-06 在 GB200 单节点 4 卡上用 ncu 2026.3 跑过 `--algo nvls`（默认参数）：communicator 正常，
> 失败在 kernel replay 保存显存，见下文「实测：NVLS + kernel replay 保存显存失败」。

## 文件

| 文件 | 作用 |
| --- | --- |
| `ncu_torch_all_gather.py` | torchrun worker：warmup，再在 NVTX range `ncu_all_gather` 里跑 `--iters` 次 all-gather，前后都校验数据 |
| `ncu_torch_ag_rs_overlap.py` | `--ag-rs` 用的 worker，只跑 AG + RS：两者各一个 NCCL communicator 和一条 CUDA stream，同时 launch（并发写法参照 `../nvls-collective/nvls_collective_benchmark.py`），NVTX range 为 `ncu_collectives`，前后校验两者的结果，并用 CUDA event 打出每次迭代 AG 和 RS 的重叠时间 |
| `run_ncu_torch_all_gather.sh` | 单节点启动器：baseline（不挂 ncu）+ ncu 两遍；默认跑 all-gather，`--ag-rs` 跑 AG + RS 同时执行 |
| `ncu_tcp_ranks.sh` | `--communicator tcp` 时由启动器调用：每个 rank 起一个 ncu、各包一个 worker，设好 torchrun 会设的 `RANK` 等变量，每行输出加 `[rank <r>]` 前缀 |
| `bisect_ncu_sections.sh` | 逐个 section 跑启动器（application replay），找出让 kernel 卡死的 section，见下文「实测：application replay + `--set full` 卡在 pass 39」 |
| `sweep_ncu_replay_modes.sh` | 逐个 replay 模式 × communicator 组合跑启动器（默认 `--ag-rs`），找出哪些组合能 profile 并发的 AG + RS，见下文「AG + RS：replay 模式 × communicator」 |
| `sweep_symm_ag_pause_blocking_metrics.sh` | 用 `.venv/python3.12-torch2.13-nccl2.32.3` + `nccl-2.32.3-pause` 逐个 `NCCL_SYM_AG_STMC_PAUSE_CYCLES` 跑启动器的 `--blocking-metrics`（symm，application + shmem），每个 pause 一个文件夹，见下文用法 |
| `sweep_symm_ag_pause_rs_blocking_metrics.sh` | 同上，但 AG（加 pause）和 RS（不加）同时执行：启动器 `--ag-rs --blocking-metrics`，ncu 2026.4 + app-range + tcp；默认 pause 点取自 4 卡 AG + RS pause 图，summary.txt 同时列出 baseline 的 AG done / RS done 和 ncu range 时长 |
| `analyse_blocking_metrics.py` | `--blocking-metrics` 结束后由启动器调用：把 `summary.csv` 按数据流经的各级（发端 SM→XBAR、NVLink、接收端 L2、HBM）排成 `blocking.txt`，字节数以每卡分片 S 的倍数显示 |
| `README-test-records.md` | 各种 ncu 版本 / replay 模式 / 算法组合的成功和失败记录，以及四种 replay 模式的区别 |

## 用法

在仓库根目录执行，需要 Nsight Compute ≥ 2026.4 和 GPU 性能计数器权限。仓库根目录下有
`NVIDIA-Nsight-Compute-2026.4/` 时，脚本会自动把它放到 PATH 最前面，不用传 `--ncu`；版本低于 2026.4 会直接报错。

`--algo` 的四种 all-gather，每次 launch 都用 TUNING 日志核对：

| `--algo` | 实现 | NCCL 设置 | kernel |
| --- | --- | --- | --- |
| `ring` | ring all-gather | `NCCL_ALGO=Ring` | `ncclDevKernel_AllGather_RING[_<proto>]` |
| `nvls` | NVLS multicast all-gather | `NCCL_ALGO=NVLS NCCL_NVLS_ENABLE=1` | 同上（算法在 kernel 内部分派） |
| `symm` | symmetric kernel，multicast 实现 | 注册 symmetric window，`NCCL_SYM_KERNEL` 固定 `AllGather_STMC` | `ncclSymkDevKernel_AllGather_STMC` |
| `p2p` | symmetric kernel，p2p 实现 | 注册 symmetric window，固定 `AllGather_ST`，`NCCL_NVLS_ENABLE=0` | `ncclSymkDevKernel_AllGather_ST` |

`symm` / `p2p` 都把 `NCCL_SYM_CE_THRESHOLD` 设成 INT64_MAX，否则 Blackwell 上大 AllGather 会走 copy engine。
`NCCL_SYM_KERNEL` 的写法取决于 torch 实际加载的 NCCL（脚本启动时探测一次）：2.32 之前是裸 kernel 名
`AllGather_STMC`，2.32 起是 `AllGather:AllGather_STMC`。

```bash
# ncu 2026.3 可用：单遍 metric 的 kernel replay，不保存显存
bash examples/gb200/ncu-torch-all-gather/run_ncu_torch_all_gather.sh --metrics gpu__time_duration.sum

# 判断保存失败是否由 multicast 映射引起：Ring 且关闭 NVLS，communicator 里没有 multicast 内存
NCCL_NVLS_ENABLE=0 bash examples/gb200/ncu-torch-all-gather/run_ncu_torch_all_gather.sh --algo ring

# ncu >= 2026.4：application replay 不保存显存，可以采完整的 --set
bash examples/gb200/ncu-torch-all-gather/run_ncu_torch_all_gather.sh --replay-mode application --set full

# ncu >= 2026.4：只采指定的 section（可重复，代替 --set）
bash examples/gb200/ncu-torch-all-gather/run_ncu_torch_all_gather.sh --replay-mode application \
    --section SpeedOfLight --section Occupancy

# ncu >= 2026.4：把 --set full 的 section 逐个单独跑，找出卡死的那个（-- 之后是启动器参数）
# 默认：symm、1 GiB、--set full 的每个 section 单独跑，结果在 results/gb200/ncu-torch-all-gather/symm-all-gather-<时间>/<section>/
bash examples/gb200/ncu-torch-all-gather/bisect_ncu_sections.sh
bash examples/gb200/ncu-torch-all-gather/bisect_ncu_sections.sh --algo p2p --size-mib 256

# AG 和 RS 同时执行：只有 range / app-range 不把两个 kernel 串行化。
# shmem + app-range 在 2026.4 上失败（#14），先用 sweep 找出能用的 replay × communicator 组合
bash examples/gb200/ncu-torch-all-gather/sweep_ncu_replay_modes.sh
bash examples/gb200/ncu-torch-all-gather/sweep_ncu_replay_modes.sh --combos range:tcp,app-range:tcp -- --warmup 2
# 单个组合：每个 rank 一个 ncu（tcp communicator），报告是 ag_rs-rank<r>.ncu-repz
bash examples/gb200/ncu-torch-all-gather/run_ncu_torch_all_gather.sh --algo symm --ag-rs \
    --replay-mode app-range --communicator tcp --size-mib 1024 --section SpeedOfLight
# 按 section 拆开采 AG + RS 重叠（换成 sweep 找到的组合）
bash examples/gb200/ncu-torch-all-gather/bisect_ncu_sections.sh --ag-rs --replay-mode app-range -- --communicator tcp

# symmetric AllGather_STMC / AllGather_ST，换成 2.32.3 的 NCCL
bash examples/gb200/ncu-torch-all-gather/run_ncu_torch_all_gather.sh --algo symm --replay-mode app-range \
    --nccl-lib nccl/nccl-2.32.3/build/lib/libnccl.so.2

# 各级反压：发端 l1tex2xbar stalled/active、warp stall、NVLink 字节（PM sampling）、L2 入口、ltcfabric、HBM，
# 以及 GB200 的实际频率、跨 die 读写、到 Grace 的 C2C / sysmem 流量
# 自动加 --clock-control none，结果在 blocking.txt；多 pass，symm / nvls 必须用 application 或 app-range
# NVLink rx 的 "% of peak" 以 ncu 的约 1106 GB/s 为分母，打满时约 81%（≈ 894 GB/s），不是 100%
bash examples/gb200/ncu-torch-all-gather/run_ncu_torch_all_gather.sh --algo symm --size-mib 1024 \
    --replay-mode application --blocking-metrics

# 逐个 pause 采各级反压：symm AllGather_STMC，nccl-2.32.3-pause（install/install-nccl-2.32.3-pause.sh），
# 激活 .venv/python3.12-torch2.13-nccl2.32.3；每个 NCCL_SYM_AG_STMC_PAUSE_CYCLES 一次启动器 --blocking-metrics，
# 结果在 results/gb200/ncu-torch-all-gather/symm-all-gather-blocking-metric-pause-<时间>/pause-<C>cyc/，
# summary.txt 每个 pause 一行：CTA 数、SM 频率、pause 折成的 ns、kernel 时长、algbw
bash examples/gb200/ncu-torch-all-gather/sweep_symm_ag_pause_blocking_metrics.sh
bash examples/gb200/ncu-torch-all-gather/sweep_symm_ag_pause_blocking_metrics.sh --pause-cycles 0,1024,4096 -- --gpus 0,1

# AG（加 pause）+ RS（不加）同时执行时逐个 pause 采各级反压：ncu 2026.4、app-range + tcp（#15 唯一能 profile
# 并发 AG + RS 的组合），metric 属于整个 NVTX range（AG + RS 合在一起），每个 rank 一个结果；
# 默认 pause 点 0,128,...,3072,4096,6144 即 4 卡 AG + RS pause 图上的点（makespan 最低在 3072）。
# 结果在 results/gb200/ncu-torch-all-gather/symm-ag-pause-rs-blocking-metric-<时间>/pause-<C>cyc/，
# summary.txt 每个 pause 一行：AG / RS 的 CTA 数、SM 频率、pause 折成的 ns、baseline 的 AG done / RS done、range 时长
bash examples/gb200/ncu-torch-all-gather/sweep_symm_ag_pause_rs_blocking_metrics.sh
bash examples/gb200/ncu-torch-all-gather/sweep_symm_ag_pause_rs_blocking_metrics.sh --pause-cycles 0,2048,3072,4096 -- --no-lockstep

# 复现不加 communicator 时的卡死（由 --timeout 兜底）
bash examples/gb200/ncu-torch-all-gather/run_ncu_torch_all_gather.sh --communicator none --timeout 300
```

全部选项见 `--help`。默认使用 torch 自带的 NCCL；`--nccl-lib` 会通过 LD_PRELOAD 换成指定的库。

## `-k regex:...` 是什么

`-k` 就是 `--kernel-name`，按 kernel 函数名过滤：只有名字匹配的 kernel 才会被 profile，其他 kernel 照常
运行，不被串行化也不被重放。`regex:` 前缀表示后面是正则表达式，按子串搜索。文档示例里的
`regex:nccl.*` 会匹配所有 NCCL kernel。

这里用的是更窄的 `regex:^nccl(Symk)?DevKernel_AllGather`，原因如下：

- `dist.barrier()` 走 NCCL 时会 launch 一个 AllReduce kernel，用 `nccl.*` 会把它也选进来。worker 改成
  通过 gloo group 做 barrier，进程里唯一的 NCCL 工作就是 all-gather。
- NCCL 2.32.3 每个集合通信只编译一个特化 kernel（`src/device/generate.py` 的 `best_kernel`）。Ring 和
  NVLS 的 AllGather 都 launch `ncclDevKernel_AllGather_RING`（不设置 `NCCL_EXACT_KERNEL_NAMES=1` 时
  kernel 名里不带 protocol），实际算法在 kernel 内部分派，所以
  **单看 kernel 名分不出 Ring 和 NVLS**。脚本靠 TUNING 日志（`AllGather: <n> Bytes -> Algo NVLS ...`）
  确认每次 launch 的算法。symmetric 路径的 kernel 名是 `ncclSymkDevKernel_AllGather_STMC`。

同一个 kernel 名会出现在 warmup 和正式迭代里，所以还要加 `--nvtx --nvtx-include "ncu_all_gather/"`，只选
NVTX range 里的那几次 launch。这样就不用按 `--launch-skip` 去数 warmup 有几次。

## `pytorch_app.py` 要怎么写

对照 `ncu_torch_all_gather.py`，有以下几条要求：

1. **每个进程一张 GPU**：先 `torch.cuda.set_device(LOCAL_RANK)`，再
   `init_process_group("nccl", device_id=...)`。传 `device_id` 会立即创建 communicator，避免把建
   communicator 的开销算进第一次 all-gather。
2. **所有 rank 的 kernel 序列必须一致**：communicator 按顺序对齐各 rank 的第 k 个被选中的 kernel。
   不能出现某个 rank 多跑一次的情况，比如只在 rank 0 上做检查。
3. **只让要 profile 的 NCCL kernel 进入过滤范围**：barrier 用 gloo group，或者把 `-k` 写窄一些。
4. **用 NVTX range 标出要 profile 的区间**：`range_push` / `range_pop` 必须在 launch kernel 的同一个线程上
   调用（torch 在调用线程里 enqueue NCCL）。
5. **symm 需要先注册 window**：buffer 从 `torch.cuda.MemPool(backend.mem_allocator)`（底层是
   `ncclMemAlloc`）分配，再调用 `register_mem_pool(pool, symm=True)` 注册；结束前要先
   `deregister_mem_pool`。
6. **application / app-range replay 每个 pass 都会重新启动整个 torchrun**：程序必须能重复执行，并且
   每次 launch 顺序都相同。启动器用了 `--standalone`，每次都会自动选一个空闲的 rendezvous 端口。

## 实测：NVLS + kernel replay 保存显存失败

2026-11-06，GB200 单节点 4 卡，ncu 2026.3，torch 2.13 自带的 NCCL 2.32.3，默认参数（`--algo nvls`、
`--set basic`、`--replay-mode kernel`）。baseline 通过，6 次 AllGather 都是 `Algo NVLS`。ncu 那一遍：

```text
==PROF== Profiling "ncclDevKernel_AllGather_RING" in process 220657
==PROF== Profiling "ncclDevKernel_AllGather_RING" in process 220658
==ERROR== Failed to save memory for replay. If the issue persists, consider using --replay-mode application ...
==ERROR== ContextSaveFailed
==PROF== Profiling "ncclDevKernel_AllGather_RING" in process 220659
==ERROR== Failed to profile "ncclDevKernel_AllGather_RING" in client process 220659
==PROF== Waiting for 3 clients
==ERROR== Failed to wait for profiling
==ERROR== The application returned an error code (9).
```

`/tmp/nsight-compute-*.log` 中的调用链：`cuda_context_state>> Failed to copy memory` → `Failed to transfer context
state!` → `Failed to save context state` → `cuda_replay>> Failed to create replay state`。

- **communicator 没有问题**：多个 rank 同时停在同一个 NCCL kernel 上，没有出现串行化导致的卡死。
- **失败在复制显存**：Profiling Guide 写明，kernel replay 需要多遍时，会在第一遍之前保存 kernel 能访问的全部
  显存；只需要一遍时完全不保存。日志里没有 `Backing up device memory in system memory`，buffer 也只有
  256 MiB，所以不是容量不足，而是某段内存复制不了。
- **推测是 multicast 映射**（尚未证实）：NVLS 用 `cuMemCreate`/`cuMemMap` 建 UC buffer，再用
  `ncclMcPartitionBindMem` 绑定到 multicast 对象（2.32.3 `src/transport/nvls.cc`），进程中同时映射着 UC 和 MC。
  `../nccl-tests/README.md` 里的 STMC（symmetric window，同样带 multicast）用 tcp 和 shmem 都是同样的错误。
  用上面 `NCCL_NVLS_ENABLE=0 --algo ring` 那条命令可以验证。
- **ncu 2026.3 下 application replay 用不了**：2026.4 的 release notes 才写了 *Added application replay
  support when profiling with the shmem communicator*。这也解释了 nccl-tests 实验中 shmem + application 只得到
  `No kernels were profiled`。脚本在 shmem + application / app-range + ncu < 2026.4 时会直接报错。注意 2026.4
  的 CLI 文档仍然写着 shmem 只支持 kernel 和 range replay；release notes 只加了 application，**没有 app-range**，
  实测 shmem + app-range 失败（见 [README-test-records.md](README-test-records.md) #14）。
- range replay 预计也走不通（未实测）：它同样要保存和恢复内存，而且文档列出的支持 API 中 Virtual Memory
  Management 和 Peer Context Memory Access 都是 *Not supported*，Execution Control 里也没有 NCCL 用的
  `cuLaunchKernelEx`。不过这些限制只针对 range **内部**的 API 调用，range 里只有 event 和 NCCL launch。

截至 2026-11-06，最新版本是 2026.4。它的 release notes 和 Known Issues 都没有提到 multicast / VMM 内存的
保存问题，所以 2026.4 下 kernel replay 的多遍采集是否仍然失败，需要实测。2026.4 的改进是可以绕开保存：
用 application replay。另外 Known Issues 写明 NVLink Topology section 不支持 NVSwitch 拓扑，也不支持
per-NVLink metrics。

## 实测：application replay + `--set full` 卡在 pass 39

2026-11-07，GB200 单节点 4 卡，ncu 2026.4，torch 2.13（运行时 NCCL 2.29.7），`--algo nvls --replay-mode
application --set full`：

- shmem 下的 application replay 能用：每个 pass 重启一次整个 torchrun，约 14 s，其中被 profile 的那一段
  （5 次 warmup + 1 次 AllGather + 校验）只要 1.2–1.7 s，各 pass 之间没有越来越慢。pass 1–38 都是 4 个 rank
  同时进入 kernel，并且通过了数据校验。
- pass 39 里 4 个 rank 都打印了 `Profiling ... pass 39`，但 kernel 一直没有跑完（没有 `output correct`），之后
  ncu 报 `Failed to wait for profiling`。最后的 `Unexpected number of profiled kernels` 是这个 pass 没有产出
  结果导致的，不是根因，调 `--app-replay-match` 没用。
- NCCL 2.29.7 的 kernel 名是 `ncclDevKernel_AllGather_RING_LL`，名字里带 protocol。`-k` 正则两种名字都能匹配。
- 重复了 4 次，都卡在 pass 39：`nvls` 256 MiB 两次，`nvls` 1 MiB 一次，`NCCL_NVLS_ENABLE=0 --algo ring` 一次。
  所以和 multicast、NVLS 无关；数据量小了 256 倍也一样，所以不是 pass 太慢，而是卡死。
- ncu 不是等到脚本的 `--timeout` 才退出：pass 39 开始后不到约 2.5 min 就自己报 `Failed to wait for profiling`。
  没有找到能调整这个等待时间的选项；而且既然是卡死，调大也没用。启动器现在把这种情况报成
  `ncu gave up waiting for the profiled kernel (hang in replay pass N)`。

ncu 不会打印每个 pass 采了哪些 metric。Profiling Guide 写明软件插桩（SASS patching）类 metric 单独占 pass，
PM sampling / warp state sampling 的 pass 数是动态决定的，这两类很可能排在 full 的最后几个 pass。剩下两种解释：

1. 某个 section 的采集方式（推测是插桩或 sampling）让 NCCL kernel 跑不完：这个 section 单独跑也会卡；
2. torchrun 重启到第 39 次时碰到 ncu 的某个次数上限：每个 section 单独都能跑完，只有合起来要 39 个以上
   pass 时才卡。

`bisect_ncu_sections.sh --set full` 从 ncu 的 `*.section` 文件里读出 full 的全部 section，每个单独用
`--section` 跑一次 application replay，在 `summary.txt` 里记下 ok / hang / fail / unknown 和用了几个 pass。
用 `A+B` 可以把几个 section 放在一次运行里，用来凑出 39 个以上 pass，检验第 2 种解释。

## 输出

默认目录是 `results/gb200/ncu-torch-all-gather/<algo>-all-gather-<时间>/`。`bisect_ncu_sections.sh` 在同一个目录下
每个 section 一个子目录（内容同下表，外加 `launcher.log`），顶层是 `summary.txt`：

| 文件 | 内容 |
| --- | --- |
| `config.txt` | ncu 版本、参数、所有 `NCCL_*` 环境变量 |
| `baseline.log`、`ncu.log` | 两遍的输出，其中有 worker 的数据校验结果和 ncu 的 `==PROF==` 输出 |
| `nccl-logs/<pass>/nccl-<host>-<pid>.log` | NCCL INFO 日志（INIT、ENV、TUNING） |
| `all_gather.ncu-rep` / `all_gather.ncu-repz` | shmem 模式下所有 rank 合在一份报告里；ncu 2026.4 默认写压缩的 `.ncu-repz`；`--communicator tcp` 时每个 rank 一份 `all_gather-rank<r>.ncu-repz` |
| `summary.csv` | 每个被 profile 的 kernel（range / app-range 下是每个 range）一行，tcp 时把各 rank 的报告接在一起；`--blocking-metrics` 时是报告里的全部 metric |
| `blocking.txt` | 仅 `--blocking-metrics`：各级反压和流量，每个结果一列 |

## 读结果时注意

- ncu 默认会锁频、刷 cache、kernel replay 时还会保存和恢复显存，所以报告里的耗时**不等于**真实性能。
  带宽请看 nsys 或 `../nvls-collective` 的 benchmark。ncu 用来看 kernel 内部的情况。
- 文档提到，shmem 模式下 `--communicator-shmem-num-peers` 不是必填，但建议填。不填时，如果各 rank 没有
  同时到达同一个 kernel，ncu 可能在参与者没到齐的情况下就开始 profile。
- 默认只 profile 每组的第一个 CUDA context；需要选 GPU 时，用 `-- --devices ...` 透传给 ncu。

## AG + RS 同时执行（`--ag-rs`）

- RS 用的实现：ring / nvls 和 AG 一样由 `NCCL_ALGO` 决定；symm 是 `ReduceScatter_LDMC`（multicast load-reduce），
  p2p 是 `ReduceScatter_LD`。TUNING 检查对 AllGather 和 ReduceScatter 分别逐条核对。
- NCCL 2.29.7 的 `NCCL_SYM_KERNEL` 只能写一个 kernel 名，固定了 AG 的 kernel 会让 RS 没有 symmetric kernel 可用，
  所以 symm / p2p 的 `--ag-rs` 在 2.29.7 上不设它，由 NCCL 自己选，再由 TUNING 检查确认选的是上面那两个；
  2.32 起写成 `AllGather:<k>;ReduceScatter:<k>` 两个都固定。
- 第一次 AG 和 RS 串行执行（中间同步），先把两个 communicator 的 channel 连好：一个 communicator 在建连接时，
  另一个的 kernel 正在等对端，可能卡死。所以 `--ag-rs` 要求 `--warmup >= 1`；`--warmup N` 是 1 次串行加
  N−1 次 AG + RS 并发。
- 只有 **range / app-range** replay 能看到重叠：kernel / application replay 一次只 profile 一个 kernel，并把它串行化。
  range 类模式的 metric 属于整个 NVTX range（AG + RS 合在一起），分不出 AG 和 RS 各自的部分。
- `baseline.log` 里有每次迭代 AG 和 RS 的起止时间和重叠时长（CUDA event，以最早的 start 为 0），用来确认确实重叠了。

## AG + RS：replay 模式 × communicator

2026-11-07 实测 shmem + app-range 失败（#14）：4 个 rank 都跑完了 warmup，一进 NVTX range，ncu 就报
`Failed to profile "range" in client process <rank 0>`，第 1 遍都没开始，退出码 9。2026.4 的文档也没说 shmem
支持 app-range（见上文）。`sweep_ncu_replay_modes.sh` 把各个组合跑了一遍（#15，`--ag-rs --algo symm`，即 symmetric
multicast kernel `AllGather_STMC` + `ReduceScatter_LDMC`，1 GiB，SpeedOfLight）：

| 组合 | 做法 | 结果（#15） |
| --- | --- | --- |
| **`app-range:tcp`** | 每个 rank 一个 ncu，`--lockstep-kernel-launch --lockstep-nvtx-include <range>/` | ✅ 8 个 pass。**唯一能 profile 并发 AG + RS 的组合**，sweep 默认第一个跑 |
| **`app-range:tcp:nolockstep`** | 同上，不加 `--lockstep-*` | ✅ 8 个 pass：app-range 下 lockstep 不是必需的 |
| `application:shmem` | 串行化的对照组 | ✅ 9 个 pass；但 AG、RS 被串行化，看不到重叠 |
| `range:shmem` / `range:tcp` | 在进程内录制、重放 range | ❌ 保存显存失败（symm window 的 multicast 映射，同 #1/#3/#4） |
| `app-range:shmem` | — | ❌ 同 #14 |
| `app-range:none` | 不用 communicator | ❌ 卡死 900 s：同一进程树里 ncu 串行化各进程 |

`--communicator tcp` 不经过 torchrun：torchrun 的 agent 进程持有 TCPStore，application / app-range 每遍只重启
worker，旧 store 里上一轮留下的 key（例如 NCCL unique id）可能被新一轮读到（推测，未验证）。启动器改为直接起
`ncu_tcp_ranks.sh`，每个 rank 设 `RANK` / `LOCAL_RANK` / `WORLD_SIZE` / `LOCAL_WORLD_SIZE`，`MASTER_ADDR=127.0.0.1`、
`MASTER_PORT` 和 `--communicator-tcp-port` 各取一个空闲端口。一个 rank 的 ncu 先退出时，其他 rank 会等到 `--timeout`。
