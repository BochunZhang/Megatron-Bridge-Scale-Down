# GB200：用 Nsight Compute 2026 profile NCCL 2.32.3 的 multicast AllGather（AllGather_STMC）

日期：2026-10-31。在单节点 4 张 GPU 上，用 nccl-tests 的 `all_gather_perf` 跑**未修改的** NCCL 2.32.3，
强制 AllGather 使用对称内核的 multicast 实现 `AllGather_STMC`（寄存器 `multimem.st`），再用 ncu
（≥ 2026.1）采集这个 kernel 的硬件指标。

> 状态：脚本还没在 GB200 上实际跑过，第一次运行时请留意文末「待验证」中的几点。

## 为什么需要 ncu 的 communicator

ncu 默认会串行化 kernel，并逐个 kernel 重放（kernel replay）。NCCL kernel 需要所有 rank 同时在跑才能
往下推进，所以直接用 ncu 包住 NCCL 程序会卡死（nccl-tests issue #101）。ncu 2026.1 新增了多进程
communicator，它让同一个 kernel 在所有 rank 上同时 launch、同时重放：

| 模式 | 启动方式 | 适用场景 |
| --- | --- | --- |
| `shmem`（默认） | 一个 ncu 包住 `mpirun -np 4`，4 个 rank 都在它的进程树里 | 单节点 |
| `tcp` | `mpirun -np 4` 给每个 rank 各起一个 ncu，通过 `127.0.0.1` 汇合，并加 `--lockstep-kernel-launch` | 只能配 `--replay-mode kernel`（MPI rank 无法脱离 mpirun 单独重启）；多节点 |
| `none` | 一个 ncu 包住 `mpirun -np 4`，不启用 communicator | 对照实验 |

shmem 走的是主机 `/dev/shm` 上的 POSIX 共享内存，tcp 走 socket。两者都只传控制消息，不占 HBM、NVLink，
也不经过 NCCL。

因为 communicator 只协调**多个进程**，所以这里用 nccl-tests 的 MPI 版本，一张卡一个进程（`-g 1`）。
单进程驱动 4 卡（`-g 4`）的形态正是 issue #101 卡死的情况，这里不用。

## 前置条件

- 单个 GB200 节点，4 张 GPU 都在同一个 NVLink 域里，并且支持 multicast（NVLS）。
- CUDA toolkit ≥ 12.8（用于编译 sm_100）。
- MPI（Open MPI、HPC-X 等），`mpicc` / `mpirun` 在 `PATH` 中。nccl-tests 只能通过 MPI 实现多进程。
- Nsight Compute ≥ 2026.1。检查方法：

  ```bash
  ncu --version
  ncu --help | grep -- --communicator
  ```

- 有权限读 GPU 性能计数器：以 root 运行，或者驱动加载时设置 `NVreg_RestrictProfilingToAdminUsers=0`。
  没有权限时会报 `ERR_NVGPUCTRPERM`，脚本会提示。

## 运行步骤

在仓库根目录执行。

### 1. 编译未修改的 NCCL 2.32.3（已有则跳过）

```bash
./install/install-nccl-2.32.3.sh
```

产物在 `nccl/nccl-2.32.3/build`。注意不是 `nccl-2.32.3-cta` 或 `nccl-2.32.3-pause`。

### 2. 为它编译最新的 nccl-tests

```bash
./install/install-nccl-tests-2.32.3.sh
# 找不到 mpicc 时： --mpi-home /path/to/mpi
# 已经 clone 过、想更新到最新提交时： --update
```

这个脚本会做三件事：

- 校验 `nccl/nccl-2.32.3` 的 tag 是 `v2.32.3-1`，并且没有被修改过的文件；
- 把 nccl-tests clone 到 `nccl/nccl-tests`；
- 编译两份：`build/`（不带 MPI）和 `build-mpi/`（MPI=1，第 3 步用这一份），最后用 `ldd` 确认加载的是
  这份 `libnccl.so`。

### 3. 用 ncu profile AllGather_STMC

```bash
bash examples/gb200/nccl-tests/run_ncu_all_gather_stmc.sh
```

脚本依次跑两遍：

1. **baseline**：不挂 ncu，打开数据校验。确认 `AllGather_STMC` 能被选中、结果正确。
2. **ncu**：只 profile `^ncclSymkDevKernel_AllGather_STMC`。默认跳过 1 次 warmup，采集 1 次，使用 `--set full`
   和 `--replay-mode application`（见下文「为什么用 application replay」）。

每一遍都会检查 NCCL 的 TUNING 日志：每次 AllGather launch 都必须是
`AllGather [Symmetric]: <n> Bytes -> Kernel AllGather_STMC`，launch 次数也要对上，否则直接报错退出。
超过 `--timeout`（默认 1800 秒）还没结束，就按卡死处理并退出。

常用变体：

```bash
# shmem 识别不到 rank 或卡住时，改用 tcp（每个 rank 一个 ncu，只能用 kernel replay）
bash examples/gb200/nccl-tests/run_ncu_all_gather_stmc.sh --communicator tcp --replay-mode kernel

# 换消息大小或数据类型；--size 是 AllGather 的总输出大小，每个 rank 发送 size/4
bash examples/gb200/nccl-tests/run_ncu_all_gather_stmc.sh --size 1G --datatype float

# 只采几个 section，需要的 pass 少，application replay 重启的次数也少
bash examples/gb200/nccl-tests/run_ncu_all_gather_stmc.sh --set basic

# 用 -- 透传额外的 ncu 参数，例如不锁频
bash examples/gb200/nccl-tests/run_ncu_all_gather_stmc.sh -- --clock-control none

# 换 GPU、换 ncu 版本、已经跑过 baseline 时跳过
bash examples/gb200/nccl-tests/run_ncu_all_gather_stmc.sh --gpus 4,5,6,7 --ncu /opt/nvidia/nsight-compute/2026.1/ncu --skip-baseline
```

全部选项见 `--help`。

## 如何强制使用 AllGather_STMC

| 设置 | 作用 |
| --- | --- |
| `all_gather_perf -R 2` | 用 `ncclMemAlloc` 分配 send/recv buffer，并用 `NCCL_WIN_COLL_SYMMETRIC` 注册为 window；对称内核需要这个 |
| `NCCL_SYM_KERNEL=AllGather:AllGather_STMC` | 把 AllGather 的对称内核固定为 STMC（`src/tuning/cost_model.cc`） |
| `NCCL_SYM_CE_THRESHOLD=9223372036854775807` | **必须设置**。Blackwell 上 count 超过这个阈值的 AllGather 会自动改走 copy engine。这条分支只看按调用传入的 algSelection，不看 `NCCL_SYM_KERNEL`（`src/enqueue/enqueue.cc`） |
| `NCCL_NVLS_ENABLE=1`、`NCCL_WIN_ENABLE=1` | 打开 multicast 和 symmetric window |
| unset `NCCL_ALGO` / `NCCL_PROTO` / `NCCL_CTA_POLICY` / `NCCL_CE_COLL_AG_MULTICAST_THRESHOLD` | 避免外部环境变量干扰 kernel 选择 |

nccl-tests 对每个消息大小都会先跑 out-of-place，再跑 in-place，每一遍各有 `-w + -n + -c` 次 launch，
所以期望的 STMC launch 数是 `2 × (w + n + c)`。被 profile 的那次 launch 在 out-of-place 那一遍里。

## 为什么用 application replay

`--set full` 需要很多遍（pass）才能采完全部计数器。ncu 默认的 kernel replay 在每一遍之前都要保存、
之后恢复这个 kernel 能访问到的全部显存。在 4 卡上实测（shmem 模式），4 个 rank 都在 STMC kernel 处
齐步停下，`==PROF== Waiting for 4 clients` 之后报错：

```text
==WARNING== Backing up device memory in system memory. Kernel replay might be slow. ...
==ERROR== Failed to save memory for replay. If the issue persists, consider using --replay-mode application ...
==ERROR== Failed to wait for profiling
```

这说明 communicator 本身没有问题，失败的是保存显存这一步。STMC kernel 能访问的不只是本卡的 buffer，
还有 symmetric window 映射进来的其他 3 张卡的显存，以及 multicast 对象。ncu 的报错没有说明是哪一块
保存失败，细节在报错里给出的 `/tmp/nsight-compute-*.log` 中。

`--replay-mode application` 不保存显存，而是每一遍都把 `mpirun` 连同 4 个 rank 整体重新启动，
所以每次都是新的 NCCL 初始化，也就没有需要恢复的状态。代价是：

- 慢：`--set full` 有几十遍，每一遍都要重新启动并初始化 NCCL；用 `--set basic` 可以少很多遍。
- 要求每次运行的 kernel 序列相同。nccl-tests 每次都按同样的顺序 launch，满足这个要求。
- TUNING 日志每一遍都会多出一份，脚本按「每次运行 `2 × (w + n)` 次 STMC」乘以运行次数来检查。

如果想试 kernel replay，可以加 `--replay-mode kernel`。

**实测结果（2026-10-31，shmem + application）：没有 profile 到 kernel。**程序完整跑完，数据校验通过，
但 ncu 全程没有输出 `Profiling "..."` 或 `Application replay pass N`，最后报 `No kernels were profiled`。
out-of-place 那次 AllGather 耗时 7.6 s（in-place 只有 1.7 ms），说明 ncu 在这个 kernel 处停了很久后放弃了。
ncu 二进制里有 `Did not find candidates for profiling across all communicator clients` 这条提示，推测是
shmem communicator 在 application replay 下没能让 4 个 rank 对齐到同一个候选 kernel，尚未证实。

所以目前两种 replay 都还没跑通，对照实验：

| 命令 | 要验证什么 |
| --- | --- |
| `--replay-mode kernel -- --metrics gpu__time_duration.sum` | 只要 1 遍、不需要 replay 时，ncu 是否还会去保存显存 |
| `--replay-mode kernel --set basic` | 少量 pass 时保存显存是否成功 |
| `--communicator none --replay-mode application --set basic` | 不用 communicator 时，application replay 能否拦到 kernel（可能卡住，由 `--timeout` 兜底） |

## 输出

默认目录是 `results/gb200/nccl-tests/ncu-all-gather-stmc/<时间>/`，也可以用 `--output-dir` 指定：

| 文件 | 内容 |
| --- | --- |
| `config.txt` | NCCL / nccl-tests / ncu 的版本与路径、参数、所有 `NCCL_*` 环境变量 |
| `baseline.log`、`ncu.log` | 两遍的 stdout/stderr，其中有 nccl-tests 的带宽表和 ncu 的 `==PROF==` 输出 |
| `nccl-logs/<pass>/nccl-<host>-<pid>.log` | NCCL INFO 日志（INIT、ENV、TUNING） |
| `ag_stmc.ncu-rep`（shmem）或 `ag_stmc.rank<R>.ncu-rep`（tcp） | ncu 报告，可以用 Nsight Compute GUI 打开 |
| `summary.csv` | 每个被 profile 的 kernel 一行：进程、设备、grid/block、`gpu__time_duration`、DRAM 利用率 |

在命令行查看报告：

```bash
ncu --import <dir>/ag_stmc.ncu-rep --page details
```

## 读结果时注意

- ncu 默认会锁频（`--clock-control base`）、在重放前刷 cache，还会保存和恢复显存，并让 kernel
  齐步 launch。因此报告里的 kernel 耗时**不等于**真实性能。带宽要看 baseline 中 nccl-tests 的 `busbw`
  或者 nsys 的结果；ncu 的数据用来看 kernel 内部的情况，例如 SM / L2 / NVLink 吞吐和 warp stall 原因。
- `--set full` 一个 kernel 要重放几十遍，比较慢。只看 Speed of Light 的话用 `--set basic`。

## 待验证（第一次运行时确认）

1. **shmem 能否识别 mpirun 的 rank**：mpirun 拉起的 rank 中间隔着 prterun，ncu 能否把它们当作同一棵
   进程树里的 peer，需要实测。如果不行（卡住，或者报 peer 数不足），就加 `--communicator tcp`。
2. **`--launch-count` 的计数方式**：在 communicator 模式下，它是按每个 rank 计数，还是 4 个 rank 合起来
   计数，还不确定。脚本期望报告里有 `4 × launch-count` 条结果，对不上时只打印 WARNING，需要对照
   `ncu.log` 判断。
3. **multicast kernel 的重放**：重放时 ncu 会保存和恢复显存，多播（multicast）对象上的数据能否被正确恢复，
   官方文档没有说明。如果 ncu 这一遍卡住，或者报错，而 baseline 正常，问题多半出在这里。
