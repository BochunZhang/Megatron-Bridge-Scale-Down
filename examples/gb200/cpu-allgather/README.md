# GB200 Copy Engine multicast AllGather

每个 rank 提供一个 shard。NCCL 将 shard 写到接收缓冲区对应槽位的 multicast 地址，每张 GPU 最终持有按 rank 排列的完整输出。

| Case | 源 → 目标 | 每轮操作 |
| --- | --- | --- |
| `case1` | GPU HBM → GPU HBM | 从本地 GPU shard 发起 multicast AllGather |
| `case2` | Pinned CPU → GPU HBM → GPU HBM | 将 shard 拷入本地 GPU 的 staging buffer，再从该 buffer 发起 multicast AllGather |
| `case3` | Pinned CPU → GPU HBM | 将映射后的 host 指针直接传给 NCCL，从 CPU shard 发起 multicast AllGather |

在 Linux GB200 节点上运行。使用同一台主机、同一 NVLink multicast clique 中至少三张 GPU。激活带 CUDA、Gloo 和 NCCL 支持的 PyTorch 环境。PyTorch 需提供 `ProcessGroupNCCL` 的 `_comm_ptr()` 和 symmetric memory pool 注册接口。默认使用 PyTorch 动态加载的 NCCL，版本至少为 2.32.2。CUDA/NCCL 组合需接受 multicast 目标拷贝，`case3` 还需接受 pinned host 源。默认通过 `nsys` 采集，可用 `NSYS_BIN` 指定可执行文件路径。

```bash
bash examples/gb200/cpu-allgather/run_cpu_allgather.sh
```

默认使用 GPU `0,1,2,3`，每个 case 预热 3 轮、计时并采集 20 轮。`--shard-mib` 是每张卡的输入大小，默认每张卡输入 256 MiB，四卡时接收 1 GiB。`--cases case2,case3` 可只运行指定路径。三个 case 分别启动，单个 case 失败后仍会尝试其余 case，最终返回非零状态。

启动前执行 `nvidia-smi -i <GPU 列表> --lock-gpu-clocks=tdp,tdp`，需要设置 GPU 时钟的权限。锁频结果保存在输出根目录的 `gpu_clock_lock.log`。锁频失败会终止运行。`--dry-run` 可预览锁频和采集命令。

结果默认保存在 `results/gb200/cpu-allgather/<timestamp>/<case>/`，可通过 `--output-dir` 指定输出根目录。每个 case 的 `results.json` 保存所有 rank 的逐轮 CUDA event 时间和最终接收缓冲区的逐 shard 校验结果。`case2` 分别记录 H2D、AllGather 和从 H2D 开始到 AllGather 完成的总时间。AllGather 时间包含 NCCL 的流内同步。每项汇总都先取该轮各 rank 的最大值，再跨轮取中位数。`output_gb_s` 用每张 GPU 完整接收字节数除以总时间中位数。对比性能时用 `--no-profile` 获取只做 CUDA event 计时的结果，TDP 锁频仍会执行。

## 核对硬件通道

默认每个 case 生成一份 Nsight Systems 报告 `<case>.nsys-rep`。可用 `--iters 3` 缩短采集。

```bash
bash examples/gb200/cpu-allgather/run_cpu_allgather.sh \
  --gpus 0,1,2,3 --shard-mib 256 --iters 3
```

在每个 case 的 `.nsys-rep` 中，从 `cpu-allgather/<case>/rank-<rank>` NVTX 范围定位各 rank 的 `multicast_allgather` 和 `h2d_stage` 提交。沿 CUDA API 和 GPU 活动的关联查看 shard 大小的 payload copy。`case1` 和 `case3` 每 rank、每轮各提交一次 multicast payload copy。`case2` 多一次同样大小的 H2D staging copy。NCCL 的同步写只有 4 B。P2P 拷贝若有 source 和 destination 两条轨道，只数一次。对每个 rank 核对 payload 的 `Channel type` 为 `Async Memcpy`，`Compute` 表示该拷贝跑在计算通道上。

启动器要求 NCCL 的 tuning 日志确认 `Copy Engine` 与 `CE synchronization with NVLS`，并核对 multicast 阈值。日志证明 NCCL 的选路，Profiler 的硬件通道证明实际执行位置。若 R580 上 rank 0 的 payload 走 `Compute`，可用仓库已有的 [NCCL 2.32.3 self-import 构建脚本](../../../install/install-nccl-2.32.3-ce-multicast-handler.sh)，再重新核对硬件通道。

```bash
NCCL_CE_MC_SELF_IMPORT=1 bash examples/gb200/cpu-allgather/run_cpu_allgather.sh \
  --gpus 0,1,2,3 --iters 3 \
  --nccl-lib "$PWD/nccl/nccl-2.32.3-ce-mc-self-import/build/lib/libnccl.so.2"
```
