# GB200 PyTorch AG/RS 融合对照

在同一 NVLink 域的 GB200 节点上，比较一个融合 kernel 与两个独立 NCCL symmetric kernel。两组实验使用同一份 NCCL 2.32.3 补丁库、BF16 数据、相同 rank 和相同消息尺寸。

- `fused` 从 PyTorch ProcessGroupNCCL 取得 communicator。Python 用 NCCL allocator 分配 AG 输出和 RS 输入，native shim 把两块缓冲注册为 symmetric window，再通过一个 Device API 调用提交两项任务。
- `split` 在两个 PyTorch NCCL communicator 上分别注册 AG 输出和 RS 输入，使用独立 CUDA stream 从同一个 CUDA event 放行 `AllGather_STMC` 和 `ReduceScatter_LDMC`。两个输出都可用后才放行下一对。

在仓库根目录运行。需要动态链接 NCCL 2.32.3 的 CUDA PyTorch，并提供 `ProcessGroupNCCL._comm_ptr()`、`mem_allocator` 和 `register_mem_pool`。运行时会核对 PyTorch 只加载了一份指定的补丁库。

```bash
bash install/install-nccl-2.32.3-fused-ag-rs.sh
bash examples/gb200/nvls-fused-ag-rs/run_nsys_fused_ag_rs.sh \
  --gpus 0,1,2,3 --ag-mib 256 --rs-mib 256 --ctas 4
```

补丁在 [install/patches](../../../install/patches/nccl-2.32.3-fused-ag-rs.patch)。`--ag-mib` 是每卡的完整 AG 输出，`--rs-mib` 是每卡的完整 RS 输入。两项均可独立指定，均须能按 GPU 数分成 16B 对齐的分片。独立版 CTA 数由 NCCL 选取，融合版由 `--ctas` 指定。可先加 `--dry-run` 核对命令。

脚本默认使用仓库根目录下的 `NsightSystems-2026.6.1/bin/nsys`，也可通过 `NSYS_BIN` 或 `--nsys` 指定同版本的其他安装位置。每个模式先做无 profiler 的 CUDA event 计时，再单独采集 CUDA 和 NVTX。两轮均验证每个 rank 的两项输出。结果写到 `results/gb200/nvls-fused-ag-rs/<timestamp>/`。

| 文件 | 内容 |
| --- | --- |
| `fused/timing.json`、`split/timing.json` | 无 profiler，每次从共同 GPU event 到两项输出都可用的时间，汇总取每次最慢 rank，再取中位数。 |
| 各模式的 `profile.nsys-rep`、`profile.sqlite` | 每卡 CUDA/NVTX 时间线。 |
| `summary.json` | 两种模式的 kernel 跨度，独立版的 AG/RS 各自耗时和同时执行时间，以及实际 grid、block 和寄存器数。分析器检查每张 GPU 的 kernel 类型与次数。 |

评估总耗时优先用 `timing.json`。GPU 时间线可用来解释是哪一项结束得晚、独立版是否真正并发，以及融合版实际发了多少 CTA。实验没有采样 SM 驻留或 NVLink/HBM 计数器。
