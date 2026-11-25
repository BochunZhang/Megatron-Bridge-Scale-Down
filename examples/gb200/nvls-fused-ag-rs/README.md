# GB200 PyTorch AG/RS 融合对照

在同一 NVLink 域的 GB200 节点上，比较四种单 kernel 融合方案和两个独立 NCCL symmetric kernel 的基线。各组使用同一份 NCCL 2.32.3 补丁库、BF16 数据、相同 rank 和相同消息尺寸。

- `fused` 从 PyTorch ProcessGroupNCCL 取得 communicator。Python 用 NCCL allocator 分配 AG 输出和 RS 输入，native shim 把两块缓冲注册为 symmetric window，再通过一个 Device API 调用提交两项任务。
- `split` 在两个 PyTorch NCCL communicator 上分别注册 AG 输出和 RS 输入，使用独立 CUDA stream 从同一个 CUDA event 放行 `AllGather_STMC` 和 `ReduceScatter_LDMC`。两个输出都可用后才放行下一对。

| 融合变体 | 每 lane 的主循环 |
| --- | --- |
| `ag-first` | 先发 8 个 AG 本地读和 8 个 multicast 写，再发 8 个 RS 归约读和 8 个本地写。 |
| `rs-first` | 先发 8 个 RS 归约读和 8 个本地写，再发 8 个 AG 本地读和 8 个 multicast 写。 |
| `warp-split` | 每个 256 线程 CTA 的 4 个 warp 做 AG，另 4 个做 RS。各自沿用独立内核的 8-pack tile 和按 rank 错开起始地址的遍历。 |
| `warp-split-pause` | 同样按 warp 平分。AG warp 每发完一个 4 KiB multicast tile 暂停指定的 SM cycles，再读下一批。 |

在仓库根目录运行。需要动态链接 NCCL 2.32.3 的 CUDA PyTorch，并提供 `ProcessGroupNCCL._comm_ptr()`、`mem_allocator` 和 `register_mem_pool`。运行时会核对 PyTorch 只加载了一份指定的补丁库。

```bash
bash install/install-nccl-2.32.3-fused-ag-rs.sh
bash examples/gb200/nvls-fused-ag-rs/run_nsys_fused_ag_rs.sh \
  --gpus 0,1,2,3 --ag-mib 1024 --rs-mib 1024 --ctas 4,8 --ag-pause-cycles 3072
```

补丁在 [install/patches](../../../install/patches/nccl-2.32.3-fused-ag-rs.patch)。若旧版补丁已应用到原有的 NCCL 源码目录，给安装脚本传 `--dest` 指向一个新 checkout，再给测试脚本传 `--nccl-build <dest>/build`。

`--ag-mib` 是每卡的完整 AG 输出，`--rs-mib` 是每卡的完整 RS 输入。两项均可独立指定，均须能按 GPU 数分成 16 B 对齐的分片。默认顺序是先跑 4 CTA 的四种融合内核，再跑 8 CTA 的四种，最后跑一份 CTA 数由 NCCL 选取的独立 AG/RS 基线。`--ctas` 和 `--variants` 支持逗号分隔的列表。pause 变体默认暂停 3072 cycles，设为 0 可测同一变体的零延迟对照。可用 `--dry-run` 核对顺序和参数。

脚本默认使用仓库根目录下的 `NsightSystems-2026.6.1/bin/nsys`，也可通过 `NSYS_BIN` 或 `--nsys` 指定同版本的其他安装位置。每个模式先做无 profiler 的 CUDA event 计时，再单独采集 CUDA 和 NVTX。两轮均验证每个 rank 的两项输出。结果写到 `results/gb200/nvls-fused-ag-rs/<timestamp>/`。

Gloo 网卡由仓库的 [单节点选择脚本](../../../scripts/scale-down/common/gloo_socket_ifname.sh) 解析。已有 `GLOO_SOCKET_IFNAME` 且网卡已启用并有 IPv4 地址时沿用该值，否则优先选 `eth0`，其次选 `bond1`。`config.txt` 记录生效网卡和采样设置。

Nsight 对 `--gpus` 中的每张卡采集 NVLink RX/TX，区分请求、响应、用户数据和协议数据。指标集提供 Bytes、GB/s 和 Nsight 原生 Throughput %。默认 `--nsys-freq 50000`，采样周期约 20 µs。汇总用 Bytes 计算每张物理 GPU 的吞吐，窗口为每对操作从首个 kernel 开始到最后一个结束。采样区间和 kernel 边界相交时按时间比例分摊。汇总的 `percent_of_nominal` 用 RX、TX 各自的总吞吐除以 GB200 单向标称 900 GB/s。短于几个采样周期的窗口只适合看量级。

| 文件 | 内容 |
| --- | --- |
| `cta-<N>/<variant>/timing.json`、`split/timing.json` | 无 profiler，每次从共同 GPU event 到两项输出都可用的时间，汇总取每次最慢 rank，再取中位数。 |
| 各模式的 `profile.nsys-rep`、`profile.sqlite` | 每卡 CUDA/NVTX 时间线和原始 NVLink GPU Metrics。 |
| `cta-<N>/<variant>/summary.json` | 校验实际融合 kernel 名称和 CTA 数，再汇总与独立版的 kernel 跨度、AG/RS 重叠、寄存器数和逐 GPU NVLink RX/TX。 |

评估总耗时优先用 `timing.json`。GPU 时间线可用来解释是哪一项结束得晚、独立版是否真正并发，以及融合期间各卡的 NVLink 流量。
