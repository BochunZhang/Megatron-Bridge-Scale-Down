# Qwen3.5 FSDP3 通信 overlap 实验

目标：观察训练中 **参数 all-gather（AG）和梯度 reduce-scatter（RS）** 能被多少计算覆盖。
这里的 FSDP3 指 Megatron FSDP 的 `optim_grads_params`，即参数、梯度、优化器状态均分片。
使用与 `../01-fine-grained-offload` 相同的 Bridge pretrain / GPT forward step，随机初始化、mock text data。

## 模型与矩阵

从下列 HF 模型的 `text_config` 构建，保留原模型宽度、GDN、attention、词表和路由 top-k；不加载权重或视觉编码器。

| 参数 | 397b | 122b | 27b |
|---|---:|---:|---:|
| HF 模型 | [397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/blob/main/config.json) | [122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B/blob/main/config.json) | [27B](https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/config.json) |
| hidden size | 4096 | 3072 | 5120 |
| attention heads / KV heads | 32 / 2 | 32 / 2 | 24 / 4 |
| GDN key / value heads | 16 / 64 | 16 / 64 | 16 / 48 |
| FFN intermediate size（MoE 为单个 expert） | 1024 | 1024 | 17408 |
| 裁剪后 layers | 8 | 8 | 8 |
| 裁剪后 routed experts / top-k | 64 / 10 | 64 / 8 | dense |

每四层为 `GDN, GDN, GDN, full attention`，共两个周期。MTP 关闭，shared expert 保留。
每个模型执行以下 8 组 CP / sequence length 配置，每组遍历 **MBS=1、2、4**，共 **72 次训练**。
1k = 1024 tokens，sequence length 是 CP 切分前的全局长度。

| CP | sequence length | 每个 CP rank 的 tokens |
|---|---|---|
| 1 | 4096, 8192, 32768, 65536 | 4096, 8192, 32768, 65536 |
| 4 | 4096, 32768, 131072, 262144 | 1024, 8192, 32768, 65536 |

固定 BF16 参数/计算、FP32 主参数/梯度/RS，TP=PP=EP=ETP=1。默认 4 GPUs，固定 **GBS=32**。
每步的 microbatch 数自动计算为 `num_microbatches = 32 / (MBS × (world_size / CP))`，必须为正整数。
默认四卡时，CP=1 的 MBS=1、2、4 分别对应 8、4、2 个 microbatch；CP=4 时分别对应 32、16、8 个。
因此跨 MBS 和 CP 配置保持相同 GBS，梯度累积次数随配置变化。
Megatron FSDP 参数在 **DP×CP** 组分片，EP=1 时 expert 的分片组也覆盖全部 GPUs；CP=4、world=4 仍有四卡 AG/RS。

关闭 activation recompute、CPU offload、CUDA graphs，避免它们改变计算量或掩盖逐层时间线。
保留 learned routing，不启用 force load balancing。标准 attention 使用 CP P2P，GDN 仍有其自身的 CP 通信。
长序列若 OOM，实验会直接失败并保留日志；不会静默降低长度或启用重计算。

当前 Megatron-LM 在 FSDP3 初始化时会强制将 `overlap_param_gather`、`overlap_grad_reduce` 设为 True。
因此本实验不提供无效的 `false` 对照。可用 `--communication-unit-size N` 调整预取/RS 队列容量（单位为参数元素数），
但这同时影响 AG 和 RS，不能作为单独关闭其中一个 overlap 的实验。默认使用 MCore 自身的容量计算。

## 运行

在已有可运行 Qwen3.5 的 Linux/CUDA Bridge 环境执行，需可用的 Transformer Engine、GDN 依赖和 `nsys`。
仅获取 HF 配置，使用同词表大小的 NullTokenizer，无需下载模型权重。
以下命令从 Bridge 仓库根目录运行；Bash 启动脚本自身也会定位根目录，并自动设置结果目录和运行时间戳。

```bash
# 无 GPU、无 HF 网络访问的矩阵检查
bash examples/scale-down/01-analyse/04-fsdp3-comm/benchmark_fsdp3.sh --dry-run

# 全部 72 组，默认每个 rank 均采集 nsys
bash examples/scale-down/01-analyse/04-fsdp3-comm/benchmark_fsdp3.sh

# 单个配置；world size 必须是 CP 的整数倍，GBS 必须是 MBS × DP 的整数倍
bash examples/scale-down/01-analyse/04-fsdp3-comm/benchmark_fsdp3.sh \
  --model 397b --cp 4 --seq-length 131072 --micro-batch-size 2 --nproc-per-node 8

# 相同配置关闭 profiler 测量干扰
bash examples/scale-down/01-analyse/04-fsdp3-comm/benchmark_fsdp3.sh \
  --model 397b --cp 4 --seq-length 131072 --micro-batch-size 2 --nproc-per-node 8 --profile none
```

默认总计 10 步，先执行 5 步 warmup，再采集 3 步（Bridge 内部 step 5、6、7，对应日志 iteration 6–8）。
可以调整 `--profile-start / --profile-end / --train-iters`。每个 torch distributed worker 单独运行 nsys，
用 cudaProfilerApi 限定采集范围；一个 rank 停止采集不会提前结束其他 rank 的报告。
默认 `CUDA_DEVICE_MAX_CONNECTIONS=32`，允许环境变量覆盖，实际值记录在 manifest 中。

多节点：在每个节点上分别运行同一命令，设置共同的 `NNODES`、`MASTER_ADDR`、`MASTER_PORT` 和各自的 `NODE_RANK`；
或使用等价的 CLI 参数。`--nproc-per-node` 指每节点 GPUs。
多节点启动时还需为所有节点设置相同的 `RUN_DATE`（格式为 `yymmdd-hhmmss`），将同一配置的各节点结果归入同一目录；
单节点运行无需设置，Bash 脚本会自动生成。启动器不申请集群资源，所有节点必须选择相同矩阵。
一个 case 失败后应停止其他节点的作业。
没有设置通信网卡、账号、集群路径等环境特定参数。

结果自动保存在仓库下的 `results/01-analyse/04-fsdp3-comm/`，无需手动指定存储位置。
每个配置使用 `{item}_{value}-{item}_{value}` 形式命名，最后追加 `date_{yymmdd-hhmmss}`，例如：

```text
results/01-analyse/04-fsdp3-comm/model_397b-fsdp_3-mbs_1-seq_4096-cp_1-gbs_32-gpus_4-profile_nsys-date_261009-123456/node0/
```

目录名中的 `gpus` 是全局 GPU 数。每次 Bash 启动使用一个时间戳，各配置的 `node<rank>/` 目录包含：

- `launch.json`、`config.yaml`、`hf-text-config-rank*.json`：命令、运行配置与实际裁剪后的 HF 配置。
- `train.log`：训练日志，运行时可另开终端 `tail -f`。
- `profile-rank*.nsys-rep`、`.sqlite`、`.json`：每个 rank 的原始报告、SQL 数据库和统计。

已有 case 目录会被拒绝，防止混入旧结果。训练或导出失败会报错，重新启动会自动使用新的时间戳。

## SQL 分析与解释

启动器自动 export 和分析，也可单独使用：

```bash
nsys export --type=sqlite --output=profile.sqlite profile.nsys-rep
uv run --no-sync python examples/scale-down/01-analyse/04-fsdp3-comm/analyse_nsys.py \
  profile.sqlite --output stats.json

# 对其他 profile，显式使用选定的时间窗口（nsys 时间戳，单位 ns）
uv run --no-sync python examples/scale-down/01-analyse/04-fsdp3-comm/analyse_nsys.py \
  profile.sqlite --start-ns 1000000000 --end-ns 2000000000 --output selected.json
```

[analyse_nsys.py](analyse_nsys.py) 只需 Python 标准库，数据库以只读方式打开，规范化数据放在 TEMP 表。
统计主体是 [gpu_timeline.sql](gpu_timeline.sql)：将所有 stream 的 kernel/memcpy/memset 区间截取到窗口，
用端点扫描和 SQL window function 计算并集与交集，**不直接累加重叠 kernel 的持续时间**。
参考 [NVIDIA Nsight Systems SQLite schema](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)。

默认窗口是每个进程完整 `megatron.bridge.training.train.train_step` NVTX ranges 的连续包络，
包括步间空隙，并通过 CUDA runtime 的进程 ID、correlation ID 和线程 ID 关联延伸至最后一个 GPU 工作结束。
不包含模型初始化。缺少匹配 NVTX 时会报错，允许显式 `--window activity` 使用首个到最后一个 GPU 活动的包络；
该退化口径**遗漏两端空闲**，结果中会标注。所有 rank 单独统计，不能直接相加比例。

| JSON 指标 | 含义 |
|---|---|
| `idle_pct` | `100 × 无任何已采集 GPU kernel/copy/memset 的时间 / 窗口时间` |
| `compute_idle_pct` | `100 × 无非 NCCL 计算 kernel 的时间 / 窗口时间`；通信或 copy 独占时也计入 |
| `ag_overlap_pct` / `rs_overlap_pct` | AG / RS kernel 时间并集中，同时有非 NCCL 计算 kernel 的比例 |
| `ag_exposed_ns` / `rs_exposed_ns` | AG / RS 存在、计算不存在的时间 |
| `ag_rs_exposed_ns` / `ag_rs_exposed_pct` | AG 或 RS 存在、计算不存在的时间并集 / 占窗口的比例 |
| `other_comm_ns` | 其他 NCCL kernel 的时间并集，包括无法按名称识别的通信 |

没有识别到 AG/RS 时对应 overlap 百分比为 `null`，不表示“全部隐藏”。所有识别到的 NCCL kernel 名称列在
`communication_kernel_names` 中供检查。分类依赖 kernel 名中的 NCCL + AllGather/ReduceScatter，
generic/SendRecv kernel 无法仅凭名字判断 collective 或张量归属；CP 等其他通信也可能混入同类 collective。
需要结合 nsys 的 FSDP ranges、通信 stream、layer ranges 确认归因。
`layer 1` … `layer 8` 使用 MCore 的 1-based 全局层号，涵盖 GDN 和 full-attention 层。
这些 NVTX ranges 是 CPU forward 调用范围，FSDP pre-forward hook 的 AG 可能在其之前；
应结合 CUDA stream 和 GPU kernel 看等待/预取，不能把 CPU range 的时长直接当作 GPU 计算时长。
缺少 memcpy/memset 表时按没有记录到此类事件处理；若采集时关闭了对应 tracing，空闲率只适用于已记录的活动。

判读时同时查看 `ag_rs_exposed_pct`、overlap 比例和未开启 profiler 时的 iteration time。
空闲率低可能只是 NCCL 在运行；高时间重叠也可能有 SM/带宽争用，**不等同于通信对 step time 的影响为零**。
这里是 trace 可见的进程/GPU 活动空闲时间，不是 SM occupancy、硬件利用率或其他未采集进程的全局 GPU 空闲率。
八层模型还包含 embedding、loss、optimizer 及首尾通信，结论只针对这个 scale-down 工作负载。

## 本地验证

```bash
uv run --no-sync python -m pytest --confcutdir=tests/unit_tests/scripts/scale_down \
  tests/unit_tests/scripts/scale_down/test_fsdp3_comm.py -q
```

覆盖 72 组矩阵、固定 GBS 与梯度累积次数、非法配置、嵌套/并发区间、拷贝覆盖、AG/RS 交叠、
进程间重复 correlation ID、NVTX 字符串两种存储方式和异步 GPU 尾部。
这些是 CPU 验证；实际训练耗时、显存和 overlap 结论需要在 GPU 上运行后填写，当前未提供实测数值。
