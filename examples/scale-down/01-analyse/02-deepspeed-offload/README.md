# DeepSpeed Offload + Recompute 组合测试

## 1. 测试目标

在 ZeRO-3 下测量 **parameter 位置 × optimizer 位置 × recompute 策略 × micro-batch size** 的组合矩阵,回答:

1. 各显存预算下能达到的**最佳吞吐**,即显存—吞吐前沿与推荐配置;
2. `ratio=1.0` 时 SuperOffload 相比 ZeRO-Offload 的收益(SuperOffload 针对 NVLink-C2C,把 CPU 更新后的 optimizer state 直接搬到 GPU 再做精度转换);
3. SuperOffload `ratio` 的最优 CPU/GPU 配比;
4. `act`(HF 逐层重计算)与 `act+cpu`(再把 checkpoint 层激活异步卸载到 CPU)的显存收益与吞吐代价;
5. 上述结论在 dense 与 MoE 模型上是否一致。

背景见 [offload-combination.md](../../../../documents/deepspeed/offload-combination.md) 与 [activation-offload-and-recompute/02-config.md](../../../../documents/deepspeed/activation-offload-and-recompute/02-config.md)。

## 2. 测试模型与配置

| 项目 | Dense | MoE |
| --- | --- | --- |
| 模型 | `Qwen/Qwen3.5-9B-Base` | `Qwen/Qwen3.5-35B-A3B-Base` |
| 启动模式 | `--mode dense` | `--mode autoep`(`--autoep_size 4`,64 专家) |
| GPU 数 | 4 | 4 |

公共配置(由 `pretrain_experiment.sh` / `pretrain.sh` 统一注入):

- 训练:`wikitext` 加载 1%,`seq_len=4096`,`steps=10`,`seed=42`,按原生层数运行;
- batch:每 GPU 每 step 16 个样本,`grad_accum = 16 / mbs`,全局 batch 恒定;
- optimizer:AdamW(`weight_decay=0.01`),lr `1e-6`,WarmupLR 从 0 起 warmup 1000 步,梯度裁剪 `1.0`;`zero_3` 用 torch AdamW,offload 策略用 `DeepSpeedCPUAdam`;
- ZeRO-3:`overlap_comm=true`,`reduce_bucket_size = sub_group_size = 4e8`,卸载均 `pin_memory=true`;SuperOffload `cpuadam_cores_perc=0.90` 并自动加 `--bind_cores_to_rank`;
- 环境变量:`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`(缺失时 `train.py` 拒绝启动)、`DS_PIN_MEMORY_BACKEND=torch`(设为 `native` 会让 side-stream DMA stall);
- 依赖:**`transformers >= 5.16.0`**,旧版本会导致 offload 配置出现 NaN,见 §6.1。

## 3. 测试命令

### 3.1 NVMe 补丁

含 `param_nvme` 的测试,运行前必须先打补丁(SuperOffload 原因见 §6.2,MoE 预取原因见 §6.3)。补丁基于 DeepSpeed `c7cc64a90`:

```bash
# 应用
bash examples/scale-down/01-analyse/02-deepspeed-offload/apply_superoffload_nvme_patch.sh
# 恢复原 DeepSpeed 代码
bash examples/scale-down/01-analyse/02-deepspeed-offload/apply_superoffload_nvme_patch.sh --recovery
```

脚本按 `PYTHON_BIN`(默认 `python`)所在环境定位 DeepSpeed,可重复执行;补丁与源码不匹配时报错且不改动任何文件。

### 3.2 运行矩阵

`pretrain_experiment.sh` 串行执行全部组合:单模型 4×3×6×3 = 216 个 run,两个模型共 432 个。单个 run 失败或 OOM 只记入 summary,不中断扫描:

```bash
# 全矩阵
bash examples/scale-down/01-analyse/02-deepspeed-offload/pretrain_experiment.sh

# 子集:按需收窄各轴
bash examples/scale-down/01-analyse/02-deepspeed-offload/pretrain_experiment.sh \
  --models Qwen/Qwen3.5-9B-Base \
  --optimizer_strategies "zero_3 zero_offload" \
  --param_positions "param_gpu param_cpu" \
  --recompute_combos recompute_act \
  --micro_batch_sizes "1 4"
```

加 `--dry_run` 只生成 ds_config 不训练。run 命名为 `<策略>-<param位置>__<recompute>__mbs<N>`,如 `super_offload_0.9-param_cpu__recompute_act__mbs4`。

配置优先级为 CLI 参数 > 环境变量 > 默认值。矩阵各轴如下,其余配置项见[附录 A](#附录-a其余配置项):

| 项目 | CLI | 默认值 | 说明 |
| --- | --- | --- | --- |
| 模型 | `--models` | `Qwen/Qwen3.5-9B-Base Qwen/Qwen3.5-35B-A3B-Base` | 只接受这两个模型 |
| micro-batch | `--micro_batch_sizes` | `1 2 4 8` | 须整除 `per_gpu_batch_size` |
| recompute | `--recompute_combos` | `recompute_none recompute_act recompute_act_cpu` | `act_cpu` 自动包含 `act` |
| optimizer 策略 | `--optimizer_strategies` | `zero_3 zero_offload super_offload_1.0 super_offload_0.9 super_offload_0.75 super_offload_0.1` | 后缀即 ratio |
| param 位置 | `--param_positions` | `param_cpu param_gpu param_nvme` | 含 `param_nvme` 时做 NVMe 预检 |

### 3.3 多实例并行

脚本没有多机协调,并行方式是把矩阵切成互不重叠的子集,每组 GPU 跑一个实例。同机多实例须分别设置 `RESULTS_ROOT`、`TMP_CONFIG_DIR`、`NVME_PATH`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 RESULTS_ROOT=results/dense TMP_CONFIG_DIR=.tmp/dense NVME_PATH=/tmp/nvme_dense \
  bash examples/scale-down/01-analyse/02-deepspeed-offload/pretrain_experiment.sh --models Qwen/Qwen3.5-9B-Base &
CUDA_VISIBLE_DEVICES=4,5,6,7 RESULTS_ROOT=results/moe TMP_CONFIG_DIR=.tmp/moe NVME_PATH=/tmp/nvme_moe \
  bash examples/scale-down/01-analyse/02-deepspeed-offload/pretrain_experiment.sh --models Qwen/Qwen3.5-35B-A3B-Base &
```

### 3.4 Smoke test

用最省 GPU/CPU 内存的组合跑通全链路(NVMe swapper、CPUAdam、激活卸载):

```bash
bash examples/scale-down/01-analyse/02-deepspeed-offload/apply_superoffload_nvme_patch.sh

bash examples/scale-down/01-analyse/02-deepspeed-offload/pretrain_experiment.sh \
  --models Qwen/Qwen3.5-9B-Base \
  --optimizer_strategies super_offload_1.0 \
  --param_positions param_nvme \
  --recompute_combos recompute_act_cpu \
  --micro_batch_sizes 1 \
  --nvme_path /tmp/deepspeed_nvme_offload \
  --nvme_device /dev/nvme2n1
```

启动前会检查 `nvme_path` 可写、挂载自 `nvme_device`、`async_io` op 可加载,任一失败即退出。

### 3.5 `--test` 非有限值诊断

追加 `--test` 后,在 `metrics.csv` 同目录写出 NaN/Inf 报告:

- 训练前:`non_finite_init_weight.json`(FP32 master)、`non_finite_init_param.json`(模型参数),只记录不退出;
- 每轮更新后:`non_finite_{loss,gradient,main_weight,param}.json`,任一 rank 有异常则全部 rank 以退出码 3 退出;
- 无 rank 后缀的文件是汇总,`*.rank<N>.json` 是该 rank 的详情:`ranges` 为异常 flat index 的 `[start, end)` 区间,`parameter_layout` 用于映射回原参数(原参数下标 = `parameter_offset + i - subgroup_offset`;`i - subgroup_offset >= valid_numel` 时属于 padding);
- NVMe 上未驻留的参数分片记入 `skipped`,不会主动 swap-in。

诊断会额外同步和写文件,开启时的吞吐不能作为正式结果。

## 4. 配置维度

| 维度 | 取值 | 对应配置 | 说明 |
| --- | --- | --- | --- |
| param 位置 | `param_gpu` / `param_cpu` / `param_nvme` | `offload_param.device` | NVMe 使用 `aio`(queue_depth=8、block_size=1MiB)与 `max_in_cpu=0` |
| optimizer 位置 | `zero_3` / `zero_offload` / `super_offload_<ratio>` | `offload_optimizer.device`、`super_offload`、`ratio` | `ratio` 为在 CPU 更新的 subgroup 比例,其余由 GPU 上的 backup AdamW 更新 |
| recompute | `recompute_none` / `recompute_act` / `recompute_act_cpu` | `--activation_checkpointing` / `--cpu_checkpointing` | `act_cpu` 用 DeepSpeed `CheckpointHiddenStatesOffload` 把 checkpoint 层输入异步卸载到 pinned CPU 内存 |
| micro-batch | `1 / 2 / 4 / 8` | `train_micro_batch_size_per_gpu` | 全局 batch 不变 |

`offload_param`、`ratio < 1`、`super_offload` 都只支持 ZeRO-3;`ratio < 1` 要求 `DeepSpeedCPUAdam`;无效组合已由驱动脚本剔除。

## 5. 输出与指标

```text
<repo_root>/results/01-analyse/02-deepspeed/
├── experiment_summary_<ts>.txt     # 每个 run 的状态:OK / OOM / FAILED(rc) / DRY_RUN
└── <model>/<TEST_NAME>/<timestamp>/
    ├── run.log                     # 完整日志,含 wall_clock_breakdown 的 fwd/bwd/step 耗时
    ├── metrics.csv                 # 每 step 指标
    └── ds_config.json              # 实际使用的配置
```

| `metrics.csv` 字段 | 含义 |
| --- | --- |
| `loss` | 前两次更新 lr = 0,`zero_3` / `zero_offload` / `super_offload_1.0` 前 3 步 loss 应一致。`ratio < 1` 的 GPU subgroup 不受 WarmupLR 控制(lr 固定为配置值),第 2 步起可能不同 |
| `iter_time_sec` | 单 step 耗时 |
| `global_tokens_per_sec` | 全局吞吐,跨配置比较的主指标 |
| `gpu_peak_gigabytes` / `cpu_peak_gigabytes` | GPU / CPU 峰值内存 |
| `cuda_*_bytes` | PyTorch allocator 的 allocated / peak allocated / peak reserved |

## 6. 问题记录

| 问题 | 根因 | 解决方式 |
| --- | --- | --- |
| offload 配置出现 NaN loss | GatedDeltaNet 初始化时 `log(0) = -inf`,CPU Adam 把 `-inf` 变成 NaN | 升级 transformers 到 >= 5.16.0 |
| `param_nvme` + SuperOffload 报错 | SuperOffload 对 NVMe 参数卸载支持不完整 | §3.1 补丁 |
| MoE + `param_nvme` backward 报 `param N already assigned swap buffer id M` | NVMe 预取未去重,同一参数在一次 `swap_in` 中出现两次 | §3.1 补丁(§6.3) |

### 6.1 NaN loss:初始化 `-inf` 被 CPU Adam 变成 NaN

**根因一:初始化产生 `-inf`**。GatedDeltaNet 的 `A_log = log(A)`,`A ~ U(0, 16)`。transformers 5.16 之前下界为 0,采样到 0 时得到 `-inf`;5.16.0 起改为 `uniform_(0.01, 16)`。由于采样在 bf16 上进行(CPU bf16 `uniform_` 只有 256 个离散值),每个元素约有 1/256 的概率取到 0,9B 模型约 768 个 `A_log` 元素,期望出现约 3 个 `-inf`。

**根因二:CPU Adam 把 `-inf` 变成 NaN**。`-inf` 元素的梯度和 Adam 更新量都是 0,差别只在 weight decay 的写法:

| 实现 | weight decay | `param = -inf` 时 |
| --- | --- | --- |
| `torch.optim.AdamW`(`zero_3`) | `param *= 1 - lr * wd` | 仍为 `-inf` |
| `DeepSpeedCPUAdam`(offload、SuperOffload) | `param += (-lr * wd) * param` | lr = 0 时 `0 × inf`,lr > 0 时 `-inf + inf`,都是 NaN |

因此 **NaN 与学习率无关**:lr = 0 也会出现,降低 lr 或开启梯度裁剪都无效。

**为什么有时训练正常**:torch AdamW 会保留 `-inf`,而 forward 中 `exp(-inf) = 0` 仍为有限值,所以 `zero_3` 不报错,但该 head 梯度恒为 0,永远不会学习。此前部分 offload 配置能正常训练,可能是 `-inf` 所在的分片恰好在 GPU 上,由 torch AdamW 更新。

`--test` 报告中更新后 `param` 计数为 5 而非 1,是因为 `A_log` 属于 persistent 参数,会被 all-gather 到 4 个 rank,同一个值被重复计数,并不是新出现的异常。

**升级方法**(只改 transformers 与 tokenizers;`pyproject.toml` 的 override 会把 tokenizers 压回 0.22,因此需要 `--no-config`):

```bash
uv pip install --no-config --no-deps "transformers==5.17.0" "tokenizers==0.23.3"
# 确认修复
grep -n "uniform_(0.01, 16)" "$(python -c 'import transformers, os; print(os.path.dirname(transformers.__file__))')/models/qwen3_5/modeling_qwen3_5.py"
# 回滚
uv pip install --no-config --no-deps "transformers==5.12.1" "tokenizers==0.22.2"
```

升级后不要在 megatron-bridge 目录执行 `uv sync` 或不带 `--no-sync` 的 `uv run`,否则会降回旧版本。

### 6.2 SuperOffload 不支持 NVMe 参数卸载

参数位于 NVMe 时,大部分 subgroup 没有 CPU flat buffer(`fp16_partitioned_groups_flat[i]` 为 `None`,由 swapper 按需换入换出),SuperOffload 没有处理这种情况:

| 补丁 | 触发条件 | 报错 | 修复 |
| --- | --- | --- | --- |
| `superoffload_nvme_override.patch` | 所有 `super_offload_*` | 写回参数时 `'NoneType' object has no attribute 'data'` | 复用 ZeRO-3 父类的 swap-out 路径 |
| `superoffload_nvme_fp32_device_override.patch` | `ratio < 1` | GPU subgroup 的 FP32 master 建在 CPU 上,`deepspeed.initialize()` 时报 `.grad` 设备不一致 | FP32 master 跟随 `subgroup_to_device` |

注意:subgroup 按 `i >= int((1 - ratio) * n)` 划入 CPU。9B 模型每个 rank 只有 6~7 个 subgroup,`ratio=0.9` 时实际全部在 CPU,与 `1.0` 等价。

### 6.3 MoE + `param_nvme`:NVMe 预取重复分配 swap buffer

**现象**:MoE 模型在 backward 中报 `AssertionError: param 689 already assigned swap buffer id 14`,调用链为 `fetch_sub_module` → `__prefetch_nvme_param_partitions` → `swap_in` → `_allocate_and_return_buffers_for_swap_in`。dense 模型不触发。

**根因**:`__prefetch_nvme_param_partitions` 沿参数 trace 收集 `NOT_AVAILABLE` 的参数后一次性 `swap_in`,但没有去重(同文件的普通 all-gather 预取有 `# Avoid duplicates`)。trace 中同一参数会重复出现(如 recompute forward 紧接 backward、最后一层 forward→backward 拐点),两次出现落在同一窗口内时被收集两次,第二次分配 buffer 时触发断言。

**为什么只有 MoE**:预取窗口按元素数限制为 `2 × numel_in_flight`。dense 每个 Linear 只有一个量级相近的参数,窗口通常只覆盖 1~2 个条目;MoE 层中巨大的 3D expert 张量在 in-flight 时会把窗口撑大,同时 router / shared expert 等小参数和多参数模块(leaf MoE block、`GroupedExperts`)让窗口内条目变多,重复出现的参数就容易同时落入窗口。

**修复**:`patches/zero3_nvme_prefetch_dedup_override.patch` 在收集时按 `ds_id` 去重。参数一旦换入,在释放前对后续所有使用都有效,不改变语义。

## 附录 A:其余配置项

| 项目 | CLI / 环境变量 | 默认值 | 说明 |
| --- | --- | --- | --- |
| 每 GPU 每 step 样本数 | `--per_gpu_batch_size` | `16` | |
| 学习率 | `--learning_rate` / `LEARNING_RATE` | `1e-6` | |
| lr warmup 步数 | `--warmup_steps` / `WARMUP_STEPS` | `1000` | |
| 测量 warmup 步数 | `--measure_warmup_steps` / `MEASURE_WARMUP_STEPS` | `2` | 只影响吞吐/显存统计窗口 |
| GPU 数 | `--num_gpus` | `4` | |
| AutoEP 大小 | `--autoep_size` | `4` | 仅 MoE |
| MoE 专家数 | `--moe_num_experts` | `64` | 只能为 64 |
| 空跑 | `--dry_run` | 关闭 | |
| NVMe 目录 | `--nvme_path` / `NVME_PATH` | `/tmp/deepspeed_nvme_offload` | 每个 run 独立子目录,结束后删除 |
| NVMe 设备 | `--nvme_device` / `NVME_DEVICE` | `/dev/nvme2n1` | |
| NVMe 设备校验 | `--nvme_device_check` / `NVME_DEVICE_CHECK` | `true` | 容器内看不到 block device 时设 `false` |
| NVMe buffer 数量 | `--nvme_buffer_count` / `NVME_BUFFER_COUNT` | `16` | 不能低于 16,否则 buffer slot 不足 |
| NVMe buffer 容量(元素数) | `--nvme_buffer_size` / `NVME_BUFFER_SIZE` | `400000000` | 须容纳最大的单参数分片,更换 GPU 数或模型后需重新校准 |
| NVMe CPU 常驻上限 | `--nvme_max_in_cpu` / `NVME_MAX_IN_CPU` | `0` | |
| 保留 swap 数据 | `--keep_nvme_data` / `KEEP_NVME_DATA` | `false` | 跑全矩阵时保持默认,否则会耗尽磁盘 |
| 通信重叠 | `OVERLAP_COMM` | `true` | |
| 训练步数 | `STEPS` | `10` | |
| 序列长度 | `SEQ_LEN` | `4096` | |
| 数据集 / 加载比例(%) | `DATASET_NAME` / `DATASET_PERCENTAGE` | `wikitext` / `1.0` | 10% 时 tokenizer 预处理约 7 分钟,因此降到 1% |
| 随机种子 | `SEED` | `42` | |
| 日志间隔 | `LOG_INTERVAL` | `1` | |
| CPU 核绑定 | `BIND_CORES_TO_RANK` | `auto` | 含 `super_offload` 时自动开启 |
| pinned memory 后端 | `DS_PIN_MEMORY_BACKEND` | `torch` | |
| Python 解释器 | `PYTHON_BIN` | `python` | |
| 输出位置 | `REPO_ROOT` / `TMP_CONFIG_DIR` / `RESULTS_ROOT` | 仓库根 / `<repo_root>/.tmp` / `<repo_root>/results/01-analyse/02-deepspeed` | |
