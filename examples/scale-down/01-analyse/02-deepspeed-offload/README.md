# DeepSpeed Offload + Recompute 组合测试

## 1. 测试目标

本实验在 ZeRO-3 分片策略下,系统测量 **parameter 存放位置 × optimizer 存放/计算位置 × recompute 策略 × micro-batch size** 的组合矩阵,回答以下问题:

1. 在每一档显存占用下,能达到的**最佳吞吐量**是多少,即显存—吞吐前沿曲线与各显存预算下的推荐配置;
2. **SuperOffload vs ZeRO-Offload**:在 `ratio=1.0`(全部 optimizer 子组在 CPU 更新)时,SuperOffload 针对 NVLink-C2C 的优化(CPU 更新的 optimizer state 直接搬到 GPU 再做精度转换,而不是在 CPU 上转换后再传输)能否带来可观的带宽/吞吐收益;
3. SuperOffload 的 `ratio` 扫描曲线:CPU/GPU 异构 optimizer 计算的最优配比,以及"每 GB 显存换到的吞吐"拐点;
4. `act`(HF 逐层重计算)与 `act+cpu`(checkpoint 层激活经 DeepSpeed `CheckpointHiddenStatesOffload` 异步卸载到 CPU)各自的显存收益与吞吐代价;
5. 以上结论在 MoE 模型与 dense 模型上是否一致。

相关背景分析见 [documents/deepspeed/offload-combination.md](../../../../documents/deepspeed/offload-combination.md)(ZeRO-Offload / ZeRO-Infinity / SuperOffload 的关系与组合约束)与 [documents/deepspeed/activation-offload-and-recompute/02-config.md](../../../../documents/deepspeed/activation-offload-and-recompute/02-config.md)(激活卸载与重计算的实现方案)。

## 2. 测试模型与配置

| 项目 | Dense | MoE |
| --- | --- | --- |
| 模型 | `Qwen/Qwen3.5-9B-Base` | `Qwen/Qwen3.5-35B-A3B-Base` |
| 启动模式 | `--mode dense` | `--mode autoep`(`--autoep_size 4`,专家数固定为 64) |
| GPU 数 | 4 | 4 |

公共控制变量(两个模型一致,全部 run 固定,均由 `pretrain_experiment.sh` / `pretrain.sh` 统一注入):

- 数据与训练:数据集 `wikitext`,加载比例 1.0%(见 §6.2),`seq_len=4096`,`steps=10`,`warmup_steps=2`,`seed=42`,只统计 warmup 之后的稳定步;
- batch:每 GPU 每 optimizer step 处理 16 个样本,`grad_accum = 16 / micro_batch_size`,保证 mbs 扫描时全局 batch 恒定;
- optimizer:AdamW(`betas=(0.9, 0.999)`、`eps=1e-8`、`weight_decay=0.01`),学习率默认 `1e-6`(见 §6.1),全局梯度裁剪阈值为 `1.0`;`zero_3` 策略配置 `torch_adam=true`,offload 策略使用 `DeepSpeedCPUAdam`;
- ZeRO-3:`overlap_comm=true`(可用 `OVERLAP_COMM` 环境变量覆盖),`reduce_bucket_size = sub_group_size = 4e8`,所有 CPU/NVMe 卸载均 `pin_memory=true`;
- SuperOffload:统一 `cpuadam_cores_perc=0.90`,`pretrain.sh` 检测到 ds_config 含 `super_offload` 时自动为 deepspeed launcher 追加 `--bind_cores_to_rank`;
- 环境变量:`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`(act+cpu 的 offload/restore 循环易产生碎片,`train.py` 缺失该项会拒绝启动),`DS_PIN_MEMORY_BACKEND=torch`(不能设为 `native`,否则 side-stream DMA 会 stall),另有 `TORCH_NCCL_AVOID_RECORD_STREAMS=1`、`NCCL_NVLS_ENABLE=0` 等由 `pretrain.sh` 统一导出;
- 模型形状:按原生层数运行,`apply_model_shape_overrides` 被强制为 `false`(层数缩减仅保留给 Megatron 对比实验);
- 观测:`wall_clock_breakdown=true`,用于拆分 fwd / bwd / optimizer step 耗时。

## 3. 测试命令

### 3.1 NVMe 测试补丁的开启与关闭

`param_nvme` 与 SuperOffload 组合会触发 DeepSpeed 的兼容性缺陷(现象与根因见 §6.3),本目录提供一对可逆补丁,针对 DeepSpeed `c7cc64a90` 的 `deepspeed/runtime/superoffload/superoffload_stage3.py`:开启补丁用 `patches/superoffload_nvme_override.patch`(覆盖 SuperOffload 方法,复用 ZeRO-3 父类的 NVMe-safe 实现),关闭补丁用 `patches/superoffload_nvme_recovery.patch`(恢复原实现)。凡是矩阵中包含 `param_nvme` 且 optimizer 策略含 `super_offload_*` 的测试,运行前必须先开启补丁:

```bash
# 开启(应用 override 补丁)
bash examples/scale-down/01-analyse/02-deepspeed-offload/apply_superoffload_nvme_patch.sh

# 关闭(恢复 DeepSpeed 原实现)
bash examples/scale-down/01-analyse/02-deepspeed-offload/apply_superoffload_nvme_patch.sh --recovery
```

脚本通过 `PYTHON_BIN` 对应的 Python 环境执行 `importlib.util.find_spec("deepspeed")` 自动定位 DeepSpeed 安装位置,默认使用当前环境的 `python`;如果训练使用另一套虚拟环境,用 `PYTHON_BIN=/path/to/venv/bin/python` 指定即可。打补丁优先使用 `git apply`,对没有 Git 元数据的 `site-packages` 父目录回退到 `patch`;补丁无法干净应用时直接报错,避免覆盖本地修改。应用后需确认训练进程实际导入的 DeepSpeed 路径就是被修改的路径。

### 3.2 单机串行运行(全部测试)

`pretrain_experiment.sh` 是串行驱动器,按 模型 → micro-batch → recompute → optimizer 策略 × param 位置 的顺序逐个执行,单模型全矩阵为 4×3×6×3 = 216 个 run,dense + MoE 共 432 个。任一 run 失败或 OOM 不会中断扫描,状态会记入 summary 后继续下一个:

```bash
# 全矩阵(dense + MoE,默认 4 GPU)
bash examples/scale-down/01-analyse/02-deepspeed-offload/pretrain_experiment.sh

# 只跑子集:每个轴参数按需收窄即可
bash examples/scale-down/01-analyse/02-deepspeed-offload/pretrain_experiment.sh \
  --models Qwen/Qwen3.5-9B-Base \
  --optimizer_strategies "zero_3 zero_offload" \
  --param_positions "param_gpu param_cpu" \
  --recompute_combos recompute_act \
  --micro_batch_sizes "1 4"
```

追加 `--dry_run` 可以只生成各 run 的 ds_config 与结果目录而不启动训练,用于跑前核对配置。每个 run 的命名格式为 `<策略>-<param位置>__<recompute组合>__mbs<N>`(例如 `super_offload_0.9-param_cpu__recompute_act__mbs4`),工作 ds_config 生成在 `<repo_root>/.tmp/` 下,实际使用的副本随结果一起归档。

### 3.3 多设备并行运行

脚本本身没有内置多机协调,并行的方式是**把矩阵切分成互不重叠的子集,每个设备(机器或 GPU 组)启动一个独立实例**,各自跑各自的子集、各自落盘结果,最后汇总分析。同一台机器上切 GPU 组并行时,除 `CUDA_VISIBLE_DEVICES` 外还必须为每个实例设置不同的 `RESULTS_ROOT`、`TMP_CONFIG_DIR`、`NVME_PATH`,避免输出与 swap 目录互相覆盖:

```bash
# 同一台 8 GPU 机器:两个实例各占 4 GPU,分别跑 dense 与 MoE
CUDA_VISIBLE_DEVICES=0,1,2,3 RESULTS_ROOT=results/dense TMP_CONFIG_DIR=.tmp/dense NVME_PATH=/tmp/nvme_dense \
  bash examples/scale-down/01-analyse/02-deepspeed-offload/pretrain_experiment.sh \
  --models Qwen/Qwen3.5-9B-Base &
CUDA_VISIBLE_DEVICES=4,5,6,7 RESULTS_ROOT=results/moe TMP_CONFIG_DIR=.tmp/moe NVME_PATH=/tmp/nvme_moe \
  bash examples/scale-down/01-analyse/02-deepspeed-offload/pretrain_experiment.sh \
  --models Qwen/Qwen3.5-35B-A3B-Base &
```

所有配置项都遵循同一规则:**不配置则运行默认值,配置了则按照配置走**。优先级为 CLI 参数 > 环境变量 > 脚本默认值(仅环境变量项除外,它们只能通过环境变量设置)。切分矩阵最常用的五个轴配置项如下表,其余配置项(batch、学习率、GPU 数、NVMe、运行环境等)收录在文末[附录 A](#附录-a其余配置项):

| 项目 | 配置方式(CLI / 环境变量) | 默认值 | 说明 |
| --- | --- | --- | --- |
| 测试模型 | `--models` | `Qwen/Qwen3.5-9B-Base Qwen/Qwen3.5-35B-A3B-Base` | 仅接受这两个对齐的 Qwen3.5 Base 模型 ID |
| micro-batch 列表 | `--micro_batch_sizes` | `1 2 4 8` | 每个值必须整除 `per_gpu_batch_size` |
| recompute 组合列表 | `--recompute_combos` | `recompute_none recompute_act recompute_act_cpu` | `act_cpu` 自动包含 `act` |
| optimizer 策略列表 | `--optimizer_strategies` | `zero_3 zero_offload super_offload_1.0 super_offload_0.9 super_offload_0.75 super_offload_0.1` | `super_offload_` 后缀即 ratio |
| param 位置列表 | `--param_positions` | `param_cpu param_gpu param_nvme` | 含 `param_nvme` 时触发 NVMe 环境预检 |

### 3.4 Smoke test

Smoke test 的目标是用最小代价验证整条链路(环境、补丁、ds_config 生成、训练与指标落盘)能跑通:每个轴只取一个值,并建议选用**最极限节省 GPU 与 CPU 内存的组合**——参数驻留 NVMe(`param_nvme`,`max_in_cpu=0`,GPU 和 CPU 都不留参数副本)、optimizer state 全部在 CPU 侧更新(`super_offload_1.0`)、激活逐层重计算并异步卸载到 CPU(`recompute_act_cpu`)、`mbs=1`,这样即使在显存紧张的机器上也能完整走一遍 NVMe swapper、CPUAdam 与激活卸载路径:

```bash
# 前置:该组合包含 param_nvme + super_offload,必须先开启 §3.1 的 override 补丁
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

启动前脚本会确认 `nvme_path` 可写、所在文件系统确实挂载自 `nvme_device`,并加载 DeepSpeed `async_io` op,任一预检失败都会直接退出而不产生半截结果。如果只想核对配置,可在上述命令后追加 `--dry_run`。

## 4. 配置维度

| 维度 | 取值 | 对应配置 | 说明 |
| --- | --- | --- | --- |
| shard | ZeRO-3(固定) | `zero_optimization.stage = 3` | `offload_param`、`ratio<1`、`super_offload` 均只支持 ZeRO-3 |
| param position | `param_gpu` / `param_cpu` / `param_nvme` | `offload_param.device = none / cpu / nvme` | CPU 与 NVMe 均为 ZeRO-Infinity 参数卸载;NVMe 附带 `aio` 配置块(queue_depth=8、block_size=1MiB、use_gds=false)与 `max_in_cpu=0` |
| optimizer position | `zero_3`(GPU)/ `zero_offload`(CPU)/ `super_offload_1.0 / 0.9 / 0.75 / 0.1`(CPU+GPU 混合) | `offload_optimizer.device`、`super_offload`、`ratio` | `ratio` 是在 CPU 侧执行 optimizer update 的参数比例:`1.0` 等价于 ZeRO-Offload 的卸载范围但走 SuperOffload Stage-3 实现;`<1` 时其余子组由 GPU backup AdamW 更新,显存占用上升、吞吐预期上升 |
| recompute | `recompute_none` / `recompute_act` / `recompute_act_cpu` | `--activation_checkpointing` + `--cpu_checkpointing` | `act` 是 HF 原生逐层 gradient checkpointing(`use_reentrant=False`,且 `use_cache=False`);`act+cpu` 在此之上用 DeepSpeed `CheckpointHiddenStatesOffload` 把 checkpoint 层输入 hidden_states 经 side stream 异步 D2H 到 pinned CPU 缓冲,backward 时再 H2D 取回,GPU 额外驻留激活峰值约 `max_fwd_stash_count + keep_last_count` 个层 |
| batch size | `mbs ∈ {1, 2, 4, 8}` | `train_micro_batch_size_per_gpu` + `gradient_accumulation_steps` | `per_gpu_batch_size=16` 固定,`grad_accum = 16 / mbs`,全局 batch 恒定,扫描只改变切分粒度 |

组合约束(源码确认,详见 [offload-combination.md](../../../../documents/deepspeed/offload-combination.md)):`offload_param` 仅支持 ZeRO-3;`ratio < 1` 仅支持 ZeRO-3 且要求使用 `DeepSpeedCPUAdam`;`super_offload` 仅支持 ZeRO-3 + NVIDIA CUDA;`act+cpu` 依赖 `act` 开启(offload ctx 只对 HF `GradientCheckpointingLayer` 的 checkpoint 层输入打标记,单独开启无意义);`super_offload + param_nvme` 必须先应用 §3.1 的 override 补丁(见 §6.3)。驱动脚本在生成矩阵时已剔除无效组合(如 `cpu_checkpoint` 不开 `act` 的档位)。

## 5. 观测指标

每个 run 的输出独立落盘、按时间戳区分,汇总状态单独成文件:

```text
<repo_root>/results/01-analyse/02-deepspeed/
├── experiment_summary_<ts>.txt                 # 全部 run 的状态汇总(OK / OOM / FAILED(rc) / DRY_RUN)
└── <model>/<TEST_NAME>/<timestamp>/
    ├── run.log                                 # 完整 stdout/stderr(含 wall_clock_breakdown 分解)
    ├── metrics.csv                             # 每 step 指标(train.py MetricsLogger)
    └── ds_config.json                          # 该 run 实际使用的精确配置(由 pretrain.sh 归档)
```

`metrics.csv` 每 step 记录以下字段:

| 字段 | 含义 |
| --- | --- |
| `step` / `loss` | 训练步与 loss;同 seed 下各组合 step 1 的 loss 应完全一致,step≥2 允许因 Adam 实现差异而分叉(见 §6.1) |
| `iter_time_sec` | 单 step 耗时,取 warmup 后窗口的稳态值 |
| `global_tokens_per_sec` | 全局吞吐(tokens/s),跨配置比较的主指标 |
| `gpu_peak_gigabytes` / `cpu_peak_gigabytes` | GPU / CPU 侧峰值内存占用 |
| `cuda_memory_allocated_bytes` / `cuda_peak_memory_allocated_bytes` / `cuda_peak_memory_reserved_bytes` | PyTorch allocator 的当前、峰值 allocated 与峰值 reserved 显存 |

除 CSV 外,`wall_clock_breakdown=true` 会在 `run.log` 中给出 fwd / bwd / optimizer step / 通信的耗时分解;run 结束后驱动器会根据日志把失败归类为 OOM 或 FAILED 写入 summary。汇总分析的产物为:每模型一张 `配置 × (吞吐, 峰值显存, 耗时分解, loss 差异)` 总表、ratio 扫描曲线(吞吐与显存 vs ratio)、显存—吞吐前沿图(标注 Pareto 最优点),以及 O1(zero_offload)vs O2(super_offload_1.0)、P2 vs P3 的 SuperOffload 增益结论(分 Dense / MoE)。目录内的 `metrics.py` / `profiling.py` 提供记录与 OOM 观察(`record_memory_history`)的公共实现。

## 6. 实验记录

### 6.1 Adam 实现差异与 non-finite 报错

**现象与排查**:部分测试出现 non-finite(NaN/Inf)报错。排查发现,DeepSpeed 初始化完成后第一轮(step 1)各配置的 loss 完全相等,但经过一轮 Adam 更新后,step 2 起的 loss 不再完全相等——各配置实际走的 optimizer 实现不同(GPU `torch.optim.AdamW` / `DeepSpeedCPUAdam` / SuperOffload 的 CPU worker + GPU backup AdamW),Adam 实现差异导致更新后的参数出现数值分歧,并随训练轨迹传播,部分情况下表现为 non-finite 报错。各配置的更新路径如下:

| 配置 | 实际 optimizer | 更新路径 |
| --- | --- | --- |
| `zero_3` | `torch.optim.AdamW`(`torch_adam=true`) | GPU 上的 FP32 master partition + torch AdamW |
| `zero_offload` | `DeepSpeedCPUAdam` | CPU 上的 FP32 master partition + CPUAdam |
| `super_offload_1.0` | `DeepSpeedCPUAdam` | CPU worker 中更新,全部 subgroup 走 CPU |
| `super_offload_0.9 / 0.75 / 0.1` | `DeepSpeedCPUAdam` + GPU `torch.optim.AdamW` backup | CPU subgroup 由 CPUAdam 更新,GPU subgroup 由 backup AdamW 更新(其超参从 CpuAdam 的配置派生,可能与 GPU 侧 AdamW 存在细微差异) |

**修正此前的归因**:此前把 non-finite 归因为 DeepSpeed FusedAdam 的实现问题(在 zero-3 中改用 torch AdamW 后错误曾消失),但后来观察到 zero_offload(走 `DeepSpeedCPUAdam`,与 FusedAdam 无关)也出现相同的 non-finite 问题,说明该归因不成立,应从其他角度继续排查。**尚未解释的现象**:观察到有两个异构(CPU+GPU 混合更新)的实验 loss 曲线完全一致,这与"实现不同必然分叉"的预期矛盾,暂时无法解释。

**学习率与梯度裁剪**:早期实验将 learning rate 设为 `1e-3`,对随机初始化的模型来说过大,一次合法的首轮更新也可能直接把权重推到 NaN/Inf,在 offload 路径被测到之前就产生 non-finite。随后将默认值降到 `1e-4`仍不能排除首轮更新不稳定,因此稳定性排查阶段使用 `1e-6` 并启用全局梯度裁剪 `1.0`;学习率仍可用 `--learning_rate` 覆盖。梯度裁剪只能限制有限的大梯度,不能修复已经包含 NaN/Inf 的梯度。该配置用于短跑稳定性测试,正式从头预训练仍应另行配置 warmup 与学习率曲线。

### 6.2 加速测试

单轮测试曾超过 10 分钟,排查发现核心原因是 tokenizer 需要对数据集做约 7 分钟的预处理,训练本身反而不是瓶颈。将数据集加载比例从 10% 降低为 1% 后,预处理时间大幅缩短,单轮耗时恢复正常。`pretrain.sh` 当前的 `DATASET_PERCENTAGE` 默认值即为 `1.0`;对短步数的稳定性/吞吐对比,1% 的数据量已经足够,如需更大的数据覆盖面再通过环境变量调回。

### 6.3 NVMe 与 SuperOffload 的兼容性问题

**buffer slot 不足**:NVMe 参数卸载配置 `buffer_count = 15` 时会出现 buffer slot 不足的报错,该值不可下调,脚本默认 `NVME_BUFFER_COUNT=16`;同时默认 `buffer_size=4e8` 用于容纳当前 4-GPU Qwen3.5 矩阵的最大单参数分片,改变 GPU 数或模型后应通过 `--nvme_buffer_size` / `--nvme_buffer_count` 重新校准。

**NVMe 与 SuperOffload 不兼容**:参数位置和 optimizer 主参数位置是两个独立维度,`offload_param.device="nvme"` 只决定低精度 parameter partition 的驻留位置,而 SuperOffload 的 `subgroup_to_device` 只会取 CPU 或 GPU、不会取 NVMe。启用 NVMe 参数卸载后,DeepSpeed 只为不超过 `max_in_cpu` 的参数建立 CPU flat buffer,超过预算的 subgroup 其 `fp16_partitioned_groups_flat[sub_group_id]` 为 `None`(数据实际驻留 NVMe,需由 swapper 按需换入换出),`max_in_cpu=0` 时大多数 subgroup 为 `None` 是预期行为。问题在于 SuperOffload Stage-3 的重载函数会直接把 main-weight 复制到 model-weight,并按 `subgroup_to_device` 判断 CPU/GPU 后无条件访问 `.data`,不会像 ZeRO-3 父类那样在 flat buffer 为 `None` 时走 `_partitioned_params_swap_out()`,于是在"参数位于 NVMe、optimizer subgroup 位于 CPU"的组合下触发 `AttributeError: 'NoneType' object has no attribute 'data'`。修复方式是增加一个 patch,让 SuperOffload 按照父类 ZeRO-3 / ZeRO-Offload 的 NVMe-safe 配置路径执行,即 §3.1 的 `patches/superoffload_nvme_override.patch`,默认应在所有含 `param_nvme + super_offload_*` 的测试前应用。

## 附录 A:其余配置项

以下为 §3.3 五个矩阵轴之外的全部配置项,规则相同:不配置则运行默认值,配置了则按照配置走。

| 类别 | 项目 | 配置方式(CLI / 环境变量) | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| 矩阵轴 | 每 GPU 每 step 样本数 | `--per_gpu_batch_size` | `16` | `grad_accum = 该值 / mbs`,全局 batch 恒定 |
| 矩阵轴 | 学习率 | `--learning_rate` / `LEARNING_RATE` | `1e-6` | 必须为正数;稳定性排查使用更保守的值,原因见 §6.1 |
| 矩阵轴 | GPU 数 | `--num_gpus` | `4` | 与 `CUDA_VISIBLE_DEVICES` 配合切分 GPU 组 |
| 矩阵轴 | AutoEP 大小 | `--autoep_size` | `4` | 仅 MoE 模型使用,须整除专家数 |
| 矩阵轴 | MoE 专家数 | `--moe_num_experts` | `64` | 对比实验固定为 64,其余取值直接报错 |
| 矩阵轴 | 层数覆盖 | `--num_layers` / `--apply_model_shape_overrides` | `8` / `false` | 形状覆盖已禁用,设为 `true` 会报错退出 |
| 矩阵轴 | 空跑 | `--dry_run` | 关闭 | 只生成 ds_config 与结果目录,不启动训练 |
| NVMe | swap 目录 | `--nvme_path` / `NVME_PATH` | `/tmp/deepspeed_nvme_offload` | 每个 run 使用独立子目录,结束或中断时默认删除 |
| NVMe | 目标设备 | `--nvme_device` / `NVME_DEVICE` | `/dev/nvme2n1` | 启动前校验目录所在文件系统的挂载源 |
| NVMe | 设备校验开关 | `--nvme_device_check` / `NVME_DEVICE_CHECK` | `true` | 容器内看不到宿主机 block device 时可设 `false`,但须先人工确认目录确实在 NVMe 上 |
| NVMe | buffer 数量 | `--nvme_buffer_count` / `NVME_BUFFER_COUNT` | `16` | 设 15 会触发 buffer slot 不足报错,见 §6.3 |
| NVMe | 单 buffer 容量(元素数) | `--nvme_buffer_size` / `NVME_BUFFER_SIZE` | `400000000` | 须容纳当前 world size 下最大单参数分片,改 GPU 数后需重新校准 |
| NVMe | CPU 常驻上限 | `--nvme_max_in_cpu` / `NVME_MAX_IN_CPU` | `0` | `0` 保证该轴实际测到 NVMe 常驻 |
| NVMe | 保留 swap 数据 | `--keep_nvme_data` / `KEEP_NVME_DATA` | `false` | 调试时设 `true`;全矩阵务必保持默认,否则耗尽磁盘 |
| 运行环境 | 通信重叠 | `OVERLAP_COMM` | `true` | 写入 `zero_optimization.overlap_comm` |
| 运行环境 | 训练步数 | `STEPS` | `10` | |
| 运行环境 | warmup 步数 | `WARMUP_STEPS` | `2` | |
| 运行环境 | 序列长度 | `SEQ_LEN` | `4096` | |
| 运行环境 | 数据集 | `DATASET_NAME` | `wikitext` | |
| 运行环境 | 数据集加载比例(%) | `DATASET_PERCENTAGE` | `1.0` | 从 10.0 降为 1.0,见 §6.2 |
| 运行环境 | 随机种子 | `SEED` | `42` | |
| 运行环境 | 日志间隔 | `LOG_INTERVAL` | `1` | |
| 运行环境 | CPU 核绑定 | `BIND_CORES_TO_RANK` | `auto` | `auto` 表示 ds_config 含 `super_offload` 时自动加 `--bind_cores_to_rank`,可用 `true`/`false` 强制 |
| 运行环境 | pinned memory 后端 | `DS_PIN_MEMORY_BACKEND` | `torch` | 不要设为 `native`(mlock、未做 cudaHostRegister,side-stream DMA 会 stall) |
| 运行环境 | Python 解释器 | `PYTHON_BIN` | `python` | 用于 NVMe async_io 预检、batch 配置解析与 DeepSpeed 定位 |
| 运行环境 | 输出位置 | `REPO_ROOT` / `TMP_CONFIG_DIR` / `RESULTS_ROOT` | 仓库根 / `<repo_root>/.tmp` / `<repo_root>/results/01-analyse/02-deepspeed` | 同机多实例并行时必须为每个实例分别设置 |
