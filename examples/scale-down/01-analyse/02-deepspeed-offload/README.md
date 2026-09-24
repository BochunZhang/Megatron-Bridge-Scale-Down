# DeepSpeed Offload + Recompute 组合测试

## 1. 测试目标

在 ZeRO-3 分片策略下，系统测量 **parameter 存放位置 × optimizer 存放/计算位置 × recompute 策略**
三个维度的组合，回答：

1. 在每一档显存占用（memory tier）下，能达到的**最佳吞吐量**是多少；
2. **SuperOffload vs ZeRO-Offload**：在 `ratio=1.0`（全部 optimizer 子组在 CPU 更新）时，
   SuperOffload 针对 NVLink-C2C 的优化（CPU 更新的 optimizer state 直接搬到 GPU 再做精度转换，
   而不是在 CPU 上转换后再传输）能否带来可观的带宽/吞吐收益；
3. SuperOffload 的 `ratio` 扫描曲线：CPU/GPU 异构 optimizer 计算的最优配比；
4. `act`（HF 逐层重计算）与 `act+cpu`（checkpoint 层激活经 DeepSpeed
   `CheckpointHiddenStatesOffload` 异步卸载到 CPU）各自的显存收益与吞吐代价；
5. 以上结论在 **MoE 模型**与 **dense 模型**上是否一致。

相关背景分析见 [documents/deepspeed/offload-combination.md](../../../../documents/deepspeed/offload-combination.md)
（ZeRO-Offload / ZeRO-Infinity / SuperOffload 的关系与组合约束）。

## 2. 测试对象与环境

| 项目 | Dense | MoE |
| --- | --- | --- |
| 模型 | Qwen3-14B | Qwen3-30B-A3B |
| 启动脚本 | `pretrain_qwen35_7b.sh` | `pretrain_qwen35_35b_a3b.sh` |
| GPU 数 | 1 | 4 |
| ZeRO leaf module | 无 | `Qwen3MoeSparseMoeBlock`（`set_z3_leaf_modules`，MoE 模型必需） |

公共控制变量（两个模型一致，全部 run 固定）：

- `seq_len = 4096`，`bf16 = true`，`gradient_accumulation_steps = 1`
- 数据集 `tatsu-lab/alpaca`（`dataset_percentage = 10.0`），`seed = 42`
- `warmup_steps = 20`，`bench_steps = 10`（只统计 warmup 之后的稳定步）
- `overlap_comm = false`、`reduce_bucket_size = 4e8`、`sub_group_size = 4e8`（先固定，避免引入额外变量；后续可作为二阶扫描项）
- `pin_memory = true`（所有 CPU offload 配置）
- SuperOffload 配置统一带 `cpuadam_cores_perc = 0.90`，启动时加 `--bind_cores_to_rank`
- `wall_clock_breakdown = true`，用于拆分 fwd / bwd / optimizer step 耗时
- 环境变量：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（所有 run 统一开启，
  act+cpu 的 offload/restore 循环尤其需要）；`DS_PIN_MEMORY_BACKEND` 保持默认 `torch`

## 3. 配置维度与 ds_config 映射

| 维度 | 取值 | 实现方式 |
| --- | --- | --- |
| shard | ZeRO-3（固定） | `zero_optimization.stage = 3` |
| param 位置 | GPU / CPU | `offload_param.device = "none" / "cpu"`（CPU 即 ZeRO-Infinity 参数卸载） |
| optimizer 位置与计算 | GPU / ZeRO-Offload(CPU) / SuperOffload(CPU+GPU 混合) | `offload_optimizer.device`、`super_offload`、`ratio` |
| recompute | none / act / act+cpu | `--activation_checkpointing`（HF 逐层 gradient checkpointing）+ DeepSpeed `CheckpointHiddenStatesOffload`（见 §3.1） |

### 3.1 Recompute + CPU 激活卸载的实现方案

HF 构建的模型采用 **HF 原生逐层重计算 + DeepSpeed 激活 CPU 卸载** 组合，
方案详见 [documents/deepspeed/activation-offload-and-recompute/02-config.md](../../../../documents/deepspeed/activation-offload-and-recompute/02-config.md)：

1. **act（逐层重计算，HF 原生）**：HF 的 gradient checkpointing 本身就是逐 decoder
   layer 包 checkpoint，`finetune_zero3.py` 现有的
   `gradient_checkpointing_enable(use_reentrant=False)` 即为此路径，无需改造。
   必须使用非重入模式（重入模式与 ZeRO-3 不兼容，且卸载 ctx 的 marker patch
   面向非重入路径），同时 `model.config.use_cache = False`。
2. **act+cpu（激活卸载，DeepSpeed `CheckpointHiddenStatesOffload`）**：
   使用 `deepspeed.runtime.activation_checkpointing.offload_activations` 的
   `get_checkpoint_hidden_states_offloading_ctx_manager()`（本地 DeepSpeed 源码
   `offload_activations.py:507` 已确认存在）。它 patch HF 的
   `GradientCheckpointingLayer.__call__`，把每个 checkpoint 层的输入 hidden_states
   异步 D2H 到 pinned CPU 缓冲（side stream，与计算重叠），backward 需要时再 H2D 取回：

   ```python
   offload_ctx = get_checkpoint_hidden_states_offloading_ctx_manager()  # 创建一次，每 step 复用

   for batch in dataloader:
       with offload_ctx:                # forward 和 backward 必须在同一个 ctx 内
           loss = model(**batch).loss
           loss.backward()
       optimizer.step()
       optimizer.zero_grad()
   ```

   可调参数用默认值（`use_pin_memory=True`、`use_streams=True`、
   `max_fwd_stash_count=2`、`keep_last_count=1` 等）；GPU 额外驻留的激活峰值
   ≈ `max_fwd_stash_count + keep_last_count` 个层的 hidden_states。
3. **环境变量（官方教程明确要求）**：
   - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（offload/restore 循环易产生碎片）；
   - `DS_PIN_MEMORY_BACKEND` 保持默认 `torch`，**不要设成 `native`**
     （native 是 mlock、未做 cudaHostRegister，side-stream DMA 会 stall）。

`ratio` 语义（SuperOffload 专有）：在 **CPU 侧执行 optimizer update 的参数比例**。
`ratio=1.0` 全部子组走 CPU（等价于 ZeRO-Offload 的卸载范围，但走 SuperOffload Stage-3 实现）；
`0 < ratio < 1` 时其余子组由 GPU backup optimizer 更新，optimizer state 相应留在 GPU，
显存占用上升、吞吐预期上升。

组合约束（源码确认，见 offload-combination.md）：

- `offload_param` 仅支持 ZeRO-3；
- `ratio < 1` 仅支持 ZeRO-3，且要求使用 `DeepSpeedCPUAdam`；
- `super_offload` 仅支持 ZeRO-3 + NVIDIA CUDA；
- act+cpu（激活卸载）依赖 act（逐层重计算）开启：offload ctx 只对 HF
  `GradientCheckpointingLayer` 的 checkpoint 层输入打标记，单独开启无意义；
  且必须 `use_reentrant=False`。

## 4. 测试策略列表

采用**分阶段单轴扫描**而非全笛卡尔积（全组合 2×5×3=30/模型，成本高且大量组合冗余）。
每个阶段固定其余两轴，只动一轴；阶段之间用上一阶段选出的最优配置传递。
下表对 Dense 和 MoE 各执行一遍（共 2×12 = 24 个基础 run + 阶段 5 的 batch 扫描）。

### Phase 0 — Baseline

| ID | param | optimizer | recompute | 目的 |
| --- | --- | --- | --- | --- |
| B0 | GPU | GPU (Adam) | none | 纯 ZeRO-3 baseline，吞吐上界、显存下界参照 |

> 若 B0 OOM（MoE 在 4 GPU 上 ZeRO-3 全 GPU 驻留 fp32 optimizer state 仍很可能 OOM），则以 R1 作为该模型的事实 baseline，并在结果中记录 B0 OOM。

### Phase 1 — Recompute 轴（param=GPU，optimizer=GPU）

| ID | recompute | 目的 |
| --- | --- | --- |
| R1 | act（HF 逐层 gradient checkpointing，`use_reentrant=False`） | 重计算的显存收益 / 吞吐代价 |
| R2 | act + cpu（`CheckpointHiddenStatesOffload` ctx，见 §3.1） | checkpoint 层输入激活异步卸载 CPU 的增量收益 / 代价；与 R1 对比隔离出 D2H/H2D 传输开销 |

**产出**：确定后续阶段默认使用的 recompute 档位（预期选 act；若 act 仍 OOM 则选 act+cpu）。

### Phase 2 — Optimizer 轴（param=GPU，recompute=Phase 1 选定值）

| ID | optimizer | 目的 |
| --- | --- | --- |
| O1 | ZeRO-Offload（`device=cpu`，`super_offload=false`） | 经典 ZeRO-Offload：CPU Adam + CPU 精度转换 |
| O2 | SuperOffload `ratio=1.0` | **与 O1 核心对比**：卸载范围相同，验证 C2C 优化（GPU 侧精度转换）的增益 |
| O3 | SuperOffload `ratio=0.9` | ratio 曲线：10% optimizer 子组回 GPU（与现有脚本默认值一致） |
| O4 | SuperOffload `ratio=0.75` | ratio 曲线：25% 子组在 GPU，混合档位的下探点 |

**产出**：O1 vs O2 的 SuperOffload 带宽优势结论；ratio—吞吐—显存曲线，找出
"每 GB 显存换到的吞吐"拐点（预期在 C2C 高带宽机器上，低 ratio 更划算）。

### Phase 3 — Parameter 轴（ZeRO-Infinity，recompute=Phase 1 选定值）

| ID | param | optimizer | 目的 |
| --- | --- | --- | --- |
| P1 | CPU | GPU (Adam) | 只卸载参数：fp32 master weights + optimizer state 仍在 GPU，验证 bf16 参数卸载本身的收益/代价 |
| P2 | CPU | ZeRO-Offload | ZeRO-Infinity 经典全卸载 |
| P3 | CPU | SuperOffload `ratio=1.0` | 与 P2 对比，全卸载场景下 SuperOffload 的增益 |
| P4 | CPU | SuperOffload `ratio=0.75`（可选） | 全卸载 + 混合 optimizer 的折中点 |

### Phase 4 — 极限省显存组合

| ID | param | optimizer | recompute | 目的 |
| --- | --- | --- | --- | --- |
| E1 | CPU | SuperOffload `ratio=1.0` | act + cpu | 显存占用最低的组合，验证能否进一步放大 batch（进入 Phase 5） |

### Phase 5 — Batch size 扫描（吞吐上界）

对以下代表性配置做 batch size 扫描（`B ∈ {1, 2, 4, 8, 16, ...}` 直到 OOM），
绘制每个配置的**显存占用—最佳吞吐**前沿曲线：

- B0（或 R1，若 B0 OOM）— 显存换吞吐的 GPU-only 上界
- O2 / O4 — SuperOffload 全卸载与混合各一档
- P3 / E1 — 参数卸载侧的两档

> "不同 memory 占用下的最佳吞吐" 最终由本阶段产出：同一显存档位上比较不同组合的
> tokens/s，得出各显存预算下的推荐配置。

## 5. 观测指标与记录

每个 run 记录（写入 `results/` 下的 CSV/JSON，复用目录内 `metrics.py` / `profiling.py`）：

| 指标 | 来源 |
| --- | --- |
| 吞吐：tokens/s/GPU、samples/s、稳态 step time | bench_steps 窗口内统计 |
| 峰值显存：`torch.cuda.max_memory_allocated` / `max_memory_reserved` | 每 step 记录，取 max |
| 时间分解：fwd / bwd / optimizer step / comm | `wall_clock_breakdown` |
| CPU Adam 耗时、D2H/H2D 传输量 | DeepSpeed flops/step log + nsys（抽样 run） |
| loss（前 50 step） | 数值正确性 sanity check：同 seed 下各组合 loss 曲线应基本重合 |
| OOM 与否、OOM 时显存快照 | `profiling.py` 的 OOM observer（`record_memory_history`） |

汇总产物：

1. 每模型一张 `ID × (throughput, peak_mem, step_time_breakdown, loss_delta)` 总表；
2. ratio 扫描曲线（吞吐 & 显存 vs ratio）；
3. 显存—吞吐前沿图（Phase 5），标注 Pareto 最优点；
4. O1 vs O2、P2 vs P3 的 SuperOffload 增益结论（分 Dense / MoE）。

### 5.1 实测发现：Adam 实现差异导致 loss 不完全一致

各配置实际使用的 optimizer 实现与更新路径如下：

| 配置 | `basic_optimizer` | ZeRO-3 外层 | 实际更新路径 |
|---|---|---|---|
| `zero_3`（无 offload） | `FusedAdam`（GPU） | `DeepSpeedZeroOptimizer_Stage3` | GPU 上的 FP32 master partition + FusedAdam |
| `zero_offload_cpu` | `DeepSpeedCPUAdam` | `DeepSpeedZeroOptimizer_Stage3` | CPU 上的 FP32 master partition + CPUAdam |
| `super_offload_1.0` | `DeepSpeedCPUAdam` | `SuperOffloadOptimizer_Stage3` | CPU worker 中的 DeepSpeedCPUAdam，全部 subgroup 走 CPU |
| `super_offload_0.9/0.75/0.1` | `DeepSpeedCPUAdam` | `SuperOffloadOptimizer_Stage3` | CPU worker 更新 CPU subgroup；GPU subgroup 通过额外的 `torch.optim.AdamW` backup optimizer 更新 |

实测结论：

- 不同的 Adam 实现，计算结果是不同的，这和 Adam 内部的实现有关系。随机初始化的参数相同，
  理论上 step 1 计算得到的 loss 是相等的，但是第 2 轮的参数是 Adam 更新后的，
  Adam 的实现差别会导致从 step 2 开始的 loss 出现差异。
- 例如 super-offload 0.75 & super-offload 0.5 算出来的 loss 是不同的。
- 但是 zero-offload / super-offload_0.9 / super-offload_1.0 算出来的结果是相同的，暂时没有找到问题所在。
- zero-3 使用 deepspeed 的 FusedAdam 训练，会稳定出现 non-finit 报错，改用 torch 的 Adam 后错误消失。
  这个错误在 dense 和 expert 模型里面都出现了，应该是 FusedAdam 的实现存在问题。
- 同样使用 torch Adam，super-offload 0.0 和 zero-3 + torch Adam 的结果不同，
  推测是 Adam 的参数配置存在差异，导致后面的 loss 有差别。
  因为 super-offload 的 torch Adam 参数是从 DeepSpeed 的 CpuAdam 派生出来的。
- 测试 zero-3 (FusedAdam) / zero-3 (torch.Adam) / super_offload 0.0 / super_offload 1.0，
  step 1 的 loss 全部相等，证明这应该是 Adam 的实现差别导致的后续 step loss 不完全一致。

## 6. 前置改造项（TODO）

当前 harness 不能直接跑完上述矩阵，需要先补齐：

1. **act+cpu 激活卸载接入（方案已定，见 §3.1）**：`finetune_zero3.py` 现有的 HF
   `gradient_checkpointing_enable(use_reentrant=False)` 即 act 路径，无需改动；
   act+cpu 需要给 train 脚本增加 `--cpu_checkpointing` 开关：创建一次
   `get_checkpoint_hidden_states_offloading_ctx_manager()`，并把训练循环中每个
   step 的 forward + backward 包进同一个 `with offload_ctx:` 内（optimizer.step
   在 ctx 外）。同时在 launcher 中导出
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，且保持
   `DS_PIN_MEMORY_BACKEND` 为默认 `torch`。
2. **launcher 参数化**：现有脚本只有 `superoffload`（ratio 硬编码 0.90）/ `zerooffload`
   两个 MODE。需要扩展为 `MODE ∈ {baseline, zerooffload, superoffload, infinity, ...}` +
   `RATIO`、`OFFLOAD_PARAM`、`RECOMPUTE`（none/act/act+cpu）作为独立参数，并生成对应 ds_config。
3. **P1 组合校验**：`offload_param=cpu` + `offload_optimizer.device=none` 在目标 DeepSpeed
   版本（v0.19.6）上先做小规模冒烟验证。
4. **SuperOffload 依赖确认**：O3/O4（ratio<1）需要 `DeepSpeedCPUAdam`，确认镜像内
   CPU Adam op 已编译；MoE + SuperOffload 走的是 ZeRO-3 leaf module 路径（非 AutoEP），
   与 offload-combination.md 中 "EP+SuperOffload 不受支持" 的限制不冲突。

## 7. 预期与假设（待验证）

- O2 ≥ O1：C2C 带宽（GB200 ~900 GB/s）下 GPU 侧精度转换优于 CPU 侧转换后再传输；PCIe 机器上差距应明显缩小 —— 结论需注明硬件。
- ratio 曲线单调：ratio 越低吞吐越高、显存越大；最优点取决于 GPU 剩余显存能否容纳对应比例的 fp32 optimizer state。
- act+cpu 显存收益大；由于 D2H/H2D 走 side stream 异步拷贝、与计算重叠，吞吐代价预期小于同步卸载，但在 C2C/PCIe 带宽不足的机器上仍会显性化。GPU 额外驻留激活峰值 ≈ `max_fwd_stash_count + keep_last_count` 个层的 hidden_states，预期只在 R2 / E1 这类省显存组合中开启。
- MoE 与 dense 的相对结论（SuperOffload 增益、ratio 拐点）方向一致，但 MoE 因 expert 参数量大、计算密度低，optimizer offload 的相对开销预期更小。
