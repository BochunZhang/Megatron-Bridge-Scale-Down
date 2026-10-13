# GB200 micro-batch size 对 GEMM 效率的影响

结论：**mbs = 1 已经能把 GB200 的 tensor core 喂饱**。把 mbs 从 1 提高到 8，在
seq 4k 下收益与损失互相抵消，在 seq 16k 下则是净损失。原因是 mbs=1 时 GPU 已经跑在
boost 满频上，继续增大 mbs 只会拉长单 case 时长，触发功率墙降频与 L2 溢出。

- 被测模型：Qwen3.8 27B（dense）、Qwen3.8 2.4T（MoE）
- 硬件：单张 NVIDIA GB200（152 SM），BF16，TP = CP = 1，无通信
- 测试框架：`examples/gb200/micro-batch-size/`，scope `GEMM`
- 原始数据：`results/gb200/micro-batch-size/`，`seq4k` / `seq16k`

测量的是训练实际使用的 Megatron 模块与 kernel。计时在 CUDA 闸门下完成，屏蔽 CPU 开销，
不使用 CUDA Graph。`fwdbwd` 包含 dgrad 与 wgrad。输入 tensor 常驻 L2，无通信争抢，
因此结果是单卡计算时间的**上界**。

---

## 1. 两个模型的 GEMM 操作

`m` 是 token 行数（= `sequence_length × mbs`，grouped case 为本地各 expert 行数之和），
`k` 是输入特征数，`n` 是输出特征数。层数取自各模型 README，`次数` 为单次训练迭代
（含 MTP 层）中该 GEMM 的出现次数。

### 1.1 Qwen3.8 27B（dense，64 层 = 48 GDN + 16 gated attention）

| case | Megatron 模块 | k | n | 次数 |
| --- | --- | --- | --- | --- |
| `gdn_in_proj` | `TELayerNormColumnParallelLinear`（融合 RMSNorm） | 5120 | 16480 | 48 |
| `gdn_out_proj` | `TERowParallelLinear` | 6144 | 5120 | 48 |
| `attention_linear_qkv` | `TELayerNormColumnParallelLinear`（融合 RMSNorm，Q+KV+输出门） | 5120 | 14336 | 16 |
| `attention_linear_proj` | `TERowParallelLinear` | 6144 | 5120 | 16 |
| `mlp_fc1` | `TELayerNormColumnParallelLinear`（融合 pre-MLP RMSNorm） | 5120 | 34816 | 64 |
| `mlp_fc2` | `TERowParallelLinear` | 17408 | 5120 | 64 |
| `mtp_eh_proj` | `TEColumnParallelLinear` | 10240 | 5120 | 1 |
| `output_layer` | `tensor_parallel.ColumnParallelLinear` | 5120 | 248320 | 1 |

每层 dense MLP，无 MoE。hidden 5120，FFN 17408（fc1 输出 2×17408 = 34816，SwiGLU 门控），
词表 248320。

### 1.2 Qwen3.8 2.4T（MoE，92 层 = 69 GDN + 23 gated attention）

| case | Megatron 模块 | k | n | 次数 |
| --- | --- | --- | --- | --- |
| `gdn_in_proj` | `TELayerNormColumnParallelLinear`（融合 RMSNorm） | 8192 | 37120 | 69 |
| `gdn_out_proj` | `TERowParallelLinear` | 16384 | 8192 | 69 |
| `attention_linear_qkv` | `TELayerNormColumnParallelLinear`（融合 RMSNorm） | 8192 | 34816 | 23 |
| `attention_linear_proj` | `TERowParallelLinear` | 16384 | 8192 | 23 |
| `router_gating` | `TopKRouter.gating`（fp32 输出） | 8192 | 512 | 92 |
| `shared_expert_fc1` | `TEColumnParallelLinear` | 8192 | 4096 | 92 |
| `shared_expert_fc2` | `TERowParallelLinear` | 2048 | 8192 | 92 |
| `shared_expert_gate` | `F.linear(x, SharedExpertMLP.gate_weight)` | 8192 | **1** | 92 |
| `mtp_eh_proj` | `TEColumnParallelLinear` | 16384 | 8192 | 1 |
| `output_layer` | `tensor_parallel.ColumnParallelLinear` | 8192 | 248320 | 1 |
| `routed_expert_fc1` | `TEColumnParallelGroupedLinear`，8 组 | 8192 | 4096 | 92 |
| `routed_expert_fc2` | `TERowParallelGroupedLinear`，8 组 | 2048 | 8192 | 92 |

每个 token 除 routed expert 外还要过 shared expert。路由为均匀路由，每个本地 expert 收到
`1.25 × tokens` 行，因此 routed expert 的 `m` 是 `8 × 1.25 × tokens`。

两个模型都有两个极端形状，是后续分析的主要"噪音源"：`shared_expert_gate` 的 **n = 1**
本质是 reduction 而非 GEMM；`output_layer` 的 **n = 248320** 远大于其他所有层。

---

## 2. 测试结果

### 2.1 整迭代 GEMM 吞吐（按出现次数加权，TFLOP/s）

用各层单次耗时 × 出现次数求和，得到一次训练迭代中 GEMM 部分的合并吞吐。

| 模型 | 序列 | mbs=1 | mbs=2 | mbs=4 | mbs=8 | m1 → m8 |
| --- | --- | --- | --- | --- | --- | --- |
| 27B | 4k | **1582** | 1551 | 1518 | 1474 | −6.8% |
| 27B | 16k | **1597** | 1473 | 1454 | 1419 | −11.1% |
| 2.4T | 4k | **1604** | 1489 | 1454 | 1445 | −9.9% |
| 2.4T | 16k | 1436 | 1454 | 1413 | 1382 | −3.8% |

**四个组合的单点最优都是 mbs=1。** 序列越长，增大 mbs 的损失越大。

### 2.2 逐 case 的 m1 → m8 变化方向

以 ±3% 为阈值统计：

| 组合 | 提升 >3% | 基本持平 | 下降 >3% |
| --- | --- | --- | --- |
| 27B seq4k（8 case） | 4 | 0 | 4 |
| 27B seq16k（8 case） | **0** | 0 | **8** |
| 2.4T seq4k（12 case） | 5 | 2 | 5 |
| 2.4T seq16k（12 case） | 1 | 1 | **10** |

seq4k 下提升与损失数量相当，互相抵消；seq16k 下几乎全线下降。

### 2.3 关键 case 明细（`fwd` 模式，TFLOP/s）

只保留能说明问题的形状。`6144×5120` 这类"方"形状随 mbs 上升，长 `k` / 窄 `n` 形状随
mbs 下降。

| case | k × n | 27B 4k | 27B 16k | 2.4T 4k | 2.4T 16k |
| --- | --- | --- | --- | --- | --- |
| `attention_linear_proj` | 6144 × 5120 | 1534 → 1761 ↑ | 1745 → 1558 ↓ | 1724 → 1658 ↓ | 1782 → 1506 ↓ |
| `mlp_fc2` | 17408 × 5120 | 1721 → 1857 ↑ | 1844 → 1492 ↓ | — | — |
| `mtp_eh_proj` | 10240 × 5120 | 1536 → 1741 ↑ | 1716 → 1692 → | 1660 → 1646 → | 1771 → 1471 ↓ |
| `attention_linear_qkv` | 5120 × 14336 | 1473 → 1547 ↑ | 1521 → 1342 ↓ | 1727 → 1381 ↓ | 1436 → 1412 → |
| `routed_expert_fc2` | 2048 × 8192 | — | — | 1745 → 1733 → | 1823 → 1453 ↓ |
| `output_layer` | 5120 / 8192 × 248320 | 1620 → 1434 ↓ | 1459 → 1401 ↓ | 1456 → 1413 ↓ | 1446 → 1395 ↓ |
| `router_gating` | 8192 × 512 | — | — | 1159 → 1416 ↑ | 1359 → 1500 ↑ |
| `shared_expert_gate` | 8192 × **1** | — | — | 4 → 6 ↑ | 5 → 7 ↑ |

### 2.4 退化 case 的绝对量级（说明为何可忽略）

`router_gating` 与 `shared_expert_gate` 是唯一在四个组合下都上涨的 case，但绝对量级极小。

| case | k × n | m1 | m8 | 涨幅 | 单次 FLOPs | 整迭代占比 |
| --- | --- | --- | --- | --- | --- | --- |
| `router_gating` | 8192 × 512 | 348 | 452 | +30% | 0.1–0.8 G | < 0.5% |
| `shared_expert_gate` | 8192 × 1 | 3 | 4 | +28% | 0.2–6.4 G | < 0.5% |

`n = 512` / `n = 1` 的窄 GEMM 在 mbs=1 时 SM 占用率极低，抬 m 确实在填 SM，但即使涨 30%
也只是把 0.15 ms 变成 0.11 ms。**这两个 case 不应作为优化目标。**

---

## 3. 为什么 mbs = 1 已经喂饱了 GB200

### 3.1 mbs=1 时 GPU 已在 boost 满频

这是整个结论的根基。`sm_clock_mhz`（case 结束时的采样值）显示，mbs=1 时多个 case 已经
躺在 2062 MHz 的 boost 频率上：

| case | k × n | mbs=1 时钟 | mbs=8 时钟 |
| --- | --- | --- | --- |
| 27B `attention_linear_proj` | 6144 × 5120 | 2062 | 2062 |
| 27B `mlp_fc2` | 17408 × 5120 | 2062 | 2062 |
| 27B `mtp_eh_proj` | 10240 × 5120 | 2062 | 2062 |
| 2.4T `gdn_out_proj` | 16384 × 8192 | 2062 | 1305 |
| 2.4T `router_gating` | 8192 × 512 | 2062 | 2062 |

mbs=1、seq=4k 时 `m` 只有 4096 行，**tensor core 已经满载**。"m 太小喂不饱 tensor core"
这个常规担忧在这里不成立，所以抬 mbs 换不到 GEMM 效率，只会把 case 时长从 <1 ms 拖到
几十 ms。

### 3.2 机制一：功率墙降频

case 时长与采样到的时钟严格相关：

| case 时长 | 平均 `sm_clock_mhz` | 平均功耗 |
| --- | --- | --- |
| < 1 ms | **2062**（boost） | 232 W |
| 1–3 ms | 1958 | 267 W |
| 3–10 ms | 1437 | 353 W |
| 10–50 ms | 1295 | 619 W |
| > 50 ms | **1221** | **1179 W** |

算力上界：152 SM × 4 tensor core × 4096 FLOP/cycle（BF16 dense）× 2.062 GHz ≈ **2567 TFLOP/s**。
实测值 1419–1604 对应的是 **~1250–1350 MHz** 的稳态频率，正好等于实测采样到的时钟区间。
`output_layer` 在 27B seq4k 各 mbs 下功耗恒定在 1084–1174 W、时钟 1155–1305 MHz，是撞了
板级功率上限的直接证据。

**所以 1500 TFLOP/s 不是 GEMM 实现不好，是这卡在持续负载下只能跑到 ~1300 MHz。** 由此也
导致 `tflops` 列不能跨 case 比较：短 case 采样到 boost、长 case 采样到稳态，两者混在一起。

### 3.3 机制二：L2 溢出

测试框架的设计前提是"输入常驻 L2"。这个前提只在 m 较小时成立：

| 配置 | 单 case 输入张量 | 与 L2 的关系 |
| --- | --- | --- |
| 27B seq4k mbs1 | 5120 × 4096 × 2 B ≈ 42 MB | 可常驻 L2 |
| 27B seq16k mbs8 | 5120 × 131072 × 2 B ≈ 1.3 GB | 远超 L2 |

`m` 增大后激活读取转为从 HBM 取，权重读的摊薄收益被抵消。这解释了为什么最吃激活读取的
case 退化最严重：27B seq16k 的 `mlp_fc2` 掉 19%、2.4T seq16k 的 `routed_expert_fc2`
掉 20%，而权重占比高、激活占比低的 case（`output_layer`）几乎不变。

### 3.4 两个机制的分工

- **seq4k**：单 case 输入仍在百 MB 量级，L2 尚可承受。mbs 增大主要触发**降频**，
  所以表现为有涨有跌（涨的是 SM 未被占满的窄 GEMM，跌的是被降频吃掉的）。
- **seq16k**：L2 溢出与降频**同时生效**，所以几乎全线下降，且降幅更大。

---

## 4. 结论与适用范围

1. **mbs = 1 已经是 GEMM 效率的甜点。** 在 seq 4k 与 16k、两个模型、`fwd` 与 `fwdbwd`
   两种模式下，单点最优全部落在 mbs=1。
2. **不要用 mbs 作为 GEMM 效率的调节旋钮。** 增大 mbs 不带来 GEMM 算力提升，只会引入
   降频与 L2 溢出两项损失。
3. **`router_gating` / `shared_expert_gate` 的低 TFLOP/s 不是问题。** N=512 / N=1 的窄
   GEMM 在逐 kernel 归因里会以"低效 kernel"跳出来，但整迭代占比 < 0.5%，优化收益上限
   就是零点几个百分点。

**适用范围 —— 以下三点必须同时成立，结论才适用：**

- 只适用于**孤立模块的单卡 GEMM**。这份数据是单卡计算时间的上界，不含通信。
- 只覆盖 **seq 4k 与 16k**。更长序列时 `output_layer` 会第一个 OOM、第一个撞功率墙
  （27B seq4k mbs8 的 `output_layer` 峰值已达 44.5 GiB）。
- **不能用来论证"训练时 mbs 不能大"。** 真实训练里抬 mbs 的价值不在 GEMM 效率，而在
  通信与计算的重叠窗口变长、optimizer step 频率下降、DP/EP allreduce 摊薄。评估 mbs
  收益要看端到端 step time 或通信重叠率。

### 4.1 若要进一步验证

1. **锁频重测**：用 `nvidia-smi -lgc 1300` 固定 SM 频率后重跑，消除时钟变量，mbs 的纯
   几何效应会直接暴露。
2. **分离 L2 与 m 两个因素**：构造超出 L2 但 m 不变的输入尺寸，即可把当前混在
   seq16k/mbs8 一列里的两个机制拆开。
