# Qwen3.8 27B（dense）：GB200 单卡 GEMM 效率测试

目标：在单张 GB200 上，用 **Megatron 训练时实际执行的线性层模块与 kernel**，测量 Qwen3.8 27B 每个
GEMM 在不同 micro-batch size / 序列长度（`tokens = seq × mbs`）下的耗时与 TFLOP/s。计时屏蔽 CPU 开销，
不使用 CUDA Graph。与 [`qwen3.8-2.4T`](../qwen3.8-2.4T/README.md) 相互独立：配置、脚本和结果目录都分开。

测量框架（计时协议、Nsys 校验、CSV 列定义）见
[`examples/gb200/benchmark/README.md`](../../benchmark/README.md)。

## 1. 文件

| 文件 | 作用 |
| --- | --- |
| `mbench.yaml` | 模型结构（`GPTModelProvider` 字段）、扫描范围、计时参数 |
| `run_all_benchmarks.sh` | 运行 scope（默认只跑 `GEMM`），最后生成 CSV；支持 `--seq`、`--scope`、`--nsys` |
| `build_csv.sh` | 只重新生成某个序列长度的 CSV |
| `_env.sh` | 路径与 `PYTHONPATH` |

## 2. 模型配置与假设

本项目的 Qwen3.8 27B recipe（`src/megatron/bridge/recipes/qwen/gb200/qwen35.py`）直接读取
`Qwen/Qwen3.5-27B` 的 text config 构建模型，`mbench.yaml` 采用同一结构：

| 项 | 取值 |
| --- | --- |
| 层 | 64 层：48 × GatedDeltaNet + 16 × gated attention（`linear_attention_freq: 4`），**每层 dense MLP，无 MoE** |
| hidden | 5120，RMSNorm（zero-centered gamma），SiLU/SwiGLU，无 bias |
| Dense MLP | FFN 17408（fc1 输出 2 × 17408） |
| Gated attention | 24 Q 头、4 KV 头、head_dim 256，QK RMSNorm，sigmoid 输出门，RoPE（`rotary_percent 0.25`） |
| GatedDeltaNet | 16 key 头、48 value 头、head_dim 128/128，conv kernel 4 |
| 词表 | 248320（不共享 embedding） |
| 精度 | BF16 参数与激活，fp32 `main_grad`，`gradient_accumulation_fusion: true` |
| 序列 / mbs | 默认 4096（`--seq` 可改为 16k / 64k / 256k …）；mbs = 1, 2, 4, 8 |
| 并行假设 | TP = CP = 1（与 4×GB200 FSDP recipe 一致），单卡执行，不含任何通信 |

## 3. `GEMM`：所有线性层（经由其所属的 Megatron 模块执行）

| case | Megatron 模块 | M | K | N | 每次迭代出现次数 |
| --- | --- | --- | --- | --- | --- |
| `gdn_in_proj` | `TELayerNormColumnParallelLinear`（含融合 RMSNorm） | tokens | 5120 | 16480 | 48 |
| `gdn_out_proj` | `TERowParallelLinear` | tokens | 6144 | 5120 | 48 |
| `attention_linear_qkv` | `TELayerNormColumnParallelLinear`（含融合 RMSNorm，Q + KV + 输出门） | tokens | 5120 | 14336 | 16 |
| `attention_linear_proj` | `TERowParallelLinear` | tokens | 6144 | 5120 | 16 |
| `mlp_fc1` | `TELayerNormColumnParallelLinear`（含融合 pre-MLP RMSNorm） | tokens | 5120 | 34816 | 64 |
| `mlp_fc2` | `TERowParallelLinear` | tokens | 17408 | 5120 | 64 |
| `mtp_eh_proj` | `TEColumnParallelLinear` | tokens | 10240 | 5120 | 每个 MTP 层 1 次 |
| `output_layer` | `tensor_parallel.ColumnParallelLinear` | tokens | 5120 | 248320 | 1 |

- N 由 Megatron 模块自身决定：`gdn_in_proj` = 2·qk(2048) + 2·v(6144) + 2·48（β、α），
  `attention_linear_qkv` = Q(6144) + 2·KV(1024) + 输出门(6144)。CSV 中的 `k`/`n` 列为实际值。
- 输入布局与训练一致（`[s, b, k]`）。`fwdbwd` 包含 dgrad 和 wgrad，TE 层的 wgrad 直接累加进 fp32 `main_grad`。
- `GatedDeltaNet`、`GatedAttention` scope 也可以用 `--scope` 运行；`MLP`、`Experts` 只适用于 MoE 模型。

## 4. 运行

```bash
# 默认：GEMM，seq 4k，mbs 1/2/4/8
examples/gb200/micro-batch-size/qwen3.8-27b/run_all_benchmarks.sh

# 长序列（k = 1024）：结果写入 seq16k/、seq256k/ …；放不下显存的 case 在 CSV 中标为 status=oom
examples/gb200/micro-batch-size/qwen3.8-27b/run_all_benchmarks.sh --seq 16k
examples/gb200/micro-batch-size/qwen3.8-27b/run_all_benchmarks.sh --seq 256k --micro-batch-sizes 1

# 加 nsys trace + SQL 交叉校验列；同时跑 GDN / attention 模块
examples/gb200/micro-batch-size/qwen3.8-27b/run_all_benchmarks.sh --nsys --scope "GEMM GatedDeltaNet GatedAttention"

# 多个序列长度一次跑完（空格或逗号分隔），含小于 1k 的 token 数
examples/gb200/micro-batch-size/qwen3.8-27b/run_all_benchmarks.sh --seq "256,1k" --scope GEMM

# 重新整理某个序列长度的 CSV
examples/gb200/micro-batch-size/qwen3.8-27b/build_csv.sh --seq 16k
```

## 5. 输出

```
results/gb200/micro-batch-size/qwen3.8_27b/seq<len>/
  gemm/                        每个 (case, mbs, mode) 一个 JSON
  nsys/gemm.nsys-rep / .sqlite 仅 --nsys
  csv/gemm.csv
```

整模型单次迭代的 GEMM 时间可以用 `gemm.csv` 中的单层耗时 × 出现次数加权求和来估算。

## 6. 已知局限

- 只覆盖 BF16；只测孤立模块（输入常驻 L2、无通信争抢），结果是单卡计算时间的上界。
- 短 case 与长 case 可能处于不同的功耗 / 频率状态（boost 突发 vs 功耗墙稳态），跨 mbs / seq 比较
  TFLOP/s 时需结合 `sm_clock_mhz`、`power_w` 与单次迭代时长判断。
