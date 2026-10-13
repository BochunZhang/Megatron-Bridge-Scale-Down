# Selective activation offload：dense / expert 模型的开销归因与规模测量

本目录测量 **fine-grained activation offload**（模块级 selective offload）在 dense MLP 和 MoE
expert 上分别带来多少吞吐损失，损失来自哪里，以及在更大规模上 offload 是否值得。

上一轮实验保留在 [../02-activation-offload-on-dense-and-expert-model-outdate-01/](../02-activation-offload-on-dense-and-expert-model-outdate-01/)，
本轮是它的重构版。

## 1. 要回答的三个问题

1. **offload 本身的代价**：排除 HybridEP 的 host 同步之后，开启 selective offload 还会掉多少吞吐。
2. **损失来源归因**：dense 和 expert 的下降分别来自 offload 的 D2H/H2D 拷贝，还是来自 MoE 的
   dispatch/combine 同步点，还是 attention 部分的 offload。
3. **规模上是否值得**：小规模（4 卡）归因，中规模（16 卡）沿 Qwen3.5 上限测，大规模（32 卡）
   用 PP 作为对照。

## 2. 前提：HybridEP 的 host 同步

expert 下降最大的原因不是 offload 本身，而是 HybridEP 在每个 MoE 层、每个 microbatch 上的一次
host 同步：`_HybridEPManager.setup_metadata` 中的
`padded_num_tokens = int(max_num_tokens_across_ep.item())`。

- 它在 `moe_hybridep_pad_uneven_dispatch_inputs=True` 时执行（EP 组内 `all_reduce(MAX)` 之后读回）。
- Bridge 的 `TransformerConfig` 在 `flex+hybridep` 且**未开启 CUDA graph** 时会自动打开这个开关
  （`src/megatron/bridge/models/transformer_config.py`）；本实验强制 `cuda_graph_impl=none`，
  所以它一定会生效。
- dispatch 内部 DeepEP 的其他同步点在实测中没有表现出阻塞，本轮不处理。

**解法（本轮新增）**：`--sync-with-sm true`（`model.moe_hybridep_sync_with_sm=true`）只替换这一处
`.item()`，用一个 SM 的 Triton kernel 把规约后的值写进 pinned 内存，再用
`atomic_xchg(scope="sys")` 发布序号，host 轮询序号（不经过驱动同步）。

| 文件 | 作用 |
| --- | --- |
| `megatron/core/transformer/moe/hybrid_ep_sm_sync.py` | SM copy kernel + host 轮询读取器 `HostCountsReader` |
| `megatron/core/transformer/moe/token_dispatcher.py` | `setup_metadata` 中 `.item()` 按开关改走 SM copy |
| `megatron/core/transformer/transformer_config.py` | 新开关 + 后端校验 |

**为什么是 SM 而不是 `.item()`**：host 仍要等 stream 跑完 `all_reduce`，所以省掉的是驱动级同步和
流水线排空的开销，不是"等待"本身。
pinned 内存 + 异构 `cudaMemcpy` 就是同一件事的另一种写法；用 SM 是因为它把拷贝和发布序号合并成
一次 kernel 启动，并且不需要专门为 tiny copy 建一条 stream。

**它对 offload 结论的影响**：这个同步在 baseline 和 offload 两个 case 里都存在，所以它单独就解释了
expert 那部分额外下降。只有把它替换掉，剩下的差值才是真正的 offload 代价。

## 3. 代价：expert 的 offload 模块与 sync-free 不相容

`MoEExperts` 在开启 `expert_fc1` / `moe_act` offload 时会强制关闭 TE op-fuser：

```python
# megatron/core/transformer/moe/experts.py
if getattr(self, "offload_expert_fc1", False) or getattr(self, "offload_moe_act", False):
    return False  # Selective expert_fc1/moe_act offload is only supported unfused.
```

而 sync-free 路径需要 op-fuser（`moe_expert_rank_capacity_factor` 会校验
`use_transformer_engine_op_fuser=True`）。因此：

- 若想同时拿到 "无 host 同步" 和 "expert activation offload"，只能用 **`fused_group_mlp`** 这一组
  （offload 整个 fused grouped MLP 的输入）。
- 本目录的 expert case 因此有三档：`offload-mlp`（`expert_fc1,moe_act`，unfused）、
  `offload-attn-mlp`（再加 attention 组）、以及 SM-sync 的 A/B（同栈对比 blocking 与 SM copy）。

## 4. 三组矩阵

每组都包含**不开启 offload 的 baseline**和开启 offload 的对照组，且在模型、dtype、MBS、GBS、
dispatcher、并行度上完全一致。

### a 组：4 卡归因（dsw）

| 子组 | 模型 | 裁剪 | 说明 |
| --- | --- | --- | --- |
| dense proxy | Qwen3.5 9B | 16 层，MTP=0 | 与 expert proxy 同深度，便于横向对比 |
| expert proxy | Qwen3.5 35B-A3B | 16 层、64 experts、`linear-attention-freq=4` | 跑 alltoall 与 hybridep 两个 dispatcher |
| deepseek dense proxy | DeepSeek-V3 | 4 层、32 experts、`moe_layer_freq=[0,0,0,0]` | MLA 架构对照 |
| deepseek expert proxy | DeepSeek-V3 | 4 层、32 experts、`moe_layer_freq=[1,1,1,1]` | MLA 架构对照 |

hybridep 子组额外跑 `baseline-sync-sm` 与 `offload-mlp-sync-sm`，用来单独量出 HybridEP
host 同步的代价。

### b 组：16 卡，FSDP1 + selective offload（dlc）

完整 Qwen3.5-35B-A3B，`--pp 1 --ep 16 --sharding fsdp1`，4 节点 × 4 卡。

### c 组：32 卡，PP 对照（dlc）

Qwen3.5-35B-A3B，`--pp 2 --ep 16 --sharding none`，8 节点 × 4 卡。与 b 组用同一个模型和
同一组数据参数，因此 "16 卡 FSDP + offload" 与 "32 卡 PP" 的对比是可比的一对。

> b/c 组用 35B-A3B 而不是 122B，是因为 122B 在 PP=2 下每卡 expert 参数约 7.4B，
> 静态显存（bf16 参数 + fp32 梯度 + Adam）已在 90 GB 量级，4 节点/8 节点的余量不足以稳定跑满
> 矩阵。若显存确认足够，可用 `--model-family qwen --mbs` 与 recipe 覆盖切到
> `qwen35_text_122b_a10b_pretrain_4gpu_gb200_bf16_fsdp1_config`。

## 5. 文件说明

| 文件 | 作用 |
| --- | --- |
| `benchmark_offload.sh` | 矩阵入口，展开 a/b/c 组并逐项调用训练脚本 |
| `run_pretrain.sh` | 单个配置的训练入口：拼 override、启动、采集显存与 TFlops、写 config/summary |
| `lib/common.sh` | 共用函数：模型梯子、recipe 名、offload 模块列表、显存监控、TFlops 导出 |
| `analyse_offload_results.py` | 按组配对 baseline 与 offload，输出 CSV 汇总与样本 |
| `../02-activation-offload-on-dense-and-expert-model-outdate-01/` | 上一轮实验（含 .mjs 版本与 32 卡专文），保留作对照 |

## 6. 运行

前置：仓库根目录 `uv sync`；dsw 4 卡可直接跑 a 组；DLC 需要共享仓库路径与结果路径。

### a 组（4 卡）

```bash
EXPERIMENT_DIR=examples/scale-down/01-analyse/01-fine-grained-offload/02-offload-on-dense-or-expert-model

# 先看展开的矩阵
bash "${EXPERIMENT_DIR}/benchmark_offload.sh" --dry-run --group a --mbs 1,2

# Qwen 归因
bash "${EXPERIMENT_DIR}/benchmark_offload.sh" --group a --mbs 1,2

# DeepSeek 归因
bash "${EXPERIMENT_DIR}/benchmark_offload.sh" --group a --model-family deepseek --mbs 1,2
```

### b 组（16 卡，DLC）

```bash
export MASTER_ADDR=<rank-0-host-or-ip>
export MASTER_PORT=29501
export RUN_TIME="20261013-16gpu-001"   # 提交端生成一次，传给全部 4 个 worker

bash "${EXPERIMENT_DIR}/benchmark_offload.sh" \
  --group b --mbs 1,2,4 --gbs 128 --results-root "${RESULTS_ROOT}"
```

### c 组（32 卡，DLC）

```bash
export MASTER_ADDR=<rank-0-host-or-ip>
export RUN_TIME="20261013-32gpu-001"   # 提交端生成一次，传给全部 8 个 worker

bash "${EXPERIMENT_DIR}/benchmark_offload.sh" \
  --group c --mbs 1,2,4 --gbs 32 --results-root "${RESULTS_ROOT}"
```

`GBS` 必须能被 `MBS × DP` 整除，脚本会在启动前校验并在不满足时报错退出（上一轮的脚本只校验了
`GBS % MBS`，导致 16/32 卡下 MBS>1 静默失败）。

### 归因用 profile

```bash
bash "${EXPERIMENT_DIR}/benchmark_offload.sh" \
  --group a --mbs 1 --profile nsys
```

只对代表性 case 开 profile；profile 运行不进入 CSV 汇总。

## 7. 结果结构与汇总

```text
<results-root>/<model>/gpu_<n>-<dtype>-pp_<p>-ep_<e>-mbs_<m>-gbs_<g>-<dispatcher>-<run-name>/<run-time>/
├── config.json          # 权威记录：规模、并行度、offload 列表、sync 模式
├── command.txt          # 实际 distributed 命令
├── train.log            # 训练日志
├── gpu_utilization.json # 每步 TFlops/GPU
├── gpu_memory/          # nvidia-smi 显存采样
├── summary.json         # status=0 表示成功
└── profile/ memory/     # 仅 profile 运行
```

非 0 号节点把节点本地文件写到 `<run-time>/rank<N>/`，共享目录不会被并发覆盖。

```bash
uv run python "${EXPERIMENT_DIR}/analyse_offload_results.py" --results-root "${RESULTS_ROOT}"
```

分析脚本的分组 key 是 `(model, dtype, gpu_count, pp, ep, gbs, mbs, dispatcher, sharding)`，
**包含规模**，因此 4 卡、16 卡、32 卡的结果不会互相配对。输出 `offload-summary.csv`
（每组各 case 的平均 TFlops 与相对 baseline 的降幅）和 `offload-samples.csv`
（最后 4 个 iteration 的原始值、均值与方差）。

## 8. 验收与解读

1. `summary.json.status == 0`，所有 rank 正常退出。
2. `train.log` 无 NaN/Inf、无 NCCL timeout、无 OOM、无 Hydra 配置错误。
3. 先看 `offload-samples.csv` 的标准差，再判断 5% 量级的差异是否真实。
4. 显存收益不在 CSV 里：读每个 run 的 `gpu_memory/`。offload 的"收益"是省下的 HBM，
   "代价"是 CSV 里的降幅，两者要一起看。
5. `--sync-with-sm true` 的 run 不能与 `--sync-with-sm false` 的 run 混在同一个结论里；
   它们回答的是不同问题（前者的差值 = offload 净代价，后者 = 实际端到端代价）。

## 9. 已知限制

- SM copy 依赖 Triton，且不能与 CUDA graph capture 同时使用；本目录所有 case 都设置了
  `cuda_graph_impl=none`。
- `fused_group_mlp` 与 `expert_fc1`/`moe_act` 互斥，且 `fused_group_mlp` 要求 TE op-fuser。
- 多机 nsys 只采集各节点 local rank 0–3（`profiling.profile_ranks=[0,1,2,3]`），跨节点通信
  需要按节点扩展该列表才能看到全部 rank。
