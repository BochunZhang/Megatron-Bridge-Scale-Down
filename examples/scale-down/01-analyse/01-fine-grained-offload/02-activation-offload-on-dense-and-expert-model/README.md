# Selective activation offload：dense / expert 模型的开销归因与规模测量

本目录测量 **fine-grained activation offload**（模块级 selective offload）在 dense MLP 和 MoE
expert 上分别带来多少吞吐损失，损失来自哪里，以及在更大规模上 offload 是否值得。

上一轮实验保留在 [../02-activation-offload-on-dense-and-expert-model-outdate-01/](../02-activation-offload-on-dense-and-expert-model-outdate-01/)，
本轮是它的重构版。

## 1. 要回答的问题

1. **offload 本身的代价**：排除 HybridEP 的 host 同步之后，开启 selective offload 还会掉多少吞吐。
2. **损失来源归因**：dense 和 expert 的下降分别来自 offload 的 D2H/H2D 拷贝，还是来自 MoE 的
   dispatch/combine 同步点，还是 attention 部分的 offload。
3. **规模上是否值得**：小规模（4 卡）归因，8 卡用 16 卡的 expert 模型减半 expert 数量做中间点，
   中规模（16 卡）沿 Qwen3.5 上限测，大规模（32 卡）用 PP 作为对照。

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

## 3. offload 与 recompute 策略

4/8/16 卡用 FSDP1（Megatron-FSDP，`data_parallel_sharding_strategy=optim`），32 卡改用 DDP
（`--sharding none`，Megatron 分布式优化器）——Megatron-FSDP 的 device mesh helper 从全局
world size 推导，而 `dp_cp` 是 per-pipeline-stage 组，只在 PP=1 时自洽，PP=4 会在
`_get_dp_tp_mesh` 里直接抛 `Shape mismatch, 32 != 8`。两者都是 ZeRO-1，模型状态占用基本持平。
**所有档位都不开 recompute**，只用 fine-grained activation offload。baseline 与 offload case
的唯一区别是 offload 开关与模块列表：

| 固定配置 | 值 |
| --- | --- |
| `recompute_granularity` / `recompute_modules` | `null` |
| `cpu_offloading`（按层 offload） | `false` |
| `optimizer_cpu_offload` | `false` |
| `cuda_graph_impl` | `none` |

| case | dense | expert |
| --- | --- | --- |
| `baseline` | 不开 offload | 不开 offload |
| `offload-mlp` | `[mlp_norm,mlp_act]` | `[mlp_norm,expert_fc1,moe_act]` |
| `offload-attn-mlp` | 再加 `attn_norm,core_attn,attn_proj` | 再加 `attn_norm,core_attn,attn_proj` |

DeepSeek 的 attention 组额外包含 `qkv_linear`。Qwen3.5 的 GatedDeltaNet 复用 attention 组名，
不加 `qkv_linear`，因此 GDN 的 `in_proj` 输入不 offload。expert 默认 `--sync-with-sm true`，
用于排除 HybridEP `.item()` 的阻塞。

## 4. 测试脚本

| 脚本 | 规模 | 模型 | 并行 | 默认矩阵 |
| --- | --- | --- | --- | --- |
| `run_4gpu_expert_activation_offload_blocking.sh` | 1 节点 × 4 卡 | Qwen3.5-35B-A3B 裁剪 | EP=4，HybridEP | `offload-attn-mlp` × {blocking, sync-sm}，MBS=1，nsys |
| `run_4gpu_compare_activation_offload_influence_on_dense_and_expert.sh` | 1 节点 × 4 卡 | Qwen3.5-9B / 35B-A3B 裁剪 | dense EP=1；expert EP=4 | 3 case × MBS{1,2} × {dense, expert} |
| `run_8gpu_compare_activation_offload_influence_on_dense_and_expert.sh` | 2 节点 × 4 卡 | Qwen3.5-9B 完整 / 35B-A3B 128 expert | PP=1；dense EP=1；expert EP=8 | 3 case × MBS{1,2} × {dense, expert}，GBS=64 |
| `run_16gpu_compare_activation_offload_influence_on_dense_and_expert.sh` | 4 节点 × 4 卡 | Qwen3.5-9B / 35B-A3B 完整 | PP=1；dense EP=1；expert EP=16 | 3 case × MBS{1,2} × {dense, expert}，GBS=128 |
| `run_32gpu_compare_activation_offload_influence_on_dense_and_expert.sh` | 8 节点 × 4 卡 | Qwen3.5-27B / 122B-A10B 完整 | PP=4；dense EP=1；expert EP=8 | 3 case × MBS{1} × {dense, expert}，GBS=128 |

### 4 卡裁剪配置

| 类型 | 模型 | 裁剪 | 并行 |
| --- | --- | --- | --- |
| dense | Qwen3.5-9B | `num_layers=16`、`mtp_num_layers=0`（12 层 GDN + 4 层 full attention） | DP=4，EP=1 |
| expert | Qwen3.5-35B-A3B | `num_layers=16`、`num_moe_experts=64`、`mtp_num_layers=0`、`linear_attention_freq=4`，top-k 保持 8 | DP=4，EP=4，每卡 16 个 expert |
| dense（`--model-family deepseek`） | DeepSeek-V3 | `num_layers=4`、`num_moe_experts=32`、`moe_layer_freq=[0,0,0,0]` | EP=1 |
| expert（`--model-family deepseek`） | DeepSeek-V3 | `num_layers=4`、`num_moe_experts=32`、`moe_layer_freq=[1,1,1,1]` | EP=4 |

两个 Qwen 模型裁到同一深度，hidden size 保持原值（9B 为 4096，35B-A3B 为 2048）。

### 8/16/32 卡模型与显存估算

估算按 bf16 参数 + fp32 梯度（ZeRO-1 不分片参数和梯度）+ fp32 Adam（按 EDP 分片）+ 无 recompute 的
activation（1F1B 下 stage 0 保留 PP 个 microbatch）+ fp32 logits，GB200 每卡约 186 GB：

| 规模 | 模型 | 并行 | MBS=1 | MBS=2 |
| --- | --- | --- | --- | --- |
| 8 卡 | Qwen3.5-9B | PP1 EP1 DP8 | ≈ 101 GB | ≈ 133 GB |
| 8 卡 | Qwen3.5-35B-A3B（128 expert） | PP1 EP8 EDP1 | ≈ 89 GB | ≈ 122 GB |
| 16 卡 | Qwen3.5-9B | PP1 EP1 DP16 | ≈ 94 GB | ≈ 126 GB |
| 16 卡 | Qwen3.5-35B-A3B | PP1 EP16 EDP1 | ≈ 87 GB | ≈ 120 GB |
| 32 卡 | Qwen3.5-27B | PP4 EP1 DP8 | ≈ 124 GB | ≈ 180 GB（stage 0），大概率 OOM |
| 32 卡 | Qwen3.5-122B-A10B | PP4 EP8 EDP1 | ≈ 137 GB | ≈ 180 GB（stage 0），大概率 OOM |

- 8 卡 expert 沿用 16 卡的 35B-A3B（40 层 + MTP、top-k 8 不变），只把 `num_moe_experts` 从 256 减到
  128，EP=8 下每卡仍是 16 个 expert，expert 参数、梯度、Adam 与 16 卡相同；只有约 2.5B 非 expert 参数的
  Adam 从 /16 变成 /8（+≈2 GB）。
- 8 卡 dense 仍用完整 Qwen3.5-9B：DP 从 16 降到 8 只让 Adam 分片从 ≈6.75 GB 增加到 ≈13.5 GB，
  静态约 68 GB，MBS=2 也有约 50 GB 余量，不需要换更小的 Qwen3.5 模型。
- GBS 保持每个 DP rank 8 条样本（4 卡 32、8 卡 64、16 卡 128）。
- 16 卡、PP=1 能放下的最大 Qwen3.5 dense / MoE 分别是 9B 和 35B-A3B；27B（≈184 GB 静态）和
  122B-A10B（≈175 GB 静态）不开 PP 放不下。
- 32 卡默认只测 MBS=1；`--mbs 1,2` 可以尝试 MBS=2，OOM 时以 MBS=1 结果为准。
- learned routing 下 token 不均衡会让 expert 的 activation 高于估算。

## 5. DLC 运行

所有 worker 执行同一条命令，不需要手动导出任何变量：

- `MASTER_ADDR`、`MASTER_PORT`、`WORLD_SIZE`（节点数）、`RANK`（节点序号）由 DLC 注入。
  `WORLD_SIZE` 与脚本要求的节点数不一致时直接报错。
- **结果目录与 master 同步**：未传 `--run-time` 时，0 号节点生成批次号，通过
  `MASTER_ADDR:MASTER_PORT` 上的 TCPStore 发给其他节点，所有节点确认收到后才开始训练。
  因此所有节点写入同一个 `<results-root>/<experiment>/.../<run-time>/`：0 号节点写根目录，其他节点写
  `rank<N>/`。
- 每个 case 用独立的 rendezvous 端口（`MASTER_PORT+1`、`+2`…），避免上一个 case 的端口
  未释放。
- `--results-root` 必须是所有节点可见的共享存储。
- **HF 缓存放在节点本地（仅 8/16/32 卡）**：所有 rank 同时从共享 CPFS 的 `HF_HOME` 读
  `tokenizer.json` 时，若缓存 blob 被并发改写，会报 `OSError: [Errno 116] Stale file handle`。
  因此 wrapper 在矩阵开始前，每个节点把用到的 HF 仓库（不含权重）下载一次到
  `LOCAL_HF_HOME`（默认 `/tmp/megatron-bridge-hf-home`，可导出覆盖），训练时使用
  `HF_HOME=$LOCAL_HF_HOME` 和 `HF_HUB_OFFLINE=1`。4 卡脚本仍使用共享的 `.cache/huggingface`。
- `NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN` 默认不设置，由 HybridEP 自动检测；只有检测错误
  时才手动导出。

拆分到多个 DLC 任务并行跑：

```bash
E=examples/scale-down/01-analyse/01-fine-grained-offload/02-activation-offload-on-dense-and-expert-model

# 8 卡：1 个 DLC 任务，2 节点
bash $E/run_8gpu_compare_activation_offload_influence_on_dense_and_expert.sh

# 16 卡：4 个 DLC 任务，各 4 节点
bash $E/run_16gpu_compare_activation_offload_influence_on_dense_and_expert.sh --mbs 1 --model-type dense
bash $E/run_16gpu_compare_activation_offload_influence_on_dense_and_expert.sh --mbs 1 --model-type expert
bash $E/run_16gpu_compare_activation_offload_influence_on_dense_and_expert.sh --mbs 2 --model-type dense
bash $E/run_16gpu_compare_activation_offload_influence_on_dense_and_expert.sh --mbs 2 --model-type expert

# 32 卡
bash $E/run_32gpu_compare_activation_offload_influence_on_dense_and_expert.sh --model-type expert
```

### DSW 4 卡

```bash
# 人工核验 .item() 阻塞是否消失（nsys）
bash $E/run_4gpu_expert_activation_offload_blocking.sh

# dense / expert 对比
bash $E/run_4gpu_compare_activation_offload_influence_on_dense_and_expert.sh
bash $E/run_4gpu_compare_activation_offload_influence_on_dense_and_expert.sh --model-family deepseek
```

所有脚本都支持 `--dry-run`，先确认矩阵再启动。共同参数：`--mbs`、`--model-type`、`--case`、
`--dtype`、`--gbs`、`--train-iters`、`--sync-with-sm`、`--profile`、`--results-root`、`--run-time`。

### blocking 核验看什么

在 nsys 的 step 7–8 中对比两个 run：

- `offload-attn-mlp-blocking`：`MoEFlexTokenDispatcher.dispatch_preprocess` 内有 D2H
  `cudaMemcpyAsync` + `cudaStreamSynchronize`。
- `offload-attn-mlp-sync-sm`：同一区间内是 Triton `_sm_copy_with_flag_kernel` 和 NVTX
  `hybrid-ep sm-sync wait`，没有 stream 同步。
- 两个 run 的 `train.log` loss 逐步一致。

## 6. 结果结构与汇总

默认 `<results-root>` 是
`results/01-analyse/01-fine-grained-offload/02-activation-offload-on-dense-and-expert-model`。
一级目录是实验名，即脚本名去掉 `run_` 前缀，标明这批结果回答的是哪个问题：

| 一级目录 | 对应脚本 | 回答的问题 |
| --- | --- | --- |
| `4gpu_expert_activation_offload_blocking` | `run_4gpu_expert_activation_offload_blocking.sh` | `--sync-with-sm` 是否消除 HybridEP `.item()` 阻塞 |
| `4gpu_compare_activation_offload_influence_on_dense_and_expert` | `run_4gpu_compare_…sh` | 小规模裁剪模型下 offload 对 dense / expert 的影响 |
| `8gpu_compare_activation_offload_influence_on_dense_and_expert` | `run_8gpu_compare_…sh` | 完整 9B / 128 expert 的 35B-A3B、FSDP1、EP=8 下的影响 |
| `16gpu_compare_activation_offload_influence_on_dense_and_expert` | `run_16gpu_compare_…sh` | 完整模型、FSDP1、EP=16 下的影响 |
| `32gpu_compare_activation_offload_influence_on_dense_and_expert` | `run_32gpu_compare_…sh` | 完整模型、DDP、PP=4、EP=8 下的影响 |

```text
<results-root>/<experiment>/<model>-<dense|expert>/gpu_<n>-<dtype>-pp_<p>-ep_<e>-mbs_<m>-gbs_<g>-<dispatcher>-<run-name>/<run-time>/
├── config.json          # 规模、并行度、offload 列表、sync 模式（0 号节点写）
├── command.txt
├── train.log
├── gpu_utilization.json # 每步 TFlops/GPU
├── gpu_memory/          # nvidia-smi 显存采样
├── summary.json         # status=0 表示成功（0 号节点写）
└── rank<N>/             # 其他节点的 train.log、gpu_utilization.json、gpu_memory/、profile/
```

```bash
# 只汇总一个实验，报告写进 <results-root>/<experiment>/
uv run --no-sync python $E/analyse_offload_results.py --experiment 16gpu_compare_activation_offload_influence_on_dense_and_expert
```

- 分组 key：`experiment, model, model_type, dtype, gpu_count, pp, ep, gbs, mbs, dispatcher, sharding,
  hybridep_sync_with_sm`。不同实验、规模、dense/expert、sync 模式互不配对。
- Megatron 在最后一个 rank 打印吞吐；PP 多机时这些数据在最后一个节点的 `rank<N>/` 下，分析脚本
  会自动选取样本最多的那份。
- 没有 baseline 的组（例如 blocking 核验）会被跳过并在 stderr 提示。
- 输出 `offload-summary.csv` 和 `offload-samples.csv`。

## 7. 验收与解读

1. `summary.json.status == 0`，`train.log` 无 NaN/Inf、NCCL timeout、OOM、Hydra 配置错误。
2. 先看 `offload-samples.csv` 的方差，再判断 5% 量级的差异是否真实。
3. 显存收益不在 CSV 里，读每个 run 的 `gpu_memory/`；offload 的收益是 HBM，代价是吞吐降幅。

## 8. 已知限制

- SM copy 依赖 Triton，不能与 CUDA graph capture 同时使用；本目录所有 case 都是 `cuda_graph_impl=none`。
- **Megatron-FSDP 与 PP>1 不兼容**：32 卡组是首次组合，已在 setup 阶段确认失败——`_get_dp_tp_mesh`
  用全局 `world_size` 展开 device mesh，但 `dp_cp` 是 per-pipeline-stage 组，PP=4 时报
  `EinopsError: Shape mismatch, 32 != 8`。因此 32 卡组固定传 `--sharding none` 走 DDP；若将来
  上游修好 mesh helper，可以去掉这个参数恢复 FSDP。
- Qwen3.5-122B-A10B 只有 bf16 recipe，32 卡 expert 不支持 `--dtype mxfp8`。
- recipe 会通过 `AutoConfig.from_pretrained` 读取 HF 配置，DLC 节点需要能访问 HF 或已缓存配置。
- 多机 nsys 只采集全局 rank 0–3（`profiling.profile_ranks=[0,1,2,3]`）。
