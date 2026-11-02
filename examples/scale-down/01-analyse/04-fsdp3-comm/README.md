# Qwen3.5 / Qwen3.8 FSDP3 通信实验

这个目录使用 Bridge 的 `scripts/training/run_recipe.py` 运行 Qwen3.5 / Qwen3.8 文本模型的 Megatron FSDP3（`optim_grads_params`）训练。GB200 recipe 保留完整 Hugging Face 模型结构，并以 FSDP1 作为默认 recipe；实验启动器在命令行阶段裁剪层数和 MoE experts。

支持四个模型：

| short name | Hugging Face model | recipe |
| --- | --- | --- |
| `397b` | `Qwen/Qwen3.5-397B-A17B` | `qwen35_text_397b_a17b_pretrain_4gpu_gb200_bf16_fsdp1_config` |
| `122b` | `Qwen/Qwen3.5-122B-A10B` | `qwen35_text_122b_a10b_pretrain_4gpu_gb200_bf16_fsdp1_config` |
| `27b` | `Qwen/Qwen3.5-27B` | `qwen35_text_27b_pretrain_4gpu_gb200_bf16_fsdp1_config` |
| `2.4t`（别名 `2p4t`） | `Qwen/Qwen3.8-2.4T-A95B` | `qwen38_text_2p4t_a95b_pretrain_4gpu_gb200_bf16_fsdp1_config` |

Qwen3.8-2.4T-A95B 的 HF 配置是纯文本的 `Qwen3_5MoeForCausalLM`（没有 `text_config`），与 Qwen3.5 MoE 同构。它的 `output_gate_type="swish"` 在 Megatron 中没有对应实现：Megatron 的 attention output gate 是 sigmoid，参数形状相同，只差一个逐元素乘法，对通信与显存分析没有影响。

## 脚本

目录只保留两个脚本：

- `run_pretrain_fsdp3.sh`：启动一个 case，`--model` 选择模型与对应的 GB200 recipe（`--recipe` 可覆盖）；
- `benchmark_fsdp3.sh`：对列表选项取笛卡尔积，逐个调用 `run_pretrain_fsdp3.sh`。启动第一个 case 前，所有 case 会先用 `run_pretrain_fsdp3.sh --print-result-dir` 做一遍和真实启动相同的校验；单个 case 失败（例如 OOM）只记录下来，sweep 继续。

### 选项

| 选项 | 取值 | 说明 |
| --- | --- | --- |
| `--model` | `27b`、`397b`、`2.4t`（别名 `2p4t`）、`122b`，可逗号分隔 | sweep 必填 |
| `--fsdp` | `ddp`、`1`、`2`、`3`，可逗号分隔 | 数据并行方式，见下表；sweep 默认 `3,1`，单 case 默认 `3` |
| `--tp` | N，可逗号分隔 | Tensor parallel；N>1 时自动打开 sequence parallel（`model.sequence_parallel=true`）。默认 1 |
| `--cp` | N，可逗号分隔 | Context parallel，`cp_comm_type=p2p`。默认 1 |
| `--cuda-graph` | `off`、`on`，可逗号分隔 | 默认 `off`，含义见「CUDA graph」 |
| `--seq` | N，可逗号分隔 | sweep 默认 `256,1024,4096` |
| `--mbs` | N，可逗号分隔 | 默认 1 |
| `--gbs` | N，可逗号分隔 | 固定 GBS，必须是 MBS×DP 的倍数；不传时 GBS = `--num-microbatches`（默认 4）× MBS × DP |
| `--experts` | N 或 `full`，可逗号分隔 | MoE 的 routed expert 数，须为 EP=4 的倍数且 ≥ top-k=10；`full` 保留 recipe 的完整数量。默认 `64,32`，27b 忽略 |
| `--num-layers` | N | 默认 27b 8 层、MoE 4 层（每层 MoE） |
| `--full` / `--linear` | — | 所有层都用 gated attention（`linear_attention_freq=1`）/ 都用 GatedDeltaNet（`linear_attention_freq=[1,...,1]`）；两者互斥。不传时为默认的每 4 层 3 层 GatedDeltaNet + 1 层 gated attention（`linear_attention_freq=4`） |
| `--mtp-num-layers` | N | 默认 0 |
| `--dispatcher` | `hybridep`、`alltoall` | MoE dispatcher，默认 `hybridep` |
| `--force-balance` | — | 见「强制均衡路由」 |
| `--dtype` / `--mxfp8` | `bf16`、`mxfp8` | 默认 BF16 |
| `--profile` / `--nsys` | `none`、`nsys` | sweep 默认 `nsys`，单 case 默认 `none` |

DP = GPU 数 / (TP × CP)；MoE 的 EP=4 与 TP×DP×CP mesh 共用同一组 GPU，只要求 EP 整除 GPU 数（PP 固定为 1）。TP 组由连续 rank 组成，TP 必须整除每节点 GPU 数，保证 TP 通信不出节点；开启 SP 时每个 CP rank 的序列片段（seq / CP）还必须能被 TP 整除。

数据并行方式：

| `--fsdp` | sharding strategy | 通信 |
| --- | --- | --- |
| `3` | `optim_grads_params` | 参数、梯度、optimizer state 都切分；每个 micro-batch 做 fwd AG、bwd AG、RS，预取并重叠 |
| `2` | `optim_grads` | 切分梯度和 optimizer state，参数保持完整；每个 micro-batch 的 backward 都做梯度 RS（Megatron-FSDP 要求重叠），optimizer step 后做一次同步的参数 AG |
| `1` | `optim`（recipe 默认） | 只切 optimizer state；整个 iteration 只在最后做一次梯度 RS、optimizer step 后做一次参数 AG，不重叠 |
| `ddp` | `no_shard` + MCore DDP | 分布式优化器；梯度按 bucket 在 backward 中 RS，参数 AG 与下一次 forward 重叠。见「DDP：逐层 backward 时间」 |

FSDP 每个 micro-batch 的单次通信量与 seq、MBS 无关，因此计算强度主要取决于“每个 micro-batch 每个 rank 的 token 数 = MBS × seq / CP”（SP 再把非 attention 部分按 TP 切分）。默认的短序列（256 / 1024 / 4096）让每层计算盖不住 FSDP 通信，用来观察 FSDP3 / FSDP2 / FSDP1 / DDP 之间的差异；每个 iteration 4 个 micro-batch，这样一个 iteration 里既有每个 micro-batch 都重复的通信，也有每个 iteration 只发生一次的开销。

### 默认矩阵与裁剪

| 模型 | 层数 | experts | 默认 case 数 |
| --- | --- | --- | --- |
| `27b` dense | 8（6 GatedDeltaNet + 2 gated attention） | — | seq 3 × FSDP 2 = 6 |
| `397b` MoE，EP=4 | 4（3 + 1，每层 MoE） | `64,32` | experts 2 × seq 3 × FSDP 2 = 12 |
| `2.4t` MoE，EP=4 | 同上 | `64,32` | 12 |

4 卡 EP=4 时，expert 的数据并行组大小为 1，routed expert 参数在任何 FSDP 级别下都不切分，也没有 FSDP 通信；各 FSDP 级别之间的差别只来自 dense 参数（attention / GDN、shared expert、router、embedding、LM head）。expert 数只改变每卡的 routed expert 参数量，不改变 EP dispatch / combine 的通信量。2.4t 的 hidden=8192，单层 dense 参数约为 397b 的 3.7 倍，短序列下通信更难被盖住。

按 BF16 混合精度 + Adam 估算，FSDP3 每参数 18 B（bf16 参数 2 + fp32 main grad 4 + fp32 master/m/v 12，`grad_reduce_in_fp32=true`），EP=4 时 expert 也按 4 路切分，每卡模型状态如下：

| 模型（4 层 + 1 层 MTP） | dense 参数 | routed expert 参数 | 每卡模型状态 |
| --- | --- | --- | --- |
| 27b | 约 4.5B | — | 约 19 GiB |
| 397b（512 experts） | 约 2.7B | 约 32.2B | 约 147 GiB |
| 2.4t，128 experts | 约 6.6B | 约 32.2B | 约 163 GiB |
| 2.4t，64 experts | 约 6.6B | 约 16.1B | 约 95 GiB |

GB200 每卡约 185 GiB。此外还要加上 FSDP double buffer、激活、248K 词表的 fp32 logits（seq 4096 时每份约 3.8 GiB），以及 CUDA graph 的内存池。FSDP1 / FSDP2 / DDP 不切分参数，2.4t 64 experts 的 dense 部分每卡约多 30 GiB。TP 会把 dense 参数和激活再按 TP 切分。

### 示例

```bash
S=examples/scale-down/01-analyse/04-fsdp3-comm/benchmark_fsdp3.sh
bash $S --model 27b --dry-run
bash $S --model 27b,397b,2.4t
# FSDP 四种方式对比，只跑一个 shape
bash $S --model 2.4t --experts 64 --seq 1024 --fsdp ddp,1,2,3
# TP+SP、CP 单独开启
bash $S --model 27b --fsdp 3 --seq 4096 --tp 1,2,4
bash $S --model 397b --experts 64 --fsdp 3 --seq 16384 --cp 1,2,4
# MBS / GBS
bash $S --model 27b --fsdp 3 --seq 1024 --mbs 1,2 --gbs 16,32
# 全 gated attention / 全 GatedDeltaNet
bash $S --model 397b --experts 64 --seq 1024 --full
bash $S --model 397b --experts 64 --seq 1024 --linear
# MXFP8
bash $S --model 27b --mxfp8
```

结果目录名中，非默认设置会追加标记，默认设置不加任何标记，因此以前的结果名仍可用 `--resume` 续跑：MXFP8 为 `-mxfp8`，强制均衡为 `-fb`，`--full` / `--linear` 为层数后的 `-attn_full` / `-attn_linear`，TP>1 为 CP 前的 `-tp_<N>`。例如 `model_27b-fsdp_3-mxfp8-layers_8-attn_full-mtp_0-experts_0-cg_off-mbs_1-seq_4096-tp_2-cp_1-...`。

### DDP：逐层 backward 时间（27b / 397b / 2.4t）

`--fsdp ddp`，把 Megatron-FSDP 换成 MCore `DistributedDataParallel`，用来在 CUDA graph 关 / 开下观察每个 layer 的 backward 时间。FSDP 下只能用 full-iteration graph（见下文「CUDA graph」），整轮 forward + backward 是一个 graph，看不到层边界；DDP 可以用 TE 的按层 graph。

| 项 | `--fsdp ddp` 的覆盖 |
| --- | --- |
| 数据并行 | `dist.use_megatron_fsdp=false`、`ddp.use_megatron_fsdp=false`、`ddp.data_parallel_sharding_strategy=no_shard`（该字段只有 FSDP 使用） |
| 优化器 | `ddp.use_distributed_optimizer=true`：参数与梯度在每卡完整保留，只切 optimizer state |
| 通信重叠 | `ddp.overlap_grad_reduce=true`、`ddp.overlap_param_gather=true`（`comm_overlap.*` 同步覆盖）：梯度按 bucket 在 backward 中 reduce-scatter，参数 all-gather 与下一轮 forward 重叠 |
| checkpoint | `checkpoint.ckpt_format=torch_dist`（`fsdp_dtensor` 只支持 Megatron-FSDP；本实验不保存） |
| 命名 | 结果目录 `model_<m>-ddp-...`，nsys 报告 `<qwen35_27b\|qwen35_397b\|qwen38_2.4t>-ddp-...`；`launch.json` 中 `fsdp` 为 `null`、`data_parallel` 为 `ddp` |

`--cuda-graph on` 时设置 `cuda_graph_impl=transformer_engine`、`cuda_graph_warmup_steps=3`、`use_te_rng_tracker=true`、`rng.te_rng_tracker=true` 与 `NCCL_GRAPH_REGISTER=0`（`TECudaGraphHelper` 在 expandable segments 下要求关闭）。每层每个 micro-batch 分别有一个 forward graph 和一个 backward graph：

- dense（27b）：`cuda_graph_modules=[attn,mlp]`，整层进 graph；
- MoE（397b / 2.4t）：`cuda_graph_modules=[attn,moe_router,moe_preprocess]`，attention、router、shared expert 与 preprocess 进 graph，dispatch、routed experts、combine 保持 eager。MoE 仍是 dropless，不需要容量因子、paged stash 或 TE op fuser，expert 走与关闭模式相同的非融合 `TEGroupedMLP`，因此不受上文 cuBLAS 13.3 的限制，hybridep 与 alltoall 都可以用；
- MXFP8 + alltoall 时 `moe_router_padding_for_fp8` 让 alltoall 的 preprocess 把 token 数拷回 host，MCore 不允许 graph 它，启动器改用 `[attn,moe_router]`；
- 不需要关闭 NaN 检查，recipe 的 `check_for_nan_in_loss` / `check_for_nan_in_grad` 保持开启；
- DDP 的参数 all-gather 是 forward pre-hook：Bridge 在 capture 时摘掉它，capture 后改为每层的 manual hook（`train.py` 的 `cuda_graph_set_manual_hooks`）。

显存：DDP 每卡保留完整的 bf16 参数与 fp32 梯度，dense 部分约 9 B/参数（2 + 4 + 12/4），EP=4 时 routed expert 的数据并行组为 1，与 FSDP 相同。2.4t 64 experts 每卡模型状态约 120 GiB，与 FSDP1 相当；397b 与 27b 余量更大。

逐层 backward 时间的读法：

- **graph 关**：nsys 模式下每层有 `L<i> bwd` NVTX range（见下文「Layer 的 forward / backward 标注」），直接投影到 GPU 即可得到每层 backward 时间；DDP 的梯度 reduce-scatter 在独立 stream 上，与后续层的 backward 重叠。
- **graph 开**：MCore 的 TE graph 路径在 capture 与 replay 时都绕过 `TransformerLayer.forward()`，直接调用 `_forward_attention` / `_forward_mlp`（`megatron/core/transformer/transformer_layer.py` 的 `_te_cuda_graph_capture` / `_te_cuda_graph_replay`），而 `L<i> fwd` / `L<i> bwd` 是在 `forward()` 里注入的，因此 capture 之后（nsys 采集的 step 7）**没有这两类 range**。此时按 graph 读：nsys 默认 `--cuda-graph-trace=graph`，每次 graph launch 在 GPU 时间线上是一个整体。27b 每层的 backward 就是该层的 backward graph；MoE 层的 backward 是 graph 部分（attention / router / shared expert）加上前后 eager 的 combine、routed experts、dispatch 的 backward kernel。
- 比较 graph 关 / 开的 iteration 时间时加 `--profile none`，以排除 nsys 与 `record_shapes` 的 host 开销。

```bash
bash examples/scale-down/01-analyse/04-fsdp3-comm/benchmark_fsdp3.sh --model 27b,397b,2.4t --fsdp ddp --cuda-graph off,on
# 与 FSDP 放在同一矩阵里（FSDP 的 graph 开需要 cuBLAS >= 13.3 编译的 TE）
bash examples/scale-down/01-analyse/04-fsdp3-comm/benchmark_fsdp3.sh --model 2.4t --fsdp 3,2,1,ddp --experts 64 --seq 1024 --cuda-graph off,on
```

第一条命令的矩阵：27b 为 seq `256,1024,4096` × graph `off,on` 共 6 次；397b / 2.4t 为 experts `64,32` × seq `256,1024,4096` × graph `off,on` 各 12 次。

尚未在 GPU 上验证的点：GatedDeltaNet 层在 TE graph 中的 capture；`--force-balance` 的随机 logits 在 router graph replay 时是否重新采样（与 FSDP full-iteration graph 的同一问题）。

### 强制均衡路由（`--force-balance`）

两个脚本都支持 `--force-balance`，它设置 MCore 的 `model.moe_router_force_load_balancing=true`：`TopKRouter.forward` 在 router GEMM 之后把 logits 换成每 rank 独立的 N(0,1) 随机数（`apply_random_logits`，反向为直通），再照常 top-k。每个 token 因此均匀随机地选 k 个 expert，各 expert 与各 EP rank 的 token 数在统计上相等；router GEMM、dispatch / combine 与 expert 计算量不变。

- 只用于性能测试：路由与输入无关，loss 没有意义；
- 只作用于 MoE case，`benchmark_fsdp3.sh` 中的 27b 不受影响，单 case 启动器对 27b 报错；
- 结果目录与 nsys 报告名在 dtype 标记后加 `-fb`，例如 `model_2.4t-fsdp_3-fb-layers_4-...` 与 `qwen38_2.4t-fsdp_3-fb-experts_64-....nsys-rep`；不加该选项时命名不变；
- 默认路由是 `global_aux_loss`，系数取 HF config 的 `router_aux_loss_coef`（缺省时为 0）。10 个 iter 的 mock 数据下，路由主要由随机初始化的 router 决定，EP rank 之间可能不均；`--force-balance` 用来排除这部分干扰；
- 与 `--cuda-graph on`（router 在 full-iteration graph 内）同时使用时，replay 是否每次重新采样随机 logits 尚未验证。

```bash
bash examples/scale-down/01-analyse/04-fsdp3-comm/benchmark_fsdp3.sh --model 397b,2.4t --force-balance
```

## CUDA graph

`--cuda-graph on` 的含义取决于数据并行方式（`--fsdp`）：

| `--cuda-graph` | `--fsdp` | dense（27b） | MoE（397b / 2.4t） |
| --- | --- | --- | --- |
| `off` | 任意 | `cuda_graph_impl=none` | `cuda_graph_impl=none` |
| `on` | `3` / `1` | `cuda_graph_impl=full_iteration`：整个 forward + backward（含 FSDP all-gather / reduce-scatter）一个 graph，optimizer step 保持 eager | 同左，另加 `moe_expert_rank_capacity_factor=1.5`、`moe_paged_stash=true`、`use_transformer_engine_op_fuser=true`；只支持 `--dispatcher hybridep` |
| `on` | `ddp` | `cuda_graph_impl=transformer_engine`，`cuda_graph_modules=[attn,mlp]`：每层整层一个 graph | `cuda_graph_impl=transformer_engine`，`cuda_graph_modules=[attn,moe_router,moe_preprocess]`：dispatch / routed experts / combine 保持 eager；hybridep 与 alltoall 都可用 |

DDP 的 TE 按层 graph 见下文「DDP：逐层 backward 时间」。下面先说明 FSDP 的 full-iteration graph。

FSDP 开启时同时设置：

- `cuda_graph_warmup_steps=3`；
- `ddp.megatron_fsdp_cuda_graph_mode=true`：Megatron-FSDP 在 optimizer step 后不解除梯度引用，保证 replay 使用的地址有效；
- `ddp.fsdp_all_gather_in_start_param_sync=false`：optimizer step 里的 `start_param_sync()` 不再发起 all-gather，否则 graph 会 wait 一个在 capture 之外 stream 上的 kernel；
- `rerun_state_machine.check_for_nan_in_loss=false`、`ddp.check_for_nan_in_grad=false`：整轮 graph 内无法检查 NaN；
- `NCCL_GRAPH_REGISTER=0`：graph 内含 NCCL 集合通信，`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 时 NCCL 注册 graph 内存可能非法访存。

Megatron-LM 在 FSDP + full-iteration graph 时会自动设置前两项 ddp 选项，Bridge 不会，所以启动器显式覆盖。recipe 本身已启用 `fsdp_double_buffer` 与 `megatron_fsdp_max_pool_double_buffer`，后者让每个 FSDP unit 的参数桶落在固定地址，是 replay 的前提。

不用按层 graph 的原因：Megatron-FSDP 在每个 `TransformerLayer` 上注册 unshard / reshard / post-backward 的 module hook。TE `make_graphed_callables` 要求被 graph 的 module 上没有任何 hook（报 `Modules must not have hooks registered`），MCore 的 `local` 按层 graph 在 replay 时只放行 MCore DDP 的 pre-forward hook；Bridge 也只允许 `local` 用于 inference。因此 FSDP 下只有 full-iteration graph 可用。

MoE 的额外设置：dropless MoE 每步把各 expert 的 token 数拷回 host、shape 随路由变化，无法进 graph。`moe_expert_rank_capacity_factor=1.5` 给每个 EP rank 固定 1.5 倍均衡负载的 token 预算，让 HybridEP 的 buffer 与 expert 输入 shape 固定（需要 TE op fuser 的 grouped MLP；启动器同时导出 `NVTE_CUTEDSL_FUSED_GROUPED_MLP=1`，否则 MCore 不构建 SwiGLU 的 fused grouped MLP，报 `Fused GroupedMLP is not supported for this configuration`）；`moe_paged_stash` 把 routed expert 反向所需的激活放进固定的分页 buffer。某一步超出预算时，`PagedStashRunner` 检测到 overflow 并以 eager 重跑该步。因此：

- 开启模式的 MoE expert 计算走 op fuser 路径（对齐到 256 token），与关闭模式的 expert kernel 不同，关 / 开对比的不只是 graph 本身；
- 不加 `--force-balance` 时，随机初始化的 router 可能让某些 rank 超出预算而触发 eager 重跑，`train.log` 中需要检查；
- graph 让中间 buffer 常驻，显存高于关闭模式；2.4t 的 expert 减半搜索会把这部分算进去；
- op fuser 的 `te.ops.GroupedLinear` 在 Blackwell + BF16/MXFP8 上固定走 cuBLASLt grouped GEMM（GPU 端读取每组 shape），TE 必须用 **cuBLAS ≥ 13.3（CUDA ≥ 13.2）** 编译。用 CUDA 13.0 编译的 TE 会在第一个 warmup iteration 报 `nvte_get_grouped_gemm_setup_workspace_size requires cuBLAS 13.3+, but compile-time cuBLAS version is 130000`，日志最上面看到的 `Attempted to pop NVTX range ...` 只是这个异常引起的连锁错误。

graph 在第 3 个 iteration 后捕获，nsys 默认采集 step 7，已处于 replay 阶段；FSDP 通信与计算 kernel 都以 graph launch 形式出现。首次运行开启模式时，建议对比同一 case 关 / 开的 `train.log` loss 是否接近。

测试 FSDP3 下 CUDA graph 的收益，用同一 case 的 graph 关 / 开对比 step 时间与 nsys 上暴露的通信，例如 2.4t 短序列（每 micro-batch 计算最少、最可能受 CPU launch 限制）：

```bash
bash examples/scale-down/01-analyse/04-fsdp3-comm/benchmark_fsdp3.sh \
  --model 2.4t --fsdp 3 --experts 64 --seq 256,1024,4096 --cuda-graph off,on
```

比较 iteration 时间时加 `--profile none`，以排除 nsys 与 `record_shapes` 的 host 开销。FSDP1 / FSDP2 加 CUDA graph（`--fsdp 1|2 --cuda-graph on`）也能启动，但尚未验证；它们的参数 all-gather 在 eager 的 optimizer step 中同步完成，不进 graph。TP+SP、CP 与 full-iteration graph 组合时，TP / SP 的 all-gather / reduce-scatter 与 CP 的 p2p 都是 NCCL kernel，随 forward / backward 一起被捕获，代码中没有禁止，但同样尚未在 GPU 上验证，建议先用 `--cuda-graph off` 确认 eager 正常。

## FSDP all-gather / reduce-scatter 的层标注

上游 Megatron-LM（`megatron/core/distributed/fsdp/src/megatron_fsdp/`）**不会**标注 all-gather / reduce-scatter 属于哪个 layer：

- `MegatronFSDP.forward` 只有一个 `record_function("CustomFSDP.forward")`；
- `AllGatherPipeline.all_gather_params` / `async_bucket_gather` 与 `GradReducePipeline._bucket_group_gradient_reduce` 在 `_coalescing_manager` 内发起 `all_gather_into_tensor` / `reduce_scatter_tensor`，没有任何 NVTX；
- 预取（prefetch）让 layer N+1 的 all-gather 在 layer N 的 forward hook 中发起，按模块 NVTX range 归属会算错；
- 带模块名的 `MFSDP <name> forward/backward` range 只存在于 `experimental/module.py`，标注的是计算而不是通信，常规 FSDP 路径也不使用它；
- `emit_nvtx(record_shapes=True)` 只能给 `c10d::*_coalesced_` / `record_param_comms` 带上 bucket 的 tensor 形状，无法反推层号。

本项目使用的 Megatron-LM（`core_v0.19.0_scale_down` 分支）在 Megatron-FSDP 中直接加入了带层号的 NVTX range：

- `megatron_fsdp/utils.py`：`configure_nvtx_profiling()` 开关与 `nvtx_range()`（未开启时不构造 label，无额外开销）；`megatron.core.utils.configure_nvtx_profiling()` 会同步转发到这个开关，因此 Bridge 的 `profiling.nvtx_ranges=true`（启动器在 nsys 模式下设置）或 Megatron-LM 的 `--nvtx-ranges` 即可开启；
- `param_and_grad_buffer.py`：`ParamAndGradBuffer` 在构建 parameter group（bucket）时用 `param_to_name` 一次性算出每个 bucket 的层、类型和参数量，存入 `ParameterGroup.nvtx_label`；运行时 `AllGatherPipeline.all_gather_nvtx_label()` / `GradReducePipeline.grad_reduce_nvtx_label()` 只按 bucket 组合查表，组合的拼接串在首次出现时生成并缓存；`AllGatherPipeline.all_gather_params` 与 `GradReducePipeline._bucket_group_gradient_reduce` 把 `_coalescing_manager` 与 `nvtx_range` 放在同一个 `with` 中，range 正好包住 coalesced NCCL kernel 的发起（发生在 manager 退出时）；
- `megatron_fsdp.py`：非 overlap 路径 `start_param_sync` 中逐 bucket 的 all-gather 也带同样的 label。

label 例如：

```text
FSDP AG fwd L1 dense 132.7M | FSDP AG fwd prefetch L2 dense 132.7M + L2 expert 1610.6M
FSDP AG bwd prefetch mtp0 dense 153.1M
FSDP RS L3 dense 119.6M
```

- `AG` / `RS`（`no_shard` 时为 `AR`）：all-gather / reduce-scatter；`fwd` / `bwd`：参数为哪个 pass 收集；
- `prefetch`：该 bucket 不是当前请求模块的参数，而是被提前收集的后续模块；
- `L<i>` / `mtp<i>` / `embedding` / `output_layer` / `final_norm`：bucket 中参数所属的 decoder 层、MTP 层或非层参数；
- `dense` / `expert`：是否为 routed expert 参数（EP 下 expert 参数单独成桶），数字是 bucket 未切分参数量；
- 同一次 coalesced 发起的多个 bucket 以 `+` 连接，请求与预取以 `|` 分隔。nsys 会把 range 投影到对应 NCCL kernel（`nsys stats --report nvtx_gpu_proj_trace`，或在 GUI 中看 NVTX 行下的 GPU 投影）。

`--profile none` 时不开启 NVTX，也就没有这些 range。

## Layer 的 forward / backward 标注

`TransformerLayer` 的 NVTX range 与上面的 FSDP label 使用同一套 0-based 层号（即参数名 `decoder.layers.<i>` / `mtp.layers.<i>` 中的 `<i>`），名称在构造 layer 时算好：

```text
L0 fwd / L0 bwd ... L3 fwd / L3 bwd      decoder 层
mtp0 fwd / mtp0 bwd                       MTP 层（开启 MTP 时）
```

- `L<i> fwd`：`TransformerLayer.forward` 的 push / pop；
- `L<i> bwd`：按 Megatron-FSDP backward hook 的方式注入。layer 输出上挂一个 tensor hook，输出梯度到达（该层 backward 开始）时 push；layer 输入经过一个恒等 autograd Function，输入梯度算完（该层 backward 结束）时 pop。range 在 autograd 线程上，包住该层全部 backward kernel 的发起；full recompute 时重算的 `L<i> fwd` 嵌套在 `L<i> bwd` 内。
- 只在 NVTX 开启、梯度开启且输入 `requires_grad` 时注入，`--profile none` 与推理没有额外开销。
- PP>1 时层号是全局层号，而 FSDP label 来自本 stage 的参数名（本地层号），两者只在 PP=1（本实验）时一致。
- `cuda_graph_impl=transformer_engine`（`--fsdp ddp --cuda-graph on`）capture 之后不再调用 `forward()`，这两类 range 都不会出现；`full_iteration` graph replay 时也不会执行 Python 代码，同样没有。

## 执行步骤

以下命令都从 Megatron Bridge 仓库根目录执行。

默认启动器使用 4 张 GPU、BF16 和 Nsight Systems。运行前需要准备好 Bridge 的 uv 环境、CUDA/Transformer Engine，以及可用或已缓存的 Hugging Face 配置与 tokenizer；脚本只加载配置，不加载模型权重。若设备没有 `nsys`，请显式添加 `--profile none`。

建议先检查命令：

```bash
bash examples/scale-down/01-analyse/04-fsdp3-comm/benchmark_fsdp3.sh --model 27b --dry-run
```

所有 case 在启动第一个之前整体校验。单个 case 失败（例如长序列 OOM）会记录下来，然后继续下一个 case；结束时打印失败列表，并以非 0 退出。`Ctrl-C` 会中止整个矩阵。

中断后可以用同一个 `RUN_DATE` 加 `--resume` 续跑：已成功（`run_info.txt` 含 `status=0`）的 case 跳过；其他失败或未完成的旧结果先重命名为 `nodeN.attempt-<时间>`，再重跑：

```bash
RUN_DATE=261031-120000 bash examples/scale-down/01-analyse/04-fsdp3-comm/benchmark_fsdp3.sh --model 27b --resume
```

先跑一个小 case 验证环境：

```bash
bash examples/scale-down/01-analyse/04-fsdp3-comm/benchmark_fsdp3.sh --model 27b --fsdp 3 --seq 4096 --profile none
```

`--train-iters`、`--gpus`、`--nproc-per-node`、`--nnodes`、`--node-rank`、`--master-addr` 和 `--master-port` 可以覆盖默认值。多节点 DLC 时，各节点使用相同的矩阵和 `RUN_DATE`；每个节点只根据自己的 `nodeN/` 目录判断是否跳过，请确保各节点的结果状态一致，否则跳过的节点和重跑的节点会在 rendezvous 处等待超时。例如 4 节点、每节点 4 卡：

```bash
RUN_DATE=261031-130000 bash examples/scale-down/01-analyse/04-fsdp3-comm/benchmark_fsdp3.sh \
  --model 2.4t --seq 4096 --experts 256 --fsdp 3 --profile none \
  --gpus 16 --nproc-per-node 4 --nnodes 4 \
  --node-rank 0 --master-addr <rank0-host> --master-port 29501
```

其他节点把 `--node-rank 0` 改为 `1`、`2`、`3`。多节点 DLC 不会由脚本设置 `GLOO_SOCKET_IFNAME`。

PP 固定为 1，MoE 的 EP 固定为 4；FSDP 在全部 DP×CP rank 上分片，TP 是 DTensor mesh 的 `tp` 维。MoE case 的 GPU 数必须是 4 的倍数。

单个 case 也可以直接调用（`--print-result-dir` 校验 case 后只打印结果目录）：

```bash
bash examples/scale-down/01-analyse/04-fsdp3-comm/run_pretrain_fsdp3.sh \
  --model 2.4t --seq 4096 --tp 2 --cp 1 --mbs 1 --gbs 8 \
  --num-layers 4 --num-experts 64 --cuda-graph on --profile nsys
```

训练命令通过 `torch.distributed.run --no-python` 让每个 worker 执行 `numarun python ...`。脚本依次检查目标设备上的 `numarun`、仓库 `.cache/numarun`（需要 `numactl`）、`numactl --localalloc` 的本地 NUMA 内存策略，都不可用时才取消绑定。四 GPU 单节点运行会设置 `GLOO_SOCKET_IFNAME=eth0`；多节点 DLC 不注入该变量。

结果目录保持为 `results/01-analyse/04-fsdp3-comm/`，case 名包含层数、MTP 层数、routed expert 数（dense 为 0，未裁剪为 `full`）和 CUDA graph 模式，并按节点保存：

```text
results/01-analyse/04-fsdp3-comm/model_2.4t-fsdp_3-layers_4-mtp_0-experts_64-cg_on-mbs_1-seq_4096-tp_2-cp_1-ep_4-disp_hybridep-gbs_8-gpus_4-profile_nsys-date_261031-120000/node0/
```

目录中包含 `launch.json`、`command.txt`、`run_info.txt` 和 `train.log`；启用 `--nsys` 时，`node0/` 下还会有唯一一份 nsys 报告。`config.yaml` 只由全局 rank 0 写出，因此多节点时只出现在 `node0/`。

## Nsys

`benchmark_fsdp3.sh` 默认 `--profile nsys`（单 case 脚本 `run_pretrain_fsdp3.sh` 默认 `none`）。`nsys profile --capture-range=cudaProfilerApi` 包在 torchrun 外面，跟踪 torchrun 拉起的全部 worker 进程；`profiling.profile_ranks` 为节点 0 的 4 个 rank（0..nproc-per-node-1，即一个 GB200 tray 的 4 张 GPU），Bridge 在 `profile_step_start`（默认 7）调用 `cudaProfilerStart`，在 step 计数到 `profile_step_end`（默认 8）时停止，因此每份报告只覆盖 step 7（0-based，对应日志里的 iteration 8）这一个完整 iteration（含 4 个 micro-batch）。只有节点 0 的命令外包 nsys，其他节点不录制，因此每个 case 只产生一份报告，同一时间线上按进程列出这 4 个 rank。文件名为 `<qwen35_397b|qwen35_27b|qwen38_2.4t>-<ddp|fsdp_1|fsdp_2|fsdp_3>[-mxfp8][-fb][-attn_<full|linear>]-experts_<E>-cg_<off|on>[-tp_<TP>]-cp_<CP>-ep_<EP>-mbs_<MBS>-gbs_<GBS>-seq_<N>k.nsys-rep`。

- trace 范围为 `cuda,nvtx`：NCCL（FSDP all-gather / reduce-scatter、TP / SP 集合通信、CP p2p）与 HybridEP dispatch / combine kernel 以 CUDA kernel 形式出现在 GPU 时间线上，可以直接与计算 kernel 对比。CUDA graph 开启时，整轮 forward + backward（含 FSDP 通信）以 graph launch 形式出现。
- 启用 nsys 时启动器设置 `profiling.record_shapes=true`，Bridge 进入 `torch.autograd.profiler.emit_nvtx(record_shapes=True)`，每个 op 的 NVTX range 带输入形状；会增加 host 开销，比较 iteration 绝对时间时，用 `--profile none` 的 `train.log` 作为基准。
- FSDP all-gather / reduce-scatter 的层归属看上一节的 `FSDP AG ...` / `FSDP RS ...` range。
- NCCL 自身在 `NCCL` NVTX domain 为每次 `ncclAllGather` / `ncclReduceScatter` 打 range，payload 含消息字节数和 comm hash；启动器在 nsys 模式下设置 `NCCL_NVTX_DISABLE=0`。nsys 不会在 kernel 上直接显示带宽，需要用 recipe 关联 NVTX 和 kernel，例如 `nsys recipe nccl_sum --input <report>.nsys-rep`。
- 4 个 rank 在同一份报告中，可以直接对比各 rank 的 FSDP AG / RS 与 TP / EP / CP 通信是否对齐、负载是否均衡。多节点时报告只包含节点 0 的 rank。
