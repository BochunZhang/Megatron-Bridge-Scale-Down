# Qwen3.5 / Qwen3.8 FSDP3 通信实验

这个目录使用 Bridge 的 `scripts/training/run_recipe.py` 运行 Qwen3.5 / Qwen3.8 文本模型的 Megatron FSDP3（`optim_grads_params`）训练。GB200 recipe 保留完整 Hugging Face 模型结构，并以 FSDP1 作为默认 recipe；实验启动器在命令行阶段裁剪层数和 MoE experts。

支持四个模型：

| short name | Hugging Face model | recipe |
| --- | --- | --- |
| `397b` | `Qwen/Qwen3.5-397B-A17B` | `qwen35_text_397b_a17b_pretrain_4gpu_gb200_bf16_fsdp1_config` |
| `122b` | `Qwen/Qwen3.5-122B-A10B` | `qwen35_text_122b_a10b_pretrain_4gpu_gb200_bf16_fsdp1_config` |
| `27b` | `Qwen/Qwen3.5-27B` | `qwen35_text_27b_pretrain_4gpu_gb200_bf16_fsdp1_config` |
| `2p4t`（别名 `2.4t`） | `Qwen/Qwen3.8-2.4T-A95B` | `qwen38_text_2p4t_a95b_pretrain_4gpu_gb200_bf16_fsdp1_config` |

Qwen3.8-2.4T-A95B 的 HF 配置是纯文本的 `Qwen3_5MoeForCausalLM`（没有 `text_config`），与 Qwen3.5 MoE 同构。它的 `output_gate_type="swish"` 在 Megatron 中没有对应实现：Megatron 的 attention output gate 是 sigmoid，参数形状相同，只差一个逐元素乘法，对通信与显存分析没有影响。

## 实验矩阵

默认运行 9 个代表性 case，每个 case 分别以 CUDA graph 关闭和开启各跑一次（共 18 次，另加 2p4t 减半 experts 的重试）。每 iter 固定 4 个 micro-batch（MBS=1、4 卡），GBS 按 `micro-batch 数 × MBS × DP` 逐 case 推算，DP=4 时 GBS=16：

| case | 模型 | seq | CP | DP | EP | GBS | 每 micro-batch 每 rank token | 观察点 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D1 / D2 / D3 | 27b dense | 4096 / 32768 / 16384 | 1 / 1 / 4 | 4 / 4 / 1 | 1 | 16 / 16 / 4 | 4096 / 32768 / 4096 | 计算最少时 FSDP 通信占比上限；长序列是否隐藏 FSDP 通信；CP 通信 |
| M1 / M2 / M3 | 397b MoE | 同上 | 同上 | 同上 | 4 | 同上 | 同上 | 同上，增加 HybridEP dispatch / combine |
| T1 / T2 / T3 | 2p4t MoE（裁剪 experts） | 同上 | 同上 | 同上 | 4 | 同上 | 同上 | hidden=8192 的大层：单层 dense 参数是 397b 的约 3.7 倍 |

模型裁剪规则：

- 所有模型裁剪到 **4 层**（`linear_attention_freq=4`，即 3 层 GatedDeltaNet + 1 层 gated attention），并开启 **1 层 MTP**（`mtp_num_layers=1`，MTP 层是 gated attention + MoE / dense MLP，与 embedding、output layer 共享权重）。
- `397b` **不裁剪 experts**（512 个 routed experts，EP=4，每 rank 128 个）。
- `2p4t` 从 512 个 experts 开始；某个 expert 数下任一 CUDA graph 模式在 `train.log` 中出现 OOM（`out of memory` / `OutOfMemoryError`），就把 experts 减半并重跑该 case 的所有 CUDA graph 模式，直到能训练（下限为 top-k=10 以上且能被 EP=4 整除，即 16）。同一 case 的 graph 关 / 开因此总是使用同一个 expert 数；每个 expert 数先跑显存更高的 graph 开。搜索结果追加到 `results/01-analyse/04-fsdp3-comm/expert_search-date_<RUN_DATE>-node<N>.tsv`（`experts=` 为最终值，`tried=` 记录 `512:oom,256:oom,...`）。`--experts-2p4t N` 固定 expert 数、不搜索，`--max-experts-2p4t N` 修改搜索起点。多节点时各节点只能看到自己的日志，减半决策可能不一致，因此多节点 2p4t 必须用 `--experts-2p4t`。

按 BF16 混合精度 + Adam、FSDP3 每参数 18 B（bf16 参数 2 + fp32 main grad 4 + fp32 master/m/v 12，`grad_reduce_in_fp32=true`）、EP=4 时 expert 也是 4 路切分估算，每卡模型状态：

| 模型（4 层 + MTP） | dense 参数 | routed expert 参数 | 每卡模型状态 |
| --- | --- | --- | --- |
| 27b | 约 4.5B | — | 约 19 GiB |
| 397b（512 experts） | 约 2.7B | 约 32.2B | 约 147 GiB |
| 2p4t，128 experts | 约 6.6B | 约 32.2B | 约 163 GiB |
| 2p4t，64 experts | 约 6.6B | 约 16.1B | 约 95 GiB |

GB200 每卡约 185 GiB。另外还有 FSDP double buffer（两个 FSDP unit 的 bf16 参数与 fp32 梯度桶）、激活、248K 词表的 fp32 logits（seq 4096 时每份约 3.8 GiB，MTP 再加一份）以及 CUDA graph 内存池。因此 397b 在 4096 上可以放下但余量不大，32768 很可能 OOM（按要求 397b 不裁剪 experts，OOM 只记为失败 case）；2p4t 预计停在 64 experts。

FSDP3 每个 micro-batch 都要 all-gather 参数并 reduce-scatter 梯度，单次通信量与 seq、MBS 无关，因此计算强度主要由“每 micro-batch 每 rank token 数 = MBS × seq / CP”决定，默认固定 MBS=1。4 个 micro-batch 保证一个 iter 内既有每个 micro-batch 重复的 FSDP 通信，也有每 iter 只发生一次的开销（optimizer step、首层 all-gather / 末层 reduce-scatter 等无法被重叠的部分）。122b 与 397b 结构相同、只是更小，默认不跑。

传入 `--model`、`--seq`、`--cp`、`--mbs` 任一参数时改为笛卡尔积矩阵：未指定的维度取完整默认值（模型 `397b,27b,2p4t`、CP `1,4`、seq `4096,16384,32768`、MBS `1,2,4`），`--model all` 加入 122b。`--cuda-graph off,on` 选择 CUDA graph 模式，`--num-layers`、`--mtp-num-layers` 修改裁剪，`--num-microbatches` 修改 micro-batch 数，`--global-batch-size N` 改为固定 GBS（此时忽略 `--num-microbatches`）。

## CUDA graph

| `--cuda-graph` | dense（27b） | MoE（397b / 2p4t） |
| --- | --- | --- |
| `off` | `cuda_graph_impl=none` | `cuda_graph_impl=none` |
| `on` | TE per-layer graph，`cuda_graph_modules=[attn,mlp]`：attention 与 MLP 计算全部进 graph | TE per-layer graph，`cuda_graph_modules=[attn,moe_router,moe_preprocess]`（部分 graph）：attention、router、shared expert、preprocess 进 graph；HybridEP dispatch / combine 与 routed experts 保持 eager |

开启时同时设置 `cuda_graph_warmup_steps=3` 与 `ddp.megatron_fsdp_cuda_graph_mode=true`。后者让 Megatron-FSDP 在 optimizer step 后不解除梯度引用，保证 graph replay 使用的地址有效；Megatron-LM 在 FSDP + CUDA graph 时会自动打开它，Bridge 不会，所以启动器显式覆盖。recipe 本身已启用 `fsdp_double_buffer` 与 `megatron_fsdp_max_pool_double_buffer`，后者让每个 FSDP unit 的参数桶落在固定地址，是 graph replay 读取 all-gather 结果的前提。TE graph 也覆盖 MTP 层。

per-layer graph 只包住模块内部计算：FSDP 的 pre-forward / post-backward hook、all-gather、reduce-scatter 都在 graph 之外 eager 执行，因此 FSDP 通信在两种模式下相同，对比的是 graph 消除的 CPU launch 开销是否改变计算与通信的重叠。graph 在第 3 个 iteration 后捕获，nsys 默认采集 step 7，已处于 replay 阶段。首次运行开启模式时，建议对比同一 case 关 / 开的 `train.log` loss 是否一致。Bridge 只支持 `transformer_engine` 按模块 graph 与 `full_iteration` graph，`full_iteration` 会把 FSDP 通信一起捕获，不在本实验范围内。

## FSDP all-gather / reduce-scatter 的层标注

上游 Megatron-LM（`megatron/core/distributed/fsdp/src/megatron_fsdp/`）**不会**标注 all-gather / reduce-scatter 属于哪个 layer：

- `MegatronFSDP.forward` 只有一个 `record_function("CustomFSDP.forward")`；
- `AllGatherPipeline.all_gather_params` / `async_bucket_gather` 与 `GradReducePipeline._bucket_group_gradient_reduce` 在 `_coalescing_manager` 内发起 `all_gather_into_tensor` / `reduce_scatter_tensor`，没有任何 NVTX；
- 预取（prefetch）让 layer N+1 的 all-gather 在 layer N 的 forward hook 中发起，按模块 NVTX range 归属会算错；
- 带模块名的 `MFSDP <name> forward/backward` range 只存在于 `experimental/module.py`，标注的是计算而不是通信，常规 FSDP 路径也不使用它；
- `emit_nvtx(record_shapes=True)` 只能给 `c10d::*_coalesced_` / `record_param_comms` 带上 bucket 的 tensor 形状，无法反推层号。

本项目使用的 Megatron-LM（`core_v0.19.0_scale_down` 分支）在 Megatron-FSDP 中直接加入了带层号的 NVTX range：

- `megatron_fsdp/utils.py`：`configure_nvtx_profiling()` 开关与 `nvtx_range()`（未开启时不构造 label，无额外开销）；`megatron.core.utils.configure_nvtx_profiling()` 会同步转发到这个开关，因此 Bridge 的 `profiling.nvtx_ranges=true`（启动器在 nsys 模式下设置）或 Megatron-LM 的 `--nvtx-ranges` 即可开启；
- `param_and_grad_buffer.py`：`ParamAndGradBuffer.bucket_nvtx_label()` 由 `param_to_name` 生成 bucket label；`AllGatherPipeline.all_gather_params` 与 `GradReducePipeline._bucket_group_gradient_reduce` 把 `_coalescing_manager` 与 `nvtx_range` 放在同一个 `with` 中，range 正好包住 coalesced NCCL kernel 的发起（发生在 manager 退出时）；
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

## 执行步骤

以下命令都从 Megatron Bridge 仓库根目录执行。

默认启动器使用 4 张 GPU、BF16 和 Nsight Systems。运行前需要准备好 Bridge 的 uv 环境、CUDA/Transformer Engine，以及可用或已缓存的 Hugging Face 配置与 tokenizer；脚本只加载配置，不加载模型权重。若设备没有 `nsys`，请显式添加 `--profile none`。

建议先检查命令：

```bash
bash examples/scale-down/01-analyse/04-fsdp3-comm/benchmark_fsdp3.sh --dry-run
```

确认命令后运行默认矩阵：

```bash
bash examples/scale-down/01-analyse/04-fsdp3-comm/benchmark_fsdp3.sh
```

所有 case 在启动第一个之前整体校验（CP 整除、`seq % (2*CP)`，固定 GBS 时还检查 GBS 整除 MBS×DP）。单个 case 失败（例如长序列 OOM）会记录下来并继续下一个 case，结束时打印失败列表并以非 0 退出；`Ctrl-C` 会中止整个矩阵。

中断后可用同一个 `RUN_DATE` 加 `--resume` 续跑：已成功（`run_info.txt` 含 `status=0`）的 case 跳过；2p4t 已 OOM 的 expert 数直接沿用结论、继续减半，不再重跑；其他失败或未完成的旧结果被重命名为 `nodeN.attempt-<时间>` 后重跑：

```bash
RUN_DATE=261031-120000 bash examples/scale-down/01-analyse/04-fsdp3-comm/benchmark_fsdp3.sh --resume
```

多节点续跑时，每个节点只根据自己的 `nodeN/` 目录判断是否跳过；请确保各节点的结果状态一致，否则跳过与重跑的节点会在 rendezvous 处等待超时。

先跑小 case 验证环境，或只跑一个模型、一种 graph 模式：

```bash
bash examples/scale-down/01-analyse/04-fsdp3-comm/benchmark_fsdp3.sh \
  --model 27b --cp 1 --seq 4096 --mbs 1 --cuda-graph off --profile none
bash examples/scale-down/01-analyse/04-fsdp3-comm/benchmark_fsdp3.sh \
  --model 2p4t --cp 1 --seq 4096 --mbs 1 --max-experts-2p4t 128
```

`--dtype mxfp8`、`--train-iters`、`--global-batch-size`、`--gpus`、`--nproc-per-node`、`--nnodes`、`--node-rank`、`--master-addr` 和 `--master-port` 可以覆盖默认值。多节点 DLC 各节点使用相同的矩阵和 `RUN_DATE`，并设置 `MASTER_ADDR`、`NODE_RANK` 等分布式参数，例如 4 节点、每节点 4 卡：

```bash
RUN_DATE=261031-130000 bash examples/scale-down/01-analyse/04-fsdp3-comm/benchmark_fsdp3.sh \
  --model 2p4t --cp 1 --seq 4096 --mbs 1 --experts-2p4t 256 --profile none \
  --gpus 16 --nproc-per-node 4 --nnodes 4 \
  --node-rank 0 --master-addr <rank0-host> --master-port 29501
```

其他节点将 `--node-rank 0` 改为 `1`、`2`、`3`。多节点 DLC 不会由脚本设置 `GLOO_SOCKET_IFNAME`。

TP、PP 固定为 1，MoE 的 EP 固定为 4，FSDP3 在全部 DP×CP rank 上分片。MoE case 的 GPU 数必须是 4 的倍数。

单个 case 也可以直接调用（`--print-result-dir` 只打印结果目录）：

```bash
bash examples/scale-down/01-analyse/04-fsdp3-comm/run_pretrain_fsdp3.sh \
  --model 2p4t \
  --recipe qwen38_text_2p4t_a95b_pretrain_4gpu_gb200_bf16_fsdp1_config \
  --seq-length 4096 --cp 1 --micro-batch-size 1 \
  --num-layers 4 --mtp-num-layers 1 --num-experts 64 --moe-layer-freq '[1,1,1,1]' \
  --cuda-graph on --profile nsys
```

训练命令通过 `torch.distributed.run --no-python` 让每个 worker 执行 `numarun python ...`。脚本依次检查目标设备上的 `numarun`、仓库 `.cache/numarun`（需要 `numactl`）、`numactl --localalloc` 的本地 NUMA 内存策略，都不可用时才取消绑定。四 GPU 单节点运行会设置 `GLOO_SOCKET_IFNAME=eth0`；多节点 DLC 不注入该变量。

结果目录保持为 `results/01-analyse/04-fsdp3-comm/`，case 名包含层数、MTP 层数、routed expert 数（dense 为 0，未裁剪为 `full`）和 CUDA graph 模式，并按节点保存：

```text
results/01-analyse/04-fsdp3-comm/model_2p4t-fsdp_3-layers_4-mtp_1-experts_64-cg_on-mbs_1-seq_4096-cp_1-ep_4-disp_hybridep-gbs_16-gpus_4-profile_nsys-date_261031-120000/node0/
```

目录中包含 `launch.json`、`command.txt`、`run_info.txt` 和 `train.log`；启用 `--nsys` 时还会生成该节点每个 rank 的 nsys 报告。`config.yaml` 只由全局 rank 0 写出，因此多节点时只出现在 `node0/`。

## Nsys

`benchmark_fsdp3.sh` 默认 `--profile nsys`（单 case 脚本 `run_pretrain_fsdp3.sh` 默认 `none`）。每个 torchrun worker 由 `nsys profile --capture-range=cudaProfilerApi` 包裹，Bridge 在 `profile_step_start`（默认 7）调用 `cudaProfilerStart`，在 step 计数到 `profile_step_end`（默认 8）时停止，因此每份报告只覆盖 step 7（0-based，对应日志里的 iteration 8）这一个完整 iteration（含 4 个 micro-batch）。每个 rank 生成一份报告，文件名为 `<qwen35_397b|qwen35_27b|qwen38_2p4t>-experts_<E>-cg_<off|on>-cp_<CP>-ep_<EP>-mbs_<MBS>-gbs_<GBS>-seq_<N>k-rank_<RANK>.nsys-rep`，其中 `%q{RANK}` 由 nsys 按各 worker 的 `RANK` 环境变量展开成全局 rank。

- trace 范围为 `cuda,nvtx`：NCCL（FSDP all-gather / reduce-scatter、CP p2p）与 HybridEP dispatch / combine kernel 以 CUDA kernel 形式出现在 GPU 时间线上，可以直接与计算 kernel 对比。CUDA graph 开启时，graph 内的 kernel 以 graph launch 形式出现。
- 启用 nsys 时启动器设置 `profiling.record_shapes=true`，Bridge 进入 `torch.autograd.profiler.emit_nvtx(record_shapes=True)`，每个 op 的 NVTX range 带输入形状；会增加 host 开销，比较 iteration 绝对时间时，用 `--profile none` 的 `train.log` 作为基准。
- FSDP all-gather / reduce-scatter 的层归属看上一节的 `FSDP AG ...` / `FSDP RS ...` range。
- NCCL 自身在 `NCCL` NVTX domain 为每次 `ncclAllGather` / `ncclReduceScatter` 打 range，payload 含消息字节数和 comm hash；启动器在 nsys 模式下设置 `NCCL_NVTX_DISABLE=0`。nsys 不会在 kernel 上直接显示带宽，需要用 recipe 关联 NVTX 和 kernel，例如 `nsys recipe nccl_sum --input <report>.nsys-rep`。
- FSDP3 下各 rank 对称，通常打开 `rank_0` 的报告即可；EP / CP case 如需看负载不均衡再对比其他 rank。
