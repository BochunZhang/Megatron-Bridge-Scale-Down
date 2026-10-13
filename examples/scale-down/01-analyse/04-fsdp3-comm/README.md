# Qwen3.5 FSDP3 通信实验

这个目录使用 Bridge 的 `scripts/training/run_recipe.py` 运行 Qwen3.5 文本模型的 Megatron FSDP3（`optim_grads_params`）训练。GB200 recipe 保留完整 Hugging Face 模型结构，并以 FSDP1 作为默认 recipe；实验启动器在命令行阶段裁剪层数和 MoE experts。

支持三个模型：

| short name | Hugging Face model | recipe |
| --- | --- | --- |
| `397b` | `Qwen/Qwen3.5-397B-A17B` | `qwen35_text_397b_a17b_pretrain_4gpu_gb200_bf16_fsdp1_config` |
| `122b` | `Qwen/Qwen3.5-122B-A10B` | `qwen35_text_122b_a10b_pretrain_4gpu_gb200_bf16_fsdp1_config` |
| `27b` | `Qwen/Qwen3.5-27B` | `qwen35_text_27b_pretrain_4gpu_gb200_bf16_fsdp1_config` |

默认运行 6 个代表性 case，每个 case 分别以每 iter 1 个和 4 个 micro-batch 运行，共 12 次（MBS=1、4 卡），用于观察 FSDP3 通信与计算的占比，以及 dense / MoE、短序列 / 长序列 / 长序列 + CP 对通信的影响。GBS 按 `micro-batch 数 × MBS × DP` 逐 case 推算：

| case | 模型 | seq | CP | DP | EP | GBS（1 / 4 个 micro-batch） | 每 micro-batch 每 rank token | 观察点 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D1 | 27b dense | 4096 | 1 | 4 | 1 | 4 / 16 | 4096 | 计算最少，FSDP 通信占比上限 |
| D2 | 27b dense | 32768 | 1 | 4 | 1 | 4 / 16 | 32768 | FSDP 通信量同 D1、计算约 8 倍以上，FSDP 通信是否被隐藏 |
| D3 | 27b dense | 32768 | 4 | 1 | 1 | 1 / 4 | 8192 | FSDP 通信量同 D2，每 rank 计算约为 D2 的 1/4，并新增 CP 通信 |
| M1 | 397b MoE | 4096 | 1 | 4 | 4 | 4 / 16 | 4096 | 同 D1，增加 EP all-to-all |
| M2 | 397b MoE | 32768 | 1 | 4 | 4 | 4 / 16 | 32768 | 同 D2 |
| M3 | 397b MoE | 32768 | 4 | 1 | 4 | 1 / 4 | 8192 | 同 D3，CP 与 EP 共用同一组 4 卡 |

同一 case 的两组 micro-batch 数用于区分每个 micro-batch 内的 FSDP 通信与每 iter 只发生一次的开销（optimizer step、首层 all-gather / 末层 reduce-scatter 等无法被重叠的部分）。PP=1 时激活按 micro-batch 释放，micro-batch 数不影响峰值显存。

FSDP3 每个 micro-batch 都要 all-gather 参数并 reduce-scatter 梯度，单次通信量与 seq、MBS 无关，因此计算强度主要由“每 micro-batch 每 rank token 数 = MBS × seq / CP”决定，MBS 维度与 seq 维度基本重复，默认固定 MBS=1。122b 与 397b 结构相同、只是更小，默认不跑。

传入 `--model`、`--seq`、`--cp`、`--mbs` 任一参数时改为笛卡尔积矩阵：未指定的维度取完整默认值（模型 `397b,122b,27b`、CP `1,4`、seq `4096,16384,32768`、MBS `1,2,4`），`--model all` 即完整 54 个组合，每个组合仍按 1、4 个 micro-batch 各跑一次（共 108 次）。`--num-microbatches` 可修改 micro-batch 数，`--global-batch-size N` 改为固定 GBS（此时忽略 `--num-microbatches`）。

所有模型裁剪到 8 层，MoE 模型裁剪到 64 个 routed experts 并固定 EP=4（dense 模型只允许 EP=1）；recipe 本身不会写入这些裁剪值。默认使用 BF16 和 Nsight Systems；`--profile none` 可关闭 profiler，`--dtype mxfp8` 会在同一个 BF16 recipe 上覆盖 MXFP8 字段，不会选择额外 recipe。

启动器相对 recipe 的主要改动：

| 项目 | recipe 默认（FSDP1） | 本实验（FSDP3） |
| --- | --- | --- |
| `ddp.data_parallel_sharding_strategy` | `optim` | `optim_grads_params` |
| grad reduce / param gather overlap | 关闭（`comm_overlap.overlap_*=false`） | 开启，`ddp.*` 与 `comm_overlap.*` 同时覆盖 |
| `ddp.average_in_collective` | `true` | `false` |
| CUDA graph | Transformer Engine（`attn` 等模块） | 关闭（`cuda_graph_impl=none`） |
| MoE EP | 1 | 4（EP 组与 DP×CP 共用同一组 4 卡） |
| MoE dispatcher | `alltoall` | `alltoall` |

recipe 显式设置了 `comm_overlap.overlap_grad_reduce/overlap_param_gather=false`，`CommOverlapConfig.setup()` 会在运行时把它们写回 `ddp`；因此只覆盖 `ddp.overlap_*` 不会生效，启动器同时覆盖了 `comm_overlap.overlap_*`。

## 执行步骤

以下命令都从 Megatron Bridge 仓库根目录执行。

默认启动器使用 4 张 GPU、BF16 和 Nsight Systems。运行前需要准备好 Bridge 的 uv 环境、CUDA/Transformer Engine，以及可用或已缓存的 Qwen3.5 配置；脚本只加载 Hugging Face 配置，不加载模型权重。若设备没有 `nsys`，请显式添加 `--profile none`。

建议先检查命令：

```bash
bash examples/scale-down/01-analyse/04-fsdp3-comm/benchmark_fsdp3.sh --dry-run
```

确认命令后运行 6 个代表性 case：

```bash
bash examples/scale-down/01-analyse/04-fsdp3-comm/benchmark_fsdp3.sh
```

所有 case 在启动第一个之前整体校验（CP 整除、`seq % (2*CP)`，固定 GBS 时还检查 GBS 整除 MBS×DP）。单个 case 失败（例如长序列 OOM）会记录下来并继续下一个 case，结束时打印失败列表并以非 0 退出；`Ctrl-C` 会中止整个矩阵。

中断后可用同一个 `RUN_DATE` 加 `--resume` 续跑：已成功（`run_info.txt` 含 `status=0`）的 case 跳过，失败或未完成的旧结果被重命名为 `nodeN.attempt-<时间>` 后重跑：

```bash
RUN_DATE=261009-120000 bash examples/scale-down/01-analyse/04-fsdp3-comm/benchmark_fsdp3.sh --resume
```

多节点续跑时，每个节点只根据自己的 `nodeN/` 目录判断是否跳过；请确保各节点的结果状态一致，否则跳过与重跑的节点会在 rendezvous 处等待超时。

也可以先运行一个小 case 验证环境：

```bash
bash examples/scale-down/01-analyse/04-fsdp3-comm/benchmark_fsdp3.sh \
  --model 27b --cp 1 --seq 4096 --mbs 1 --profile none
```

自定义矩阵，或运行完整矩阵：

```bash
bash examples/scale-down/01-analyse/04-fsdp3-comm/benchmark_fsdp3.sh \
  --model 397b --cp 4 --seq 4096,16384 --mbs 1,2 --nsys
bash examples/scale-down/01-analyse/04-fsdp3-comm/benchmark_fsdp3.sh --model all
```

`--dtype mxfp8`、`--train-iters`、`--global-batch-size`、`--gpus`、`--nproc-per-node`、`--nnodes`、`--node-rank`、`--master-addr` 和 `--master-port` 可以覆盖默认值。多节点 DLC 各节点使用相同的矩阵和 `RUN_DATE`，并设置 `MASTER_ADDR`、`NODE_RANK` 等分布式参数。

例如 4 节点、每节点 4 卡时，在每个节点执行同一命令，只修改 `--node-rank`；`--master-addr` 填写 rank 0 节点地址：

```bash
RUN_DATE=261009-130000 bash examples/scale-down/01-analyse/04-fsdp3-comm/benchmark_fsdp3.sh \
  --model 27b --cp 1 --seq 4096 --mbs 1 --profile none \
  --gpus 16 --nproc-per-node 4 --nnodes 4 \
  --node-rank 0 --master-addr <rank0-host> --master-port 29501
```

其他节点将 `--node-rank 0` 改为 `1`、`2`、`3`。多节点 DLC 不会由脚本设置 `GLOO_SOCKET_IFNAME`。

TP、PP 固定为 1，MoE 的 EP 固定为 4，FSDP3 在全部 DP×CP rank 上分片。MoE case 的 GPU 数必须是 4 的倍数。默认按 micro-batch 数推算 GBS，任意卡数都满足整除；若用 `--global-batch-size` 固定 GBS，它必须是 `MBS × GPUs / CP` 的倍数。

单个 case 也可以直接调用：

```bash
bash examples/scale-down/01-analyse/04-fsdp3-comm/run_pretrain_fsdp3.sh \
  --model 27b \
  --recipe qwen35_text_27b_pretrain_4gpu_gb200_bf16_fsdp1_config \
  --seq-length 16384 --cp 1 --micro-batch-size 1 --num-layers 8
```

训练命令通过 `torch.distributed.run --no-python` 让每个 worker 执行 `numarun python ...`。脚本依次检查目标设备上的 `numarun`、仓库 `.cache/numarun`（需要 `numactl`）、`numactl --localalloc` 的本地 NUMA 内存策略，都不可用时才取消绑定。四 GPU 单节点运行会设置 `GLOO_SOCKET_IFNAME=eth0`；多节点 DLC 不注入该变量。

结果目录保持为 `results/01-analyse/04-fsdp3-comm/`。每个 case 使用以下格式，并按节点保存：

```text
results/01-analyse/04-fsdp3-comm/model_397b-fsdp_3-mbs_1-seq_4096-cp_1-ep_4-gbs_4-gpus_4-profile_nsys-date_261009-120000/node0/
```

目录中包含 `launch.json`、`command.txt`、`run_info.txt` 和 `train.log`；启用 `--nsys` 时还会生成该节点每个 local rank 的 nsys 报告。`config.yaml` 只由全局 rank 0 写出，因此多节点时只出现在 `node0/`。

## Nsys

`benchmark_fsdp3.sh` 默认 `--profile nsys`（单 case 脚本 `run_pretrain_fsdp3.sh` 默认 `none`）。每个 torchrun worker 由 `nsys profile --capture-range=cudaProfilerApi` 包裹，Bridge 在 `profile_step_start`（默认 5）调用 `cudaProfilerStart`，在 step 计数到 `profile_step_end`（默认 8）时停止，因此每份报告覆盖第 5–7 三个完整 iteration。每个 local rank 生成一份 `nsys-rank<LOCAL_RANK>.nsys-rep`。

- trace 范围为 `cuda,nvtx`：NCCL（FSDP all-gather / reduce-scatter、CP p2p、EP all-to-all）以 CUDA kernel 形式出现在 GPU 时间线上，可以直接与计算 kernel 对比。
- 启用 nsys 时 Bridge 会进入 `torch.autograd.profiler.emit_nvtx()`，为每个 autograd op 打 NVTX range，会增加 host 开销；比较 iteration 绝对时间时，用 `--profile none` 的 `train.log` 作为基准。
- FSDP3 下各 rank 对称，通常打开 `nsys-rank0` 即可；EP / CP case 如需看负载不均衡再对比其他 rank。
