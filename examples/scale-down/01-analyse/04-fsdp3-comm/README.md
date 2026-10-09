# Qwen3.5 FSDP3 通信实验

这个目录使用 Bridge 的 `scripts/training/run_recipe.py` 运行 Qwen3.5 文本模型的 Megatron FSDP3（`optim_grads_params`）训练。GB200 recipe 保留完整 Hugging Face 模型结构，并以 FSDP1 作为默认 recipe；实验启动器在命令行阶段裁剪层数和 MoE experts。

支持三个模型：

| short name | Hugging Face model | recipe |
| --- | --- | --- |
| `397b` | `Qwen/Qwen3.5-397B-A17B` | `qwen35_text_397b_a17b_pretrain_4gpu_gb200_bf16_fsdp1_config` |
| `122b` | `Qwen/Qwen3.5-122B-A10B` | `qwen35_text_122b_a10b_pretrain_4gpu_gb200_bf16_fsdp1_config` |
| `27b` | `Qwen/Qwen3.5-27B` | `qwen35_text_27b_pretrain_4gpu_gb200_bf16_fsdp1_config` |

默认矩阵遍历 CP=`1,4`、sequence length=`4096,16384,32768`、MBS=`1,2,4`，GBS=`32`。三个模型默认裁剪到 8 层，两个 MoE 模型裁剪到 64 个 routed experts；recipe 本身不会写入这些裁剪值。默认使用 BF16 和 Nsight Systems；`--profile none` 可关闭 profiler，`--dtype mxfp8` 会在同一个 BF16 recipe 上覆盖 MXFP8 字段，不会选择额外 recipe。

## 执行步骤

以下命令都从仓库根目录执行：

```bash
cd /Users/zhangbochun/Desktop/AI-Infra/Megatron/Megatron-Scale-Down/megatron-bridge
```

默认启动器使用 4 张 GPU、BF16 和 Nsight Systems。运行前需要准备好 Bridge 的 uv 环境、CUDA/Transformer Engine，以及可用或已缓存的 Qwen3.5 配置；脚本只加载 Hugging Face 配置，不加载模型权重。若设备没有 `nsys`，请显式添加 `--profile none`。

建议先检查命令：

```bash
bash examples/scale-down/01-analyse/04-fsdp3-comm/benchmark_fsdp3.sh --dry-run
```

确认命令后运行完整矩阵：

```bash
bash examples/scale-down/01-analyse/04-fsdp3-comm/benchmark_fsdp3.sh
```

也可以先运行一个小 case 验证环境：

```bash
bash examples/scale-down/01-analyse/04-fsdp3-comm/benchmark_fsdp3.sh \
  --model 27b --cp 1 --seq 4096 --mbs 1 --profile none
```

选择模型、序列长度、MBS 或 profiler：

```bash
bash examples/scale-down/01-analyse/04-fsdp3-comm/benchmark_fsdp3.sh \
  --model 397b --cp 4 --seq 4096,16384 --mbs 1,2 --nsys
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
results/01-analyse/04-fsdp3-comm/model_397b-fsdp_3-mbs_1-seq_4096-cp_1-gbs_32-gpus_4-profile_nsys-date_261009-120000/node0/
```

目录中包含 `launch.json`、`command.txt`、`run_info.txt`、`train.log` 和 recipe 写出的 `config.yaml`；启用 `--nsys` 时还会生成每个 rank 的 nsys 报告。
