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

通用启动器和原综合测试：

- `run_pretrain_fsdp3.sh`：启动一个 case，`--model` 选择模型与对应的 GB200 recipe（`--recipe` 可覆盖）；
- `benchmark_fsdp3.sh`：对列表选项取笛卡尔积，逐个调用 `run_pretrain_fsdp3.sh`。启动第一个 case 前，所有 case 会先用 `run_pretrain_fsdp3.sh --print-result-dir` 做一遍和真实启动相同的校验；单个 case 失败（例如 OOM）只记录下来，sweep 继续。
- `benchmark_fsdp3_suite.sh`：原 8 项综合矩阵，使用 GBS=8。

### FSDP3 offload 专项实验

以下五组在一台 4 卡 GB200 上运行。固定 BF16、full attention、GBS=16、27B 裁剪 8 层、2.4T 裁剪 4 层和 64 experts。默认 MBS=1，只有 `02_mbs.sh` 扫描 1、2、4。TP=CP=1 用 seq=4096，CP=4 或 TP=4 用 seq=16384。所有案例采集 Nsight Systems CUDA/NVTX trace，关闭 GPU metrics。

| 组 | 脚本 | 对照 |
| --- | --- | --- |
| `01-baseline` | [01_cuda_graph.sh](01-baseline/01_cuda_graph.sh)、[02_mbs.sh](01-baseline/02_mbs.sh)、[03_fsdp3.sh](01-baseline/03_fsdp3.sh) | FSDP3 eager/graph、固定拓扑扫 MBS、FSDP1/FSDP3。27B 两边用 full-iteration graph，2.4T 两边用原生 MoE partial graph。 |
| `02-comm-overlap` | [01_fsdp3_cp_overlap.sh](02-comm-overlap/01_fsdp3_cp_overlap.sh)、[02_ddp_cp_tp.sh](02-comm-overlap/02_ddp_cp_tp.sh) | FSDP3 在 CP=1/4 内分别比较 AG/RS 共用与独立 communicator。27B 用 full-iteration graph，2.4T 用 eager。DDP + graph 对比 CP4 与 TP4 的通信。 |
| `03-cpu-offload` | [01_weight_offload.sh](03-cpu-offload/01_weight_offload.sh)、[02_cpu_grad_optimizer.sh](03-cpu-offload/02_cpu_grad_optimizer.sh) | 两个模型都在 CP=1 和 CP=4 测权重回载 × AG/RS overlap。固定 CP=1、卸载组件和各模型的 graph 范围，逐项把梯度累加和 AdamW 移到 CPU。 |
| `04-nvme` | [01_dram_vs_nvme.sh](04-nvme/01_dram_vs_nvme.sh) | eager CPU 累加 + CPU AdamW，比较 DRAM 与 NVMe。默认用 `/tmp/megatron-fsdp3-nvme`，启动前检查默认目录所在设备。可用 `--nvme-path DIR` 或 `NVME_PATH` 指向节点的本地 NVMe 目录。 |
| `05-nccl` | [01_priority.sh](05-nccl/01_priority.sh)、[02_symmetric_multicast.sh](05-nccl/02_symmetric_multicast.sh)、[03_ag_pause.sh](05-nccl/03_ag_pause.sh) | 27B graph 下的默认/高优先级、AG 和 RS 独立 communicator × symmetric multicast、AG pause 1536/3072/4608 cycles。三脚本默认加载已有的 `nccl/nccl-2.32.3-cta-with-pause/build/lib/libnccl.so.2`。可用 `--nccl-lib FILE` 或 `NCCL_LIB` 覆盖。 |

从仓库根目录运行。默认结果根目录为 `results/01-analyse/04-fsdp3-comm/`。所有编号脚本使用 `<组名>/<小项目>/model_<配置>/node0/`。小项目名取脚本名去掉 `.sh` 并将下划线换成连字符。编号实验省略时间戳，`01-baseline` 省略 `-baseline`，其余组在 case 目录名中保留用于区分实验变体的标签。

编号脚本只检查小项目下的目标 case 目录。目标 `model_…/` 已存在就跳过，包括空目录和上次失败的目录。缺少目录的 case 会继续运行。需要复跑时可指定新的 `--output-dir DIR`。可用 `--model 27b` 限定模型，或用 `--dry-run` 查看完整计划。

用 `migrate_results.py` 迁移原先直接保存在组目录下的结果。脚本按当前测试矩阵匹配，完整迁移 case 目录及历史尝试。同一个旧 case 对应多个小项目时，每个目标都获得一份结果。若某个目标已有结果，该 case 的源目录与目标目录都保留，脚本报告冲突。未匹配的旧目录会列出来供检查。使用过自定义结果根目录时，给迁移脚本传同一个 `--output-dir DIR`。

```bash
study=examples/scale-down/01-analyse/04-fsdp3-comm
uv run --no-sync python "$study/migrate_results.py" --dry-run
uv run --no-sync python "$study/migrate_results.py"
bash "$study/01-baseline/01_cuda_graph.sh" --dry-run
bash "$study/01-baseline/01_cuda_graph.sh"
bash "$study/04-nvme/01_dram_vs_nvme.sh"
bash "$study/04-nvme/01_dram_vs_nvme.sh" --nvme-path /mnt/local-nvme/megatron-swap
MEGATRON_LM_DIR=/path/to/megatron-lm bash "$study/05-nccl/03_ag_pause.sh" --output-dir /path/to/new-results
```

4 卡 TP=4 时 dense FSDP 分片组只有 1 个 rank，因此 TP4 使用 DDP 观测 TP 通信。

`05-nccl` 的 split case 同时启用 `--overlap` 和 `--separate-rs`，需要包含[独立 RS 组支持](https://github.com/BochunZhang/Megatron-LM-Scale-Down/commit/bb5a346f93f6da398e16cb864a7736d376154583)的 Megatron-LM checkout。`MEGATRON_LM_DIR` 指向该 checkout。实验让 NCCL 自动选核，symmetric case 在首步完成 window 注册后才具备 MC 条件。第 7 步的 Nsight Systems trace 要核对 FSDP 参数 AG 和梯度 RS 分别使用 `AllGather_STMC` 和 `ReduceScatter_LDMC`，选到其他内核的结果不纳入 MC 对照。已有 pause 库按进程读取 pause cycles。若 trace 中其他通信也使用受该设置影响的 AG 内核，该轮不能归因于仅 FSDP AG 的 pause。

### 选项

| 选项 | 取值 | 说明 |
| --- | --- | --- |
| `--model` | `27b`、`397b`、`2.4t`（别名 `2p4t`）、`122b`，可逗号分隔 | sweep 必填 |
| `--fsdp` | `ddp`、`1`、`2`、`3`，可逗号分隔 | 数据并行方式，见下表；sweep 默认 `3,1`，单 case 默认 `3` |
| `--ddp-overlap` | — | DDP 在首个 micro-batch 的 forward 预取参数。默认在 optimizer step 内同步聚合参数，默认模式的结果名带 `-sync-ag` |
| `--overlap` | — | 见「AG / RS 通信组分离」；只作用于 FSDP case（sweep 中 `ddp` case 不受影响），名字中数据并行方式后加 `-overlap` |
| `--separate-rs` | — | 仅 FSDP3。梯度 RS 使用独立 DP communicator，结果名加 `-separate-rs` |
| `--symmetric` | — | 见「Symmetric memory」；只作用于 FSDP case（sweep 中 `ddp` case 不受影响），名字中加 `-symm`（与 `--overlap` 同用时为 `-overlap-symm`） |
| `--tp` | N，可逗号分隔 | Tensor parallel；N>1 时自动打开 sequence parallel（`model.sequence_parallel=true`）。默认 1 |
| `--tp-comm-overlap` | — | 为 TP > 1 的 case 设置 `comm_overlap.tp_comm_overlap=true`，结果目录和 nsys 报告名带 `-tp-overlap`。混合 sweep 中的 TP=1 case 作为基线，单 case 启动时要求 TP > 1 |
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
| `--offload` | `optimizer`、`weight`、`gradient`，可逗号组合 | 每个 case 卸载指定模块，目录和 nsys 报告名追加 `-offload_<模块>` |
| `--profile` / `--nsys` | `none`、`nsys` | sweep 默认 `nsys`，单 case 默认 `none` |
| `--metric [HZ]` | 10–200000 Hz | 启用 nsys 并采集 node 0 上所有 GPU 的硬件指标。默认 20000 Hz，可用 `--metric 10000` 指定频率。结果目录和报告名追加 `-metric_<HZ>` |
| `--merge` | — | nsys 不展开 CUDA graph，每次 graph launch 画成一个整体（`--cuda-graph-trace=graph`）；只对 nsys 且 `--cuda-graph on` 的 case 生效，在 graph 标记后追加 `-merge`。见「Nsys」 |

DP = GPU 数 / (TP × CP)；MoE 的 EP=4 与 TP×DP×CP mesh 共用同一组 GPU，只要求 EP 整除 GPU 数（PP 固定为 1）。TP 组由连续 rank 组成，TP 必须整除每节点 GPU 数，保证 TP 通信不出节点；开启 SP 时每个 CP rank 的序列片段（seq / CP）还必须能被 TP 整除。

数据并行方式：

| `--fsdp` | sharding strategy | 通信 |
| --- | --- | --- |
| `3` | `optim_grads_params` | 参数、梯度、optimizer state 都切分；每个 micro-batch 做 fwd AG、bwd AG、RS，预取并重叠 |
| `2` | `optim_grads` | 切分梯度和 optimizer state，参数保持完整；每个 micro-batch 的 backward 都做梯度 RS（Megatron-FSDP 要求重叠），optimizer step 后做一次同步的参数 AG |
| `1` | `optim`（recipe 默认） | 只切 optimizer state；整个 iteration 只在最后做一次梯度 RS、optimizer step 后做一次参数 AG，不重叠 |
| `ddp` | `no_shard` + MCore DDP | 分布式优化器。梯度按 bucket 在 backward 中 RS，参数 AG 默认在 optimizer step 内同步执行。见「DDP：逐层 backward 时间」 |

FSDP 每个 micro-batch 的单次通信量与 seq、MBS 无关，因此计算强度主要取决于“每个 micro-batch 每个 rank 的 token 数 = MBS × seq / CP”（SP 再把非 attention 部分按 TP 切分）。默认的短序列（256 / 1024 / 4096）让每层计算盖不住 FSDP 通信，用来观察 FSDP3 / FSDP2 / FSDP1 / DDP 之间的差异；每个 iteration 4 个 micro-batch，这样一个 iteration 里既有每个 micro-batch 都重复的通信，也有每个 iteration 只发生一次的开销。

### 默认矩阵与裁剪

| 模型 | 层数 | experts | 默认 case 数 |
| --- | --- | --- | --- |
| `27b` dense | 8（6 GatedDeltaNet + 2 gated attention） | — | seq 3 × FSDP 2 = 6 |
| `397b` MoE，EP=4 | 4（3 + 1，每层 MoE） | `64,32` | experts 2 × seq 3 × FSDP 2 = 12 |
| `2.4t` MoE，EP=4 | 同上 | `64,32` | 12 |

4 卡 EP=4 时，expert 的数据并行组大小为 1，routed expert 参数在任何 FSDP 级别下都不切分，也没有 FSDP 通信；各 FSDP 级别之间的差别只来自 dense 参数（attention / GDN、shared expert、router、embedding、LM head）。expert 数只改变每卡的 routed expert 参数量，不改变 EP dispatch / combine 的通信量。2.4t 的 hidden=8192，单层 dense 参数约为 397b 的 3.7 倍，短序列下通信更难被盖住。

按 BF16 混合精度 + Adam 估算，FSDP3 每参数 16 B（bf16 参数 2 + bf16 main grad 2 + fp32 master/m/v 12），EP=4 时 expert 也按 4 路切分，每卡模型状态如下：

| 模型（4 层 + 1 层 MTP） | dense 参数 | routed expert 参数 | 每卡模型状态 |
| --- | --- | --- | --- |
| 27b | 约 4.5B | — | 约 17 GiB |
| 397b（512 experts） | 约 2.7B | 约 32.2B | 约 130 GiB |
| 2.4t，128 experts | 约 6.6B | 约 32.2B | 约 145 GiB |
| 2.4t，64 experts | 约 6.6B | 约 16.1B | 约 85 GiB |

梯度精度：启动器传的 `grad_reduce_in_fp32=true` 只作用于 MCore DDP（`--fsdp ddp` 的梯度 buffer 与 reduce-scatter 都是 fp32）。Megatron-FSDP 不读这个字段，梯度 buffer 的 dtype 由 `ddp.megatron_fsdp_main_grads_dtype` 决定，默认 `None` 即跟随参数 dtype；Bridge 在包 FSDP 之前已用 `Float16Module` 把参数转成 bf16，所以 FSDP 的梯度累加、reduce-scatter（`megatron_fsdp_grad_comm_dtype=None` 时等于梯度 dtype）和参数 all-gather 都是 bf16，只有 main weight（`megatron_fsdp_main_params_dtype`，默认 fp32）与 Adam 状态是 fp32。

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
# FSDP3 CPU offload，四卡每步两个 microbatch
bash $S --model 27b --fsdp 3 --seq 1024 --num-layers 4 --gbs 8 \
  --profile none --offload optimizer,weight,gradient
# 全 gated attention / 全 GatedDeltaNet
bash $S --model 397b --experts 64 --seq 1024 --full
bash $S --model 397b --experts 64 --seq 1024 --linear
# MXFP8
bash $S --model 27b --mxfp8
# AG 独立 communicator + symmetric memory，观察 FSDP3 中预取 AG 与 RS 的重叠
bash $S --model 2.4t --experts 64 --seq 1024 --fsdp 3 --overlap --symmetric
```

结果目录名中，非默认设置会追加标记，默认设置不加任何标记，因此以前的结果名仍可用 `--resume` 续跑：MXFP8 为 `-mxfp8`，强制均衡为 `-fb`，`--full` / `--linear` 为层数后的 `-attn_full` / `-attn_linear`，TP>1 为 CP 前的 `-tp_<N>`。例如 `model_27b-fsdp_3-mxfp8-layers_8-attn_full-mtp_0-experts_0-cg_off-mbs_1-seq_4096-tp_2-cp_1-...`。

两个启动器都接受 `--offload`。它将所选模块传给 `ddp.fsdp_cpu_offload`，并将模块列表写入 `launch.json`。同一组模块采用固定顺序命名，例如 `...-offload_optimizer_weight_gradient/node0`。省略此选项可跑 baseline。运行环境需包含支持该字段的 Megatron-LM（`9b2c5ba3d` 或更新版本）。Offload 要求 Megatron-FSDP 并关闭 NCCL symmetric buffer。FSDP1 选择 `gradient` 时，启动器会开启梯度规约重叠。与 `--cuda-graph on` 同用时，需要 `--fsdp 3` 和支持该组合的 Megatron-LM 源码。

### CPU 梯度累加、CPU AdamW 和 NVMe

用本次修改的 Megatron-LM checkout 跑同一组 4 GPU、4 microbatch 的 mock-data 实验。以下命令从 Megatron-Bridge 根目录运行。

```bash
export MEGATRON_LM_DIR="$PWD/../megatron-lm"
launcher=examples/scale-down/01-analyse/04-fsdp3-comm/benchmark_fsdp3.sh
common=(--model 27b --fsdp 3 --num-layers 4 --seq 256 --num-microbatches 4 --profile nsys)

# 已有路径：GPU 累加和 Adam 更新，CPU 保存 partial sum 和 optimizer 数据
bash "$launcher" "${common[@]}" --offload optimizer,weight,gradient

# 同一组卸载组件，改在 CPU 累加和更新
bash "$launcher" "${common[@]}" --cpu-optimizer --offload weight

# 同一组 CPU 运算，使用本机 NVMe 保存 shard
bash "$launcher" "${common[@]}" --nvme-path /mnt/local-nvme/megatron-swap
```

单独使用 `--cpu-grad-accumulation` 可保留 GPU optimizer，用于比较梯度累加的位置。CPU 端按通信 dtype 的最大 shard 尺寸分配两个 pinned 槽位，D2H 完成后由 CPU worker 累加到 main grad，两个操作完成后才复用该槽位。梯度同步、unscale、范数和裁剪沿用 Megatron 的 GPU 路径。每步结束会回载梯度到 GPU，CPU AdamW 再按 bucket 收取裁剪后的 shard。CPU AdamW 和 NVMe 用 eager BF16 FSDP3，单独的 CPU 梯度累加也支持 eager FSDP2。

NVMe 模式需要 Linux、已编译的 Megatron-LM `megatron_nvme_aio` 扩展及支持 direct I/O 的本地 NVMe 目录。每个 rank 在该目录内创建独立的临时 swap 子目录。计算权重、FP32 master、累加梯度和 Adam 一阶/二阶矩分别按 bucket 交换，Adam 的步数留在 CPU。读取权重时复用 FSDP 的 all-gather 预取顺序，梯度最终回载也提前提交下一 bucket 的磁盘读取。正常进程退出会清理临时目录。

Nsight Systems 在 CPU 线程上显示 `fsdp.cpu_grad.d2h`、`fsdp.cpu_grad.wait_d2h`、`fsdp.cpu_grad.accumulate`、`fsdp.cpu_grad.finalize_h2d`、`fsdp.cpu_optimizer.gradient_d2h`、`fsdp.cpu_optimizer.adamw`、`fsdp.cpu_optimizer.publish` 和 `fsdp.nvme.read/write`。各模式的结果目录和报告名带有对应后缀。

#### 编译 NVMe 扩展并查看 worker 时间线

在 Linux GPU 容器内编译 Megatron-LM 自带的 `src/nvme/deepspeed`。扩展沿用 DeepSpeed 的 libaio 队列调度，编译依赖 C++17、`libaio-dev` 和 NVTX3 头文件。下面从 Megatron-Bridge 根目录执行，将扩展装到启动器使用的 Python 环境。若 NVTX3 头文件在 Nsight Systems 安装目录中，将 `NVTX_INCLUDE_DIR` 指向包含 `nvtx3/` 的 include 目录。

```bash
export MEGATRON_LM_DIR="$PWD/../megatron-lm"
train_python="$(uv run --no-sync python -c 'import sys; print(sys.executable)')"
uv pip install --python "$train_python" setuptools wheel
uv pip install --python "$train_python" --no-build-isolation \
  --no-deps --no-cache --reinstall "$MEGATRON_LM_DIR/src/nvme/deepspeed"

MEGATRON_FSDP_TEST_NVME_PATH=/mnt/local-nvme/megatron-swap \
  uv run --no-sync python "$MEGATRON_LM_DIR/src/nvme/deepspeed/verify_aio.py"
```

重跑上面的 `--nvme-path ... --profile nsys` 命令。在 Nsight Systems 中展开 `mfsdp-nvme-aio` 线程，`NVMe AIO read/write` 显示 worker 处理每个 shard 的区间和字节数，内层 `io_submit` 和 `io_getevents` 分别显示向内核提交和等待完成的时间。Python 线程上的 `fsdp.nvme.read/write` 同时包含 CPU staging 拷贝。SSD 控制器上的物理传输需要设备侧观测。

### DDP：逐层 backward 时间（27b / 397b / 2.4t）

在 sweep 脚本中用 `--fsdp ddp --cuda-graph off,on` 对比 CUDA graph 开关下的逐层 backward 时间。开启后，TE 为每层、每个 micro-batch 分别生成 forward 和 backward graph，可以按 graph launch 对齐层内 kernel。Dense FSDP 的 `full_iteration` 把整步 forward 和 backward 放进同一个 graph，逐层计时需要从 node trace 中还原层边界。

对 397b / 2.4t，实验还要让两组都沿用相同的 **dropless MoE** 路径。DDP + TE partial graph 只捕获 attention、router 等可捕获部分，动态 token dispatch、routed expert 计算和 combine 仍在 eager 路径执行。因此保持 dispatcher 和其他参数一致时，graph 开关两组使用相同的 expert kernel。

MoE FSDP3 的原生 partial graph 捕获 attention、router、preprocess 和 MoE postprocess，也可以单独捕获符合条件的 FSDP 参数预取。动态 dispatch、routed expert 计算及 combine 走 eager。

| 项 | `--fsdp ddp` 的覆盖 |
| --- | --- |
| 数据并行 | `dist.use_megatron_fsdp=false`、`ddp.use_megatron_fsdp=false`、`ddp.data_parallel_sharding_strategy=no_shard`（该字段只有 FSDP 使用） |
| 优化器 | `ddp.use_distributed_optimizer=true`：参数与梯度在每卡完整保留，只切 optimizer state |
| 通信重叠 | `ddp.overlap_grad_reduce=true`。`ddp.overlap_param_gather` 默认 `false`，加 `--ddp-overlap` 后设为 `true`（`comm_overlap.*` 同步覆盖） |
| checkpoint | `checkpoint.ckpt_format=torch_dist`（`fsdp_dtensor` 只支持 Megatron-FSDP；本实验不保存） |
| 命名 | 结果目录 `model_<m>-ddp-...`，nsys 报告 `<qwen35_27b\|qwen35_397b\|qwen38_2.4t>-ddp-...`；`launch.json` 中 `fsdp` 为 `null`、`data_parallel` 为 `ddp` |

默认由 distributed optimizer 在更新参数分片后同步执行参数 all-gather，下一轮第一个 micro-batch 直接使用已聚合的参数。加 `--ddp-overlap` 后，参数 bucket 在下一轮第一个 micro-batch 的 forward 中预取。两种模式都开启梯度 reduce-scatter 与 backward 的重叠。混合 sweep 只给 `ddp` case 传 `--ddp-overlap`。

同步模式的结果目录和 nsys 报告名带 `ddp-sync-ag`。前向预取模式沿用 `ddp` 命名，可续跑已有的前向预取实验。

`--cuda-graph on` 时设置 `cuda_graph_impl=transformer_engine`、`cuda_graph_warmup_steps=3`、`use_te_rng_tracker=true`、`rng.te_rng_tracker=true` 与 `NCCL_GRAPH_REGISTER=0`（`TECudaGraphHelper` 在 expandable segments 下要求关闭）。每层每个 micro-batch 分别有一个 forward graph 和一个 backward graph：

- dense（27b）：`cuda_graph_modules=[attn,mlp]`，整层进 graph；
- MoE（397b / 2.4t）：`cuda_graph_modules=[attn,moe_router,moe_preprocess]`，attention、router、shared expert 与 preprocess 进 graph，dispatch、routed experts、combine 保持 eager。HybridEP 与 alltoall 都可以用，expert 沿用非融合 `TEGroupedMLP`。
- MXFP8 + alltoall 时 `moe_router_padding_for_fp8` 让 alltoall 的 preprocess 把 token 数拷回 host，MCore 不允许 graph 它，启动器改用 `[attn,moe_router]`；
- 不需要关闭 NaN 检查，recipe 的 `check_for_nan_in_loss` / `check_for_nan_in_grad` 保持开启；
- 使用 `--ddp-overlap` 时，Bridge 在 capture 时摘掉 DDP 的 forward pre-hook，capture 后改为每层的 manual hook（`train.py` 的 `cuda_graph_set_manual_hooks`）。

显存：DDP 每卡保留完整的 bf16 参数与 fp32 梯度，dense 部分约 9 B/参数（2 + 4 + 12/4），EP=4 时 routed expert 的数据并行组为 1，与 FSDP 相同。2.4t 64 experts 每卡模型状态约 120 GiB，与 FSDP1 相当；397b 与 27b 余量更大。

逐层 backward 时间的读法：

- **graph 关**：nsys 模式下每层有 `L<i> bwd` NVTX range（见下文「Layer 的 forward / backward 标注」），直接投影到 GPU 即可得到每层 backward 时间；DDP 的梯度 reduce-scatter 在独立 stream 上，与后续层的 backward 重叠。
- **graph 开**：MCore 的 TE graph 路径在 capture 与 replay 时都绕过 `TransformerLayer.forward()`，直接调用 `_forward_attention` / `_forward_mlp`（`megatron/core/transformer/transformer_layer.py` 的 `_te_cuda_graph_capture` / `_te_cuda_graph_replay`），而 `L<i> fwd` / `L<i> bwd` 是在 `forward()` 里注入的，因此 capture 之后（nsys 采集的 step 7）**没有这两类 range**。此时按 graph 读：启动器在 graph 开启时给 nsys 加 `--cuda-graph-trace=node`，graph 内每个 kernel（GEMM、NCCL 等）按名字单独出现在 GPU 时间线上，同一 graph launch 的 kernel 共享 graph ID。27b 每层的 backward 就是该层 backward graph 内的全部 kernel；MoE 层的 backward 是 graph 部分（attention / router / shared expert）加上前后 eager 的 combine、routed experts、dispatch 的 backward kernel。
- 比较 graph 关 / 开的 iteration 时间时加 `--profile none`，以排除 nsys 与 `record_shapes` 的 host 开销。

```bash
bash examples/scale-down/01-analyse/04-fsdp3-comm/benchmark_fsdp3.sh --model 27b,397b,2.4t --fsdp ddp --cuda-graph off,on
# 对比 MoE FSDP3 原生 partial graph 和 DDP TE graph
bash examples/scale-down/01-analyse/04-fsdp3-comm/benchmark_fsdp3.sh --model 2.4t --fsdp 3,ddp --experts 64 --seq 1024 --cuda-graph off,on
# 开启 DDP 的首个 micro-batch forward 参数预取
bash examples/scale-down/01-analyse/04-fsdp3-comm/benchmark_fsdp3.sh --model 27b --fsdp ddp --seq 4096 --ddp-overlap
```

第一条命令的矩阵：27b 为 seq `256,1024,4096` × graph `off,on` 共 6 次；397b / 2.4t 为 experts `64,32` × seq `256,1024,4096` × graph `off,on` 各 12 次。

GatedDeltaNet 的图捕获仍待在目标 GPU 栈验证。DDP 捕获 router 时，`--force-balance` 的随机 logits 是否在每次 replay 时重新采样也待验证。

### 强制均衡路由（`--force-balance`）

两个脚本都支持 `--force-balance`，它设置 MCore 的 `model.moe_router_force_load_balancing=true`：`TopKRouter.forward` 在 router GEMM 之后把 logits 换成每 rank 独立的 N(0,1) 随机数（`apply_random_logits`，反向为直通），再照常 top-k。每个 token 因此均匀随机地选 k 个 expert，各 expert 与各 EP rank 的 token 数在统计上相等；router GEMM、dispatch / combine 与 expert 计算量不变。

- 只用于性能测试：路由与输入无关，loss 没有意义；
- 只作用于 MoE case，`benchmark_fsdp3.sh` 中的 27b 不受影响，单 case 启动器对 27b 报错；
- 结果目录与 nsys 报告名在 dtype 标记后加 `-fb`，例如 `model_2.4t-fsdp_3-fb-layers_4-...` 与 `qwen38_2.4t-fsdp_3-fb-experts_64-....nsys-rep`；不加该选项时命名不变；
- 默认路由是 `global_aux_loss`，系数取 HF config 的 `router_aux_loss_coef`（缺省时为 0）。10 个 iter 的 mock 数据下，路由主要由随机初始化的 router 决定，EP rank 之间可能不均；`--force-balance` 用来排除这部分干扰；
- MoE FSDP3 的 router 在图外执行，每步仍通过原来的 eager 路径生成 logits。

```bash
bash examples/scale-down/01-analyse/04-fsdp3-comm/benchmark_fsdp3.sh --model 397b,2.4t --force-balance
```

## CUDA graph

`--cuda-graph on` 按模型和数据并行方式选图。

| `--cuda-graph` | `--fsdp` | dense（27b） | MoE（397b / 2.4t / 122b） |
| --- | --- | --- | --- |
| `off` | 任意 | `cuda_graph_impl=none` | `cuda_graph_impl=none` |
| `on` | `3` | `cuda_graph_impl=full_iteration`，捕获整轮 forward + backward | `cuda_graph_impl=local`、`cuda_graph_modules=[attn,moe_router,moe_preprocess]`，符合条件的 FSDP 参数预取单独进图 |
| `on` | `1` | `cuda_graph_impl=full_iteration` | 同样的原生 MoE partial graph，读取常驻 GPU 的计算权重 |
| `on` | `2` | `cuda_graph_impl=full_iteration` | MoE partial graph 支持 FSDP1 和 FSDP3 |
| `on` | `ddp` | `cuda_graph_impl=transformer_engine`，`cuda_graph_modules=[attn,mlp]`：每层整层一个 graph | `cuda_graph_impl=transformer_engine`，`cuda_graph_modules=[attn,moe_router,moe_preprocess]`：dispatch / routed experts / combine 保持 eager；hybridep 与 alltoall 都可用 |

### FSDP1 和 FSDP3 的 partial graph

使用 BF16、TP=PP=1，关闭 `--symmetric`。CP>1 时，attention graph 包含前后向的 CP 通信。两种 FSDP 策略均关闭 fused gradient accumulation，并保留 NaN 检查。前 3 轮走 eager，第 4 轮开始 replay。默认 MoE 范围的结果名包含 `cg_on-partial-moe`。显式使用 `--local-attention-graph` 时只捕获 attention，结果名包含 `cg_on-partial`。

- FSDP1 的 graph 读取常驻 GPU 的计算权重。Optimizer step 后同步 all-gather 更新这些地址中的权重。
- FSDP3 启用 max-pool double buffering。预取图在 all-gather stream 上写入槽位，attention 和 router graph 通过外层 FSDP hook 等待权重就绪。选择 `--offload weight` 时，参数 H2D 和 all-gather 走 eager，仍写入同一组固定槽位。只卸载 gradient 或 optimizer 时，参数预取图仍然启用。

梯度通信、EP dispatch/combine 和 routed expert 计算留在图外。MoE 的默认 sweep 在无 offload 时对 FSDP1 和 FSDP3 都启用 partial graph。FSDP1 的 partial graph 要求 GPU 常驻权重和同步参数 all-gather。

运行环境必须能导入支持对应 FSDP partial graph 范围的 Megatron-LM checkout。若当前环境指向其他版本，可在启动前将带改动的 checkout 放在 `PYTHONPATH` 首位。

```bash
export PYTHONPATH=/path/to/patched/megatron-lm${PYTHONPATH:+:$PYTHONPATH}
bash examples/scale-down/01-analyse/04-fsdp3-comm/benchmark_fsdp3.sh \
  --model 2.4t --fsdp 1,3 --experts 64 --seq 256,1024,4096 --cuda-graph off,on
```

### Dense FSDP full-iteration graph

第 4 轮捕获并首次 replay 整轮 forward 和 backward，optimizer step 走 eager。启动器打开 `ddp.megatron_fsdp_cuda_graph_mode` 保留梯度引用，关闭 `ddp.fsdp_all_gather_in_start_param_sync` 和 loss／梯度 NaN 检查。FSDP1／FSDP2 的参数 all-gather 在 optimizer step 中同步完成。

启动器统一设置 `NCCL_GRAPH_REGISTER=0`。默认 nsys 采集 replay 阶段的 step 7，比较 iteration 时间时用 `--profile none` 去掉 profiler 开销。

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

## AG / RS 通信组分离（`--overlap`）

`--overlap` 设置 Bridge 的 `dist.create_all_gather_group=true`（对应 Megatron-LM `--create-all-gather-group`）：初始化时再建一组与 `dp_cp` / `expt_dp` rank 完全相同的 communicator（`DATA_PARALLEL_GROUP_WITH_CP_AG`、`EXPERT_DATA_PARALLEL_GROUP_AG`），Megatron-FSDP 的参数 all-gather 改走这组 communicator，梯度 reduce-scatter 仍走原来的。

- 不加时：RS 发在 `side_stream_for_buffer_copy_and_grad_accum`、AG 发在 `side_stream_for_param_gather`，两条 stream 不同，但 NCCL 对同一个 communicator 的每次 launch 都让 user stream 先等 `comm->sharedRes->deviceStream`、launch 后再把它推进到本次结束（NCCL `src/enqueue.cc` 的 `ncclLaunchPrepare` / `ncclLaunchFinish`），同一 communicator 上的操作因此按发起顺序串行。backward 中 layer i 的 pre-backward hook 预取 layer i-1 的 AG，它要排在 layer i+1（及更早未完成）的 RS 之后；
- 加上后：AG 与 RS 属于两个 communicator，各自的 `deviceStream` 互不等待，可以在 GPU 上同时执行（仍共享 SM 与 NVLink 带宽）。RS 之间仍按顺序执行；
- 若设置了 `NCCL_COMM_SPLIT_SHARE_RESOURCES=1`，split 出来的 communicator 共享父 communicator 的 `sharedRes`（含 `deviceStream`），AG / RS 又会串行，`--overlap` 失效；
- 只有 FSDP3 在 backward 中同时有 AG 预取与 RS，FSDP1 / FSDP2 只是换了 AG 用的 communicator；`ddp` 不使用这组 communicator，单 case 脚本直接拒绝 `--fsdp ddp --overlap`；
- nsys 中可以看到 `ncclDevKernel_AllGather_*` 与 `ncclDevKernel_ReduceScatter_*`（`--symmetric` 时为 `ncclSymkDevKernel_*`）落在两条不同的 stream 上，时间区间可以重叠；不加时两者在同一条 stream 上首尾相接。

`--separate-rs` 另建与原 DP 分片组 rank 顺序相同的 RS communicator。仅 FSDP3 梯度 ReduceScatter 走该组，DTensor 分片和其他 DP collective 仍用原组。可以单独启用，也可以与 `--overlap` 同用。

## Symmetric memory（`--symmetric`）

`--symmetric` 用 Megatron 已有的 `ddp.nccl_ub` 把 Megatron-FSDP 的通信 buffer 注册为 NCCL symmetric window（`ncclCommWindowRegister(..., NCCL_WIN_COLL_SYMMETRIC)`），让 NCCL 选用 symmetric kernel。启动器加的覆盖：

| 覆盖 | 作用 |
| --- | --- |
| `ddp.nccl_ub=true` | 创建全局 `NCCL_MEMORY_POOL`（MCore `nccl_allocator`，JIT 编译的 `ncclMemAlloc` 分配器）；Bridge 随之强制 `fsdp_manual_registration=true` |
| `ddp.disable_symmetric_registration=false` | 注册为 symmetric window（为 true 时退回 `ncclCommRegister` 的普通 user buffer 注册） |
| `ddp.fsdp_double_buffer=true`、`ddp.megatron_fsdp_max_pool_double_buffer=true` | MaxPool double buffer：full attention 层与 GatedDeltaNet 层的 bucket 大小不同，FixedPool 只能覆盖其中一种，MaxPool 按最大 bucket 建槽位，所有层共用两份固定地址的 buffer |
| `ddp.fsdp_db_use_persist_buf_on_alloc_fail=true` | 不是 FSDP unit 的模块（embedding、LM head、final norm）在池内分配常驻 buffer；为 false 时它们走动态分配，不在池里，通信不走 symmetric |
| `NCCL_NVLS_ENABLE=1` | `AllGather_STMC` / `ReduceScatter_LDMC` 依赖 multicast；`run_recipe.py` 只 `setdefault`，shell 里设成 0 时会退化为单播的 `AllGather_ST` / `ReduceScatter_LD`，所以启动器强制 export |

以上 double buffer 选项与 recipe 当前值相同，显式写出是为了 `--recipe` 换成别的 recipe 时行为不变。`NCCL_ALGO` 会把集合通信钉在经典 ring / NVLS kernel 上，`--symmetric` 下启动器直接拒绝；`--fsdp ddp` 也拒绝（MCore DDP 不用 MaxPool）。

注册以内存池为单位。`manual_buffer_registration()` 在第 1 个 iteration 之后把池注册到各 FSDP communicator，包括 `--overlap` 创建的 AG 组和 `--separate-rs` 创建的 RS 组。池里有：

- 常驻的参数分片（`model_weight_buffer`，AG 的 send buffer）；
- 常驻的梯度分片（all-in-one `main_grad_buffer`，RS 结果累加的目标）；
- MaxPool double buffer：`fsdp_params_*`（FSDP3 AG 的 recv buffer）、`fsdp_grads_*`（RS 的 send buffer，RS 的输出是同一 bucket 中本 rank 那一段）、MXFP8 时的 `fsdp_fp8_transpose_params_*`；
- 非 FSDP unit 的常驻 fallback buffer。

fp32 main weight（`main_weight_buffer`）在 `mem_alloc_context` 之外分配，不在池里，它也不参与 FSDP 通信。NCCL 选 kernel 时只看 AG 的 recv、RS 的 send 是否落在 symmetric window 中，其余已注册的 buffer 只是顺带。

**确认生效：**

- 第 1 个 iteration 在注册之前，必然走经典 kernel，计时与 profile 都要跳过它（默认 profile 第 7 步）；
- 在 FSDP 的 AG/RS NVTX 区间核对 NCCL 的实际选核。MC 对照要求 `ncclSymkDevKernel_AllGather_STMC` 和 `ncclSymkDevKernel_ReduceScatter_LDMC`；
- NCCL 2.29 在 Blackwell 上会把每 rank 元素数超过 `NCCL_SYM_CE_THRESHOLD`（默认 8Mi）的 AllGather 交给 copy engine（nsys 中是 memcpy，而不是 kernel），想强制 SM kernel 可以 `export NCCL_SYM_CE_THRESHOLD=<更大的值>`；
- 启动器把 `NCCL_DEBUG` 固定为 `WARN`，看不到 `NCCL_DEBUG_SUBSYS=TUNING` 的选核日志，以 nsys kernel 名为准。

**注意：**

- MaxPool 的槽位按需增长；若注册之后又有更大的 bucket 请求，新分配的 segment 不在已注册的 window 中，对应通信退回经典 kernel；
- 4 卡 EP=4 时 expert FSDP group 只有 1 个 rank，也会注册一次（没有通信）；
- 与 `--cuda-graph on` 同用时：注册发生在第 1 步，graph 在 `cuda_graph_warmup_steps=3` 之后 capture，replay 直接使用 capture 时选定的 symmetric kernel；`NCCL_GRAPH_REGISTER=0` 只关闭 capture 期间对经典算法 buffer 的自动注册，不影响 symmetric window。这一组合尚未实测；
- `--symmetric --overlap` 时 AG 与 RS 分属两个 communicator，各自在自己注册的 window 上运行，可以在 nsys 中看到两条 stream 上的 `ncclSymkDevKernel_*` 重叠。

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
- DDP 的 `transformer_engine` 图和 dense 的 `full_iteration` 图在 replay 时跳过这些 Python 层标注。MoE FSDP3 的 native partial graph 仍执行 `TransformerLayer.forward()`，因此保留层级 NVTX range。

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

其他节点把 `--node-rank 0` 改为 `1`、`2`、`3`。

PP 固定为 1，MoE 的 EP 固定为 4；FSDP 在全部 DP×CP rank 上分片，TP 是 DTensor mesh 的 `tp` 维。MoE case 的 GPU 数必须是 4 的倍数。

单个 case 也可以直接调用（`--print-result-dir` 校验 case 后只打印结果目录）：

```bash
bash examples/scale-down/01-analyse/04-fsdp3-comm/run_pretrain_fsdp3.sh \
  --model 2.4t --seq 4096 --tp 1 --cp 1 --mbs 1 --gbs 8 \
  --num-layers 4 --num-experts 64 --cuda-graph on --profile nsys
```

训练命令通过 `torch.distributed.run --no-python` 让每个 worker 执行 `numarun python ...`。脚本依次检查目标设备上的 `numarun`、仓库 `.cache/numarun`（需要 `numactl`）、`numactl --localalloc` 的本地 NUMA 内存策略，都不可用时才取消绑定。

实际启动时，[Gloo 网卡 helper](../../../../scripts/scale-down/common/gloo_socket_ifname.sh) 用 `ip -o -4 addr show up` 验证环境中的 `GLOO_SOCKET_IFNAME`。需要自动选择时，单节点优先 `eth0`，多节点优先 `bond1`，另一个接口作为备选。最终选择写入 `run_info.txt`。

结果目录保持为 `results/01-analyse/04-fsdp3-comm/`，case 名包含层数、MTP 层数、routed expert 数（dense 为 0，未裁剪为 `full`）和 CUDA graph 模式，并按节点保存：

```text
results/01-analyse/04-fsdp3-comm/model_2.4t-fsdp_3-layers_4-mtp_0-experts_64-cg_on-partial-moe-mbs_1-seq_4096-cp_1-ep_4-disp_hybridep-gbs_8-gpus_4-profile_nsys-date_261031-120000/node0/
```

目录中包含 `launch.json`、`command.txt`、`run_info.txt` 和 `train.log`；启用 `--nsys` 时，`node0/` 下还会有唯一一份 nsys 报告。`config.yaml` 只由全局 rank 0 写出，因此多节点时只出现在 `node0/`。

## Nsys

`benchmark_fsdp3.sh` 默认 `--profile nsys`（单 case 脚本 `run_pretrain_fsdp3.sh` 默认 `none`）。`nsys profile --capture-range=cudaProfilerApi` 包在 torchrun 外面，跟踪 torchrun 拉起的全部 worker 进程；`profiling.profile_ranks` 为节点 0 的 4 个 rank（0..nproc-per-node-1，即一个 GB200 tray 的 4 张 GPU），Bridge 在 `profile_step_start`（默认 7）调用 `cudaProfilerStart`，在 step 计数到 `profile_step_end`（默认 8）时停止，因此每份报告只覆盖 step 7（0-based，对应日志里的 iteration 8）这一个完整 iteration（含 4 个 micro-batch）。只有节点 0 的命令外包 nsys，其他节点不录制，因此每个 case 只产生一份报告，同一时间线上按进程列出这 4 个 rank。文件名为 `<qwen35_397b|qwen35_27b|qwen38_2.4t>-<ddp|fsdp_1|fsdp_2|fsdp_3>[-mxfp8][-fb][-attn_<full|linear>]-experts_<E>-cg_<off|on|on-partial|on-partial-moe>[-merge][-tp_<TP>]-cp_<CP>-ep_<EP>-mbs_<MBS>-gbs_<GBS>-seq_<N>k.nsys-rep`。

- trace 范围为 `cuda,nvtx`：NCCL（FSDP all-gather / reduce-scatter、TP / SP 集合通信、CP p2p）与 HybridEP dispatch / combine kernel 以 CUDA kernel 形式出现在 GPU 时间线上，可以直接与计算 kernel 对比。CUDA graph 开启时，启动器追加 `--cuda-graph-trace=node`（nsys 默认 `graph` 只把每次 graph launch 画成一个整体），graph 内每个 kernel 都带名字出现，例如 `ncclDevKernel_AllGather_RING_LL`、`ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL` 或 symmetric memory 下的 `ncclSymkDevKernel_*`，可以直接在 Events View 按 `nccl` 过滤得到通信区间，或用 `nsys stats --report cuda_gpu_kern_sum` 汇总。图内的通信 kernel 在 replay 时由 CUDA graph 发射，`nsys recipe nccl_sum` 可能缺少这些 kernel 的 host 侧字节数。MoE partial 模式仍有 FSDP 层级 NVTX 和图外通信，可以用它们与 graph launch 对齐。node 模式的 trace 开销略高于 graph 模式，比较 iteration 时间仍以 `--profile none` 为准。加 `--merge` 时改用 nsys 默认的 `--cuda-graph-trace=graph`，每次 graph launch 只显示为一个整体，看不到其中的 kernel，但 trace 开销更低；结果目录与报告名的 graph 标记追加 `-merge`，与展开模式分别保存。`--cuda-graph off` 或 `--profile none` 时 `--merge` 不起作用，命名也不变，因此 sweep 可以与 `--cuda-graph off,on` 一起使用。
- 启用 nsys 时启动器设置 `profiling.record_shapes=true`，Bridge 进入 `torch.autograd.profiler.emit_nvtx(record_shapes=True)`，每个 op 的 NVTX range 带输入形状；会增加 host 开销，比较 iteration 绝对时间时，用 `--profile none` 的 `train.log` 作为基准。
- FSDP all-gather / reduce-scatter 的层归属看上一节的 `FSDP AG ...` / `FSDP RS ...` range。
- NCCL 自身在 `NCCL` NVTX domain 为每次 `ncclAllGather` / `ncclReduceScatter` 打 range，payload 含消息字节数和 comm hash；启动器在 nsys 模式下设置 `NCCL_NVTX_DISABLE=0`。nsys 不会在 kernel 上直接显示带宽，需要用 recipe 关联 NVTX 和 kernel，例如 `nsys recipe nccl_sum --input <report>.nsys-rep`。
- 4 个 rank 在同一份报告中，可以直接对比各 rank 的 FSDP AG / RS 与 TP / EP / CP 通信是否对齐、负载是否均衡。多节点时报告只包含节点 0 的 rank。
