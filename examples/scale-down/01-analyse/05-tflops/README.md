# 量化训练的 TFLOP/s 与通信时间线

在单节点 4 张 GB200 上，使用 Megatron-FSDP 和 mock 数据运行 Qwen3.8-27B 与 Qwen3.8-2.4T-A95B。默认矩阵为 2 个模型 × 2 个 FSDP 等级 × 3 种计算精度 × 3 种并行配置，共 36 个 case。

| `--parallel` | DP | CP | TP | Sequence length | 每轮 micro-batch 数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dp4` | 4 | 1 | 1 | 4096 | 8 |
| `cp4` | 1 | 4 | 1 | 16384 | 32 |
| `tp4` | 1 | 1 | 4 | 16384 | 32 |

两个模型都裁到 4 层全注意力，无 MTP，GBS=32，MBS=1。2.4T 每层保留 64 个 routed experts，使用 HybridEP、EP=4、expert TP=1。27B 是 dense 模型，EP=1。两个模型都读取各自的 Qwen3.8 配置，随机初始化。当前 Megatron 注意力输出门使用 sigmoid，官方配置中的 swish 门未建模。

| `--fsdp` | Sharding strategy | FSDP 梯度规约 |
| --- | --- | --- |
| `1` | `optim` | 最后一个 micro-batch 的 backward 中开始规约 |
| `2` | `optim_grads` | 每个 micro-batch 的 backward 中开始规约 |

两级都开梯度规约 overlap，使用 BF16 reduce-scatter 和 FP32 main gradient，optimizer step 后同步做参数 all-gather。四卡下 dense FSDP 的 DP×CP 组在 DP4/CP4 为 4 rank，在 TP4 为 1 rank。MoE expert DP 组在三种布局下均为 1 rank。

BF16、MXFP8、NVFP4 共用 BF16 参数通信。CUDA Graph 路径与 [04-fsdp3-comm](../04-fsdp3-comm/README.md#cuda-graph) 对齐。

- 27B 使用 `full_iteration`，捕获整轮 forward 和 backward。Optimizer step 在图外执行。
- 2.4T 使用原生 `local` partial graph，范围为 `[attn,moe_router,moe_preprocess]`。HybridEP dispatch、routed expert 计算、combine 和 FSDP 梯度规约在图外执行。量化 MoE 使用对应精度的 token 对齐。

## 运行

运行环境需要支持 MXFP8/NVFP4 的 Transformer Engine、HybridEP 和 Nsight Systems。Megatron-LM 使用 `core_v0.19.0_scale_down` 的 `8e771a34f` 或包含该提交的版本，它包含 FSDP2 单 rank 梯度 bucket 修复，以及支持 FSDP1/2 常驻权重、TP 和量化计算的 MoE local partial graph 实现。启动器优先使用相邻的 `megatron-lm` checkout，也可用 `MEGATRON_LM_DIR` 指定部署的源码位置。 [Megatron-LM 原生 partial graph 8e771a34f](https://github.com/BochunZhang/Megatron-LM-Scale-Down/commit/8e771a34f38afa6e7dd744d6b94cd87ed944b056)

```bash
study=examples/scale-down/01-analyse/05-tflops/benchmark_tflops.sh

bash "$study" --dry-run
bash "$study"
bash "$study" --precision bf16 nvfp4
bash "$study" --model 2.4t --fsdp 2 --parallel tp4 --precision mxfp8
bash "$study" --profile none --precision bf16,mxfp8
```

每个 case 跑 20 个训练 iteration，预热 3 轮，第 4 轮捕获 CUDA Graph。默认与 `04-fsdp3-comm` 对齐，只采集从 0 开始计数的 `[7, 8)`，即训练日志的第 8 轮。NSYS 记录 4 个 rank，并展开 CUDA Graph 内的 kernel。比较稳态 TFLOP/s 或迭代耗时时，使用同一筛选条件加 `--profile none`，查看图捕获之后的日志数据。

结果写入 `results/01-analyse/05-tflops/<RUN_DATE>/`。结果目录和报告名包含模型、FSDP 等级、精度和并行布局。每个 case 的 `node0/` 下有 `train.log`（逐轮耗时及 TFLOP/s/GPU）、`config.yaml`、`launch.json`、`command.txt` 和 NSYS 模式下的 `.nsys-rep`。指定同一个 `RUN_DATE` 加 `--resume` 可跳过已完成的 case。
