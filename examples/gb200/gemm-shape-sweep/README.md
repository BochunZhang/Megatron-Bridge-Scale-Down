# GB200 GEMM 形状扫描（mbench `GemmSweep`）

目标：在单张 GB200 上测量 **不同规模、不同长宽比** 的 BF16 GEMM 的耗时与 TFLOP/s，回答
"什么形状能吃满 Tensor Core、什么形状受访存 / wave 量化 / launch 开销限制"。与具体模型无关；
模型实际用到的 GEMM 见 [`qwen3.8-2.4T`](../micro-batch-size/qwen3.8-2.4T/README.md)、
[`qwen3.8-27b`](../micro-batch-size/qwen3.8-27b/README.md)。

计时协议（`cuStreamWaitValue32` 闸门 + CUDA Event、Nsys 交叉校验、CSV 列）与其他 scope 完全相同，见
[`examples/gb200/benchmark/README.md`](../benchmark/README.md)。

## 1. 执行内容

每个 case 是一次 `torch.nn.functional.linear`：`Y[M, N] = X[M, K] @ W[N, K]^T`，即所有 Linear
前向执行的 cuBLAS "TN" GEMM（BF16 输入 / 输出，fp32 累加）。`benchmark.modes` 加上 `fwdbwd` 后
还会计入 dgrad（`dY @ W`，NN）和 wgrad（`dY^T @ X`，NT；BF16 输出，不累加 fp32 `main_grad`）。

## 2. 形状族（`gemm_sweep.yaml`）

| family | 形状 | 关注点 |
| --- | --- | --- |
| `square` | M = N = K ∈ {256 … 32768} | 均匀矩阵随规模变化：launch 受限 → wave 量化 → 计算饱和 |
| `tall_skinny` | M = 65536, K = 8192, N ∈ {16 … 4096} | 细长输出（router / gate / 小词表头）：N 太小时变成访存受限 |
| `small_m` | M ∈ {16 … 4096}, K = N = 8192 | 少行 × 大权重（decode、小 expert 组、小 mbs）：权重读取主导 |
| `thin_k` | M = N = 16384, K ∈ {64 … 4096} | 短规约：输出 tile 多但每字节复用少，epilogue / 写回占比高 |
| `deep_k` | M = N = 2048, K ∈ {8192 … 524288} | 长规约、输出小（wgrad 形态）：输出 tile 少，难以填满 148 个 SM |
| `iso_flops` | M·K·N = 2^39 的 10 种长宽比 | 固定计算量，只改变形状，分离"形状效应"与"规模效应" |

每个 family 三种写法任选其一：`square: [...]`、`m/k/n` 笛卡尔积、`shapes: [[M, K, N], ...]`。
case 名为 `<family>_m<M>_k<K>_n<N>`，可直接追加新的 family。

CSV 额外列：`family`、`arithmetic_intensity`（前向 GEMM 的 FLOPs / 字节，按每个操作数恰好读写
HBM 一次计算，用于判断访存受限还是计算受限）、`operand_mib`（X、W、Y 总大小）。

## 3. 运行

```bash
# 全部 family
examples/gb200/gemm-shape-sweep/run_gemm_sweep.sh

# 加 nsys trace + SQL 交叉校验列（kernel 名可看出 cuBLAS 选择的 tile / split-K）
examples/gb200/gemm-shape-sweep/run_gemm_sweep.sh --nsys

# 只跑几个 case
examples/gb200/gemm-shape-sweep/run_gemm_sweep.sh --cases square_m8192_k8192_n8192,deep_k_m2048_k524288_n2048

# 自定义形状表 / 结果目录
CONFIG=/path/to/my_sweep.yaml RESULTS_ROOT=/path/to/out examples/gb200/gemm-shape-sweep/run_gemm_sweep.sh
```

## 4. 输出

```
results/gb200/gemm-shape-sweep/
  gemm_sweep/                         每个 (case, mode) 一个 JSON（micro_batch_size 固定记为 1）
  nsys/gemm_sweep.nsys-rep / .sqlite  仅 --nsys
  csv/gemm_sweep.csv
```

## 5. 解读注意

- **功耗 / 频率状态**：小形状一轮只有几毫秒，GPU 处在 boost 突发；大形状（如 `square` 32768、
  `iso_flops`）一轮上百毫秒，会进入功耗墙稳态，SM 频率明显下降。不同形状之间的 TFLOP/s 差异
  因此混有频率差异，比较时请同时看 `sm_clock_mhz`、`power_w` 和单次迭代时长。
- 只测孤立 GEMM：输入在测量期间反复复用（小形状可常驻 126 MB L2），没有通信 kernel 争抢 SM / 带宽。
- 只覆盖 BF16。
