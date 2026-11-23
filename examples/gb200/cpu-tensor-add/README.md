# GB200 Grace CPU 混合精度张量加法

测量两个等长、连续的一维 CPU 张量逐元素相加的线程扩展性。输入为 BF16 和 FP32，结果写入预分配的 FP32 张量。测试使用 PyTorch 的 `torch.add(bf16, fp32, out=output)`，只使用 Grace CPU 和主机内存。

## 运行

在 GB200 计算节点的仓库根目录执行。启动器用 `uv run --no-sync --active`，优先使用已激活的 PyTorch 虚拟环境，否则使用项目的虚拟环境。

```bash
bash examples/gb200/cpu-tensor-add/run_cpu_tensor_add_benchmark.sh
```

要在单个 Grace NUMA 节点上对比线程数，先用 `numactl --hardware` 查看可用节点，再设置 `CPU_ADD_NUMA_NODE`。下面以节点 0 为例。

```bash
CPU_ADD_NUMA_NODE=0 bash examples/gb200/cpu-tensor-add/run_cpu_tensor_add_benchmark.sh
```

默认每个张量分别包含 `32M、64M、128M、256M、512M、1024M` 个元素，`1M = 2²⁰`。线程数默认取 `1,2,4,8,16,32,48,64` 中 CPU affinity 允许的档位，再加入可用逻辑 CPU 总数。每组预热 3 次，计时 10 次。可用以下命令先跑一组小矩阵。

```bash
bash examples/gb200/cpu-tensor-add/run_cpu_tensor_add_benchmark.sh \
  --sizes-m 32,64 --threads 1,2,4,8 --warmup 2 --iters 5
```

1024M 档的两个输入和预分配输出合计占 10 GiB。采用 CPU TensorIterator 整张量类型提升的 PyTorch 构建还会在每次调用中生成一个 4 GiB 的 FP32 临时张量。运行最大档前，给所选 NUMA 节点和作业内存额度留出超过 14 GiB 的空间。

## 结果

默认写入 `results/gb200/cpu-tensor-add/<timestamp>/`，也可通过 `--output-dir` 指定新目录。

- `summary.csv` 记录每组的中位耗时、有效 GB/s、相对同尺寸单线程的加速比，以及并行效率（加速比除以线程数）。自选线程档位时，包含 `1` 才会计算后两项。
- `samples.csv` 保存每次同步 CPU 加法调用的纳秒耗时。
- `config.json` 记录 PyTorch 版本、并行后端、CPU affinity 和测试矩阵。线程数为 PyTorch intra-op 配置值。

计时包含 PyTorch 加法内部的隐式 BF16 转 FP32、临时缓冲区及加法。输入和输出的预分配、预热和抽样结果校验在计时外完成。有效带宽依据逻辑输入和输出流量计算，每元素为 `2 + 4 + 4 = 10` 字节。实际内存流量还受临时缓冲区及缓存行为影响。
