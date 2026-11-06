# GB200：用 Nsight Compute 2026 profile torch all-gather 的 NCCL kernel

日期：2026-11-06。按 Nsight Compute CLI 文档（5.3.2 Multi-Process Support）的写法，用**一个 ncu 包住整个
torchrun**，通过 shmem communicator 让各 rank 的 NCCL kernel 同时 launch、同时重放：

```bash
ncu --communicator shmem --communicator-shmem-num-peers 2 -f -o report -k regex:nccl.* \
    torchrun --nnodes=1 --nproc_per_node=2 pytorch_app.py
```

本目录把文档里的 `pytorch_app.py` 换成 `ncu_torch_all_gather.py`，并加上算法固定、TUNING 校验和卡死检测。

> 状态：还没在 GB200 上实际跑过。`../nccl-tests/README.md` 记录过，STMC kernel 在 kernel replay 下会报
> `Failed to save memory for replay`，`--algo symm` 大概率需要换 replay 模式。

## 文件

| 文件 | 作用 |
| --- | --- |
| `ncu_torch_all_gather.py` | torchrun worker：warmup，再在 NVTX range `ncu_all_gather` 里跑 `--iters` 次 all-gather，前后都校验数据 |
| `run_ncu_torch_all_gather.sh` | 单节点启动器：baseline（不挂 ncu）+ ncu 两遍 |

## 用法

在仓库根目录执行，需要 Nsight Compute ≥ 2026.1，并且有 GPU 性能计数器权限：

```bash
# 默认：4 卡、NVLS、256 MiB bf16、kernel replay、--set basic
bash examples/gb200/ncu-torch-all-gather/run_ncu_torch_all_gather.sh

# Ring 对照
bash examples/gb200/ncu-torch-all-gather/run_ncu_torch_all_gather.sh --algo ring

# symmetric AllGather_STMC（NCCL >= 2.32；kernel replay 保存显存会失败，改用 app-range）
bash examples/gb200/ncu-torch-all-gather/run_ncu_torch_all_gather.sh --algo symm --replay-mode app-range \
    --nccl-lib nccl/nccl-2.32.3/build/lib/libnccl.so.2

# 复现不加 communicator 时的卡死（由 --timeout 兜底）
bash examples/gb200/ncu-torch-all-gather/run_ncu_torch_all_gather.sh --communicator none --timeout 300
```

全部选项见 `--help`。默认使用 torch 自带的 NCCL；`--nccl-lib` 会通过 LD_PRELOAD 换成指定的库。

## `-k regex:...` 是什么

`-k` 就是 `--kernel-name`，按 kernel 函数名过滤：只有名字匹配的 kernel 才会被 profile，其他 kernel 照常
运行，不被串行化也不被重放。`regex:` 前缀表示后面是正则表达式，按子串搜索。文档示例里的
`regex:nccl.*` 会匹配所有 NCCL kernel。

这里用的是更窄的 `regex:^nccl(Symk)?DevKernel_AllGather`，原因如下：

- `dist.barrier()` 走 NCCL 时会 launch 一个 AllReduce kernel，用 `nccl.*` 会把它也选进来。worker 改成
  通过 gloo group 做 barrier，进程里唯一的 NCCL 工作就是 all-gather。
- NCCL 2.32.3 每个集合通信只编译一个特化 kernel（`src/device/generate.py` 的 `best_kernel`）。Ring 和
  NVLS 的 AllGather 都 launch `ncclDevKernel_AllGather_RING_LL`，实际算法在 kernel 内部分派，所以
  **单看 kernel 名分不出 Ring 和 NVLS**。脚本靠 TUNING 日志（`AllGather: <n> Bytes -> Algo NVLS ...`）
  确认每次 launch 的算法。symmetric 路径的 kernel 名是 `ncclSymkDevKernel_AllGather_STMC`。

同一个 kernel 名会出现在 warmup 和正式迭代里，所以还要加 `--nvtx --nvtx-include "ncu_all_gather/"`，只选
NVTX range 里的那几次 launch。这样就不用按 `--launch-skip` 去数 warmup 有几次。

## `pytorch_app.py` 要怎么写

对照 `ncu_torch_all_gather.py`，有以下几条要求：

1. **每个进程一张 GPU**：先 `torch.cuda.set_device(LOCAL_RANK)`，再
   `init_process_group("nccl", device_id=...)`。传 `device_id` 会立即创建 communicator，避免把建
   communicator 的开销算进第一次 all-gather。
2. **所有 rank 的 kernel 序列必须一致**：communicator 按顺序对齐各 rank 的第 k 个被选中的 kernel。
   不能出现某个 rank 多跑一次的情况，比如只在 rank 0 上做检查。
3. **只让要 profile 的 NCCL kernel 进入过滤范围**：barrier 用 gloo group，或者把 `-k` 写窄一些。
4. **用 NVTX range 标出要 profile 的区间**：`range_push` / `range_pop` 必须在 launch kernel 的同一个线程上
   调用（torch 在调用线程里 enqueue NCCL）。
5. **symm 需要先注册 window**：buffer 从 `torch.cuda.MemPool(backend.mem_allocator)`（底层是
   `ncclMemAlloc`）分配，再调用 `register_mem_pool(pool, symm=True)` 注册；结束前要先
   `deregister_mem_pool`。
6. **application / app-range replay 每个 pass 都会重新启动整个 torchrun**：程序必须能重复执行，并且
   每次 launch 顺序都相同。启动器用了 `--standalone`，每次都会自动选一个空闲的 rendezvous 端口。

## 输出

默认目录是 `results/gb200/ncu-torch-all-gather/<algo>-<时间>/`：

| 文件 | 内容 |
| --- | --- |
| `config.txt` | ncu 版本、参数、所有 `NCCL_*` 环境变量 |
| `baseline.log`、`ncu.log` | 两遍的输出，其中有 worker 的数据校验结果和 ncu 的 `==PROF==` 输出 |
| `nccl-logs/<pass>/nccl-<host>-<pid>.log` | NCCL INFO 日志（INIT、ENV、TUNING） |
| `all_gather.ncu-rep` | shmem 模式下所有 rank 合在一份报告里 |
| `summary.csv` | 每个被 profile 的 kernel（app-range 下是每个 range）一行 |

## 读结果时注意

- ncu 默认会锁频、刷 cache、kernel replay 时还会保存和恢复显存，所以报告里的耗时**不等于**真实性能。
  带宽请看 nsys 或 `../nvls-collective` 的 benchmark。ncu 用来看 kernel 内部的情况。
- 文档提到，shmem 模式下 `--communicator-shmem-num-peers` 不是必填，但建议填。不填时，如果各 rank 没有
  同时到达同一个 kernel，ncu 可能在参与者没到齐的情况下就开始 profile。
- 默认只 profile 每组的第一个 CUDA context；需要选 GPU 时，用 `-- --devices ...` 透传给 ncu。
