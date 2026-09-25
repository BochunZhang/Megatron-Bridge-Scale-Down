# CUDA Kernel Launch 流程与 CPU-GPU 通信

本文总结 PyTorch launch kernel 时 CPU 与 GPU 之间的通信内容、数据流向，eager 模式与 CUDA Graph 模式的差异，以及命令面（PBDMA）与数据面（Copy Engine）的关系。分析环境为 GB200（Grace CPU + Blackwell GPU，NVLink-C2C 900 GB/s 双向，cache-coherent）。

源码路径：

- PyTorch: `/Users/zhangbochun/Desktop/AI-Infra/envs/ali-gb200/python-package/PyTorch`
- nvbandwidth: `/Users/zhangbochun/Desktop/AI-Infra/NVLINK-C2C/nvbandwidth-v0.8`

## 结论速览

| 问题 | 结论 |
| --- | --- |
| launch kernel 会传输 tensor 数据吗 | 不会。只传输命令描述符（QMD ~512B）、参数值（tensor 的 `data_ptr` 指针、标量，≤4KB）、doorbell（8B）、semaphore（8B） |
| 这些通信走 C2C 吗 | 走。GB200 上 CPU↔GPU 唯一通用链路是 NVLink-C2C：doorbell MMIO 写、GPU fetch sysmem 中的命令、GPU 写回 semaphore 全部经过 C2C |
| launch 的瓶颈是什么 | CPU 侧 driver 软件路径（3~10µs/launch）为主，链路延迟（亚 µs~2µs）为辅；带宽几乎不构成因素 |
| QMD 放在哪里 | eager：CPU 每次 launch 就地写入 sysmem pushbuffer；CUDA Graph：`cudaGraphInstantiate` 时一次性烘焙进 vidmem（HBM） |
| 命令 fetch 用 H2D Copy Engine 吗 | 不用。命令 fetch 走 Host Interface 的 PBDMA（控制面），CE 是数据面引擎，且 CE 自己的命令也要靠 PBDMA 取回 |
| 会被正在进行的 H2D 传输阻塞吗 | 不会阻塞（无队列依赖），但 C2C / HBM / Grace 内存带宽接近饱和时命令面小事务延迟会膨胀（µs 级带宽竞争） |

## 1. Launch Kernel 流程

### 1.1 软件路径

```
PyTorch aten op
  → kernel<<<grid, block, smem, getCurrentCUDAStream()>>>(args)   # nvcc 生成代码
  → cudaLaunchKernel (cudart, 静态链入)
  → cuLaunchKernel  (libcuda driver, 用户态)
```

说明：

- PyTorch 不直接封装 `cuLaunchKernel`（`c10/cuda/driver_api.h` 只封装 `cuMemMap` 等其他 driver 入口），launch 经 `<<<>>>` 语法走 cudart 静态路径。
- 整条 fast path 在**用户态**完成，无 syscall：driver 初始化时已把 pushbuffer、GPFIFO、doorbell mmap 进进程地址空间（user-mode submission）。

### 1.2 Channel 数据结构

```
CUDA stream ──映射──▶ GPU Channel（每 context 一组，TSG/runlist 调度）
Channel 包含：
 ├─ GPFIFO ring      ：环形索引表，每项 8B = {pushbuffer 段的 GPU VA, 长度}，位于 sysmem
 ├─ Pushbuffer       ：存放命令字（含 QMD + 参数）的内存段，默认位于 sysmem（pinned）
 ├─ Doorbell 寄存器   ：映射到 CPU 地址空间的 MMIO（GPU BAR 空间）
 └─ Semaphore 槽位   ：完成通知的 8B 内存位置，通常在 sysmem
```

### 1.3 四类通信详解

一次 launch 涉及 4 类 CPU↔GPU 通信：

| # | 通信 | 载荷 | 源 → 目的 | GB200 上是否跨 C2C |
| --- | --- | --- | --- | --- |
| 1 | 命令包写入 | QMD(~512B) + 参数区(几十 B~4KB) + semaphore release 命令 | CPU store → pushbuffer（默认 sysmem） | sysmem 放置：写入时不跨；GPU fetch 时跨 |
| 2 | Doorbell | 8B = {channel ID, GPFIFO PUT 指针} | CPU store（MMIO）→ GPU Host Interface doorbell 寄存器 | **必跨** |
| 3 | 命令 fetch | 8B GPFIFO entry + 几百 B~几 KB pushbuffer 段 | GPU PBDMA DMA read → GPU 前端 FIFO | sysmem 放置：**跨**（GPU 读 Grace 内存） |
| 4 | 完成通知 | 8B semaphore（sequence number 或 timestamp） | GPU semaphore 单元 → sysmem（CPU 等待）或 vidmem（GPU 等待） | sysmem 目标：跨；vidmem 目标：不跨 |

各类通信要点：

1. **命令包写入（CPU → pushbuffer）**：driver 把 launch 描述打包为 QMD（Queue Meta Data：kernel 入口 GPU VA、gridDim/blockDim、shared memory 大小、寄存器数、constant buffer 绑定表）+ 参数区（`void* kernelParams[]` 按 ABI 打包，tensor 只写 8B 的 `data_ptr()` 指针值）。kernel 代码本身在 `cuModuleLoad` 时已整体拷入 vidmem，launch 只传函数地址。CPU 用普通 store 指令写入（WC 映射）。
2. **Doorbell（CPU → GPU MMIO）**：driver 更新 GPFIFO producer index 后，向 doorbell 寄存器做一次 8B posted write，通知 Host Interface "该 channel 有新工作"。这是 launch 路径上第一处必然跨链路的操作：PCIe 平台约 1~2µs，C2C 为亚 µs 级。
3. **命令 fetch（GPU DMA read）**：Host Interface 的 PBDMA 引擎读 GPFIFO entry 拿到 pushbuffer 段的 VA 和长度，再读取命令段 → Command Processor 解析 → QMD 进入 CWD（Compute Work Distributor）队列 → CTA 分发到 SM。此步为硬件 DMA，CPU 不参与，不占 CPU 时间。前端有预取机制，会超前 fetch 命令段。
4. **完成通知（GPU → semaphore）**：driver 在构建命令流时已在 kernel 命令后附加 SEMAPHORE_RELEASE method；kernel 完成后由 GPU 前端 semaphore 单元把 8B 值（seq/timestamp）写到指定地址。CPU 侧 `cudaStreamSynchronize` 默认 spin-poll **本地内存**（poll 本身不跨链路）；`cudaDeviceScheduleBlockingSync` 则改走 MSI-X 中断 + futex。CUDA event 的 `elapsed_time` 就是读两个 event release 的 GPU timestamp 相减。

### 1.4 完整时序（GB200，pushbuffer 在 sysmem 的典型配置）

```
CPU (Grace)                                        GPU (Blackwell)
─────────────────────                              ─────────────────────
① cudaLaunchKernel：打包 QMD+参数
   → store 进 sysmem pushbuffer   (不跨链路)
② 更新 GPFIFO entry               (不跨链路)
③ store doorbell ═══8B posted write═══C2C═══▶ Host Interface 标记 channel runnable
                                           ④ PBDMA 读 GPFIFO entry ◀══C2C DMA read
                                           ⑤ fetch pushbuffer 段(QMD+param) ◀══C2C
                                           ⑥ CP 解析 → CWD → CTA 分发到 SM
                                           ⑦ kernel 执行（代码/参数/tensor 全在 vidmem，
                                              不跨链路；offload host tensor 除外）
                                           ⑧ semaphore release ═══8B write═══C2C═══▶ sysmem
⑨ sync: spin-poll 读本地 semaphore（不跨链路）
```

典型配置下一次 launch 跨链路流量 ≈ doorbell 8B + fetch 几百 B~几 KB + semaphore 8B。nsys 中 "`cudaLaunchKernel` API 结束 → kernel 开始" 的 gap 主要由 ③→④→⑤→⑥ 的链路延迟 + 解析排队构成。

### 1.5 QMD 的位置：sysmem 还是 vidmem

eager 路径下 QMD 不存在"拷贝"动作——它是 pushbuffer 命令流的一部分（inline method data），CPU 在每次 launch 时**就地生成**。QMD 在哪 = pushbuffer 在哪。

| 时机 | 写入者 | 内容 | 落点 | 是否跨 C2C |
| --- | --- | --- | --- | --- |
| channel/context 初始化 | driver | pushbuffer、GPFIFO、USERD 分配 | sysmem (pinned) | — |
| 每次 eager launch | CPU store | QMD + 参数 inline 写入 pushbuffer | **sysmem** | 写时不跨；GPU fetch 时跨 |
| `cudaGraphInstantiate`（一次） | CPU 构建 + 上传 | 全部节点的 QMD/参数/依赖结构 | **vidmem (HBM)** | 跨（一次性） |
| 每次 `cudaGraphLaunch` | CPU store | 一条 graph-launch 命令 | sysmem pushbuffer | doorbell 跨；GPU 遍历 QMD 在 vidmem 内部，不跨 |
| graph 参数 patch（`cudaGraphExecKernelNodeSetParams`） | CPU/driver | 修改已烘焙 QMD 的少数字段 | vidmem | 跨（少量小写） |

pushbuffer 默认放 sysmem 的原因（libcuda 闭源策略，不可用户调节，但动因明确）：

- CPU 是写方且写极频繁（训练循环每秒几万次 launch），写本地内存零链路开销；
- GPFIFO/USERD 必须 CPU 可写，pushbuffer 同层管理最简单；
- vidmem（192GB HBM）是稀缺资源，要留给权重/激活/优化器状态；
- GPU fetch 是 DMA read，不占 CPU；GB200 的 C2C 一致性还省掉了 PCIe 平台所需的 WC flush/`sfence` 序列。

术语说明：vidmem 是逻辑/驱动术语（GPU 本地设备内存，`cudaMalloc` 的地址空间，nvidia-smi 显示为 FB memory）；GB200 上物理实现为 HBM3e（8 stack，192 GB/GPU，~8 TB/s）。Grace 侧 LPDDR5X sysmem（~480 GB）不属于 vidmem，但因 C2C cache 一致性可被 GPU 直接寻址——这是 CPU offload 时 kernel 直读 host 数据的硬件基础。

## 2. Eager 模式与 Graph 模式的通信过程

### 2.1 Eager 模式

每个 kernel 独立走一遍 1.3 节的 4 类通信：N 个 kernel = 4×N 次通信交互，每次 launch 都要付出完整的 CPU 软件路径（打包 QMD、更新 GPFIFO、敲 doorbell）+ 跨链路 fetch。

### 2.2 CUDA Graph 模式

capture/instantiate 与 replay 分离：

1. **`cudaGraphInstantiate`（一次性）**：driver 把 capture 到的全部 kernel 节点的 QMD、参数段、节点依赖构建成 GPU 前端可直接遍历的执行结构，上传到 **vidmem 常驻**。Bridge 中对应 `create_cudagraphs()`（`train.py:338-350`）与 `CUDAGraph.cpp:241` 的 `cudaGraphInstantiateWithFlags`。
2. **每次 replay（`cudaGraphLaunch`，`CUDAGraph.cpp:268`）**：CPU 只往 sysmem pushbuffer 写一条很小的 graph-launch 命令 + 敲一次 doorbell；GPU 前端在 **vidmem 内部**逐节点读取预烘焙的 QMD 自行 dispatch，不再跨链路 fetch 每个 kernel 的命令。

即：**N 个 kernel 的 4×N 次通信坍缩为约 2 次跨链路交互**（一次 doorbell + graph 内部遍历）。

### 2.3 对比

| 维度 | Eager | CUDA Graph |
| --- | --- | --- |
| QMD 生成 | 每次 launch CPU 就地写 sysmem | instantiate 时一次性烘焙进 vidmem |
| 跨链路 fetch | 每 kernel 一次 | replay 时无（GPU 内部遍历） |
| doorbell | 每 kernel 一次 | 每 graph 一次 |
| CPU 软件路径 | 每 kernel 完整 driver 路径 | replay 仅一条 graph-launch 命令 |
| 代价 | — | vidmem 常驻（graph 结构 + 中间 buffer 钉住不复用），capture 一次性耗时，shape 必须静态 |

### 2.4 Bridge 中的配置与实测

```python
cfg.model.cuda_graph_impl = "transformer_engine"   # 或 "local" + full_iteration
cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]  # MoE 示例
cfg.model.cuda_graph_warmup_steps = 3
cfg.model.use_te_rng_tracker = True
cfg.rng.te_rng_tracker = True
```

skill 实测数据（`skills/nemo-mbridge-perf-cuda-graphs/card.yaml`）：

- launch-bound workload 收益显著：Qwen3-30B-A3B（TP2PP2EP4, 2×H100 节点）step time 623→484ms（-22%）；GPT-OSS-20B 467-520→391-399ms（-16~24%）
- 非 launch-bound workload 收益为零甚至为负：Qwen3-30B-A3B H100 BF16 alltoall 短测 replay 42.00s vs eager 41.36s
- **"graph 有没有收益"本身就是 launch 开销是否为瓶颈的判据**
- 内存代价：TE-scoped 增加数 GB，full-iteration 峰值可达 1.5~2×（GPT-OSS-120B 因此 OOM）；`_delete_cuda_graphs()`（`train.py:1414`）负责释放 vidmem 常驻结构
- CPU offloading 与 CUDA graphs 互斥（MCore `transformer_config.py:1907`）——offload 实验场景无法用 graph A/B 法测量 launch 开销

## 3. 命令面 PBDMA vs 数据面 CE：不阻塞，但有带宽竞争

### 3.1 两套硬件独立

nvbandwidth 中的 CE（Copy Engine，`CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT`，`host_to_device_memcpy_ce` 所测引擎）是**数据面**批量搬运引擎，服务于 `cudaMemcpy/Async`、H2D/D2H/P2P 大块传输。

命令 fetch 由 GPU **Host Interface 中的 PBDMA（Push Buffer DMA）单元**完成——**控制面**专用取指引擎，服务 GPFIFO channel，与 CE 是不同的硬件 requester：

```
                     ┌──────────────── GPU 前端 ────────────────┐
CPU doorbell ══C2C══▶│ Host Interface                           │
                     │   └─ PBDMA (多个) ──fetch──▶ sysmem      │  ← 控制面：取 GPFIFO/pushbuffer/QMD
                     │        └─▶ Command Processor ─▶ CWD ─▶ SM│
                     │   CE × N (asyncEngineCount)              │  ← 数据面：cudaMemcpy 批量搬运
                     └──────────────────────────────────────────┘
```

层级关系：**CE 本身也被 pushbuffer 驱动**。`cudaMemcpyAsync(H2D)` 同样是 driver 往某 channel 的 pushbuffer 写一条 CE method（LAUNCH_DMA 类命令），由 PBDMA 取回解析后才派给 CE 执行。PBDMA 在 CE 的上游，命令 fetch 不可能排 CE 的队。

### 3.2 为什么不会被 H2D 传输阻塞

- **不同队列、不同引擎**：PBDMA fetch 与 CE 传输是并发 requester，链路上无共享 FIFO，无 head-of-line blocking；
- **C2C 为 NVLink 协议、多 virtual channel**：小的 request/response 读（fetch 几百 B~几 KB）与大块 posted write 流走不同虚通道/流控类；
- **前端预取**：PBDMA 超前 fetch pushbuffer 段，流水线可吸收部分延迟；
- **stream 内顺序不靠 fetch 阻塞实现**：同一 stream 中 memcpy H2D 后跟 kernel，kernel 的 QMD 早已被 fetch 进 GPU，先后顺序由 GPU 内部 semaphore/依赖逻辑（CWD 等待）保证。

### 3.3 带宽竞争的来源

物理资源共享，极端情况下命令面延迟会有 µs 级膨胀（竞争，非阻塞）：

| 共享资源 | 影响 |
| --- | --- |
| C2C 链路带宽（900 GB/s） | CE 将 H2D 打到接近饱和（如 offload 参数预取风暴）时，doorbell/fetch/semaphore 小事务延迟略升 |
| GPU 内部 fabric / HBM 控制器 | CE 写 vidmem、PBDMA 读命令、SM 读数据共享内部交叉开关 |
| Grace 侧 LPDDR5X 带宽 | CE 读 host 内存（H2D 源头）与 CPU 写 pushbuffer、spin-poll semaphore 共享 CPU 内存带宽（offload 重负载下易被忽略） |

量级判断：命令面流量为 KB 级/launch，数据面为 GB 级/step，相差约 6 个数量级，正常训练下竞争可忽略；仅在 DeepSpeed offload 等 C2C 长期高水位场景值得实测。

## 4. 测量方法

### 4.1 CPU 侧单次 launch 成本（纯软件路径）

```python
import torch, time
x = torch.zeros(1, device='cuda')
for _ in range(100): x.add_(0)          # warmup
torch.cuda.synchronize()
N = 5000                                 # 不要太大，避免 launch queue(~1024) 打满反压 CPU
t0 = time.perf_counter()
for _ in range(N): x.add_(0)
t1 = time.perf_counter()
torch.cuda.synchronize()
print(f"CPU-side: {(t1-t0)/N*1e6:.2f} us/launch")
```

### 4.2 完整往返延迟（doorbell → GPU 执行 → CPU 感知）

```python
torch.cuda.synchronize()
t0 = time.perf_counter()
x.add_(0)
torch.cuda.synchronize()                 # ≈ launch + C2C doorbell + kernel + 完成通知
t1 = time.perf_counter()

# 配合 CUDA event 分离 GPU 时间线：
s, e = torch.cuda.Event(True), torch.cuda.Event(True)
s.record(); x.add_(0); e.record(); torch.cuda.synchronize()
gpu_us = s.elapsed_time(e) * 1000        # wall - gpu ≈ launch 延迟 + sync 开销
```

### 4.3 Nsight Systems 定位 launch gap（推荐）

```bash
nsys profile --trace=cuda,nvtx -o launch_test python your_bench.py
nsys stats launch_test.nsys-rep
```

看三个指标：

- CPU 线程上 `cudaLaunchKernel` API 行的持续时间 = CPU 软件成本；
- API 结束 → GPU 上 kernel 开始的时差 = doorbell/fetch 链路延迟（可跨 GB200 vs PCIe 平台对比，直接量化 C2C 控制面）；
- GPU timeline 相邻 kernel 间的 gap：若 gap 规则出现且 ≈ 单次 launch 成本，workload 即 **launch-bound**。

判别表：

| 现象 | 结论 |
| --- | --- |
| kernel 间 gap ≈ launch 成本，CUDA graph 显著加速 | launch-bound，CPU→GPU 控制面通信是瓶颈 |
| kernel 本身变慢（offload 时），graph 无收益 | C2C 数据面带宽/延迟瓶颈（访存），不是 launch 问题 |
| `cudaLaunchKernel` API 时长远大于 gap | CPU 软件路径（dispatcher/driver）主导，与链路无关 |

### 4.4 CUDA Graph A/B 量化端到端影响

graph replay 把 N 次 launch 压成 1 次提交，eager 与 replay 的 steady-state step time 差值 ≈ launch 相关 CPU-GPU 交互总开销（capture step 不计入计时）。Bridge 配置见 2.4 节。

### 4.5 命令面 vs 数据面竞争实验（GB200）

```bash
# 终端 A：用 CE 持续打满 H2D（数据面负载）
./nvbandwidth -t host_to_device_memcpy_ce -i 60

# 终端 B：同时跑 4.2 的 launch 延迟探针
python launch_latency_probe.py
```

- 差值 ≈ 0（几个 % 以内）→ 证实命令面与数据面无阻塞、竞争可忽略；
- 差值显著 → 带宽竞争，可换 `-t host_to_device_memcpy_sm`（SM 拷贝，不占 CE 但占 SM 和链路）对照，或降低 CE 并发度找拐点；
- 先用 `query_async_engines` 确认 B200 的 `asyncEngineCount`（CE 数量），多 CE 并发才容易打到竞争水位；
- nsys 中 H2D CE 传输显示在独立 Memcpy 轨，与 Compute 轨时间重叠即证明 CE 与 kernel 执行并发；对比饱和/空闲时段 launch gap 是否变宽。

### 4.6 C2C 链路基线

- `nvidia-smi topo -m`：确认 CPU↔GPU 为 NV18/C2C 而非 PIX/PHB；
- `nvbandwidth`：host↔device memcpy 带宽基线（offload 场景 kernel 读 host 数据受此约束）；
- `ncu`：kernel 访存中 remote/host 内存占比（区分 launch 瓶颈与访存瓶颈）。

## 参考锚点

| 内容 | 位置 |
| --- | --- |
| PyTorch launch 路径（`<<<>>>` → cudaLaunchKernel） | `PyTorch/aten/src/ATen/native/cuda/`，stream 封装 `PyTorch/c10/cuda/CUDAStream.h` |
| driver API 封装（不含 launch） | `PyTorch/c10/cuda/driver_api.h` |
| `cudaGraphLaunch` / instantiate | `PyTorch/aten/src/ATen/cuda/CUDAGraph.cpp:241,268` |
| CE 定义与查询 | `NVLINK-C2C/nvbandwidth-v0.8/query_async_engines.cpp` |
| Bridge graph 创建/清理 | `megatron-bridge/src/megatron/bridge/training/train.py:231-255,338-350,1414` |
| graph 实测数据 | `megatron-bridge/skills/nemo-mbridge-perf-cuda-graphs/card.yaml` |
| CPU offload 与 graph 互斥 | MCore `transformer_config.py:1907` |

> 注：libcuda/driver 内部实现闭源，pushbuffer 放置策略等细节基于公开文档、open-gpu-kernel-modules 行为与观测推断；QMD/PBDMA/CWD 命名沿用 NVIDIA 架构文档惯例。
