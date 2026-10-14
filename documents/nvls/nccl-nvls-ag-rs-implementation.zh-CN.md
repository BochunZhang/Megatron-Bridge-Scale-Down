# NCCL NVLS All-Gather / Reduce-Scatter 实现与实验依据

日期：2026-10-08。本文是[原始重叠报告](nvls-fsdp3-overlap-report.zh-CN.md)的源码核验补充；实验入口见 [GB200 实验 README](../../examples/gb200/nvls-base-ccl/README.md)。本文没有 GPU 实测结果。

主审计版本为 NVIDIA NCCL **v2.32.3-1**，commit [`12df1a11afad322be5a204a2db890161cbf8131d`](https://github.com/NVIDIA/nccl/tree/12df1a11afad322be5a204a2db890161cbf8131d)。同时核对实验最低版本 **v2.28.9-1**，commit [`dbc86fd06e8b0c4517b95d8958a09ccacf9520c9`](https://github.com/NVIDIA/nccl/tree/dbc86fd06e8b0c4517b95d8958a09ccacf9520c9)。以下源码链接固定 commit，不将移动的 `master` 当成部署版本。

## 1. 结论与术语

NCCL 经典 `NVLS + Simple` 算法已经同时实现完整 AG 和 RS。AG 的数据面由每个 rank 广播自己的 shard 组成；RS 的数据面由每个 rank 对属于自己的输出 shard 发起跨副本归约读取组成。两者分别利用 `multimem.st` 和 `multimem.ld_reduce`，并不要求应用调用多次 `ncclBroadcast` / `ncclReduce`。

| 层次 | 对象或操作 | 实验中怎样计数 |
|---|---|---|
| Fabric | multicast team / slot 配额 | 平台管理面实际已分配数量；有配额、成员和拓扑约束 |
| CUDA | `cuMulticastCreate` 创建的 object | 分别数 created、added、bound、mapped、可用对象 |
| NCCL | communicator、NVLS resources、user-buffer 注册 | 记录数量和版本；不能直接换算成硬件 slot |
| GPU kernel | channel / CTA / chunk / FIFO step | 属于并行度、流水与同步，不能当成 multicast team |

GB200 partition guide 第 9 章给出 **1,024 个全局 multicast teams，由各 partition 共享**，并通过 `multicastGroupsLimit` 预留，通过 `GetPartitionInfoList.allocatedMulticastGroups` 查询已消耗数量。这不是每 GPU 1,024 个独占 slot，也不是 1,024 路归约带宽。[官方 partition guide，PDF 第 56–57 页](https://docs.nvidia.com/multi-node-nvlink-systems/partition-guide-v1-2.pdf#page=56)

## 2. 从 Host API 到 NVLS kernel

固定版本中的调用与选择流程如下；两次 API 的 `count` 都表示 **每 rank shard 的元素数**，AG 接收总量和 RS 发送总量都是 `P × count`。

```text
ncclAllGather / ncclReduceScatter
  → ncclInfo → ncclEnqueueCheck
  → task preparation / tuning（拓扑、dtype、op、buffer 注册、算法白名单）
  → NVLS transport / channel setup；必要时建立 multicast 映射
  → ncclDevWorkColl（regUsed、sendbuff、recvbuff、channel/chunk）
  → RunWorkColl<AllGather 或 ReduceScatter, ..., NVLS, SIMPLE>
  → Primitives → reduceCopy → multimem store / reduce-load
```

2.32.3 的 API 位于 [`collectives.cc`](https://github.com/NVIDIA/nccl/blob/12df1a11afad322be5a204a2db890161cbf8131d/src/collectives.cc#L130)；pre-tuning、`ncclTuningCompute`、classification、post-tuning 的顺序见 [`task_prep.cc`](https://github.com/NVIDIA/nccl/blob/12df1a11afad322be5a204a2db890161cbf8131d/src/enqueue/task_prep/task_prep.cc#L11)。2.28.9 使用旧版 [`enqueue.cc`](https://github.com/NVIDIA/nccl/blob/dbc86fd06e8b0c4517b95d8958a09ccacf9520c9/src/enqueue.cc) 和 [`graph/tuning.cc`](https://github.com/NVIDIA/nccl/blob/dbc86fd06e8b0c4517b95d8958a09ccacf9520c9/src/graph/tuning.cc#L273)，核心数据面相同。

NVLS 并非任何输入都能选中：需要 multicast 能力、有效 NVLS 拓扑，以及受支持的 reduction / dtype。经典 NVLS 仅使用 Simple 协议；NVLSTree 在这两版中只用于 AllReduce，不能用 `NVLSTree` 冒充 AG/RS。2.32.3 的 AG/RS 还受 `NCCL_MAX_NVLS_ARITY=32` 约束；4/8/16 均在范围内。跨多个 NVLink 域的经典 NVLS AG/RS 需要额外 CollNet 支持，本实验限定同一 NVLink domain/clique。[NVLS tuning](https://github.com/NVIDIA/nccl/blob/12df1a11afad322be5a204a2db890161cbf8131d/src/tuning/nvls.cc#L25)、[arity](https://github.com/NVIDIA/nccl/blob/12df1a11afad322be5a204a2db890161cbf8131d/src/include/device.h#L216)

### 如何强制路径并识别失败

经典路径使用：

```bash
unset NCCL_SYM_KERNEL
export NCCL_NVLS_ENABLE=1
export NCCL_ALGO=NVLS
export NCCL_PROTO=Simple
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH,TUNING,COLL,NVLS,REG
```

`NCCL_NVLS_ENABLE=1` 本身只开启/要求 NVLS 能力，不能证明某次 collective 选了 NVLS；`NCCL_ALGO=NVLS` 才限制经典算法候选。2.32.3 的 parser 在设置算法时清空其他算法和 symmetric 候选，除非同时显式设置 `NCCL_SYM_KERNEL`；没有有效候选时返回错误，不应接受 Ring 结果作为 NVLS 结果。2.28.9 和 2.32.3 的 `NCCL_SYM_KERNEL` parser 不同，不用 `NCCL_SYM_KERNEL=0` 充当跨版本通用禁用开关。[候选过滤](https://github.com/NVIDIA/nccl/blob/12df1a11afad322be5a204a2db890161cbf8131d/src/tuning/cost_model.cc#L310)、[无候选时报错](https://github.com/NVIDIA/nccl/blob/12df1a11afad322be5a204a2db890161cbf8131d/src/tuning/tuning.cc#L299)

每个测量配置应有三层证据：初始化显示 NVLS multicast support / channels；每次被测 collective 的 tuning 或 dispatch 确认 NVLS/Simple；Nsight 时间线看到相应 NVLS kernel。新旧版日志格式不同，不能仅搜索任意一行含 `NVLS` 就宣布验证成功。TRACE 用于短验证运行；正式计时降低日志级别，并保存相同环境与版本。

## 3. Multicast 地址如何支持 AG 和 RS

一个 P-GPU multicast object 在每张 GPU 上有本地物理副本。每张 GPU 可以同时建立本地 unicast 映射 UC 和面向所有副本的 multicast 映射 MC。MC 地址不是普通指针：只能用 multimem 指令访问，不能用普通 `ld` / `st` 直接读取或写入。[CUDA multicast memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html#multicast-memory-sharing)

设每 rank shard 有 `N` 个元素：

```text
AG: rank r 读取 send_r[j]
    → multimem.st recv_mc[r*N+j]
    → 所有 rank 的 recv[r*N+j] = send_r[j]

RS: 各 rank 提供 send_r[q*N+j]
    → rank q 发起 multimem.ld_reduce.add send_mc[q*N+j]
    → rank q 将返回寄存器写到本地 recv_q[j]
```

RS 是由结果接收方发起的 **pull**。完整 RS 中，每个 rank 都对自己的输出区间发起读取，同时向别人的归约提供其他区间。不能把它简化为只有 GPU0 收结果的单 root 实验。

另外，`multimem.red` 是把调用方提供的 operand 分别应用于所有副本的原子归约；它不等价于“先把各副本归约成一个结果，再广播结果”。跨副本 reduce-load 的结果在发起线程的寄存器中，若需要 AllReduce，才再通过 multicast store 分发。[PTX multimem 语义](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-multimem-ld-reduce-multimem-st-multimem-red)

### 3.1 经典 AG：staging 与注册 buffer

[`all_gather.h` 的 NVLS specialization](https://github.com/NVIDIA/nccl/blob/12df1a11afad322be5a204a2db890161cbf8131d/src/device/all_gather.h#L254) 在单 NVLink 域内分两组线程工作。

当 `work->regUsed == 0`：broadcast 线程执行 `prims.send`，用 `ProtoSimple<..., MultimemSrcs=0, MultimemDsts=1>` 把本 rank 输入写到该 rank/head 对应的 MC staging 区；各 rank 的 gather 线程从 UC 副本读取各 head 的 chunk，按 rank 顺序拷到最终 `recvbuff`。这仍是 NVLS broadcast，只是增加 staging→用户输出的本地 HBM 拷贝。

当 `regUsed == 1`：输出 buffer 可以直接经 MC 地址写入，`directSend(inpOffset, inpOffset + rank*count, nelem)` 放到正确 shard。gather 分支保留零长度 scatter/gather 作为同步，省去数据拷贝。所谓 “zero-copy” 是省掉 NCCL 中间 staging，并非没有 GPU load/store、同步或 HBM 流量。

底层 4/8/16-byte store specialization 生成 `multimem.st.global.b32/b64/v4.f32`。[`op128.h`](https://github.com/NVIDIA/nccl/blob/12df1a11afad322be5a204a2db890161cbf8131d/src/device/op128.h#L412)

### 3.2 经典 RS：staging 与直接 reduce-load

[`reduce_scatter.h` 的 NVLS specialization](https://github.com/NVIDIA/nccl/blob/12df1a11afad322be5a204a2db890161cbf8131d/src/device/reduce_scatter.h#L232) 同样分 scatter 和 reduce 两组线程。

未注册时，scatter 线程把输入各输出 shard 写入相应 head 的 UC staging 区；reduce 线程使用 `ProtoSimple<..., MultimemSrcs=1, MultimemDsts=0>`，通过 `prims.recv` 从 MC 地址归约读取自己的 shard，然后写本地输出。注册成功时，scatter 保留为同步，`directRecvCopy(rank*count + outOffset, outOffset, nelem)` 直接读取用户输入的 MC 映射。

数值归约由 [`reduce_kernel.h`](https://github.com/NVIDIA/nccl/blob/12df1a11afad322be5a204a2db890161cbf8131d/src/device/reduce_kernel.h#L1120) 的 `applyLoadMultimem` specialization 生成，例如 FP32 sum 的 `multimem.ld_reduce.relaxed.sys.global.add.f32` / 向量形式；不是每个 GPU 在 SM 上串行加载 P 份数据做软件求和。

### 3.3 数据与 credit 不可混为一谈

经典 transport 给每个 head/channel 划分 reduce 和 broadcast 两个 half；`send[1]`/`recv[0]` 连接 reduce 的 UC 写入与 MC 读取，`send[0]`/`recv[1]` 连接 broadcast 的 MC 写入与 UC 读取。[buffer layout](https://github.com/NVIDIA/nccl/blob/12df1a11afad322be5a204a2db890161cbf8131d/src/transport/nvls.cc#L319)

除数据指令外，`Primitives::loadStepValue` 可以用 `multimem.ld_reduce.acquire.sys.global.min.u64` 读取所有副本的最小进度，确保参与者都就绪；`postPeer` 在发布 step 前进行 system-scope fence。因此 AG 虽不对 payload 求和，协议仍可能使用归约形式的 credit 指令。看到某条 reduce-load 不能单独证明运行了 RS。[`prims_simple.h`](https://github.com/NVIDIA/nccl/blob/12df1a11afad322be5a204a2db890161cbf8131d/src/device/prims_simple.h#L86)

原始 CUDA 实验还必须处理 UC/MC alias：同 kernel 内通过别名访问同一物理内存需 `fence.proxy.alias`，同时用 release/acquire 与跨 GPU barrier 确保输入可见、输出完成、下一轮不覆盖仍在使用的数据。单独一个 fence 不等于所有 rank 已到达 barrier。[CUDA virtual aliasing](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html#virtual-aliasing-support)

## 4. 注册用户 buffer 与 symmetric kernels 是两层机制

`ncclMemAlloc + ncclCommRegister` 为经典 NVLS 的 direct path 提供条件：每个 rank 都必须遵守注册、分配粒度和一致 offset 要求。不能一部分 rank 注册、一部分不注册；通信完成后先 deregister，再释放物理内存。[NCCL buffer registration](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html)

`ncclCommWindowRegister(..., NCCL_WIN_COLL_SYMMETRIC)` 则提供 symmetric window。2.28.9 的 scheduler 通过 window 标志进入独立的 symmetric kernel 路径，不能仅凭 `ncclCommRegister` 成功就把输出标为 `AllGather_STMC` / `ReduceScatter_LDMC`。[2.28.9 symmetric scheduler](https://github.com/NVIDIA/nccl/blob/dbc86fd06e8b0c4517b95d8958a09ccacf9520c9/src/scheduler/symmetric_sched.cc#L12)

2.32.3 的 `ncclSymkRun_AllGather_STMC_impl` 先 acquire barrier，随后 `bcastMultimem` 将输入放到 `output + rank*nAllElts`，最后 release barrier；`AllGather_LLMC` 将小消息数据和低延迟协议结合。`ncclSymkRun_ReduceScatter_LDMC` 先 acquire barrier，再从 `input.multimemPtr + rank*nAllElts` reduce-load 到本地输出，最后 barrier 结束。这些是不同于经典 `RunWorkColl` 的 kernel 与调度机制。[symmetric AG](https://github.com/NVIDIA/nccl/blob/12df1a11afad322be5a204a2db890161cbf8131d/src/device/symmetric/all_gather.cuh#L289)、[symmetric RS](https://github.com/NVIDIA/nccl/blob/12df1a11afad322be5a204a2db890161cbf8131d/src/device/symmetric/reduce_scatter.cuh)

本实验的 NCCL 基线使用经典 NVLS；原始 multicast 实验直接测试相同硬件原语。二者与 symmetric-window NCCL 是可比较但不可混名的三条路径。Blackwell 新版还存在 TMA/CE 变体，应按实际 kernel 标识分类，不能统称“用了 NVLS 所以路径完全相同”。

## 5. 同一 NVLink 域可以跨 OS 主机

原报告中“跨节点只能 NVLS + IB”的说法需要限定为**跨 NVLink domain/clique**。GB200 NVL 系统的多个 OS 主机可在同一 NVLink domain；NCCL 的 MNNVL 检查 fabric UUID、clique、fabric state、cuMem 与 FABRIC handle/IMEX 可用性，并通过 FABRIC handle 共享 multicast 对象。16 卡无需被放进一台 OS 主机。[`mnnvl.cc`](https://github.com/NVIDIA/nccl/blob/12df1a11afad322be5a204a2db890161cbf8131d/src/mnnvl.cc#L13)

但**经典 NVLS user-buffer registration 在审计的两版中都因 MNNVL 被关闭**：2.28.9 [`init.cc`](https://github.com/NVIDIA/nccl/blob/dbc86fd06e8b0c4517b95d8958a09ccacf9520c9/src/init.cc#L949) 与 2.32.3 [`init.cc`](https://github.com/NVIDIA/nccl/blob/12df1a11afad322be5a204a2db890161cbf8131d/src/init.cc#L1353) 都设置 `comm->nvlsRegSupport=0`。所以 8/16 卡跨主机运行中，即使应用申请注册成功，也应预期可能走经典 staging NVLS；必须把 `requested_registration` 和实际 direct/registered path 分开解释。这不等于 MNNVL 不支持 NVLS，也不能推广为 symmetric window 机制一律不支持 MNNVL。

实验保存每 rank 的物理 GPU 和 OS host 映射；fabric/clique 状态需结合 NCCL 日志和管理员查询另行保存。按 MPI rank 连续切半与交错切半可能跨不同 tray/链路。不能根据 `nvidia-smi topo -m` 的单机输出认定整个 16 卡分配属于同一 multicast clique。

## 6. `cuMulticastCreate`、绑定与 slot 消耗

一个可实际执行的对象按如下顺序建立：

1. 查询 `CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED`；查询 UC 和 MC 最小/推荐粒度。
2. leader `cuMulticastCreate` 指定 `numDevices`、对齐后的 size、handle type；导出对象供其他 rank 导入。
3. 每个 rank `cuMulticastAddDevice`，所有成员完成后同步。
4. 每个 rank `cuMemCreate` 本地物理内存、建立 UC 映射；用 `cuMulticastBindMem` 绑定到同一 MC offset。
5. `cuMemAddressReserve`、`cuMemMap`、`cuMemSetAccess` 建立 MC VA，执行数据与同步验证。
6. 通信全部结束后解除 MC 映射/绑定，释放所有 import/create handle、UC 映射与物理分配；只有完整生命周期结束才算资源释放。

跨主机需要 `CU_MEM_HANDLE_TYPE_FABRIC` 与正确 IMEX 权限；POSIX FD 的整数不能通过 MPI 广播后直接在其他进程当作同一个 FD 使用。本实验原始 API 路径使用 FABRIC handle，适用于 GB200 同一 NVLink fabric 的多进程场景。[CUDA Driver multicast API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__MULTICAST.html)、[NVIDIA MNNVL 示例](https://github.com/NVIDIA/multi-gpu-programming-models/tree/master/multi_node_p2p)

`cuMulticastCreate` 成功只说明对象描述创建成功，不说明所有硬件映射和物理 backing 都建立成功。资源不足可在 add、bind 或 map 阶段暴露；特别要保存 `cuMulticastBindMem` 的 CUDA 数字错误码、符号和错误字符串。`CUDA_ERROR_MULTICAST_RESOURCE_FULL` 或 OOM 都不能仅凭名称断言具体哪片 switch ASIC 的 SRAM 已满。需结合 phase、可用显存和管理面 team 增量判断。

容量实验至少区分 `create-only` 与完整 bind/map；达到预设上限且全部成功只能报告“可用数 ≥ 本次上限”，不能报告硬件最大值。每个独立对象由一个 leader 创建；P 个进程导入同一个对象仍只有一个逻辑对象，而不是 P 个 team。

### 版本为何影响一个 communicator 的资源成本

2.28.9 的 [`nvlsAllocateMem`](https://github.com/NVIDIA/nccl/blob/dbc86fd06e8b0c4517b95d8958a09ccacf9520c9/src/transport/nvls.cc#L223) 为 credit 和 data 分别建立 MC 对象，user-buffer 注册又有独立的资源生命周期；资源共享与延迟初始化使固定“每 comm 几个 slot”也不可靠。

2.32.3 的 [`ncclNvlsSetup`](https://github.com/NVIDIA/nccl/blob/12df1a11afad322be5a204a2db890161cbf8131d/src/transport/nvls.cc#L428) 把 credit、data、UB arena 切为同一个 MC object 的不同区间，data 可稍后绑定；[`ncclMcGroupBuildPartitions`](https://github.com/NVIDIA/nccl/blob/12df1a11afad322be5a204a2db890161cbf8131d/src/transport/multicast.cc) 管理创建、共享、粒度和映射。这是 **该 NVLS resource group 内** 的合并，不代表机器上的所有独立 communicator 自动共用唯一一个全局 MC 对象。[2.32.3 release notes](https://github.com/NVIDIA/nccl/releases/tag/v2.32.3-1)也明确说明降低了多 communicator 的 NVLS slot 消耗。

同样，增加 channel 数通常扩大同一个对象的 buffer 分片，而不是创建同样数量的 multicast teams。设 `C` 为 channels、`H` 为 heads、`B = chunkSize × NCCL_STEPS`，经典 data staging 的每 rank 分配量为 `2 × C × H × B`，credit 为 `2 × C × H × 64` 字节（随后按分配粒度上取整）。这些是 GPU HBM 字节数，不能当成 switch scratch SRAM。[transport 分配公式](https://github.com/NVIDIA/nccl/blob/12df1a11afad322be5a204a2db890161cbf8131d/src/transport/nvls.cc#L297)

## 7. 4 / 8 / 16 卡实验怎样回答四个问题

| 总 GPU 数 | 完整 AG/RS 组 | 两个互斥组 | 同组 AG+RS |
|---|---|---|---|
| 4 | 4 | 2 + 2 | 两个独立 4-rank communicator / MC 对象 |
| 8 | 8 | 4 + 4 | 两个独立 8-rank communicator / MC 对象 |
| 16 | 16 | 8 + 8 | 两个独立 16-rank communicator / MC 对象 |

**单操作性能与行为。** 先测完整多 root AG、完整多输出 RS；扫描每 rank shard 大小，验证每 rank 的输出内容，区分 staged / requested registered / raw PTX 路径。低消息尺寸测延迟，大消息测带宽平台；至少使用多个大尺寸判断平台，不拿一个小消息点代表极限带宽。

**对象容量与带宽。** 一条轴只增加存活、已绑定但空闲的 MC 对象，保持一个固定工作对象、消息尺寸、CTA 数；另一条轴增加实际参与工作的对象，把固定总字节数分片到 K 个对象。再独立扫描 CTA/chunk 并行度。这样可区分“空闲 slot 占用影响”“更多并行工作影响”和“物理分配/缓存影响”。若 K=1 已到平台，继续增加对象数并不预期线性加速。

**两个互斥组相互影响。** 在同一 partition 中先只运行 A，再只运行 B，然后同时运行，分别测 AG+AG、RS+RS。每个 K 都用同样存活对象集合重测单组基线与并发值；连续切半和交错切半作为拓扑对照。对象不足导致初始化失败是容量现象，已成功初始化后的 slowdown 才是运行时争用证据。即使 GPU 完全不重叠，两个组也可能共享 switch 资源。

**同组 AG+RS overlap。** 同一 rank 集合内分别建立独立 AG、RS communicator/对象、输入输出及 stream，保留相同工作量的串行基线；同时观测每个操作时间和两者共同完成时间。同一 communicator 的双 stream 仅是排序控制组，不作为硬件并发能力结论。

### 计量定义

若每 rank shard 为 `S` 字节、组大小为 `P`、单次 collective 完成时间为 `T`，报告明确的口径：

```text
AG / RS logical aggregate bytes per rank = P*S
algorithm bandwidth                     = P*S/T
nccl-tests-style normalized busbw       = (P-1)*S/T
RS output bandwidth                     = S/T
```

归一化 busbw 不是物理 NVLink 端口测量；复制和交换机归约会改变数据路径。单个定向 reduce 的 `P*S/T` operand 口径也不能未经说明与全组总流量相加。FSDP 等量通信比较应保持 AG 总接收量和 RS 总发送量一致。

```text
T_serial_expected = T_AG + T_RS
T_ideal           = max(T_AG, T_RS)
speedup           = (T_AG + T_RS) / T_both
normalized_overlap = (T_AG + T_RS - T_both) / min(T_AG, T_RS)
group_slowdown     = T_group_concurrent / T_group_alone
```

`speedup` 的理想值为 `(T_AG+T_RS)/max(T_AG,T_RS)`；只有两者时长接近时才接近 2。保留负的 normalized overlap，它表示并发反而更慢。不能把全局 `T_both` 代替每组独立完成时间后声称测到了组间公平性。

## 8. overlap 的软件顺序边界

不同 communicator 的 collective 在所有 rank 上必须保持一致的 host 发起顺序。NCCL 2.26 起支持 `NCCL_LAUNCH_ORDER_IMPLICIT=1`，较新 CUDA 用 launch-completion event 排序 launch，从而允许两个 kernel 并行；这不要求第一个 kernel 已执行结束。[NCCL concurrent communicators](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#using-multiple-nccl-communicators-concurrently)

但是审计版本的 `getImplicitOrder` 对 **CUDA Graph capture 且 driver API version < 12090** 明确降为 execution serialization；所以 graph overlap 实验额外要求 driver ≥ 12.9，且 NCCL 构建使用支持 launch event 的 CUDA toolkit。仅看到两个 graph 分支不能证明实际 overlap。[2.32.3 launch ordering](https://github.com/NVIDIA/nccl/blob/12df1a11afad322be5a204a2db890161cbf8131d/src/enqueue/enqueue.cc#L1665)

保留 NCCL 的 graph ordering 默认值。不能通过 `NCCL_GRAPH_STREAM_ORDERING=0` 强行追求 overlap：当前文档要求关闭后应用保证同 GPU NCCL operation 不重叠。默认排序、版本实现、共享资源与 SM 调度都可能限制并发，应以 Nsight 时间线为准；NCCL 观察到串行只能说明本配置未 overlap，不能单独证明 NVSwitch 硬件不支持两种原语重叠。[graph ordering 契约](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-graph-stream-ordering)

原始 multicast 双 stream 实验用于进一步排除 NCCL 调度因素，但必须保留独立数据区和同步状态。最终只有 `T_both` 接近 `T_ideal`、输出正确且 kernel/NVLink 计数支持同一时间段工作，才能称为接近理想 overlap。两组带宽是否下降、对象数多少时开始下降、单对象能否到平台，都留给 GB200 实测回答。
