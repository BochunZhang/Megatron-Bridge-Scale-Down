# NVLS-based 集合通信

NCCL 版本: 2.29.7 (开发机版本) & 2.32.3 (最新版本)

详细分析见 [nccl-2.29.7-nvlink-domain-collective-communication.md](nccl-2.29.7-nvlink-domain-collective-communication.md)

场景: 通信组的 P 个 rank 都在同一个 NVL72 内 (MNNVL 把整个 clique 视为一个节点, `nNodes == 1`)



发展历史

- NCCL 2.17 首次引入 nvls-based collective communication，在数据通路上利用了 NVSwitch 的硬件聚合与多播能力实现了 nvlink 域内的高效集合通信。

- NCCL 2.27 首次引入 symmetric memory (对称内存)，提供了 AllReduce、AllGather、ReduceScatter 等集合通信的 symmetric kernel 实现。

- NCCL 2.30 (最晚 2.30.4，2.29.7 中没有) 引入 TMA 版 symmetric kernel：`AllGather_TmaST` / `AllGather_TmaSTMC` / `ReduceScatter_TmaLD` / `AllReduce_RSxTmaLD_AGxTmaST`，其中只有 `AllGather_TmaSTMC` 写 multicast address，其余都是单播

- NCCL 2.32.3 中 CE AllGather 新增 multicast 分支 (2.30.4 中 CE 只用 multicast address 写同步 flag，数据仍是单播)



# PTX 核心指令

multicast & in-network reduce

- `multimem.st` rank r 向 nvswitch 发送报文，nvswitch 检测到 multicast address，将数据复制并发送给全部 rank (包括 rank r 自己)

- `multimem.ld_reduce` rank r 向 nvswitch 发起请求，nvswitch 读取各 rank 的数据，在 nvswitch 上聚合后返回给 rank r

- 两个指令操作的都是 multicast address

- multicast address 只是一段虚拟地址，不占物理显存；它背后是每个成员 GPU 同一偏移处的物理内存，各卡的副本不会自动保持一致

- multimem.st 需要 sm 先把数据加载到寄存器再发给 nvswitch，nvswitch 把数据写到每个成员 GPU 同一偏移处的 HBM

- multimem.ld_reduce 由 nvswitch 读取每个成员 GPU 同一偏移处的数据，归约结果存储到发起线程的寄存器上，要进入 HBM 还需要再发一条 st

- 只有 SM 能发 `multimem.ld_reduce`，所以只有 SM 能利用 nvswitch 做归约；TMA 和 CE 只能把 multicast address 当作写目标



multimem.st 直接发送 data & address 给 nvswitch 即可，nvswitch 检查到 multicast address 就会广播。同理，TMA / CE 只要写目标是 multicast address，也会被广播

- CE 通过 `cudaMemcpyBatchAsync` 提交的拷贝任务目标指向 multicast address，从而完成广播 (2.32.3)

- TMA 分两步：lane 0 先用 `memcpy_async_tx` 把 HBM 数据加载到 shared memory，再执行 `cp.async.bulk.global.shared` 把 shared memory 写到 multicast address，触发广播 (TMA 不能直接 global → global)



几种 send/recv 操作的差别

- sm + LSU (load store unit): 线程执行 ld 把 HBM 数据加载到寄存器，再执行 st 由 LSU 把 data & address 发往内存子系统 / NVLink；每个线程每条指令 4–16B

- sm + TMA: sm 把搬运任务卸载给 TMA，每个 warp 只有 lane 0 发一条指令搬一整个 tile；load 只能 global → shared memory，store 只能 shared memory → global (e.g. 对端地址 / multicast address)

- ce (copy engine): Host API 提交的任务，直接放在 CE 上，等待 stream 上的依赖满足后发起，不占用 SM



# Legacy Kernel

- FIFO slot: NCCL 在 HBM 维护的中转缓冲区。每卡共 `nHeads × 2 (A: reduce 方向 / B: gather 方向) × nvlsChannels` 条 FIFO，每条 FIFO 有 8 slot (对应 NCCL_STEPS = 8)

- 每条 FIFO 有 UC (只指向本卡) 和 MC (指向所有卡) 两个虚拟地址，multicast address 只映射在 FIFO 上

- Legacy Kernel 接收的输入是用户的 send_buff & recv_buff，MNNVL 下不能把它们注册为 multicast address (`nvlsRegSupport = 0`)，所以必须经过 FIFO 中转

- head: 每个 head 负责 1/nHeads 的数据，NVL72 上 nHeads = min(P, 32)

- channel: 每个 channel 独占一个 sm (1 CTA)，SM100 单 NVLink 域时为 24 个，与 P 无关



## AllReduce

AllReduce = ReduceScatter + AllGather

- ReduceScatter 只能处理 multicast address 上的数据，而 send_buff 没有 multicast 映射，因此需要先复制 send_buff 数据到 reduce FIFO slot (A)

- AllGather 的广播结果存在 gather FIFO slot (B)，因此要将数据复制到 recv_buff



Legacy Kernel 中，每个 channel 使用 1 CTA (640 线程)，分为 20 warp。P > 6 时分为 8 scatter & 7 gather & 5 reduce (P ≤ 6 时 reduce 为 7 warp)

- scatter[0 - 7] 负责 wait send & copy & post send

    - scatter[0] 前 nHead lane 各自检查一条 reduce FIFO slot 是否已被 head 释放 `WaitSend[h=lane]` → 即 8 step 前写入这个 slot 的数据已被 head 用 ld_reduce 读完

    - scatter[0] - scatter[6] 在全部 nHead slot 都空闲后 barrier，统一开始从 send_buff copy 数据到 FIFO slot

    - scatter[7] 后 nHead lane 在 copy 完成后先 fence，再写本卡 tail，标记 slot 数据已写好 `PostSend[h=lane]`

- gather[8 - 14] 负责 copy 数据到 recv_buff

    - gather[8] 前 nHead lane 检查 gather FIFO slot 是否准备好数据

    - gather[8] - gather[14] 在全部 nHead slot 都准备好数据后完成 barrier，复制数据到 recv_buff

    - gather[14] 后 nHead lane 写本卡 head，标记 slot 数据已读走

- reduce[15 - 19] 负责做 multimem.ld_reduce & multimem.st，是唯一跨卡的组

    - reduce[15].lane[0] 通过 multimem.ld_reduce.min 读取全部 rank 的 tail，等待全部 rank 的 reduce FIFO slot 都准备好了 reduce 数据

    - reduce[15].lane[1] 通过 multimem.ld_reduce.min 读取全部 rank 的 head，等待全部 rank 的 gather 已经把 gather FIFO slot 复制到了 recv_buff (slot 可以覆盖)

    - reduce[15-18] 每个线程先做 8 次 multimem.ld_reduce，然后做 8 次 multimem.st，将 reduce 结果广播出去；st 是 posted write，发射后即可开始下一个 hunk 的 ld_reduce

    - reduce[19].lane[30] 跨卡同步，写 MC 地址的 head，释放全部 rank 的 reduce FIFO slot → 和 scatter 联动

    - reduce[19].lane[31] 跨卡同步，先 fence 再写 MC 地址的 tail，通知全部 rank 的 gather FIFO slot 数据已到达 → 和 gather 联动



几点观察

1. scatter & gather 搬运本 rank 的全部数据 (N)，分配了更多 warp；reduce 只处理本 rank 作为 head 负责的 1/nHeads，占用资源更少

2. 流程上需要跨卡和本地的 credit 握手 (head/tail 计数器，不是 barrier)

    1. 需要 FIFO 中转，是因为 multicast address 只映射在 FIFO slot 上

    2. 每个 step 都要握手，是因为 FIFO 只有 8 slot 需要循环复用：既要确认 slot 数据好了没有，也要确认 slot 能不能被覆盖

3. legacy kernel 允许将 recv buffer / send buffer 注册为 multicast address，绕开 FIFO slot，但 MNNVL 下不支持 (`init.cc:1024`)，只在 compute tray 内部 (≤ 4 卡) 有效，不如直接看 symmetric kernel 的逻辑

4. 8 slot 支持流水线，scatter 最多可以比 reduce 快 8 step

5. 每个 channel 每个 step 每个 head 有 4 次跨卡同步 (reduce[15].lane[0/1]、reduce[19].lane[30/31])



## AllGather

Legacy Kernel 中，每个 channel 使用 1 CTA (640 线程)，分为 20 warp：gather[0-7] 8 warp & bcast[8-19] 12 warp

**限制**：要求 P ≤ 32 (`enqueue.cc:1914`)，NVL72 上 72 卡的通信组用不了

AllGather 只有 Gather & Bcast 两个角色，只用 B 方向的 FIFO

- Bcast 先执行 8x 加载 send_buffer 数据，再执行 8x multimem.st，交替循环，结果存储在全部 rank 的 gather FIFO slot

- Gather 将数据从本卡 P 条 gather FIFO slot 存储到 recv_buffer



## ReduceScatter

Legacy Kernel 中，每个 channel 使用 1 CTA (640 线程)，分为 20 warp：scatter[0-7] 8 warp & reduce[8-19] 12 warp

**限制**：要求 P ≤ 32 (`enqueue.cc:1914`)，同 AllGather

ReduceScatter 只有 Scatter & Reduce 两个角色，只用 A 方向的 FIFO，没有 multimem.st

- Scatter 将数据从 send buffer 的 P 段分别搬运到本卡 P 条 reduce FIFO slot

- Reduce 先执行 8x multimem.ld_reduce，然后执行 8x 将收到的数据从 sm 寄存器写入本卡 recv buffer，交替执行



# Symmetric Kernel

symmetric memory 使用连续的虚拟地址把所有 rank 的 buffer 平铺起来

- 每个 rank 在自己的虚拟地址空间预留 `P × bigSize` 的连续区域 `lsaFlatBase`，`bigSize = alignUp(max(各卡 HBM 容量), 4GB)`，与 buffer 大小无关

- 每个 window 注册时分配一个所有 rank 一致的偏移 `bigOffset`，rank r 的 buffer 映射到 `lsaFlatBase + r × bigSize + bigOffset`

- e.g. 4 gpu，bigSize = 192GB，某个 window 的 bigOffset = 1MB

    - gpu 0: lsaFlatBase + 0 × 192GB + 1MB

    - gpu 1: lsaFlatBase + 1 × 192GB + 1MB

    - gpu 2: lsaFlatBase + 2 × 192GB + 1MB

    - gpu 3: lsaFlatBase + 3 × 192GB + 1MB

- 可以根据初始地址和偏置，计算每个 rank 的地址，kernel 访问 gpu 2 的地址会通过 nvlink 访问 GPU 2 的 HBM

- 同时每个 window 绑定到一个 multicast 对象，multicast address = `mcBase + bigOffset`



- 非 LL kernel 不需要 FIFO 中转，可以直接操作 send_buffer & recv_buffer，只在 kernel 首尾各做一次 LSA barrier，运行过程中不需要同步 → 效率显著高于 legacy kernel

- LL kernel (`AllReduce_AGxLL_R` / `AGxLLMC_R`、`AllGather_LL` / `LLMC`、`ReduceScatter_LL`) 用于小消息，数据要经过 LL scratch 中转，没有 barrier，靠每个 16B 包里的 epoch flag 同步

- symmetric kernel 支持非 multicast 实现版本，依赖 p2p unicast 实现

- symmetric kernel 不使用 LL / LL128 / Simple 协议枚举，只是按实现方式分为 LL 风格 & Simple 风格

- 每个 CTA 512 线程 (16 warp)，所有线程都发 multimem 指令，没有角色分工；数据以 4KB chunk 为单位，按全局 warp 号 `W = rank + P × (block + nBlocks × warp)` 交错分给各 rank

- 限制

    - multicast 版本要求 P > 2

    - multicast 版本每卡只有 `divUp(32, P)` 个 CTA，P = 72 时只有 1 个

    - 非 LL kernel 要求注册：AllReduce 要求 send & recv 都注册，AllGather 要求 recv 注册，ReduceScatter 要求 send 注册；否则只能走 LL kernel 或回退到 legacy kernel



AllReduce (`AllReduce_RSxLDMC_AGxSTMC`)

- 每个线程执行 multimem.ld_reduce 读取自己负责的 chunk → send_buffer 注册为 multicast address，nvswitch 直接读取并归约

- 同一个线程执行 multimem.st 将寄存器中的结果广播出去 → recv_buffer 注册为 multicast address，nvswitch 直接存入

- 小消息走 `AllReduce_AGxLLMC_R`：用普通 st 把 LL 包写到 LL scratch 的 multicast address，每个 rank 本地归约全部数据



AllGather (`AllGather_STMC`)

- 每个线程读取本卡 send_buffer 数据到寄存器

- 每个线程执行 multimem.st 将加载的数据广播出去 → recv_buffer 注册为 multicast address，nvswitch 直接存入

- 2.32.3 的 `AllGather_TmaSTMC`：每个 warp 的 lane 0 用 TMA 把 send_buffer 加载到 shared memory，再用 `cp.async.bulk` 写到 recv_buffer 的 multicast address



ReduceScatter (`ReduceScatter_LDMC`)

- 每个线程执行 multimem.ld_reduce 读取自己负责的 chunk → send_buffer 注册为 multicast address，nvswitch 直接读取并归约

- 每个线程将寄存器中的结果用普通 st 写入本卡 recv_buffer
