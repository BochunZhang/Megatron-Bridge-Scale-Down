# PTX ISA 9.4 中的 multimem 指令

日期：2026-10-12。

依据：

- PTX ISA **9.4**（CUDA 13.4，驱动 r615）。
- CUDA Driver API 中 Multicast Object Management 与 Logical Endpoint 两章（2026-09-14 版）。
- CUDA Programming Guide 的 VMM / Multicast 章节。

本文只讲**指令和硬件语义**。NCCL 怎样用这些指令实现 AllGather / ReduceScatter / AllReduce，见 [nccl-2.29.7-nvls-collectives.zh-CN.md](nccl-2.29.7-nvls-collectives.zh-CN.md)。交换机侧的资源（multicast group 配额、SHARP ALU、scratch SRAM），见 [nccl-nvls-resources.zh-CN.md](nccl-nvls-resources.zh-CN.md)。

本文结论来自文档和源码阅读，**没有 GPU 实测**。

---

## 0. 结论速览

| 问题 | 结论 |
|---|---|
| multimem 指令访问的是什么 | multicast object 的 **MC 地址**。一次访问由 NVSwitch 展开到所有成员 GPU 的物理副本上 |
| 有哪些指令 | sm_90 起：`multimem.ld_reduce`、`multimem.st`、`multimem.red`；较新：`multimem.st.async`、`multimem.red.async`、`multimem.cp.async.bulk`、`multimem.cp.reduce.async.bulk`、`fabric.try_put / try_red / try_pullred .multimem` |
| 哪条能做“归约后拿到结果” | **只有 `ld_reduce`**（以及 `fabric.try_pullred`）：N 个副本 → 1 个结果，结果回到**发起者** |
| `red` 能不能代替 `ld_reduce` | 语义不同：`red` 是“1 个操作数作用到 N 个副本”，每个副本都被 `+=`。用它做 AllReduce 的写流量是 `ld_reduce` 方案的 P 倍 |
| 能否把自己排除在外 | **不能**。所有指令都作用于**全部成员**，发起者必须是成员，没有成员掩码 |
| Copy Engine 能否用 | CE 写 MC 地址会被复制到所有副本（可做广播）；但**归约只能由 SM 指令触发**，CE 做不了 |
| NCCL 2.29.7 用了哪些 | 只用了 `ld_reduce`、`st`、`red` 三条（第 5 节） |

---

## 1. multicast 对象模型

### 1.1 UC 与 MC 两种映射

一个 multicast object 绑定在 P 张 GPU 各自的物理内存上。每张 GPU 都拿到两种虚拟地址：

| 映射 | 指向 | 访问方式 |
|---|---|---|
| **UC（unicast）** | 只指向**本卡**的物理副本 | 普通 `ld` / `st` / 原子操作 |
| **MC（multicast / multimem）** | 同时指向**全部 P 个**副本 | **只能用 `multimem.*` 指令** |

PTX 原文：“Accessing a multimem address with ld, st or any other memory operations results in undefined behavior.”

NVSwitch 负责把一次 multimem 访问展开成 P 次访问：

- `st`：复制写入每个副本。
- `ld_reduce`：读取每个副本，在交换机内归约，返回一份结果。
- `red`：对每个副本执行一次原子归约。

### 1.2 建立 multicast object 的 Driver API 流程

1. `cuMulticastCreate(&mc, prop)`：`prop.numDevices` 是成员数，`prop.size` 要按 `cuMulticastGetGranularity` 对齐。
2. 通过 `cuMemExportToShareableHandle` 导出 handle 并分给其他成员。MNNVL（跨 tray 的 NVL72）下要用 FABRIC 类型，并需要 IMEX channel。
3. 每个成员调用 `cuMulticastAddDevice(mc, dev)`。**必须所有成员都 add 完，才能开始 bind。**
4. 每个成员用 `cuMulticastBindMem` 或 `cuMulticastBindAddr` 把自己的物理内存绑定到 multicast object。偏移和大小都要按粒度对齐。
5. `cuMemAddressReserve` + `cuMemMap(mc)` + `cuMemSetAccess`，得到 MC VA。

补充：

- 按照 Programming Guide，每个参与的设备都必须 `cuMulticastAddDevice` 并绑定物理内存。**只有成员才能发起 multimem 访问。**
- Rubin（SM_107+）把绑定的对齐要求放宽到 256B。
- `CUmulticastObjectProp.flags` 和 `cuMulticastBind*.flags` 目前都是 “Flags for future use, must be zero”。

---

## 2. 指令全集

### 2.1 一览表

| 指令 | 语义 | 主要 op / type | 引入版本 / 最低架构 |
|---|---|---|---|
| `multimem.ld_reduce{.sem}{.scope}.op{.acc}{.vec}.type d, [a]` | 从全部副本读取，在交换机中归约，**结果返回发起线程的寄存器**；内存本身不变 | `.add`：u32/s32/u64/f16(x2)/bf16(x2)/f32/f64/fp8；`.min/.max`：u32/s32/u64/s64/f16(x2)/bf16(x2)/fp8；`.and/.or/.xor`：b32/b64；`.acc::f32`（f16/bf16）、`.acc::f16`（fp8）；`.v2/.v4/.v8`（总宽 32/64/128 bit） | PTX 8.1，sm_90；fp8 需要 sm_100a/f 系列 |
| `multimem.st{.sem}{.scope}{.vec}.type [a], b` | 把 b 写到全部副本，即广播 | b32/b64/u32/…/f16(x2)/bf16(x2)/f32/f64，可向量化到 128 bit | PTX 8.1，sm_90 |
| `multimem.red{.sem}{.scope}.op{.vec}.type [a], b` | 对**每个副本**原子执行 `*copy op= b` | 整数 add/min/max/and/or/xor；浮点只有 `.add`（f16/bf16/f32/f64） | PTX 8.1，sm_90 |
| `multimem.st.async.release.{gpu,sys}.type` | 非阻塞的 release 广播写 | b8–b64 / u / s / f32 / f64 | PTX 9.3，sm_100 |
| `multimem.red.async.release.{gpu,sys}.add.{u32,s32,u64}` | 非阻塞的 release 原子归约 | 整数 add | PTX 9.3，sm_100 |
| `multimem.cp.async.bulk.global.shared::cta.bulk_group [dst], [src], size` | TMA 风格：smem → MC global 的批量广播 | 16B 对齐，可加 `.cp_mask` | PTX 9.1，sm_90（`.cp_mask` 需要 sm_100） |
| `multimem.cp.reduce.async.bulk….redOp.type [dst], [src], size` | smem → MC global 的**批量原子归约**，每个副本都执行 `op=` | add / min / max / inc / dec / and / or / xor；f16/bf16/f32/f64/整数 | PTX 9.1，sm_90 |
| `fabric.try_put / try_red{.async}.multimem …`（含 `.tensor` 变体） | 基于 fabric handle / logical endpoint 的 multicast put / red，带错误报告 | b128 | PTX 9.3，sm_100 |
| `fabric.try_pullred.async.multimem` | 从 multicast logical endpoint 的所有资源读取并归约，结果直接写入 smem，带错误上报 | — | PTX 9.3，sm_100 |

### 2.2 修饰符

| 修饰符 | 取值 | 说明 |
|---|---|---|
| `.sem` | `ld_reduce`：`.weak` / `.relaxed` / `.acquire`；`st`：`.weak` / `.relaxed` / `.release`；`red`：`.relaxed` / `.release` | 内存序。NCCL 数据面用 `.relaxed`，同步计数用 `.acquire` / `.release` |
| `.scope` | `.cta` / `.cluster` / `.gpu` / `.sys` | 跨 GPU 同步要用 `.sys` |
| `.global` | — | 状态空间。NCCL 总是显式写出 |
| `.acc::f32` / `.acc::f16` | 只用于 `ld_reduce .add` | 交换机内部用更高精度累加，最后只舍入一次 |
| `.v2 / .v4 / .v8` | 向量宽度，总宽不超过 128 bit | 一条指令最多搬 16B |

### 2.3 三种“作用方向”

把所有指令按数据流向分成三类：

| 类别 | 指令 | 数据流 |
|---|---|---|
| **pull-reduce（N → 1）** | `ld_reduce`、`fabric.try_pullred` | 交换机读 N 个副本 → 归约 → 1 份结果回发起者 |
| **push-broadcast（1 → N）** | `st`、`st.async`、`cp.async.bulk`、`fabric.try_put` | 发起者写 1 次 → 交换机复制到 N 个副本 |
| **push-reduce（1 → N，原子）** | `red`、`red.async`、`cp.reduce.async.bulk`、`fabric.try_red` | 发起者给 1 个操作数 → N 个副本各自 `op=` |

`ld_reduce` 和 `red` 的区别要特别注意：

- **`ld_reduce`**：N 个副本 → 1 个结果（寄存器），内存不变。天然适合 ReduceScatter：谁需要结果，谁就发起读取。
- **`red` / `cp.reduce.async.bulk`**：1 个操作数 → 作用到 N 个副本。如果 P 个 rank 都对同一 MC 地址执行 `red.add`，每个副本最终都等于“初值 + 所有操作数之和”。这相当于基于原子操作的 AllReduce，但：
  - 每个贡献者都要向全部 P 个副本写入，NVLink 写流量是 `ld_reduce` 方案的 P 倍。
  - 浮点只有 `.add`，而且累加顺序不确定。
  - 适合做**计数器 / barrier**，不适合做数据面归约。

---

## 3. 链路流量模型

记号：P 为成员数，N 为一次操作的字节数。每张 GPU 到 NVSwitch 的链路是全双工，TX（出）和 RX（入）各自计带宽。

| 指令 | 发起者 TX | 发起者 RX | 每个非发起成员 TX | 每个非发起成员 RX |
|---|---|---|---|---|
| `ld_reduce`（N 字节结果） | N（自己的副本也被读） | N（结果） | N | 0 |
| `st`（N 字节） | N | N（自己的副本也被写） | 0 | N |
| `red`（N 字节操作数） | N | N | 0 | N |

**自环假设**：

- `ld_reduce` 读取**所有**副本，**发起者自己的副本也要走一次 GPU→Switch 链路**。
- `st` 写入**所有**副本，**发起者自己的副本也要走一次 Switch→GPU 链路**。

文档没有说明本地副本是否在 GPU 内部短路。NCCL 自己的代价模型注释是 “Should be nRanks*nBytes”（`nccl-2.29.7/src/sym_kernels.cc:355,367`），支持“经过链路”的判断。可以用 DCGM 的 `DCGM_FI_PROF_NVLINK_TX_BYTES` / `RX_BYTES` 实测：P 卡 multicast 广播时，每卡 RX 若约等于 P·N，说明自环经过链路；若约等于 (P−1)N，说明本地短路。

---

## 4. 成员语义：能否“排除自己”

### 4.1 逐条检查

| 指令 / 接口 | 是否作用于全部成员 | 有无排除成员的手段 |
|---|---|---|
| `multimem.ld_reduce` / `st` / `red` | 是，访问 “the multiple memory locations pointed to by the multimem address” | 无 |
| `multimem.st.async` / `red.async` | 是，写入 “each GPU’s memory referred to by the multimem address” | 无，只是把同步写改为异步 release 写 |
| `multimem.cp.async.bulk` | 是 | `.cp_mask` 是**每 16B 内的字节掩码**，不是成员掩码 |
| `multimem.cp.reduce.async.bulk` | 是，对每个副本执行 `op=` | 无 |
| `fabric.try_put / try_red{.multimem}` | 是，“performed on all member devices of the multicast endpoint” | 无；而且要求 “the device that issues the operation must be a member” |
| `fabric.try_pullred.async.multimem` | 是，“reduction is performed on the values read from **all resources** referred to by the multicast logical endpoint” | `imm-membermask` 是 **warp lane 掩码**，规定必须为 `0xFFFFFFFF`，不是组成员掩码 |
| `CUmulticastObjectProp.flags` / `cuMulticastBind*.flags` | — | 都是 “must be zero” |
| `CUlogicalEndpointProp` | — | 只有 `numDevices` 和 `flags`（`NONE` 或 `COUNTED_OPS`），没有排除成员的选项 |

结论：**现有 ISA 和驱动接口中，没有“对多播组中除自己以外的成员执行 st / ld_reduce”的语义。**发起者必须是成员，成员的副本必然参与。

由此推出两条硬约束：

1. **`ld_reduce` 的结果只能回到发起者**。交换机没有“对组 A 归约、把结果投递给非成员 B”的原语。要让非成员拿到结果，只能由成员发起后再用 unicast 转发。
2. 把某个 GPU 加入组但让它贡献单位元（Sum 用 0），交换机**仍会读取**它的副本，TX 省不下来。

### 4.2 一个未文档化、不建议依赖的角落

`cuMulticastBindMem_v2` / `cuMulticastBindAddr_v2` 支持按 `dev` 逐设备绑定。理论上可以设想：为每个 shard 建一个 multicast object，在其中**不绑定发起者自己那段副本**，以此去掉自环。但是：

- 文档没有说明成员在某个偏移处未绑定物理内存时 multimem 访问会怎样，属于未定义行为。可能触发 fault，也可能静默丢弃。
- 文档要求 “All devices must be added”，只约束了加入组，没有保证可以部分绑定。
- P 个 shard 需要 P 个 multicast object，会消耗 fabric 上全局共享的 multicast group 配额（见资源文档）。

只能作为在实验机上探测硬件行为的手段，不能用于生产。

---

## 5. Copy Engine 与 multimem

| 操作 | CE 能否做 | 原因 |
|---|---|---|
| 写 MC 地址（广播） | **能** | CE 写 MC 地址时，交换机会把数据复制到所有副本。NCCL 2.29.7 用这一点广播 4 字节同步 flag（`src/ce_coll.cc:120-160`）。大块 D2D 拷贝写 MC 地址的性能需要实测 |
| 归约（ld_reduce / red） | **不能** | 交换机内归约只能由 SM 指令触发。CE 只能执行 copy，没有归约语义 |
| `multimem.cp.async.bulk` | — | 这是 SM 发起的 TMA 指令，可以只用少量线程完成广播，但仍然占用 CTA |
| `multimem.cp.reduce.async.bulk` | — | 也是 SM 发起，语义是“对所有副本 `op=`”（push），不是“从所有副本拉取并归约”，不能用来做 ReduceScatter |

---

## 6. NCCL 2.29.7 实际使用的 multimem 指令

源码位置相对 `Megatron-Scale-Down/nccl-2.29.7`。

| 位置 | 指令 | 用途 |
|---|---|---|
| `src/device/reduce_kernel.h:861-1018`（`Apply_LoadMultimem`） | `multimem.ld_reduce.relaxed.sys.global.{add,min,max}[.acc::f32][.v4].{u32,s32,u64,s64,f32,f64,f16x2,bf16x2,e4m3x4,e5m2x4}` | 经典 NVLS kernel 的归约读取 |
| `src/device/op128.h:374-410`（`multimem_st_global`） | `multimem.st.global.{b32,b64,v4.f32}` | 经典 NVLS kernel 的广播写；1/2 字节回退为对 MC 地址的普通 `st` |
| `src/device/prims_simple.h:96-107`（`loadStepValue`） | `multimem.ld_reduce.acquire.sys.global.min.u64` | **NVLS credit 轮询**：一次读取所有 rank 的计数并取最小值，相当于“等最慢的 rank” |
| `src/include/nccl_device/impl/multimem__funcs.h` | `ld_reduce.global.add[.acc::f32 \| .acc::f16][.v2/.v4]…`、`st.global.{b32,b64,v4.f32}` | Device API 与 symmetric kernel 共用的 multimem 实现 |
| `src/include/nccl_device/impl/lsa_barrier__funcs.h:63-80` | `multimem.red.{release,relaxed}.sys.add.u32` | LSA barrier 的 arrive：一条指令让所有副本的计数器 +1；wait 只轮询本地副本是否达到 `epoch + nRanks` |

**没有用到**：`st.async`、`red.async`、`cp.async.bulk`、`cp.reduce.async.bulk`、`fabric.*`。

### 6.1 pack 宽度

`ld_reduce` 一次处理的字节数由 `LoadMultimem_BigPackSize` 决定（`src/device/reduce_kernel.h:973`）：

| 情况 | pack 宽度 |
|---|---|
| 浮点 sum，元素 < 8B（f16/bf16/f32） | 16B（`.v4`） |
| f64 | 8B |
| f16/bf16 的 min/max | 16B |
| 32/64 位整数 | 元素宽度 |
| 8 位整数 | 不支持 |
| fp8 | 只在 sm_100/101/120 系列启用 |

### 6.2 精度

- NCCL 对 bf16/f16 的 sum 使用 `.acc::f32`（`src/device/reduce_kernel.h:846-847`）：**交换机内用 fp32 累加 P 份输入，只在最后舍入一次**。
- fp8 使用 `.acc::f16`。
- 对比：Ring AllReduce 每一跳都要把累加结果舍入回 bf16，共舍入 P−1 次。

### 6.3 用普通 `st` 写 MC 地址

NCCL 的 credit 通知（`postPeer` 中的 `st.relaxed.sys`，`src/device/prims_simple.h:174-182`）和 `multimem_st_global<1/2>` 都是对 MC 地址执行**普通 store**，依赖的是“对 MC 地址的普通写也会被复制到所有副本”。这与 PTX 文档“只能用 multimem 指令访问 MC 地址”的说法不一致，属于 NCCL 依赖的实现行为，自己写代码时不建议照搬。
