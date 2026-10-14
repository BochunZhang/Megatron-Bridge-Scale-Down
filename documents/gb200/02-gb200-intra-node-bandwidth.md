# GB200 compute tray 节点内带宽：NVLink、C2C 与 CPU 中转

结论：**在当前驱动模式（CDMM）下，"CPU 中转、数据不落 CPU 内存"的 GPU→GPU 路径不存在。** 本轮 relay
和 host 测试测到的其实是 Grace LPDDR 带宽，而不是 C2C。可信的数字如下：

| 路径 | 带宽（payload） | 说明 |
| --- | --- | --- |
| GPU→GPU，NVLink，copy engine | **782 GB/s** | 约为 NV18 单向 900 GB/s 的 87%，同 Grace 与跨 Grace 相同 |
| GPU→GPU，NVLink，SM copy | 718 GB/s | 约 80% |
| D2H，GPU→本地 Grace pinned | **194 GB/s** | 不到 C2C 单向约 450 GB/s 的一半 |
| H2D，本地 Grace pinned→GPU | **212 GB/s** | 同上 |
| D2H / H2D，跨 Grace | **123 / 125 GB/s** | 受 CPU0↔CPU1 链路限制 |
| GPU→pinned→GPU（staged，同 Grace） | 170 GB/s | 达到较慢一段（D2H）的 88% |
| GPU→pinned→GPU（staged，跨 Grace） | 122 GB/s | 达到跨 Grace 段的 98–99% |
| GPU 读本地 HBM（SM） | 7.13–7.20 TB/s | 约为 8 TB/s 标称值的 89% |

经 CPU 中转的 GPU→GPU 拷贝，最好情况只有 NVLink 直连的约 1/5（170 vs 782 GB/s），只能作为 NVLink
之外的附加通道，不能取代 NVLink。

- 硬件：1 个 GB200 compute tray，2 × Grace（Neoverse-V2，各 72 核）+ 4 × B200（各 152 SM），64K 页内核
- 软件：driver 580.105.08，CUDA 13.0，`CoherentGPUMemoryMode: "driver"`（CDMM），Addressing Mode ATS
- 测试框架：`examples/gb200/intra-super-node/gb_benchmark/`，`./run_gb200.sh`（默认 relay 套件）
- 原始数据：`results/gb200/intra-super-node/261013-183202/`
- 计时：每项 4 GiB，warmup 2 轮，计时 10 轮，报中位数；GB/s 按 10^9 计

---

## 1. 拓扑

```text
GPU0 ─C2C─┐                ┌─C2C─ GPU2
          ├─ CPU0 ── CPU1 ─┤
GPU1 ─C2C─┘                └─C2C─ GPU3

4 张 GPU 之间均为 NV18（经 NVSwitch）
```

| GPU | 所挂 CPU | Host NUMA | CPU 核 |
| --- | --- | --- | --- |
| gpu0、gpu1 | cpu0 | 0 | 0–71 |
| gpu2、gpu3 | cpu1 | 1 | 72–143 |

`numactl -H` 共显示 34 个 NUMA 节点：node 0/1 是两个 Grace 的 LPDDR（各约 478 GiB），node 2–33 是
GPU 对应的节点，内存全部为 0 MB。node 0 与 node 1 的 distance 是 40。

## 2. 运行环境：CDMM 模式下 CPU 不能原地访问 HBM

### 2.1 什么是 CDMM

GB200 默认有两种 GPU 内存管理模式：

- **NUMA 模式**：HBM 作为 NUMA 节点交给操作系统，`malloc`/`mmap`/`numa_alloc_onnode` 都能分配到 HBM 上，
  CPU 经 C2C 原地 load/store。
- **CDMM 模式**（Coherent Driver-based Memory Management，`NVreg_CoherentGPUMemoryMode=driver`）：HBM 由
  NVIDIA 驱动管理，不上线为 NUMA 节点。GPU 的 NUMA 节点仍然存在，但报告内存为 0。从 driver 580.65.06 起，
  K8s GPU Operator 部署默认使用 CDMM。

本机 `/proc/driver/nvidia/params` 中 `CoherentGPUMemoryMode: "driver"`，node 2–33 均为 0 MB，确认是 CDMM。

### 2.2 hbm_probe：每种分配方式下 CPU 能否原地访问 HBM

4 张 GPU 的判定完全一致：

| 方法 | 分配方式 | 判定 |
| --- | --- | --- |
| `numa` | `mmap` + `mbind` 到 HBM NUMA 节点 | UNAVAILABLE：HBM 不是在线的 NUMA 节点（`cudaDevAttrNumaConfig=0`） |
| `managed` | `cudaMallocManaged` + PreferredLocation=GPU + AccessedBy=CPU | **MIGRATES**：CPU 访问会把页从 HBM 迁走 |
| `managed_noadvise` | `cudaMallocManaged`，prefetch 到 GPU | **MIGRATES** |
| `cudamalloc` | `cudaMalloc`，CPU 直接解引用 | CPU CANNOT ACCESS：SIGSEGV |
| `vmm` | `cuMemCreate(DEVICE)` + `cuMemSetAccess(HOST / HOST_NUMA)` | CPU CANNOT ACCESS：`cuMemSetAccess` 返回 `CUDA_ERROR_NOT_SUPPORTED`，解引用 SIGSEGV |

managed 内存的迁移代价（probe，1 GiB）：

| 指标 | 数值 |
| --- | --- |
| GPU 读，CPU 访问前 | 7.13–7.20 TB/s |
| CPU 首轮读（触发迁移） | 43–60 GB/s |
| CPU 稳态读（页已在 LPDDR） | 290–325 GB/s |
| CPU 访问后 GPU 首轮读（页迁回 HBM） | 11–13 GB/s |

`directManagedMemAccessFromHost=0`，意味着 CPU 不能隔着 C2C 访问留在 HBM 上的 managed 页。CPU 一访问就会
缺页，驱动把页迁到发起访问的 CPU 所在节点的 LPDDR。probe 结束后抽样的 256 页全部位于 node 1，与此一致。

**结论：CDMM 下 CPU 只能借 managed 内存碰到 HBM 上的数据，而 managed 内存靠迁移来实现访问。**
不迁移的 `cudaMalloc` 和 VMM 内存，CPU 根本访问不了。按需迁移的带宽只有 10–60 GB/s，不应依赖 UVM
自动迁移来做 offload。

## 3. 有效结果

以下测试只用 `cudaMalloc`、pinned 内存和 copy engine/SM，不涉及 managed 内存，结果有效。

### 3.1 NVLink 直连

| 路径 | copy engine（`cudaMemcpyAsync`） | SM copy（608×512 blocks） |
| --- | --- | --- |
| gpu0→gpu1（同 Grace） | 782.26 GB/s | 718.47 GB/s |
| gpu0→gpu2（跨 Grace） | 782.11 GB/s | 718.41 GB/s |

- NV18 = 18 条 NVLink5，单向约 900 GB/s。copy engine 达到 87%，SM copy 达到 80%。
- 同 Grace 与跨 Grace 没有差别：GPU 之间都经 NVSwitch 互连，与 CPU 拓扑无关。
- 10 轮的 min/max 差距不到 0.2%，非常稳定。

### 3.2 C2C：copy engine D2H / H2D

pinned 内存用 `numa_alloc_onnode` + `cudaHostRegister` 分配，抽样确认页全部落在指定节点。

| 方向 | 本地 Grace | 跨 Grace |
| --- | --- | --- |
| D2H cuda0→pinned | 193.66 GB/s（pinned0） | 123.09 GB/s（pinned1） |
| H2D pinned→cuda1 | 211.79 GB/s（pinned0） | 124.99 GB/s（pinned1） |
| H2D pinned→cuda3 | 211.74 GB/s（pinned1） | 125.40 GB/s（pinned0） |

- **本地 D2H/H2D 只有 C2C 单向约 450 GB/s 的 43%/47%。** D2H 每轮耗时稳定在 22.177 ms（10 轮差距
  < 0.01 ms），表现为某个固定上限。可能是单条 copy engine/单 stream 的上限，也可能是 LPDDR 或 pinned
  内存注册方式的限制，尚未区分（见 §7）。
- **跨 Grace 统一在 123–125 GB/s**，是 CPU0↔CPU1 链路在 DMA 下的上限。
- pinned 内存必须放在 GPU 所挂的 NUMA 节点上，放错会损失约 40% 带宽。

### 3.3 staged：cudaMalloc→pinned→cudaMalloc

数据按 chunk 切块，在 2 个槽之间轮转。D2H 在源 GPU 的 stream 上发起，H2D 在目的 GPU 的 stream 上发起，
用 event 串起来，一块的 D2H 与上一块的 H2D 重叠。中转区大小为 chunk × 2。

| chunk（中转区） | gpu0→gpu1 经 pinned0 | gpu0→gpu3 经 pinned0 | gpu0→gpu3 经 pinned1 |
| --- | --- | --- | --- |
| 4 MiB（8 MiB） | 138.50 GB/s | 112.57 GB/s | 112.33 GB/s |
| 16 MiB（32 MiB） | 165.82 GB/s | **122.28 GB/s** | 120.44 GB/s |
| 64 MiB（128 MiB） | **169.90 GB/s** | 122.07 GB/s | **121.46 GB/s** |
| 256 MiB（512 MiB） | 164.37 GB/s | 118.89 GB/s | 118.88 GB/s |
| 理论上限（较慢一段） | 193.66（D2H 本地） | 125.40（H2D 跨 Grace） | 123.09（D2H 跨 Grace） |
| 最佳效率 | 88% | 98% | 99% |

- **跨 Grace**：无论在哪个 CPU 中转，都必须跨一次 socket，所以两种放法都在约 122 GB/s，几乎打满跨 Grace 段。
- **同 Grace**：只有 88%。D2H 写和 H2D 读同时压在 node0 的 LPDDR 上，合计约 340 GB/s，接近 §4.3 测到的
  本地 LPDDR 量级，LPDDR 可能是瓶颈。
- **4 MiB 太小**：发起所有拷贝耗时约 15 ms，占总时间 31 ms 的一半，受主机发起开销拖累。
- **256 MiB 偏大**：4 GiB 只切成 16 块，流水首尾无法重叠的部分约占 1/16。
- **没有看到 Grace L3 的收益**：32 MiB 中转区能放进单个 socket 的 L3（114 MiB），但并不比放不进的
  128 MiB 快。可能 copy engine 的写入不在 L3 中分配，也可能瓶颈在 copy engine 本身，仅凭带宽无法区分。

## 4. 无效结果：relay 与 host 分段

### 4.1 原设计

relay 想测的是 CPU 线程从源 HBM 读 64 B 到寄存器，再直接写到目的 HBM，数据不落 CPU 内存：

```text
GPU0 HBM ─C2C→ CPU0 寄存器 ─C2C→ GPU1 HBM              （同 Grace）
GPU0 HBM ─C2C→ CPU0 → CPU1 ─C2C→ GPU2 HBM              （跨 Grace）
```

前提是 CPU 能原地 load/store HBM，CDMM 下这个前提不成立（§2）。

### 4.2 实际发生了什么

`--hbm-backend` 默认是 `auto`：HBM 不是 NUMA 节点时，`hbm*` 改用 `cudaMallocManaged`。
`prefetch_managed()` 只在填充 buffer 时调用一次，计时循环中不会把页 prefetch 回 GPU。因此每个 case：

1. **填充**：GPU 写入 pattern，prefetch 到 HBM，此时数据确实在 HBM 上；
2. **warmup 第 1 轮**：CPU 线程访问时逐页缺页，页迁到发起访问的 CPU 所在节点的 LPDDR，迁移代价被 warmup 吸收；
3. **warmup 第 2 轮与 10 轮计时**：页全部在 LPDDR，测的是 LPDDR 到 LPDDR；
4. **另一个 CPU 再访问**：页已在系统内存中，不再迁移，变成跨 socket 远端访问。

### 4.3 证据与重新解读

| 测试项 | ldp_stp | ldnp_stnp | memcpy | 实际测到的 |
| --- | --- | --- | --- | --- |
| read hbm0 by cpu0 | 301.96 | 309.45 | 302.66 | 本地 LPDDR 读 |
| read hbm0 by cpu1 | 93.09 | 93.90 | 92.74 | 跨 socket 读 node0 LPDDR |
| write hbm1 by cpu0 | 354.62 | 358.17 | 355.06 | 本地 LPDDR 写 |
| write hbm1 by cpu1 | 107.58 | 107.66 | 107.86 | 跨 socket 写 |
| write hbm2 by cpu0 | 353.64 | 355.76 | 358.96 | 本地 LPDDR 写 |
| write hbm2 by cpu1 | 107.65 | 107.48 | 107.74 | 跨 socket 写 |
| relay gpu0→gpu1 via cpu0 | 155.85 | 157.90 | 158.63 | 本地 LPDDR copy |
| relay gpu0→gpu2 via cpu0 | 160.95 | 158.10 | 159.24 | 本地 LPDDR copy |
| relay gpu0→gpu2 via cpu1 | 91.93 | 92.94 | 92.68 | 跨 socket copy |

单位 GB/s，72 线程。三处数据可以直接排除 C2C：

- **写 hbm2 与写 hbm1 一样快。** 由 cpu0 写，分别是 354 和 355 GB/s，但 hbm2 挂在 cpu1 下。
  真经过 C2C 的话，写 hbm2 必须跨 CPU0↔CPU1 链路，不可能和本地一样快。
- **relay 带宽与目的 GPU 无关。** 经 cpu0 中转时，gpu0→gpu1 是 156 GB/s，gpu0→gpu2 是 161 GB/s，
  说明数据根本没到 GPU。
- **出现了"先访问者占页"。** cpu0 先访问，页落到 node0；cpu1 之后访问只有 93/108 GB/s，与跨 socket
  访问一致。probe 中 attached CPU 写的 112 GB/s 也是同一机制。

这些数字虽然不是 C2C 带宽，但可以作为 Grace LPDDR 的参考：

| 操作（72 线程） | 本地 | 跨 socket |
| --- | --- | --- |
| 读 | 约 300–310 GB/s | 约 93 GB/s |
| 写 | 约 355 GB/s | 约 108 GB/s |
| copy（payload） | 约 156–161 GB/s | 约 92 GB/s |

### 4.4 staged direct

staged 的 direct 模式让 CPU 线程直接 load/store `cudaMalloc` 内存。CPU 访问时 SIGSEGV，被正确记为
`[skip]`，与 probe 的 `cudamalloc` 判定一致。

## 5. 能否由 SM 驱动、经 CPU 中转且不落地

不能。GPU SM 发出的访存只能落到三类目标，路由由地址映射决定，用户无法指定：

| 目标 | 地址来源 | 物理路径 |
| --- | --- | --- |
| 本地 HBM | `cudaMalloc` | GPU 内部 |
| 对端 GPU 的 HBM | peer 映射（`cudaDeviceEnablePeerAccess` / IPC / VMM） | 固定走 NVLink/NVSwitch |
| 系统内存 | pinned / 可分页内存（ATS） | C2C 到 Grace |

要让 GPU0 的 SM 写的数据经 Grace 落到 GPU1 的 HBM，GPU1 的 HBM 必须以系统物理地址的形式出现在 GPU0
可用的系统页表里，而这只有 NUMA 模式才满足。即使在 NUMA 模式下，GPU0 访问 GPU1 的 NUMA 内存是走
C2C→Grace→C2C 还是 NVLink，NVIDIA 也没有公开文档，需要实测。

最接近的方案是：SM 驱动、经一小块 pinned 中转区接力。这在结构上就是 staged pinned（§3.3），区别只有三点：

1. **由 SM 而不是 copy engine 搬运**：这是唯一可能提高上限的变量。staged 已经达到较慢一段的 88%，
   再优化流水，收益也有限。
2. **用 GPU 之间的细粒度 flag 同步，代替主机 event**：主要改善小 chunk 时的主机开销。
3. **用 Grace PMU 统计 LPDDR 流量**，确认数据是否只经过 L3、不落 DRAM。这一点现有的 staged 也可以加。

所以应先测 SM 驱动的两段（§7 第 2 项）：只有当它明显超过 copy engine 的 194/212 GB/s 时，才值得实现
SM 接力。

## 6. 对 Scale-Down offload 的启示

- **单卡与本地 Grace 之间可用约 190–210 GB/s**（copy engine，单 stream）。activation/optimizer offload
  的 D2H/H2D 都受此限制。
- **pinned buffer 必须放在 GPU 所挂的 NUMA 节点上**，跨 Grace 只有约 123 GB/s。
- **同一 Grace 下的两张 GPU 同时 offload 时，会共享约 300–355 GB/s 的 LPDDR 带宽**，加上 CPU 侧 optimizer
  等计算访存，很可能是实际瓶颈。本轮没有测并发，需要补测。
- **不要依赖 managed 内存或 UVM 的自动迁移**：CDMM 下按需迁移只有 10–60 GB/s。

## 7. 后续工作

1. **修正 gb_benchmark**：`hbm*` 为 managed 且 `directManagedMemAccessFromHost=0` 时，relay 与 host 应像
   staged direct 一样记为 `[skip]`，不再产出误导数据。managed buffer 的 `placement` 应复用 probe 的页节点
   抽样，替代现在的 `"managed (not checked)"`。
2. **测 SM 驱动的 C2C 两段**，判断 194/212 GB/s 是否是 copy engine 的上限：

   ```bash
   ./build/gb_benchmark device --gpu 0 --op copy --src cuda0 --dst pinned0 --blocks 8,16,32,64,152,auto
   ./build/gb_benchmark device --gpu 1 --op copy --src pinned0 --dst cuda1 --blocks 8,16,32,64,152,auto
   ```

3. **并发测试**：
   - 同一 GPU 多 stream 的 D2H，判断是否为单 copy engine 的上限；
   - gpu0 与 gpu1 同时向 pinned0 做 D2H，测共享 LPDDR 的上限；
   - 同一 GPU 的 D2H 与 H2D 双向并发，对应 offload 与 reload 重叠的场景。
4. **比较 pinned 内存的分配方式**：`cudaHostAlloc` vs `numa_alloc_onnode` + `cudaHostRegister`（64K 页）。
5. **给 staged 加 Grace PMU 统计**（`perf` 的 `nvidia_scf_pmu_*`），确认中转数据是否落 LPDDR。
6. **更新 `gb_benchmark/README.md` 的验证状态**：GPU 路径与 staged 已在 GB200 上运行；relay/host 在 CDMM 下无效。

## 附录：复现与数据

```bash
cd examples/gb200/intra-super-node/gb_benchmark
make
unset CUDA_VISIBLE_DEVICES
./run_gb200.sh        # results/gb200/intra-super-node/YYMMDD-HHMMSS/
```

| 文件 | 内容 |
| --- | --- |
| `summary.txt` | 所有 probe 判定与带宽结果，一行一条 |
| `env.txt`、`hbm_probe/env.txt` | `nvidia-smi`、`numactl -H`、`lscpu`、驱动参数（含 `CoherentGPUMemoryMode`） |
| `topology.json` | 运行时发现的 CPU/GPU/NUMA 拓扑与 P2P 能力 |
| `hbm_probe/probe-gpu<g>.{log,json}` | 每张 GPU 的 HBM 访问方式探针 |
| `gpu0_to_gpu{1,2}/{relay,baseline_memcpy,baseline_sm_copy}/` | relay（无效）与 NVLink baseline |
| `gpu0_to_gpu{1,3}/staged/` | staged pinned；direct 被跳过 |
| `legs/{read_hbm0,write_hbm1,write_hbm2}/` | host 分段（实为 LPDDR） |
| `legs/{d2h_cuda0,h2d_cuda1,h2d_cuda3}/` | copy engine D2H/H2D 分段 |

参考：

- [Understanding Memory Management on Hardware-Coherent Platforms](https://developer.nvidia.com/blog/understanding-memory-management-on-hardware-coherent-platforms/)（NVIDIA Developer Blog）
- [NVIDIA Data Center GPU Driver 580.82.07 Release Notes](https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-580-82-07/index.html)
- [arXiv 2408.11556](https://arxiv.org/abs/2408.11556)（gh_benchmark 原始论文，GH200）
