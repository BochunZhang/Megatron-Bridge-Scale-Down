# Megatron 训练中的 CPU↔GPU（H2D/D2H）通信全景

本文汇总对 megatron-bridge 仓库（含 `3rdparty/Megatron-LM`，core_v0.19.0 系）
的全量扫描结果：**除 offload 特性以外**，还有哪些操作会产生参数 / 权重 /
梯度 / 优化器状态 / checkpoint 数据在 CPU 与 GPU 之间的搬运，以及训练循环内
小标量（grad-norm、found-inf、num-tokens、MoE 路由元数据）D2H 同步的确切
时序和底层机制。

已排除的 offload 特性（另有专门文档）：

- 层级激活 offload（`model.cpu_offloading*`，`cpu_offloading_context`）
- 细粒度激活 offload（`fine_grained_activation_offloading`）
- 优化器 offload（`optimizer_cpu_offload` / `HybridDeviceOptimizer` /
  `overlap_cpu_optimizer_d2h_h2d`）

路径约定：下文 `core:` 指 `3rdparty/Megatron-LM/megatron/core/`，
`training:` 指 `3rdparty/Megatron-LM/megatron/training/`，
`bridge:` 指 `src/megatron/bridge/`。

## 1. 总览：按数据量与触发时机分类

| 类别 | 数据量 | 触发时机 | 方向 |
|---|---|---|---|
| 模型初始化（CPU init → GPU） | 全模型权重 | 冷启动一次 | H2D |
| HF 权重导入 | 全模型权重 | 加载 HF checkpoint 一次 | disk→CPU→GPU |
| HF 导出 / `also_save_hf_checkpoint` | 全模型权重 | 每次导出 / 每次 save | GPU→CPU→disk |
| Checkpoint save（torch_dist / legacy） | 权重 + fp32 主参数 + Adam 态 + RNG | 每个 save interval | GPU→CPU→disk |
| Checkpoint load | 同上 | resume / finetune 一次 | disk→CPU→GPU |
| RL 相位切换 | 整优化器 / 参数 buffer / 推理权重 | 每次 train↔inference 切换（或每步 refit） | 双向 |
| PEFT adapter 导出 | adapter 权重 | 导出 / 每次 save（`also_save_hf_checkpoint`） | GPU→CPU |
| 推理入口 / diffusion pipeline | 整模型或组件（T5/DiT/VAE） | setup / 每 prompt / 每 timestep | 双向 |
| 调试路径（param hash、wgrad/dgrad dump、router trace） | 全参数 / 全梯度 | 按 interval 或开关 | GPU→CPU |
| 训练循环内标量（grad-norm、found-inf、num-tokens、MoE splits） | 字节级 | 每步 / 每 MoE 层 | 主要 D2H，**贵在同步** |

对 scale-down / 显存分析最重要的排序：**checkpoint save/load > RL 相位切换 >
HF 导入导出 > CPU 初始化路径 > 训练循环内隐性同步**。

## 2. 大块数据路径

### 2.1 模型初始化（一次性全模型 H2D）

`use_cpu_initialization=True` 时所有权重在 host 内存构建并初始化，随后整机
搬上 GPU：

- CPU 侧构建：`core:tensor_parallel/layers.py:184-230`
  `_initialize_affine_weight_cpu()`（VocabParallelEmbedding /
  ColumnParallelLinear / RowParallelLinear 共用）；TE 层统一传
  `device="cpu"`（`core:extensions/transformer_engine.py:403-413`）。
- H2D 搬运点：
  - `training:training.py:1812-1818` / bridge
    `models/model_provider.py:683-689`：`model_module.cuda()`；
  - DDP：`core:distributed/param_and_grad_buffer.py:1259-1302`，
    `param.data` 替换为 GPU buffer 视图后 `copy_(old_param_data)`
    （CPU 参数 / CPU checkpoint 加载后的参数都经此上卡）；
  - Megatron-FSDP：`fsdp/.../megatron_fsdp.py:249-253` `module.to(device)` +
    `param_and_grad_buffer.py:2954,2985`；
  - meta-device 初始化：`to_empty_if_meta_device()`
    （`training:models/dist_utils.py:426-450`）。
- 延迟小权重 H2D（首次 forward）：MoE router 权重
  （`core:transformer/moe/router.py:96-100`）、RoPE `inv_freq`
  （`core:models/common/embeddings/rotary_pos_embedding.py:160-162`）、
  shared embedding tie（`core:models/common/language_module/language_module.py:305-308`）。
- Bridge 特有：MIMO 冻结参数 `_move_frozen_params_to_device`
  （`bridge:models/megatron_mimo/megatron_mimo_provider.py:669-686`，DDP 只搬
  `requires_grad=True` 的参数）；`hf_pretrained/causal_lm.py:171`
  `model.to(self.device)`——访问 `hf_pretrained.model` 会把**整个 HF 模型**
  搬上 GPU（默认 device 为 cuda）。

### 2.2 HF 权重导入（CPU→GPU，逐权重）

safetensors 以 `device="cpu"` 读入（`bridge:models/hf_pretrained/state.py:584-645`），
转换时逐权重上卡：

- `bridge:models/conversion/param_mapping.py`
  - `scatter_to_tp_ranks()` L578-615：TP>1 时切分后 `.to(device)`；
  - `ReplicatedMapping.hf_to_megatron()` L1292（router.weight 特判 L1296-1298）；
  - `broadcast_from_pp_rank()` L419-421：PP broadcast 的接收 buffer。
- 训练启动加载 HF 预训练权重：
  `bridge:training/checkpointing.py::_load_hf_pretrained_checkpoint` L2598-2624。
- 模型特例：MiniMax-M2 QK-norm（`minimax_m2_bridge.py:51-70`）、Kimi INT4
  CPU 反量化（`kimi_k25_vl_bridge.py:148-164`）、DSV3 `inv_freq`
  （`deepseek_v3_bridge.py:193`）。

### 2.3 HF 导出（GPU→CPU，逐权重流式）

- 主通道：`bridge:models/conversion/model_bridge.py::HFWeightTuple.iter_finalized`
  L133 `exported_tensor.cpu()`——`save_hf_pretrained` / `export_hf_checkpoint`
  的每个权重都经此 D2H。
- MoE grouped 导出 CUDA OOM 回退：`_accumulate_grouped_export` L985-999，
  全部 expert 权重 `.cpu()` 后在 host 上 `torch.stack`。
- 非流式回退：`auto_bridge.py::save_hf_weights` L1167-1189——**整模型收集进
  host RAM**（代码注释自认 >70B 有 RAM 风险）。
- Step3.5 lm_head 主动 `.cpu()` 以压低导出峰值显存（`step35_bridge.py:253`）。
- ModelOpt 量化导出：amax/scale `_clone_cpu()`；EP all-gather 前的 CPU→GPU
  staging（`modelopt_utils.py:1001-1009`，pinned 时 `non_blocking`）。
- MIMO 导出：非 rank0 的 CPU tensor 经 `dist.gather_object`（host pickle）
  汇聚（`megatron_mimo/conversion/orchestrator.py:945-1020`）。
- 纯 CPU 转换通道（设计上不经 GPU）：`export_ckpt` / `import_ckpt` /
  `export_adapter_ckpt`（gloo + `use_cpu_initialization`）、
  `scripts/conversion/cpu_backend.py`。
- TRT-LLM 导出（MCore 侧）：每个权重 GPU→**pinned CPU buffer**
  （`core:export/trtllm/.../distributed_trtllm_model_weights_converter.py:82-98`）。

### 2.4 Checkpoint 保存（GPU→CPU→disk，每个 save interval）

**torch_dist（主路径）**：`core:dist_checkpointing/strategies/filesystem_async.py:227-250`
`FileSystemWriterAsync.preload_tensors()`——每个写 bucket 内**所有张量**
（权重 + 优化器态 + extra_state + RNG）`to("cpu", non_blocking)` +
`synchronize`。同步 save、异步 fork save、常驻子进程 save 三种模式都经过
这一步（异步模式的 D2H 发生在 fork 前的训练进程或 preload 子进程中）。

**分布式优化器 CPU world buffer**（`--dist-ckpt-optim-sharding-type` 为
`dp_zero_gather_scatter` / `fully_reshardable` 时）：
`core:optimizer/distrib_optimizer.py::get_parameter_state_dp_zero`
L1352-1355——**所有 fp32 主参数 + exp_avg + exp_avg_sq 逐张量 `.detach().cpu()`**
进 CPU world buffer，再由 `save_parameter_state()` `torch.save` 落盘。

**legacy torch 格式**：`torch.save(GPU state_dict)`（`training:checkpointing.py:948`；
bridge `training/checkpointing.py:1456-1462`）——pickle CUDA storage 时
隐式整体 D2H。

**local checkpoint（nvrx）**：`core:dist_checkpointing/tensor_aware_state_dict.py:245-273`
`copy_tensors_to_cpu()` 保存前全部 D2H，异步写完后 `restore_tensor_device()`
再 H2D 还原。

**`also_save_hf_checkpoint=True`**：bridge `training/checkpointing.py::_save_hf_weights`
L886-1005——在每次 save 的关键路径上追加一次全模型 HF 导出（走 §2.3 的
GPU→CPU）；PEFT 运行则走 `_save_hf_adapter_weights` L1007-1057。

**其他**：FP8 extra-state pickle（`core:extensions/transformer_engine.py:2440-2449`）；
RNG 状态 `torch.cuda.get_rng_state()` 返回 CPU 字节（小）；MoE/Mamba 工厂合并
OOM 回退 `cat_with_oom_fallback`（`core:transformer/utils.py:89`，PEFT 对应
bridge `peft/utils.py:1655-1666`）；调试转储——wgrad dump
（`training:training.py:2439-2456` 全部 `main_grad.cpu()`）、dgrad logging
（`training:dgrad_logging.py:63-80`）。

### 2.5 Checkpoint 加载（disk→CPU→GPU）

**torch_dist**：DCP 读入后，fully-parallel load 的交换会把 CPU 分片
`.cuda()` 走 NCCL 再 `.to(orig_device)` 弹回
（`core:dist_checkpointing/strategies/exchange_utils.py:339,369-371,506`）。

**优化器态最终 H2D**：`distrib_optimizer.py::_set_main_param_and_optimizer_states`
L1223-1225——CPU shard → GPU fp32 主参数 / exp_avg / exp_avg_sq；各
`load_parameter_state_from_*`（dp_zero / fully_reshardable / dp_reshardable /
fs_model_space）殊途同归。

**legacy**：`torch.load(map_location='cpu')` + `module.load_state_dict`
（全参数 H2D）+ `optimizer.load_state_dict`（全状态 H2D）
（`training:checkpointing.py:1739,2506-2573`；bridge 对应
`training/checkpointing.py:3093,3109`）。

**FP8 extra-state decode**：`transformer_engine.py:2451-2464`
（TE 2.0 路径 `.cpu().numpy().tobytes()`；legacy 路径
`torch.load(map_location="cuda")`）。

### 2.6 RL / RLHF（相位切换级，非 offload 特性中数据量最大者）

| 路径 | 代码 | 方向 | 触发 |
|---|---|---|---|
| 整优化器 offload/restore | `core:optimizer/optimizer.py:555-593` `MegatronOptimizer.offload_to_cpu()/restore_from_cpu()` | 全部优化器态 + inner fp32 主参数 GPU↔CPU | `--rl-offload-optimizer-during-inference`，每次相位切换（`megatron/rl/rl_utils.py:624,1901,2191,2241`） |
| DDP 参数/梯度 buffer offload | `core:distributed/param_and_grad_buffer.py:1619-1649` `offload_to_cpu/reload_from_cpu`（pinned + `non_blocking`） | 参数 buffer GPU→CPU(pinned)、grad 存储释放/重建 | RL 推理相位前后（`distributed_data_parallel.py:636-672`） |
| UVM 托管内存权重预取 | `core:inference/unified_memory.py`（`cudaMallocManaged` + `cudaMemPrefetchAsync`） | host↔device | `--rl-offload-inference-model-weights-when-idle`，每次相位切换 |
| 参考策略快照 | `training:training.py:3396-3417` `{k: v.cpu()}` 全 state_dict | GPU→CPU | RL setup 一次 |
| 每步 refit | bridge `examples/rl/rlhf_with_bridge.py::refit_hf_from_megatron` L177-196 | 全权重 GPU→CPU→HF 模型再 H2D | **每训练步** |
| 在线 resharding | `core:resharding/copy_services/gloo_copy_service.py:88-124` | GPU→pinned CPU→gloo→CPU→GPU | resharding / 权重传输 |

### 2.7 PEFT / LoRA

- adapter 导出：`bridge:models/conversion/peft_bridge.py:843-1087`
  `.cpu()` 流式导出；`auto_bridge.py::save_hf_adapter` L813-921 收集进 host
  后 rank0 写 safetensors。
- shared-expert 同步的 CPU↔GPU 往返：`bridge:peft/utils.py:1285-1314`
  `_synchronize_shared_expert_parameters`——adapter 权重驻留 CPU 而 EP 组是
  NCCL 时，`weight.to(cuda)` → broadcast → `copy_(staged.cpu())`，
  **每个权重一次完整往返**。

### 2.8 推理与 diffusion pipeline

- 推理入口整模型上卡：bridge `inference/text_generation.py:202`、
  `inference/vlm/base.py:94,147`、`scripts/inference/vlm_generation.py:229`
  及各 examples（gloo/CPU 加载后 `model.cuda()`）。
- WAN diffusion pipeline（`bridge:diffusion/models/wan/flow_matching/flow_inference_pipeline.py`）
  默认 `offload_model=True`：每个 prompt T5 编码器 CPU→GPU→CPU（L381-389），
  采样循环内 DiT 每 timestep `.to(device)` / `.cpu()`（L493,540-542）——
  组件权重反复往返（属推理显存管理，非训练 offload 特性）。
- FLUX pipeline 初始化时 T5/CLIP/VAE 上卡（`flux_inference_pipeline.py:412-435`）。

### 2.9 调试 / 校验路径

- **参数哈希一致性检查**：`core:utils.py:891-960`
  `check_param_hashes_across_dp_replicas`——**每个参数**
  `.to("cpu").float().numpy()` 后 sha1；按
  `--check-weight-hash-across-dp-replicas-interval` 周期触发（eval 时也触发）。
- **Router trace**：`core:transformer/moe/router_trace.py:341-351`——router
  `weight` / `expert_bias` GPU→CPU 落盘。

## 3. 训练循环内的小标量 D2H：时序与根因

这些传输只有字节级数据量，**成本在于附带的流同步**（见 §4）。根因是同一
类：值在 GPU 上算出，但消费者是 CPU 侧 Python 控制流或 host-only API 参数。

### 3.1 一次迭代的时间轴

```text
forward (每 microbatch)          backward           grad sync            optimizer.step()
┌──────────────────────┐   ┌──────────────┐   ┌─────────────────┐   ┌──────────────────────────┐
│ loss_func 算出        │   │              │   │ DP reduce-scatter│   │ prepare_grads:           │
│ num_tokens (GPU)     │   │              │   │ /all-reduce      │   │   unscale + found_inf    │← D2H #1 (仅 fp16)
│ loss /= num_tokens   │   │              │   │ (overlap 时分散在 │   │ clip_grad_norm:          │← D2H #2
│ total_num_tokens +=  │   │              │   │  backward 中)     │   │   total_norm.item()      │
└──────────────────────┘   └──────────────┘   └─────────────────┘   │   inner Adam step (GPU)  │
                                              finalize_model_grads:  │ copy main→model (GPU)    │
                                              num_tokens 广播+AR+缩放 └──────────────────────────┘
                                              (全程 GPU，无 D2H)
```

### 3.2 grad-norm 与 found-inf / loss scale：都在 `optimizer.step()` 内部

调用链：`core:optimizer/optimizer.py::MegatronOptimizer.step()` L816-846，
位于**梯度同步全部完成之后、inner Adam step 之前**：

1. **`prepare_grads()`（L820 → L745-772）——found-inf / loss scale**
   - 仅当有 `grad_scaler`（**fp16 + DynamicGradScaler**）时执行
     `_unscale_main_grads_and_check_for_nan()`（L713-738）：
     - `torch._amp_foreach_non_finite_check_and_unscale_` 在 GPU 写
       `found_inf`（L724），跨 DP all-reduce（NCCL，GPU↔GPU）；
     - **`found_inf_flag = self.found_inf.item() > 0`（L736）——D2H 同步点**；
     - `grad_scaler.update(found_inf_flag)`（L770）：growth/hysteresis 计数器
       是 Python int，scale 升降决策在 host 上。
   - `if found_inf_flag: return False, None, None`（L820-821）——溢出时整个
     step 被跳过。**这个 Python 提前返回就是必须 D2H 的根本原因**。
   - BF16 训练（bridge 默认）没有 grad_scaler，此段直接 `return False`，
     **不发生** found-inf D2H。
2. **`clip_grad_norm()`（L830 → L374-430）——grad-norm**
   - `get_grad_norm_fp32`：GPU kernel 算 L2 norm + NCCL all-reduce；
   - `total_norm.item() ** (1.0/norm_type)`（`core:optimizer/clip_grads.py:138`）
     ——**D2H 同步点**，host 得到 Python float 后计算 clip 系数；
   - 注意 L135-137：`multi_tensor_scale_tensor_impl` 可用时 `pow` **留在
     GPU**，fused clip 路径不需要每步 `.item()`，host 值推迟到日志上报。
3. **`step_with_ready_grads()`（L777）**——inner Adam +
   `_copy_main_params_to_model_params`，全 GPU，无 host 值需求。

`overlap_grad_reduce` 隐藏的是 backward 期间的 bucket reduce 本身，
**隐藏不了 step 内的这两个同步点**——`.item()` 必须等前序 NCCL 落地。

### 3.3 MoE dispatcher：区分两种"通信"

**(a) token 数据本身是 GPU↔GPU**：dispatch/combine 走 NCCL
`all_to_all_single`（all-to-all-v，NVLink/IB），数据从不经过 host。

**(b) 路由元数据是 GPU→CPU，且卡在关键路径上**：

- `core:transformer/moe/token_dispatcher.py:317`
  `tokens_per_expert = self.local_map.sum(dim=0).long().cpu()`；
- `_maybe_dtoh_and_synchronize()` L913-951 搬运 `input_splits` /
  `output_splits` / `output_splits_tp` / `num_out_tokens` /
  `num_global_tokens_per_local_expert`。

必须在 host 上拿到这些值的三个原因：

1. **NCCL all-to-all-v 的 `input/output_split_sizes` 是 host API 参数**
   （`ncclSend/ncclRecv` 的 count 在 launch 时必须确定）——每 rank 收发多少
   token 是 router 在 GPU 上算出的直方图；
2. **输出 buffer 的 shape 依赖它**：`torch.empty(num_out_tokens, hidden)`；
3. permute kernel 的 `num_out_tokens=tokens_per_expert.sum().item()`。

因此 dropless alltoall dispatcher 每个 MoE 层都有一次
"router 算完 → D2H splits → CPU 发起 alltoall"的往返，D2H 隐式要求 router
的前序 kernel 全部完成，排空 CPU 异步 launch 流水线，形成 GPU 气泡。

**MCore 的缓解机制**（都在这段代码里）：

- `cuda_dtoh_point` / `cuda_sync_point`：D2H 放到专用 side stream 上以
  `non_blocking=True` 提前发出（`core:transformer/moe/moe_utils.py:1169-1190`
  `maybe_move_tensor_to_cpu`），记 event（`d2h_event`），直到真正要用 splits
  才 `d2h_event.synchronize()`，让 router 之后的其他 kernel 与 D2H 重叠；
- `drop_and_pad`（固定 capacity）：splits 静态可知 → **完全无 D2H**，
  这也是 full CUDA graph 只支持 drop-and-pad MoE 的原因；
- flex dispatcher（DeepEP / HybridEP，`moe_token_dispatcher_type="flex"`）：
  **GPU-side routing**，splits 留在 device 上，通信 kernel 直接读 device
  内存，消掉 host 往返。

### 3.4 num-tokens：产生于 forward，标准路径全程留在 GPU

生命周期分三段：

1. **产生**：每 microbatch 的 loss_func。bridge `training/losses.py:102`
   `num_tokens = loss_mask.sum().clone().detach().to(torch.int)`——GPU
   tensor，**无 D2H**（真实 token 数只有 GPU 上的 loss_mask 知道）。
2. **消费点一（forward 内）**：`core:pipeline_parallel/schedules.py:335`
   `output_tensor /= torch.clamp(num_tokens, min=1)`——默认（非
   per-token-loss）路径到此结束，GPU↔GPU。
3. **消费点二（schedule 末尾，`calculate_per_token_loss=True` 时）**：
   `total_num_tokens`（GPU tensor，逐 microbatch 累加）传入
   `finalize_model_grads`（`schedules.py:861-868` →
   `core:distributed/finalize_model_grads.py:680-696`）：
   - PP last stage → 其他 stage `broadcast`（NCCL，GPU↔GPU）；
   - DP×CP `all_reduce`（NCCL，GPU↔GPU）；
   - `scaling = 1.0 / safe_num_tokens` → `scale_gradients(scaling)`（GPU）。
   - 代码注释明言（L691-692）：clamp 是为了避免 host 分支——
     *"which would otherwise cause a sync that is illegal during CUDA graph
     capture"*。**主路径是刻意全程留在 GPU 上的，没有 CPU↔GPU 通信**。

真正发生 D2H 的只有两条旁路：

- **日志上报**：`losses.py:103` 的 `reporting_loss = cat([loss, num_tokens])`
  进 `forward_data_store`，训练循环在 log interval all-reduce 后
  `.item()` / `.cpu()`（如 bridge `training/eval.py:295`）——按上报周期触发；
- **hierarchical / hybrid CP schedule**：
  `core:pipeline_parallel/hybrid_cp_schedule.py:609,642,665`
  `total_num_tokens += num_tokens.item()`——host 端累加，每 microbatch 一次 D2H。

### 3.5 其他每步标量（汇总）

grad-norm 变体（per-group flag，`optimizer.py:694` `bool(flag.item() > 0)`）、
count_zeros、loss scale 上报、num-tokens 上报（见上）、MoE `paged_stash.py:713`、
`token_dispatcher.py:1074`（drop-and-pad 的 `max_num_tokens_across_ep.item()`）、
data broadcast 的 sizes 列表（`core:tensor_parallel/data.py:37-42`）、
loss/日志 `.item()`（bridge `training/eval.py:295`、`flop_utils.py:84` 等）。
量小，但每一处都是潜在的流水线排空点。

## 4. 底层机制：是不是都走 DMA？

**字节移动层面都是 DMA（GPU copy engine），但同步行为决定性能。**
`.item()` / `.cpu()` / `.to()` 最终都落到 `cudaMemcpy(Async)` D2H/H2D。

| 路径 | 机制 | 同步行为 |
|---|---|---|
| `.item()` / 阻塞 `.cpu()`（pageable 目标内存） | copy engine DMA 进驱动内部 **pinned staging buffer**，再 CPU memcpy 到 pageable 目标 | **同步**：等流上前序 kernel 全部完成才返回。几字节的拷贝，成本几乎全是 sync 延迟 + launch 流水线排空，不是带宽 |
| pinned memory + `non_blocking=True`（MoE dtoh side stream、DDP buffer offload、TRT-LLM 导出 buffer） | 真正异步 DMA，copy engine 直写 pinned host 内存，与计算 kernel 跨 stream 并行 | 异步，配合 event 延迟同步（§3.3） |
| NCCL all-to-all / all-reduce（token、梯度、num_tokens 广播） | GPU↔GPU DMA（NVLink copy / IB GPUDirect RDMA），**不经过 host 内存** | host 只负责 launch |
| H2D 小标量（clip 系数、`torch.tensor([...], device='cuda')`） | copy engine DMA（pageable 源经 staging） | 通常异步，成本可忽略 |

要点：

1. 标量路径的瓶颈是**同步**而非带宽——优化手段（side stream + pinned +
   event、`drop_and_pad`、GPU-side routing 的 flex dispatcher、fused clip）
   针对的都是消除或推迟同步点；
2. **GB200（Grace-Blackwell）**：H2D/D2H 走 NVLink-C2C（~900 GB/s，内存语义
   一致），大块数据（checkpoint、optimizer offload）搬运带宽远好于 PCIe
   平台；但标量路径的同步延迟依旧存在，C2C 帮助有限；
3. pageable 内存的"DMA + staging + CPU memcpy"两跳解释了为什么 pinned
   buffer 在所有大块 D2H 路径（checkpoint preload、TRT-LLM 导出、RL buffer
   offload）中都是标配。

## 5. 扫描方法与已知修正

- 本文由两次全仓库扫描（megatron-bridge 自身代码 + `3rdparty/Megatron-LM`
  的 `megatron/core`、`megatron/training`）汇总而成，搜索模式包括 `.cpu()` /
  `.cuda()` / `.to(device)` / `pin_memory` / `non_blocking` /
  `use_cpu_initialization` / `torch.save` / `state_dict` 等。
- 修正记录：`schedules.py:773/1146/2318` 的 `total_num_tokens` 行是 GPU 零
  tensor 的**创建**而非 D2H；标准 per-token-loss 路径的 num_tokens 全程留在
  GPU（§3.4），早期扫描将其归为每步标量 D2H 不准确。
- 未计入：输入 batch 的 H2D（dataloader `.cuda(non_blocking=True)` /
  `pin_memory`，搬运的是 token/pixel 而非参数）、GPU↔GPU 集合通信、
  以及开头列出的三类 offload 特性。
