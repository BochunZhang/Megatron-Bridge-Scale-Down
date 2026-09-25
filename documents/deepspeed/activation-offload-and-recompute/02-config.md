推荐配置：HF 逐层重计算 + DeepSpeed 激活 CPU 卸载

1. 逐层重计算（HF 原生）
HF 的 gradient checkpointing 本身就是逐 decoder layer 包 checkpoint，直接开启即可，但务必用非重入模式（重入模式与 ZeRO-3、冻结层不兼容，且卸载 ctx 的 marker patch 面向非重入路径）：

```python
model = AutoModelForCausalLM.from_pretrained(...)

model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
model.config.use_cache = False   # 与 gradient checkpointing 冲突，必须关
model.enable_input_require_grads()  # 若有冻结 embedding / LoRA
```


2. CPU 卸载（DeepSpeed CheckpointHiddenStatesOffload）

```python
from deepspeed.runtime.activation_checkpointing.offload_activations import (
    get_checkpoint_hidden_states_offloading_ctx_manager,
)
# 创建一次，每个训练 step 复用同一个 manager
offload_ctx = get_checkpoint_hidden_states_offloading_ctx_manager()

for batch in dataloader:
    with offload_ctx:                       # forward 和 backward 必须在同一个 ctx 内
        loss = model(**batch).loss
        loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

它的工作方式：进入 ctx 时 patch HF 的 GradientCheckpointingLayer.__call__，把每个 checkpoint 层的输入 hidden_states 打标记；pack hook 只把被标记的层间激活异步 D2H 到 pinned CPU 缓冲（side stream，与计算重叠），backward 需要时再 H2D 取回。其余 saved tensors（final norm、lm_head 等）原样透传。

可调参数（一般用默认即可）：

```python
get_checkpoint_hidden_states_offloading_ctx_manager(
    use_pin_memory=True,        # pinned host buffer，保持 True
    use_streams=True,           # side stream 异步拷贝，保持 True
    min_offload_bytes=1024,     # 小于此尺寸的张量不卸载
    max_fwd_stash_count=2,      # forward 在途 GPU 引用数
    keep_last_count=1,          # 最近 N 个输入留 GPU（backward 立刻要用）
    max_cpu_buffer_pool_count=64,
)
```

GPU 上额外驻留的激活峰值 ≈ max_fwd_stash_count + keep_last_count 个层的 hidden_states，其余全部在 CPU。

3. 环境变量（官方教程明确要求）

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # offload/restore 循环易产生碎片
# DS_PIN_MEMORY_BACKEND 保持默认 torch —— 不要设成 native！
# native 后端是 mlock、未做 cudaHostRegister，side-stream DMA 会 stall