# DeepSpeed Offload 组合说明

本文基于 DeepSpeed v0.19.6 源码，说明 ZeRO-Offload、ZeRO-Infinity 和 SuperOffload 的关系，以及如何配置参数、optimizer state 和 optimizer computation 的卸载位置。

## 结论速览

| 说法 | 判断 | 准确表述 |
| --- | --- | --- |
| ZeRO-Offload 将 optimizer 卸载到 CPU 运行 | 基本正确 | `offload_optimizer` 会将 optimizer state 放到 CPU，并在 CPU 执行 optimizer update。 |
| SuperOffload 将部分 optimizer 放到 CPU、部分留在 GPU | 有条件正确 | `0 < ratio < 1` 时才是 CPU/GPU 混合更新；默认 `ratio=1.0`，所有 optimizer 子组都走 CPU。 |
| ZeRO-Infinity 将参数放到 CPU/NVMe，需要时加载到 GPU | 基本正确 | ZeRO-Infinity 是基于 ZeRO-3 的完整 model-state offload 体系，参数只是其中一部分。 |
| ZeRO-Infinity 可以和 ZeRO-Offload/SuperOffload 直接叠加 | 需要修正 | 这些不是彼此完全独立的 stage 开关，而是通过同一个 ZeRO-3 配置分别指定参数和 optimizer 的 offload 行为。 |

## 1. ZeRO-Offload

`offload_optimizer` 控制 optimizer state 的存放位置，同时控制 optimizer update 的执行位置：

- `device: "cpu"`：optimizer state 在 CPU，optimizer computation 在 CPU。
- `device: "nvme"`：optimizer state 可以换出到 NVMe，但 optimizer computation 仍然在 CPU。
- CPU offload 支持 ZeRO stage 1、2、3；optimizer state 的 NVMe offload 只支持 ZeRO-3。

ZeRO-Offload 不等于把模型参数全部放到 CPU。典型的 ZeRO-1/2 配置中，模型参数仍然参与 GPU 上的 forward/backward，主要卸载的是 optimizer 相关状态和计算；ZeRO-2 还会处理分区后的梯度状态。

DeepSpeed 配置文档明确写明，`offload_optimizer` 同时涉及 optimizer state 和 optimizer computation，且无论 state 的 device 选项是什么，optimizer computation 都在 CPU 执行：

- [optimizer offloading 配置](/Users/zhangbochun/Desktop/AI-Infra/Megatron/Megatron-Scale-Down/deepspeed/docs/_pages/config-json.md:685)
- [ZeRO-Offload 教程](/Users/zhangbochun/Desktop/AI-Infra/Megatron/Megatron-Scale-Down/deepspeed/docs/_tutorials/zero-offload.md:9)

## 2. SuperOffload

SuperOffload 是基于 ZeRO-3 的 optimizer 执行优化。它使用 CPU optimizer worker，并可以同时使用 GPU optimizer 处理另一部分 optimizer 子组，从而形成异构 CPU/GPU optimizer computation。

### `ratio` 的语义

`zero_optimization.offload_optimizer.ratio` 表示在 CPU 侧执行 optimizer update 的参数比例：

- `ratio = 1.0`：所有 optimizer 子组都分配到 CPU；这是默认值。
- `0 < ratio < 1`：一部分 optimizer 子组在 CPU，另一部分在 GPU。
- `ratio = 0.0`：没有 optimizer 子组分配到 CPU；通常不适合作为 SuperOffload 的实际配置。

源码按 optimizer 子组建立 CPU/GPU 映射，并在 GPU 子组上使用 backup optimizer：

- [ratio 默认值和 SuperOffload 配置字段](/Users/zhangbochun/Desktop/AI-Infra/Megatron/Megatron-Scale-Down/deepspeed/deepspeed/runtime/zero/offload_config.py:99)
- [CPU/GPU 子组分配](/Users/zhangbochun/Desktop/AI-Infra/Megatron/Megatron-Scale-Down/deepspeed/deepspeed/runtime/zero/stage3.py:1053)
- [CPU/GPU optimizer step 分支](/Users/zhangbochun/Desktop/AI-Infra/Megatron/Megatron-Scale-Down/deepspeed/deepspeed/runtime/zero/stage3.py:1203)
- [SuperOffload 选择 Stage-3 实现](/Users/zhangbochun/Desktop/AI-Infra/Megatron/Megatron-Scale-Down/deepspeed/deepspeed/runtime/engine.py:2566)

因此，“SuperOffload 总是把 optimizer 一部分放 CPU、一部分放 GPU”并不准确。只有设置合适的 `ratio` 时才成立；启用 `super_offload` 本身不会自动选择 50/50 的比例。

SuperOffload 当前实现还有两个重要限制：

- 只支持 ZeRO-3。
- 只支持 NVIDIA CUDA accelerator，见 [SuperOffload accelerator 校验](/Users/zhangbochun/Desktop/AI-Infra/Megatron/Megatron-Scale-Down/deepspeed/deepspeed/runtime/superoffload/superoffload_stage3.py:20)。

SuperOffload 的设计文档也明确描述了 GPU/CPU optimizer computation 的分区：

- [SuperOffload 异构 optimizer computation](/Users/zhangbochun/Desktop/AI-Infra/Megatron/Megatron-Scale-Down/deepspeed/blogs/deepspeed-superoffload/README.md:84)

## 3. ZeRO-Infinity

ZeRO-Infinity 是基于 ZeRO-3 的完整 model-state offload 方案。它可以分别处理：

- 参数：通过 `offload_param.device` 放到 CPU 或 NVMe。
- optimizer state：通过 `offload_optimizer.device` 放到 CPU 或 NVMe。
- 分区梯度和其他 ZeRO-3 状态：由 ZeRO-3 的分区和 offload 逻辑管理。

参数不需要永久驻留在 GPU。ZeRO-3 会在 forward/backward 需要参数时将对应分区取回 accelerator，计算完成后根据持久化、复用距离和内存策略释放或换出。

因此，“ZeRO-Infinity 只负责参数 offload”是不完整的；它的价值在于能够把更多 model state 分层放到 GPU、CPU 和 NVMe。官方教程将其描述为 ZeRO-3 的 full model-state offload：

- [ZeRO-Infinity 配置和说明](/Users/zhangbochun/Desktop/AI-Infra/Megatron/Megatron-Scale-Down/deepspeed/docs/_tutorials/zero.md:123)
- [参数 offload 配置](/Users/zhangbochun/Desktop/AI-Infra/Megatron/Megatron-Scale-Down/deepspeed/docs/_pages/config-json.md:634)

## 4. 如何组合

不要把 ZeRO-Offload、ZeRO-Infinity 和 SuperOffload 当成三个可以任意同时打开的独立 stage。实际组合关系是：

```text
ZeRO stage
  └── stage 3
       ├── offload_param.device       -> 参数放 CPU 或 NVMe
       ├── offload_optimizer.device   -> optimizer state 放 CPU 或 NVMe
       ├── offload_optimizer.ratio    -> CPU/GPU optimizer update 比例
       └── offload_optimizer.super_offload
                                      -> 使用 SuperOffload Stage-3 实现
```

例如，下面的配置表示参数放 NVMe，并让 optimizer state/update 的一部分子组在 CPU、另一部分在 GPU：

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_param": {
      "device": "nvme",
      "nvme_path": "/local_nvme"
    },
    "offload_optimizer": {
      "device": "cpu",
      "ratio": 0.5,
      "super_offload": true
    }
  }
}
```

这个配置可以理解为“ZeRO-3 参数 offload + optimizer offload + SuperOffload”，而不是同时运行一个 ZeRO-2 ZeRO-Offload stage 和一个 ZeRO-3 ZeRO-Infinity stage。这里的 NVMe 参数 offload 与 SuperOffload 的组合应在目标 DeepSpeed 版本和硬件上实际验证；源码会将两类配置传入同一个 Stage-3 optimizer，但没有针对该组合的单独兼容性校验。

### 组合时的限制

- `offload_param` 只支持 ZeRO-3。
- `offload_optimizer` 的 NVMe state offload 只支持 ZeRO-3。
- `ratio < 1` 的 partial offload 只支持 ZeRO-3，见 [ratio 校验](/Users/zhangbochun/Desktop/AI-Infra/Megatron/Megatron-Scale-Down/deepspeed/deepspeed/runtime/zero/config.py:389)。
- `super_offload` 只支持 ZeRO-3，并且当前只支持 NVIDIA CUDA。
- SuperOffload 的 optimizer offload 依赖 CPU optimizer；部分 offload 场景还要求使用 `DeepSpeedCPUAdam`。

## 5. EP 与 Offload 的组合

这里需要区分 Native DeepSpeed MoE EP 和 AutoEP。两者的 ZeRO-3 支持边界不同。

| EP 路径 | ZeRO-Offload | SuperOffload |
| --- | --- | --- |
| Native DeepSpeed MoE EP | 支持 ZeRO-1/2，推荐 ZeRO-2 | 不支持 |
| AutoEP | 支持 ZeRO-1/2；也支持受限的 ZeRO-3 CPU offload | 没有正式验证，不应视为稳定支持 |
| AutoEP + AutoTP folding | offload 被明确拒绝 | 不支持 |

### Native MoE EP + ZeRO-Offload

这是 DeepSpeed 文档明确列出的组合：`Expert + ZeRO-Offload + Model`，见 [MoE 并行配置](/Users/zhangbochun/Desktop/AI-Infra/Megatron/Megatron-Scale-Down/deepspeed/docs/_tutorials/mixture-of-experts.md:23)。

ZeRO-1/2 初始化时会同时传入 expert-parallel group、expert-data-parallel group 和 `offload_optimizer_config`，见 [engine.py:2488](/Users/zhangbochun/Desktop/AI-Infra/Megatron/Megatron-Scale-Down/deepspeed/deepspeed/runtime/engine.py:2488)。对于 expert 参数，optimizer 会使用对应的 expert-data-parallel group 进行分区，而普通参数继续使用普通 data-parallel group，见 [stage_1_and_2.py:778](/Users/zhangbochun/Desktop/AI-Infra/Megatron/Megatron-Scale-Down/deepspeed/deepspeed/runtime/zero/stage_1_and_2.py:778)。

ZeRO-2 + MoE 要求 `contiguous_gradients=true` 和 `reduce_scatter=true`；ZeRO-1 的 MoE 路径仍被源码标记为实验性。因此，EP + CPU optimizer offload 的首选是 ZeRO-2。

示意配置：

```json
{
  "expert_parallel": {
    "enabled": true,
    "autoep_size": 4,
    "preset_model": "mixtral"
  },
  "zero_optimization": {
    "stage": 2,
    "contiguous_gradients": true,
    "reduce_scatter": true,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    }
  }
}
```

### AutoEP + ZeRO-3 Offload

AutoEP 文档声明支持 ZeRO-0/1/2 以及受限的 ZeRO-3。ZeRO-3 会将 AutoEP expert 参数绑定到 expert-data-parallel group，普通参数绑定到普通 data-parallel group，见 [engine.py:2012](/Users/zhangbochun/Desktop/AI-Infra/Megatron/Megatron-Scale-Down/deepspeed/deepspeed/runtime/engine.py:2012)。CPU offload 的 ZeRO-3 梯度范数逻辑也会额外处理 expert-parallel group，仓库有对应单测，见 [test_autoep_unit.py:513](/Users/zhangbochun/Desktop/AI-Infra/Megatron/Megatron-Scale-Down/deepspeed/tests/unit/v1/moe/test_autoep_unit.py:513)。

但此路径是受限支持：

- Native DeepSpeed MoE 不能使用 ZeRO-3，必须改用 AutoEP，见 [engine.py:1958](/Users/zhangbochun/Desktop/AI-Infra/Megatron/Megatron-Scale-Down/deepspeed/deepspeed/runtime/engine.py:1958)。
- AutoEP + ZeRO-3 不能同时使用 AutoTP、sequence parallel、MiCS、hpZeRO secondary groups 或 quantized gradients。
- AutoEP + ZeRO-3 的 NVMe optimizer swapping 不能正常保存 checkpoint，见 [engine.py:5132](/Users/zhangbochun/Desktop/AI-Infra/Megatron/Megatron-Scale-Down/deepspeed/deepspeed/runtime/engine.py:5132)。因此，优先使用 CPU optimizer offload。

### EP + SuperOffload

Native MoE EP + SuperOffload 不支持。SuperOffload 只工作在 ZeRO-3，而 Native DeepSpeed MoE 会被 ZeRO-3 兼容性检查明确拒绝。

AutoEP + SuperOffload 在源码结构上没有显式拒绝：SuperOffload 选择的是 ZeRO-3 子类，而该子类继承了 AutoEP 的 ZeRO-3 参数分组和梯度逻辑。但当前仓库没有 SuperOffload + AutoEP/EP 的专门文档、单元测试或端到端测试，因此不能把它视为正式稳定支持，尤其不应直接假设它覆盖 NVMe 参数/optimizer offload、checkpoint 和复杂并行组合。

如果使用 AutoTP folding，ZeRO optimizer/parameter offload 会被直接拒绝，见 [auto_ep_folding.py:248](/Users/zhangbochun/Desktop/AI-Infra/Megatron/Megatron-Scale-Down/deepspeed/deepspeed/module_inject/auto_ep_folding.py:248)。

## 最终判断

1. **ZeRO-Offload 将 optimizer state 和 optimizer update 放到 CPU：正确。**
2. **SuperOffload 将 optimizer 分到 CPU/GPU：有条件正确，必须结合 `ratio`；默认不是部分分配。**
3. **ZeRO-Infinity 将参数按需从 CPU/NVMe 取到 GPU：正确，但它还覆盖 optimizer state 等更多 model state。**
4. **可以按参数和 optimizer 分别配置 offload，也可以在 ZeRO-3 上启用 SuperOffload；但不是把多个 ZeRO stage 直接叠加。**
5. **EP + ZeRO-Offload：Native MoE EP 在 ZeRO-1/2 有明确支持，推荐 ZeRO-2；AutoEP 还可使用受限的 ZeRO-3 CPU offload。**
6. **EP + SuperOffload：Native MoE EP 不支持；AutoEP 目前只有未验证的源码组合，不应视为生产级支持。**
