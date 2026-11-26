# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configure and report Megatron-FSDP layer-by-layer training."""

import json
import logging
import math
from typing import Literal

import torch
from megatron.core.optimizer import MegatronOptimizer
from torch import distributed as dist

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.training.config import ConfigContainer


logger = logging.getLogger(__name__)


def _normalize_tier(value: object, field_name: str, *, legacy_suffix: str = "") -> Literal["gpu", "cpu", "nvme"]:
    tier = value.removesuffix(legacy_suffix) if isinstance(value, str) else None
    if tier == "gpu" or tier == "cpu" or tier == "nvme":
        return tier
    raise ValueError(f"Layer-by-layer {field_name} must be gpu, cpu, or nvme, got {value!r}.")


def configure_layer_by_layer(config: ConfigContainer) -> None:
    """Enable the native schedule and configure Megatron-FSDP storage tiers."""
    offload = config.layer_by_layer
    assert offload is not None
    offload.parameter_tier = _normalize_tier(offload.parameter_tier, "parameter_tier", legacy_suffix="_param")
    offload.activation_tier = _normalize_tier(offload.activation_tier, "activation_tier", legacy_suffix="_act")
    offload.optimizer_tier = (
        offload.parameter_tier
        if offload.optimizer_tier is None
        else _normalize_tier(offload.optimizer_tier, "optimizer_tier")
    )
    nvme_parameters = offload.parameter_tier == "nvme"
    nvme_activations = offload.activation_tier == "nvme"
    if offload.optimizer_tier == "nvme" and not nvme_parameters:
        raise ValueError("Layer-by-layer NVMe optimizer state requires NVMe parameters.")
    if nvme_parameters and offload.optimizer_tier == "gpu":
        raise ValueError("Layer-by-layer NVMe parameters require the CPU or NVMe optimizer tier.")
    if (nvme_parameters or nvme_activations) and not offload.nvme_dir:
        raise ValueError("Layer-by-layer NVMe requires layer_by_layer.nvme_dir.")
    if offload.activation_nvme_backend not in ("native", "deepspeed"):
        raise ValueError("Layer-by-layer activation_nvme_backend must be native or deepspeed.")
    if offload.activation_nvme_backend == "deepspeed" and not nvme_activations:
        raise ValueError("Layer-by-layer DeepSpeed AIO requires activation_tier=nvme.")
    if offload.cpu_optimizer not in ("grace_adam", "torch") or offload.cpu_threads < 1:
        raise ValueError("Layer-by-layer requires a valid CPU optimizer and a positive CPU thread count.")

    model = config.model
    if not isinstance(model, GPTModelProvider):
        raise ValueError("Layer-by-layer training requires a GPT model provider.")
    if not hasattr(model, "layer_by_layer_schedule"):
        raise ValueError("Layer-by-layer training requires the updated Megatron-LM layer-by-layer branch.")
    cpu_activations = offload.activation_tier == "cpu"
    if cpu_activations and not hasattr(model, "layer_by_layer_activation_offload"):
        raise ValueError(
            "Layer-by-layer CPU activation offload requires the updated Megatron-LM layer-by-layer branch."
        )
    if nvme_activations and not hasattr(model, "layer_by_layer_activation_nvme_path"):
        raise ValueError(
            "Layer-by-layer NVMe activation offload requires the updated Megatron-LM layer-by-layer branch."
        )
    if offload.activation_nvme_backend == "deepspeed" and not hasattr(model, "layer_by_layer_activation_nvme_backend"):
        raise ValueError("Layer-by-layer DeepSpeed AIO requires the updated Megatron-LM layer-by-layer branch.")
    if (
        (
            model.tensor_model_parallel_size,
            model.pipeline_model_parallel_size,
            model.context_parallel_size,
        )
        != (1, 1, 1)
        or model.expert_tensor_parallel_size not in (None, 1)
        or model.virtual_pipeline_model_parallel_size is not None
        or model.pipeline_model_parallel_layout is not None
        or model.sequence_parallel
    ):
        raise ValueError("Layer-by-layer training requires TP=PP=CP=ETP=1 without sequence or virtual parallelism.")
    if model.position_embedding_type != "rope" or model.share_embeddings_and_output_weights:
        raise ValueError("Layer-by-layer training requires rotary embeddings and untied output weights.")
    if not model.bf16 or model.params_dtype != torch.bfloat16 or model.fp16 or model.fp8 or model.fp4:
        raise ValueError("Layer-by-layer training requires BF16 model parameters and compute.")
    if (
        config.peft is not None
        or config.dist.use_torch_fsdp2
        or config.dist.use_decentralized_pg
        or model.cuda_graph_impl != "none"
        or model.cuda_graph_scope
        or config.ddp.megatron_fsdp_cuda_graph_mode
        or config.optimizer.optimizer_cpu_offload
        or config.optimizer.optimizer_cuda_graph
        or model.cpu_offloading
        or model.fine_grained_activation_offloading
        or model.hybrid_context_parallel
        or model.overlap_moe_expert_parallel_comm
        or model.mtp_num_layers
    ):
        raise ValueError("Layer-by-layer training requires eager Megatron-FSDP and standard GPT layers.")
    if getattr(model, "moe_enable_routing_replay", False) or getattr(model, "moe_paged_stash", False):
        raise ValueError("Layer-by-layer training does not support MoE routing replay or paged stash.")
    if model.recompute_granularity is not None and (
        model.recompute_granularity,
        model.recompute_method,
        model.recompute_num_layers,
    ) != ("full", "uniform", 1):
        raise ValueError("The layer-by-layer schedule checkpoints each decoder layer individually.")

    ddp = config.ddp
    optimizer = config.optimizer
    nvme_weights_only = nvme_parameters and offload.optimizer_tier == "cpu"
    if nvme_weights_only and not hasattr(ddp, "fsdp_nvme_weights_only"):
        raise ValueError(
            "Layer-by-layer NVMe parameters with the CPU optimizer require the updated Megatron-LM backend."
        )
    cpu_optimizer = offload.optimizer_tier != "gpu"
    layerwise_grace_adam = offload.optimizer_tier == "cpu" and offload.cpu_optimizer == "grace_adam"
    if hasattr(ddp, "fsdp_layer_by_layer_cpu_optimizer"):
        ddp.fsdp_layer_by_layer_cpu_optimizer = layerwise_grace_adam
    elif layerwise_grace_adam:
        raise ValueError("Layer-by-layer GraceAdam requires the updated Megatron-FSDP layer-by-layer backend.")
    if offload.overlap_cpu_optimizer:
        if not layerwise_grace_adam or offload.parameter_tier == "gpu":
            raise ValueError(
                "Layer-by-layer optimizer overlap requires CPU or NVMe weights and GraceAdam on the CPU tier."
            )
        if not hasattr(ddp, "fsdp_layer_by_layer_overlap_cpu_optimizer"):
            raise ValueError("Layer-by-layer optimizer overlap requires the updated Megatron-FSDP backend.")
        if model.calculate_per_token_loss or model.qk_clip:
            raise ValueError(
                "Layer-by-layer optimizer overlap requires per-microbatch loss normalization and qk_clip=False."
            )

        optimizer.clip_grad = 0.0
        optimizer.log_num_zeros_in_grad = False
        optimizer.loss_scale = None
        optimizer.grad_norm_skip_threshold = float("inf")
        ddp.check_for_nan_in_grad = False
        ddp.check_for_large_grads = False
        config.rerun_state_machine.rerun_mode = "disabled"
    if hasattr(ddp, "fsdp_layer_by_layer_overlap_cpu_optimizer"):
        ddp.fsdp_layer_by_layer_overlap_cpu_optimizer = offload.overlap_cpu_optimizer
    components = []
    if offload.parameter_tier != "gpu":
        components.append("weight")
    if cpu_optimizer:
        if optimizer.optimizer != "adam" or not optimizer.decoupled_weight_decay:
            raise ValueError("Layer-by-layer CPU and NVMe optimizer tiers require AdamW.")
        components.extend(("gradient", "optimizer"))
        optimizer.use_precision_aware_optimizer = False
        ddp.megatron_fsdp_use_decoupled_grad = False
        if hasattr(optimizer, "fsdp_cpu_optimizer_backend"):
            optimizer.fsdp_cpu_optimizer_backend = offload.cpu_optimizer
        elif offload.cpu_optimizer == "grace_adam":
            raise ValueError("Layer-by-layer GraceAdam requires the native Megatron-FSDP GraceAdam backend.")
        torch.set_num_threads(offload.cpu_threads)

    config.dist.use_megatron_fsdp = True
    ddp.use_megatron_fsdp = True
    ddp.use_distributed_optimizer = True
    optimizer.use_distributed_optimizer = True
    ddp.data_parallel_sharding_strategy = "optim_grads_params"
    ddp.average_in_collective = False
    ddp.fsdp_double_buffer = True
    ddp.megatron_fsdp_max_pool_double_buffer = True
    ddp.overlap_grad_reduce = True
    ddp.overlap_param_gather = True
    optimizer.overlap_param_gather = True
    ddp.megatron_fsdp_main_params_dtype = torch.float32
    ddp.megatron_fsdp_main_grads_dtype = torch.float32
    if layerwise_grace_adam:
        ddp.megatron_fsdp_grad_comm_dtype = torch.bfloat16
    ddp.fsdp_cpu_offload = tuple(components)
    ddp.fsdp_cpu_grad_accumulation = cpu_optimizer and not layerwise_grace_adam
    ddp.fsdp_cpu_optimizer = cpu_optimizer
    ddp.fsdp_nvme_path = offload.nvme_dir if nvme_parameters else None
    if hasattr(ddp, "fsdp_nvme_weights_only"):
        ddp.fsdp_nvme_weights_only = nvme_weights_only
    config.checkpoint.ckpt_format = "fsdp_dtensor"

    _apply_workload_geometry(model, tiny=offload.tiny)
    model.layer_by_layer_schedule = True
    if hasattr(model, "layer_by_layer_activation_offload"):
        model.layer_by_layer_activation_offload = cpu_activations
    if hasattr(model, "layer_by_layer_activation_nvme_path"):
        model.layer_by_layer_activation_nvme_path = offload.nvme_dir if nvme_activations else None
    if hasattr(model, "layer_by_layer_activation_nvme_backend"):
        model.layer_by_layer_activation_nvme_backend = offload.activation_nvme_backend
    if config.tokenizer.tokenizer_type == "NullTokenizer":
        config.tokenizer.vocab_size = model.vocab_size


def _apply_workload_geometry(model: GPTModelProvider, *, tiny: bool) -> None:
    from megatron.training.layer_by_layer_workload import (
        QWEN35_LAYER_BY_LAYER_TINY_VOCAB_SIZE,
        QWEN35_LAYER_BY_LAYER_VOCAB_SIZE,
        build_qwen35_layer_by_layer_config,
    )

    if tiny:
        tiny_model = build_qwen35_layer_by_layer_config(model.num_layers, tiny=True)
        for field in (
            "hidden_size",
            "ffn_hidden_size",
            "num_attention_heads",
            "num_query_groups",
            "kv_channels",
            "linear_key_head_dim",
            "linear_value_head_dim",
            "linear_num_key_heads",
            "linear_num_value_heads",
        ):
            setattr(model, field, getattr(tiny_model, field))
        model.vocab_size = QWEN35_LAYER_BY_LAYER_TINY_VOCAB_SIZE
    elif model.vocab_size is None:
        model.vocab_size = QWEN35_LAYER_BY_LAYER_VOCAB_SIZE
    if isinstance(model.linear_attention_freq, list):
        model.linear_attention_freq = model.linear_attention_freq[: model.num_layers]


def log_layer_by_layer_result(
    config: ConfigContainer, optimizer: MegatronOptimizer, *, steps: int, loss: torch.Tensor | None
) -> None:
    """Log completed optimizer steps, final loss, and maximum training peaks across ranks."""
    offload = config.layer_by_layer
    assert offload is not None
    device = torch.cuda.current_device()
    torch.cuda.synchronize(device)
    peaks = torch.tensor(
        [torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)],
        dtype=torch.int64,
        device=torch.device("cuda", device),
    )
    dist.all_reduce(peaks, op=dist.ReduceOp.MAX)
    if dist.get_rank() != 0:
        return
    allocated, reserved = peaks.tolist()
    if offload.optimizer_tier == "gpu":
        optimizers = getattr(optimizer, "chained_optimizers", [optimizer])
        optimizer_backend = "+".join(sorted({type(opt.optimizer).__name__ for opt in optimizers}))
    else:
        optimizer_backend = offload.cpu_optimizer
    final_loss = loss.item() if loss is not None else None
    logger.info(
        "LAYER_BY_LAYER_RESULT %s",
        json.dumps(
            {
                "schedule": "layer_by_layer",
                "parameter_backend": "megatron_fsdp",
                "fsdp_strategy": config.ddp.data_parallel_sharding_strategy,
                "fsdp_cpu_offload": config.ddp.fsdp_cpu_offload,
                "fsdp_cpu_grad_accumulation": config.ddp.fsdp_cpu_grad_accumulation,
                "fsdp_nvme_weights_only": getattr(config.ddp, "fsdp_nvme_weights_only", False),
                "fsdp_layer_by_layer_cpu_optimizer": getattr(config.ddp, "fsdp_layer_by_layer_cpu_optimizer", False),
                "fsdp_layer_by_layer_overlap_cpu_optimizer": getattr(
                    config.ddp, "fsdp_layer_by_layer_overlap_cpu_optimizer", False
                ),
                "clip_grad": config.optimizer.clip_grad,
                "parameter_tier": offload.parameter_tier,
                "activation_tier": offload.activation_tier,
                "activation_nvme_backend": getattr(config.model, "layer_by_layer_activation_nvme_backend", "native"),
                "optimizer_tier": offload.optimizer_tier,
                "optimizer_backend": optimizer_backend,
                "steps": steps,
                "layers": config.model.num_layers,
                "num_moe_experts": config.model.num_moe_experts,
                "expert_model_parallel_size": config.model.expert_model_parallel_size,
                "moe_token_dispatcher_type": config.model.moe_token_dispatcher_type,
                "moe_flex_dispatcher_backend": config.model.moe_flex_dispatcher_backend,
                "loss": final_loss if final_loss is not None and math.isfinite(final_loss) else None,
                "peak_allocated_bytes": allocated,
                "peak_reserved_bytes": reserved,
                "world_size": dist.get_world_size(),
            },
            allow_nan=False,
        ),
    )
