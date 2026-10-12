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
"""Build the exact Megatron modules a Qwen3.5/3.8-style hybrid model trains with.

Every module comes from the same spec functions training uses
(``get_transformer_layer_with_experimental_attention_variant_spec`` and the
MoE/MTP specs), with a Bridge ``GPTModelProvider`` as the config, on a
single-rank process group. Parameters receive fp32 ``main_grad`` buffers and
the Megatron DDP backward hook, so TE layers take the fused wgrad-accumulation
path that training takes.
"""

from __future__ import annotations

import copy
import logging
import os
import socket
from functools import cached_property
from typing import Any

import torch
import torch.distributed as dist
from megatron.core import parallel_state, tensor_parallel
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_linear_attention_pattern,
    get_transformer_layer_with_experimental_attention_variant_spec,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.spec_utils import build_module

from megatron.bridge.models.gpt_provider import GPTModelProvider


LOGGER = logging.getLogger(__name__)
_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def resolve_dtype(name: str) -> torch.dtype:
    """Map a config dtype name to a torch dtype."""

    try:
        return _DTYPES[name]
    except KeyError:
        raise ValueError(f"Unsupported dtype {name!r}; expected one of {sorted(_DTYPES)}") from None


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def init_single_rank(*, device_index: int, seed: int) -> ProcessGroupCollection:
    """Initialize a world-size-1 NCCL group and Megatron parallel state (all parallel sizes 1)."""

    torch.cuda.set_device(device_index)
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(_free_port()))
        dist.init_process_group(backend="nccl", rank=0, world_size=1, device_id=torch.device("cuda", device_index))
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(seed)
    return ProcessGroupCollection.use_mpu_process_groups()


def build_provider(model_fields: dict[str, Any], *, sequence_length: int, dtype: torch.dtype) -> GPTModelProvider:
    """Create and finalize the Bridge model config for a single-GPU run.

    Args:
        model_fields: ``GPTModelProvider`` fields from the YAML ``model`` section.
        sequence_length: Benchmark sequence length.
        dtype: Parameter / activation dtype (mixed-precision ``params_dtype``).

    Returns:
        The finalized provider, used directly as the Megatron ``TransformerConfig``.
    """

    fields = dict(model_fields)
    attention_backend = AttnBackend[fields.pop("attention_backend", "auto")]
    provider = GPTModelProvider(**fields)
    provider.attention_backend = attention_backend
    provider.seq_length = sequence_length
    # Mirrors megatron.bridge.training.mixed_precision.bf16_mixed for bf16.
    provider.bf16 = dtype == torch.bfloat16
    provider.fp16 = dtype == torch.float16
    provider.params_dtype = dtype
    provider.pipeline_dtype = dtype
    provider.autocast_enabled = False
    # The benchmark runs on one GPU; the formal EP layout only shapes the expert inputs.
    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    provider.context_parallel_size = 1
    provider.expert_model_parallel_size = 1
    provider.expert_tensor_parallel_size = 1
    provider.sequence_parallel = False
    provider.finalize()
    return provider


def _make_backward_post_hook(param: torch.nn.Parameter):
    """Same accumulate-and-free logic as Megatron ``DistributedDataParallel._make_backward_post_hook``."""

    def hook(*_unused: object) -> None:
        if param.grad is not None and not param.grad_added_to_main_grad:
            param.main_grad.add_(param.grad.data)
        param.grad = None

    return hook


class MegatronModules:
    """Lazily built Megatron modules of one hybrid GDN/attention + MoE decoder.

    Args:
        config: Finalized provider (Megatron ``TransformerConfig``).
        pg_collection: Single-rank process groups.
        local_experts: Routed experts per rank under the formal EP layout.
    """

    def __init__(self, config: GPTModelProvider, pg_collection: ProcessGroupCollection, *, local_experts: int) -> None:
        self.config = config
        self.pg_collection = pg_collection
        self.local_experts = local_experts
        self._layer_specs = get_transformer_layer_with_experimental_attention_variant_spec(config)
        pattern = get_linear_attention_pattern(config)
        if 1 not in pattern or 0 not in pattern:
            raise ValueError("linear_attention_freq must produce both GatedDeltaNet and attention layers")
        self.gdn_layer_number = pattern.index(1) + 1
        self.attention_layer_number = pattern.index(0) + 1
        self._grad_accumulators: list[object] = []

    def _prepare(self, module: torch.nn.Module) -> torch.nn.Module:
        module.cuda()
        module.train()
        for param in module.parameters():
            if not param.requires_grad:
                continue
            param.main_grad = torch.zeros_like(param, dtype=torch.float32)
            param.grad_added_to_main_grad = False
            # Hook the AccumulateGrad node exactly like Megatron DDP does.
            with torch.enable_grad():
                accumulator = param.expand_as(param).grad_fn.next_functions[0][0]
            accumulator.register_hook(_make_backward_post_hook(param))
            self._grad_accumulators.append(accumulator)
        return module

    def _moe_submodules(self) -> Any:
        builder = self._layer_specs[self.gdn_layer_number - 1].submodules.mlp
        submodules = getattr(builder, "keywords", {}).get("submodules")
        if submodules is None:
            raise ValueError("expected an MoE layer spec; set num_moe_experts in the model config")
        return submodules

    @cached_property
    def gated_delta_net(self) -> torch.nn.Module:
        """``GatedDeltaNet`` (TELayerNormColumnParallelLinear in_proj, FLA kernels, TERowParallelLinear out_proj)."""

        spec = self._layer_specs[self.gdn_layer_number - 1].submodules.self_attention
        return self._prepare(
            build_module(
                spec, config=self.config, layer_number=self.gdn_layer_number, pg_collection=self.pg_collection
            )
        )

    @cached_property
    def attention(self) -> torch.nn.Module:
        """``SelfAttention`` with fused input norm, QK norm, RoPE, ``TEDotProductAttention`` and output gate."""

        spec = self._layer_specs[self.attention_layer_number - 1].submodules.self_attention
        return self._prepare(
            build_module(
                spec, config=self.config, layer_number=self.attention_layer_number, pg_collection=self.pg_collection
            )
        )

    @cached_property
    def router(self) -> torch.nn.Module:
        """``TopKRouter`` over all ``num_moe_experts`` (the router is never sharded by EP)."""

        router = self._moe_submodules().router(
            config=self.config, pg_collection=self.pg_collection, is_mtp_layer=False
        )
        router.set_layer_number(self.gdn_layer_number)
        return self._prepare(router)

    @cached_property
    def routed_experts(self) -> torch.nn.Module:
        """``TEGroupedMLP`` holding the local experts of one EP rank."""

        return self._prepare(
            self._moe_submodules().experts(self.local_experts, self.config, pg_collection=self.pg_collection)
        )

    @cached_property
    def shared_experts(self) -> torch.nn.Module:
        """``SharedExpertMLP`` including the sigmoid shared-expert gate."""

        return self._prepare(
            self._moe_submodules().shared_experts(
                config=self.config, pg_collection=self.pg_collection, gate=self.config.moe_shared_expert_gate
            )
        )

    @cached_property
    def mtp_eh_proj(self) -> torch.nn.Module:
        """MTP ``eh_proj`` built with the same arguments as ``MultiTokenPredictionLayer``."""

        return self._prepare(
            build_module(
                TESpecProvider().column_parallel_linear(),
                self.config.hidden_size * 2,
                self.config.hidden_size,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name="mtp_eh_proj",
                tp_group=self.pg_collection.tp,
            )
        )

    @cached_property
    def output_layer(self) -> torch.nn.Module:
        """``tensor_parallel.ColumnParallelLinear`` LM head, as built by ``GPTModel`` for BF16."""

        config = self.config
        if config.gradient_accumulation_fusion and not tensor_parallel.layers._grad_accum_fusion_available:
            # Without APEX fused_weight_gradient_mlp_cuda, ColumnParallelLinear refuses fused wgrad.
            LOGGER.warning("APEX fused wgrad unavailable: output layer uses unfused wgrad")
            config = copy.copy(config)
            config.gradient_accumulation_fusion = False
        return self._prepare(
            tensor_parallel.ColumnParallelLinear(
                self.config.hidden_size,
                self.config.vocab_size,
                config=config,
                init_method=self.config.init_method,
                bias=False,
                skip_bias_add=False,
                gather_output=False,
                skip_weight_param_allocation=False,
                embedding_activation_buffer=None,
                grad_output_buffer=None,
                tp_group=self.pg_collection.tp,
            )
        )

    @property
    def output_layer_fused_wgrad(self) -> bool:
        """Whether the LM head accumulates wgrad into ``main_grad`` with the fused kernel."""

        return bool(self.output_layer.gradient_accumulation_fusion)

    def rotary_pos_emb(self, sequence_length: int) -> torch.Tensor:
        """RoPE frequencies as ``GPTModel`` computes them for ``position_embedding_type='rope'``."""

        rotary = RotaryEmbedding(
            kv_channels=self.config.kv_channels,
            rotary_percent=self.config.rotary_percent,
            rotary_interleaved=self.config.rotary_interleaved,
            rotary_base=self.config.rotary_base,
            use_cpu_initialization=self.config.use_cpu_initialization,
            cp_group=self.pg_collection.cp,
        )
        return rotary(sequence_length)
