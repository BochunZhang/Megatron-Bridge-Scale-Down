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
"""Every linear layer of the model, executed through the Megatron module that owns it."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import torch
import torch.nn.functional as F

from mbench.cases import BenchCase, BenchContext, ParamValue, first, leaf, linear_flops, local_expert_token_counts
from mbench.timing import Workload


CATEGORY = "gemm"


def _dense_case(
    ctx: BenchContext,
    *,
    name: str,
    megatron_module: str,
    in_features: int,
    out_features: int,
    call: Callable[[torch.Tensor], torch.Tensor],
    mbs: int,
    mode: str,
    extra: dict[str, ParamValue],
) -> BenchCase:
    tokens = ctx.sequence_length * mbs

    def build() -> Workload:
        hidden = leaf((ctx.sequence_length, mbs, in_features), dtype=ctx.dtype)
        return Workload(forward=lambda: (call(hidden),), leaves=[hidden])

    return BenchCase(
        name=name,
        megatron_module=megatron_module,
        micro_batch_size=mbs,
        mode=mode,
        params={"m": tokens, "k": in_features, "n": out_features, "groups": 1, "input_layout": "[s,b,k]", **extra},
        flops=linear_flops(m=tokens, n=out_features, k=in_features, mode=mode),
        build=build,
    )


def _grouped_case(
    ctx: BenchContext,
    *,
    name: str,
    megatron_module: str,
    layer: torch.nn.Module,
    token_counts: list[int],
    mbs: int,
    mode: str,
) -> BenchCase:
    rows = sum(token_counts)

    def build() -> Workload:
        hidden = leaf((rows, layer.in_features), dtype=ctx.dtype)
        return Workload(forward=lambda: (first(layer(hidden, token_counts)),), leaves=[hidden])

    return BenchCase(
        name=name,
        megatron_module=megatron_module,
        micro_batch_size=mbs,
        mode=mode,
        params={
            "m": rows,
            "k": layer.in_features,
            "n": layer.out_features,
            "groups": len(token_counts),
            "rows_per_group": ",".join(str(count) for count in token_counts),
            "input_layout": "[rows,k]",
            "compute_dtype": str(ctx.dtype).removeprefix("torch."),
            "fused_wgrad_accumulation": bool(layer.fuse_wgrad_accumulation),
        },
        flops=linear_flops(m=rows, n=layer.out_features, k=layer.in_features, mode=mode),
        build=build,
    )


def iter_cases(ctx: BenchContext, micro_batch_sizes: list[int], modes: list[str]) -> Iterator[BenchCase]:
    """Yield GEMM cases in (layer, mbs, mode) order."""

    modules = ctx.modules
    config = modules.config
    dtype_name = str(ctx.dtype).removeprefix("torch.")

    def te_linear(layer: torch.nn.Module) -> dict[str, object]:
        return {
            "in_features": layer.in_features,
            "out_features": layer.out_features,
            "call": lambda hidden: first(layer(hidden)),
            "extra": {"compute_dtype": dtype_name, "fused_wgrad_accumulation": bool(layer.fuse_wgrad_accumulation)},
        }

    def te_norm_linear(layer: torch.nn.Module) -> dict[str, object]:
        spec = te_linear(layer)
        spec["extra"] = {**spec["extra"], "fused_input_norm": config.normalization}
        return spec

    dense_layers: list[tuple[str, str, Callable[[], dict[str, object]]]] = [
        (
            "gdn_in_proj",
            "TELayerNormColumnParallelLinear",
            lambda: te_norm_linear(modules.gated_delta_net.in_proj),
        ),
        ("gdn_out_proj", "TERowParallelLinear", lambda: te_linear(modules.gated_delta_net.out_proj)),
        (
            "attention_linear_qkv",
            "TELayerNormColumnParallelLinear",
            lambda: te_norm_linear(modules.attention.linear_qkv),
        ),
        ("attention_linear_proj", "TERowParallelLinear", lambda: te_linear(modules.attention.linear_proj)),
        (
            "router_gating",
            "TopKRouter.gating (router_gating_linear)",
            lambda: {
                "in_features": config.hidden_size,
                "out_features": config.num_moe_experts,
                "call": modules.router.gating,
                "extra": {"compute_dtype": config.moe_router_dtype or dtype_name, "fused_wgrad_accumulation": False},
            },
        ),
        ("shared_expert_fc1", "TEColumnParallelLinear", lambda: te_linear(modules.shared_experts.linear_fc1)),
        ("shared_expert_fc2", "TERowParallelLinear", lambda: te_linear(modules.shared_experts.linear_fc2)),
        (
            "shared_expert_gate",
            "SharedExpertMLP.gate_weight (F.linear)",
            lambda: {
                "in_features": config.hidden_size,
                "out_features": modules.shared_experts.gate_weight.shape[0],
                "call": lambda hidden: F.linear(hidden, modules.shared_experts.gate_weight),
                "extra": {"compute_dtype": dtype_name, "fused_wgrad_accumulation": False},
            },
        ),
        ("mtp_eh_proj", "TEColumnParallelLinear", lambda: te_linear(modules.mtp_eh_proj)),
        (
            "output_layer",
            "tensor_parallel.ColumnParallelLinear",
            lambda: {
                "in_features": modules.output_layer.input_size,
                "out_features": modules.output_layer.output_size_per_partition,
                "call": lambda hidden: first(modules.output_layer(hidden)),
                "extra": {"compute_dtype": dtype_name, "fused_wgrad_accumulation": modules.output_layer_fused_wgrad},
            },
        ),
    ]
    for name, megatron_module, describe in dense_layers:
        if not ctx.wants(name):
            continue
        spec = describe()
        for mbs in micro_batch_sizes:
            for mode in modes:
                yield _dense_case(
                    ctx,
                    name=name,
                    megatron_module=megatron_module,
                    in_features=spec["in_features"],
                    out_features=spec["out_features"],
                    call=spec["call"],
                    mbs=mbs,
                    mode=mode,
                    extra=spec["extra"],
                )

    for name, megatron_module, fc in (
        ("routed_expert_fc1", "TEColumnParallelGroupedLinear", "linear_fc1"),
        ("routed_expert_fc2", "TERowParallelGroupedLinear", "linear_fc2"),
    ):
        if not ctx.wants(name):
            continue
        layer = getattr(modules.routed_experts, fc)
        for mbs in micro_batch_sizes:
            token_counts = local_expert_token_counts(
                tokens=ctx.sequence_length * mbs,
                topk=config.moe_router_topk,
                num_experts=config.num_moe_experts,
                expert_parallel_size=ctx.formal_expert_parallel_size,
                routing={"name": "uniform"},
            )
            for mode in modes:
                yield _grouped_case(
                    ctx,
                    name=name,
                    megatron_module=megatron_module,
                    layer=layer,
                    token_counts=token_counts,
                    mbs=mbs,
                    mode=mode,
                )
