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
"""Gated (softmax) attention: the TE core-attention kernel and the full Megatron ``SelfAttention``."""

from __future__ import annotations

from collections.abc import Iterator

from megatron.core.transformer.enums import AttnMaskType

from mbench.cases import BenchCase, BenchContext, first, leaf, linear_flops
from mbench.timing import Workload


CATEGORY = "gated_attention"


def _core_attention_flops(*, batch: int, seq: int, heads: int, head_dim: int, mode: str) -> float:
    # QK^T and PV are 2*b*s*s*h*d each; the causal mask halves the useful work.
    # Backward recomputes QK^T and adds dV, dP, dQ, dK: 2.5x forward (FlashAttention convention).
    forward = 2.0 * batch * seq * seq * heads * head_dim
    return forward if mode == "fwd" else 3.5 * forward


def iter_cases(ctx: BenchContext, micro_batch_sizes: list[int], modes: list[str]) -> Iterator[BenchCase]:
    """Yield ``core_attention`` then ``self_attention`` cases."""

    modules = ctx.modules
    config = modules.config
    seq = ctx.sequence_length
    heads = config.num_attention_heads
    groups = config.num_query_groups
    head_dim = config.kv_channels
    common = {
        "seq_len": seq,
        "num_query_heads": heads,
        "num_kv_heads": groups,
        "head_dim": head_dim,
        "causal": True,
        "dtype": str(ctx.dtype).removeprefix("torch."),
        "attention_backend": config.attention_backend.name,
    }

    for mbs in micro_batch_sizes:
        for mode in modes:

            def build_core(mbs: int = mbs) -> Workload:
                core = modules.attention.core_attention
                query = leaf((seq, mbs, heads, head_dim), dtype=ctx.dtype)
                key = leaf((seq, mbs, groups, head_dim), dtype=ctx.dtype)
                value = leaf((seq, mbs, groups, head_dim), dtype=ctx.dtype)
                return Workload(
                    forward=lambda: (core(query, key, value, None, attn_mask_type=AttnMaskType.causal),),
                    leaves=[query, key, value],
                )

            yield BenchCase(
                name="core_attention",
                megatron_module="TEDotProductAttention",
                micro_batch_size=mbs,
                mode=mode,
                params={**common, "batch": mbs, "qkv_layout": "sbhd"},
                flops=_core_attention_flops(batch=mbs, seq=seq, heads=heads, head_dim=head_dim, mode=mode),
                build=build_core,
            )

    hidden = config.hidden_size
    qkv_out = (heads * 2 + groups * 2) * head_dim if config.attention_output_gate else (heads + 2 * groups) * head_dim
    for mbs in micro_batch_sizes:
        tokens = seq * mbs
        for mode in modes:

            def build_full(mbs: int = mbs) -> Workload:
                attention = modules.attention
                rotary_pos_emb = modules.rotary_pos_emb(seq)
                hidden_states = leaf((seq, mbs, hidden), dtype=ctx.dtype)
                return Workload(
                    forward=lambda: (first(attention(hidden_states, None, rotary_pos_emb=rotary_pos_emb)),),
                    leaves=[hidden_states],
                )

            yield BenchCase(
                name="self_attention",
                megatron_module="SelfAttention",
                micro_batch_size=mbs,
                mode=mode,
                params={
                    **common,
                    "batch": mbs,
                    "hidden_size": hidden,
                    "linear_qkv_n": qkv_out,
                    "qk_layernorm": config.qk_layernorm,
                    "attention_output_gate": config.attention_output_gate,
                    "rotary_percent": config.rotary_percent,
                    "includes": "input_norm+linear_qkv+qk_norm+rope+core_attention+output_gate+linear_proj",
                    "flops_scope": "linear_qkv+core_attention(causal)+linear_proj",
                },
                flops=_core_attention_flops(batch=mbs, seq=seq, heads=heads, head_dim=head_dim, mode=mode)
                + linear_flops(m=tokens, n=qkv_out, k=hidden, mode=mode)
                + linear_flops(m=tokens, n=hidden, k=heads * head_dim, mode=mode),
                build=build_full,
            )
