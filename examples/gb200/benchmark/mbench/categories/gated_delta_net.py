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
"""GatedDeltaNet: the FLA chunked kernel as Megatron calls it, and the full Megatron ``GatedDeltaNet``."""

from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn.functional as F

from mbench.cases import BenchCase, BenchContext, first, leaf, linear_flops
from mbench.timing import Workload


CATEGORY = "gated_delta_net"


def iter_cases(ctx: BenchContext, micro_batch_sizes: list[int], modes: list[str]) -> Iterator[BenchCase]:
    """Yield ``gated_delta_rule`` then ``gated_delta_net`` cases."""

    modules = ctx.modules
    config = modules.config
    seq = ctx.sequence_length
    key_heads = config.linear_num_key_heads
    value_heads = config.linear_num_value_heads
    key_dim = config.linear_key_head_dim
    value_dim = config.linear_value_head_dim
    common = {
        "seq_len": seq,
        "num_key_heads": key_heads,
        "num_value_heads": value_heads,
        "key_head_dim": key_dim,
        "value_head_dim": value_dim,
        "dtype": str(ctx.dtype).removeprefix("torch."),
    }

    for mbs in micro_batch_sizes:
        for mode in modes:

            def build_kernel(mbs: int = mbs) -> Workload:
                gdn = modules.gated_delta_net
                # GatedDeltaNet L2-normalizes q/k, repeat_interleaves them to the value heads,
                # computes g in fp32 and beta in the activation dtype, then calls the kernel.
                query = F.normalize(torch.randn(mbs, seq, value_heads, key_dim, device="cuda"), dim=-1)
                key = F.normalize(torch.randn(mbs, seq, value_heads, key_dim, device="cuda"), dim=-1)
                query = query.to(ctx.dtype).requires_grad_(True)
                key = key.to(ctx.dtype).requires_grad_(True)
                value = leaf((mbs, seq, value_heads, value_dim), dtype=ctx.dtype)
                g = F.logsigmoid(torch.randn(mbs, seq, value_heads, device="cuda")).requires_grad_(True)
                beta = torch.rand(mbs, seq, value_heads, device="cuda").to(ctx.dtype).requires_grad_(True)

                def forward() -> tuple[torch.Tensor]:
                    output, _ = gdn.gated_delta_rule(
                        query,
                        key,
                        value,
                        g=g,
                        beta=beta,
                        initial_state=None,
                        output_final_state=False,
                        use_qk_l2norm_in_kernel=False,
                        cu_seqlens=None,
                    )
                    return (output,)

                return Workload(forward=forward, leaves=[query, key, value, g, beta])

            yield BenchCase(
                name="gated_delta_rule",
                megatron_module="GatedDeltaNet.gated_delta_rule (fla chunk_gated_delta_rule)",
                micro_batch_size=mbs,
                mode=mode,
                params={
                    **common,
                    "batch": mbs,
                    "kernel_q_heads": value_heads,
                    "layout": "bshd",
                    "initial_state": False,
                    "g_dtype": "float32",
                },
                flops=None,
                build=build_kernel,
            )

    hidden = config.hidden_size
    gdn_in = 2 * key_heads * key_dim + 2 * value_heads * value_dim + 2 * value_heads
    gdn_inner = value_heads * value_dim
    for mbs in micro_batch_sizes:
        tokens = seq * mbs
        for mode in modes:

            def build_full(mbs: int = mbs) -> Workload:
                gdn = modules.gated_delta_net
                hidden_states = leaf((seq, mbs, hidden), dtype=ctx.dtype)
                return Workload(forward=lambda: (first(gdn(hidden_states, None)),), leaves=[hidden_states])

            yield BenchCase(
                name="gated_delta_net",
                megatron_module="GatedDeltaNet",
                micro_batch_size=mbs,
                mode=mode,
                params={
                    **common,
                    "batch": mbs,
                    "hidden_size": hidden,
                    "in_proj_n": gdn_in,
                    "conv_kernel_dim": config.linear_conv_kernel_dim,
                    "includes": "input_norm+in_proj+conv1d+l2norm+gdr_kernel+gated_rmsnorm+out_proj",
                    "flops_scope": "in_proj+out_proj GEMMs only",
                },
                flops=linear_flops(m=tokens, n=gdn_in, k=hidden, mode=mode)
                + linear_flops(m=tokens, n=hidden, k=gdn_inner, mode=mode),
                build=build_full,
            )
