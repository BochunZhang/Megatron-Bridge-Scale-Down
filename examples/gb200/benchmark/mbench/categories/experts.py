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
"""MoE ``Experts`` scope: ``TopKRouter`` and ``TEGroupedMLP`` (routed experts of one EP rank).

Token dispatch/combine communication and permutation are not part of these cases.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch

from mbench.cases import BenchCase, BenchContext, first, leaf, linear_flops, local_expert_token_counts
from mbench.timing import Workload


CATEGORY = "experts"


def iter_cases(ctx: BenchContext, micro_batch_sizes: list[int], modes: list[str]) -> Iterator[BenchCase]:
    """Yield ``router`` then ``routed_experts_<routing>`` cases."""

    modules = ctx.modules
    config = modules.config
    seq = ctx.sequence_length
    hidden = config.hidden_size
    dtype_name = str(ctx.dtype).removeprefix("torch.")
    experts = config.num_moe_experts
    topk = config.moe_router_topk
    ffn = config.moe_ffn_hidden_size
    glu = 2 if config.gated_linear_unit else 1

    for mbs in micro_batch_sizes:
        tokens = seq * mbs
        for mode in modes:

            def build_router(mbs: int = mbs) -> Workload:
                router = modules.router
                hidden_states = leaf((seq, mbs, hidden), dtype=ctx.dtype)

                def forward() -> tuple[torch.Tensor]:
                    probs, _routing_map = router(hidden_states)
                    return (probs,)

                return Workload(forward=forward, leaves=[hidden_states])

            yield BenchCase(
                name="router",
                megatron_module="TopKRouter",
                micro_batch_size=mbs,
                mode=mode,
                params={
                    "tokens": tokens,
                    "hidden_size": hidden,
                    "num_experts": experts,
                    "topk": topk,
                    "score_function": config.moe_router_score_function,
                    "load_balancing": str(config.moe_router_load_balancing_type),
                    "router_dtype": config.moe_router_dtype or dtype_name,
                    "router_fusion": config.moe_router_fusion,
                    "flops_scope": "gating GEMM only",
                },
                flops=linear_flops(m=tokens, n=experts, k=hidden, mode=mode),
                build=build_router,
            )

    for routing in ctx.expert_routings:
        routing_name = str(routing["name"])
        for mbs in micro_batch_sizes:
            token_counts = local_expert_token_counts(
                tokens=seq * mbs,
                topk=topk,
                num_experts=experts,
                expert_parallel_size=ctx.formal_expert_parallel_size,
                routing=routing,
            )
            rows = sum(token_counts)
            for mode in modes:

                def build_experts(token_counts: list[int] = token_counts, rows: int = rows) -> Workload:
                    grouped_mlp = modules.routed_experts
                    # The alltoall dispatcher hands tokens_per_expert to the experts on the host.
                    tokens_per_expert = torch.tensor(token_counts, dtype=torch.long)
                    permuted_hidden = leaf((rows, hidden), dtype=ctx.dtype)
                    permuted_probs = torch.rand(rows, device="cuda", dtype=torch.float32).requires_grad_(True)
                    return Workload(
                        forward=lambda: (first(grouped_mlp(permuted_hidden, tokens_per_expert, permuted_probs)),),
                        leaves=[permuted_hidden, permuted_probs],
                    )

                yield BenchCase(
                    name=f"routed_experts_{routing_name}",
                    megatron_module="TEGroupedMLP",
                    micro_batch_size=mbs,
                    mode=mode,
                    params={
                        "tokens_per_rank": seq * mbs,
                        "rows": rows,
                        "local_experts": len(token_counts),
                        "num_experts": experts,
                        "formal_expert_parallel_size": ctx.formal_expert_parallel_size,
                        "topk": topk,
                        "routing": routing_name,
                        "routing_seed": routing.get("seed"),
                        "rows_per_expert": ",".join(str(count) for count in token_counts),
                        "max_rows_per_expert": max(token_counts),
                        "hidden_size": hidden,
                        "ffn_hidden_size": ffn,
                        "fc1": f"[{rows},{hidden}]x[{hidden},{glu * ffn}]",
                        "fc2": f"[{rows},{ffn}]x[{ffn},{hidden}]",
                        "dtype": dtype_name,
                    },
                    flops=linear_flops(m=rows, n=glu * ffn, k=hidden, mode=mode)
                    + linear_flops(m=rows, n=hidden, k=ffn, mode=mode),
                    build=build_experts,
                )
