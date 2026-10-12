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
"""MoE ``MLP`` scope: the shared expert (``SharedExpertMLP``, including its sigmoid gate).

Qwen3.8 has no dense-MLP layer; the shared expert is the dense MLP every token passes through.
"""

from __future__ import annotations

from collections.abc import Iterator

from mbench.cases import BenchCase, BenchContext, leaf, linear_flops
from mbench.timing import Workload


CATEGORY = "mlp"


def iter_cases(ctx: BenchContext, micro_batch_sizes: list[int], modes: list[str]) -> Iterator[BenchCase]:
    """Yield ``shared_experts`` cases."""

    modules = ctx.modules
    config = modules.config
    seq = ctx.sequence_length
    hidden = config.hidden_size
    dtype_name = str(ctx.dtype).removeprefix("torch.")
    glu = 2 if config.gated_linear_unit else 1

    shared_ffn = config.moe_shared_expert_intermediate_size
    for mbs in micro_batch_sizes:
        tokens = seq * mbs
        for mode in modes:

            def build_shared(mbs: int = mbs) -> Workload:
                shared = modules.shared_experts
                hidden_states = leaf((seq, mbs, hidden), dtype=ctx.dtype)
                return Workload(forward=lambda: (shared(hidden_states),), leaves=[hidden_states])

            yield BenchCase(
                name="shared_experts",
                megatron_module="SharedExpertMLP",
                micro_batch_size=mbs,
                mode=mode,
                params={
                    "tokens": tokens,
                    "hidden_size": hidden,
                    "ffn_hidden_size": shared_ffn,
                    "shared_expert_gate": config.moe_shared_expert_gate,
                    "fc1": f"[{tokens},{hidden}]x[{hidden},{glu * shared_ffn}]",
                    "fc2": f"[{tokens},{shared_ffn}]x[{shared_ffn},{hidden}]",
                    "dtype": dtype_name,
                },
                flops=linear_flops(m=tokens, n=glu * shared_ffn, k=hidden, mode=mode)
                + linear_flops(m=tokens, n=hidden, k=shared_ffn, mode=mode)
                + linear_flops(m=tokens, n=1, k=hidden, mode=mode),
                build=build_shared,
            )
