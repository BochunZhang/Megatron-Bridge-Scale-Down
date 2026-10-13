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
"""Case description shared by all benchmark categories."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from mbench.megatron_setup import MegatronModules
from mbench.timing import Workload


ParamValue = int | float | str | bool | None


@dataclass(frozen=True)
class BenchContext:
    """Everything a category needs to build its cases.

    Attributes:
        modules: Lazily built Megatron modules; ``None`` for model-free scopes (GemmSweep).
        sequence_length: Tokens per sample; ``None`` for model-free scopes.
        dtype: Activation dtype.
        formal_expert_parallel_size: EP size of the real training layout; ``None`` for dense models.
        expert_routings: Routing distributions for the routed-expert cases.
        seed: Seed for inputs and routing histograms.
        case_filter: Case names to run; ``None`` runs all. Categories must not build
            modules for filtered-out cases.
        settings: The config section named after the category (e.g. ``gemm_sweep``), if any.
    """

    modules: MegatronModules | None
    sequence_length: int | None
    dtype: torch.dtype
    formal_expert_parallel_size: int | None
    expert_routings: list[dict[str, ParamValue]]
    seed: int
    case_filter: frozenset[str] | None
    settings: dict[str, Any] = field(default_factory=dict)

    def wants(self, name: str) -> bool:
        """Whether case ``name`` is selected."""

        return self.case_filter is None or name in self.case_filter


@dataclass(frozen=True)
class BenchCase:
    """One (operator, micro-batch size, mode) measurement.

    Attributes:
        name: Case name, unique within the category.
        megatron_module: Megatron class or function that is executed.
        micro_batch_size: Micro-batch size.
        mode: ``fwd`` or ``fwdbwd``.
        params: Operator parameters written to the CSV (shapes, dtypes, options).
        flops: Model FLOPs of one iteration in ``mode``; ``None`` when not well defined.
        build: Allocates inputs and returns the workload.
    """

    name: str
    megatron_module: str
    micro_batch_size: int
    mode: str
    params: dict[str, ParamValue]
    flops: float | None
    build: Callable[[], Workload]


CaseFactory = Callable[[BenchContext, list[int], list[str]], Iterator[BenchCase]]


def leaf(shape: tuple[int, ...], *, dtype: torch.dtype, scale: float = 1.0) -> torch.Tensor:
    """Random CUDA leaf tensor that requires grad (dgrad of every input is computed, as in training)."""

    return (torch.randn(shape, device="cuda", dtype=dtype) * scale).requires_grad_(True)


def first(output: torch.Tensor | tuple[torch.Tensor | None, ...]) -> torch.Tensor:
    """Megatron linears return ``(output, bias)``; keep the output."""

    return output[0] if isinstance(output, tuple) else output


def linear_flops(*, m: int, n: int, k: int, mode: str) -> float:
    """2MNK forward; backward adds dgrad and wgrad (another 4MNK)."""

    return (2.0 if mode == "fwd" else 6.0) * m * n * k


def local_expert_token_counts(
    *, tokens: int, topk: int, num_experts: int, expert_parallel_size: int, routing: dict[str, ParamValue]
) -> list[int]:
    """Tokens received by each local expert of EP rank 0 after the all-to-all.

    Every EP rank contributes ``tokens * topk`` assignments; rank 0 owns experts
    ``[0, num_experts / EP)``.

    Args:
        tokens: Tokens per micro-batch on one rank.
        topk: Experts per token.
        num_experts: Total routed experts.
        expert_parallel_size: Formal EP size.
        routing: ``{"name": "uniform"}`` or ``{"name": "multinomial", "seed": int}``.

    Returns:
        Token count per local expert.
    """

    local_experts = num_experts // expert_parallel_size
    assignments = tokens * topk * expert_parallel_size
    if routing["name"] == "uniform":
        if assignments % num_experts:
            raise ValueError("uniform routing needs tokens * topk * EP divisible by num_experts")
        return [assignments // num_experts] * local_experts
    if routing["name"] == "multinomial":
        generator = np.random.default_rng(int(routing["seed"]))
        counts = generator.multinomial(assignments, np.full(num_experts, 1.0 / num_experts))
        return [int(count) for count in counts[:local_experts]]
    raise ValueError(f"unknown routing {routing['name']!r}")
