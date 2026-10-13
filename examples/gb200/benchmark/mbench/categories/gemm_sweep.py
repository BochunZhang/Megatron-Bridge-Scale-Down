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
"""Model-free GEMM shape sweep: plain ``F.linear`` over configurable shape families.

``Y[M, N] = X[M, K] @ W[N, K]^T`` is exactly what ``nn.Linear`` / TE linears run in
the forward pass (cuBLAS "TN"). ``fwdbwd`` adds the dgrad (``dY @ W``) and wgrad
(``dY^T @ X``, BF16 output, no fp32 ``main_grad`` accumulation) GEMMs.

The ``gemm_sweep`` config section lists families; each family uses exactly one of::

    - {name: square, square: [1024, 2048]}                       # M = N = K
    - {name: small_m, m: [16, 32], k: [8192], n: [8192]}          # Cartesian product of m x k x n
    - {name: iso_flops, shapes: [[8192, 8192, 8192], [32768, 8192, 2048]]}   # explicit [M, K, N]
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from typing import Any

import torch.nn.functional as F

from mbench.cases import BenchCase, BenchContext, leaf, linear_flops
from mbench.timing import Workload


CATEGORY = "gemm_sweep"
_MODULE = "torch.nn.functional.linear (cuBLAS)"


def family_shapes(family: dict[str, Any]) -> list[tuple[int, int, int]]:
    """Return the ``(M, K, N)`` shapes of one ``gemm_sweep.families`` entry, in config order.

    Args:
        family: ``{"name": ..., "square": [...]}``, ``{"name": ..., "m": [...], "k": [...], "n": [...]}``
            or ``{"name": ..., "shapes": [[M, K, N], ...]}``.

    Returns:
        Shapes with duplicates removed.
    """

    name = family.get("name")
    if not name:
        raise ValueError(f"gemm_sweep family without a name: {family}")
    forms = [key for key in ("square", "shapes", "m") if key in family]
    if len(forms) != 1:
        raise ValueError(f"gemm_sweep family {name!r} needs exactly one of 'square', 'shapes' or 'm'/'k'/'n'")
    if forms[0] == "square":
        shapes = [(int(size), int(size), int(size)) for size in family["square"]]
    elif forms[0] == "shapes":
        shapes = []
        for shape in family["shapes"]:
            if len(shape) != 3:
                raise ValueError(f"gemm_sweep family {name!r}: shape {shape} is not [M, K, N]")
            shapes.append((int(shape[0]), int(shape[1]), int(shape[2])))
    else:
        missing = [key for key in ("m", "k", "n") if key not in family]
        if missing:
            raise ValueError(f"gemm_sweep family {name!r} is missing {missing}")
        shapes = [(int(m), int(k), int(n)) for m, k, n in itertools.product(family["m"], family["k"], family["n"])]
    if any(dim <= 0 for shape in shapes for dim in shape):
        raise ValueError(f"gemm_sweep family {name!r} has a non-positive dimension")
    return list(dict.fromkeys(shapes))


def _case(ctx: BenchContext, *, family: str, m: int, k: int, n: int, mode: str) -> BenchCase:
    element_bytes = ctx.dtype.itemsize
    operand_bytes = (m * k + k * n + m * n) * element_bytes

    def build() -> Workload:
        hidden = leaf((m, k), dtype=ctx.dtype)
        weight = leaf((n, k), dtype=ctx.dtype)
        return Workload(forward=lambda: (F.linear(hidden, weight),), leaves=[hidden, weight])

    return BenchCase(
        name=f"{family}_m{m}_k{k}_n{n}",
        megatron_module=_MODULE,
        micro_batch_size=1,
        mode=mode,
        params={
            "family": family,
            "m": m,
            "k": k,
            "n": n,
            "compute_dtype": str(ctx.dtype).removeprefix("torch."),
            # Forward GEMM: FLOPs per byte if every operand crossed HBM exactly once.
            "arithmetic_intensity": 2.0 * m * n * k / operand_bytes,
            "operand_mib": operand_bytes / 2**20,
        },
        flops=linear_flops(m=m, n=n, k=k, mode=mode),
        build=build,
    )


def iter_cases(ctx: BenchContext, micro_batch_sizes: list[int], modes: list[str]) -> Iterator[BenchCase]:
    """Yield one case per (family, shape, mode); the micro-batch size is not used (recorded as 1)."""

    del micro_batch_sizes
    families = ctx.settings.get("families")
    if not families:
        raise ValueError("GemmSweep needs a non-empty 'gemm_sweep.families' list in the config")
    names = [str(family.get("name")) for family in families]
    if len(set(names)) != len(names):
        raise ValueError(f"gemm_sweep family names must be unique: {names}")
    for family in families:
        for m, k, n in family_shapes(family):
            for mode in modes:
                yield _case(ctx, family=str(family["name"]), m=m, k=k, n=n, mode=mode)
