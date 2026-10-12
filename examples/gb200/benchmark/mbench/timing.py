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
"""Eager-mode timing of one benchmark case with host-gated CUDA events.

One *round* is: synchronize -> arm the gate -> enqueue ``W`` warmup and ``P``
profile iterations (a CUDA event after the warmup and after every profile
iteration) -> release the gate. The GPU starts only after everything is
queued, so the event deltas contain GPU work only. Rounds repeat until
``profile_iters`` samples exist.

The CUDA launch queue is finite. If one round does not fit, the host blocks
while the GPU is gated; the watchdog then releases the gate, the round is
discarded and the number of iterations per round is halved.
"""

from __future__ import annotations

import logging
import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from mbench.gate import LaunchGate


LOGGER = logging.getLogger(__name__)
NVTX_PREFIX = "mbench"
_SYNC_ERROR_MARKER = "synchronizing CUDA operation"


@dataclass(frozen=True)
class TimingConfig:
    """Iteration counts for one case.

    Attributes:
        pre_run_iters: Ungated eager iterations before timing (autotuning, cuBLAS
            heuristics, allocator warmup, lazy initialization).
        warmup_iters: Gated warmup iterations at the start of every round.
        profile_iters: Total timed iterations collected over all rounds.
        max_iters_per_round: Initial upper bound of warmup + profile iterations
            in one gated round; halved automatically if the launch queue fills.
    """

    pre_run_iters: int
    warmup_iters: int
    profile_iters: int
    max_iters_per_round: int


@dataclass
class Workload:
    """Tensors and callable for one case.

    Attributes:
        forward: Runs the Megatron module forward and returns the differentiable
            outputs that backward starts from.
        leaves: Leaf input tensors whose ``.grad`` is reset before each iteration.
    """

    forward: Callable[[], tuple[torch.Tensor, ...]]
    leaves: list[torch.Tensor]


@dataclass
class TimingResult:
    """Per-iteration GPU times of the profile iterations and how they were taken."""

    iter_ms: list[float]
    gated: bool
    iters_per_round: int
    rounds: list[dict[str, object]] = field(default_factory=list)
    discarded_rounds: int = 0
    gate_note: str = ""

    def summary(self) -> dict[str, float | int]:
        """Return order statistics of ``iter_ms``."""

        values = sorted(self.iter_ms)
        p90_index = min(len(values) - 1, math.ceil(0.9 * len(values)) - 1)
        return {
            "median_ms": statistics.median(values),
            "mean_ms": statistics.fmean(values),
            "min_ms": values[0],
            "p90_ms": values[p90_index],
            "max_ms": values[-1],
            "std_ms": statistics.pstdev(values),
            "samples": len(values),
        }


def nvtx_label(*, category: str, case: str, mbs: int, mode: str, phase: str, round_index: int, iteration: int) -> str:
    """NVTX range text parsed by ``build_csv.py``; one range per enqueued iteration."""

    return (
        f"{NVTX_PREFIX}|cat={category}|case={case}|mbs={mbs}|mode={mode}"
        f"|phase={phase}|round={round_index}|iter={iteration}"
    )


def make_step(workload: Workload, mode: str) -> Callable[[], None]:
    """Build one training-like iteration: forward (autograd recording) and optional backward."""

    if mode not in {"fwd", "fwdbwd"}:
        raise ValueError(f"mode must be fwd or fwdbwd, got {mode}")
    for leaf in workload.leaves:
        leaf.grad = None
    outputs = workload.forward()
    if not all(output.requires_grad for output in outputs):
        raise ValueError("every returned output must require grad")
    if mode == "fwd":

        def forward_step() -> None:
            # Training forward: graph recorded and activations saved, then dropped.
            workload.forward()

        return forward_step

    grads = tuple(torch.randn_like(output) for output in outputs)
    del outputs

    def forward_backward_step() -> None:
        for leaf in workload.leaves:
            leaf.grad = None
        torch.autograd.backward(workload.forward(), grads)

    return forward_backward_step


def _enqueue(
    step: Callable[[], None], *, warmup: int, profile: int, label: Callable[[str, int], str]
) -> list[torch.cuda.Event]:
    events = [torch.cuda.Event(enable_timing=True) for _ in range(profile + 1)]
    for iteration in range(warmup):
        with torch.cuda.nvtx.range(label("warmup", iteration)):
            step()
    events[0].record()
    for iteration in range(profile):
        with torch.cuda.nvtx.range(label("profile", iteration)):
            step()
        events[iteration + 1].record()
    return events


def _elapsed(events: list[torch.cuda.Event]) -> list[float]:
    return [start.elapsed_time(end) for start, end in zip(events[:-1], events[1:], strict=True)]


def measure(
    step: Callable[[], None],
    *,
    config: TimingConfig,
    gate: LaunchGate,
    label: Callable[[str, int, int], str],
    iters_per_round: int,
) -> TimingResult:
    """Time ``step`` with gated rounds; fall back to ungated rounds if the case syncs the host.

    Args:
        step: One iteration, typically from :func:`make_step`.
        config: Iteration counts.
        gate: Launch gate used for every round.
        label: ``label(phase, round, iteration)`` returns the NVTX range text.
        iters_per_round: Starting round size (cached per case by the caller).

    Returns:
        Profile-iteration GPU times and the round layout actually used.
    """

    for _ in range(config.pre_run_iters):
        step()
    torch.cuda.synchronize()

    stream = torch.cuda.current_stream()
    result = TimingResult(iter_ms=[], gated=True, iters_per_round=max(2, iters_per_round))
    # Every attempt gets its own NVTX round index, so discarded rounds never alias accepted ones.
    attempt = -1
    while len(result.iter_ms) < config.profile_iters:
        attempt += 1
        cap = result.iters_per_round
        warmup = min(config.warmup_iters, cap - 1)
        profile = min(config.profile_iters - len(result.iter_ms), cap - warmup)

        def round_label(phase: str, iteration: int, round_index: int = attempt) -> str:
            return label(phase, round_index, iteration)

        torch.cuda.synchronize()
        if result.gated:
            previous_mode = torch.cuda.get_sync_debug_mode()
            gate.arm(stream)
            try:
                # A host sync on a gated stream would block until the watchdog fires.
                torch.cuda.set_sync_debug_mode("error")
                events = _enqueue(step, warmup=warmup, profile=profile, label=round_label)
            except RuntimeError as error:
                if _SYNC_ERROR_MARKER not in str(error):
                    raise
                result.gated = False
                result.gate_note = f"host sync inside the operation, measured ungated: {error}"
                LOGGER.warning("%s", result.gate_note)
                continue
            finally:
                torch.cuda.set_sync_debug_mode(previous_mode)
                gate.release()
                torch.cuda.synchronize()
            if gate.watchdog_fired():
                result.discarded_rounds += 1
                if cap <= 2:
                    raise RuntimeError(
                        "the host blocked while enqueuing even 2 iterations behind the gate; "
                        "increase timing.gate_timeout_s or check for host synchronization"
                    )
                result.iters_per_round = cap // 2
                LOGGER.info("launch queue full at %d iterations/round, retrying with %d", cap, cap // 2)
                continue
        else:
            events = _enqueue(step, warmup=warmup, profile=profile, label=round_label)
            torch.cuda.synchronize()
        round_ms = _elapsed(events)
        result.iter_ms.extend(round_ms)
        result.rounds.append({"round": attempt, "warmup": warmup, "profile": profile, "iter_ms": round_ms})
    return result
