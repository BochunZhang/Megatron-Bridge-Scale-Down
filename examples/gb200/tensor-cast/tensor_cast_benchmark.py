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
"""Benchmark casting 1 GiB of FP32 weights to BF16 on the same CPU or CUDA device."""

from __future__ import annotations

import argparse
import logging
import platform
from functools import partial
from statistics import median
from time import perf_counter_ns

import torch


logger = logging.getLogger(__name__)
NUM_WEIGHTS = (1 << 30) // 4


def _report(operation: str, timer: str, samples_ms: list[float], *, traffic_gib: float) -> None:
    median_ms = median(samples_ms)
    logger.info(
        "%s [%s]  median %.3f ms  min %.3f ms  max %.3f ms  %.2f effective GiB/s",
        operation,
        timer,
        median_ms,
        min(samples_ms),
        max(samples_ms),
        traffic_gib * 1000 / median_ms,
    )


def _benchmark(weights: torch.Tensor, *, preallocate: bool, warmup: int, iters: int) -> None:
    if preallocate:
        destination = torch.empty_like(weights, dtype=torch.bfloat16)
        cast = partial(destination.copy_, weights)
        operation = "copy_ (preallocated)"
    else:
        cast = partial(weights.to, dtype=torch.bfloat16)
        operation = "to (allocating)"

    stream = torch.cuda.current_stream(weights.device) if weights.is_cuda else None
    for _ in range(warmup):
        warmup_output = cast()
        if stream is not None:
            stream.synchronize()
        del warmup_output

    start_event = end_event = None
    if stream is not None:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        # CUDA events are initialized lazily on their first record.
        start_event.record(stream)
        end_event.record(stream)
        end_event.synchronize()

    wall_ms = []
    cuda_ms = []
    result: torch.Tensor | None = None
    for _ in range(iters):
        # Release the previous output before timing another allocation.
        result = None
        start_ns = perf_counter_ns()
        if start_event is not None:
            start_event.record(stream)
        result = cast()
        if end_event is not None:
            end_event.record(stream)
            end_event.synchronize()
        wall_ms.append((perf_counter_ns() - start_ns) / 1e6)
        if start_event is not None and end_event is not None:
            cuda_ms.append(start_event.elapsed_time(end_event))

    assert result is not None
    if result.shape != weights.shape or result.dtype != torch.bfloat16 or result.device != weights.device:
        raise RuntimeError(f"Unexpected output shape, dtype or device for {operation}")
    probe = slice(None, None, max(1, weights.numel() // 32))
    expected = weights[probe].cpu().to(dtype=torch.bfloat16)
    torch.testing.assert_close(result[probe].cpu(), expected, rtol=0, atol=0)
    traffic_gib = (weights.nbytes + result.nbytes) / (1 << 30)
    _report(operation, "wall", wall_ms, traffic_gib=traffic_gib)
    if cuda_ms:
        _report(operation, "CUDA events", cuda_ms, traffic_gib=traffic_gib)


def main() -> None:
    """Run both conversion paths and report latency and effective bandwidth."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu or a CUDA device such as cuda:0 (default: cpu)")
    parser.add_argument("--threads", type=int, help="CPU intra-op threads (default: PyTorch setting)")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations per path (default: 3)")
    parser.add_argument("--iters", type=int, default=10, help="Timed iterations per path (default: 10)")
    args = parser.parse_args()

    if args.warmup < 1 or args.iters < 1:
        parser.error("--warmup and --iters must be positive")
    try:
        device = torch.device(args.device)
    except RuntimeError as exc:
        parser.error(str(exc))
    if device.type not in {"cpu", "cuda"}:
        parser.error("--device must select CPU or CUDA")
    if args.threads is not None:
        if device.type != "cpu" or args.threads < 1:
            parser.error("--threads requires a positive integer and --device cpu")
        torch.set_num_threads(args.threads)

    if device.type == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA is unavailable in this PyTorch environment")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= torch.cuda.device_count():
            parser.error(f"CUDA device {index} is outside the visible device range")
        device = torch.device("cuda", index)
        device_info = torch.cuda.get_device_name(device)
    else:
        device_info = f"{platform.platform()}, {torch.get_num_threads()} intra-op threads"

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("PyTorch %s | %s (%s)", torch.__version__, device, device_info)
    logger.info(
        "%s elements | 1024 MiB FP32 -> 512 MiB BF16 | %s warmups, %s samples per path",
        f"{NUM_WEIGHTS:,}",
        args.warmup,
        args.iters,
    )
    weights = torch.randn(NUM_WEIGHTS, dtype=torch.float32, device=device)
    _benchmark(weights, preallocate=True, warmup=args.warmup, iters=args.iters)
    _benchmark(weights, preallocate=False, warmup=args.warmup, iters=args.iters)


if __name__ == "__main__":
    main()
