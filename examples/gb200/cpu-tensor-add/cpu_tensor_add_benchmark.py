#!/usr/bin/env python3
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
"""Measure CPU thread scaling for PyTorch BF16 + FP32 elementwise add."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import platform
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import torch


logger = logging.getLogger(__name__)
DEFAULT_SIZES_M = [32, 64, 128, 256, 512, 1024]
RESULTS_ROOT = Path(__file__).resolve().parents[3] / "results" / "gb200" / "cpu-tensor-add"


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def _positive_list(value: str) -> list[int]:
    try:
        numbers = [_positive_int(item) for item in value.split(",")]
    except (ValueError, argparse.ArgumentTypeError) as exc:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers") from exc
    if len(numbers) != len(set(numbers)):
        raise argparse.ArgumentTypeError("list values must be unique")
    return sorted(numbers)


def _default_threads(cpu_count: int) -> list[int]:
    return sorted({count for count in (1, 2, 4, 8, 16, 32, 48, 64, cpu_count) if count <= cpu_count})


def _measure_size(numel: int, threads: list[int], *, warmup: int, iters: int) -> Iterator[tuple[int, list[int]]]:
    # First-touch the same input buffers once for all thread counts at this size.
    torch.set_num_threads(threads[-1])
    bf16 = torch.full((numel,), 1.5, dtype=torch.bfloat16, device="cpu")
    fp32 = torch.full((numel,), 2.25, dtype=torch.float32, device="cpu")
    output = torch.empty_like(fp32)

    for count in threads:
        torch.set_num_threads(count)
        if torch.get_num_threads() != count:
            raise RuntimeError(f"PyTorch did not set the requested {count} intra-op threads")

        for index in (0, numel // 2, numel - 1):
            output[index] = float("nan")
        for _ in range(warmup):
            torch.add(bf16, fp32, out=output)

        durations_ns = []
        for _ in range(iters):
            start = perf_counter_ns()
            torch.add(bf16, fp32, out=output)
            durations_ns.append(perf_counter_ns() - start)

        for index in (0, numel // 2, numel - 1):
            if output[index].item() != 3.75:
                raise RuntimeError(f"Incorrect FP32 result at index {index} for {numel} elements, {count} threads")
        yield count, durations_ns


def main() -> None:
    """Run the size and thread matrix and save timings and run configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes-m", type=_positive_list, default=DEFAULT_SIZES_M, help="Sizes in units of 2^20 elements"
    )
    parser.add_argument("--threads", type=_positive_list, help="Comma-separated intra-op thread counts")
    parser.add_argument("--warmup", type=_positive_int, default=3, help="Warmup rounds per case (default: 3)")
    parser.add_argument("--iters", type=_positive_int, default=10, help="Timed rounds per case (default: 10)")
    parser.add_argument("--output-dir", type=Path, help="New directory for CSV and JSON results")
    args = parser.parse_args()

    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    cpu_count = len(affinity) if affinity is not None else (os.cpu_count() or 1)
    threads = args.threads or _default_threads(cpu_count)
    if threads[-1] > cpu_count:
        parser.error(f"--threads exceeds the {cpu_count} available logical CPUs")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.set_num_interop_threads(1)
    torch.set_num_threads(threads[-1])
    output_dir = args.output_dir or RESULTS_ROOT / datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
    output_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "operation": "torch.add(bf16, fp32, out=fp32_output)",
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "torch_parallel_info": torch.__config__.parallel_info(),
        "cpu_affinity": affinity,
        "available_logical_cpus": cpu_count,
        "sizes_m": args.sizes_m,
        "threads": threads,
        "warmup": args.warmup,
        "iters": args.iters,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    logger.info("CPU intra-op threads: %s. Results: %s", threads, output_dir)
    with (
        (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as summary_file,
        (output_dir / "samples.csv").open("w", newline="", encoding="utf-8") as samples_file,
    ):
        summary = csv.writer(summary_file)
        samples = csv.writer(samples_file)
        summary.writerow(
            ["size_m", "numel", "threads", "median_ms", "effective_gb_s", "speedup_vs_1t", "parallel_efficiency_pct"]
        )
        samples.writerow(["size_m", "numel", "threads", "iteration", "elapsed_ns"])

        for size_m in args.sizes_m:
            numel = size_m << 20
            baseline_ns = None
            for count, durations_ns in _measure_size(numel, threads, warmup=args.warmup, iters=args.iters):
                median_ns = median(durations_ns)
                if count == 1:
                    baseline_ns = median_ns
                speedup = baseline_ns / median_ns if baseline_ns is not None else None
                efficiency_pct = 100 * speedup / count if speedup is not None else None
                median_ms = median_ns / 1e6
                effective_gb_s = 10 * numel / median_ns
                samples.writerows(
                    (size_m, numel, count, iteration, elapsed_ns)
                    for iteration, elapsed_ns in enumerate(durations_ns, start=1)
                )
                summary.writerow((size_m, numel, count, median_ms, effective_gb_s, speedup, efficiency_pct))
                samples_file.flush()
                summary_file.flush()
                logger.info("%4dM  %3dt  %9.3f ms  %8.2f effective GB/s", size_m, count, median_ms, effective_gb_s)


if __name__ == "__main__":
    main()
