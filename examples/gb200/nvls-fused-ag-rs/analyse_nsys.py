# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate and compare the fused and split AG/RS CUDA traces."""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from collections import defaultdict
from contextlib import closing
from pathlib import Path
from statistics import median
from typing import NamedTuple


LOGGER = logging.getLogger(__name__)
KERNEL_NAMES = {
    "fused": "ncclDevKernel_FusedAllGatherReduceScatter",
    "ag": "ncclSymkDevKernel_AllGather_STMC",
    "rs": "ncclSymkDevKernel_ReduceScatter_LDMC_sum_bf16",
}
KERNEL_QUERY = """
SELECT k.deviceId, k.start, k.end, k.gridX, k.blockX, k.registersPerThread, k.streamId, s.value
FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
JOIN StringIds AS s ON s.id = k.shortName
WHERE instr(s.value, ?) > 0 OR instr(s.value, ?) > 0 OR instr(s.value, ?) > 0
ORDER BY k.deviceId, k.start
"""


class _Launch(NamedTuple):
    start: int
    end: int
    grid_x: int
    block_x: int
    registers_per_thread: int
    stream_id: int


def _read_launches(path: Path) -> dict[int, dict[str, list[_Launch]]]:
    launches = defaultdict(lambda: defaultdict(list))
    try:
        with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as connection:
            for device, start, end, grid, block, registers, stream, name in connection.execute(
                KERNEL_QUERY, tuple(KERNEL_NAMES.values())
            ):
                if start < 0 or end <= start or grid < 1 or block < 1:
                    raise ValueError(f"{path} GPU {device} has an invalid launch for {name}")
                op = next(op for op, kernel in KERNEL_NAMES.items() if kernel in name)
                launches[device][op].append(_Launch(start, end, grid, block, registers, stream))
    except sqlite3.Error as error:
        raise ValueError(f"Cannot read CUDA kernel records from {path}: {error}") from error
    return dict(launches)


def _case_summary(
    launches: dict[int, dict[str, list[_Launch]]], *, path: Path, ops: tuple[str, ...], iters: int
) -> tuple[dict[int, dict[str, object]], list[float]]:
    devices = {}
    pair_samples = []
    for device, by_op in launches.items():
        counts = {op: len(samples) for op, samples in by_op.items()}
        expected = dict.fromkeys(ops, iters)
        if counts != expected:
            raise ValueError(f"{path} GPU {device} has kernel counts {counts}, expected {expected}")

        pair_us = []
        overlaps = []
        previous_end = 0
        # The worker joins its streams before releasing the next pair.
        for pair in zip(*(by_op[op] for op in ops), strict=True):
            first_start = min(kernel.start for kernel in pair)
            last_end = max(kernel.end for kernel in pair)
            if first_start < previous_end:
                raise ValueError(f"{path} GPU {device} has kernels from successive iterations overlapping")
            pair_us.append((last_end - first_start) / 1_000)
            if len(pair) == 2:
                last_start = max(kernel.start for kernel in pair)
                first_end = min(kernel.end for kernel in pair)
                overlaps.append(max(0, first_end - last_start) / 1_000)
            previous_end = last_end
        pair_samples.append(pair_us)
        devices[device] = {
            "median_pair_span_us": median(pair_us),
            "median_overlap_us": median(overlaps) if overlaps else None,
            "kernels": {
                op: {
                    "median_duration_us": median((kernel.end - kernel.start) / 1_000 for kernel in samples),
                    "launch_configs": [
                        {"grid_x": grid, "block_x": block, "registers_per_thread": registers}
                        for grid, block, registers in sorted(
                            {(kernel.grid_x, kernel.block_x, kernel.registers_per_thread) for kernel in samples}
                        )
                    ],
                    "stream_ids": sorted({kernel.stream_id for kernel in samples}),
                }
                for op, samples in by_op.items()
            },
        }
    slowest_device_us = [max(iteration) for iteration in zip(*pair_samples, strict=True)]
    return devices, slowest_device_us


def main() -> None:
    """Write GPU kernel timing and launch geometry for two Nsight Systems exports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fused", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--iters", type=int, required=True)
    parser.add_argument("--ranks", type=int, help="Expected number of GPU ranks in each capture")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iters < 1 or (args.ranks is not None and args.ranks < 2):
        parser.error("--iters must be positive and --ranks must be at least two")

    try:
        fused = _read_launches(args.fused)
        split = _read_launches(args.split)
        if fused.keys() != split.keys() or len(fused) < 2 or (args.ranks is not None and len(fused) != args.ranks):
            raise ValueError(
                f"Mismatched GPU coverage (expected {args.ranks or 'at least 2'}): "
                f"fused {sorted(fused)}, split {sorted(split)}"
            )
        fused_devices, fused_slowest = _case_summary(fused, path=args.fused, ops=("fused",), iters=args.iters)
        split_devices, split_slowest = _case_summary(split, path=args.split, ops=("ag", "rs"), iters=args.iters)
    except ValueError as error:
        parser.error(str(error))

    fused_us = median(fused_slowest)
    split_us = median(split_slowest)
    speedup = split_us / fused_us
    summary = {
        "iterations": args.iters,
        "fused": {"median_slowest_device_pair_span_us": fused_us, "devices": fused_devices},
        "split": {"median_slowest_device_pair_span_us": split_us, "devices": split_devices},
        "fused_vs_split_speedup": speedup,
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    LOGGER.info("Median slowest GPU pair span: fused %.3f us, split %.3f us (%.3fx)", fused_us, split_us, speedup)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
