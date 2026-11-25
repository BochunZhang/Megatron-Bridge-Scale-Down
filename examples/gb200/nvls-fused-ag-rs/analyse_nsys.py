# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate and compare the fused and split AG/RS CUDA traces."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sqlite3
from bisect import bisect_left, bisect_right
from collections import defaultdict
from contextlib import closing
from itertools import pairwise
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
SELECT {device}, k.start, k.end, k.gridX, k.blockX, k.registersPerThread, k.streamId, s.value
FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
JOIN StringIds AS s ON s.id = k.shortName
{devices}
WHERE instr(s.value, ?) > 0 OR instr(s.value, ?) > 0 OR instr(s.value, ?) > 0
ORDER BY 1, k.start
"""
CUDA_DEVICE_JOINS = """
LEFT JOIN (
    SELECT pid, cudaId, MIN(gpuId) AS gpuId, COUNT(DISTINCT gpuId) AS matches
    FROM TARGET_INFO_CUDA_DEVICE WHERE gpuId IS NOT NULL GROUP BY pid, cudaId
) AS ordinal ON ordinal.pid = ((k.globalPid >> 24) & 16777215) AND ordinal.cudaId = k.deviceId
LEFT JOIN (
    SELECT DISTINCT pid, gpuId FROM TARGET_INFO_CUDA_DEVICE WHERE gpuId IS NOT NULL
) AS physical ON physical.pid = ((k.globalPid >> 24) & 16777215) AND physical.gpuId = k.deviceId
"""
PHYSICAL_DEVICE = (
    "CASE WHEN ordinal.matches = 1 THEN ordinal.gpuId WHEN ordinal.matches IS NULL THEN physical.gpuId END"
)
GB200_NVLINK_NOMINAL_GB_PER_S = 900.0
NVLINK_BYTES = {
    f"NVLink {direction.upper()} {packet.title()}s {kind.title()} Data [Bytes]": (direction, packet, kind)
    for direction in ("rx", "tx")
    for packet in ("request", "response")
    for kind in ("user", "protocol")
}


class _Launch(NamedTuple):
    start: int
    end: int
    grid_x: int
    block_x: int
    registers_per_thread: int
    stream_id: int


def _read_launches(path: Path, *, physical_ids: bool) -> dict[int, dict[str, list[_Launch]]]:
    launches = defaultdict(lambda: defaultdict(list))
    # CUDA_VISIBLE_DEVICES can remap kernel.deviceId. Use the process-local map for NVLink metrics.
    query = KERNEL_QUERY.format(
        device=PHYSICAL_DEVICE if physical_ids else "k.deviceId", devices=CUDA_DEVICE_JOINS if physical_ids else ""
    )
    try:
        with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as connection:
            for device, start, end, grid, block, registers, stream, name in connection.execute(
                query, tuple(KERNEL_NAMES.values())
            ):
                if device is None:
                    raise ValueError(f"{path} cannot resolve the physical GPU for {name} at {start} ns")
                if start < 0 or end <= start or grid < 1 or block < 1:
                    raise ValueError(f"{path} GPU {device} has an invalid launch for {name}")
                op = next(op for op, kernel in KERNEL_NAMES.items() if kernel in name)
                launches[device][op].append(_Launch(start, end, grid, block, registers, stream))
    except sqlite3.Error as error:
        raise ValueError(f"Cannot read CUDA kernel records from {path}: {error}") from error
    return dict(launches)


def _case_summary(
    launches: dict[int, dict[str, list[_Launch]]], *, path: Path, ops: tuple[str, ...], iters: int
) -> tuple[dict[int, dict[str, object]], list[float], dict[int, list[tuple[int, int]]]]:
    devices = {}
    pair_samples = []
    spans = {}
    for device, by_op in launches.items():
        counts = {op: len(samples) for op, samples in by_op.items()}
        expected = dict.fromkeys(ops, iters)
        if counts != expected:
            raise ValueError(f"{path} GPU {device} has kernel counts {counts}, expected {expected}")

        pair_us = []
        overlaps = []
        spans[device] = []
        previous_end = 0
        # The worker joins its streams before releasing the next pair.
        for pair in zip(*(by_op[op] for op in ops), strict=True):
            first_start = min(kernel.start for kernel in pair)
            last_end = max(kernel.end for kernel in pair)
            if first_start < previous_end:
                raise ValueError(f"{path} GPU {device} has kernels from successive iterations overlapping")
            spans[device].append((first_start, last_end))
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
    return devices, slowest_device_us, spans


def _window_bytes(
    samples: list[tuple[int, float | None]], spans: list[tuple[int, int]], *, max_gap_ns: float
) -> tuple[float, int, int]:
    """Estimate bytes within the spans, using the interval ending at each sample timestamp."""
    timestamps = [timestamp for timestamp, _ in samples]
    total_bytes = []
    interval_counts = []
    longest_interval_ns = 0
    for start, end in spans:
        left = bisect_right(timestamps, start) - 1
        right = bisect_left(timestamps, end)
        if left < 0 or right == len(samples):
            raise ValueError(f"byte samples do not bracket the kernel span {start}–{end} ns")
        interval_counts.append(right - left)
        for (previous, _), (timestamp, value) in pairwise(samples[left : right + 1]):
            interval_ns = timestamp - previous
            if not 0 < interval_ns <= max_gap_ns:
                raise ValueError(f"invalid or missing sample interval of {interval_ns} ns at {timestamp} ns")
            if value is None or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid byte sample {value!r} at {timestamp} ns")
            overlap_ns = min(timestamp, end) - max(previous, start)
            total_bytes.append(value * overlap_ns / interval_ns)
            longest_interval_ns = max(longest_interval_ns, interval_ns)
    return math.fsum(total_bytes), min(interval_counts), longest_interval_ns


def _nvlink_summary(
    path: Path, spans: dict[int, list[tuple[int, int]]], *, frequency: int
) -> dict[int, dict[str, object]]:
    period_ns = 1_000_000_000 / frequency
    padding = math.ceil(2 * period_ns)
    start = min(windows[0][0] for windows in spans.values()) - padding
    end = max(windows[-1][1] for windows in spans.values()) + padding
    placeholders = ", ".join("?" for _ in NVLINK_BYTES)
    try:
        with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as connection:
            # Nsight stores the physical TARGET_INFO_CUDA_DEVICE.gpuId in the low byte of typeId.
            metrics = defaultdict(dict)
            for type_id, metric_id, name in connection.execute(
                f"SELECT DISTINCT typeId, metricId, metricName FROM TARGET_INFO_GPU_METRICS "
                f"WHERE metricName IN ({placeholders})",
                tuple(NVLINK_BYTES),
            ):
                gpu = type_id & 0xFF
                if name in metrics[gpu]:
                    raise ValueError(f"{path} GPU {gpu} has multiple byte series for {name}")
                metrics[gpu][name] = (type_id, metric_id)
            if metrics.keys() != spans.keys():
                raise ValueError(f"{path} NVLink byte metrics cover GPUs {sorted(metrics)}, expected {sorted(spans)}")
            samples = defaultdict(list)
            for type_id, metric_id, timestamp, value in connection.execute(
                f"SELECT g.typeId, g.metricId, g.timestamp, g.value FROM GPU_METRICS AS g "
                f"JOIN (SELECT DISTINCT typeId, metricId FROM TARGET_INFO_GPU_METRICS "
                f"WHERE metricName IN ({placeholders})) AS n USING (typeId, metricId) "
                f"WHERE g.timestamp BETWEEN ? AND ? ORDER BY g.timestamp",
                (*NVLINK_BYTES, start, end),
            ):
                samples[type_id, metric_id].append((timestamp, value))
    except sqlite3.Error as error:
        raise ValueError(f"Cannot read NVLink metrics from {path}: {error}") from error

    summary = {}
    for gpu, windows in spans.items():
        if missing := NVLINK_BYTES.keys() - metrics[gpu].keys():
            raise ValueError(f"{path} GPU {gpu} is missing NVLink byte metrics {sorted(missing)}")
        window_ns = sum(last - first for first, last in windows)
        categories = {"rx": {}, "tx": {}}
        counts = []
        interval_ns = []
        for name, (direction, packet, kind) in NVLINK_BYTES.items():
            try:
                bytes_in_window, count, longest = _window_bytes(
                    samples[metrics[gpu][name]], windows, max_gap_ns=1.5 * period_ns
                )
            except ValueError as error:
                raise ValueError(f"{path} GPU {gpu}, {name}: {error}") from error
            # Decimal GB/s equals bytes/ns. Prorating boundary samples estimates their share.
            categories[direction][f"{packet}_{kind}_gb_per_s"] = bytes_in_window / window_ns
            counts.append(count)
            interval_ns.append(longest)
        for traffic in categories.values():
            traffic["user_gb_per_s"] = traffic["request_user_gb_per_s"] + traffic["response_user_gb_per_s"]
            traffic["protocol_gb_per_s"] = traffic["request_protocol_gb_per_s"] + traffic["response_protocol_gb_per_s"]
            traffic["total_gb_per_s"] = traffic["user_gb_per_s"] + traffic["protocol_gb_per_s"]
            traffic["percent_of_nominal"] = traffic["total_gb_per_s"] / GB200_NVLINK_NOMINAL_GB_PER_S * 100
        summary[gpu] = {
            "window_us": window_ns / 1_000,
            "min_sample_intervals_per_metric_per_pair": min(counts),
            "max_sample_interval_ns": max(interval_ns),
            **categories,
        }
    return summary


def main() -> None:
    """Write GPU kernel timing and launch geometry for two Nsight Systems exports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fused", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--iters", type=int, required=True)
    parser.add_argument("--ranks", type=int, help="Expected number of GPU ranks in each capture")
    parser.add_argument("--gpus", help="Physical GPU IDs in the NVLink metric capture")
    parser.add_argument("--frequency", type=int, help="NVLink metric sampling frequency in Hz")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iters < 1 or (args.ranks is not None and args.ranks < 2):
        parser.error("--iters must be positive and --ranks must be at least two")
    if (args.gpus is None) != (args.frequency is None):
        parser.error("--gpus and --frequency must be supplied together")
    gpus = None
    if args.gpus is not None:
        try:
            gpu_ids = [int(gpu) for gpu in args.gpus.split(",")]
        except ValueError:
            parser.error("--gpus must be comma-separated physical GPU IDs")
        gpus = set(gpu_ids)
        if len(gpus) < 2 or len(gpus) != len(gpu_ids) or any(not 0 <= gpu <= 255 for gpu in gpus):
            parser.error("--gpus must contain at least two distinct physical GPU IDs between 0 and 255")
        if not 10 <= args.frequency <= 200_000:
            parser.error("--frequency must be between 10 and 200000 Hz")

    try:
        fused = _read_launches(args.fused, physical_ids=gpus is not None)
        split = _read_launches(args.split, physical_ids=gpus is not None)
        if fused.keys() != split.keys() or len(fused) < 2 or (args.ranks is not None and len(fused) != args.ranks):
            raise ValueError(
                f"Mismatched GPU coverage (expected {args.ranks or 'at least 2'}): "
                f"fused {sorted(fused)}, split {sorted(split)}"
            )
        if gpus is not None and fused.keys() != gpus:
            raise ValueError(f"Expected physical GPUs {sorted(gpus)}, found kernels on {sorted(fused)}")
        fused_devices, fused_slowest, fused_spans = _case_summary(
            fused, path=args.fused, ops=("fused",), iters=args.iters
        )
        split_devices, split_slowest, split_spans = _case_summary(
            split, path=args.split, ops=("ag", "rs"), iters=args.iters
        )
        if gpus is not None:
            for path, spans, devices in (
                (args.fused, fused_spans, fused_devices),
                (args.split, split_spans, split_devices),
            ):
                for gpu, traffic in _nvlink_summary(path, spans, frequency=args.frequency).items():
                    devices[gpu]["nvlink"] = traffic
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
    if gpus is not None:
        summary["nvlink"] = {
            "nominal_gb_per_s_per_direction": GB200_NVLINK_NOMINAL_GB_PER_S,
            "sampling_frequency_hz": args.frequency,
            "window": "sum of per-iteration pair spans from earliest kernel start to latest end",
        }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    LOGGER.info("Median slowest GPU pair span: fused %.3f us, split %.3f us (%.3fx)", fused_us, split_us, speedup)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
