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
"""Summarize timed GEMM kernels and GPU metrics from Nsight Systems SQLite exports."""

from __future__ import annotations

import argparse
import csv
import logging
import math
import re
import sqlite3
import statistics
from collections import defaultdict
from contextlib import closing
from itertools import pairwise
from pathlib import Path


G_LOGGER = logging.getLogger(__name__)
CASE_DIR = re.compile(r"case([123])-b([1-9][0-9]*)")
METRICS = {
    "dram_read": "DRAM Read Bandwidth",
    "dram_write": "DRAM Write Bandwidth",
    "c2c_rx": "CTC RX Throughput User Data",
    "c2c_tx": "CTC TX Throughput User Data",
}
SUMMARY_FIELDS = [
    "case",
    "b",
    "kernel_count",
    "kernel_mean_us",
    "kernel_median_us",
    "kernel_us_per_gemm",
    "kernel_span_us",
    *(f"{metric}_{field}" for metric in METRICS for field in ("mean", "unit", "samples")),
    "error",
]
KERNEL_FIELDS = ["case", "b", "index", "kernel", "start_ns", "duration_us"]


def _bandwidth(
    samples: dict[str, list[tuple[int, float]]], *, metric: str, start_ns: int, end_ns: int, period_ns: float
) -> tuple[float, str, int]:
    choices: tuple[tuple[str, str, int | None], ...]
    if metric.startswith("DRAM"):
        choices = (("[Sectors]", "GB/s", 32), ("[Throughput %]", "% active cycles", None))
    else:
        choices = (("[Bytes]", "GB/s", 1),)
    for suffix, unit, bytes_per_count in choices:
        readings = samples.get(f"{metric} {suffix}") or samples.get(f"{metric} {suffix[1:-1]}")
        if readings:
            break
    else:
        raise ValueError(f"no samples for {metric}")

    weighted = []
    covered_ns = 0
    # A sample describes the interval ending at its timestamp. Attribute boundary
    # intervals proportionally and tolerate one sampling period on either edge.
    for (previous, _), (timestamp, value) in pairwise(readings):
        if timestamp <= previous:
            raise ValueError(f"duplicate {metric} samples at {timestamp} ns")
        overlap_ns = min(timestamp, end_ns) - max(previous, start_ns)
        if overlap_ns <= 0:
            continue
        interval_ns = timestamp - previous
        if interval_ns > 1.5 * period_ns or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid or missing {metric} sample at {timestamp} ns")
        if bytes_per_count is not None:
            value = value * bytes_per_count / interval_ns  # B/ns equals decimal GB/s.
        weighted.append(value * overlap_ns)
        covered_ns += overlap_ns
    if not weighted or readings[0][0] > start_ns + period_ns or readings[-1][0] < end_ns - period_ns:
        raise ValueError(f"{metric} samples do not cover the GEMM window")
    return math.fsum(weighted) / covered_ns, unit, len(weighted)


def _analyse(
    database: Path, *, case: int, batch: int, gpu: int, period_ns: float, iters: int | None
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]:
    summary: dict[str, float | int | str] = {"case": case, "b": batch}
    with closing(sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)) as conn:
        tables = {name for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"CUPTI_ACTIVITY_KIND_KERNEL", "StringIds"} <= tables:
            raise ValueError("no kernel activity in capture")
        kernels = conn.execute(
            """SELECT k.start, k.end, s.value, k.deviceId
               FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.shortName
               WHERE k.end > k.start ORDER BY k.start"""
        ).fetchall()
        if not kernels or len({row[3] for row in kernels}) != 1:
            raise ValueError("expected kernel activity from one CUDA device")
        start_ns, end_ns = kernels[0][0], max(row[1] for row in kernels)
        kernel_rows: list[dict[str, float | int | str]] = [
            dict(case=case, b=batch, index=index, kernel=name, start_ns=start, duration_us=(end - start) / 1000)
            for index, (start, end, name, _) in enumerate(kernels, start=1)
        ]
        times_us = [(end - start) / 1000 for start, end, _, _ in kernels]
        summary.update(
            kernel_count=len(kernels),
            kernel_mean_us=statistics.fmean(times_us),
            kernel_median_us=statistics.median(times_us),
            kernel_span_us=(end_ns - start_ns) / 1000,
        )
        errors = []
        if iters is not None:
            if len(kernels) < iters:
                errors.append(f"expected at least {iters} kernels, captured {len(kernels)}")
            else:
                summary["kernel_us_per_gemm"] = math.fsum(times_us) / iters
        for table in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMCPY2"):
            if table in tables and conn.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone():
                errors.append(f"CUDA memcpy in timed capture ({table})")

        samples: dict[str, list[tuple[int, float]]] = defaultdict(list)
        if {"GPU_METRICS", "TARGET_INFO_GPU_METRICS"} <= tables:
            padding = math.ceil(2 * period_ns)
            for name, timestamp, value in conn.execute(
                """SELECT DISTINCT n.metricName, g.timestamp, g.value
                   FROM GPU_METRICS g JOIN TARGET_INFO_GPU_METRICS n USING (typeId, metricId)
                   WHERE (g.typeId & 255) = ? AND g.timestamp BETWEEN ? AND ? AND g.value IS NOT NULL
                   ORDER BY g.timestamp""",
                (gpu, start_ns - padding, end_ns + padding),
            ):
                samples[name].append((timestamp, value))
        for key, name in METRICS.items():
            try:
                mean, unit, count = _bandwidth(
                    samples, metric=name, start_ns=start_ns, end_ns=end_ns, period_ns=period_ns
                )
            except ValueError as exc:
                errors.append(str(exc))
            else:
                summary.update({f"{key}_mean": mean, f"{key}_unit": unit, f"{key}_samples": count})
        summary["error"] = " | ".join(errors)
    return summary, kernel_rows


def main() -> int:
    """Write per-case summaries and kernel durations alongside the captures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--gpu", type=int, required=True, help="physical Nsight Systems metric device index")
    parser.add_argument("--frequency", type=int, default=50_000, help="GPU metric sample frequency in Hz")
    parser.add_argument("--iters", type=int, help="number of timed GEMM calls")
    args = parser.parse_args()
    if args.gpu < 0 or args.frequency <= 0 or not args.output_dir.is_dir():
        parser.error("expected a results directory, non-negative GPU index and positive sampling frequency")
    if args.iters is not None and args.iters <= 0:
        parser.error("--iters must be positive")
    cases = sorted(
        (int(match[1]), int(match[2]), path)
        for path in args.output_dir.iterdir()
        if path.is_dir() and (match := CASE_DIR.fullmatch(path.name))
    )
    if not cases:
        parser.error("no caseN-bN capture directories found")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    failed = False
    with (
        (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as summary_file,
        (args.output_dir / "kernels.csv").open("w", newline="", encoding="utf-8") as kernels_file,
    ):
        summary_writer = csv.DictWriter(summary_file, SUMMARY_FIELDS)
        kernels_writer = csv.DictWriter(kernels_file, KERNEL_FIELDS)
        summary_writer.writeheader()
        kernels_writer.writeheader()
        for case, batch, path in cases:
            try:
                summary, kernels = _analyse(
                    path / "profile.sqlite",
                    case=case,
                    batch=batch,
                    gpu=args.gpu,
                    period_ns=1e9 / args.frequency,
                    iters=args.iters,
                )
            except (sqlite3.Error, ValueError) as exc:
                summary, kernels = {"case": case, "b": batch, "error": str(exc)}, []
            if summary["error"]:
                G_LOGGER.error("%s: %s", path.name, summary["error"])
                failed = True
            summary_writer.writerow(summary)
            kernels_writer.writerows(kernels)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
