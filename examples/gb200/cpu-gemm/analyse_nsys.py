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
"""Summarize timed GEMM kernels and GPU metrics from Nsight Systems captures."""

from __future__ import annotations

import argparse
import csv
import logging
import math
import re
import sqlite3
import statistics
import subprocess
import tempfile
from collections import defaultdict
from contextlib import closing
from itertools import pairwise
from pathlib import Path


G_LOGGER = logging.getLogger(__name__)
CASE_DIR = re.compile(r"case([1234])-b([1-9][0-9]*)")
RUN_SETTING = re.compile(r"(?:^|\s)(gpu|frequency_hz|iters|n|k|k_panel)=([0-9]+)(?=\s|$)")
ALGORITHM_SETTING = re.compile(r"(?:^|\s)algorithm=(\S*)")
OUTPUT_STATIONARY = "output-stationary"
K_PANEL = "k-panel"
MEMCPY_TABLES = ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMCPY2")
H2D_COPY_KIND = 1  # CUDA_MEMCPY_KIND_HTOD in ENUM_CUDA_MEMCPY_OPER.
METRICS = {
    "dram_read": "DRAM Read Bandwidth",
    "dram_write": "DRAM Write Bandwidth",
    "c2c_rx": "CTC RX Throughput User Data",
    "c2c_tx": "CTC TX Throughput User Data",
}
SUMMARY_FIELDS = [
    "case",
    "b",
    "algorithm",
    "kernel_count",
    "kernel_mean_us",
    "kernel_median_us",
    "kernel_us_per_gemm",
    "kernel_span_us",
    "capture_span_us",
    "capture_us_per_gemm",
    "h2d_copy_count",
    "h2d_bytes_per_gemm",
    "h2d_us_per_gemm",
    *(f"{metric}_{field}" for metric in METRICS for field in ("mean", "unit", "samples")),
    "c2c_rx_bytes_per_gemm",
    "c2c_tx_bytes_per_gemm",
    "error",
]
KERNEL_FIELDS = ["case", "b", "index", "kernel", "start_ns", "duration_us"]


def _run_settings(output_dir: Path) -> tuple[dict[str, int], str | None]:
    config = output_dir / "config.txt"
    if not config.is_file():
        return {}, None
    for line in config.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("m="):
            algorithm = ALGORITHM_SETTING.search(line)
            return {name: int(value) for name, value in RUN_SETTING.findall(line)}, algorithm[1] if algorithm else None
    return {}, None


def _capture_database(case_dir: Path, *, nsys: str | None) -> Path:
    database = case_dir / "profile.sqlite"
    if database.is_file() and database.stat().st_size:
        return database
    if nsys is None:
        raise ValueError("profile.sqlite is missing or empty; pass --nsys to export the raw capture")
    reports = sorted(path for path in case_dir.glob("*.nsys-rep") if path.is_file())
    if not reports:
        raise ValueError("no .nsys-rep capture in case directory")
    if len(reports) > 1:
        raise ValueError(f"multiple .nsys-rep captures: {', '.join(path.name for path in reports)}")
    report = reports[0]
    if not report.stat().st_size:
        raise ValueError(f"{report.name} is empty")
    log = case_dir / "profile.log"
    if not log.is_file() or "validation=passed" not in log.read_text(encoding="utf-8", errors="replace"):
        raise ValueError("profile.log does not confirm GEMM validation")

    with tempfile.TemporaryDirectory(prefix=".nsys-export-", dir=case_dir) as directory:
        staged = Path(directory) / database.name
        result = subprocess.run(
            [nsys, "export", "--type=sqlite", f"--output={staged}", str(report)],
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
        )
        if result.returncode:
            detail = result.stderr.strip() or result.stdout.strip()
            raise ValueError(f"nsys export failed (exit {result.returncode}): {detail}")
        if not staged.is_file() or not staged.stat().st_size:
            raise ValueError("nsys export did not produce a SQLite file")
        staged.replace(database)
    return database


def _bandwidth(
    samples: dict[str, list[tuple[int, float]]], *, metric: str, start_ns: int, end_ns: int, period_ns: float
) -> tuple[float, str, int, float]:
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
    integral = math.fsum(weighted)
    return integral / covered_ns, unit, len(weighted), integral


def _analyse(
    database: Path,
    *,
    case: int,
    batch: int,
    algorithm: str,
    gpu: int,
    period_ns: float,
    iters: int | None,
    settings: dict[str, int],
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]:
    summary: dict[str, float | int | str] = {"case": case, "b": batch, "algorithm": algorithm}
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
        min_kernels = iters or 0
        panels = 0
        n, k, k_panel = (settings.get(key, 0) for key in ("n", "k", "k_panel"))
        if algorithm == K_PANEL:
            if iters is None or min(n, k, k_panel) <= 0:
                errors.append("k-panel mode requires positive n, k, k_panel and iters in the run settings")
            else:
                panels = (k + k_panel - 1) // k_panel
                min_kernels = iters * (panels + 1)
        copy_tables = [table for table in MEMCPY_TABLES if table in tables]
        if algorithm == K_PANEL and case in (2, 4):
            # UNION deduplicates activities exported into both memcpy tables.
            copy_query = " UNION ".join(
                f"SELECT start, end, deviceId, streamId, correlationId, bytes, copyKind FROM {table}"
                for table in copy_tables
            )
            copies = list(conn.execute(copy_query)) if copy_query else []
            h2d = [
                (begin, end, size)
                for begin, end, device, _, _, size, kind in copies
                if kind == H2D_COPY_KIND and device == kernels[0][3] and end >= begin and size > 0
            ]
            if len(h2d) != len(copies):
                errors.append("unexpected memcpy direction, CUDA device or invalid activity in k-panel capture")
            if h2d:
                start_ns = min(start_ns, min(begin for begin, _, _ in h2d))
                end_ns = max(end_ns, max(end for _, end, _ in h2d))
            total_bytes = sum(size for _, _, size in h2d)
            summary["h2d_copy_count"] = len(h2d)
            if iters is not None:
                summary["h2d_bytes_per_gemm"] = total_bytes / iters
                summary["h2d_us_per_gemm"] = sum(end - begin for begin, end, _ in h2d) / 1000 / iters
            if panels and iters is not None:
                expected_copies = iters * panels
                expected_bytes = iters * 2 * k * n
                if len(h2d) != expected_copies:
                    errors.append(f"expected {expected_copies} H2D copies, captured {len(h2d)}")
                if total_bytes != expected_bytes:
                    errors.append(f"expected {expected_bytes} H2D bytes, captured {total_bytes}")
                elif len(h2d) == expected_copies:
                    # Copies are ordered on one stream. Each GEMM may end with a shorter panel.
                    for index, (_, _, size) in enumerate(sorted(h2d)):
                        width = min(k_panel, k - (index % panels) * k_panel)
                        if size != 2 * width * n:
                            errors.append(f"H2D copy {index + 1} expected {2 * width * n} bytes, captured {size}")
                            break
        else:
            for table in copy_tables:
                if conn.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone():
                    errors.append(f"CUDA memcpy in timed capture ({table})")

        capture_span = (end_ns - start_ns) / 1000
        summary["capture_span_us"] = capture_span
        if iters is not None:
            summary["capture_us_per_gemm"] = capture_span / iters
            if len(kernels) < min_kernels:
                errors.append(f"expected at least {min_kernels} kernels, captured {len(kernels)}")
            else:
                summary["kernel_us_per_gemm"] = math.fsum(times_us) / iters

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
                mean, unit, count, integral = _bandwidth(
                    samples, metric=name, start_ns=start_ns, end_ns=end_ns, period_ns=period_ns
                )
            except ValueError as exc:
                errors.append(str(exc))
            else:
                summary.update({f"{key}_mean": mean, f"{key}_unit": unit, f"{key}_samples": count})
                if key.startswith("c2c_") and iters is not None:
                    # GB/s × ns equals bytes. This estimate includes only covered sample intervals.
                    summary[f"{key}_bytes_per_gemm"] = integral / iters
        summary["error"] = " | ".join(errors)
    return summary, kernel_rows


def main() -> int:
    """Write per-case summaries and kernel durations alongside the captures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--gpu", type=int, help="physical Nsight Systems metric device index")
    parser.add_argument("--frequency", type=int, help="GPU metric sample frequency in Hz")
    parser.add_argument("--iters", type=int, help="number of timed GEMM calls")
    parser.add_argument("--nsys", metavar="PATH", help="Nsight Systems executable to export missing SQLite files")
    args = parser.parse_args()
    if not args.output_dir.is_dir():
        parser.error("expected a results directory")
    settings, run_algorithm = _run_settings(args.output_dir)
    if run_algorithm is not None and run_algorithm not in (OUTPUT_STATIONARY, K_PANEL):
        parser.error(f"unsupported algorithm in config.txt: {run_algorithm!r}")
    gpu = args.gpu if args.gpu is not None else settings.get("gpu")
    frequency = args.frequency if args.frequency is not None else settings.get("frequency_hz", 50_000)
    iters = args.iters if args.iters is not None else settings.get("iters")
    if gpu is None or gpu < 0:
        parser.error("a non-negative --gpu is required when config.txt has no GPU index")
    if frequency <= 0 or (iters is not None and iters <= 0):
        parser.error("sample frequency and iteration count must be positive")
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
            algorithm = run_algorithm or (K_PANEL if case == 4 else OUTPUT_STATIONARY)
            try:
                if case == 4 and run_algorithm is not None:
                    raise ValueError("case4 is supported only for legacy runs without an algorithm setting")
                summary, kernels = _analyse(
                    _capture_database(path, nsys=args.nsys),
                    case=case,
                    batch=batch,
                    algorithm=algorithm,
                    gpu=gpu,
                    period_ns=1e9 / frequency,
                    iters=iters,
                    settings=settings,
                )
            except (OSError, sqlite3.Error, ValueError) as exc:
                summary, kernels = {"case": case, "b": batch, "algorithm": algorithm, "error": str(exc)}, []
            if summary["error"]:
                G_LOGGER.error("%s: %s", path.name, summary["error"])
                failed = True
            summary_writer.writerow(summary)
            kernels_writer.writerows(kernels)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
