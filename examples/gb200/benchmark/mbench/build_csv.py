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
"""Collect benchmark JSON records into one CSV per category, optionally cross-checked with Nsys.

Usage::

    python -m mbench.build_csv --input-dir results/... [--nsys-dir results/.../nsys]

For every category directory ``<input-dir>/<category>/*.json`` this writes
``<input-dir>/csv/<category>.csv``. When ``<nsys-dir>/<category>.sqlite`` exists
(``nsys export --type sqlite``), the ``nsys_*`` columns are filled from SQL:

* Each NVTX range ``mbench|...|iter=i`` covers the host-side enqueue of one
  iteration. The CUDA runtime/driver calls made inside that time window (on any
  thread of the process, which includes the autograd backward thread) are
  joined by ``correlationId`` to the kernels, memsets and memcpys they launched.
  This attributes GPU work correctly even though, with the launch gate, GPU
  execution happens long after the NVTX range ended.
* ``nsys_iter_ms`` = end of the last GPU activity of iteration ``i`` minus that
  of iteration ``i - 1``; this is the quantity the CUDA events measure.
* ``nsys_span_ms`` / ``nsys_kernel_union_ms`` / ``nsys_kernel_sum_ms`` are the
  first-start-to-last-end span, the interval union (GPU busy time) and the plain
  sum of the activity durations of one iteration.
Only rounds accepted by the timer (listed in the JSON) are used.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import logging
import math
import re
import sqlite3
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)
_RANGE = re.compile(
    r"mbench\|cat=(?P<cat>[^|]+)\|case=(?P<case>[^|]+)\|mbs=(?P<mbs>\d+)\|mode=(?P<mode>[^|]+)"
    r"\|phase=(?P<phase>warmup|profile)\|round=(?P<round>\d+)\|iter=(?P<iter>\d+)"
)
_GPU_TABLES = ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMSET", "CUPTI_ACTIVITY_KIND_MEMCPY")
_BASE_COLUMNS = [
    "category",
    "case",
    "megatron_module",
    "micro_batch_size",
    "mode",
]
_RESULT_COLUMNS = [
    "flops",
    "event_median_ms",
    "event_mean_ms",
    "event_min_ms",
    "event_p90_ms",
    "event_max_ms",
    "event_std_ms",
    "samples",
    "tflops_median",
    "gated",
    "iters_per_round",
    "discarded_rounds",
    "peak_memory_gib",
    "sm_clock_mhz",
    "power_w",
]
_NSYS_COLUMNS = [
    "nsys_samples",
    "nsys_iter_median_ms",
    "nsys_iter_p90_ms",
    "nsys_span_median_ms",
    "nsys_kernel_union_median_ms",
    "nsys_kernel_sum_median_ms",
    "nsys_gpu_busy_ratio",
    "nsys_activities_per_iter",
    "nsys_vs_event_ratio",
    "nsys_top_kernels",
]


@dataclass
class IterationActivity:
    """GPU activities launched while one NVTX iteration range was open."""

    phase: str
    iteration: int
    intervals: list[tuple[int, int]] = field(default_factory=list)
    kernel_names: list[tuple[str, int]] = field(default_factory=list)


def _tables(connection: sqlite3.Connection) -> set[str]:
    return {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')}


def _union_ns(intervals: list[tuple[int, int]]) -> int:
    total = 0
    current_start, current_end = None, None
    for start, end in sorted(intervals):
        if current_end is None or start > current_end:
            if current_end is not None:
                total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_end is not None:
        total += current_end - current_start
    return total


def _process_id(global_tid: int) -> int:
    # Nsys packs (hardware id, pid, tid) into globalTid / globalPid.
    return (int(global_tid) >> 24) & 0xFFFFFF


def load_nsys(sqlite_path: Path) -> dict[tuple[str, str, int, str, int], list[IterationActivity]]:
    """Map (category, case, mbs, mode, round) to its iterations' GPU activities, in enqueue order."""

    connection = sqlite3.connect(sqlite_path)
    tables = _tables(connection)
    strings: dict[int, str] = {}
    if "StringIds" in tables:
        strings = {int(row[0]): str(row[1]) for row in connection.execute("SELECT id, value FROM StringIds")}

    nvtx_columns = _columns(connection, "NVTX_EVENTS")
    text_expr = "text" if "text" in nvtx_columns else "NULL"
    text_id_expr = "textId" if "textId" in nvtx_columns else "NULL"
    ranges = []
    query = f"SELECT start, end, globalTid, {text_expr}, {text_id_expr} FROM NVTX_EVENTS WHERE end IS NOT NULL"
    for start, end, global_tid, text, text_id in connection.execute(query):
        label = text if text else strings.get(int(text_id or 0), "")
        match = _RANGE.fullmatch(label or "")
        if match:
            ranges.append((int(start), int(end), _process_id(global_tid), match.groupdict()))

    calls_by_pid: dict[int, tuple[list[int], list[int]]] = {}
    rows = connection.execute(
        "SELECT start, globalTid, correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME ORDER BY start"
    ).fetchall()
    for start, global_tid, correlation_id in rows:
        starts, correlations = calls_by_pid.setdefault(_process_id(global_tid), ([], []))
        starts.append(int(start))
        correlations.append(int(correlation_id))

    activities: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    for table in _GPU_TABLES:
        if table not in tables:
            continue
        name_column = "shortName" if "shortName" in _columns(connection, table) else None
        select_name = name_column or "NULL"
        for start, end, correlation_id, name_id in connection.execute(
            f"SELECT start, end, correlationId, {select_name} FROM {table}"
        ):
            name = strings.get(int(name_id), "?") if name_id is not None else table.rsplit("_", 1)[-1].lower()
            activities[int(correlation_id)].append((int(start), int(end), name))
    connection.close()

    result: dict[tuple[str, str, int, str, int], list[IterationActivity]] = defaultdict(list)
    for start, end, pid, fields in sorted(ranges, key=lambda item: item[0]):
        starts, correlations = calls_by_pid.get(pid, ([], []))
        first_call = bisect.bisect_left(starts, start)
        last_call = bisect.bisect_right(starts, end)
        iteration = IterationActivity(phase=fields["phase"], iteration=int(fields["iter"]))
        for correlation_id in correlations[first_call:last_call]:
            for gpu_start, gpu_end, name in activities.get(correlation_id, ()):
                iteration.intervals.append((gpu_start, gpu_end))
                iteration.kernel_names.append((name, gpu_end - gpu_start))
        key = (fields["cat"], fields["case"], int(fields["mbs"]), fields["mode"], int(fields["round"]))
        result[key].append(iteration)
    return result


def _percentile(values: list[float], fraction: float) -> float:
    # Nearest-rank, identical to mbench.timing.TimingResult.summary.
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def nsys_columns(
    record: dict[str, Any], nsys: dict[tuple[str, str, int, str, int], list[IterationActivity]]
) -> dict[str, Any]:
    """Aggregate the Nsys view of one case over its accepted rounds."""

    iter_ms: list[float] = []
    span_ms: list[float] = []
    union_ms: list[float] = []
    sum_ms: list[float] = []
    counts: list[int] = []
    kernel_time: dict[str, int] = defaultdict(int)
    for round_info in record["rounds"]:
        key = (
            record["category"],
            record["case"],
            int(record["micro_batch_size"]),
            record["mode"],
            int(round_info["round"]),
        )
        previous_end: int | None = None
        for iteration in nsys.get(key, []):
            if not iteration.intervals:
                continue
            last_end = max(end for _, end in iteration.intervals)
            if iteration.phase == "profile":
                if previous_end is not None:
                    iter_ms.append((last_end - previous_end) / 1e6)
                first_start = min(start for start, _ in iteration.intervals)
                span_ms.append((last_end - first_start) / 1e6)
                union_ms.append(_union_ns(iteration.intervals) / 1e6)
                sum_ms.append(sum(end - start for start, end in iteration.intervals) / 1e6)
                counts.append(len(iteration.intervals))
                for name, duration in iteration.kernel_names:
                    kernel_time[name] += duration
            previous_end = last_end
    if not iter_ms:
        return {}
    total_kernel_time = sum(kernel_time.values()) or 1
    top = sorted(kernel_time.items(), key=lambda item: item[1], reverse=True)[:5]
    iter_median = statistics.median(iter_ms)
    union_median = statistics.median(union_ms)
    event_median = record["timing"]["median_ms"]
    return {
        "nsys_samples": len(iter_ms),
        "nsys_iter_median_ms": iter_median,
        "nsys_iter_p90_ms": _percentile(iter_ms, 0.9),
        "nsys_span_median_ms": statistics.median(span_ms),
        "nsys_kernel_union_median_ms": union_median,
        "nsys_kernel_sum_median_ms": statistics.median(sum_ms),
        "nsys_gpu_busy_ratio": union_median / iter_median if iter_median else None,
        "nsys_activities_per_iter": statistics.median(counts),
        "nsys_vs_event_ratio": iter_median / event_median if event_median else None,
        "nsys_top_kernels": "; ".join(f"{name}:{100.0 * time / total_kernel_time:.1f}%" for name, time in top),
    }


def _row(record: dict[str, Any]) -> dict[str, Any]:
    timing = record["timing"]
    telemetry = record.get("telemetry", {})
    return {
        "category": record["category"],
        "case": record["case"],
        "megatron_module": record["megatron_module"],
        "micro_batch_size": record["micro_batch_size"],
        "mode": record["mode"],
        **record["params"],
        "flops": record["flops"],
        "event_median_ms": timing["median_ms"],
        "event_mean_ms": timing["mean_ms"],
        "event_min_ms": timing["min_ms"],
        "event_p90_ms": timing["p90_ms"],
        "event_max_ms": timing["max_ms"],
        "event_std_ms": timing["std_ms"],
        "samples": timing["samples"],
        "tflops_median": record["tflops_median"],
        "gated": timing["gated"],
        "iters_per_round": timing["iters_per_round"],
        "discarded_rounds": timing["discarded_rounds"],
        "peak_memory_gib": record["peak_memory_gib"],
        "sm_clock_mhz": telemetry.get("sm_clock_mhz"),
        "power_w": telemetry.get("power_w"),
    }


def build_category(category_dir: Path, *, sqlite_path: Path | None, output_path: Path) -> int:
    """Write the CSV of one category and return its row count."""

    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(category_dir.glob("*.json"))]
    if not records:
        return 0
    nsys = load_nsys(sqlite_path) if sqlite_path is not None else {}
    case_order: dict[str, int] = {}
    for record in records:
        case_order.setdefault(record["case"], len(case_order))
    records.sort(
        key=lambda record: (
            case_order[record["case"]],
            int(record["micro_batch_size"]),
            0 if record["mode"] == "fwd" else 1,
        )
    )
    rows = []
    param_columns: list[str] = []
    for record in records:
        row = _row(record)
        if nsys:
            row.update(nsys_columns(record, nsys))
        for name in record["params"]:
            if name not in param_columns:
                param_columns.append(name)
        rows.append(row)
    columns = _BASE_COLUMNS + param_columns + _RESULT_COLUMNS + (_NSYS_COLUMNS if nsys else [])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    """Write ``<input-dir>/csv/<category>.csv`` for every category directory."""

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--nsys-dir", type=Path, default=None, help="default: <input-dir>/nsys")
    parser.add_argument("--output-dir", type=Path, default=None, help="default: <input-dir>/csv")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    nsys_dir = args.nsys_dir or args.input_dir / "nsys"
    output_dir = args.output_dir or args.input_dir / "csv"
    for category_dir in sorted(path for path in args.input_dir.iterdir() if path.is_dir()):
        if category_dir.resolve() in {nsys_dir.resolve(), output_dir.resolve()}:
            continue
        sqlite_path = nsys_dir / f"{category_dir.name}.sqlite"
        rows = build_category(
            category_dir,
            sqlite_path=sqlite_path if sqlite_path.is_file() else None,
            output_path=output_dir / f"{category_dir.name}.csv",
        )
        if rows:
            LOGGER.info(
                "%s: %d rows%s",
                category_dir.name,
                rows,
                f" (nsys: {sqlite_path})" if sqlite_path.is_file() else " (no nsys export)",
            )


if __name__ == "__main__":
    main()
