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
"""Summarize GEMM and copy-engine interference from rank-local Nsight Systems captures."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import sqlite3
import statistics
import subprocess
from collections import defaultdict
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path
from typing import NamedTuple

from analyse_hbm_conflict import Sample, export_sqlite, merge_intervals, overlap_ns, report_rank, window_metrics


LOGGER = logging.getLogger(__name__)
_RANGE = re.compile(r"(gemm|comm)(\d+)")
# Older suites used task5 for the host-source multicast case.
_CASE_DIR = re.compile(r"(?:task[1-5]|baseline[1-3])-.+")
_REQUIRED_METRICS = {
    "DRAM read": re.compile(r"dram.*read|read.*dram", re.I),
    "DRAM write": re.compile(r"dram.*write|write.*dram", re.I),
    "NVLink RX": re.compile(r"^(?!.*(?:c2c|ctc)).*(?:nvlink.*(?:\brx\b|receiv)|\bnvlrx(?:\b|_))", re.I),
    "NVLink TX": re.compile(r"^(?!.*(?:c2c|ctc)).*(?:nvlink.*(?:\btx\b|transmit)|\bnvltx(?:\b|_))", re.I),
}
_C2C_RX = re.compile(r"(?:c2c|ctc).*?(?:rx|receiv|read).*?user", re.I)
_GPU_ACTIVITY = ("KERNEL", "MEMCPY", "MEMCPY2")
_CONFIG_FIELDS = ("case", "mode", "world_size", "gemm_mnk", "size_bytes", "comm_payload_bytes")


class Activity(NamedTuple):
    """A GPU kernel or payload memcpy with a host launch in a measured NVTX range."""

    start: int
    end: int
    device: int
    name: str
    size_bytes: int
    copy_kind: str
    src_kind: str
    dst_kind: str


def _enum_name(values: dict[int, str], value: int | None) -> str:
    if value is None:
        return "unavailable"
    return f"{value} ({values[value]})" if value in values else str(value)


def _read_capture(sqlite_path: Path, *, case: str) -> tuple[dict[str, list[Activity]], dict[int, list[Sample]]]:
    with closing(sqlite3.connect(f"{sqlite_path.resolve().as_uri()}?mode=ro", uri=True)) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        labels = conn.execute(
            """SELECT DISTINCT COALESCE(n.text, s.value) FROM NVTX_EVENTS n
               LEFT JOIN StringIds s ON s.id = n.textId
               WHERE n.end IS NOT NULL AND COALESCE(n.text, s.value) LIKE 'hbm-ce/%'"""
        ).fetchall()
        if labels != [(f"hbm-ce/{case}",)]:
            raise ValueError(f"expected NVTX case hbm-ce/{case}, found {labels}")

        api_tables = [
            f"CUPTI_ACTIVITY_KIND_{kind}" for kind in ("RUNTIME", "DRIVER") if f"CUPTI_ACTIVITY_KIND_{kind}" in tables
        ]
        if not api_tables:
            raise ValueError("capture has no CUDA runtime or driver API activity")
        api_calls = " UNION ALL ".join(f"SELECT start, globalTid, correlationId FROM {table}" for table in api_tables)
        activities = defaultdict(list)
        seen_copies = set()
        copy_enum, memory_enum = (
            dict(conn.execute(f"SELECT id, COALESCE(label, name) FROM {table}")) if table in tables else {}
            for table in ("ENUM_CUDA_MEMCPY_OPER", "ENUM_CUDA_MEM_KIND")
        )
        for kind in _GPU_ACTIVITY:
            table = f"CUPTI_ACTIVITY_KIND_{kind}"
            if table not in tables:
                continue
            is_copy = kind != "KERNEL"
            name = f"'{kind}'" if is_copy else "COALESCE(s.value, 'kernel')"
            bytes_copied = "g.bytes" if is_copy else "0"
            kernel_names = "" if is_copy else "LEFT JOIN StringIds s ON s.id = g.shortName"
            # CE synchronization can contain small flag writes, which are separate from the payload.
            payload_filter = "AND g.bytes > 8" if is_copy else ""
            columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")} if is_copy else set()
            kind_fields = ", ".join(
                f"g.{col}" if col in columns else "NULL" for col in ("copyKind", "srcKind", "dstKind")
            )
            query = f"""
                WITH api AS ({api_calls}),
                ranges AS (
                    SELECT n.start, n.end, n.globalTid, COALESCE(n.text, s.value) AS label
                    FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id = n.textId
                    WHERE n.end IS NOT NULL
                )
                SELECT DISTINCT n.label, g.start, g.end, g.deviceId, {name}, {bytes_copied},
                                g.correlationId, g.streamId, {kind_fields}
                FROM {table} g
                JOIN api a ON a.correlationId = g.correlationId
                    AND ((a.globalTid >> 24) & 16777215) = ((g.globalPid >> 24) & 16777215)
                JOIN ranges n ON n.globalTid = a.globalTid AND a.start >= n.start AND a.start < n.end
                JOIN ranges c ON (c.globalTid >> 24) = (n.globalTid >> 24)
                    AND c.label = ? AND n.start >= c.start AND n.end <= c.end
                {kernel_names}
                WHERE (n.label GLOB 'gemm[0-9]*' OR n.label GLOB 'comm[0-9]*')
                      AND g.end > g.start {payload_filter}
                ORDER BY g.start
            """
            for label, start, end, device, activity_name, size, correlation, stream, copy, src, dst in conn.execute(
                query, (f"hbm-ce/{case}",)
            ):
                if not _RANGE.fullmatch(label):
                    continue
                if is_copy:
                    key = (label, start, end, device, size, correlation, stream)
                    if key in seen_copies:
                        continue
                    seen_copies.add(key)
                activities[label].append(
                    Activity(
                        start,
                        end,
                        device,
                        activity_name,
                        size,
                        _enum_name(copy_enum, copy),
                        _enum_name(memory_enum, src),
                        _enum_name(memory_enum, dst),
                    )
                )

        samples = defaultdict(list)
        if {"GPU_METRICS", "TARGET_INFO_GPU_METRICS"} <= tables:
            query = """
                SELECT DISTINCT g.timestamp, g.typeId & 255, n.metricName, g.value
                FROM GPU_METRICS g
                JOIN TARGET_INFO_GPU_METRICS n ON (g.typeId & 255) = (n.typeId & 255) AND g.metricId = n.metricId
                ORDER BY g.timestamp
            """
            for timestamp, device, metric, value in conn.execute(query):
                if value is not None and math.isfinite(value):
                    samples[device].append(Sample(timestamp, metric, value))
        return dict(activities), dict(samples)


def _event_intervals(record: dict, kind: str) -> list[tuple[int, int]]:
    intervals = [(round(start * 1e6), round(end * 1e6)) for start, end in record[f"{kind}_ms"]]
    if any(start < 0 or end <= start for start, end in intervals):
        raise ValueError(f"invalid CUDA event timestamps for {kind}")
    return intervals


def _iterations(activities: dict[str, list[Activity]], *, label_prefix: str) -> dict[int, list[Activity]]:
    iterations = {}
    for label, group in activities.items():
        match = _RANGE.fullmatch(label)
        if match and match[1] == label_prefix:
            selected = [activity for activity in group if bool(activity.size_bytes) == (label_prefix == "comm")]
            if selected:
                iterations[int(match[2])] = selected
    return iterations


def _busy(activities: Sequence[Activity]) -> list[tuple[int, int]]:
    return merge_intervals([(activity.start, activity.end) for activity in activities])


def _analyse_rank(record: dict, activities: dict[str, list[Activity]], samples: dict[int, list[Sample]]) -> dict:
    errors = []
    gemm_events = _event_intervals(record, "gemm")
    comm_events = _event_intervals(record, "comm")
    gemm = _iterations(activities, label_prefix="gemm")
    copies = _iterations(activities, label_prefix="comm")
    gemm_complete = sorted(gemm) == list(range(len(gemm_events)))
    copies_complete = sorted(copies) == list(range(len(comm_events)))
    if not gemm_complete:
        errors.append(f"expected {len(gemm_events)} GEMM kernel ranges, found indices {sorted(gemm)}")
    if not copies_complete:
        errors.append(f"expected {len(comm_events)} payload copy ranges, found indices {sorted(copies)}")
    if record["mode"] != "gemm" and not comm_events:
        errors.append("no communication rounds measured")
    if record.get("comm_rounds_measured", len(comm_events)) != len(comm_events):
        errors.append("communication round count differs from event count")
    if record["mode"].startswith("gemm") != (record["gemm_mnk"] is not None and bool(gemm_events)):
        errors.append("recorded GEMM shape or rounds do not match the selected mode")
    if record["output_correct"] is not True:
        errors.append("worker reported incorrect output")
    comm_kernels = sorted(
        {op.name for name, group in activities.items() if name.startswith("comm") for op in group if not op.size_bytes}
    )
    if comm_kernels:
        errors.append(f"communication launched kernels instead of a CE-only path: {comm_kernels}")
    source_bytes = record["size_bytes"]
    if "ag-" in record["mode"]:
        source_bytes //= record["world_size"]
    for index, group in sorted(copies.items()):
        total = sum(op.size_bytes for op in group)
        if total < source_bytes:
            errors.append(f"comm{index:02d}: traced {total} copy bytes, expected at least {source_bytes} source bytes")
            copies_complete = False
    compute_ops = [op for group in gemm.values() for op in group]
    copy_ops = [op for group in copies.values() for op in group]
    if record["mode"].endswith("ag-hbm") and any(
        re.search(r"host|pinned|pageable", op.src_kind, re.I) for op in copy_ops
    ):
        errors.append("HBM-source AG has a payload copy with a CUPTI host source")
    if record["mode"].endswith("h2d") and any(
        re.search(r"device|unified|managed", op.src_kind, re.I) for op in copy_ops
    ):
        errors.append("H2D has a payload copy with a CUPTI non-host source")
    devices = {op.device for op in compute_ops or copy_ops}
    device = next(iter(devices)) if len(devices) == 1 else None
    if device is None:
        errors.append(f"cannot identify a unique rank-local GPU from activity devices {sorted(devices)}")
    if device is not None and any(op.device != device for op in copy_ops):
        errors.append(f"copy activities include a device other than the rank-local GPU {device}")

    comm_union = _busy(copy_ops)
    if gemm_events and comm_events and gemm_complete and copies_complete:
        if not any(overlap_ns(start, end, comm_union) for start, end in _busy(compute_ops)):
            errors.append("no GEMM kernel overlapped a traced local payload copy")
    event_union = merge_intervals(comm_events)
    m, n, k = record["gemm_mnk"] or (0, 0, 0)
    gemms = []
    for index, (event_start, event_end) in enumerate(gemm_events):
        group = gemm.get(index, [])
        gpu_intervals = _busy(group)
        gpu_ns = sum(end - start for start, end in gpu_intervals)
        event_ns = event_end - event_start
        has_trace = bool(gpu_intervals) and copies_complete
        covered_ns = sum(overlap_ns(start, end, comm_union) for start, end in gpu_intervals)
        gemms.append(
            {
                "index": index,
                "nsys_kernel_ms": gpu_ns / 1e6 if group else None,
                "nsys_kernel_tflops": 2 * m * n * k / gpu_ns / 1000 if group else None,
                "nsys_copy_coverage": covered_ns / gpu_ns if has_trace else None,
                "event_ms": event_ns / 1e6,
                "event_tflops": 2 * m * n * k / event_ns / 1000,
                "event_copy_coverage": overlap_ns(event_start, event_end, event_union) / event_ns,
                "kernels": sorted({op.name for op in group}),
            }
        )

    # The launcher requests one physical GPU's counter set in each rank capture. The CUDA
    # activity ordinal can be remapped by CUDA_VISIBLE_DEVICES, so keep the identifiers separate.
    metric_device = next(iter(samples)) if len(samples) == 1 else None
    requested_gpu = record.get("physical_gpu")
    if metric_device is None:
        errors.append(f"expected one rank-local GPU counter source, found {sorted(samples)}")
    elif requested_gpu is not None and metric_device != requested_gpu:
        errors.append(f"requested physical GPU {requested_gpu}, found counters for GPU {metric_device}")
    metrics = []
    for window, ops in (("gemm_span", compute_ops), ("copy_span", copy_ops)):
        if not ops or metric_device is None:
            continue
        start, end = min(op.start for op in ops), max(op.end for op in ops)
        counters = window_metrics(samples[metric_device], start, end)
        missing = [kind for kind, pattern in _REQUIRED_METRICS.items() if not any(pattern.search(n) for n in counters)]
        if missing:
            errors.append(f"{window}: no sampled {', '.join(missing)} counters on GPU {metric_device}")
        if window == "copy_span" and record["mode"].endswith(("h2d", "ag-host")):
            if not any(_C2C_RX.search(name) and values["peak"] > 0 for name, values in counters.items()):
                errors.append(f"{window}: no positive C2C user RX samples for host-sourced copies")
        if window == "copy_span" and "ag-" in record["mode"]:
            for direction in ("NVLink RX", "NVLink TX"):
                if not any(
                    _REQUIRED_METRICS[direction].search(name) and values["peak"] > 0
                    for name, values in counters.items()
                ):
                    errors.append(f"{window}: no positive {direction} samples for multicast AG")
        metrics.extend(
            {
                "case": record["case"],
                "rank": record["rank"],
                "gpu": metric_device,
                "window": window,
                "start_ns": start,
                "end_ns": end,
                "metric": name,
                "unit": "nsys_raw",
                **values,
            }
            for name, values in counters.items()
        )

    return {
        "rank": record["rank"],
        "cuda_activity_device": device,
        "nsys_metrics_device": metric_device,
        "physical_gpu": requested_gpu,
        "output_correct": record["output_correct"],
        "errors": errors,
        "gemms": gemms,
        "metrics": metrics,
        "communication": [
            {
                "index": index,
                "event_ms": (end - start) / 1e6,
                "event_source_GBps": source_bytes / (end - start),
                "event_receive_payload_proxy_GBps": record["comm_payload_bytes"] / (end - start),
                "nsys_payload_copy_bytes": sum(op.size_bytes for op in copies.get(index, [])),
                "nsys_payload_copy_intervals_ns": _busy(copies.get(index, [])),
                "nsys_copy_kinds": sorted({(op.copy_kind, op.src_kind, op.dst_kind) for op in copies.get(index, [])}),
            }
            for index, (start, end) in enumerate(comm_events)
        ],
    }


def _analyse_case(case_dir: Path, *, reports_dir: Path, nsys_bin: str) -> dict:
    case = case_dir.name
    records = {}
    for path in sorted(case_dir.glob("events.rank_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        rank = int(path.stem.rsplit("_", 1)[1])
        if rank != record["rank"] or rank in records:
            raise ValueError(f"{path.name}: inconsistent or duplicate rank")
        records[rank] = record
    if not records:
        raise ValueError("missing events.rank_*.json")
    config = {key: records[min(records)][key] for key in _CONFIG_FIELDS}
    expected_ranks = set(range(config["world_size"]))
    if config["case"] != case or set(records) != expected_ranks:
        raise ValueError(
            f"inconsistent case or missing event ranks, expected {sorted(expected_ranks)}, got {sorted(records)}"
        )
    if any(any(record[key] != val for key, val in config.items()) for record in records.values()):
        raise ValueError("worker case configuration differs between ranks")

    reports = {}
    paths = [*reports_dir.glob(f"{case}.rank_*.nsys-rep"), *(case_dir / "nsys").glob(f"{case}.rank_*.nsys-rep")]
    for path in sorted(paths):
        rank = report_rank(path)
        if rank in reports:
            raise ValueError(f"duplicate nsys report for rank {rank}")
        reports[rank] = path
    errors = []
    if set(reports) != expected_ranks:
        errors.append(f"expected nsys reports for {sorted(expected_ranks)}, got {sorted(reports)}")

    ranks = []
    for rank, record in sorted(records.items()):
        try:
            report = reports[rank]
            capture = _read_capture(export_sqlite(report, nsys_bin=nsys_bin), case=case)
        except (KeyError, OSError, ValueError, subprocess.CalledProcessError, sqlite3.Error) as error:
            errors.append(f"rank {rank}: capture unavailable or invalid: {error}")
            capture = {}, {}
        ranks.append(_analyse_rank(record, *capture))

    return {"case": case, "config": config, "ranks": ranks, "errors": errors}


def _median(values: Sequence[float | None]) -> str:
    measured = [value for value in values if value is not None]
    return f"{statistics.median(measured):.3f}" if measured else "—"


def _render(cases: Sequence[dict], *, root: Path) -> str:
    lines = [
        "# GB200 GEMM and copy-engine overlap",
        "",
        "GEMM throughput is 2MNK divided by the union of launch-attributed GPU kernel intervals.",
        "Coverage is the fraction of those intervals overlapping this rank's traced payload copies, excluding signal writes.",
        "Copies issued by peer ranks are absent from the rank-local trace. GPU counters are device-wide.",
        "CUDA-event coverage uses the stream-ordered interval between the operation's start and end events.",
        "Source GB/s uses the size of one outgoing shard or H2D transfer and its CUDA-event duration.",
        "Receive payload proxy GB/s normalizes logical receive bytes by the rank's outgoing event duration.",
        "For multicast AG the logical receive bytes exclude the local shard. Peer completion is outside this timing.",
        "CUPTI copy bytes and source/destination enums in summary.json describe outgoing copies as traced.",
        "GPU metric CSV windows span the first to last traced kernel or payload copy and include any intervening gaps.",
        "Counter units and scopes follow Nsight Systems. DRAM bandwidth metrics indicate interface activity and NVLink",
        "throughput metrics are relative to peak. CTC metrics measure the CPU–GPU NVLink-C2C path.",
        "Raw samples are retained in their original units.",
        "",
        "| Case | Status | GPU GEMM TFLOP/s | Event GEMM TFLOP/s | GPU copy coverage | Event coverage "
        "| Source GB/s | Receive payload proxy GB/s |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    failures = []
    counter_rows = []
    for case in cases:
        gemms = [item for rank in case["ranks"] for item in rank["gemms"]]
        copies = [item for rank in case["ranks"] for item in rank["communication"]]
        errors = case["errors"] + [
            f"rank {rank['rank']}: {error}" for rank in case["ranks"] for error in rank["errors"]
        ]
        status = "failed" if errors else "ok"
        cells = [
            case["case"],
            status,
            _median([g["nsys_kernel_tflops"] for g in gemms]),
            _median([g["event_tflops"] for g in gemms]),
            _median([g["nsys_copy_coverage"] for g in gemms]),
            _median([g["event_copy_coverage"] for g in gemms]),
            _median([c["event_source_GBps"] for c in copies]),
            _median([c["event_receive_payload_proxy_GBps"] for c in copies]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
        failures.extend(f"- {case['case']} / {error}" for error in errors)
        means = defaultdict(list)
        for rank in case["ranks"]:
            for metric in rank["metrics"]:
                name = metric["metric"]
                if re.search(r"dram|nvlink|nvl[rt]x|c2c|ctc|clock", name, re.I):
                    means[metric["window"], name].append(metric["mean"])
        for (window, name), values in sorted(means.items()):
            counter_rows.append(f"| {case['case']} | {window} | {name} | {_median(values)} | {len(values)} |")
    if failures:
        lines.extend(["", "## Evidence gaps", "", *failures])
    clock_files = [name for name in ("gpu_clocks.before.csv", "gpu_clocks.after.csv") if (root / name).is_file()]
    if clock_files:
        links = ", ".join(f"[{name}]({name})" for name in clock_files)
        lines.extend(["", f"Suite clock readings from nvidia-smi are available in {links}."])
    if counter_rows:
        lines.extend(
            [
                "",
                "## GPU counters",
                "",
                "Each value is the median of rank-local sample means over the indicated GPU window.",
                "",
                "| Case | Window | Metric | Median mean (raw) | Rank count |",
                "| --- | --- | --- | ---: | ---: |",
                *counter_rows,
            ]
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    """Analyse every benchmark case and fail if a required trace or counter is missing."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="Suite root with cases and nsys reports")
    parser.add_argument("--nsys", required=True, help="Nsight Systems executable used to export SQLite")
    args = parser.parse_args(argv)

    root = args.output_dir
    case_dirs = sorted(path for path in root.iterdir() if path.is_dir() and _CASE_DIR.fullmatch(path.name))
    if not case_dirs:
        parser.error(f"{root}: no task case directories")
    cases = []
    for path in case_dirs:
        try:
            cases.append(_analyse_case(path, reports_dir=root / "nsys", nsys_bin=args.nsys))
        except (KeyError, OSError, TypeError, ValueError) as error:
            cases.append({"case": path.name, "ranks": [], "errors": [str(error)]})

    (root / "summary.json").write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")
    (root / "report.md").write_text(_render(cases, root=root), encoding="utf-8")
    fields = ("case", "rank", "gpu", "window", "start_ns", "end_ns", "metric", "unit", "samples", "mean", "peak")
    with (root / "gpu_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row for case in cases for rank in case["ranks"] for row in rank["metrics"])
    failed = [case["case"] for case in cases if case["errors"] or any(rank["errors"] for rank in case["ranks"])]
    if failed:
        parser.exit(1, f"missing or invalid evidence in {', '.join(failed)}; see {root / 'report.md'}\n")
    LOGGER.info("wrote %s", root / "report.md")


if __name__ == "__main__":
    main()
