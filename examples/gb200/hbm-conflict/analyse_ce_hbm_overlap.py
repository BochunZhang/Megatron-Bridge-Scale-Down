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
"""Summarize GB200 counters during trace-verified GEMM and CE overlap."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import shutil
import sqlite3
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path

from analyse_ce_hbm import (
    Activity,
    _busy,
    _read_combined,
    _read_hardware_channels,
    _validate_unicast_trace,
)
from analyse_ce_hbm_kernel_trace import _read_records, _validate_operations
from analyse_hbm_conflict import Sample, overlap_ns


_LOG = logging.getLogger(__name__)
_PERIOD_NS = 5_000
_MAX_SAMPLE_NS = 1.5 * _PERIOD_NS
_MODES = {"gemm", "gemm-ag-hbm", "gemm-ag-host", "gemm-ag-hbm-unicast"}
_TRACE_TABLES = (
    "^CUPTI_ACTIVITY_KIND_RUNTIME$,^CUPTI_ACTIVITY_KIND_DRIVER$,^CUPTI_ACTIVITY_KIND_KERNEL$,"
    "^CUPTI_ACTIVITY_KIND_MEMCPY$,^CUPTI_ACTIVITY_KIND_MEMCPY2$,"
    "^NVTX_EVENTS$,^StringIds$,^ENUM_CUDA_MEM.*$,^GPU_METRICS$,^TARGET_INFO_GPU_METRICS$"
)
_METRIC_COLUMNS = {
    "sm_active_pct": ("SM Active",),
    "dram_read_gbps": ("DRAM Read",),
    "dram_write_gbps": ("DRAM Write",),
    "c2c_rx_user_gbps": ("C2C RX User Data",),
    "c2c_tx_user_gbps": ("C2C TX User Data",),
    "nvlink_rx_user_gbps": ("NVLink RX Requests User Data", "NVLink RX Responses User Data"),
    "nvlink_tx_user_gbps": ("NVLink TX Requests User Data", "NVLink TX Responses User Data"),
}
_RAW_METRICS = {name for names in _METRIC_COLUMNS.values() for name in names}
_CSV_COLUMNS = ("gpu", "gemm", "start_ns", "end_ns", *_METRIC_COLUMNS)


def _export(report: Path, *, nsys_bin: str) -> Path:
    target = report.with_suffix(".overlap.sqlite")
    if target.is_file() and target.stat().st_mtime_ns >= report.stat().st_mtime_ns:
        return target
    with tempfile.TemporaryDirectory(prefix=".overlap-", dir=target.parent) as temp:
        staged = Path(temp) / target.name
        subprocess.run(
            [nsys_bin, "export", "--type=sqlite", f"--tables={_TRACE_TABLES}", f"--output={staged}", str(report)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        staged.replace(target)
    return target


def _sample_values(samples: Sequence[Sample]) -> dict[int, dict[str, float]]:
    readings: dict[int, dict[str, float]] = defaultdict(dict)
    for sample in samples:
        name = sample.metric
        for suffix in (" [Throughput %]", " [Sectors]", " [Bytes]", " Throughput %", " Sectors", " Bytes"):
            name = name.removesuffix(suffix)
        if name in _RAW_METRICS:
            values = readings[sample.timestamp]
            if name in values and values[name] != sample.value:
                raise ValueError(f"conflicting samples for {name} at {sample.timestamp} ns")
            values[name] = sample.value
    return readings


def _gemm_samples(
    readings: dict[int, dict[str, float]],
    *,
    gpu: int,
    gemm: int,
    windows: Sequence[tuple[int, int]],
) -> list[dict[str, int | float]]:
    rows = []
    # Sample values apply to the interval beginning at their timestamp. Keep only
    # full intervals so inactive edges contribute neither bytes nor elapsed time.
    for start, end in pairwise(sorted(readings)):
        covered = overlap_ns(start, end, windows)
        if covered and end - start > _MAX_SAMPLE_NS:
            raise ValueError(f"GPU {gpu} GEMM {gemm}: gap in GPU metrics at {start} ns")
        if covered != end - start:
            continue
        raw = readings[start]
        if raw.keys() != _RAW_METRICS or any(not math.isfinite(v) or v < 0 for v in raw.values()):
            raise ValueError(f"GPU {gpu} GEMM {gemm}: incomplete GPU metrics at {start} ns")
        result: dict[str, int | float] = {"gpu": gpu, "gemm": gemm, "start_ns": start, "end_ns": end}
        for metric, names in _METRIC_COLUMNS.items():
            count = sum(raw[name] for name in names)
            if metric == "sm_active_pct":
                result[metric] = count
            else:
                # DRAM counters count 32-byte sectors. Bytes/ns equals decimal GB/s.
                multiplier = 32 if metric.startswith("dram_") else 1
                result[metric] = count * multiplier / (end - start)
        rows.append(result)
    return rows


def _weighted_metrics(rows: Sequence[dict[str, int | float]]) -> dict[str, float]:
    durations = [row["end_ns"] - row["start_ns"] for row in rows]
    total = sum(durations)
    if not total:
        return {}
    return {
        metric: math.fsum(row[metric] * duration for row, duration in zip(rows, durations, strict=True)) / total
        for metric in _METRIC_COLUMNS
    }


def _channel_tool(nsys_bin: str) -> str | None:
    nsys_path = Path(shutil.which(nsys_bin) or nsys_bin).resolve()
    return (
        os.environ.get("DUMP_TIMELINE")
        or shutil.which(str(nsys_path.with_name("DumpTimeline")))
        or shutil.which("DumpTimeline")
    )


def _copy_channel(op: Activity, *, channels: dict[tuple[int, int, int, int, int], set[str]]) -> set[str]:
    return channels.get((op.pid, op.device, op.start, op.end, op.size_bytes), set())


def _analyse(
    *,
    case_dir: Path,
    nsys_bin: str,
    report_gpus: set[int],
    dump_timeline: str | None,
) -> tuple[dict, list[dict[str, int | float]]]:
    records = _read_records(case_dir)
    first = records[0]
    mode = first["mode"]
    if mode not in _MODES:
        raise ValueError(f"unsupported case mode {mode}")
    solo = mode == "gemm"
    unicast = mode == "gemm-ag-hbm-unicast"
    size_bytes = first["size_bytes"]
    rounds = first["comm_rounds_measured"]
    gemm_rounds = len(first["gemm_ms"])
    if not gemm_rounds or not first["gemm_mnk"] or (solo and (rounds or len(records) != 1)):
        raise ValueError("unexpected GEMM count, shape, or standalone communication")
    if not solo and (
        not rounds
        or len(records) < 3
        or size_bytes % len(records)
        or any(
            record.get("all_gather_out_of_place") is not True or record.get("ce_lead_in_rounds") != 1
            for record in records.values()
        )
    ):
        raise ValueError("expected timed out-of-place AllGather and one CE lead-in per rank")
    gpus = {record["physical_gpu"] for record in records.values()}
    if not report_gpus <= gpus:
        raise ValueError(f"requested physical GPUs {sorted(report_gpus)} missing from {sorted(gpus)}")
    report = case_dir / "nsys" / f"{case_dir.name}.nsys-rep"
    errors: list[str] = []
    captures = _read_combined(_export(report, nsys_bin=nsys_bin), case=case_dir.name, records=records, errors=errors)
    if errors or set(captures) != set(records):
        raise ValueError(f"incomplete CUDA/NVTX capture {', '.join(errors)}")

    gemms_by_rank: dict[int, dict[int, list[Activity]]] = {}
    copies_by_rank: dict[int, list[Activity]] = {}
    issuing_devices = set()
    selected_devices = set()
    for rank, record in records.items():
        expected_copy_bytes = size_bytes if unicast else size_bytes // len(records)
        gemms, copies = _validate_operations(
            record, captures[rank][0], expected_gemms=gemm_rounds, expected_bytes=expected_copy_bytes
        )
        devices = {op.device for groups in (gemms, copies) for group in groups.values() for op in group}
        if len(devices) != 1 or devices & issuing_devices:
            raise ValueError(f"rank {rank}: expected one distinct CUDA device, found {sorted(devices)}")
        issuing_devices.update(devices)
        if record["physical_gpu"] in report_gpus:
            selected_devices.update(devices)
        gemms_by_rank[rank] = gemms
        copies_by_rank[rank] = [op for group in copies.values() for op in group]

    tool = dump_timeline or _channel_tool(nsys_bin)
    channels = _read_hardware_channels(report, nsys_bin=nsys_bin, dump_timeline=tool) if not solo and tool else None
    if unicast:
        validation, incoming = _validate_unicast_trace(
            records, {r: captures[r][0] for r in records}, channels, channel_devices=selected_devices
        )
        if any(validation.values()):
            raise ValueError(f"invalid out-of-place unicast payloads {validation}")
        if channels is not None:
            incoming = {
                rank: [op for op in ops if _copy_channel(op, channels=channels) == {"Async Memcpy"}]
                for rank, ops in incoming.items()
            }
    else:
        # One multicast shard from each source writes to every rank's receive window.
        # Nsight exposes the source DMA interval rather than separate destination intervals.
        multicast = [
            op
            for rank, copies in copies_by_rank.items()
            if records[rank]["physical_gpu"] in report_gpus
            for op in copies
        ]
        if channels is not None:
            for op in multicast:
                channel = _copy_channel(op, channels=channels)
                if channel != {"Async Memcpy"}:
                    raise ValueError(
                        f"multicast CUDA device {op.device}: expected Async Memcpy payload, found {channel}"
                    )
        incoming = dict.fromkeys(records, multicast)

    summaries = []
    rows = []
    for rank, record in records.items():
        gpu = record["physical_gpu"]
        if gpu not in report_gpus:
            continue
        readings = _sample_values(captures[rank][1].get(gpu, []))
        if not readings:
            raise ValueError(f"GPU {gpu}: no requested GPU metrics in capture")
        first_sample, last_sample = min(readings), max(readings)
        payload = _busy(incoming[rank])
        gpu_rows = []
        gemm_results = []
        for index, operations in sorted(gemms_by_rank[rank].items()):
            busy = _busy(operations)
            windows = (
                busy
                if solo
                else [
                    (max(start, left), min(end, right))
                    for start, end in busy
                    for left, right in payload
                    if max(start, left) < min(end, right)
                ]
            )
            kernel_ns = sum(end - start for start, end in busy)
            overlap = sum(end - start for start, end in windows)
            samples = _gemm_samples(readings, gpu=gpu, gemm=index, windows=windows)
            if kernel_ns == overlap and (
                not samples
                or first_sample - windows[0][0] > _MAX_SAMPLE_NS
                or windows[-1][1] - last_sample > _MAX_SAMPLE_NS
            ):
                raise ValueError(f"GPU {gpu} GEMM {index}: incomplete 5-us GPU metric sample coverage")
            gpu_rows.extend(samples)
            gemm_results.append(
                {
                    "index": index,
                    "kernel_ns": kernel_ns,
                    "overlap_ns": None if solo else overlap,
                    "coverage": None if solo else overlap / kernel_ns,
                    "sample_count": len(samples),
                    "metrics": _weighted_metrics(samples),
                }
            )
        rows.extend(gpu_rows)
        summaries.append(
            {
                "gpu": gpu,
                "gemms": gemm_results,
                "sample_count": len(gpu_rows),
                "metrics": _weighted_metrics(gpu_rows),
            }
        )
    status = (
        "ok"
        if solo or all(gemm["coverage"] == 1.0 for gpu in summaries for gemm in gpu["gemms"])
        else "insufficient_overlap"
    )
    return {
        "case": case_dir.name,
        "mode": mode,
        "gemm_mnk": first["gemm_mnk"],
        "gemm_rounds": gemm_rounds,
        "size_mib": size_bytes >> 20,
        "comm_rounds": rounds,
        "ce_lead_in_rounds": first.get("ce_lead_in_rounds", 0),
        "status": status,
        "sampling_period_ns": _PERIOD_NS,
        "metric_units": {metric: "%" if metric == "sm_active_pct" else "GB/s" for metric in _METRIC_COLUMNS},
        "payload_timing": "none" if solo else "destination memcpy" if unicast else "multicast source DMA",
        "multicast_coverage_source_gpus": sorted(report_gpus) if not solo and not unicast else [],
        "hardware_channels_verified": channels is not None,
        "gpus": summaries,
    }, rows


def main(argv: Sequence[str] | None = None) -> int:
    """Write metrics for one sweep attempt and return its validation status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nsys", required=True)
    parser.add_argument("--report-gpus", required=True, help="Physical GPU IDs, comma-separated.")
    parser.add_argument(
        "--dump-timeline", help="Optional DumpTimeline executable to verify payload hardware channels."
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        gpu_ids = args.report_gpus.split(",")
        if not all(gpu.isdecimal() for gpu in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids):
            raise ValueError("expected distinct physical GPU IDs")
        cases = [path for path in args.output_dir.iterdir() if path.is_dir() and path.name.startswith("task")]
        if len(cases) != 1:
            raise ValueError(f"expected one captured case directory, found {len(cases)}")
        summary, rows = _analyse(
            case_dir=cases[0],
            nsys_bin=args.nsys,
            report_gpus={int(gpu) for gpu in gpu_ids},
            dump_timeline=args.dump_timeline,
        )
    except (OSError, KeyError, TypeError, ValueError, sqlite3.Error, subprocess.SubprocessError) as error:
        _LOG.error("%s: %s", args.output_dir, error)
        if args.output_dir.is_dir():
            (args.output_dir / "summary.json").write_text(
                json.dumps({"status": "error", "error": str(error)}, indent=2) + "\n", encoding="utf-8"
            )
        return 1
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "gpu_metrics.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    _LOG.info("%s: %s on physical GPUs %s", args.output_dir, summary["status"], args.report_gpus)
    return 0 if summary["status"] == "ok" else 3


if __name__ == "__main__":
    raise SystemExit(main())
