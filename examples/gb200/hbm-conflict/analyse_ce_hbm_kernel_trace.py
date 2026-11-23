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

"""Read GEMM and same-round AllGather overlap and summarize a kernel sweep."""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import statistics
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

from analyse_ce_hbm import Activity, _event_intervals, _iterations, _read_combined
from analyse_hbm_conflict import merge_intervals, overlap_ns


_AG_MODES = ("gemm-ag-hbm", "gemm-ag-host", "gemm-ag-hbm-unicast", "gemm-ag-host-unicast")
_TRACE_TABLES = "^CUPTI_ACTIVITY_KIND_.*$,^NVTX_EVENTS$,^StringIds$,^ENUM_CUDA_MEM.*$"
_CASE_NAMES = {
    "task1": "GEMM only",
    "task3": "HBM-source multicast",
    "task4": "CPU-source multicast",
    "task7": "HBM-source unicast",
    "task8": "CPU-source unicast",
}
_LOG = logging.getLogger(__name__)


def _export_kernel_trace(report: Path, *, nsys_bin: str) -> Path:
    target = report.with_suffix(".kernel-trace.sqlite")
    if target.is_file() and target.stat().st_mtime_ns >= report.stat().st_mtime_ns:
        return target
    with tempfile.TemporaryDirectory(prefix=".kernel-trace-", dir=target.parent) as temp:
        staging = Path(temp) / target.name
        subprocess.run(
            [nsys_bin, "export", "--type=sqlite", f"--tables={_TRACE_TABLES}", f"--output={staging}", str(report)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        staging.replace(target)
    return target


def _read_records(case_dir: Path) -> dict[int, dict]:
    records = {}
    for path in sorted(case_dir.glob("events.rank_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        rank = record["rank"]
        if path.name != f"events.rank_{rank:02d}.json" or rank in records or record["case"] != case_dir.name:
            raise ValueError(f"{path}: inconsistent case or rank")
        if record["output_correct"] is not True:
            raise ValueError(f"{path}: worker reported incorrect output")
        records[rank] = record
    if 0 not in records or set(records) != set(range(records[0]["world_size"])):
        raise ValueError(f"{case_dir}: missing or unexpected worker ranks")
    fields = ("mode", "gemm_mnk", "world_size", "size_bytes", "comm_rounds_measured")
    if any(any(record[key] != records[0][key] for key in fields) for record in records.values()):
        raise ValueError(f"{case_dir}: inconsistent worker configurations")
    return records


def _validate_operations(
    record: dict, activities: dict[str, list[Activity]], *, expected_gemms: int, expected_bytes: int
) -> tuple[dict[int, list[Activity]], dict[int, list[Activity]]]:
    rank = record["rank"]
    gemms = _iterations(activities, label_prefix="gemm")
    copies = _iterations(activities, label_prefix="comm")
    if len(_event_intervals(record, "gemm")) != expected_gemms or sorted(gemms) != list(range(expected_gemms)):
        raise ValueError(f"rank {rank}: expected {expected_gemms} GEMM rounds, traced indices {sorted(gemms)}")
    comm_rounds = record["comm_rounds_measured"]
    if len(_event_intervals(record, "comm")) != comm_rounds or sorted(copies) != list(range(comm_rounds)):
        raise ValueError(f"rank {rank}: expected {comm_rounds} communication rounds, traced indices {sorted(copies)}")
    for label, ops in activities.items():
        if label.startswith("gemm") and any(op.size_bytes for op in ops):
            raise ValueError(f"rank {rank} {label}: unexpected memcpy in measured GEMM")
        if label.startswith("comm") and any(not op.size_bytes for op in ops):
            raise ValueError(f"rank {rank} {label}: unexpected CUDA kernel in copy-engine AllGather")
    source_kind = "pinned" if "-host" in record["mode"] else "device"
    for index, ops in copies.items():
        payload_bytes = sum(op.size_bytes for op in ops)
        if payload_bytes != expected_bytes:
            raise ValueError(f"rank {rank} comm{index:02d}: traced {payload_bytes} B, expected {expected_bytes} B")
        if any(source_kind not in op.src_kind.lower() or "device" not in op.dst_kind.lower() for op in ops):
            raise ValueError(
                f"rank {rank} comm{index:02d}: expected {source_kind} payload source and device destination"
            )
    return gemms, copies


def analyse_kernel_case(case_dir: Path, *, nsys_bin: str) -> dict:
    """Measure first-rank GEMMs and their full coverage by individual AllGather rounds.

    Args:
        case_dir: HBM benchmark result directory containing the combined Nsight report.
        nsys_bin: Nsight Systems executable used to export CUDA and NVTX tables.

    Returns:
        Trace-verified timings and coverage. Zero overlap is a valid result.

    Raises:
        ValueError: The capture or worker metadata cannot establish the measured operations.
    """
    records = _read_records(case_dir)
    first = records[0]
    mode = first["mode"]
    if mode not in ("gemm", *_AG_MODES):
        raise ValueError(f"{case_dir}: expected standalone GEMM or GEMM with CE AllGather, found {mode!r}")
    expected_gemms = len(first["gemm_ms"])
    comm_rounds = first["comm_rounds_measured"]
    if not expected_gemms or not first["gemm_mnk"] or (mode == "gemm") != (comm_rounds == 0):
        raise ValueError(f"{case_dir}: GEMM or communication rounds do not match the selected mode")
    world_size = first["world_size"]
    if mode == "gemm":
        if world_size != 1:
            raise ValueError(f"{case_dir}: the standalone baseline must use one worker")
        expected_bytes = 0
    else:
        shard_bytes, remainder = divmod(first["size_bytes"], world_size)
        if world_size < 2 or remainder or shard_bytes <= 8:
            raise ValueError(f"{case_dir}: invalid AllGather output size or world size")
        expected_bytes = shard_bytes
        if mode.endswith("-unicast"):
            expected_bytes = first["size_bytes"] if "-host" in mode else first["size_bytes"] - shard_bytes
            if any(record["comm_source_bytes"] != expected_bytes for record in records.values()):
                raise ValueError(f"{case_dir}: incorrect unicast source byte count")

    report = case_dir / "nsys" / f"{case_dir.name}.nsys-rep"
    db = _export_kernel_trace(report, nsys_bin=nsys_bin)
    errors: list[str] = []
    try:
        captures = _read_combined(db, case=case_dir.name, records=records, errors=errors)
    except sqlite3.Error as error:
        raise ValueError(f"{case_dir}: unable to read measured CUDA/NVTX activities: {error}") from error
    if errors or set(captures) != set(records):
        raise ValueError(f"{case_dir}: incomplete CUDA/NVTX capture {', '.join(errors)}")

    gpu0_operations = {}
    copies_by_round: list[list[Activity]] = [[] for _ in range(comm_rounds)]
    gemm_devices: set[int] = set()
    for rank, record in records.items():
        gemms, copies = _validate_operations(
            record, captures[rank][0], expected_gemms=expected_gemms, expected_bytes=expected_bytes
        )
        devices = {op.device for ops in gemms.values() for op in ops}
        if len(devices) != 1 or not devices.isdisjoint(gemm_devices):
            raise ValueError(f"{case_dir}: rank {rank} GEMMs must run on a distinct CUDA device, traced {devices}")
        gemm_devices.update(devices)
        if rank == 0:
            gpu0_operations = gemms
        for index, ops in copies.items():
            copies_by_round[index].extend(ops)

    intervals = [merge_intervals([(op.start, op.end) for op in ops]) for ops in copies_by_round]
    ag_rounds = [
        {"index": index, "payload_intervals_ns": spans, "gpu0_gemms": []} for index, spans in enumerate(intervals)
    ]
    gpu0_gemms = []
    for index, ops in sorted(gpu0_operations.items()):
        kernel_spans = merge_intervals([(op.start, op.end) for op in ops])
        start, end = kernel_spans[0][0], kernel_spans[-1][1]
        # A logical GEMM may launch several kernels. Include any gaps in the coverage requirement.
        overlapping_rounds = [
            wave for wave, spans in enumerate(intervals) if overlap_ns(start, end, spans) == end - start
        ]
        for wave in overlapping_rounds:
            ag_rounds[wave]["gpu0_gemms"].append(index)
        gpu0_gemms.append(
            {
                "index": index,
                "start_ns": start,
                "end_ns": end,
                "duration_us": sum(b - a for a, b in kernel_spans) / 1000,
                "kernel_count": len(ops),
                "ag_rounds": overlapping_rounds,
            }
        )
    covered = [gemm for gemm in gpu0_gemms if gemm["ag_rounds"]]
    selected = gpu0_gemms if mode == "gemm" else covered
    return {
        "case": case_dir.name,
        "mode": mode,
        "gemm_mnk": first["gemm_mnk"],
        "size_bytes": first["size_bytes"],
        "world_size": world_size,
        "physical_gpu": first["physical_gpu"],
        "comm_rounds_measured": comm_rounds,
        "gpu0_gemm_count": len(gpu0_gemms),
        "gpu0_covered_gemm_count": len(covered),
        "gpu0_max_gemms_per_ag_round": max((len(ag_round["gpu0_gemms"]) for ag_round in ag_rounds), default=0),
        "gpu0_gemm_mean_us": statistics.fmean(gemm["duration_us"] for gemm in selected) if selected else None,
        "gpu0_gemms": gpu0_gemms,
        "ag_rounds": ag_rounds,
    }


def _analyse_attempt(args: argparse.Namespace) -> int:
    attempt = {"output_dir": str(args.output_dir), "size_mib": args.size_mib}
    try:
        if args.launch_exit_code:
            raise ValueError(f"CE benchmark launcher exited with status {args.launch_exit_code}")
        produced = [path for path in args.output_dir.glob(f"{args.case}-*") if path.is_dir()]
        if len(produced) != 1:
            raise ValueError(f"expected one captured case, found {len(produced)}")
        trace = analyse_kernel_case(produced[0], nsys_bin=args.nsys)
        if (
            trace["gpu0_gemm_count"] != args.gemm_rounds
            or trace["gemm_mnk"] != [args.m, 8192, 7168]
            or trace["size_bytes"] != args.size_mib << 20
            or (args.case != "task1" and trace["comm_rounds_measured"] < args.comm_rounds)
        ):
            raise ValueError(f"{produced[0]}: captured workload differs from requested shape, size, or rounds")
        attempt["trace"] = trace
        accepted = args.case == "task1" or trace["gpu0_max_gemms_per_ag_round"] >= 2
        attempt["status"] = "ok" if accepted else "insufficient_overlap"
        result = 0 if accepted else 3
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error, subprocess.CalledProcessError) as error:
        attempt.update(status="error", error=str(error))
        _LOG.error("%s M=%d, %d MiB. %s", args.case, args.m, args.size_mib, error)
        result = 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    record = {"case_id": args.case, "m": args.m, "attempt": attempt}
    (args.output_dir / "attempt.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return result


def _write_sweep_report(root: Path) -> None:
    cases = {}
    for path in root.glob("m*/task*/size-*mib/attempt.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        key = (record["m"], record["case_id"])
        cases.setdefault(key, []).append(record["attempt"])
    rows = [
        {"case_id": case, "m": m, "attempts": sorted(attempts, key=lambda attempt: attempt["size_mib"])}
        for (m, case), attempts in sorted(cases.items())
    ]
    baseline_means = {}
    for row in rows:
        attempt = row["attempts"][-1]
        if row["case_id"] == "task1" and attempt["status"] == "ok":
            baseline_means[row["m"]] = attempt["trace"]["gpu0_gemm_mean_us"]
    lines = [
        "# GB200 GEMM and CE AllGather kernel sweep",
        "",
        "GPU means use the first selected GPU. Each AG result includes GEMMs fully inside a single AG payload round.",
        "The reported delta pairs each accepted case with the same-shape GEMM-only baseline.",
        "",
        "| M | Case | Status | AG MiB/GPU | AG rounds | GEMMs in mean | Max per AG round | Mean µs | Δ vs solo |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        attempt = row["attempts"][-1]
        trace = attempt.get("trace", {})
        baseline = baseline_means.get(row["m"])
        solo = row["case_id"] == "task1"
        duration = trace.get("gpu0_gemm_mean_us") if attempt["status"] == "ok" else None
        delta = (duration / baseline - 1) * 100 if not solo and duration is not None and baseline else None
        row["gpu0_slowdown_percent"] = delta
        cells = [
            str(row["m"]),
            _CASE_NAMES[row["case_id"]],
            attempt["status"],
            str(attempt["size_mib"]) if not solo else "—",
            str(trace.get("comm_rounds_measured", "—")),
            str(trace.get("gpu0_covered_gemm_count" if not solo else "gpu0_gemm_count", "—")),
            str(trace.get("gpu0_max_gemms_per_ag_round", "—")) if not solo else "—",
            f"{duration:.3f}" if duration is not None else "—",
            f"{delta:+.1f}%" if delta is not None else "—",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    (root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Analyze a captured sweep attempt or regenerate its run summary."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    attempt = subparsers.add_parser("attempt", help="Validate and store one captured attempt")
    attempt.add_argument("--output-dir", type=Path, required=True)
    attempt.add_argument("--case", choices=tuple(_CASE_NAMES), required=True)
    attempt.add_argument("--m", type=int, required=True)
    attempt.add_argument("--size-mib", type=int, required=True)
    attempt.add_argument("--gemm-rounds", type=int, required=True)
    attempt.add_argument("--comm-rounds", type=int, required=True)
    attempt.add_argument("--nsys", required=True)
    attempt.add_argument("--launch-exit-code", type=int, default=0)
    report = subparsers.add_parser("report", help="Summarize all stored attempts")
    report.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "attempt":
        return _analyse_attempt(args)
    _write_sweep_report(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
