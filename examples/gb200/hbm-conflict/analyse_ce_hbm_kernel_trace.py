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

"""Read GEMM execution and same-round AllGather overlap from a CUDA trace."""

from __future__ import annotations

import json
import sqlite3
import statistics
import subprocess
import tempfile
from pathlib import Path

from analyse_ce_hbm import Activity, _event_intervals, _iterations, _read_combined
from analyse_hbm_conflict import merge_intervals, overlap_ns


_AG_MODES = ("gemm-ag-hbm", "gemm-ag-host", "gemm-ag-hbm-unicast", "gemm-ag-host-unicast")
_TRACE_TABLES = "^CUPTI_ACTIVITY_KIND_.*$,^NVTX_EVENTS$,^StringIds$,^ENUM_CUDA_MEM.*$"


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
