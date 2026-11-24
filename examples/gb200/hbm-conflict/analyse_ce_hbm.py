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
import html
import json
import logging
import math
import os
import re
import shutil
import sqlite3
import statistics
import subprocess
from collections import Counter, defaultdict
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path
from typing import NamedTuple

from analyse_hbm_conflict import Sample, export_sqlite, merge_intervals, overlap_ns, report_rank, window_metrics


LOGGER = logging.getLogger(__name__)
_RANGE = re.compile(r"(gemm|comm)(\d+)")
# Keep historical results readable. Older runs used task5 for host-source multicast
# and task6 for host-destination D2H. The worker's mode determines the analysis.
_CASE_DIR = re.compile(r"(?:task[1-8]|baseline[1-6])-.+")
_UNICAST_MODES = ("ag-hbm-unicast", "ag-host-unicast")
_MEMCPY_TRACK = re.compile(r"/Processes/(\d+)/CudaDevices/(\d+)/.*/Streams/\d+/Memory/Memcpy/")
_MEMCPY_BYTES = re.compile(r"\bmemcpy (\d+) bytes\b")
_MEMCPY_CHANNEL = re.compile(r"\bChannel type: ([^<]+)")
_REQUIRED_METRICS = {
    "DRAM read": re.compile(r"dram.*read|read.*dram", re.I),
    "DRAM write": re.compile(r"dram.*write|write.*dram", re.I),
    "NVLink RX": re.compile(r"^(?!.*(?:c2c|ctc)).*(?:nvlink.*(?:\brx\b|receiv)|\bnvlrx(?:\b|_))", re.I),
    "NVLink TX": re.compile(r"^(?!.*(?:c2c|ctc)).*(?:nvlink.*(?:\btx\b|transmit)|\bnvltx(?:\b|_))", re.I),
}
_C2C_RX = re.compile(r"(?:c2c|ctc).*?(?:rx|receiv|read).*?user", re.I)
_C2C_TX = re.compile(r"(?:c2c|ctc).*?(?:tx|transmit|write).*?user", re.I)
_D2H_COPY = re.compile(r"\b(?:dtoh|d2h|device[- ]to[- ]host)\b", re.I)
_H2D_COPY = re.compile(r"\b(?:htod|h2d|host[- ]to[- ]device)\b", re.I)
_P2P_COPY = re.compile(r"\b(?:ptop|p2p|peer[- ]to[- ]peer)\b", re.I)
_D2D_COPY = re.compile(r"\b(?:dtod|d2d|device[- ]to[- ]device)\b", re.I)
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
    pid: int | None = None
    src_device: int | None = None
    dst_device: int | None = None
    copy_count: int | None = None


class Capture(NamedTuple):
    """Rank labels, launch-attributed activities, and counters in one SQLite export."""

    worker_processes: dict[str, list[int]]
    activities: dict[str, dict[str, list[Activity]]]
    samples: dict[int, list[Sample]]


def _enum_name(values: dict[int, str], value: int | None) -> str:
    if value is None:
        return "unavailable"
    return f"{value} ({values[value]})" if value in values else str(value)


def _read_trace(sqlite_path: Path, *, labels: Sequence[str], physical_gpus: set[int] | None = None) -> Capture:
    with closing(sqlite3.connect(f"{sqlite_path.resolve().as_uri()}?mode=ro", uri=True)) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        worker_processes = defaultdict(list)
        for label, process in conn.execute(
            """SELECT COALESCE(n.text, s.value), n.globalTid >> 24 FROM NVTX_EVENTS n
               LEFT JOIN StringIds s ON s.id = n.textId
               WHERE n.end IS NOT NULL AND COALESCE(n.text, s.value) LIKE 'hbm-ce/%'"""
        ):
            worker_processes[label].append(process)

        api_tables = [
            f"CUPTI_ACTIVITY_KIND_{kind}" for kind in ("RUNTIME", "DRIVER") if f"CUPTI_ACTIVITY_KIND_{kind}" in tables
        ]
        if not api_tables:
            raise ValueError("capture has no CUDA runtime or driver API activity")
        api_calls = " UNION ALL ".join(f"SELECT start, globalTid, correlationId FROM {table}" for table in api_tables)
        activities = {label: defaultdict(list) for label in labels}
        label_params = ", ".join("?" for _ in labels)
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
                f"g.{col}" if col in columns else "NULL"
                for col in ("copyKind", "srcKind", "dstKind", "srcDeviceId", "dstDeviceId", "copyCount")
            )
            # Nsight's serialized global ID packs the OS PID into bits 24–47.
            # DumpTimeline uses the OS PID in the process path.
            query = f"""
                WITH api AS ({api_calls}),
                ranges AS (
                    SELECT n.start, n.end, n.globalTid, COALESCE(n.text, s.value) AS label
                    FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id = n.textId
                    WHERE n.end IS NOT NULL
                )
                SELECT DISTINCT c.label, n.label, g.start, g.end, g.deviceId, {name}, {bytes_copied},
                                g.correlationId, g.streamId, (g.globalPid >> 24) & 16777215, {kind_fields}
                FROM {table} g
                JOIN api a ON a.correlationId = g.correlationId
                    AND (a.globalTid >> 24) = (g.globalPid >> 24)
                JOIN ranges n ON n.globalTid = a.globalTid AND a.start >= n.start AND a.start < n.end
                JOIN ranges c ON (c.globalTid >> 24) = (n.globalTid >> 24)
                    AND c.label IN ({label_params}) AND n.start >= c.start AND n.end <= c.end
                {kernel_names}
                WHERE (n.label GLOB 'gemm[0-9]*' OR n.label GLOB 'comm[0-9]*')
                      AND g.end > g.start {payload_filter}
                ORDER BY g.start
            """
            for (
                worker,
                label,
                start,
                end,
                device,
                activity_name,
                size,
                correlation,
                stream,
                pid,
                copy,
                src,
                dst,
                src_device,
                dst_device,
                copy_count,
            ) in conn.execute(query, labels):
                if not _RANGE.fullmatch(label):
                    continue
                if is_copy:
                    key = (worker, label, start, end, device, size, correlation, stream, src_device, dst_device)
                    if key in seen_copies:
                        continue
                    seen_copies.add(key)
                activities[worker][label].append(
                    Activity(
                        start,
                        end,
                        device,
                        activity_name,
                        size,
                        _enum_name(copy_enum, copy),
                        _enum_name(memory_enum, src),
                        _enum_name(memory_enum, dst),
                        pid,
                        src_device,
                        dst_device,
                        copy_count,
                    )
                )

        samples = defaultdict(list)
        if {"GPU_METRICS", "TARGET_INFO_GPU_METRICS"} <= tables:
            query = """
                SELECT DISTINCT g.timestamp, g.typeId & 255, n.metricName, g.value
                FROM GPU_METRICS g
                JOIN TARGET_INFO_GPU_METRICS n ON (g.typeId & 255) = (n.typeId & 255) AND g.metricId = n.metricId
            """
            params = ()
            if physical_gpus is not None:
                # Combined reports can contain a long host rendezvous before the first GPU work.
                # Read only the requested devices and the time range used by the GPU metric windows.
                bounds = [
                    (min(op.start for op in group), max(op.end for op in group))
                    for worker in activities.values()
                    for group in worker.values()
                ]
                if not bounds or not physical_gpus:
                    return Capture(dict(worker_processes), activities, {})
                gpu_params = ", ".join("?" for _ in physical_gpus)
                query += f" WHERE g.timestamp BETWEEN ? AND ? AND (g.typeId & 255) IN ({gpu_params})"
                params = (min(start for start, _ in bounds), max(end for _, end in bounds), *sorted(physical_gpus))
            for timestamp, device, metric, value in conn.execute(query, params):
                if value is not None and math.isfinite(value):
                    samples[device].append(Sample(timestamp, metric, value))
        return Capture(dict(worker_processes), activities, dict(samples))


def _read_capture(sqlite_path: Path, *, case: str) -> tuple[dict[str, list[Activity]], dict[int, list[Sample]]]:
    label = f"hbm-ce/{case}"
    capture = _read_trace(sqlite_path, labels=(label,))
    if set(capture.worker_processes) != {label}:
        raise ValueError(f"expected NVTX case {label}, found {sorted(capture.worker_processes)}")
    return capture.activities[label], capture.samples


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


def _hardware_channels(document: dict) -> dict[tuple[int, int, int, int, int], set[str]]:
    channels = defaultdict(set)
    for track in document["GenericHierarchyRows"]:
        match = _MEMCPY_TRACK.search(track.get("Path", ""))
        if match is None:
            continue
        pid, device = map(int, match.groups())
        for row in track["Rows"]:
            for item in row["ViewDataElements"]:
                if item["Text"].endswith(" (destination)"):
                    continue
                tooltip = html.unescape(item["Tooltip"]).replace("\xa0", " ")
                size_match = _MEMCPY_BYTES.search(tooltip)
                if size_match is None:
                    continue
                start = item["Start"]
                end = start + item["Duration"]
                channel = _MEMCPY_CHANNEL.search(tooltip)
                channels[pid, device, start, end, int(size_match[1])].add(
                    channel[1].strip() if channel is not None else "unavailable"
                )
    return dict(channels)


def _read_hardware_channels(
    report: Path, *, nsys_bin: str, dump_timeline: str | None
) -> dict[tuple[int, int, int, int, int], set[str]]:
    nsys_path = Path(shutil.which(nsys_bin) or nsys_bin).resolve()
    sibling = nsys_path.with_name("DumpTimeline")
    tool = (
        dump_timeline or os.environ.get("DUMP_TIMELINE") or shutil.which(str(sibling)) or shutil.which("DumpTimeline")
    )
    if tool is None:
        raise ValueError("unicast hardware channel evidence requires --dump-timeline or DUMP_TIMELINE")
    result = subprocess.run(
        [tool, "--input", str(report), "--full", "--format=n,d", "--show-all", "--correlation=0"],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if result.returncode:
        raise ValueError(f"DumpTimeline failed with status {result.returncode}: {result.stderr.strip()}")
    return _hardware_channels(json.loads(result.stdout))


def _validate_unicast_trace(
    records: dict[int, dict],
    activities_by_rank: dict[int, dict[str, list[Activity]]],
    channels: dict[tuple[int, int, int, int, int], set[str]] | None,
    *,
    channel_devices: set[int] | None = None,
) -> tuple[dict[int, list[str]], dict[int, list[Activity]]]:
    errors = {rank: [] for rank in records}
    incoming = {rank: [] for rank in records}
    ranks_by_device = {}
    for rank in records:
        activities = activities_by_rank.get(rank, {})
        devices = {op.device for group in activities.values() for op in group}
        if len(devices) == 1:
            ranks_by_device[next(iter(devices))] = rank
    if len(ranks_by_device) != len(records):
        message = "unicast trace needs one distinct issuing CUDA device per rank"
        return {rank: [message] for rank in records}, incoming

    for device, rank in ranks_by_device.items():
        record = records[rank]
        host_source = record["mode"].endswith("ag-host-unicast")
        out_of_place = record.get("all_gather_out_of_place", False)
        count = record["world_size"]
        shard, remainder = divmod(record["size_bytes"], count)
        peers = set(ranks_by_device)
        if not host_source and not out_of_place:
            peers.remove(device)
        expected_bytes = shard * len(peers)
        if (
            remainder
            or shard <= 8
            or any(record.get(key) != expected_bytes for key in ("comm_source_bytes", "comm_receive_bytes"))
        ):
            errors[rank].append(
                f"invalid unicast byte contract, expected {expected_bytes} sent and received per round"
            )
        rounds = _iterations(activities_by_rank[rank], label_prefix="comm")
        if set(rounds) != set(range(record["comm_rounds_measured"])):
            errors[rank].append("unicast payload ranges do not match the measured communication rounds")
        for index, copies in sorted(rounds.items()):
            destinations = []
            matched_activities = set()
            for op in copies:
                key = (op.pid, op.device, op.start, op.end, op.size_bytes)
                observed_channels = channels.get(key, set()) if channels is not None else None
                if (
                    observed_channels is not None
                    and (channel_devices is None or device in channel_devices)
                    and observed_channels != {"Async Memcpy"}
                ):
                    errors[rank].append(
                        f"comm{index:02d}: payload hardware channel must be Async Memcpy, "
                        f"found {', '.join(sorted(observed_channels)) or 'no matching evidence'}"
                    )
                if channels is not None and key in matched_activities:
                    errors[rank].append(f"comm{index:02d}: payloads share the same hardware channel trace activity")
                matched_activities.add(key)
                is_h2d = bool(_H2D_COPY.search(op.copy_kind))
                is_p2p = bool(_P2P_COPY.search(op.copy_kind))
                is_self_copy = out_of_place and not host_source and bool(_D2D_COPY.search(op.copy_kind))
                destination = op.dst_device
                if destination is None and ((host_source and is_h2d) or is_self_copy):
                    destination = op.device
                destinations.append(destination)
                source_matches = (
                    "pinned" in op.src_kind.lower() and (is_h2d or is_p2p)
                    if host_source
                    else "device" in op.src_kind.lower()
                    and (
                        (is_p2p and op.src_device == device and destination != device)
                        or (is_self_copy and op.src_device in (None, device) and destination == device)
                    )
                )
                if (
                    not source_matches
                    or "device" not in op.dst_kind.lower()
                    or op.size_bytes != shard
                    or op.copy_count not in (None, 1)
                ):
                    errors[rank].append(
                        f"comm{index:02d}: expected one {shard} B "
                        f"{'pinned-host' if host_source else 'peer-device'} shard per copy to device memory"
                    )
                if destination in ranks_by_device:
                    incoming[ranks_by_device[destination]].append(op)
            if Counter(destinations) != Counter(peers):
                errors[rank].append(
                    f"comm{index:02d}: expected one payload per destination {sorted(peers)}, found {destinations}"
                )
    return errors, incoming


def _analyse_rank(
    record: dict,
    activities: dict[str, list[Activity]],
    samples: dict[int, list[Sample]],
    *,
    unicast_errors: Sequence[str] = (),
    incoming: Sequence[Activity] | None = None,
) -> dict:
    errors = list(unicast_errors)
    mode = record["mode"]
    unicast = mode.endswith(_UNICAST_MODES)
    host_destination = mode.endswith("ag-d2h")
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
    if mode != "gemm" and not comm_events:
        errors.append("no communication rounds measured")
    if record.get("comm_rounds_measured", len(comm_events)) != len(comm_events):
        errors.append("communication round count differs from event count")
    if mode.startswith("gemm") != (record["gemm_mnk"] is not None and bool(gemm_events)):
        errors.append("recorded GEMM shape or rounds do not match the selected mode")
    if record["output_correct"] is not True:
        errors.append("worker reported incorrect output")
    comm_kernels = sorted(
        {op.name for name, group in activities.items() if name.startswith("comm") for op in group if not op.size_bytes}
    )
    if comm_kernels:
        errors.append(f"communication launched kernels instead of a CE-only path: {comm_kernels}")
    source_bytes = record["size_bytes"]
    if "ag-" in mode and not host_destination:
        source_bytes //= record["world_size"]
    if unicast:
        source_bytes = record["comm_source_bytes"]
        copies_complete = copies_complete and not unicast_errors
    native_ag = record.get("nccl_version") is not None
    for index, group in sorted(copies.items()):
        total = sum(op.size_bytes for op in group)
        if unicast:
            continue  # The unicast trace checks the byte count at every destination.
        if host_destination and total != source_bytes:
            errors.append(f"comm{index:02d}: traced {total} copy bytes, expected {source_bytes} bytes of D2H fan-out")
            copies_complete = False
        elif native_ag and total != source_bytes:
            errors.append(
                f"comm{index:02d}: traced {total} copy bytes, expected one multicast shard of {source_bytes}"
            )
            copies_complete = False
        elif total < source_bytes:
            errors.append(f"comm{index:02d}: traced {total} copy bytes, expected at least {source_bytes} source bytes")
            copies_complete = False
    compute_ops = [op for group in gemm.values() for op in group]
    copy_ops = [op for group in copies.values() for op in group]
    if mode.endswith("ag-hbm") and any(re.search(r"host|pinned|pageable", op.src_kind, re.I) for op in copy_ops):
        errors.append("HBM-source AG has a payload copy with a CUPTI host source")
    if native_ag and mode.endswith("ag-host") and any("pinned" not in op.src_kind.lower() for op in copy_ops):
        errors.append("host-source AG has a payload copy without a CUPTI pinned source")
    if mode.endswith("h2d") and any(re.search(r"device|unified|managed", op.src_kind, re.I) for op in copy_ops):
        errors.append("H2D has a payload copy with a CUPTI non-host source")
    if host_destination and any(
        not _D2H_COPY.search(op.copy_kind)
        or "device" not in op.src_kind.lower()
        or "pinned" not in op.dst_kind.lower()
        for op in copy_ops
    ):
        errors.append("host-destination AG has a payload copy outside the CUPTI device-to-pinned-host path")
    devices = {op.device for op in compute_ops or copy_ops}
    device = next(iter(devices)) if len(devices) == 1 else None
    if device is None:
        errors.append(f"cannot identify a unique rank-local GPU from activity devices {sorted(devices)}")
    if device is not None and any(op.device != device for op in copy_ops):
        errors.append(f"copy activities include a device other than the rank-local GPU {device}")

    destination_ops = copy_ops if incoming is None else incoming
    comm_union = _busy(destination_ops)
    outgoing_union = _busy(copy_ops)
    if gemm_events and comm_events and gemm_complete and copies_complete:
        if not any(overlap_ns(start, end, comm_union) for start, end in _busy(compute_ops)):
            errors.append(
                "no GEMM kernel overlapped a traced local HBM write"
                if unicast
                else "no GEMM kernel overlapped a traced local payload copy"
            )
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
        outgoing_ns = sum(overlap_ns(start, end, outgoing_union) for start, end in gpu_intervals)
        gemms.append(
            {
                "index": index,
                "nsys_kernel_ms": gpu_ns / 1e6 if group else None,
                "nsys_kernel_tflops": 2 * m * n * k / gpu_ns / 1000 if group else None,
                "nsys_copy_coverage": covered_ns / gpu_ns if has_trace else None,
                **({"nsys_outgoing_copy_coverage": outgoing_ns / gpu_ns if has_trace else None} if unicast else {}),
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
    copy_window = [*copy_ops, *destination_ops] if unicast else copy_ops
    for window, ops in (("gemm_span", compute_ops), ("copy_span", copy_window)):
        if not ops or metric_device is None:
            continue
        start, end = min(op.start for op in ops), max(op.end for op in ops)
        counters = window_metrics(samples[metric_device], start, end)
        missing = [kind for kind, pattern in _REQUIRED_METRICS.items() if not any(pattern.search(n) for n in counters)]
        if missing:
            errors.append(f"{window}: no sampled {', '.join(missing)} counters on GPU {metric_device}")
        if window == "copy_span" and mode.endswith(("h2d", "ag-host", "ag-host-unicast")):
            if not any(_C2C_RX.search(name) and values["peak"] > 0 for name, values in counters.items()):
                errors.append(f"{window}: no positive C2C user RX samples for host-sourced copies")
        if window == "copy_span" and host_destination:
            if not any(_C2C_TX.search(name) and values["peak"] > 0 for name, values in counters.items()):
                errors.append(f"{window}: no positive C2C user TX samples for host-destination copies")
        requires_nvlink = ("ag-" in mode and not host_destination and not unicast) or mode.endswith("ag-hbm-unicast")
        if window == "copy_span" and requires_nvlink:
            for direction in ("NVLink RX", "NVLink TX"):
                if not any(
                    _REQUIRED_METRICS[direction].search(name) and values["peak"] > 0
                    for name, values in counters.items()
                ):
                    errors.append(
                        f"{window}: no positive {direction} samples for {'unicast' if unicast else 'multicast'} AG"
                    )
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
                **({"event_receive_GBps": record["comm_receive_bytes"] / (end - start)} if unicast else {}),
                "nsys_payload_copy_bytes": sum(op.size_bytes for op in copies.get(index, [])),
                "nsys_payload_copy_intervals_ns": _busy(copies.get(index, [])),
                "nsys_copy_kinds": sorted({(op.copy_kind, op.src_kind, op.dst_kind) for op in copies.get(index, [])}),
            }
            for index, (start, end) in enumerate(comm_events)
        ],
    }


def _read_combined(
    sqlite_path: Path, *, case: str, records: dict[int, dict], errors: list[str]
) -> dict[int, tuple[dict[str, list[Activity]], dict[int, list[Sample]]]]:
    labels = {rank: f"hbm-ce/{case}/rank-{rank}" for rank in records}
    gpu_counts = Counter(
        gpu for record in records.values() if type(gpu := record.get("physical_gpu")) is int and gpu >= 0
    )
    capture = _read_trace(sqlite_path, labels=tuple(labels.values()), physical_gpus=set(gpu_counts))
    unexpected_labels = set(capture.worker_processes) - set(labels.values())
    if unexpected_labels:
        errors.append(f"unexpected combined report NVTX labels {sorted(unexpected_labels)}")
    worker_counts = Counter(process for workers in capture.worker_processes.values() for process in workers)
    rank_captures = {}
    for rank, label in labels.items():
        workers = capture.worker_processes.get(label, [])
        if len(workers) != 1 or worker_counts[workers[0]] != 1:
            errors.append(f"rank {rank}: expected one distinct worker process in NVTX range {label}, found {workers}")
            continue
        gpu = records[rank].get("physical_gpu")
        if type(gpu) is not int or gpu_counts[gpu] != 1:
            errors.append(f"rank {rank}: expected a unique physical_gpu in worker events, found {gpu!r}")
            samples = {}
        else:
            samples = {gpu: capture.samples[gpu]} if gpu in capture.samples else {}
        rank_captures[rank] = capture.activities[label], samples
    return rank_captures


def _analyse_case(case_dir: Path, *, reports_dir: Path, nsys_bin: str, dump_timeline: str | None = None) -> dict:
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
    first = records[min(records)]
    unicast = first["mode"].endswith(_UNICAST_MODES)
    config_fields = _CONFIG_FIELDS
    if unicast:
        config_fields += ("comm_source_bytes", "comm_receive_bytes", "comm_rounds_measured")
    config = {key: first[key] for key in config_fields}
    config["nccl_version"] = records[min(records)].get("nccl_version")
    expected_ranks = set(range(config["world_size"]))
    if config["case"] != case or set(records) != expected_ranks:
        raise ValueError(
            f"inconsistent case or missing event ranks, expected {sorted(expected_ranks)}, got {sorted(records)}"
        )
    if any(any(record.get(key) != val for key, val in config.items()) for record in records.values()):
        raise ValueError("worker case configuration differs between ranks")

    combined_reports = [
        directory / f"{case}.nsys-rep"
        for directory in (case_dir / "nsys", reports_dir)
        if (directory / f"{case}.nsys-rep").is_file()
    ]
    paths = [*reports_dir.glob(f"{case}.rank_*.nsys-rep"), *(case_dir / "nsys").glob(f"{case}.rank_*.nsys-rep")]
    errors = []
    rank_captures = {}
    channels = {}
    capture_errors = (OSError, ValueError, subprocess.CalledProcessError, sqlite3.Error)
    if combined_reports:
        if len(combined_reports) != 1 or paths:
            raise ValueError("expected one combined nsys report or per-rank reports, found duplicate inputs")
        try:
            sqlite_path = export_sqlite(combined_reports[0], nsys_bin=nsys_bin)
            rank_captures = _read_combined(sqlite_path, case=case, records=records, errors=errors)
        except capture_errors as error:
            errors.append(f"combined capture unavailable or invalid: {error}")
        if unicast:
            try:
                channels = _read_hardware_channels(combined_reports[0], nsys_bin=nsys_bin, dump_timeline=dump_timeline)
            except (KeyError, OSError, TypeError, ValueError, subprocess.TimeoutExpired) as error:
                errors.append(f"unicast hardware channel evidence unavailable: {error}")
    else:
        if unicast:
            errors.append("unicast validation requires one combined Nsight Systems report")
        reports = {}
        for path in sorted(paths):
            rank = report_rank(path)
            if rank in reports:
                raise ValueError(f"duplicate nsys report for rank {rank}")
            reports[rank] = path
        if set(reports) != expected_ranks:
            errors.append(f"expected nsys reports for {sorted(expected_ranks)}, got {sorted(reports)}")
        for rank in sorted(expected_ranks & reports.keys()):
            try:
                rank_captures[rank] = _read_capture(export_sqlite(reports[rank], nsys_bin=nsys_bin), case=case)
            except capture_errors as error:
                errors.append(f"rank {rank}: capture unavailable or invalid: {error}")

    unicast_errors, incoming = {}, {}
    if unicast:
        activities_by_rank = {rank: capture[0] for rank, capture in rank_captures.items()}
        unicast_errors, incoming = _validate_unicast_trace(records, activities_by_rank, channels)
    ranks = [
        _analyse_rank(
            record,
            *rank_captures.get(rank, ({}, {})),
            unicast_errors=unicast_errors.get(rank, ()),
            incoming=incoming.get(rank),
        )
        for rank, record in sorted(records.items())
    ]
    return {"case": case, "config": config, "ranks": ranks, "errors": errors}


def _median(values: Sequence[float | None]) -> str:
    measured = [value for value in values if value is not None]
    return f"{statistics.median(measured):.3f}" if measured else "—"


def _render(cases: Sequence[dict], *, root: Path) -> str:
    lines = [
        "# GB200 GEMM and copy-engine overlap",
        "",
        "GEMM throughput is 2MNK divided by the union of launch-attributed GPU kernel intervals.",
        "GPU copy coverage is the fraction of those intervals overlapping traced payload copies.",
        "Multicast, H2D, and D2H coverage uses rank-local launches. GPU counters are device-wide.",
        "CUDA-event coverage uses the stream-ordered interval between the operation's start and end events.",
        "Source GB/s uses outgoing bytes divided by CUDA-event duration. Multicast sends one shard per rank.",
        "Direct D2H AllGather sends one shard per destination rank, including the local rank.",
        "Unicast AllGather receive GB/s uses traced destination bytes per rank normalized by its event duration.",
        "Unicast GPU copy coverage uses inbound HBM writes. Outbound coverage is available in summary.json.",
        "NCCL AG event durations include stream-ordered peer synchronization. Earlier direct multicast runs timed only copy.",
        "Receive payload proxy GB/s normalizes logical receive bytes by the rank's event duration.",
        "The logical receive proxy used for the other AllGather modes excludes the local shard.",
        "CUPTI copy bytes and source/destination enums in summary.json describe outgoing copies as traced.",
        "GPU metric CSV windows span the first to last traced kernel or payload copy and include any intervening gaps.",
        "Counter units and scopes follow Nsight Systems. DRAM bandwidth metrics indicate interface activity and NVLink",
        "throughput metrics are relative to peak. CTC metrics measure the CPU–GPU NVLink-C2C path.",
        "Raw samples are retained in their original units.",
        "",
        "| Case | Status | GPU GEMM TFLOP/s | Event GEMM TFLOP/s | GPU copy coverage | Event coverage "
        "| Source GB/s | Receive GB/s |",
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
        show_performance = not errors or not case.get("config", {}).get("mode", "").endswith(_UNICAST_MODES)
        if not show_performance:
            gemms, copies = [], []
        cells = [
            case["case"],
            status,
            _median([g["nsys_kernel_tflops"] for g in gemms]),
            _median([g["event_tflops"] for g in gemms]),
            _median([g["nsys_copy_coverage"] for g in gemms]),
            _median([g["event_copy_coverage"] for g in gemms]),
            _median([c["event_source_GBps"] for c in copies]),
            _median([c.get("event_receive_GBps", c["event_receive_payload_proxy_GBps"]) for c in copies]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
        failures.extend(f"- {case['case']} / {error}" for error in errors)
        means = defaultdict(list)
        for rank in case["ranks"] if show_performance else []:
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
    parser.add_argument("--dump-timeline", help="Nsight Systems DumpTimeline executable for unicast channel checks")
    args = parser.parse_args(argv)

    root = args.output_dir
    case_dirs = sorted(path for path in root.iterdir() if path.is_dir() and _CASE_DIR.fullmatch(path.name))
    if not case_dirs:
        parser.error(f"{root}: no task case directories")
    cases = []
    for path in case_dirs:
        try:
            cases.append(
                _analyse_case(path, reports_dir=root / "nsys", nsys_bin=args.nsys, dump_timeline=args.dump_timeline)
            )
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
