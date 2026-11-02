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
"""Read the single nsys report of a ``run_p2p_conflict_benchmark.sh`` run: per-flow kernels and NVLink counters.

The report holds all four ranks and the GPU metrics of all four GPUs. This script exports it with
``nsys export --type sqlite`` and

* maps every ``ncclDevKernel_SendRecv`` / gate ``AllReduce`` kernel to the NVTX range its launch
  call started in (same process): ``<scenario>/trial<k>/flow <src>><dst> <send|recv>`` or
  ``<scenario>/trial<k>/gate``. ``gridX`` of a SendRecv kernel is its CTA count (one per NCCL
  channel), the measurement NCCL does not log for P2P;
* times each flow end from the end of the gate kernel on the same GPU (start offset, duration,
  end offset), and checks the kernel ran on the GPU the flow's role was assigned to;
* averages every NVLink GPU metric per GPU over each trial window, from the first flow kernel
  start to the last flow kernel end on any GPU. GPU metric rows belong to GPU ``typeId & 0xFF``
  (as in nsys's own ``nvlink_sum`` recipe) and are percent of peak, summed over all NVLink links.

Outputs in ``--output-dir``: ``nsys_flow_kernels.csv`` (one row per kernel), ``nvlink_windows.csv``
(one row per trial, GPU and metric), ``nvlink_samples.csv`` (raw samples inside the trial windows,
for plotting) and ``nsys_summary.md``.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import logging
import re
import sqlite3
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from statistics import mean, median


LOGGER = logging.getLogger(__name__)
REPORT_NAME = "p2p-conflict"
BYTES_PER_MIB = 1024 * 1024
FLOW_RANGE = re.compile(
    r"^(?P<scenario>[^/]+)/trial(?P<trial>\d+)/flow (?P<src>\d+)>(?P<dst>\d+) (?P<side>send|recv)$"
)
GATE_RANGE = re.compile(r"^(?P<scenario>[^/]+)/trial(?P<trial>\d+)/gate$")
# Name suffix newer nsys versions append to the metric names (see nsys_recipe nvlink_sum).
METRIC_SUFFIX = " [Throughput %]"
NVLINK_METRICS = (
    "NVLink TX Requests User Data",
    "NVLink TX Requests Protocol Data",
    "NVLink TX Responses User Data",
    "NVLink TX Responses Protocol Data",
    "NVLink RX Requests User Data",
    "NVLink RX Requests Protocol Data",
    "NVLink RX Responses User Data",
    "NVLink RX Responses Protocol Data",
)

RANGES_SQL = """
SELECT n.start, n.end, (n.globalTid >> 24) & 16777215 AS pid, COALESCE(n.text, s.value) AS name
FROM NVTX_EVENTS AS n
LEFT JOIN StringIds AS s ON s.id = n.textId
WHERE n.end IS NOT NULL
"""
KERNELS_SQL = """
SELECT
    r.start AS launch_start,
    (r.globalTid >> 24) & 16777215 AS pid,
    k.start, k.end, k.deviceId, k.streamId, k.gridX, k.blockX, s.value AS kernel
FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
JOIN CUPTI_ACTIVITY_KIND_RUNTIME AS r
    ON r.correlationId = k.correlationId
    AND ((r.globalTid >> 24) & 16777215) = ((k.globalPid >> 24) & 16777215)
JOIN StringIds AS s ON s.id = k.shortName
WHERE s.value LIKE 'ncclDevKernel_SendRecv%' OR s.value LIKE 'nccl%AllReduce%'
"""
METRIC_NAMES_SQL = "SELECT DISTINCT typeId, metricId, metricName FROM TARGET_INFO_GPU_METRICS"
GPU_INFO_SQL = "SELECT id, name, busLocation FROM TARGET_INFO_GPU"
WINDOW_SQL = """
SELECT typeId & 255 AS gpu, metricId, AVG(value), COUNT(*)
FROM GPU_METRICS
WHERE timestamp BETWEEN ? AND ?
GROUP BY gpu, metricId
"""
SAMPLES_SQL = "SELECT timestamp, typeId & 255 AS gpu, metricId, value FROM GPU_METRICS WHERE timestamp BETWEEN ? AND ?"


@dataclass(frozen=True)
class Range:
    """One NVTX push/pop range of one process."""

    start: int
    end: int
    pid: int
    name: str


@dataclass(frozen=True)
class Kernel:
    """One NCCL kernel with the host time its launch call started."""

    launch_start: int
    pid: int
    start: int
    end: int
    device_id: int
    stream_id: int
    grid_x: int
    block_x: int
    name: str


@dataclass(frozen=True)
class FlowKernel:
    """A SendRecv kernel attributed to one end of one flow in one trial; times in us from the gate kernel end."""

    scenario: str
    trial: int
    flow: str
    side: str
    role: str
    gpu: int
    expected_gpu: int
    grid_x: int
    block_x: int
    start_us: float
    duration_us: float
    end_us: float
    start_ns: int
    end_ns: int


@dataclass(frozen=True)
class WindowMetric:
    """Mean of one NVLink metric on one GPU over one trial window (percent of peak)."""

    scenario: str
    trial: int
    gpu: int
    metric: str
    mean_pct: float
    samples: int
    window_us: float


def metric_name(raw: str) -> str:
    """Metric name without the ``[Throughput %]`` suffix newer nsys versions add."""
    return raw[: -len(METRIC_SUFFIX)] if raw.endswith(METRIC_SUFFIX) else raw


def attribute(ranges: list[Range], kernels: list[Kernel]) -> tuple[dict[tuple[str, int, str, str], Kernel], dict]:
    """Map kernels to flow ends and gates by the NVTX range (same process) their launch call started in.

    Returns ``{(scenario, trial, flow, side): kernel}`` for SendRecv kernels and
    ``{(scenario, trial, gpu): kernel}`` for gate AllReduce kernels. A flow end with more than one
    kernel is an error: each end is one isend/irecv, i.e. one launch.
    """
    by_pid: dict[int, list[Range]] = defaultdict(list)
    for item in ranges:
        if FLOW_RANGE.match(item.name) or GATE_RANGE.match(item.name):
            by_pid[item.pid].append(item)
    for items in by_pid.values():
        items.sort(key=lambda r: r.start)
    starts = {pid: [r.start for r in items] for pid, items in by_pid.items()}

    flows: dict[tuple[str, int, str, str], Kernel] = {}
    gates: dict[tuple[str, int, int], Kernel] = {}
    for kernel in kernels:
        items = by_pid.get(kernel.pid, [])
        i = bisect.bisect_right(starts.get(kernel.pid, []), kernel.launch_start) - 1
        if i < 0 or kernel.launch_start > items[i].end:
            continue
        name = items[i].name
        is_send_recv = "SendRecv" in kernel.name
        match = FLOW_RANGE.match(name)
        if match and is_send_recv:
            key = (match["scenario"], int(match["trial"]), f"{match['src']}>{match['dst']}", match["side"])
            if key in flows:
                raise RuntimeError(f"two SendRecv kernels launched in NVTX range {name!r}")
            flows[key] = kernel
            continue
        match = GATE_RANGE.match(name)
        if match and not is_send_recv:
            gates[(match["scenario"], int(match["trial"]), kernel.device_id)] = kernel
    return flows, gates


def flow_kernels(
    flows: dict[tuple[str, int, str, str], Kernel], gates: dict[tuple[str, int, int], Kernel], *, gpus: list[int]
) -> list[FlowKernel]:
    """One :class:`FlowKernel` per attributed kernel, timed from the gate kernel on the same GPU."""
    rows = []
    for (scenario, trial, flow, side), kernel in sorted(flows.items()):
        src, dst = (int(x) for x in flow.split(">"))
        role = src if side == "send" else dst
        gate = gates.get((scenario, trial, kernel.device_id))
        if gate is None:
            raise RuntimeError(f"{scenario} trial {trial}: no gate kernel on GPU{kernel.device_id}")
        rows.append(
            FlowKernel(
                scenario=scenario,
                trial=trial,
                flow=flow,
                side=side,
                role=f"G{role}",
                gpu=kernel.device_id,
                expected_gpu=gpus[role],
                grid_x=kernel.grid_x,
                block_x=kernel.block_x,
                start_us=(kernel.start - gate.end) / 1e3,
                duration_us=(kernel.end - kernel.start) / 1e3,
                end_us=(kernel.end - gate.end) / 1e3,
                start_ns=kernel.start,
                end_ns=kernel.end,
            )
        )
    return rows


def check_coverage(rows: list[FlowKernel], scenarios: dict[str, list[str]], *, trials: int) -> list[str]:
    """Problems with the attribution: missing flow ends (capture window too small) or kernels on the wrong GPU."""
    seen = {(r.scenario, r.trial, r.flow, r.side) for r in rows}
    problems = []
    for scenario, flows in scenarios.items():
        for trial in range(trials):
            for flow in flows:
                for side in ("send", "recv"):
                    if (scenario, trial, flow, side) not in seen:
                        problems.append(f"{scenario} trial {trial}: no kernel for flow {flow} {side}")
    problems += [
        f"{r.scenario} trial {r.trial} flow {r.flow} {r.side}: ran on GPU{r.gpu}, {r.role} is GPU{r.expected_gpu}"
        for r in rows
        if r.gpu != r.expected_gpu
    ]
    return problems


def trial_windows(rows: list[FlowKernel]) -> dict[tuple[str, int], tuple[int, int]]:
    """``{(scenario, trial): (first flow kernel start, last flow kernel end)}`` over all GPUs, in ns."""
    windows: dict[tuple[str, int], tuple[int, int]] = {}
    for r in rows:
        key = (r.scenario, r.trial)
        lo, hi = windows.get(key, (r.start_ns, r.end_ns))
        windows[key] = (min(lo, r.start_ns), max(hi, r.end_ns))
    return windows


def summarize_flows(rows: list[FlowKernel], *, size_bytes: int) -> list[dict[str, object]]:
    """Per (scenario, flow, side): medians over trials, kernel-time GB/s and the CTA counts seen."""
    groups: dict[tuple[str, str, str], list[FlowKernel]] = defaultdict(list)
    for r in rows:
        groups[(r.scenario, r.flow, r.side)].append(r)
    out = []
    for (scenario, flow, side), items in groups.items():
        duration = median(r.duration_us for r in items)
        out.append(
            {
                "scenario": scenario,
                "flow": flow,
                "side": side,
                "gpu": items[0].gpu,
                "trials": len(items),
                "grid_x": sorted({r.grid_x for r in items}),
                "block_x": sorted({r.block_x for r in items}),
                "start_us": median(r.start_us for r in items),
                "duration_us": duration,
                "end_us": median(r.end_us for r in items),
                "kernel_gbps": size_bytes / (duration * 1e-6) / 1e9,
            }
        )
    return out


def summarize_metrics(windows: list[WindowMetric]) -> dict[tuple[str, int], dict[str, float]]:
    """``{(scenario, gpu): {metric: mean over trials of the window means}}`` for the NVLink metrics."""
    groups: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for w in windows:
        groups[(w.scenario, w.gpu, w.metric)].append(w.mean_pct)
    out: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for (scenario, gpu, metric), values in groups.items():
        out[(scenario, gpu)][metric] = mean(values)
    return out


def render(
    flows: list[dict[str, object]],
    metrics: dict[tuple[str, int], dict[str, float]],
    *,
    scenarios: list[str],
    gpus: list[int],
    gpu_info: list[tuple[int, str, str]],
    problems: list[str],
) -> str:
    """``nsys_summary.md``."""
    order = {name: i for i, name in enumerate(scenarios)}
    lines = [
        "# P2P conflict: kernels and NVLink counters from the single nsys report",
        "",
        "Roles: " + ", ".join(f"G{r} = GPU{g}" for r, g in enumerate(gpus)) + ".",
        "Kernel times are in us from the end of the gate AllReduce kernel on the same GPU (medians over trials).",
        "kernel GB/s = flow bytes / SendRecv kernel duration (no launch latency; 1e9 bytes/s).",
        "",
        "## Flow kernels",
        "",
        "| scenario | flow | side | GPU | CTAs (gridX) | threads/CTA | start us | duration us | end us | kernel GB/s |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for f in sorted(flows, key=lambda f: (order.get(f["scenario"], 99), f["flow"] != "0>3", f["flow"], f["side"])):
        lines.append(
            f"| {f['scenario']} | G{f['flow']} | {f['side']} | GPU{f['gpu']} | {','.join(map(str, f['grid_x']))} | "
            f"{','.join(map(str, f['block_x']))} | {f['start_us']:.1f} | {f['duration_us']:.1f} | "
            f"{f['end_us']:.1f} | {f['kernel_gbps']:.1f} |"
        )
    short = [m.replace("NVLink ", "").replace(" Data", "") for m in NVLINK_METRICS]
    lines += [
        "",
        "## NVLink counters per GPU (mean over the trial windows, percent of peak, all links)",
        "",
        "Push (P2P write): the sender's data leaves as TX Requests User Data and arrives as RX Requests",
        "User Data; the write acknowledgements travel back as Responses. Pull (P2P read): the reader sends",
        "requests and the data comes back as Responses User Data.",
        "",
        "| scenario | GPU | " + " | ".join(short) + " |",
        "| --- | --- | " + " | ".join("---" for _ in short) + " |",
    ]
    for (scenario, gpu), values in sorted(metrics.items(), key=lambda kv: (order.get(kv[0][0], 99), kv[0][1])):
        role = f"G{gpus.index(gpu)}" if gpu in gpus else "-"
        cells = " | ".join(f"{values[m]:.1f}" if m in values else "-" for m in NVLINK_METRICS)
        lines.append(f"| {scenario} | GPU{gpu} ({role}) | {cells} |")
    lines += ["", "## GPUs in the report (TARGET_INFO_GPU)", ""]
    lines += [f"- id {gid}: {name} at {bus}" for gid, name, bus in gpu_info] or ["- (none recorded)"]
    lines += ["", "## Checks", ""]
    lines += [f"- {p}" for p in problems] or ["- every flow end of every trial has one kernel on its role's GPU"]
    lines.append("")
    return "\n".join(lines)


def _write_csv(path: Path, row_type: type, rows: list[object]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[f.name for f in fields(row_type)])
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)  # type: ignore[call-overload]


def export_sqlite(report: Path, *, nsys_bin: str) -> Path:
    """``nsys export`` the report next to it, unless an export newer than the report exists."""
    target = report.with_suffix(".sqlite")
    if target.exists() and target.stat().st_mtime >= report.stat().st_mtime:
        return target
    subprocess.run(
        [nsys_bin, "export", "--type=sqlite", "--force-overwrite=true", f"--output={target}", str(report)],
        check=True,
    )
    return target


def analyse(output_dir: Path, *, nsys_bin: str) -> Path:
    """Analyse the run in ``output_dir``; returns the path of ``nsys_summary.md``."""
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    gpus: list[int] = metadata["gpus"]
    scenarios: dict[str, list[str]] = metadata["scenarios"]
    trials = int(metadata["args"]["trials"])
    size_bytes = int(metadata["args"]["size_mib"]) * BYTES_PER_MIB
    report = output_dir / "nsys" / f"{REPORT_NAME}.nsys-rep"
    if not report.exists():
        raise FileNotFoundError(f"{report} is missing; was the run profiled?")
    connection = sqlite3.connect(export_sqlite(report, nsys_bin=nsys_bin))
    try:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        ranges = [Range(*row) for row in connection.execute(RANGES_SQL)]
        kernels = [Kernel(*row) for row in connection.execute(KERNELS_SQL)]
        flows, gates = attribute(ranges, kernels)
        rows = flow_kernels(flows, gates, gpus=gpus)
        problems = check_coverage(rows, scenarios, trials=trials)
        windows: list[WindowMetric] = []
        samples: list[tuple[object, ...]] = []
        gpu_info: list[tuple[int, str, str]] = []
        if {"GPU_METRICS", "TARGET_INFO_GPU_METRICS"} <= tables:
            names = {
                (type_id & 255, metric_id): metric_name(raw)
                for type_id, metric_id, raw in connection.execute(METRIC_NAMES_SQL)
            }
            for (scenario, trial), (lo, hi) in sorted(trial_windows(rows).items()):
                for gpu, metric_id, value, count in connection.execute(WINDOW_SQL, (lo, hi)):
                    name = names.get((gpu, metric_id), "")
                    if name in NVLINK_METRICS:
                        windows.append(WindowMetric(scenario, trial, gpu, name, value, count, (hi - lo) / 1e3))
                for timestamp, gpu, metric_id, value in connection.execute(SAMPLES_SQL, (lo, hi)):
                    name = names.get((gpu, metric_id), "")
                    if name in NVLINK_METRICS:
                        samples.append((scenario, trial, gpu, name, (timestamp - lo) / 1e3, value))
            if not windows:
                problems.append("the GPU metric set recorded no NVLink metric; pick one with --nsys-gpu-metrics-set")
        else:
            problems.append("the report has no GPU metrics (nsys ran without --gpu-metrics-devices)")
        if "TARGET_INFO_GPU" in tables:
            gpu_info = list(connection.execute(GPU_INFO_SQL))
    finally:
        connection.close()

    _write_csv(output_dir / "nsys_flow_kernels.csv", FlowKernel, rows)  # type: ignore[arg-type]
    _write_csv(output_dir / "nvlink_windows.csv", WindowMetric, windows)  # type: ignore[arg-type]
    with (output_dir / "nvlink_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scenario", "trial", "gpu", "metric", "t_us", "value_pct"])
        writer.writerows(samples)
    summary = output_dir / "nsys_summary.md"
    summary.write_text(
        render(
            summarize_flows(rows, size_bytes=size_bytes),
            summarize_metrics(windows),
            scenarios=list(scenarios),
            gpus=gpus,
            gpu_info=gpu_info,
            problems=problems,
        ),
        encoding="utf-8",
    )
    for problem in problems:
        LOGGER.warning("%s", problem)
    return summary


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True, help="Run directory of the benchmark")
    parser.add_argument("--nsys", default="nsys", help="nsys binary used for the sqlite export")
    args = parser.parse_args(argv)
    LOGGER.info("summary: %s", analyse(args.output_dir, nsys_bin=args.nsys))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
