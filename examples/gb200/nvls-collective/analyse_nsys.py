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
"""File the ``--nsys`` reports under their cases, then query AG / RS / AR kernel cost with SQL.

nsys writes one report per rank per case as ``<algo>/nsys/rank<R>.<K>.nsys-rep``, where ``K`` is
the rank's ``K``-th ``cudaProfilerStart``/``Stop`` window. Every rank opens one window per case
in the same order, and rank 0 lists them in ``<algo>/nsys/index.tsv`` (``capture_index`` -> NVTX
label ``<algo>/<case>``), so report ``K`` of every rank is moved to
``<algo>/cases/<algo>-<case>/<algo>-<case>.rank_<RR>.nsys-rep``, next to that case's NCCL logs,
without exporting it. ``--file-only`` stops there; the launcher runs it on node 0 after each pass.

The analysis exports every filed report with ``nsys export --type sqlite``, checks that its outer
NVTX range is the case it was filed under, runs the queries in ``nsys_collective_kernels.sql`` and
joins the result back to ``<algo>/results.jsonl`` through the case's NVTX label and the rank's
group. A rank outside every group of a case launches no kernel inside its window, so its export
has no kernel tables and contributes nothing. Only the Python standard library is used::

    python analyse_nsys.py --output-dir <dir>                         # file what is left, analyse
    python analyse_nsys.py --output-dir <dir> --algo ring --file-only

Outputs ``<algo>/nsys/kernels.csv`` (one row per report, trial, op and kernel) and
``<output-dir>/nsys_kernels.md`` (per case and group: median over trials of the slowest rank).
Reports recorded with GPU metrics (``--gpu-metrics-devices``, which the launcher's ``--nsys`` turns
on) also give ``<algo>/nsys/gpu_metrics.csv`` (one row per report, trial and metric: the metric's
mean over the trial's collective kernels) and ``<output-dir>/nsys_gpu_metrics.md`` (NVLink, DRAM
and C2C metrics per case and group: mean / min / max over the group's ranks, median over trials).
Running it again reuses the filed reports and their exports. One pass that fails to analyse does
not stop the others; the script exits non-zero after writing what it could.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sqlite3
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))  # the sibling module, also when this file is loaded by path

from collective_metrics import ALGORITHMS, case_dir_name, rank_file, workload_ops  # noqa: E402


LOGGER = logging.getLogger(__name__)
SQL_PATH = HERE / "nsys_collective_kernels.sql"
REQUIRED_COLUMNS = {
    "StringIds": {"id", "value"},
    "NVTX_EVENTS": {"start", "end", "text", "textId", "globalTid"},
    "CUPTI_ACTIVITY_KIND_RUNTIME": {"start", "globalTid", "correlationId"},
    "CUPTI_ACTIVITY_KIND_KERNEL": {"start", "end", "deviceId", "streamId", "correlationId", "globalPid", "shortName"},
    "GPU_METRICS": {"timestamp", "metricId", "value"},
    "TARGET_INFO_GPU_METRICS": {"metricId", "metricName"},
}
# nsys only creates these when the window recorded CUDA activity. A rank outside every group of
# a case waits in the Gloo barrier and synchronizes, so it can have neither.
KERNEL_TABLES = {"CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_KERNEL"}
# Only in reports recorded with --gpu-metrics-devices (not in runs from before the launcher set it).
GPU_METRIC_TABLES = {"GPU_METRICS", "TARGET_INFO_GPU_METRICS"}
# GPU metrics reported in nsys_gpu_metrics.md (gpu_metrics.csv keeps every metric of the set):
# "NVLink bytes received/transmitted", "DRAM Read/Write Bandwidth", C2C link throughput.
REPORTED_METRICS = re.compile(r"nvl|dram|c2c", re.IGNORECASE)
_RAW_REPORT = re.compile(r"^rank(\d+)\.(\d+)\.nsys-rep$")
_FILED_RANK = re.compile(r"\.rank_(\d+)$")
QUERY_NAMES = {"kernels", "overlap", "label", "gpu_metrics"}
NS_PER_MS = 1e6


@dataclass(frozen=True)
class KernelRow:
    """One (trial, op, kernel) aggregate from one rank's report."""

    algo: str
    rank: int
    report: str
    label: str
    trial: str
    op: str
    kernel: str
    device_id: int
    kernels: int
    busy_ns: int
    min_ns: int
    max_ns: int
    first_start_ns: int
    last_end_ns: int
    streams: int


@dataclass(frozen=True)
class MetricRow:
    """One GPU metric averaged over one trial's collective kernels, from one rank's report."""

    algo: str
    rank: int
    report: str
    label: str
    trial: str
    metric: str
    samples: int
    mean: float
    peak: float
    window_ns: int


def load_queries(path: Path = SQL_PATH) -> dict[str, str]:
    """Split the SQL file on ``-- name: <query>`` markers."""
    queries = {}
    for block in path.read_text(encoding="utf-8").split("-- name: ")[1:]:
        name, body = block.split("\n", 1)
        queries[name.strip()] = body.strip()
    if set(queries) != QUERY_NAMES:
        raise ValueError(f"{path} must define exactly the {sorted(QUERY_NAMES)} queries, got {sorted(queries)}")
    return queries


def raw_capture(path: Path) -> tuple[int, int]:
    """(rank, capture index) of a report named by the launcher's ``--output=.../rank%q{RANK}``.

    With ``--capture-range-end=repeat`` nsys appends ``.<K>`` for the ``K``-th capture, from 1.
    """
    match = _RAW_REPORT.match(path.name)
    if match is None:
        raise ValueError(f"cannot read a rank and capture index from report name {path.name!r}")
    return int(match.group(1)), int(match.group(2))


def filed_rank(path: Path) -> int:
    """Rank of a report already filed as ``<algo>-<case>.rank_<RR>.nsys-rep``."""
    match = _FILED_RANK.search(path.stem)
    if match is None:
        raise ValueError(f"cannot read a rank from filed report {path.name!r}")
    return int(match.group(1))


def filed_label(algo: str, report: Path) -> str:
    """The NVTX label ``<algo>/<case>`` a report filed under ``cases/<algo>-<case>/`` must carry."""
    prefix = f"{algo}-"
    if not report.parent.name.startswith(prefix):
        raise ValueError(f"{report}: case directory does not start with {prefix!r}")
    return f"{algo}/{report.parent.name[len(prefix) :]}"


def read_index(nsys_dir: Path) -> list[str]:
    """Case labels in capture order: element ``K - 1`` is the label of capture ``K``."""
    with (nsys_dir / "index.tsv").open(encoding="utf-8", newline="") as handle:
        rows = sorted((int(row["capture_index"]), row["label"]) for row in csv.DictReader(handle, delimiter="\t"))
    if [index for index, _ in rows] != list(range(1, len(rows) + 1)):
        raise RuntimeError(f"{nsys_dir / 'index.tsv'}: capture indices are not 1..{len(rows)}")
    return [label for _, label in rows]


def wait_for_reports(nsys_dir: Path, *, expected: int, timeout_s: float, poll_s: float = 5.0) -> list[Path]:
    """Wait until ``expected`` reports exist and their sizes stop changing.

    Each node's nsys sessions finish before that node's torchrun returns, but node 0 can get here
    while other nodes are still finalizing their reports on the shared file system.
    """
    deadline = time.monotonic() + timeout_s
    previous: dict[Path, int] = {}
    while True:
        reports = sorted(nsys_dir.glob("rank*.nsys-rep"))
        sizes = {path: path.stat().st_size for path in reports}
        if len(reports) >= expected and sizes == previous:
            return reports
        if time.monotonic() > deadline:
            LOGGER.warning(
                "%s: %d of %d reports after %.0fs; filing what exists", nsys_dir, len(reports), expected, timeout_s
            )
            return reports
        previous = sizes
        time.sleep(poll_s)


def export_sqlite(report: Path, *, nsys_bin: str) -> Path:
    """Export one report next to itself, reusing an export that is newer than the report."""
    target = report.with_suffix(".sqlite")
    if target.exists() and target.stat().st_mtime >= report.stat().st_mtime:
        return target
    subprocess.run(
        [nsys_bin, "export", "--type=sqlite", "--force-overwrite=true", f"--output={target}", str(report)],
        check=True,
    )
    return target


def check_schema(connection: sqlite3.Connection, path: Path) -> set[str]:
    """Fail with the missing table/column instead of an opaque SQL error; return the tables present."""
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for table, columns in REQUIRED_COLUMNS.items():
        if table not in tables:
            if table in KERNEL_TABLES | GPU_METRIC_TABLES:
                continue
            raise RuntimeError(f"{path}: table {table} is missing; was the run profiled with --trace=cuda,nvtx?")
        present = {row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')}
        if columns - present:
            raise RuntimeError(
                f"{path}: {table} lacks columns {sorted(columns - present)} (nsys export schema changed)"
            )
    return tables


def file_reports(algo_dir: Path, reports: Sequence[Path], labels: Sequence[str], *, world_size: int) -> list[Path]:
    """Move raw ``nsys/rank<R>.<K>`` reports (and any export of them) into their case directories."""
    algo = algo_dir.name
    filed = []
    for report in reports:
        rank, capture = raw_capture(report)
        if not 1 <= capture <= len(labels):
            raise RuntimeError(f"{report}: capture {capture} is not in nsys/index.tsv ({len(labels)} captures)")
        label_algo, _, case = labels[capture - 1].partition("/")
        if label_algo != algo or not case or "/" in case:
            raise RuntimeError(f"{report}: NVTX label {labels[capture - 1]} does not name a {algo} case")
        name = case_dir_name(algo, case)
        target = (
            algo_dir / "cases" / name / rank_file(algo, case, rank=rank, world_size=world_size, suffix=".nsys-rep")
        )
        if target.exists():
            raise FileExistsError(f"{report} and {target} are both rank {rank} of {name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        export = report.with_suffix(".sqlite")
        if export.exists():
            # rename keeps mtimes, so export_sqlite still recognizes the moved export as current.
            export.rename(target.with_suffix(".sqlite"))
        report.rename(target)
        filed.append(target)
    return filed


def query_report(
    sqlite_path: Path, queries: dict[str, str], *, algo: str, rank: int, label: str
) -> tuple[list[KernelRow], dict[tuple[str, str], int]]:
    """Check the report's case label, then run both queries on it.

    A report without kernel tables (a rank that sat the case out) yields no rows.
    """
    connection = sqlite3.connect(sqlite_path)
    try:
        tables = check_schema(connection, sqlite_path)
        labels = [row[0] for row in connection.execute(queries["label"])]
        if labels != [label]:
            raise RuntimeError(f"{sqlite_path}: filed as {label} but its case NVTX ranges are {labels}")
        if not KERNEL_TABLES <= tables:
            return [], {}
        kernels = [
            KernelRow(algo, rank, sqlite_path.stem, *row) for row in connection.execute(queries["kernels"]).fetchall()
        ]
        overlap = {(lbl, trial): int(ns) for lbl, trial, ns in connection.execute(queries["overlap"])}
    finally:
        connection.close()
    return kernels, overlap


def query_gpu_metrics(sqlite_path: Path, queries: dict[str, str], *, algo: str, rank: int) -> list[MetricRow]:
    """Per-trial GPU metric means of one report; empty without kernels or without GPU metrics."""
    connection = sqlite3.connect(sqlite_path)
    try:
        tables = check_schema(connection, sqlite_path)
        if not (KERNEL_TABLES | GPU_METRIC_TABLES) <= tables:
            return []
        return [MetricRow(algo, rank, sqlite_path.stem, *row) for row in connection.execute(queries["gpu_metrics"])]
    finally:
        connection.close()


def _rank_records(results: Sequence[dict[str, object]]) -> dict[tuple[str, int], dict[str, object]]:
    """Map (nsys label, rank) to the results record of the group that rank ran in."""
    by_rank = {}
    for record in results:
        label = record.get("nsys_label")
        if label:
            for rank in record["ranks"]:  # type: ignore[union-attr]
                by_rank[(str(label), int(rank))] = record
    return by_rank


def summarize(
    kernels: Sequence[KernelRow],
    overlap: dict[tuple[int, str, str], int],
    results: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Per case and group: median over trials of the slowest rank's per-iteration kernel times.

    Only the ops of the case's own workload are counted. That drops start-gate AllReduce kernels
    from AG/RS cases of reports whose trial range still contains the gate (the C++ benchmark).
    """
    records = _rank_records(results)
    per_trial: dict[tuple[str, str, str, str], dict[int, dict[str, float]]] = defaultdict(dict)
    names: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in kernels:
        record = records.get((row.label, row.rank))
        if record is None:
            continue  # idle rank of a split case, or a label with no results row
        if row.op not in workload_ops(str(record["workload"])):
            continue  # e.g. the start gate's AllReduce inside an AG/RS trial
        group = str(record["group_name"])
        sample = per_trial[(row.algo, row.label, group, row.trial)].setdefault(
            row.rank, {"ag": 0.0, "rs": 0.0, "ar": 0.0, "first": float("inf"), "last": 0.0}
        )
        sample[row.op] += row.busy_ns
        sample["first"] = min(sample["first"], row.first_start_ns)
        sample["last"] = max(sample["last"], row.last_end_ns)
        names[(row.algo, row.label, group)].add(row.kernel)

    cases: dict[tuple[str, str, str], list[dict[str, float]]] = defaultdict(list)
    for (algo, label, group, trial), ranks in per_trial.items():
        record = records[(label, next(iter(ranks)))]
        iters = int(record["iters"])  # type: ignore[arg-type]
        slowest: dict[str, float] = {}
        for rank, sample in ranks.items():
            ov = overlap.get((rank, label, trial), 0)
            values = {
                "ag_ms": sample["ag"] / iters / NS_PER_MS,
                "rs_ms": sample["rs"] / iters / NS_PER_MS,
                "ar_ms": sample["ar"] / iters / NS_PER_MS,
                "overlap_ms": ov / iters / NS_PER_MS,
                "span_ms": (sample["last"] - sample["first"]) / iters / NS_PER_MS,
            }
            for key, value in values.items():
                slowest[key] = max(slowest.get(key, 0.0), value)
        cases[(algo, label, group)].append(slowest)

    rows = []
    for (algo, label, group), trials in sorted(cases.items(), key=lambda item: (algo_order(item[0][0]), item[0][1:])):
        record = next(r for (lbl, _), r in records.items() if lbl == label and r["group_name"] == group)
        stats = {key: median(trial[key] for trial in trials) for key in trials[0]}
        shorter = min(stats["ag_ms"], stats["rs_ms"])
        event_ms = float(record["makespan_ms"])  # type: ignore[arg-type]
        rows.append(
            {
                "algo": algo,
                "label": label,
                "group": group,
                "group_size": record["group_size"],
                "workload": record["workload"],
                "phase": record.get("phase"),
                "trials": len(trials),
                "kernels": ";".join(sorted(names[(algo, label, group)])),
                **stats,
                "overlap_fraction": stats["overlap_ms"] / shorter if shorter > 0 else None,
                "event_makespan_ms": event_ms,
                "span_over_event": stats["span_ms"] / event_ms if event_ms > 0 else None,
            }
        )
    return rows


def summarize_metrics(metrics: Sequence[MetricRow], results: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Per case, group and reported metric: mean / min / max over the group's ranks, median over trials.

    Unlike kernel time, a throughput has no "slowest rank": how far the ranks of one group drift
    apart (one GPU's links or HBM busier than the others') is itself the finding, so it is kept
    as min and max.
    """
    records = _rank_records(results)
    per_trial: dict[tuple[str, str, str, str], dict[str, list[MetricRow]]] = defaultdict(lambda: defaultdict(list))
    for row in metrics:
        record = records.get((row.label, row.rank))
        if record is None or not REPORTED_METRICS.search(row.metric):
            continue  # idle rank of a split case, a label with no results row, or another metric
        per_trial[(row.algo, row.label, str(record["group_name"]), row.metric)][row.trial].append(row)

    rows = []
    for (algo, label, group, metric), trials in sorted(
        per_trial.items(), key=lambda item: (algo_order(item[0][0]), item[0][1:])
    ):
        record = next(r for (lbl, _), r in records.items() if lbl == label and r["group_name"] == group)
        # One list per trial, holding that trial's row of every rank of the group.
        by_trial = list(trials.values())
        rows.append(
            {
                "algo": algo,
                "label": label,
                "group": group,
                "workload": record["workload"],
                "phase": record.get("phase"),
                "metric": metric,
                "trials": len(by_trial),
                "ranks": max(len(ranks) for ranks in by_trial),
                "mean": median(mean(r.mean for r in ranks) for ranks in by_trial),
                "rank_min": median(min(r.mean for r in ranks) for ranks in by_trial),
                "rank_max": median(max(r.mean for r in ranks) for ranks in by_trial),
                "peak": median(max(r.peak for r in ranks) for ranks in by_trial),
                "min_samples": min(r.samples for ranks in by_trial for r in ranks),
                "window_ms": median(max(r.window_ns for r in ranks) for ranks in by_trial) / NS_PER_MS,
            }
        )
    return rows


def algo_order(algo: str) -> tuple[int, str]:
    """Sort key: this directory's families first, then any other pass (e.g. ../nvls-collective-cpp's tma, ce)."""
    return (ALGORITHMS.index(algo), "") if algo in ALGORITHMS else (len(ALGORITHMS), algo)


def _fmt(value: object, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def render(rows: Sequence[dict[str, object]]) -> str:
    """Markdown table of the per-case summary."""
    header = [
        "algo",
        "case",
        "group",
        "phase",
        "AG kernel ms/iter",
        "RS kernel ms/iter",
        "AR kernel ms/iter",
        "AG∩RS ms/iter",
        "overlap_fraction",
        "kernel span ms/iter",
        "CUDA-event ms/iter",
        "span/event",
        "kernels",
    ]
    lines = [
        "# AG / RS / AR kernel cost from nsys (SQL)",
        "",
        "每格：trial 中位数，trial 内取组内最慢 rank；ms/iter = 该 trial 内 kernel 时长之和 / iters。",
        "kernel 时长含等待对端的时间。经典 NCCL kernel 名不反映算法（NVLS 也显示 *_RING_LL），算法看 compare.md 2b。",
        "以下数字来自 --nsys 采集，含 Nsight 开销，只用于和同一次运行的 CUDA-event 时间对照。",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for row in rows:
        cells = [
            row["algo"],
            row["label"],
            row["group"],
            row["phase"],
            _fmt(row["ag_ms"]),
            _fmt(row["rs_ms"]),
            _fmt(row["ar_ms"]),
            _fmt(row["overlap_ms"]),
            _fmt(row["overlap_fraction"], 2),
            _fmt(row["span_ms"]),
            _fmt(row["event_makespan_ms"]),
            _fmt(row["span_over_event"], 2),
            row["kernels"],
        ]
        lines.append("| " + " | ".join(str(cell) for cell in cells) + " |")
    return "\n".join(lines) + "\n"


def render_metrics(rows: Sequence[dict[str, object]]) -> str:
    """Markdown table of the per-case GPU metric summary."""
    header = [
        "algo",
        "case",
        "group",
        "phase",
        "metric",
        "mean %",
        "rank min %",
        "rank max %",
        "peak sample %",
        "min samples",
        "window ms",
    ]
    lines = [
        "# NVLink / DRAM / C2C throughput from nsys GPU metrics (SQL)",
        "",
        "每格：nsys GPU Metrics 采样值（占峰值的百分比，整数采样）在每个 trial 的 GPU 窗口内取平均，",
        "窗口 = 该 rank 本 trial 第一个集合通信 kernel 开始到最后一个结束（含 iters 次背靠背迭代）。",
        "mean / rank min / rank max：组内各 rank 的平均、最小、最大；peak sample：窗口内单个采样的最大值；",
        "再对 trial 取中位数。min samples：所有 trial、所有 rank 中窗口内最少的采样数，太少（< 5）时只看量级。",
        "NVLink bytes received / transmitted 是本卡**所有 NVLink 链路合计**（含协议开销），不能区分单条链路；",
        "GB200 每方向峰值约 900 GB/s（18 × 50 GB/s），GB/s ≈ % × 9。DRAM 是 HBM 接口忙碌周期占比。",
        "以下数字来自 --nsys 采集，含 Nsight 开销。",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for row in rows:
        cells = [
            row["algo"],
            row["label"],
            row["group"],
            row["phase"],
            row["metric"],
            _fmt(row["mean"], 1),
            _fmt(row["rank_min"], 1),
            _fmt(row["rank_max"], 1),
            _fmt(row["peak"], 1),
            row["min_samples"],
            _fmt(row["window_ms"]),
        ]
        lines.append("| " + " | ".join(str(cell) for cell in cells) + " |")
    return "\n".join(lines) + "\n"


def _load_results(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_algo(algo_dir: Path, *, timeout_s: float) -> list[Path]:
    """File this pass's raw reports under ``cases/``; return every filed report of the pass."""
    nsys_dir = algo_dir / "nsys"
    labels = read_index(nsys_dir)
    world_size = int(json.loads((algo_dir / "metadata.json").read_text(encoding="utf-8"))["world_size"])
    filed = sorted(algo_dir.glob("cases/*/*.nsys-rep"))
    raw = wait_for_reports(nsys_dir, expected=world_size * len(labels) - len(filed), timeout_s=timeout_s)
    filed += file_reports(algo_dir, raw, labels, world_size=world_size)
    LOGGER.info("%s: %d reports filed under %s", algo_dir.name, len(filed), algo_dir / "cases")
    return sorted(filed)


def _write_csv(path: Path, row_type: type, rows: Sequence[object]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(row_type.__dataclass_fields__)
        writer.writerows([getattr(row, name) for name in row_type.__dataclass_fields__] for row in rows)


def analyse_algo(
    algo_dir: Path, queries: dict[str, str], *, nsys_bin: str, timeout_s: float
) -> tuple[list[KernelRow], list[dict[str, object]], list[dict[str, object]]]:
    """File, export and query every report of one algorithm pass.

    Returns the kernel rows, the per-case kernel summary and the per-case GPU metric summary.
    """
    algo = algo_dir.name
    results = _load_results(algo_dir / "results.jsonl")
    kernels: list[KernelRow] = []
    metrics: list[MetricRow] = []
    overlap: dict[tuple[int, str, str], int] = {}
    filed = file_algo(algo_dir, timeout_s=timeout_s)
    for report in filed:
        rank = filed_rank(report)
        sqlite_path = export_sqlite(report, nsys_bin=nsys_bin)
        rows, report_overlap = query_report(
            sqlite_path, queries, algo=algo, rank=rank, label=filed_label(algo, report)
        )
        kernels += rows
        overlap.update({(rank, label, trial): ns for (label, trial), ns in report_overlap.items()})
        metrics += query_gpu_metrics(sqlite_path, queries, algo=algo, rank=rank)
    _write_csv(algo_dir / "nsys" / "kernels.csv", KernelRow, kernels)
    if not kernels:
        LOGGER.warning("%s: no AllGather/ReduceScatter/AllReduce kernels found in %d reports", algo, len(filed))
    if metrics:
        _write_csv(algo_dir / "nsys" / "gpu_metrics.csv", MetricRow, metrics)
        if not any(REPORTED_METRICS.search(row.metric) for row in metrics):
            LOGGER.warning(
                "%s: the GPU metric set has no NVLink/DRAM/C2C metric (%s); pick another with --nsys-gpu-metrics-set",
                algo,
                ", ".join(sorted({row.metric for row in metrics})),
            )
    elif kernels:
        LOGGER.info("%s: no GPU metric samples in the reports (recorded without --gpu-metrics-devices)", algo)
    return kernels, summarize(kernels, overlap, results), summarize_metrics(metrics, results)


def main(argv: Sequence[str] | None = None) -> None:
    """File and analyse every algorithm / impl directory that was run with ``--nsys``."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True, help="Launcher --output-dir")
    parser.add_argument("--algo", help="Only this pass directory (default: every <algo>/nsys/index.tsv)")
    parser.add_argument("--file-only", action="store_true", help="Move the reports into cases/, skip the SQL")
    parser.add_argument("--nsys-bin", default="nsys")
    parser.add_argument("--wait-seconds", type=float, default=300.0, help="Wait for other nodes' reports")
    args = parser.parse_args(argv)

    # Every pass directory that was profiled; the C++ benchmark writes the same layout per impl.
    algos = sorted((p.parent.parent.name for p in args.output_dir.glob("*/nsys/index.tsv")), key=algo_order)
    if args.algo is not None:
        algos = [algo for algo in algos if algo == args.algo]
    if not algos:
        raise FileNotFoundError(f"no {args.algo or '<algo>'}/nsys/index.tsv under {args.output_dir}")
    queries = load_queries()
    rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    failed = []
    for algo in algos:
        algo_dir = args.output_dir / algo
        try:
            if args.file_only:
                file_algo(algo_dir, timeout_s=args.wait_seconds)
            else:
                _, case_rows, case_metric_rows = analyse_algo(
                    algo_dir, queries, nsys_bin=args.nsys_bin, timeout_s=args.wait_seconds
                )
                rows += case_rows
                metric_rows += case_metric_rows
        except Exception:
            LOGGER.exception("%s: nsys %s failed", algo, "filing" if args.file_only else "analysis")
            failed.append(algo)
    if not args.file_only and rows:
        (args.output_dir / "nsys_kernels.md").write_text(render(rows), encoding="utf-8")
        LOGGER.info("wrote %s (%d cases)", args.output_dir / "nsys_kernels.md", len(rows))
    if not args.file_only and metric_rows:
        (args.output_dir / "nsys_gpu_metrics.md").write_text(render_metrics(metric_rows), encoding="utf-8")
        LOGGER.info("wrote %s (%d rows)", args.output_dir / "nsys_gpu_metrics.md", len(metric_rows))
    if failed:
        raise RuntimeError(f"nsys {'filing' if args.file_only else 'analysis'} failed for {', '.join(failed)}")
    if not args.file_only and not rows:
        raise RuntimeError(
            f"no AllGather/ReduceScatter/AllReduce kernels matched results.jsonl under {args.output_dir}"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
