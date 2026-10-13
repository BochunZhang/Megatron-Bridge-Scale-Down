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
"""Query all-gather / reduce-scatter kernel cost from the ``--nsys`` reports with SQL.

For every ``<output-dir>/<algo>/nsys/rank*.nsys-rep`` this script runs
``nsys export --type sqlite``, executes the two queries in ``nsys_collective_kernels.sql``
and joins the result back to ``<algo>/results.jsonl`` through the case's NVTX label and the
rank's group. Only the Python standard library is used::

    python analyse_nsys.py --output-dir <dir>

Outputs ``<algo>/nsys/kernels.csv`` (one row per report, trial, op and kernel) and
``<output-dir>/nsys_kernels.md`` (per case and group: median over trials of the slowest rank).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sqlite3
import subprocess
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median


LOGGER = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent
SQL_PATH = HERE / "nsys_collective_kernels.sql"
ALGORITHMS = ("ring", "nvls", "symm")
REQUIRED_COLUMNS = {
    "StringIds": {"id", "value"},
    "NVTX_EVENTS": {"start", "end", "text", "textId", "globalTid"},
    "CUPTI_ACTIVITY_KIND_RUNTIME": {"start", "globalTid", "correlationId"},
    "CUPTI_ACTIVITY_KIND_KERNEL": {"start", "end", "deviceId", "streamId", "correlationId", "globalPid", "shortName"},
}
_RANK = re.compile(r"^rank(\d+)(?!\d)")
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


def load_queries(path: Path = SQL_PATH) -> dict[str, str]:
    """Split the SQL file on ``-- name: <query>`` markers."""
    queries = {}
    for block in path.read_text(encoding="utf-8").split("-- name: ")[1:]:
        name, body = block.split("\n", 1)
        queries[name.strip()] = body.strip()
    if set(queries) != {"kernels", "overlap"}:
        raise ValueError(f"{path} must define exactly the 'kernels' and 'overlap' queries, got {sorted(queries)}")
    return queries


def report_rank(path: Path) -> int:
    """Rank encoded by the launcher's ``--output=.../rank%q{RANK}``; nsys appends its own suffix."""
    match = _RANK.match(path.name)
    if match is None:
        raise ValueError(f"cannot read a rank from report name {path.name!r}")
    return int(match.group(1))


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
                "%s: %d of %d reports after %.0fs; analysing what exists", nsys_dir, len(reports), expected, timeout_s
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


def check_schema(connection: sqlite3.Connection, path: Path) -> None:
    """Fail with the missing table/column instead of an opaque SQL error."""
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for table, columns in REQUIRED_COLUMNS.items():
        if table not in tables:
            raise RuntimeError(f"{path}: table {table} is missing; was the run profiled with --trace=cuda,nvtx?")
        present = {row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')}
        if columns - present:
            raise RuntimeError(
                f"{path}: {table} lacks columns {sorted(columns - present)} (nsys export schema changed)"
            )


def query_report(
    sqlite_path: Path, queries: dict[str, str], *, algo: str, rank: int
) -> tuple[list[KernelRow], dict[tuple[str, str], int]]:
    """Run both queries on one exported report."""
    connection = sqlite3.connect(sqlite_path)
    try:
        check_schema(connection, sqlite_path)
        kernels = [
            KernelRow(algo, rank, sqlite_path.stem, *row) for row in connection.execute(queries["kernels"]).fetchall()
        ]
        overlap = {(label, trial): int(ns) for label, trial, ns in connection.execute(queries["overlap"])}
    finally:
        connection.close()
    return kernels, overlap


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
    """Per case and group: median over trials of the slowest rank's per-iteration kernel times."""
    records = _rank_records(results)
    per_trial: dict[tuple[str, str, str, str], dict[int, dict[str, float]]] = defaultdict(dict)
    names: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in kernels:
        record = records.get((row.label, row.rank))
        if record is None:
            continue  # idle rank of a split case, or a label with no results row
        group = str(record["group_name"])
        sample = per_trial[(row.algo, row.label, group, row.trial)].setdefault(
            row.rank, {"ag": 0.0, "rs": 0.0, "first": float("inf"), "last": 0.0}
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
                "overlap_ms": ov / iters / NS_PER_MS,
                "span_ms": (sample["last"] - sample["first"]) / iters / NS_PER_MS,
            }
            for key, value in values.items():
                slowest[key] = max(slowest.get(key, 0.0), value)
        cases[(algo, label, group)].append(slowest)

    rows = []
    for (algo, label, group), trials in sorted(
        cases.items(), key=lambda item: (ALGORITHMS.index(item[0][0]), item[0][1:])
    ):
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
        "AG∩RS ms/iter",
        "overlap_fraction",
        "kernel span ms/iter",
        "CUDA-event ms/iter",
        "span/event",
        "kernels",
    ]
    lines = [
        "# AG / RS kernel cost from nsys (SQL)",
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
            _fmt(row["overlap_ms"]),
            _fmt(row["overlap_fraction"], 2),
            _fmt(row["span_ms"]),
            _fmt(row["event_makespan_ms"]),
            _fmt(row["span_over_event"], 2),
            row["kernels"],
        ]
        lines.append("| " + " | ".join(str(cell) for cell in cells) + " |")
    return "\n".join(lines) + "\n"


def _load_results(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def analyse_algo(
    algo_dir: Path, queries: dict[str, str], *, nsys_bin: str, timeout_s: float
) -> tuple[list[KernelRow], list[dict[str, object]]]:
    """Export and query every report of one algorithm pass."""
    algo = algo_dir.name
    nsys_dir = algo_dir / "nsys"
    index_path = nsys_dir / "index.tsv"
    results = _load_results(algo_dir / "results.jsonl")
    world_size = int(json.loads((algo_dir / "metadata.json").read_text(encoding="utf-8"))["world_size"])
    captures = len(index_path.read_text(encoding="utf-8").splitlines()) - 1
    reports = wait_for_reports(nsys_dir, expected=world_size * captures, timeout_s=timeout_s)
    kernels: list[KernelRow] = []
    overlap: dict[tuple[int, str, str], int] = {}
    for report in reports:
        rank = report_rank(report)
        rows, report_overlap = query_report(export_sqlite(report, nsys_bin=nsys_bin), queries, algo=algo, rank=rank)
        kernels += rows
        overlap.update({(rank, label, trial): ns for (label, trial), ns in report_overlap.items()})
    with (nsys_dir / "kernels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(KernelRow.__dataclass_fields__)
        writer.writerows([getattr(row, name) for name in KernelRow.__dataclass_fields__] for row in kernels)
    if not kernels:
        LOGGER.warning("%s: no AllGather/ReduceScatter kernels found in %d reports", algo, len(reports))
    return kernels, summarize(kernels, overlap, results)


def main(argv: Sequence[str] | None = None) -> None:
    """Analyse every algorithm directory that was run with ``--nsys``."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True, help="Launcher --output-dir")
    parser.add_argument("--nsys-bin", default="nsys")
    parser.add_argument("--wait-seconds", type=float, default=300.0, help="Wait for other nodes' reports")
    args = parser.parse_args(argv)

    queries = load_queries()
    rows: list[dict[str, object]] = []
    for algo in ALGORITHMS:
        algo_dir = args.output_dir / algo
        if (algo_dir / "nsys" / "index.tsv").exists():
            rows += analyse_algo(algo_dir, queries, nsys_bin=args.nsys_bin, timeout_s=args.wait_seconds)[1]
    if not rows:
        raise FileNotFoundError(f"no <algo>/nsys/index.tsv with matching kernels under {args.output_dir}")
    (args.output_dir / "nsys_kernels.md").write_text(render(rows), encoding="utf-8")
    LOGGER.info("wrote %s (%d cases)", args.output_dir / "nsys_kernels.md", len(rows))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
