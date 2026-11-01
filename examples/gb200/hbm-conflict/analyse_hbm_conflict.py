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
"""Report GEMM time under AllGather traffic from the nsys reports of run_hbm_conflict_benchmark.sh.

For every case directory (``baseline``, ``pause<C>cyc``) of a run, every rank's report
``<case>/nsys/<case>.rank_<R>.nsys-rep`` is exported with ``nsys export --type sqlite`` (an export
newer than its report is reused) and queried with ``hbm_conflict.sql``:

* AllGather rounds: the kernel of every ``ag<II>`` range, its time and CTA count (gridX). A round's
  time is the slowest rank's kernel; bandwidth uses the gathered size (algbw) and
  ``algbw * (n - 1) / n`` (busbw).
* GEMMs: the kernels of every ``gemm<JJ>`` range, first start to last end. Each GEMM is checked
  against the union of the same GPU's AllGather kernels: ``coverage`` is the fraction of the GEMM
  that ran while an AllGather kernel was resident.
* GPU metrics: the NVLink / DRAM / SM samples averaged over the isolated AllGather rounds and over
  the GEMMs (first GEMM start to last GEMM end) of each rank.

The CUDA-event times of the worker (``events.rank_<RR>.json``) are listed next to the nsys times
as a cross-check. Output in the run directory: ``report.md``, ``summary.json``, ``ag_rounds.csv``,
``gemms.csv``, ``gpu_metrics.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import shutil
import sqlite3
import statistics
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


LOGGER = logging.getLogger(__name__)

SQL_PATH = Path(__file__).with_name("hbm_conflict.sql")
QUERY_NAMES = {"label", "kernels", "gpu_metrics"}
AG_KERNEL = "ncclSymkDevKernel_AllGather_ST"
_REPORT_RANK = re.compile(r"\.rank_(\d+)\.nsys-rep$")
_AG_RANGE = re.compile(r"^ag(\d+)$")
_GEMM_RANGE = re.compile(r"^gemm(\d+)$")
# Metrics reported in report.md (gpu_metrics.csv keeps every metric of the set).
REPORTED_METRICS = re.compile(r"nvl|dram|c2c|sms? active", re.IGNORECASE)
WINDOWS = ("ag_isolated", "gemm")
NS_PER_MS = 1e6
DTYPE_BYTES = {"bf16": 2, "fp16": 2}


@dataclass(frozen=True)
class Kernel:
    """One kernel of the ``kernels`` query."""

    range_name: str
    kernel: str
    start: int
    end: int
    device_id: int
    stream_id: int
    grid_x: int
    block_x: int


@dataclass(frozen=True)
class Sample:
    """One GPU metric sample."""

    timestamp: int
    metric: str
    value: float


def load_queries(path: Path = SQL_PATH) -> dict[str, str]:
    """Split the SQL file on ``-- name: <query>`` markers."""
    queries = {}
    for block in path.read_text(encoding="utf-8").split("-- name: ")[1:]:
        name, body = block.split("\n", 1)
        queries[name.strip()] = body.strip()
    if set(queries) != QUERY_NAMES:
        raise ValueError(f"{path} must define exactly the {sorted(QUERY_NAMES)} queries, got {sorted(queries)}")
    return queries


def report_rank(path: Path) -> int:
    """Rank of a report named ``<case>.rank_<R>.nsys-rep``."""
    match = _REPORT_RANK.search(path.name)
    if match is None:
        raise ValueError(f"cannot read a rank from report name {path.name!r}")
    return int(match.group(1))


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


def query_report(sqlite_path: Path, queries: dict[str, str], *, case: str) -> tuple[list[Kernel], list[Sample]]:
    """Kernels and GPU metric samples of one report; fails if it is not a report of ``case``."""
    connection = sqlite3.connect(sqlite_path)
    try:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        missing = {"NVTX_EVENTS", "StringIds", "CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_RUNTIME"} - tables
        if missing:
            raise RuntimeError(f"{sqlite_path}: no {sorted(missing)} table; the capture window recorded no kernels")
        labels = [row[0] for row in connection.execute(queries["label"])]
        if labels != [f"hbm-conflict/{case}"]:
            raise RuntimeError(f"{sqlite_path}: filed under {case} but its case NVTX ranges are {labels}")
        kernels = [Kernel(*row) for row in connection.execute(queries["kernels"])]
        samples = []
        if {"GPU_METRICS", "TARGET_INFO_GPU_METRICS"} <= tables:
            samples = [Sample(*row) for row in connection.execute(queries["gpu_metrics"])]
        return kernels, samples
    finally:
        connection.close()


def merge_intervals(intervals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Union of ``[start, end]`` intervals as sorted, disjoint intervals."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def overlap_ns(start: int, end: int, intervals: Sequence[tuple[int, int]]) -> int:
    """Length of ``[start, end]`` covered by disjoint ``intervals``."""
    return sum(max(0, min(end, b) - max(start, a)) for a, b in intervals)


def round_phase(index: int, *, warmup: int, isolated: int) -> str:
    """``warmup``, ``isolated`` or ``background`` for AllGather round ``index``."""
    if index < warmup:
        return "warmup"
    if index < warmup + isolated:
        return "isolated"
    return "background"


def window_metrics(samples: Sequence[Sample], start: int, end: int) -> dict[str, dict[str, float]]:
    """Per metric: sample count, mean and peak of the samples in ``[start, end]``."""
    values: dict[str, list[float]] = {}
    for sample in samples:
        if start <= sample.timestamp <= end:
            values.setdefault(sample.metric, []).append(sample.value)
    return {
        metric: {"samples": len(v), "mean": statistics.fmean(v), "peak": max(v)}
        for metric, v in sorted(values.items())
    }


def rank_summary(kernels: Sequence[Kernel], samples: Sequence[Sample], *, warmup: int, isolated: int) -> dict:
    """AllGather rounds, GEMMs (with their AllGather coverage) and window metrics of one rank."""
    ag: dict[int, list[Kernel]] = {}
    gemm: dict[int, list[Kernel]] = {}
    for kernel in kernels:
        if match := _AG_RANGE.match(kernel.range_name):
            ag.setdefault(int(match.group(1)), []).append(kernel)
        elif match := _GEMM_RANGE.match(kernel.range_name):
            gemm.setdefault(int(match.group(1)), []).append(kernel)

    rounds = []
    for index in sorted(ag):
        # One AllGather launches one NCCL kernel; anything else in the range is reported, not timed.
        nccl = [k for k in ag[index] if k.kernel.startswith("nccl")]
        if len(nccl) != 1:
            raise RuntimeError(f"ag{index:02d}: expected one NCCL kernel, got {[k.kernel for k in ag[index]]}")
        k = nccl[0]
        rounds.append(
            {
                "round": index,
                "phase": round_phase(index, warmup=warmup, isolated=isolated),
                "kernel": k.kernel,
                "grid_x": k.grid_x,
                "block_x": k.block_x,
                "start": k.start,
                "end": k.end,
                "ns": k.end - k.start,
            }
        )
    ag_union = merge_intervals([(r["start"], r["end"]) for r in rounds])

    gemms = []
    for index in sorted(gemm):
        start = min(k.start for k in gemm[index])
        end = max(k.end for k in gemm[index])
        covered = overlap_ns(start, end, ag_union)
        gemms.append(
            {
                "index": index,
                "kernels": sorted({k.kernel for k in gemm[index]}),
                "start": start,
                "end": end,
                "ns": end - start,
                "covered_ns": covered,
                "coverage": covered / (end - start) if end > start else 0.0,
                "ag_rounds": [r["round"] for r in rounds if r["start"] < end and start < r["end"]],
            }
        )

    windows: dict[str, tuple[int, int]] = {}
    isolated_rounds = [r for r in rounds if r["phase"] == "isolated"]
    if isolated_rounds:
        windows["ag_isolated"] = (isolated_rounds[0]["start"], isolated_rounds[-1]["end"])
    if gemms:
        windows["gemm"] = (gemms[0]["start"], gemms[-1]["end"])
    metrics = {name: window_metrics(samples, *span) for name, span in windows.items()}
    return {"ag_rounds": rounds, "gemms": gemms, "windows": windows, "metrics": metrics}


def _stats(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def summarize_case(case: str, ranks: dict[int, dict], events: dict[int, dict]) -> dict:
    """Combine the ranks of one case; ``events`` are the worker records (configuration and CUDA events)."""
    config = events[min(events)]
    world_size = config["world_size"]
    size_bytes = config["size_mib"] << 20
    m, k, n = config["gemm_mkn"]
    elem = DTYPE_BYTES[config["dtype"]]
    flops = 2.0 * m * k * n
    gemm_min_bytes = float(m * k + k * n + m * n) * elem

    # AllGather: per round, the slowest rank.
    by_round: dict[int, list[dict]] = {}
    for summary in ranks.values():
        for r in summary["ag_rounds"]:
            by_round.setdefault(r["round"], []).append(r)
    gemm_rounds = {i for s in ranks.values() for g in s["gemms"] for i in g["ag_rounds"]}
    rounds = []
    for index in sorted(by_round):
        rows = by_round[index]
        ms = max(r["ns"] for r in rows) / NS_PER_MS
        algbw = size_bytes / (ms * 1e6)
        rounds.append(
            {
                "case": case,
                "round": index,
                "phase": rows[0]["phase"],
                "with_gemm": index in gemm_rounds,
                "ranks": len(rows),
                "ms": ms,
                "rank_min_ms": min(r["ns"] for r in rows) / NS_PER_MS,
                "algbw_gbps": algbw,
                "busbw_gbps": algbw * (world_size - 1) / world_size,
                "kernels": sorted({r["kernel"] for r in rows}),
                "grid_x": sorted({r["grid_x"] for r in rows}),
            }
        )

    def ag_phase(phase: str, *, with_gemm: bool | None = None) -> dict | None:
        rows = [r for r in rounds if r["phase"] == phase and (with_gemm is None or r["with_gemm"] == with_gemm)]
        if not rows:
            return None
        ms = statistics.fmean(r["ms"] for r in rows)
        algbw = size_bytes / (ms * 1e6)
        return {
            "rounds": [r["round"] for r in rows],
            "ms": ms,
            "algbw_gbps": algbw,
            "busbw_gbps": algbw * (world_size - 1) / world_size,
        }

    gemm_rows = []
    for rank, summary in sorted(ranks.items()):
        for g in summary["gemms"]:
            gemm_rows.append({"case": case, "rank": rank, **{key: g[key] for key in g if key not in ("start", "end")}})
    gemm_ms = [g["ns"] / NS_PER_MS for g in gemm_rows]
    gemm = None
    if gemm_ms:
        gemm = _stats(gemm_ms)
        gemm["tflops"] = flops / (gemm["mean"] * 1e9)
        gemm["min_traffic_gbps"] = gemm_min_bytes / (gemm["mean"] * 1e6)
        gemm["slowest_rank_mean_ms"] = max(
            statistics.fmean(g["ns"] / NS_PER_MS for g in gemm_rows if g["rank"] == rank) for rank in ranks
        )
        gemm["count"] = len(gemm_rows)
        gemm["fully_covered"] = sum(1 for g in gemm_rows if g["covered_ns"] == g["ns"])
        gemm["min_coverage"] = min(g["coverage"] for g in gemm_rows)
        gemm["kernels"] = sorted({name for g in gemm_rows for name in g["kernels"]})

    # Window metrics, averaged over the ranks.
    metric_rows = []
    for rank, summary in sorted(ranks.items()):
        for window, values in summary["metrics"].items():
            for metric, v in values.items():
                metric_rows.append({"case": case, "rank": rank, "window": window, "metric": metric, **v})
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    for window in WINDOWS:
        names = sorted({r["metric"] for r in metric_rows if r["window"] == window})
        for name in names:
            rows = [r for r in metric_rows if r["window"] == window and r["metric"] == name]
            metrics.setdefault(window, {})[name] = {
                "mean": statistics.fmean(r["mean"] for r in rows),
                "rank_min": min(r["mean"] for r in rows),
                "rank_max": max(r["mean"] for r in rows),
                "peak": max(r["peak"] for r in rows),
                "min_samples": min(r["samples"] for r in rows),
            }

    # CUDA events of the worker: [start_ms, end_ms] per round, per rank.
    event_gemm = [end - start for e in events.values() for start, end in e["gemm_ms"]]
    warmup, isolated = config["ag_warmup"], config["ag_isolated"]
    event_ag = [
        max(e["ag_ms"][i][1] - e["ag_ms"][i][0] for e in events.values())
        for i in range(warmup, warmup + isolated)
        if all(len(e["ag_ms"]) > i for e in events.values())
    ]
    return {
        "case": case,
        "mode": config["mode"],
        "pause_cycles": config.get("NCCL_SYM_AG_ST_PAUSE_CYCLES"),
        "world_size": world_size,
        "ranks": sorted(ranks),
        "size_mib": config["size_mib"],
        "dtype": config["dtype"],
        "gemm_mkn": [m, k, n],
        "gemm_delay_us": config["gemm_delay_us"],
        "ag_kernels": sorted({name for r in rounds for name in r["kernels"]}),
        "ag_grid_x": sorted({x for r in rounds for x in r["grid_x"]}),
        "ag_isolated": ag_phase("isolated"),
        "ag_background_with_gemm": ag_phase("background", with_gemm=True),
        "ag_background_without_gemm": ag_phase("background", with_gemm=False),
        "gemm": gemm,
        "metrics": metrics,
        "events": {
            "gemm_mean_ms": statistics.fmean(event_gemm) if event_gemm else None,
            "ag_isolated_ms": statistics.fmean(event_ag) if event_ag else None,
            "ag_correct": all(e.get("ag_correct") is not False for e in events.values()),
        },
        "_ag_rounds": rounds,
        "_gemms": gemm_rows,
        "_metric_rows": metric_rows,
    }


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def _case_order(case: dict) -> tuple[int, int]:
    pause = case.get("pause_cycles")
    return (0 if case["mode"] == "baseline" else 1, int(pause) if pause not in (None, "") else -1)


def render(cases: Sequence[dict]) -> str:
    """``report.md``: the summary against the baseline, then per-case rounds, GEMMs and metrics."""
    cases = sorted(cases, key=_case_order)
    baseline = next((c for c in cases if c["mode"] == "baseline" and c["gemm"]), None)
    base_ms = baseline["gemm"]["mean"] if baseline else None
    first = cases[0]
    m, k, n = first["gemm_mkn"]
    lines = [
        "# GB200：AllGather 背景流量下的 GEMM（HBM 争用）",
        "",
        f"- GEMM：`[{m}, {k}] x [{k}, {n}]` {first['dtype']}，每次 {2 * m * k * n / 1e9:.1f} GFLOP；"
        f"最小 HBM 流量（A + B + C 各读写一次）{(m * k + k * n + m * n) * DTYPE_BYTES[first['dtype']] / 2**20:.1f} MiB。",
        f"- AllGather：{first['world_size']} GPU、gathered {first['size_mib']} MiB，单播对称内核 `AllGather_ST`"
        "（`NCCL_NVLS_ENABLE=0`，非 multicast），流量由 `NCCL_SYM_AG_ST_PAUSE_CYCLES` 控制。",
        "- 时间全部来自 nsys（kernel 的 GPU 起止）；AG 每轮取最慢 rank，GEMM 对所有 rank × 所有 GEMM 取统计。",
        "- `覆盖` = GEMM 期间同一 GPU 上有 AllGather kernel 驻留的时间占比；`10/10` 表示每个 GEMM 都完全在 AG 期间执行。",
        "- `vs baseline` = GEMM 平均时间 / baseline GEMM 平均时间（>1 表示变慢）。",
        "",
        "## 1. 汇总",
        "",
        "| case | pause (cycles) | AG kernel | AG CTAs | AG 独立 ms | AG 独立 algbw GB/s | AG 独立 busbw GB/s "
        "| AG 与 GEMM 重叠轮 ms | GEMM mean ms | GEMM min / max ms | GEMM TFLOPS | GEMM 最小流量 GB/s "
        "| vs baseline | 覆盖（完全 / 最小） | event GEMM ms | event AG 独立 ms |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for case in cases:
        iso, with_gemm, gemm = case["ag_isolated"], case["ag_background_with_gemm"], case["gemm"]
        ratio = gemm["mean"] / base_ms if gemm and base_ms else None
        coverage = "-"
        if gemm and case["mode"] == "sweep":
            coverage = f"{gemm['fully_covered']}/{gemm['count']} / {_pct(gemm['min_coverage'])}"
        lines.append(
            "| "
            + " | ".join(
                [
                    case["case"],
                    _fmt(case["pause_cycles"]),
                    ", ".join(n.removeprefix("ncclSymkDevKernel_") for n in case["ag_kernels"]) or "-",
                    "|".join(str(x) for x in case["ag_grid_x"]) or "-",
                    _fmt(iso and iso["ms"]),
                    _fmt(iso and iso["algbw_gbps"], 1),
                    _fmt(iso and iso["busbw_gbps"], 1),
                    _fmt(with_gemm and with_gemm["ms"]),
                    _fmt(gemm and gemm["mean"], 4),
                    f"{gemm['min']:.4f} / {gemm['max']:.4f}" if gemm else "-",
                    _fmt(gemm and gemm["tflops"], 1),
                    _fmt(gemm and gemm["min_traffic_gbps"], 1),
                    _fmt(ratio),
                    coverage,
                    _fmt(case["events"]["gemm_mean_ms"], 4),
                    _fmt(case["events"]["ag_isolated_ms"]),
                ]
            )
            + " |"
        )

    metric_names = sorted(
        {name for c in cases for w in c["metrics"].values() for name in w if REPORTED_METRICS.search(name)}
    )
    lines += [
        "",
        "## 2. NVLink / DRAM 吞吐（nsys GPU metrics，窗口内采样均值，% of peak）",
        "",
        "每格是各 rank 窗口均值的平均（括号里是 rank 最小 ~ 最大）。`ag_isolated` = 独立执行的 AG 轮，"
        "`gemm` = 第一个 GEMM 开始到最后一个 GEMM 结束（sweep 下 AG 同时在跑）。NVLink 是本卡 18 条链路合计，"
        "GB/s ≈ % × 9；DRAM 是 HBM 接口忙碌周期占比。",
        "",
    ]
    if metric_names:
        lines += [
            "| case | window | min samples | " + " | ".join(metric_names) + " |",
            "| --- | --- | ---: | " + " | ".join("---:" for _ in metric_names) + " |",
        ]
        for case in cases:
            for window in WINDOWS:
                values = case["metrics"].get(window)
                if not values:
                    continue
                cells = []
                for name in metric_names:
                    v = values.get(name)
                    cells.append("-" if v is None else f"{v['mean']:.1f} ({v['rank_min']:.1f}~{v['rank_max']:.1f})")
                samples = min(v["min_samples"] for v in values.values())
                lines.append(f"| {case['case']} | {window} | {samples} | " + " | ".join(cells) + " |")
    else:
        lines.append("报告里没有 NVLink / DRAM metric（没有 GPU metrics 表，或 metric set 不含它们）。")

    lines += ["", "## 3. 每轮 AllGather（最慢 rank）", ""]
    for case in cases:
        if not case["_ag_rounds"]:
            continue
        lines += [
            f"### {case['case']}",
            "",
            "| round | phase | 与 GEMM 重叠 | ms | rank 最快 ms | algbw GB/s | busbw GB/s | CTAs |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
        for r in case["_ag_rounds"]:
            lines.append(
                f"| {r['round']} | {r['phase']} | {'yes' if r['with_gemm'] else ''} | {r['ms']:.3f} | "
                f"{r['rank_min_ms']:.3f} | {r['algbw_gbps']:.1f} | {r['busbw_gbps']:.1f} | "
                + "|".join(str(x) for x in r["grid_x"])
                + " |"
            )
        lines.append("")

    lines += ["## 4. 每个 GEMM（所有 rank）", ""]
    for case in cases:
        if not case["_gemms"]:
            continue
        lines += [
            f"### {case['case']}",
            "",
            "| GEMM | rank 平均 ms | rank 最大 ms | TFLOPS（平均） | 最小覆盖 | 重叠的 AG 轮 |",
            "| ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        m, k, n = case["gemm_mkn"]
        by_index: dict[int, list[dict]] = {}
        for g in case["_gemms"]:
            by_index.setdefault(g["index"], []).append(g)
        for index, rows in sorted(by_index.items()):
            mean_ms = statistics.fmean(g["ns"] / NS_PER_MS for g in rows)
            rounds = sorted({i for g in rows for i in g["ag_rounds"]})
            lines.append(
                f"| {index} | {mean_ms:.4f} | {max(g['ns'] for g in rows) / NS_PER_MS:.4f} | "
                f"{2.0 * m * k * n / (mean_ms * 1e9):.1f} | "
                f"{_pct(min(g['coverage'] for g in rows)) if case['mode'] == 'sweep' else '-'} | "
                f"{', '.join(str(i) for i in rounds) or '-'} |"
            )
        lines.append("")

    warnings = []
    for case in cases:
        if case["mode"] == "sweep" and case["ag_kernels"] != [AG_KERNEL]:
            warnings.append(f"{case['case']}: AllGather ran {case['ag_kernels']}, not only {AG_KERNEL}")
        if case["mode"] == "sweep" and case["gemm"] and case["gemm"]["fully_covered"] != case["gemm"]["count"]:
            warnings.append(
                f"{case['case']}: {case['gemm']['count'] - case['gemm']['fully_covered']} GEMM(s) were not fully "
                "inside AllGather kernels"
            )
        if not case["events"]["ag_correct"]:
            warnings.append(f"{case['case']}: the AllGather output was wrong on some rank")
    if warnings:
        lines += ["## 警告", ""] + [f"- {w}" for w in warnings] + [""]
    return "\n".join(lines).rstrip() + "\n"


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(v) if isinstance(v, list) else v for key, v in row.items()})


def analyse_case(case_dir: Path, queries: dict[str, str], *, nsys_bin: str) -> dict:
    """Summarize one case directory of a run."""
    case = case_dir.name
    events = {}
    for path in sorted(case_dir.glob("events.rank_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        events[record["rank"]] = record
    if not events:
        raise RuntimeError(f"{case_dir}: no events.rank_*.json; the worker did not finish")
    config = events[min(events)]
    reports = sorted((case_dir / "nsys").glob("*.rank_*.nsys-rep"))
    if not reports:
        raise RuntimeError(f"{case_dir}/nsys: no nsys reports")
    ranks = {}
    for report in reports:
        kernels, samples = query_report(export_sqlite(report, nsys_bin=nsys_bin), queries, case=case)
        ranks[report_rank(report)] = rank_summary(
            kernels, samples, warmup=config["ag_warmup"], isolated=config["ag_isolated"]
        )
    if sorted(ranks) != sorted(events):
        LOGGER.warning("%s: nsys reports for ranks %s, events for ranks %s", case_dir, sorted(ranks), sorted(events))
    return summarize_case(case, ranks, events)


def main(argv: Sequence[str] | None = None) -> None:
    """Analyse every case of ``--output-dir`` and write the report files there."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True, help="Run directory of the launcher")
    parser.add_argument("--nsys", default=shutil.which("nsys") or "nsys", help="nsys binary (default: from PATH)")
    args = parser.parse_args(argv)

    queries = load_queries()
    case_dirs = sorted(p.parent for p in args.output_dir.glob("*/case.json"))
    if not case_dirs:
        raise SystemExit(f"{args.output_dir}: no <case>/case.json")
    cases, failed = [], []
    for case_dir in case_dirs:
        try:
            cases.append(analyse_case(case_dir, queries, nsys_bin=args.nsys))
        except (RuntimeError, ValueError, subprocess.CalledProcessError, sqlite3.Error) as error:
            LOGGER.error("%s: %s", case_dir.name, error)
            failed.append(case_dir.name)
    if cases:
        out = args.output_dir
        (out / "report.md").write_text(render(cases), encoding="utf-8")
        public = [{key: v for key, v in c.items() if not key.startswith("_")} for c in cases]
        (out / "summary.json").write_text(json.dumps(public, indent=2) + "\n", encoding="utf-8")
        _write_csv(out / "ag_rounds.csv", [r for c in cases for r in c["_ag_rounds"]])
        _write_csv(out / "gemms.csv", [g for c in cases for g in c["_gemms"]])
        _write_csv(out / "gpu_metrics.csv", [r for c in cases for r in c["_metric_rows"]])
        LOGGER.info("wrote %s", out / "report.md")
    if failed:
        raise SystemExit(f"analysis failed for {failed}")


if __name__ == "__main__":
    main()
