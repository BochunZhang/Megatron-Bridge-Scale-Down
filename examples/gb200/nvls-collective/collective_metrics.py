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
"""Hardware-independent helpers and the ring-vs-NVLS report for the collective benchmark.

Only the Python standard library is used so the layout, bandwidth formulas and the
report can be checked on a machine without CUDA. Run as a script to compare
``results.jsonl`` files produced by ``nvls_collective_benchmark.py``::

    python collective_metrics.py --input ring=<dir>/results.jsonl \\
        --input nvls=<dir>/results.jsonl --output <dir>/compare.md
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path


LOGGER = logging.getLogger(__name__)
BYTES_PER_MIB = 1024 * 1024
BYTES_PER_GB = 1_000_000_000
# Algorithm families. "symm" is the window-registered symmetric-memory path; its report rows
# sit alongside ring and nvls because it answers the same questions with a third implementation.
ALGORITHMS = ("ring", "nvls", "symm")
WORKLOADS = ("ag", "rs", "ag+rs")
SPLIT_LAYOUTS = ("contiguous", "interleaved")
NCCL_ALGO_ENV = {"ring": "Ring", "nvls": "NVLS", "symm": None}


def workload_ops(workload: str) -> tuple[str, ...]:
    """Return the collectives issued by a workload, in launch order."""
    if workload not in WORKLOADS:
        raise ValueError(f"unknown workload {workload!r}; expected one of {WORKLOADS}")
    return ("ag", "rs") if workload == "ag+rs" else (workload,)


def split_groups(total_ranks: int, layout: str) -> tuple[list[int], list[int]]:
    """Split ``range(total_ranks)`` into two equal, disjoint groups.

    Args:
        total_ranks: Number of ranks shared by both groups; must be even.
        layout: ``contiguous`` (first half / second half) or ``interleaved`` (even / odd).

    Returns:
        Global ranks of group A and group B.
    """
    if total_ranks < 2 or total_ranks % 2:
        raise ValueError(f"split needs an even rank count >= 2, got {total_ranks}")
    ranks = list(range(total_ranks))
    half = total_ranks // 2
    if layout == "contiguous":
        return ranks[:half], ranks[half:]
    if layout == "interleaved":
        return ranks[0::2], ranks[1::2]
    raise ValueError(f"unknown split layout {layout!r}; expected one of {SPLIT_LAYOUTS}")


def alg_bandwidth_gb_s(*, total_bytes: int, seconds: float) -> float:
    """Full-tensor payload rate: AG output bytes or RS input bytes per second."""
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    return total_bytes / seconds / BYTES_PER_GB


def bus_bandwidth_gb_s(*, total_bytes: int, group_size: int, seconds: float) -> float:
    """nccl-tests bus bandwidth for AG/RS: ``total_bytes * (n - 1) / n / t``."""
    if group_size < 2:
        raise ValueError("group_size must be at least 2")
    return alg_bandwidth_gb_s(total_bytes=total_bytes, seconds=seconds) * (group_size - 1) / group_size


def overlap_metrics(*, ag_s: float, rs_s: float, both_s: float) -> dict[str, float]:
    """Compare a concurrent AG+RS makespan against the isolated AG and RS times.

    ``speedup`` is 1 for fully serialized execution and ``(ag+rs)/max(ag,rs)`` for a perfect
    overlap. ``overlap_fraction`` is 1 for a perfect overlap of the shorter op; it can be
    negative when running both is slower than running them one after another.
    """
    if min(ag_s, rs_s, both_s) <= 0:
        raise ValueError("all times must be positive")
    serial_s = ag_s + rs_s
    return {
        "serial_ms": serial_s * 1e3,
        "ideal_ms": max(ag_s, rs_s) * 1e3,
        "speedup": serial_s / both_s,
        "overlap_fraction": (serial_s - both_s) / min(ag_s, rs_s),
    }


def load_records(path: Path) -> list[dict[str, object]]:
    """Read one JSON object per non-empty line."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _key(record: dict[str, object], *fields: str) -> tuple[object, ...]:
    return tuple(record[field] for field in fields)


def _fmt(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---:|" * len(header)]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _single_index(records: Iterable[dict[str, object]]) -> dict[tuple[object, ...], dict[str, object]]:
    return {_key(r, "algo", "group_size", "size_bytes", "workload"): r for r in records if r["scenario"] == "single"}


def single_rows(records: Sequence[dict[str, object]]) -> list[list[str]]:
    """Question 1: isolated AG / RS versus AG and RS issued together on one group."""
    index = _single_index(records)
    rows = []
    for algo, group_size, size_bytes in sorted({k[:3] for k in index}, key=lambda k: (ALGORITHMS.index(k[0]), *k[1:])):
        ag = index.get((algo, group_size, size_bytes, "ag"))
        rs = index.get((algo, group_size, size_bytes, "rs"))
        both = index.get((algo, group_size, size_bytes, "ag+rs"))
        overlap = None
        if ag and rs and both:
            overlap = overlap_metrics(
                ag_s=float(ag["makespan_ms"]) / 1e3,
                rs_s=float(rs["makespan_ms"]) / 1e3,
                both_s=float(both["makespan_ms"]) / 1e3,
            )
        rows.append(
            [
                str(algo),
                str(group_size),
                str(int(size_bytes) // BYTES_PER_MIB),
                _fmt(ag and float(ag["ag_busbw_gb_s"])),
                _fmt(rs and float(rs["rs_busbw_gb_s"])),
                _fmt(both and float(both["ag_busbw_gb_s"])),
                _fmt(both and float(both["rs_busbw_gb_s"])),
                _fmt(both and float(both["aggregate_busbw_gb_s"])),
                _fmt(overlap and overlap["speedup"], 2),
                _fmt(overlap and overlap["overlap_fraction"], 2),
            ]
        )
    return rows


def _best_of(records: Sequence[dict[str, object]]) -> list[list[str]]:
    """Absolute throughput of every family side by side, plus which one won at that shape."""
    index = _single_index(records)
    rows = []
    for group_size, size_bytes, workload in sorted(
        {k[1:] for k in index}, key=lambda k: (k[0], k[1], WORKLOADS.index(str(k[2])))
    ):
        cells: list[str] = []
        best: tuple[float, str] | None = None
        for algo in ALGORITHMS:
            record = index.get((algo, group_size, size_bytes, workload))
            if record is None:
                cells.append("-")
                continue
            bus = float(record["aggregate_busbw_gb_s"])
            cells.append(_fmt(bus))
            if best is None or bus > best[0]:
                best = (bus, algo)
        rows.append(
            [str(group_size), str(int(size_bytes) // BYTES_PER_MIB), str(workload), *cells, best[1] if best else "-"]
        )
    return rows


def algo_rows(records: Sequence[dict[str, object]]) -> list[list[str]]:
    """Question 2: every family against ring, as aggregate bus bandwidth, per shape."""
    index = _single_index(records)
    rows = []
    for group_size, size_bytes, workload in sorted(
        {k[1:] for k in index}, key=lambda k: (k[0], k[1], WORKLOADS.index(str(k[2])))
    ):
        ring = index.get(("ring", group_size, size_bytes, workload))
        ring_bw = ring and float(ring["aggregate_busbw_gb_s"])
        cells: list[str] = []
        for algo in ALGORITHMS:
            if algo == "ring":
                continue
            record = index.get((algo, group_size, size_bytes, workload))
            other_bw = record and float(record["aggregate_busbw_gb_s"])
            cells += [_fmt(other_bw), _fmt(other_bw / ring_bw if ring_bw and other_bw else None, 2)]
        rows.append([str(group_size), str(int(size_bytes) // BYTES_PER_MIB), str(workload), _fmt(ring_bw), *cells])
    return rows


def kernel_rows(records: Sequence[dict[str, object]]) -> list[list[str]]:
    """Question 2, second half: which concrete kernel each family actually ran.

    The kernel name comes from NCCL's own log lines: ``Algo ... proto ...`` for the classic
    collectives and ``[Symmetric]: ... -> Kernel ...`` for the window path. An empty cell means
    the log line was not found, so the case must not be reported as that family.
    """
    rows = []
    keys = {(str(r["algo"]), str(r["group_name"])) for r in records}
    for algo, group_name in sorted(keys, key=lambda k: (ALGORITHMS.index(k[0]), k[1])):
        mine = [r for r in records if r["algo"] == algo and r["group_name"] == group_name]
        evidence = {
            str(r.get("nccl_kernel") or r.get("nccl_algo")) for r in mine if r.get("nccl_kernel") or r.get("nccl_algo")
        }
        n_nodes = sorted({int(r["n_nodes"]) for r in mine if r.get("n_nodes") is not None})
        rows.append([algo, group_name, ",".join(map(str, n_nodes)) or "-", "; ".join(sorted(evidence)) or "-"])
    return rows


def split_rows(records: Sequence[dict[str, object]]) -> list[list[str]]:
    """Question 3: each half-group alone versus both half-groups at the same time."""
    fields = ("algo", "total_ranks", "layout", "size_bytes", "workload", "group_index")
    index = {_key(r, *fields, "phase"): r for r in records if r["scenario"] == "split"}
    rows = []
    for key in sorted(
        {k[:-1] for k in index},
        key=lambda k: (ALGORITHMS.index(str(k[0])), k[1], k[2], k[3], WORKLOADS.index(str(k[4])), k[5]),
    ):
        isolated = index.get((*key, "isolated"))
        concurrent = index.get((*key, "concurrent"))
        iso_bw = isolated and float(isolated["aggregate_busbw_gb_s"])
        conc_bw = concurrent and float(concurrent["aggregate_busbw_gb_s"])
        slowdown = None
        if isolated and concurrent:
            slowdown = float(concurrent["makespan_ms"]) / float(isolated["makespan_ms"])
        algo, total_ranks, layout, size_bytes, workload, group_index = key
        rows.append(
            [
                str(algo),
                str(total_ranks),
                str(layout),
                str(int(size_bytes) // BYTES_PER_MIB),
                str(workload),
                "AB"[int(group_index)],
                _fmt(iso_bw),
                _fmt(conc_bw),
                _fmt(slowdown, 3),
            ]
        )
    return rows


def render_report(records: Sequence[dict[str, object]]) -> str:
    """Render the three comparison tables as Markdown."""
    profiled = sorted({str(r["algo"]) for r in records if r.get("profiled")})
    lines = [
        "# AG / RS: ring vs NVLS vs symmetric memory",
        "",
        "busbw = size * (n-1)/n / t, decimal GB/s, median over trials; size is the full tensor.",
        "",
    ]
    if profiled:
        lines += [
            f"> 注意：{', '.join(profiled)} 是 `--nsys` 采集下的结果，含 Nsight 开销，只用于看 trace，"
            "不要和非 profile 的数字放进同一张结论表。",
            "",
        ]
    lines += [
        "## 1. AG、RS 单独与同时发起",
        "",
        "speedup = (T_AG + T_RS) / T_AG+RS；1 表示完全串行，overlap_fraction = 1 表示短的操作被完全隐藏。",
        "",
    ]
    lines += _table(
        [
            "algo",
            "ranks",
            "MiB",
            "AG busbw",
            "RS busbw",
            "AG in pair",
            "RS in pair",
            "pair aggregate",
            "speedup",
            "overlap_fraction",
        ],
        single_rows(records),
    )
    lines += ["", "## 1b. 各实现的绝对吞吐（aggregate busbw）", ""]
    lines += _table(["ranks", "MiB", "workload", *ALGORITHMS, "best"], _best_of(records))
    lines += [
        "",
        "## 2. 各实现相对 ring（aggregate busbw）",
        "",
        "symm 是 window 注册的对称内存路径，与 ring/NVLS 走不同内核家族；它的绝对列见 1b，",
        "上面的比值只是和 ring 放在同一口径下的对照。",
        "",
    ]
    lines += _table(["ranks", "MiB", "workload", "ring", "nvls", "nvls/ring", "symm", "symm/ring"], algo_rows(records))
    kernel_table = kernel_rows(records)
    if kernel_table:
        lines += [
            "",
            "### 2b. 实际执行的内核（来自 NCCL TUNING 日志，非环境变量推断）",
            "",
            "nNodes > 1 的 symm 行不是平铺对称内存结果：NCCL 2.28.9 对多节点 communicator 摘掉 LSA 内核。",
            "",
        ]
        lines += _table(["algo", "group", "nNodes", "kernel evidence"], kernel_table)
    lines += [
        "",
        "## 3. 两个半组同时运行",
        "",
        "slowdown = T_concurrent / T_isolated（同一组自身的 makespan），1 表示没有相互干扰。",
        "",
    ]
    lines += _table(
        ["algo", "total", "layout", "MiB", "workload", "group", "isolated busbw", "concurrent busbw", "slowdown"],
        split_rows(records),
    )
    return "\n".join(lines) + "\n"


def _parse_input(value: str) -> tuple[str, Path]:
    algo, sep, path = value.partition("=")
    if not sep or algo not in ALGORITHMS:
        raise argparse.ArgumentTypeError(f"expected ALGO=PATH with ALGO in {ALGORITHMS}, got {value!r}")
    return algo, Path(path)


def main(argv: Sequence[str] | None = None) -> None:
    """Merge per-algorithm result files and write the Markdown comparison."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=_parse_input, action="append", required=True, help="ALGO=results.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="Markdown report path")
    args = parser.parse_args(argv)

    records: list[dict[str, object]] = []
    for algo, path in args.input:
        loaded = load_records(path)
        mismatched = {str(r["algo"]) for r in loaded} - {algo}
        if mismatched:
            raise ValueError(f"{path} contains records for {sorted(mismatched)}, expected only {algo}")
        records.extend(loaded)
    args.output.write_text(render_report(records), encoding="utf-8")
    LOGGER.info("wrote %s (%d records)", args.output, len(records))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
