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
# Split cases: each half-group alone, both half-groups running the same workload at the same
# time, and "cross": group A all-gathers while group B reduce-scatters, at the same time.
SPLIT_PARTS = ("A-isolated", "B-isolated", "AB-concurrent", "AB-cross")
CROSS_PART = "AB-cross"
CROSS_WORKLOAD = "agA-rsB"
CROSS_GROUP_WORKLOADS = ("ag", "rs")  # per split group, A then B
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


def scenario_name(scenario: str, *, total_ranks: int, layout: str | None = None) -> str:
    """Name one set of communicators: ``single-n8`` or ``split-n16-contiguous``."""
    if scenario == "single" and layout is None:
        return f"single-n{total_ranks}"
    if scenario == "split" and layout in SPLIT_LAYOUTS:
        return f"split-n{total_ranks}-{layout}"
    raise ValueError(f"unknown scenario {scenario!r} with layout {layout!r}")


def case_name(scenario: str, *, part: str | None, workload: str, dtype: str, size_mib: int) -> str:
    """Name one test, e.g. ``split-n16-contiguous-AB-concurrent_ag+rs_bf16_1024MiB``.

    The name is the case directory, the stem of every per-rank file in it, and (prefixed with
    the algorithm) the outer NVTX range of the case's nsys report, so it must stay a single
    path component. ``size_mib`` is the full tensor: AG output / RS input. The cross part takes
    ``CROSS_WORKLOAD`` (``agA-rsB``), every other case one of ``WORKLOADS``.
    """
    if part is not None and part not in SPLIT_PARTS:
        raise ValueError(f"unknown split part {part!r}; expected one of {SPLIT_PARTS}")
    if (part == CROSS_PART) != (workload == CROSS_WORKLOAD):
        raise ValueError(f"workload {CROSS_WORKLOAD!r} belongs to part {CROSS_PART!r} only, got {part!r}/{workload!r}")
    if workload != CROSS_WORKLOAD:
        workload_ops(workload)
    prefix = scenario if part is None else f"{scenario}-{part}"
    return f"{prefix}_{workload}_{dtype}_{size_mib}MiB"


def rank_file(algo: str, name: str, *, rank: int, world_size: int, suffix: str) -> str:
    """Per-rank file name ``<algo>_<name>_rank<RR><suffix>``, zero-padded so files sort by rank."""
    width = max(2, len(str(world_size - 1)))
    return f"{algo}_{name}_rank{rank:0{width}d}{suffix}"


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


def trial_concurrency(spans_by_group: Sequence[Sequence[float]]) -> list[float]:
    """Per trial, ``min / max`` of the groups' times measured from the shared start gate.

    ``spans_by_group[g][t]`` is group ``g``'s slowest-rank time in trial ``t``. Because every
    group starts at the gate, the shorter group was running during exactly that share of the
    longer group's time: 1 means both groups moved data for the whole trial.
    """
    if len(spans_by_group) < 2:
        raise ValueError("concurrency needs at least two groups")
    return [min(spans) / max(spans) for spans in zip(*spans_by_group, strict=True)]


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
    collectives and ``[Symmetric]: ... -> Kernel ...`` for the window path. ``nccl_impl`` is
    the value the benchmark checked against every launch of the case; older records only
    carry the last line seen. An empty cell means the log line was not found, so the case
    must not be reported as that family.
    """
    rows = []
    keys = {(str(r["algo"]), str(r["group_name"])) for r in records}
    for algo, group_name in sorted(keys, key=lambda k: (ALGORITHMS.index(k[0]), k[1])):
        mine = [r for r in records if r["algo"] == algo and r["group_name"] == group_name]
        evidence = {
            str(found) for r in mine if (found := r.get("nccl_impl") or r.get("nccl_kernel") or r.get("nccl_algo"))
        }
        n_nodes = sorted({int(r["n_nodes"]) for r in mine if r.get("n_nodes") is not None})
        rows.append([algo, group_name, ",".join(map(str, n_nodes)) or "-", "; ".join(sorted(evidence)) or "-"])
    return rows


def split_rows(records: Sequence[dict[str, object]]) -> list[list[str]]:
    """Question 3: each half-group alone versus both half-groups at the same time."""
    fields = ("algo", "total_ranks", "layout", "size_bytes", "workload", "group_index")
    index = {
        _key(r, *fields, "phase"): r
        for r in records
        if r["scenario"] == "split" and r["phase"] in ("isolated", "concurrent")
    }
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
                _fmt(concurrent and _optional_float(concurrent.get("concurrency")), 3),
            ]
        )
    return rows


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)  # type: ignore[arg-type]


def cross_rows(records: Sequence[dict[str, object]]) -> list[list[str]]:
    """Question 4: group A all-gathers while group B reduce-scatters, each against itself alone.

    The isolated references are the split cases where A ran ``ag`` alone and B ran ``rs`` alone;
    ``slowdown = T_cross / T_isolated`` of the same group and op.
    """
    fields = ("algo", "total_ranks", "layout", "size_bytes", "phase", "group_index", "workload")
    index = {_key(r, *fields): r for r in records if r["scenario"] == "split"}
    shapes = {key[:4] for key in index if key[4] == "cross"}
    rows = []
    for shape in sorted(shapes, key=lambda k: (ALGORITHMS.index(str(k[0])), *k[1:])):
        cells: list[str] = []
        cross_bw = 0.0
        for group_index, workload in enumerate(CROSS_GROUP_WORKLOADS):
            cross = index.get((*shape, "cross", group_index, workload))
            isolated = index.get((*shape, "isolated", group_index, workload))
            slowdown = None
            if cross and isolated:
                slowdown = float(cross["makespan_ms"]) / float(isolated["makespan_ms"])
            if cross:
                cross_bw += float(cross["aggregate_busbw_gb_s"])
            cells += [
                _fmt(isolated and float(isolated["aggregate_busbw_gb_s"])),
                _fmt(cross and float(cross["aggregate_busbw_gb_s"])),
                _fmt(cross and float(cross["makespan_ms"]), 3),
                _fmt(slowdown, 3),
            ]
        any_cross = index.get((*shape, "cross", 0, CROSS_GROUP_WORKLOADS[0])) or index.get(
            (*shape, "cross", 1, CROSS_GROUP_WORKLOADS[1])
        )
        algo, total_ranks, layout, size_bytes = shape
        rows.append(
            [
                str(algo),
                str(total_ranks),
                str(layout),
                str(int(size_bytes) // BYTES_PER_MIB),
                *cells,
                _fmt(cross_bw),
                _fmt(any_cross and _optional_float(any_cross.get("concurrency")), 3),
            ]
        )
    return rows


def completion_rows(records: Sequence[dict[str, object]]) -> list[list[str]]:
    """Table 0: every test and group with its completion time per iteration over the trials.

    ``mean`` is the arithmetic mean over trials of the slowest rank's per-iteration time; the
    bandwidth columns in this table are computed from that mean, every op of the workload
    contributing ``size`` (algbw) or ``size * (n-1)/n`` (busbw).
    """
    rows = []
    ordered = sorted(
        records,
        key=lambda r: (ALGORITHMS.index(str(r["algo"])), str(r.get("case") or ""), str(r["group_name"])),
    )
    for r in ordered:
        trials = [float(t) for t in r["trial_makespan_ms"]]  # type: ignore[union-attr]
        mean_ms = float(r.get("mean_makespan_ms") or sum(trials) / len(trials))  # type: ignore[arg-type]
        size_bytes, group_size = int(r["size_bytes"]), int(r["group_size"])  # type: ignore[arg-type]
        n_ops = len(workload_ops(str(r["workload"])))
        algbw = n_ops * alg_bandwidth_gb_s(total_bytes=size_bytes, seconds=mean_ms / 1e3)
        rows.append(
            [
                str(r["algo"]),
                str(r.get("case") or "-"),
                str(r["group_name"]),
                str(r.get("phase") or "-"),
                str(group_size),
                str(r["workload"]),
                str(size_bytes // BYTES_PER_MIB),
                f"{len(trials)}x{r['iters']}",
                _fmt(mean_ms, 3),
                _fmt(float(r["makespan_ms"]), 3),  # type: ignore[arg-type]
                _fmt(min(trials), 3),
                _fmt(max(trials), 3),
                _fmt(algbw),
                _fmt(algbw * (group_size - 1) / group_size),
                _fmt(_optional_float(r.get("concurrency")), 3),
            ]
        )
    return rows


def render_report(records: Sequence[dict[str, object]]) -> str:
    """Render the comparison tables as Markdown."""
    profiled = sorted({str(r["algo"]) for r in records if r.get("profiled")})
    lines = [
        "# AG / RS: ring vs NVLS vs symmetric memory",
        "",
        "计时：每个 trial 先过一次全 scenario 的 GPU 同步起点（1 元素 all-reduce），再背靠背发 iters 次；",
        "t = (该 rank 所有 op 完成 − 起点) / iters，trial 内取组内最慢 rank。除表 0 外各表取 trials 的中位数。",
        "",
        "换算：size = 完整张量（AG 每卡接收总量 / RS 每卡输入总量），每卡 shard = size/n，十进制 GB/s：",
        "algbw = size / t；busbw = size·(n−1)/n / t（nccl-tests 定义，= 每卡实际收/发字节数 / t）；",
        "ag+rs 的 aggregate busbw = 两个 op 的 size·(n−1)/n 之和 / makespan；不同组的带宽不相加，需要时自己把各组行相加。",
        "concurrency = min(T_A, T_B) / max(T_A, T_B)：两组从同一起点出发，1 表示两组全程同时在跑。",
        "",
    ]
    if profiled:
        lines += [
            f"> 注意：{', '.join(profiled)} 是 `--nsys` 采集下的结果，含 Nsight 开销，只用于看 trace，"
            "不要和非 profile 的数字放进同一张结论表。",
            "",
        ]
    lines += [
        "## 0. 每个测试的完成时间",
        "",
        "mean/median/min/max：各 trial（每个 trial 取组内最慢 rank）的每次迭代完成时间，单位 ms；",
        "本表的 algbw/busbw 用 mean 计算。",
        "",
    ]
    lines += _table(
        [
            "algo",
            "case",
            "group",
            "phase",
            "ranks",
            "workload",
            "MiB",
            "trials x iters",
            "mean ms",
            "median ms",
            "min ms",
            "max ms",
            "algbw",
            "busbw",
            "concurrency",
        ],
        completion_rows(records),
    )
    lines += [
        "",
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
            "每个 case 的每次 launch 都已核对：ring 只能是 Ring/*，nvls 只能是 NVLS/*，symm 只能是对称内核，"
            "任何 CopyEngine 或回退到经典内核的 launch 都会让该 case 失败，不会出现在这里。",
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
        [
            "algo",
            "total",
            "layout",
            "MiB",
            "workload",
            "group",
            "isolated busbw",
            "concurrent busbw",
            "slowdown",
            "concurrency",
        ],
        split_rows(records),
    )
    lines += [
        "",
        "## 4. 两组同时：A 组 all-gather，B 组 reduce-scatter",
        "",
        "isolated 是同一组单独跑同一个 op（split 的 A-isolated ag、B-isolated rs）；"
        "slowdown = T_cross / T_isolated；A+B busbw 是两组 busbw 之和。",
        "",
    ]
    lines += _table(
        [
            "algo",
            "total",
            "layout",
            "MiB",
            "A AG isolated busbw",
            "A AG cross busbw",
            "A AG cross ms",
            "A slowdown",
            "B RS isolated busbw",
            "B RS cross busbw",
            "B RS cross ms",
            "B slowdown",
            "A+B cross busbw",
            "concurrency",
        ],
        cross_rows(records),
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
