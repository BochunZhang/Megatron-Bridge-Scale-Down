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
"""Plot the AllGather_STMC pause sweep: per-iteration AG / RS completion inside ``ag+rs``.

Reads a ``run_symm_cta_pause_sweep.sh`` results tree (``gpu-num-<N>/<id>-pause<C>cyc/symm/``, or
its ``simplify_results.py`` copy) and, for every GPU count, draws one PNG with three curves over the
pause: the ag+rs makespan and, inside the same iterations, when the AG and when the RS finished.

The benchmark's own ``ag_ms`` / ``rs_ms`` of an ag+rs record cannot give the last two: ag+rs runs
in lockstep (each iteration starts once both ops of the previous one ended), and ``<op>_ms`` is the
time from the trial start to the op's *last* end divided by ``iters``, so it contains ``iters - 1``
whole makespans. The script therefore reads rank 0's nsys report of the ag+rs case instead. Every
iteration's origin is the end of the previous iteration (both ops ended; the trial's gate
all-reduce for the first one), and::

    ag_done = AG kernel end - origin,  rs_done = RS kernel end - origin,  makespan = max of the two

Each trial averages its iterations and the trials are summarised by their median, like
``compare.md``. The sum of the per-iteration makespans is checked against the event-timed
``makespan_ms`` of ``results.jsonl``.

The report is read with Nsight Systems' ``DumpTimeline`` (shipped with the macOS host app; pass
``--dump-timeline`` elsewhere), so no ``nsys export`` is needed. ``matplotlib`` must be importable::

    python plot_pause_sweep.py --results-dir results/nvls-collective-1024MiB-2.32.3-pause-simplify-1 \\
        --plots-dir documents/gb200/figures/03-pause-cycle

Outputs ``pause-sweep-gpu-<N>.png`` per GPU count, ``pause_sweep_iterations.csv`` (one row per
GPU count, pause, trial and iteration) and ``pause_sweep_summary.csv`` (one row per GPU count and
pause) under ``--plots-dir``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


LOGGER = logging.getLogger(__name__)

DEFAULT_DUMP_TIMELINE = "/Applications/NVIDIA Nsight Systems.app/Contents/MacOS/DumpTimeline"
_RUN_DIR = re.compile(r"-pause(\d+)cyc$")
_ALL_STREAMS_ROW = re.compile(r"/AllStreams;$")
# Kernel name prefix -> role. The gate is the 1-element all-reduce every trial starts with.
_KERNEL_ROLE = {
    "ncclSymkDevKernel_AllGather_STMC": "ag",
    "ncclSymkDevKernel_ReduceScatter_LDMC": "rs",
    "ncclDevKernel_AllReduce": "gate",
}
# Categorical slots 1-3 of the dataviz reference palette (validated all-pairs, light mode).
_SERIES = (
    ("makespan", "AG+RS makespan", "#2a78d6", "o"),
    ("ag_done", "AG done (in AG+RS)", "#eb6834", "s"),
    ("rs_done", "RS done (in AG+RS)", "#1baf7a", "^"),
)
_TEXT_PRIMARY = "#0b0b0b"
_TEXT_SECONDARY = "#52514e"
_GRID = "#e4e3df"


def _kernels(report: Path, dump_timeline: str) -> list[tuple[str, int, int]]:
    """Return ``(role, start_ns, end_ns)`` of every AG, RS and gate kernel in ``report``, by start."""
    out = subprocess.run(
        [dump_timeline, "-i", str(report), "--full", "--format", "n", "--correlation", "0"],
        check=True,
        capture_output=True,
    ).stdout
    kernels = []
    # [All Streams] holds every kernel once; DumpTimeline folds the less busy per-stream rows into an
    # aggregate row without their kernels, so those rows can miss an op.
    for row in json.loads(out)["GenericHierarchyRows"]:
        if not _ALL_STREAMS_ROW.search(row["Path"]):
            continue
        for sub in row["Rows"]:
            for element in sub.get("ViewDataElements", []):
                role = next((r for p, r in _KERNEL_ROLE.items() if element["Text"].startswith(p)), None)
                if role:
                    kernels.append((role, element["Start"], element["Start"] + element["Duration"]))
    return sorted(kernels, key=lambda k: k[1])


def _iterations(kernels: list[tuple[str, int, int]], iters: int, trials: int) -> list[dict[str, float]]:
    """Split the kernels into trials at the gates and time every lockstep iteration, in ms."""
    gates = [k for k in kernels if k[0] == "gate"]
    if len(gates) != trials:
        raise ValueError(f"expected {trials} gate all-reduces, found {len(gates)}")
    rows = []
    for t, gate in enumerate(gates):
        until = gates[t + 1][1] if t + 1 < len(gates) else float("inf")
        ops = {op: [k for k in kernels if k[0] == op and gate[2] <= k[1] < until] for op in ("ag", "rs")}
        if any(len(v) != iters for v in ops.values()):
            raise ValueError(f"trial {t}: expected {iters} AG and RS kernels, found {[len(v) for v in ops.values()]}")
        origin = gate[2]
        for i, (ag, rs) in enumerate(zip(ops["ag"], ops["rs"])):
            end = max(ag[2], rs[2])
            rows.append(
                {
                    "trial": t,
                    "iter": i,
                    "ag_done": (ag[2] - origin) / 1e6,
                    "rs_done": (rs[2] - origin) / 1e6,
                    "makespan": (end - origin) / 1e6,
                    "ag_kernel": (ag[2] - ag[1]) / 1e6,
                    "rs_kernel": (rs[2] - rs[1]) / 1e6,
                }
            )
            origin = end
    return rows


def _records(run: Path) -> dict[str, dict]:
    """Return the ag / rs / ag+rs records of the run's ``symm/results.jsonl`` by workload."""
    lines = (run / "symm" / "results.jsonl").read_text(encoding="utf-8").splitlines()
    return {r["workload"]: r for r in map(json.loads, lines) if r.get("scenario") == "single"}


def _summarise(n: int, pause: int, iterations: list[dict[str, float]], records: dict[str, dict]) -> dict:
    """Median over trials of each trial's mean, next to the event-timed numbers of results.jsonl."""
    by_trial = defaultdict(list)
    for row in iterations:
        by_trial[row["trial"]].append(row)
    keys = ("makespan", "ag_done", "rs_done", "ag_kernel", "rs_kernel")
    summary = {"gpus": n, "pause_cycles": pause}
    for key in keys:
        summary[f"{key}_ms"] = median(mean(r[key] for r in rows) for rows in by_trial.values())
    summary["ag_first_fraction"] = mean(r["ag_done"] < r["rs_done"] for r in iterations)
    pair = records["ag+rs"]
    summary["event_makespan_ms"] = pair["makespan_ms"]
    summary["event_ag_ms"] = pair["ag_ms"]
    summary["event_rs_ms"] = pair["rs_ms"]
    summary["ag_alone_ms"] = records["ag"]["makespan_ms"]
    summary["rs_alone_ms"] = records["rs"]["makespan_ms"]
    return summary


def _plot(n: int, rows: list[dict], path: Path) -> None:
    """Draw the three curves of one GPU count over the pause."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda r: r["pause_cycles"])
    x = [r["pause_cycles"] for r in rows]
    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for key, label, colour, marker in _SERIES:
        y = [r[f"{key}_ms"] for r in rows]
        if key == "makespan":
            # The makespan is the later of the two ops, so it runs under one of them all along: draw it
            # as a wide band beneath them and mark its minimum instead of labelling its end.
            ax.plot(x, y, color=colour, linewidth=9, alpha=0.3, solid_capstyle="round", label=label)
            best = min(range(len(y)), key=y.__getitem__)
            ax.plot(x[best], y[best], marker="o", markersize=13, markerfacecolor="none", color=colour, mew=2)
            ax.annotate(
                f"min {y[best]:.3f} ms @ {x[best]} cycles\n(pause 0: {y[0]:.3f} ms)",
                (x[best], y[best]),
                # Below every curve: the lowest of them stays above ~1.5 ms.
                xytext=(x[best], 0.4 * y[best]),
                ha="center",
                va="top",
                fontsize=9,
                color=_TEXT_PRIMARY,
                arrowprops={"arrowstyle": "-", "color": _TEXT_SECONDARY, "linewidth": 0.8, "shrinkB": 8},
            )
            continue
        # The surface-coloured marker edge keeps overlapping markers apart.
        ax.plot(
            x,
            y,
            color=colour,
            linewidth=2,
            marker=marker,
            markersize=7,
            markeredgecolor="#fcfcfb",
            markeredgewidth=1.5,
            label=label,
        )
        ax.annotate(
            label.split(" (")[0],
            (x[-1], y[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=_TEXT_SECONDARY,
        )
    ax.set_title(
        f"{n} GPUs, 1024 MiB bf16: AG+RS per iteration vs AllGather_STMC pause",
        loc="left",
        fontsize=11,
        color=_TEXT_PRIMARY,
    )
    ax.set_xlabel("pause cycle (SM cycles per warp after each 4 KiB multimem store chunk)", color=_TEXT_SECONDARY)
    ax.set_ylabel("time per iteration (ms)", color=_TEXT_SECONDARY)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0, right=x[-1] * 1.13)
    ax.grid(axis="y", color=_GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.tick_params(colors=_TEXT_SECONDARY, labelsize=9)
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=_TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    """Extract every ag+rs report of the sweep, write the CSVs and one plot per GPU count."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, required=True, help="directory holding gpu-num-<N>/")
    parser.add_argument("--plots-dir", type=Path, required=True)
    parser.add_argument("--dump-timeline", default=DEFAULT_DUMP_TIMELINE, help="Nsight Systems DumpTimeline")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    summaries, all_iterations = [], []
    for gpu_dir in sorted(args.results_dir.glob("gpu-num-*"), key=lambda p: int(p.name.split("-")[-1])):
        n = int(gpu_dir.name.split("-")[-1])
        for run in sorted(gpu_dir.iterdir()):
            match = _RUN_DIR.search(run.name)
            if not match:
                continue
            pause = int(match.group(1))
            records = _records(run)
            pair = records["ag+rs"]
            reports = list((run / "symm" / pair["case_dir"]).glob("*.rank_00.nsys-rep"))
            if len(reports) != 1:
                raise FileNotFoundError(f"{run}: expected one rank 0 ag+rs report, found {reports}")
            iterations = _iterations(_kernels(reports[0], args.dump_timeline), pair["iters"], pair["trials"])
            summary = _summarise(n, pause, iterations, records)
            LOGGER.info(
                "%2d GPUs pause %5d: makespan %.3f (events %.3f) AG %.3f RS %.3f ms",
                n,
                pause,
                summary["makespan_ms"],
                summary["event_makespan_ms"],
                summary["ag_done_ms"],
                summary["rs_done_ms"],
            )
            summaries.append(summary)
            all_iterations += [{"gpus": n, "pause_cycles": pause, **row} for row in iterations]

    for name, rows in (("pause_sweep_summary.csv", summaries), ("pause_sweep_iterations.csv", all_iterations)):
        with (args.plots_dir / name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for n in sorted({s["gpus"] for s in summaries}):
        _plot(n, [s for s in summaries if s["gpus"] == n], args.plots_dir / f"pause-sweep-gpu-{n}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
