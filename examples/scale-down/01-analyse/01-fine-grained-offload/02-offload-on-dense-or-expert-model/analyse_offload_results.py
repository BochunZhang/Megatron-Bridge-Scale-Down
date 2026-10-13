#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Compare selective activation offload against its baseline per run group.

Reads every successful, non-profiled run under the results root, groups the runs
by (model, dtype, gpu_count, pp, ep, gbs, mbs, dispatcher, sharding), and pairs
each offload run with the baseline that shares its group. The grouping key
includes the launch scale on purpose: a 4-GPU attribution run and a 32-GPU
pipeline run must never be paired together just because both are "baseline".

Reports:
    Summary  mean TFlops for every case in a group and the drop against baseline.
    Samples  the per-iteration samples behind each mean, with variance.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, variance


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]
DEFAULT_RESULTS_ROOT = (
    REPO_ROOT
    / "results/01-analyse/01-fine-grained-offload/02-offload-on-dense-or-expert-model"
)
# One sample per logged training iteration; the analyser needs enough of them to
# skip warmup and still average a stable window.
MIN_SAMPLES = 10
SAMPLES_TO_AVERAGE = 4
REQUIRED_KEYS = ("model", "dtype", "gpu_count", "pp", "ep", "gbs", "mbs", "dispatcher", "sharding")

SUMMARY_FIELDS = (
    "model",
    "dtype",
    "gpu_count",
    "pp",
    "ep",
    "gbs",
    "mbs",
    "dispatcher",
    "sharding",
    "case",
    "run_name",
    "mean TFlops/GPU",
    "vs baseline",
    "sample count",
)


@dataclass(frozen=True)
class Run:
    """One successful, non-profiled run."""

    config: dict
    result_dir: Path
    run_time: str

    @property
    def group(self) -> tuple:
        """Grouping key; every field here must match for a run to be paired."""
        return tuple(self.config[key] for key in REQUIRED_KEYS)

    @property
    def is_baseline(self) -> bool:
        """Whether this run is an offload-free baseline."""
        return str(self.config["run_name"]).startswith("baseline")


def read_json(path: Path) -> dict:
    """Read a JSON object."""
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def read_samples(result_dir: Path) -> tuple[float, tuple[float, ...]]:
    """Average the last few logged iterations of a run.

    Returns:
        The mean of the averaged window and the window itself.
    """
    metrics_path = result_dir / "gpu_utilization.json"
    if not metrics_path.exists():
        raise ValueError(f"{metrics_path} is missing; the run did not log utilization")
    metrics = read_json(metrics_path)
    samples: list[tuple[int, float]] = []
    for raw_step, raw_value in metrics.items():
        try:
            step = int(raw_step)
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{metrics_path} has a non-numeric entry: {raw_step!r}") from exc
        if step < 0 or not math.isfinite(value):
            raise ValueError(f"{metrics_path} has an invalid entry at step {raw_step!r}")
        samples.append((step, value))
    samples.sort(key=lambda item: item[0])
    if len(samples) < MIN_SAMPLES:
        raise ValueError(
            f"{metrics_path} has {len(samples)} samples; at least {MIN_SAMPLES} are required"
        )
    window = tuple(value for _, value in samples[-SAMPLES_TO_AVERAGE:])
    return mean(window), window


def discover_runs(results_root: Path) -> list[Run]:
    """Find every successful run that should take part in a comparison."""
    runs: list[Run] = []
    for config_path in sorted(results_root.rglob("config.json")):
        config = read_json(config_path)
        if str(config.get("profile", "none")) != "none":
            continue
        missing = [key for key in REQUIRED_KEYS if key not in config]
        if missing:
            raise ValueError(f"{config_path} is missing keys: {', '.join(missing)}")
        if not str(config.get("run_name", "")).strip():
            raise ValueError(f"{config_path} has an empty run_name")
        result_dir = config_path.parent
        summary_path = result_dir / "summary.json"
        if not summary_path.exists():
            continue
        if int(read_json(summary_path).get("status", 1)) != 0:
            continue
        runs.append(Run(config=config, result_dir=result_dir, run_time=str(config.get("run_time", ""))))
    return runs


def build_summary(runs: list[Run]) -> list[dict]:
    """Pair each offload run with its baseline inside the same group."""
    groups: dict[tuple, list[Run]] = {}
    for run in runs:
        groups.setdefault(run.group, []).append(run)

    rows: list[dict] = []
    for group, members in sorted(groups.items(), key=lambda item: str(item[0])):
        baselines = [run for run in members if run.is_baseline]
        if not baselines:
            raise ValueError(f"group {group} has no baseline run; cannot compute a drop")
        # Prefer the newest baseline, but only pair runs that share a batch id when
        # the batch id is uniform, so a mixed-result-root sweep cannot silently mix
        # runs from different campaigns.
        baselines.sort(key=lambda run: run.run_time)
        baseline = baselines[-1]
        baseline_mean, _ = read_samples(baseline.result_dir)
        if baseline_mean <= 0:
            raise ValueError(f"baseline for {group} has non-positive mean TFlops")

        base_row = _row(group, baseline, baseline_mean, 0.0)
        rows.append(base_row)
        for run in sorted(members, key=lambda item: str(item.config["run_name"])):
            if run is baseline:
                continue
            run_mean, _ = read_samples(run.result_dir)
            drop = (baseline_mean - run_mean) / baseline_mean
            rows.append(_row(group, run, run_mean, drop))
    return rows


def _row(group: tuple, run: Run, run_mean: float, drop: float) -> dict:
    row = dict(zip(REQUIRED_KEYS, group))
    row.update(
        {
            "case": "baseline" if run.is_baseline else "offload",
            "run_name": run.config["run_name"],
            "mean TFlops/GPU": run_mean,
            "vs baseline": drop,
            "sample count": SAMPLES_TO_AVERAGE,
        }
    )
    return row


def write_csv(rows: list[dict], output: Path) -> None:
    """Write the summary as CSV."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_samples_csv(runs: list[Run], output: Path) -> None:
    """Write the raw per-iteration samples behind each mean."""
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        *REQUIRED_KEYS,
        "run_name",
        "run_time",
        *(f"iteration {index + 1} (TFlops)" for index in range(SAMPLES_TO_AVERAGE)),
        "mean TFlops/GPU",
        "variance",
    )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in sorted(runs, key=lambda item: (str(item.group), str(item.config["run_name"]))):
            _, window = read_samples(run.result_dir)
            row = dict(zip(REQUIRED_KEYS, run.group))
            row["run_name"] = run.config["run_name"]
            row["run_time"] = run.run_time
            for index, value in enumerate(window):
                row[f"iteration {index + 1} (TFlops)"] = value
            row["mean TFlops/GPU"] = mean(window)
            row["variance"] = variance(window) if len(window) > 1 else 0.0
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=f"Result tree root (default: {DEFAULT_RESULTS_ROOT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV summary path (default: <results-root>/offload-summary.csv)",
    )
    parser.add_argument(
        "--samples-output",
        type=Path,
        default=None,
        help="CSV sample path (default: <results-root>/offload-samples.csv)",
    )
    parser.add_argument("--model", default=None, help="Only analyse runs whose model matches")
    return parser


def main() -> int:
    """Write the comparison reports."""
    args = build_parser().parse_args()
    results_root = args.results_root.resolve()
    if not results_root.exists():
        print(f"results root does not exist: {results_root}", file=sys.stderr)
        return 2
    runs = discover_runs(results_root)
    if args.model is not None:
        runs = [run for run in runs if run.config["model"] == args.model]
    if not runs:
        print(f"no successful non-profiled runs under {results_root}", file=sys.stderr)
        return 1

    rows = build_summary(runs)
    summary_path = args.output or (results_root / "offload-summary.csv")
    samples_path = args.samples_output or (results_root / "offload-samples.csv")
    write_csv(rows, summary_path)
    write_samples_csv(runs, samples_path)

    for row in rows:
        print(
            f"{row['model']} gpu={row['gpu_count']} pp={row['pp']} ep={row['ep']} "
            f"mbs={row['mbs']} {row['dispatcher']} {row['run_name']}: "
            f"{row['mean TFlops/GPU']:.1f} TFlops/GPU "
            f"({row['vs baseline'] * 100:+.1f}% vs baseline)"
        )
    print(f"\nsummary: {summary_path}\nsamples: {samples_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
