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

"""Summarize MLP fine-grained offload benchmark runs."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast


LOGGER = logging.getLogger(__name__)
RUN_NAME_PATTERN = re.compile(r"^(dense|expert)-(default|alltoall|hybridep)-(.+)-r(\d+)$")
STEP_TIME_PATTERN = re.compile(r"Step Time\s*:\s*([\d.]+)s")
ITERATION_TIME_PATTERN = re.compile(r"elapsed time per iteration \(ms\):\s*([\d.]+)")
GPU_UTILIZATION_PATTERN = re.compile(r"GPU utilization:\s*([\d.]+)")
LOSS_PATTERN = re.compile(r"lm loss:\s*([\d.E+\-]+)")
MAX_ALLOC_PATTERN = re.compile(r"mem-max-allocated-gigabytes:\s*([\d.]+)")
MAX_RESERVED_PATTERN = re.compile(r"mem-max-reserved-gigabytes:\s*([\d.]+)")


@dataclass(frozen=True)
class RunResult:
    """Metrics extracted from one benchmark repeat."""

    model_kind: str
    model: str
    precision: str
    profile: str
    dispatcher: str
    case: str
    repeat: int
    status: int
    samples: int
    mean_step_time_ms: float | None
    step_time_std_ms: float | None
    p95_step_time_ms: float | None
    tokens_per_second: float | None
    mean_tflops_per_gpu: float | None
    peak_gpu_memory_gib: float | None
    max_allocated_gib: float | None
    max_reserved_gib: float | None
    final_loss: float | None
    result_dir: str


@dataclass(frozen=True)
class AggregateResult:
    """Metrics averaged across successful repeats of one matrix case."""

    model_kind: str
    model: str
    precision: str
    profile: str
    dispatcher: str
    case: str
    successful_repeats: int
    total_repeats: int
    mean_step_time_ms: float | None
    repeat_step_time_std_ms: float | None
    tokens_per_second: float | None
    mean_tflops_per_gpu: float | None
    peak_gpu_memory_gib: float | None
    max_allocated_gib: float | None
    final_loss: float | None
    step_time_vs_baseline_pct: float | None
    throughput_vs_baseline_pct: float | None
    memory_saved_vs_baseline_gib: float | None
    memory_saved_vs_baseline_pct: float | None
    step_time_vs_alltoall_pct: float | None


def _load_json(path: Path) -> dict[str, object]:
    return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _stdev(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else (0.0 if values else None)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _metric_values(pattern: re.Pattern[str], text: str) -> list[float]:
    return [float(value) for value in pattern.findall(text)]


def _peak_gpu_memory(result_dir: Path) -> float | None:
    peaks_mib: list[float] = []
    for trace_path in sorted((result_dir / "gpu_memory").glob("gpu_*.json")):
        trace = _load_json(trace_path)
        value = trace.get("max_memory_used_mib")
        if isinstance(value, int | float):
            peaks_mib.append(float(value))
    return max(peaks_mib) / 1024.0 if peaks_mib else None


def _parse_identity(run_name: str) -> tuple[str, str, str, int]:
    match = RUN_NAME_PATTERN.fullmatch(run_name)
    if match is None:
        raise ValueError(f"Unrecognized benchmark run name: {run_name}")
    model_kind, dispatcher, case_name, repeat = match.groups()
    return model_kind, dispatcher, case_name, int(repeat)


def parse_run(config_path: Path) -> RunResult:
    """Parse one run directory into comparable metrics.

    Args:
        config_path: Path to the runner's config.json.

    Returns:
        Parsed metrics and run identity.
    """
    config = _load_json(config_path)
    result_dir = config_path.parent
    run_name = str(config["run_name"])
    model_kind, dispatcher_from_name, case_name, repeat = _parse_identity(run_name)
    dispatcher = str(config.get("dispatcher", dispatcher_from_name))
    summary_path = result_dir / "summary.json"
    status = int(_load_json(summary_path).get("status", 1)) if summary_path.exists() else 1
    log_path = result_dir / "train.log"
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""

    iteration_times = _metric_values(ITERATION_TIME_PATTERN, log_text)
    step_times_ms = iteration_times or [seconds * 1000.0 for seconds in _metric_values(STEP_TIME_PATTERN, log_text)]
    measured_step_times = step_times_ms
    measured_utilization = _metric_values(GPU_UTILIZATION_PATTERN, log_text)
    losses = _metric_values(LOSS_PATTERN, log_text)
    max_allocated = _metric_values(MAX_ALLOC_PATTERN, log_text)
    max_reserved = _metric_values(MAX_RESERVED_PATTERN, log_text)
    mean_step_time_ms = _mean(measured_step_times)
    global_batch_size = int(config.get("global_batch_size", 0))
    sequence_length = int(config.get("sequence_length", 4096))
    tokens_per_second = None
    if mean_step_time_ms and global_batch_size:
        tokens_per_second = global_batch_size * sequence_length * 1000.0 / mean_step_time_ms

    return RunResult(
        model_kind=model_kind,
        model=str(config["model"]),
        precision=str(config["precision"]),
        profile=str(config.get("profile", "none")),
        dispatcher=dispatcher,
        case=case_name,
        repeat=repeat,
        status=status,
        samples=len(measured_step_times),
        mean_step_time_ms=mean_step_time_ms,
        step_time_std_ms=_stdev(measured_step_times),
        p95_step_time_ms=_percentile(measured_step_times, 0.95),
        tokens_per_second=tokens_per_second,
        mean_tflops_per_gpu=_mean(measured_utilization),
        peak_gpu_memory_gib=_peak_gpu_memory(result_dir),
        max_allocated_gib=max(max_allocated) if max_allocated else None,
        max_reserved_gib=max(max_reserved) if max_reserved else None,
        final_loss=losses[-1] if losses else None,
        result_dir=str(result_dir),
    )


def discover_runs(results_root: Path, *, run_time: str) -> list[RunResult]:
    """Find and parse all benchmark runs belonging to a batch id."""
    runs: list[RunResult] = []
    for config_path in sorted(results_root.glob(f"*/*/*/{run_time}/config.json")):
        try:
            runs.append(parse_run(config_path))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            LOGGER.warning("Skipping %s: %s", config_path, exc)
    return runs


def _ratio_delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline in (None, 0.0):
        return None
    return (value / baseline - 1.0) * 100.0


def aggregate_runs(runs: list[RunResult]) -> list[AggregateResult]:
    """Aggregate repeats and compute baseline and dispatcher deltas."""
    grouped: dict[tuple[str, str, str, str, str, str], list[RunResult]] = {}
    for run in runs:
        key = (run.model_kind, run.model, run.precision, run.profile, run.dispatcher, run.case)
        grouped.setdefault(key, []).append(run)

    base_rows: dict[tuple[str, str, str, str, str, str], dict[str, float | int | None]] = {}
    for key, group in grouped.items():
        successful = [run for run in group if run.status == 0 and run.mean_step_time_ms is not None]
        base_rows[key] = {
            "successful_repeats": len(successful),
            "total_repeats": len(group),
            "mean_step_time_ms": _mean(
                [run.mean_step_time_ms for run in successful if run.mean_step_time_ms is not None]
            ),
            "repeat_step_time_std_ms": _stdev(
                [run.mean_step_time_ms for run in successful if run.mean_step_time_ms is not None]
            ),
            "tokens_per_second": _mean(
                [run.tokens_per_second for run in successful if run.tokens_per_second is not None]
            ),
            "mean_tflops_per_gpu": _mean(
                [run.mean_tflops_per_gpu for run in successful if run.mean_tflops_per_gpu is not None]
            ),
            "peak_gpu_memory_gib": _mean(
                [run.peak_gpu_memory_gib for run in successful if run.peak_gpu_memory_gib is not None]
            ),
            "max_allocated_gib": _mean(
                [run.max_allocated_gib for run in successful if run.max_allocated_gib is not None]
            ),
            "final_loss": _mean([run.final_loss for run in successful if run.final_loss is not None]),
        }

    aggregates: list[AggregateResult] = []
    for key, values in sorted(base_rows.items()):
        model_kind, model, precision, profile, dispatcher, case_name = key
        baseline = base_rows.get((model_kind, model, precision, profile, dispatcher, "baseline"), {})
        alltoall = base_rows.get((model_kind, model, precision, profile, "alltoall", case_name), {})
        peak_memory = cast(float | None, values["peak_gpu_memory_gib"])
        baseline_memory = cast(float | None, baseline.get("peak_gpu_memory_gib"))
        memory_saved = None
        memory_saved_pct = None
        if peak_memory is not None and baseline_memory is not None:
            memory_saved = baseline_memory - peak_memory
            memory_saved_pct = memory_saved / baseline_memory * 100.0 if baseline_memory else None
        aggregates.append(
            AggregateResult(
                model_kind=model_kind,
                model=model,
                precision=precision,
                profile=profile,
                dispatcher=dispatcher,
                case=case_name,
                successful_repeats=int(values["successful_repeats"]),
                total_repeats=int(values["total_repeats"]),
                mean_step_time_ms=cast(float | None, values["mean_step_time_ms"]),
                repeat_step_time_std_ms=cast(float | None, values["repeat_step_time_std_ms"]),
                tokens_per_second=cast(float | None, values["tokens_per_second"]),
                mean_tflops_per_gpu=cast(float | None, values["mean_tflops_per_gpu"]),
                peak_gpu_memory_gib=peak_memory,
                max_allocated_gib=cast(float | None, values["max_allocated_gib"]),
                final_loss=cast(float | None, values["final_loss"]),
                step_time_vs_baseline_pct=_ratio_delta(
                    cast(float | None, values["mean_step_time_ms"]),
                    cast(float | None, baseline.get("mean_step_time_ms")),
                ),
                throughput_vs_baseline_pct=_ratio_delta(
                    cast(float | None, values["tokens_per_second"]),
                    cast(float | None, baseline.get("tokens_per_second")),
                ),
                memory_saved_vs_baseline_gib=memory_saved,
                memory_saved_vs_baseline_pct=memory_saved_pct,
                step_time_vs_alltoall_pct=(
                    _ratio_delta(
                        cast(float | None, values["mean_step_time_ms"]),
                        cast(float | None, alltoall.get("mean_step_time_ms")),
                    )
                    if dispatcher == "hybridep"
                    else None
                ),
            )
        )
    return aggregates


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _format(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _format_percent(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}%"


def _write_markdown(path: Path, aggregates: list[AggregateResult], *, run_time: str) -> None:
    lines = [
        f"# MLP offload benchmark `{run_time}`",
        "",
        "Metrics include every logged training iteration. Positive step-time delta is slower; positive memory saved is better.",
        "",
        "| Model | Profile | Dispatcher | Case | Repeats | Step ms | vs baseline | Tokens/s | TFLOP/s/GPU | Peak HBM GiB | HBM saved | vs alltoall |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.model,
                    row.profile,
                    row.dispatcher,
                    row.case,
                    f"{row.successful_repeats}/{row.total_repeats}",
                    _format(row.mean_step_time_ms),
                    _format_percent(row.step_time_vs_baseline_pct),
                    _format(row.tokens_per_second, 0),
                    _format(row.mean_tflops_per_gpu),
                    _format(row.peak_gpu_memory_gib),
                    f"{_format(row.memory_saved_vs_baseline_gib)} GiB",
                    _format_percent(row.step_time_vs_alltoall_pct),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Bottleneck isolation",
            "",
            "- `baseline -> offload` measures fine-grained activation offload with activation recompute disabled.",
            "- Dense offload moves `mlp_norm` and `mlp_act`; expert offload moves `mlp_norm`, `expert_fc1`, and `moe_act`.",
            "- For expert rows, compare HybridEP with all-to-all for the same case; compare that delta with the baseline dispatcher delta to identify an offload/dispatcher interaction.",
            "- Timing and HBM identify the tradeoff, but not copy overlap. Use the matching nsys or torch trace before attributing a regression to D2H/H2D or synchronization.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--run-time", required=True, help="Batch id stored in config.json")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main() -> int:
    """Summarize one benchmark batch."""
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    runs = discover_runs(args.results_root, run_time=args.run_time)
    if not runs:
        LOGGER.error("No benchmark runs found for run_time=%s under %s", args.run_time, args.results_root)
        return 1
    aggregates = aggregate_runs(runs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "runs.csv", [asdict(run) for run in runs])
    _write_csv(args.output_dir / "summary.csv", [asdict(row) for row in aggregates])
    (args.output_dir / "summary.json").write_text(
        json.dumps([asdict(row) for row in aggregates], indent=2) + "\n", encoding="utf-8"
    )
    _write_markdown(args.output_dir / "summary.md", aggregates, run_time=args.run_time)
    LOGGER.info("Wrote %d runs and %d aggregate rows to %s", len(runs), len(aggregates), args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
