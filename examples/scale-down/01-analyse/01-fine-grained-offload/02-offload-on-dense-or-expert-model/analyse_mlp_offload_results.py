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

"""Analyze GPU utilization samples and write an MLP throughput XLSX workbook."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

from collect_mlp_offload_results import (
    DEFAULT_RESULTS_ROOT,
    column_name,
    content_types_xml,
    inline_string_cell,
    make_sheet_xml,
    number_cell,
    package_rels_xml,
    styles_xml,
)


MIN_GPU_UTILIZATION_SAMPLES = 10
GPU_UTILIZATION_SAMPLES_TO_AVERAGE = 4
RUN_NAME_PATTERN = re.compile(
    r"^(dense|expert)-(default|alltoall|hybridep)-(baseline|offload-mlp|offload-attn-mlp)(?:-mbs(\d+))?-r\d+$"
)
HEADERS = (
    "model",
    "dispatcher",
    "mbs",
    "dtype",
    "type",
    "baseline (TFlops)",
    "offload-mlp (TFlops)",
    "offload-attn-mlp (TFlops)",
    "offload-mlp 相对于 baseline 的性能下降幅度",
    "offload-attn-mlp 相对于 baseline 的性能下降幅度",
)


@dataclass(frozen=True)
class Identity:
    """Validated metadata identifying one benchmark run."""

    model: str
    dispatcher: str
    mbs: int
    dtype: str
    model_type: str
    case_name: str
    run_time: str


@dataclass(frozen=True)
class Candidate:
    """A run identity and its result directory."""

    identity: Identity
    result_dir: Path


@dataclass(frozen=True)
class ThroughputRow:
    """Comparison of the latest baseline and both offload runs."""

    model: str
    dispatcher: str
    mbs: int
    dtype: str
    model_type: str
    baseline_tflops: float
    offload_mlp_tflops: float
    offload_attn_mlp_tflops: float
    offload_mlp_performance_drop: float
    offload_attn_mlp_performance_drop: float
    baseline_run_time: str
    offload_mlp_run_time: str
    offload_attn_mlp_run_time: str
    baseline_samples: tuple[float, ...]
    offload_mlp_samples: tuple[float, ...]
    offload_attn_mlp_samples: tuple[float, ...]


def read_json(path: Path) -> dict[str, object]:
    """Read a JSON object from a file."""

    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def matches_model_family(model: str, model_family: str) -> bool:
    """Return whether a model belongs to the requested family."""

    normalized = model.lower()
    return normalized.startswith("qwen") if model_family == "qwen" else normalized.startswith("deepseek")


def parse_identity(config: dict[str, object], config_path: Path) -> Identity:
    """Validate run metadata and return its analysis identity."""

    run_name = str(config.get("run_name", ""))
    match = RUN_NAME_PATTERN.fullmatch(run_name)
    if match is None:
        raise ValueError(f"Unrecognized run_name in {config_path}: {run_name}")
    try:
        mbs = int(config.get("micro_batch_size", match.group(4) or 0))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid micro_batch_size in {config_path}") from exc
    run_name_mbs = None if match.group(4) is None else int(match.group(4))
    if mbs <= 0 or (run_name_mbs is not None and run_name_mbs != mbs):
        raise ValueError(f"Invalid or mismatched micro_batch_size in {config_path}")
    dispatcher = str(config.get("dispatcher", match.group(2)))
    if dispatcher != match.group(2):
        raise ValueError(f"run_name and config disagree on dispatcher in {config_path}")
    recorded_dtype = str(config.get("dtype", config.get("precision", "")))
    dtype = "mxfp8" if recorded_dtype == "fp8mx" else recorded_dtype
    if dtype not in {"bf16", "mxfp8"}:
        raise ValueError(f"Invalid dtype in {config_path}: {recorded_dtype}")
    model = str(config.get("model", ""))
    run_time = str(config.get("run_time", ""))
    if not model or not run_time:
        raise ValueError(f"Missing model or run_time in {config_path}")
    return Identity(model, dispatcher, mbs, dtype, match.group(1), match.group(3), run_time)


def read_mean_tflops(result_dir: Path) -> tuple[float, tuple[float, ...]]:
    """Read all GPU utilization samples and average the final four."""

    metrics_path = result_dir / "gpu_utilization.json"
    metrics = read_json(metrics_path)
    samples: list[tuple[int, float]] = []
    for raw_step, raw_value in metrics.items():
        try:
            step = int(raw_step)
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{metrics_path} contains invalid GPU utilization data") from exc
        if str(step) != str(raw_step) and not (isinstance(raw_step, str) and raw_step.isdigit()):
            raise ValueError(f"{metrics_path} contains invalid GPU utilization data")
        if step < 0 or not math.isfinite(value):
            raise ValueError(f"{metrics_path} contains invalid GPU utilization data")
        samples.append((step, value))
    samples.sort(key=lambda item: item[0])
    if len(samples) < MIN_GPU_UTILIZATION_SAMPLES:
        raise ValueError(
            f"{metrics_path} has {len(samples)} GPU utilization values; "
            f"expected at least {MIN_GPU_UTILIZATION_SAMPLES}"
        )
    values = tuple(value for _, value in samples[-GPU_UTILIZATION_SAMPLES_TO_AVERAGE:])
    return sum(values) / len(values), values


def discover_rows(results_root: Path, model_family: str) -> list[ThroughputRow]:
    """Discover successful runs and pair the latest baseline/offload entries."""

    groups: dict[tuple[object, ...], dict[str, dict[str, Candidate]]] = {}
    for config_path in results_root.rglob("config.json"):
        config = read_json(config_path)
        if str(config.get("profile", "none")) != "none":
            continue
        model = str(config.get("model", ""))
        if not matches_model_family(model, model_family):
            continue
        result_dir = config_path.parent
        summary_path = result_dir / "summary.json"
        if not summary_path.exists() or int(read_json(summary_path).get("status", 1)) != 0:
            continue
        identity = parse_identity(config, config_path)
        key = (identity.model, identity.dispatcher, identity.mbs, identity.dtype, identity.model_type)
        case_runs = groups.setdefault(
            key,
            {"baseline": {}, "offload-mlp": {}, "offload-attn-mlp": {}},
        )
        target = case_runs[identity.case_name]
        if identity.run_time in target:
            raise ValueError(f"Duplicate {identity.case_name} run for {key} at {identity.run_time}")
        target[identity.run_time] = Candidate(identity, result_dir)

    rows: list[ThroughputRow] = []
    for key, case_runs in groups.items():
        if (
            not case_runs["baseline"]
            or not case_runs["offload-mlp"]
            or not case_runs["offload-attn-mlp"]
        ):
            raise ValueError(f"Missing baseline or offload case run for {key}")
        baseline_time = sorted(case_runs["baseline"])[-1]
        offload_mlp_time = sorted(case_runs["offload-mlp"])[-1]
        offload_attn_mlp_time = sorted(case_runs["offload-attn-mlp"])[-1]
        baseline = case_runs["baseline"][baseline_time]
        offload_mlp = case_runs["offload-mlp"][offload_mlp_time]
        offload_attn_mlp = case_runs["offload-attn-mlp"][offload_attn_mlp_time]
        baseline_mean, baseline_samples = read_mean_tflops(baseline.result_dir)
        offload_mlp_mean, offload_mlp_samples = read_mean_tflops(offload_mlp.result_dir)
        offload_attn_mlp_mean, offload_attn_mlp_samples = read_mean_tflops(offload_attn_mlp.result_dir)
        if baseline_mean <= 0:
            raise ValueError(f"Baseline mean TFlops must be positive for {key} at {baseline_time}")
        rows.append(
            ThroughputRow(
                model=baseline.identity.model,
                dispatcher=baseline.identity.dispatcher,
                mbs=baseline.identity.mbs,
                dtype=baseline.identity.dtype,
                model_type=baseline.identity.model_type,
                baseline_tflops=baseline_mean,
                offload_mlp_tflops=offload_mlp_mean,
                offload_attn_mlp_tflops=offload_attn_mlp_mean,
                offload_mlp_performance_drop=(baseline_mean - offload_mlp_mean) / baseline_mean,
                offload_attn_mlp_performance_drop=(baseline_mean - offload_attn_mlp_mean) / baseline_mean,
                baseline_run_time=baseline_time,
                offload_mlp_run_time=offload_mlp_time,
                offload_attn_mlp_run_time=offload_attn_mlp_time,
                baseline_samples=baseline_samples,
                offload_mlp_samples=offload_mlp_samples,
                offload_attn_mlp_samples=offload_attn_mlp_samples,
            )
        )
    if not rows:
        raise ValueError(f"No successful non-profiled {model_family} runs found under {results_root}")
    return sorted(
        rows,
        key=lambda row: (
            {"default": 0, "alltoall": 1, "hybridep": 2}[row.dispatcher],
            row.mbs,
            row.dtype,
            {"dense": 0, "expert": 1}[row.model_type],
            row.model,
        ),
    )


def workbook_xml() -> str:
    """Return one-sheet workbook metadata."""

    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Throughput" sheetId="1" r:id="rId1"/></sheets><calcPr calcMode="auto" fullCalcOnLoad="1" forceFullCalc="1"/></workbook>'


def workbook_rels_xml() -> str:
    """Return one-sheet workbook relationships."""

    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>'


def write_workbook(rows: list[ThroughputRow], output_path: Path) -> None:
    """Write throughput rows to a formatted XLSX workbook."""

    sheet_rows = [[inline_string_cell(f"{column_name(index)}1", header) for index, header in enumerate(HEADERS)]]
    for row_number, row in enumerate(rows, start=2):
        sheet_rows.append([
            inline_string_cell(f"A{row_number}", row.model),
            inline_string_cell(f"B{row_number}", row.dispatcher),
            number_cell(f"C{row_number}", row.mbs),
            inline_string_cell(f"D{row_number}", row.dtype),
            inline_string_cell(f"E{row_number}", row.model_type),
            number_cell(f"F{row_number}", row.baseline_tflops),
            number_cell(f"G{row_number}", row.offload_mlp_tflops),
            number_cell(f"H{row_number}", row.offload_attn_mlp_tflops),
            number_cell(f"I{row_number}", row.offload_mlp_performance_drop),
            number_cell(f"J{row_number}", row.offload_attn_mlp_performance_drop),
        ])
    sheet = make_sheet_xml(
        sheet_rows,
        max_column=10,
        max_row=max(1, len(sheet_rows)),
    )
    content_types = content_types_xml().replace(
        '<Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>',
        "",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", package_rels_xml())
        archive.writestr("xl/workbook.xml", workbook_xml())
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml())
        archive.writestr("xl/styles.xml", styles_xml())
        archive.writestr("xl/worksheets/sheet1.xml", sheet)


def run_cli(argv: list[str]) -> int:
    """Parse options, analyze runs, and write the throughput workbook."""

    parser = argparse.ArgumentParser(description="Analyze MLP offload GPU utilization and write an XLSX workbook.")
    parser.add_argument("--model", required=True, choices=("qwen", "deepseek"))
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    results_root = args.results_root.resolve()
    rows = discover_rows(results_root, args.model)
    output_path = (args.output or results_root / f"offload-throughput-{args.model}.xlsx").resolve()
    if args.dry_run:
        print(json.dumps({
            "modelFamily": args.model,
            "outputPath": str(output_path),
            "headers": HEADERS,
            "rows": [
                {
                    "model": row.model,
                    "dispatcher": row.dispatcher,
                    "mbs": row.mbs,
                    "dtype": row.dtype,
                    "type": row.model_type,
                    "baselineTflops": row.baseline_tflops,
                    "offloadMlpTflops": row.offload_mlp_tflops,
                    "offloadAttnMlpTflops": row.offload_attn_mlp_tflops,
                    "offloadMlpPerformanceDrop": row.offload_mlp_performance_drop,
                    "offloadAttnMlpPerformanceDrop": row.offload_attn_mlp_performance_drop,
                    "baselineRunTime": row.baseline_run_time,
                    "offloadMlpRunTime": row.offload_mlp_run_time,
                    "offloadAttnMlpRunTime": row.offload_attn_mlp_run_time,
                    "baselineSamples": row.baseline_samples,
                    "offloadMlpSamples": row.offload_mlp_samples,
                    "offloadAttnMlpSamples": row.offload_attn_mlp_samples,
                }
                for row in rows
            ],
        }, indent=2))
        return 0
    write_workbook(rows, output_path)
    print(f"Wrote {len(rows)} throughput comparison row(s): {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run_cli(sys.argv[1:]))
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error
