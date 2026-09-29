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

"""Collect Megatron experiment metrics and write an XLSX workbook."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import statistics
import sys
import zipfile
from dataclasses import dataclass
from html import escape
from pathlib import Path


LOGGER = logging.getLogger(__name__)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results/01-analyse/03-megatron-vs-deepspeed"
SAMPLES_TO_AVERAGE = 4
BYTES_PER_GIB = 1024**3
ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*m")
RUN_NAME_PATTERN = re.compile(
    r"^model_(dense|expert)__activation_(baseline|recompute|recompute_offload)__"
    r"(optimizer_none|optimizer_cpu_090|optimizer_cpu_075|optimizer_cpu_100)__"
    r"mbs_(\d+)__gbs_(\d+)__repeat_(\d+)$"
)
ITERATION_PATTERN = re.compile(r"iteration\s+(\d+)\s*/\s*\d+\s*\|", re.IGNORECASE)
METRIC_PATTERNS = {
    "step_time_ms": re.compile(r"elapsed time per iteration \(ms\):\s*([\d.eE+-]+)", re.IGNORECASE),
    "tflops_per_gpu": re.compile(r"throughput per GPU \(TFLOP/s/GPU\):\s*([\d.eE+-]+)", re.IGNORECASE),
    "cuda_peak_bytes": re.compile(r"cuda_peak_memory_allocated_bytes:\s*(\d+)", re.IGNORECASE),
    "host_peak_bytes": re.compile(r"host_peak_memory_allocated_bytes:\s*(\d+)", re.IGNORECASE),
}


@dataclass(frozen=True)
class IterationSample:
    """Metrics parsed for one training iteration."""

    iteration: int
    step_time_ms: float
    tflops_per_gpu: float | None
    tokens_per_second: float
    cuda_peak_bytes: int
    host_peak_bytes: int


@dataclass(frozen=True)
class RunResult:
    """One experiment run and the metrics computed from its final iterations."""

    run_time: str
    test_name: str
    status: str
    model_kind: str
    model: str
    precision: str
    activation_strategy: str
    optimizer_strategy: str
    micro_batch_size: int
    global_batch_size: int
    sequence_length: int
    repeat: int
    result_dir: str
    samples: tuple[IterationSample, ...]
    error: str

    @property
    def mean_step_time_ms(self) -> float | None:
        """Return the mean step time over the selected samples."""

        return statistics.fmean(sample.step_time_ms for sample in self.samples) if self.samples else None

    @property
    def mean_tflops_per_gpu(self) -> float | None:
        """Return mean TFLOP/s/GPU when all selected samples contain it."""

        values = [sample.tflops_per_gpu for sample in self.samples]
        if not values or any(value is None for value in values):
            return None
        return statistics.fmean(value for value in values if value is not None)

    @property
    def mean_tokens_per_second(self) -> float | None:
        """Return mean global token throughput over the selected samples."""

        return statistics.fmean(sample.tokens_per_second for sample in self.samples) if self.samples else None

    @property
    def peak_cuda_bytes(self) -> int | None:
        """Return the largest CUDA allocator peak in the selected samples."""

        return max((sample.cuda_peak_bytes for sample in self.samples), default=None)

    @property
    def peak_host_bytes(self) -> int | None:
        """Return the largest pinned-host allocator peak in the selected samples."""

        return max((sample.host_peak_bytes for sample in self.samples), default=None)


def read_json(path: Path) -> dict[str, object]:
    """Read and validate a JSON object."""

    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _parse_numeric(pattern: re.Pattern[str], line: str) -> float | None:
    match = pattern.search(line)
    if match is None:
        return None
    value = float(match.group(1))
    return value if math.isfinite(value) else None


def parse_iteration_samples(
    log_text: str,
    *,
    global_batch_size: int,
    sequence_length: int,
) -> tuple[IterationSample, ...]:
    """Parse the final four complete iteration records from a training log."""

    grouped: dict[int, dict[str, list[float]]] = {}
    for line in ANSI_PATTERN.sub("", log_text).splitlines():
        iteration_match = ITERATION_PATTERN.search(line)
        if iteration_match is None:
            continue
        iteration = int(iteration_match.group(1))
        metrics = grouped.setdefault(iteration, {name: [] for name in METRIC_PATTERNS})
        for name, pattern in METRIC_PATTERNS.items():
            value = _parse_numeric(pattern, line)
            if value is not None:
                metrics[name].append(value)

    complete_iterations = [
        iteration
        for iteration, metrics in grouped.items()
        if metrics["step_time_ms"] and metrics["cuda_peak_bytes"] and metrics["host_peak_bytes"]
    ]
    selected_iterations = sorted(complete_iterations)[-SAMPLES_TO_AVERAGE:]
    if len(selected_iterations) != SAMPLES_TO_AVERAGE:
        raise ValueError(f"Expected {SAMPLES_TO_AVERAGE} complete iterations, found {len(selected_iterations)}")

    samples: list[IterationSample] = []
    for iteration in selected_iterations:
        metrics = grouped[iteration]
        step_time_ms = statistics.fmean(metrics["step_time_ms"])
        if step_time_ms <= 0:
            raise ValueError(f"Iteration {iteration} has invalid step time: {step_time_ms}")
        tflops_values = metrics["tflops_per_gpu"]
        samples.append(
            IterationSample(
                iteration=iteration,
                step_time_ms=step_time_ms,
                tflops_per_gpu=statistics.fmean(tflops_values) if tflops_values else None,
                tokens_per_second=global_batch_size * sequence_length * 1000.0 / step_time_ms,
                cuda_peak_bytes=int(max(metrics["cuda_peak_bytes"])),
                host_peak_bytes=int(max(metrics["host_peak_bytes"])),
            )
        )
    return tuple(samples)


def parse_run(config_path: Path) -> RunResult:
    """Parse one run directory, retaining failed or incomplete runs."""

    config = read_json(config_path)
    result_dir = config_path.parent
    test_name = str(config.get("run_name", ""))
    identity = RUN_NAME_PATTERN.fullmatch(test_name)
    if identity is None:
        raise ValueError(f"Unrecognized run name in {config_path}: {test_name}")

    model_kind, activation, optimizer, name_mbs, name_gbs, repeat = identity.groups()
    micro_batch_size = int(config.get("micro_batch_size", 0))
    global_batch_size = int(config.get("global_batch_size", 0))
    sequence_length = int(config.get("sequence_length", 4096))
    if micro_batch_size != int(name_mbs) or global_batch_size != int(name_gbs):
        raise ValueError(f"Run name and config batch sizes disagree: {config_path}")
    if min(micro_batch_size, global_batch_size, sequence_length) <= 0:
        raise ValueError(f"Invalid batch size or sequence length in {config_path}")

    summary_path = result_dir / "summary.json"
    summary = read_json(summary_path) if summary_path.exists() else {}
    return_code = int(summary.get("status", 1))
    samples: tuple[IterationSample, ...] = ()
    error = ""
    status = "success" if return_code == 0 else "failed"
    if return_code != 0:
        error = f"training exit status {return_code}"
    else:
        try:
            log_text = (result_dir / "train.log").read_text(encoding="utf-8", errors="replace")
            samples = parse_iteration_samples(
                log_text,
                global_batch_size=global_batch_size,
                sequence_length=sequence_length,
            )
        except (OSError, ValueError) as exc:
            status = "incomplete"
            error = str(exc)

    return RunResult(
        run_time=str(config.get("run_time", "")),
        test_name=test_name,
        status=status,
        model_kind=model_kind,
        model=str(config.get("model", "")),
        precision=str(config.get("precision", "")),
        activation_strategy=str(config.get("activation_strategy", activation)),
        optimizer_strategy=str(config.get("optimizer_strategy", optimizer)),
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        sequence_length=sequence_length,
        repeat=int(repeat),
        result_dir=str(result_dir),
        samples=samples,
        error=error,
    )


def _sort_key(run: RunResult) -> tuple[object, ...]:
    return (
        {"dense": 0, "expert": 1}[run.model_kind],
        {"baseline": 0, "recompute": 1, "recompute_offload": 2}[run.activation_strategy],
        {
            "optimizer_none": 0,
            "optimizer_cpu_090": 1,
            "optimizer_cpu_075": 2,
            "optimizer_cpu_100": 3,
        }[run.optimizer_strategy],
        run.micro_batch_size,
        run.repeat,
    )


def discover_runs(results_root: Path, requested_run_time: str | None) -> tuple[str, list[RunResult]]:
    """Discover and parse one timestamped experiment batch."""

    candidates: list[tuple[Path, str]] = []
    for config_path in results_root.rglob("config.json"):
        config = read_json(config_path)
        run_name = str(config.get("run_name", ""))
        if RUN_NAME_PATTERN.fullmatch(run_name) is not None:
            candidates.append((config_path, str(config.get("run_time", ""))))
    if not candidates:
        raise ValueError(f"No experiment runs found under {results_root}")

    run_times = sorted({run_time for _, run_time in candidates if run_time})
    if not run_times:
        raise ValueError(f"Runs under {results_root} have no run_time")
    run_time = requested_run_time or run_times[-1]
    selected = sorted((parse_run(path) for path, value in candidates if value == run_time), key=_sort_key)
    if not selected:
        raise ValueError(f"No experiment runs found for run_time={run_time}")
    names = [run.test_name for run in selected]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate test names found for run_time={run_time}")
    return run_time, selected


def _column_name(index: int) -> str:
    result = ""
    while index >= 0:
        index, remainder = divmod(index, 26)
        result = chr(65 + remainder) + result
        index -= 1
    return result


def _string_cell(reference: str, value: object, style: int = 0) -> str:
    style_attr = f' s="{style}"' if style else ""
    return f'<c r="{reference}"{style_attr} t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'


def _number_cell(reference: str, value: int | float | None, style: int) -> str:
    if value is None:
        return f'<c r="{reference}" s="{style}"/>'
    return f'<c r="{reference}" s="{style}"><v>{value}</v></c>'


def _sheet_xml(
    rows: list[list[str]],
    *,
    max_columns: int,
    widths: tuple[float, ...],
    frozen_rows: int,
    auto_filter_row: int,
) -> str:
    max_row = max(1, len(rows))
    columns = "".join(
        f'<col min="{index}" max="{index}" width="{width}" customWidth="1"/>'
        for index, width in enumerate(widths, start=1)
    )
    rendered_rows_list = []
    for index, cells in enumerate(rows, start=1):
        height = ' ht="28" customHeight="1"' if index in {1, auto_filter_row} else ""
        rendered_rows_list.append(f'<row r="{index}"{height}>{"".join(cells)}</row>')
    rendered_rows = "".join(rendered_rows_list)
    last_column = _column_name(max_columns - 1)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<dimension ref="A1:{last_column}{max_row}"/>'
        f'<sheetViews><sheetView workbookViewId="0"><pane xSplit="2" ySplit="{frozen_rows}" '
        f'topLeftCell="C{frozen_rows + 1}" activePane="bottomRight" state="frozen"/>'
        '<selection pane="bottomRight" activeCell="C5" sqref="C5"/></sheetView></sheetViews>'
        '<sheetFormatPr defaultRowHeight="18"/>'
        f"<cols>{columns}</cols><sheetData>{rendered_rows}</sheetData>"
        f'<autoFilter ref="A{auto_filter_row}:{last_column}{max_row}"/>'
        f'<mergeCells count="2"><mergeCell ref="A1:{last_column}1"/><mergeCell ref="A2:{last_column}2"/></mergeCells>'
        "</worksheet>"
    )


def _styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <numFmts count="4"><numFmt numFmtId="164" formatCode="0.0"/><numFmt numFmtId="165" formatCode="0.00"/><numFmt numFmtId="166" formatCode="#,##0"/><numFmt numFmtId="167" formatCode="#,##0.0"/></numFmts>
  <fonts count="4"><font><sz val="10"/><name val="Arial"/></font><font><b/><sz val="14"/><color rgb="FF1F4E78"/><name val="Arial"/></font><font><i/><sz val="10"/><color rgb="FF595959"/><name val="Arial"/></font><font><b/><sz val="10"/><color rgb="FFFFFFFF"/><name val="Arial"/></font></fonts>
  <fills count="3"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/><bgColor indexed="64"/></patternFill></fill></fills>
  <borders count="2"><border/><border><bottom style="thin"><color rgb="FFD9E2F3"/></bottom></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="8"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="0" borderId="1" xfId="0"/><xf numFmtId="0" fontId="2" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="3" fillId="2" borderId="0" xfId="0"><alignment horizontal="center" vertical="center" wrapText="1"/></xf><xf numFmtId="166" fontId="0" fillId="0" borderId="0" xfId="0"><alignment horizontal="right"/></xf><xf numFmtId="164" fontId="0" fillId="0" borderId="0" xfId="0"><alignment horizontal="right"/></xf><xf numFmtId="167" fontId="0" fillId="0" borderId="0" xfId="0"><alignment horizontal="right"/></xf><xf numFmtId="165" fontId="0" fillId="0" borderId="0" xfId="0"><alignment horizontal="right"/></xf></cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>"""


def _workbook_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Summary" sheetId="1" r:id="rId1"/><sheet name="Samples" sheetId="2" r:id="rId2"/></sheets></workbook>"""


def _workbook_relationships_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/><Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>"""


def _content_types_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/></Types>"""


def _package_relationships_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>"""


def write_workbook(runs: list[RunResult], run_time: str, output_path: Path) -> None:
    """Write run summaries and final-iteration samples to an XLSX workbook."""

    summary_headers = (
        "Run Time",
        "Test Name",
        "Status",
        "Model Type",
        "Model",
        "Precision",
        "Activation",
        "Optimizer",
        "MBS",
        "GBS",
        "Sequence Length",
        "Repeat",
        "Sample Iterations",
        "Samples",
        "Avg Iteration Time (ms)",
        "Avg TFLOP/s/GPU",
        "Avg Tokens/s",
        "GPU Peak (GiB)",
        "CPU Pinned Peak (GiB)",
        "GPU Peak (bytes)",
        "CPU Pinned Peak (bytes)",
        "Result Directory",
        "Error",
    )
    summary_rows: list[list[str]] = [
        [_string_cell("A1", "Megatron ZeRO-3 Offload Experiment", 1)],
        [_string_cell("A2", f"run_time={run_time}; averages use the final 4 complete iterations", 2)],
        [],
        [_string_cell(f"{_column_name(index)}4", header, 3) for index, header in enumerate(summary_headers)],
    ]
    for run in runs:
        row = len(summary_rows) + 1
        iterations = ",".join(str(sample.iteration) for sample in run.samples)
        peak_cuda = run.peak_cuda_bytes
        peak_host = run.peak_host_bytes
        summary_rows.append(
            [
                _string_cell(f"A{row}", run.run_time),
                _string_cell(f"B{row}", run.test_name),
                _string_cell(f"C{row}", run.status),
                _string_cell(f"D{row}", run.model_kind),
                _string_cell(f"E{row}", run.model),
                _string_cell(f"F{row}", run.precision),
                _string_cell(f"G{row}", run.activation_strategy),
                _string_cell(f"H{row}", run.optimizer_strategy),
                _number_cell(f"I{row}", run.micro_batch_size, 4),
                _number_cell(f"J{row}", run.global_batch_size, 4),
                _number_cell(f"K{row}", run.sequence_length, 4),
                _number_cell(f"L{row}", run.repeat, 4),
                _string_cell(f"M{row}", iterations),
                _number_cell(f"N{row}", len(run.samples), 4),
                _number_cell(f"O{row}", run.mean_step_time_ms, 5),
                _number_cell(f"P{row}", run.mean_tflops_per_gpu, 5),
                _number_cell(f"Q{row}", run.mean_tokens_per_second, 6),
                _number_cell(f"R{row}", peak_cuda / BYTES_PER_GIB if peak_cuda is not None else None, 7),
                _number_cell(f"S{row}", peak_host / BYTES_PER_GIB if peak_host is not None else None, 7),
                _number_cell(f"T{row}", peak_cuda, 4),
                _number_cell(f"U{row}", peak_host, 4),
                _string_cell(f"V{row}", run.result_dir),
                _string_cell(f"W{row}", run.error),
            ]
        )

    sample_headers = (
        "Run Time",
        "Test Name",
        "Iteration",
        "Iteration Time (ms)",
        "TFLOP/s/GPU",
        "Tokens/s",
        "GPU Peak (GiB)",
        "CPU Pinned Peak (GiB)",
        "GPU Peak (bytes)",
        "CPU Pinned Peak (bytes)",
    )
    sample_rows: list[list[str]] = [
        [_string_cell("A1", "Final Iteration Samples", 1)],
        [_string_cell("A2", "CUDA allocator and CUDA pinned-host allocator peak counters", 2)],
        [],
        [_string_cell(f"{_column_name(index)}4", header, 3) for index, header in enumerate(sample_headers)],
    ]
    for run in runs:
        for sample in run.samples:
            row = len(sample_rows) + 1
            sample_rows.append(
                [
                    _string_cell(f"A{row}", run.run_time),
                    _string_cell(f"B{row}", run.test_name),
                    _number_cell(f"C{row}", sample.iteration, 4),
                    _number_cell(f"D{row}", sample.step_time_ms, 5),
                    _number_cell(f"E{row}", sample.tflops_per_gpu, 5),
                    _number_cell(f"F{row}", sample.tokens_per_second, 6),
                    _number_cell(f"G{row}", sample.cuda_peak_bytes / BYTES_PER_GIB, 7),
                    _number_cell(f"H{row}", sample.host_peak_bytes / BYTES_PER_GIB, 7),
                    _number_cell(f"I{row}", sample.cuda_peak_bytes, 4),
                    _number_cell(f"J{row}", sample.host_peak_bytes, 4),
                ]
            )

    summary_xml = _sheet_xml(
        summary_rows,
        max_columns=len(summary_headers),
        widths=(20, 72, 12, 12, 24, 10, 20, 22, 8, 8, 16, 8, 20, 9, 22, 18, 18, 17, 22, 20, 24, 72, 44),
        frozen_rows=4,
        auto_filter_row=4,
    )
    samples_xml = _sheet_xml(
        sample_rows,
        max_columns=len(sample_headers),
        widths=(20, 72, 10, 20, 16, 18, 17, 22, 20, 24),
        frozen_rows=4,
        auto_filter_row=4,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", _content_types_xml())
        archive.writestr("_rels/.rels", _package_relationships_xml())
        archive.writestr("xl/workbook.xml", _workbook_xml())
        archive.writestr("xl/_rels/workbook.xml.rels", _workbook_relationships_xml())
        archive.writestr("xl/styles.xml", _styles_xml())
        archive.writestr("xl/worksheets/sheet1.xml", summary_xml)
        archive.writestr("xl/worksheets/sheet2.xml", samples_xml)


def _dry_run_payload(run_time: str, runs: list[RunResult], output_path: Path) -> dict[str, object]:
    return {
        "runTime": run_time,
        "outputPath": str(output_path),
        "runs": [
            {
                "testName": run.test_name,
                "status": run.status,
                "sampleIterations": [sample.iteration for sample in run.samples],
                "meanIterationTimeMs": run.mean_step_time_ms,
                "meanTflopsPerGpu": run.mean_tflops_per_gpu,
                "meanTokensPerSecond": run.mean_tokens_per_second,
                "peakCudaBytes": run.peak_cuda_bytes,
                "peakHostBytes": run.peak_host_bytes,
                "error": run.error,
            }
            for run in runs
        ],
    }


def run_cli(argv: list[str]) -> int:
    """Collect one experiment batch and optionally write an XLSX workbook."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-time", help="Experiment timestamp; defaults to the latest discovered value")
    parser.add_argument("--output", type=Path, help="XLSX output path")
    parser.add_argument("--dry-run", action="store_true", help="Print parsed JSON without writing XLSX")
    args = parser.parse_args(argv)

    results_root = args.results_root.resolve()
    run_time, runs = discover_runs(results_root, args.run_time)
    output_path = (args.output or results_root / f"megatron-vs-deepspeed-{run_time}.xlsx").resolve()
    if args.dry_run:
        sys.stdout.write(json.dumps(_dry_run_payload(run_time, runs, output_path), indent=2) + "\n")
        return 0

    write_workbook(runs, run_time, output_path)
    success_count = sum(run.status == "success" for run in runs)
    LOGGER.info("Wrote %s runs (%s successful) to %s", len(runs), success_count, output_path)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        raise SystemExit(run_cli(sys.argv[1:]))
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        LOGGER.error("%s", error)
        raise SystemExit(1) from error
