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

"""Collect MLP offload benchmark runs and write an XLSX comparison workbook."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import zipfile
from dataclasses import dataclass
from html import escape
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]
DEFAULT_RESULTS_ROOT = (
    REPO_ROOT
    / "results/01-analyse/01-fine-grained-offload/02-offload-on-dense-or-expert-model"
)
SAMPLE_ITERATIONS = (5, 6, 7, 8, 9)
RUN_NAME_PATTERN = re.compile(
    r"^(dense|expert)-(default|alltoall|hybridep)-(baseline|offload)(?:-mbs(\d+))?-r(\d+)$"
)
ITERATION_PATTERN = re.compile(
    r"iteration\s+(\d+)\s*/\s*\d+\s*\|.*?elapsed time per iteration \(ms\):\s*([\d.]+)",
    re.IGNORECASE,
)
ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*m")


@dataclass(frozen=True)
class Sample:
    """One measured training iteration and its derived throughput."""

    iteration: int
    step_time_ms: float
    tokens_per_second: float


@dataclass(frozen=True)
class Run:
    """Validated metadata and samples for one benchmark run."""

    run_time: str
    model_kind: str
    model: str
    dtype: str
    dispatcher: str
    case_name: str
    repeat: int
    global_batch_size: int
    micro_batch_size: int
    sequence_length: int
    result_dir: str
    samples: tuple[Sample, ...]


def parse_identity(run_name: str) -> tuple[str, str, str, int | None, int]:
    """Parse the model, dispatcher, case, micro-batch size, and repeat."""

    match = RUN_NAME_PATTERN.fullmatch(run_name)
    if match is None:
        raise ValueError(f"Unrecognized run name: {run_name}")
    return (
        match.group(1),
        match.group(2),
        match.group(3),
        None if match.group(4) is None else int(match.group(4)),
        int(match.group(5)),
    )


def parse_iteration_times(log_text: str) -> dict[int, float]:
    """Extract iteration step times from a training log."""

    times: dict[int, float] = {}
    for line in ANSI_PATTERN.sub("", log_text).splitlines():
        match = ITERATION_PATTERN.search(line)
        if match is None:
            continue
        iteration = int(match.group(1))
        step_time_ms = float(match.group(2))
        if not math.isfinite(step_time_ms) or step_time_ms <= 0:
            raise ValueError(f"Invalid step time for iteration {iteration}: {step_time_ms}")
        if iteration in times and times[iteration] != step_time_ms:
            raise ValueError(f"Iteration {iteration} has conflicting step times in train.log")
        times[iteration] = step_time_ms
    return times


def read_json(path: Path) -> dict[str, object]:
    """Read a JSON object from a file."""

    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def parse_run(config_path: Path) -> Run | None:
    """Validate and parse one run, skipping profiled runs."""

    config = read_json(config_path)
    if str(config.get("profile", "none")) != "none":
        return None

    result_dir = config_path.parent
    run_name = str(config.get("run_name", ""))
    model_kind, dispatcher_from_name, case_name, mbs_from_name, repeat = parse_identity(run_name)
    summary_path = result_dir / "summary.json"
    summary = read_json(summary_path) if summary_path.exists() else {"status": 1}
    if int(summary.get("status", 1)) != 0:
        raise ValueError(f"Training did not complete successfully: {result_dir}")

    try:
        log_text = (result_dir / "train.log").read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Unable to read train.log: {result_dir}") from exc
    iteration_times = parse_iteration_times(log_text)
    missing = [iteration for iteration in SAMPLE_ITERATIONS if iteration not in iteration_times]
    if missing:
        missing_text = ", ".join(str(iteration) for iteration in missing)
        raise ValueError(f"{result_dir} is missing iteration(s): {missing_text}")

    try:
        global_batch_size = int(config["global_batch_size"])
        micro_batch_size = int(config["micro_batch_size"])
        sequence_length = int(config["sequence_length"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid batch size or sequence_length in {config_path}") from exc
    if global_batch_size <= 0 or micro_batch_size <= 0 or sequence_length <= 0:
        raise ValueError(f"Invalid batch size or sequence_length in {config_path}")

    recorded_dtype = str(config.get("dtype", config.get("precision", "")))
    dtype = "mxfp8" if recorded_dtype == "fp8mx" else recorded_dtype
    if dtype not in {"bf16", "mxfp8"}:
        raise ValueError(f"Invalid or missing dtype in {config_path}")
    if mbs_from_name is not None and mbs_from_name != micro_batch_size:
        raise ValueError(f"Run name and config disagree on micro batch size: {config_path}")

    samples = tuple(
        Sample(
            iteration=iteration,
            step_time_ms=iteration_times[iteration],
            tokens_per_second=(global_batch_size * sequence_length * 1000) / iteration_times[iteration],
        )
        for iteration in SAMPLE_ITERATIONS
    )
    return Run(
        run_time=str(config.get("run_time", "")),
        model_kind=model_kind,
        model=str(config.get("model", "")),
        dtype=dtype,
        dispatcher=str(config.get("dispatcher", dispatcher_from_name)),
        case_name=case_name,
        repeat=repeat,
        global_batch_size=global_batch_size,
        micro_batch_size=micro_batch_size,
        sequence_length=sequence_length,
        result_dir=str(result_dir),
        samples=samples,
    )


def sort_key(run: Run) -> tuple[object, ...]:
    """Return the stable display order for a benchmark run."""

    return (
        {"dense": 0, "expert": 1}[run.model_kind],
        {"default": 0, "alltoall": 1, "hybridep": 2}[run.dispatcher],
        run.micro_batch_size,
        {"baseline": 0, "offload": 1}[run.case_name],
        run.repeat,
    )


def discover_runs(results_root: Path, requested_run_time: str | None) -> tuple[str, list[Run]]:
    """Find, validate, deduplicate, and sort runs for one batch."""

    candidates: list[tuple[Path, str]] = []
    for config_path in results_root.rglob("config.json"):
        config = read_json(config_path)
        if str(config.get("profile", "none")) != "none":
            continue
        run_name = str(config.get("run_name", ""))
        if RUN_NAME_PATTERN.fullmatch(run_name) is None:
            continue
        candidates.append((config_path, str(config.get("run_time", ""))))
    if not candidates:
        raise ValueError(f"No successful non-profiled runs found under {results_root}")

    run_times = sorted({run_time for _, run_time in candidates})
    run_time = requested_run_time or run_times[-1]
    selected_paths = [path for path, candidate_time in candidates if candidate_time == run_time]
    selected = sorted(
        (run for path in selected_paths if (run := parse_run(path)) is not None),
        key=sort_key,
    )
    if not selected:
        raise ValueError(f"No successful non-profiled runs found for run_time={run_time}")

    identities: set[tuple[object, ...]] = set()
    for run in selected:
        identity = (
            run.model_kind,
            run.model,
            run.dtype,
            run.dispatcher,
            run.case_name,
            run.micro_batch_size,
            run.repeat,
        )
        if identity in identities:
            raise ValueError(f"Duplicate run identity for run_time={run_time}: {identity}")
        identities.add(identity)
    return run_time, selected


def column_name(index: int) -> str:
    """Convert a zero-based column index to an Excel column name."""

    result = ""
    while index >= 0:
        index, remainder = divmod(index, 26)
        result = chr(65 + remainder) + result
        index -= 1
    return result


def xml_text(value: object) -> str:
    """Escape a value for use in XML text."""

    return escape(str(value), quote=False)


def inline_string_cell(reference: str, value: object, style: int = 0) -> str:
    """Build an inline-string worksheet cell."""

    return f'<c r="{reference}" s="{style}" t="inlineStr"><is><t>{xml_text(value)}</t></is></c>'


def number_cell(reference: str, value: object, style: int = 0) -> str:
    """Build a numeric worksheet cell."""

    return f'<c r="{reference}" s="{style}"><v>{xml_text(value)}</v></c>'


def formula_cell(reference: str, formula: str, style: int = 0) -> str:
    """Build a worksheet formula cell."""

    return f'<c r="{reference}" s="{style}"><f>{xml_text(formula)}</f><v></v></c>'


def make_sheet_xml(
    rows: list[list[str]],
    *,
    max_column: int,
    max_row: int,
    merges: tuple[str, ...] = (),
    widths: tuple[tuple[int, float], ...] = (),
    frozen_rows: int | None = None,
    row_heights: tuple[tuple[int, float], ...] = (),
) -> str:
    """Build worksheet XML from already-rendered cells."""

    columns = "".join(
        f'<col min="{index + 1}" max="{index + 1}" width="{width}" customWidth="1"/>'
        for index, width in widths
    )
    sheet_rows = []
    heights = dict(row_heights)
    for row_number, cells in enumerate(rows, start=1):
        height = f' ht="{heights[row_number]}" customHeight="1"' if row_number in heights else ""
        sheet_rows.append(f'<row r="{row_number}"{height}>{"".join(cells)}</row>')
    pane = ""
    if frozen_rows is not None:
        pane = (
            f'<sheetViews><sheetView workbookViewId="0"><pane ySplit="{frozen_rows}" '
            f'topLeftCell="A{frozen_rows + 1}" activePane="bottomLeft" state="frozen"/>'
            "<selection pane=\"bottomLeft\" activeCell=\"A1\" sqref=\"A1\"/>"
            "</sheetView></sheetViews>"
        )
    merge_xml = "" if not merges else f'<mergeCells count="{len(merges)}">' + "".join(
        f'<mergeCell ref="{merge}"/>' for merge in merges
    ) + "</mergeCells>"
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<dimension ref="A1:{column_name(max_column - 1)}{max_row}"/>'
        f'{pane}<sheetFormatPr defaultRowHeight="15"/><cols>{columns}</cols>'
        f'<sheetData>{"".join(sheet_rows)}</sheetData>{merge_xml}'
        "</worksheet>"
    )


def styles_xml() -> str:
    """Return the workbook's compact style table."""

    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <numFmts count="2"><numFmt numFmtId="164" formatCode="#,##0.00"/><numFmt numFmtId="165" formatCode="0.00%"/></numFmts>
  <fonts count="4"><font><sz val="11"/><color theme="1"/><name val="Calibri"/></font><font><b/><sz val="16"/><color rgb="FFFFFFFF"/><name val="Calibri"/></font><font><i/><sz val="11"/><color rgb="FF1F1F1F"/><name val="Calibri"/></font><font><b/><sz val="11"/><color rgb="FFFFFFFF"/><name val="Calibri"/></font></fonts>
  <fills count="4"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/></patternFill></fill><fill><patternFill patternType="solid"><fgColor rgb="FFD9EAF7"/></patternFill></fill></fills>
  <borders count="2"><border/><border><top style="thin"><color rgb="FFD9E1F2"/></top><bottom style="thin"><color rgb="FFD9E1F2"/></bottom></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="8"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/><xf numFmtId="0" fontId="1" fillId="2" borderId="0" applyAlignment="1"><alignment horizontal="left"/></xf><xf numFmtId="0" fontId="2" fillId="3" borderId="0"/><xf numFmtId="0" fontId="3" fillId="2" borderId="0" applyAlignment="1"><alignment wrapText="1"/></xf><xf numFmtId="164" fontId="0" fillId="0" borderId="1"/><xf numFmtId="165" fontId="0" fillId="0" borderId="1"/><xf numFmtId="0" fontId="0" fillId="0" borderId="1"/><xf numFmtId="164" fontId="0" fillId="0" borderId="0"/></cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>'''


def workbook_xml() -> str:
    """Return workbook metadata and recalculation settings."""

    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Summary" sheetId="1" r:id="rId1"/><sheet name="Samples" sheetId="2" r:id="rId2"/></sheets><calcPr calcMode="auto" fullCalcOnLoad="1" forceFullCalc="1"/></workbook>'''


def workbook_rels_xml() -> str:
    """Return workbook relationship metadata."""

    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/><Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>'''


def content_types_xml() -> str:
    """Return package content type declarations."""

    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/></Types>'''


def package_rels_xml() -> str:
    """Return the package root relationship metadata."""

    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>'''


def write_workbook(runs: list[Run], run_time: str, output_path: Path) -> None:
    """Write the summary and samples sheets to an XLSX package."""

    summary_headers = [
        "Model Type", "Model", "DType", "Dispatcher", "Case", "Repeat", "MBS",
        "Iter 5", "Iter 6", "Iter 7", "Iter 8", "Iter 9", "Mean", "Median", "Min",
        "Max", "Std Dev", "CV", "vs Baseline", "vs All-to-All",
    ]
    summary_rows: list[list[str]] = []
    summary_rows.append([inline_string_cell("A1", "MLP Offload Throughput Comparison", 1)])
    summary_rows.append([inline_string_cell("A2", f"run_time={run_time}; successful profile=none runs only; throughput uses iterations 5-9", 2)])
    summary_rows.append([])
    summary_rows.append([
        inline_string_cell(f"{column_name(index)}4", header, 3) for index, header in enumerate(summary_headers)
    ])

    baseline_rows: dict[tuple[object, ...], int] = {}
    alltoall_rows: dict[tuple[object, ...], int] = {}
    for index, run in enumerate(runs, start=5):
        baseline_key = (run.model_kind, run.model, run.dtype, run.dispatcher, run.micro_batch_size, run.repeat)
        alltoall_key = (run.model_kind, run.model, run.dtype, run.case_name, run.micro_batch_size, run.repeat)
        if run.case_name == "baseline":
            baseline_rows[baseline_key] = index
        if run.dispatcher == "alltoall":
            alltoall_rows[alltoall_key] = index

    for index, run in enumerate(runs, start=5):
        cells = [
            inline_string_cell(f"A{index}", run.model_kind),
            inline_string_cell(f"B{index}", run.model),
            inline_string_cell(f"C{index}", run.dtype),
            inline_string_cell(f"D{index}", run.dispatcher),
            inline_string_cell(f"E{index}", run.case_name),
            number_cell(f"F{index}", run.repeat),
            number_cell(f"G{index}", run.micro_batch_size),
        ]
        cells.extend(number_cell(f"{column_name(column)}{index}", sample.tokens_per_second, 4) for column, sample in enumerate(run.samples, start=7))
        cells.extend([
            formula_cell(f"M{index}", f"AVERAGE(H{index}:L{index})", 4),
            formula_cell(f"N{index}", f"MEDIAN(H{index}:L{index})", 4),
            formula_cell(f"O{index}", f"MIN(H{index}:L{index})", 4),
            formula_cell(f"P{index}", f"MAX(H{index}:L{index})", 4),
            formula_cell(f"Q{index}", f"STDEV.S(H{index}:L{index})", 4),
            formula_cell(f"R{index}", f'IFERROR(Q{index}/M{index},"")', 5),
        ])
        baseline_row = baseline_rows.get((run.model_kind, run.model, run.dtype, run.dispatcher, run.micro_batch_size, run.repeat))
        cells.append(formula_cell(f"S{index}", f'IFERROR(M{index}/M{baseline_row}-1,"")' if baseline_row else '""', 5))
        alltoall_row = alltoall_rows.get((run.model_kind, run.model, run.dtype, run.case_name, run.micro_batch_size, run.repeat))
        cells.append(formula_cell(f"T{index}", f'IFERROR(M{index}/M{alltoall_row}-1,"")' if run.model_kind == "expert" and alltoall_row else '""', 5))
        summary_rows.append(cells)

    sample_headers = [
        "Run Time", "Model Type", "Model", "DType", "Dispatcher", "Case", "Repeat", "MBS",
        "Iteration", "Step Time (ms)", "Global Batch Size", "Sequence Length", "Throughput (tokens/s)", "Result Directory",
    ]
    sample_rows = [[inline_string_cell(f"{column_name(index)}1", header, 3) for index, header in enumerate(sample_headers)]]
    for run in runs:
        for sample in run.samples:
            row = len(sample_rows) + 1
            values: list[str] = [
                inline_string_cell(f"A{row}", run.run_time), inline_string_cell(f"B{row}", run.model_kind),
                inline_string_cell(f"C{row}", run.model), inline_string_cell(f"D{row}", run.dtype),
                inline_string_cell(f"E{row}", run.dispatcher), inline_string_cell(f"F{row}", run.case_name),
                number_cell(f"G{row}", run.repeat), number_cell(f"H{row}", run.micro_batch_size),
                number_cell(f"I{row}", sample.iteration), number_cell(f"J{row}", sample.step_time_ms, 4),
                number_cell(f"K{row}", run.global_batch_size), number_cell(f"L{row}", run.sequence_length),
                number_cell(f"M{row}", sample.tokens_per_second, 4), inline_string_cell(f"N{row}", run.result_dir),
            ]
            sample_rows.append(values)

    summary_xml = make_sheet_xml(
        summary_rows,
        max_column=20,
        max_row=max(4, len(summary_rows)),
        merges=("A1:T1", "A2:T2"),
        widths=tuple((index, width) for index, width in enumerate((12, 24, 10, 13, 11, 8, 8, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 11, 14, 15))),
        frozen_rows=4,
        row_heights=((1, 28), (4, 30)),
    )
    samples_xml = make_sheet_xml(
        sample_rows,
        max_column=14,
        max_row=max(1, len(sample_rows)),
        widths=tuple((index, width) for index, width in enumerate((20, 12, 24, 10, 13, 11, 8, 8, 10, 16, 17, 16, 22, 72))),
        frozen_rows=1,
        row_heights=((1, 30),),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types_xml())
        archive.writestr("_rels/.rels", package_rels_xml())
        archive.writestr("xl/workbook.xml", workbook_xml())
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml())
        archive.writestr("xl/styles.xml", styles_xml())
        archive.writestr("xl/worksheets/sheet1.xml", summary_xml)
        archive.writestr("xl/worksheets/sheet2.xml", samples_xml)


def run_cli(argv: list[str]) -> int:
    """Parse command-line options, collect runs, and write the workbook."""

    parser = argparse.ArgumentParser(
        description="Collect successful non-profiled MLP offload runs and write an XLSX comparison workbook."
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-time")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Parse and validate data, then print JSON without writing XLSX")
    args = parser.parse_args(argv)
    results_root = args.results_root.resolve()
    run_time, runs = discover_runs(results_root, args.run_time)
    output_path = (args.output or results_root / f"offload-comparison-{run_time}.xlsx").resolve()
    if args.dry_run:
        payload = {
            "runTime": run_time,
            "outputPath": str(output_path),
            "runs": [
                {
                    "runTime": run.run_time,
                    "modelKind": run.model_kind,
                    "model": run.model,
                    "dtype": run.dtype,
                    "dispatcher": run.dispatcher,
                    "caseName": run.case_name,
                    "repeat": run.repeat,
                    "globalBatchSize": run.global_batch_size,
                    "microBatchSize": run.micro_batch_size,
                    "sequenceLength": run.sequence_length,
                    "resultDir": run.result_dir,
                    "samples": [
                        {"iteration": sample.iteration, "stepTimeMs": sample.step_time_ms, "tokensPerSecond": sample.tokens_per_second}
                        for sample in run.samples
                    ],
                }
                for run in runs
            ],
        }
        print(json.dumps(payload, indent=2))
        return 0
    write_workbook(runs, run_time, output_path)
    print(f"Collected {len(runs)} group(s), 5 iterations per group: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run_cli(sys.argv[1:]))
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error
