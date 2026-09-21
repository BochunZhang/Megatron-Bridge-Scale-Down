"""Convert DeepSpeed training memory-profile JSON files to an XLSX workbook."""

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

from __future__ import annotations

import argparse
import json
import logging
import zipfile
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

logger = logging.getLogger(__name__)
GIB = 1024**3
REPLAY_HEADERS = [
    "phase",
    "start time (ms)",
    "end time (ms)",
    "duration (ms)",
    "start memory (GiB)",
    "end memory (GiB)",
    "peak memory (GiB)",
    "end - start memory (GiB)",
    "peak - start memory (GiB)",
    "alloc cnt",
    "free cnt",
    "overlap",
]


def _gib(value: int) -> float:
    return round(value / GIB, 3)


def _profile_paths(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for input_path in inputs:
        if input_path.is_dir():
            paths.extend(sorted(input_path.glob("memory_profile*.json")))
        elif input_path.is_file():
            paths.append(input_path)
        else:
            raise FileNotFoundError(f"Profile input does not exist: {input_path}")
    unique = sorted(set(paths))
    if not unique:
        raise FileNotFoundError("No memory_profile*.json files found")
    return unique


def _load_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
            raise ValueError(f"Invalid memory profile JSON: {path}")
        rank = payload.get("rank", "unknown")
        for record in payload["records"]:
            if not isinstance(record, dict):
                raise ValueError(f"Invalid record in memory profile JSON: {path}")
            records.append({"rank": rank, **record})
    return sorted(records, key=lambda record: (str(record["rank"]), record.get("step", 0)))


def _cuda_row(record: dict[str, Any]) -> list[Any]:
    start = int(record.get("cuda_start_bytes", 0))
    end = int(record.get("cuda_end_bytes", 0))
    peak = int(record.get("cuda_peak_bytes", 0))
    reserved_start = int(record.get("cuda_reserved_start_bytes", 0))
    reserved_end = int(record.get("cuda_reserved_end_bytes", 0))
    reserved_peak = int(record.get("cuda_reserved_peak_bytes", 0))
    return [
        f"rank-{record['rank']}:{record.get('phase', 'step')}",
        record.get("start_time_ms"),
        record.get("end_time_ms"),
        record.get("duration_ms"),
        _gib(start),
        _gib(end),
        _gib(peak),
        _gib(end - start),
        _gib(peak - start),
        record.get("cuda_alloc_count", 0),
        record.get("cuda_free_count", 0),
        f"reserved start={_gib(reserved_start):.3f} GiB; "
        f"reserved end={_gib(reserved_end):.3f} GiB; "
        f"reserved peak={_gib(reserved_peak):.3f} GiB",
    ]


def _pinned_row(record: dict[str, Any]) -> list[Any]:
    start = int(record.get("pinned_start_bytes", 0))
    end = int(record.get("pinned_end_bytes", 0))
    peak = int(record.get("pinned_peak_bytes", 0))
    return [
        f"rank-{record['rank']}:{record.get('phase', 'step')}",
        record.get("start_time_ms"),
        record.get("end_time_ms"),
        record.get("duration_ms"),
        _gib(start),
        _gib(end),
        _gib(peak),
        _gib(end - start),
        _gib(peak - start),
        len(record.get("samples", [])),
        "",
        (
            f"process start={_gib(int(record.get('pinned_process_start_bytes', 0))):.3f} GiB; "
            f"process end={_gib(int(record.get('pinned_process_end_bytes', 0))):.3f} GiB; "
            f"process peak={_gib(int(record.get('pinned_process_peak_bytes', 0))):.3f} GiB; "
            f"locked start={_gib(int(record.get('pinned_locked_start_bytes', 0))):.3f} GiB; "
            f"locked end={_gib(int(record.get('pinned_locked_end_bytes', 0))):.3f} GiB; "
            f"tensor peak={_gib(int(record.get('pinned_tensor_peak_bytes', 0))):.3f} GiB"
        ),
    ]


def _xlsx_cell(value: object, row: int, column: int) -> str:
    column_name = ""
    number = column
    while number:
        number, remainder = divmod(number - 1, 26)
        column_name = chr(65 + remainder) + column_name
    cell_ref = f"{column_name}{row}"
    if value is None or value == "":
        return f'<c r="{cell_ref}"/>'
    if isinstance(value, bool):
        return f'<c r="{cell_ref}" t="b"><v>{int(value)}</v></c>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{cell_ref}"><v>{value}</v></c>'
    return f'<c r="{cell_ref}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'


def _sheet_xml(rows: list[list[object]]) -> str:
    row_xml = []
    for row_number, values in enumerate(rows, 1):
        cells = "".join(
            _xlsx_cell(value, row_number, column)
            for column, value in enumerate(values, 1)
        )
        row_xml.append(f'<row r="{row_number}">{cells}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{''.join(row_xml)}</sheetData>"
        "</worksheet>"
    )


def write_xlsx(records: list[dict[str, Any]], output_path: Path) -> None:
    """Write CUDA and pinned memory tables as separate workbook sheets."""
    sheets = {
        "cuda_memory": [_cuda_row(record) for record in records],
        "pinned_memory": [_pinned_row(record) for record in records],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    relationships = []
    overrides = []
    sheet_xml: dict[int, str] = {}
    for index, (name, rows) in enumerate(sheets.items(), 1):
        relationships.append(
            f'<Relationship Id="rId{index}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{index}.xml"/>'
        )
        overrides.append(
            f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
        sheet_xml[index] = _sheet_xml([REPLAY_HEADERS, *rows])
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        "<sheets>"
        + "".join(
            f'<sheet name="{name}" sheetId="{index}" r:id="rId{index}"/>'
            for index, name in enumerate(sheets, 1)
        )
        + "</sheets></workbook>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        + "".join(overrides)
        + "</Types>"
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/></Relationships>'
    )
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(relationships)
        + "</Relationships>"
    )
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        for index, xml in sheet_xml.items():
            archive.writestr(f"xl/worksheets/sheet{index}.xml", xml)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export memory profile JSON to XLSX.")
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="Profile JSON file(s) or directories containing memory_profile*.json.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output XLSX path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = _profile_paths(args.input)
    records = _load_records(paths)
    write_xlsx(records, args.output)
    logger.info("Wrote %s records from %s profile(s) to %s", len(records), len(paths), args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        main()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.error("%s", exc)
        raise SystemExit(2) from exc
