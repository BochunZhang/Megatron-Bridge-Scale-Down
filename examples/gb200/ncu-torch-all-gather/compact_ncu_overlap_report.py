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
"""Compact v7 Nsight Compute range reports with all-zero inst_executed PC tables.

Drop per-PC SourceCounters metrics in those ranges when the report has no source mapping.
Keep scalar metrics, PM timelines and warp PC samples. Ranges with nonzero instruction
counts or source mappings pass through intact. Requires zstd on PATH.

Pass a pause results directory to create a sibling <path>-simplify directory. Keep
the same layout and filenames, compact every .ncu-repz, and copy the other files.
"""

from __future__ import annotations

import argparse
import logging
import mmap
import os
import re
import shutil
import struct
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import suppress
from pathlib import Path
from typing import BinaryIO, NamedTuple


LOGGER = logging.getLogger(__name__)
U32 = struct.Struct("<I")
Span = tuple[int, int]
ZERO_VALUES = {b"\x20\x00", b"\x28\x00", b"\x15" + bytes(4), b"\x19" + bytes(8)}

# ProfileMetricListElement with a uint64 PC and a uint64 zero value. Match bounded
# runs in C so millions of zero rows do not need to be decoded into Python objects.
ZERO_PC_ROWS = re.compile(
    b"(?:"
    + b"|".join(
        re.escape(bytes((0x1A, pc_bytes + 7, 0x0A, pc_bytes + 1, 0x28)))
        + rb"[\x80-\xff]" * (pc_bytes - 1)
        + rb"[\x00-\x7f]\x12\x02\x28\x00"
        for pc_bytes in range(1, 11)
    )
    + b"){1,4096}"
)


class Field(NamedTuple):
    """A protobuf field and its original byte extent."""

    number: int
    start: int
    end: int
    value: int | Span


def _varint(data: mmap.mmap, offset: int, end: int) -> tuple[int, int]:
    value = 0
    for shift in range(0, 70, 7):
        if offset >= end:
            break
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
    raise ValueError("Invalid protobuf varint")


def _fields(data: mmap.mmap, offset: int, end: int) -> Iterator[Field]:
    while offset < end:
        start = offset
        tag, offset = _varint(data, offset, end)
        wire = tag & 7
        if wire == 0:
            value, offset = _varint(data, offset, end)
        else:
            if wire == 2:
                size, offset = _varint(data, offset, end)
            elif wire in (1, 5):
                size = 8 if wire == 1 else 4
            else:
                raise ValueError(f"Unsupported protobuf wire type {wire}")
            value = (offset, offset + size)
            offset += size
        if offset > end:
            raise ValueError("Truncated protobuf field")
        yield Field(tag >> 3, start, offset, value)


def _span(value: int | Span) -> Span:
    if isinstance(value, int):
        raise ValueError("Expected a length-delimited protobuf field")
    return value


def _message(data: mmap.mmap, offset: int, end: int) -> Span:
    if end - offset < U32.size:
        raise ValueError("Truncated report message length")
    start = offset + U32.size
    stop = start + U32.unpack_from(data, offset)[0]
    if stop > end:
        raise ValueError("Truncated report message")
    return start, stop


def _encode_varint(value: int) -> bytes:
    encoded = bytearray()
    while value >= 0x80:
        encoded.append((value & 0x7F) | 0x80)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


def _source_names(data: mmap.mmap, sections: list[Field]) -> set[bytes]:
    names = set()
    for section in sections:
        fields = list(_fields(data, *_span(section.value)))
        if not any(f.number == 1 and data[slice(*_span(f.value))] == b"SourceCounters" for f in fields):
            continue
        for field in fields:
            if field.number == 13:  # ProfilerSection.SourceMetrics
                for metric in _fields(data, *_span(field.value)):
                    if metric.number == 1:
                        names.update(
                            data[slice(*_span(name.value))]
                            for name in _fields(data, *_span(metric.value))
                            if name.number == 1
                        )
    return names


def _all_zero_pcs(data: mmap.mmap, start: int, end: int) -> bool:
    seen = False
    while start < end:
        run = ZERO_PC_ROWS.match(data, start, end)
        if run:
            seen = True
            start = run.end()
            continue
        field = next(_fields(data, start, end))
        if field.number == 3:  # A nonzero or unfamiliar instance must be preserved.
            return False
        if field.number == 2 and data[slice(*_span(field.value))] not in ZERO_VALUES:
            return False
        start = field.end
    return seen


def _has_discardable_pc_rows(data: mmap.mmap, metric: Span, name: bytes) -> bool:
    # SourceCounters computes this derived value as 32 minus the average
    # predicated-on count, which also produces 32 on unexecuted PCs.
    has_filler_aggregate = name == b"derived__avg_thread_unexecuted_true"
    for field in _fields(data, *metric):
        if field.number == 2 and not has_filler_aggregate and data[slice(*_span(field.value))] not in ZERO_VALUES:
            return False
        if field.number == 3:
            # An instance without a correlation ID is positional.
            return next(_fields(data, *_span(field.value))).number == 1
    return False


def _compact_range(data: mmap.mmap, bounds: Span, names: list[bytes]) -> tuple[list[Span], int]:
    fields = list(_fields(data, *bounds))
    if any(field.number == 15 for field in fields):  # RangeResult.References
        return [bounds], 0
    source_names = _source_names(data, [field for field in fields if field.number == 7])
    if not source_names:
        return [bounds], 0

    metrics = {}
    for field in fields:
        if field.number == 4:  # RangeResult.MetricResults
            name = next(_fields(data, *_span(field.value)))
            if name.number != 1 or not isinstance(name.value, int) or name.value >= len(names):
                raise ValueError("Invalid metric name ID")
            metrics[field.start] = names[name.value]

    counts = [field for field in fields if metrics.get(field.start) == b"inst_executed"]
    if not counts or not all(_all_zero_pcs(data, *_span(field.value)) for field in counts):
        return [bounds], 0

    kept = []
    removed = 0
    for field in fields:
        name = metrics.get(field.start, b"")
        if (
            name in source_names
            and not name.startswith((b"group:", b"pmsampling:", b"profiler__"))
            and b"pcsamp" not in name
            and _has_discardable_pc_rows(data, _span(field.value), name)
        ):
            removed += 1
        else:
            kept.append((field.start, field.end))
    return kept, removed


def _write_compact(data: mmap.mmap, output: BinaryIO) -> int:
    if data[:4] != b"NVR\0":
        raise ValueError("Expected an Nsight Compute report")
    start, offset = _message(data, 4, len(data))
    if not any(f.number == 1 and f.value == 7 for f in _fields(data, start, offset)):
        raise ValueError("Only Nsight Compute report format v7 is supported")
    output.write(data[:offset])
    names = []
    removed = 0

    # NVIDIA's v7 payload stores source records, kernel results, then range results.
    # Other payload types follow them. Copy those records and unknown fields verbatim.
    with memoryview(data) as view:
        while offset < len(data):
            block_start = offset
            start, payload = _message(data, offset, len(data))
            header = list(_fields(data, start, payload))
            values = {f.number: f.value for f in header}
            if 4 in values:  # BlockHeader.StringTable
                names = [data[slice(*_span(f.value))] for f in _fields(data, *_span(values[4])) if f.number == 1]
            size = values.get(5, 0)  # BlockHeader.PayloadSize
            if not isinstance(size, int) or payload + size > len(data):
                raise ValueError("Invalid report block size")
            offset = payload + size
            cursor = payload
            ranges = []
            block_removed = 0
            for _ in range(int(values.get(1, 0)) + int(values.get(2, 0))):
                _, cursor = _message(data, cursor, offset)
            prefix_end = cursor
            for _ in range(int(values.get(7, 0))):
                bounds = _message(data, cursor, offset)
                kept, count = _compact_range(data, bounds, names)
                ranges.append((sum(stop - start for start, stop in kept), kept))
                block_removed += count
                cursor = bounds[1]
            if not block_removed:
                output.write(view[block_start:offset])
                continue

            new_size = prefix_end - payload + offset - cursor + sum(U32.size + n for n, _ in ranges)
            new_header = b"".join(
                b"\x28" + _encode_varint(new_size) if f.number == 5 else data[f.start : f.end] for f in header
            )
            output.write(U32.pack(len(new_header)))
            output.write(new_header)
            output.write(view[payload:prefix_end])
            for size, kept in ranges:
                output.write(U32.pack(size))
                for start, stop in kept:
                    output.write(view[start:stop])
            output.write(view[cursor:offset])
            removed += block_removed
    return removed


def _compact_report(path: Path, output: Path) -> None:
    if output.suffix != ".ncu-repz" or output.exists() or output.is_symlink():
        raise ValueError("Output must be a new .ncu-repz file")
    with tempfile.TemporaryFile() as raw, tempfile.TemporaryFile() as compact:
        with path.open("rb", buffering=0) as report:
            magic = report.read(4)
            report.seek(0)
            if magic == b"NVR\0":
                with mmap.mmap(report.fileno(), 0, access=mmap.ACCESS_READ) as data:
                    removed = _write_compact(data, compact)
            else:
                subprocess.run(["zstd", "-dcq"], stdin=report, stdout=raw, check=True)
                with mmap.mmap(raw.fileno(), 0, access=mmap.ACCESS_READ) as data:
                    removed = _write_compact(data, compact)
        compact.seek(0)
        with tempfile.NamedTemporaryFile(dir=output.parent) as encoded:
            subprocess.run(["zstd", "-cq"], stdin=compact, stdout=encoded, check=True)
            os.link(encoded.name, output)
    LOGGER.info(
        "%s: removed %d metric records, %d -> %d bytes", path, removed, path.stat().st_size, output.stat().st_size
    )


def _compact_directory(path: Path) -> None:
    output = path.with_name(f"{path.name}-simplify")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"{output} already exists")

    def copy_file(source: str, destination: str) -> None:
        report = Path(source)
        if report.suffix == ".ncu-repz":
            try:
                _compact_report(report, Path(destination))
            except (OSError, ValueError, subprocess.CalledProcessError) as exc:
                raise ValueError(f"{report}: {exc}") from exc
        else:
            shutil.copy2(source, destination)

    with tempfile.TemporaryDirectory(prefix=".ncu-simplify-", dir=output.parent) as staging:
        staged = Path(staging) / "results"
        shutil.copytree(path, staged, symlinks=True, copy_function=copy_file)
        output.mkdir()
        try:
            staged.rename(output)
        except OSError:
            with suppress(OSError):
                output.rmdir()
            raise
    LOGGER.info("Simplified results: %s", output)


def main() -> None:
    """Create a simplified results tree or compact a single report."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, help="Pause results directory or input .ncu-rep/.ncu-repz")
    parser.add_argument("output", type=Path, nargs="?", help="New .ncu-repz file in single-report mode")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    path = args.path.expanduser().resolve()
    try:
        if path.is_dir():
            if args.output is not None:
                parser.error("Directory mode writes to <path>-simplify and takes no output argument")
            _compact_directory(path)
        else:
            if args.output is None:
                parser.error("Pass a results directory or provide both an input report and an output file")
            _compact_report(path, args.output.expanduser())
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
