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
"""Copy Nsight Systems reports to a sibling directory with the original layout."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory


logger = logging.getLogger(__name__)
REPORT_SUFFIXES = {".nsys", ".nsys-rep"}


def main(argv: list[str] | None = None) -> int:
    """Copy reports to `SOURCE-nsys` and refresh reports already copied there."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Results directory to scan recursively")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    count = 0
    try:
        supplied_path = Path(os.path.abspath(args.source.expanduser()))
        source = supplied_path.resolve(strict=True)
        if not source.is_dir() or not supplied_path.name:
            raise ValueError(f"Expected a source directory with a parent, got {supplied_path}")
        output = supplied_path.with_name(f"{supplied_path.name}-nsys")
        if output.is_symlink():
            raise ValueError(f"Output directory is a symbolic link: {output}")
        output_root = output.resolve()
        if output_root.is_relative_to(source):
            raise ValueError(f"Output directory is inside the source directory: {output}")
        output.mkdir(exist_ok=True)
        for report in source.rglob("*"):
            if report.suffix not in REPORT_SUFFIXES or not report.is_file():
                continue
            target = output / report.relative_to(source)
            if not target.parent.resolve().is_relative_to(output_root):
                raise ValueError(f"Report destination escapes the output directory: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            # Publish the report after a complete copy, including on reruns.
            with TemporaryDirectory(dir=target.parent) as staging:
                temporary = Path(staging) / target.name
                shutil.copy2(report, temporary)
                temporary.replace(target)
            count += 1
    except (OSError, ValueError) as exc:
        logger.error("Copied %d report(s) before stopping: %s", count, exc)
        return 1
    logger.info("Copied %d Nsight Systems report(s) to %s", count, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
