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

"""Export per-step GPU utilization values from training logs as JSON."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import cast


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.performance.utils.evaluate import get_metrics_from_logfiles  # noqa: E402


LOGGER = logging.getLogger(__name__)


def export_gpu_utilization(log_paths: list[Path], *, output_path: Path) -> dict[str, float]:
    """Extract per-step GPU utilization metrics and write them as JSON.

    Args:
        log_paths: Training log files to parse.
        output_path: Destination JSON file.

    Returns:
        GPU utilization values keyed by zero-based training step.
    """
    metrics = cast(
        dict[str, float],
        get_metrics_from_logfiles([str(log_path) for log_path in log_paths], "GPU utilization"),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Wrote %d GPU utilization values to %s", len(metrics), output_path)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-file",
        action="append",
        required=True,
        type=Path,
        dest="log_files",
        help="Training log to parse; repeat for multiple log files.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Destination JSON file.")
    return parser


def main() -> int:
    """Run the GPU utilization exporter."""
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    export_gpu_utilization(args.log_files, output_path=args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
