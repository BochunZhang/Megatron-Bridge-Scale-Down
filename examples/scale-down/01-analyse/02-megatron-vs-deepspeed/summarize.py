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

"""Collect one row per Megatron run for comparison with DeepSpeed metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = (
    "status",
    "model",
    "precision",
    "run_name",
    "recipe",
    "train_iters",
    "warmup_steps",
    "global_batch_size",
    "micro_batch_size",
    "recompute_granularity",
    "recompute_modules",
    "fine_grained_offload",
    "offload_modules",
    "optimizer_cpu_offload",
    "optimizer_offload_fraction",
    "overlap_cpu_optimizer_d2h_h2d",
    "result_dir",
)


def _row(summary_path: Path) -> dict[str, Any]:
    """Merge the run summary and launcher metadata."""
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config_path = summary_path.with_name("config.json")
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    merged = {**config, **summary}
    return {field: merged.get(field, "") for field in FIELDS}


def main() -> None:
    """Write ``summary.csv`` below a result tree."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    output = args.output or args.results_root / "summary.csv"
    rows = [_row(path) for path in sorted(args.results_root.rglob("summary.json"))]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
