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

"""Validate rank reports from the concurrent AG+RS lockstep comparison."""

import argparse
import csv
import importlib
import logging
import math
import sys
from pathlib import Path
from types import ModuleType


LOGGER = logging.getLogger(__name__)
LAUNCH_COUNT = "launch__count"
LAUNCH_NAMES = "launch__sub_launch_name"
WARPS_ACTIVE = "smsp__warps_active.sum"
LG_THROTTLE = "smsp__warps_issue_stalled_lg_throttle.sum"
LONG_SCOREBOARD = "smsp__warps_issue_stalled_long_scoreboard.sum"
EXPECTED_KERNELS = ("ncclSymkDevKernel_AllGather_STMC", "ncclSymkDevKernel_ReduceScatter_LDMC")


def _read_rank_report(directory: Path, rank: int, *, ncu_report: ModuleType) -> tuple[str, float, float, float]:
    base = directory / f"ag_rs-rank{rank}"
    report = next((path for ext in (".ncu-repz", ".ncu-rep") if (path := base.with_suffix(ext)).is_file()), None)
    if report is None:
        raise ValueError(f"no report for {base.name}")

    context = ncu_report.load_report(str(report))
    if context.num_ranges() != 1:
        raise ValueError("expected exactly one range")
    capture = context.range_by_idx(0)
    if capture.num_actions() != 1:
        raise ValueError("expected exactly one result")
    action = capture.action_by_idx(0)
    if action.workload_type() != ncu_report.IAction.WorkloadType_RANGE:
        raise ValueError("expected an app-range result")

    def counter(name: str) -> float:
        metric = action.metric_by_name(name)
        value = metric.value() if metric is not None else None
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid {name}={value!r}")
        return float(value)

    count = counter(LAUNCH_COUNT)
    if count != 2:
        raise ValueError(f"expected 2 launches, got {count:g}")
    names = action.metric_by_name(LAUNCH_NAMES)
    kernels = [str(names.value(i)) for i in range(names.num_instances())] if names is not None else []
    if len(kernels) != 2 or not all(any(name in kernel for kernel in kernels) for name in EXPECTED_KERNELS):
        raise ValueError(f"expected AllGather_STMC and ReduceScatter_LDMC, got {kernels!r}")

    active = counter(WARPS_ACTIVE)
    if active == 0:
        raise ValueError(f"invalid {WARPS_ACTIVE}=0")
    return "; ".join(kernels), active, counter(LG_THROTTLE), counter(LONG_SCOREBOARD)


def main() -> int:
    """Check both capture directories and write the rank-level acceptance results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="Directory containing lockstep-off and lockstep-on captures")
    parser.add_argument(
        "--ncu", type=Path, required=True, help="Nsight Compute executable for locating its report API"
    )
    parser.add_argument("--off-exit", type=int, default=0, help="Exit status of the lockstep-off launcher")
    parser.add_argument("--on-exit", type=int, default=0, help="Exit status of the lockstep-on launcher")
    args = parser.parse_args()

    sys.path.insert(0, str(args.ncu.resolve().parent / "extras/python"))
    try:
        ncu_report = importlib.import_module("ncu_report")
    except ImportError as exc:
        parser.error(f"cannot load ncu_report from the Nsight Compute installation: {exc}")

    status = 0
    summary = args.output_dir / "acceptance.tsv"
    with summary.open("w", newline="") as stream:
        writer = csv.writer(stream, dialect="excel-tab")
        writer.writerow(
            (
                "mode",
                "launcher_exit",
                "rank",
                "status",
                LAUNCH_COUNT,
                "kernels",
                WARPS_ACTIVE,
                LG_THROTTLE,
                LONG_SCOREBOARD,
                "error",
            )
        )
        for mode, launcher_exit in (("lockstep-off", args.off_exit), ("lockstep-on", args.on_exit)):
            directory = args.output_dir / mode
            if launcher_exit:
                LOGGER.error("%s launcher exited with %d", mode, launcher_exit)
                status = 1
            for rank in range(4):
                try:
                    kernels, active, lg, long_scoreboard = _read_rank_report(directory, rank, ncu_report=ncu_report)
                except (OSError, RuntimeError, ValueError) as exc:
                    LOGGER.error("%s rank %d FAIL: %s", mode, rank, exc)
                    writer.writerow((mode, launcher_exit, rank, "FAIL", "", "", "", "", "", str(exc)))
                    status = 1
                else:
                    LOGGER.info(
                        "%s rank %d PASS: lg_throttle=%.2f%% long_scoreboard=%.2f%%",
                        mode,
                        rank,
                        100 * lg / active,
                        100 * long_scoreboard / active,
                    )
                    writer.writerow((mode, launcher_exit, rank, "PASS", 2, kernels, active, lg, long_scoreboard, ""))
    LOGGER.info("Rank acceptance: %s", summary)
    return status


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(main())
