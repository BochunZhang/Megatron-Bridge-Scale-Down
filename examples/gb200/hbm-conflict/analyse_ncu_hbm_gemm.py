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
"""Select NCU counters and summarize standalone GEMM profile data."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import statistics
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple


LOGGER = logging.getLogger(__name__)
DURATION = "gpu__time_duration.sum"
DRAM = {
    "dram_read": (("dram__bytes_read.sum", 1), ("dram__sectors_read.sum", 32)),
    "dram_write": (("dram__bytes_write.sum", 1), ("dram__sectors_write.sum", 32)),
}
LINKS = {
    "nvlink_rx_wire": "nvlrx__bytes.sum",
    "nvlink_tx_wire": "nvltx__bytes.sum",
    "c2c_rx_user": "ctc__rx_bytes_data_user.sum",
    "c2c_tx_user": "ctc__tx_bytes_data_user.sum",
}
PM = "pmsampling:"
PM_INTERVAL_NS = 1000
METRIC_LINE = re.compile(r"^\s*(?P<metric>[a-zA-Z0-9_]+__[a-zA-Z0-9_.]+)(?:\s|$)", re.M)
GEMM_LABEL = re.compile(r"gemm\d+")
TIME_NS = {"ns": 1.0, "nsecond": 1.0, "us": 1e3, "usecond": 1e3, "ms": 1e6, "msecond": 1e6, "s": 1e9, "second": 1e9}


class Counter(NamedTuple):
    """NCU metric and its unit conversion to bytes."""

    metric: str
    unit_bytes: int


def _catalog(path: Path) -> set[str]:
    return {match["metric"] for match in METRIC_LINE.finditer(path.read_text(encoding="utf-8"))}


def _available_metrics(profiler: set[str], sampled: set[str]) -> dict[str, Counter]:
    chosen = {}
    for group, options in DRAM.items():
        for name, unit_bytes in options:
            if name in profiler:
                chosen[group] = Counter(name, unit_bytes)
                break
    for group, name in LINKS.items():
        if name in sampled:
            chosen[group] = Counter(PM + name, 1)
    if not DRAM.keys() <= chosen.keys():
        raise ValueError("NCU must expose directional DRAM byte or sector counters. See metrics-profiling.txt")
    return chosen


def _number(text: str) -> float | None:
    try:
        result = float(text.strip().replace(",", ""))
    except ValueError:
        return None
    return result if math.isfinite(result) and result >= 0 else None


def _summary(raw_csv: Path, counters: dict[str, Counter], *, rounds: int) -> dict:
    with raw_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next((line for line in reader if "Kernel Name" in line and DURATION in line), None)
        if header is None:
            raise ValueError(f"{raw_csv}: no NCU raw-page CSV header")
        units = dict(zip(header, next(reader, [])))
        time_unit = units.get(DURATION, "")
        if time_unit not in TIME_NS:
            raise ValueError(f"{raw_csv}: unexpected NCU duration unit {time_unit!r}")
        groups = defaultdict(list)
        for line in reader:
            if not any(line):
                continue
            row = dict(zip(header, line))
            label = row.get("Kernel Name", "")
            if not GEMM_LABEL.fullmatch(label):
                raise ValueError(f"{raw_csv}: expected a GEMM NVTX label, got {label!r}")
            groups[label].append(row)
    expected = {f"gemm{index:02d}" for index in range(rounds)}
    if groups.keys() != expected:
        raise ValueError(f"{raw_csv}: expected {rounds} measured GEMMs, got {sorted(groups)}")

    per_gemm = []
    for label in sorted(groups, key=lambda key: int(key[4:])):
        group = groups[label]
        times = [_number(row.get(DURATION, "")) for row in group]
        if any(value is None or value <= 0 for value in times):
            raise ValueError(f"{raw_csv}: missing positive kernel time in {label}")
        metrics = {}
        for network in (*DRAM, *LINKS):
            counter = counters.get(network)
            size = None
            if counter is not None:
                column = counter.metric
                # Some NCU raw pages remove the PM sampling prefix on export.
                if column not in header and column.startswith(PM):
                    column = column.removeprefix(PM)
                values = [_number(row.get(column, "")) for row in group]
                if all(value is not None for value in values):
                    size = sum(values) * counter.unit_bytes
            metrics[network] = size
        per_gemm.append(
            {"name": label, "kernels": len(group), "kernel_ns": sum(times) * TIME_NS[time_unit], "bytes": metrics}
        )

    totals = {}
    for key in (*DRAM, *LINKS):
        valid = [(gemm["bytes"][key], gemm["kernel_ns"]) for gemm in per_gemm if gemm["bytes"][key] is not None]
        complete = len(valid) == rounds
        counter = counters.get(key)
        totals[key] = {
            "metric": counter.metric if counter is not None else None,
            "gemms_with_data": len(valid),
            "mean_bytes": statistics.fmean(size for size, _ in valid) if complete else None,
            "total_bytes": sum(size for size, _ in valid) if complete else None,
            "mean_gb_per_s": statistics.fmean(size / ns for size, ns in valid) if complete else None,
        }
    return {
        "mean_ncu_kernel_us": statistics.fmean(gemm["kernel_ns"] for gemm in per_gemm) / 1000,
        "gemms": per_gemm,
        "traffic": totals,
    }


def _record_case(args: argparse.Namespace) -> int:
    case_dir = args.case_dir
    record = {"gemm_mnk": [args.m, 8192, 7168], "case": case_dir.name}
    try:
        if args.error is not None:
            raise ValueError(args.error)
        metrics = set(args.metrics.split(","))
        sampled = {name.removeprefix(PM) for name in metrics if name.startswith(PM)}
        counters = _available_metrics(metrics, sampled)
        worker_record = json.loads((case_dir / "events.rank_00.json").read_text(encoding="utf-8"))
        worker_times = worker_record["gemm_ms"]
        if (
            worker_record["output_correct"] is not True
            or worker_record["world_size"] != 1
            or worker_record["mode"] != "gemm"
            or worker_record["gemm_mnk"] != record["gemm_mnk"]
            or len(worker_times) != args.gemm_rounds
        ):
            raise ValueError(f"invalid worker result in {case_dir}")
        record.update(_summary(case_dir / "gemm-raw.csv", counters, rounds=args.gemm_rounds))
        record["worker_cuda_event_mean_us"] = statistics.fmean(stop - start for start, stop in worker_times) * 1000
        if any(record["traffic"][key]["mean_bytes"] is None for key in DRAM):
            raise ValueError("directional DRAM counts are missing for one or more GEMMs")
    except (OSError, ValueError, KeyError) as error:
        LOGGER.error("%s failed: %s", case_dir, error)
        record["error"] = str(error)

    output = case_dir.parent / "summary.json"
    cases = json.loads(output.read_text(encoding="utf-8"))["cases"] if output.is_file() else []
    cases.append(record)
    summary = {
        "gpu": args.gpu,
        "warmup": args.warmup,
        "gemm_rounds": args.gemm_rounds,
        "link_pm_sampling_interval_ns": PM_INTERVAL_NS,
        "measurement": "DRAM uses kernel counters. NVLink/C2C use device-wide PM samples during isolated GEMM profiling.",
        "bandwidth_basis": "Per GEMM, bytes divided by summed NCU kernel duration in ns, then averaged in decimal GB/s.",
        "cases": cases,
    }
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return int("error" in record)


def main(argv: Sequence[str] | None = None) -> int:
    """Print selected metric names or update the sweep summary from existing files."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    selection = commands.add_parser("metrics", help="Select supported counters from the NCU catalogs")
    selection.add_argument("--profiler-catalog", type=Path, required=True)
    selection.add_argument("--pm-catalog", type=Path)
    summary = commands.add_parser("summarize", help="Update summary.json from one captured case")
    summary.add_argument("--case-dir", type=Path, required=True)
    summary.add_argument("--metrics", required=True)
    summary.add_argument("--m", type=int, required=True)
    summary.add_argument("--gpu", type=int, required=True)
    summary.add_argument("--warmup", type=int, required=True)
    summary.add_argument("--gemm-rounds", type=int, required=True)
    summary.add_argument("--error")
    args = parser.parse_args(argv)
    if args.command == "summarize":
        return _record_case(args)
    try:
        profiler = _catalog(args.profiler_catalog)
        sampled = _catalog(args.pm_catalog) if args.pm_catalog is not None else set()
        counters = _available_metrics(profiler, sampled)
        for group in LINKS.keys() - counters.keys():
            LOGGER.warning("%s is unavailable in the PM sampling catalog", group)
    except (OSError, ValueError) as error:
        LOGGER.error("%s", error)
        return 1
    sys.stdout.write(",".join([DURATION, *(counter.metric for counter in counters.values())]) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
