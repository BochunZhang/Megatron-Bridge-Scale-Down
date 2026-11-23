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
"""Profile the standalone GB200 BF16 GEMM sweep with Nsight Compute."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import NamedTuple


LOGGER = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent
DEFAULT_M_VALUES = tuple(1 << power for power in range(14))
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


def _default_metrics() -> dict[str, Counter]:
    return {
        **{group: Counter(*options[0]) for group, options in DRAM.items()},
        **{group: Counter(PM + metric, 1) for group, metric in LINKS.items()},
    }


def _catalog(ncu: str, collection: str, env: dict[str, str], output_dir: Path) -> set[str]:
    result = subprocess.run(
        [ncu, "--query-metrics-mode", "all", "--query-metrics-collection", collection, "--devices", "0"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    raw_catalog = output_dir / f"metrics-{collection}.txt"
    raw_catalog.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        LOGGER.warning("NCU metric query for %s failed. See %s", collection, raw_catalog)
        return set()
    return {match["metric"] for match in METRIC_LINE.finditer(result.stdout)}


def _available_metrics(ncu: str, env: dict[str, str], output_dir: Path) -> dict[str, Counter]:
    profiler = _catalog(ncu, "profiling", env, output_dir)
    sampled = _catalog(ncu, "pmsampling", env, output_dir)
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
        raise RuntimeError(f"NCU must expose directional DRAM byte or sector counters. See {output_dir}/metrics-*.txt")
    for group in LINKS.keys() - chosen.keys():
        LOGGER.warning("%s is unavailable in the PM sampling catalog", group)
    return chosen


def _command(args: argparse.Namespace, *, case_dir: Path, m: int, metrics: dict[str, Counter]) -> list[str]:
    includes = []
    for index in range(args.gemm_rounds):
        includes.extend(("--nvtx-include", f"gemm{index:02d}/"))
    pm_args = (
        ["--pm-sampling-interval", str(PM_INTERVAL_NS)]
        if any(counter.metric.startswith(PM) for counter in metrics.values())
        else []
    )
    return [
        args.ncu,
        "--replay-mode",
        "application",
        "--cache-control",
        "none",
        "--clock-control",
        "none",
        "--target-processes",
        "all",
        "--nvtx",
        *includes,
        *pm_args,
        "--metrics",
        ",".join([DURATION, *(counter.metric for counter in metrics.values())]),
        "--export",
        str(case_dir / "gemm"),
        "--",
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=1",
        str(HERE / "ce_hbm_benchmark.py"),
        "--mode",
        "gemm",
        "--ncu-gemm",
        "--case",
        case_dir.name,
        "--gemm",
        f"[{m},8192,7168]",
        "--warmup",
        str(args.warmup),
        "--gemm-rounds",
        str(args.gemm_rounds),
        "--comm-rounds",
        "1",
        "--size-mib",
        "1024",
        "--output-dir",
        str(case_dir),
    ]


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


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0, help="Physical GPU index in PCI bus order")
    parser.add_argument(
        "--m-values", default=",".join(map(str, DEFAULT_M_VALUES)), help="Comma-separated GEMM M values"
    )
    parser.add_argument("--gemm-rounds", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--ncu", default=os.environ.get("NCU_BIN", "ncu"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--dry-run", action="store_true", help="Show commands without creating files or querying the GPU"
    )
    args = parser.parse_args(argv)
    try:
        args.m_values = [int(item) for item in args.m_values.split(",")]
    except ValueError:
        parser.error("--m-values expects comma-separated integers")
    if any(not 1 <= m <= 8192 for m in args.m_values) or len(set(args.m_values)) != len(args.m_values):
        parser.error("--m-values requires distinct integers in 1..8192")
    if args.gpu < 0 or args.gemm_rounds < 1 or args.warmup < 1:
        parser.error("--gpu must be nonnegative and --gemm-rounds/--warmup must be positive")
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_dir = HERE.parents[2] / "results/gb200/hbm-conflict" / f"{stamp}-ncu-gemm"
    args.output_dir = args.output_dir.resolve()
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Capture one NCU report per shape and summarize its kernel traffic."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stdout)
    args = _parse_args(argv)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID", LC_ALL="C")
    lock = ["nvidia-smi", "-i", str(args.gpu), "--lock-gpu-clocks=tdp,tdp"]
    counters = _default_metrics()
    if args.dry_run:
        LOGGER.info("%s", shlex.join(lock))
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        version = subprocess.run([args.ncu, "--version"], env=env, capture_output=True, text=True, check=True)
        (args.output_dir / "ncu-version.txt").write_text(version.stdout, encoding="utf-8")
        counters = _available_metrics(args.ncu, env, args.output_dir)
        with (args.output_dir / "gpu_clock_lock.log").open("w", encoding="utf-8") as handle:
            subprocess.run(lock, stdout=handle, stderr=subprocess.STDOUT, check=True)
    cases = []
    failed = False
    for m in args.m_values:
        case_dir = args.output_dir / f"gemm-m{m}-n8192-k7168"
        command = _command(args, case_dir=case_dir, m=m, metrics=counters)
        command_text = shlex.join(
            ["env", f"CUDA_VISIBLE_DEVICES={args.gpu}", "CUDA_DEVICE_ORDER=PCI_BUS_ID", *command]
        )
        LOGGER.info("%s", command_text)
        if args.dry_run:
            continue
        case_dir.mkdir()
        (case_dir / "command.txt").write_text(command_text + "\n", encoding="utf-8")
        record = {"gemm_mnk": [m, 8192, 7168], "case": case_dir.name}
        try:
            with (case_dir / "ncu.log").open("w", encoding="utf-8") as handle:
                subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
            with (case_dir / "gemm-raw.csv").open("w", encoding="utf-8") as handle:
                subprocess.run(
                    [
                        args.ncu,
                        "--import",
                        str(case_dir / "gemm.ncu-rep"),
                        "--page",
                        "raw",
                        "--csv",
                        "--print-units",
                        "base",
                        "--print-nvtx-rename",
                        "kernel",
                    ],
                    env=env,
                    stdout=handle,
                    check=True,
                )
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
        except (OSError, subprocess.CalledProcessError, ValueError, KeyError) as error:
            LOGGER.error("%s failed: %s", case_dir, error)
            record["error"] = str(error)
            failed = True
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
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
