# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure GEMM-only and CE AllGather overlap using CUDA kernel and payload traces."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sqlite3
import subprocess
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from analyse_ce_hbm_kernel_trace import analyse_kernel_case


_LOG = logging.getLogger(__name__)
_HERE = Path(__file__).resolve().parent
_M_VALUES = tuple(2**power for power in range(14))
_CASE_NAMES = {
    "task1": "GEMM only",
    "task3": "HBM-source multicast",
    "task4": "CPU-source multicast",
    "task7": "HBM-source unicast",
    "task8": "CPU-source unicast",
}
_FIRST_SIZE_MIB = 1024
_MIN_GEMMS_PER_AG_ROUND = 2


def _m_values(value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("use comma-separated powers of two from 1 to 8192") from error
    if len(set(sizes)) != len(sizes) or any(size not in _M_VALUES for size in sizes):
        raise argparse.ArgumentTypeError("use distinct powers of two from 1 to 8192")
    return tuple(sorted(sizes))


def _cases(value: str) -> tuple[str, ...]:
    ids = value.split(",")
    if len(set(ids)) != len(ids) or any(case not in _CASE_NAMES for case in ids):
        raise argparse.ArgumentTypeError("use distinct case IDs from task1,task3,task4,task7,task8")
    return tuple(case for case in _CASE_NAMES if case == "task1" or case in ids)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--m-values", type=_m_values, default=_M_VALUES, help="Subset of the default M=1,2,…,8192 sweep"
    )
    default_gpus = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0,1,2,3").split(",")[:4]
    parser.add_argument("--gpus", default=",".join(default_gpus), help="Ordered list of physical GPU IDs")
    parser.add_argument("--cases", type=_cases, default=tuple(_CASE_NAMES), help="Case subset. Always includes task1")
    parser.add_argument("--gemm-rounds", type=int, default=16)
    parser.add_argument("--comm-rounds", type=int, default=25, help="Minimum measured AG rounds per attempt")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--max-size-mib", type=int, default=8192, help="Maximum AG output MiB per GPU when retrying")
    parser.add_argument("--nsys", default=os.environ.get("NSYS_BIN", "NsightSystems-2026.6.1/bin/nsys"))
    parser.add_argument("--nccl-lib", type=Path, help="NCCL shared library for the AG cases")
    parser.add_argument("--mc-self-import", choices=("0", "1"), help="Self-import setting for multicast cases")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Show the initial 1 GiB commands")
    args = parser.parse_args(argv)
    if args.gemm_rounds < _MIN_GEMMS_PER_AG_ROUND or args.comm_rounds < 1 or args.warmup < 1:
        parser.error("use at least 2 GEMM rounds, 1 communication round, and 1 warmup round")
    if args.max_size_mib < _FIRST_SIZE_MIB:
        parser.error("--max-size-mib must be at least 1024")
    return args


def _launch(args: argparse.Namespace, *, case: str, m: int, size_mib: int, output_dir: Path) -> None:
    command = [
        "bash",
        str(_HERE / "run_ce_hbm_benchmark.sh"),
        "--kernel-trace-only",
        "--gpus",
        args.gpus,
        "--nsys",
        args.nsys,
        "--cases",
        case,
        "--gemm",
        f"[{m},8192,7168]",
        "--size-mib",
        str(size_mib),
        "--gemm-rounds",
        str(args.gemm_rounds),
        "--comm-rounds",
        str(args.comm_rounds),
        "--warmup",
        str(args.warmup),
        "--output-dir",
        str(output_dir),
    ]
    if case != "task1" and args.nccl_lib is not None:
        command.extend(("--nccl-lib", str(args.nccl_lib.resolve())))
    if case in ("task3", "task4") and args.mc_self_import is not None:
        command.extend(("--mc-self-import", args.mc_self_import))
    if args.dry_run:
        command.append("--dry-run")
    subprocess.run(command, check=True)


def _run_shape(args: argparse.Namespace, *, case: str, m: int, root: Path) -> dict:
    row = {"case_id": case, "m": m, "attempts": []}
    size_mib = _FIRST_SIZE_MIB
    while True:
        attempt_dir = root / f"m{m:04d}" / case / f"size-{size_mib}mib"
        attempt = {"output_dir": str(attempt_dir), "size_mib": size_mib}
        row["attempts"].append(attempt)
        try:
            _launch(args, case=case, m=m, size_mib=size_mib, output_dir=attempt_dir)
            if args.dry_run:
                attempt["status"] = "dry_run"
                return row
            produced = [path for path in attempt_dir.glob(f"{case}-*") if path.is_dir()]
            if len(produced) != 1:
                raise ValueError(f"{attempt_dir}: expected one captured case, found {len(produced)}")
            trace = analyse_kernel_case(produced[0], nsys_bin=args.nsys)
            if (
                trace["gpu0_gemm_count"] != args.gemm_rounds
                or trace["gemm_mnk"] != [m, 8192, 7168]
                or trace["size_bytes"] != size_mib << 20
                or (case != "task1" and trace["comm_rounds_measured"] < args.comm_rounds)
            ):
                raise ValueError(f"{produced[0]}: captured workload differs from requested shape, size, or rounds")
            attempt["trace"] = trace
        except (OSError, ValueError, sqlite3.Error, subprocess.CalledProcessError) as error:
            attempt.update(status="error", error=str(error))
            _LOG.error("%s M=%d, %d MiB. %s", case, m, size_mib, error)
            return row

        covers_two = trace["gpu0_max_gemms_per_ag_round"] >= _MIN_GEMMS_PER_AG_ROUND
        if case == "task1" or covers_two:
            attempt["status"] = "ok"
            return row
        attempt["status"] = "insufficient_overlap"
        if size_mib == args.max_size_mib:
            _LOG.error("%s M=%d reached %d MiB without two fully covered GEMMs in one AG round", case, m, size_mib)
            return row
        size_mib = min(2 * size_mib, args.max_size_mib)
        _LOG.info("%s M=%d has insufficient overlap. Retrying with %d MiB per GPU", case, m, size_mib)


def _write_results(root: Path, rows: list[dict]) -> None:
    baseline_means = {}
    for row in rows:
        attempt = row["attempts"][-1]
        if row["case_id"] == "task1" and attempt["status"] == "ok":
            baseline_means[row["m"]] = attempt["trace"]["gpu0_gemm_mean_us"]
    lines = [
        "# GB200 GEMM and CE AllGather kernel sweep",
        "",
        "GPU means use the first selected GPU. Each AG result includes GEMMs fully inside a single AG payload round.",
        "The reported delta pairs each accepted case with the same-shape GEMM-only baseline.",
        "",
        "| M | Case | Status | AG MiB/GPU | AG rounds | GEMMs in mean | Max per AG round | Mean µs | Δ vs solo |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        attempt = row["attempts"][-1]
        trace = attempt.get("trace", {})
        baseline = baseline_means.get(row["m"])
        solo = row["case_id"] == "task1"
        duration = trace.get("gpu0_gemm_mean_us") if attempt["status"] == "ok" else None
        delta = (duration / baseline - 1) * 100 if not solo and duration is not None and baseline else None
        row["gpu0_slowdown_percent"] = delta
        cells = [
            str(row["m"]),
            _CASE_NAMES[row["case_id"]],
            attempt["status"],
            str(attempt["size_mib"]) if not solo else "—",
            str(trace.get("comm_rounds_measured", "—")),
            str(trace.get("gpu0_covered_gemm_count" if not solo else "gpu0_gemm_count", "—")),
            str(trace.get("gpu0_max_gemms_per_ag_round", "—")) if not solo else "—",
            f"{duration:.3f}" if duration is not None else "—",
            f"{delta:+.1f}%" if delta is not None else "—",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    (root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Run selected GEMM shapes and report AG-qualified GPU kernel timing."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)
    args.nsys = str(Path(shutil.which(args.nsys) or args.nsys).resolve())
    root = args.output_dir or (
        _HERE.parents[2]
        / "results"
        / "gb200"
        / "hbm-conflict"
        / datetime.now().strftime("%Y%m%d-%H%M%S-ce-kernel-sweep")
    )
    root = root.resolve()
    if not args.dry_run:
        root.mkdir(parents=True)
    rows = []
    for m in args.m_values:
        for case in args.cases:
            rows.append(_run_shape(args, case=case, m=m, root=root))
            if not args.dry_run:
                _write_results(root, rows)
    if not args.dry_run:
        _LOG.info("Results in %s", root)
    return int(any(row["attempts"][-1]["status"] not in ("ok", "dry_run") for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
