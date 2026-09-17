# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Poll all NVIDIA GPUs and write per-GPU memory traces as JSON files.

The monitor samples every visible GPU until a training PID exits, a duration
expires, a stop file appears, or the monitor receives SIGINT/SIGTERM. Samples
are kept in memory during the run and written atomically when monitoring ends,
so each JSON file is a complete trace rather than a partially written log.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import TypedDict


LOGGER = logging.getLogger(__name__)
NVIDIA_SMI_QUERY = "index,uuid,name,memory.used,memory.total"


class GpuSample(TypedDict):
    """One point in a GPU memory trace."""

    timestamp: str
    elapsed_seconds: float
    memory_used_mib: int
    memory_total_mib: int
    memory_used_percent: float | None


class GpuTrace(TypedDict):
    """Metadata and samples written for one GPU."""

    schema_version: int
    gpu_index: int
    gpu_uuid: str
    gpu_name: str
    memory_total_mib: int
    started_at: str
    ended_at: str
    sample_interval_seconds: float
    sample_count: int
    max_memory_used_mib: int
    max_memory_used_percent: float | None
    max_memory_at: str | None
    samples: list[GpuSample]


def utc_now() -> datetime:
    """Return the current UTC time."""

    return datetime.now(timezone.utc)


def parse_mib(value: str, field_name: str) -> int:
    """Parse a unit-less MiB field returned by ``nvidia-smi``."""

    try:
        return int(value.strip())
    except ValueError as exc:
        raise RuntimeError(f"nvidia-smi returned invalid {field_name}: {value!r}") from exc


def query_gpu_memory() -> list[dict[str, int | str]]:
    """Query memory usage and metadata for every GPU reported by ``nvidia-smi``.

    Returns:
        One dictionary per GPU, including its index, UUID, name, used memory,
        and total memory in MiB.

    Raises:
        RuntimeError: If ``nvidia-smi`` is unavailable, fails, or returns an
            unexpected row.
    """

    command = [
        "nvidia-smi",
        f"--query-gpu={NVIDIA_SMI_QUERY}",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        detail = getattr(exc, "stderr", None) or str(exc)
        raise RuntimeError(f"failed to query NVIDIA GPUs: {detail.strip()}") from exc

    rows = [row for row in csv.reader(io.StringIO(result.stdout)) if row]
    if not rows:
        raise RuntimeError("nvidia-smi returned no GPUs")

    gpus: list[dict[str, int | str]] = []
    for row in rows:
        if len(row) != 5:
            raise RuntimeError(f"unexpected nvidia-smi row with {len(row)} fields: {row!r}")
        gpus.append(
            {
                "index": parse_mib(row[0], "GPU index"),
                "uuid": row[1].strip(),
                "name": row[2].strip(),
                "memory_used_mib": parse_mib(row[3], "memory.used"),
                "memory_total_mib": parse_mib(row[4], "memory.total"),
            }
        )
    return gpus


def process_is_alive(pid: int) -> bool:
    """Return whether a process still exists and is signalable."""

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def append_samples(
    traces: dict[int, GpuTrace],
    gpu_rows: list[dict[str, int | str]],
    *,
    started_monotonic: float,
    started_at: datetime,
) -> None:
    """Append one timestamped sample for every queried GPU."""

    timestamp = utc_now()
    elapsed_seconds = round(time.monotonic() - started_monotonic, 6)
    timestamp_text = timestamp.isoformat()

    for row in gpu_rows:
        gpu_index = int(row["index"])
        memory_used_mib = int(row["memory_used_mib"])
        memory_total_mib = int(row["memory_total_mib"])
        memory_used_percent = round(memory_used_mib * 100.0 / memory_total_mib, 3) if memory_total_mib else None
        sample: GpuSample = {
            "timestamp": timestamp_text,
            "elapsed_seconds": elapsed_seconds,
            "memory_used_mib": memory_used_mib,
            "memory_total_mib": memory_total_mib,
            "memory_used_percent": memory_used_percent,
        }

        if gpu_index not in traces:
            traces[gpu_index] = {
                "schema_version": 1,
                "gpu_index": gpu_index,
                "gpu_uuid": str(row["uuid"]),
                "gpu_name": str(row["name"]),
                "memory_total_mib": memory_total_mib,
                "started_at": started_at.isoformat(),
                "ended_at": "",
                "sample_interval_seconds": 0.0,
                "sample_count": 0,
                "max_memory_used_mib": memory_used_mib,
                "max_memory_used_percent": memory_used_percent,
                "max_memory_at": timestamp_text,
                "samples": [],
            }

        trace = traces[gpu_index]
        trace["samples"].append(sample)
        trace["sample_count"] = len(trace["samples"])
        if memory_used_mib >= trace["max_memory_used_mib"]:
            trace["max_memory_used_mib"] = memory_used_mib
            trace["max_memory_used_percent"] = memory_used_percent
            trace["max_memory_at"] = timestamp_text


def write_traces(
    traces: dict[int, GpuTrace],
    output_dir: Path,
    ended_at: datetime,
    interval: float,
) -> None:
    """Write one atomically replaced JSON file for each GPU."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for gpu_index, trace in sorted(traces.items()):
        trace["ended_at"] = ended_at.isoformat()
        trace["sample_interval_seconds"] = interval
        output_path = output_dir / f"gpu_{gpu_index}.json"
        temporary_path = output_dir / f".gpu_{gpu_index}.json.tmp"
        temporary_path.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
        temporary_path.replace(output_path)
        LOGGER.info(
            "wrote GPU %s trace: %s (%d samples, peak %d MiB)",
            gpu_index,
            output_path,
            trace["sample_count"],
            trace["max_memory_used_mib"],
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    default_output_dir = Path("gpu_memory_traces") / utc_now().strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Sampling interval in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--pid",
        type=int,
        help="Stop after this training process exits.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="Stop after this many seconds; otherwise run until --pid, --stop-file, or a signal.",
    )
    parser.add_argument(
        "--stop-file",
        type=Path,
        help="Stop after this file exists. The monitor does not delete the file.",
    )
    parser.add_argument(
        "--ready-file",
        type=Path,
        help="Create this file after the first successful sample; useful for launcher synchronization.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help=f"Directory for gpu_<index>.json files (default: {default_output_dir}).",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Console log level (default: INFO).",
    )
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate positive numeric command-line options."""

    if args.interval <= 0:
        parser.error("--interval must be greater than zero")
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be greater than zero")
    if args.pid is not None and args.pid <= 0:
        parser.error("--pid must be greater than zero")


def monitor(args: argparse.Namespace, stop_event: Event) -> int:
    """Run the polling loop and write completed traces."""

    started_at = utc_now()
    started_monotonic = time.monotonic()
    traces: dict[int, GpuTrace] = {}

    try:
        while True:
            rows = query_gpu_memory()
            append_samples(
                traces,
                rows,
                started_monotonic=started_monotonic,
                started_at=started_at,
            )
            if args.ready_file is not None and not args.ready_file.exists():
                args.ready_file.parent.mkdir(parents=True, exist_ok=True)
                args.ready_file.touch()
            LOGGER.debug(
                "sampled GPU memory: %s",
                ", ".join(f"GPU {row['index']}={row['memory_used_mib']} MiB" for row in rows),
            )

            elapsed = time.monotonic() - started_monotonic
            if stop_event.is_set():
                LOGGER.info("stopping after signal")
                break
            if args.pid is not None and not process_is_alive(args.pid):
                LOGGER.info("training process %s exited", args.pid)
                break
            if args.stop_file is not None and args.stop_file.exists():
                LOGGER.info("stopping because %s exists", args.stop_file)
                break
            if args.duration is not None and elapsed >= args.duration:
                LOGGER.info("stopping after %.3f seconds", elapsed)
                break

            sleep_for = args.interval
            if args.duration is not None:
                sleep_for = min(sleep_for, max(args.duration - elapsed, 0.0))
            stop_event.wait(sleep_for)
    except RuntimeError as exc:
        LOGGER.error("%s", exc)
        return 1
    finally:
        if traces:
            write_traces(traces, args.output_dir, utc_now(), args.interval)

    return 0


def main() -> int:
    """Parse arguments, install signal handlers, and run the monitor."""

    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    stop_event = Event()

    def request_stop(signum: int, _frame: object) -> None:
        LOGGER.info("received signal %s", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    return monitor(args, stop_event)


if __name__ == "__main__":
    sys.exit(main())
