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

"""Measure C2C bandwidth while CPU Gloo/ibverbs P2P traffic runs in the background.

The CPU background traffic continuously sends and receives CPU tensors over
fixed pairs ``0 -> 1`` and ``2 -> 3``. Gloo is configured with its ibverbs
transport, so the CPU payload path uses RDMA Verbs. The benchmark measures GPU
H2D and D2H copies while the CPU P2P traffic is active. No background tensor
is allocated on the GPU; the GPU is used only for the C2C copy under test.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import cast

import torch
import torch.distributed as dist


LOGGER = logging.getLogger(__name__)
BYTES_PER_MIB = 1024 * 1024
BYTES_PER_GB = 1_000_000_000
EXPECTED_WORLD_SIZE = 4
BACKGROUND_STOP_CHECK_INTERVAL = 32
CPU_P2P = "p2p"
GLOO_IBVERBS_TRANSPORT = "IBVERBS"


@dataclass(frozen=True)
class CopyMeasurement:
    """Result of one host/device copy direction."""

    direction: str
    bandwidth_gb_s: float
    elapsed_seconds: float
    transferred_bytes: int


@dataclass(frozen=True)
class BackgroundMeasurement:
    """Result of one CPU background operation window."""

    direction: str
    peer_rank: int | None
    iterations: int
    elapsed_seconds: float
    average_completion_ms: float
    bandwidth_gb_s: float
    transferred_bytes: int
    started_monotonic_seconds: float
    finished_monotonic_seconds: float


@dataclass(frozen=True)
class RankMeasurement:
    """All measurements and topology metadata for one distributed rank."""

    rank: int
    hostname: str
    gpu_name: str
    gpu_pci_bus_id: str
    background_mode: str
    background_role: str
    peer_rank: int | None
    c2c_buffer_mib: int
    background_buffer_mib: int
    baseline: dict[str, CopyMeasurement]
    concurrent: dict[str, CopyMeasurement]
    background_alone: BackgroundMeasurement
    background_concurrent: BackgroundMeasurement


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive number, got {value}")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c2c-buffer-mib", type=_positive_int, default=512)
    parser.add_argument("--background-buffer-mib", type=_positive_int, default=256)
    parser.add_argument("--warmup-iterations", type=_positive_int, default=5)
    parser.add_argument("--copy-iterations", type=_positive_int, default=20)
    parser.add_argument("--background-warmup-seconds", type=_positive_float, default=3.0)
    parser.add_argument("--background-ready-timeout-seconds", type=_positive_float, default=120.0)
    parser.add_argument("--background-alone-iterations", type=_positive_int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _bandwidth_gb_s(*, transferred_bytes: int, elapsed_seconds: float) -> float:
    if transferred_bytes <= 0:
        raise ValueError("transferred_bytes must be positive")
    if elapsed_seconds <= 0:
        raise ValueError("elapsed_seconds must be positive")
    return transferred_bytes / elapsed_seconds / BYTES_PER_GB


def _drop_percent(*, baseline_gb_s: float, concurrent_gb_s: float) -> float:
    if baseline_gb_s <= 0:
        raise ValueError("baseline_gb_s must be positive")
    return (baseline_gb_s - concurrent_gb_s) / baseline_gb_s * 100.0


@contextmanager
def _nvtx_range(name: str) -> Iterator[None]:
    """Annotate a host scope for Nsight Systems through PyTorch NVTX APIs."""
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def _validate_environment() -> None:
    """Require Gloo to use its ibverbs transport for CPU payloads."""
    if os.environ.get("GLOO_DEVICE_TRANSPORT") != GLOO_IBVERBS_TRANSPORT:
        raise RuntimeError(
            "CPU background benchmark requires "
            f"GLOO_DEVICE_TRANSPORT={GLOO_IBVERBS_TRANSPORT!r}; "
            f"got {os.environ.get('GLOO_DEVICE_TRANSPORT')!r}"
        )
    if os.environ.get("GLOO_SOCKET_IFNAME"):
        raise RuntimeError(
            f"GLOO_SOCKET_IFNAME must be unset for Gloo ibverbs; got {os.environ['GLOO_SOCKET_IFNAME']!r}"
        )


def _resolve_gpu_pci_bus_id(local_rank: int) -> str:
    """Resolve the physical PCI bus ID for the rank's visible GPU."""
    bus_ids = [value for value in os.environ.get("BENCHMARK_GPU_PCI_BUS_IDS", "").split(",") if value]
    if local_rank < len(bus_ids):
        return bus_ids[local_rank]
    return os.environ.get("BENCHMARK_GPU_PCI_BUS_ID", "unknown")


def _measure_copy(
    *,
    direction: str,
    host_buffer: torch.Tensor,
    device_buffer: torch.Tensor,
    stream: torch.cuda.Stream,
    warmup_iterations: int,
    copy_iterations: int,
) -> CopyMeasurement:
    if direction not in {"h2d", "d2h"}:
        raise ValueError(f"unsupported copy direction: {direction}")
    source, destination = (host_buffer, device_buffer) if direction == "h2d" else (device_buffer, host_buffer)

    with _nvtx_range(f"c2c_{direction}_warmup"):
        with torch.cuda.stream(stream):
            for _ in range(warmup_iterations):
                destination.copy_(source, non_blocking=True)
    stream.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    with _nvtx_range(f"c2c_{direction}_timed"):
        with torch.cuda.stream(stream):
            start_event.record(stream)
            for _ in range(copy_iterations):
                destination.copy_(source, non_blocking=True)
            end_event.record(stream)
    end_event.synchronize()
    elapsed_seconds = start_event.elapsed_time(end_event) / 1000.0
    transferred_bytes = host_buffer.numel() * host_buffer.element_size() * copy_iterations
    return CopyMeasurement(
        direction=direction,
        bandwidth_gb_s=_bandwidth_gb_s(transferred_bytes=transferred_bytes, elapsed_seconds=elapsed_seconds),
        elapsed_seconds=elapsed_seconds,
        transferred_bytes=transferred_bytes,
    )


def _measure_c2c_pair(
    *,
    host_buffer: torch.Tensor,
    device_buffer: torch.Tensor,
    stream: torch.cuda.Stream,
    warmup_iterations: int,
    copy_iterations: int,
) -> dict[str, CopyMeasurement]:
    return {
        direction: _measure_copy(
            direction=direction,
            host_buffer=host_buffer,
            device_buffer=device_buffer,
            stream=stream,
            warmup_iterations=warmup_iterations,
            copy_iterations=copy_iterations,
        )
        for direction in ("h2d", "d2h")
    }


def _p2p_role(rank: int) -> tuple[str, int]:
    """Return the role and peer for the fixed ``0 -> 1`` and ``2 -> 3`` pairs."""
    if rank % 2 == 0:
        return "send", rank + 1
    return "recv", rank - 1


def _measurement_from_windows(
    *,
    direction: str,
    peer_rank: int | None,
    windows: list[tuple[float, float]],
    buffer_bytes: int,
) -> BackgroundMeasurement:
    if not windows:
        raise RuntimeError(f"no completed CPU background {direction} operations were recorded")
    completion_seconds = sum(finished - started for started, finished in windows)
    if completion_seconds <= 0:
        raise RuntimeError("CPU background completion time must be positive")
    elapsed_seconds = windows[-1][1] - windows[0][0]
    if elapsed_seconds <= 0:
        raise RuntimeError("CPU background elapsed time must be positive")
    iterations = len(windows)
    transferred_bytes = iterations * buffer_bytes
    return BackgroundMeasurement(
        direction=direction,
        peer_rank=peer_rank,
        iterations=iterations,
        elapsed_seconds=elapsed_seconds,
        average_completion_ms=completion_seconds / iterations * 1000.0,
        bandwidth_gb_s=_bandwidth_gb_s(transferred_bytes=transferred_bytes, elapsed_seconds=elapsed_seconds),
        transferred_bytes=transferred_bytes,
        started_monotonic_seconds=windows[0][0],
        finished_monotonic_seconds=windows[-1][1],
    )


class _CpuBackgroundLoad:
    """Run CPU Gloo/ibverbs P2P traffic on a background thread."""

    def __init__(
        self,
        *,
        rank: int,
        buffer_bytes: int,
        data_group: dist.ProcessGroup,
        control_group: dist.ProcessGroup,
        max_iterations: int | None = None,
    ) -> None:
        self._mode = CPU_P2P
        self._rank = rank
        self._buffer_bytes = buffer_bytes
        self._data_group = data_group
        self._control_group = control_group
        self._max_iterations = max_iterations
        self._role: str
        self._peer_rank: int | None
        self._role, self._peer_rank = _p2p_role(rank)
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="cpu-p2p-load", daemon=True)
        self._error: BaseException | None = None
        self._measurement: BackgroundMeasurement | None = None
        self._iteration_windows: list[tuple[float, float]] = []

    @property
    def role(self) -> str:
        """Return this rank's CPU background role."""
        return self._role

    @property
    def peer_rank(self) -> int | None:
        """Return this rank's P2P peer, if the mode is P2P."""
        return self._peer_rank

    def start(self) -> None:
        self._thread.start()

    def wait_until_ready(self, *, timeout_seconds: float) -> None:
        if not self._ready.wait(timeout=timeout_seconds):
            raise TimeoutError(f"CPU {self._mode} load did not complete its first operation within {timeout_seconds}s")
        if self._error is not None:
            raise RuntimeError(f"CPU {self._mode} load failed during startup") from self._error

    def stop_and_join(self) -> BackgroundMeasurement:
        self._stop.set()
        self._thread.join()
        if self._error is not None:
            raise RuntimeError(f"CPU {self._mode} load failed") from self._error
        if self._measurement is None:
            raise RuntimeError(f"CPU {self._mode} load stopped without producing a measurement")
        return self._measurement

    def join(self) -> BackgroundMeasurement:
        """Wait for a fixed-iteration load to finish."""
        self._thread.join()
        if self._error is not None:
            raise RuntimeError(f"CPU {self._mode} load failed") from self._error
        if self._measurement is None:
            raise RuntimeError(f"CPU {self._mode} load stopped without producing a measurement")
        return self._measurement

    def measurement_from_iteration(self, *, start_index: int) -> BackgroundMeasurement:
        """Return the fixed-iteration measurement after warmup operations."""
        return _measurement_from_windows(
            direction=self._role,
            peer_rank=self._peer_rank,
            windows=self._iteration_windows[start_index:],
            buffer_bytes=self._buffer_bytes,
        )

    def measurement_for_window(self, *, started_at: float, finished_at: float) -> BackgroundMeasurement:
        """Return CPU operations whose execution overlaps a C2C window."""
        windows = [
            window for window in self._iteration_windows if window[1] >= started_at and window[0] <= finished_at
        ]
        return _measurement_from_windows(
            direction=self._role,
            peer_rank=self._peer_rank,
            windows=windows,
            buffer_bytes=self._buffer_bytes,
        )

    def _make_buffer(self) -> torch.Tensor:
        return torch.zeros(self._buffer_bytes, dtype=torch.uint8, device="cpu")

    def _run_operation(self, buffer: torch.Tensor) -> None:
        if self._role == "send":
            work = dist.isend(buffer, dst=cast(int, self._peer_rank), group=self._data_group, tag=0)
        else:
            work = dist.irecv(buffer, src=cast(int, self._peer_rank), group=self._data_group, tag=0)
        work.wait()

    def _run(self) -> None:
        try:
            background_buffer = self._make_buffer()
            iterations = 0
            started_at = time.monotonic()
            with _nvtx_range(f"cpu_{self._mode}_background"):
                while True:
                    for _ in range(BACKGROUND_STOP_CHECK_INTERVAL):
                        if self._max_iterations is not None and iterations >= self._max_iterations:
                            break
                        iteration_started_at = time.monotonic()
                        with _nvtx_range(f"cpu_{self._mode}_{self._role}"):
                            self._run_operation(background_buffer)
                        self._iteration_windows.append((iteration_started_at, time.monotonic()))
                        iterations += 1
                        if iterations == 1:
                            self._ready.set()

                    if self._max_iterations is not None and iterations >= self._max_iterations:
                        break
                    dist.barrier(group=self._control_group)
                    if self._stop.is_set():
                        break

            self._measurement = _measurement_from_windows(
                direction=self._role,
                peer_rank=self._peer_rank,
                windows=self._iteration_windows,
                buffer_bytes=self._buffer_bytes,
            )
            LOGGER.debug(
                "rank=%d CPU %s background ran for %.3f seconds",
                self._rank,
                self._mode,
                time.monotonic() - started_at,
            )
        except BaseException as error:
            self._error = error
            self._ready.set()


def _initialize_distributed() -> tuple[
    int,
    int,
    torch.device,
    dist.ProcessGroup,
    dist.ProcessGroup,
    dist.ProcessGroup,
]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the H2D/D2H measurement")
    if not dist.is_gloo_available():
        raise RuntimeError("PyTorch was built without Gloo support")

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    try:
        dist.init_process_group(backend="gloo")
    except RuntimeError as error:
        raise RuntimeError(
            "Gloo ibverbs initialization failed. Build PyTorch/Gloo with ibverbs "
            "support (USE_GLOO_IBVERBS=1), expose an RDMA device, and set "
            "TORCH_GLOO_IBV_NAME when automatic device selection is unsuitable."
        ) from error
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != EXPECTED_WORLD_SIZE:
        raise RuntimeError(f"expected exactly {EXPECTED_WORLD_SIZE} ranks on one node, got {world_size}")
    data_group = cast(dist.ProcessGroup, dist.group.WORLD)
    phase_group = cast(dist.ProcessGroup, dist.new_group(backend="gloo"))
    background_control_group = cast(dist.ProcessGroup, dist.new_group(backend="gloo"))
    return rank, world_size, device, data_group, phase_group, background_control_group


def _summarize_c2c(records: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for direction in ("h2d", "d2h"):
        baseline = [
            cast(dict[str, float], cast(dict[str, object], record["baseline"])[direction])["bandwidth_gb_s"]
            for record in records
        ]
        concurrent = [
            cast(dict[str, float], cast(dict[str, object], record["concurrent"])[direction])["bandwidth_gb_s"]
            for record in records
        ]
        baseline_mean = fmean(baseline)
        concurrent_mean = fmean(concurrent)
        result[direction] = {
            "baseline_mean_gb_s": baseline_mean,
            "concurrent_mean_gb_s": concurrent_mean,
            "drop_percent": _drop_percent(
                baseline_gb_s=baseline_mean,
                concurrent_gb_s=concurrent_mean,
            ),
        }
    return result


def _summarize_background(records: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for direction in ("send", "recv"):
        alone = [
            cast(dict[str, object], record["background_alone"])
            for record in records
            if cast(dict[str, object], record["background_alone"])["direction"] == direction
        ]
        concurrent = [
            cast(dict[str, object], record["background_concurrent"])
            for record in records
            if cast(dict[str, object], record["background_concurrent"])["direction"] == direction
        ]
        alone_completion = fmean(cast(float, value["average_completion_ms"]) for value in alone)
        concurrent_completion = fmean(cast(float, value["average_completion_ms"]) for value in concurrent)
        alone_bandwidth = fmean(cast(float, value["bandwidth_gb_s"]) for value in alone)
        concurrent_bandwidth = fmean(cast(float, value["bandwidth_gb_s"]) for value in concurrent)
        result[direction] = {
            "alone_mean_completion_ms": alone_completion,
            "concurrent_mean_completion_ms": concurrent_completion,
            "completion_slowdown_percent": (concurrent_completion - alone_completion) / alone_completion * 100.0,
            "alone_mean_bandwidth_gb_s": alone_bandwidth,
            "concurrent_mean_bandwidth_gb_s": concurrent_bandwidth,
            "bandwidth_drop_percent": _drop_percent(
                baseline_gb_s=alone_bandwidth,
                concurrent_gb_s=concurrent_bandwidth,
            ),
            "alone_mean_iterations": fmean(cast(int, value["iterations"]) for value in alone),
            "concurrent_mean_iterations": fmean(cast(int, value["iterations"]) for value in concurrent),
        }
    return result


def _run(args: argparse.Namespace) -> None:
    _validate_environment()
    rank, world_size, device, data_group, phase_group, background_control_group = _initialize_distributed()
    local_rank = int(os.environ["LOCAL_RANK"])

    try:
        c2c_buffer_bytes = args.c2c_buffer_mib * BYTES_PER_MIB
        background_buffer_bytes = args.background_buffer_mib * BYTES_PER_MIB
        host_buffer = torch.empty(c2c_buffer_bytes, dtype=torch.uint8, pin_memory=True)
        device_buffer = torch.empty(c2c_buffer_bytes, dtype=torch.uint8, device=device)
        host_buffer.fill_(1)
        device_buffer.fill_(2)
        copy_stream = torch.cuda.Stream(device=device)
        copy_stream.wait_stream(torch.cuda.current_stream(device=device))

        with _nvtx_range("phase_baseline"):
            dist.barrier(group=phase_group)
            LOGGER.info("rank=%d phase=baseline start", rank)
            baseline = _measure_c2c_pair(
                host_buffer=host_buffer,
                device_buffer=device_buffer,
                stream=copy_stream,
                warmup_iterations=args.warmup_iterations,
                copy_iterations=args.copy_iterations,
            )
            LOGGER.info("rank=%d phase=baseline complete", rank)
            dist.barrier(group=phase_group)

        with _nvtx_range("phase_cpu_background_alone"):
            dist.barrier(group=phase_group)
            alone_load = _CpuBackgroundLoad(
                rank=rank,
                buffer_bytes=background_buffer_bytes,
                data_group=data_group,
                control_group=background_control_group,
                max_iterations=args.background_alone_iterations + args.warmup_iterations,
            )
            alone_load.start()
            alone_load.wait_until_ready(timeout_seconds=args.background_ready_timeout_seconds)
            alone_load.join()
            background_alone = alone_load.measurement_from_iteration(start_index=args.warmup_iterations)
            dist.barrier(group=phase_group)

        background_load = _CpuBackgroundLoad(
            rank=rank,
            buffer_bytes=background_buffer_bytes,
            data_group=data_group,
            control_group=background_control_group,
        )
        with _nvtx_range("phase_cpu_background_warmup"):
            background_load.start()
            background_load.wait_until_ready(timeout_seconds=args.background_ready_timeout_seconds)
            dist.barrier(group=phase_group)
            time.sleep(args.background_warmup_seconds)
            dist.barrier(group=phase_group)

        concurrent_started_at = time.monotonic()
        with _nvtx_range("phase_concurrent"):
            LOGGER.info("rank=%d phase=concurrent start", rank)
            concurrent = _measure_c2c_pair(
                host_buffer=host_buffer,
                device_buffer=device_buffer,
                stream=copy_stream,
                warmup_iterations=args.warmup_iterations,
                copy_iterations=args.copy_iterations,
            )
        concurrent_finished_at = time.monotonic()
        LOGGER.info("rank=%d phase=concurrent complete", rank)

        with _nvtx_range("phase_cpu_background_shutdown"):
            dist.barrier(group=phase_group)
            background_load.stop_and_join()
            background_concurrent = background_load.measurement_for_window(
                started_at=concurrent_started_at,
                finished_at=concurrent_finished_at,
            )
            dist.barrier(group=phase_group)

        properties = torch.cuda.get_device_properties(device)
        rank_measurement = RankMeasurement(
            rank=rank,
            hostname=socket.gethostname(),
            gpu_name=properties.name,
            gpu_pci_bus_id=_resolve_gpu_pci_bus_id(local_rank),
            background_mode=CPU_P2P,
            background_role=background_load.role,
            peer_rank=background_load.peer_rank,
            c2c_buffer_mib=args.c2c_buffer_mib,
            background_buffer_mib=args.background_buffer_mib,
            baseline=baseline,
            concurrent=concurrent,
            background_alone=background_alone,
            background_concurrent=background_concurrent,
        )
        local_record = cast(dict[str, object], asdict(rank_measurement))
        gathered_records: list[object] = [None] * world_size
        dist.all_gather_object(gathered_records, local_record, group=phase_group)

        if rank == 0:
            records = [cast(dict[str, object], record) for record in gathered_records]
            hostnames = {cast(str, record["hostname"]) for record in records}
            if len(hostnames) != 1:
                raise RuntimeError(f"expected all four ranks on one node, got hosts: {sorted(hostnames)}")
            output = {
                "background_device": "cpu",
                "background_backend": "gloo-ibverbs",
                "background_mode": CPU_P2P,
                "summary": _summarize_c2c(records),
                "background_summary": _summarize_background(records),
                "ranks": records,
                "transport": {
                    "background": "cpu/gloo-ibverbs",
                    "GLOO_DEVICE_TRANSPORT": os.environ["GLOO_DEVICE_TRANSPORT"],
                    "TORCH_GLOO_IBV_NAME": os.environ.get("TORCH_GLOO_IBV_NAME", "auto"),
                },
                "metric_note": (
                    "background traffic uses CPU tensors and Gloo ibverbs; C2C bandwidth uses CUDA events; "
                    "background bandwidth is completed tensor payload, not wire-level traffic"
                ),
                "torch_version": str(torch.__version__),
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
            LOGGER.info("wrote CPU background benchmark result to %s", args.output)
            for direction, values in cast(dict[str, dict[str, float]], output["summary"]).items():
                LOGGER.info(
                    "%s baseline=%.2f GB/s concurrent=%.2f GB/s drop=%.2f%%",
                    direction.upper(),
                    values["baseline_mean_gb_s"],
                    values["concurrent_mean_gb_s"],
                    values["drop_percent"],
                )
    finally:
        dist.destroy_process_group(background_control_group)
        dist.destroy_process_group(phase_group)
        dist.destroy_process_group()


def main() -> None:
    """Parse arguments and run the CPU background bandwidth benchmark."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    _run(_build_parser().parse_args())


if __name__ == "__main__":
    main()
