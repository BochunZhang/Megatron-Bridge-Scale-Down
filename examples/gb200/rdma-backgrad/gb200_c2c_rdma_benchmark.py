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
"""Measure GB200 C2C bandwidth with and without sustained RDMA traffic.

Run this program through ``run_gb200_c2c_rdma_benchmark.sh`` on one node with
four GPUs. The NCCL process group generates GPU-buffer IB traffic while a
separate Gloo process group coordinates the measurement lifecycle. P2P/NVLink
and shared-memory NCCL paths are disabled by the wrapper so the collective
must use the configured RDMA network.
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
BITS_PER_BYTE = 8
EXPECTED_WORLD_SIZE = 4
RDMA_STOP_CHECK_INTERVAL = 32


@dataclass(frozen=True)
class CopyMeasurement:
    """Result of one host/device copy direction.

    Args:
        direction: Copy direction, either ``h2d`` or ``d2h``.
        bandwidth_gb_s: Decimal gigabytes transferred per second.
        elapsed_seconds: CUDA event elapsed time for all timed copies.
        transferred_bytes: Total bytes transferred during timed copies.
    """

    direction: str
    bandwidth_gb_s: float
    elapsed_seconds: float
    transferred_bytes: int


@dataclass(frozen=True)
class RdmaMeasurement:
    """Sustained NCCL load statistics from one rank.

    Args:
        iterations: Completed all-reduce operations.
        elapsed_seconds: Wall-clock span used for this measurement; window-derived metrics
            cover the selected all-reduce operations, while the full background metric
            covers the load lifecycle.
        average_all_reduce_ms: Mean host-observed completion time of one all-reduce.
        payload_gbit_s: Tensor payload rate, not an estimate of wire-level bytes.
        started_monotonic_seconds: Local monotonic timestamp before the first all-reduce.
        finished_monotonic_seconds: Local monotonic timestamp after the last all-reduce.
    """

    iterations: int
    elapsed_seconds: float
    average_all_reduce_ms: float
    payload_gbit_s: float
    started_monotonic_seconds: float
    finished_monotonic_seconds: float


@dataclass(frozen=True)
class RankMeasurement:
    """All measurements and topology metadata for one distributed rank."""

    rank: int
    hostname: str
    gpu_name: str
    gpu_pci_bus_id: str
    hca: str
    c2c_buffer_mib: int
    rdma_buffer_mib: int
    baseline: dict[str, CopyMeasurement]
    concurrent: dict[str, CopyMeasurement]
    rdma: RdmaMeasurement
    rdma_alone: RdmaMeasurement
    rdma_concurrent: RdmaMeasurement
    concurrent_c2c_started_monotonic_seconds: float
    concurrent_c2c_finished_monotonic_seconds: float
    rdma_covers_entire_concurrent_c2c: bool


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
    parser.add_argument("--rdma-buffer-mib", type=_positive_int, default=256)
    parser.add_argument("--warmup-iterations", type=_positive_int, default=5)
    parser.add_argument("--copy-iterations", type=_positive_int, default=20)
    parser.add_argument("--rdma-warmup-seconds", type=_positive_float, default=3.0)
    parser.add_argument("--rdma-ready-timeout-seconds", type=_positive_float, default=120.0)
    parser.add_argument("--rdma-alone-iterations", type=_positive_int, default=20)
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


def _validate_transport_environment() -> str:
    required_values = {
        "NCCL_IB_DISABLE": "0",
        "NCCL_MNNVL_ENABLE": "0",
        "NCCL_NET": "IB",
        "NCCL_NET_GDR_LEVEL": "PHB",
        "NCCL_NET_GDR_C2C": "1",
        "NCCL_P2P_DISABLE": "1",
        "NCCL_NVB_DISABLE": "1",
        "NCCL_PXN_DISABLE": "1",
        "NCCL_SHM_DISABLE": "1",
        "NCCL_NVLS_ENABLE": "0",
        "GLOO_SOCKET_IFNAME": "lo",
        "NCCL_SOCKET_IFNAME": "lo",
        "NCCL_SOCKET_FAMILY": "AF_INET",
    }
    mismatches = {
        name: os.environ.get(name) for name, expected in required_values.items() if os.environ.get(name) != expected
    }
    if mismatches:
        details = ", ".join(f"{name}={value!r}" for name, value in sorted(mismatches.items()))
        raise RuntimeError(f"RDMA transport environment is not enforced: {details}")

    return os.environ.get("NCCL_IB_HCA", "")


def _resolve_gpu_pci_bus_id(local_rank: int) -> str:
    """Resolve the physical PCI bus ID for the rank's visible GPU.

    Args:
        local_rank: Rank index assigned by torchrun on this node.

    Returns:
        The PCI bus ID recorded by the launcher, or ``"unknown"`` when the
        launcher did not provide one.
    """
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
        bandwidth_gb_s=_bandwidth_gb_s(
            transferred_bytes=transferred_bytes,
            elapsed_seconds=elapsed_seconds,
        ),
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


class _RdmaLoad:
    def __init__(
        self,
        *,
        device: torch.device,
        buffer_bytes: int,
        control_group: dist.ProcessGroup,
        max_iterations: int | None = None,
    ) -> None:
        self._device = device
        self._buffer_bytes = buffer_bytes
        self._control_group = control_group
        self._max_iterations = max_iterations
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="nccl-rdma-load", daemon=True)
        self._error: BaseException | None = None
        self._measurement: RdmaMeasurement | None = None
        self._iteration_windows: list[tuple[float, float]] = []

    def start(self) -> None:
        self._thread.start()

    def wait_until_ready(self, *, timeout_seconds: float) -> None:
        if not self._ready.wait(timeout=timeout_seconds):
            raise TimeoutError(f"RDMA load did not complete its first collective within {timeout_seconds}s")
        if self._error is not None:
            raise RuntimeError("RDMA load failed during startup") from self._error

    def stop_and_join(self) -> RdmaMeasurement:
        self._stop.set()
        self._thread.join()
        if self._error is not None:
            raise RuntimeError("RDMA load failed") from self._error
        if self._measurement is None:
            raise RuntimeError("RDMA load stopped without producing a measurement")
        return self._measurement

    def join(self) -> RdmaMeasurement:
        """Wait for a fixed-iteration load to finish without requesting stop."""
        self._thread.join()
        if self._error is not None:
            raise RuntimeError("RDMA load failed") from self._error
        if self._measurement is None:
            raise RuntimeError("RDMA load stopped without producing a measurement")
        return self._measurement

    def measurement_for_window(self, *, started_at: float, finished_at: float) -> RdmaMeasurement:
        """Return all-reduce timings whose execution overlaps a C2C window."""
        windows = [
            (iteration_started_at, iteration_finished_at)
            for iteration_started_at, iteration_finished_at in self._iteration_windows
            if iteration_finished_at >= started_at and iteration_started_at <= finished_at
        ]
        return _measurement_from_windows(
            windows=windows,
            buffer_bytes=self._buffer_bytes,
        )

    def measurement_from_iteration(self, *, start_index: int) -> RdmaMeasurement:
        """Return timings after skipping a specified number of warmup operations."""
        return _measurement_from_windows(
            windows=self._iteration_windows[start_index:],
            buffer_bytes=self._buffer_bytes,
        )

    def _run(self) -> None:
        try:
            torch.cuda.set_device(self._device)
            rdma_stream = torch.cuda.Stream(device=self._device)
            rdma_buffer = torch.zeros(self._buffer_bytes, dtype=torch.uint8, device=self._device)
            stop_tensor = torch.zeros(1, dtype=torch.int32)
            iterations = 0
            started_at = time.monotonic()

            with _nvtx_range("rdma_background"):
                while True:
                    for _ in range(RDMA_STOP_CHECK_INTERVAL):
                        if self._max_iterations is not None and iterations >= self._max_iterations:
                            break
                        iteration_started_at = time.monotonic()
                        with _nvtx_range("rdma_all_reduce"):
                            with torch.cuda.stream(rdma_stream):
                                work = dist.all_reduce(rdma_buffer, op=dist.ReduceOp.SUM, async_op=True)
                            work.wait()
                            rdma_stream.synchronize()
                        self._iteration_windows.append((iteration_started_at, time.monotonic()))
                        iterations += 1
                        if iterations == 1:
                            self._ready.set()

                    if self._max_iterations is not None and iterations >= self._max_iterations:
                        break
                    stop_tensor.fill_(int(self._stop.is_set()))
                    dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX, group=self._control_group)
                    if stop_tensor.item() != 0:
                        break

            rdma_stream.synchronize()
            finished_at = time.monotonic()
            self._measurement = _measurement_from_windows(
                windows=self._iteration_windows,
                buffer_bytes=self._buffer_bytes,
                started_at=started_at,
                finished_at=finished_at,
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
]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not dist.is_nccl_available():
        raise RuntimeError("PyTorch was built without NCCL support")
    if not dist.is_gloo_available():
        raise RuntimeError("PyTorch was built without Gloo support")

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != EXPECTED_WORLD_SIZE:
        raise RuntimeError(f"expected exactly {EXPECTED_WORLD_SIZE} ranks on one node, got {world_size}")

    control_group = cast(dist.ProcessGroup, dist.new_group(backend="gloo"))
    rdma_control_group = cast(dist.ProcessGroup, dist.new_group(backend="gloo"))
    return rank, world_size, device, control_group, rdma_control_group


def _measurement_from_windows(
    *,
    windows: list[tuple[float, float]],
    buffer_bytes: int,
    started_at: float | None = None,
    finished_at: float | None = None,
) -> RdmaMeasurement:
    if not windows:
        raise RuntimeError("no completed all-reduce operations were recorded")
    completion_seconds = sum(finished - started for started, finished in windows)
    if completion_seconds <= 0:
        raise RuntimeError("all-reduce completion time must be positive")
    first_started = windows[0][0] if started_at is None else started_at
    last_finished = windows[-1][1] if finished_at is None else finished_at
    elapsed_seconds = last_finished - first_started
    if elapsed_seconds <= 0:
        raise RuntimeError("all-reduce elapsed time must be positive")
    iterations = len(windows)
    payload_bits = iterations * buffer_bytes * BITS_PER_BYTE
    return RdmaMeasurement(
        iterations=iterations,
        elapsed_seconds=elapsed_seconds,
        average_all_reduce_ms=completion_seconds / iterations * 1000.0,
        payload_gbit_s=payload_bits / elapsed_seconds / BYTES_PER_GB,
        started_monotonic_seconds=first_started,
        finished_monotonic_seconds=last_finished,
    )


def _summarize(rank_measurements: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for direction in ("h2d", "d2h"):
        baseline_values = [
            cast(dict[str, dict[str, float]], measurement["baseline"])[direction]["bandwidth_gb_s"]
            for measurement in rank_measurements
        ]
        concurrent_values = [
            cast(dict[str, dict[str, float]], measurement["concurrent"])[direction]["bandwidth_gb_s"]
            for measurement in rank_measurements
        ]
        baseline_mean = fmean(baseline_values)
        concurrent_mean = fmean(concurrent_values)
        summary[direction] = {
            "baseline_mean_gb_s": baseline_mean,
            "concurrent_mean_gb_s": concurrent_mean,
            "drop_percent": _drop_percent(
                baseline_gb_s=baseline_mean,
                concurrent_gb_s=concurrent_mean,
            ),
        }
    return summary


def _summarize_rdma_completion(rank_measurements: list[dict[str, object]]) -> dict[str, float]:
    alone_values = [
        cast(dict[str, float], measurement["rdma_alone"])["average_all_reduce_ms"] for measurement in rank_measurements
    ]
    concurrent_values = [
        cast(dict[str, float], measurement["rdma_concurrent"])["average_all_reduce_ms"]
        for measurement in rank_measurements
    ]
    alone_iterations = [
        cast(dict[str, float], measurement["rdma_alone"])["iterations"] for measurement in rank_measurements
    ]
    concurrent_iterations = [
        cast(dict[str, float], measurement["rdma_concurrent"])["iterations"] for measurement in rank_measurements
    ]
    alone_mean = fmean(alone_values)
    concurrent_mean = fmean(concurrent_values)
    if alone_mean <= 0:
        raise ValueError("alone all-reduce completion time must be positive")
    return {
        "alone_mean_completion_ms": alone_mean,
        "concurrent_mean_completion_ms": concurrent_mean,
        "slowdown_percent": (concurrent_mean - alone_mean) / alone_mean * 100.0,
        "alone_mean_iterations": fmean(alone_iterations),
        "concurrent_mean_iterations": fmean(concurrent_iterations),
    }


def _run(args: argparse.Namespace) -> None:
    hca = _validate_transport_environment()
    rank, world_size, device, control_group, rdma_control_group = _initialize_distributed()
    local_rank = int(os.environ["LOCAL_RANK"])

    try:
        c2c_buffer_bytes = args.c2c_buffer_mib * BYTES_PER_MIB
        rdma_buffer_bytes = args.rdma_buffer_mib * BYTES_PER_MIB
        host_buffer = torch.empty(c2c_buffer_bytes, dtype=torch.uint8, pin_memory=True)
        device_buffer = torch.empty(c2c_buffer_bytes, dtype=torch.uint8, device=device)
        host_buffer.fill_(1)
        device_buffer.fill_(2)
        copy_stream = torch.cuda.Stream(device=device)

        with _nvtx_range("phase_baseline"):
            dist.barrier(group=control_group)
            LOGGER.info("rank=%d phase=baseline start", rank)
            baseline = _measure_c2c_pair(
                host_buffer=host_buffer,
                device_buffer=device_buffer,
                stream=copy_stream,
                warmup_iterations=args.warmup_iterations,
                copy_iterations=args.copy_iterations,
            )
            LOGGER.info("rank=%d phase=baseline complete", rank)
            dist.barrier(group=control_group)

        with _nvtx_range("phase_rdma_alone"):
            dist.barrier(group=control_group)
            LOGGER.info("rank=%d phase=rdma_alone start", rank)
            alone_load = _RdmaLoad(
                device=device,
                buffer_bytes=rdma_buffer_bytes,
                control_group=rdma_control_group,
                max_iterations=args.rdma_alone_iterations + 1,
            )
            alone_load.start()
            alone_load.wait_until_ready(timeout_seconds=args.rdma_ready_timeout_seconds)
            alone_load.join()
            rdma_alone = alone_load.measurement_from_iteration(start_index=1)
            LOGGER.info("rank=%d phase=rdma_alone complete", rank)
            dist.barrier(group=control_group)

        rdma_load = _RdmaLoad(
            device=device,
            buffer_bytes=rdma_buffer_bytes,
            control_group=rdma_control_group,
        )
        with _nvtx_range("phase_rdma_warmup"):
            rdma_load.start()
            rdma_load.wait_until_ready(timeout_seconds=args.rdma_ready_timeout_seconds)
            dist.barrier(group=control_group)
            time.sleep(args.rdma_warmup_seconds)
            dist.barrier(group=control_group)

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

        # This Gloo barrier is independent of the NCCL load. All ranks keep
        # issuing NCCL operations until every rank has completed both C2C directions.
        with _nvtx_range("phase_rdma_shutdown"):
            dist.barrier(group=control_group)
            rdma = rdma_load.stop_and_join()
            rdma_concurrent = rdma_load.measurement_for_window(
                started_at=concurrent_started_at,
                finished_at=concurrent_finished_at,
            )
            dist.barrier(group=control_group)

        properties = torch.cuda.get_device_properties(device)
        rank_measurement = RankMeasurement(
            rank=rank,
            hostname=socket.gethostname(),
            gpu_name=properties.name,
            gpu_pci_bus_id=_resolve_gpu_pci_bus_id(local_rank),
            hca=hca,
            c2c_buffer_mib=args.c2c_buffer_mib,
            rdma_buffer_mib=args.rdma_buffer_mib,
            baseline=baseline,
            concurrent=concurrent,
            rdma=rdma,
            rdma_alone=rdma_alone,
            rdma_concurrent=rdma_concurrent,
            concurrent_c2c_started_monotonic_seconds=concurrent_started_at,
            concurrent_c2c_finished_monotonic_seconds=concurrent_finished_at,
            rdma_covers_entire_concurrent_c2c=(
                rdma.started_monotonic_seconds <= concurrent_started_at
                and rdma.finished_monotonic_seconds >= concurrent_finished_at
            ),
        )
        local_record = cast(dict[str, object], asdict(rank_measurement))
        gathered_records: list[object] = [None] * world_size
        dist.all_gather_object(gathered_records, local_record, group=control_group)

        if rank == 0:
            records = [cast(dict[str, object], record) for record in gathered_records]
            hostnames = {cast(str, record["hostname"]) for record in records}
            if len(hostnames) != 1:
                raise RuntimeError(f"expected all four ranks on one node, got hosts: {sorted(hostnames)}")
            output = {
                "summary": _summarize(records),
                "rdma_completion": _summarize_rdma_completion(records),
                "ranks": records,
                "rdma_covers_entire_concurrent_c2c": all(
                    cast(bool, record["rdma_covers_entire_concurrent_c2c"]) for record in records
                ),
                "rdma_metric_note": "payload_gbit_s is completed tensor payload, not wire-level traffic",
                "transport": {
                    "NCCL_IB_HCA": hca,
                    "NCCL_IB_DISABLE": os.environ["NCCL_IB_DISABLE"],
                    "NCCL_MNNVL_ENABLE": os.environ["NCCL_MNNVL_ENABLE"],
                    "NCCL_NET": os.environ["NCCL_NET"],
                    "NCCL_NET_GDR_LEVEL": os.environ["NCCL_NET_GDR_LEVEL"],
                    "NCCL_NET_GDR_C2C": os.environ["NCCL_NET_GDR_C2C"],
                    "NCCL_P2P_DISABLE": os.environ["NCCL_P2P_DISABLE"],
                    "NCCL_NVB_DISABLE": os.environ["NCCL_NVB_DISABLE"],
                    "NCCL_PXN_DISABLE": os.environ["NCCL_PXN_DISABLE"],
                    "NCCL_SHM_DISABLE": os.environ["NCCL_SHM_DISABLE"],
                    "NCCL_NVLS_ENABLE": os.environ["NCCL_NVLS_ENABLE"],
                    "GLOO_SOCKET_IFNAME": os.environ["GLOO_SOCKET_IFNAME"],
                    "NCCL_SOCKET_IFNAME": os.environ["NCCL_SOCKET_IFNAME"],
                    "NCCL_SOCKET_FAMILY": os.environ["NCCL_SOCKET_FAMILY"],
                },
                "torch_version": str(torch.__version__),
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
            LOGGER.info("wrote benchmark result to %s", args.output)
            for direction, values in cast(dict[str, dict[str, float]], output["summary"]).items():
                LOGGER.info(
                    "%s baseline=%.2f GB/s concurrent=%.2f GB/s drop=%.2f%%",
                    direction.upper(),
                    values["baseline_mean_gb_s"],
                    values["concurrent_mean_gb_s"],
                    values["drop_percent"],
                )
            rdma_completion = cast(dict[str, float], output["rdma_completion"])
            LOGGER.info(
                "RDMA all_reduce alone=%.3f ms concurrent=%.3f ms slowdown=%.2f%%",
                rdma_completion["alone_mean_completion_ms"],
                rdma_completion["concurrent_mean_completion_ms"],
                rdma_completion["slowdown_percent"],
            )
    finally:
        dist.destroy_process_group(rdma_control_group)
        dist.destroy_process_group(control_group)
        dist.destroy_process_group()


def main() -> None:
    """Parse arguments and run the distributed bandwidth benchmark."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    _run(_build_parser().parse_args())


if __name__ == "__main__":
    main()
