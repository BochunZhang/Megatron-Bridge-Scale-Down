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
"""Measure NCCL P2P send/recv interference with C2C H2D/D2H copies.

Run this program through ``run_gb200_c2c_p2p_benchmark.sh`` on one node with
four GPUs. Ranks are paired as ``0 -> 1`` and ``2 -> 3``. The sender ranks
run D2H copies in the send contention phase; the receiver ranks run H2D copies
in the receive contention phase. NCCL transport settings are supplied by the
wrapper and force the P2P data path through the configured IB/GDRDMA network.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
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


@dataclass(frozen=True)
class CopyMeasurement:
    """Result of one host/device copy direction."""

    direction: str
    bandwidth_gb_s: float
    elapsed_seconds: float
    transferred_bytes: int


@dataclass(frozen=True)
class P2PMeasurement:
    """Result of one rank's NCCL P2P send or receive operations.

    ``average_completion_ms`` and ``elapsed_seconds`` use CUDA events on the
    P2P stream so they can be compared with Nsight CUDA activity. The host
    fields retain the end-to-end Python/request/synchronization timing.
    """

    direction: str
    peer_rank: int
    iterations: int
    elapsed_seconds: float
    average_completion_ms: float
    host_elapsed_seconds: float
    host_average_completion_ms: float
    bandwidth_gb_s: float
    transferred_bytes: int


@dataclass(frozen=True)
class RankMeasurement:
    """All measurements and topology metadata for one distributed rank."""

    rank: int
    hostname: str
    gpu_name: str
    gpu_pci_bus_id: str
    hca: str
    c2c_buffer_mib: int
    p2p_buffer_mib: int
    baseline: dict[str, CopyMeasurement]
    p2p_send_alone: P2PMeasurement | None
    p2p_recv_alone: P2PMeasurement | None
    p2p_send_with_d2h: P2PMeasurement | None
    p2p_recv_with_h2d: P2PMeasurement | None
    send_d2h_copy: CopyMeasurement | None
    recv_h2d_copy: CopyMeasurement | None


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c2c-buffer-mib", type=_positive_int, default=512)
    parser.add_argument("--p2p-buffer-mib", type=_positive_int, default=256)
    parser.add_argument("--warmup-iterations", type=_positive_int, default=5)
    parser.add_argument("--copy-iterations", type=_positive_int, default=20)
    parser.add_argument("--p2p-iterations", type=_positive_int, default=20)
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


def _measurement_from_windows(
    *,
    direction: str,
    peer_rank: int,
    windows: list[tuple[float, float]],
    buffer_bytes: int,
    gpu_windows: list[tuple[torch.cuda.Event, torch.cuda.Event]] | None = None,
) -> P2PMeasurement:
    if not windows:
        raise RuntimeError(f"no completed P2P {direction} operations were recorded")
    completion_seconds = sum(finished - started for started, finished in windows)
    if completion_seconds <= 0:
        raise RuntimeError("P2P completion time must be positive")
    elapsed_seconds = windows[-1][1] - windows[0][0]
    if elapsed_seconds <= 0:
        raise RuntimeError("P2P elapsed time must be positive")
    iterations = len(windows)
    transferred_bytes = iterations * buffer_bytes
    host_average_completion_ms = completion_seconds / iterations * 1000.0
    if gpu_windows is None:
        gpu_average_completion_ms = host_average_completion_ms
        gpu_elapsed_seconds = elapsed_seconds
    else:
        if len(gpu_windows) != iterations:
            raise ValueError("GPU and host P2P timing windows must have equal lengths")
        gpu_completion_ms = [start.elapsed_time(end) for start, end in gpu_windows]
        gpu_average_completion_ms = fmean(gpu_completion_ms)
        gpu_elapsed_seconds = gpu_windows[0][0].elapsed_time(gpu_windows[-1][1]) / 1000.0
        if gpu_elapsed_seconds <= 0:
            raise RuntimeError("GPU P2P elapsed time must be positive")
    return P2PMeasurement(
        direction=direction,
        peer_rank=peer_rank,
        iterations=iterations,
        elapsed_seconds=gpu_elapsed_seconds,
        average_completion_ms=gpu_average_completion_ms,
        host_elapsed_seconds=elapsed_seconds,
        host_average_completion_ms=host_average_completion_ms,
        bandwidth_gb_s=_bandwidth_gb_s(transferred_bytes=transferred_bytes, elapsed_seconds=gpu_elapsed_seconds),
        transferred_bytes=transferred_bytes,
    )


def _p2p_role(rank: int) -> tuple[str, int]:
    """Return the role and paired rank for the fixed 0-1 and 2-3 topology."""
    if rank % 2 == 0:
        return "send", rank + 1
    return "recv", rank - 1


def _launch_p2p(
    *,
    direction: str,
    tensor: torch.Tensor,
    peer_rank: int,
    stream: torch.cuda.Stream,
) -> list[dist.Work]:
    if direction == "send":
        operation = dist.P2POp(dist.isend, tensor, peer_rank)
    elif direction == "recv":
        operation = dist.P2POp(dist.irecv, tensor, peer_rank)
    else:
        raise ValueError(f"unsupported P2P direction: {direction}")
    with torch.cuda.stream(stream):
        requests = dist.batch_isend_irecv([operation])
    return requests


def _run_p2p_iterations(
    *,
    rank: int,
    p2p_buffer: torch.Tensor,
    p2p_stream: torch.cuda.Stream,
    host_buffer: torch.Tensor,
    device_buffer: torch.Tensor,
    copy_stream: torch.cuda.Stream,
    iterations: int,
    p2p_warmup_iterations: int,
    copy_direction: str | None,
    copy_warmup_iterations: int,
    nvtx_name: str,
) -> tuple[
    list[tuple[float, float]],
    list[tuple[torch.cuda.Event, torch.cuda.Event]],
    CopyMeasurement | None,
]:
    direction, peer_rank = _p2p_role(rank)
    source, destination = (host_buffer, device_buffer) if copy_direction == "h2d" else (device_buffer, host_buffer)
    if copy_direction is not None and copy_direction not in {"h2d", "d2h"}:
        raise ValueError(f"unsupported copy direction: {copy_direction}")

    if copy_direction is not None:
        with _nvtx_range(f"{nvtx_name}_{copy_direction}_warmup"):
            with torch.cuda.stream(copy_stream):
                for _ in range(copy_warmup_iterations):
                    destination.copy_(source, non_blocking=True)
        copy_stream.synchronize()

    for _ in range(p2p_warmup_iterations):
        with _nvtx_range(f"{nvtx_name}_{direction}_warmup"):
            with torch.cuda.stream(p2p_stream):
                requests = _launch_p2p(
                    direction=direction,
                    tensor=p2p_buffer,
                    peer_rank=peer_rank,
                    stream=p2p_stream,
                )
                if copy_direction is not None:
                    with _nvtx_range(f"{nvtx_name}_{copy_direction}_warmup"):
                        with torch.cuda.stream(copy_stream):
                            destination.copy_(source, non_blocking=True)
                for request in requests:
                    request.wait()
            p2p_stream.synchronize()
            if copy_direction is not None:
                copy_stream.synchronize()

    copy_start_event = torch.cuda.Event(enable_timing=True) if copy_direction is not None else None
    copy_end_event = torch.cuda.Event(enable_timing=True) if copy_direction is not None else None
    if copy_start_event is not None:
        with torch.cuda.stream(copy_stream):
            copy_start_event.record(copy_stream)

    windows: list[tuple[float, float]] = []
    gpu_windows: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    for _ in range(iterations):
        started_at = time.monotonic()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        with _nvtx_range(f"{nvtx_name}_{direction}"):
            with torch.cuda.stream(p2p_stream):
                start_event.record(p2p_stream)
                requests = _launch_p2p(
                    direction=direction,
                    tensor=p2p_buffer,
                    peer_rank=peer_rank,
                    stream=p2p_stream,
                )
                if copy_direction is not None:
                    with _nvtx_range(f"{nvtx_name}_{copy_direction}"):
                        with torch.cuda.stream(copy_stream):
                            destination.copy_(source, non_blocking=True)
                for request in requests:
                    request.wait()
                end_event.record(p2p_stream)
            p2p_stream.synchronize()
        windows.append((started_at, time.monotonic()))
        gpu_windows.append((start_event, end_event))

    copy_measurement = None
    if copy_end_event is not None and copy_start_event is not None:
        with torch.cuda.stream(copy_stream):
            copy_end_event.record(copy_stream)
        copy_end_event.synchronize()
        elapsed_seconds = copy_start_event.elapsed_time(copy_end_event) / 1000.0
        transferred_bytes = host_buffer.numel() * host_buffer.element_size() * iterations
        copy_measurement = CopyMeasurement(
            direction=cast(str, copy_direction),
            bandwidth_gb_s=_bandwidth_gb_s(
                transferred_bytes=transferred_bytes,
                elapsed_seconds=elapsed_seconds,
            ),
            elapsed_seconds=elapsed_seconds,
            transferred_bytes=transferred_bytes,
        )
    return windows, gpu_windows, copy_measurement


def _initialize_distributed() -> tuple[int, int, torch.device, dist.ProcessGroup]:
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
    return rank, world_size, device, control_group


def _summary_values(records: list[dict[str, object]], key: str) -> list[dict[str, object]]:
    return [cast(dict[str, object], record[key]) for record in records if record[key] is not None]


def _summarize_p2p(rank_measurements: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for direction, alone_key, concurrent_key, copy_key in (
        ("send", "p2p_send_alone", "p2p_send_with_d2h", "send_d2h_copy"),
        ("recv", "p2p_recv_alone", "p2p_recv_with_h2d", "recv_h2d_copy"),
    ):
        alone = _summary_values(rank_measurements, alone_key)
        concurrent = _summary_values(rank_measurements, concurrent_key)
        copy_values = _summary_values(rank_measurements, copy_key)
        alone_completion = fmean(cast(float, value["average_completion_ms"]) for value in alone)
        concurrent_completion = fmean(cast(float, value["average_completion_ms"]) for value in concurrent)
        alone_host_completion = fmean(
            cast(float, value.get("host_average_completion_ms", value["average_completion_ms"])) for value in alone
        )
        concurrent_host_completion = fmean(
            cast(float, value.get("host_average_completion_ms", value["average_completion_ms"]))
            for value in concurrent
        )
        alone_bandwidth = fmean(cast(float, value["bandwidth_gb_s"]) for value in alone)
        concurrent_bandwidth = fmean(cast(float, value["bandwidth_gb_s"]) for value in concurrent)
        copy_direction = "d2h" if direction == "send" else "h2d"
        result[direction] = {
            "alone_mean_completion_ms": alone_completion,
            f"with_{copy_direction}_mean_completion_ms": concurrent_completion,
            "alone_mean_host_completion_ms": alone_host_completion,
            f"with_{copy_direction}_mean_host_completion_ms": concurrent_host_completion,
            "completion_slowdown_percent": (concurrent_completion - alone_completion) / alone_completion * 100.0,
            "alone_mean_bandwidth_gb_s": alone_bandwidth,
            f"with_{copy_direction}_mean_bandwidth_gb_s": concurrent_bandwidth,
            "bandwidth_drop_percent": _drop_percent(
                baseline_gb_s=alone_bandwidth,
                concurrent_gb_s=concurrent_bandwidth,
            ),
            "alone_mean_iterations": fmean(cast(int, value["iterations"]) for value in alone),
            f"with_{copy_direction}_mean_iterations": fmean(cast(int, value["iterations"]) for value in concurrent),
            "copy_mean_bandwidth_gb_s": fmean(cast(float, value["bandwidth_gb_s"]) for value in copy_values),
        }
    return result


def _summarize_copy(rank_measurements: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for direction, concurrent_key in (("d2h", "send_d2h_copy"), ("h2d", "recv_h2d_copy")):
        baseline = [
            cast(dict[str, object], record["baseline"])[direction]["bandwidth_gb_s"]
            for record in rank_measurements
            if record[concurrent_key] is not None
        ]
        concurrent = [
            cast(dict[str, object], value)["bandwidth_gb_s"]
            for value in _summary_values(rank_measurements, concurrent_key)
        ]
        baseline_mean = fmean(cast(float, value) for value in baseline)
        concurrent_mean = fmean(cast(float, value) for value in concurrent)
        result[direction] = {
            "baseline_mean_gb_s": baseline_mean,
            "with_p2p_mean_gb_s": concurrent_mean,
            "drop_percent": _drop_percent(baseline_gb_s=baseline_mean, concurrent_gb_s=concurrent_mean),
        }
    return result


def _run(args: argparse.Namespace) -> None:
    hca = _validate_transport_environment()
    rank, world_size, device, control_group = _initialize_distributed()
    local_rank = int(os.environ["LOCAL_RANK"])
    direction, peer_rank = _p2p_role(rank)

    try:
        c2c_buffer_bytes = args.c2c_buffer_mib * BYTES_PER_MIB
        p2p_buffer_bytes = args.p2p_buffer_mib * BYTES_PER_MIB
        host_buffer = torch.empty(c2c_buffer_bytes, dtype=torch.uint8, pin_memory=True)
        device_buffer = torch.empty(c2c_buffer_bytes, dtype=torch.uint8, device=device)
        host_buffer.fill_(1)
        device_buffer.fill_(2)
        p2p_buffer = torch.ones(p2p_buffer_bytes, dtype=torch.uint8, device=device)
        copy_stream = torch.cuda.Stream(device=device)
        p2p_stream = torch.cuda.Stream(device=device)
        current_stream = torch.cuda.current_stream(device=device)
        copy_stream.wait_stream(current_stream)
        p2p_stream.wait_stream(current_stream)

        with _nvtx_range("phase_baseline"):
            dist.barrier(group=control_group)
            baseline = {
                copy_direction: _measure_copy(
                    direction=copy_direction,
                    host_buffer=host_buffer,
                    device_buffer=device_buffer,
                    stream=copy_stream,
                    warmup_iterations=args.warmup_iterations,
                    copy_iterations=args.copy_iterations,
                )
                for copy_direction in ("h2d", "d2h")
            }
            dist.barrier(group=control_group)

        with _nvtx_range("phase_p2p_alone"):
            dist.barrier(group=control_group)
            alone_windows, alone_gpu_windows, _ = _run_p2p_iterations(
                rank=rank,
                p2p_buffer=p2p_buffer,
                p2p_stream=p2p_stream,
                host_buffer=host_buffer,
                device_buffer=device_buffer,
                copy_stream=copy_stream,
                iterations=args.p2p_iterations,
                p2p_warmup_iterations=args.warmup_iterations,
                copy_direction=None,
                copy_warmup_iterations=0,
                nvtx_name="p2p_alone",
            )
            alone = _measurement_from_windows(
                direction=direction,
                peer_rank=peer_rank,
                windows=alone_windows,
                buffer_bytes=p2p_buffer_bytes,
                gpu_windows=alone_gpu_windows,
            )
            dist.barrier(group=control_group)

        with _nvtx_range("phase_p2p_send_d2h"):
            dist.barrier(group=control_group)
            send_windows, send_gpu_windows, send_copy = _run_p2p_iterations(
                rank=rank,
                p2p_buffer=p2p_buffer,
                p2p_stream=p2p_stream,
                host_buffer=host_buffer,
                device_buffer=device_buffer,
                copy_stream=copy_stream,
                iterations=args.p2p_iterations,
                p2p_warmup_iterations=args.warmup_iterations,
                copy_direction="d2h" if direction == "send" else None,
                copy_warmup_iterations=args.warmup_iterations,
                nvtx_name="p2p_send_with_d2h",
            )
            send_with_d2h = _measurement_from_windows(
                direction=direction,
                peer_rank=peer_rank,
                windows=send_windows,
                buffer_bytes=p2p_buffer_bytes,
                gpu_windows=send_gpu_windows,
            )
            dist.barrier(group=control_group)

        with _nvtx_range("phase_p2p_recv_h2d"):
            dist.barrier(group=control_group)
            recv_windows, recv_gpu_windows, recv_copy = _run_p2p_iterations(
                rank=rank,
                p2p_buffer=p2p_buffer,
                p2p_stream=p2p_stream,
                host_buffer=host_buffer,
                device_buffer=device_buffer,
                copy_stream=copy_stream,
                iterations=args.p2p_iterations,
                p2p_warmup_iterations=args.warmup_iterations,
                copy_direction="h2d" if direction == "recv" else None,
                copy_warmup_iterations=args.warmup_iterations,
                nvtx_name="p2p_recv_with_h2d",
            )
            recv_with_h2d = _measurement_from_windows(
                direction=direction,
                peer_rank=peer_rank,
                windows=recv_windows,
                buffer_bytes=p2p_buffer_bytes,
                gpu_windows=recv_gpu_windows,
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
            p2p_buffer_mib=args.p2p_buffer_mib,
            baseline=baseline,
            p2p_send_alone=alone if direction == "send" else None,
            p2p_recv_alone=alone if direction == "recv" else None,
            p2p_send_with_d2h=send_with_d2h if direction == "send" else None,
            p2p_recv_with_h2d=recv_with_h2d if direction == "recv" else None,
            send_d2h_copy=send_copy if direction == "send" else None,
            recv_h2d_copy=recv_copy if direction == "recv" else None,
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
                "copy_summary": _summarize_copy(records),
                "p2p_summary": _summarize_p2p(records),
                "ranks": records,
                "p2p_topology": {"pairs": [[0, 1], [2, 3]], "sender_ranks": [0, 2], "receiver_ranks": [1, 3]},
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
                "p2p_metric_note": "bandwidth_gb_s is completed tensor payload, not wire-level traffic",
                "torch_version": str(torch.__version__),
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
            LOGGER.info("wrote P2P benchmark result to %s", args.output)
            p2p_summary = cast(dict[str, dict[str, float]], output["p2p_summary"])
            for p2p_direction, values in p2p_summary.items():
                copy_direction = "d2h" if p2p_direction == "send" else "h2d"
                LOGGER.info(
                    "%s alone=%.3f ms/%.2f GB/s with-%s=%.3f ms/%.2f GB/s "
                    "device-slowdown=%.2f%% host-alone=%.3f ms host-with-%s=%.3f ms",
                    p2p_direction,
                    values["alone_mean_completion_ms"],
                    values["alone_mean_bandwidth_gb_s"],
                    copy_direction,
                    values[f"with_{copy_direction}_mean_completion_ms"],
                    values[f"with_{copy_direction}_mean_bandwidth_gb_s"],
                    values["completion_slowdown_percent"],
                    values["alone_mean_host_completion_ms"],
                    copy_direction,
                    values[f"with_{copy_direction}_mean_host_completion_ms"],
                )
    finally:
        dist.destroy_process_group(control_group)
        dist.destroy_process_group()


def main() -> None:
    """Parse arguments and run the distributed P2P benchmark."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    _run(_build_parser().parse_args())


if __name__ == "__main__":
    main()
