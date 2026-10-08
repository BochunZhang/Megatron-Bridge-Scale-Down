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

"""Measure GPU0 C2C bandwidth without and with persistent host or CUDA RDMA traffic.

The launcher first runs a baseline with ``--rdma-role none`` and no background
traffic. For each RDMA direction, it then starts the ``ib_write_bw`` receiver and
sender, waits ten seconds, and invokes this benchmark with ``--rdma-role recv``
or ``send`` to identify GPU0's NIC role. Every round uses
``CUDA_VISIBLE_DEVICES=0`` and measures D2H/H2D copies on ``cuda:0`` using CUDA
events. The launcher owns the background processes and stops them after this
script exits before starting the next round. ``--rdma-memory`` records whether
perftest uses host memory or CUDA buffers for its background traffic. RDMA
throughput is recorded separately by perftest.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


LOGGER = logging.getLogger(__name__)
BYTES_PER_MIB = 1024 * 1024
BYTES_PER_GB = 1_000_000_000


@dataclass(frozen=True)
class CopyMeasurement:
    """Result of one host/device copy direction."""

    direction: str
    bandwidth_gb_s: float
    elapsed_seconds: float
    transferred_bytes: int


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c2c-buffer-mib", type=_positive_int, default=512)
    parser.add_argument("--warmup-iterations", type=_positive_int, default=5)
    parser.add_argument("--copy-iterations", type=_positive_int, default=20)
    parser.add_argument(
        "--rdma-role",
        choices=("none", "recv", "send"),
        required=True,
        help="GPU0 NIC's role in the external RDMA traffic; none measures the baseline without background traffic",
    )
    parser.add_argument(
        "--rdma-memory",
        choices=("host", "cuda"),
        default="host",
        help="Memory used by the external RDMA background (default: host)",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _bandwidth_gb_s(*, transferred_bytes: int, elapsed_seconds: float) -> float:
    if transferred_bytes <= 0:
        raise ValueError("transferred_bytes must be positive")
    if elapsed_seconds <= 0:
        raise ValueError("elapsed_seconds must be positive")
    return transferred_bytes / elapsed_seconds / BYTES_PER_GB


@contextmanager
def _nvtx_range(name: str) -> Iterator[None]:
    """Annotate a host scope for Nsight Systems through PyTorch NVTX APIs."""
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def _measure_copy(
    *,
    direction: str,
    host_buffer: torch.Tensor,
    device_buffer: torch.Tensor,
    stream: torch.cuda.Stream,
    warmup_iterations: int,
    copy_iterations: int,
) -> CopyMeasurement:
    """Measure fixed-size asynchronous H2D or D2H copies."""
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


def _run(args: argparse.Namespace) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("run through the shell launcher with CUDA_VISIBLE_DEVICES=0 to measure physical GPU0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the GPU0 C2C benchmark")
    torch.cuda.set_device(0)
    host_buffer = torch.empty(args.c2c_buffer_mib * BYTES_PER_MIB, dtype=torch.uint8, pin_memory=True)
    device_buffer = torch.empty_like(host_buffer, device="cuda:0")
    stream = torch.cuda.Stream(device=device_buffer.device)
    measurements = [
        _measure_copy(
            direction=direction,
            host_buffer=host_buffer,
            device_buffer=device_buffer,
            stream=stream,
            warmup_iterations=args.warmup_iterations,
            copy_iterations=args.copy_iterations,
        )
        for direction in ("d2h", "h2d")
    ]
    has_background = args.rdma_role != "none"
    measurement_suffix = args.rdma_role if has_background else "baseline"
    output = {
        "benchmark": "gb200_c2c_with_rdma_p2p",
        "background_backend": "perftest/ib_write_bw" if has_background else None,
        "background_managed_by": "shell" if has_background else None,
        "measurement_phase": "with_background" if has_background else "baseline",
        "rdma_role": args.rdma_role,
        "rdma_memory": args.rdma_memory if has_background else None,
        "rdma_sender_gpu_index": {"recv": 3, "send": 0}.get(args.rdma_role),
        "rdma_receiver_gpu_index": {"recv": 0, "send": 3}.get(args.rdma_role),
        "gpu_data_transport": "measured host-device C2C copies",
        "hostname": socket.gethostname(),
        "gpu_index": 0,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_pci_bus_id": os.environ.get("BENCHMARK_GPU_PCI_BUS_ID", "unknown"),
        "c2c_config": {
            "buffer_mib": args.c2c_buffer_mib,
            "warmup_iterations": args.warmup_iterations,
            "copy_iterations": args.copy_iterations,
        },
        "measurements": [
            {"name": f"{measurement.direction}_{measurement_suffix}", **asdict(measurement)}
            for measurement in measurements
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("wrote GPU0 C2C %s measurements to %s", measurement_suffix, args.output)
    for measurement in measurements:
        LOGGER.info(
            "%s_%s: %.3f GB/s (%.6f seconds, %d bytes)",
            measurement.direction,
            measurement_suffix,
            measurement.bandwidth_gb_s,
            measurement.elapsed_seconds,
            measurement.transferred_bytes,
        )


def main() -> None:
    """Measure GPU0 copies without background traffic or while its NIC sends or receives RDMA."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _run(_build_parser().parse_args())


if __name__ == "__main__":
    main()
