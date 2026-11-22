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
"""Measure BF16 GEMMs with local H2D or multicast copy-engine background traffic.

Run one torchrun worker per GPU on one host. The AG modes repeatedly write immutable,
rank-disjoint shards and use a rank barrier before releasing the measured batch and after
draining it. Each CUDA-event copy interval measures this rank's outgoing operation.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from collections.abc import Callable
from pathlib import Path

import torch
import torch.distributed as dist
from torch.cuda import nvtx


logger = logging.getLogger(__name__)
MODES = ("gemm", "gemm-h2d", "gemm-ag-hbm", "gemm-ag-host", "h2d", "ag-hbm", "ag-host")
TimingPair = tuple[torch.cuda.Event, torch.cuda.Event]


def _parse_shape(value: str) -> tuple[int, int, int]:
    try:
        shape = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("--gemm expects a JSON list [M,N,K]") from exc
    if not isinstance(shape, list) or len(shape) != 3 or any(type(dim) is not int or dim <= 0 for dim in shape):
        raise argparse.ArgumentTypeError("--gemm expects three positive integers [M,N,K]")
    return shape[0], shape[1], shape[2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse one case and its GEMM and communication settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gemm", type=_parse_shape, help="One GEMM shape as a JSON list [M,N,K]")
    parser.add_argument("--size-mib", type=int, required=True, help="Full AllGather output or local H2D size in MiB")
    parser.add_argument("--warmup", type=int, required=True, help="Warmup rounds per active stream")
    parser.add_argument("--gemm-rounds", type=int, required=True)
    parser.add_argument(
        "--comm-rounds", type=int, required=True, help="Communication rounds, extended if needed in mixed modes"
    )
    args = parser.parse_args(argv)
    if args.mode.startswith("gemm") and args.gemm is None:
        parser.error("--gemm is required in GEMM modes")
    for name in ("size_mib", "gemm_rounds", "comm_rounds"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    return args


def _event() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=True)


def _calibrate(op: Callable[[], object], stream: torch.cuda.Stream, *, warmup: int) -> float:
    start, end = _event(), _event()
    with torch.cuda.stream(stream):
        for _ in range(max(1, warmup)):
            op()
        start.record()
        for _ in range(3):
            op()
        end.record()
    end.synchronize()
    return start.elapsed_time(end) / 3


def _enqueue(op: Callable[[], object], pairs: list[TimingPair], label: str) -> None:
    for index, (start, end) in enumerate(pairs):
        with nvtx.range(f"{label}{index:02d}"):
            start.record()
            op()
            end.record()


def _elapsed(origin: torch.cuda.Event, pairs: list[TimingPair]) -> list[list[float]]:
    return [[origin.elapsed_time(start), origin.elapsed_time(end)] for start, end in pairs]


def _overlap_fractions(gemms: list[list[float]], copies: list[list[float]]) -> list[float]:
    fractions = []
    for start, end in gemms:
        overlap = sum(max(0.0, min(end, stop) - max(start, begin)) for begin, stop in copies)
        fractions.append(min(1.0, overlap / (end - start)) if end > start else 0.0)
    return fractions


def _run_case(args: argparse.Namespace, device: torch.device) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
    from ce_multicast import MulticastAllGather
    from mbench.gate import LaunchGate

    rank, world_size = dist.get_rank(), dist.get_world_size()
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    physical_gpu = int(visible_devices[device.index]) if all(value.isdecimal() for value in visible_devices) else None
    do_gemm = args.mode.startswith("gemm")
    comm_kind = args.mode.removeprefix("gemm-") if args.mode != "gemm" else None
    size_bytes = args.size_mib << 20
    comm_payload_bytes = 0
    gemm_stream, comm_stream = torch.cuda.Stream(), torch.cuda.Stream()
    control_stream = torch.cuda.Stream()
    gemm: Callable[[], object] | None = None
    comm: Callable[[], object] | None = None
    validate_comm: Callable[[], None] | None = None
    reset_comm: Callable[[], object] | None = None
    if do_gemm:
        m, n, k = args.gemm
        torch.backends.cuda.preferred_blas_library("cublaslt")
        torch.manual_seed(0)
        a = torch.randn((m, k), dtype=torch.bfloat16, device=device)
        b = torch.randn((k, n), dtype=torch.bfloat16, device=device)
        c = torch.empty((m, n), dtype=torch.bfloat16, device=device)

        def run_gemm() -> torch.Tensor:
            return torch.mm(a, b, out=c)

        gemm = run_gemm

    if comm_kind == "h2d":
        source = torch.empty(size_bytes, dtype=torch.uint8, pin_memory=True).fill_(1)
        destination = torch.empty(size_bytes, dtype=torch.uint8, device=device)
        comm_payload_bytes = size_bytes

        def copy_h2d() -> torch.Tensor:
            return destination.copy_(source, non_blocking=True)

        def validate_h2d() -> None:
            if not torch.all(destination == 1).item():
                raise RuntimeError(f"Rank {rank} received incorrect H2D bytes")

        comm, reset_comm, validate_comm = copy_h2d, destination.zero_, validate_h2d
    elif comm_kind is not None:
        all_gather = MulticastAllGather(device, rank, world_size, size_bytes, comm_kind == "ag-host")
        comm, reset_comm, validate_comm = all_gather.run, all_gather.reset_output, all_gather.validate
        comm_payload_bytes = size_bytes - size_bytes // world_size

    torch.cuda.synchronize(device)
    dist.barrier()
    gemm_ms = _calibrate(gemm, gemm_stream, warmup=args.warmup) if gemm is not None else 0.0
    dist.barrier()
    comm_ms = _calibrate(comm, comm_stream, warmup=args.warmup) if comm is not None else 0.0
    torch.cuda.synchronize(device)
    dist.barrier()

    comm_rounds = args.comm_rounds if comm is not None else 0
    if gemm is not None and comm is not None:
        if comm_ms <= 0:
            raise RuntimeError("Copy calibration did not produce a positive duration")
        target = math.ceil(1.5 * args.gemm_rounds * gemm_ms / comm_ms)
        required = torch.tensor(max(comm_rounds, target), dtype=torch.int64)
        dist.all_reduce(required, op=dist.ReduceOp.MAX)
        comm_rounds = int(required.item())
    if reset_comm is not None:
        reset_comm()
    if gemm is not None:
        # Validate representative corners with a separate FP32 calculation outside the capture.
        sample_rows, sample_cols = min(4, m), min(4, n)
        reference = (a[-sample_rows:].float() @ b[:, -sample_cols:].float()).to(c.dtype)
        c.fill_(float("nan"))

    gemm_events = [(_event(), _event()) for _ in range(args.gemm_rounds if gemm is not None else 0)]
    comm_events = [(_event(), _event()) for _ in range(comm_rounds)]
    origin = _event()
    # PyTorch lazily creates CUDA events on first record, so instantiate all of them before capture.
    origin.record()
    for start, end in gemm_events + comm_events:
        start.record()
        end.record()
    torch.cuda.synchronize(device)
    dist.barrier()

    gate = LaunchGate(timeout_s=30.0)
    try:
        torch.cuda.profiler.start()
        try:
            with nvtx.range(f"hbm-ce/{args.case}"):
                gate.arm(control_stream)
                try:
                    origin.record(control_stream)
                    if comm is not None:
                        with torch.cuda.stream(comm_stream):
                            comm_stream.wait_event(origin)
                            _enqueue(comm, comm_events, "comm")
                    if gemm is not None:
                        with torch.cuda.stream(gemm_stream):
                            gemm_stream.wait_event(origin)
                            _enqueue(gemm, gemm_events, "gemm")
                    # Host rendezvous releases only fully queued streams across all ranks.
                    dist.barrier()
                finally:
                    gate.release()
                torch.cuda.synchronize(device)
        finally:
            torch.cuda.profiler.stop()
        watchdog = torch.tensor(int(gate.watchdog_fired()))
        dist.all_reduce(watchdog, op=dist.ReduceOp.MAX)
        if watchdog.item():
            raise RuntimeError("CUDA launch gate timed out before all ranks queued the measured batch")
    finally:
        gate.close()

    # Each local sync guarantees completion of its outgoing peer writes. This
    # barrier makes the received data safe to inspect on every destination.
    dist.barrier()
    comm_correct = True
    if validate_comm is not None:
        try:
            validate_comm()
        except RuntimeError as exc:
            logger.error("%s", exc)
            comm_correct = False
    gemm_correct = True
    if gemm is not None:
        gemm_correct = torch.allclose(c[-sample_rows:, -sample_cols:], reference, rtol=0.016, atol=0.1)
    gemm_intervals = _elapsed(origin, gemm_events)
    comm_intervals = _elapsed(origin, comm_events)
    props = torch.cuda.get_device_properties(device)
    pci_address = None
    if hasattr(props, "pci_bus_id"):
        pci_address = f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
    result = {
        "rank": rank,
        "world_size": world_size,
        "physical_gpu": physical_gpu,
        "device_name": props.name,
        "pci_address": pci_address,
        "case": args.case,
        "mode": args.mode,
        "gemm_mnk": args.gemm if do_gemm else None,
        "gemm_ms": gemm_intervals,
        "comm_ms": comm_intervals,
        "gemm_comm_overlap": _overlap_fractions(gemm_intervals, comm_intervals),
        "size_bytes": size_bytes,
        "comm_payload_bytes": comm_payload_bytes,
        "comm_rounds_requested": args.comm_rounds,
        "comm_rounds_measured": comm_rounds,
        "comm_output_correct": comm_correct,
        "gemm_output_correct": gemm_correct,
        "output_correct": comm_correct and gemm_correct,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"events.rank_{rank:02d}.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    logger.info("Rank %d recorded %d GEMMs and %d copies in %s", rank, len(gemm_events), comm_rounds, output)
    dist.barrier()
    if not result["output_correct"]:
        raise RuntimeError(f"Rank {rank} output validation failed. See {output}")


def main(argv: list[str] | None = None) -> None:
    """Initialize one torchrun CUDA worker and run its case."""
    logging.basicConfig(level=logging.INFO)
    args = parse_args(argv)
    if int(os.environ["LOCAL_WORLD_SIZE"]) != int(os.environ["WORLD_SIZE"]):
        raise ValueError("The copy-engine benchmark expects one torchrun host")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("gloo")
    try:
        _run_case(args, device)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
