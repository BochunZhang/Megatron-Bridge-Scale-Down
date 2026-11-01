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
"""GEMM time on GB200 while a few SMs move unicast symmetric AllGather traffic between the GPUs.

One torchrun worker per GPU; ``run_hbm_conflict_benchmark.sh`` launches one job per case and wraps
every worker in ``nsys profile`` (``../nvls-collective/nsys_worker.sh``), so each case leaves one
report per rank. The whole measured sequence is one ``cudaProfilerStart``/``Stop`` window.

* ``--mode baseline``: no AllGather. Each rank runs ``--gemm-rounds`` back-to-back GEMMs
  ``[M, K] x [K, N]`` (bf16) on the GEMM stream.
* ``--mode sweep``: ``--ag-rounds`` AllGathers of ``--size-mib`` (the gathered tensor) back to
  back on the comm stream. The first ``--ag-warmup`` are warmup, the next ``--ag-isolated`` run
  alone. An event recorded after the last isolated round releases the GEMM stream, which sleeps
  ``--gemm-delay-us`` on the GPU (so the next AllGather holds its SMs first) and then runs the same
  GEMMs while the remaining rounds keep the NVLink / HBM traffic going.

The AllGather is the unicast symmetric kernel ``AllGather_ST``: the buffers live in a
``torch.cuda.MemPool`` backed by ``ncclMemAlloc`` and registered as a ``NCCL_WIN_COLL_SYMMETRIC``
window, and the launcher pins ``NCCL_SYM_KERNEL=allgather:AllGather_ST`` with ``NCCL_NVLS_ENABLE=0``
(no multicast), ``NCCL_SYM_TMA_ENABLE=0`` and the copy-engine threshold at ``INT64_MAX``. Each
rank stores its shard into every peer's window. The traffic is set by
``NCCL_SYM_AG_ST_PAUSE_CYCLES`` of nccl-2.32.3-pause: every AllGather warp spins that many SM
cycles after each tile it has stored to all peers.

The comm stream has the highest priority, so when an AllGather ends while GEMM CTAs are waiting,
the next AllGather's CTAs are scheduled first and keep their SMs.

Every launch sits in its own NVTX range (``ag<II>``, ``gemm<JJ>``, ``gemm-delay``) inside the case
range ``hbm-conflict/<case>``; ``analyse_hbm_conflict.py`` attributes kernels to them by launch
time. CUDA events time every round as well; each rank writes them to ``events.rank_<RR>.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.cuda import nvtx


logger = logging.getLogger(__name__)

PAUSE_ENV = "NCCL_SYM_AG_ST_PAUSE_CYCLES"
SLEEP_CALIBRATION_CYCLES = 100_000_000
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the worker options; the launcher passes every one of them explicitly."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("baseline", "sweep"), required=True)
    parser.add_argument("--case", required=True, help="Case name, e.g. baseline or pause1024cyc")
    parser.add_argument("--output-dir", type=Path, required=True, help="The case directory")
    parser.add_argument("--size-mib", type=int, required=True, help="AllGather output (gathered tensor) in MiB")
    parser.add_argument("--dtype", choices=sorted(DTYPES), required=True)
    parser.add_argument("--gemm-m", type=int, required=True)
    parser.add_argument("--gemm-k", type=int, required=True)
    parser.add_argument("--gemm-n", type=int, required=True)
    parser.add_argument("--gemm-rounds", type=int, required=True, help="Measured GEMMs")
    parser.add_argument("--gemm-warmup", type=int, required=True, help="GEMMs before the capture window")
    parser.add_argument("--gemm-delay-us", type=float, required=True, help="GPU sleep before the first GEMM (sweep)")
    parser.add_argument("--ag-rounds", type=int, required=True, help="AllGathers in total (sweep)")
    parser.add_argument("--ag-warmup", type=int, required=True)
    parser.add_argument("--ag-isolated", type=int, required=True)
    args = parser.parse_args(argv)
    for name in ("size_mib", "gemm_m", "gemm_k", "gemm_n", "gemm_rounds"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.gemm_warmup < 0 or args.ag_warmup < 0 or args.ag_isolated < 0 or args.gemm_delay_us < 0:
        parser.error("--gemm-warmup, --ag-warmup, --ag-isolated and --gemm-delay-us must not be negative")
    if args.mode == "sweep" and args.ag_warmup + args.ag_isolated >= args.ag_rounds:
        parser.error("--ag-rounds must leave background rounds after --ag-warmup + --ag-isolated")
    return args


def release_round(args: argparse.Namespace) -> int:
    """Index of the AllGather after which the GEMM stream is released (the last isolated round)."""
    return args.ag_warmup + args.ag_isolated - 1


def _calibrate_sleep_cycles_per_us() -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda._sleep(SLEEP_CALIBRATION_CYCLES // 10)
    start.record()
    torch.cuda._sleep(SLEEP_CALIBRATION_CYCLES)
    end.record()
    end.synchronize()
    return SLEEP_CALIBRATION_CYCLES / (start.elapsed_time(end) * 1000.0)


def _symmetric_buffers(
    backend: dist.ProcessGroupNCCL, *, rank: int, world_size: int, size_bytes: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.cuda.MemPool]:
    """AllGather input / output inside one ncclMemAlloc pool, registered as a symmetric window.

    The pool needs NCCL's own allocator: NCCL maps to peers only the cuMem allocation that holds
    the window base, so an expandable-segments pool would leave most of the window unmapped
    (see ../nvls-collective/nvls_collective_benchmark.py ``_make_buffers``).
    """
    elem = torch.empty((), dtype=dtype).element_size()
    if size_bytes % (world_size * elem):
        raise ValueError(f"{size_bytes} bytes cannot be split into {world_size} shards of {dtype}")
    numel = size_bytes // elem
    pool = torch.cuda.MemPool(backend.mem_allocator)
    with torch.cuda.use_mem_pool(pool):
        # Chunk k of the gathered tensor must equal k + 1.
        ag_in = torch.full((numel // world_size,), float(rank + 1), dtype=dtype, device="cuda")
        ag_out = torch.empty(numel, dtype=dtype, device="cuda")
    torch.cuda.synchronize()
    backend.register_mem_pool(pool, symm=True)
    torch.cuda.synchronize()
    return ag_in, ag_out, pool


def _elapsed(origin: torch.cuda.Event, events: list[tuple[torch.cuda.Event, torch.cuda.Event]]) -> list[list[float]]:
    return [[origin.elapsed_time(start), origin.elapsed_time(end)] for start, end in events]


def main(argv: list[str] | None = None) -> None:
    """Run one case on this rank and write ``events.rank_<RR>.json``."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [rank%(process)d] %(message)s")
    args = parse_args(argv)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    # device_id creates the NCCL communicator now; register_mem_pool needs it.
    dist.init_process_group(backend="nccl", device_id=device)
    rank, world_size = dist.get_rank(), dist.get_world_size()
    backend = dist.group.WORLD._get_backend(device)
    dtype = DTYPES[args.dtype]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Highest priority for the comm stream (lower number = higher priority).
    comm_stream = torch.cuda.Stream(priority=torch.cuda.Stream.priority_range()[1])
    gemm_stream = torch.cuda.Stream()
    a = torch.randn(args.gemm_m, args.gemm_k, dtype=dtype, device=device)
    b = torch.randn(args.gemm_k, args.gemm_n, dtype=dtype, device=device)
    c = torch.empty(args.gemm_m, args.gemm_n, dtype=dtype, device=device)
    with torch.cuda.stream(gemm_stream):
        for _ in range(args.gemm_warmup):
            torch.mm(a, b, out=c)
    torch.cuda.synchronize()
    cycles_per_us = _calibrate_sleep_cycles_per_us()

    ag_in = ag_out = pool = None
    if args.mode == "sweep":
        ag_in, ag_out, pool = _symmetric_buffers(
            backend, rank=rank, world_size=world_size, size_bytes=args.size_mib << 20, dtype=dtype
        )

    def timing_pair() -> tuple[torch.cuda.Event, torch.cuda.Event]:
        return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    ag_events = [timing_pair() for _ in range(args.ag_rounds if args.mode == "sweep" else 0)]
    gemm_events = [timing_pair() for _ in range(args.gemm_rounds)]
    origin = torch.cuda.Event(enable_timing=True)
    release = torch.cuda.Event()

    dist.barrier(device_ids=[local_rank])
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    nvtx.range_push(f"hbm-conflict/{args.case}")
    if args.mode == "sweep":
        with torch.cuda.stream(comm_stream):
            origin.record()
            for i, (start, end) in enumerate(ag_events):
                nvtx.range_push(f"ag{i:02d}")
                start.record()
                # async_op=False runs NCCL on the current stream without blocking the host.
                dist.all_gather_into_tensor(ag_out, ag_in)
                end.record()
                nvtx.range_pop()
                if i == release_round(args):
                    release.record()
        gemm_stream.wait_event(release)
    else:
        with torch.cuda.stream(gemm_stream):
            origin.record()
    with torch.cuda.stream(gemm_stream):
        if args.mode == "sweep" and args.gemm_delay_us > 0:
            nvtx.range_push("gemm-delay")
            torch.cuda._sleep(int(args.gemm_delay_us * cycles_per_us))
            nvtx.range_pop()
        for j, (start, end) in enumerate(gemm_events):
            nvtx.range_push(f"gemm{j:02d}")
            start.record()
            torch.mm(a, b, out=c)
            end.record()
            nvtx.range_pop()
    torch.cuda.synchronize()
    nvtx.range_pop()
    torch.cuda.profiler.stop()

    ag_ok = None
    if ag_out is not None:
        expected = torch.arange(1, world_size + 1, dtype=dtype, device=device).view(world_size, 1)
        ag_ok = bool((ag_out.view(world_size, -1) == expected).all().item())
        if not ag_ok:
            logger.error("rank %d: the AllGather output is wrong", rank)

    props = torch.cuda.get_device_properties(device)
    record = {
        "case": args.case,
        "mode": args.mode,
        "rank": rank,
        "world_size": world_size,
        "device": props.name,
        "pci_bus_id": props.pci_bus_id,
        "sm_count": props.multi_processor_count,
        PAUSE_ENV: os.environ.get(PAUSE_ENV),
        "size_mib": args.size_mib,
        "dtype": args.dtype,
        "gemm_mkn": [args.gemm_m, args.gemm_k, args.gemm_n],
        "gemm_delay_us": args.gemm_delay_us if args.mode == "sweep" else 0.0,
        "sleep_cycles_per_us": cycles_per_us,
        "ag_warmup": args.ag_warmup,
        "ag_isolated": args.ag_isolated,
        # [start_ms, end_ms] of every round, relative to the first event of the window.
        "ag_ms": _elapsed(origin, ag_events),
        "gemm_ms": _elapsed(origin, gemm_events),
        "ag_correct": ag_ok,
    }
    (args.output_dir / f"events.rank_{rank:02d}.json").write_text(json.dumps(record, indent=2) + "\n")

    if pool is not None:
        torch.cuda.synchronize()
        backend.deregister_mem_pool(pool)
    dist.barrier(device_ids=[local_rank])
    dist.destroy_process_group()
    if ag_ok is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
