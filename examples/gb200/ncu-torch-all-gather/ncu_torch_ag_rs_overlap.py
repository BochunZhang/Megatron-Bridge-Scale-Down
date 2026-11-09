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
"""torch all-gather + reduce-scatter worker: an Nsight Compute capture of the two running at once.

Started by ``run_ncu_torch_all_gather.sh --ag-rs`` as ``ncu ... torchrun ... ncu_torch_ag_rs_overlap.py``.
Every launch is one all-gather and one reduce-scatter started together. The concurrent part follows
``../nvls-collective/nvls_collective_benchmark.py``:

* the default process group is gloo, so the only NCCL communicators are one for the all-gather
  and one for the reduce-scatter, each created eagerly (``device_id``);
* the AG and the RS are launched from two CUDA streams (``torch.cuda.stream``), so the two NCCL
  kernels have no stream dependency and run concurrently on the GPU;
* the first AG and the first RS run one after the other with a device sync in between: NCCL
  connects a communicator's channels in its first collective, and doing that while the other
  communicator's kernel waits for peers can deadlock.

Every rank runs the same sequence, so ncu sees the same launches on every rank:

1. the serial connect (one AG, then one RS);
2. ``--warmup - 1`` concurrent AG + RS launches outside any NVTX range;
3. ``--iters`` concurrent AG + RS launches inside the NVTX range ``ncu_collectives``. In app-range
   replay that range is the profiled workload; ncu does not serialize the kernels inside it, so the
   AG and the RS still overlap. Kernel and application replay profile (and serialize) single kernels.

CUDA events around each op report how much the AG and RS overlapped in every profiled launch.

``--algo`` must match the NCCL environment the launcher exported:

* ``ring`` / ``nvls``: ``NCCL_ALGO`` is forced for both functions, buffers are plain device memory;
* ``symm`` / ``p2p``: buffers live in a ``torch.cuda.MemPool`` backed by ``ncclMemAlloc``,
  registered as ``NCCL_WIN_COLL_SYMMETRIC`` windows on both communicators. symm runs the multicast
  kernels (``AllGather_STMC`` / ``ReduceScatter_LDMC``), p2p (``NCCL_NVLS_ENABLE=0``) the unicast
  ones (``AllGather_ST`` / ``ReduceScatter_LD``); the launcher's TUNING check enforces which.
"""

import argparse
import contextlib
import logging
import os

import torch
import torch.distributed as dist
from symm_clock64_csv import Clock64Capture
from torch.cuda import nvtx


logger = logging.getLogger(__name__)

# The NVTX range around the profiled launches; the launcher's --nvtx-include matches it.
NVTX_RANGE = "ncu_collectives"

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
# Launched together, each from a stream of its own.
OPS = ("ag", "rs")

# Algorithms that run NCCL's symmetric kernels on registered windows.
SYMMETRIC_ALGOS = ("symm", "p2p")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algo", choices=("ring", "nvls", "symm", "p2p"), required=True)
    parser.add_argument(
        "--size-mib", type=int, required=True, help="All-gather output = reduce-scatter input size per rank in MiB"
    )
    parser.add_argument("--dtype", choices=tuple(DTYPES), required=True)
    parser.add_argument(
        "--warmup", type=int, required=True, help="AG + RS launches before the NVTX range, the serial one included"
    )
    parser.add_argument("--iters", type=int, required=True, help="Concurrent AG + RS launches inside the NVTX range")
    args = parser.parse_args()
    if args.warmup < 1:
        parser.error("--warmup must be at least 1: the first launch connects the communicators serially")
    return args


def _check_env(algo: str) -> None:
    """Fail when the NCCL environment does not select ``algo``."""
    nccl_algo = os.environ.get("NCCL_ALGO", "")
    if algo in ("ring", "nvls") and nccl_algo.lower() != algo:
        raise RuntimeError(f"--algo {algo} requires NCCL_ALGO={algo}, got {nccl_algo!r}")
    if algo in SYMMETRIC_ALGOS:
        if nccl_algo:
            raise RuntimeError(f"--algo {algo} requires NCCL_ALGO to be unset, got {nccl_algo!r}")
        if os.environ.get("NCCL_WIN_ENABLE") != "1":
            raise RuntimeError(f"--algo {algo} requires NCCL_WIN_ENABLE=1 to register symmetric windows")
        # NCCL_SYM_KERNEL may be unset: NCCL 2.29 takes a single kernel name, which would leave the
        # other function without a symmetric kernel, so the launcher pins only what it can.
    nvls = os.environ.get("NCCL_NVLS_ENABLE")
    if algo == "p2p" and nvls != "0":
        raise RuntimeError(f"--algo p2p requires NCCL_NVLS_ENABLE=0, got {nvls!r}")


def _nccl_backend(pg: dist.ProcessGroup) -> dist.ProcessGroupNCCL:
    """Return the ``ProcessGroupNCCL`` behind ``pg``; ``mem_allocator`` and pool registration live there."""
    return pg._get_backend(torch.device("cuda"))


def _check_ag(out: torch.Tensor, world_size: int, shard: int) -> None:
    """Chunk ``k`` of the gathered tensor must hold ``k + 1`` everywhere."""
    expected = torch.arange(1, world_size + 1, device=out.device, dtype=out.dtype).repeat_interleave(shard)
    if not torch.equal(out, expected):
        bad = (out != expected).nonzero()[0].item()
        raise RuntimeError(f"all-gather output wrong at element {bad}: {out[bad].item()} != {expected[bad].item()}")


def _check_rs(out: torch.Tensor, world_size: int) -> None:
    """Every element is the sum of the ranks' inputs ``rank % 2 + 1``, a small integer exact in bf16."""
    expected = float(sum(rank % 2 + 1 for rank in range(world_size)))
    if not torch.all(out == expected):
        bad = (out != expected).nonzero()[0].item()
        raise RuntimeError(f"reduce-scatter output wrong at element {bad}: {out[bad].item()} != {expected}")


def main() -> None:
    """Connect both communicators, run the warmup and the NVTX-ranged launches, and check the results."""
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    _check_env(args.algo)

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    # Gloo carries the barriers, so no world-sized NCCL communicator exists besides the two below.
    dist.init_process_group(backend="gloo")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    ranks = list(range(world_size))
    # new_group is collective over the default group: every rank creates both, in the same order.
    # device_id makes each NCCL communicator eager, so windows can be registered before any collective.
    ag_pg = dist.new_group(ranks=ranks, backend="nccl", device_id=device)
    rs_pg = dist.new_group(ranks=ranks, backend="nccl", device_id=device)

    dtype = DTYPES[args.dtype]
    elem = torch.empty((), dtype=dtype).element_size()
    size_bytes = args.size_mib << 20
    if size_bytes % (world_size * elem):
        raise ValueError(f"{size_bytes} bytes cannot be split into {world_size} {args.dtype} shards")
    numel = size_bytes // elem
    shard = numel // world_size

    # One pool, registered as a symmetric window on both communicators. It must allocate through
    # ncclMemAlloc: a pool with its own allocator never uses expandable segments, whose
    # multi-handle segments NCCL would only partly map to peers.
    pool = torch.cuda.MemPool(_nccl_backend(ag_pg).mem_allocator) if args.algo in SYMMETRIC_ALGOS else None
    with torch.cuda.use_mem_pool(pool) if pool else contextlib.nullcontext():
        ag_in = torch.full((shard,), float(rank + 1), dtype=dtype, device=device)
        ag_out = torch.empty(numel, dtype=dtype, device=device)
        rs_in = torch.full((numel,), float(rank % 2 + 1), dtype=dtype, device=device)
        rs_out = torch.empty(shard, dtype=dtype, device=device)
    if pool is not None:
        torch.cuda.synchronize()
        # Collective over each communicator: ncclCommWindowRegister ends in a bootstrap barrier.
        for pg in (ag_pg, rs_pg):
            _nccl_backend(pg).register_mem_pool(pool, symm=True)
        torch.cuda.synchronize()

    streams = {op: torch.cuda.Stream() for op in OPS}

    def launch() -> dict[str, tuple[torch.cuda.Event, torch.cuda.Event]]:
        """Launch the AG and the RS, each on its own stream; return their start/end events."""
        events = {}
        for op in OPS:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(streams[op]):
                start.record()
                # async_op=False runs NCCL on the current stream and does not block the host.
                if op == "ag":
                    dist.all_gather_into_tensor(ag_out, ag_in, group=ag_pg)
                else:
                    dist.reduce_scatter_tensor(rs_out, rs_in, group=rs_pg)
                end.record()
            events[op] = (start, end)
        return events

    if rank == 0:
        logger.info(
            "torch %s, NCCL %s, algo %s, AG + RS, %d ranks, %d MiB %s per rank, warmup %d, iters %d",
            torch.__version__,
            ".".join(map(str, torch.cuda.nccl.version())),
            args.algo,
            world_size,
            args.size_mib,
            args.dtype,
            args.warmup,
            args.iters,
        )

    # 1. Serial connect: one op at a time with a device sync, so no communicator connects its
    #    channels while the other one's kernel is waiting for peers.
    for op in OPS:
        with torch.cuda.stream(streams[op]):
            if op == "ag":
                dist.all_gather_into_tensor(ag_out, ag_in, group=ag_pg)
            else:
                dist.reduce_scatter_tensor(rs_out, rs_in, group=rs_pg)
        torch.cuda.synchronize()
        dist.barrier()
    # 2. The rest of the warmup, concurrent like the profiled launches.
    for _ in range(args.warmup - 1):
        launch()
    torch.cuda.synchronize()
    _check_ag(ag_out, world_size, shard)
    _check_rs(rs_out, world_size)
    dist.barrier()

    ag_out.zero_()
    rs_out.zero_()
    capture = None
    if profile := os.environ.get("NCCL_SYM_CLOCK64_CSV_PROFILE"):
        if args.algo != "symm":
            raise ValueError("Clock64 CSV capture requires --algo symm")
        capture = Clock64Capture(profile=profile, device=device, shard_bytes=shard * elem, launches=args.iters)
    torch.cuda.synchronize()
    # 3. The profiled launches. range_push/pop must run on the thread that launches the kernels.
    with capture.arm() if capture is not None else contextlib.nullcontext():
        nvtx.range_push(NVTX_RANGE)
        try:
            timings = [launch() for _ in range(args.iters)]
            torch.cuda.synchronize()
        finally:
            nvtx.range_pop()
    _check_ag(ag_out, world_size, shard)
    _check_rs(rs_out, world_size)
    if capture is not None:
        capture.write(rank=rank)
    dist.barrier()

    if rank == 0:
        for i, events in enumerate(timings):
            # All events are on one device, so elapsed_time works across streams. Times are relative
            # to the earliest start; under ncu they include the profiler's overhead.
            ref = next(iter(events.values()))[0]
            raw = {op: (ref.elapsed_time(s), ref.elapsed_time(e)) for op, (s, e) in events.items()}
            first = min(a for a, _ in raw.values())
            spans = {op: (a - first, b - first) for op, (a, b) in raw.items()}
            (a0, a1), (r0, r1) = spans["ag"], spans["rs"]
            overlap = max(0.0, min(a1, r1) - max(a0, r0))
            logger.info(
                "iter %d: ag %.3f-%.3f ms, rs %.3f-%.3f ms, overlap %.3f ms (%.0f%% of the longer op)",
                i,
                a0,
                a1,
                r0,
                r1,
                overlap,
                100 * overlap / max(a1 - a0, r1 - r0),
            )
        logger.info("AG and RS output correct after warmup and the %s range", NVTX_RANGE)

    if pool is not None:
        # Release the windows before the pool's memory goes away.
        for pg in (ag_pg, rs_pg):
            _nccl_backend(pg).deregister_mem_pool(pool)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
