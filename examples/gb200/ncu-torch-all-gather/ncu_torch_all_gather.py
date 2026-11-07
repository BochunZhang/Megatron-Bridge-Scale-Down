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
"""torch all-gather worker for an Nsight Compute capture of the NCCL AllGather kernel.

Started by ``run_ncu_torch_all_gather.sh`` as ``ncu ... torchrun ... ncu_torch_all_gather.py``.
One ncu wraps the whole torchrun process tree; its shmem communicator launches and replays
each selected NCCL kernel on all ranks together, which the kernel needs to make progress.

Every rank runs the same sequence, so ncu sees the same kernel order on every rank:

1. ``--warmup`` all-gathers outside any NVTX range (communicator setup, first-launch costs).
2. ``--iters`` all-gathers inside the NVTX range ``ncu_all_gather``. The launcher selects the
   profiled work with ``--nvtx-include ncu_all_gather/``: in kernel replay together with
   ``-k regex:^nccl(Symk)?DevKernel_AllGather``, in app-range replay the range itself.

The only NCCL work in the process is all-gather. CPU barriers go over a gloo group, so no
NCCL all-reduce kernel of ``dist.barrier`` enters a kernel filter or a range.

``--algo`` must match the NCCL environment the launcher exported:

* ``ring`` / ``nvls``: ``NCCL_ALGO`` is forced, payload buffers are plain device memory. Both
  launch the kernel symbol ``ncclDevKernel_AllGather_RING`` (NCCL specializes one AllGather
  kernel per build, ``best_kernel`` in ``src/device/generate.py``, and leaves the protocol out of
  the name unless ``NCCL_EXACT_KERNEL_NAMES=1``); the algorithm is dispatched inside it, so only
  the TUNING log tells Ring and NVLS apart.
* ``symm`` / ``p2p``: buffers are allocated from a ``torch.cuda.MemPool`` backed by
  ``ncclMemAlloc`` and registered as ``NCCL_WIN_COLL_SYMMETRIC`` windows; NCCL runs the symmetric
  kernel the launcher pins with ``NCCL_SYM_KERNEL``: ``ncclSymkDevKernel_AllGather_STMC``
  (multicast stores) for ``symm``, ``ncclSymkDevKernel_AllGather_ST`` (peer stores) for ``p2p``.
"""

import argparse
import contextlib
import logging
import os

import torch
import torch.distributed as dist
from torch.cuda import nvtx


logger = logging.getLogger(__name__)

# The NVTX range around the profiled all-gathers; the launcher's --nvtx-include matches it.
NVTX_RANGE = "ncu_all_gather"

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}

# Algorithms that run NCCL's symmetric kernels on registered windows.
SYMMETRIC_ALGOS = ("symm", "p2p")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algo", choices=("ring", "nvls", "symm", "p2p"), required=True)
    parser.add_argument("--size-mib", type=int, required=True, help="All-gather output size per rank in MiB")
    parser.add_argument("--dtype", choices=tuple(DTYPES), required=True)
    parser.add_argument("--warmup", type=int, required=True, help="All-gathers before the NVTX range")
    parser.add_argument("--iters", type=int, required=True, help="All-gathers inside the NVTX range")
    return parser.parse_args()


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
        if not os.environ.get("NCCL_SYM_KERNEL"):
            raise RuntimeError(f"--algo {algo} requires NCCL_SYM_KERNEL to pin the symmetric kernel")


def _nccl_backend(pg: dist.ProcessGroup) -> dist.ProcessGroupNCCL:
    """Return the ``ProcessGroupNCCL`` behind ``pg``; ``mem_allocator`` and pool registration live there."""
    return pg._get_backend(torch.device("cuda"))


def _check_output(out: torch.Tensor, world_size: int, shard: int) -> None:
    """Chunk ``k`` of the gathered tensor must hold ``k + 1`` everywhere."""
    expected = torch.arange(1, world_size + 1, device=out.device, dtype=out.dtype).repeat_interleave(shard)
    if not torch.equal(out, expected):
        bad = (out != expected).nonzero()[0].item()
        raise RuntimeError(f"all-gather output wrong at element {bad}: {out[bad].item()} != {expected[bad].item()}")


def main() -> None:
    """Run the warmup and the NVTX-ranged all-gathers, checking the result after each phase."""
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    _check_env(args.algo)

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    # device_id makes the NCCL communicator eager, so its setup is not inside the first all-gather.
    dist.init_process_group("nccl", device_id=device)
    rank, world_size = dist.get_rank(), dist.get_world_size()
    cpu_group = dist.new_group(backend="gloo")

    dtype = DTYPES[args.dtype]
    elem = torch.empty((), dtype=dtype).element_size()
    size_bytes = args.size_mib << 20
    if size_bytes % (world_size * elem):
        raise ValueError(f"{size_bytes} bytes cannot be split into {world_size} {args.dtype} shards")
    numel = size_bytes // elem
    shard = numel // world_size

    backend = _nccl_backend(dist.group.WORLD)
    # Symmetric windows need ncclMemAlloc memory; a pool with its own allocator never uses
    # expandable segments, whose multi-handle segments NCCL would only partly map to peers.
    pool = torch.cuda.MemPool(backend.mem_allocator) if args.algo in SYMMETRIC_ALGOS else None
    with torch.cuda.use_mem_pool(pool) if pool else contextlib.nullcontext():
        inp = torch.full((shard,), float(rank + 1), dtype=dtype, device=device)
        out = torch.empty(numel, dtype=dtype, device=device)
    if pool is not None:
        torch.cuda.synchronize()
        # Collective over the communicator: ncclCommWindowRegister ends in a bootstrap barrier.
        backend.register_mem_pool(pool, symm=True)
        torch.cuda.synchronize()

    if rank == 0:
        logger.info(
            "torch %s, NCCL %s, algo %s, %d ranks, %d MiB %s output (%d B per rank's shard), warmup %d, iters %d",
            torch.__version__,
            ".".join(map(str, torch.cuda.nccl.version())),
            args.algo,
            world_size,
            args.size_mib,
            args.dtype,
            shard * elem,
            args.warmup,
            args.iters,
        )

    for _ in range(args.warmup):
        dist.all_gather_into_tensor(out, inp)
    torch.cuda.synchronize()
    if args.warmup:
        _check_output(out, world_size, shard)
    dist.barrier(group=cpu_group)

    out.zero_()
    # range_push/pop must run on the thread that launches the kernels; torch enqueues NCCL from it.
    nvtx.range_push(NVTX_RANGE)
    for _ in range(args.iters):
        dist.all_gather_into_tensor(out, inp)
    torch.cuda.synchronize()
    nvtx.range_pop()
    _check_output(out, world_size, shard)
    dist.barrier(group=cpu_group)
    if rank == 0:
        logger.info("all-gather output correct after warmup and the %s range", NVTX_RANGE)

    if pool is not None:
        # Release the windows before the pool's memory goes away.
        backend.deregister_mem_pool(pool)
    del inp, out
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
