#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run a standalone NCCL ReduceScatter_LDMC capture with registered symmetric buffers."""

import argparse
import contextlib
import logging
import os

import torch
import torch.distributed as dist
from symm_clock64_csv import Clock64Capture
from torch.cuda import nvtx


logger = logging.getLogger(__name__)

NVTX_RANGE = "ncu_reduce_scatter"
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size-mib", type=int, required=True, help="Reduce-scatter input size per rank in MiB")
    parser.add_argument("--dtype", choices=tuple(DTYPES), required=True)
    parser.add_argument("--warmup", type=int, required=True, help="Reduce-scatters before the NVTX range")
    args = parser.parse_args()
    if args.size_mib < 1 or args.warmup < 1:
        parser.error("--size-mib and --warmup must both be positive")
    return args


def _check_output(out: torch.Tensor, *, rank: int, world_size: int) -> None:
    # Every rank contributes rank % 2 + 1. Rank 0 also adds the destination rank to each shard.
    expected = world_size + world_size // 2 + rank
    if not torch.all(out == expected):
        raise RuntimeError(f"rank {rank}: reduce-scatter shard should contain only {expected}")


def main() -> None:
    """Warm up, profile one reduce-scatter, and check the output after each phase."""
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 2:
        raise ValueError("ReduceScatter_LDMC requires at least two ranks")
    dtype = DTYPES[args.dtype]
    size_bytes = args.size_mib << 20
    elem = torch.empty((), dtype=dtype).element_size()
    shard, remainder = divmod(size_bytes, world_size * elem)
    if remainder:
        raise ValueError(f"{size_bytes} bytes cannot be split into {world_size} equal {args.dtype} shards")

    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    # Eager NCCL initialization lets us register the pool before the first collective.
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    cpu_group = dist.new_group(backend="gloo")

    backend = dist.group.WORLD._get_backend(torch.device("cuda"))
    # ncclMemAlloc supplies the multicast-capable memory for symmetric window registration.
    pool = torch.cuda.MemPool(backend.mem_allocator)
    with torch.cuda.use_mem_pool(pool):
        inp = torch.full((world_size, shard), rank % 2 + 1, dtype=dtype, device=device)
        out = torch.empty(shard, dtype=dtype, device=device)
    if rank == 0:
        inp.add_(torch.arange(world_size, dtype=dtype, device=device)[:, None])
    inp = inp.view(-1)
    torch.cuda.synchronize()
    backend.register_mem_pool(pool, symm=True)
    torch.cuda.synchronize()

    for _ in range(args.warmup):
        dist.reduce_scatter_tensor(out, inp)
    torch.cuda.synchronize()
    _check_output(out, rank=rank, world_size=world_size)

    out.zero_()
    capture = None
    if profile := os.environ.get("NCCL_SYM_CLOCK64_CSV_PROFILE"):
        capture = Clock64Capture(
            profile=profile, device=device, shard_bytes=shard * elem, launches=1, kernels=("ReduceScatter_LDMC",)
        )
    torch.cuda.synchronize()
    dist.barrier(group=cpu_group)
    with capture.arm() if capture is not None else contextlib.nullcontext():
        with nvtx.range(NVTX_RANGE):
            dist.reduce_scatter_tensor(out, inp)
            torch.cuda.synchronize()
    _check_output(out, rank=rank, world_size=world_size)
    if capture is not None:
        capture.write(rank=rank)
    dist.barrier(group=cpu_group)
    if rank == 0:
        logger.info(
            "Reduce-scatter verified on %d ranks, %d MiB %s input per rank", world_size, args.size_mib, args.dtype
        )

    backend.deregister_mem_pool(pool)
    del inp, out
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
