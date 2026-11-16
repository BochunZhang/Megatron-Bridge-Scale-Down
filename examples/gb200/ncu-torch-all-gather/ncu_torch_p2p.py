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

"""Send/recv worker for an Nsight Compute capture of one NCCL P2P direction.

Rank 0 sends to rank 1. The launcher selects push or pull via NCCL_P2P_READ_ENABLE.
Only the profiled SendRecv calls are inside the ncu_p2p NVTX range. Gloo carries
the CPU barrier and correctness checks outside the range.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.cuda import nvtx


# The separately runnable NVLink benchmarks share these NCCL log checks.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nvls-collective"))
from nvls_collective_benchmark import NcclLog, check_p2p_connections  # noqa: E402
from p2p_push_pull_benchmark import (  # noqa: E402
    DTYPES,
    PAYLOAD,
    RECEIVER,
    SENDER,
    WORKLOADS,
    check_environment,
    physical_gpu,
)


logger = logging.getLogger(__name__)
NVTX_RANGE = "ncu_p2p"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=WORKLOADS, required=True)
    parser.add_argument("--size-mib", type=int, required=True)
    parser.add_argument("--dtype", choices=tuple(DTYPES), required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--iters", type=int, required=True)
    args = parser.parse_args()
    if args.size_mib <= 0 or args.warmup < 0 or args.iters <= 0:
        parser.error("--size-mib and --iters must be positive, and --warmup must be nonnegative")
    return args


def _transfer(buffer: torch.Tensor, *, rank: int) -> None:
    if rank == SENDER:
        work = dist.isend(buffer, dst=RECEIVER)
    else:
        work = dist.irecv(buffer, src=SENDER)
    work.wait()


def _check_payload(buffer: torch.Tensor, *, rank: int, cpu_group: dist.ProcessGroup, phase: str) -> None:
    correct = [rank == RECEIVER and bool(torch.all(buffer == PAYLOAD).item())]
    dist.broadcast_object_list(correct, src=RECEIVER, group=cpu_group)
    if not correct[0]:
        raise RuntimeError(f"rank {RECEIVER} received incorrect P2P data after {phase}")
    logger.info("rank %d P2P payload verified after %s", rank, phase)


def main() -> None:
    """Warm up and profile a P2P transfer, checking transport and payload."""
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    logger.info(
        "%s rank %d of %d pid=%d starting on GPU%d",
        args.workload,
        rank,
        world_size,
        os.getpid(),
        physical_gpu(local_rank),
    )
    check_environment(args.workload, world_size)

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    cpu_group = dist.new_group(backend="gloo")

    # Connect the P2P channels and verify the sender's NCCL connection log on both ranks.
    nccl_log = NcclLog()
    _transfer(torch.zeros(1024, device=device), rank=rank)
    torch.cuda.synchronize(device)
    verdict: list[tuple[list[str], list[str]] | None] = [None]
    if rank == SENDER:
        transports, errors = check_p2p_connections(nccl_log.consume(), mode=args.workload)
        if not transports:
            errors.append(f"no P2P connection to rank {RECEIVER} in {nccl_log.path}")
        verdict[0] = (transports, errors)
    dist.broadcast_object_list(verdict, src=SENDER, group=cpu_group)
    if verdict[0] is None:
        raise RuntimeError(f"missing {args.workload} P2P transport verdict")
    transports, errors = verdict[0]
    if errors:
        raise RuntimeError(f"{args.workload} P2P transport check failed: {', '.join(errors)}")
    logger.info("%s rank %d P2P transports %s", args.workload, rank, transports)

    dtype = DTYPES[args.dtype]
    numel = (args.size_mib << 20) // torch.empty((), dtype=dtype).element_size()
    fill = PAYLOAD if rank == SENDER else 0.0
    buffer = torch.full((numel,), fill, dtype=dtype, device=device)
    for _ in range(args.warmup):
        _transfer(buffer, rank=rank)
    torch.cuda.synchronize(device)
    if args.warmup:
        _check_payload(buffer, rank=rank, cpu_group=cpu_group, phase="warmup")
    if rank == RECEIVER:
        buffer.zero_()
    torch.cuda.synchronize(device)
    dist.barrier(group=cpu_group)

    with nvtx.range(NVTX_RANGE):
        for _ in range(args.iters):
            _transfer(buffer, rank=rank)
        torch.cuda.synchronize(device)
    _check_payload(buffer, rank=rank, cpu_group=cpu_group, phase=NVTX_RANGE)
    dist.barrier(group=cpu_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
