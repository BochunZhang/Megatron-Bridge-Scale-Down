#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Compare GPU, staged CPU, and direct CPU inputs for NCCL multicast AllGather."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from statistics import median

import torch
import torch.distributed as dist
from torch.cuda import nvtx


logger = logging.getLogger(__name__)


def _run_case(args: argparse.Namespace, device: torch.device, cleanup: ExitStack) -> None:
    gb200_dir = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(gb200_dir / name) for name in ("hbm-conflict", "benchmark")]
    from ce_multicast import NcclCopyEngineAllGather
    from mbench.gate import LaunchGate

    rank, world_size = dist.get_rank(), dist.get_world_size()
    shard_bytes = args.shard_mib << 20
    output_bytes = shard_bytes * world_size
    staged = args.case == "case2"
    all_gather = NcclCopyEngineAllGather(
        device, rank, world_size, output_bytes, host_input=args.case != "case1", stage_host_input=staged
    )
    cleanup.callback(all_gather.close)
    stream = torch.cuda.Stream(device=device)
    dist.barrier()
    with torch.cuda.stream(stream):
        for _ in range(args.warmup):
            if staged:
                all_gather.stage_input()
            all_gather.run()
        all_gather.reset_output()

    # Initialize CUDA event handles before capture and before blocking the stream.
    events = [
        tuple(torch.cuda.Event(enable_timing=True) for _ in range(3 if staged else 2)) for _ in range(args.iters)
    ]
    for round_events in events:
        for event in round_events:
            event.record(stream)
    stream.synchronize()

    gate = LaunchGate(timeout_s=30.0)
    cleanup.callback(gate.close)
    dist.barrier()
    if args.profile and rank == 0:
        torch.cuda.profiler.start()
    try:
        dist.barrier()
        with torch.cuda.stream(stream), nvtx.range(f"cpu-allgather/{args.case}/rank-{rank}"):
            gate.arm(stream)
            try:
                for iteration, round_events in enumerate(events):
                    with nvtx.range(f"iteration-{iteration}"):
                        round_events[0].record(stream)
                        if staged:
                            with nvtx.range("h2d_stage"):
                                all_gather.stage_input()
                            round_events[1].record(stream)
                        with nvtx.range("multicast_allgather"):
                            all_gather.run()
                        round_events[-1].record(stream)
                dist.barrier()
            finally:
                gate.release()
            stream.synchronize()
        dist.barrier()
    finally:
        if args.profile and rank == 0:
            torch.cuda.profiler.stop()

    try:
        all_gather.validate()
    except RuntimeError as exc:
        logger.error("Rank %d: %s", rank, exc)
        output_correct = False
    else:
        output_correct = True

    rank_result = {
        "rank": rank,
        "gpu": torch.cuda.get_device_name(device),
        "nccl_version": all_gather.version,
        "output_correct": output_correct,
        "watchdog_fired": gate.watchdog_fired(),
        "total_ms": [row[0].elapsed_time(row[-1]) for row in events],
        "allgather_ms": [row[-2].elapsed_time(row[-1]) for row in events],
        "staging_ms": [row[0].elapsed_time(row[1]) for row in events] if staged else None,
    }
    ranks = [None] * world_size if rank == 0 else None
    dist.gather_object(rank_result, object_gather_list=ranks, dst=0)
    if rank == 0:
        assert ranks is not None
        output_correct = all(row["output_correct"] for row in ranks)
        timings_valid = not any(row["watchdog_fired"] for row in ranks)
        result = {
            "case": args.case,
            "world_size": world_size,
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "shard_bytes": shard_bytes,
            "output_bytes": output_bytes,
            "warmup": args.warmup,
            "iterations": args.iters,
            "profiled": args.profile,
            "output_correct": output_correct,
            "timings_valid": timings_valid,
            "ranks": ranks,
        }
        if output_correct and timings_valid:
            for phase in ("staging_ms", "allgather_ms", "total_ms"):
                if phase != "staging_ms" or staged:
                    rank_times = [row[phase] for row in ranks]
                    slowest_rank_ms = [max(sample) for sample in zip(*rank_times, strict=True)]
                    phase_ms = median(slowest_rank_ms)
                    result[f"median_max_rank_{phase}"] = phase_ms
                    logger.info("%s %s (median of per-round max rank): %.4f ms", args.case, phase, phase_ms)
            total_ms = result["median_max_rank_total_ms"]
            result["output_gb_s"] = output_bytes / total_ms / 1e6
            logger.info("%s effective output throughput: %.2f GB/s", args.case, result["output_gb_s"])
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output_dir / "results.json"
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if not output_correct or not timings_valid:
            raise RuntimeError(f"Output validation or CUDA launch gate failed. See {output_path}")


def main() -> None:
    """Run a single source placement with one local worker per CUDA GPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("case1", "case2", "case3"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-mib", type=int, default=256, help="Input MiB per rank")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--profile", action="store_true", help="Capture timed rounds with Nsight Systems")
    args = parser.parse_args()
    if args.shard_mib <= 0 or args.iters <= 0 or args.warmup < 0:
        parser.error("--shard-mib and --iters must be positive, and --warmup must be nonnegative")
    logging.basicConfig(level=logging.INFO)
    if os.environ["LOCAL_WORLD_SIZE"] != os.environ["WORLD_SIZE"]:
        raise ValueError("The multicast benchmark expects one torchrun host")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("gloo")
    try:
        with ExitStack() as cleanup:
            _run_case(args, device, cleanup)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
