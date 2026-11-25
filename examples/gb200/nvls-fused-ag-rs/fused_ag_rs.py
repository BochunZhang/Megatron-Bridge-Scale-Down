# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare one fused NCCL launch with two concurrent PyTorch collectives."""

from __future__ import annotations

import argparse
import ctypes
import json
import logging
import os
import time
from collections.abc import Callable
from contextlib import ExitStack, nullcontext
from pathlib import Path
from statistics import median

import torch
import torch.distributed as dist


logger = logging.getLogger(__name__)
# Mirrors ncclDevFusedAgRsVariant_t in the patched NCCL Device API.
FUSED_VARIANTS = {"ag-first": 0, "rs-first": 1, "warp-split": 2, "warp-split-pause": 3}


def _loaded_nccl(expected: Path) -> ctypes.CDLL:
    # The raw communicator can only be used by the same library that created it.
    loaded = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and Path(fields[5]).name.startswith("libnccl.so"):
            loaded.add(Path(fields[5]).resolve())
    if loaded != {expected.resolve()}:
        raise RuntimeError(f"Expected only {expected.resolve()} in PyTorch, found {sorted(loaded)}")
    library = ctypes.CDLL(str(expected.resolve()), mode=os.RTLD_NOW | os.RTLD_NOLOAD)
    library.ncclGetErrorString.argtypes = [ctypes.c_int]
    library.ncclGetErrorString.restype = ctypes.c_char_p
    # The presence of the experimental API also catches a stale patch/build.
    if not hasattr(library, "ncclDevFusedAllGatherReduceScatterWithConfig"):
        raise RuntimeError(f"{expected} does not export ncclDevFusedAllGatherReduceScatterWithConfig")
    return library


class _FusedPair:
    def __init__(
        self,
        shim_path: Path,
        nccl: ctypes.CDLL,
        comm: int,
        *,
        ag_out: torch.Tensor,
        rs_in: torch.Tensor,
        ctas: int,
        variant: str,
        ag_pause_cycles: int,
    ) -> None:
        self._nccl = nccl
        self._variant = FUSED_VARIANTS[variant]
        self._ag_pause_cycles = ag_pause_cycles
        self._shim = ctypes.CDLL(str(shim_path.resolve()))
        ptr, size = ctypes.c_void_p, ctypes.c_size_t
        self._shim.fused_ag_rs_create.argtypes = [ptr, ptr, size, ptr, size, ctypes.c_int, ctypes.POINTER(ptr)]
        self._shim.fused_ag_rs_launch.argtypes = [ptr, ptr, size, ptr, size, ctypes.c_int, ctypes.c_uint32, ptr]
        self._shim.fused_ag_rs_destroy.argtypes = [ptr]
        self._state = ptr()
        self._check(
            self._shim.fused_ag_rs_create(
                comm, ag_out.data_ptr(), ag_out.nbytes, rs_in.data_ptr(), rs_in.nbytes, ctas, ctypes.byref(self._state)
            ),
            "register symmetric windows and create device communicator",
        )

    def _check(self, result: int, operation: str) -> None:
        if result != 0:
            message = self._nccl.ncclGetErrorString(result)
            detail = message.decode() if message else "unknown error"
            raise RuntimeError(f"{operation} failed (NCCL {result}: {detail})")

    def launch(self, *, ag_in: torch.Tensor, rs_out: torch.Tensor, stream: torch.cuda.Stream) -> None:
        self._check(
            self._shim.fused_ag_rs_launch(
                self._state,
                ag_in.data_ptr(),
                ag_in.nbytes,
                rs_out.data_ptr(),
                rs_out.numel(),
                self._variant,
                self._ag_pause_cycles,
                stream.cuda_stream,
            ),
            "launch fused AG/RS",
        )

    def close(self) -> None:
        self._check(self._shim.fused_ag_rs_destroy(self._state), "destroy device communicator and symmetric windows")


def _new_group(device: torch.device, cleanup: ExitStack) -> dist.ProcessGroup:
    options = dist.ProcessGroupNCCL.Options()
    # The fused call has no PyTorch Work object to poll a nonblocking NCCL communicator.
    options.config.blocking = 1
    group = dist.new_group(backend="nccl", device_id=device, pg_options=options)
    cleanup.callback(dist.destroy_process_group, group)
    return group


def _symmetric_tensor(
    backend: dist.ProcessGroupNCCL, count: int, *, device: torch.device
) -> tuple[torch.Tensor, torch.cuda.MemPool]:
    pool = torch.cuda.MemPool(backend.mem_allocator)
    with torch.cuda.use_mem_pool(pool):
        tensor = torch.empty(count, device=device, dtype=torch.bfloat16)
    return tensor, pool


def _run_pairs(
    control: torch.cuda.Stream, operations: list[tuple[torch.cuda.Stream, Callable[[], None]]], count: int
) -> list[float]:
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(count)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(count)]
    completions = [[torch.cuda.Event() for _ in operations] for _ in range(count)]
    for start, end, done in zip(starts, ends, completions, strict=True):
        start.record(control)
        for (stream, launch), completion in zip(operations, done, strict=True):
            with torch.cuda.stream(stream):
                stream.wait_event(start)
                launch()
                completion.record()
            control.wait_event(completion)
        # The next pair starts only after both outputs from this pair are usable.
        end.record(control)
    control.synchronize()
    return [start.elapsed_time(end) * 1_000 for start, end in zip(starts, ends, strict=True)]


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fused", "split"), required=True)
    parser.add_argument("--ag-mib", type=int, required=True, help="Full AG output per GPU")
    parser.add_argument("--rs-mib", type=int, required=True, help="Full RS input per GPU")
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--iters", type=int, required=True)
    parser.add_argument("--ctas", type=int, required=True, help="Fused grid size")
    parser.add_argument("--variant", choices=FUSED_VARIANTS, default="ag-first")
    parser.add_argument("--ag-pause-cycles", type=int, default=0, help="Cycles per AG warp tile in warp-split-pause")
    parser.add_argument("--nccl-lib", type=Path, required=True)
    parser.add_argument("--shim", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-frequency", type=int, default=50_000, help="Nsight GPU metrics sampling rate in Hz")
    args = parser.parse_args()
    if min(args.ag_mib, args.rs_mib, args.warmup, args.iters) < 1 or not 1 <= args.ctas <= 64:
        parser.error("sizes and iteration counts must be positive, and CTAs must be 1–64")
    if not 10 <= args.profile_frequency <= 200_000:
        parser.error("profile frequency must be 10–200000 Hz")
    if not 0 <= args.ag_pause_cycles <= 0xFFFFFFFF:
        parser.error("AG pause cycles must fit an unsigned 32-bit integer")
    if args.ag_pause_cycles and (args.mode != "fused" or args.variant != "warp-split-pause"):
        parser.error("nonzero AG pause cycles require fused warp-split-pause")
    return args


def main() -> None:
    """Run and validate the selected collective pair on each torchrun rank."""
    args = _args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (10, 0):
        raise RuntimeError("Run with GB200 sm_100 GPUs to match the patched NCCL build")
    nccl = _loaded_nccl(args.nccl_lib)

    with ExitStack() as cleanup:
        dist.init_process_group(backend="gloo")
        cleanup.callback(dist.destroy_process_group)
        rank, world = dist.get_rank(), dist.get_world_size()
        ag_bytes, rs_bytes = args.ag_mib << 20, args.rs_mib << 20
        if not 2 <= world <= 64 or ag_bytes % (world * 16) or rs_bytes % (world * 16):
            raise ValueError("Use 2–64 GPUs and AG/RS sizes that split into 16-byte-aligned shards")
        ag_count, rs_count = ag_bytes // 2, rs_bytes // 2

        ag_group = _new_group(device, cleanup)
        ag_backend = ag_group._get_backend(device)
        rs_group = _new_group(device, cleanup) if args.mode == "split" else ag_group
        rs_backend = rs_group._get_backend(device)
        # Distinct pools make each tensor the first allocation in its own NCCL-backed segment.
        ag_out, ag_pool = _symmetric_tensor(ag_backend, ag_count, device=device)
        rs_in, rs_pool = _symmetric_tensor(rs_backend, rs_count, device=device)
        ag_in = torch.empty(ag_count // world, device=device, dtype=torch.bfloat16)
        rs_out = torch.empty(rs_count // world, device=device, dtype=torch.bfloat16)
        torch.cuda.synchronize(device)

        ag_stream = torch.cuda.Stream(device=device)
        if args.mode == "fused":
            comm = ag_backend._comm_ptr()
            if not comm:
                raise RuntimeError("PyTorch did not initialize the NCCL communicator")
            fused = _FusedPair(
                args.shim,
                nccl,
                comm,
                ag_out=ag_out,
                rs_in=rs_in,
                ctas=args.ctas,
                variant=args.variant,
                ag_pause_cycles=args.ag_pause_cycles,
            )
            cleanup.callback(fused.close)
            operations = [(ag_stream, lambda: fused.launch(ag_in=ag_in, rs_out=rs_out, stream=ag_stream))]
        else:
            ag_backend.register_mem_pool(ag_pool, symm=True)
            cleanup.callback(ag_backend.deregister_mem_pool, ag_pool)
            rs_backend.register_mem_pool(rs_pool, symm=True)
            cleanup.callback(rs_backend.deregister_mem_pool, rs_pool)
            operations = [
                (ag_stream, lambda: dist.all_gather_into_tensor(ag_out, ag_in, group=ag_group)),
                (
                    torch.cuda.Stream(device=device),
                    lambda: dist.reduce_scatter_tensor(rs_out, rs_in, op=dist.ReduceOp.SUM, group=rs_group),
                ),
            ]
        cleanup.callback(torch.cuda.synchronize, device)
        control = torch.cuda.Stream(device=device)

        def prepare(epoch: int) -> None:
            ag_in.fill_(rank + epoch)
            for owner, shard in enumerate(rs_in.chunk(world)):
                shard.fill_(rank % 2 + epoch + owner % 2)
            ag_out.zero_()
            rs_out.zero_()
            torch.cuda.synchronize(device)

        def validate(epoch: int) -> None:
            for peer, shard in enumerate(ag_out.chunk(world)):
                if not shard.eq(peer + epoch).all().item():
                    raise RuntimeError(f"rank {rank}: AG output mismatch for peer {peer}")
            expected_rs = world * (epoch + rank % 2) + world // 2
            if not rs_out.eq(expected_rs).all().item():
                raise RuntimeError(f"rank {rank}: RS output mismatch, expected {expected_rs}")

        prepare(1)
        dist.barrier()
        _run_pairs(control, operations, args.warmup)
        validate(1)
        prepare(2)
        dist.barrier()
        if args.profile and rank == 0:
            torch.cuda.profiler.start()
            # Byte counters describe sampling intervals. Collect boundary samples outside the kernels.
            time.sleep(2 / args.profile_frequency)
        dist.barrier()
        run_name = args.variant if args.mode == "fused" else "split"
        with torch.cuda.nvtx.range(f"fused_ag_rs/{run_name}/timed") if args.profile else nullcontext():
            local_us = _run_pairs(control, operations, args.iters)
        dist.barrier()
        if args.profile:
            if rank == 0:
                time.sleep(2 / args.profile_frequency)
                torch.cuda.profiler.stop()
            dist.barrier()
        validate(2)
        all_rank_us: list[list[float]] = [[] for _ in range(world)]
        dist.all_gather_object(all_rank_us, local_us)
        if rank == 0:
            slowest_rank_us = [max(row) for row in zip(*all_rank_us, strict=True)]
            report = {
                "mode": args.mode,
                "profiled": args.profile,
                "torch_version": torch.__version__,
                "nccl_version": torch.cuda.nccl.version(),
                "ag_output_mib": args.ag_mib,
                "rs_input_mib": args.rs_mib,
                "fused_ctas": args.ctas if args.mode == "fused" else None,
                "fused_variant": args.variant if args.mode == "fused" else None,
                "ag_pause_cycles": args.ag_pause_cycles,
                "all_rank_pair_us": all_rank_us,
                "median_slowest_rank_us": median(slowest_rank_us),
                "validation": "passed",
            }
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            logger.info(
                "%s: AG and RS correct on %d ranks, median slowest-rank pair %.3f us",
                run_name,
                world,
                report["median_slowest_rank_us"],
            )


if __name__ == "__main__":
    main()
