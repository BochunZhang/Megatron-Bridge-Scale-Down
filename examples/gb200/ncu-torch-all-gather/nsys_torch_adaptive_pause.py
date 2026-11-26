# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise NCCL adaptive pause and validate the Nsight Systems capture."""

import argparse
import csv
import logging
import math
import os
import sqlite3
from collections import defaultdict
from contextlib import closing
from functools import partial
from itertools import pairwise
from pathlib import Path


LOGGER = logging.getLogger(__name__)
NUM_GPUS = 4
AG_BYTES = 4 << 30
RS_BYTES = 1 << 30
SAMPLE_PERIOD_NS = 5_000
METRICS = tuple(
    f"NVLink {direction} {packet} {data} Data"
    for direction in ("RX", "TX")
    for packet in ("Requests", "Responses")
    for data in ("User", "Protocol")
)
KERNEL_SQL = """
SELECT DISTINCT d.cudaId, k.deviceId, k.start, k.end
FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
JOIN StringIds AS s ON s.id = k.shortName
JOIN TARGET_INFO_CUDA_DEVICE AS d
  ON d.pid = ((k.globalPid >> 24) & 16777215) AND d.gpuId = k.deviceId
WHERE s.value = :kernel OR s.value GLOB :kernel || '_*'
ORDER BY d.cudaId
"""


def _check_nccl_library() -> None:
    expected = Path(os.environ["ADAPTIVE_PAUSE_NCCL_LIB"]).resolve(strict=True)
    with Path("/proc/self/maps").open() as maps:
        paths = (line.split(maxsplit=5)[-1].strip() for line in maps)
        loaded = {Path(path).resolve() for path in paths if Path(path).name.startswith("libnccl.so")}
    if loaded != {expected}:
        raise RuntimeError(f"torch mapped NCCL libraries {sorted(loaded)}, expected only {expected}")


def benchmark(rs_delay_ms: float) -> None:
    """Capture one AG followed by an overlapping, delayed RS on four GPUs."""
    import torch
    import torch.distributed as dist
    from torch.cuda import nvtx

    _check_nccl_library()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    if dist.get_world_size() != NUM_GPUS:
        raise RuntimeError("launch with four torch workers on one node")
    ranks = list(range(NUM_GPUS))
    ag_pg = dist.new_group(ranks=ranks, backend="nccl", device_id=device)
    rs_pg = dist.new_group(ranks=ranks, backend="nccl", device_id=device)
    gate_pg = dist.new_group(ranks=ranks, backend="nccl", device_id=device)
    ag_backend, rs_backend = (pg._get_backend(device) for pg in (ag_pg, rs_pg))

    dtype = torch.bfloat16
    ag_elts, rs_elts = AG_BYTES // 2, RS_BYTES // 2
    pool = torch.cuda.MemPool(ag_backend.mem_allocator)
    with torch.cuda.use_mem_pool(pool):
        ag_in = torch.full((ag_elts // NUM_GPUS,), rank + 1, device=device, dtype=dtype)
        ag_out = torch.empty(ag_elts, device=device, dtype=dtype)
        rs_in = torch.empty(rs_elts, device=device, dtype=dtype)
        rs_out = torch.empty(rs_elts // NUM_GPUS, device=device, dtype=dtype)
        for dest, shard in enumerate(rs_in.chunk(NUM_GPUS)):
            shard.fill_(rank + 1 + dest)
    torch.cuda.synchronize()
    for backend in (ag_backend, rs_backend):
        backend.register_mem_pool(pool, symm=True)

    ag = partial(dist.all_gather_into_tensor, ag_out, ag_in, group=ag_pg)
    rs = partial(dist.reduce_scatter_tensor, rs_out, rs_in, group=rs_pg)
    gate = torch.zeros(1, device=device)
    # Connect each communicator before any concurrent launch.
    for operation in (ag, rs, partial(dist.all_reduce, gate, group=gate_pg)):
        operation()
        torch.cuda.synchronize()
        dist.barrier()
    ag_out.zero_()
    rs_out.zero_()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda._sleep(1_000_000)
    start.record()
    torch.cuda._sleep(10_000_000)
    end.record()
    end.synchronize()
    cycles_per_ms = 10_000_000 / start.elapsed_time(end)
    ag_stream, rs_stream = torch.cuda.Stream(), torch.cuda.Stream()
    release = torch.cuda.Event()

    dist.barrier()
    if rank == 0:
        torch.cuda.profiler.start()
    dist.barrier()
    with nvtx.range("adaptive_pause_ag4g_rs1g"):
        # Queue both launches behind a common GPU start gate. Only the RS stream has the join delay.
        torch.cuda._sleep(round(5 * cycles_per_ms))
        dist.all_reduce(gate, group=gate_pg)
        release.record()
        with torch.cuda.stream(ag_stream):
            ag_stream.wait_event(release)
            ag_work = ag(async_op=True)
        with torch.cuda.stream(rs_stream):
            rs_stream.wait_event(release)
            torch.cuda._sleep(max(1, round(rs_delay_ms * cycles_per_ms)))
            rs_work = rs(async_op=True)
        torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        torch.cuda.profiler.stop()
    ag_work.wait()
    rs_work.wait()

    ag_ok = all(bool(torch.all(shard == src + 1).item()) for src, shard in enumerate(ag_out.chunk(NUM_GPUS)))
    expected_sum = sum(src + 1 + rank for src in ranks)
    rs_ok = bool(torch.all(rs_out == expected_sum).item())
    checks = [None] * NUM_GPUS
    dist.all_gather_object(checks, (ag_ok, rs_ok))
    for backend in (ag_backend, rs_backend):
        backend.deregister_mem_pool(pool)
    dist.destroy_process_group()
    if not all(ag_ok and rs_ok for ag_ok, rs_ok in checks):
        raise RuntimeError(f"per-rank AG and RS correctness checks failed: {checks}")
    if rank == 0:
        LOGGER.info("Verified 4 GiB AG and 1 GiB RS on all 4 ranks")


def _kernels(conn: sqlite3.Connection, gpus: tuple[int, ...], *, kernel: str) -> list[tuple[int, int]]:
    rows = conn.execute(KERNEL_SQL, {"kernel": kernel}).fetchall()
    if [(rank, gpu) for rank, gpu, *_ in rows] != list(enumerate(gpus)):
        raise ValueError(f"expected one {kernel} per rank on GPUs {gpus}, got {rows}")
    return [(start, end) for _, _, start, end in rows]


def _phases(ag: list[tuple[int, int]], rs: list[tuple[int, int]], *, adaptive: bool) -> dict[str, tuple[int, int]]:
    last_ag_start = max(start for start, _ in ag)
    first_ag_end = min(end for _, end in ag)
    first_rs_start = min(start for start, _ in rs)
    last_rs_start = max(start for start, _ in rs)
    first_rs_end = min(end for _, end in rs)
    last_rs_end = max(end for _, end in rs)
    overlap_end = min(first_ag_end, first_rs_end)
    if not last_ag_start < first_rs_start <= last_rs_start < overlap_end:
        raise ValueError("expected all AG kernels to start before RS and all GPUs to overlap")
    if adaptive and last_rs_end >= first_ag_end:
        raise ValueError("adaptive capture failed: the last RS must finish before the first AG finishes")
    phases = {"ag_before_rs": (last_ag_start, first_rs_start), "ag_rs_overlap": (last_rs_start, overlap_end)}
    if last_rs_end < first_ag_end:
        phases["ag_after_rs"] = last_rs_end, first_ag_end
    elif max(end for _, end in ag) < first_rs_end:
        phases["rs_after_ag"] = max(end for _, end in ag), first_rs_end
    return phases


def _phase_rate(samples: list[tuple[int, float]], *, start: int, end: int) -> tuple[float, int, int]:
    if not samples or samples[0][0] > start or samples[-1][0] < end:
        raise ValueError("samples do not bracket the phase")
    byte_count = duration = sample_count = longest = 0
    for (previous, _), (timestamp, value) in pairwise(samples):
        if timestamp <= start or previous >= end:
            continue
        interval = timestamp - previous
        if not 0 < interval <= 1.5 * SAMPLE_PERIOD_NS:
            raise ValueError(f"invalid or missing NVLink sample interval of {interval} ns")
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid NVLink byte count {value!r} at {timestamp} ns")
        longest = max(longest, interval)
        # Exclude samples straddling a change in which kernels are active.
        if previous >= start and timestamp <= end:
            byte_count += value
            duration += interval
            sample_count += 1
    if sample_count < 2:
        raise ValueError("phase contains fewer than two full NVLink samples")
    return byte_count / duration, sample_count, longest


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[tuple]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def analyse(mode_dir: Path, gpus: tuple[int, ...], *, adaptive: bool) -> None:
    """Validate kernel order and summarize every NVLink byte counter for each common phase."""
    database = mode_dir / "profile.sqlite"
    with closing(sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)) as conn:
        ag = _kernels(conn, gpus, kernel="ncclSymkDevKernel_AllGather_STMC")
        rs = _kernels(conn, gpus, kernel="ncclSymkDevKernel_ReduceScatter_LDMC")
        _write_csv(
            mode_dir / "kernel_timing.csv",
            ("rank", "gpu", "ag_start_ns", "ag_end_ns", "rs_start_ns", "rs_end_ns"),
            [(rank, gpu, *ag[rank], *rs[rank]) for rank, gpu in enumerate(gpus)],
        )
        phases = _phases(ag, rs, adaptive=adaptive)
        metric_ids = defaultdict(list)
        for type_id, metric_id, name in conn.execute(
            "SELECT DISTINCT typeId, metricId, metricName FROM TARGET_INFO_GPU_METRICS"
        ):
            name = name.removesuffix(" [Bytes]").removesuffix(" Bytes")
            metric_ids[type_id & 255, name].append((type_id, metric_id))

        window_start = min(start for start, _ in phases.values()) - 2 * SAMPLE_PERIOD_NS
        window_end = max(end for _, end in phases.values()) + 2 * SAMPLE_PERIOD_NS
        samples = defaultdict(list)
        for type_id, metric_id, timestamp, value in conn.execute(
            "SELECT typeId, metricId, timestamp, value FROM GPU_METRICS "
            "WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp",
            (window_start, window_end),
        ):
            samples[type_id, metric_id].append((timestamp, value))

    rows = []
    for rank, gpu in enumerate(gpus):
        for name in METRICS:
            ids = metric_ids[gpu, name]
            if len(ids) != 1:
                raise ValueError(f"GPU {gpu}: expected one bytes metric for {name!r}, got {ids}")
            for phase, (start, end) in phases.items():
                try:
                    rate, count, longest = _phase_rate(samples[ids[0]], start=start, end=end)
                except ValueError as exc:
                    raise ValueError(f"GPU {gpu}, {phase}, {name}: {exc}") from exc
                rows.append((rank, gpu, phase, start, end, name, round(rate, 3), count, longest))
    _write_csv(
        mode_dir / "nvlink_by_phase.csv",
        ("rank", "gpu", "phase", "start_ns", "end_ns", "metric", "GB/s", "samples", "max_sample_gap_ns"),
        rows,
    )
    LOGGER.info("%s phase spans (ms): %s", mode_dir, {p: round((e - s) / 1e6, 3) for p, (s, e) in phases.items()})


def main() -> None:
    """Run the CUDA worker or analyse both CPU-side SQLite exports."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    worker = commands.add_parser("benchmark")
    worker.add_argument("--rs-delay-ms", type=float, required=True)
    analysis = commands.add_parser("analyse")
    analysis.add_argument("--output-dir", type=Path, required=True)
    analysis.add_argument("--gpus", required=True)
    args = parser.parse_args()
    if args.command == "benchmark":
        if not math.isfinite(args.rs_delay_ms) or args.rs_delay_ms <= 0:
            parser.error("--rs-delay-ms must be finite and positive")
        benchmark(args.rs_delay_ms)
    else:
        gpus = tuple(int(index) for index in args.gpus.split(","))
        for mode in ("adaptive", "control"):
            mode_dir = args.output_dir / mode
            try:
                analyse(mode_dir, gpus, adaptive=(mode == "adaptive"))
            except (sqlite3.Error, ValueError) as exc:
                parser.exit(1, f"{mode_dir}: {exc}\n")


if __name__ == "__main__":
    main()
