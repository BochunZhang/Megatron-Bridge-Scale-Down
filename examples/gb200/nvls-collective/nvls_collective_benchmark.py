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
"""Measure torch all-gather / reduce-scatter throughput with ring or NVLS on GB200.

One launch measures one algorithm family, because ``NCCL_ALGO`` is fixed when a
communicator is created and the symmetric-memory path needs its own buffer registration:

* ``ring`` / ``nvls``: ``NCCL_ALGO`` is forced and payload buffers are plain device memory.
* ``symm``: ``NCCL_WIN_ENABLE`` / ``NCCL_SYM_KERNEL`` are forced, ``NCCL_ALGO`` is left
  unset, and every buffer is allocated inside a ``torch.cuda.MemPool`` registered as a
  ``NCCL_WIN_COLL_SYMMETRIC`` window. NCCL then dispatches to its symmetric kernels
  (``AllGather_STMC`` / ``ReduceScatter_LDMC`` on a NVLink domain), which are unrelated to
  the classic ring and NVLS collectives.

``run_nvls_collective_benchmark.sh`` launches one pass per family and merges them with
``collective_metrics.py``.

Scenarios (all inside one torchrun job; ranks outside a group wait at a Gloo barrier):

* ``single``: ranks ``[0, n)`` for every ``n`` in ``--group-sizes`` run AG alone, RS alone,
  and AG + RS at the same time on two streams and two communicators.
* ``split``: ranks ``[0, T)`` for every ``T`` in ``--split-totals`` are cut into two groups of
  ``T/2``; each group runs alone, then both run at the same time.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import logging
import os
import socket
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from statistics import median

import torch
import torch.cuda.profiler
import torch.distributed as dist
from collective_metrics import (
    BYTES_PER_MIB,
    NCCL_ALGO_ENV,
    SPLIT_LAYOUTS,
    WORKLOADS,
    alg_bandwidth_gb_s,
    bus_bandwidth_gb_s,
    split_groups,
    workload_ops,
)
from torch.cuda import nvtx


LOGGER = logging.getLogger(__name__)
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
# Which symmetric kernels are eligible per collective; None leaves NCCL_SYM_KERNEL unset
# so the user/default kernel mask applies.
SYMM_KERNELS = {
    "ag": ["AllGather_ST", "AllGather_LL", "AllGather_STMC", "AllGather_LLMC"],
    "rs": ["ReduceScatter_LD", "ReduceScatter_LL", "ReduceScatter_LDMC"],
}
PROBE_FIELDS = ("nccl_algo", "nccl_kernel", "n_nodes", "mnnvl")

SLEEP_CALIBRATION_CYCLES = 10_000_000


@dataclasses.dataclass
class CollectiveGroup:
    """One rank set with a dedicated communicator per collective so AG and RS can overlap."""

    name: str
    ranks: list[int]
    ag_pg: dist.ProcessGroup | None
    rs_pg: dist.ProcessGroup | None

    @property
    def size(self) -> int:
        """Number of ranks in the group."""
        return len(self.ranks)


@dataclasses.dataclass
class Buffers:
    """Pre-filled inputs and outputs for one group/size; inputs are never overwritten."""

    ag_in: torch.Tensor
    ag_out: torch.Tensor
    rs_in: torch.Tensor
    rs_out: torch.Tensor
    pool: torch.cuda.MemPool | None = None


class Topology:
    """The collective groups of one job, in rank order, by name.

    ``comm_info`` holds this rank's ``nNodes``/``MNNVL`` as NCCL logged them when the group's
    communicators were created; it is empty on ranks outside every group.
    """

    def __init__(self, groups: list[CollectiveGroup], rank: int, comm_info: dict[str, object]) -> None:
        self.rank = rank
        self.groups = groups
        self.group_by_rank = {member: group for group in groups for member in group.ranks}
        self.comm_info = comm_info

    @property
    def mine(self) -> CollectiveGroup | None:
        """The group ``rank`` participates in, or ``None`` if it sits this scenario out."""
        return self.group_by_rank.get(self.rank)


def _csv_ints(value: str) -> list[int]:
    try:
        parsed = list(dict.fromkeys(int(item) for item in value.split(",") if item))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers, got {value!r}") from error
    if not parsed or min(parsed) < 1:
        raise argparse.ArgumentTypeError("all entries must be positive")
    return parsed


def _csv_ints_or_none(value: str) -> list[int]:
    return [] if value == "none" else _csv_ints(value)


def _csv_choices(choices: tuple[str, ...]) -> Callable[[str], list[str]]:
    def parse(value: str) -> list[str]:
        parsed = list(dict.fromkeys(item for item in value.split(",") if item))
        if not parsed or set(parsed) - set(choices):
            raise argparse.ArgumentTypeError(f"expected comma-separated values from {choices}, got {value!r}")
        return parsed

    return parse


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--algo",
        choices=sorted(NCCL_ALGO_ENV),
        required=True,
        help="Algorithm family; must match the NCCL_ALGO/window environment set by the launcher",
    )
    parser.add_argument(
        "--symm-kernel",
        choices=["default", "ag", "rs"],
        default="default",
        help="For --algo symm: assert the pinned NCCL_SYM_KERNEL matches SYMM_KERNELS[...]",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group-sizes", type=_csv_ints, default=[4, 8, 16, 32], help="Single-group sizes")
    parser.add_argument(
        "--split-totals",
        type=_csv_ints_or_none,
        default=[16],
        help="Ranks shared by two concurrent half-groups; 'none' skips the split scenario",
    )
    parser.add_argument("--split-layouts", type=_csv_choices(SPLIT_LAYOUTS), default=["contiguous"])
    parser.add_argument("--workloads", type=_csv_choices(WORKLOADS), default=list(WORKLOADS))
    parser.add_argument(
        "--sizes-mib", type=_csv_ints, default=[16, 64, 256, 1024], help="Full tensor: AG output / RS input per rank"
    )
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--warmup", type=int, default=5, help="Untimed iterations before every trial set")
    parser.add_argument("--iters", type=int, default=20, help="Back-to-back iterations per timed trial")
    parser.add_argument("--trials", type=int, default=5, help="Timed trials; the median is reported")
    parser.add_argument(
        "--launch-delay-ms",
        type=float,
        default=5.0,
        help="GPU spin before the timed region so all launches are queued before timing starts",
    )
    parser.add_argument(
        "--nsys",
        action="store_true",
        help="Wrap every timed case in an NVTX range and a cudaProfilerStart/Stop capture window",
    )
    args = parser.parse_args(argv)
    if min(args.warmup + 1, args.iters, args.trials) < 1:
        parser.error("--warmup must be >= 0, --iters and --trials must be >= 1")
    if args.launch_delay_ms < 0:
        parser.error("--launch-delay-ms must be >= 0")
    return args


def _check_environment(algo: str, world_size: int, args: argparse.Namespace) -> None:
    expected = NCCL_ALGO_ENV[algo]
    actual = os.environ.get("NCCL_ALGO", "").strip()
    nvls_enable = os.environ.get("NCCL_NVLS_ENABLE")
    if algo == "symm":
        # This path is selected by window registration, not by NCCL_ALGO; forcing an algorithm
        # would hide exactly the dispatch this pass is meant to observe.
        if actual:
            raise RuntimeError(f"--algo symm requires NCCL_ALGO to be unset, got {actual!r}")
        if os.environ.get("NCCL_WIN_ENABLE") != "1":
            raise RuntimeError("--algo symm requires NCCL_WIN_ENABLE=1 to register symmetric windows")
        pinned = os.environ.get("NCCL_SYM_KERNEL")
        if args.symm_kernel != "default" and pinned not in SYMM_KERNELS[args.symm_kernel]:
            raise RuntimeError(
                f"--symm-kernel {args.symm_kernel} requires NCCL_SYM_KERNEL in"
                f" {SYMM_KERNELS[args.symm_kernel]}, got {pinned!r}"
            )
    else:
        assert expected is not None
        if actual.lower() != expected.lower():
            raise RuntimeError(f"--algo {algo} requires NCCL_ALGO={expected}, got {actual!r}")
        if algo == "nvls" and nvls_enable != "1":
            raise RuntimeError("--algo nvls requires NCCL_NVLS_ENABLE=1 so a missing NVLS path fails loudly")
        if algo == "ring" and nvls_enable != "0":
            raise RuntimeError("--algo ring requires NCCL_NVLS_ENABLE=0 so no multicast buffers are allocated")
    too_large = [n for n in args.group_sizes + args.split_totals if n > world_size]
    if too_large:
        raise ValueError(f"group sizes {too_large} exceed WORLD_SIZE={world_size}")
    bad_split = [n for n in args.split_totals if n % 2 or n < 4]
    if bad_split:
        raise ValueError(f"--split-totals must be even and >= 4, got {bad_split}")
    if any(n < 2 for n in args.group_sizes):
        raise ValueError("--group-sizes entries must be >= 2")


def _new_group(name: str, ranks: list[int]) -> CollectiveGroup:
    # new_group is collective over the default (Gloo) group: every rank calls it in the same order.
    ag_pg = dist.new_group(ranks=ranks, backend="nccl")
    rs_pg = dist.new_group(ranks=ranks, backend="nccl")
    member = dist.get_rank() in ranks
    return CollectiveGroup(name=name, ranks=ranks, ag_pg=ag_pg if member else None, rs_pg=rs_pg if member else None)


def nccl_debug_path(template: str, *, hostname: str, pid: int) -> Path:
    """Expand ``NCCL_DEBUG_FILE`` the way NCCL's ``debug.cc`` does.

    ``%h`` is the host name cut at the first ``.`` (``getHostName(..., '.')``), ``%p`` is the pid,
    ``%%`` is a literal ``%``; anything else after ``%`` is echoed unchanged.
    """
    out: list[str] = []
    chars = iter(template)
    for char in chars:
        if char != "%":
            out.append(char)
            continue
        code = next(chars, "")
        if code == "h":
            out.append(hostname.split(".", 1)[0])
        elif code == "p":
            out.append(str(pid))
        elif code == "%":
            out.append("%")
        else:
            out.append("%" + code)
    return Path("".join(out))


def parse_nccl_lines(lines: list[str]) -> dict[str, object]:
    """Pull the executed algorithm/kernel and the communicator topology out of NCCL log lines.

    * ``<Func>: <n> Bytes -> Algo <algo> proto <proto> ...`` (``enqueue.cc``, classic path).
    * ``<Func> [Symmetric]: <n> Bytes -> Kernel <name> ...`` (``scheduler/symmetric_sched.cc``).
    * ``comm <p> rank <r> nRanks <n> nNodes <n> localRanks <n> localRank <r> MNNVL <f>`` (``init.cc``).

    The first two are ``INFO(NCCL_TUNING, ...)`` lines emitted by comm rank 0 only, so the
    launcher must include ``TUNING`` in ``NCCL_DEBUG_SUBSYS``. Kernels are reported per function,
    e.g. ``AllGather=NVLS/Simple;ReduceScatter=NVLS/Simple``, so an AG+RS case shows both.
    """
    algos: dict[str, str] = {}
    kernels: dict[str, str] = {}
    result: dict[str, object] = dict.fromkeys(PROBE_FIELDS)
    for line in lines:
        if "[Symmetric]:" in line and "-> Kernel " in line:
            func = line.split("[Symmetric]:", 1)[0].split()[-1]
            kernels[func] = line.split("-> Kernel ", 1)[1].split()[0]
        elif "Bytes -> Algo " in line:
            func = line.split("Bytes -> Algo ", 1)[0].rsplit(":", 1)[0].split()[-1]
            fields = line.split("Bytes -> Algo ", 1)[1].split()
            if len(fields) >= 3 and fields[1] == "proto":
                algos[func] = f"{fields[0]}/{fields[2]}"
        elif " nNodes " in line and " MNNVL " in line:
            fields = line.split()
            result["n_nodes"] = int(fields[fields.index("nNodes") + 1])
            result["mnnvl"] = bool(int(fields[fields.index("MNNVL") + 1]))
    if algos:
        result["nccl_algo"] = ";".join(f"{func}={value}" for func, value in sorted(algos.items()))
    if kernels:
        result["nccl_kernel"] = ";".join(f"{func}={value}" for func, value in sorted(kernels.items()))
    return result


class NcclLog:
    """Read this process's NCCL debug file incrementally, so each case sees only its own lines.

    One process writes one file for all of its communicators. Reading from the last offset
    attributes communicator init lines to the ``_connect`` that created them and kernel lines to
    the case whose warmup launched them.
    """

    def __init__(self) -> None:
        template = os.environ.get("NCCL_DEBUG_FILE", "")
        self.path = nccl_debug_path(template, hostname=socket.gethostname(), pid=os.getpid()) if template else None
        self.offset = 0

    def consume(self) -> list[str]:
        """Return the lines appended since the previous call."""
        if self.path is None or not self.path.exists():
            return []
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            data = handle.read()
        # Keep a partial trailing line for the next read.
        cut = data.rfind(b"\n") + 1
        self.offset += cut
        return data[:cut].decode("utf-8", errors="replace").splitlines()


def _connect(groups: list[CollectiveGroup], rank: int) -> None:
    """Create and connect every communicator serially before any concurrent launch.

    NCCL connects ring/NVLS channels on the first collective of each algorithm. Doing that
    while another communicator's kernel waits for peers can deadlock, so it happens here,
    one communicator at a time, with a device sync in between.
    """
    for group in groups:
        if group.contains(rank):
            index = group.ranks.index(rank)
            shard = torch.full((1024,), float(index), device="cuda")
            gathered = torch.empty(1024 * group.size, device="cuda")
            dist.all_gather_into_tensor(gathered, shard, group=group.ag_pg)
            torch.cuda.synchronize()
            reduced = torch.empty(1024, device="cuda")
            dist.reduce_scatter_tensor(reduced, gathered, group=group.rs_pg)
            torch.cuda.synchronize()
        torch.distributed.barrier()


def _register_pool(group: CollectiveGroup, pool: torch.cuda.MemPool | None) -> None:
    """Register every segment of ``pool`` on both communicators as a symmetric window.

    torch walks the allocator's snapshot for that pool and calls ``ncclCommWindowRegister``
    with ``NCCL_WIN_COLL_SYMMETRIC``; NCCL ends that call with a ``bootstrapBarrier``
    (``dev_runtime.cc``), so registration is collective: all ranks of the group register
    before any of them launches the first collective.
    """
    if pool is None:
        return
    torch.cuda.synchronize()
    for pg in (group.ag_pg, group.rs_pg):
        # register_mem_pool is a ProcessGroupNCCL method; the Gloo default group has no windows.
        pg.register_mem_pool(pool, symm=True)  # type: ignore[attr-defined]
    torch.cuda.synchronize()


def _make_buffers(group: CollectiveGroup, rank: int, size_bytes: int, dtype: torch.dtype, *, symm: bool) -> Buffers:
    """Allocate one group's inputs and outputs, inside a registered MemPool when ``symm``.

    For the symmetric path every buffer must live in a ``torch.cuda.MemPool`` that has been
    registered on the communicator as an ``NCCL_WIN_COLL_SYMMETRIC`` window; NCCL only
    dispatches to its symmetric kernels for windows with that flag.
    """
    elem = torch.empty((), dtype=dtype).element_size()
    if size_bytes % (group.size * elem):
        raise ValueError(f"{size_bytes} bytes cannot be split into {group.size} {dtype} shards")
    numel = size_bytes // elem
    shard = numel // group.size
    index = group.ranks.index(rank)
    pool = torch.cuda.MemPool() if symm else None
    with torch.cuda.use_mem_pool(pool) if pool else contextlib.nullcontext():
        return Buffers(
            # AG chunk k must equal k + 1; RS input is 1 on even and 2 on odd group ranks so every
            # partial sum stays a small integer that is exact in bf16 for any reduction order.
            ag_in=torch.full((shard,), float(index + 1), dtype=dtype, device="cuda"),
            ag_out=torch.empty(numel, dtype=dtype, device="cuda"),
            rs_in=torch.full((numel,), float(index % 2 + 1), dtype=dtype, device="cuda"),
            rs_out=torch.empty(shard, dtype=dtype, device="cuda"),
            pool=pool,
        )


def _verify(group: CollectiveGroup, buffers: Buffers, ops: tuple[str, ...]) -> bool:
    ok = True
    if "ag" in ops:
        expected = torch.arange(1, group.size + 1, device="cuda", dtype=torch.float32)
        ok &= bool(torch.equal(buffers.ag_out.view(group.size, -1).float(), expected[:, None].expand(group.size, -1)))
    if "rs" in ops:
        expected_sum = group.size + group.size // 2
        ok &= bool(torch.all(buffers.rs_out.float() == expected_sum).item())
    return ok


def _launch(
    group: CollectiveGroup, buffers: Buffers, ops: tuple[str, ...], streams: dict[str, torch.cuda.Stream]
) -> None:
    for op in ops:
        with torch.cuda.stream(streams[op]):
            # async_op=False runs NCCL on the current stream and does not block the host.
            if op == "ag":
                dist.all_gather_into_tensor(buffers.ag_out, buffers.ag_in, group=group.ag_pg)
            else:
                dist.reduce_scatter_tensor(buffers.rs_out, buffers.rs_in, group=group.rs_pg)


def _calibrate_sleep_cycles_per_ms() -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda._sleep(SLEEP_CALIBRATION_CYCLES // 10)
    start.record()
    torch.cuda._sleep(SLEEP_CALIBRATION_CYCLES)
    end.record()
    end.synchronize()
    return SLEEP_CALIBRATION_CYCLES / start.elapsed_time(end)


def _timed_trial(
    group: CollectiveGroup,
    buffers: Buffers,
    ops: tuple[str, ...],
    streams: dict[str, torch.cuda.Stream],
    *,
    iters: int,
    delay_cycles: int,
) -> dict[str, float]:
    """Run ``iters`` back-to-back launches and return per-iteration times in ms."""
    main = torch.cuda.current_stream()
    start = torch.cuda.Event(enable_timing=True)
    op_end = {op: torch.cuda.Event(enable_timing=True) for op in ops}
    end = torch.cuda.Event(enable_timing=True)
    if delay_cycles:
        torch.cuda._sleep(delay_cycles)
    start.record(main)
    for op in ops:
        streams[op].wait_event(start)
    host_start = time.perf_counter()
    for _ in range(iters):
        _launch(group, buffers, ops, streams)
    enqueue_ms = (time.perf_counter() - host_start) * 1e3
    for op in ops:
        op_end[op].record(streams[op])
        main.wait_event(op_end[op])
    end.record(main)
    end.synchronize()
    result = {"makespan_ms": start.elapsed_time(end) / iters, "enqueue_ms": enqueue_ms}
    result.update({f"{op}_ms": start.elapsed_time(op_end[op]) / iters for op in ops})
    return result


class CaptureWindows:
    """Open one Nsight capture window per case, with the same numbering on every rank.

    The launcher runs ``nsys profile --capture-range=cudaProfilerApi --capture-range-end=repeat``,
    which writes a new report for every ``cudaProfilerStart``/``Stop`` pair. Every rank opens the
    window for every case, including ranks that sit the case out, so the ``k``-th report any rank
    produces belongs to case ``k``. The case label is also the outermost NVTX range inside each
    report, and rank 0 records the window's wall-clock span, so a report can be matched to its
    case without relying on how a given nsys version numbers its output files.
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.index = 0
        self.window_unix: tuple[float, float] | None = None

    @contextlib.contextmanager
    def case(self, name: str) -> Iterator[int | None]:
        """Capture ``name`` and yield its 1-based report index, or ``None`` when profiling is off."""
        if not self.enabled:
            yield None
            return
        self.index += 1
        torch.cuda.synchronize()
        started = time.time()
        torch.cuda.profiler.start()
        nvtx.range_push(name)
        try:
            yield self.index
        finally:
            torch.cuda.synchronize()
            nvtx.range_pop()
            torch.cuda.profiler.stop()
            self.window_unix = (started, time.time())


def _deregister_pool(group: CollectiveGroup, pool: torch.cuda.MemPool | None) -> None:
    """Release the window registration before the pool's memory goes away.

    torch deregisters every segment first and only then lets the allocator free them; skipping
    this would leave NCCL holding windows over memory it no longer owns.
    """
    if pool is None:
        return
    torch.cuda.synchronize()
    for pg in (group.ag_pg, group.rs_pg):
        pg.deregister_mem_pool(pool)  # type: ignore[attr-defined]
    torch.cuda.synchronize()


def run_case(
    topology: Topology,
    *,
    size_bytes: int,
    workload: str,
    args: argparse.Namespace,
    delay_cycles: int,
    streams: dict[str, torch.cuda.Stream],
    captures: CaptureWindows,
    nccl_log: NcclLog,
) -> list[dict[str, object]]:
    """Run one workload on every group of ``topology`` at the same time; return records on rank 0."""
    ops = workload_ops(workload)
    rank = topology.rank
    group = topology.mine
    probe: dict[str, object] = dict(topology.comm_info)
    buffers = _make_buffers(group, rank, size_bytes, DTYPES[args.dtype], symm=args.algo == "symm") if group else None
    if group:
        _register_pool(group, buffers.pool)
    torch.cuda.synchronize()
    torch.distributed.barrier()

    if group:
        for _ in range(args.warmup):
            _launch(group, buffers, ops, streams)
        torch.cuda.synchronize()
        # Warmup is the first launch at this size, so its TUNING lines name the chosen kernel.
        tuned = parse_nccl_lines(nccl_log.consume())
        probe.update({key: tuned[key] for key in ("nccl_algo", "nccl_kernel")})
    trials: list[dict[str, float]] = []
    label = f"{args.algo}/{'+'.join(g.name for g in topology.groups)}/{workload}/{size_bytes // BYTES_PER_MIB}MiB"
    # Warmup and setup stay outside the window: the report holds exactly the timed trials.
    with captures.case(label) as capture_index:
        for trial in range(args.trials):
            torch.distributed.barrier()
            if group:
                with nvtx.range(f"trial{trial}"):
                    trials.append(
                        _timed_trial(group, buffers, ops, streams, iters=args.iters, delay_cycles=delay_cycles)
                    )
    passed = _verify(group, buffers, ops) if group else True
    local = {
        "rank": rank,
        "group": group.name if group else None,
        "trials": trials,
        "passed": passed,
        **{key: probe.get(key) for key in PROBE_FIELDS},
    }
    gathered: list[dict[str, object] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    if group:
        _deregister_pool(group, buffers.pool)
    del buffers
    # Every rank raises together so no rank is left waiting at the next barrier.
    failed = [item["rank"] for item in gathered if item and not item["passed"]]
    if failed:
        raise RuntimeError(
            f"{workload} {size_bytes}B on {[g.name for g in topology.groups]}: wrong result on ranks {failed}"
        )
    if rank != 0:
        return []

    delay_ms = args.launch_delay_ms
    records = []
    for g in topology.groups:
        members = [item for item in gathered if item and item["group"] == g.name]
        # The slowest rank defines each trial; the median over trials is reported.
        per_trial = [
            {key: max(item["trials"][t][key] for item in members) for key in members[0]["trials"][t]}
            for t in range(args.trials)
        ]
        stats = {key: median(trial[key] for trial in per_trial) for key in per_trial[0]}
        evidence = {key: next((m[key] for m in members if m.get(key) is not None), None) for key in PROBE_FIELDS}
        record: dict[str, object] = {
            "group_name": g.name,
            "group_size": g.size,
            "ranks": g.ranks,
            "size_bytes": size_bytes,
            "size_mib": size_bytes // BYTES_PER_MIB,
            "dtype": args.dtype,
            "workload": workload,
            "iters": args.iters,
            "trials": args.trials,
            "trial_makespan_ms": [trial["makespan_ms"] for trial in per_trial],
            "host_bound": max(trial["enqueue_ms"] for trial in per_trial) > delay_ms,
            "check_passed": True,
            # Profiled numbers include Nsight overhead; the report keeps them apart from timing runs.
            "profiled": args.nsys,
            "nsys_capture_index": capture_index,
            "nsys_label": label if args.nsys else None,
            "nsys_window_unix": list(captures.window_unix) if args.nsys and captures.window_unix else None,
            **evidence,
            **stats,
        }
        aggregate_bus = 0.0
        for op in ops:
            seconds = stats[f"{op}_ms"] / 1e3
            record[f"{op}_algbw_gb_s"] = alg_bandwidth_gb_s(total_bytes=size_bytes, seconds=seconds)
            record[f"{op}_busbw_gb_s"] = bus_bandwidth_gb_s(total_bytes=size_bytes, group_size=g.size, seconds=seconds)
            aggregate_bus += bus_bandwidth_gb_s(
                total_bytes=size_bytes, group_size=g.size, seconds=stats["makespan_ms"] / 1e3
            )
        record["aggregate_busbw_gb_s"] = aggregate_bus
        if args.algo == "symm":
            n_nodes = record.get("n_nodes")
            if n_nodes is None:
                LOGGER.warning(
                    "%s: no nNodes line in %s; symmetric kernels use local-shared-address "
                    "paths and their eligibility is unverified for this case",
                    g.name,
                    os.environ.get("NCCL_DEBUG_FILE", "<unset>"),
                )
            elif n_nodes > 1:
                LOGGER.warning(
                    "%s spans %d OS-hosts: NCCL drops the LSA symmetric kernels for multi-node "
                    "communicators, so this case is not a flat symmetric-memory result",
                    g.name,
                    n_nodes,
                )
        if record["nccl_kernel"] is None and record["nccl_algo"] is None:
            LOGGER.warning("%s %s: no kernel evidence in the NCCL log; treat as unconfirmed", g.name, workload)
        records.append(record)
    return records


def _metadata(args: argparse.Namespace, rank: int, world_size: int) -> dict[str, object]:
    props = torch.cuda.get_device_properties()
    local = {
        "rank": rank,
        "local_rank": int(os.environ["LOCAL_RANK"]),
        "hostname": socket.gethostname(),
        "gpu": props.name,
        "gpu_uuid": str(props.uuid),
        "pci_bus_id": props.pci_bus_id,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    ranks: list[dict[str, object] | None] = [None] * world_size
    dist.all_gather_object(ranks, local)
    return {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "world_size": world_size,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "nccl": ".".join(map(str, torch.cuda.nccl.version())),
        "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith(("NCCL_", "TORCH_NCCL_", "CUDA_"))},
        "ranks": ranks,
    }


def main(argv: list[str] | None = None) -> None:
    """Run the benchmark under torchrun."""
    args = parse_args(argv)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    # Gloo carries barriers and result gathering so no world-sized NCCL communicator exists.
    dist.init_process_group(backend="gloo")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    _check_environment(args.algo, world_size, args)
    output = args.output_dir / "results.jsonl"
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"{output} already exists; use a fresh --output-dir")
        metadata = _metadata(args, rank, world_size)
        (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    else:
        _metadata(args, rank, world_size)

    cycles_per_ms = _calibrate_sleep_cycles_per_ms()
    delay_cycles = int(cycles_per_ms * args.launch_delay_ms)
    streams = {"ag": torch.cuda.Stream(), "rs": torch.cuda.Stream()}
    captures = CaptureWindows(args.nsys)
    nccl_log = NcclLog()
    nccl_log.consume()  # skip lines from the Gloo/default-group setup
    sizes = [mib * BYTES_PER_MIB for mib in args.sizes_mib]

    nsys_index = args.output_dir / "nsys" / "index.tsv"
    indexed_captures: set[int] = set()
    if rank == 0 and args.nsys:
        nsys_index.parent.mkdir(parents=True, exist_ok=True)
        nsys_index.write_text("capture_index\tlabel\tscenario\tphase\tstart_unix\tend_unix\n", encoding="utf-8")

    def emit(records: list[dict[str, object]], **fields: object) -> None:
        if rank != 0:
            return
        with output.open("a", encoding="utf-8") as handle:
            for record in records:
                full = {"algo": args.algo, **fields, **record}
                handle.write(json.dumps(full) + "\n")
                index = record["nsys_capture_index"]
                # Concurrent split cases emit one record per group for the same capture window.
                if index is not None and index not in indexed_captures:
                    indexed_captures.add(index)
                    with nsys_index.open("a", encoding="utf-8") as tsv:
                        start, end = record["nsys_window_unix"] or (float("nan"), float("nan"))
                        tsv.write(
                            f"{index}\t{record['nsys_label']}\t{fields['scenario']}\t{fields.get('phase')}"
                            f"\t{start:.3f}\t{end:.3f}\n"
                        )
                LOGGER.info(
                    "%s %-6s %-11s n=%-2d %5d MiB %-5s %-10s makespan %7.3f ms  busbw %7.1f GB/s  %s%s",
                    args.algo,
                    fields["scenario"],
                    fields.get("phase", ""),
                    record["group_size"],
                    record["size_mib"],
                    record["workload"],
                    record["group_name"],
                    record["makespan_ms"],
                    record["aggregate_busbw_gb_s"],
                    record["nccl_kernel"] or record["nccl_algo"] or "kernel=?",
                    "  [HOST-BOUND]" if record["host_bound"] else "",
                )

    # Every group of one scenario exists at the same time: torch builds a group one rank at a
    # time, so group A must be joined before creating group B on the ranks that belong to both.
    def case_kwargs(size_bytes: int, workload: str) -> dict[str, object]:
        return {
            "size_bytes": size_bytes,
            "workload": workload,
            "args": args,
            "delay_cycles": delay_cycles,
            "streams": streams,
            "captures": captures,
            "nccl_log": nccl_log,
        }

    for n in args.group_sizes:
        group = _new_group(f"single{n}", list(range(n)))
        _connect([group], rank)
        # The connect collectives' TUNING lines are tiny warm-up sizes; keep only the comm init.
        comm_info = {k: v for k, v in parse_nccl_lines(nccl_log.consume()).items() if k in ("n_nodes", "mnnvl")}
        topology = Topology([group], rank, comm_info)
        for size_bytes in sizes:
            for workload in args.workloads:
                records = run_case(topology, **case_kwargs(size_bytes, workload))
                emit(records, scenario="single", phase="isolated", total_ranks=n, layout=None, group_index=0)
        _destroy([group])

    for total in args.split_totals:
        for layout in args.split_layouts:
            ranks_a, ranks_b = split_groups(total, layout)
            groups = [_new_group(f"{layout}{total}-A", ranks_a), _new_group(f"{layout}{total}-B", ranks_b)]
            _connect(groups, rank)
            comm_info = {k: v for k, v in parse_nccl_lines(nccl_log.consume()).items() if k in ("n_nodes", "mnnvl")}
            topology = Topology(groups, rank, comm_info)
            common = {"scenario": "split", "total_ranks": total, "layout": layout}
            for size_bytes in sizes:
                for workload in args.workloads:
                    kwargs = case_kwargs(size_bytes, workload)
                    for index, group in enumerate(groups):
                        emit(
                            run_case(Topology([group], rank, comm_info), **kwargs),
                            phase="isolated",
                            group_index=index,
                            **common,
                        )
                    for index, record in enumerate(run_case(topology, **kwargs)):
                        emit([record], phase="concurrent", group_index=index, **common)
            _destroy(groups)

    torch.distributed.barrier()
    dist.destroy_process_group()


def _destroy(groups: list[CollectiveGroup]) -> None:
    """Release communicators (and NVLS multicast memory) before the next group size."""
    torch.cuda.synchronize()
    torch.distributed.barrier()
    for group in groups:
        for pg in (group.ag_pg, group.rs_pg):
            if pg is not None:
                dist.destroy_process_group(pg)
    torch.distributed.barrier()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
