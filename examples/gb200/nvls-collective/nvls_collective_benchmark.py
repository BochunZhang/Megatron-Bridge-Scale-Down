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
  unset, and every buffer is allocated inside a ``torch.cuda.MemPool`` backed by
  ``ncclMemAlloc`` and registered as a ``NCCL_WIN_COLL_SYMMETRIC`` window. NCCL then dispatches to its symmetric kernels
  (``AllGather_STMC`` / ``ReduceScatter_LDMC`` on a NVLink domain), which are unrelated to
  the classic ring and NVLS collectives. ``NCCL_SYM_CE_THRESHOLD`` must exceed every payload,
  otherwise large AllGathers go to the copy engines instead.

``run_nvls_collective_benchmark.sh`` launches one pass per family and merges them with
``collective_metrics.py``.

Every launch is checked against NCCL's own TUNING log on comm rank 0: a case fails unless
each op ran exactly warmup + trials * iters times, with the expected shard size, on the
family's implementation (``Ring/*``, ``NVLS/*`` or a symmetric kernel; never ``CopyEngine``).
Each case gets ``<output-dir>/cases/<case>/`` holding every member rank's slice of its NCCL
log, ``case.json`` and, after ``analyse_nsys.py``, the case's nsys reports.

Scenarios (all inside one torchrun job; ranks outside a group wait at a Gloo barrier):

* ``single``: ranks ``[0, n)`` for every ``n`` in ``--group-sizes`` run AG alone, RS alone,
  and AG + RS at the same time on two streams and two communicators.
* ``split``: ranks ``[0, T)`` for every ``T`` in ``--split-totals`` are cut into two groups of
  ``T/2``; each group runs alone, then both run the same workload at the same time, then
  (``cross``) group A all-gathers while group B reduce-scatters.

Every timed trial opens with a start gate: a one-element all-reduce over all ranks of the
scenario, after which each rank records its start event. Groups that share no communicator
therefore start their traffic together, and their times share one origin, which is what
``concurrency = min(T_A, T_B) / max(T_A, T_B)`` relies on.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import logging
import os
import re
import socket
import time
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path
from statistics import mean, median

import torch
import torch.cuda.profiler
import torch.distributed as dist
from collective_metrics import (
    BYTES_PER_MIB,
    CROSS_GROUP_WORKLOADS,
    CROSS_PART,
    CROSS_WORKLOAD,
    NCCL_ALGO_ENV,
    SPLIT_LAYOUTS,
    SPLIT_PARTS,
    WORKLOADS,
    alg_bandwidth_gb_s,
    bus_bandwidth_gb_s,
    case_name,
    rank_file,
    scenario_name,
    split_groups,
    trial_concurrency,
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
OP_FUNCS = {"ag": "AllGather", "rs": "ReduceScatter"}
PROBE_FIELDS = ("nccl_algo", "nccl_kernel", "n_nodes", "mnnvl")
# The three INFO(NCCL_TUNING, ...) lines NCCL 2.29 prints on comm rank 0 for every launch:
#   "<Func>: <n> Bytes -> Algo <algo> proto <proto> ..."      enqueue.cc, classic kernels
#   "<Func> [Symmetric]: <n> Bytes -> Kernel <name> ..."      scheduler/symmetric_sched.cc
#   "<Func> [Copy Engine]: <n> Bytes -> cudaMemcpy..."        enqueue.cc, CE collectives
# <n> is the per-rank count times the element size, i.e. the shard: AG sendcount, RS recvcount.
_TUNING_LINE = re.compile(
    r"NCCL INFO (\w+)(?: \[(Symmetric|Copy Engine)\])?: (\d+) Bytes -> "
    r"(?:Algo (\S+) proto (\S+)|Kernel (\S+)|cudaMemcpy)"
)
COPY_ENGINE = "CopyEngine"
# enqueue.cc sends an AllGather with sendcount > NCCL_SYM_CE_THRESHOLD (elements, default 8Mi)
# on a window-registered Blackwell NVLink communicator to the copy engines, before any
# symmetric kernel is considered and regardless of NCCL_SYM_KERNEL.
SYM_CE_THRESHOLD_ENV = "NCCL_SYM_CE_THRESHOLD"

# The start gate: a one-element fp32 all-reduce; its TUNING line is "AllReduce: 4 Bytes -> ...".
GATE_FUNC = "AllReduce"
GATE_BYTES = 4

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


@dataclasses.dataclass
class StartGate:
    """A one-element all-reduce over every rank of a scenario that opens each timed trial.

    An all-reduce completes on every rank only after every rank has entered it, so all groups
    of the scenario, which share no communicator, leave it within a few microseconds of each
    other. Each rank records its start event right after the gate, so every group's times are
    measured from the same instant and the groups' traffic starts together.
    """

    ranks: list[int]
    pg: dist.ProcessGroup | None
    tensor: torch.Tensor | None


class Topology:
    """The collective groups of one job, in rank order, by name.

    ``comm_info`` holds this rank's ``nNodes``/``MNNVL`` as NCCL logged them when the group's
    communicators were created; it is empty on ranks outside every group. ``gate`` spans every
    rank of the scenario, including the other half-group of an isolated split case.
    """

    def __init__(
        self, groups: list[CollectiveGroup], rank: int, comm_info: dict[str, object], gate: StartGate
    ) -> None:
        self.rank = rank
        self.groups = groups
        self.group_by_rank = {member: group for group in groups for member in group.ranks}
        self.comm_info = comm_info
        self.gate = gate

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
    parser.add_argument(
        "--workloads",
        type=_csv_choices(WORKLOADS),
        default=list(WORKLOADS),
        help="Per-group workloads; split scenarios also run the AG-on-A / RS-on-B cross case when ag and rs are both listed",
    )
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
        # NCCL_SYM_KERNEL is a single-kernel mask: pinning an AG kernel leaves RS with no
        # symmetric kernel at all, so RS would silently run a classic kernel.
        if args.symm_kernel != "default" and args.workloads != [args.symm_kernel]:
            raise ValueError(
                f"--symm-kernel {args.symm_kernel} masks every other symmetric kernel;"
                f" run it with --workloads {args.symm_kernel}, got {args.workloads}"
            )
        # The CE branch compares the element count with the threshold; a byte bound at least
        # as large as the biggest payload keeps every AllGather on the symmetric kernels.
        largest = max(args.sizes_mib) * BYTES_PER_MIB
        ce_threshold = os.environ.get(SYM_CE_THRESHOLD_ENV, "")
        if not ce_threshold.isdigit() or int(ce_threshold) < largest:
            raise RuntimeError(
                f"--algo symm requires {SYM_CE_THRESHOLD_ENV} >= {largest} so no AllGather takes the"
                f" copy-engine path, got {ce_threshold!r}"
            )
    else:
        assert expected is not None
        if actual.lower() != expected.lower():
            raise RuntimeError(f"--algo {algo} requires NCCL_ALGO={expected}, got {actual!r}")
        if algo == "nvls" and nvls_enable != "1":
            raise RuntimeError("--algo nvls requires NCCL_NVLS_ENABLE=1 so a missing NVLS path fails loudly")
        if algo == "ring" and nvls_enable != "0":
            raise RuntimeError("--algo ring requires NCCL_NVLS_ENABLE=0 so no multicast buffers are allocated")
    # Every case is checked against the TUNING lines of this process's own debug file.
    if os.environ.get("NCCL_DEBUG", "").upper() not in ("INFO", "TRACE"):
        raise RuntimeError("NCCL_DEBUG=INFO is required: the executed implementation is read from NCCL's log")
    subsys = {item.strip().upper() for item in os.environ.get("NCCL_DEBUG_SUBSYS", "").split(",")}
    if not subsys & {"TUNING", "ALL"}:
        raise RuntimeError("NCCL_DEBUG_SUBSYS must include TUNING: it carries the per-launch algorithm/kernel line")
    if "%p" not in os.environ.get("NCCL_DEBUG_FILE", ""):
        raise RuntimeError("NCCL_DEBUG_FILE must be set and contain %p so every process reads only its own log")
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


def _new_gate(ranks: list[int]) -> StartGate:
    # Collective over the default group like _new_group; only members get the NCCL group.
    pg = dist.new_group(ranks=ranks, backend="nccl")
    if dist.get_rank() not in ranks:
        return StartGate(ranks=ranks, pg=None, tensor=None)
    return StartGate(ranks=ranks, pg=pg, tensor=torch.zeros(1, dtype=torch.float32, device="cuda"))


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


def parse_tuning_line(line: str) -> tuple[str, str, int] | None:
    """Return ``(func, implementation, shard_bytes)`` for one NCCL TUNING launch line.

    The implementation is ``<Algo>/<Proto>`` for classic kernels (``NVLS/Simple``),
    ``Symmetric/<kernel>`` for the window path and ``CopyEngine`` for CE collectives.
    """
    match = _TUNING_LINE.search(line)
    if match is None:
        return None
    func, path, nbytes, algo, proto, kernel = match.groups()
    if path == "Symmetric":
        implementation = f"Symmetric/{kernel}"
    elif path == "Copy Engine":
        implementation = COPY_ENGINE
    else:
        implementation = f"{algo}/{proto}"
    return func, implementation, int(nbytes)


def parse_nccl_lines(lines: list[str]) -> dict[str, object]:
    """Pull the executed algorithm/kernel and the communicator topology out of NCCL log lines.

    * The per-launch TUNING lines described at ``_TUNING_LINE``.
    * ``comm <p> rank <r> nRanks <n> nNodes <n> localRanks <n> localRank <r> MNNVL <f>`` (``init.cc``).

    TUNING lines are emitted by comm rank 0 only, so the launcher must include ``TUNING`` in
    ``NCCL_DEBUG_SUBSYS``. Kernels are reported per function, e.g.
    ``AllGather=NVLS/Simple;ReduceScatter=NVLS/Simple``, so an AG+RS case shows both; the
    symmetric and copy-engine paths go to ``nccl_kernel``.
    """
    algos: dict[str, str] = {}
    kernels: dict[str, str] = {}
    result: dict[str, object] = dict.fromkeys(PROBE_FIELDS)
    for line in lines:
        launch = parse_tuning_line(line)
        if launch is not None:
            func, implementation, _ = launch
            if implementation == COPY_ENGINE:
                kernels[func] = COPY_ENGINE
            elif implementation.startswith("Symmetric/"):
                kernels[func] = implementation.split("/", 1)[1]
            else:
                algos[func] = implementation
        elif " nNodes " in line and " MNNVL " in line:
            fields = line.split()
            result["n_nodes"] = int(fields[fields.index("nNodes") + 1])
            result["mnnvl"] = bool(int(fields[fields.index("MNNVL") + 1]))
    if algos:
        result["nccl_algo"] = ";".join(f"{func}={value}" for func, value in sorted(algos.items()))
    if kernels:
        result["nccl_kernel"] = ";".join(f"{func}={value}" for func, value in sorted(kernels.items()))
    return result


def implementation_matches(algo: str, op: str, implementation: str, *, pinned_kernel: str | None) -> bool:
    """Whether one launch ran the implementation the ``algo`` pass is meant to measure."""
    if algo == "symm":
        family, _, kernel = implementation.partition("/")
        return family == "Symmetric" and kernel in SYMM_KERNELS[op] and pinned_kernel in (None, kernel)
    expected = NCCL_ALGO_ENV[algo]
    return implementation.split("/", 1)[0] == expected


def check_implementation(
    lines: list[str],
    *,
    algo: str,
    ops: tuple[str, ...],
    launches: int,
    shard_bytes: int,
    pinned_kernel: str | None,
    gate_launches: int = 0,
) -> tuple[str | None, list[str]]:
    """Check every collective launch of one case against the family under test.

    ``lines`` are the NCCL log lines comm rank 0 of the group wrote during the case (warmup
    and timed trials). Each op must appear exactly ``launches`` times, every launch must move
    ``shard_bytes`` per rank and use the expected implementation, and no other collective may
    run except the start gate: exactly ``gate_launches`` AllReduce lines, which only the gate's
    comm rank 0 logs. Returns the observed implementations
    (``AllGather=Symmetric/AllGather_STMC;...``) and the list of violations; an empty list means
    the case measured what its name says.
    """
    expected_funcs = {OP_FUNCS[op]: op for op in ops}
    seen: dict[str, Counter[str]] = {func: Counter() for func in expected_funcs}
    sizes: dict[str, Counter[int]] = {func: Counter() for func in expected_funcs}
    foreign: Counter[str] = Counter()
    gates = 0
    for line in lines:
        launch = parse_tuning_line(line)
        if launch is None:
            continue
        func, implementation, nbytes = launch
        if func == GATE_FUNC and nbytes == GATE_BYTES:
            gates += 1
            continue
        if func not in expected_funcs:
            foreign[f"{func}={implementation}"] += 1
            continue
        seen[func][implementation] += 1
        sizes[func][nbytes] += 1

    errors = [f"unexpected collective {name} x{count}" for name, count in sorted(foreign.items())]
    if gates != gate_launches:
        errors.append(f"start gate: {gates} {GATE_FUNC} launches in the NCCL log, expected {gate_launches}")
    for func, op in expected_funcs.items():
        total = sum(seen[func].values())
        if total != launches:
            errors.append(f"{func}: {total} launches in the NCCL log, expected {launches}")
        wrong = {
            impl: n
            for impl, n in seen[func].items()
            if not implementation_matches(algo, op, impl, pinned_kernel=pinned_kernel)
        }
        if wrong:
            if algo == "symm":
                want = f"Symmetric/{pinned_kernel or '|'.join(SYMM_KERNELS[op])}"
            else:
                want = f"{NCCL_ALGO_ENV[algo]}/*"
            got = ", ".join(f"{impl} x{n}" for impl, n in sorted(wrong.items()))
            errors.append(f"{func}: expected {want}, got {got}")
        bad_sizes = {size: n for size, n in sizes[func].items() if size != shard_bytes}
        if bad_sizes:
            got = ", ".join(f"{size} B x{n}" for size, n in sorted(bad_sizes.items()))
            errors.append(f"{func}: expected {shard_bytes} B per rank, got {got}")
    observed = ";".join(f"{func}={'|'.join(sorted(seen[func]))}" for func in sorted(seen) if seen[func])
    return observed or None, errors


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


def _connect(groups: list[CollectiveGroup], gate: StartGate, rank: int) -> None:
    """Create and connect every communicator serially before any concurrent launch.

    NCCL connects ring/NVLS channels on the first collective of each algorithm. Doing that
    while another communicator's kernel waits for peers can deadlock, so it happens here,
    one communicator at a time, with a device sync in between. The start gate goes first.
    """
    if gate.pg is not None:
        dist.all_reduce(gate.tensor, group=gate.pg)
        torch.cuda.synchronize()
    torch.distributed.barrier()
    for group in groups:
        if rank in group.ranks:
            index = group.ranks.index(rank)
            shard = torch.full((1024,), float(index), device="cuda")
            gathered = torch.empty(1024 * group.size, device="cuda")
            dist.all_gather_into_tensor(gathered, shard, group=group.ag_pg)
            torch.cuda.synchronize()
            reduced = torch.empty(1024, device="cuda")
            dist.reduce_scatter_tensor(reduced, gathered, group=group.rs_pg)
            torch.cuda.synchronize()
        torch.distributed.barrier()


def _nccl_backend(pg: dist.ProcessGroup) -> dist.ProcessGroupNCCL:
    """Return the ``ProcessGroupNCCL`` behind ``pg``.

    ``dist.new_group`` returns the generic ``ProcessGroup`` wrapper; ``mem_allocator``,
    ``register_mem_pool`` and ``deregister_mem_pool`` are bound only on the NCCL backend.
    """
    return pg._get_backend(torch.device("cuda"))


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
        _nccl_backend(pg).register_mem_pool(pool, symm=True)
    torch.cuda.synchronize()


def _make_buffers(group: CollectiveGroup, rank: int, size_bytes: int, dtype: torch.dtype, *, symm: bool) -> Buffers:
    """Allocate one group's inputs and outputs, inside a registered MemPool when ``symm``.

    For the symmetric path every buffer must live in a ``torch.cuda.MemPool`` that has been
    registered on the communicator as an ``NCCL_WIN_COLL_SYMMETRIC`` window; NCCL only
    dispatches to its symmetric kernels for windows with that flag.

    The pool must allocate through ``ncclMemAlloc``. NCCL shares and multicast-binds only the
    single cuMem allocation that contains the window base (``cuMemGetAddressRange`` +
    ``cuMemRetainAllocationHandle`` in ``dev_runtime.cc``). A default pool under
    ``expandable_segments`` builds one segment out of many cuMem handles, so registration
    succeeds but everything past the first handle is never mapped to peers, and the
    symmetric collective silently returns wrong data. A pool with its own allocator never
    uses expandable segments.
    """
    elem = torch.empty((), dtype=dtype).element_size()
    if size_bytes % (group.size * elem):
        raise ValueError(f"{size_bytes} bytes cannot be split into {group.size} {dtype} shards")
    numel = size_bytes // elem
    shard = numel // group.size
    index = group.ranks.index(rank)
    pool = None
    if symm:
        pool = torch.cuda.MemPool(_nccl_backend(group.ag_pg).mem_allocator)
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
        ok &= bool(torch.all(buffers.ag_out.view(group.size, -1).float() == expected[:, None]).item())
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
    group: CollectiveGroup | None,
    buffers: Buffers | None,
    ops: tuple[str, ...],
    streams: dict[str, torch.cuda.Stream],
    *,
    iters: int,
    delay_cycles: int,
    gate: StartGate,
) -> dict[str, float] | None:
    """Pass the start gate, run ``iters`` back-to-back launches and return per-iteration times in ms.

    The GPU sleep keeps the gate and every launch queued before any of them runs, so host
    enqueue time is not measured. Every rank of the scenario passes the gate, including the idle
    half of an isolated split case (which then returns ``None``), so the gate has the same
    members in every case. ``start`` is recorded after the gate: all groups' clocks start
    within a few microseconds of each other, and their end times can be compared directly.
    """
    main = torch.cuda.current_stream()
    if delay_cycles:
        torch.cuda._sleep(delay_cycles)
    # Everything enqueued after the sleep must be queued before the sleep ends.
    host_start = time.perf_counter()
    dist.all_reduce(gate.tensor, group=gate.pg)
    if group is None:
        main.synchronize()
        return None
    start = torch.cuda.Event(enable_timing=True)
    op_end = {op: torch.cuda.Event(enable_timing=True) for op in ops}
    end = torch.cuda.Event(enable_timing=True)
    start.record(main)
    for op in ops:
        streams[op].wait_event(start)
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
        _nccl_backend(pg).deregister_mem_pool(pool)
    torch.cuda.synchronize()


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.writelines(line + "\n" for line in lines)


def run_case(
    topology: Topology,
    *,
    case: str,
    size_bytes: int,
    workloads: list[str],
    args: argparse.Namespace,
    delay_cycles: int,
    streams: dict[str, torch.cuda.Stream],
    captures: CaptureWindows,
    nccl_log: NcclLog,
) -> list[dict[str, object]]:
    """Run ``workloads[i]`` on ``topology.groups[i]``, all at the same time; return records on rank 0.

    Every group of a concurrent case runs the same workload except the cross case, where group A
    all-gathers while group B reduce-scatters. All of them start at the scenario's start gate.

    Everything about the case lands in ``<output-dir>/cases/<case>/``: each member rank's slice
    of its NCCL debug log, rank 0's ``case.json`` and, after ``analyse_nsys.py``, the nsys reports.
    The case fails on every rank if the result is wrong or if comm rank 0 of any group logged a
    launch that is not the implementation this pass measures.
    """
    if len(workloads) != len(topology.groups):
        raise ValueError(f"{case}: {len(workloads)} workloads for {len(topology.groups)} groups")
    group_ops = {g.name: workload_ops(w) for g, w in zip(topology.groups, workloads)}
    rank = topology.rank
    world_size = dist.get_world_size()
    group = topology.mine
    ops = group_ops[group.name] if group else ()
    gate = topology.gate
    case_dir = args.output_dir / "cases" / case
    label = f"{args.algo}/{case}"
    probe: dict[str, object] = dict(topology.comm_info)
    buffers = _make_buffers(group, rank, size_bytes, DTYPES[args.dtype], symm=args.algo == "symm") if group else None
    if group:
        _register_pool(group, buffers.pool)
    torch.cuda.synchronize()
    torch.distributed.barrier()
    # Window registration and anything since the previous case belong to this case's log.
    case_lines = nccl_log.consume()

    if group:
        for _ in range(args.warmup):
            _launch(group, buffers, ops, streams)
        torch.cuda.synchronize()
    trials: list[dict[str, float]] = []
    # Warmup and setup stay outside the window: the report holds exactly the timed trials.
    with captures.case(label) as capture_index:
        for trial in range(args.trials):
            torch.distributed.barrier()
            if gate.pg is not None:
                with nvtx.range(f"trial{trial}"):
                    result = _timed_trial(
                        group, buffers, ops, streams, iters=args.iters, delay_cycles=delay_cycles, gate=gate
                    )
                if result is not None:
                    trials.append(result)
    case_lines += nccl_log.consume()
    passed = _verify(group, buffers, ops) if group else True
    implementation, impl_errors = None, []
    if group:
        tuned = parse_nccl_lines(case_lines)
        probe.update({key: tuned[key] for key in ("nccl_algo", "nccl_kernel")})
    if group and rank == group.ranks[0]:
        # dist.new_group keeps the rank order, so the group's first rank is comm rank 0 of both
        # communicators, the only rank NCCL writes TUNING lines on.
        implementation, impl_errors = check_implementation(
            case_lines,
            algo=args.algo,
            ops=ops,
            launches=args.warmup + args.trials * args.iters,
            shard_bytes=size_bytes // group.size,
            pinned_kernel=(os.environ.get("NCCL_SYM_KERNEL") or None) if args.algo == "symm" else None,
            # Gate comm rank 0 is the scenario's first rank; it logs one AllReduce per trial.
            gate_launches=args.trials if rank == gate.ranks[0] else 0,
        )
    local = {
        "rank": rank,
        "group": group.name if group else None,
        "trials": trials,
        "passed": passed,
        "nccl_impl": implementation,
        "impl_errors": impl_errors,
        **{key: probe.get(key) for key in PROBE_FIELDS},
    }
    gathered: list[dict[str, object] | None] = [None] * world_size
    dist.all_gather_object(gathered, local)
    if group:
        _deregister_pool(group, buffers.pool)
        case_lines += nccl_log.consume()
        _write_lines(
            case_dir / rank_file(args.algo, case, rank=rank, world_size=world_size, suffix=".nccl.log"), case_lines
        )
    del buffers
    # Every rank raises together so no rank is left waiting at the next barrier.
    failed = [item["rank"] for item in gathered if item and not item["passed"]]
    if failed:
        raise RuntimeError(f"{label}: wrong result on ranks {failed}; NCCL logs in {case_dir}")
    impl_failures = [f"{item['group']}: {error}" for item in gathered if item for error in item["impl_errors"]]
    if impl_failures:
        if rank == 0:
            case_dir.mkdir(parents=True, exist_ok=True)
            report = {"label": label, "implementation_errors": impl_failures}
            (case_dir / "case.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise RuntimeError(
            f"{label}: NCCL did not run the {args.algo} implementation\n  "
            + "\n  ".join(impl_failures)
            + f"\nNCCL logs in {case_dir}"
        )
    if rank != 0:
        return []

    delay_ms = args.launch_delay_ms
    # The slowest rank defines each trial of each group.
    group_trials: dict[str, list[dict[str, float]]] = {}
    for g in topology.groups:
        members = [item for item in gathered if item and item["group"] == g.name]
        group_trials[g.name] = [
            {key: max(item["trials"][t][key] for item in members) for key in members[0]["trials"][t]}
            for t in range(args.trials)
        ]
    # All groups start at the gate, so their makespans share an origin: min/max is the share of
    # the longer group's time during which the other group was also moving data.
    concurrency = None
    if len(topology.groups) > 1:
        concurrency = trial_concurrency(
            [[trial["makespan_ms"] for trial in trials] for trials in group_trials.values()]
        )
    records = []
    for g, workload in zip(topology.groups, workloads):
        members = [item for item in gathered if item and item["group"] == g.name]
        per_trial = group_trials[g.name]
        stats = {key: median(trial[key] for trial in per_trial) for key in per_trial[0]}
        means = {f"mean_{key}": mean(trial[key] for trial in per_trial) for key in per_trial[0]}
        evidence = {
            key: next((m[key] for m in members if m.get(key) is not None), None)
            for key in (*PROBE_FIELDS, "nccl_impl")
        }
        record: dict[str, object] = {
            "case": case,
            "case_dir": str(case_dir.relative_to(args.output_dir)),
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
            "trial_concurrency": concurrency,
            "concurrency": median(concurrency) if concurrency else None,
            **evidence,
            **stats,
            **means,
        }
        aggregate_bus = 0.0
        for op in group_ops[g.name]:
            seconds = stats[f"{op}_ms"] / 1e3
            record[f"{op}_algbw_gb_s"] = alg_bandwidth_gb_s(total_bytes=size_bytes, seconds=seconds)
            record[f"{op}_busbw_gb_s"] = bus_bandwidth_gb_s(total_bytes=size_bytes, group_size=g.size, seconds=seconds)
            aggregate_bus += bus_bandwidth_gb_s(
                total_bytes=size_bytes, group_size=g.size, seconds=stats["makespan_ms"] / 1e3
            )
        record["aggregate_busbw_gb_s"] = aggregate_bus
        records.append(record)
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "case.json").write_text(json.dumps({"label": label, "records": records}, indent=2), encoding="utf-8")
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
                    record["nccl_impl"] or "kernel=?",
                    "  [HOST-BOUND]" if record["host_bound"] else "",
                )

    # Every group of one scenario exists at the same time: torch builds a group one rank at a
    # time, so group A must be joined before creating group B on the ranks that belong to both.
    def case_kwargs(
        scenario: str, part: str | None, size_bytes: int, workload: str, workloads: list[str]
    ) -> dict[str, object]:
        """``workload`` names the case; ``workloads`` is what each group of the topology runs."""
        return {
            "case": case_name(
                scenario, part=part, workload=workload, dtype=args.dtype, size_mib=size_bytes // BYTES_PER_MIB
            ),
            "size_bytes": size_bytes,
            "workloads": workloads,
            "args": args,
            "delay_cycles": delay_cycles,
            "streams": streams,
            "captures": captures,
            "nccl_log": nccl_log,
        }

    def connect(scenario: str, groups: list[CollectiveGroup]) -> tuple[Topology, Path | None]:
        """Create the scenario's communicators and start gate; keep their init lines in ``comm/``."""
        gate = _new_gate(sorted(member for group in groups for member in group.ranks))
        _connect(groups, gate, rank)
        lines = nccl_log.consume()
        # The connect collectives' TUNING lines are tiny warm-up sizes; keep only the comm init.
        comm_info = {k: v for k, v in parse_nccl_lines(lines).items() if k in ("n_nodes", "mnnvl")}
        topology = Topology(groups, rank, comm_info, gate)
        comm_log = None
        if topology.mine:
            name = rank_file(args.algo, scenario, rank=rank, world_size=world_size, suffix=".nccl.log")
            comm_log = args.output_dir / "comm" / name
            _write_lines(comm_log, lines)
        return topology, comm_log

    def destroy(topology: Topology, comm_log: Path | None) -> None:
        _destroy(topology.groups, topology.gate)
        lines = nccl_log.consume()
        if comm_log:
            _write_lines(comm_log, lines)

    for n in args.group_sizes:
        scenario = scenario_name("single", total_ranks=n)
        group = _new_group(f"single{n}", list(range(n)))
        topology, comm_log = connect(scenario, [group])
        for size_bytes in sizes:
            for workload in args.workloads:
                records = run_case(topology, **case_kwargs(scenario, None, size_bytes, workload, [workload]))
                emit(records, scenario="single", phase="isolated", total_ranks=n, layout=None, group_index=0)
        destroy(topology, comm_log)

    for total in args.split_totals:
        for layout in args.split_layouts:
            scenario = scenario_name("split", total_ranks=total, layout=layout)
            ranks_a, ranks_b = split_groups(total, layout)
            groups = [_new_group(f"{layout}{total}-A", ranks_a), _new_group(f"{layout}{total}-B", ranks_b)]
            topology, comm_log = connect(scenario, groups)
            common = {"scenario": "split", "total_ranks": total, "layout": layout}
            isolated_parts, concurrent_part = SPLIT_PARTS[:2], SPLIT_PARTS[2]
            for size_bytes in sizes:
                for workload in args.workloads:
                    for index, group in enumerate(groups):
                        # The other half-group still passes the gate, so it stays idle, not absent.
                        isolated = Topology([group], rank, topology.comm_info, topology.gate)
                        emit(
                            run_case(
                                isolated,
                                **case_kwargs(scenario, isolated_parts[index], size_bytes, workload, [workload]),
                            ),
                            phase="isolated",
                            group_index=index,
                            **common,
                        )
                    concurrent = run_case(
                        topology, **case_kwargs(scenario, concurrent_part, size_bytes, workload, [workload] * 2)
                    )
                    for index, record in enumerate(concurrent):
                        emit([record], phase="concurrent", group_index=index, **common)
                # A all-gathers while B reduce-scatters; its references are A-isolated ag and
                # B-isolated rs above, so it runs only when both of those were measured.
                if set(CROSS_GROUP_WORKLOADS) <= set(args.workloads):
                    cross = run_case(
                        topology,
                        **case_kwargs(scenario, CROSS_PART, size_bytes, CROSS_WORKLOAD, list(CROSS_GROUP_WORKLOADS)),
                    )
                    for index, record in enumerate(cross):
                        emit([record], phase="cross", group_index=index, **common)
            destroy(topology, comm_log)

    torch.distributed.barrier()
    dist.destroy_process_group()


def _destroy(groups: list[CollectiveGroup], gate: StartGate) -> None:
    """Release communicators (and NVLS multicast memory) before the next group size."""
    torch.cuda.synchronize()
    torch.distributed.barrier()
    for pg in [pg for group in groups for pg in (group.ag_pg, group.rs_pg)] + [gate.pg]:
        if pg is not None:
            dist.destroy_process_group(pg)
    torch.distributed.barrier()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
