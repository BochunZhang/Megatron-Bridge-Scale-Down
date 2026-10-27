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
"""Hardware-independent helpers and the ring-vs-NVLS report for the collective benchmark.

Only the Python standard library is used so the layout, bandwidth formulas and the
report can be checked on a machine without CUDA. Run as a script to compare
``results.jsonl`` files produced by ``nvls_collective_benchmark.py``::

    python collective_metrics.py --input ring=<dir>/results.jsonl \\
        --input nvls=<dir>/results.jsonl --output <dir>/compare.md
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path


LOGGER = logging.getLogger(__name__)
BYTES_PER_MIB = 1024 * 1024
BYTES_PER_GB = 1_000_000_000
# Algorithm families. "symm" is the window-registered symmetric-memory path; its report rows
# sit alongside ring and nvls because it answers the same questions with a third implementation.
# "symm-p2p" runs the same symmetric kernels with multicast off (NCCL_NVLS_ENABLE=0), so NCCL
# picks their unicast LSA load/store variants (AllGather_ST, ReduceScatter_LD, AllReduce_RSxLD_AGxST,
# and the LL ones for small sizes): a direct all-to-all over NVLink, against ring's n-1 hop chain.
# "symm-tma" (AllGather_TmaSTMC) and "copy-engine" (CE AllGather to the multicast address)
# are two more AllGather implementations on the same windows; both pair them with the multicast
# ReduceScatter_LDMC, because NCCL has neither a TMA multicast nor a CE reduce-scatter.
ALGORITHMS = ("ring", "nvls", "symm", "symm-p2p", "symm-tma", "copy-engine")
# Collectives, in launch order of the ag+rs pair. "ar" is an all-reduce of the whole per-rank
# tensor, so it moves twice the payload of one AG or RS shard exchange (see bus_bandwidth_gb_s).
OPS = ("ag", "rs", "ar")
# Extended ops used by the dual-collective workloads: ag2/rs2 are second communicators running
# the same collective concurrently; read/write are NCCL point-to-point sends/receives.
EXTENDED_OPS = ("ag2", "rs2", "read", "write")
# read: the pair's first rank receives one AG shard from the second; write: it sends one to it.
P2P_OPS = frozenset({"read", "write"})
WORKLOADS = (
    "ag", "rs", "ar", "ag+rs",
    "ag+ag", "rs+rs",
    "ag+read", "ag+write",
    "rs+read", "rs+write",
    "read", "write",
)
# Workloads each family measures. symm-tma and copy-engine only differ from symm in the
# AllGather: their RS alone would be symm's ReduceScatter_LDMC again, and they have no all-reduce.
ALGO_WORKLOADS = {
    "ring": WORKLOADS,
    "nvls": WORKLOADS,
    "symm": WORKLOADS,
    "symm-p2p": WORKLOADS,
    "symm-tma": ("ag", "ag+rs", "ag+ag", "ag+read", "ag+write", "read", "write"),
    "copy-engine": ("ag", "ag+rs", "ag+ag", "ag+read", "ag+write", "read", "write"),
}
# How a read/write moves its bytes over NVLink. push: the rank that owns the data stores it into
# the peer (NCCL's P2P write, its default on Blackwell): NVLink TX/RX *Requests* Data. pull: the
# rank that needs the data loads it from the peer (NCCL_P2P_READ_ENABLE=1, transport/p2p.cc): the
# owner's TX and the reader's RX *Responses* Data. Only the Simple protocol reads; LL always writes.
P2P_MODES = ("push", "pull")
DEFAULT_P2P_MODE = "push"
# NCCL_P2P_LL_THRESHOLD default (enqueue.cc): a send of at most nChannels * this many bytes uses LL.
P2P_LL_THRESHOLD = 16384
# Outside-P2P scenario (--outside-p2p-sizes): ranks [0, n) run the collective while rank 0 also
# exchanges with rank n, the first rank outside the collective group.
OUTSIDE_P2P_SCENARIO = "outside-p2p"
SPLIT_LAYOUTS = ("contiguous", "interleaved")
# Split cases: both half-groups running the same workload at the same time, and "cross": group A
# all-gathers while group B reduce-scatters, at the same time. Neither half runs alone: the
# reference for a half-group of T/2 ranks is the single group of T/2 ranks (single-n<T/2>).
CONCURRENT_PART = "AB-concurrent"
CROSS_PART = "AB-cross"
SPLIT_PARTS = (CONCURRENT_PART, CROSS_PART)
CROSS_WORKLOAD = "agA-rsB"
CROSS_GROUP_WORKLOADS = ("ag", "rs")  # per split group, A then B
NCCL_ALGO_ENV = {
    "ring": "Ring",
    "nvls": "NVLS",
    "symm": None,
    "symm-p2p": None,
    "symm-tma": None,
    "copy-engine": None,
}
# Late join (--late-join): one group reduce-scatters a tensor LATE_JOIN_RS_SCALE times the AG size
# and all-gathers on the same ranks once the RS is --join-fraction (default LATE_JOIN_FRACTION) of its
# isolated time in; 0 launches both together.
# Its two references, AG of size S and RS of size LATE_JOIN_RS_SCALE * S, run alone in the same
# scenario with the same per-iteration timing.
LATE_JOIN_RS_SCALE = 4
LATE_JOIN_FRACTION = 0.25
LATE_JOIN_WORKLOAD = f"rs-x{LATE_JOIN_RS_SCALE}+late-ag"
# Spelled-out workloads for case directory and report names.
WORKLOAD_NAMES = {
    "ag": "all_gather",
    "rs": "reduce_scatter",
    "ar": "all_reduce",
    "ag+rs": "all_gather+reduce_scatter",
    "ag+ag": "all_gather+all_gather",
    "rs+rs": "reduce_scatter+reduce_scatter",
    "ag+read": "all_gather+read",
    "ag+write": "all_gather+write",
    "rs+read": "reduce_scatter+read",
    "rs+write": "reduce_scatter+write",
    "read": "read",
    "write": "write",
    CROSS_WORKLOAD: "all_gather_A+reduce_scatter_B",
    LATE_JOIN_WORKLOAD: f"reduce_scatter_x{LATE_JOIN_RS_SCALE}+late_all_gather",
}

# --cta-configs: one entry sets the CTA count of each op's communicator through ncclConfig_t
# (minCTAs/maxCTAs, passed per process group as ProcessGroupNCCL.Options.config). ag2/rs2 follow
# ag/rs; read and write share the "p2p" key. An op without a key keeps NCCL's own choice.
DEFAULT_CTA_CONFIG = "default"
CTA_KEYS = ("ag", "rs", "ar", "p2p")
CTA_KEY_OF_OP = {"ag": "ag", "ag2": "ag", "rs": "rs", "rs2": "rs", "ar": "ar", "read": "p2p", "write": "p2p"}
# ncclConfig_t minCTAs/maxCTAs are capped at MAXCHANNELS (src/include/device.h).
NCCL_MAX_CTAS = 64
# NCCL rounds the P2P channel counts up to powers of two (paths.cc ncclTopoComputeP2pChannels).
P2P_CTAS = (1, 2, 4, 8, 16, 32)
# Every communicator of this benchmark on a GB200 NVL72 domain logs "32 coll channels, ...,
# 32 p2p channels, 32 p2p channels per peer" at init (NCCL 2.29.7 init.cc), with or without MNNVL.
GB200_COMM_CHANNELS = 32
GB200_P2P_CHANNELS_PER_PEER = 32
# P2P step sizes (init.cc): NCCL_P2P_NVL_CHUNKSIZE within one node; across nodes
# NCCL_P2P_NET_CHUNKSIZE, doubled on ARM Blackwell (GB200 trays have only 4 GPUs per host).
P2P_NVL_CHUNK_BYTES = 512 * 1024
P2P_NET_CHUNK_BYTES = 2 * 128 * 1024
# A send/recv pair is spread over at most NCCL_MAX_DEV_WORK_P2P_PER_BATCH peers per round.
P2P_PEERS_PER_ROUND = 8


def workload_ops(workload: str) -> tuple[str, ...]:
    """Return the collectives issued by a workload, in launch order.

    The late-join workload launches its RS first: the AG joins it later on the same ranks.
    The dual-collective workloads issue two ops concurrently on separate streams/communicators:
    ``ag+ag``/``rs+rs`` use a second AG/RS communicator (``ag2``/``rs2``), and the ``p2p_*``
    workloads pair a collective with a point-to-point send or receive.
    """
    if workload == LATE_JOIN_WORKLOAD:
        return ("rs", "ag")
    if workload == "ag+ag":
        return ("ag", "ag2")
    if workload == "rs+rs":
        return ("rs", "rs2")
    if workload == "ag+read":
        return ("ag", "read")
    if workload == "ag+write":
        return ("ag", "write")
    if workload == "rs+read":
        return ("rs", "read")
    if workload == "rs+write":
        return ("rs", "write")
    if workload not in WORKLOADS:
        raise ValueError(f"unknown workload {workload!r}; expected one of {WORKLOADS}")
    return ("ag", "rs") if workload == "ag+rs" else (workload,)


def has_p2p(workload: str) -> bool:
    """Whether ``workload`` issues a read or write (the late-join and cross workloads never do)."""
    return workload in WORKLOADS and bool(P2P_OPS & set(workload_ops(workload)))


def record_p2p_mode(record: Mapping[str, object]) -> str | None:
    """The P2P mode of a record with a read/write; records from before ``--p2p-modes`` were push."""
    if not has_p2p(str(record["workload"])):
        return None
    return str(record.get("p2p_mode") or DEFAULT_P2P_MODE)


def measured_workloads(algo: str, requested: Sequence[str]) -> list[str]:
    """The requested workloads that ``algo`` measures (``ALGO_WORKLOADS``), in the requested order."""
    return [workload for workload in requested if workload in ALGO_WORKLOADS[algo]]


def parse_cta_config(text: str) -> dict[str, int]:
    """Parse one ``--cta-configs`` entry: ``default`` or ``ag=3,rs=6,p2p=32`` (keys from ``CTA_KEYS``).

    A collective value N is the CTA count of every launch of that op (1..``NCCL_MAX_CTAS``; NCCL
    may cap it lower, which the benchmark then reports as an error). A ``p2p`` value is the CTA
    count of one send/recv and must be one of ``P2P_CTAS``.
    """
    text = text.strip()
    if text == DEFAULT_CTA_CONFIG:
        return {}
    ctas: dict[str, int] = {}
    for item in text.split(","):
        key, sep, value = item.strip().partition("=")
        if not sep or key not in CTA_KEYS:
            raise ValueError(f"CTA entry {item!r} must be <op>=<ctas> with <op> in {CTA_KEYS}")
        if key in ctas:
            raise ValueError(f"CTA config {text!r} sets {key} twice")
        if not value.isdigit() or not 1 <= int(value) <= NCCL_MAX_CTAS:
            raise ValueError(f"{key}={value!r}: CTAs must be an integer in 1..{NCCL_MAX_CTAS}")
        if key == "p2p" and int(value) not in P2P_CTAS:
            raise ValueError(f"p2p={value}: NCCL uses power-of-two P2P channel counts, pick one of {P2P_CTAS}")
        ctas[key] = int(value)
    return ctas


def cta_config_name(ctas: Mapping[str, int]) -> str:
    """``default``, or the case-name suffix of a CTA config, e.g. ``cta-ag3-rs6-p2p32``."""
    if not ctas:
        return DEFAULT_CTA_CONFIG
    return "cta-" + "-".join(f"{key}{ctas[key]}" for key in CTA_KEYS if key in ctas)


def p2p_comm_ctas(target: int) -> tuple[int | None, int | None]:
    """``(min_ctas, max_ctas)`` of the P2P communicator that makes a large send/recv use ``target`` CTAs.

    Up to the per-peer limit, ``maxCTAs = target`` caps the communicator at ``target`` channels,
    so a send that is large enough to need more than half of them uses exactly ``target``. The
    per-peer limit itself (32) is only reached when the communicator has twice as many P2P
    channels as a 2-rank send/recv starts with, hence ``minCTAs = 64``. ``None`` keeps NCCL's default.
    See :func:`predicted_p2p_ctas` for what a given size actually gets.
    """
    if target not in P2P_CTAS:
        raise ValueError(f"p2p CTAs must be one of {P2P_CTAS}, got {target}")
    if target >= GB200_P2P_CHANNELS_PER_PEER:
        return NCCL_MAX_CTAS, None
    return None, target


def _pow2_up(value: int) -> int:
    return 1 << max(0, value - 1).bit_length()


def predicted_p2p_ctas(
    nbytes: int, *, n_ranks: int, multi_node: bool, min_ctas: int | None, max_ctas: int | None
) -> int:
    """CTAs (= channels) of one ncclSend/ncclRecv of ``nbytes``, replaying NCCL 2.29.7 on GB200.

    NCCL logs no channel count for send/recv, so the benchmark predicts it; nsys ``gridX`` of
    ``ncclDevKernel_SendRecv`` is the measurement. Steps, with ``min_ctas``/``max_ctas`` the
    communicator's ncclConfig_t (``None`` = NCCL default 1 / 64):

    1. channels = clamp(32 topology channels, max_ctas), then duplicated up to min_ctas (connect.cc).
    2. p2pnChannels = pow2Up(channels); channels per peer = 32, capped by p2pnChannels within a
       node, or halved until every peer fits one round across nodes (paths.cc).
    3. Per op (enqueue.cc): start with the per-peer count halved until ``start * n_ranks <=
       p2pnChannels``, then double while a channel's part exceeds 32 chunks (one chunk across
       nodes) and the per-peer count allows it.

    Args:
        nbytes: Bytes of the send (the benchmark sends one AG shard).
        n_ranks: Ranks of the communicator that runs the send: 2 for torch's lazily created
            pair communicator, the group size for an eagerly initialized group.
        multi_node: Whether that communicator spans more than one tray (NCCL ``nNodes > 1``).
        min_ctas: ncclConfig_t.minCTAs of that communicator.
        max_ctas: ncclConfig_t.maxCTAs of that communicator.
    """
    if nbytes <= 0 or n_ranks < 2:
        raise ValueError("nbytes must be positive and n_ranks at least 2")
    channels = min(GB200_COMM_CHANNELS, max_ctas or NCCL_MAX_CTAS)
    channels = min(max(channels, min_ctas or 1), NCCL_MAX_CTAS)
    p2p_channels = _pow2_up(channels)
    per_peer = GB200_P2P_CHANNELS_PER_PEER
    if multi_node:
        rounds = -(-n_ranks // P2P_PEERS_PER_ROUND)
        while per_peer * rounds > p2p_channels and per_peer > 1:
            per_peer //= 2
    else:
        per_peer = min(p2p_channels, per_peer)
    start = per_peer
    while start * n_ranks > p2p_channels and start > 1:
        start //= 2
    chunk = P2P_NET_CHUNK_BYTES if multi_node else P2P_NVL_CHUNK_BYTES
    min_part, max_part = (chunk // 2, chunk) if multi_node else (chunk // 8, chunk * 32)
    count = min(start, -(-nbytes // min_part))
    part = max(min_part, -(-nbytes // count))
    while part > max_part and count <= per_peer // 2:
        count *= 2
        part = -(-nbytes // count)
    return count


def predicted_p2p_protocol(nbytes: int, *, ctas: int) -> str:
    """``LL`` if NCCL 2.29.7 may send ``nbytes`` over ``ctas`` channels with LL, else ``Simple``.

    ``enqueue.cc`` keeps LL only while ``nbytes <= nChannels * NCCL_P2P_LL_THRESHOLD``. Only Simple
    honors P2P read: LL always writes into the receiver's buffer, so a ``pull`` send must be Simple.
    """
    if nbytes <= 0 or ctas < 1:
        raise ValueError("nbytes and ctas must be positive")
    return "LL" if nbytes <= ctas * P2P_LL_THRESHOLD else "Simple"


def split_groups(total_ranks: int, layout: str) -> tuple[list[int], list[int]]:
    """Split ``range(total_ranks)`` into two equal, disjoint groups.

    Args:
        total_ranks: Number of ranks shared by both groups; must be even.
        layout: ``contiguous`` (first half / second half) or ``interleaved`` (even / odd).

    Returns:
        Global ranks of group A and group B.
    """
    if total_ranks < 2 or total_ranks % 2:
        raise ValueError(f"split needs an even rank count >= 2, got {total_ranks}")
    ranks = list(range(total_ranks))
    half = total_ranks // 2
    if layout == "contiguous":
        return ranks[:half], ranks[half:]
    if layout == "interleaved":
        return ranks[0::2], ranks[1::2]
    raise ValueError(f"unknown split layout {layout!r}; expected one of {SPLIT_LAYOUTS}")


def scenario_name(scenario: str, *, total_ranks: int, layout: str | None = None) -> str:
    """Name one set of communicators: ``single-n8``, ``late-join-n8``, ``split-n16-contiguous`` or
    ``outside-p2p-n4`` (``total_ranks`` is the collective group there; the P2P peer is one more rank)."""
    if scenario in ("single", "late-join", OUTSIDE_P2P_SCENARIO) and layout is None:
        return f"{scenario}-n{total_ranks}"
    if scenario == "split" and layout in SPLIT_LAYOUTS:
        return f"split-n{total_ranks}-{layout}"
    raise ValueError(f"unknown scenario {scenario!r} with layout {layout!r}")


def case_name(
    scenario: str,
    *,
    part: str | None,
    workload: str,
    dtype: str,
    size_mib: int,
    cta_config: str = DEFAULT_CTA_CONFIG,
    p2p_mode: str | None = None,
) -> str:
    """Name one test, e.g. ``all_gather+reduce_scatter-1024MiB-bf16-split-n16-contiguous-AB-concurrent``.

    Prefixed with the algorithm (``case_dir_name``) it is the case directory and the stem of every
    per-rank file in it; ``<algo>/<case>`` is the outer NVTX range of the case's nsys report.
    It must stay a single path component. ``size_mib`` is the full tensor: AG output / RS input /
    AR input. The cross part takes ``CROSS_WORKLOAD`` (``agA-rsB``), every other case one of ``WORKLOADS``
    or ``LATE_JOIN_WORKLOAD``, whose ``size_mib`` is the AG size. A read/write workload in ``pull``
    mode gets ``-pull`` (``push``, NCCL's default, adds nothing), and a CTA config other than the
    default (:func:`cta_config_name`) is appended last, e.g. ``...-single-n4-pull-cta-ag3-p2p32``.
    """
    if part is not None and part not in SPLIT_PARTS:
        raise ValueError(f"unknown split part {part!r}; expected one of {SPLIT_PARTS}")
    if (part == CROSS_PART) != (workload == CROSS_WORKLOAD):
        raise ValueError(f"workload {CROSS_WORKLOAD!r} belongs to part {CROSS_PART!r} only, got {part!r}/{workload!r}")
    if workload != CROSS_WORKLOAD:
        workload_ops(workload)
    if p2p_mode is not None and p2p_mode not in P2P_MODES:
        raise ValueError(f"unknown P2P mode {p2p_mode!r}; expected one of {P2P_MODES}")
    if p2p_mode is not None and not has_p2p(workload):
        raise ValueError(f"workload {workload!r} has no read/write, so it takes no P2P mode")
    suffix = scenario if part is None else f"{scenario}-{part}"
    if p2p_mode == "pull":
        suffix += "-pull"
    if cta_config != DEFAULT_CTA_CONFIG:
        if not cta_config.startswith("cta-"):
            raise ValueError(f"unknown CTA config name {cta_config!r}; expected default or cta-...")
        suffix += f"-{cta_config}"
    return f"{WORKLOAD_NAMES[workload]}-{size_mib}MiB-{dtype}-{suffix}"


def case_dir_name(algo: str, case: str) -> str:
    """The case directory, e.g. ``nvls-all_reduce-1024MiB-bf16-single-n16``: algorithm first,
    so a directory or report copied out of the run still says what it measured."""
    return f"{algo}-{case}"


def rank_file(algo: str, name: str, *, rank: int, world_size: int, suffix: str) -> str:
    """Per-rank file name ``<algo>-<name>.rank_<RR><suffix>``, zero-padded so files sort by rank."""
    width = max(2, len(str(world_size - 1)))
    return f"{case_dir_name(algo, name)}.rank_{rank:0{width}d}{suffix}"


def alg_bandwidth_gb_s(*, total_bytes: int, seconds: float) -> float:
    """Full-tensor payload rate: AG output bytes, RS input bytes or AR input bytes per second."""
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    return total_bytes / seconds / BYTES_PER_GB


def bus_bandwidth_gb_s(*, total_bytes: int, group_size: int, seconds: float) -> float:
    """nccl-tests bus bandwidth: ``total_bytes * (n - 1) / n / t`` for AG/RS, twice that for AR.

    An all-reduce of ``total_bytes`` per rank sends ``(n - 1) / n`` of them and receives the same
    amount back, so its bus traffic is ``2 * total_bytes * (n - 1) / n`` -- the nccl-tests
    all-reduce factor. AG and RS move each payload byte once per rank.
    """
    if group_size < 2:
        raise ValueError("group_size must be at least 2")
    return alg_bandwidth_gb_s(total_bytes=total_bytes, seconds=seconds) * (group_size - 1) / group_size


def op_bus_bandwidth_gb_s(*, op: str, total_bytes: int, group_size: int, seconds: float) -> float:
    """Bus bandwidth of one op of a workload; all-reduce counts both directions."""
    if op not in OPS:
        raise ValueError(f"unknown op {op!r}; expected one of {OPS}")
    bus = bus_bandwidth_gb_s(total_bytes=total_bytes, group_size=group_size, seconds=seconds)
    return 2 * bus if op == "ar" else bus


def per_rank_bytes(op: str, size_bytes: int, group_size: int) -> int:
    """Bytes one rank moves in one launch: an AG/RS shard, or the whole AR tensor.

    ``size_bytes`` is the full tensor: the AG output and the RS input per rank. An all-reduce
    has no sharding, so every rank reduces the whole tensor.
    """
    if op not in OPS:
        raise ValueError(f"unknown op {op!r}; expected one of {OPS}")
    return size_bytes if op == "ar" else size_bytes // group_size


def trial_concurrency(spans_by_group: Sequence[Sequence[float]]) -> list[float]:
    """Per trial, ``min / max`` of the groups' times measured from the shared start gate.

    ``spans_by_group[g][t]`` is group ``g``'s slowest-rank time in trial ``t``. Because every
    group starts at the gate, the shorter group was running during exactly that share of the
    longer group's time: 1 means both groups moved data for the whole trial.
    """
    if len(spans_by_group) < 2:
        raise ValueError("concurrency needs at least two groups")
    return [min(spans) / max(spans) for spans in zip(*spans_by_group, strict=True)]


def overlap_metrics(*, ag_s: float, rs_s: float, both_s: float) -> dict[str, float]:
    """Compare a concurrent AG+RS makespan against the isolated AG and RS times.

    ``speedup`` is 1 for fully serialized execution and ``(ag+rs)/max(ag,rs)`` for a perfect
    overlap. ``overlap_fraction`` is 1 for a perfect overlap of the shorter op; it can be
    negative when running both is slower than running them one after another.
    """
    if min(ag_s, rs_s, both_s) <= 0:
        raise ValueError("all times must be positive")
    serial_s = ag_s + rs_s
    return {
        "serial_ms": serial_s * 1e3,
        "ideal_ms": max(ag_s, rs_s) * 1e3,
        "speedup": serial_s / both_s,
        "overlap_fraction": (serial_s - both_s) / min(ag_s, rs_s),
    }


def op_size_bytes(record: dict[str, object], op: str) -> int:
    """Full tensor of ``op`` in a record: ``<op>_size_bytes`` if set (late-join RS), else ``size_bytes``."""
    return int(record.get(f"{op}_size_bytes") or record["size_bytes"])  # type: ignore[arg-type]


def late_join_metrics(
    *,
    ag_bytes: int,
    rs_bytes: int,
    ag_iso_s: float,
    rs_iso_s: float,
    ag_begin_s: float,
    ag_end_s: float,
    rs_end_s: float,
) -> dict[str, float | bool | None]:
    """Split a late-join iteration into RS alone / AG and RS together / the rest, and rate each op.

    Times are measured from the iteration start, when the RS is launched; the AG is released at
    ``ag_begin_s``. The overlap window is ``[ag_begin, min(ag_end, rs_end)]``. Outside it each op is
    assumed to run at its isolated rate (``bytes / iso time``), so the bytes it moved inside the
    window are the rest. ``*_rel_speed`` is that in-window rate over the isolated rate;
    ``overlap_capacity`` is their sum: 2 means both ran at full speed together (they do not share
    a bottleneck), 1 means they time-shared one. ``ag_share`` is the AG's part of that capacity.

    Without an overlap window (the AG started after the RS ended) the in-window values are ``None``.
    """
    if min(ag_iso_s, rs_iso_s, rs_end_s) <= 0 or ag_end_s <= ag_begin_s or ag_begin_s < 0:
        raise ValueError("times must be positive and the AG must end after it begins")
    ag_rate, rs_rate = ag_bytes / ag_iso_s, rs_bytes / rs_iso_s
    ag_s = ag_end_s - ag_begin_s
    overlap_s = min(ag_end_s, rs_end_s) - ag_begin_s
    result: dict[str, float | bool | None] = {
        "join_fraction": ag_begin_s / rs_iso_s,
        "ag_join_ms": ag_s * 1e3,
        "ag_slowdown": ag_s / ag_iso_s,
        "rs_slowdown": rs_end_s / rs_iso_s,
        "ag_inside_rs": ag_end_s <= rs_end_s,
        "overlap_ms": max(overlap_s, 0.0) * 1e3,
        "ag_algbw_overlap_gb_s": None,
        "rs_algbw_overlap_gb_s": None,
        "ag_rel_speed": None,
        "rs_rel_speed": None,
        "overlap_capacity": None,
        "ag_share": None,
    }
    if overlap_s <= 0:
        return result
    ag_in = max(ag_bytes - ag_rate * (ag_s - overlap_s), 0.0)
    rs_in = max(rs_bytes - rs_rate * (rs_end_s - overlap_s), 0.0)
    ag_rel, rs_rel = ag_in / overlap_s / ag_rate, rs_in / overlap_s / rs_rate
    capacity = ag_rel + rs_rel
    result.update(
        {
            "ag_algbw_overlap_gb_s": ag_in / overlap_s / BYTES_PER_GB,
            "rs_algbw_overlap_gb_s": rs_in / overlap_s / BYTES_PER_GB,
            "ag_rel_speed": ag_rel,
            "rs_rel_speed": rs_rel,
            "overlap_capacity": capacity,
            "ag_share": ag_rel / capacity if capacity > 0 else None,
        }
    )
    return result


FUNC_OPS = {"AllGather": "ag", "ReduceScatter": "rs", "AllReduce": "ar"}
OP_LABELS = {"ag": "AG", "rs": "RS", "ar": "AR"}
# Benchmark dtype -> the type suffix NCCL 2.29.7 puts in generated kernel names (src/device/generate.py).
NCCL_KERNEL_TYPES = {"bf16": "bf16", "fp32": "f32"}


def kernel_name(func: str, implementation: str, dtype: str) -> str:
    """The CUDA function that did the work for one implementation read from the NCCL TUNING log.

    Names follow NCCL 2.29.7's generators. The symmetric path launches one kernel per variant,
    ``ncclSymkDevKernel_<Coll>_<algo>[_sum_<ty>]`` (src/device/symmetric/generate.py), so the
    launched kernel names the variant. The classic path launches a single specialized kernel per
    collective (``ncclDevKernel_<Coll>[_Sum_<ty>]_RING_LL``, ``best_kernel`` in
    src/device/generate.py) that dispatches on its funcId to ``ncclDevFunc_<Coll>[_Sum_<ty>]_<ALGO>_<PROTO>``;
    that device function is what ran, so it is the name reported here.

    Args:
        func: NCCL function name as logged, e.g. ``AllGather``.
        implementation: ``<ALGO>/<PROTO>``, ``Symmetric/<kernel>``, ``CopyEngine`` or
            ``CopyEngine+NVLSSync`` (CE whose ready/complete flags go through multicast).
        dtype: Benchmark dtype, ``bf16`` or ``fp32``.
    """
    ty = NCCL_KERNEL_TYPES.get(dtype, dtype)
    reduces = func != "AllGather"
    family, _, detail = implementation.partition("/")
    if family == "Symmetric":
        algo = detail.removeprefix(f"{func}_")
        return f"ncclSymkDevKernel_{func}_{algo}" + (f"_sum_{ty}" if reduces else "")
    if family.startswith("CopyEngine"):
        return f"{family}（cudaMemcpy，无 kernel）"
    return f"ncclDevFunc_{func}" + (f"_Sum_{ty}" if reduces else "") + f"_{family.upper()}_{detail.upper()}"


def record_kernels(record: dict[str, object]) -> dict[str, str]:
    """Map each op of a record to the kernel(s) its checked ``nccl_impl`` evidence names.

    ``nccl_impl`` is written by the benchmark after it checked every launch of the case in the
    NCCL log of the group's comm rank 0, e.g. ``AllGather=NVLS/SIMPLE;ReduceScatter=NVLS/SIMPLE``.
    Several implementations of one op (``|``) are all listed. Records without evidence map to
    nothing, so the report shows ``-`` instead of a guess.
    """
    evidence = str(record.get("nccl_impl") or "")
    dtype = str(record.get("dtype") or "")
    kernels: dict[str, str] = {}
    for entry in filter(None, evidence.split(";")):
        func, _, implementations = entry.partition("=")
        if func in FUNC_OPS:
            names = [kernel_name(func, impl, dtype) for impl in implementations.split("|") if impl]
            kernels[FUNC_OPS[func]] = " \\| ".join(names)
    return kernels


def _kernel_summary(record: dict[str, object]) -> str:
    kernels = record_kernels(record)
    if not kernels:
        return "-"
    if len(kernels) == 1:
        return next(iter(kernels.values()))
    return "; ".join(f"{OP_LABELS[op]} {kernels[op]}" for op in OPS if op in kernels)


def _kernel_cell(op: str, isolated: dict[str, object] | None, pair: dict[str, object] | None = None) -> str:
    """Kernel of ``op`` in its isolated case; the AG+RS pair is added only if it ran another one."""
    alone = record_kernels(isolated).get(op) if isolated else None
    together = record_kernels(pair).get(op) if pair else None
    if together and together != alone:
        return f"{alone or '-'}（pair：{together}）"
    return alone or "-"


def load_records(path: Path) -> list[dict[str, object]]:
    """Read one JSON object per non-empty line."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _key(record: dict[str, object], *fields: str) -> tuple[object, ...]:
    return tuple(record[field] for field in fields)


def _fmt(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---:|" * len(header)]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _single_index(records: Iterable[dict[str, object]]) -> dict[tuple[object, ...], dict[str, object]]:
    return {_key(r, "algo", "group_size", "size_bytes", "workload"): r for r in records if r["scenario"] == "single"}


def single_rows(records: Sequence[dict[str, object]]) -> list[list[str]]:
    """Question 1: isolated AG / RS / AR versus AG and RS issued together on one group."""
    index = _single_index(records)
    rows = []
    for algo, group_size, size_bytes in sorted({k[:3] for k in index}, key=lambda k: (ALGORITHMS.index(k[0]), *k[1:])):
        ag = index.get((algo, group_size, size_bytes, "ag"))
        rs = index.get((algo, group_size, size_bytes, "rs"))
        ar = index.get((algo, group_size, size_bytes, "ar"))
        both = index.get((algo, group_size, size_bytes, "ag+rs"))
        overlap = None
        if ag and rs and both:
            overlap = overlap_metrics(
                ag_s=float(ag["makespan_ms"]) / 1e3,
                rs_s=float(rs["makespan_ms"]) / 1e3,
                both_s=float(both["makespan_ms"]) / 1e3,
            )
        rows.append(
            [
                str(algo),
                str(group_size),
                str(int(size_bytes) // BYTES_PER_MIB),
                _fmt(ag and float(ag["ag_busbw_gb_s"])),
                _fmt(rs and float(rs["rs_busbw_gb_s"])),
                _fmt(ar and float(ar["ar_busbw_gb_s"])),
                _fmt(both and float(both["ag_busbw_gb_s"])),
                _fmt(both and float(both["rs_busbw_gb_s"])),
                _fmt(both and float(both["aggregate_busbw_gb_s"])),
                _fmt(overlap and overlap["speedup"], 2),
                _fmt(overlap and overlap["overlap_fraction"], 2),
                _kernel_cell("ag", ag, both),
                _kernel_cell("rs", rs, both),
                _kernel_cell("ar", ar),
            ]
        )
    return rows


def _best_of(records: Sequence[dict[str, object]]) -> list[list[str]]:
    """Absolute throughput of every family side by side, plus which one won at that shape."""
    index = _single_index(records)
    rows = []
    for group_size, size_bytes, workload in sorted(
        {k[1:] for k in index}, key=lambda k: (k[0], k[1], WORKLOADS.index(str(k[2])))
    ):
        cells: list[str] = []
        best: tuple[float, str] | None = None
        for algo in ALGORITHMS:
            record = index.get((algo, group_size, size_bytes, workload))
            if record is None:
                cells.append("-")
                continue
            bus = float(record["aggregate_busbw_gb_s"])
            cells.append(_fmt(bus))
            if best is None or bus > best[0]:
                best = (bus, algo)
        rows.append(
            [str(group_size), str(int(size_bytes) // BYTES_PER_MIB), str(workload), *cells, best[1] if best else "-"]
        )
    return rows


def algo_rows(records: Sequence[dict[str, object]]) -> list[list[str]]:
    """Question 2: every family against ring, as aggregate bus bandwidth, per shape."""
    index = _single_index(records)
    rows = []
    for group_size, size_bytes, workload in sorted(
        {k[1:] for k in index}, key=lambda k: (k[0], k[1], WORKLOADS.index(str(k[2])))
    ):
        ring = index.get(("ring", group_size, size_bytes, workload))
        ring_bw = ring and float(ring["aggregate_busbw_gb_s"])
        cells: list[str] = []
        for algo in ALGORITHMS:
            if algo == "ring":
                continue
            record = index.get((algo, group_size, size_bytes, workload))
            other_bw = record and float(record["aggregate_busbw_gb_s"])
            cells += [_fmt(other_bw), _fmt(other_bw / ring_bw if ring_bw and other_bw else None, 2)]
        rows.append([str(group_size), str(int(size_bytes) // BYTES_PER_MIB), str(workload), _fmt(ring_bw), *cells])
    return rows


def symm_p2p_rows(records: Sequence[dict[str, object]]) -> list[list[str]]:
    """symm-p2p against ring and against symm, as aggregate bus bandwidth, per shape.

    Ring and the unicast symmetric kernels move the same bytes per rank ((n-1)/n of the tensor for
    AG and RS, twice that for AR), so symm-p2p/ring isolates the cost of ring's n-1 store-and-forward
    steps. symm/symm-p2p is what multicast adds on top of the same windows. Only shapes symm-p2p
    measured appear.
    """
    index = _single_index(records)
    shapes = {k[1:] for k in index if k[0] == "symm-p2p"}
    rows = []
    for group_size, size_bytes, workload in sorted(shapes, key=lambda k: (k[0], k[1], WORKLOADS.index(str(k[2])))):
        ring, p2p, symm = (
            index.get((algo, group_size, size_bytes, workload)) for algo in ("ring", "symm-p2p", "symm")
        )
        ring_bw = ring and float(ring["aggregate_busbw_gb_s"])
        p2p_bw = p2p and float(p2p["aggregate_busbw_gb_s"])
        symm_bw = symm and float(symm["aggregate_busbw_gb_s"])
        rows.append(
            [
                str(group_size),
                str(int(size_bytes) // BYTES_PER_MIB),
                str(workload),
                _fmt(ring_bw),
                _fmt(p2p_bw),
                _fmt(p2p_bw / ring_bw if ring_bw and p2p_bw else None, 2),
                _fmt(symm_bw),
                _fmt(symm_bw / p2p_bw if p2p_bw and symm_bw else None, 2),
            ]
        )
    return rows


# The families that replace symm's AllGather; table 2a compares them with symm.
SYMM_VARIANTS = ("symm-tma", "copy-engine")


def symm_variant_rows(records: Sequence[dict[str, object]]) -> list[list[str]]:
    """TMA and copy-engine AllGather against symm's, as aggregate bus bandwidth, per shape.

    All three share the windows and, in ag+rs, the multicast ReduceScatter_LDMC, so the ratio
    isolates the AllGather implementation. Only shapes one of the variants measured appear.
    """
    index = _single_index(records)
    shapes = {k[1:] for k in index if k[0] in SYMM_VARIANTS}
    rows = []
    for group_size, size_bytes, workload in sorted(shapes, key=lambda k: (k[0], k[1], WORKLOADS.index(str(k[2])))):
        symm = index.get(("symm", group_size, size_bytes, workload))
        symm_bw = symm and float(symm["aggregate_busbw_gb_s"])
        cells: list[str] = []
        for algo in SYMM_VARIANTS:
            record = index.get((algo, group_size, size_bytes, workload))
            bw = record and float(record["aggregate_busbw_gb_s"])
            cells += [_fmt(bw), _fmt(bw / symm_bw if symm_bw and bw else None, 2)]
        rows.append([str(group_size), str(int(size_bytes) // BYTES_PER_MIB), str(workload), _fmt(symm_bw), *cells])
    return rows


def kernel_rows(records: Sequence[dict[str, object]]) -> list[list[str]]:
    """Question 2, second half: which concrete kernel each family actually ran.

    The kernel name comes from NCCL's own log lines: ``Algo ... proto ...`` for the classic
    collectives and ``[Symmetric]: ... -> Kernel ...`` for the window path. ``nccl_impl`` is
    the value the benchmark checked against every launch of the case; older records only
    carry the last line seen. An empty cell means the log line was not found, so the case
    must not be reported as that family.
    """
    rows = []
    keys = {(str(r["algo"]), str(r["group_name"])) for r in records}
    for algo, group_name in sorted(keys, key=lambda k: (ALGORITHMS.index(k[0]), k[1])):
        mine = [r for r in records if r["algo"] == algo and r["group_name"] == group_name]
        evidence = {
            str(found) for r in mine if (found := r.get("nccl_impl") or r.get("nccl_kernel") or r.get("nccl_algo"))
        }
        n_nodes = sorted({int(r["n_nodes"]) for r in mine if r.get("n_nodes") is not None})
        rows.append([algo, group_name, ",".join(map(str, n_nodes)) or "-", "; ".join(sorted(evidence)) or "-"])
    return rows


def _half_group_reference(
    single: dict[tuple[object, ...], dict[str, object]],
    *,
    algo: object,
    total_ranks: object,
    size_bytes: object,
    workload: str,
) -> dict[str, object] | None:
    """The single-group record a half-group of a ``total_ranks`` split is compared with."""
    return single.get((algo, int(total_ranks) // 2, size_bytes, workload))  # type: ignore[arg-type]


def split_rows(records: Sequence[dict[str, object]]) -> list[list[str]]:
    """Question 3: both half-groups at the same time versus one group of the same size alone.

    The isolated reference of either half of a ``T``-rank split is the single group of ``T/2``
    ranks with the same size and workload; ``slowdown = T_concurrent / T_single``.
    """
    single = _single_index(records)
    fields = ("algo", "total_ranks", "layout", "size_bytes", "workload", "group_index")
    index = {_key(r, *fields): r for r in records if r["scenario"] == "split" and r["phase"] == "concurrent"}
    rows = []
    for key in sorted(
        index,
        key=lambda k: (ALGORITHMS.index(str(k[0])), k[1], k[2], k[3], WORKLOADS.index(str(k[4])), k[5]),
    ):
        algo, total_ranks, layout, size_bytes, workload, group_index = key
        concurrent = index[key]
        isolated = _half_group_reference(
            single, algo=algo, total_ranks=total_ranks, size_bytes=size_bytes, workload=str(workload)
        )
        iso_bw = isolated and float(isolated["aggregate_busbw_gb_s"])
        conc_bw = float(concurrent["aggregate_busbw_gb_s"])
        slowdown = None
        if isolated:
            slowdown = float(concurrent["makespan_ms"]) / float(isolated["makespan_ms"])
        rows.append(
            [
                str(algo),
                str(total_ranks),
                str(layout),
                str(int(size_bytes) // BYTES_PER_MIB),
                str(workload),
                "AB"[int(group_index)],
                _fmt(iso_bw),
                _fmt(conc_bw),
                _fmt(slowdown, 3),
                _fmt(concurrent and _optional_float(concurrent.get("concurrency")), 3),
            ]
        )
    return rows


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)  # type: ignore[arg-type]


def cross_rows(records: Sequence[dict[str, object]]) -> list[list[str]]:
    """Question 4: group A all-gathers while group B reduce-scatters, each against the op alone.

    The isolated references are the single group of ``T/2`` ranks running ``ag`` (for A) and
    ``rs`` (for B) at the same size; ``slowdown = T_cross / T_single`` of that op.
    """
    single = _single_index(records)
    fields = ("algo", "total_ranks", "layout", "size_bytes", "phase", "group_index", "workload")
    index = {_key(r, *fields): r for r in records if r["scenario"] == "split"}
    shapes = {key[:4] for key in index if key[4] == "cross"}
    rows = []
    for shape in sorted(shapes, key=lambda k: (ALGORITHMS.index(str(k[0])), *k[1:])):
        cells: list[str] = []
        cross_bw = 0.0
        for group_index, workload in enumerate(CROSS_GROUP_WORKLOADS):
            cross = index.get((*shape, "cross", group_index, workload))
            isolated = _half_group_reference(
                single, algo=shape[0], total_ranks=shape[1], size_bytes=shape[3], workload=workload
            )
            slowdown = None
            if cross and isolated:
                slowdown = float(cross["makespan_ms"]) / float(isolated["makespan_ms"])
            if cross:
                cross_bw += float(cross["aggregate_busbw_gb_s"])
            cells += [
                _fmt(isolated and float(isolated["aggregate_busbw_gb_s"])),
                _fmt(cross and float(cross["aggregate_busbw_gb_s"])),
                _fmt(cross and float(cross["makespan_ms"]), 3),
                _fmt(slowdown, 3),
            ]
        any_cross = index.get((*shape, "cross", 0, CROSS_GROUP_WORKLOADS[0])) or index.get(
            (*shape, "cross", 1, CROSS_GROUP_WORKLOADS[1])
        )
        algo, total_ranks, layout, size_bytes = shape
        rows.append(
            [
                str(algo),
                str(total_ranks),
                str(layout),
                str(int(size_bytes) // BYTES_PER_MIB),
                *cells,
                _fmt(cross_bw),
                _fmt(any_cross and _optional_float(any_cross.get("concurrency")), 3),
            ]
        )
    return rows


def late_join_rows(records: Sequence[dict[str, object]]) -> list[list[str]]:
    """Question 5: an AG joining a running RS of ``LATE_JOIN_RS_SCALE`` times its size on the same ranks.

    The benchmark stores the isolated references and ``late_join_metrics`` in each join record;
    the bandwidths are algbw (full tensor / time), the unit of documents/gb200/03-*.
    """
    joins = [r for r in records if r["scenario"] == "late-join" and r.get("phase") == "join"]
    rows = []
    for r in sorted(joins, key=lambda r: (ALGORITHMS.index(str(r["algo"])), r["group_size"], r["size_bytes"])):
        ag_bytes, rs_bytes = op_size_bytes(r, "ag"), op_size_bytes(r, "rs")
        ag_iso_ms, rs_iso_ms = float(r["ag_iso_ms"]), float(r["rs_iso_ms"])  # type: ignore[arg-type]
        rows.append(
            [
                str(r["algo"]),
                str(r["group_size"]),
                str(ag_bytes // BYTES_PER_MIB),
                str(rs_bytes // BYTES_PER_MIB),
                _fmt(ag_iso_ms, 3),
                _fmt(rs_iso_ms, 3),
                # --join-fraction 0 records no ag_begin_ms: the AG is launched with the RS.
                f"{_fmt(float(r.get('ag_begin_ms', 0.0)), 3)} ({_fmt(_optional_float(r.get('join_fraction')), 2)})",  # type: ignore[arg-type]
                _fmt(_optional_float(r.get("ag_join_ms")), 3),
                _fmt(_optional_float(r.get("ag_slowdown")), 2),
                _fmt(float(r["rs_ms"]), 3),  # type: ignore[arg-type]
                _fmt(_optional_float(r.get("rs_slowdown")), 2),
                _fmt(alg_bandwidth_gb_s(total_bytes=ag_bytes, seconds=ag_iso_ms / 1e3)),
                _fmt(_optional_float(r.get("ag_algbw_overlap_gb_s"))),
                _fmt(alg_bandwidth_gb_s(total_bytes=rs_bytes, seconds=rs_iso_ms / 1e3)),
                _fmt(_optional_float(r.get("rs_algbw_overlap_gb_s"))),
                _fmt(_optional_float(r.get("overlap_capacity")), 2),
                _fmt(_optional_float(r.get("ag_share")), 2),
                "yes" if r.get("ag_inside_rs") else "no",
                "†" if r.get("host_bound") else "",
            ]
        )
    return rows


def record_cta_config(record: Mapping[str, object]) -> str:
    """The ``--cta-configs`` entry of a record; records from before the option are ``default``."""
    return str(record.get("cta_config") or DEFAULT_CTA_CONFIG)


# Op columns of table 6, in launch order of every workload that has them.
CTA_TABLE_OPS = ("ag", "ag2", "rs", "rs2", "ar", "read", "write")


def cta_rows(records: Sequence[dict[str, object]]) -> list[list[str]]:
    """Table 6: every non-default ``--cta-configs`` case against the default config of the same case.

    For each op of the workload: its CTAs (NCCL's TUNING log for collectives, the prediction for
    a send/recv), its per-iteration time, and that time relative to the same op in the default
    config. The pairing key is (algo, scenario, phase, ranks, layout, group, size, workload).
    """
    fields = ("algo", "scenario", "phase", "group_size", "layout", "group_index", "size_bytes", "workload")

    def key(record: Mapping[str, object]) -> tuple[object, ...]:
        return tuple(record.get(field) for field in fields)

    defaults = {key(r): r for r in records if record_cta_config(r) == DEFAULT_CTA_CONFIG}
    if len({record_cta_config(r) for r in records}) < 2:
        return []
    rows = []
    for r in sorted(
        records,
        key=lambda r: (ALGORITHMS.index(str(r["algo"])), *(str(r.get(f)) for f in fields[1:]), record_cta_config(r)),
    ):
        base = defaults.get(key(r))
        cells: list[str] = []
        for op in CTA_TABLE_OPS:
            if op not in workload_ops(str(r["workload"])):
                cells.append("-")
                continue
            ctas = (r.get("observed_ctas") or {}).get(op)  # type: ignore[union-attr]
            if op in ("read", "write"):
                ctas = r.get("predicted_p2p_ctas")
                ctas = None if ctas is None else f"~{ctas}"
            ms = _optional_float(r.get(f"{op}_ms"))
            base_ms = _optional_float(base.get(f"{op}_ms")) if base else None
            ratio = f" ({ms / base_ms:.2f}x)" if ms and base_ms and base is not r else ""
            cells.append(f"{ctas or '?'} CTA / {_fmt(ms, 3)} ms{ratio}")
        makespan = float(r["makespan_ms"])  # type: ignore[arg-type]
        base_makespan = float(base["makespan_ms"]) if base else None  # type: ignore[arg-type]
        rows.append(
            [
                str(r["algo"]),
                str(r["scenario"]),
                str(r["group_size"]),
                str(int(r["size_bytes"]) // BYTES_PER_MIB),  # type: ignore[arg-type]
                str(r["workload"]),
                str(r.get("group_name") or "-"),
                record_cta_config(r),
                *cells,
                _fmt(makespan, 3),
                _fmt(makespan / base_makespan if base_makespan and base is not r else None, 2),
                _fmt(float(r["aggregate_busbw_gb_s"])),  # type: ignore[arg-type]
            ]
        )
    return rows


def completion_rows(records: Sequence[dict[str, object]]) -> list[list[str]]:
    """Table 0: every test and group with its completion time per iteration over the trials.

    ``mean`` is the arithmetic mean over trials of the slowest rank's per-iteration time; the
    bandwidth columns in this table are computed from that mean, every op of the workload
    contributing its own bus traffic: ``size * (n-1)/n`` for AG and RS, twice that for AR.
    """
    rows = []
    ordered = sorted(
        records,
        key=lambda r: (ALGORITHMS.index(str(r["algo"])), str(r.get("case") or ""), str(r["group_name"])),
    )
    for r in ordered:
        trials = [float(t) for t in r["trial_makespan_ms"]]  # type: ignore[union-attr]
        mean_ms = float(r.get("mean_makespan_ms") or sum(trials) / len(trials))  # type: ignore[arg-type]
        size_bytes, group_size = int(r["size_bytes"]), int(r["group_size"])  # type: ignore[arg-type]
        ops = workload_ops(str(r["workload"]))
        algbw = busbw = 0.0
        for op in ops:
            # Same accounting as the benchmark's per-op columns: ag2/rs2 are a second AG/RS, and a
            # read/write moves one AG shard between two ranks (algbw = busbw, no fan-out).
            if op in ("read", "write"):
                shard = alg_bandwidth_gb_s(
                    total_bytes=per_rank_bytes("ag", size_bytes, group_size), seconds=mean_ms / 1e3
                )
                algbw += shard
                busbw += shard
                continue
            canonical = {"ag2": "ag", "rs2": "rs"}.get(op, op)
            total = op_size_bytes(r, canonical)
            algbw += alg_bandwidth_gb_s(total_bytes=total, seconds=mean_ms / 1e3)
            busbw += op_bus_bandwidth_gb_s(
                op=canonical, total_bytes=total, group_size=group_size, seconds=mean_ms / 1e3
            )
        rows.append(
            [
                str(r["algo"]),
                str(r.get("case") or "-"),
                str(r["group_name"]),
                str(r.get("phase") or "-"),
                str(group_size),
                str(r["workload"]),
                str(size_bytes // BYTES_PER_MIB),
                f"{len(trials)}x{r['iters']}",
                _fmt(mean_ms, 3),
                _fmt(float(r["makespan_ms"]), 3),  # type: ignore[arg-type]
                _fmt(min(trials), 3),
                _fmt(max(trials), 3),
                _fmt(algbw),
                _fmt(busbw),
                _fmt(_optional_float(r.get("concurrency")), 3),
                _kernel_summary(r),
            ]
        )
    return rows


def render_report(records: Sequence[dict[str, object]]) -> str:
    """Render the comparison tables as Markdown.

    Table 0 lists every case; tables 1-5 compare implementations at NCCL's own CTA choice (the
    ``default`` CTA config) and table 6 compares every other ``--cta-configs`` entry with it.
    """
    all_records = list(records)
    records = [r for r in all_records if record_cta_config(r) == DEFAULT_CTA_CONFIG]
    profiled = sorted({str(r["algo"]) for r in all_records if r.get("profiled")})
    lines = [
        "# AG / RS / AR: ring vs NVLS vs symmetric memory (register / TMA / copy engine)",
        "",
        "计时：每个 trial 先过一次全 scenario 的 GPU 同步起点（1 元素 all-reduce），再背靠背发 iters 次；",
        "t = (该 rank 所有 op 完成 − 起点) / iters，trial 内取组内最慢 rank。除表 0 外各表取 trials 的中位数。",
        "",
        "换算：size = 完整张量（AG 每卡接收总量 / RS 每卡输入总量 / AR 每卡归约总量），十进制 GB/s：",
        "AG、RS 每卡 shard = size/n，busbw = size·(n−1)/n / t（nccl-tests 定义，= 每卡实际收/发字节数 / t）；",
        "AR 没有分片，每卡归约整块 size，busbw = 2·size·(n−1)/n / t（收、发各算一次，nccl-tests 的 all-reduce 口径）；",
        "ag+rs 的 aggregate busbw = 两个 op 的 busbw 之和 / makespan；不同组的带宽不相加，需要时自己把各组行相加。",
        "concurrency = min(T_A, T_B) / max(T_A, T_B)：两组从同一起点出发，1 表示两组全程同时在跑。",
        "",
    ]
    if profiled:
        lines += [
            f"> 注意：{', '.join(profiled)} 是 `--nsys` / `--ncu` 采集下的结果，含 Nsight 开销"
            "（ncu 还会串行化并重放 trial0 的 kernel），只用于看 trace / kernel 细节，"
            "不要和非 profile 的数字放进同一张结论表。",
            "",
        ]
    lines += [
        "## 0. 每个测试的完成时间",
        "",
        "mean/median/min/max：各 trial（每个 trial 取组内最慢 rank）的每次迭代完成时间，单位 ms；",
        "本表的 algbw/busbw 用 mean 计算。",
        "",
        "kernel：该组 comm rank 0 的 NCCL TUNING 日志里，本 case 每一次 launch（含 warmup）记录的实现，",
        "换算成 NCCL 2.29.7 实际运行的函数名，不是由环境变量推断：",
        "symm 是对称内核 `ncclSymkDevKernel_<Coll>_<algo>`（launch 的就是它）；ring/nvls 是经典路径真正执行的",
        "设备函数 `ncclDevFunc_<Coll>_<ALGO>_<PROTO>`——经典路径每个 collective 只 launch 一个特化 kernel",
        "（如 `ncclDevKernel_AllGather_RING_LL`，nsys 里看到的就是这个名字），再按 funcId 分发到该设备函数，",
        "所以 nvls 的 AG 在 nsys 里也叫 `*_RING_LL`。copy-engine 的 AG 是 cudaMemcpy，没有 kernel。",
        "一个 op 出现多种实现时用 `\\|` 全部列出；`-` 表示没有日志证据。",
        "",
    ]
    lines += _table(
        [
            "algo",
            "case",
            "group",
            "phase",
            "ranks",
            "workload",
            "MiB",
            "trials x iters",
            "mean ms",
            "median ms",
            "min ms",
            "max ms",
            "algbw",
            "busbw",
            "concurrency",
            "kernel",
        ],
        completion_rows(all_records),
    )
    lines += [
        "",
        "## 1. AG、RS、AR 单独与 AG+RS 同时发起",
        "",
        "speedup = (T_AG + T_RS) / T_AG+RS；1 表示完全串行，overlap_fraction = 1 表示短的操作被完全隐藏。",
        "AR busbw 已按 all-reduce 口径乘 2，可直接和 AG/RS 的 busbw 比。",
        "AG/RS/AR kernel：单独跑该 op 那个 case 的实际函数（来源同表 0）；AG+RS 同时发起时若换了实现，括号里注明。",
        "symm-tma、copy-engine 只测 ag 和 ag+rs，所以 RS/AR 列和 speedup、overlap_fraction 为 `-`；"
        "pair 里的 RS 是 `ReduceScatter_LDMC`（见 RS kernel 列的 pair 注明）。",
        "",
    ]
    lines += _table(
        [
            "algo",
            "ranks",
            "MiB",
            "AG busbw",
            "RS busbw",
            "AR busbw",
            "AG in pair",
            "RS in pair",
            "pair aggregate",
            "speedup",
            "overlap_fraction",
            "AG kernel",
            "RS kernel",
            "AR kernel",
        ],
        single_rows(records),
    )
    lines += ["", "## 1b. 各实现的绝对吞吐（aggregate busbw）", ""]
    lines += _table(["ranks", "MiB", "workload", *ALGORITHMS, "best"], _best_of(records))
    lines += [
        "",
        "## 2. 各实现相对 ring（aggregate busbw）",
        "",
        "symm 是 window 注册的对称内存路径，与 ring/NVLS 走不同内核家族；它的绝对列见 1b，",
        "symm-p2p 是同一套对称内核关掉 multicast（`NCCL_NVLS_ENABLE=0`）后的单播 load/store 变体。",
        "上面的比值只是和 ring 放在同一口径下的对照。",
        "",
    ]
    others = [algo for algo in ALGORITHMS if algo != "ring"]
    lines += _table(
        ["ranks", "MiB", "workload", "ring", *(col for algo in others for col in (algo, f"{algo}/ring"))],
        algo_rows(records),
    )
    variant_table = symm_variant_rows(records)
    if variant_table:
        lines += [
            "",
            "### 2a. TMA / copy engine 的 AllGather 相对 symm（aggregate busbw）",
            "",
            "三者都在同一种 window 上跑，ag+rs 里的 RS 也都是 `ReduceScatter_LDMC`，比值只反映 AllGather 实现："
            "symm 是寄存器 `multimem.st` 的对称内核，symm-tma 是 `AllGather_TmaSTMC`，"
            "copy-engine 是 CE 写 multicast 地址。",
            "",
        ]
        lines += _table(
            ["ranks", "MiB", "workload", "symm", *(col for algo in SYMM_VARIANTS for col in (algo, f"{algo}/symm"))],
            variant_table,
        )
    kernel_table = kernel_rows(records)
    if kernel_table:
        lines += [
            "",
            "### 2b. 实际执行的内核（来自 NCCL TUNING 日志，非环境变量推断）",
            "",
            "每个 case 的每次 launch 都已核对：ring 只能是 Ring/*，nvls 只能是 NVLS/*，symm 只能是对称内核；"
            "symm-p2p 只能是不带 multicast 的对称内核（`*_ST`、`*_LD`、`*_LL`、`*_TmaST`、`*_TmaLD` 及其 AR 组合）；"
            "symm-tma 的 AG 只能是 `AllGather_TmaSTMC`，copy-engine 的 AG 只能是带 NVLS 同步的 CopyEngine，"
            "二者的 RS 只能是 `ReduceScatter_LDMC`。不符合的 launch 会让该 case 失败，不会出现在这里。",
            "",
        ]
        lines += _table(["algo", "group", "nNodes", "kernel evidence"], kernel_table)
    p2p_table = symm_p2p_rows(records)
    if p2p_table:
        lines += [
            "",
            "### 2c. 对称内核的 P2P 方案：相对 ring 与相对 multicast（aggregate busbw）",
            "",
            "symm-p2p 关掉 multicast，对称内核退回单播 LSA load/store：AG 是 `AllGather_ST`（每卡把自己的 shard "
            "直接写进其余 n−1 卡），RS 是 `ReduceScatter_LD`（每卡从其余 n−1 卡直接读自己那份再归约），"
            "AR 是 `AllReduce_RSxLD_AGxST`（两者串起来），小尺寸是对应的 LL 变体。",
            "ST/LD 下 AG/RS 每卡收发 (n−1)/n·size、AR 每卡 2·(n−1)/n·size，与 ring 完全相同"
            "（LL 变体每 8 字节带 8 字节 flag，线上字节约翻倍；`AllReduce_AGxLL_R` 把整块发给每张卡再本地归约，"
            "每卡收 (n−1)·size，只用于小尺寸），"
            "所以 symm-p2p/ring 只反映 ring 的 n−1 步逐跳转发（每步多一次 HBM 读写和步间同步）与直接全互连的差别；"
            "symm/symm-p2p 是 multicast（`multimem.st` / `multimem.ld_reduce` 让一侧只需 size/n）带来的增益。",
            "",
        ]
        lines += _table(
            ["ranks", "MiB", "workload", "ring", "symm-p2p", "symm-p2p/ring", "symm", "symm/symm-p2p"],
            p2p_table,
        )
    lines += [
        "",
        "## 3. 两个半组同时运行",
        "",
        "T 卡切成两个 T/2 组同时跑同一负载；isolated 取 single 里同样 T/2 卡、同大小、同负载的结果（rank 0..T/2-1）。",
        "slowdown = T_concurrent / T_isolated，1 表示没有相互干扰。",
        "",
    ]
    lines += _table(
        [
            "algo",
            "total",
            "layout",
            "MiB",
            "workload",
            "group",
            "isolated busbw",
            "concurrent busbw",
            "slowdown",
            "concurrency",
        ],
        split_rows(records),
    )
    lines += [
        "",
        "## 4. 两组同时：A 组 all-gather，B 组 reduce-scatter",
        "",
        "isolated 取 single 里 T/2 卡单独跑同一个 op（A 对 ag、B 对 rs）；"
        "slowdown = T_cross / T_isolated；A+B busbw 是两组 busbw 之和。",
        "symm-tma、copy-engine 不单独测 rs，B 的 isolated 为 `-`；B 跑的是 `ReduceScatter_LDMC`，"
        "表 1 里 RS kernel 同为它的 symm 行可作参照。",
        "",
    ]
    lines += _table(
        [
            "algo",
            "total",
            "layout",
            "MiB",
            "A AG isolated busbw",
            "A AG cross busbw",
            "A AG cross ms",
            "A slowdown",
            "B RS isolated busbw",
            "B RS cross busbw",
            "B RS cross ms",
            "B slowdown",
            "A+B cross busbw",
            "concurrency",
        ],
        cross_rows(records),
    )
    join_table = late_join_rows(records)
    if join_table:
        lines += [
            "",
            f"## 5. late join：RS（{LATE_JOIN_RS_SCALE}×S）跑到 `--join-fraction`（默认 {LATE_JOIN_FRACTION:g}）时，同一组 rank 加入 AG（S）",
            "",
            "每次迭代：RS 先发，AG 的 stream 先 GPU sleep `join_fraction × T_RS iso` 再发 AG，迭代结束两条 stream 汇合；",
            "`--join-fraction 0` 不 sleep，AG 与 RS 每次迭代同时发射（join 列为 0）；",
            "isolated 是同一 scenario 里用同样的逐迭代计时单独跑的 AG(S)、RS(4S)。时间都从迭代起点（RS 发射）算，",
            "trial 内取组内最慢 rank，再取 trial 中位数。带宽是 algbw = 张量大小 / 时间（GB/s，10^9）。",
            "",
            "- join：AG 被放行的时刻（括号里是相对 T_RS iso 的比例）；T_AG join：AG 从放行到完成的时间；",
            "  T_RS join：RS 从迭代起点到完成的时间。slowdown = join 下的时间 / isolated。",
            "- overlap 窗口 = [AG 放行, min(AG 结束, RS 结束)]。窗口外假设各自按 isolated 速率跑，窗口内的字节数是剩下的，",
            "  由此得到窗口内的 AG / RS algbw（是估算，nsys 的 NVLink 采样是直接证据）。",
            "- capacity = 窗口内 AG 速率 / AG isolated 速率 + RS 速率 / RS isolated 速率：2 表示两者互不影响、都跑满，",
            "  1 表示两者在分时共用同一瓶颈；AG share = AG 那一项 / capacity，0.5 是平分。",
            "- AG in RS：AG 是否在 RS 结束前完成（否则窗口只覆盖 AG 的一部分）；† 表示 host 入队比 GPU sleep 慢"
            "（`enqueue_ms > --launch-delay-ms`），AG 可能晚于放行时刻才被发射，该行不可靠。",
            "",
        ]
        lines += _table(
            [
                "algo",
                "ranks",
                "AG MiB",
                "RS MiB",
                "T_AG iso ms",
                "T_RS iso ms",
                "join ms (frac)",
                "T_AG join ms",
                "AG slowdown",
                "T_RS join ms",
                "RS slowdown",
                "AG algbw iso",
                "AG algbw overlap",
                "RS algbw iso",
                "RS algbw overlap",
                "capacity",
                "AG share",
                "AG in RS",
                "host",
            ],
            join_table,
        )
    cta_table = cta_rows(all_records)
    if cta_table:
        lines += [
            "",
            "## 6. 不同 CTA 配置（`--cta-configs`）相对 default",
            "",
            "表 1–5 只用 `default`（NCCL 自己选 CTA 数）的结果；本表列出其余每个 CTA 配置的 case，和同一 case 的",
            "default 结果配对（algo、scenario、phase、组、大小、负载都相同）。每个 op 一格：`CTA 数 / 每次迭代 ms (相对 default)`。",
            "集合通信的 CTA 数是该组 comm rank 0 的 TUNING 日志里每次 launch 的实测值（多种时用 `|` 列出）；",
            "send/recv 的 NCCL 不打日志，`~N` 是按 NCCL 2.29.7 源码推算的值，以 nsys 里 `ncclDevKernel_SendRecv` 的",
            "`gridX`（`nsys_kernels.md` 的 grid 列）为准。makespan / default < 1 表示整体变快。",
            "",
        ]
        lines += _table(
            [
                "algo",
                "scenario",
                "ranks",
                "MiB",
                "workload",
                "group",
                "cta config",
                *(f"{op}" for op in CTA_TABLE_OPS),
                "makespan ms",
                "makespan / default",
                "aggregate busbw",
            ],
            cta_table,
        )
    return "\n".join(lines) + "\n"


def _parse_input(value: str) -> tuple[str, Path]:
    algo, sep, path = value.partition("=")
    if not sep or algo not in ALGORITHMS:
        raise argparse.ArgumentTypeError(f"expected ALGO=PATH with ALGO in {ALGORITHMS}, got {value!r}")
    return algo, Path(path)


def main(argv: Sequence[str] | None = None) -> None:
    """Merge per-algorithm result files and write the Markdown comparison."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=_parse_input, action="append", required=True, help="ALGO=results.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="Markdown report path")
    args = parser.parse_args(argv)

    records: list[dict[str, object]] = []
    for algo, path in args.input:
        loaded = load_records(path)
        mismatched = {str(r["algo"]) for r in loaded} - {algo}
        if mismatched:
            raise ValueError(f"{path} contains records for {sorted(mismatched)}, expected only {algo}")
        records.extend(loaded)
    args.output.write_text(render_report(records), encoding="utf-8")
    LOGGER.info("wrote %s (%d records)", args.output, len(records))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
