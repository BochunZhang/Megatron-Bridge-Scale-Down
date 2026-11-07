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
"""Helpers of ``run_all_gather_nccl_2_32_3_ring_vs_symm_p2p.sh``: AllGather, NCCL 2.32.3, ring vs symm-p2p.

The launcher runs the ring pass first, then the symm-p2p pass with every symmetric-kernel
communicator pinned (``minCTAs = maxCTAs``) to the CTA count ring used for the same group size
(ranks) and message size.

* ``ring-ctas``: the CTA count of ring's AllGather per (ranks, size), from ``ring/results.jsonl``
  (``observed_ctas``, parsed from comm rank 0's TUNING lines). Only global rank 0 writes TUNING
  lines, so node 0 reads the file and hands the counts to the other nodes through a TCPStore;
  every node writes them as JSON ``[[ranks, size_mib, ctas], ...]`` to ``--out``.
* ``summary``: pairs every (ranks, size) ring case with the symm-p2p case that ran at ring's CTA
  count, checks the CTA count and the symmetric kernel, and writes ``ring_vs_symm_p2p_all_gather.md``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
import time
from pathlib import Path

from collective_metrics import cta_config_name


logger = logging.getLogger(__name__)

RING = "ring"
SYMM_P2P = "symm-p2p"
WORKLOAD = "ag"
# NCCL_SYM_TMA_ENABLE=0 masks AllGather_TmaST, so the cost model is left with AllGather_ST.
P2P_KERNELS = {"st": "AllGather_ST", "tma": "AllGather_TmaST"}
SUMMARY_NAME = "ring_vs_symm_p2p_all_gather.md"
STORE_KEY = "ring_ag_ctas"


Key = tuple[int, int]  # (group size in ranks, full AllGather size in MiB)


def _ints(text: str) -> list[int]:
    return [int(value) for value in text.split(",")]


def _key(record: dict[str, object]) -> Key:
    return int(record["group_size"]), int(record["size_mib"])


def _single_ag_records(path: Path) -> list[dict[str, object]]:
    """The AllGather records of the single groups (ranks ``[0, n)``, ``single`` scenario) in ``path``."""
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; the pass did not finish")
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("workload") == WORKLOAD and "-single-n" in str(record.get("case")):
            records.append(record)
    return records


def _one_count(record: dict[str, object]) -> int:
    """The single AllGather CTA count NCCL logged for ``record``; several counts are an error."""
    observed = (record.get("observed_ctas") or {}).get(WORKLOAD)
    if observed is None or not str(observed).isdigit():
        raise ValueError(f"{record['case']}: expected one AllGather CTA count, NCCL logged {observed!r}")
    return int(observed)


def read_ring_ctas(output_dir: Path, group_sizes: list[int], sizes_mib: list[int]) -> dict[Key, int]:
    """``{(ranks, size_mib): CTAs}`` of ring's AllGather at NCCL's own (default) CTA choice."""
    found: dict[Key, int] = {}
    for record in _single_ag_records(output_dir / RING / "results.jsonl"):
        if record.get("cta_config") != "default":
            continue
        algo = str(record.get("nccl_algo") or "")
        if "AllGather=RING/" not in algo:
            raise ValueError(f"{record['case']}: ring pass ran {algo!r}, not RING")
        found[_key(record)] = _one_count(record)
    wanted = [(ranks, size) for ranks in group_sizes for size in sizes_mib]
    missing = [key for key in wanted if key not in found]
    if missing:
        raise ValueError(f"ring/results.jsonl has no AllGather case for (ranks, MiB) {missing}")
    return {key: found[key] for key in wanted}


def share_ring_ctas(args: argparse.Namespace) -> None:
    """Node 0 reads ring's CTA counts and publishes them; every node writes them to ``--out``."""
    # torch is only needed for the store; the summary runs without it.
    import torch.distributed as dist

    timeout = datetime.timedelta(minutes=args.timeout_min)
    is_master = args.node_rank == 0
    ctas = (
        [
            [*key, n]
            for key, n in read_ring_ctas(args.output_dir, _ints(args.group_sizes), _ints(args.sizes_mib)).items()
        ]
        if is_master
        else None
    )
    store = dist.TCPStore(
        args.master_addr,
        args.port,
        world_size=args.nnodes,
        is_master=is_master,
        timeout=timeout,
        wait_for_workers=False,
    )
    if is_master:
        store.set(STORE_KEY, json.dumps(ctas))
        deadline = time.monotonic() + timeout.total_seconds()
        while store.add("acks", 0) < args.nnodes - 1:
            if time.monotonic() > deadline:
                raise TimeoutError("timed out waiting for every node to read ring's CTA counts")
            time.sleep(0.2)
    else:
        ctas = json.loads(store.get(STORE_KEY).decode())
        store.add("acks", 1)
    args.out.write_text(json.dumps(ctas), encoding="utf-8")
    logger.info("ring AllGather CTAs per [ranks, MiB, CTAs]: %s", ctas)


def _fmt(value: object, digits: int) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def _ag_algo(record: dict[str, object]) -> str:
    """The AllGather entry of ``nccl_algo`` (``AllGather=RING/SIMPLE;AllReduce=...`` -> ``RING/SIMPLE``)."""
    for entry in str(record.get("nccl_algo") or "").split(";"):
        func, _, algo = entry.partition("=")
        if func == "AllGather":
            return algo
    return "-"


def summarize(args: argparse.Namespace) -> int:
    """Write the ring vs symm-p2p table; return 1 if a pair did not run as required."""
    ring_ctas = read_ring_ctas(args.output_dir, _ints(args.group_sizes), _ints(args.sizes_mib))
    ring = {
        _key(r): r
        for r in _single_ag_records(args.output_dir / RING / "results.jsonl")
        if r.get("cta_config") == "default"
    }
    p2p = {
        (*_key(r), str(r["cta_config"])): r for r in _single_ag_records(args.output_dir / SYMM_P2P / "results.jsonl")
    }
    want_kernel = P2P_KERNELS[args.p2p_kernel]
    rows = [
        "| ranks | MiB | CTAs | ring algo | ring ms | ring busbw | p2p kernel | p2p CTAs | p2p ms | p2p busbw "
        "| p2p/ring time |",
        "| ---: | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    errors: list[str] = []
    for (ranks, size), ctas in ring_ctas.items():
        r = ring[(ranks, size)]
        q = p2p.get((ranks, size, cta_config_name({WORKLOAD: ctas})))
        where = f"{ranks} ranks, {size} MiB"
        if q is None:
            errors.append(f"{where}: no symm-p2p case at ring's {ctas} CTAs")
            continue
        kernel = str(q.get("nccl_kernel") or "")
        if kernel != f"AllGather={want_kernel}":
            errors.append(f"{where}: symm-p2p ran {kernel!r}, expected AllGather={want_kernel}")
        if _one_count(q) != ctas:
            errors.append(f"{where}: symm-p2p ran {_one_count(q)} CTAs, ring {ctas}")
        for record in (r, q):
            if not record.get("check_passed"):
                errors.append(f"{record['case']}: output check failed")
        ratio = float(q["ag_ms"]) / float(r["ag_ms"])
        rows.append(
            f"| {ranks} | {size} | {ctas} | {_ag_algo(r)} | {_fmt(r['ag_ms'], 3)} "
            f"| {_fmt(r.get('ag_busbw_gb_s'), 1)} | {kernel.split('=', 1)[-1]} | {_one_count(q)} "
            f"| {_fmt(q['ag_ms'], 3)} | {_fmt(q.get('ag_busbw_gb_s'), 1)} | {ratio:.2f} |"
        )
    text = "\n".join(
        [
            "# AllGather, NCCL 2.32.3: ring vs symm-p2p at ring's CTA count",
            "",
            f"symm-p2p kernel: {want_kernel} (`--p2p-kernel {args.p2p_kernel}`); busbw in GB/s;"
            " ms = mean time of one AllGather in the median trial.",
            "",
            *rows,
            "",
            *(["## Errors", "", *(f"- {e}" for e in errors), ""] if errors else []),
        ]
    )
    out = args.output_dir / SUMMARY_NAME
    out.write_text(text, encoding="utf-8")
    logger.info("%s\n%s", out, text)
    return 1 if errors else 0


def main() -> int:
    """Command-line entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    share = sub.add_parser("ring-ctas", help="share ring's AllGather CTA count per (ranks, size) with every node")
    share.add_argument("--output-dir", type=Path, required=True)
    share.add_argument("--group-sizes", required=True)
    share.add_argument("--sizes-mib", required=True)
    share.add_argument("--nnodes", type=int, required=True)
    share.add_argument("--node-rank", type=int, required=True)
    share.add_argument("--master-addr", required=True)
    share.add_argument("--port", type=int, required=True)
    share.add_argument("--timeout-min", type=int, required=True)
    share.add_argument("--out", type=Path, required=True)

    summary = sub.add_parser("summary", help="write the ring vs symm-p2p table")
    summary.add_argument("--output-dir", type=Path, required=True)
    summary.add_argument("--group-sizes", required=True)
    summary.add_argument("--sizes-mib", required=True)
    summary.add_argument("--p2p-kernel", choices=sorted(P2P_KERNELS), required=True)

    args = parser.parse_args()
    if args.command == "ring-ctas":
        share_ring_ctas(args)
        return 0
    return summarize(args)


if __name__ == "__main__":
    sys.exit(main())
