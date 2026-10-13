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
"""Pick ``<root>/gpu-num-<W>/run-<NN>`` once and give every node the same directory.

Node 0 takes the next free ``run-NN`` by creating it with ``mkdir`` (atomic, so two jobs
started at the same time cannot share one). With more than one node it then publishes the
path through a ``torch.distributed.TCPStore`` on ``MASTER_ADDR`` and waits until every other
node has read it. Scanning the directory independently on each node would race: a node that
starts after node 0 already created ``run-01`` would pick ``run-02``.

Prints the chosen directory on stdout. torch is only imported for the multi-node handshake.
"""

from __future__ import annotations

import argparse
import re
from datetime import timedelta
from pathlib import Path


RUN_PATTERN = re.compile(r"^run-(\d+)$")
STORE_KEY = "nvls_collective_output_dir"
SHARED_MARKER = ".node0"


def gpu_dir(root: Path, world_size: int) -> Path:
    """``<root>/gpu-num-<world_size>``."""
    return root / f"gpu-num-{world_size}"


def next_run_index(parent: Path) -> int:
    """One past the highest existing ``run-NN`` (gaps are not reused)."""
    if not parent.is_dir():
        return 1
    indices = [int(m.group(1)) for p in parent.iterdir() if (m := RUN_PATTERN.match(p.name)) and p.is_dir()]
    return max(indices, default=0) + 1


def run_name(index: int) -> str:
    """``run-01`` ... ``run-99``, then ``run-100``."""
    return f"run-{index:02d}"


def claim_run_dir(parent: Path) -> Path:
    """Create and return the next ``run-NN``; retry if another job created it first."""
    parent.mkdir(parents=True, exist_ok=True)
    index = next_run_index(parent)
    while True:
        candidate = parent / run_name(index)
        try:
            candidate.mkdir()
        except FileExistsError:
            index += 1
            continue
        (candidate / SHARED_MARKER).write_text("created by node 0\n", encoding="utf-8")
        return candidate


def share_with_nodes(path: Path | None, *, nnodes: int, node_rank: int, host: str, port: int, timeout_s: int) -> Path:
    """Node 0 publishes ``path``; the others receive it. Returns the agreed path on every node."""
    from torch.distributed import TCPStore

    timeout = timedelta(seconds=timeout_s)
    store = TCPStore(host, port, nnodes, node_rank == 0, timeout, wait_for_workers=True)
    if node_rank == 0:
        assert path is not None
        store.set(STORE_KEY, str(path))
        # Keep the server alive until every node has read the key.
        store.wait([f"ack-{rank}" for rank in range(1, nnodes)], timeout)
        return path
    agreed = Path(store.get(STORE_KEY).decode())
    store.set(f"ack-{node_rank}", "1")
    return agreed


def main(argv: list[str] | None = None) -> None:
    """Resolve and print the run directory."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="results/gb200/nvls-collective")
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="TCPStore port; required when --nnodes > 1")
    parser.add_argument("--timeout-s", type=int, default=600)
    parser.add_argument("--dry-run", action="store_true", help="Print the next directory without creating it")
    args = parser.parse_args(argv)

    parent = gpu_dir(args.root, args.world_size)
    if args.dry_run:
        print(parent / run_name(next_run_index(parent)))
        return
    path = claim_run_dir(parent) if args.node_rank == 0 else None
    if args.nnodes > 1:
        if args.port <= 0:
            parser.error("--port is required when --nnodes > 1")
        path = share_with_nodes(
            path,
            nnodes=args.nnodes,
            node_rank=args.node_rank,
            host=args.master_addr,
            port=args.port,
            timeout_s=args.timeout_s,
        )
        # Every node writes into this directory and node 0 merges from it, so it must be the same
        # directory, not a same-named local copy. Node 0 created the marker before publishing.
        if args.node_rank != 0 and not (path / SHARED_MARKER).is_file():
            raise SystemExit(
                f"{path} is not on storage shared with node 0 ({SHARED_MARKER} is missing); "
                "put the repo on a shared file system"
            )
    print(path)


if __name__ == "__main__":
    main()
