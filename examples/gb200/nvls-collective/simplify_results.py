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
"""Copy a results tree keeping only the first ``N`` ranks' copy of every per-rank file.

``<PATH>`` is ``results/gb200/nvls-collective`` (``gpu-num-<W>/<yyyymmdd-hhmmss>/...`` below it)
or any directory inside it. The copy goes to ``<PATH>-simplify-<N>``; ``<PATH>`` is not touched::

    python simplify_results.py <PATH>             # N = 1
    python simplify_results.py <PATH> --num 2

Per-rank files are grouped by directory and by their name with the rank taken out, and only the
``N`` lowest ranks present in each group are copied:

- ``<algo>-<name>.rank_<RR><suffix>``: ``comm/`` and ``cases/<algo>-<case>/`` NCCL logs, nsys
  reports and their ``.sqlite`` exports;
- ``rank<R><suffix>``: ``nsys/rank<R>.<K>.nsys-rep`` (grouped by ``K``) and ``ncu/rank<R>.ncu-rep``;
- ``nccl-raw/nccl-<host>-<pid>.log``: the rank comes from the ``[<cudaDev>]`` field of the log
  lines (the benchmark sets the device to ``LOCAL_RANK``) and the host's ranks in the pass's
  ``metadata.json``. A log whose rank cannot be found is kept.

Everything else (``results.jsonl``, ``metadata.json``, ``case.json``, the reports, the per-node
``torchrun-node<k>.log``) has one copy and is copied as is. Only the standard library is used.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from collections import defaultdict
from pathlib import Path


LOGGER = logging.getLogger(__name__)

RANK_FILE_RE = re.compile(r"^(?P<prefix>.*\.rank_|rank)(?P<rank>\d+)(?P<suffix>(?:\..*)?)$")
NCCL_RAW_RE = re.compile(r"^nccl-(?P<host>.+)-(?P<pid>\d+)\.log$")
# NCCL's debug.cc prefix: "<host>:<pid>:<tid> [<cudaDev>] NCCL INFO|WARN|TRACE ...".
NCCL_DEV_RE = re.compile(r":\d+:\d+ \[(?P<dev>\d+)\] NCCL ")
NCCL_DEV_SCAN_LINES = 1000


def nccl_raw_ranks(algo_dir: Path) -> dict[tuple[str, int], int]:
    """Map ``(host cut at the first '.', local_rank)`` to the global rank from ``metadata.json``."""
    path = algo_dir / "metadata.json"
    if not path.is_file():
        return {}
    ranks = json.loads(path.read_text(encoding="utf-8")).get("ranks") or []
    return {(str(r["hostname"]).split(".", 1)[0], int(r["local_rank"])): int(r["rank"]) for r in ranks if r}


def nccl_raw_rank(path: Path, host: str, ranks: dict[tuple[str, int], int]) -> int | None:
    """Global rank of a raw ``nccl-<host>-<pid>.log``, or ``None`` if it cannot be told."""
    with path.open(encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= NCCL_DEV_SCAN_LINES:
                break
            m = NCCL_DEV_RE.search(line)
            if m:
                return ranks.get((host, int(m["dev"])))
    return None


class RankFilter:
    """``shutil.copytree`` ignore callback dropping all but the first ``num`` ranks' files."""

    def __init__(self, num: int) -> None:
        self.num = num
        self.kept: list[Path] = []
        self.dropped: list[Path] = []
        self.unknown: list[Path] = []

    def _select(self, groups: dict[object, dict[int, list[str]]]) -> set[str]:
        ignored: set[str] = set()
        for by_rank in groups.values():
            for rank in sorted(by_rank)[self.num :]:
                ignored.update(by_rank[rank])
        return ignored

    def __call__(self, directory: str, names: list[str]) -> set[str]:
        """Return the names in ``directory`` that are not copied."""
        dir_path = Path(directory)
        groups: dict[object, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
        raw_ranks = nccl_raw_ranks(dir_path.parent) if dir_path.name == "nccl-raw" else None
        for name in names:
            path = dir_path / name
            if not path.is_file():
                continue
            if raw_ranks is not None and (m := NCCL_RAW_RE.match(name)):
                rank = nccl_raw_rank(path, m["host"], raw_ranks)
                if rank is None:
                    self.unknown.append(path)
                    continue
                groups["nccl-raw"][rank].append(name)
            elif m := RANK_FILE_RE.match(name):
                groups[(m["prefix"], m["suffix"])][int(m["rank"])].append(name)
        ignored = self._select(groups)
        for by_rank in groups.values():
            for files in by_rank.values():
                for name in files:
                    (self.dropped if name in ignored else self.kept).append(dir_path / name)
        return ignored


def total_bytes(paths: list[Path]) -> int:
    """Sum of the file sizes."""
    return sum(p.stat().st_size for p in paths)


def human(num_bytes: int) -> str:
    """``num_bytes`` as e.g. ``1.5 GiB``."""
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, help="results directory, e.g. results/gb200/nvls-collective")
    parser.add_argument("--num", type=int, default=1, help="ranks kept per per-rank file group (default: 1)")
    args = parser.parse_args(argv)
    if args.num < 1:
        parser.error("--num must be >= 1")
    return args


def main(argv: list[str] | None = None) -> None:
    """Copy ``<path>`` to ``<path>-simplify-<num>`` without the extra ranks' files."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    src = args.path.expanduser().resolve()
    if not src.is_dir():
        raise SystemExit(f"{src} is not a directory")
    dst = src.with_name(f"{src.name}-simplify-{args.num}")
    if dst.exists():
        raise SystemExit(f"{dst} already exists; remove it first")

    keep = RankFilter(args.num)
    shutil.copytree(src, dst, ignore=keep)

    LOGGER.info("copied %s -> %s", src, dst)
    LOGGER.info(
        "per-rank files: kept %d (%s), dropped %d (%s)",
        len(keep.kept),
        human(total_bytes(keep.kept)),
        len(keep.dropped),
        human(total_bytes(keep.dropped)),
    )
    if keep.unknown:
        LOGGER.warning(
            "kept %d raw NCCL logs whose rank could not be determined, e.g. %s", len(keep.unknown), keep.unknown[0]
        )
    LOGGER.info(
        "source %s, copy %s",
        human(total_bytes([p for p in src.rglob("*") if p.is_file()])),
        human(total_bytes([p for p in dst.rglob("*") if p.is_file()])),
    )


if __name__ == "__main__":
    main()
