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
"""One AllGather_STMC of ``--size-mib`` with nccl-2.32.3-pause-with-debug timing its multimem.st.

``run_ag_st_debug.sh`` starts this once per pause value, with the library preloaded and NCCL_SYM_KERNEL
pinning the AllGather to ``AllGather_STMC``. Every rank of the job is in one group. The input
(``size / n`` bytes) and output (``size`` bytes) live in a ``torch.cuda.MemPool`` backed by
``ncclMemAlloc`` and registered as an ``NCCL_WIN_COLL_SYMMETRIC`` window, so NCCL runs the symmetric
multicast kernel. The default process group is Gloo and carries every exchange of results: with the
pin, an NCCL AllGather on plain memory has no implementation.

``--warmup`` AllGathers run first, then one measured AllGather between CUDA events. The library counts
the AllGather_STMC launches of each communicator and prints only those after
``NCCL_SYM_AG_STMC_ST_PRINT_SKIP`` (``NCCL_SYM_AG_STMC_ST_PRINT_LAUNCHES`` of them); the launcher sets
SKIP = ``--warmup`` and LAUNCHES = 1, so only the measured launch prints.

Every rank writes into ``--scratch-dir`` (node-local, removed by the launcher after a good run): its
device printf (stdout, pointed there before CUDA starts) and its NCCL log (``NCCL_DEBUG_FILE``, set
here before NCCL reads it). It then checks its own run: the AllGather output, the printed lines
covering every chunk it stored (``size / n / 4 KiB``; a line dropped by a full printf FIFO fails),
the library banner, the four ``NCCL_SYM_AG_STMC_*`` values, no process-wide CTA variable and, on rank 0,
the kernel and CTA count of every AllGather launch. Rank 0 gathers everything over Gloo and writes the
one output file of this pause, ``--output-file`` (``<stem>.failed.log`` next to it if a check failed)::

    # ag-st-debug status=ok
    # nccl=2.32.3-pause-with-debug+cuda13.0 kernel=AllGather_STMC ctas=4 warps_per_cta=16 world_size=4 ...
    # rank=0 host=... elapsed_ms=... busbw_gbps=... lines=1024 warps=64 chunks=65536 ... st_cycles_mean=...
    [nccl-ag-stmc-st] launch=2 rank=0 cta=0 warp=0 iter=64 n=64 sum=... min=... max=... last=... pause=0 globaltimer=...

The ``# key=value`` lines describe the run and summarize each rank; then come the raw lines of every
rank, sorted by rank, CTA, warp and chunk, in the library's own format (``parse_lines`` reads them back).
"""

from __future__ import annotations

import argparse
import ctypes
import logging
import os
import re
import socket
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist


LOGGER = logging.getLogger(__name__)

BYTES_PER_MIB = 1024 * 1024
# bcastMultimem stores one UnrollPacks (8) x WARP_SIZE (32) x BytePerPack (16) = 4 KiB chunk per warp and iteration.
CHUNK_BYTES = 4096
WARP_SIZE = 32
LINE_PREFIX = "[nccl-ag-stmc-st]"
LINE_RE = re.compile(
    r"\[nccl-ag-stmc-st\] launch=(?P<launch>\d+) rank=(?P<rank>-?\d+) cta=(?P<cta>\d+) warp=(?P<warp>\d+)"
    r" iter=(?P<iter>\d+) n=(?P<n>\d+) sum=(?P<sum>\d+) min=(?P<min>\d+) max=(?P<max>\d+) last=(?P<last>\d+)"
    r" pause=(?P<pause>\d+) globaltimer=(?P<globaltimer>\d+)"
)
PAUSE_ENV = "NCCL_SYM_AG_STMC_PAUSE_CYCLES"
INTERVAL_ENV = "NCCL_SYM_AG_STMC_ST_PRINT_INTERVAL"
SKIP_ENV = "NCCL_SYM_AG_STMC_ST_PRINT_SKIP"
LAUNCHES_ENV = "NCCL_SYM_AG_STMC_ST_PRINT_LAUNCHES"
DEBUG_ENVS = (PAUSE_ENV, INTERVAL_ENV, SKIP_ENV, LAUNCHES_ENV)
SYM_KERNEL_PIN = "allgather:AllGather_STMC"
NCCL_SUFFIX = "-pause-with-debug"
# Process-wide CTA variables; the launcher unsets them so NCCL picks its own AllGather_STMC CTA count.
CTA_ENVS = (
    "NCCL_MIN_CTAS",
    "NCCL_MAX_CTAS",
    "NCCL_MIN_NCHANNELS",
    "NCCL_MAX_NCHANNELS",
    "NCCL_MIN_NRINGS",
    "NCCL_MAX_NRINGS",
    "NCCL_NVLS_NCHANNELS",
    "NCCL_MIN_P2P_NCHANNELS",
    "NCCL_MAX_P2P_NCHANNELS",
    "NCCL_SYM_CTAS",
)
BANNER_RE = re.compile(r"NCCL version (\S+)")
# symmetric_sched.cc, comm rank 0 only: one line per symmetric launch.
TUNING_RE = re.compile(
    r"AllGather \[Symmetric\]: (?P<bytes>\d+) Bytes -> Kernel (?P<kernel>\w+) nchannels (?P<ctas>\d+)"
    r" nthreads (?P<threads>\d+)"
)


@dataclass(frozen=True)
class StLine:
    """One printed window of a warp: ``n`` chunks whose multimem.st took ``sum`` cycles in total."""

    launch: int
    rank: int
    cta: int
    warp: int
    iter: int
    n: int
    sum: int
    min: int
    max: int
    last: int
    pause: int
    globaltimer: int

    def render(self) -> str:
        """The line in the library's printf format."""
        return (
            f"{LINE_PREFIX} launch={self.launch} rank={self.rank} cta={self.cta} warp={self.warp} iter={self.iter}"
            f" n={self.n} sum={self.sum} min={self.min} max={self.max} last={self.last} pause={self.pause}"
            f" globaltimer={self.globaltimer}"
        )


@dataclass(frozen=True)
class RankSummary:
    """The measured launch of one rank: its lines folded into chunks and cycles."""

    rank: int
    lines: int
    warps: int
    chunks: int
    expected_chunks: int
    cycles_sum: int
    cycles_min: int
    cycles_max: int
    chunks_per_warp_min: int
    chunks_per_warp_max: int
    span_ns: int

    @property
    def cycles_mean(self) -> float:
        """Mean multimem.st cycles of one 4 KiB chunk (8 stores of one warp)."""
        return self.cycles_sum / self.chunks if self.chunks else float("nan")


@dataclass(frozen=True)
class NcclLogCheck:
    """What one rank's NCCL log says about the run."""

    banner: str | None
    # (bytes, kernel, ctas, threads) of every AllGather launch; only comm rank 0 logs them.
    tunings: list[tuple[int, str, int, int]]
    errors: list[str]


def chunks_per_rank(size_bytes: int, world_size: int) -> int:
    """4 KiB chunks one rank stores in an AllGather of ``size_bytes`` output (its ``size / n`` shard)."""
    return size_bytes // world_size // CHUNK_BYTES


def expected_lines(size_bytes: int, world_size: int, interval: int) -> int:
    """Lines one rank prints, not counting the one tail line a warp may add (at most one per warp).

    Every chunk of the rank belongs to exactly one warp, so the count does not depend on how many CTAs
    and warps NCCL picks: chunks / interval, rounded down per warp, plus at most one tail line per warp.
    """
    return chunks_per_rank(size_bytes, world_size) // interval


def parse_lines(text: str) -> list[StLine]:
    """Every ``[nccl-ag-stmc-st]`` line of a device printf log; a line that does not parse raises."""
    lines = []
    for raw in text.splitlines():
        if LINE_PREFIX not in raw:
            continue
        match = LINE_RE.search(raw)
        if match is None:
            raise ValueError(f"malformed {LINE_PREFIX} line: {raw!r}")
        lines.append(StLine(**{key: int(value) for key, value in match.groupdict().items()}))
    return lines


def summarize(lines: list[StLine], *, rank: int, expected_chunks: int) -> RankSummary:
    """Fold one rank's lines of one launch; fails if they miss or repeat a chunk of some warp."""
    if not lines:
        raise ValueError(f"rank {rank}: no {LINE_PREFIX} lines")
    per_warp: dict[tuple[int, int], list[StLine]] = {}
    for line in lines:
        if line.rank != rank:
            raise ValueError(f"rank {rank}: a line of rank {line.rank} in its log")
        per_warp.setdefault((line.cta, line.warp), []).append(line)
    warp_chunks = []
    for (cta, warp), windows in sorted(per_warp.items()):
        windows.sort(key=lambda line: line.iter)
        done = 0
        for line in windows:
            # iter is the warp's running chunk count, so consecutive windows must tile it without a gap.
            if line.iter != done + line.n:
                raise ValueError(
                    f"rank {rank} cta {cta} warp {warp}: window ending at chunk {line.iter} has n={line.n},"
                    f" but the previous one ended at {done} (a line was dropped?)"
                )
            done = line.iter
        warp_chunks.append(done)
    chunks = sum(warp_chunks)
    if chunks != expected_chunks:
        raise ValueError(
            f"rank {rank}: the lines cover {chunks} chunks, expected {expected_chunks}"
            " (the printf FIFO dropped lines? raise the interval)"
        )
    timers = [line.globaltimer for line in lines]
    return RankSummary(
        rank=rank,
        lines=len(lines),
        warps=len(per_warp),
        chunks=chunks,
        expected_chunks=expected_chunks,
        cycles_sum=sum(line.sum for line in lines),
        cycles_min=min(line.min for line in lines),
        cycles_max=max(line.max for line in lines),
        chunks_per_warp_min=min(warp_chunks),
        chunks_per_warp_max=max(warp_chunks),
        span_ns=max(timers) - min(timers),
    )


def check_nccl_log(text: str, env: dict[str, int], *, comm_rank0: bool, launches: int) -> NcclLogCheck:
    """Check one rank's NCCL log: the debug build, its four values, no CTA variable, and (rank 0) the launches."""
    errors = []
    banners = sorted(set(BANNER_RE.findall(text)))
    banner = banners[0] if len(banners) == 1 else None
    if banner is None or not banner.startswith(f"2.32.3{NCCL_SUFFIX}+"):
        errors.append(f"NCCL banner {banners or 'missing'}, expected 2.32.3{NCCL_SUFFIX}+cuda...")
    for name, value in env.items():
        if f"{name} set by environment to {value}." not in text:
            errors.append(f"NCCL did not log {name}={value}")
    for name in CTA_ENVS:
        if f"{name} set by environment to" in text:
            errors.append(f"NCCL read {name}, so the AllGather is not at NCCL's own CTA count")
    tunings = [(int(m["bytes"]), m["kernel"], int(m["ctas"]), int(m["threads"])) for m in TUNING_RE.finditer(text)]
    if comm_rank0:
        if len(tunings) != launches:
            errors.append(f"{len(tunings)} AllGather launches in the TUNING log, expected {launches}")
        other = sorted({kernel for _, kernel, _, _ in tunings if kernel != "AllGather_STMC"})
        if other:
            errors.append(f"AllGathers ran {other}, not AllGather_STMC")
    return NcclLogCheck(banner=banner, tunings=tunings, errors=errors)


def render_output(meta: dict[str, object], records: list[dict], errors: list[str]) -> str:
    """The output file of one pause: ``# key=value`` header, one summary line per rank, then every raw line."""
    out = [f"# ag-st-debug status={'failed' if errors else 'ok'}"]
    out.append("# " + " ".join(f"{key}={value}" for key, value in meta.items()))
    for record in records:
        fields: dict[str, object] = {
            "rank": record["rank"],
            "host": record["host"],
            "elapsed_ms": f"{record['elapsed_ms']:.4f}",
            "busbw_gbps": f"{record['busbw_gbps']:.1f}",
        }
        stats = record["summary"]
        if stats is not None:
            fields |= {
                "lines": stats["lines"],
                "warps": stats["warps"],
                "chunks": stats["chunks"],
                "chunks_per_warp": f"{stats['chunks_per_warp_min']}-{stats['chunks_per_warp_max']}",
                "st_cycles_mean": f"{stats['cycles_sum'] / stats['chunks']:.1f}",
                "st_cycles_min": stats["cycles_min"],
                "st_cycles_max": stats["cycles_max"],
                "span_ns": stats["span_ns"],
            }
        out.append("# " + " ".join(f"{key}={value}" for key, value in fields.items()))
    out.extend(f"# error: {error}" for error in errors)
    for record in records:
        out.extend(record["lines"])
    return "\n".join(out) + "\n"


def failed_path(output_file: Path) -> Path:
    """Where a run whose checks failed writes its output: ``pause0cyc.log`` -> ``pause0cyc.failed.log``."""
    return output_file.with_name(f"{output_file.stem}.failed{output_file.suffix}")


def _env_int(name: str) -> int:
    value = os.environ.get(name, "")
    if not value.isdigit():
        raise RuntimeError(f"{name} must be set to a non-negative integer by run_ag_st_debug.sh, got {value!r}")
    return int(value)


def check_environment(warmup: int) -> dict[str, int]:
    """The library's variables, checked so that exactly the measured launch prints."""
    debug = {name: _env_int(name) for name in DEBUG_ENVS}
    if debug[INTERVAL_ENV] == 0:
        raise RuntimeError(f"{INTERVAL_ENV}=0 times nothing")
    if debug[SKIP_ENV] != warmup or debug[LAUNCHES_ENV] != 1:
        raise RuntimeError(
            f"{SKIP_ENV} must equal --warmup ({warmup}) and {LAUNCHES_ENV} must be 1, so only the measured launch"
            f" prints; got {debug[SKIP_ENV]} and {debug[LAUNCHES_ENV]}"
        )
    if os.environ.get("NCCL_SYM_KERNEL") != SYM_KERNEL_PIN:
        raise RuntimeError(f"NCCL_SYM_KERNEL must be {SYM_KERNEL_PIN!r}, got {os.environ.get('NCCL_SYM_KERNEL')!r}")
    if os.environ.get("NCCL_WIN_ENABLE") != "1" or os.environ.get("NCCL_NVLS_ENABLE") != "1":
        raise RuntimeError("AllGather_STMC needs NCCL_WIN_ENABLE=1 and NCCL_NVLS_ENABLE=1 (windows + multicast)")
    return debug


def _flush_c_stdout() -> None:
    """Flush the C stdio buffer the CUDA runtime wrote the device printf lines into."""
    ctypes.CDLL(None).fflush(None)


def _run(args: argparse.Namespace, printf_log: Path, nccl_log: Path) -> None:
    debug = check_environment(args.warmup)
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    # Gloo carries the barriers and the results; the NCCL group is created now (device_id), so the window is
    # registered before its first collective.
    dist.init_process_group(backend="gloo")
    ag_pg = dist.new_group(backend="nccl", device_id=device)
    rank, world_size = dist.get_rank(), dist.get_world_size()
    size_bytes = args.size_mib * BYTES_PER_MIB
    elem = torch.empty((), dtype=torch.bfloat16).element_size()
    if size_bytes % (world_size * CHUNK_BYTES):
        raise ValueError(f"--size-mib {args.size_mib} does not split into {world_size} shards of whole 4 KiB chunks")
    numel = size_bytes // elem
    shard = numel // world_size

    # register_mem_pool and mem_allocator are bound only on the NCCL backend behind the generic ProcessGroup.
    backend = ag_pg._get_backend(torch.device("cuda"))
    # ncclMemAlloc: NCCL multicast-binds only the cuMem allocation that holds the window base, so the pool
    # must not use expandable segments (see nvls_collective_benchmark._make_buffers).
    pool = torch.cuda.MemPool(backend.mem_allocator)
    with torch.cuda.use_mem_pool(pool):
        ag_in = torch.full((shard,), float(rank + 1), dtype=torch.bfloat16, device=device)
        ag_out = torch.empty(numel, dtype=torch.bfloat16, device=device)
    torch.cuda.synchronize()
    backend.register_mem_pool(pool, symm=True)
    torch.cuda.synchronize()

    errors: list[str] = []
    for _ in range(args.warmup):
        dist.all_gather_into_tensor(ag_out, ag_in, group=ag_pg)
    torch.cuda.synchronize()
    _flush_c_stdout()
    if LINE_PREFIX in printf_log.read_text(errors="replace"):
        errors.append(f"rank {rank}: a warmup launch printed; is {SKIP_ENV} honoured?")
    dist.barrier()

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    dist.all_gather_into_tensor(ag_out, ag_in, group=ag_pg)
    end.record()
    torch.cuda.synchronize()
    _flush_c_stdout()
    elapsed_ms = start.elapsed_time(end)

    expected = torch.arange(1, world_size + 1, dtype=torch.bfloat16, device=device).repeat_interleave(shard)
    if not torch.equal(ag_out, expected):
        errors.append(f"rank {rank}: wrong AllGather output")
    lines = parse_lines(printf_log.read_text(errors="replace"))
    lines.sort(key=lambda line: (line.cta, line.warp, line.iter))
    launches = sorted({line.launch for line in lines})
    summary = None
    if launches != [args.warmup]:
        errors.append(f"rank {rank}: lines of launches {launches}, expected only the measured launch {args.warmup}")
    else:
        try:
            summary = summarize(lines, rank=rank, expected_chunks=chunks_per_rank(size_bytes, world_size))
        except ValueError as exc:
            errors.append(str(exc))
    backend.deregister_mem_pool(pool)
    # NCCL logs line-buffered, so the log is complete once the launches are done.
    log_check = check_nccl_log(
        nccl_log.read_text(errors="replace") if nccl_log.exists() else "",
        debug,
        comm_rank0=rank == 0,
        launches=args.warmup + 1,
    )
    errors.extend(f"rank {rank}: {error}" for error in log_check.errors)

    record = {
        "rank": rank,
        "host": socket.gethostname(),
        "elapsed_ms": elapsed_ms,
        "busbw_gbps": size_bytes * (world_size - 1) / world_size / (elapsed_ms * 1e6),
        "summary": asdict(summary) if summary else None,
        "banner": log_check.banner,
        "tunings": log_check.tunings,
        "lines": [line.render() for line in lines],
        "errors": errors,
    }
    records: list[dict] | None = [{} for _ in range(world_size)] if rank == 0 else None
    dist.gather_object(record, records, dst=0)

    all_errors: list[str] = []
    if rank == 0:
        assert records is not None
        all_errors = [error for item in records for error in item["errors"]]
        # The measured launch is the last one rank 0 logged; its CTAs x warps are the warps that must print.
        tuning = records[0]["tunings"][-1] if records[0]["tunings"] else None
        ctas, warps_per_cta = (tuning[2], tuning[3] // WARP_SIZE) if tuning else (None, None)
        # Each warp prints at least its tail line once it has a chunk at all.
        if tuning is not None and chunks_per_rank(size_bytes, world_size) >= ctas * warps_per_cta:
            for item in records:
                if item["summary"] is not None and item["summary"]["warps"] != ctas * warps_per_cta:
                    all_errors.append(
                        f"rank {item['rank']}: {item['summary']['warps']} warps printed,"
                        f" expected {ctas} CTAs x {warps_per_cta} warps"
                    )
        meta = {
            "nccl": records[0]["banner"],
            "kernel": tuning[1] if tuning else None,
            "ctas": ctas,
            "warps_per_cta": warps_per_cta,
            "world_size": world_size,
            "size_mib": args.size_mib,
            "warmup": args.warmup,
            "chunk_bytes": CHUNK_BYTES,
            "chunks_per_rank": chunks_per_rank(size_bytes, world_size),
        } | debug
        path = failed_path(args.output_file) if all_errors else args.output_file
        path.parent.mkdir(parents=True, exist_ok=True)
        # A rerun of the same pause leaves exactly one of the two files.
        for stale in {args.output_file, failed_path(args.output_file)} - {path}:
            stale.unlink(missing_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(render_output(meta, records, all_errors))
        tmp.replace(path)
        LOGGER.info(
            "AllGather %d MiB over %d ranks, %s CTAs, pause %d cycles, one line per %d chunks: %s",
            args.size_mib,
            world_size,
            ctas,
            debug[PAUSE_ENV],
            debug[INTERVAL_ENV],
            path,
        )
        for item in records:
            stats = item["summary"]
            mean = f"{stats['cycles_sum'] / stats['chunks']:.1f}" if stats else "n/a"
            LOGGER.info(
                "  rank %d: %.3f ms, %.1f GB/s busbw, multimem.st %s cycles per chunk",
                item["rank"],
                item["elapsed_ms"],
                item["busbw_gbps"],
                mean,
            )
    # Every rank exits with rank 0's verdict, so the torchrun of every node reports the same status.
    verdict = [all_errors]
    dist.broadcast_object_list(verdict, src=0)
    dist.destroy_process_group()
    if verdict[0]:
        raise RuntimeError("; ".join(verdict[0]))


def main() -> None:
    """Point stdout and the NCCL log at this rank's scratch files, then run the warmup and the measured AllGather."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-file", type=Path, required=True, help="The one output file of this pause (rank 0)")
    parser.add_argument("--scratch-dir", type=Path, required=True, help="Node-local per-rank printf and NCCL logs")
    parser.add_argument("--size-mib", type=int, default=1024, help="AllGather output size (every rank's shard: / n)")
    parser.add_argument("--warmup", type=int, default=2, help="AllGathers before the measured one, not printed")
    args = parser.parse_args()
    if args.size_mib <= 0 or args.warmup < 0:
        parser.error("--size-mib must be > 0 and --warmup >= 0")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    rank = int(os.environ["RANK"])
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    # NCCL opens NCCL_DEBUG_FILE at its first log line, after this; one file per rank, known by name.
    nccl_log = args.scratch_dir / f"nccl.rank{rank:02d}.log"
    os.environ["NCCL_DEBUG_FILE"] = str(nccl_log)
    # The CUDA runtime prints the device FIFO to fd 1 at each synchronization. Python's logging goes to stderr
    # and stays on the console.
    printf_log = args.scratch_dir / f"device-printf.rank{rank:02d}.log"
    fd = os.open(printf_log, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(fd, 1)
    os.close(fd)
    _run(args, printf_log, nccl_log)


if __name__ == "__main__":
    main()
