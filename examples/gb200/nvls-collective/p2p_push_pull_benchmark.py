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
"""Profile one-way NCCL P2P between two GPUs of a GB200 tray: push (P2P write) vs pull (P2P read).

``run_p2p_push_pull_benchmark.sh`` starts this once per workload, with two ranks on two GPUs of
one 4-GPU tray. Rank 0 is always the sender and rank 1 the receiver; the workload only changes
which side's SMs move the data across NVLink (``transport/p2p.cc`` in NCCL 2.32.3):

* ``push`` (``NCCL_P2P_READ_ENABLE=0``, P2P write): the SIMPLE staging buffer lives on the
  receiver. The sender's SMs store into it over NVLink and the receiver copies it locally into
  the user buffer, so the sender is the NVLink initiator (its NVLink TX carries the data).
* ``pull`` (``NCCL_P2P_READ_ENABLE=1``, P2P read): ``p2pSendSetup`` appends the SIMPLE buffer to
  the sender's ``ncclSendMem``. The sender copies its user buffer into it locally and the
  receiver's SMs load from it over NVLink, so the receiver is the initiator (its NVLink RX
  carries the data, as read responses).

NCCL fixes read vs write when the pair's channels connect, so the variable is process-wide and
each workload is its own torchrun job. One small send connects the channels before anything is
timed; the sender's INIT line ``Channel <c>/1 : 0[..] -> 1[..] via P2P/...[/read]`` must match
the workload, otherwise the job fails.

Each size is one case: ``--warmup`` sends, then ``--trials`` timed trials. A trial opens with a
4-byte all-reduce that lines the two GPUs up, then times ``--iters`` back-to-back sends (sender)
or receives (receiver) of the whole message with CUDA events. The receiver's buffer is zeroed
before the warmup and before the trials and checked after both. With ``--nsys`` the trials of a
case are one ``cudaProfilerStart``/``Stop`` window; nsys (``--capture-range-end=repeat``) writes
``rank<R>.<K>.nsys-rep`` for case ``K``, and rank 0 lists in ``nsys/index.tsv`` the name the
launcher gives each report, e.g. ``push.1024MiB.gpu0.sender.initiator``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
from pathlib import Path
from statistics import median

import torch
import torch.distributed as dist
from nvls_collective_benchmark import CaptureWindows, NcclLog, check_p2p_connections
from torch.cuda import nvtx


LOGGER = logging.getLogger(__name__)

WORKLOADS = ("push", "pull")
SENDER, RECEIVER = 0, 1
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
BYTES_PER_MIB = 1024 * 1024
# NCCL decides read vs write from this when a P2P channel connects (p2pGetInfo in transport/p2p.cc).
P2P_READ_ENV = "NCCL_P2P_READ_ENABLE"
# What the sender sends; the receiver's zero-filled buffer cannot hold it by accident (exact in bf16).
PAYLOAD = 3.0

# What each GPU's SMs do in a workload; the NVTX case range carries it into the nsys report.
ACTIVITY = {
    ("push", SENDER): "NVLink initiator: SMs store into the receiver's staging buffer",
    ("push", RECEIVER): "passive: NVLink writes land here, SMs copy the staging buffer locally",
    ("pull", SENDER): "passive: SMs copy into the local staging buffer the receiver loads",
    ("pull", RECEIVER): "NVLink initiator: SMs load the sender's staging buffer",
}
INDEX_HEADER = "capture_index\tsize_mib\trank\tgpu\trole\tinitiator\treport\n"


def role(rank: int) -> str:
    """``sender`` for rank 0, ``receiver`` for rank 1."""
    return "sender" if rank == SENDER else "receiver"


def initiator(workload: str) -> int:
    """Rank whose SMs drive the NVLink transfer: the sender stores (push), the receiver loads (pull)."""
    return SENDER if workload == "push" else RECEIVER


def report_name(workload: str, size_mib: int, *, rank: int, gpu: int) -> str:
    """Name of ``rank``'s report for one case, e.g. ``push.1024MiB.gpu0.sender.initiator``."""
    name = f"{workload}.{size_mib}MiB.gpu{gpu}.{role(rank)}"
    return f"{name}.initiator" if rank == initiator(workload) else name


def index_rows(workload: str, size_mib: int, *, capture_index: int, gpus: list[int]) -> list[str]:
    """``nsys/index.tsv`` lines of one case, one per rank (see :data:`INDEX_HEADER`)."""
    return [
        f"{capture_index}\t{size_mib}\t{rank}\t{gpu}\t{role(rank)}\t{'yes' if rank == initiator(workload) else 'no'}"
        f"\t{report_name(workload, size_mib, rank=rank, gpu=gpu)}\n"
        for rank, gpu in enumerate(gpus)
    ]


def physical_gpu(local_rank: int) -> int:
    """The nvidia-smi id of ``cuda:<local_rank>`` (``CUDA_VISIBLE_DEVICES`` must list numeric ids)."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible:
        return local_rank
    entry = visible.split(",")[local_rank].strip()
    if not entry.isdigit():
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES={visible!r}: use numeric GPU ids, got {entry!r}")
    return int(entry)


def check_environment(workload: str, world_size: int) -> None:
    """Refuse to run unless the job is two ranks and NCCL is set up the way ``workload`` needs."""
    if world_size != 2:
        raise RuntimeError(f"push/pull runs one sender and one receiver; got {world_size} ranks, need 2")
    want = "1" if workload == "pull" else "0"
    got = os.environ.get(P2P_READ_ENV)
    if got != want:
        raise RuntimeError(f"--workload {workload} needs {P2P_READ_ENV}={want}, got {got!r}")
    if os.environ.get("NCCL_DEBUG", "").upper() != "INFO":
        raise RuntimeError("NCCL_DEBUG=INFO is required: the P2P transport is read from NCCL's INIT log")
    if "%p" not in os.environ.get("NCCL_DEBUG_FILE", ""):
        raise RuntimeError("NCCL_DEBUG_FILE must be set and contain %p so every process reads only its own log")
    subsys = os.environ.get("NCCL_DEBUG_SUBSYS")
    # NCCL's default (unset) includes INIT; the connect line is INFO(NCCL_INIT | NCCL_P2P).
    if subsys is not None and (
        subsys.startswith("^") or not {"INIT", "P2P", "ALL"} & {s.strip().upper() for s in subsys.split(",")}
    ):
        raise RuntimeError(f"NCCL_DEBUG_SUBSYS={subsys!r} hides the P2P connect lines; include INIT")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the worker options (the launcher passes all of them)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workload", choices=WORKLOADS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sizes-mib", type=lambda s: [int(x) for x in s.split(",")], default=[1024], help="Message sizes in MiB"
    )
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--warmup", type=int, default=5, help="Untimed sends per size")
    parser.add_argument("--trials", type=int, default=5, help="Timed trials per size")
    parser.add_argument("--iters", type=int, default=20, help="Back-to-back sends per trial")
    parser.add_argument(
        "--nsys", action="store_true", help="Open one cudaProfilerStart/Stop window around each size's trials"
    )
    args = parser.parse_args(argv)
    if any(size <= 0 for size in args.sizes_mib) or len(set(args.sizes_mib)) != len(args.sizes_mib):
        parser.error("--sizes-mib must be distinct positive sizes")
    if min(args.warmup, args.trials, args.iters) < 0 or args.trials == 0 or args.iters == 0:
        parser.error("--trials and --iters must be positive, --warmup non-negative")
    return args


def _transfer(pg: dist.ProcessGroup, rank: int, tensor: torch.Tensor, count: int) -> None:
    """``count`` back-to-back sends (sender) or receives (receiver) of ``tensor``, ordered on the current stream."""
    for _ in range(count):
        if rank == SENDER:
            work = dist.isend(tensor, dst=RECEIVER, group=pg)
        else:
            work = dist.irecv(tensor, src=SENDER, group=pg)
        # Makes the current stream wait for NCCL's P2P stream; the host does not block.
        work.wait()


def _received(tensor: torch.Tensor) -> bool:
    return bool(torch.all(tensor == PAYLOAD).item())


def connect(pg: dist.ProcessGroup, rank: int, workload: str, nccl_log: NcclLog) -> list[str]:
    """Connect the pair's P2P channels with one small send and check them against ``workload``.

    NCCL connects every channel to the peer on the first send and never reconnects them, so this
    fixes read vs write for the whole job. Only the sender logs the connection (``p2pSendSetup``);
    rank 0 checks it and both ranks fail together. Returns the transports, e.g. ``P2P/CUMEM/read``.
    """
    nccl_log.consume()
    tensor = torch.zeros(1024, device="cuda")
    _transfer(pg, rank, tensor, 1)
    torch.cuda.synchronize()
    lines = nccl_log.consume()
    verdict: list[object] = [None]
    if rank == SENDER:
        transports, errors = check_p2p_connections(lines, mode=workload)
        if not transports:
            errors.append(f"rank 0 logged no P2P connection to rank 1 in {nccl_log.path}")
        verdict = [(transports, errors)]
    dist.broadcast_object_list(verdict, src=SENDER)
    transports, errors = verdict[0]  # type: ignore[misc]
    if errors:
        raise RuntimeError(f"--workload {workload}: " + "; ".join(errors))
    return transports


def run_size(
    args: argparse.Namespace,
    *,
    pg: dist.ProcessGroup,
    rank: int,
    gpus: list[int],
    size_mib: int,
    captures: CaptureWindows,
) -> dict[str, object]:
    """Warm up, check, time and check one message size; returns rank 0's record (others get ``{}``)."""
    dtype = DTYPES[args.dtype]
    numel = size_mib * BYTES_PER_MIB // torch.empty((), dtype=dtype).element_size()
    if rank == SENDER:
        tensor = torch.full((numel,), PAYLOAD, dtype=dtype, device="cuda")
    else:
        tensor = torch.zeros(numel, dtype=dtype, device="cuda")
    gate = torch.zeros(1, dtype=torch.float32, device="cuda")

    _transfer(pg, rank, tensor, max(args.warmup, 1))
    torch.cuda.synchronize()
    ok = [rank == SENDER or _received(tensor)]
    if rank == RECEIVER:
        tensor.zero_()
    torch.cuda.synchronize()

    label = report_name(args.workload, size_mib, rank=rank, gpu=gpus[rank])
    times_ms: list[float] = []
    with captures.case(f"{label} | {ACTIVITY[(args.workload, rank)]}") as capture_index:
        for trial in range(args.trials):
            # The gate is an all-reduce on the pair, so the NVTX trial range opens after it.
            dist.all_reduce(gate, group=pg)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            with nvtx.range(f"trial{trial}"):
                start.record()
                _transfer(pg, rank, tensor, args.iters)
                end.record()
                end.synchronize()
            times_ms.append(start.elapsed_time(end))
    ok.append(rank == SENDER or _received(tensor))

    gathered: list[object] = [None, None]
    dist.all_gather_object(gathered, {"times_ms": times_ms, "ok": all(ok), "capture_index": capture_index})
    if rank != 0:
        return {}
    sender, receiver = gathered  # type: ignore[misc]
    if not receiver["ok"]:
        raise RuntimeError(f"{args.workload} {size_mib} MiB: GPU{gpus[RECEIVER]} did not receive the payload")
    if sender["capture_index"] != receiver["capture_index"]:
        raise RuntimeError(f"capture windows diverged: {sender['capture_index']} vs {receiver['capture_index']}")
    moved_bytes = size_mib * BYTES_PER_MIB * args.iters
    sender_ms, receiver_ms = median(sender["times_ms"]), median(receiver["times_ms"])
    return {
        "workload": args.workload,
        "size_mib": size_mib,
        "dtype": args.dtype,
        "iters": args.iters,
        "trials": args.trials,
        "sender_gpu": gpus[SENDER],
        "receiver_gpu": gpus[RECEIVER],
        "initiator": role(initiator(args.workload)),
        "sender_ms": sender["times_ms"],
        "receiver_ms": receiver["times_ms"],
        "sender_median_ms": sender_ms,
        "receiver_median_ms": receiver_ms,
        # The receiver's time ends when the last byte has landed in its buffer.
        "algbw_gbps": moved_bytes / (receiver_ms * 1e-3) / 1e9,
        "nsys_capture_index": capture_index,
        "nsys_reports": [report_name(args.workload, size_mib, rank=r, gpu=gpus[r]) for r in (SENDER, RECEIVER)],
    }


def main(argv: list[str] | None = None) -> None:
    """Run one workload under torchrun with two ranks."""
    args = parse_args(argv)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    # Gloo carries the result gathering; the NCCL communicator below is the pair's only one.
    dist.init_process_group(backend="gloo")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    check_environment(args.workload, world_size)
    gpus: list[object] = [None, None]
    dist.all_gather_object(gpus, physical_gpu(local_rank))
    if gpus[0] == gpus[1]:
        raise RuntimeError(f"sender and receiver share GPU{gpus[0]}")

    output = args.output_dir / "results.jsonl"
    nsys_index = args.output_dir / "nsys" / "index.tsv"
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"{output} already exists; use a fresh --output-dir")
        metadata = {
            "workload": args.workload,
            "hostname": socket.gethostname(),
            "torch": torch.__version__,
            "nccl": ".".join(map(str, torch.cuda.nccl.version())),
            "gpus": {"sender": gpus[SENDER], "receiver": gpus[RECEIVER]},
            "initiator": role(initiator(args.workload)),
            "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith(("NCCL_", "CUDA_"))},
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        }
        (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if args.nsys:
            nsys_index.parent.mkdir(parents=True, exist_ok=True)
            nsys_index.write_text(INDEX_HEADER, encoding="utf-8")

    nccl_log = NcclLog()
    pg = dist.new_group([SENDER, RECEIVER], backend="nccl", device_id=torch.device("cuda", local_rank))
    transports = connect(pg, rank, args.workload, nccl_log)
    LOGGER.info(
        "rank %d GPU%d %s: %s, transports %s",
        rank,
        gpus[rank],
        role(rank),
        ACTIVITY[(args.workload, rank)],
        transports,
    )

    captures = CaptureWindows(args.nsys)
    for size_mib in args.sizes_mib:
        record = run_size(args, pg=pg, rank=rank, gpus=gpus, size_mib=size_mib, captures=captures)  # type: ignore[arg-type]
        if rank != 0:
            continue
        record["transports"] = transports
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if args.nsys:
            with nsys_index.open("a", encoding="utf-8") as tsv:
                tsv.writelines(
                    index_rows(args.workload, size_mib, capture_index=record["nsys_capture_index"], gpus=gpus)  # type: ignore[arg-type]
                )
        LOGGER.info(
            "%s %5d MiB GPU%d -> GPU%d (initiator: %s)  sender %8.3f ms  receiver %8.3f ms  algbw %7.1f GB/s",
            args.workload,
            size_mib,
            gpus[SENDER],
            gpus[RECEIVER],
            record["initiator"],
            record["sender_median_ms"],
            record["receiver_median_ms"],
            record["algbw_gbps"],
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
