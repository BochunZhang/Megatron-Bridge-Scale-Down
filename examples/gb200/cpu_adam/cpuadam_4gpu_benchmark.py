# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare CPUAdam in a GIL-releasing thread and DeepSpeed's SuperOffload process on four GPU ranks."""

import argparse
import contextlib
import csv
import json
import math
import os
import shutil
import statistics
import sys
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cpuadam_benchmark import BYTES_PER_PARAM


if TYPE_CHECKING:
    from torch import Tensor

WORLD_SIZE = 4
MIB = 1 << 20
ADAM_GROUP = {"lr": 1e-4, "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.1, "amsgrad": False}


@contextlib.contextmanager
def _thread_step(
    *, parameter: "Tensor", gradient: "Tensor", rank_cpus: list[int], threads: int
) -> Iterator[tuple[Callable[[], float], dict[str, Any]]]:
    from deepspeed.ops.adam import DeepSpeedCPUAdam
    from torch.nn import Parameter

    weight = Parameter(parameter)
    weight.grad = gradient
    optimizer = DeepSpeedCPUAdam([weight], adamw_mode=True, fp32_optimizer_states=True, **ADAM_GROUP)

    def update() -> float:
        start = time.perf_counter_ns()
        optimizer.step()
        return (time.perf_counter_ns() - start) / 1e6

    controller_cpus, worker_cpus = rank_cpus[:-threads], rank_cpus[-threads:]
    os.sched_setaffinity(0, controller_cpus)
    try:
        with ThreadPoolExecutor(
            max_workers=1, initializer=os.sched_setaffinity, initargs=(0, worker_cpus)
        ) as executor:
            affinity = sorted(executor.submit(os.sched_getaffinity, 0).result())
            if affinity != worker_cpus:
                raise RuntimeError(f"Unexpected CPUAdam thread affinity {affinity}. Expected {worker_cpus}.")
            yield (
                lambda: executor.submit(update).result(),
                {
                    "controller_cpus": controller_cpus,
                    "worker_cpus": affinity,
                    "worker_pid": os.getpid(),
                    "worker_tid": executor.submit(threading.get_native_id).result(),
                    "timing_scope": "submit_adam_result",
                },
            )
    finally:
        os.sched_setaffinity(0, rank_cpus)


def _cpu_time_ns(pid: int) -> int:
    if pid == os.getpid():
        return time.process_time_ns()
    # The command name in parentheses can contain spaces. utime and stime are fields 14 and 15.
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    ticks = int(fields[11]) + int(fields[12])
    return ticks * 1_000_000_000 // os.sysconf("SC_CLK_TCK")


def _run_rank(args: argparse.Namespace) -> dict[str, Any] | None:
    import torch
    import torch.distributed as dist
    from deepspeed.ops.op_builder import CPUAdamBuilder

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != WORLD_SIZE or int(os.environ["LOCAL_WORLD_SIZE"]) != WORLD_SIZE:
        raise RuntimeError("Launch four ranks on one host with torch.distributed.run.")
    if torch.cuda.device_count() < WORLD_SIZE:
        raise RuntimeError("Make at least four CUDA devices visible to the four-rank benchmark.")
    if os.environ.get("OMP_NUM_THREADS") != str(args.threads):
        raise RuntimeError("Set OMP_NUM_THREADS to --threads in the rank environment before launch.")
    if not hasattr(os, "sched_getaffinity"):
        raise RuntimeError("The CPUAdam benchmark requires Linux CPU affinity support.")

    rank_cpus = sorted(os.sched_getaffinity(0))
    if len(rank_cpus) < args.threads + 1:
        raise RuntimeError(
            f"Rank {rank} has {len(rank_cpus)} CPUs. Allocate {args.threads} Adam CPUs plus one controller CPU."
        )

    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo", timeout=timedelta(minutes=5))
    try:
        if args.mode == "process":
            # Every rank shares its FP32 parameter and gradient with its child.
            required = WORLD_SIZE * 2 * args.param_mib * MIB
            available = [shutil.disk_usage("/dev/shm").free if rank == 0 else None]
            dist.broadcast_object_list(available)
            if available[0] < required:
                raise RuntimeError(
                    f"SuperOffload needs at least {required // MIB} MiB free in /dev/shm for this case. "
                    f"Found {available[0] // MIB} MiB."
                )

        torch.set_num_threads(args.threads)
        native = CPUAdamBuilder().load()
        if not getattr(native, "adam_update_releases_gil", False):
            raise RuntimeError("CPUAdam must be built with the Bridge GIL-release patch for this comparison.")

        numel = args.param_mib * MIB // 4
        parameter = torch.full((numel,), 0.01, dtype=torch.float32, device="cpu")
        gradient = torch.full_like(parameter, 0.001)
        initial = parameter[0].item()
        if args.mode == "thread":
            context = _thread_step(parameter=parameter, gradient=gradient, rank_cpus=rank_cpus, threads=args.threads)
        else:
            from cpuadam_superoffload import superoffload_step

            context = superoffload_step(
                parameter=parameter, gradient=gradient, config=ADAM_GROUP, threads=args.threads, rank_cpus=rank_cpus
            )

        samples = []
        with context as (step, worker):
            worker_pid = worker["worker_pid"]
            for _ in range(args.warmup):
                step()
            for _ in range(args.steps):
                dist.barrier()
                start_cpu = _cpu_time_ns(worker_pid)
                start = time.perf_counter_ns()
                native_ms = step()
                end = time.perf_counter_ns()
                cpu_ns = _cpu_time_ns(worker_pid) - start_cpu
                samples.append({"start_ns": start, "end_ns": end, "native_ms": native_ms, "cpu_time_ns": cpu_ns})
            updated = parameter[0].item()
            if not math.isfinite(updated) or updated == initial:
                raise RuntimeError(f"CPUAdam on rank {rank} did not produce a finite parameter update.")

        local = {
            "rank": rank,
            "local_rank": local_rank,
            "gpu_name": torch.cuda.get_device_name(local_rank),
            "rank_pid": os.getpid(),
            "rank_cpus": rank_cpus,
            "cpuadam_module": native.__file__,
            "worker": worker,
            "samples": samples,
            "first_weight_after_updates": updated,
        }
        ranks = [None] * WORLD_SIZE
        dist.all_gather_object(ranks, local)
        if rank == 0:
            return _report(args, ranks, numel)
        return None
    finally:
        dist.destroy_process_group()


def _report(args: argparse.Namespace, ranks: list[dict[str, Any]], numel: int) -> dict[str, Any]:
    if len({rank["cpuadam_module"] for rank in ranks}) != 1:
        raise RuntimeError("Four ranks did not load the same CPUAdam extension path.")
    joint_ms = []
    for group in zip(*(rank["samples"] for rank in ranks), strict=True):
        joint_ms.append(
            (max(sample["end_ns"] for sample in group) - min(sample["start_ns"] for sample in group)) / 1e6
        )
    for rank in ranks:
        wall_ns = [sample["end_ns"] - sample["start_ns"] for sample in rank["samples"]]
        rank["median_step_ms"] = statistics.median(wall_ns) / 1e6
        rank["cpu_time_per_wall"] = sum(sample["cpu_time_ns"] for sample in rank["samples"]) / sum(wall_ns)
        native = [sample["native_ms"] for sample in rank["samples"] if sample["native_ms"] is not None]
        rank["median_native_ms"] = statistics.median(native) if native else None
    median_ms = statistics.median(joint_ms)
    return {
        "mode": args.mode,
        "threads_per_rank": args.threads,
        "param_mib_per_rank": args.param_mib,
        "numel_per_rank": numel,
        "warmup": args.warmup,
        "steps": args.steps,
        "gil_release": True,
        "median_four_rank_ms": median_ms,
        "aggregate_params_per_s": WORLD_SIZE * numel / (median_ms / 1000),
        "aggregate_adam_effective_gb_per_s": WORLD_SIZE * numel * BYTES_PER_PARAM / (median_ms / 1000) / 1e9,
        "ranks": ranks,
    }


def _summarize(directory: Path) -> None:
    records = [json.loads(path.read_text()) for path in directory.glob("*-t*.json")]
    if not records:
        raise RuntimeError(f"No benchmark cases in {directory}")
    fields = (
        "mode",
        "threads_per_rank",
        "param_mib_per_rank",
        "median_four_rank_ms",
        "aggregate_params_per_s",
        "aggregate_adam_effective_gb_per_s",
    )
    with (directory / "summary.tsv").open("w", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter="\t", lineterminator="\n")
        writer.writerow((*fields, "median_native_ms", "median_cpu_time_per_wall"))
        for record in sorted(records, key=lambda row: (row["mode"], row["threads_per_rank"])):
            native = [rank["median_native_ms"] for rank in record["ranks"] if rank["median_native_ms"] is not None]
            values = [f"{record[key]:.6g}" if isinstance(record[key], float) else record[key] for key in fields]
            cpu_time = statistics.median(rank["cpu_time_per_wall"] for rank in record["ranks"])
            writer.writerow([*values, f"{statistics.median(native):.4f}" if native else "NA", f"{cpu_time:.3f}"])
    sys.stdout.write((directory / "summary.tsv").read_text())


def main() -> None:
    """Run one distributed case or summarize completed cases."""
    parser = argparse.ArgumentParser(description=__doc__)
    entry = parser.add_mutually_exclusive_group(required=True)
    entry.add_argument("--mode", choices=("thread", "process"))
    entry.add_argument("--summarize", type=Path)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--param-mib", type=int, default=320)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.summarize is not None:
        _summarize(args.summarize)
        return
    if args.out is None or args.threads is None or min(args.threads, args.param_mib, args.warmup, args.steps) <= 0:
        parser.error("--mode requires --out, --threads and positive size and iteration counts")
    report = _run_rank(args)
    if report is not None:
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
