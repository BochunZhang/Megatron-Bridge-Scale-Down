#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Launch the 72 Qwen3.5 FSDP3 communication experiments at GBS=32, one at a time."""

import argparse
import json
import logging
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path


logger = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
MODELS = {
    "397b": "Qwen/Qwen3.5-397B-A17B",
    "122b": "Qwen/Qwen3.5-122B-A10B",
    "27b": "Qwen/Qwen3.5-27B",
}
SEQUENCES = {1: (4096, 8192, 32768, 65536), 4: (4096, 32768, 131072, 262144)}
MICRO_BATCH_SIZES = (1, 2, 4)
GLOBAL_BATCH_SIZE = 32


def main() -> None:
    """Validate the matrix, record commands, launch training, and export traces."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["all", *MODELS], default="all")
    parser.add_argument("--cp", type=int, choices=(1, 4))
    parser.add_argument("--seq-length", type=int)
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--nnodes", type=int, default=int(os.environ.get("NNODES", "1")))
    parser.add_argument("--node-rank", type=int, default=int(os.environ.get("NODE_RANK", "0")))
    parser.add_argument("--master-addr", default=os.environ.get("MASTER_ADDR"))
    parser.add_argument("--master-port", type=int, default=int(os.environ.get("MASTER_PORT", "29500")))
    parser.add_argument(
        "--micro-batch-size", type=int, choices=MICRO_BATCH_SIZES, help="Select one MBS (default: all)"
    )
    parser.add_argument("--train-iters", type=int, default=10)
    parser.add_argument("--profile-start", type=int, default=5)
    parser.add_argument("--profile-end", type=int, default=8)
    parser.add_argument("--profile", choices=("nsys", "none"), default="nsys")
    parser.add_argument("--communication-unit-size", type=int, help="FSDP prefetch/RS queue size in elements")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/01-analyse/04-fsdp3-comm")
    parser.add_argument("--run-date", default=datetime.now().strftime("%y%m%d-%H%M%S"), help="Run date: yymmdd-hhmmss")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for key in ("nproc_per_node", "nnodes", "train_iters"):
        if getattr(args, key) < 1:
            parser.error(f"{key} must be positive")
    try:
        run_date = datetime.strptime(args.run_date, "%y%m%d-%H%M%S")
    except ValueError:
        parser.error("run-date must use yymmdd-hhmmss")
    if run_date.strftime("%y%m%d-%H%M%S") != args.run_date:
        parser.error("run-date must use yymmdd-hhmmss")
    if not 0 <= args.node_rank < args.nnodes:
        parser.error("node-rank must be in [0, nnodes)")
    if args.nnodes > 1 and not args.master_addr:
        parser.error("multi-node runs require --master-addr / MASTER_ADDR")
    if args.profile == "nsys" and not 0 < args.profile_start < args.profile_end < args.train_iters:
        parser.error("require 0 < profile-start < profile-end < train-iters")
    if args.communication_unit_size is not None and args.communication_unit_size < 1:
        parser.error("communication-unit-size must be positive")
    world_size = args.nnodes * args.nproc_per_node
    cases = [
        (model, cp, seq, mbs)
        for model in MODELS
        for cp, lengths in SEQUENCES.items()
        for seq in lengths
        for mbs in MICRO_BATCH_SIZES
        if args.model in ("all", model)
        and args.cp in (None, cp)
        and args.seq_length in (None, seq)
        and args.micro_batch_size in (None, mbs)
    ]
    if not cases:
        parser.error("no case matches the requested CP/sequence-length matrix")
    if world_size < 2 or any(world_size % cp for _, cp, _, _ in cases):
        parser.error("FSDP needs at least 2 GPUs; world size must be divisible by each selected CP")
    if any(GLOBAL_BATCH_SIZE % (mbs * (world_size // cp)) for _, cp, _, mbs in cases):
        parser.error("GBS=32 must be divisible by MBS * (world size / CP) for every selected case")
    env = os.environ.copy()
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "32")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), str(ROOT / "3rdparty/Megatron-LM"), env.get("PYTHONPATH", "")]
    )
    for model, cp, seq, mbs in cases:
        case_name = (
            f"model_{model}-fsdp_3-mbs_{mbs}-seq_{seq}-cp_{cp}-gbs_{GLOBAL_BATCH_SIZE}"
            f"-gpus_{world_size}-profile_{args.profile}-date_{args.run_date}"
        )
        run_dir = args.output_dir.resolve() / case_name / f"node{args.node_rank}"
        num_microbatches = GLOBAL_BATCH_SIZE // (mbs * (world_size // cp))
        command = ["uv", "run", "--no-sync", "python", "-m", "torch.distributed.run"]
        command += [f"--nproc_per_node={args.nproc_per_node}", f"--nnodes={args.nnodes}"]
        if args.nnodes == 1:
            command += ["--standalone"]
        else:
            command += [
                f"--node_rank={args.node_rank}",
                f"--master_addr={args.master_addr}",
                f"--master_port={args.master_port}",
            ]
        if args.profile == "nsys":
            # One profiler per worker: one rank stopping capture cannot truncate another rank.
            command += [
                "--no-python",
                "nsys",
                "profile",
                "--trace=cuda,nvtx",
                "--sample=none",
                "--cpuctxsw=none",
                "--capture-range=cudaProfilerApi",
                "--capture-range-end=stop",
                f"--output={run_dir}/profile-rank%q{{RANK}}",
                "uv",
                "run",
                "--no-sync",
                "python",
            ]
        command += [
            str(HERE / "train.py"),
            "--model",
            model,
            "--cp",
            str(cp),
            "--seq-length",
            str(seq),
            "--micro-batch-size",
            str(mbs),
            "--global-batch-size",
            str(GLOBAL_BATCH_SIZE),
            "--train-iters",
            str(args.train_iters),
            "--profile",
            args.profile,
            "--profile-start",
            str(args.profile_start),
            "--profile-end",
            str(args.profile_end),
            "--output-dir",
            str(run_dir),
        ]
        if args.communication_unit_size is not None:
            command += ["--communication-unit-size", str(args.communication_unit_size)]
        logger.info(
            "model=%s cp=%d seq=%d mbs=%d layers=8 experts=%s dp=%d fsdp=%d gbs=%d num_microbatches=%d\n%s",
            model,
            cp,
            seq,
            mbs,
            "dense" if model == "27b" else 64,
            world_size // cp,
            world_size,
            GLOBAL_BATCH_SIZE,
            num_microbatches,
            shlex.join(command),
        )
        if args.dry_run:
            continue
        run_dir.mkdir(parents=True, exist_ok=False)
        manifest = dict(
            vars(args),
            output_dir=str(run_dir),
            model=MODELS[model],
            cp=cp,
            seq_length=seq,
            micro_batch_size=mbs,
            global_batch_size=GLOBAL_BATCH_SIZE,
            num_microbatches=num_microbatches,
            world_size=world_size,
            command=command,
            cuda_device_max_connections=env["CUDA_DEVICE_MAX_CONNECTIONS"],
        )
        (run_dir / "launch.json").write_text(json.dumps(manifest, indent=2) + "\n")
        with (run_dir / "train.log").open("w") as log:
            # Abort on failure, including OOM. All nodes must run the same matrix in the same order.
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        if args.profile == "nsys":
            reports = sorted(run_dir.glob("*.nsys-rep"))
            if len(reports) != args.nproc_per_node:
                raise RuntimeError(f"Expected {args.nproc_per_node} reports, found {len(reports)} in {run_dir}")
            for report in reports:
                database = report.with_suffix(".sqlite")
                subprocess.run(["nsys", "export", "--type=sqlite", f"--output={database}", str(report)], check=True)
                subprocess.run(
                    [
                        "uv",
                        "run",
                        "--no-sync",
                        "python",
                        str(HERE / "analyse_nsys.py"),
                        str(database),
                        "--output",
                        str(report.with_suffix(".json")),
                    ],
                    cwd=ROOT,
                    env=env,
                    check=True,
                )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
