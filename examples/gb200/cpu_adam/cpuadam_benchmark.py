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

"""Measure steady-state updates of one FP32 tensor with DeepSpeedCPUAdam."""

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path


# Read the parameter, gradient and both moments. Write the parameter and both moments.
BYTES_PER_PARAM = 28


def _step_samples(*, params: int, threads: int, duration: float, warmup: float) -> list[float]:
    import torch
    from deepspeed.ops.adam import DeepSpeedCPUAdam

    torch.set_num_threads(threads)
    param = torch.nn.Parameter(torch.full((params,), 0.01, dtype=torch.float32, device="cpu"))
    param.grad = torch.full_like(param, 1e-3)
    optimizer = DeepSpeedCPUAdam([param], lr=1e-4, weight_decay=0.01, adamw_mode=True, fp32_optimizer_states=True)

    # This first update allocates and touches the two optimizer-state tensors.
    optimizer.step()
    start = time.perf_counter()
    measure_from = start + warmup
    end = start + duration
    samples = []
    while (before := time.perf_counter()) < end:
        optimizer.step()
        after = time.perf_counter()
        if before >= measure_from and after <= end:
            samples.append(after - before)
    return samples


def main() -> None:
    """Run a single thread configuration and write its results to JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--warmup", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.params <= 0 or args.threads <= 0:
        parser.error("--params and --threads must be positive")
    if not math.isfinite(args.duration) or not 0 <= args.warmup < args.duration:
        parser.error("--duration must be finite and larger than the nonnegative --warmup")

    # Capture the mask inherited from numactl before OpenMP pins the main thread.
    binding = {
        "pid": os.getpid(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "cpus": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
    }
    samples = _step_samples(params=args.params, threads=args.threads, duration=args.duration, warmup=args.warmup)
    median_step = statistics.median(samples) if samples else None
    params_per_s = args.params / median_step if median_step is not None else None
    result = {
        "kind": "cpuadam",
        "threads": args.threads,
        "params": args.params,
        "tensor_numel": args.params,
        "duration": args.duration,
        "warmup": args.warmup,
        "binding": binding,
        "iters": len(samples),
        "step_s_median": median_step,
        "params_per_s": params_per_s,
        "gbps_median": statistics.median(args.params * BYTES_PER_PARAM / sample / 1e9 for sample in samples)
        if samples
        else None,
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
