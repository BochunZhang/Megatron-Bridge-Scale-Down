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
"""Run one benchmark scope for every configured micro-batch size and mode.

Usage::

    python -m mbench.run --config mbench.yaml --scope GatedDeltaNet --output-dir results/...

Scopes: GEMM, GatedDeltaNet, GatedAttention, MLP, Experts (case-insensitive).
Writes one JSON record per (case, mbs, mode) to ``<output-dir>/<category>/`` where
``<category>`` is the scope's directory name (gemm, gated_delta_net, ...).
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import platform
from importlib import metadata
from pathlib import Path
from typing import Any

import torch
import yaml

from mbench.cases import BenchCase, BenchContext, CaseFactory
from mbench.categories import experts, gated_attention, gated_delta_net, gemm, mlp
from mbench.gate import LaunchGate
from mbench.megatron_setup import MegatronModules, build_provider, init_single_rank, resolve_dtype
from mbench.timing import TimingConfig, make_step, measure, nvtx_label


LOGGER = logging.getLogger(__name__)
# Scope name (CLI) -> (result / CSV / nsys file name, case factory).
SCOPES: dict[str, tuple[str, CaseFactory]] = {
    "GEMM": (gemm.CATEGORY, gemm.iter_cases),
    "GatedDeltaNet": (gated_delta_net.CATEGORY, gated_delta_net.iter_cases),
    "GatedAttention": (gated_attention.CATEGORY, gated_attention.iter_cases),
    "MLP": (mlp.CATEGORY, mlp.iter_cases),
    "Experts": (experts.CATEGORY, experts.iter_cases),
}


def resolve_scope(name: str) -> str:
    """Return the canonical scope name for a case-insensitive scope or category name."""

    for scope, (category, _factory) in SCOPES.items():
        if name.lower() in {scope.lower(), category}:
            return scope
    raise ValueError(f"unknown scope {name!r}; expected one of {list(SCOPES)}")


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _environment() -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "device": properties.name,
        "sm_count": properties.multi_processor_count,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformer_engine": _package_version("transformer-engine") or _package_version("transformer_engine"),
        "megatron_core": _package_version("megatron-core"),
        "flash_linear_attention": _package_version("flash-linear-attention") or _package_version("fla-core"),
        "host": platform.node(),
    }


def _gpu_telemetry() -> dict[str, float | None]:
    """SM clock and power right after a case (needs pynvml; None otherwise)."""

    readings: dict[str, float | None] = {"sm_clock_mhz": None, "power_w": None}
    try:
        readings["sm_clock_mhz"] = float(torch.cuda.clock_rate())
        readings["power_w"] = float(torch.cuda.power_draw()) / 1000.0
    except (ModuleNotFoundError, RuntimeError, AttributeError):
        pass
    return readings


def _run_case(
    case: BenchCase,
    *,
    category: str,
    timing: TimingConfig,
    gate: LaunchGate,
    round_sizes: dict[str, int],
    output_dir: Path,
    environment: dict[str, Any],
) -> None:
    torch.cuda.reset_peak_memory_stats()
    workload = case.build()
    step = make_step(workload, case.mode)
    result = measure(
        step,
        config=timing,
        gate=gate,
        label=lambda phase, round_index, iteration: nvtx_label(
            category=category,
            case=case.name,
            mbs=case.micro_batch_size,
            mode=case.mode,
            phase=phase,
            round_index=round_index,
            iteration=iteration,
        ),
        iters_per_round=round_sizes.get(case.name, timing.max_iters_per_round),
    )
    round_sizes[case.name] = result.iters_per_round
    summary = result.summary()
    tflops = None if case.flops is None else case.flops / (summary["median_ms"] * 1e-3) / 1e12
    record = {
        "category": category,
        "case": case.name,
        "megatron_module": case.megatron_module,
        "micro_batch_size": case.micro_batch_size,
        "mode": case.mode,
        "params": case.params,
        "flops": case.flops,
        "tflops_median": tflops,
        "timing": {
            **summary,
            "gated": result.gated,
            "gate_note": result.gate_note,
            "iters_per_round": result.iters_per_round,
            "discarded_rounds": result.discarded_rounds,
            "pre_run_iters": timing.pre_run_iters,
            "warmup_iters_per_round": timing.warmup_iters,
        },
        "rounds": result.rounds,
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "telemetry": _gpu_telemetry(),
        "environment": environment,
    }
    path = output_dir / f"{case.name}.mbs{case.micro_batch_size}.{case.mode}.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    LOGGER.info(
        "%s/%s mbs=%d %s: median %.4f ms (p90 %.4f, n=%d, gated=%s)%s",
        category,
        case.name,
        case.micro_batch_size,
        case.mode,
        summary["median_ms"],
        summary["p90_ms"],
        summary["samples"],
        result.gated,
        "" if tflops is None else f", {tflops:.1f} TFLOP/s",
    )


def main() -> None:
    """Parse arguments, build the Megatron modules and time every case of one scope."""

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scope", type=resolve_scope, required=True, help=f"one of {list(SCOPES)}")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--micro-batch-sizes", type=str, default=None, help="comma list overriding the config")
    parser.add_argument("--cases", type=str, default=None, help="comma list of case names to run (default: all)")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    category, iter_cases = SCOPES[args.scope]

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    bench = config["benchmark"]
    timing_section = config["timing"]
    dtype = resolve_dtype(bench["dtype"])
    micro_batch_sizes = (
        [int(value) for value in args.micro_batch_sizes.split(",")]
        if args.micro_batch_sizes
        else [int(value) for value in bench["micro_batch_sizes"]]
    )
    modes = [str(mode) for mode in bench["modes"]]
    timing = TimingConfig(
        pre_run_iters=int(timing_section["pre_run_iters"]),
        warmup_iters=int(timing_section["warmup_iters"]),
        profile_iters=int(timing_section["profile_iters"]),
        max_iters_per_round=int(timing_section["max_iters_per_round"]),
    )

    pg_collection = init_single_rank(device_index=args.device, seed=int(bench["seed"]))
    provider = build_provider(config["model"], sequence_length=int(bench["sequence_length"]), dtype=dtype)
    expert_parallel_size = int(bench["formal_expert_parallel_size"])
    if provider.num_moe_experts % expert_parallel_size:
        raise ValueError("num_moe_experts must be divisible by formal_expert_parallel_size")
    modules = MegatronModules(provider, pg_collection, local_experts=provider.num_moe_experts // expert_parallel_size)
    context = BenchContext(
        modules=modules,
        sequence_length=int(bench["sequence_length"]),
        dtype=dtype,
        formal_expert_parallel_size=expert_parallel_size,
        expert_routings=list(bench["expert_routings"]),
        seed=int(bench["seed"]),
        case_filter=None if args.cases is None else frozenset(args.cases.split(",")),
    )
    output_dir = args.output_dir / category
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = _environment()
    gate = LaunchGate(timeout_s=float(timing_section["gate_timeout_s"]))
    round_sizes: dict[str, int] = {}
    try:
        for case in iter_cases(context, micro_batch_sizes, modes):
            if not context.wants(case.name):
                continue
            torch.manual_seed(int(bench["seed"]))
            _run_case(
                case,
                category=category,
                timing=timing,
                gate=gate,
                round_sizes=round_sizes,
                output_dir=output_dir,
                environment=environment,
            )
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        gate.close()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
