"""Compact MoE causal LM training example for AutoEP and ZeRO-3 leaf.

Launch with DeepSpeed:

    deepspeed --num_gpus 8 train.py --mode autoep --autoep_size 8
    deepspeed --num_gpus 8 train.py --mode zero3_leaf
"""

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

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    Llama4ForCausalLM,
    Llama4TextConfig,
    MixtralConfig,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeTextConfig,
)

import deepspeed

from data_utils import (
    build_hf_batch_generator,
    build_model_config,
    get_tokenizer,
    MockBatchGenerator,
    validate_tokenizer_vocab_size,
)
from init_weights import load_init_weights_artifact
from metrics import MetricsLogger, reduce_loss, reduce_max

logger = logging.getLogger(__name__)


MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "mixtral": {
        "architecture": "mixtral",
        "config_cls": MixtralConfig,
        "display_name": "Mixtral 8x7B",
        "default_tokenizer_name": "mistralai/Mixtral-8x7B-v0.1",
    },
    "qwen3_5_moe": {
        "architecture": "qwen3_5_moe",
        "config_cls": Qwen3_5MoeTextConfig,
        "display_name": "Qwen3.5 MoE",
        "default_tokenizer_name": "Qwen/Qwen3-0.6B",
    },
    "llama4": {
        "architecture": "llama4",
        "config_cls": Llama4TextConfig,
        "display_name": "Llama4 Scout",
        "default_tokenizer_name": "meta-llama/Llama-4-Scout-17B-16E",
    },
}

DEEPSPEED_LEAF_MOE_BLOCK_CLASS = {
    "llama4": "transformers.models.llama4.modeling_llama4.Llama4TextMoe",
    "mixtral": "transformers.models.mixtral.modeling_mixtral.MixtralSparseMoeBlock",
    "qwen3_5_moe": (
        "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe."
        "Qwen3_5MoeSparseMoeBlock"
    ),
}


class ModelPreset(NamedTuple):
    architecture: str
    config_cls: type[Any]
    display_name: str
    num_layers_overridden: bool


class TrainingState(NamedTuple):
    rank: int
    dp_world_size: int
    engine: Any
    batch_gen: Any
    memory_profiler: "MemoryProfiler | None"


def _pinned_tensor_bytes(value: Any) -> int:
    """Return bytes held by pinned tensors in a nested batch structure."""
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size() if value.is_pinned() else 0
    if isinstance(value, dict):
        return sum(_pinned_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_pinned_tensor_bytes(item) for item in value)
    if hasattr(value, "__dict__"):
        return sum(_pinned_tensor_bytes(item) for item in vars(value).values())
    return 0


def _proc_memory_bytes(label: str) -> int:
    """Read a byte-valued field from Linux ``/proc/self/status``."""
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith(f"{label}:"):
                fields = line.split()
                return int(fields[1]) * 1024 if len(fields) >= 2 else 0
    except (OSError, ValueError):
        pass
    return 0


def _pinned_process_bytes() -> tuple[int, int, int]:
    """Return process pinned bytes, VmPin bytes, and VmLck bytes."""
    vm_pin = _proc_memory_bytes("VmPin")
    vm_lck = _proc_memory_bytes("VmLck")
    return max(vm_pin, vm_lck), vm_pin, vm_lck


@dataclass
class MemoryProfiler:
    """Collect step-level CUDA and host-pinned memory samples for one rank."""

    output_path: str
    rank: int
    records: list[dict[str, Any]] = field(default_factory=list)
    _step_start: dict[str, Any] | None = None
    _step_samples: list[dict[str, Any]] = field(default_factory=list)
    _run_start: float = field(default_factory=time.perf_counter)

    def _sample(self, step: int, phase: str, payload: Any = None) -> dict[str, Any]:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            cuda_allocated = torch.cuda.memory_allocated()
            cuda_reserved = torch.cuda.memory_reserved()
            cuda_peak_allocated = torch.cuda.max_memory_allocated()
            cuda_peak_reserved = torch.cuda.max_memory_reserved()
            stats = torch.cuda.memory_stats()
            alloc_count = int(stats.get("allocation.all.current", 0))
            free_count = int(stats.get("free_requests.all.current", 0))
        else:
            cuda_allocated = cuda_reserved = cuda_peak_allocated = cuda_peak_reserved = 0
            alloc_count = free_count = 0
        pinned_tensor = _pinned_tensor_bytes(payload)
        pinned_process, pinned_proc_pin, pinned_proc_lock = _pinned_process_bytes()
        sample = {
            "step": step,
            "phase": phase,
            "time_ms": round((time.perf_counter() - self._run_start) * 1000, 3),
            "cuda_allocated_bytes": cuda_allocated,
            "cuda_reserved_bytes": cuda_reserved,
            "cuda_peak_allocated_bytes": cuda_peak_allocated,
            "cuda_peak_reserved_bytes": cuda_peak_reserved,
            "cuda_allocation_count": alloc_count,
            "cuda_free_count": free_count,
            "pinned_memory_process_bytes": pinned_process,
            "pinned_memory_pinned_bytes": pinned_proc_pin,
            "pinned_memory_locked_bytes": pinned_proc_lock,
            "pinned_tensor_bytes": pinned_tensor,
        }
        return sample

    def start_step(self, step: int) -> None:
        """Start a step window and reset CUDA per-step peak counters."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._step_samples = [self._sample(step, "step_start")]
        self._step_start = self._step_samples[0]

    def sample(self, step: int, phase: str, payload: Any = None) -> None:
        """Capture an intermediate phase sample, such as forward or backward."""
        self._step_samples.append(self._sample(step, phase, payload))

    def end_step(self, step: int) -> None:
        """Close a step and retain a replay_step-compatible summary row."""
        self._step_samples.append(self._sample(step, "step_end"))
        start = self._step_start or self._step_samples[0]
        end = self._step_samples[-1]
        cuda_peak = max(item["cuda_peak_allocated_bytes"] for item in self._step_samples)
        pinned_peak = max(
            max(item["pinned_memory_process_bytes"] for item in self._step_samples),
            max(item["pinned_memory_locked_bytes"] for item in self._step_samples),
            max(item["pinned_tensor_bytes"] for item in self._step_samples),
        )
        self.records.append(
            {
                "step": step,
                "phase": f"step[{step}]",
                "start_time_ms": start["time_ms"],
                "end_time_ms": end["time_ms"],
                "duration_ms": end["time_ms"] - start["time_ms"],
                "cuda_start_bytes": start["cuda_allocated_bytes"],
                "cuda_end_bytes": end["cuda_allocated_bytes"],
                "cuda_peak_bytes": cuda_peak,
                "cuda_reserved_start_bytes": start["cuda_reserved_bytes"],
                "cuda_reserved_end_bytes": end["cuda_reserved_bytes"],
                "cuda_reserved_peak_bytes": max(
                    item["cuda_peak_reserved_bytes"] for item in self._step_samples
                ),
                "cuda_alloc_count": end["cuda_allocation_count"],
                "cuda_free_count": end["cuda_free_count"],
                "pinned_start_bytes": max(
                    start["pinned_memory_process_bytes"], start["pinned_tensor_bytes"]
                ),
                "pinned_end_bytes": max(
                    end["pinned_memory_process_bytes"], end["pinned_tensor_bytes"]
                ),
                "pinned_peak_bytes": pinned_peak,
                "pinned_process_start_bytes": start["pinned_memory_process_bytes"],
                "pinned_process_end_bytes": end["pinned_memory_process_bytes"],
                "pinned_process_peak_bytes": max(
                    item["pinned_memory_process_bytes"] for item in self._step_samples
                ),
                "pinned_locked_start_bytes": start["pinned_memory_locked_bytes"],
                "pinned_locked_end_bytes": end["pinned_memory_locked_bytes"],
                "pinned_tensor_peak_bytes": max(
                    item["pinned_tensor_bytes"] for item in self._step_samples
                ),
                "samples": self._step_samples,
            }
        )
        self._step_start = None
        self._step_samples = []

    def write(self) -> Path:
        """Write the rank-local profile JSON and return its path."""
        path = Path(self.output_path)
        if path.suffix.lower() == ".json":
            path = path.with_name(f"{path.stem}_rank-{self.rank}{path.suffix}")
        else:
            path = path / f"memory_profile_rank-{self.rank}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"rank": self.rank, "records": self.records}
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AutoEP / ZeRO-3 leaf MoE training")
    parser.add_argument("--mode", choices=["autoep", "zero3_leaf"], default="autoep")
    parser.add_argument("--model", choices=sorted(MODEL_PRESETS), default="qwen3_5_moe")
    parser.add_argument(
        "--model_config",
        default=None,
        help="Local Hugging Face config.json; supports a nested text_config object.",
    )
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--num_experts", type=int, default=None)
    parser.add_argument("--autoep_size", type=int, default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--micro_batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_name", default="wikitext")
    parser.add_argument("--dataset_percentage", type=float, default=10.0)
    parser.add_argument(
        "--mock_data",
        action="store_true",
        help="Use random token batches and skip tokenizer/dataset downloads.",
    )
    parser.add_argument("--tokenizer_name", default=None)
    parser.add_argument("--hf_num_dataloader_workers", type=int, default=0)
    parser.add_argument(
        "--load_init_weights",
        type=str,
        default=None,
        help="Load a shared initialization artifact created by utils/prepare_init_weights.py.",
    )
    parser.add_argument("--metrics_out", default=None)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Record per-step CUDA and pinned-memory usage and enable DeepSpeed memory breakdown.",
    )
    parser.add_argument(
        "--profile_out",
        default="memory_profile.json",
        help="Profile JSON path or directory; rank suffixes are added automatically.",
    )
    parser.add_argument(
        "--deepspeed_config",
        default=None,
        help="Optional DeepSpeed JSON config; otherwise an in-memory config is used.",
    )
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    if args.mode == "autoep" and args.autoep_size is None:
        parser.error("--autoep_size is required in AutoEP mode.")
    if args.load_init_weights is not None:
        if not args.load_init_weights.endswith(".safetensors"):
            parser.error("--load_init_weights path must end with '.safetensors'.")
        if not os.path.isfile(args.load_init_weights):
            parser.error(f"--load_init_weights file does not exist: {args.load_init_weights}")

    return args


def resolve_model_preset(args: argparse.Namespace) -> ModelPreset:
    preset = MODEL_PRESETS[args.model]
    architecture = preset["architecture"]
    config_cls = preset["config_cls"]
    num_layers_overridden = args.num_layers is not None
    if args.num_layers is None:
        original_config = build_model_config(
            config_cls,
            None,
            config_json=args.model_config,
            num_experts=args.num_experts,
        )
        args.num_layers = int(original_config.num_hidden_layers)
    if args.tokenizer_name is None:
        args.tokenizer_name = preset["default_tokenizer_name"]
    return ModelPreset(
        architecture,
        config_cls,
        preset["display_name"],
        num_layers_overridden,
    )


def build_model(architecture: str, model_config: Any) -> torch.nn.Module:
    if architecture == "mixtral":
        return AutoModelForCausalLM.from_config(model_config)
    if architecture == "qwen3_5_moe":
        return Qwen3_5MoeForCausalLM(model_config)
    if architecture == "llama4":
        return Llama4ForCausalLM(model_config)
    raise ValueError(f"Unsupported architecture: {architecture!r}")


def num_experts_for_config(architecture: str, model_config: Any) -> int:
    if architecture in {"mixtral", "llama4"}:
        return int(model_config.num_local_experts)
    if architecture == "qwen3_5_moe":
        return int(model_config.num_experts)
    raise ValueError(f"Unsupported architecture: {architecture!r}")


def build_deepspeed_config(
    mode: str,
    architecture: str,
    micro_batch_size: int,
    grad_accum: int,
    autoep_size: int | None,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "bf16": {"enabled": True},
        "optimizer": {"type": "AdamW", "params": {"lr": 1e-4}},
        "scheduler": {
            "type": "WarmupCosineLR",
            "params": {
                "total_num_steps": 1000,
                "warmup_min_ratio": 0,
                "warmup_num_steps": 100,
                "cos_min_ratio": 0.001,
                "warmup_type": "linear",
            },
        },
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "gradient_accumulation_steps": grad_accum,
        "steps_per_print": 10,
    }
    if mode == "autoep":
        config["zero_optimization"] = {"stage": 1}
        config["expert_parallel"] = {
            "enabled": True,
            "autoep_size": autoep_size,
            "preset_model": architecture,
        }
    else:
        config["zero_optimization"] = {
            "stage": 3,
            "stage3_param_persistence_threshold": 1e5,
            "leaf_module": {"classes": [DEEPSPEED_LEAF_MOE_BLOCK_CLASS[architecture]]},
        }
    return config


def load_deepspeed_config(path: str | None, fallback: dict[str, Any]) -> dict[str, Any]:
    """Load a user-provided DeepSpeed JSON config or return the generated config."""
    if path is None:
        return fallback
    with open(path, encoding="utf-8") as config_file:
        config = json.load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"DeepSpeed config must be a JSON object: {path}")
    return config


def enable_memory_profile(config: dict[str, Any]) -> dict[str, Any]:
    """Enable DeepSpeed's actual ``memory_breakdown`` option.

    ``memory_break`` is accepted as a user-facing alias for this example, but
    DeepSpeed's configuration key is ``memory_breakdown``.
    """
    config["memory_break"] = True
    config["memory_breakdown"] = True
    return config


def validate_autoep_args(
    architecture: str,
    autoep_size: int,
    num_experts: int,
    world_size: int,
) -> None:
    valid_sizes = [
        size
        for size in range(1, min(num_experts, world_size) + 1)
        if num_experts % size == 0 and world_size % size == 0
    ]
    if autoep_size not in valid_sizes:
        raise ValueError(
            f"Invalid autoep_size={autoep_size} for architecture={architecture!r}; "
            f"num_experts={num_experts}, world_size={world_size}, "
            f"valid sizes={valid_sizes}"
        )

    from deepspeed.module_inject.auto_ep_config import PRESET_MODELS

    preset_id = architecture
    if preset_id not in PRESET_MODELS:
        raise ValueError(
            f"DeepSpeed does not provide AutoEP preset_model={preset_id!r}; "
            f"available presets={sorted(PRESET_MODELS)}"
        )


def setup_distributed(args: argparse.Namespace) -> tuple[int, int]:
    deepspeed.init_distributed()
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
        if local_rank >= 0:
            torch.cuda.set_device(local_rank)
    return rank, world_size


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_initial_weights(
    path: str | None,
    model: torch.nn.Module,
    args: argparse.Namespace,
    model_config: Any,
    rank: int,
) -> None:
    if path is None:
        return
    try:
        context = load_init_weights_artifact(path, model, args=args, model_config=model_config)
    except Exception as exc:
        logger.error("Failed to load init weights artifact: %s", exc)
        sys.exit(2)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    if rank == 0:
        logger.info(
            "Loaded init weights artifact %s (sha256=%s)",
            context["init_weights_path"],
            context["init_weights_sha256"],
        )


def prepare_training(args: argparse.Namespace) -> TrainingState:
    if args.metrics_out is None:
        args.metrics_out = f"metrics_{args.mode}.csv"

    rank, world_size = setup_distributed(args)
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format=f"[rank {rank}] %(levelname)s: %(message)s",
    )
    seed_everything(args.seed)

    preset = resolve_model_preset(args)
    model_config = build_model_config(
        preset.config_cls,
        args.num_layers,
        config_json=args.model_config,
        num_experts=args.num_experts,
    )
    num_experts = num_experts_for_config(preset.architecture, model_config)
    autoep_size = args.autoep_size if args.mode == "autoep" else None

    if args.mode == "autoep":
        try:
            validate_autoep_args(preset.architecture, autoep_size, num_experts, world_size)
        except ValueError as exc:
            logger.error("AutoEP preflight failed: %s", exc)
            sys.exit(2)

    generated_ds_config = build_deepspeed_config(
        args.mode,
        preset.architecture,
        args.micro_batch_size,
        args.grad_accum,
        autoep_size,
    )
    try:
        ds_config = load_deepspeed_config(args.deepspeed_config, generated_ds_config)
        if args.profile:
            ds_config = enable_memory_profile(ds_config)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.error("Failed to load DeepSpeed config: %s", exc)
        sys.exit(2)

    tokenizer_info = None
    if not args.mock_data:
        try:
            tokenizer = get_tokenizer(args.tokenizer_name, trust_remote_code=True)
            tokenizer_info = validate_tokenizer_vocab_size(
                tokenizer,
                args.tokenizer_name,
                model_config.vocab_size,
            )
        except ValueError as exc:
            logger.error("Tokenizer validation failed: %s", exc)
            sys.exit(2)

    if rank == 0:
        logger.info("Mode: %s", args.mode)
        logger.info(
            "Model: %s (%s), layers=%s%s, hidden=%s, experts=%s",
            args.model,
            preset.display_name,
            args.num_layers,
            " from --num_layers" if preset.num_layers_overridden else " original default",
            model_config.hidden_size,
            num_experts,
        )
        if tokenizer_info is not None:
            logger.info(
                "Tokenizer %s: len=%s, vocab_size=%s, model_vocab_size=%s",
                args.tokenizer_name,
                tokenizer_info["tokenizer_len"],
                tokenizer_info["tokenizer_vocab_size"],
                tokenizer_info["model_vocab_size"],
            )
        else:
            logger.info("Using random mock data; no tokenizer or dataset will be downloaded.")
        logger.info(
            "Seq len=%s, micro batch=%s, grad_accum=%s, steps=%s",
            args.seq_len,
            args.micro_batch_size,
            args.grad_accum,
            args.steps,
        )

    model = build_model(preset.architecture, model_config)
    load_initial_weights(args.load_init_weights, model, args, model_config, rank)

    try:
        engine, _, _, _ = deepspeed.initialize(
            model=model,
            config=ds_config,
            model_parameters=model.parameters(),
        )
    except Exception as exc:
        logger.error("deepspeed.initialize() failed: %s", exc)
        sys.exit(2)

    import deepspeed.comm as dist_comm

    dp_rank = dist_comm.get_rank(engine.data_parallel_group)
    dp_world_size = engine.dp_world_size
    memory_profiler = MemoryProfiler(args.profile_out, rank) if args.profile else None
    if args.mock_data:
        batch_gen = MockBatchGenerator(
            vocab_size=model_config.vocab_size,
            seq_len=args.seq_len,
            micro_batch_size=args.micro_batch_size,
            seed=args.seed,
            rank=rank,
        )
    else:
        batch_gen = build_hf_batch_generator(
            dataset_name=args.dataset_name,
            dataset_percentage=args.dataset_percentage,
            tokenizer_name=args.tokenizer_name,
            expected_vocab_size=model_config.vocab_size,
            seq_len=args.seq_len,
            micro_batch_size=args.micro_batch_size,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            seed=args.seed,
            rank=rank,
            hf_num_dataloader_workers=args.hf_num_dataloader_workers,
        )
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    return TrainingState(
        rank=rank,
        dp_world_size=dp_world_size,
        engine=engine,
        batch_gen=batch_gen,
        memory_profiler=memory_profiler,
    )


def train(args: argparse.Namespace, state: TrainingState) -> None:
    metrics_logger = MetricsLogger(args.metrics_out, state.rank)
    if state.rank == 0:
        logger.info(
            "Starting training for %s optimizer steps (warmup=%s).",
            args.steps,
            args.warmup_steps,
        )

    for step in range(args.steps):
        if state.memory_profiler is not None:
            state.memory_profiler.start_step(step)
        sync_cuda()
        step_start = time.time()
        last_loss = None

        for accum_idx in range(args.grad_accum):
            batch = state.batch_gen.get_batch(step, accum_idx)
            if state.memory_profiler is not None:
                state.memory_profiler.sample(step, "batch_loaded", batch)
            outputs = state.engine(
                input_ids=batch.input_ids.to(state.engine.device),
                attention_mask=batch.attention_mask.to(state.engine.device),
                labels=batch.labels.to(state.engine.device),
            )
            if state.memory_profiler is not None:
                state.memory_profiler.sample(step, "forward_end")
            loss = outputs.loss
            last_loss = loss.detach().clone()
            state.engine.backward(loss)
            if state.memory_profiler is not None:
                state.memory_profiler.sample(step, "backward_end")
            state.engine.step()
            if state.memory_profiler is not None:
                state.memory_profiler.sample(step, "optimizer_end")

        sync_cuda()
        iter_time = time.time() - step_start
        reduced_loss = reduce_loss(
            last_loss,
            state.dp_world_size,
            group=state.engine.data_parallel_group,
        )
        if not math.isfinite(reduced_loss):
            if state.rank == 0:
                logger.error("Non-finite loss at step %s: loss=%s", step, reduced_loss)
            sys.exit(3)

        if step == args.warmup_steps - 1 and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        if step >= args.warmup_steps and step % args.log_interval == 0:
            max_iter_time = reduce_max(iter_time)
            mem_allocated = torch.cuda.memory_allocated()
            mem_peak_allocated = torch.cuda.max_memory_allocated()
            mem_peak_reserved = torch.cuda.max_memory_reserved()
            global_tokens_per_sec = (
                args.seq_len
                * args.micro_batch_size
                * args.grad_accum
                * state.dp_world_size
                / max_iter_time
                if max_iter_time > 0
                else 0
            )

            metrics_logger.log_step(
                {
                    "step": step,
                    "loss": reduced_loss,
                    "iter_time_sec": max_iter_time,
                    "global_tokens_per_sec": global_tokens_per_sec,
                    "cuda_memory_allocated_bytes": mem_allocated,
                    "cuda_peak_memory_allocated_bytes": mem_peak_allocated,
                    "cuda_peak_memory_reserved_bytes": mem_peak_reserved,
                }
            )
            if state.rank == 0:
                logger.info(
                    "Step %s: loss=%.6f, time=%.3fs, global_tps=%.0f, peak_mem=%.2f GiB",
                    step,
                    reduced_loss,
                    max_iter_time,
                    global_tokens_per_sec,
                    mem_peak_allocated / (1024**3),
                )

        if state.memory_profiler is not None:
            state.memory_profiler.end_step(step)

    metrics_logger.close()
    if state.memory_profiler is not None:
        profile_path = state.memory_profiler.write()
        if state.rank == 0:
            logger.info("Memory profile written to %s", profile_path)
    if state.rank == 0:
        logger.info("Metrics written to %s", args.metrics_out)


def main() -> None:
    args = parse_args()
    state = prepare_training(args)
    train(args, state)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        logging.error("Unhandled exception: %s", exc, exc_info=True)
        sys.exit(1)
