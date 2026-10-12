"""Compact causal LM training example for AutoEP, ZeRO-3 leaf, and Qwen3.5 text.

The DeepSpeed runtime config is loaded from a JSON file via --deepspeed_config
(same convention as finetune_zero3.py); see ds_config.json for an example.
Only model-dependent keys (AutoEP expert_parallel, ZeRO-3 leaf module classes)
are applied on top of the loaded file. Batch settings remain in the JSON file.

Every run requires PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (checked
at startup): ZeRO-3 and the act+cpu offload/restore cycle fragment the
default allocator.

Launch with DeepSpeed:

    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    deepspeed --num_gpus 8 train.py --deepspeed_config ds_config.json \
        --mode autoep --autoep_size 8
    deepspeed --num_gpus 8 train.py --deepspeed_config ds_config.json \
        --mode zero3_leaf
    deepspeed --num_gpus 1 train.py --deepspeed_config ds_config.json \
        --model Qwen/Qwen3.5-9B --mode zero3_leaf

Launch with profiling / memory snapshots:

    deepspeed --num_gpus 1 train.py --deepspeed_config ds_config.json \
        --model Qwen/Qwen3.5-9B --mode zero3_leaf \
        --use_pytorch_profiler --record_memory_history \
        --profile_step_start 5 --profile_step_end 10 \
        --profile_ranks 0 --memory_snapshot_path snapshot.pickle

Launch with recompute (act) and recompute + CPU activation offload (act+cpu):

    # act: HF layerwise gradient checkpointing (use_reentrant=False)
    deepspeed --num_gpus 4 train.py --deepspeed_config ds_config.json \
        --model Qwen/Qwen3.5-35B-A3B --mode zero3_leaf --activation_checkpointing

    # act+cpu: additionally offload checkpointed layer inputs to CPU via
    # DeepSpeed CheckpointHiddenStatesOffload (keep the default
    # DS_PIN_MEMORY_BACKEND=torch)
    deepspeed --num_gpus 4 train.py --deepspeed_config ds_config.json \
        --model Qwen/Qwen3.5-35B-A3B --mode zero3_leaf \
        --activation_checkpointing --cpu_checkpointing
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import pickle
import random
import sys
import time
from collections.abc import Iterator
from typing import Any, NamedTuple

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    enable_full_determinism,
)

import deepspeed

from config import ProfilingConfig
from data_utils import (
    build_hf_batch_generator,
    build_model_config,
    get_tokenizer,
    validate_tokenizer_vocab_size,
)
from init_weights import load_init_weights_artifact
from metrics import MetricsLogger, reduce_loss, reduce_max
from profiling import (
    handle_profiling_step,
    handle_profiling_stop,
    initialize_pytorch_profiler,
    should_profile_rank,
)
from train_utils import start_memory_history_recording

logger = logging.getLogger(__name__)


class TrainingState(NamedTuple):
    rank: int
    dp_world_size: int
    engine: Any
    batch_gen: Any
    profiling: ProfilingConfig
    offload_ctx: Any = None


def expandable_segments_enabled() -> bool:
    """Return True when PYTORCH_CUDA_ALLOC_CONF requests expandable_segments:True.

    Parses the comma-separated ``key:value`` options of the env var so that
    combined settings (e.g. ``expandable_segments:True,max_split_size_mb:512``)
    are recognized as well.
    """
    for option in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "").split(","):
        key, _, value = option.partition(":")
        if key.strip().lower() == "expandable_segments":
            return value.strip().lower() == "true"
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AutoEP / ZeRO-3 leaf MoE training")
    # Adds --deepspeed_config (same convention as finetune_zero3.py).
    parser = deepspeed.add_config_arguments(parser)
    parser.add_argument("--mode", choices=["autoep", "zero3_leaf", "dense"], default="autoep")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-9B",
        help=(
            "Hugging Face model name or local path, e.g. Qwen/Qwen3.5-9B (dense) or "
            "Qwen/Qwen3.5-35B-A3B (MoE). The config is loaded via "
            "AutoConfig.from_pretrained(...).text_config; MoE is detected by an "
            "int num_experts field in the config."
        ),
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a Hugging Face config field; may be repeated.",
    )
    parser.add_argument("--autoep_size", type=int, default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--test",
        action="store_true",
        help="Diagnose non-finite loss, master gradients, master weights and resident model parameters; write JSON beside metrics_out.",
    )
    parser.add_argument("--dataset_name", default="wikitext")
    parser.add_argument("--dataset_percentage", type=float, default=10.0)
    parser.add_argument("--tokenizer_name", default=None)
    parser.add_argument("--hf_num_dataloader_workers", type=int, default=0)
    parser.add_argument(
        "--load_init_weights",
        type=str,
        default=None,
        help="Load a shared initialization artifact created by utils/prepare_init_weights.py.",
    )
    parser.add_argument(
        "--activation_checkpointing",
        action="store_true",
        help=(
            "Enable HF layerwise gradient checkpointing (act) with use_reentrant=False; "
            "also disables model.config.use_cache."
        ),
    )
    parser.add_argument(
        "--cpu_checkpointing",
        action="store_true",
        help=(
            "Offload checkpointed layer input hidden_states to CPU (act+cpu) via DeepSpeed "
            "CheckpointHiddenStatesOffload; requires --activation_checkpointing."
        ),
    )
    parser.add_argument("--metrics_out", default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    # Profiling arguments. All default to None so ProfilingConfig owns the
    # actual defaults; ProfilingConfig.from_args(args) parses them generically.
    parser.add_argument(
        "--profile",
        choices=["nsys"],
        default=None,
        help="Enable Nsight Systems profiling; capture is controlled by the profile step range.",
    )
    parser.add_argument(
        "--profile_step_start",
        type=int,
        default=None,
        help="Global step to start profiling.",
    )
    parser.add_argument(
        "--profile_step_end",
        type=int,
        default=None,
        help="Global step to stop profiling; memory snapshot is dumped at this step.",
    )
    parser.add_argument(
        "--profile_ranks",
        type=int,
        nargs="+",
        default=None,
        help="Global ranks to profile.",
    )
    parser.add_argument(
        "--record_memory_history",
        action="store_true",
        default=None,
        help="Record CUDA memory history and dump a snapshot pickle at profile_step_end.",
    )
    parser.add_argument(
        "--memory_snapshot_path",
        type=str,
        default=None,
        help="Memory history pickle path; the rank is inserted before the extension.",
    )
    parser.add_argument(
        "--use_pytorch_profiler",
        action="store_true",
        default=None,
        help="Enable the built-in PyTorch profiler (chrome traces under ./torch_profile).",
    )
    parser.add_argument(
        "--tensorboard_dir",
        type=str,
        default=None,
        help="Directory for PyTorch profiler TensorBoard output.",
    )
    args = parser.parse_args()

    if args.deepspeed_config is None:
        parser.error("--deepspeed_config is required: pass the path to a DeepSpeed JSON config.")
    if not os.path.isfile(args.deepspeed_config):
        parser.error(f"--deepspeed_config file does not exist: {args.deepspeed_config}")
    try:
        args.micro_batch_size, args.grad_accum = load_batch_settings(args.deepspeed_config)
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(f"invalid batch settings in --deepspeed_config: {exc}")
    if not expandable_segments_enabled():
        parser.error(
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is required for every run: "
            "ZeRO-3 and the act+cpu offload/restore cycle fragment the default allocator "
            "(see README)."
        )
    
    if args.mode == "autoep" and args.autoep_size is None:
        parser.error("--autoep_size is required in AutoEP mode.")
    if args.load_init_weights is not None:
        if not args.load_init_weights.endswith(".safetensors"):
            parser.error("--load_init_weights path must end with '.safetensors'.")
        if not os.path.isfile(args.load_init_weights):
            parser.error(f"--load_init_weights file does not exist: {args.load_init_weights}")
    if args.cpu_checkpointing and not args.activation_checkpointing:
        parser.error(
            "--cpu_checkpointing requires --activation_checkpointing: the offload ctx only "
            "marks inputs of HF GradientCheckpointingLayer checkpointed layers."
        )
    if args.cpu_checkpointing and os.environ.get("DS_PIN_MEMORY_BACKEND", "torch").lower() == "native":
        parser.error(
            "DS_PIN_MEMORY_BACKEND=native is incompatible with --cpu_checkpointing: the native "
            "backend uses mlock without cudaHostRegister, which stalls side-stream DMA. "
            "Keep the default 'torch' backend."
        )

    return args


def is_expert_config(model_config: Any) -> bool:
    """Return True when the HF config describes an expert (MoE) model.

    Detection rule: an ``int``-typed ``num_experts`` field on the (text)
    config, e.g. ``Qwen3_5MoeTextConfig.num_experts``. Dense configs such as
    ``Qwen3_5TextConfig`` do not define the field.
    """
    num_experts = getattr(model_config, "num_experts", None)
    return isinstance(num_experts, int) and not isinstance(num_experts, bool)


def build_model(model_config: Any) -> torch.nn.Module:
    """Instantiate a randomly initialized causal LM from the (overridden) config.

    The text-backbone model types (e.g. ``qwen3_5_text`` / ``qwen3_5_moe_text``)
    are registered in the Auto causal-LM mapping, so no per-family model
    presets are needed.
    """
    return AutoModelForCausalLM.from_config(model_config)


def resolve_autoep_preset_name(model_type: str) -> str:
    """Map an HF ``model_type`` to the DeepSpeed AutoEP preset name.

    AutoEP presets declare the HF model types they support via
    ``MoEModelPreset.hf_model_types`` (e.g. preset ``qwen3_5_moe`` matches
    ``qwen3_5_moe_text``), so the preset is resolved from the loaded config
    instead of a hardcoded architecture table.
    """
    from deepspeed.module_inject.auto_ep_config import PRESET_MODELS

    for preset_name, preset in PRESET_MODELS.items():
        if model_type in tuple(getattr(preset, "hf_model_types", ()) or ()):
            return preset_name
    raise ValueError(
        f"No DeepSpeed AutoEP preset supports model_type={model_type!r}; "
        f"available presets={sorted(PRESET_MODELS)}"
    )


def find_moe_leaf_module_class(model: torch.nn.Module) -> str | None:
    """Locate the MoE block class path for ZeRO-3 leaf-module injection.

    Replaces the old static ``DEEPSPEED_LEAF_MOE_BLOCK_CLASS`` table: scans
    the instantiated model for the HF MoE block module (class names like
    ``*MoeBlock`` / ``*SparseMoeBlock`` / ``*TextMoe``) and returns its
    fully-qualified ``module.Class`` path, or None when not found.
    """
    for module in model.modules():
        class_name = type(module).__name__
        if "MoeBlock" in class_name or class_name.endswith("TextMoe"):
            return f"{type(module).__module__}.{class_name}"
    return None

def optimizer_offload_enabled(ds_config: dict[str, Any]) -> bool:
    """Return True when zero-offload (cpu/nvme) or super-offload is enabled.

    Mirrors DeepSpeed ``engine.zero_use_cpu_optimizer()`` (offload_optimizer
    device is cpu/nvme) plus the SuperOffload flag. When enabled, the engine
    builds its own ``DeepSpeedCPUAdam`` from the ds_config ``optimizer``
    section (``get_optimizer_configuration``), and SuperOffload additionally
    runs the real CPU Adam inside a spawned worker process
    (``SuperOffloadCPUOptimizer``) — a client-side CPUAdam would only
    duplicate that state, so train.py must not create one.
    """
    zero_cfg = ds_config.get("zero_optimization") or {}
    offload_optimizer = zero_cfg.get("offload_optimizer") or {}
    device = str(offload_optimizer.get("device", "none")).lower()
    return device in ("cpu", "nvme") or bool(offload_optimizer.get("super_offload", False))


def setup_activation_checkpointing(model: torch.nn.Module) -> None:
    """Enable HF layerwise gradient checkpointing (act), mirroring finetune_zero3.py.

    HF gradient checkpointing wraps each decoder layer in a checkpoint. The
    non-reentrant mode is required: reentrant checkpointing is incompatible
    with ZeRO-3, and the CheckpointHiddenStatesOffload marker patch targets
    the non-reentrant path. ``use_cache`` conflicts with gradient checkpointing
    and must be disabled.
    """
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )


def create_offload_ctx_manager() -> Any:
    """Create the DeepSpeed CheckpointHiddenStatesOffload ctx manager (act+cpu).

    Create once and reuse the same manager for every training step. Entering
    the ctx patches HF ``GradientCheckpointingLayer.__call__`` so each
    checkpointed layer's input hidden_states are marked; pack hooks then
    asynchronously D2H the marked activations into pinned CPU buffers (side
    stream, overlapped with compute) and H2D them back when backward needs
    them. Forward and backward of a step must run inside the same ctx.
    """
    from deepspeed.runtime.activation_checkpointing.offload_activations import (
        get_checkpoint_hidden_states_offloading_ctx_manager,
    )

    return get_checkpoint_hidden_states_offloading_ctx_manager(
        use_pin_memory=True,
        use_streams=True,
    )


def build_deepspeed_config(
    config_path: str,
    mode: str,
    *,
    autoep_size: int | None = None,
    autoep_preset: str | None = None,
    leaf_module_class: str | None = None,
) -> dict[str, Any]:
    """Load the DeepSpeed JSON config from ``config_path`` and adjust it per model/mode.

    Mirrors finetune_zero3.py: the base config (bf16, optimizer, scheduler,
    zero_optimization, offload knobs) comes from the given file instead of
    being constructed here. Adjustments applied on top of the loaded file:

    - Batch settings remain exclusively in the JSON file, so the DeepSpeed
      engine, the batch generator, and the accumulation loop share one source
      of truth.
    - ``autoep`` mode: injects the ``expert_parallel`` block with the preset
      resolved from the HF config ``model_type`` (see
      ``resolve_autoep_preset_name``).
    - ``zero3_leaf`` mode with an MoE model: injects the ZeRO-3
      ``leaf_module`` class list detected from the instantiated model (see
      ``find_moe_leaf_module_class``).
    """
    with open(config_path, encoding="utf-8") as f:
        config: dict[str, Any] = json.load(f)

    config.pop("train_batch_size", None)
    validate_batch_settings(config)

    if mode == "autoep":
        if autoep_preset is None:
            raise ValueError("autoep mode requires a resolved AutoEP preset (MoE model).")
        config["expert_parallel"] = {
            "enabled": True,
            "autoep_size": autoep_size,
            "preset_model": autoep_preset,
        }
    elif mode == "zero3_leaf":
        if leaf_module_class is not None:
            config["zero_optimization"]["leaf_module"] = {
                "classes": [leaf_module_class]
            }

    return config


def load_batch_settings(config_path: str) -> tuple[int, int]:
    """Read the per-GPU micro-batch and accumulation settings from JSON."""
    with open(config_path, encoding="utf-8") as f:
        config: dict[str, Any] = json.load(f)
    return validate_batch_settings(config)


def validate_batch_settings(config: dict[str, Any]) -> tuple[int, int]:
    """Validate and return the batch settings from a DeepSpeed config."""
    micro_batch_size = config["train_micro_batch_size_per_gpu"]
    grad_accum = config["gradient_accumulation_steps"]
    if (
        isinstance(micro_batch_size, bool)
        or not isinstance(micro_batch_size, int)
        or micro_batch_size <= 0
    ):
        raise ValueError("train_micro_batch_size_per_gpu must be a positive integer")
    if (
        isinstance(grad_accum, bool)
        or not isinstance(grad_accum, int)
        or grad_accum <= 0
    ):
        raise ValueError("gradient_accumulation_steps must be a positive integer")
    return micro_batch_size, grad_accum


def validate_autoep_args(
    preset_name: str,
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
            f"Invalid autoep_size={autoep_size} for AutoEP preset={preset_name!r}; "
            f"num_experts={num_experts}, world_size={world_size}, "
            f"valid sizes={valid_sizes}"
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
    enable_full_determinism(seed)


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def _non_finite_tensor_report(tensor: torch.Tensor, *, chunk_size: int = 1024 * 1024) -> dict[str, Any]:
    """Record every bad flat index, compressing consecutive equal values into ranges."""
    flat = tensor.detach().reshape(-1)
    counts = {"NaN": 0, "+Inf": 0, "-Inf": 0}
    ranges_by_value = {value: [] for value in counts}
    for offset in range(0, flat.numel(), chunk_size):
        chunk = flat[offset : offset + chunk_size]
        if torch.isfinite(chunk).all().item():
            continue
        for value, predicate in (("NaN", torch.isnan), ("+Inf", torch.isposinf), ("-Inf", torch.isneginf)):
            indices = predicate(chunk).nonzero(as_tuple=True)[0]
            counts[value] += indices.numel()
            if not indices.numel():
                continue
            breaks = (indices[1:] != indices[:-1] + 1).nonzero(as_tuple=True)[0] + 1
            starts = torch.cat((indices[:1], indices[breaks])).cpu().tolist()
            ends = torch.cat((indices[breaks - 1] + 1, indices[-1:] + 1)).cpu().tolist()
            ranges = ranges_by_value[value]
            for start, end in zip(starts, ends):
                start, end = offset + start, offset + end
                if ranges and ranges[-1]["end"] == start:
                    ranges[-1]["end"] = end
                else:
                    ranges.append({"start": start, "end": end, "value": value})
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": tensor.numel(),
        "counts": counts,
        "non_finite_count": sum(counts.values()),
        "ranges": sorted(
            (item for ranges in ranges_by_value.values() for item in ranges), key=lambda item: item["start"]
        ),
    }


class _NonFiniteDiagnostics:
    """Collect local diagnostics without aborting inside an optimizer subgroup."""

    def __init__(self, state: TrainingState, metrics_out: str) -> None:
        self.state = state
        self.optimizer = state.engine.optimizer
        self.output_dir = os.path.dirname(os.path.abspath(metrics_out))
        self.names = {id(param): name for name, param in state.engine.module.named_parameters()}
        self.layouts = []
        for group in self.optimizer.fp16_groups:
            offset = 0
            layout = []
            for param in group:
                entry = self._param_location(param)
                entry["subgroup_offset"] = offset
                layout.append(entry)
                offset += entry["partition_numel"]
            self.layouts.append(layout)
        self.step = -1
        self.gradient_report = self._new_report("gradient")
        self.losses = []

    def _param_location(self, param: torch.Tensor) -> dict[str, Any]:
        numel = int(getattr(param, "ds_numel", param.numel()))
        partition = getattr(param, "ds_tensor", None)
        partition_numel = int(partition.ds_numel) if partition is not None else numel
        partition_rank = self.optimizer._get_param_partition_rank(param) if partition is not None else 0
        parameter_offset = partition_rank * partition_numel
        valid_numel = max(0, min(partition_numel, numel - parameter_offset))
        return {
            "name": self.names[id(param)],
            "ds_id": getattr(param, "ds_id", None),
            "parameter_shape": list(getattr(param, "ds_shape", param.shape)),
            "parameter_numel": numel,
            "partition_rank": partition_rank,
            "partition_numel": partition_numel,
            "parameter_offset": parameter_offset,
            "valid_numel": valid_numel,
            "padding_numel": partition_numel - valid_numel,
        }

    def _new_report(self, kind: str) -> dict[str, Any]:
        return {
            "kind": kind,
            "rank": self.state.rank,
            "step": self.step,
            "optimizer_step": self.state.engine.global_steps,
            "counts": {"NaN": 0, "+Inf": 0, "-Inf": 0},
            "non_finite_count": 0,
            "checked_numel": 0,
            "tensors": [],
            "skipped": [],
            "index_convention": "Zero-based flat indices; ranges are [start, end), without truncation. Counts include padding.",
        }

    def _add_tensor(self, report: dict[str, Any], tensor: torch.Tensor | None, **location: Any) -> None:
        if tensor is None or tensor.numel() == 0:
            report["skipped"].append({**location, "reason": "Tensor is absent or not materialized in CPU/GPU memory"})
            return
        entry = {**location, **_non_finite_tensor_report(tensor)}
        report["tensors"].append(entry)
        report["checked_numel"] += entry["numel"]
        report["non_finite_count"] += entry["non_finite_count"]
        for value, count in entry["counts"].items():
            report["counts"][value] += count
        if entry["non_finite_count"]:
            logger.error(
                "Non-finite %s: rank=%s step=%s source=%s subgroup=%s name=%s count=%s first=%s",
                report["kind"],
                self.state.rank,
                self.step,
                location.get("source"),
                location.get("subgroup_id"),
                location.get("name"),
                entry["non_finite_count"],
                entry["ranges"][0],
            )

    def _master_report(self, kind: str) -> dict[str, Any]:
        report = self._new_report(kind)
        for subgroup_id, master in enumerate(self.optimizer.fp32_partitioned_groups_flat):
            self._add_tensor(
                report,
                master,
                source="fp32_partitioned_groups_flat",
                subgroup_id=subgroup_id,
                parameter_layout=self.layouts[subgroup_id],
            )
        return report

    def _param_report(self, kind: str) -> dict[str, Any]:
        report = self._new_report(kind)
        for param in self.state.engine.module.parameters():
            location = self._param_location(param)
            # An NVMe partition can retain a placeholder/stale data buffer. Its
            # status, not just numel/device, determines whether it is readable.
            partition = getattr(param, "ds_tensor", None)
            if partition is not None:
                status = getattr(getattr(partition, "status", None), "name", "AVAILABLE")
                if status == "AVAILABLE":
                    self._add_tensor(report, partition, source="param.ds_tensor", **location)
                else:
                    report["skipped"].append({**location, "source": "param.ds_tensor", "reason": status})
            status = getattr(getattr(param, "ds_status", None), "name", "AVAILABLE")
            if status == "AVAILABLE":
                # A gathered full tensor may coexist with its local partition;
                # record both copies separately, with their own index origin.
                self._add_tensor(
                    report,
                    param.data,
                    **{
                        **location,
                        "source": "param.data",
                        "parameter_offset": 0,
                        "valid_numel": location["parameter_numel"],
                        "padding_numel": 0,
                    },
                )
            else:
                report["skipped"].append({**location, "source": "param.data", "reason": status})
        return report

    def _save_reports(self, reports: list[dict[str, Any]]) -> bool:
        os.makedirs(self.output_dir, exist_ok=True)
        local = {}
        for report in reports:
            kind = report["kind"]
            report["optimizer_step"] = self.state.engine.global_steps
            filename = f"non_finite_{kind}.rank{self.state.rank}.json"
            with open(os.path.join(self.output_dir, filename), "w", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2, allow_nan=False)
                stream.write("\n")
            local[kind] = {
                "rank": self.state.rank,
                "counts": report["counts"],
                "non_finite_count": report["non_finite_count"],
                "checked_numel": report["checked_numel"],
                "skipped_tensors": len(report["skipped"]),
                "details_file": filename,
            }
        # Gather only small summaries: a corrupt multi-billion-parameter tensor
        # can have far too many isolated bad positions to gather its full JSON.
        distributed = torch.distributed.is_initialized()
        ranks = [None] * torch.distributed.get_world_size() if distributed else [local]
        if distributed:
            torch.distributed.all_gather_object(ranks, local)
        failed = False
        for report in reports:
            kind = report["kind"]
            summaries = [rank[kind] for rank in ranks]
            counts = {value: sum(rank["counts"][value] for rank in summaries) for value in report["counts"]}
            total = sum(counts.values())
            failed |= total > 0
            if self.state.rank == 0:
                filename = os.path.join(self.output_dir, f"non_finite_{kind}.json")
                with open(filename, "w", encoding="utf-8") as stream:
                    json.dump(
                        {
                            "kind": kind,
                            "step": self.step,
                            "optimizer_step": self.state.engine.global_steps,
                            "counts": counts,
                            "non_finite_count": total,
                            "ranks": summaries,
                            "count_scope": "Scanned buffers across ranks, including padding and any resident full/partition copies",
                        },
                        stream,
                        indent=2,
                        allow_nan=False,
                    )
                    stream.write("\n")
                logger.info(
                    "Non-finite diagnostics: kind=%s step=%s count=%s checked_numel=%s skipped_tensors=%s report=%s",
                    kind,
                    self.step,
                    total,
                    sum(rank["checked_numel"] for rank in summaries),
                    sum(rank["skipped_tensors"] for rank in summaries),
                    filename,
                )
        if distributed:
            torch.distributed.barrier()  # Finish all four reports before any rank exits.
        return failed

    def check_initial(self) -> None:
        """Write initial master/model parameter reports and always continue training."""
        sync_cuda()
        if self._save_reports([self._master_report("init_weight"), self._param_report("init_param")]):
            logger.warning(
                "Initial non-finite values recorded; continuing to the first optimizer update as requested."
            )

    def begin_step(self, step: int) -> None:
        """Discard the previous iteration's diagnostics, retaining initialization files."""
        self.step = step
        self.gradient_report = self._new_report("gradient")
        self.losses = []

    @contextlib.contextmanager
    def capture_optimizer_gradients(self) -> Iterator[None]:
        """Inspect master.grad after each subgroup update, before DeepSpeed releases it."""
        original = self.optimizer._release_sub_group
        had_override = "_release_sub_group" in vars(self.optimizer)

        def release_with_check(sub_group_id: int, timer_names: Any) -> Any:
            # ZeroOffload's CPU Adam has returned here. SuperOffload has already
            # joined its worker results before releasing a subgroup. CUDA sync
            # only waits for the remaining GPU kernels/copies, not CPU workers.
            sync_cuda()
            self._add_tensor(
                self.gradient_report,
                self.optimizer.fp32_partitioned_groups_flat[sub_group_id].grad,
                source="fp32_partitioned_groups_flat.grad",
                subgroup_id=sub_group_id,
                parameter_layout=self.layouts[sub_group_id],
                observed_at="after_subgroup_update_before_gradient_release",
            )
            return original(sub_group_id, timer_names)

        self.optimizer._release_sub_group = release_with_check
        try:
            yield
        finally:
            if had_override:
                self.optimizer._release_sub_group = original
            else:
                del self.optimizer._release_sub_group

    def finish_step(self, reduced_loss: float) -> None:
        """Complete and write all four diagnostics, then exit collectively on any failure."""
        sync_cuda()
        loss_report = self._new_report("loss")
        for accum_idx, loss in enumerate(self.losses):
            self._add_tensor(loss_report, loss, source="local_loss_before_backward", micro_step=accum_idx)
        self._add_tensor(
            loss_report, torch.tensor(reduced_loss, dtype=torch.float64), source="reduced_last_micro_step_loss"
        )
        captured = {
            entry["subgroup_id"] for entry in self.gradient_report["tensors"] + self.gradient_report["skipped"]
        }
        for subgroup_id, master in enumerate(self.optimizer.fp32_partitioned_groups_flat):
            if subgroup_id not in captured:
                # Also diagnose skipped updates (e.g. overflow). A missing buffer
                # is explicitly reported instead of claiming its gradients passed.
                self._add_tensor(
                    self.gradient_report,
                    master.grad,
                    source="fp32_partitioned_groups_flat.grad",
                    subgroup_id=subgroup_id,
                    parameter_layout=self.layouts[subgroup_id],
                    observed_at="after_engine_step_without_subgroup_update",
                )
        reports = [loss_report, self.gradient_report, self._master_report("main_weight"), self._param_report("param")]
        if self._save_reports(reports):
            logger.error(
                "All step=%s diagnostics written to %s; non-finite values found, stopping.", self.step, self.output_dir
            )
            sys.exit(3)


def load_initial_weights(
    path: str | None,
    model: torch.nn.Module,
    args: argparse.Namespace,
    model_config: Any,
    rank: int,
) -> None:
    if path is None:
        return
    assert False, "init weights loading is disabled for now"
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

    # ProfilingConfig parses the matching CLI arguments from the namespace
    # generically (see ProfilingConfig.from_args); unknown args are ignored.
    prof_config = ProfilingConfig.from_args(args)
    # Must start before model construction so weight/optimizer allocations
    # are captured in the memory history trace.
    start_memory_history_recording(prof_config)

    if args.tokenizer_name is None:
        args.tokenizer_name = args.model

    model_config = build_model_config(args.model, args.override)
    expert_model = is_expert_config(model_config)
    num_experts = int(model_config.num_experts) if expert_model else None
    autoep_size = args.autoep_size if args.mode == "autoep" else None

    autoep_preset = None
    if args.mode == "autoep":
        if not expert_model:
            logger.error(
                "AutoEP requires an MoE model (no int num_experts in the config); "
                "use --mode dense or zero3_leaf for %s.",
                args.model,
            )
            sys.exit(2)
        try:
            assert autoep_size is not None and num_experts is not None
            autoep_preset = resolve_autoep_preset_name(model_config.model_type)
            validate_autoep_args(autoep_preset, autoep_size, num_experts, world_size)
        except ValueError as exc:
            logger.error("AutoEP preflight failed: %s", exc)
            sys.exit(2)

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
        logger.info("DeepSpeed config: %s", args.deepspeed_config)
        logger.info("Model Config: \n%s", model_config)
        logger.info(
            "Model: %s (model_type=%s, expert_model=%s), layers=%s, hidden=%s, experts=%s, overrides=%s",
            args.model,
            model_config.model_type,
            expert_model,
            model_config.num_hidden_layers,
            model_config.hidden_size,
            num_experts,
            args.override,
        )
        logger.info(
            "Tokenizer %s: len=%s, vocab_size=%s, model_vocab_size=%s",
            args.tokenizer_name,
            tokenizer_info["tokenizer_len"],
            tokenizer_info["tokenizer_vocab_size"],
            tokenizer_info["model_vocab_size"],
        )
        logger.info(
            "Seq len=%s, micro batch=%s, grad_accum=%s, steps=%s",
            args.seq_len,
            args.micro_batch_size,
            args.grad_accum,
            args.steps,
        )

    model = build_model(model_config)
    leaf_module_class = find_moe_leaf_module_class(model) if expert_model else None
    load_initial_weights(args.load_init_weights, model, args, model_config, rank)

    # ds_config is built after the model: zero3_leaf injection needs the MoE
    # block class detected from the instantiated module graph.
    ds_config = build_deepspeed_config(
        args.deepspeed_config,
        args.mode,
        autoep_size=autoep_size,
        autoep_preset=autoep_preset,
        leaf_module_class=leaf_module_class,
    )

    # act: HF layerwise gradient checkpointing; must be enabled before
    # deepspeed.initialize so ZeRO-3 partitions the checkpointed module graph.
    offload_ctx = contextlib.nullcontext()
    if args.activation_checkpointing:
        setup_activation_checkpointing(model)
        # act+cpu: DeepSpeed CheckpointHiddenStatesOffload ctx, created once and
        # reused by every training step (forward + backward inside the same ctx).
        if args.cpu_checkpointing:
            offload_ctx = create_offload_ctx_manager()
        if rank == 0:
            logger.info(
                "Activation checkpointing enabled (use_reentrant=False); cpu_checkpointing=%s",
                args.cpu_checkpointing,
            )

    # Offload-aware optimizer setup — no client-side optimizer is created:
    # - zero-offload / super-offload enabled: do NOT create a DeepSpeedCPUAdam
    #   here. The engine builds its own CPU Adam from the ds_config "optimizer"
    #   section; with super_offload the real CPU Adam runs in a spawned
    #   SuperOffloadCPUOptimizer worker process, and the engine-side CPUAdam
    #   only serves as hyper-parameter donor / backup-optimizer base (stage3
    #   asserts its type when ratio < 1).
    # - no offload: the same config path builds a GPU FusedAdam.
    offload_enabled = optimizer_offload_enabled(ds_config)
    if "optimizer" not in ds_config:
        logger.error(
            "ds_config must contain an 'optimizer' section: train.py does not create a "
            "client optimizer (offload_enabled=%s); DeepSpeed builds DeepSpeedCPUAdam "
            "(offload) or FusedAdam (no offload) from the config.",
            offload_enabled,
        )
        sys.exit(2)
    if rank == 0:
        logger.info(
            "Optimizer offload enabled: %s; the optimizer is built by DeepSpeed from "
            "the ds_config 'optimizer' section (type=%s).",
            offload_enabled,
            ds_config["optimizer"].get("type"),
        )

    try:
        engine, _, _, _ = deepspeed.initialize(
            model=model,
            config=ds_config,
            optimizer=None,
            model_parameters=model.parameters(),
        )
    except Exception as exc:
        logger.error("deepspeed.initialize() failed: %s", exc)
        sys.exit(2)

    import deepspeed.comm as dist_comm

    dp_rank = dist_comm.get_rank(engine.data_parallel_group)
    dp_world_size = engine.dp_world_size
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
        profiling=prof_config,
        offload_ctx=offload_ctx,
    )


def train(args: argparse.Namespace, state: TrainingState) -> None:
    metrics_logger = MetricsLogger(args.metrics_out, state.rank)
    profiling = state.profiling
    nsys_nvtx_context = None  # NVTX context for nsys profiling, set at profile_step_start
    if state.rank == 0:
        logger.info(
            "Starting training for %s optimizer steps (warmup=%s).",
            args.steps,
            args.warmup_steps,
        )
        if profiling.use_pytorch_profiler or profiling.use_nsys_profiler or profiling.record_memory_history:
            logger.info("Profiling config: %s", profiling)

    prof = None
    prof_config = profiling
    if prof_config and should_profile_rank(prof_config, torch.distributed.get_rank()):
        if prof_config.use_pytorch_profiler:
            prof = initialize_pytorch_profiler(prof_config, prof_config.tensorboard_dir)
            prof.start()


    diagnostics = _NonFiniteDiagnostics(state, args.metrics_out) if args.test else None
    if diagnostics is not None:
        if state.rank == 0:
            logger.warning("Non-finite diagnostics enabled; iteration timings include diagnostic overhead.")
        diagnostics.check_initial()

    for step in range(args.steps):
        sync_cuda()

        # Handle profiling for this step
        nvtx_ctx = handle_profiling_step(
            prof_config,
            step,
            torch.distributed.get_rank(),
            prof,
        )
        if nvtx_ctx is not None:
            nsys_nvtx_context = nvtx_ctx

        step_start = time.time()
        last_loss = None
        if diagnostics is not None:
            diagnostics.begin_step(step)

        for accum_idx in range(args.grad_accum):
            # act+cpu: forward and backward of a step must run inside the same
            # offload ctx so marked hidden_states can be restored on backward.
            # engine.step() stays outside the ctx.
            fwd_bwd_ctx = state.offload_ctx
            with fwd_bwd_ctx:
                msg = f"forward_step[{accum_idx}]"
                profiler_handle = torch.autograd.profiler.record_function(msg)
                profiler_handle.__enter__()
                batch = state.batch_gen.get_batch(step, accum_idx)
                outputs = state.engine(
                    input_ids=batch.input_ids.to(state.engine.device),
                    attention_mask=batch.attention_mask.to(state.engine.device),
                    labels=batch.labels.to(state.engine.device),
                )
                loss = outputs.loss
                last_loss = loss.detach().clone()
                if diagnostics is not None:
                    diagnostics.losses.append(last_loss)
                profiler_handle.__exit__(None, None, None)

                msg = f"backward_step[{accum_idx}]"
                profiler_handle = torch.autograd.profiler.record_function(msg)
                profiler_handle.__enter__()
                state.engine.backward(loss)
                profiler_handle.__exit__(None, None, None)

            gradient_check = (
                diagnostics.capture_optimizer_gradients() if diagnostics is not None else contextlib.nullcontext()
            )
            with gradient_check:
                if accum_idx == args.grad_accum - 1:
                    with torch.autograd.profiler.record_function("optimizer_step"):
                        state.engine.step()
                else:
                    state.engine.step()

        sync_cuda()
        iter_time = time.time() - step_start
        reduced_loss = reduce_loss(
            last_loss,
            state.dp_world_size,
            group=state.engine.data_parallel_group,
        )
        if diagnostics is not None:
            diagnostics.finish_step(reduced_loss)
        elif not math.isfinite(reduced_loss):
            if state.rank == 0:
                logger.error("Non-finite loss at step %s: loss=%s", step, reduced_loss)
            sys.exit(3)

        if step == args.warmup_steps - 1 and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_peak_host_memory_stats()

        if step < args.warmup_steps:
            if state.rank == 0:
                logger.info(
                    "Warming up for %s steps (step=%s), loss=%.6f",
                    args.warmup_steps,
                    step,
                    reduced_loss,
                )
        if step >= args.warmup_steps and step % args.log_interval == 0:
            max_iter_time = reduce_max(iter_time)
            mem_allocated = torch.cuda.memory_allocated()
            mem_peak_allocated = torch.cuda.max_memory_allocated()
            mem_peak_reserved = torch.cuda.max_memory_reserved()

            gpu_allocated = torch.cuda.memory_stats()
            cpu_allocated = torch.cuda.host_memory_stats()

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
                    "gpu_peak_gigabytes": gpu_allocated['allocated_bytes.all.peak'] / (1024**3),
                    "cpu_peak_gigabytes": cpu_allocated['allocated_bytes.peak'] / (1024**3),
                    "cuda_memory_allocated_bytes": mem_allocated,
                    "cuda_peak_memory_allocated_bytes": mem_peak_allocated,
                    "cuda_peak_memory_reserved_bytes": mem_peak_reserved,
                }
            )
            if state.rank == 0:
                logger.info(
                    "Step %s: loss=%.6f, time=%.3fs, global_tps=%.0f, peak_mem=%.2f GiB, gpu.peak=%.3f GiB, cpu.peak=%.3f MiB",
                    step,
                    reduced_loss,
                    max_iter_time,
                    global_tokens_per_sec,
                    mem_peak_allocated / (1024**3),
                    gpu_allocated['allocated_bytes.all.peak'] / (1024**3),
                    cpu_allocated['allocated_bytes.peak'] / (1024**3),
                )


        if profiling and profiling.record_memory_history and step == profiling.profile_step_end:
            rank = state.rank
            if rank in profiling.profile_ranks:
                snapshot = torch.cuda.memory._snapshot()
                from pickle import dump

                filename, ext = os.path.splitext(profiling.memory_snapshot_path)
                filename = f"{filename}_{rank}{ext}"
                with open(filename, "wb") as f:
                    dump(snapshot, f)

        handle_profiling_stop(
            profiling,
            step,
            state.rank,
            prof,
            nsys_nvtx_context,
        )

    metrics_logger.close()
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
