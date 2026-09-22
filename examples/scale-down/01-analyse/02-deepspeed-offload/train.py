"""Compact causal LM training example for AutoEP, ZeRO-3 leaf, and Qwen3.5 text.

The DeepSpeed runtime config is loaded from a JSON file via --deepspeed_config
(same convention as finetune_zero3.py); see ds_config.json for an example.
Only model-dependent keys (AutoEP expert_parallel, ZeRO-3 leaf module classes)
and the CLI batch settings are applied on top of the loaded file.

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
        --model qwen3_5 --mode zero3_leaf

Launch with profiling / memory snapshots:

    deepspeed --num_gpus 1 train.py --deepspeed_config ds_config.json \
        --model qwen3_5 --mode zero3_leaf \
        --use_pytorch_profiler --record_memory_history \
        --profile_step_start 5 --profile_step_end 10 \
        --profile_ranks 0 --memory_snapshot_path snapshot.pickle

Launch with recompute (act) and recompute + CPU activation offload (act+cpu):

    # act: HF layerwise gradient checkpointing (use_reentrant=False)
    deepspeed --num_gpus 4 train.py --deepspeed_config ds_config.json \
        --model qwen3_5_moe --mode zero3_leaf --activation_checkpointing

    # act+cpu: additionally offload checkpointed layer inputs to CPU via
    # DeepSpeed CheckpointHiddenStatesOffload (keep the default
    # DS_PIN_MEMORY_BACKEND=torch)
    deepspeed --num_gpus 4 train.py --deepspeed_config ds_config.json \
        --model qwen3_5_moe --mode zero3_leaf \
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
from typing import Any, NamedTuple

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    Llama4ForCausalLM,
    Llama4TextConfig,
    MixtralConfig,
    Qwen3_5ForCausalLM,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeTextConfig,
    Qwen3_5TextConfig,
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
    "qwen3_5": {
        "architecture": "qwen3_5",
        "config_cls": Qwen3_5TextConfig,
        "display_name": "Qwen3.5 Text (Dense)",
        "default_tokenizer_name": "Qwen/Qwen3.5-0.8B",
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

DEFAULT_OPTIMIZER_LR = 0.001
DEFAULT_OPTIMIZER_BETAS = (0.9, 0.999)


class ModelPreset(NamedTuple):
    architecture: str
    config_cls: type[Any]
    display_name: str


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
    parser.add_argument("--model", choices=sorted(MODEL_PRESETS), default="qwen3_5_moe")
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
    parser.add_argument("--micro_batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
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


def resolve_model_preset(args: argparse.Namespace) -> ModelPreset:
    preset = MODEL_PRESETS[args.model]
    architecture = preset["architecture"]
    config_cls = preset["config_cls"]
    if args.tokenizer_name is None:
        args.tokenizer_name = preset["default_tokenizer_name"]
    return ModelPreset(
        architecture,
        config_cls,
        preset["display_name"],
    )


def build_model(architecture: str, model_config: Any) -> torch.nn.Module:
    if architecture == "mixtral":
        return AutoModelForCausalLM.from_config(model_config)
    if architecture == "qwen3_5_moe":
        return Qwen3_5MoeForCausalLM(model_config)
    if architecture == "qwen3_5":
        return Qwen3_5ForCausalLM(model_config)
    if architecture == "llama4":
        return Llama4ForCausalLM(model_config)
    raise ValueError(f"Unsupported architecture: {architecture!r}")


def num_experts_for_config(architecture: str, model_config: Any) -> int | None:
    if architecture in {"mixtral", "llama4"}:
        return int(model_config.num_local_experts)
    if architecture == "qwen3_5_moe":
        return int(model_config.num_experts)
    if architecture == "qwen3_5":
        return None
    raise ValueError(f"Unsupported architecture: {architecture!r}")

def create_optimizer(model: AutoModelForCausalLM) -> Any:
    from deepspeed.ops.adam import DeepSpeedCPUAdam
    optimizer = DeepSpeedCPUAdam(
        model.parameters(),
        lr=DEFAULT_OPTIMIZER_LR,
        betas=DEFAULT_OPTIMIZER_BETAS
    )
    return optimizer


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
    architecture: str,
    micro_batch_size: int,
    grad_accum: int,
    autoep_size: int | None,
) -> dict[str, Any]:
    """Load the DeepSpeed JSON config from ``config_path`` and adjust it per model/mode.

    Mirrors finetune_zero3.py: the base config (bf16, optimizer, scheduler,
    zero_optimization, offload knobs) comes from the given file instead of
    being constructed here. Adjustments applied on top of the loaded file:

    - CLI batch settings win: ``train_micro_batch_size_per_gpu`` and
      ``gradient_accumulation_steps`` are set from the CLI (and any file-level
      ``train_batch_size`` is dropped), so the DeepSpeed engine, the batch
      generator, and the accumulation loop stay consistent.
    - ``autoep`` mode: forces ZeRO stage 3 — AutoEP must run on ZeRO-3, never
      downgraded to stage 1/2 — and injects the ``expert_parallel`` preset
      block for the model architecture.
    - ``zero3_leaf`` mode with a MoE architecture: injects the ZeRO-3
      ``leaf_module`` class list.
    """
    with open(config_path, encoding="utf-8") as f:
        config: dict[str, Any] = json.load(f)

    config.pop("train_batch_size", None)
    config["train_micro_batch_size_per_gpu"] = micro_batch_size
    config["gradient_accumulation_steps"] = grad_accum

    if mode == "autoep":
        if architecture not in DEEPSPEED_LEAF_MOE_BLOCK_CLASS:
            raise ValueError("AutoEP is only supported for MoE model presets.")
        config["expert_parallel"] = {
            "enabled": True,
            "autoep_size": autoep_size,
            "preset_model": architecture,
        }
    elif mode == "zero3_leaf":
        if architecture in DEEPSPEED_LEAF_MOE_BLOCK_CLASS:
            config["zero_optimization"]["leaf_module"] = {
                "classes": [DEEPSPEED_LEAF_MOE_BLOCK_CLASS[architecture]]
            }

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

    # ProfilingConfig parses the matching CLI arguments from the namespace
    # generically (see ProfilingConfig.from_args); unknown args are ignored.
    prof_config = ProfilingConfig.from_args(args)
    # Must start before model construction so weight/optimizer allocations
    # are captured in the memory history trace.
    start_memory_history_recording(prof_config)

    preset = resolve_model_preset(args)
    model_config = build_model_config(preset.config_cls, args.override)
    num_experts = num_experts_for_config(preset.architecture, model_config)
    autoep_size = args.autoep_size if args.mode == "autoep" else None

    if args.mode == "autoep":
        if num_experts is None:
            logger.error("AutoEP requires an MoE model; use --mode zero3_leaf for %s.", args.model)
            sys.exit(2)
        try:
            assert autoep_size is not None
            validate_autoep_args(preset.architecture, autoep_size, num_experts, world_size)
        except ValueError as exc:
            logger.error("AutoEP preflight failed: %s", exc)
            sys.exit(2)

    ds_config = build_deepspeed_config(
        args.deepspeed_config,
        args.mode,
        preset.architecture,
        args.micro_batch_size,
        args.grad_accum,
        autoep_size,
    )

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
        logger.info(
            "Model: %s (%s), layers=%s, hidden=%s, experts=%s, overrides=%s",
            args.model,
            preset.display_name,
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

    model = build_model(preset.architecture, model_config)
    load_initial_weights(args.load_init_weights, model, args, model_config, rank)

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

    # 支持 super-offload, 创建一个 CPU Adam
    # deepspeed 会根据 offload 比例额外创建 GPU Adam
    optimizer = create_optimizer(model)

    try:
        engine, _, _, _ = deepspeed.initialize(
            model=model,
            config=ds_config,
            optimizer=optimizer,
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
                profiler_handle.__exit__(None, None, None)

                msg = f"backward_step[{accum_idx}]"
                profiler_handle = torch.autograd.profiler.record_function(msg)
                profiler_handle.__enter__()
                state.engine.backward(loss)
                profiler_handle.__exit__(None, None, None)

            msg = f"optimizer_step"
            profiler_handle = torch.autograd.profiler.record_function(msg)

            if accum_idx == args.grad_accum - 1:
                profiler_handle.__enter__()
                state.engine.step()
                profiler_handle.__exit__(None, None, None)
            else:
                state.engine.step()

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
            torch.cuda.reset_peak_host_memory_stats()

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
