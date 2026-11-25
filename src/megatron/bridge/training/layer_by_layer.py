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

"""Run Megatron-LM's layer-by-layer backend from Bridge pretraining."""

import json
import logging
import math
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import torch
from megatron.core.process_groups_config import ProcessGroupCollection
from torch import distributed as dist

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.training.config import ConfigContainer, MockGPTDatasetConfig
from megatron.bridge.training.state import GlobalState
from megatron.bridge.utils.common_utils import get_world_size_safe


logger = logging.getLogger(__name__)


def _normalize_tier(value: object, field_name: str, *, legacy_suffix: str = "") -> Literal["gpu", "cpu", "nvme"]:
    tier = value.removesuffix(legacy_suffix) if isinstance(value, str) else None
    if tier == "gpu" or tier == "cpu" or tier == "nvme":
        return tier
    raise ValueError(f"Layer-by-layer {field_name} must be gpu, cpu, or nvme, got {value!r}.")


def configure_layer_by_layer(config: ConfigContainer) -> None:
    """Validate backend capabilities and set the geometry used by Bridge metrics."""
    offload = config.layer_by_layer
    assert offload is not None
    offload.parameter_tier = _normalize_tier(offload.parameter_tier, "parameter_tier", legacy_suffix="_param")
    offload.activation_tier = _normalize_tier(offload.activation_tier, "activation_tier", legacy_suffix="_act")
    offload.optimizer_tier = _normalize_tier(offload.optimizer_tier, "optimizer_tier")
    if not isinstance(config.dataset, MockGPTDatasetConfig) or config.dataset.blend or config.dataset.blend_per_split:
        raise ValueError("Layer-by-layer training requires a mock dataset without data blends.")
    if config.peft is not None or any(
        (config.checkpoint.load, config.checkpoint.save, config.checkpoint.pretrained_checkpoint)
    ):
        raise ValueError("Layer-by-layer training does not support PEFT or checkpoint loading and saving.")
    if config.validation.eval_iters or config.validation.skip_train:
        raise ValueError("Layer-by-layer training requires training without evaluation.")

    model = config.model
    if not isinstance(model, GPTModelProvider) or model.num_moe_experts:
        raise ValueError("Layer-by-layer training requires a dense GPT model provider.")
    if (
        (
            model.tensor_model_parallel_size,
            model.pipeline_model_parallel_size,
            model.context_parallel_size,
            model.expert_model_parallel_size,
            model.expert_tensor_parallel_size,
        )
        != (1, 1, 1, 1, 1)
        or model.virtual_pipeline_model_parallel_size is not None
        or model.pipeline_model_parallel_layout is not None
        or model.sequence_parallel
    ):
        raise ValueError("Layer-by-layer training requires TP=PP=CP=EP=ETP=1 without sequence or virtual parallelism.")
    if model.position_embedding_type != "rope" or model.share_embeddings_and_output_weights:
        raise ValueError("Layer-by-layer training requires rotary embeddings and untied output weights.")
    if (
        model.cuda_graph_impl != "none"
        or model.cuda_graph_scope
        or config.optimizer.optimizer_cuda_graph
        or config.ddp.use_megatron_fsdp
        or config.dist.use_megatron_fsdp
        or config.dist.use_torch_fsdp2
        or config.dist.use_decentralized_pg
        or config.dist.lazy_mpu_init
    ):
        raise ValueError("Layer-by-layer training owns sharding and requires MPU groups without CUDA graphs.")
    if config.rng.data_parallel_random_init:
        raise ValueError("Layer-by-layer training requires the same parameter initialization on every data rank.")
    if not model.bf16 or model.params_dtype != torch.bfloat16 or model.fp16 or model.fp8 or model.fp4:
        raise ValueError("Layer-by-layer training requires BF16 model parameters and compute.")

    training = config.train
    world_size = get_world_size_safe()
    if world_size != 4 or int(os.environ.get("LOCAL_WORLD_SIZE", world_size)) != world_size:
        raise ValueError("Layer-by-layer training requires one torchrun node with four GPU ranks.")
    if training.train_iters is None or min(training.train_iters, config.dataset.seq_length, model.num_layers) < 1:
        raise ValueError("Layer-by-layer training requires positive iterations, sequence length, and layer count.")
    samples_per_microbatch = training.micro_batch_size * world_size
    if (
        samples_per_microbatch < 1
        or training.global_batch_size < samples_per_microbatch
        or training.global_batch_size % samples_per_microbatch
    ):
        raise ValueError(
            "Layer-by-layer global batch size must be a positive multiple of four times micro batch size."
        )
    if (
        training.rampup_batch_size
        or training.iterations_to_skip
        or training.check_weight_hash_across_dp_replicas_interval
    ):
        raise ValueError(
            "Layer-by-layer training requires fixed batches without skipped steps or parameter hash checks."
        )
    if config.logger.log_params_norm or config.logger.log_l2_norm_grad_to_tensorboard:
        raise ValueError("Layer-by-layer training does not expose resident model or gradient norms.")
    if config.logger.log_progress:
        raise ValueError("Layer-by-layer training does not support checkpoint progress logs.")
    if config.inprocess_restart and config.inprocess_restart.enabled:
        raise ValueError("Layer-by-layer training does not support in-process restart.")

    optimizer = config.optimizer
    if (
        optimizer.optimizer != "adam"
        or not optimizer.decoupled_weight_decay
        or optimizer.clip_grad
        or optimizer.use_distributed_optimizer
        or config.ddp.use_distributed_optimizer
        or optimizer.lr is None
        or not math.isfinite(optimizer.lr)
        or optimizer.lr <= 0
    ):
        raise ValueError("Layer-by-layer training requires its own AdamW with positive learning rate and no clipping.")
    if (
        not all(0 <= beta < 1 for beta in (optimizer.adam_beta1, optimizer.adam_beta2))
        or not math.isfinite(optimizer.adam_eps)
        or optimizer.adam_eps <= 0
        or not math.isfinite(optimizer.weight_decay)
        or optimizer.weight_decay < 0
    ):
        raise ValueError("Layer-by-layer AdamW requires valid betas, positive epsilon, and nonnegative weight decay.")
    scheduler = config.scheduler
    if (
        scheduler.lr_decay_style != "constant"
        or scheduler.lr_warmup_iters
        or scheduler.lr_warmup_samples
        or scheduler.lr_warmup_fraction
        or scheduler.weight_decay_incr_style != "constant"
    ):
        raise ValueError("Layer-by-layer training requires constant learning rate and weight decay without warmup.")
    if offload.cpu_threads < 1:
        raise ValueError("Layer-by-layer training requires a positive CPU thread count.")

    profiling = config.profiling
    if profiling.use_pytorch_profiler or profiling.record_memory_history:
        raise ValueError("Layer-by-layer training supports only Nsight Systems profiling.")
    if profiling.use_nsys_profiler and (
        (profiling.profile_ranks and sorted(profiling.profile_ranks) != list(range(world_size)))
        or not 0 <= profiling.profile_step_start < profiling.profile_step_end <= training.train_iters
    ):
        raise ValueError("Layer-by-layer profiling requires all ranks and 0 <= start < end <= train_iters.")

    from megatron.training.layer_by_layer_workload import (
        QWEN35_LAYER_BY_LAYER_TINY_VOCAB_SIZE,
        QWEN35_LAYER_BY_LAYER_VOCAB_SIZE,
        build_qwen35_layer_by_layer_config,
    )

    # Standard FLOP accounting and saved config must reflect the actual cropped workload.
    if offload.tiny:
        tiny_model = build_qwen35_layer_by_layer_config(model.num_layers, tiny=True)
        for field in fields(tiny_model):
            if field.init:
                setattr(model, field.name, getattr(tiny_model, field.name))
        model.vocab_size = QWEN35_LAYER_BY_LAYER_TINY_VOCAB_SIZE
    elif model.vocab_size is None:
        model.vocab_size = QWEN35_LAYER_BY_LAYER_VOCAB_SIZE
    model.mtp_num_layers = None
    model.make_vocab_size_divisible_by = 1
    model.should_pad_vocab = False
    if isinstance(model.linear_attention_freq, list):
        model.linear_attention_freq = model.linear_attention_freq[: model.num_layers]


@contextmanager
def layer_by_layer_step(
    state: GlobalState, pg_collection: ProcessGroupCollection
) -> Iterator[Callable[[int, int], dict[str, torch.Tensor]]]:
    """Open the offload backend and provide whole optimizer steps to Bridge."""
    from megatron.core.distributed.layer_by_layer import LayerByLayerTrainer
    from megatron.training.layer_by_layer_workload import Qwen35DenseWorkload

    config = state.cfg
    offload, model, optimizer = config.layer_by_layer, config.model, config.optimizer
    assert offload is not None and isinstance(model, GPTModelProvider) and optimizer.lr is not None
    torch.set_num_threads(offload.cpu_threads)
    device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.reset_peak_memory_stats(device)
    group = pg_collection.dp
    rank = group.rank()
    workload = Qwen35DenseWorkload(
        num_layers=model.num_layers,
        seq_length=config.dataset.seq_length,
        micro_batch_size=config.train.micro_batch_size,
        seed=config.rng.seed,
        device=device,
        pg_collection=pg_collection,
        model_config=model,
        vocab_size=model.vocab_size,
        rotary_base=model.rotary_base,
        rotary_percent=model.rotary_percent,
    )
    loss = None
    with TemporaryDirectory(prefix=f"rank-{rank}-", dir=offload.nvme_dir) as scratch:
        with LayerByLayerTrainer(
            workload.make_units(),
            group=group,
            device=device,
            directory=Path(scratch),
            tier=offload.parameter_tier,
            activation_tier=offload.activation_tier,
            optimizer_tier=offload.optimizer_tier,
            betas=(optimizer.adam_beta1, optimizer.adam_beta2),
            eps=optimizer.adam_eps,
            weight_decay=optimizer.weight_decay,
        ) as trainer:

            def step(iteration: int, microbatches: int) -> dict[str, torch.Tensor]:
                nonlocal loss
                batches = [workload.make_batch(iteration, mb, rank) for mb in range(microbatches)]
                loss = trainer.step(batches, lr=optimizer.lr, iteration=iteration)
                if config.profiling.use_nsys_profiler and iteration + 1 == config.profiling.profile_step_end:
                    # Keep the first rank to stop capture from cutting off another rank's final step.
                    dist.barrier(group=group, device_ids=[device.index])
                return {"lm loss": torch.tensor(loss, dtype=torch.float32, device=device)}

            yield step
            torch.cuda.synchronize(device)
            parameter_count = trainer.param_count
            transfers = trainer.activation_transfer_counts

    steps = state.train_state.step
    if steps != config.train.train_iters or loss is None:
        raise RuntimeError("Layer-by-layer training did not complete all requested steps.")
    peaks = torch.tensor(
        [torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)],
        dtype=torch.int64,
        device=device,
    )
    dist.all_reduce(peaks, op=dist.ReduceOp.MAX, group=group)
    local_transfers = torch.tensor(transfers, dtype=torch.int64, device=device)
    all_transfers = torch.empty(group.size() * local_transfers.numel(), dtype=torch.int64, device=device)
    dist.all_gather_into_tensor(all_transfers, local_transfers, group=group)
    transfers_by_rank = all_transfers.view(group.size(), 2).tolist()
    microbatches = config.train.global_batch_size // (config.train.micro_batch_size * group.size())
    expected_d2h = 0 if offload.activation_tier == "gpu" else (model.num_layers + 1) * microbatches * steps
    expected = [expected_d2h, 2 * expected_d2h]
    if any(counts != expected for counts in transfers_by_rank):
        raise RuntimeError(f"Expected {expected} D2H/H2D transfers per rank, got {transfers_by_rank}.")
    if dist.get_rank() == 0:
        allocated, reserved = peaks.tolist()
        result = {
            "layers": model.num_layers,
            "steps": steps,
            "parameter_count": parameter_count,
            "parameter_tier": offload.parameter_tier,
            "activation_tier": offload.activation_tier,
            "optimizer_tier": offload.optimizer_tier,
            "activation_d2h_completed_by_rank": [counts[0] for counts in transfers_by_rank],
            "activation_h2d_completed_by_rank": [counts[1] for counts in transfers_by_rank],
            "loss": loss,
            "peak_allocated_bytes": allocated,
            "peak_reserved_bytes": reserved,
        }
        logger.info("LAYER_BY_LAYER_RESULT %s", json.dumps(result, allow_nan=False))
