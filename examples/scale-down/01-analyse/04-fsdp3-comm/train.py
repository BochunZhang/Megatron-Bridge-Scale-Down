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

"""Text-only, randomly initialized Qwen3.5 FSDP3 training on mock data."""

import argparse
import os
from pathlib import Path

from benchmark_fsdp3 import MODELS, SEQUENCES


def main() -> None:
    """Build a scaled HF text configuration and run the standard Bridge step."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--cp", type=int, choices=SEQUENCES, required=True)
    parser.add_argument("--seq-length", type=int, required=True)
    parser.add_argument("--micro-batch-size", type=int, required=True)
    parser.add_argument("--global-batch-size", type=int, required=True)
    parser.add_argument("--train-iters", type=int, required=True)
    parser.add_argument("--profile", choices=("nsys", "none"), required=True)
    parser.add_argument("--profile-start", type=int, required=True)
    parser.add_argument("--profile-end", type=int, required=True)
    parser.add_argument("--communication-unit-size", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.seq_length not in SEQUENCES[args.cp]:
        parser.error("sequence length is outside the requested CP matrix")

    import torch
    from transformers import AutoConfig

    from megatron.bridge import AutoBridge
    from megatron.bridge.recipes.common import _pretrain_common
    from megatron.bridge.training.config import ProfilingConfig, TokenizerConfig
    from megatron.bridge.training.gpt_step import forward_step
    from megatron.bridge.training.mixed_precision import bf16_mixed
    from megatron.bridge.training.pretrain import pretrain

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world < 2 or world % args.cp or args.global_batch_size % (args.micro_batch_size * (world // args.cp)):
        raise ValueError("Invalid world size / CP / batch-size combination")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = _pretrain_common()
    text = AutoConfig.from_pretrained(MODELS[args.model]).text_config
    expected = ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 2
    if list(text.layer_types[:8]) != expected:
        raise ValueError("HF config no longer has the expected four-layer attention cycle")
    text.num_hidden_layers = 8
    text.layer_types = expected
    text.mtp_num_hidden_layers = 0
    is_moe = args.model != "27b"
    text.architectures = ["Qwen3_5MoeForCausalLM" if is_moe else "Qwen3_5ForCausalLM"]
    if is_moe:
        text.num_experts = 64
    text.to_json_file(str(args.output_dir / f"hf-text-config-rank{rank}.json"))
    cfg.model = AutoBridge.from_hf_config(text).to_megatron_provider(load_weights=False)
    model = cfg.model
    model.num_layers = 8
    model.linear_attention_freq = 4
    model.mtp_num_layers = 0
    model.tensor_model_parallel_size = 1
    model.pipeline_model_parallel_size = 1
    model.virtual_pipeline_model_parallel_size = None
    model.pipeline_model_parallel_layout = None
    model.context_parallel_size = args.cp
    model.cp_comm_type = "p2p"
    model.expert_model_parallel_size = 1
    model.expert_tensor_parallel_size = 1
    model.sequence_parallel = False
    model.pipeline_dtype = torch.bfloat16
    model.seq_length = args.seq_length
    model.transformer_impl = "transformer_engine"
    model.bias_activation_fusion = True
    model.apply_rope_fusion = True
    model.cross_entropy_loss_fusion = True
    model.cross_entropy_fusion_impl = "native"
    model.recompute_granularity = None
    model.recompute_method = None
    model.recompute_num_layers = None
    model.recompute_modules = []
    model.fine_grained_activation_offloading = False
    model.offload_modules = []
    model.cuda_graph_impl = "none"
    model.cuda_graph_scope = None
    model.cuda_graph_modules = []
    if is_moe:
        model.num_moe_experts = 64
        model.moe_grouped_gemm = True
        model.moe_router_fusion = True
        model.moe_permute_fusion = True
        model.moe_token_dispatcher_type = "alltoall"
        model.moe_flex_dispatcher_backend = None
        model.moe_shared_expert_overlap = False
        model.moe_router_force_load_balancing = False
        model.moe_router_dtype = "fp32"
    # EP=1 keeps expert parameters sharded across the same world-size group.
    # No automatic CommOverlapConfig rewrite: CP=4 on four GPUs has training DP=1.
    cfg.comm_overlap = None
    cfg.dist.use_megatron_fsdp = True
    cfg.ddp.use_megatron_fsdp = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.average_in_collective = False
    cfg.ddp.check_for_nan_in_grad = False
    cfg.ddp.fsdp_double_buffer = True
    cfg.ddp.megatron_fsdp_max_pool_double_buffer = True
    cfg.ddp.nccl_ub = False
    cfg.ddp.suggested_communication_unit_size = args.communication_unit_size
    cfg.ddp.megatron_fsdp_main_params_dtype = torch.float32
    cfg.ddp.megatron_fsdp_main_grads_dtype = torch.float32
    cfg.ddp.megatron_fsdp_grad_comm_dtype = torch.float32
    cfg.mixed_precision = bf16_mixed()
    cfg.mixed_precision.grad_reduce_in_fp32 = True
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32
    cfg.optimizer.optimizer_cpu_offload = False
    cfg.optimizer.overlap_param_gather_with_optimizer_step = False
    cfg.tokenizer = TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=text.vocab_size)
    cfg.dataset.seq_length = args.seq_length
    cfg.dataset.blend = None
    cfg.dataset.num_workers = 1
    cfg.train.train_iters = args.train_iters
    cfg.train.micro_batch_size = args.micro_batch_size
    cfg.train.global_batch_size = args.global_batch_size
    cfg.scheduler.lr_warmup_iters = 1
    cfg.scheduler.lr_decay_iters = args.train_iters
    cfg.validation.eval_iters = 0
    cfg.validation.eval_interval = 0
    cfg.checkpoint.ckpt_format = "fsdp_dtensor"
    cfg.checkpoint.save = None
    cfg.checkpoint.load = None
    cfg.logger.log_interval = 1
    cfg.logger.tensorboard_dir = None
    cfg.logger.save_config_filepath = str(args.output_dir / "config.yaml")
    cfg.profiling = ProfilingConfig(
        use_nsys_profiler=args.profile == "nsys",
        use_pytorch_profiler=False,
        profile_step_start=args.profile_start,
        profile_step_end=args.profile_end,
        profile_ranks=list(range(world)),
        nvtx_ranges=args.profile == "nsys",
        record_shapes=False,
        record_memory_history=False,
    )
    pretrain(cfg, forward_step)


if __name__ == "__main__":
    main()
