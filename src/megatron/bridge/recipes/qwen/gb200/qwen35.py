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

"""GB200 text-only pretraining recipes for Qwen3.5 dense and MoE models."""

from __future__ import annotations

import torch
from transformers import AutoConfig

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_mixed, bf16_with_mxfp8_mixed


_QWEN35_9B_BASE = "Qwen/Qwen3.5-9B-Base"
_QWEN35_35B_A3B_BASE = "Qwen/Qwen3.5-35B-A3B-Base"
_QWEN35_27B_BASE = "Qwen/Qwen3.5-27B"

def qwen35_text_9b_pretrain_8gpu_gb200_bf16_config() -> ConfigContainer:
    """Return a text-only Qwen3.5-9B pretraining config for eight GB200 GPUs."""
    cfg = _pretrain_common()

    text_config = AutoConfig.from_pretrained(_QWEN35_9B_BASE).text_config
    # The nested text config intentionally omits ``architectures``. AutoBridge
    # needs it to select the registered causal-LM bridge instead of the VLM.
    text_config.architectures = ["Qwen3_5ForCausalLM"]
    cfg.model = AutoBridge.from_hf_config(text_config).to_megatron_provider(load_weights=False)
    cfg.tokenizer.tokenizer_model = _QWEN35_9B_BASE
    cfg.dataset.seq_length = 4096
    cfg.dataset.blend = None
    cfg.dataset.num_workers = 8

    # Follow the Llama 3 8B GB200 topology: keep model parallelism at one and
    # use all eight GPUs for data parallelism.
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 1
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = 4096
    cfg.model.init_method_std = 0.02
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 2

    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.bias_activation_fusion = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.apply_rope_fusion = True

    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Capture the dense attention and MLP modules. Keep cross entropy on the
    # native fused path validated by the 64-GPU GB200 performance run.
    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = None
    cfg.model.cuda_graph_modules = ["attn", "mlp"]
    cfg.model.cuda_graph_warmup_steps = 3
    cfg.model.use_te_rng_tracker = True
    cfg.rng.te_rng_tracker = True

    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    cfg.mixed_precision = bf16_mixed()
    cfg.mixed_precision.grad_reduce_in_fp32 = False

    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.check_for_nan_in_grad = False
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.use_megatron_fsdp = False
    cfg.rerun_state_machine.check_for_nan_in_loss = False

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
    return cfg


def qwen35_text_35b_a3b_pretrain_8gpu_gb200_bf16_config() -> ConfigContainer:
    """Return a text-only Qwen3.5-35B-A3B pretraining config for eight GB200 GPUs."""
    cfg = _pretrain_common()

    text_config = AutoConfig.from_pretrained(_QWEN35_35B_A3B_BASE).text_config
    # The nested text config intentionally omits ``architectures``. AutoBridge
    # needs it to select the registered causal-LM bridge instead of the VLM.
    text_config.architectures = ["Qwen3_5MoeForCausalLM"]
    cfg.model = AutoBridge.from_hf_config(text_config).to_megatron_provider(load_weights=False)
    cfg.tokenizer.tokenizer_model = _QWEN35_35B_A3B_BASE
    cfg.dataset.seq_length = 4096
    cfg.dataset.blend = None
    cfg.dataset.num_workers = 8

    # Match the Qwen3.5-VL GB200 topology while training only the text model.
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = 4096
    cfg.model.init_method_std = 0.02
    cfg.train.global_batch_size = 512
    # MBS4 is suitable for force-balanced throughput benchmarking, but OOMs
    # with learned routing. MBS1 was validated with real RP2 data on GB200.
    cfg.train.micro_batch_size = 1

    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.bias_activation_fusion = True
    cfg.model.moe_router_fusion = True
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.cross_entropy_loss_fusion = True
    # Keep the library-safe native implementation instead of the performance
    # harness's TE cross-entropy path, which currently warns about stability.
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.apply_rope_fusion = True

    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Fixed-length text batches can use the scopes that the VLM recipe must
    # disable for variable-length multimodal inputs.
    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = None
    cfg.model.cuda_graph_modules = ["attn", "moe_router", "moe_preprocess"]
    cfg.model.cuda_graph_warmup_steps = 3
    cfg.model.use_te_rng_tracker = True
    cfg.rng.te_rng_tracker = True

    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_flex_dispatcher_num_sms = 32
    cfg.model.moe_hybridep_num_sms = None
    cfg.model.moe_router_dtype = "fp32"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.moe_router_padding_for_fp8 = False

    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32
    cfg.optimizer.overlap_param_gather_with_optimizer_step = False

    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.use_megatron_fsdp = False

    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=True,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
    )
    return cfg


def qwen35_text_35b_a3b_pretrain_4gpu_gb200_bf16_fsdp1_config() -> ConfigContainer:
    """Return a 4-GPU GB200 Qwen3.5 mock-data FSDP training config.

    The provider is initialized from the Qwen3.5 35B-A3B architecture because
    that is the model definition used by the corresponding Qwen3.5 recipe.
    The recipe intentionally does not load a checkpoint and uses mock data.
    
    fork from
    - base config: qwen35_text_35b_a3b_pretrain_8gpu_gb200_bf16_config
    - fsdp config: qwen35_vl_35b_a3b_sft_2gpu_h100_bf16_fsdp_config
    - bf16 config: deepseek_v4_flash_pretrain_64gpu_gb200_bf16_muon_config
    """
    cfg = _pretrain_common()

    text_config = AutoConfig.from_pretrained(_QWEN35_35B_A3B_BASE).text_config
    # The nested text config intentionally omits ``architectures``. AutoBridge
    # needs it to select the registered causal-LM bridge instead of the VLM.
    text_config.architectures = ["Qwen3_5MoeForCausalLM"]
    cfg.model = AutoBridge.from_hf_config(text_config).to_megatron_provider(load_weights=False)
    cfg.tokenizer.tokenizer_model = _QWEN35_35B_A3B_BASE
    cfg.dataset.seq_length = 4096
    cfg.dataset.blend = None  # Declarative mock-data mode.
    cfg.dataset.num_workers = 8

    # Four-GPU GB200 topology: one data-parallel group with four experts.
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 4
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = 4096
    cfg.model.init_method_std = 0.02
    cfg.train.global_batch_size = 512
    # MBS4 is suitable for force-balanced throughput benchmarking, but OOMs
    # with learned routing. MBS1 was validated with real RP2 data on GB200.
    cfg.train.micro_batch_size = 1

    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.bias_activation_fusion = True
    cfg.model.moe_router_fusion = True
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.cross_entropy_loss_fusion = True
    # Keep the library-safe native implementation instead of the performance
    # harness's TE cross-entropy path, which currently warns about stability.
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.apply_rope_fusion = True

    # Fine-grained activation offload is compatible with PP=1 and no recompute.
    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Fixed-length text batches can use the scopes that the VLM recipe must
    # disable for variable-length multimodal inputs.
    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = None
    cfg.model.cuda_graph_modules = ["attn", "moe_router", "moe_preprocess"]
    cfg.model.cuda_graph_warmup_steps = 3
    cfg.model.use_te_rng_tracker = True
    cfg.rng.te_rng_tracker = True

    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_flex_dispatcher_num_sms = 32
    cfg.model.moe_hybridep_num_sms = None
    cfg.model.moe_router_dtype = "fp32"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.moe_router_padding_for_fp8 = False

    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32
    cfg.optimizer.overlap_param_gather_with_optimizer_step = False

    cfg.checkpoint.ckpt_format = "fsdp_dtensor"
    cfg.checkpoint.load = None
    cfg.checkpoint.save = None
    cfg.rerun_state_machine.check_for_nan_in_loss = True

    # Megatron FSDP settings
    # for fsdp 1, disable overlap_grad_reduce and overlap_param_gather
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True

    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "optim"
    cfg.ddp.use_megatron_fsdp = True
    cfg.ddp.fsdp_double_buffer = True
    cfg.ddp.megatron_fsdp_max_pool_double_buffer = True
    cfg.ddp.nccl_ub = False
    cfg.ddp.fsdp_db_use_persist_buf_on_alloc_fail = True
    cfg.ddp.num_distributed_optimizer_instances = 1

    cfg.mixed_precision = bf16_mixed()
    cfg.mixed_precision.grad_reduce_in_fp32 = True

    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=False,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
    )
    return cfg


def qwen35_text_35b_a3b_pretrain_4gpu_gb200_fp8mx_fsdp1_config() -> ConfigContainer:
    """Qwen3.8 35B-A3B pretrain: 4× GB200, MXFP8.

    fork from
    - mxfp8 config: deepseek_v4_flash_pretrain_64gpu_gb200_fp8mx_config
    """

    cfg = qwen35_text_35b_a3b_pretrain_4gpu_gb200_bf16_fsdp1_config()
    cfg.mixed_precision = bf16_with_mxfp8_mixed()
    cfg.mixed_precision.fp8_param_gather = False
    cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False
    cfg.model.moe_router_padding_for_fp8 = True
    return cfg


def qwen35_text_27b_pretrain_4gpu_gb200_bf16_fsdp1_config() -> ConfigContainer:
    """Return a 4-GPU GB200 Qwen3.8 27B dense model FSDP training config.

    The provider is initialized from the Qwen3.8 27B architecture.
    The recipe intentionally does not load a checkpoint and uses mock data.

    fork from
    - base config: qwen38_text_35b_a3b_pretrain_4gpu_gb200_bf16_fsdp1_config
    """
    cfg = _pretrain_common()

    text_config = AutoConfig.from_pretrained(_QWEN35_27B_BASE).text_config
    # Set architecture for AutoBridge to select the correct bridge
    text_config.architectures = ["Qwen3_5ForCausalLM"]
    cfg.model = AutoBridge.from_hf_config(text_config).to_megatron_provider(load_weights=False)
    cfg.tokenizer.tokenizer_model = _QWEN35_27B_BASE
    cfg.dataset.seq_length = 4096
    cfg.dataset.blend = None  # Declarative mock-data mode.
    cfg.dataset.num_workers = 8

    # Four-GPU GB200 topology: one data-parallel group.
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 1  # Dense model, no MoE
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = 4096
    cfg.model.init_method_std = 0.02
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 1

    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.bias_activation_fusion = True
    cfg.model.apply_rope_fusion = True

    # Dense model settings - disable MoE specific settings
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = False
    cfg.model.moe_grouped_gemm = False
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Fine-grained activation offload
    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # CUDA graph settings
    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = None
    cfg.model.cuda_graph_modules = ["attn"]
    cfg.model.cuda_graph_warmup_steps = 3
    cfg.model.use_te_rng_tracker = True
    cfg.rng.te_rng_tracker = True

    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32
    cfg.optimizer.overlap_param_gather_with_optimizer_step = False

    cfg.checkpoint.ckpt_format = "fsdp_dtensor"
    cfg.checkpoint.load = None
    cfg.checkpoint.save = None
    cfg.rerun_state_machine.check_for_nan_in_loss = True

    # Megatron FSDP settings
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True

    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "optim"
    cfg.ddp.use_megatron_fsdp = True
    cfg.ddp.fsdp_double_buffer = True
    cfg.ddp.megatron_fsdp_max_pool_double_buffer = True
    cfg.ddp.nccl_ub = False
    cfg.ddp.fsdp_db_use_persist_buf_on_alloc_fail = True
    cfg.ddp.num_distributed_optimizer_instances = 1

    cfg.mixed_precision = bf16_mixed()
    cfg.mixed_precision.grad_reduce_in_fp32 = True

    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=False,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
    )
    return cfg


def qwen35_text_27b_pretrain_4gpu_gb200_fp8mx_fsdp1_config() -> ConfigContainer:
    """Qwen3.5 27B dense model pretrain: 4× GB200, MXFP8.

    fork from
    - base config: qwen35_text_27b_pretrain_4gpu_gb200_bf16_fsdp1_config
    - mxfp8 config: deepseek_v4_flash_pretrain_64gpu_gb200_fp8mx_config
    """
    cfg = qwen35_text_27b_pretrain_4gpu_gb200_bf16_fsdp1_config()
    cfg.mixed_precision = bf16_with_mxfp8_mixed()
    cfg.mixed_precision.fp8_param_gather = False
    cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False
    return cfg


def qwen35_text_9b_pretrain_4gpu_gb200_bf16_fsdp1_config() -> ConfigContainer:
    """Return a 4-GPU GB200 Qwen3.5 9B dense model FSDP1 config."""
    cfg = _pretrain_common()

    text_config = AutoConfig.from_pretrained(_QWEN35_9B_BASE).text_config
    # Set architecture for AutoBridge to select the correct bridge
    text_config.architectures = ["Qwen3_5ForCausalLM"]
    cfg.model = AutoBridge.from_hf_config(text_config).to_megatron_provider(load_weights=False)
    cfg.tokenizer.tokenizer_model = _QWEN35_9B_BASE
    cfg.dataset.seq_length = 4096
    cfg.dataset.blend = None
    cfg.dataset.num_workers = 8

    # Four-GPU GB200 topology: one data-parallel group.
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 1
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = 4096
    cfg.model.init_method_std = 0.02
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 1

    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.bias_activation_fusion = True
    cfg.model.apply_rope_fusion = True
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = False
    cfg.model.moe_grouped_gemm = False
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = None
    cfg.model.cuda_graph_modules = ["attn"]
    cfg.model.cuda_graph_warmup_steps = 3
    cfg.model.use_te_rng_tracker = True
    cfg.rng.te_rng_tracker = True

    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32
    cfg.optimizer.overlap_param_gather_with_optimizer_step = False

    cfg.checkpoint.ckpt_format = "fsdp_dtensor"
    cfg.checkpoint.load = None
    cfg.checkpoint.save = None
    cfg.rerun_state_machine.check_for_nan_in_loss = True

    # Megatron FSDP1 settings.
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True

    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "optim"
    cfg.ddp.use_megatron_fsdp = True
    cfg.ddp.fsdp_double_buffer = True
    cfg.ddp.megatron_fsdp_max_pool_double_buffer = True
    cfg.ddp.nccl_ub = False
    cfg.ddp.fsdp_db_use_persist_buf_on_alloc_fail = True
    cfg.ddp.num_distributed_optimizer_instances = 1

    cfg.mixed_precision = bf16_mixed()
    cfg.mixed_precision.grad_reduce_in_fp32 = True
    
    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=False,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
    )
    return cfg


def qwen35_text_9b_pretrain_4gpu_gb200_fp8mx_fsdp1_config() -> ConfigContainer:
    """Return a 4-GPU GB200 Qwen3.5 9B dense model MXFP8 FSDP1 config."""
    cfg = qwen35_text_9b_pretrain_4gpu_gb200_bf16_fsdp1_config()
    cfg.mixed_precision = bf16_with_mxfp8_mixed()
    cfg.mixed_precision.fp8_param_gather = False
    cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False
    return cfg
