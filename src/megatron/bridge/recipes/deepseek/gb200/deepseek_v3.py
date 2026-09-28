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

"""Four-GPU GB200 DeepSeek-V3 proxy recipes for offload analysis."""

from megatron.bridge.recipes.deepseek.h100.deepseek_v3 import deepseek_v3_pretrain_1024gpu_h100_bf16_config
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_with_mxfp8_mixed


def deepseek_v3_pretrain_4gpu_gb200_bf16_fsdp1_config() -> ConfigContainer:
    """Return a reduced DeepSeek-V3 config for four GB200 GPUs.

    This is a throughput-analysis proxy, not the full 671B training shape. It
    retains DeepSeek-V3's MLA/MoE blocks while reducing the model to eight MoE
    layers and 64 experts so baseline and activation-offload runs fit on four
    GPUs. Unlike the full model, this proxy has no dense transformer layers.
    """
    cfg = deepseek_v3_pretrain_1024gpu_h100_bf16_config()

    cfg.model.num_layers = 8
    cfg.model.moe_layer_freq = [1] * 8
    cfg.model.num_moe_experts = 64
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 4
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = 4096

    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_flex_dispatcher_num_sms = 32
    cfg.model.moe_hybridep_num_sms = None
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_router_force_load_balancing = False

    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = None
    cfg.model.cuda_graph_modules = []
    cfg.model.init_model_with_meta_device = True

    cfg.dataset.seq_length = 4096
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 1

    cfg.dist.use_megatron_fsdp = True
    cfg.dist.enable_megatron_core_experimental = True
    cfg.ddp.use_megatron_fsdp = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
    cfg.ddp.num_distributed_optimizer_instances = 1
    cfg.ddp.outer_dp_sharding_strategy = "no_shard"
    cfg.ddp.average_in_collective = False
    cfg.ddp.check_for_nan_in_grad = False
    cfg.ddp.grad_reduce_in_fp32 = False

    cfg.comm_overlap.overlap_grad_reduce = True
    cfg.comm_overlap.overlap_param_gather = True
    cfg.comm_overlap.overlap_param_gather_with_optimizer_step = False

    cfg.checkpoint.ckpt_format = "fsdp_dtensor"
    cfg.checkpoint.load = None
    cfg.checkpoint.save = None
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 4,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 0,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 0,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 0,
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
        "USE_MNNVL": 1,
    }
    return cfg


def deepseek_v3_pretrain_4gpu_gb200_fp8mx_fsdp1_config() -> ConfigContainer:
    """Return the MXFP8 variant of the four-GPU DeepSeek-V3 proxy."""
    cfg = deepseek_v3_pretrain_4gpu_gb200_bf16_fsdp1_config()
    cfg.mixed_precision = bf16_with_mxfp8_mixed()
    cfg.mixed_precision.fp8_param_gather = True
    cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False
    cfg.model.moe_router_padding_for_fp8 = True
    return cfg
