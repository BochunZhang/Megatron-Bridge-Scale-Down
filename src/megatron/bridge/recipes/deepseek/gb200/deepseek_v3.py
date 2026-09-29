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

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.flex_dispatcher_backend import apply_flex_dispatcher_backend
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig, bf16_with_mxfp8_mixed


def deepseek_v3_pretrain_4gpu_gb200_bf16_fsdp1_config() -> ConfigContainer:
    """Return a reduced DeepSeek-V3 config for four GB200 GPUs.

    This is a throughput-analysis proxy, not the full 671B training shape. It
    retains DeepSeek-V3's MLA/MoE blocks while reducing the model to four
    transformer layers and 32 experts so baseline and activation-offload runs
    fit on four GPUs. MTP settings are inherited from the DeepSeek provider so
    the runtime can reuse the final main layer's dense/expert type.
    """

    cfg = _pretrain_common()

    # Model config
    cfg.model = AutoBridge.from_hf_pretrained("deepseek-ai/DeepSeek-V3").to_megatron_provider(load_weights=False)

    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

    # Dataset config - mock data by default
    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.num_workers = 8

    # Model config, moe_layer_freq, 0: disable moe, 1: enable moe.
    # cfg.model.num_layers = 4
    # cfg.model.moe_layer_freq = [1] * 4
    # cfg.model.num_moe_experts = 32

    # Parallelism settings (32 nodes configuration)
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

    # MTP (Multi-Token Prediction) configuration
    cfg.model.mtp_num_layers = 1
    cfg.model.mtp_loss_scaling_factor = 0.1

    # Model-specific settings
    cfg.model.init_method_std = 0.006
    cfg.model.rotary_base = 10000.0
    cfg.model.rotary_scaling_factor = 40
    cfg.model.rotary_base = float(cfg.model.rotary_base)
    cfg.model.rotary_scaling_factor = int(cfg.model.rotary_scaling_factor)

    # Pipeline split settings
    cfg.model.account_for_embedding_in_pipeline_split = False
    cfg.model.account_for_loss_in_pipeline_split = False
    cfg.model.num_layers_in_first_pipeline_stage = None
    cfg.model.num_layers_in_last_pipeline_stage = None

    # MoE Token Dispatcher settings
    # Note: moe_token_dispatcher_type may be overridden by apply_flex_dispatcher_backend at the end
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_flex_dispatcher_num_sms = 16
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

    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.attention_backend = None
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"
    cfg.model.moe_router_padding_for_fp8 = False

    cfg.dataset.seq_length = 4096
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 5
    cfg.train.manual_gc_eval = 5
    cfg.scheduler.lr_warmup_iters = 2000

    cfg.mixed_precision = MixedPrecisionConfig(
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_enabled=False,
        grad_reduce_in_fp32=False,
    )

    cfg.optimizer.use_precision_aware_optimizer = True
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.main_grads_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16

    cfg.dist.use_megatron_fsdp = True
    cfg.dist.enable_megatron_core_experimental = True
    cfg.ddp.use_megatron_fsdp = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.data_parallel_sharding_strategy = "optim"
    cfg.ddp.num_distributed_optimizer_instances = 1
    cfg.ddp.outer_dp_sharding_strategy = "no_shard"
    cfg.ddp.average_in_collective = False
    cfg.ddp.check_for_nan_in_grad = False
    cfg.ddp.grad_reduce_in_fp32 = False

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
    cfg.comm_overlap.overlap_grad_reduce = True
    cfg.comm_overlap.overlap_param_gather = True
    cfg.comm_overlap.delay_wgrad_compute = False
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = False

    cfg.checkpoint.ckpt_format = "fsdp_dtensor"
    cfg.checkpoint.load = None
    cfg.checkpoint.save = None
    cfg.checkpoint.save_interval = 2000
    cfg.checkpoint.async_save = False

    cfg.validation.eval_interval = 0
    cfg.validation.eval_iters = 0

    cfg.model.moe_router_force_load_balancing = False
    if cfg.model.apply_rope_fusion:
        cfg.dist.enable_megatron_core_experimental = True

    apply_flex_dispatcher_backend(cfg.model, cfg.model.moe_flex_dispatcher_backend)

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
