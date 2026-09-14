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
# ruff: noqa: F401
"""Compatibility alias for the canonical Qwen3.5 text recipe."""

from __future__ import annotations

from megatron.bridge.recipes.qwen.gb200.qwen38 import (
    qwen38_text_27b_pretrain_4gpu_gb200_bf16_fsdp1_config,
    qwen38_text_27b_pretrain_4gpu_gb200_fp8mx_fsdp1_config,
)
from megatron.bridge.recipes.qwen.gb200.qwen38 import (
    qwen38_text_35b_a3b_pretrain_4gpu_gb200_bf16_fsdp1_config,
    qwen38_text_35b_a3b_pretrain_4gpu_gb200_fp8mx_fsdp1_config, 
)


__all__ = [
    "qwen38_text_27b_pretrain_4gpu_gb200_bf16_fsdp1_config", 
    "qwen38_text_27b_pretrain_4gpu_gb200_fp8mx_fsdp1_config", 
    "qwen38_text_35b_a3b_pretrain_4gpu_gb200_bf16_fsdp1_config", 
    "qwen38_text_35b_a3b_pretrain_4gpu_gb200_fp8mx_fsdp1_config",
]
