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

"""Write the canonical HF model contract shared by both experiment runners."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
from pathlib import Path
from typing import Any

from transformers import AutoConfig


LOGGER = logging.getLogger(__name__)
QWEN35_RECIPE_MODULE = "megatron.bridge.recipes.qwen.gb200.qwen35"


def _text_config(model_id: str) -> Any:
    """Load a model config and select its text backbone when present."""
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    return getattr(config, "text_config", config)


def _rope_value(config: Any, field: str) -> Any:
    """Read a RoPE value from either the legacy attribute or nested mapping."""
    value = getattr(config, field, None)
    if value is not None:
        return value
    rope_parameters = getattr(config, "rope_parameters", None)
    return rope_parameters.get(field) if isinstance(rope_parameters, dict) else None


def _layer_types(num_layers: int, linear_attention_freq: int | list[int]) -> list[str]:
    """Expand Megatron's interval or per-layer pattern into HF layer types."""
    if isinstance(linear_attention_freq, int) and not isinstance(linear_attention_freq, bool):
        if linear_attention_freq <= 0:
            raise ValueError(f"linear_attention_freq must be positive, got {linear_attention_freq}")
        pattern = [
            0 if (layer_index + 1) % linear_attention_freq == 0 else 1 for layer_index in range(num_layers)
        ]
    elif isinstance(linear_attention_freq, list):
        if len(linear_attention_freq) != num_layers:
            raise ValueError(
                "linear_attention_freq pattern length must match num_layers: "
                f"got {len(linear_attention_freq)}, expected {num_layers}"
            )
        pattern = linear_attention_freq
    else:
        raise TypeError(
            "linear_attention_freq must be an int or list[int], "
            f"got {type(linear_attention_freq).__name__}"
        )

    unsupported = sorted(
        {
            repr(value)
            for value in pattern
            if not isinstance(value, int) or isinstance(value, bool) or value not in {0, 1}
        }
    )
    if unsupported:
        raise ValueError(f"linear_attention_freq pattern must contain only 0 or 1, got {unsupported}")
    layer_type_by_pattern = {0: "full_attention", 1: "linear_attention"}
    return [layer_type_by_pattern[value] for value in pattern]


def _hf_contract(config: Any, *, expert: bool) -> dict[str, Any]:
    """Normalize a post-override DeepSpeed HF config into common model fields."""
    contract = {
        "num_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "num_attention_heads": config.num_attention_heads,
        "num_query_groups": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "vocab_size": config.vocab_size,
        "layer_types": list(config.layer_types),
        "linear_conv_kernel_dim": config.linear_conv_kernel_dim,
        "linear_key_head_dim": config.linear_key_head_dim,
        "linear_value_head_dim": config.linear_value_head_dim,
        "linear_num_key_heads": config.linear_num_key_heads,
        "linear_num_value_heads": config.linear_num_value_heads,
        "mtp_num_layers": config.mtp_num_hidden_layers,
        "rms_norm_eps": config.rms_norm_eps,
        "initializer_range": config.initializer_range,
        "attention_dropout": config.attention_dropout,
        "tie_word_embeddings": config.tie_word_embeddings,
        "attention_bias": config.attention_bias,
        "rope_theta": _rope_value(config, "rope_theta"),
        "partial_rotary_factor": _rope_value(config, "partial_rotary_factor"),
    }
    if expert:
        contract.update(
            {
                "num_experts": config.num_experts,
                "num_experts_per_tok": config.num_experts_per_tok,
                "moe_intermediate_size": config.moe_intermediate_size,
                "shared_expert_intermediate_size": config.shared_expert_intermediate_size,
                "router_aux_loss_coef": config.router_aux_loss_coef,
            }
        )
    else:
        contract["intermediate_size"] = config.intermediate_size
    return contract


def _megatron_contract(provider: Any, *, expert: bool) -> dict[str, Any]:
    """Normalize a post-override Megatron provider into common model fields."""
    num_layers = int(provider.num_layers)
    contract = {
        "num_layers": num_layers,
        "hidden_size": provider.hidden_size,
        "num_attention_heads": provider.num_attention_heads,
        "num_query_groups": provider.num_query_groups,
        "head_dim": provider.kv_channels,
        "vocab_size": provider.vocab_size,
        "layer_types": _layer_types(num_layers, provider.linear_attention_freq),
        "linear_conv_kernel_dim": provider.linear_conv_kernel_dim,
        "linear_key_head_dim": provider.linear_key_head_dim,
        "linear_value_head_dim": provider.linear_value_head_dim,
        "linear_num_key_heads": provider.linear_num_key_heads,
        "linear_num_value_heads": provider.linear_num_value_heads,
        "mtp_num_layers": provider.mtp_num_layers,
        "rms_norm_eps": provider.layernorm_epsilon,
        "initializer_range": provider.init_method_std,
        "attention_dropout": provider.attention_dropout,
        "tie_word_embeddings": provider.share_embeddings_and_output_weights,
        "attention_bias": provider.add_qkv_bias,
        "rope_theta": provider.rotary_base,
        "partial_rotary_factor": provider.rotary_percent,
    }
    if expert:
        contract.update(
            {
                "num_experts": provider.num_moe_experts,
                "num_experts_per_tok": provider.moe_router_topk,
                "moe_intermediate_size": provider.moe_ffn_hidden_size,
                "shared_expert_intermediate_size": provider.moe_shared_expert_intermediate_size,
                "router_aux_loss_coef": provider.moe_aux_loss_coeff,
            }
        )
    else:
        contract["intermediate_size"] = provider.ffn_hidden_size
    return contract


def _mismatches(deepspeed: dict[str, Any], megatron: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return every differing canonical model field."""
    return {
        field: {"deepspeed": deepspeed.get(field), "megatron": megatron.get(field)}
        for field in sorted(deepspeed.keys() | megatron.keys())
        if deepspeed.get(field) != megatron.get(field)
    }


def _model_contract(
    model_id: str,
    recipe_name: str,
    *,
    expert: bool,
    num_experts: int | None,
) -> dict[str, Any]:
    """Build and compare the exact post-override configs used by both runners."""
    hf_config = _text_config(model_id)
    hf_overrides: dict[str, Any] = {}
    megatron_overrides: dict[str, Any] = {}
    if num_experts is not None:
        if not hasattr(hf_config, "num_experts"):
            raise ValueError(f"{model_id} has no num_experts field")
        hf_config.num_experts = num_experts
        hf_overrides["num_experts"] = num_experts

    recipe_module = importlib.import_module(QWEN35_RECIPE_MODULE)
    recipe_factory = getattr(recipe_module, recipe_name, None)
    if not callable(recipe_factory):
        raise ValueError(f"Unknown Megatron recipe: {recipe_name}")
    provider = recipe_factory().model
    if num_experts is not None:
        provider.num_moe_experts = num_experts
        megatron_overrides["num_moe_experts"] = num_experts

    deepspeed = _hf_contract(hf_config, expert=expert)
    megatron = _megatron_contract(provider, expert=expert)
    mismatches = _mismatches(deepspeed, megatron)
    return {
        "model_id": model_id,
        "hf_commit_hash": getattr(hf_config, "_commit_hash", None),
        "megatron_recipe": recipe_name,
        "deepspeed_overrides": hf_overrides,
        "megatron_overrides": megatron_overrides,
        "deepspeed_effective": deepspeed,
        "megatron_effective": megatron,
        "aligned": not mismatches,
        "mismatches": mismatches,
    }


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    """Build a JSON-serializable model and training contract."""
    models = {
        "dense": _model_contract(
            args.dense_model,
            args.dense_recipe,
            expert=False,
            num_experts=None,
        ),
        "expert": _model_contract(
            args.moe_model,
            args.moe_recipe,
            expert=True,
            num_experts=args.num_experts,
        ),
    }
    return {
        "schema_version": 2,
        "model_alignment": {
            "status": "passed" if all(model["aligned"] for model in models.values()) else "failed",
            "scope": "post-override model architecture fields shared by Hugging Face and Megatron",
        },
        "models": models,
        "training": {
            "sequence_length": args.sequence_length,
            "dtype": "bf16",
            "seed": args.seed,
            "optimizer": {
                "name": "AdamW",
                "lr": 0.001,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01,
            },
        },
        "parallelism": {
            "num_gpus": args.num_gpus,
            "dense": {"tensor": 1, "pipeline": 1, "context": 1, "expert": 1},
            "expert": {"tensor": 1, "pipeline": 1, "context": 1, "expert": args.num_gpus},
        },
        "comparability": {
            "deepspeed_param_cpu": "not_available_in_megatron_runner",
            "deepspeed_act_cpu": "module_fine_grained_offload_plus_selective_recompute",
            "optimizer_offload": "disabled_while_megatron_fsdp_and_cpu_optimizer_are_incompatible",
            "optimizer_placement": "gpu",
            "megatron_sharding": "optim_grads_params (ZeRO-3 equivalent)",
        },
    }


def main() -> None:
    """Parse arguments and write the manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dense-model", default="Qwen/Qwen3.5-9B-Base")
    parser.add_argument("--moe-model", default="Qwen/Qwen3.5-35B-A3B-Base")
    parser.add_argument("--dense-recipe", required=True)
    parser.add_argument("--moe-recipe", required=True)
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if manifest["model_alignment"]["status"] != "passed":
        mismatches = {name: model["mismatches"] for name, model in manifest["models"].items()}
        raise ValueError(f"DeepSpeed and Megatron effective model configs differ: {mismatches}")
    LOGGER.info("Wrote model manifest to %s", args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
