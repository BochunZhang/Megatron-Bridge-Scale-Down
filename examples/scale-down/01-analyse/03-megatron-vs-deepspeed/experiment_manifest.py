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
import json
import logging
from pathlib import Path
from typing import Any

from transformers import AutoConfig


LOGGER = logging.getLogger(__name__)
CONTRACT_FIELDS = (
    "model_type",
    "num_hidden_layers",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "layer_types",
    "num_experts",
    "num_experts_per_tok",
    "vocab_size",
    "rope_theta",
    "max_position_embeddings",
)


def _text_config(model_id: str) -> Any:
    """Load a model config and select its text backbone when present."""
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    return getattr(config, "text_config", config)


def _snapshot(model_id: str) -> dict[str, Any]:
    """Extract stable architecture fields from a Hugging Face config."""
    config = _text_config(model_id)
    values: dict[str, Any] = {"model_id": model_id}
    for field in CONTRACT_FIELDS:
        value = getattr(config, field, None)
        if value is not None:
            values[field] = value
    return values


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    """Build a JSON-serializable model and training contract."""
    models = {
        "dense": _snapshot(args.dense_model),
        "expert": _snapshot(args.moe_model),
    }
    if args.num_experts is not None:
        models["expert"]["experiment_num_experts"] = args.num_experts
    return {
        "schema_version": 1,
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
            "optimizer_offload": "compare_memory_and_throughput_trend_only",
            "megatron_sharding": "optim_grads_params (ZeRO-3 equivalent)",
        },
    }


def main() -> None:
    """Parse arguments and write the manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dense-model", default="Qwen/Qwen3.5-9B-Base")
    parser.add_argument("--moe-model", default="Qwen/Qwen3.5-35B-A3B-Base")
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOGGER.info("Wrote model manifest to %s", args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
