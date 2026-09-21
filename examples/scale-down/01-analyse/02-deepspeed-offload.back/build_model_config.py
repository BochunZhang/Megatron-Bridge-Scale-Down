"""Build a smaller Hugging Face model config for the DeepSpeed example."""

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

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _json_value(value: str) -> Any:
    """Parse a JSON scalar/list/object, retaining plain strings as strings."""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _set_dotted(config: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    if not all(parts):
        raise ValueError(f"Invalid --set path: {path!r}")
    target = config
    for part in parts[:-1]:
        child = target.get(part)
        if not isinstance(child, dict):
            child = {}
            target[part] = child
        target = child
    target[parts[-1]] = value


def _select_config(document: dict[str, Any], text_config_only: bool) -> tuple[dict[str, Any], dict[str, Any] | None]:
    nested = document.get("text_config")
    if isinstance(nested, dict):
        if text_config_only:
            return dict(nested), None
        return nested, document
    return document, None


def _truncate_layer_metadata(config: dict[str, Any], num_layers: int) -> None:
    layer_types = config.get("layer_types")
    if isinstance(layer_types, list):
        config["layer_types"] = layer_types[:num_layers]
    max_window_layers = config.get("max_window_layers")
    if isinstance(max_window_layers, int):
        config["max_window_layers"] = min(max_window_layers, num_layers)
    mlp_only_layers = config.get("mlp_only_layers")
    if isinstance(mlp_only_layers, list):
        config["mlp_only_layers"] = [index for index in mlp_only_layers if index < num_layers]


def _apply_args(config: dict[str, Any], args: argparse.Namespace) -> None:
    updates = {
        "num_hidden_layers": args.num_hidden_layers,
        "hidden_size": args.hidden_size,
        "linear_attention_freq": args.linear_attention_freq,
        "num_experts": args.num_experts,
        "num_experts_per_tok": args.num_experts_per_tok,
        "intermediate_size": args.intermediate_size,
        "moe_intermediate_size": args.moe_intermediate_size,
    }
    for key, value in updates.items():
        if value is not None:
            config[key] = value
    if args.num_hidden_layers is not None:
        _truncate_layer_metadata(config, args.num_hidden_layers)
    if args.architecture is not None:
        config["architectures"] = [args.architecture]
    for override in args.set_values:
        if "=" not in override:
            raise ValueError(f"--set must use KEY=VALUE: {override!r}")
        path, value = override.split("=", 1)
        _set_dotted(config, path, _json_value(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a reduced model config without downloading model weights."
    )
    parser.add_argument("--input-config", required=True, type=Path)
    parser.add_argument("--output-config", required=True, type=Path)
    parser.add_argument("--text-config-only", action="store_true")
    parser.add_argument("--num-hidden-layers", type=_positive_int)
    parser.add_argument("--hidden-size", type=_positive_int)
    parser.add_argument("--linear-attention-freq", type=_positive_int)
    parser.add_argument("--num-experts", type=_positive_int)
    parser.add_argument("--num-experts-per-tok", type=_positive_int)
    parser.add_argument("--intermediate-size", type=_positive_int)
    parser.add_argument("--moe-intermediate-size", type=_positive_int)
    parser.add_argument("--architecture")
    parser.add_argument(
        "--set",
        dest="set_values",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Set any additional dotted JSON field; VALUE is parsed as JSON when possible.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input_config.open(encoding="utf-8") as config_file:
        document = json.load(config_file)
    if not isinstance(document, dict):
        raise ValueError("Input config must be a JSON object")
    config, wrapper = _select_config(document, args.text_config_only)
    _apply_args(config, args)
    output = config if wrapper is None else wrapper
    if wrapper is not None:
        wrapper["text_config"] = config
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    with args.output_config.open("w", encoding="utf-8") as output_file:
        json.dump(output, output_file, indent=2, ensure_ascii=False)
        output_file.write("\n")
    logger.info("Wrote model config to %s", args.output_config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        main()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.error("%s", exc)
        raise SystemExit(2) from exc
