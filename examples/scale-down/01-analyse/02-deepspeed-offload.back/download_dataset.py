"""Download and cache the text dataset used by ``train.py``."""

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
import logging
import os
from pathlib import Path

from datasets import DownloadConfig, load_dataset

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache a Hugging Face dataset locally.")
    parser.add_argument("--dataset-name", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    parser.add_argument("--split", default="train[:10%]")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Override HF cache root; download_dataset.sh sets this to .cache/huggingface.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir or os.environ.get("HF_CACHE", "~/.cache/huggingface")).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir / "datasets"))
    logger.info(
        "Downloading %s%s split=%s into %s",
        args.dataset_name,
        f"[{args.dataset_config}]" if args.dataset_config else "",
        args.split,
        cache_dir,
    )
    download_config = DownloadConfig(cache_dir=str(cache_dir / "datasets"))
    if args.dataset_config:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config,
            split=args.split,
            download_config=download_config,
        )
    else:
        dataset = load_dataset(
            args.dataset_name,
            split=args.split,
            download_config=download_config,
        )
    logger.info("Cached %s rows with columns=%s", len(dataset), dataset.column_names)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
