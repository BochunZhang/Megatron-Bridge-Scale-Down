"""Hugging Face text data loading and MoE config helpers for the AutoEP example."""

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

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from datasets import DownloadConfig, load_dataset
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def build_model_config(
    config_cls: type[Any],
    num_hidden_layers: int | None = None,
    config_json: str | None = None,
    num_experts: int | None = None,
) -> Any:
    """Build a config from defaults or a local Hugging Face JSON file."""
    if config_json is None:
        kwargs: dict[str, Any] = {}
    else:
        config_path = Path(config_json)
        if not config_path.is_file():
            raise FileNotFoundError(f"Model config JSON does not exist: {config_path}")
        with config_path.open(encoding="utf-8") as config_file:
            loaded = json.load(config_file)
        if not isinstance(loaded, dict):
            raise ValueError(f"Model config must be a JSON object: {config_path}")
        text_config = loaded.get("text_config")
        kwargs = dict(text_config if isinstance(text_config, dict) else loaded)
        logger.info("Loaded model config from %s", config_path)
    if num_hidden_layers is not None:
        kwargs["num_hidden_layers"] = num_hidden_layers
    if num_experts is not None:
        if "num_local_experts" in kwargs:
            expert_key = "num_local_experts"
        elif "num_experts" in kwargs:
            expert_key = "num_experts"
        else:
            default_config = config_cls()
            expert_key = "num_local_experts" if hasattr(default_config, "num_local_experts") else "num_experts"
        kwargs[expert_key] = num_experts
    if config_json is None:
        return config_cls(**kwargs)
    from_dict = getattr(config_cls, "from_dict", None)
    return from_dict(kwargs) if callable(from_dict) else config_cls(**kwargs)


@dataclass
class CausalLmBatch:
    """One micro-batch for causal LM training (CPU tensors until the training loop moves them)."""

    input_ids: torch.Tensor  # [micro_batch_size, seq_len], dtype=torch.long
    attention_mask: torch.Tensor  # [micro_batch_size, seq_len], dtype=torch.long
    labels: torch.Tensor  # [micro_batch_size, seq_len], dtype=torch.long


def get_tokenizer(model_name: str, *, trust_remote_code: bool = True) -> Any:
    """Load tokenizer; pad token follows ds_verify_loss behavior."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.convert_ids_to_tokens(2)
    return tokenizer


def validate_tokenizer_vocab_size(
    tokenizer: Any,
    tokenizer_name: str,
    expected_vocab_size: int,
) -> dict[str, Any]:
    """Validate that tokenizer ids fit inside the model embedding table."""
    tokenizer_len = len(tokenizer)
    tokenizer_vocab_size = getattr(tokenizer, "vocab_size", None)
    if tokenizer_len > expected_vocab_size:
        raise ValueError(
            f"Tokenizer {tokenizer_name!r} len(tokenizer)={tokenizer_len} "
            f"(vocab_size={tokenizer_vocab_size}) exceeds model "
            f"vocab_size={expected_vocab_size}. "
            "Pick a tokenizer whose ids fit within the model config."
        )
    return {
        "tokenizer_name": tokenizer_name,
        "tokenizer_len": tokenizer_len,
        "tokenizer_vocab_size": tokenizer_vocab_size,
        "model_vocab_size": expected_vocab_size,
        "exact_vocab_match": tokenizer_len == expected_vocab_size,
    }


def _hf_train_split_string(dataset_fraction: float) -> str:
    """Map fraction in (0, 1] to datasets split slice (matches ds_verify_loss)."""
    if dataset_fraction >= 1.0:
        return "train"
    percentage_int = int(dataset_fraction * 100)
    return f"train[:{percentage_int}%]"


def load_hf_text_dataset_rows(
    dataset_name: str,
    dataset_fraction: float,
    *,
    is_main_process: bool,
) -> tuple[Any, str]:
    """Load raw text rows from Hugging Face (same presets as ds_verify_loss)."""
    if not is_main_process:
        disable_progress_bar()

    split_str = _hf_train_split_string(dataset_fraction)
    dl_cfg = DownloadConfig(disable_tqdm=True)

    if is_main_process:
        logger.info("Loading HF dataset: %s split=%r ...", dataset_name, split_str)

    if dataset_name == "wikitext":
        dataset = load_dataset(
            "wikitext",
            "wikitext-103-raw-v1",
            split=split_str,
            download_config=dl_cfg,
        )
        text_column = "text"
    elif dataset_name == "openwebtext":
        if dataset_fraction >= 1.0:
            split_str = "train[:1%]"
        dataset = load_dataset(
            "openwebtext", split=split_str, download_config=dl_cfg
        )
        text_column = "text"
    elif dataset_name == "c4":
        if dataset_fraction >= 1.0:
            split_str = "train[:0.1%]"
        dataset = load_dataset(
            "c4", "en", split=split_str, download_config=dl_cfg
        )
        text_column = "text"
    elif dataset_name == "ag_news":
        dataset = load_dataset(
            "ag_news", split=split_str, download_config=dl_cfg
        )
        text_column = "text"
    else:
        try:
            dataset = load_dataset(
                dataset_name, split=split_str, download_config=dl_cfg
            )
            if "text" in dataset.column_names:
                text_column = "text"
            elif "content" in dataset.column_names:
                text_column = "content"
            elif "body" in dataset.column_names:
                text_column = "body"
            else:
                text_column = dataset.column_names[0]
                if is_main_process:
                    logger.warning(
                        "Using column %r; columns=%s", text_column, dataset.column_names
                    )
        except Exception as e:
            if is_main_process:
                logger.warning(
                    "Error loading %r: %s; falling back to wikitext.", dataset_name, e
                )
            dataset = load_dataset(
                "wikitext",
                "wikitext-103-raw-v1",
                split=split_str,
                download_config=dl_cfg,
            )
            text_column = "text"

    if is_main_process:
        logger.info("HF dataset rows: %s (text column=%r)", len(dataset), text_column)
    return dataset, text_column


def tokenize_hf_dataset(
    dataset: Any,
    text_column: str,
    tokenizer: Any,
    seq_len: int,
    *,
    is_main_process: bool,
) -> Any:
    """Tokenize text column to fixed length (padding=max_length), torch columns."""

    def has_text(example: dict[str, Any]) -> bool:
        value = example.get(text_column)
        return isinstance(value, str) and bool(value.strip())

    def tokenize_fn(examples: dict[str, list]) -> dict[str, list]:
        return tokenizer(
            examples[text_column],
            padding="max_length",
            max_length=seq_len,
            truncation=True,
        )

    if is_main_process:
        logger.info("Filtering empty HF text rows...")
    dataset = dataset.filter(
        has_text,
        num_proc=1,
        keep_in_memory=True,
    )
    if len(dataset) == 0:
        raise ValueError(
            "HF dataset has no non-empty text rows; pick another dataset."
        )
    if is_main_process:
        logger.info("Non-empty HF dataset rows: %s.", len(dataset))
        logger.info("Tokenizing HF dataset...")
    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=1,
        remove_columns=dataset.column_names,
        keep_in_memory=True,
    )
    tokenized.set_format(
        type="torch", columns=["input_ids", "attention_mask"]
    )
    if is_main_process:
        logger.info("Tokenization complete: %s rows.", len(tokenized))
    if len(tokenized) == 0:
        raise ValueError(
            "Tokenized HF dataset is empty; increase dataset_percentage or pick another dataset."
        )
    return tokenized


class HFBatchGenerator:
    """Infinite iterator over a DataLoader; returns CausalLmBatch on CPU."""

    def __init__(
        self,
        dataloader: DataLoader,
        sampler: DistributedSampler,
    ) -> None:
        self.dataloader = dataloader
        self.sampler = sampler
        self._epoch = 0
        self._iter: Iterator | None = None

    def _next_raw_batch(self) -> dict[str, torch.Tensor]:
        if self._iter is None:
            self.sampler.set_epoch(self._epoch)
            self._iter = iter(self.dataloader)
        try:
            return next(self._iter)
        except StopIteration:
            self._epoch += 1
            self.sampler.set_epoch(self._epoch)
            self._iter = iter(self.dataloader)
            return next(self._iter)

    def get_batch(self, optimizer_step: int, accum_idx: int) -> CausalLmBatch:
        del optimizer_step, accum_idx  # sequential consumption (like an infinite epoch)
        batch = self._next_raw_batch()
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = input_ids.clone()
        labels = labels.masked_fill(attention_mask == 0, -100)
        return CausalLmBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )


def build_hf_batch_generator(
    *,
    dataset_name: str,
    dataset_percentage: float,
    tokenizer_name: str,
    expected_vocab_size: int,
    seq_len: int,
    micro_batch_size: int,
    dp_world_size: int,
    dp_rank: int,
    seed: int,
    rank: int,
    hf_num_dataloader_workers: int = 0,
) -> HFBatchGenerator:
    """Load HF text data, tokenize, and build a per-DP-rank batch generator."""
    is_main = rank == 0
    if dataset_percentage <= 0:
        raise ValueError("dataset_percentage must be positive")
    if dataset_percentage < 1.0:
        raise ValueError(
            "dataset_percentage must be at least 1.0 because Hugging Face "
            "split slicing uses whole percentages."
        )
    fraction = min(dataset_percentage / 100.0, 1.0)

    tokenizer = get_tokenizer(tokenizer_name, trust_remote_code=True)
    validate_tokenizer_vocab_size(tokenizer, tokenizer_name, expected_vocab_size)
    raw, text_col = load_hf_text_dataset_rows(
        dataset_name, fraction, is_main_process=is_main
    )
    tokenized = tokenize_hf_dataset(
        raw,
        text_col,
        tokenizer,
        seq_len,
        is_main_process=is_main,
    )
    sampler = DistributedSampler(
        tokenized,
        num_replicas=dp_world_size,
        rank=dp_rank,
        shuffle=True,
        seed=seed,
    )
    loader = DataLoader(
        tokenized,
        batch_size=micro_batch_size,
        sampler=sampler,
        num_workers=hf_num_dataloader_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return HFBatchGenerator(loader, sampler)


class MockBatchGenerator:
    """Generate deterministic random token batches without HF downloads."""

    def __init__(
        self,
        *,
        vocab_size: int,
        seq_len: int,
        micro_batch_size: int,
        seed: int,
        rank: int,
    ) -> None:
        if vocab_size <= 0 or seq_len <= 0 or micro_batch_size <= 0:
            raise ValueError("vocab_size, seq_len, and micro_batch_size must be positive")
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.micro_batch_size = micro_batch_size
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed + rank)

    def get_batch(self, optimizer_step: int, accum_idx: int) -> CausalLmBatch:
        """Return one random CPU batch; the arguments keep the training API stable."""
        del optimizer_step, accum_idx
        input_ids = torch.randint(
            self.vocab_size,
            (self.micro_batch_size, self.seq_len),
            generator=self.generator,
            dtype=torch.long,
        )
        attention_mask = torch.ones_like(input_ids)
        return CausalLmBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids.clone(),
        )
