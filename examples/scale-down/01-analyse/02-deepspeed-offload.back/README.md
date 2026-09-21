# DeepSpeed MoE Offload Example

This directory is a runnable copy of the external DeepSpeed expert-parallel
example. `train.py` still initializes model weights randomly unless
`--load_init_weights` is supplied.

## Download the dataset

The default dataset matches `train.py` (`wikitext-103-raw-v1`, 10% of the
training split):

```bash
./download_dataset.sh
```

The script sets `HF_CACHE`, `HF_HOME`, `HF_HUB_CACHE`, `HF_DATASETS_CACHE`, and
`TRANSFORMERS_CACHE` under `${REPO_ROOT}/.cache/huggingface`. Override the
dataset with, for example, `./download_dataset.sh --dataset-name ag_news
--dataset-config '' --split train`.

## Build a reduced model config

Yes, a Hugging Face `config.json` can be downloaded without model weights and
used to construct a randomly initialized model. For a local file, use the
config builder:

```bash
uv run python build_model_config.py \
  --input-config /path/to/config.json \
  --output-config ./qwen35-small-config.json \
  --text-config-only \
  --num-hidden-layers 8 \
  --hidden-size 2048 \
  --linear-attention-freq 4 \
  --num-experts 64 \
  --num-experts-per-tok 4
```

For a multimodal Qwen3.5 wrapper, `--text-config-only` writes the nested
`text_config` object expected by `Qwen3_5MoeTextConfig`. Without that option,
the wrapper is preserved and only its nested text config is changed. Use
`--set key=value` for any additional JSON field.

## Launch with mock data

`--mock_data` skips tokenizer and dataset downloads. The model is built from
the config and randomly initialized by the Transformers constructor:

```bash
deepspeed --num_gpus 8 train.py \
  --mode autoep \
  --autoep_size 8 \
  --model_config ./qwen35-small-config.json \
  --mock_data \
  --num_layers 8 \
  --num_experts 64 \
  --steps 50
```

`--num_layers` and `--num_experts` are runtime overrides; the generated JSON is
the reproducible source of truth. `--deepspeed_config path/to/ds_config.json`
loads a user-authored DeepSpeed JSON instead of the in-memory default.

Downloading only `config.json` does not download weights. A later call to
`from_pretrained` may download weights, but this example uses a config
constructor and therefore does not require them.

## Profile CUDA and pinned memory

Enable profiling on the training command:

```bash
deepspeed --num_gpus 8 train.py \
  --mode autoep \
  --autoep_size 8 \
  --model_config ./qwen35-small-config.json \
  --mock_data \
  --profile \
  --profile_out ./memory-profile.json \
  --steps 10
```

`--profile` does three things:

1. Sets DeepSpeed's real configuration key `memory_breakdown=true`. The
   example also carries the requested alias `memory_break=true`; DeepSpeed
   itself consumes `memory_breakdown`.
2. Samples CUDA allocated/reserved/peak memory and allocator counts at step,
   batch, forward, backward, and optimizer boundaries.
3. Samples Linux `VmPin` (falling back to `VmLck`) and bytes in pinned tensors.
   The process-level pinned, locked, and tensor-level values are all reported
   in the JSON so measurements remain interpretable on platforms without
   `/proc`.

Each rank writes a file such as
`memory-profile_rank-0.json`. Convert one rank or a directory of rank files to
an XLSX workbook:

```bash
uv run python export_memory_profile.py \
  --input ./memory-profile_rank-0.json \
  --output ./memory-profile.xlsx
```

The workbook has two sheets, `cuda_memory` and `pinned_memory`. Their phase,
time, start/end/peak, delta, allocation-count, and free-count columns follow
the layout produced by `scripts/scale-down/analyse/replay_step.py`. The pinned
sheet additionally puts locked-memory and pinned-tensor details in the
`overlap` column so it remains compatible with the replay table shape.
