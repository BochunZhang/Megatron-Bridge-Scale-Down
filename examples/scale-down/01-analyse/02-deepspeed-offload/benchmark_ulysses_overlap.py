#!/usr/bin/env python3
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
"""Measure QKV projection overlap in DeepSpeed Ulysses on CUDA with NCCL.

Both modes use the same split Q/K/V and output projections, causal Flash SDPA,
and DeepSpeed DistributedAttention. Serial enqueues projections on the caller's
stream. Overlap records Q/K completion events on a projection stream for
DistributedAttention to wait on before each exchange. The module waits for the
full projection stream before exchanging V and uses its native asynchronous
Q/K gradient exchanges in backward.

A small FP32 run checks outputs, input gradients and summed weight gradients
against unsharded attention. Timings cover one synthetic attention layer with
no optimizer. The reported median uses the slowest rank in each iteration.
Run this file directly to launch four local GPU workers by default. Use --nvtx
when collecting a separate Nsight trace.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from statistics import median
from typing import Literal

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


logger = logging.getLogger(__name__)
Mode = Literal["serial", "overlap"]
MODES: tuple[Mode, ...] = ("serial", "overlap")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nproc-per-node", type=int, default=4, help="Local GPU workers (default: 4)")
    parser.add_argument("--seq-len", type=int, default=16384, help="Global sequence length")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=32, help="Number of query heads")
    parser.add_argument("--kv-heads", type=int, help="Number of KV heads, defaults to --heads")
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations per mode")
    parser.add_argument("--iters", type=int, default=20, help="Measured iterations per mode")
    parser.add_argument("--check-only", action="store_true", help="Stop after the FP32 correctness check")
    parser.add_argument("--nvtx", action="store_true", help="Label projection, attention and iteration ranges")
    args = parser.parse_args(argv)
    if args.kv_heads is None:
        args.kv_heads = args.heads
    for name in ("seq_len", "batch_size", "heads", "kv_heads", "head_dim", "iters"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.heads % args.kv_heads:
        parser.error("--heads must be divisible by --kv-heads")
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    return args


def _range(enabled: bool, name: str):
    return torch.cuda.nvtx.range(name) if enabled else nullcontext()


def _local_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    result = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        is_causal=True,
        enable_gqa=query.size(2) != key.size(2),
    )
    return result.transpose(1, 2)


class _AttentionLayer(nn.Module):
    def __init__(self, *, heads: int, kv_heads: int, head_dim: int, device: torch.device, dtype: torch.dtype) -> None:
        from deepspeed.sequence.layer import DistributedAttention

        super().__init__()
        self.head_dim = head_dim
        self.width = heads * head_dim
        self.q_proj = nn.Linear(self.width, self.width, bias=False, device=device, dtype=dtype)
        self.k_proj = nn.Linear(self.width, kv_heads * head_dim, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(self.width, kv_heads * head_dim, bias=False, device=device, dtype=dtype)
        self.out_proj = nn.Linear(self.width, self.width, bias=False, device=device, dtype=dtype)

        self.projection_stream = torch.cuda.Stream(device=device)
        self.serial_attention = DistributedAttention(_local_attention, dist.group.WORLD, scatter_idx=2, gather_idx=1)
        self.overlap_attention = DistributedAttention(
            _local_attention, dist.group.WORLD, scatter_idx=2, gather_idx=1, sp_stream=self.projection_stream
        )

    def forward(self, hidden_states: Tensor, *, mode: Mode | Literal["reference"], nvtx: bool = False) -> Tensor:
        consumer_stream = torch.cuda.current_stream(hidden_states.device)
        projection_stream = self.projection_stream if mode == "overlap" else None
        if projection_stream is not None:
            projection_stream.wait_stream(consumer_stream)

        projections = []
        with torch.cuda.stream(projection_stream):
            for name, linear in (("q_proj", self.q_proj), ("k_proj", self.k_proj), ("v_proj", self.v_proj)):
                with _range(nvtx, name):
                    tensor = linear(hidden_states).unflatten(-1, (-1, self.head_dim))
                    if projection_stream is not None:
                        if name != "v_proj":
                            # DistributedAttention.layer_sync reads this event before Q/K A2A.
                            tensor.done_event = projection_stream.record_event()
                        tensor.record_stream(consumer_stream)
                    projections.append(tensor)

        with _range(nvtx, "distributed_attention"):
            if mode == "reference":
                context = _local_attention(*projections)
            else:
                attention = self.overlap_attention if mode == "overlap" else self.serial_attention
                context = attention(*projections, batch_dim_idx=0)
        with _range(nvtx, "out_proj"):
            return self.out_proj(context.flatten(-2))


def _make_case(
    args: argparse.Namespace,
    device: torch.device,
    *,
    local_seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
) -> tuple[_AttentionLayer, Tensor, Tensor]:
    # The weights are replicated and agree across ranks. Each sequence shard is distinct.
    torch.manual_seed(2026)
    layer = _AttentionLayer(heads=args.heads, kv_heads=args.kv_heads, head_dim=head_dim, device=device, dtype=dtype)
    generator = torch.Generator(device=device).manual_seed(2026 + dist.get_rank())
    shape = (args.batch_size, local_seq_len, layer.width)
    hidden_states = torch.randn(shape, device=device, dtype=dtype, generator=generator, requires_grad=True)
    output_grad = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    return layer, hidden_states, output_grad


def _gather_sequence(shard: Tensor) -> Tensor:
    shards = [torch.empty_like(shard) for _ in range(dist.get_world_size())]
    dist.all_gather(shards, shard)
    return torch.cat(shards, dim=1)


def _check_reference(args: argparse.Namespace, device: torch.device) -> None:
    world_size = dist.get_world_size()
    local_seq_len = min(16, args.seq_len // world_size)
    layer, hidden_states, output_grad = _make_case(
        args, device, local_seq_len=local_seq_len, head_dim=min(16, args.head_dim), dtype=torch.float32
    )
    parameters = dict(layer.named_parameters())
    offset = dist.get_rank() * local_seq_len
    full_input = _gather_sequence(hidden_states.detach()).requires_grad_()
    full_output_grad = _gather_sequence(output_grad)
    with sdpa_kernel(SDPBackend.MATH):
        expected = layer(full_input, mode="reference")
        expected_grads = torch.autograd.grad(expected, (full_input, *parameters.values()), full_output_grad)
        expected_shards = (
            expected.detach().narrow(1, offset, local_seq_len),
            expected_grads[0].narrow(1, offset, local_seq_len),
            *expected_grads[1:],
        )
        for mode in MODES:
            actual = layer(hidden_states, mode=mode)
            grads = torch.autograd.grad(actual, (hidden_states, *parameters.values()), output_grad)
            if mode == "overlap":
                torch.cuda.current_stream(device).wait_stream(layer.projection_stream)
            for grad in grads[1:]:
                dist.all_reduce(grad)
            names = ("output", "input_grad", *parameters)
            for name, value, expected_value in zip(names, (actual, *grads), expected_shards, strict=True):
                torch.testing.assert_close(
                    value, expected_value, rtol=2e-4, atol=2e-4, msg=lambda error: f"{mode} {name}: {error}"
                )


def _measure(
    layer: _AttentionLayer,
    hidden_states: Tensor,
    output_grad: Tensor,
    args: argparse.Namespace,
    *,
    mode: Mode,
) -> tuple[float, float, float]:
    device = hidden_states.device
    parameters = (hidden_states, *layer.parameters())
    for _ in range(args.warmup):
        torch.autograd.grad(layer(hidden_states, mode=mode), parameters, output_grad)
    torch.cuda.synchronize(device)
    dist.barrier(device_ids=[device.index])

    timings = []
    with _range(args.nvtx, f"ulysses/{mode}/measure"):
        for _ in range(args.iters):
            start = torch.cuda.Event(enable_timing=True)
            forward_end = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            actual = layer(hidden_states, mode=mode, nvtx=args.nvtx)
            forward_end.record()
            with _range(args.nvtx, "backward"):
                torch.autograd.grad(actual, parameters, output_grad)
                if mode == "overlap":
                    torch.cuda.current_stream(device).wait_stream(layer.projection_stream)
            end.record()
            timings.append((start, forward_end, end))

    torch.cuda.synchronize(device)
    rank_times = torch.tensor(
        [[start.elapsed_time(mid), mid.elapsed_time(end), start.elapsed_time(end)] for start, mid, end in timings],
        device=device,
    )
    dist.all_reduce(rank_times, op=dist.ReduceOp.MAX)
    fwd_ms, bwd_ms, total_ms = (median(phase) for phase in rank_times.T.tolist())
    return fwd_ms, bwd_ms, total_ms


def main(argv: list[str] | None = None) -> None:
    """Launch local workers or run the benchmark on the current distributed rank."""
    argv = sys.argv[1:] if argv is None else argv
    args = _parse_args(argv)
    if not torch.cuda.is_available() or not dist.is_nccl_available():
        raise SystemExit("The benchmark requires CUDA with NCCL support.")

    is_worker = all(name in os.environ for name in ("LOCAL_RANK", "RANK", "WORLD_SIZE"))
    world_size = int(os.environ["WORLD_SIZE"]) if is_worker else args.nproc_per_node
    if world_size < 2:
        raise SystemExit("Ulysses overlap requires at least two ranks.")
    if any(size % world_size for size in (args.seq_len, args.heads, args.kv_heads)):
        raise SystemExit("--seq-len, --heads and --kv-heads must each be divisible by the world size.")
    if not is_worker:
        visible_gpus = torch.cuda.device_count()
        if world_size > visible_gpus:
            raise SystemExit(f"Requested {world_size} GPU workers but only {visible_gpus} CUDA devices are visible.")

        from torch.distributed.run import main as launch

        launch(["--standalone", f"--nproc-per-node={world_size}", str(Path(__file__).resolve()), *argv])
        return

    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)

    import deepspeed

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    deepspeed.init_distributed("nccl", auto_mpi_discovery=False)
    try:
        rank = dist.get_rank()
        if rank == 0:
            logger.info(
                "DeepSpeed %s (%s, %s), PyTorch %s, NCCL %s",
                deepspeed.__version__,
                deepspeed.__git_hash__,
                deepspeed.__file__,
                torch.__version__,
                torch.cuda.nccl.version(),
            )
            logger.info(
                "GPU %s, CP %d, CUDA_DEVICE_MAX_CONNECTIONS=%s, %s",
                torch.cuda.get_device_name(device),
                world_size,
                os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS", "unset"),
                vars(args),
            )

        _check_reference(args, device)
        dist.barrier(device_ids=[device.index])
        if rank == 0:
            logger.info("FP32 reference passed on all ranks for output, input gradient and reduced weight gradients.")
        if args.check_only:
            return

        dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
        layer, hidden_states, output_grad = _make_case(
            args, device, local_seq_len=args.seq_len // world_size, head_dim=args.head_dim, dtype=dtype
        )
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            results = {mode: _measure(layer, hidden_states, output_grad, args, mode=mode) for mode in MODES}
        if rank == 0:
            for mode, (fwd, bwd, total) in results.items():
                logger.info("%s  forward %.3f ms  backward %.3f ms  total %.3f ms", mode, fwd, bwd, total)
            logger.info(
                "serial / overlap  forward %.3fx  total %.3fx",
                results["serial"][0] / results["overlap"][0],
                results["serial"][2] / results["overlap"][2],
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
