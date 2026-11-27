#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train a one-block language model with 4-rank DeepSpeed Ulysses CUDA Graphs.

Each configuration captures forward, backward, asynchronous DeepSpeed collectives
and AdamW in one graph. The timed steps replay the graph on fixed synthetic token
shards. Attention uses BF16 with Flash SDPA and model weights stay in FP32.

A2A bytes come from the actual buffers submitted while capturing the graph. The
reported remote send excludes the self chunk of each equal-split all-to-all.
Gradient all-reduce bytes describe its input tensor per rank.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from statistics import median
from types import ModuleType

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


logger = logging.getLogger(__name__)
CP = 4
Q_HEADS = 16
HEAD_DIM = 128
WIDTH = Q_HEADS * HEAD_DIM
VOCAB = 1024
SEED = 2026
DEFAULT_RESULTS = (
    Path(__file__).resolve().parents[4] / "results/scale-down/01-analyse/02-deepspeed-offload/ulysses_overlap"
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[2**power for power in range(12, 19)],
        help="Global lengths, 4K to 256K by default",
    )
    parser.add_argument("--kv-heads", type=int, nargs="+", default=[16, 4], help="KV head counts, 16 and 4 by default")
    parser.add_argument("--warmup", type=int, default=3, help="Eager training steps before capture")
    parser.add_argument("--replay-warmup", type=int, default=2, help="Untimed replay steps")
    parser.add_argument("--iters", type=int, default=10, help="Timed replay steps per configuration")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--nvtx", action="store_true", help="Mark the timed replay range in Nsight Systems")
    args = parser.parse_args(argv)
    if any(length <= 0 or length % CP for length in args.seq_lens):
        parser.error("--seq-lens must be positive multiples of 4")
    if any(heads <= 0 or heads % CP or Q_HEADS % heads for heads in args.kv_heads):
        parser.error("--kv-heads must be 4, 8 or 16")
    if args.warmup < 1 or args.replay_warmup < 0 or args.iters < 1:
        parser.error("--warmup and --iters must be positive and --replay-warmup must be nonnegative")
    return args


def _local_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    return F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        is_causal=True,
        enable_gqa=query.size(2) != key.size(2),
    ).transpose(1, 2)


class _LanguageModel(nn.Module):
    def __init__(self, kv_heads: int, compute_stream: torch.cuda.Stream) -> None:
        from deepspeed.sequence.layer import DistributedAttention

        super().__init__()
        device = compute_stream.device
        factory_kwargs = {"device": device, "dtype": torch.float32}
        self.embedding = nn.Embedding(VOCAB, WIDTH, **factory_kwargs)
        self.attention_norm = nn.LayerNorm(WIDTH, **factory_kwargs)
        self.q_proj = nn.Linear(WIDTH, WIDTH, bias=False, **factory_kwargs)
        self.k_proj = nn.Linear(WIDTH, kv_heads * HEAD_DIM, bias=False, **factory_kwargs)
        self.v_proj = nn.Linear(WIDTH, kv_heads * HEAD_DIM, bias=False, **factory_kwargs)
        self.out_proj = nn.Linear(WIDTH, WIDTH, bias=False, **factory_kwargs)
        self.mlp_norm = nn.LayerNorm(WIDTH, **factory_kwargs)
        self.mlp = nn.Sequential(
            nn.Linear(WIDTH, 4 * WIDTH, bias=False, **factory_kwargs),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * WIDTH, WIDTH, bias=False, **factory_kwargs),
        )
        self.lm_head = nn.Linear(WIDTH, VOCAB, bias=False, **factory_kwargs)
        self.projection_stream = torch.cuda.Stream(device=device)
        self.attention = DistributedAttention(
            _local_attention, dist.group.WORLD, scatter_idx=2, gather_idx=1, sp_stream=self.projection_stream
        )
        # DeepSpeed uses this stream for its projection events and gradient hooks.
        self.attention.default_stream = compute_stream

    def forward(self, tokens: Tensor) -> Tensor:
        residual = self.embedding(tokens)
        hidden = self.attention_norm(residual)
        compute_stream = torch.cuda.current_stream(hidden.device)
        self.projection_stream.wait_stream(compute_stream)
        with torch.cuda.stream(self.projection_stream):
            query = self.q_proj(hidden).unflatten(-1, (-1, HEAD_DIM))
            query.done_event = self.projection_stream.record_event()
            key = self.k_proj(hidden).unflatten(-1, (-1, HEAD_DIM))
            key.done_event = self.projection_stream.record_event()
            value = self.v_proj(hidden).unflatten(-1, (-1, HEAD_DIM))
            for projection in (query, key, value):
                projection.record_stream(compute_stream)
        context = self.attention(query, key, value, batch_dim_idx=0)
        hidden = residual + self.out_proj(context.flatten(-2))
        return self.lm_head(hidden + self.mlp(self.mlp_norm(hidden)))


class _Traffic:
    def __init__(self) -> None:
        self.forward: list[int] = []
        self.backward: list[int] = []
        self.buffers = self.forward

    @contextmanager
    def async_a2a(self) -> Iterator[None]:
        """Use asynchronous transport for the native Ulysses call sites."""
        from deepspeed import comm

        original = comm.all_to_all_single

        def launch(
            output: Tensor, tensor: Tensor, *, group: dist.ProcessGroup, async_op: bool = False
        ) -> dist.Work | None:
            work = original(output, tensor, group=group, async_op=True)
            self.buffers.append(tensor.numel() * tensor.element_size())
            if async_op:
                return work
            # Preserve DeepSpeed's synchronous call sites as stream dependencies.
            work.wait()
            return None

        comm.all_to_all_single = launch
        try:
            yield
        finally:
            comm.all_to_all_single = original

    def remote_send(self, local_seq: int, kv_heads: int) -> tuple[int, int]:
        # Q/O and K/V each contribute two BF16 buffers per phase.
        expected = local_seq * HEAD_DIM * (2 * Q_HEADS + 2 * kv_heads) * 2
        for phase, buffers in (("forward", self.forward), ("backward", self.backward)):
            if len(buffers) != 4 or sum(buffers) != expected:
                raise RuntimeError(
                    f"Unexpected {phase} A2A buffers {buffers}, expected four totaling {expected} bytes"
                )
        return sum(self.forward) * (CP - 1) // CP, sum(self.backward) * (CP - 1) // CP


def _benchmark(
    args: argparse.Namespace, seq_len: int, kv_heads: int, device: torch.device
) -> dict[str, str | int | float]:
    from deepspeed import comm

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    compute_stream = torch.cuda.Stream(device=device)
    local_seq = seq_len // CP
    offset = dist.get_rank() * local_seq
    with torch.cuda.stream(compute_stream):
        torch.cuda.manual_seed(SEED)
        model = _LanguageModel(kv_heads, compute_stream)
        params = tuple(model.parameters())
        optimizer = torch.optim.AdamW(params, lr=3e-4, fused=True, capturable=True)
        param_sizes = [param.numel() for param in params]
        flat_grad = torch.empty(sum(param_sizes), device=device, dtype=torch.float32)
        grad_views = tuple(shard.view_as(param) for shard, param in zip(flat_grad.split(param_sizes), params))
        for param, grad in zip(params, grad_views):
            param.grad = grad

        generator = torch.Generator(device=device).manual_seed(SEED)
        token_stream = torch.randint(VOCAB, (1, seq_len + 1), generator=generator, device=device)
        tokens = token_stream.narrow(1, offset, local_seq)
        targets = token_stream.narrow(1, offset + 1, local_seq)
    traffic = _Traffic()

    def train_step() -> Tensor:
        traffic.buffers = traffic.forward
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            logits = model(tokens)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1))
        traffic.buffers = traffic.backward
        gradients = torch.autograd.grad(loss, params)
        compute_stream.wait_stream(model.projection_stream)
        for view, grad in zip(grad_views, gradients):
            view.copy_(grad)
        comm.all_reduce(flat_grad, async_op=True).wait()
        flat_grad.mul_(1 / CP)
        optimizer.step()
        return loss.detach()

    graph = torch.cuda.CUDAGraph()
    with traffic.async_a2a():
        with torch.cuda.stream(compute_stream):
            for _ in range(args.warmup):
                train_step()
        torch.cuda.synchronize(device)
        before = model.lm_head.weight[0, :32].detach().clone()
        dist.barrier(device_ids=[device.index])
        traffic.forward.clear()
        traffic.backward.clear()
        with torch.cuda.graph(graph, stream=compute_stream):
            loss = train_step()
    forward_bytes, backward_bytes = traffic.remote_send(local_seq, kv_heads)

    with torch.cuda.stream(compute_stream):
        for _ in range(args.replay_warmup):
            graph.replay()
    torch.cuda.synchronize(device)
    dist.barrier(device_ids=[device.index])

    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(args.iters)]
    label = f"ulysses/replay/seq={seq_len}/kv_heads={kv_heads}"
    replay_range = torch.cuda.nvtx.range(label) if args.nvtx else nullcontext()
    with torch.cuda.stream(compute_stream), replay_range:
        for start, end in events:
            start.record()
            graph.replay()
            end.record()
    torch.cuda.synchronize(device)

    step_count = int(optimizer.state[model.lm_head.weight]["step"].item())
    expected_steps = args.warmup + args.replay_warmup + args.iters
    probe = model.lm_head.weight[0, :32].detach()
    replicas = [torch.empty_like(probe) for _ in range(CP)]
    dist.all_gather(replicas, probe)
    if step_count != expected_steps or torch.equal(before, probe):
        raise RuntimeError(f"Optimizer update check failed at step {step_count}, expected {expected_steps}")
    if any(not torch.equal(probe, other) for other in replicas):
        raise RuntimeError("Replicated weights diverged after CP gradient reduction")
    if not torch.isfinite(loss).item():
        raise RuntimeError("Non-finite loss after graph replay")

    rank_ms = torch.tensor([start.elapsed_time(end) for start, end in events], device=device)
    peak_bytes = torch.tensor(torch.cuda.max_memory_allocated(device), device=device, dtype=torch.int64)
    dist.all_reduce(rank_ms, op=dist.ReduceOp.MAX)
    dist.all_reduce(peak_bytes, op=dist.ReduceOp.MAX)
    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
    samples = rank_ms.tolist()
    p50 = median(samples)
    row = {
        "seq_len": seq_len,
        "local_seq_len": local_seq,
        "q_heads": Q_HEADS,
        "kv_heads": kv_heads,
        "forward_a2a_calls": len(traffic.forward),
        "backward_a2a_calls": len(traffic.backward),
        "a2a_forward_remote_send_bytes_per_rank": forward_bytes,
        "a2a_backward_remote_send_bytes_per_rank": backward_bytes,
        "a2a_remote_send_bytes_per_rank": forward_bytes + backward_bytes,
        "grad_allreduce_calls": 1,
        "grad_allreduce_input_bytes_per_rank": flat_grad.numel() * flat_grad.element_size(),
        "replay_p50_ms": p50,
        "replay_p90_ms": sorted(samples)[math.ceil(0.9 * len(samples)) - 1],
        "tokens_per_second": seq_len * 1000 / p50,
        "case_peak_allocated_bytes_rank_max": peak_bytes.item(),
        "mean_loss_last_step": loss.item() / CP,
        "optimizer_steps": step_count,
        "replay_samples_ms_rank_max": json.dumps(samples),
    }
    graph.reset()
    return row


def _run_matrix(args: argparse.Namespace, device: torch.device, deepspeed: ModuleType) -> None:
    rank = dist.get_rank()
    output_context = nullcontext()
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix="training-", dir=args.output_dir))
        metadata = {
            **vars(args),
            "gpu": torch.cuda.get_device_name(device),
            "context_parallel_size": CP,
            "num_layers": 1,
            "micro_batch_size": 1,
            "head_dim": HEAD_DIM,
            "vocab_size": VOCAB,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "nccl_version": torch.cuda.nccl.version(),
            "deepspeed_version": deepspeed.__version__,
            "deepspeed_git_hash": deepspeed.__git_hash__,
            "cuda_device_max_connections": os.getenv("CUDA_DEVICE_MAX_CONNECTIONS"),
            "a2a_bytes": "Equal-split remote send payload per rank and replay, excluding the self chunk",
            "gradient_bytes": "All-reduce input buffer per rank and replay",
            "replay_timing": "CUDA events around each replay, reduced to the slowest rank per sample",
        }
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
        logger.info("Results %s", run_dir)
        output_context = (run_dir / "summary.csv").open("w", newline="")

    with output_context as output:
        writer = None
        for seq_len in args.seq_lens:
            for kv_heads in args.kv_heads:
                if rank == 0:
                    logger.info("Training with seq_len=%d, Q/KV heads=%d/%d", seq_len, Q_HEADS, kv_heads)
                row = _benchmark(args, seq_len, kv_heads, device)
                if output is not None:
                    if writer is None:
                        writer = csv.DictWriter(output, fieldnames=tuple(row))
                        writer.writeheader()
                    writer.writerow(row)
                    output.flush()
                    logger.info(
                        "Replay %.3f ms p50, A2A %.2f MiB sent per rank, gradient AR %.2f MiB input per rank",
                        row["replay_p50_ms"],
                        row["a2a_remote_send_bytes_per_rank"] / 2**20,
                        row["grad_allreduce_input_bytes_per_rank"] / 2**20,
                    )


def main(argv: list[str] | None = None) -> None:
    """Launch four GPU workers and run the training matrix."""
    argv = sys.argv[1:] if argv is None else argv
    args = _parse_args(argv)
    if not torch.cuda.is_available() or not dist.is_nccl_available():
        raise SystemExit("This training benchmark requires CUDA and NCCL.")
    if not all(name in os.environ for name in ("LOCAL_RANK", "RANK", "WORLD_SIZE")):
        if torch.cuda.device_count() < CP:
            raise SystemExit("This training benchmark requires four visible CUDA devices.")
        from torch.distributed.run import main as launch

        launch(["--standalone", f"--nproc-per-node={CP}", str(Path(__file__).resolve()), *argv])
        return
    if int(os.environ["WORLD_SIZE"]) != CP:
        raise SystemExit("Expected exactly four training ranks.")
    if any(os.getenv(key, "0").lower() in ("1", "true") for key in ("TORCH_NCCL_BLOCKING_WAIT", "NCCL_BLOCKING_WAIT")):
        raise SystemExit("CUDA Graph capture requires NCCL Work.wait() to use CUDA stream waits.")

    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    import deepspeed

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    deepspeed.init_distributed("nccl", auto_mpi_discovery=False)
    try:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            _run_matrix(args, device, deepspeed)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
