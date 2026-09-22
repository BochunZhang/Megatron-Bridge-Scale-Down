import os
import torch
import logging
from typing import Optional

from config import ProfilingConfig

logger = logging.getLogger(__name__)

def get_rank_safe() -> int:
    """Get the distributed rank safely, even if torch.distributed is not initialized.

    Fallback order:
    1. torch.distributed.get_rank() (if initialized)
    2. RANK environment variable (torchrun/torchelastic)
    3. SLURM_PROCID environment variable (SLURM)
    4. Default: 0 (with warning)

    Returns:
        int: The rank of the current process.
    """
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    raise RuntimeError("torch.distributed is not initialized. Please ensure that you are running in a distributed environment.")


def print_rank_0(message: str) -> None:
    """Print a message only on global rank 0.

    Args:
        message: The message string to print.
    """
    rank = get_rank_safe()
    if rank == 0:
        print(message, flush=True)


def start_memory_history_recording(profiling: ProfilingConfig | None) -> None:
    """Enable the CUDA caching allocator trace so memory snapshots contain history.

    ``torch.cuda.memory._snapshot()`` only includes allocation/free events and
    Python stack context after ``_record_memory_history()`` has been enabled.
    Without this call, dumped snapshots contain only the current live
    allocations — no timeline, no call sites.

    Must be invoked before model construction so every tensor allocation is
    captured. Guarded by ``profile_ranks`` so only ranks that will dump a
    snapshot pay the recording overhead.
    """
    if profiling is None or not profiling.record_memory_history:
        return
    if get_rank_safe() not in profiling.profile_ranks:
        return

    torch.cuda.memory._record_memory_history(
        True,
        # Retain up to 100k alloc/free events.
        trace_alloc_max_entries=100_000,
        # Record the Python stack at each event — lets memory_viz show call sites.
        trace_alloc_record_context=True,
    )

    def _oom_observer(device: int, alloc: int, device_alloc: int, device_free: int) -> None:
        """Dump a snapshot on OOM so we can inspect what was live at the failure."""
        import pickle

        rank = get_rank_safe()
        base, ext = os.path.splitext(profiling.memory_snapshot_path)
        filename = f"{base}_oom_rank-{rank}{ext}"
        snapshot = torch.cuda.memory._snapshot()
        with open(filename, "wb") as f:
            pickle.dump(snapshot, f)
        # logger.info so the message reaches stderr on any profiled rank, not just rank 0.
        logger.info(f"[OOM] rank {rank} saved memory snapshot to {filename}")

    torch._C._cuda_attach_out_of_memory_observer(_oom_observer)
    print_rank_0(
        f"Memory history recording enabled (rank {get_rank_safe()}); "
        f"snapshots will be written to '{profiling.memory_snapshot_path}'."
    )
