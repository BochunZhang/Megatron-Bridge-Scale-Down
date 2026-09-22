from dataclasses import dataclass, field


@dataclass(kw_only=True)
class ProfilingConfig:
    """Configuration settings for profiling the training process."""

    use_nsys_profiler: bool = field(default=False, metadata={"argparse_meta": {"arg_names": ["--profile"], "dest": "profile"}})
    """Enable nsys profiling. When using this option, nsys options should be specified in
    commandline. An example nsys commandline is
    `nsys profile -s none -t nvtx,cuda -o <path/to/output_file> --force-overwrite true
    --capture-range=cudaProfilerApi --capture-range-end=stop`.
    """

    profile_step_start: int = 10
    """Global step to start profiling."""

    profile_step_end: int = 12
    """Global step to stop profiling."""

    use_pytorch_profiler: bool = False
    """Use the built-in pytorch profiler. Useful if you wish to view profiles in tensorboard."""

    pytorch_profiler_collect_shapes: bool = False
    """Collect tensor shape in pytorch profiler."""
  
    pytorch_profiler_collect_callstack: bool = False
    """Collect callstack in pytorch profiler."""
  
    pytorch_profiler_collect_chakra: bool = False
    """Collect chakra trace in pytorch profiler."""

    profile_ranks: list[int] = field(default_factory=lambda: [])
    """Global ranks to profile."""

    record_memory_history: bool = False
    """Record memory history in last rank."""

    memory_snapshot_path: str = "snapshot.pickle"
    """Specifies where to dump the memory history pickle."""

    record_shapes: bool = False
    """Record shapes of tensors in `torch.autograd.profiler.emit_nvtx` for the Nsys profiler."""

    nvtx_ranges: bool = False
    """Enable NVTX range annotations for profiling. When enabled, inserts NVTX markers
    to categorize execution in profiler output."""