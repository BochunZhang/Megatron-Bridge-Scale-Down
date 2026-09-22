import argparse
from dataclasses import dataclass, field, fields


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

    profile_ranks: list[int] = field(default_factory=lambda: [0])
    """Global ranks to profile. Memory-snapshot and recording-start guards use a
    strict membership check, so an empty list disables capture; the default
    ``[0]`` gives rank-0 capture with no further override required."""

    record_memory_history: bool = False
    """Record memory history in last rank."""

    memory_snapshot_path: str = "snapshot.pickle"
    """Specifies where to dump the memory history pickle."""

    record_shapes: bool = False
    """Record shapes of tensors in `torch.autograd.profiler.emit_nvtx` for the Nsys profiler."""

    nvtx_ranges: bool = False
    """Enable NVTX range annotations for profiling. When enabled, inserts NVTX markers
    to categorize execution in profiler output."""

    tensorboard_dir: str | None = None

    def finalize(self) -> None:
        """Validate profiling configuration."""
        assert not (self.use_pytorch_profiler and self.use_nsys_profiler), (
            "Exactly one of pytorch or nsys profiler should be enabled, not both."
        )
        assert self.profile_step_start >= 0, f"profile_step_start must be >= 0, got {self.profile_step_start}"
        assert self.profile_step_end >= 0, f"profile_step_end must be >= 0, got {self.profile_step_end}"
        assert self.profile_step_end >= self.profile_step_start, (
            f"profile_step_end ({self.profile_step_end}) must be >= profile_step_start ({self.profile_step_start})"
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ProfilingConfig":
        """Build a ProfilingConfig from an argparse namespace, parsing it generically.

        Every dataclass field is looked up on the namespace by its argparse dest
        (``metadata["argparse_meta"]["dest"]`` when present, otherwise the field
        name). Attributes that are missing or ``None`` fall back to the dataclass
        default, and unknown attributes are ignored — so callers can pass their
        full namespace through without per-field mapping.

        Args:
            args: Parsed argparse namespace (may contain unrelated arguments).

        Returns:
            A validated ProfilingConfig instance.
        """
        kwargs = {}
        for f in fields(cls):
            dest = f.metadata.get("argparse_meta", {}).get("dest", f.name)
            value = getattr(args, dest, None)
            if value is not None:
                kwargs[f.name] = value
        config = cls(**kwargs)
        config.finalize()
        return config