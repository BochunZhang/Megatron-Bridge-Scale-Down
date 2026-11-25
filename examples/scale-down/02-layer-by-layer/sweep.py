# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Sweep transformer depth on a four-GPU layer-by-layer training node."""

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from contextlib import nullcontext
from importlib.util import find_spec
from pathlib import Path

from launch_network import resolve_network


_LOG = logging.getLogger(__name__)
_REPO = Path(__file__).resolve().parents[3]
_WORLD_SIZE = 4
_NSYS_VERSION = "2026.6.1"
_PLACEMENTS = ("gpu", "cpu", "nvme")
_EXAMPLE_DIR = "examples/scale-down/02-layer-by-layer"
_RECIPE = "qwen35_text_27b_pretrain_4gpu_gb200_bf16_layer_by_layer_config"
_RECIPE_RUNNER = "scripts/training/run_recipe.py"
_LM_FSDP_SOURCE = "megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py"
_BRIDGE_SOURCE_FILES = (
    f"{_EXAMPLE_DIR}/run_sweep.sh",
    f"{_EXAMPLE_DIR}/run_pretrain_layer_by_layer.sh",
    f"{_EXAMPLE_DIR}/sweep.py",
    f"{_EXAMPLE_DIR}/launch_network.py",
    _RECIPE_RUNNER,
    "scripts/training/recipe_runner.py",
    "src/megatron/bridge/recipes/qwen/__init__.py",
    "src/megatron/bridge/recipes/qwen/gb200/__init__.py",
    "src/megatron/bridge/recipes/qwen/gb200/qwen35.py",
    "src/megatron/bridge/recipes/qwen/qwen35.py",
    "src/megatron/bridge/training/config.py",
    "src/megatron/bridge/training/layer_by_layer.py",
    "src/megatron/bridge/training/pretrain.py",
    "src/megatron/bridge/training/profiling.py",
    "src/megatron/bridge/training/setup.py",
    "src/megatron/bridge/training/train.py",
    "src/megatron/bridge/training/utils/flop_utils.py",
    "src/megatron/bridge/training/utils/train_utils.py",
)
_LM_SOURCE_FILES = (
    "megatron/core/distributed/distributed_data_parallel_config.py",
    "megatron/core/distributed/fsdp/mcore_fsdp_adapter.py",
    "megatron/core/distributed/fsdp/src/megatron_fsdp/distributed_data_parallel_config.py",
    _LM_FSDP_SOURCE,
    "megatron/core/distributed/fsdp/src/megatron_fsdp/offload.py",
    "megatron/core/distributed/fsdp/src/megatron_fsdp/param_and_grad_buffer.py",
    "megatron/core/distributed/fsdp/src/megatron_fsdp/utils.py",
    "megatron/core/models/gpt/gpt_model.py",
    "megatron/core/optimizer/__init__.py",
    "megatron/core/optimizer/clip_grads.py",
    "megatron/core/optimizer/cpu_offloading/fsdp_adam.py",
    "megatron/core/optimizer/distrib_optimizer.py",
    "megatron/core/optimizer/optimizer.py",
    "megatron/core/optimizer/optimizer_config.py",
    "megatron/core/pipeline_parallel/fine_grained_activation_offload.py",
    "megatron/core/pipeline_parallel/schedules.py",
    "megatron/core/tensor_parallel/random.py",
    "megatron/core/transformer/transformer_config.py",
    "megatron/training/layer_by_layer_workload.py",
)
_CASE_SCRIPTS = tuple(
    f"{_EXAMPLE_DIR}/01-baseline/{name}"
    for name in (
        "01_param_gpu_act_gpu_opt_gpu.sh",
        "02_param_cpu_act_gpu_opt_gpu.sh",
        "03_param_cpu_act_cpu_opt_gpu.sh",
        "04_param_cpu_act_cpu_opt_cpu.sh",
        "05_param_cpu_act_cpu_opt_cpu_overlap.sh",
        "offload_common.sh",
    )
)
_RESULT_PREFIX = "LAYER_BY_LAYER_RESULT "
_RESOURCE_ERRORS = {
    "gpu_oom": re.compile(
        r"cuda out of memory|cuda error: out of memory|cublas_status_alloc_failed", re.I
    ),
    "cpu_oom": re.compile(
        r"defaultcpuallocator: (?:can't|not enough memory)|std::bad_alloc|\bMemoryError\b", re.I
    ),
    "nvme_full": re.compile(r"no space left on device|disk quota exceeded", re.I),
}
_RESULT_COLUMNS = (
    "layers",
    "parameter_tier",
    "activation_tier",
    "optimizer_tier",
    "optimizer_backend",
    "overlap_cpu_optimizer",
    "status",
    "returncode",
    "steps",
    "loss",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
    "schedule",
    "parameter_backend",
    "fsdp_strategy",
    "fsdp_cpu_offload",
    "world_size",
    "log",
)


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def _module_source(module: str) -> Path | None:
    # Resolve the installed files without importing the GPU-dependent modules.
    parts = module.split(".")
    spec = find_spec(parts[0])
    for end in range(2, len(parts) + 1):
        if spec is None or spec.submodule_search_locations is None:
            return None
        fullname = ".".join(parts[:end])
        # An editable install can resolve a child via a meta path finder.
        search_paths = spec.submodule_search_locations
        spec = None
        for finder in sys.meta_path:
            find = getattr(finder, "find_spec", None)
            if find is not None:
                spec = find(fullname, search_paths, None)
                if spec is not None:
                    break
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin).resolve()


def _installed_megatron_lm() -> Path:
    workload = _module_source("megatron.training.layer_by_layer_workload")
    schedules = _module_source("megatron.core.pipeline_parallel.schedules")
    if workload is not None and schedules is not None:
        root = workload.parents[2]
        if (
            schedules == root / "megatron/core/pipeline_parallel/schedules.py"
            and all((root / source).is_file() for source in _LM_SOURCE_FILES)
            and "def layer_by_layer_forward_backward_no_pipelining(" in schedules.read_text(encoding="utf-8")
            and "def layer_by_layer_unit(" in (root / _LM_FSDP_SOURCE).read_text(encoding="utf-8")
        ):
            return root
    training_source = workload or _module_source("megatron.training")
    core_source = schedules or _module_source("megatron.core")
    raise RuntimeError(
        "The Bridge Python environment does not expose the native layer-by-layer schedule and Megatron-FSDP sources.\n"
        f"Python: {sys.executable}\n"
        f"Megatron training source: {training_source or 'missing'}\n"
        f"Megatron Core source: {core_source or 'missing'}\n"
        "Check out layer-by-layer in the Megatron-LM already installed in this environment."
    )


def _lm_revision(lm_dir: Path) -> str | None:
    revision = subprocess.run(
        ["git", "rev-parse", "--show-toplevel", "HEAD"],
        cwd=lm_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if revision.returncode == 0:
        top_level, head = revision.stdout.splitlines()
        if Path(top_level).resolve() == lm_dir:
            return head
    return None


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nvme-dir", "--nvme", type=Path, help="existing local NVMe directory for the NVMe weight and optimizer case"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="new output directory (default: unique directory under ./results)",
    )
    parser.add_argument(
        "--layers", type=_positive, nargs="+", default=[8], help="depths to attempt"
    )
    parser.add_argument("--seq-length", type=_positive, default=4096)
    parser.add_argument("--micro-batch-size", type=_positive, default=1)
    parser.add_argument(
        "--microbatches", type=_positive, default=4, help="micro-batches per rank per step"
    )
    parser.add_argument("--steps", type=_positive, default=3)
    parser.add_argument(
        "--parameter-tier", type=lambda tier: tier.removesuffix("_param"), choices=_PLACEMENTS, default="gpu"
    )
    parser.add_argument(
        "--activation-tier", type=lambda tier: tier.removesuffix("_act"), choices=("gpu", "cpu"), default="gpu"
    )
    parser.add_argument("--optimizer-tier", choices=_PLACEMENTS, default="gpu")
    parser.add_argument("--cpu-optimizer", choices=("torch", "grace_adam"), default="torch")
    parser.add_argument(
        "--overlap-cpu-optimizer", action="store_true", help="start GraceAdam when its CPU main gradients are ready"
    )
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--cpu-threads", type=_positive, default=32)
    parser.add_argument("--profile", choices=("none", "nsys"), default="none")
    parser.add_argument("--profile-start", type=int, help="first profiled step, zero-based")
    parser.add_argument("--profile-end", type=int, help="exclusive end of the capture")
    parser.add_argument("--tiny", action="store_true", help="use the small smoke model")
    parser.add_argument(
        "--single-case-layout", action="store_true", help="use node-level files for one depth"
    )
    parser.add_argument("--prepare-case", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--dry-run", action="store_true", help="show commands without writing or launching"
    )
    args = parser.parse_args()
    if "nvme" in (args.parameter_tier, args.optimizer_tier):
        if (args.parameter_tier, args.optimizer_tier) != ("nvme", "nvme"):
            parser.error("NVMe requires --parameter-tier nvme and --optimizer-tier nvme")
        if args.nvme_dir is None:
            parser.error("NVMe requires --nvme-dir")
    args.optimizer_backend = None if args.optimizer_tier == "gpu" else args.cpu_optimizer
    if args.overlap_cpu_optimizer and (args.parameter_tier, args.optimizer_tier, args.optimizer_backend) != (
        "cpu",
        "cpu",
        "grace_adam",
    ):
        parser.error(
            "--overlap-cpu-optimizer requires --parameter-tier cpu --optimizer-tier cpu --cpu-optimizer grace_adam"
        )
    if len(set(args.layers)) != len(args.layers):
        parser.error("--layers must contain distinct values")
    if args.single_case_layout and (len(args.layers) != 1 or args.results_dir is None):
        parser.error("--single-case-layout requires exactly one depth and --results-dir")
    if args.prepare_case and not args.single_case_layout:
        parser.error("--prepare-case requires --single-case-layout")
    if args.profile == "nsys":
        if args.profile_start is None:
            args.profile_start = min(7, args.steps - 1)
        if args.profile_end is None:
            args.profile_end = args.profile_start + 1
        if not 0 <= args.profile_start < args.profile_end <= args.steps:
            parser.error("profiling requires 0 <= profile-start < profile-end <= steps")
    elif args.profile_start is not None or args.profile_end is not None:
        parser.error("--profile-start and --profile-end require --profile nsys")
    return args


def _nsys_binary() -> str:
    return os.environ.get("NSYS_BIN") or str(_REPO / f"NsightSystems-{_NSYS_VERSION}/bin/nsys")


def _check_nsys_version() -> str:
    binary = _nsys_binary()
    try:
        version = subprocess.check_output(
            [binary, "--version"], cwd=_REPO, stderr=subprocess.STDOUT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"Cannot run {binary}. Set NSYS_BIN to Nsight Systems {_NSYS_VERSION}."
        ) from exc
    if not re.match(
        rf"^NVIDIA Nsight Systems version {re.escape(_NSYS_VERSION)}([.\s-]|$)", version
    ):
        raise RuntimeError(f"Expected Nsight Systems {_NSYS_VERSION} at {binary}. Got {version!r}.")
    return version


def _profile_report(args: argparse.Namespace, layers: int, results: Path) -> Path:
    model = "qwen35_tiny" if args.tiny else "qwen35_27b"
    seq = f"{args.seq_length // 1024}k" if args.seq_length % 1024 == 0 else str(args.seq_length)
    gbs = args.micro_batch_size * args.microbatches * _WORLD_SIZE
    overlap = "-opt_overlap" if args.overlap_cpu_optimizer else ""
    return results / (
        f"{model}-lbl-layers_{layers}-attn_full-bf16-cg_off-tp_1-cp_1-pp_1-dp_{_WORLD_SIZE}"
        f"-mbs_{args.micro_batch_size}-gbs_{gbs}-seq_{seq}-iters_{args.steps}"
        f"-param_{args.parameter_tier}-act_{args.activation_tier}-opt_{args.optimizer_tier}{overlap}.nsys-rep"
    )


def _numa_worker() -> tuple[str, list[str]]:
    if shutil.which("numarun"):
        return "numarun", ["numarun"]
    if shutil.which("numactl"):
        local_wrapper = _REPO / ".cache/numarun"
        if local_wrapper.is_file() and os.access(local_wrapper, os.R_OK):
            return "repo-numarun", ["bash", str(local_wrapper)]
        return "numactl", ["numactl", "--localalloc"]
    return "unbound", []


def _command(
    args: argparse.Namespace,
    layers: int,
    scratch: Path | None,
    report: Path,
    worker_overrides: dict[str, str] | None = None,
) -> list[str]:
    python = (
        [sys.executable] if os.environ.get("PYTHON") else ["uv", "run", "--project", str(_REPO), "--no-sync", "python"]
    )
    config_path = report.parent / ("config.yaml" if args.single_case_layout else f"layers-{layers}.config.yaml")
    tensorboard_dir = report.parent / ("tensorboard" if args.single_case_layout else f"layers-{layers}.tensorboard")
    _, numa_wrapper = _numa_worker()
    worker_env = []
    if args.optimizer_backend is not None:
        worker_env = ["env", f"OMP_NUM_THREADS={args.cpu_threads}"]
    command = [
        *python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={_WORLD_SIZE}",
        "--no-python",
        *numa_wrapper,
        *worker_env,
        *python,
        str(_REPO / _RECIPE_RUNNER),
        "--recipe",
        _RECIPE,
        "--mode",
        "pretrain",
        "--dataset",
        "mock",
        f"model.num_layers={layers}",
        "model.linear_attention_freq=1",
        f"dataset.seq_length={args.seq_length}",
        f"train.micro_batch_size={args.micro_batch_size}",
        f"train.global_batch_size={args.micro_batch_size * args.microbatches * _WORLD_SIZE}",
        f"train.train_iters={args.steps}",
        f"layer_by_layer.parameter_tier={args.parameter_tier}",
        f"layer_by_layer.activation_tier={args.activation_tier}",
        f"layer_by_layer.optimizer_tier={args.optimizer_tier}",
        "logger.log_interval=1",
        "logger.log_throughput=true",
        "logger.log_throughput_to_tensorboard=true",
        "logger.throughput_window_size=2",
        f"logger.tensorboard_dir={json.dumps(str(tensorboard_dir))}",
        f"logger.save_config_filepath={json.dumps(str(config_path))}",
    ]
    if args.tiny:
        command.append("layer_by_layer.tiny=true")
    if args.learning_rate is not None:
        command.append(f"optimizer.lr={args.learning_rate}")
    if args.optimizer_backend is not None:
        command.extend(
            (f"layer_by_layer.cpu_optimizer={args.cpu_optimizer}", f"layer_by_layer.cpu_threads={args.cpu_threads}")
        )
    if args.overlap_cpu_optimizer:
        command.append("layer_by_layer.overlap_cpu_optimizer=true")
    if scratch is not None:
        command.append(f"layer_by_layer.nvme_dir={json.dumps(str(scratch))}")
    if args.profile == "nsys":
        command.extend(
            (
                "profiling.use_nsys_profiler=true",
                f"profiling.profile_step_start={args.profile_start}",
                f"profiling.profile_step_end={args.profile_end}",
            )
        )
        if args.optimizer_tier == "cpu" and args.optimizer_backend == "grace_adam":
            command.append("profiling.nvtx_ranges=true")
        command = [
            _nsys_binary(),
            "profile",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--cpuctxsw=process-tree",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            f"--output={report.with_suffix('')}",
            "--gpu-metrics-devices=none",
            *command,
        ]
    return [
        "env",
        "-u",
        "NCCL_DEBUG_SUBSYS",
        "-u",
        "NCCL_DEBUG_FILE",
        "NCCL_DEBUG=WARN",
        "PYTHONUNBUFFERED=1",
        *(f"{key}={value}" for key, value in (worker_overrides or {}).items()),
        *command,
    ]


def _read_result(
    log_path: Path,
    returncode: int,
    layers: int,
    *,
    steps: int,
    parameter_tier: str,
    activation_tier: str,
    optimizer_tier: str,
    optimizer_backend: str | None,
    overlap_cpu_optimizer: bool,
) -> dict:
    result = {
        "layers": layers,
        "parameter_tier": parameter_tier,
        "activation_tier": activation_tier,
        "optimizer_tier": optimizer_tier,
        "optimizer_backend": optimizer_backend,
        "overlap_cpu_optimizer": overlap_cpu_optimizer,
        "status": "failed",
        "returncode": returncode,
        "log": log_path.name,
    }
    resource_errors = set()
    records = []
    with log_path.open(encoding="utf-8") as log:
        for line in log:
            _, marker, payload = line.partition(_RESULT_PREFIX)
            if marker:
                try:
                    records.append(json.loads(payload))
                except ValueError:
                    records.append(None)
            resource_errors.update(
                name for name, pattern in _RESOURCE_ERRORS.items() if pattern.search(line)
            )
    if returncode:
        if len(resource_errors) == 1:
            result["status"] = resource_errors.pop()
        return result
    result["status"] = "invalid_result"
    if len(records) != 1 or not isinstance(records[0], dict):
        return result
    metrics = records[0]
    loss = metrics.get("loss")
    allocated = metrics.get("peak_allocated_bytes")
    reserved = metrics.get("peak_reserved_bytes")
    actual_optimizer = metrics.get("optimizer_backend")
    offload = metrics.get("fsdp_cpu_offload")
    layerwise_grace_adam = optimizer_tier == "cpu" and optimizer_backend == "grace_adam"
    expected_cpu_accumulation = optimizer_tier != "gpu" and not layerwise_grace_adam
    expected_offload = []
    if optimizer_tier != "gpu":
        expected_offload.extend(("gradient", "optimizer"))
    if parameter_tier != "gpu":
        expected_offload.append("weight")
    if (
        metrics.get("schedule") == "layer_by_layer"
        and metrics.get("parameter_backend") == "megatron_fsdp"
        and metrics.get("fsdp_strategy") == "optim_grads_params"
        and metrics.get("world_size") == _WORLD_SIZE
        and isinstance(offload, list)
        and all(isinstance(component, str) for component in offload)
        and sorted(offload) == expected_offload
        and (optimizer_tier == "gpu" or metrics.get("fsdp_cpu_grad_accumulation") is expected_cpu_accumulation)
        and metrics.get("layers") == layers
        and metrics.get("steps") == steps
        and metrics.get("parameter_tier") == parameter_tier
        and metrics.get("activation_tier") == activation_tier
        and metrics.get("optimizer_tier") == optimizer_tier
        and isinstance(actual_optimizer, str)
        and bool(actual_optimizer.strip())
        and (optimizer_backend is None or actual_optimizer == optimizer_backend)
        and (not layerwise_grace_adam or metrics.get("fsdp_layer_by_layer_cpu_optimizer") is True)
        and metrics.get("fsdp_layer_by_layer_overlap_cpu_optimizer", False) is overlap_cpu_optimizer
        and (not overlap_cpu_optimizer or metrics.get("clip_grad") == 0.0)
        and type(loss) in (float, int)
        and math.isfinite(loss)
        and type(allocated) is int
        and type(reserved) is int
        and 0 < allocated <= reserved
    ):
        result.update(
            status="passed",
            steps=steps,
            loss=loss,
            peak_allocated_bytes=allocated,
            peak_reserved_bytes=reserved,
            optimizer_backend=actual_optimizer,
            schedule=metrics["schedule"],
            parameter_backend=metrics["parameter_backend"],
            fsdp_strategy=metrics["fsdp_strategy"],
            fsdp_cpu_offload=",".join(expected_offload),
            world_size=_WORLD_SIZE,
        )
    return result


def _write_launch(
    args: argparse.Namespace,
    results: Path,
    nvme_dir: Path | None,
    worker_overrides: dict[str, str],
    nsys_version: str | None,
    source_metadata: dict[str, str | None],
) -> None:
    launch = {
        "model": "Qwen/Qwen3.5-27B",
        "recipe": _RECIPE,
        "dtype": "bf16",
        "num_layers": args.layers[0],
        "attention": "full",
        "linear_attention_freq": 1,
        "tiny": args.tiny,
        "schedule": "layer_by_layer",
        "fully_sharded": True,
        "data_parallel_size": _WORLD_SIZE,
        "pipeline_model_parallel_size": 1,
        "tensor_model_parallel_size": 1,
        "context_parallel_size": 1,
        "seq_length": args.seq_length,
        "micro_batch_size": args.micro_batch_size,
        "global_batch_size": args.micro_batch_size * args.microbatches * _WORLD_SIZE,
        "num_microbatches": args.microbatches,
        "train_iters": args.steps,
        "gpu_count": _WORLD_SIZE,
        "nnodes": 1,
        "node_rank": 0,
        "numa_mode": _numa_worker()[0],
        "parameter_tier": args.parameter_tier,
        "activation_tier": args.activation_tier,
        "optimizer_tier": args.optimizer_tier,
        "optimizer_backend": args.optimizer_backend,
        "overlap_cpu_optimizer": args.overlap_cpu_optimizer,
        "cpu_threads": args.cpu_threads if args.optimizer_backend is not None else None,
        "nvme_path": str(nvme_dir) if nvme_dir is not None else None,
        "profile": args.profile,
        "profile_start": args.profile_start,
        "profile_end": args.profile_end,
        "nsys_version": nsys_version,
        "nsys_report": _profile_report(args, args.layers[0], results).name if nsys_version else None,
        "gpu_metrics_hz": None,
        "gloo_socket_ifname": worker_overrides["GLOO_SOCKET_IFNAME"],
        "nccl_socket_ifname": worker_overrides["NCCL_SOCKET_IFNAME"],
        "cuda_graph": "off",
        **source_metadata,
    }
    (results / "launch.json").write_text(json.dumps(launch, indent=2) + "\n", encoding="utf-8")


def _case_command(
    args: argparse.Namespace,
    layers: int,
    results: Path,
    scratch: Path | None,
    worker_overrides: dict[str, str],
) -> tuple[Path, list[str]]:
    log_name, command_name = (
        ("train.log", "command.txt")
        if args.single_case_layout
        else (f"layers-{layers}.log", f"layers-{layers}.command.txt")
    )
    log_path = results / log_name
    command = _command(args, layers, scratch, _profile_report(args, layers, results), worker_overrides)
    (results / command_name).write_text(shlex.join(command) + "\n", encoding="utf-8")
    _LOG.info("Running %d layers. Log at %s", layers, log_path)
    return log_path, command


def _run_case(
    args: argparse.Namespace,
    layers: int,
    results: Path,
    nvme_dir: Path | None,
    worker_overrides: dict[str, str],
) -> dict:
    scratch_dir = (
        tempfile.TemporaryDirectory(prefix=f"layer-by-layer-{layers}-", dir=nvme_dir)
        if nvme_dir is not None
        else nullcontext()
    )
    with scratch_dir as scratch:
        log_path, command = _case_command(args, layers, results, Path(scratch) if scratch else None, worker_overrides)
        with log_path.open("w", encoding="utf-8") as log:
            returncode = subprocess.run(
                command,
                cwd=_REPO,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            ).returncode
    return _collect_case(args, layers, results, log_path, returncode)


def _collect_case(args: argparse.Namespace, layers: int, results: Path, log_path: Path, returncode: int) -> dict:
    case = _read_result(
        log_path,
        returncode,
        layers,
        steps=args.steps,
        parameter_tier=args.parameter_tier,
        activation_tier=args.activation_tier,
        optimizer_tier=args.optimizer_tier,
        optimizer_backend=args.optimizer_backend,
        overlap_cpu_optimizer=args.overlap_cpu_optimizer,
    )
    if args.profile == "nsys" and case["status"] == "passed":
        report = _profile_report(args, layers, results)
        if not report.is_file() or report.stat().st_size == 0:
            _LOG.error("Missing or empty Nsight Systems report: %s", report)
            case["status"] = "invalid_profile"
    return case


def _write_run_info(results: Path, case: dict, worker_overrides: dict[str, str], nsys_version: str | None) -> None:
    status = 0 if case["status"] == "passed" else case["returncode"] or 1
    info = [
        f"case_status={case['status']}",
        f"returncode={case['returncode']}",
        f"gloo_socket_ifname={worker_overrides['GLOO_SOCKET_IFNAME']}",
        f"nccl_socket_ifname={worker_overrides['NCCL_SOCKET_IFNAME']}",
    ]
    if nsys_version:
        info.append(f"nsys_version={nsys_version}")
    info.append(f"status={status}")
    (results / "run_info.txt").write_text("\n".join(info) + "\n", encoding="utf-8")


def _prepare_case(
    args: argparse.Namespace,
    results: Path,
    nvme_dir: Path | None,
    worker_overrides: dict[str, str],
    nsys_version: str | None,
) -> None:
    """Hand one validated launch command to the test's Bash launcher."""
    layers = args.layers[0]
    state_dir = args.prepare_case
    scratch = None
    if nvme_dir is not None:
        scratch = Path(tempfile.mkdtemp(prefix=f"layer-by-layer-{layers}-", dir=nvme_dir))
        try:
            (state_dir / "scratch").write_text(f"{scratch}\n", encoding="utf-8")
        except OSError:
            scratch.rmdir()
            raise
    log_path, command = _case_command(args, layers, results, scratch, worker_overrides)
    state = {
        "results": str(results),
        "layers": layers,
        "steps": args.steps,
        "microbatches": args.microbatches,
        "micro_batch_size": args.micro_batch_size,
        "seq_length": args.seq_length,
        "tiny": args.tiny,
        "parameter_tier": args.parameter_tier,
        "activation_tier": args.activation_tier,
        "optimizer_tier": args.optimizer_tier,
        "optimizer_backend": args.optimizer_backend,
        "overlap_cpu_optimizer": args.overlap_cpu_optimizer,
        "profile": args.profile,
        "worker_overrides": worker_overrides,
        "nsys_version": nsys_version,
    }
    (state_dir / "case.json").write_text(json.dumps(state), encoding="utf-8")
    (state_dir / "command").write_bytes(b"".join(os.fsencode(word) + b"\0" for word in (str(log_path), *command)))


def _finish_case(state_dir: Path, returncode: int) -> int:
    """Validate the worker's output and write the numbered case's summary."""
    state = json.loads((state_dir / "case.json").read_text(encoding="utf-8"))
    state.setdefault("overlap_cpu_optimizer", False)
    args = argparse.Namespace(single_case_layout=True, **state)
    results = Path(args.results)
    case = _collect_case(args, args.layers, results, results / "train.log", returncode)
    with (results / "summary.csv").open("w", encoding="utf-8", newline="") as summary:
        writer = csv.DictWriter(summary, fieldnames=_RESULT_COLUMNS)
        writer.writeheader()
        writer.writerow(case)
    _write_run_info(results, case, args.worker_overrides, args.nsys_version)
    _LOG.info("%d layers: %s. Results at %s", args.layers, case["status"], results)
    return int(case["status"] != "passed")


def main() -> int:
    """Run selected depths and report the largest completed case."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if sys.argv[1:2] == ["--finish-case"]:
        return _finish_case(Path(sys.argv[2]), int(sys.argv[3]))
    args = _arguments()
    nvme_dir = args.nvme_dir.absolute() if args.parameter_tier == "nvme" else None
    if args.dry_run:
        results = args.results_dir or Path("results/02-layer-by-layer-<unique>")
        for layers in args.layers:
            scratch = nvme_dir / f"layer-by-layer-{layers}-<unique>" if nvme_dir is not None else None
            report = _profile_report(args, layers, results.absolute())
            _LOG.info("%s", shlex.join(_command(args, layers, scratch, report)))
        return 0
    if nvme_dir is not None and not nvme_dir.is_dir():
        _LOG.error("NVMe directory does not exist: %s", nvme_dir)
        return 2
    try:
        lm_dir = _installed_megatron_lm()
        nsys_version = _check_nsys_version() if args.profile == "nsys" else None
        worker_overrides = (
            resolve_network(os.environ) if args.single_case_layout or nsys_version else {}
        )
    except RuntimeError as exc:
        _LOG.error("%s", exc)
        return 2
    _LOG.info("Megatron-LM in the launch environment | %s", lm_dir)
    if worker_overrides:
        _LOG.info(
            "Socket interfaces | Gloo %s | NCCL %s",
            worker_overrides["GLOO_SOCKET_IFNAME"],
            worker_overrides["NCCL_SOCKET_IFNAME"],
        )
    if nsys_version:
        _LOG.info("%s", nsys_version)
        worker_overrides["NCCL_NVTX_DISABLE"] = "0"
    if args.results_dir is None:
        root = Path("results").resolve()
        root.mkdir(parents=True, exist_ok=True)
        results = Path(tempfile.mkdtemp(prefix="02-layer-by-layer-", dir=root))
    else:
        results = args.results_dir.resolve()
        results.mkdir(parents=True, exist_ok=False)

    bridge_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True
    ).strip()
    lm_revision = _lm_revision(lm_dir)
    (results / "revision.txt").write_text(bridge_revision + "\n", encoding="utf-8")
    (results / "megatron_lm_revision.txt").write_text(
        (lm_revision or "unavailable") + "\n", encoding="utf-8"
    )
    bridge_source_files = _BRIDGE_SOURCE_FILES
    if args.single_case_layout:
        _write_launch(
            args,
            results,
            nvme_dir,
            worker_overrides,
            nsys_version,
            {
                "bridge_dir": str(_REPO),
                "bridge_revision": bridge_revision,
                "megatron_lm_dir": str(lm_dir),
                "megatron_lm_revision": lm_revision,
            },
        )
        bridge_source_files += _CASE_SCRIPTS
    source_hashes = {
        f"{prefix}/{path}": hashlib.sha256((root / path).read_bytes()).hexdigest()
        for prefix, root, paths in (
            ("bridge", _REPO, bridge_source_files),
            ("megatron-lm", lm_dir, _LM_SOURCE_FILES),
        )
        for path in paths
    }
    (results / "source_sha256.json").write_text(
        json.dumps(source_hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.prepare_case:
        _prepare_case(args, results, nvme_dir, worker_overrides, nsys_version)
        return 0
    largest = None
    had_error = False
    with (results / "summary.csv").open("w", encoding="utf-8", newline="") as summary:
        writer = csv.DictWriter(summary, fieldnames=_RESULT_COLUMNS)
        writer.writeheader()
        for layers in args.layers:
            case = _run_case(args, layers, results, nvme_dir, worker_overrides)
            writer.writerow(case)
            summary.flush()
            if args.single_case_layout:
                _write_run_info(results, case, worker_overrides, nsys_version)
            _LOG.info("%d layers: %s", layers, case["status"])
            if case["status"] == "passed":
                largest = layers if largest is None else max(largest, layers)
            elif case["status"] not in _RESOURCE_ERRORS:
                had_error = True
    _LOG.info("Largest successful attempted depth: %s. Results at %s", largest, results)
    return int(had_error or largest is None)


if __name__ == "__main__":
    sys.exit(main())
