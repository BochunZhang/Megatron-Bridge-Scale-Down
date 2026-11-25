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
import subprocess
import sys
import tempfile
from importlib.util import find_spec
from pathlib import Path

from launch_network import resolve_network

_LOG = logging.getLogger(__name__)
_REPO = Path(__file__).resolve().parents[3]
_WORLD_SIZE = 4
_NSYS_VERSION = "2026.6.1"
_GPU_METRICS_HZ = 20000
_EXAMPLE_DIR = "examples/scale-down/02-layer-by-layer"
_BRIDGE_SOURCE_FILES = (
    f"{_EXAMPLE_DIR}/run_sweep.sh",
    f"{_EXAMPLE_DIR}/sweep.py",
    f"{_EXAMPLE_DIR}/launch_network.py",
)
_LM_SOURCE_FILES = (
    "megatron/core/distributed/layer_by_layer.py",
    "megatron/core/distributed/layer_by_layer_storage.py",
    "megatron/training/layer_by_layer.py",
    "megatron/training/layer_by_layer_workload.py",
)
_CASE_SCRIPTS = tuple(
    f"{_EXAMPLE_DIR}/01-baseline/{name}"
    for name in ("01_cpu_offload.sh", "02_nvme_offload.sh", "offload_common.sh")
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
    "activation_tier",
    "status",
    "returncode",
    "steps",
    "loss",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
    "activation_d2h_completed_per_rank",
    "activation_h2d_completed_per_rank",
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
    driver = _module_source("megatron.training.layer_by_layer")
    trainer = _module_source("megatron.core.distributed.layer_by_layer")
    if driver is not None and trainer is not None:
        root = driver.parents[2]
        if trainer == root / _LM_SOURCE_FILES[0] and all(
            (root / source).is_file() for source in _LM_SOURCE_FILES
        ):
            return root
    training_source = driver or _module_source("megatron.training")
    core_source = trainer or _module_source("megatron.core")
    raise RuntimeError(
        "The Bridge Python environment does not expose the complete layer-by-layer trainer.\n"
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
        "--nvme-dir", "--nvme", type=Path, required=True, help="existing directory on local NVMe"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="new output directory (default: unique directory under ./results)",
    )
    parser.add_argument(
        "--layers", type=_positive, nargs="+", default=[16, 32, 64], help="depths to attempt"
    )
    parser.add_argument("--seq-length", type=_positive, default=4096)
    parser.add_argument("--micro-batch-size", type=_positive, default=1)
    parser.add_argument(
        "--microbatches", type=_positive, default=4, help="micro-batches per rank per step"
    )
    parser.add_argument("--steps", type=_positive, default=3)
    parser.add_argument(
        "--parameter-tier", choices=("cpu_param", "nvme_param"), default="nvme_param"
    )
    parser.add_argument(
        "--activation-tier", choices=("gpu_act", "cpu_act", "nvme_act"), default="nvme_act"
    )
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--cpu-threads", type=_positive)
    parser.add_argument("--profile", choices=("none", "nsys"), default="none")
    parser.add_argument("--profile-start", type=int, help="first profiled step, zero-based")
    parser.add_argument("--profile-end", type=int, help="exclusive end of the capture")
    parser.add_argument("--tiny", action="store_true", help="use the small smoke model")
    parser.add_argument(
        "--single-case-layout", action="store_true", help="use node-level files for one depth"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="show commands without writing or launching"
    )
    args = parser.parse_args()
    if len(set(args.layers)) != len(args.layers):
        parser.error("--layers must contain distinct values")
    if args.single_case_layout and (len(args.layers) != 1 or args.results_dir is None):
        parser.error("--single-case-layout requires exactly one depth and --results-dir")
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
    return results / ("train.nsys-rep" if args.single_case_layout else f"layers-{layers}.nsys-rep")


def _command(args: argparse.Namespace, layers: int, scratch: Path, report: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={_WORLD_SIZE}",
        "-m",
        "megatron.training.layer_by_layer",
        "--num-layers",
        str(layers),
        "--seq-length",
        str(args.seq_length),
        "--micro-batch-size",
        str(args.micro_batch_size),
        "--microbatches",
        str(args.microbatches),
        "--steps",
        str(args.steps),
        "--parameter-tier",
        args.parameter_tier,
        "--activation-tier",
        args.activation_tier,
        "--nvme-dir",
        str(scratch),
    ]
    if args.tiny:
        command.append("--tiny")
    if args.learning_rate is not None:
        command.extend(("--learning-rate", str(args.learning_rate)))
    if args.cpu_threads is not None:
        command.extend(("--cpu-threads", str(args.cpu_threads)))
    if args.profile == "nsys":
        command.extend(
            ("--profile-start", str(args.profile_start), "--profile-end", str(args.profile_end))
        )
        devices = "cuda-visible" if "CUDA_VISIBLE_DEVICES" in os.environ else "all"
        command = [
            _nsys_binary(),
            "profile",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--cpuctxsw=none",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            f"--output={report.with_suffix('')}",
            f"--gpu-metrics-devices={devices}",
            f"--gpu-metrics-frequency={_GPU_METRICS_HZ}",
            *command,
        ]
    return command


def _read_result(
    log_path: Path,
    returncode: int,
    layers: int,
    *,
    steps: int,
    microbatches: int,
    activation_tier: str,
) -> dict:
    result = {
        "layers": layers,
        "activation_tier": activation_tier,
        "status": "failed",
        "returncode": returncode,
        "log": log_path.name,
    }
    resource_errors = set()
    records = []
    with log_path.open(encoding="utf-8") as log:
        for line in log:
            if line.startswith(_RESULT_PREFIX):
                try:
                    records.append(json.loads(line[len(_RESULT_PREFIX) :]))
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
    d2h = metrics.get("activation_d2h_completed_by_rank")
    h2d = metrics.get("activation_h2d_completed_by_rank")
    expected_d2h = 0 if activation_tier == "gpu_act" else (layers + 1) * microbatches * steps
    if (
        metrics.get("layers") == layers
        and metrics.get("steps") == steps
        and metrics.get("activation_tier") == activation_tier
        and d2h == [expected_d2h] * _WORLD_SIZE
        and h2d == [2 * expected_d2h] * _WORLD_SIZE
        and all(type(count) is int for count in (*d2h, *h2d))
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
            activation_d2h_completed_per_rank=d2h[0],
            activation_h2d_completed_per_rank=h2d[0],
        )
    return result


def _write_launch(
    args: argparse.Namespace,
    results: Path,
    nvme_dir: Path,
    worker_overrides: dict[str, str],
    nsys_version: str | None,
    source_metadata: dict[str, str | None],
) -> None:
    launch = {
        "model": "Qwen/Qwen3.5-27B",
        "dtype": "bf16",
        "num_layers": args.layers[0],
        "tiny": args.tiny,
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
        "parameter_tier": args.parameter_tier,
        "activation_tier": args.activation_tier,
        "nvme_path": str(nvme_dir),
        "profile": args.profile,
        "profile_start": args.profile_start,
        "profile_end": args.profile_end,
        "nsys_version": nsys_version,
        "gpu_metrics_hz": _GPU_METRICS_HZ if nsys_version else None,
        "gloo_socket_ifname": worker_overrides["GLOO_SOCKET_IFNAME"],
        "nccl_socket_ifname": worker_overrides["NCCL_SOCKET_IFNAME"],
        "cuda_graph": "off",
        **source_metadata,
    }
    (results / "launch.json").write_text(json.dumps(launch, indent=2) + "\n", encoding="utf-8")


def _run_case(
    args: argparse.Namespace,
    layers: int,
    results: Path,
    nvme_dir: Path,
    worker_overrides: dict[str, str],
) -> dict:
    log_name, command_name = (
        ("train.log", "command.txt")
        if args.single_case_layout
        else (f"layers-{layers}.log", f"layers-{layers}.command.txt")
    )
    log_path = results / log_name
    report = _profile_report(args, layers, results)
    with tempfile.TemporaryDirectory(prefix=f"layer-by-layer-{layers}-", dir=nvme_dir) as scratch:
        command = _command(args, layers, Path(scratch), report)
        rendered = command
        if worker_overrides:
            rendered = [
                "env",
                *(f"{key}={value}" for key, value in worker_overrides.items()),
                *command,
            ]
        (results / command_name).write_text(shlex.join(rendered) + "\n", encoding="utf-8")
        _LOG.info("Running %d layers. Log at %s", layers, log_path)
        with log_path.open("w", encoding="utf-8") as log:
            returncode = subprocess.run(
                command,
                cwd=_REPO,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=dict(os.environ, PYTHONUNBUFFERED="1", **worker_overrides),
                check=False,
            ).returncode
    case = _read_result(
        log_path,
        returncode,
        layers,
        steps=args.steps,
        microbatches=args.microbatches,
        activation_tier=args.activation_tier,
    )
    if args.profile == "nsys" and case["status"] == "passed":
        if not report.is_file() or report.stat().st_size == 0:
            _LOG.error("Missing or empty Nsight Systems report: %s", report)
            case["status"] = "invalid_profile"
    return case


def main() -> int:
    """Run selected depths and report the largest completed case."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _arguments()
    nvme_dir = args.nvme_dir.absolute()
    if args.dry_run:
        results = args.results_dir or Path("results/02-layer-by-layer-<unique>")
        for layers in args.layers:
            scratch = nvme_dir / f"layer-by-layer-{layers}-<unique>"
            report = _profile_report(args, layers, results.absolute())
            _LOG.info("%s", shlex.join(_command(args, layers, scratch, report)))
        return 0
    if not nvme_dir.is_dir():
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
            _LOG.info("%d layers: %s", layers, case["status"])
            if case["status"] == "passed":
                largest = layers if largest is None else max(largest, layers)
            elif case["status"] not in _RESOURCE_ERRORS:
                had_error = True
    _LOG.info("Largest successful attempted depth: %s. Results at %s", largest, results)
    return int(had_error or largest is None)


if __name__ == "__main__":
    sys.exit(main())
