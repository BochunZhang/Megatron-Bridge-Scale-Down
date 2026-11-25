import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest


BASELINE_DIR = Path(__file__).resolve().parents[4] / "examples/scale-down/02-layer-by-layer/01-baseline"
SWEEP = BASELINE_DIR.parent / "sweep.py"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("case", "parameter_tier", "activation_tier", "optimizer_tier"),
    [
        ("01_param_gpu_act_gpu_opt_gpu", "gpu", "gpu", "gpu"),
        ("02_param_cpu_act_gpu_opt_gpu", "cpu", "gpu", "gpu"),
        ("03_param_cpu_act_cpu_opt_gpu", "cpu", "cpu", "gpu"),
        ("04_param_cpu_act_cpu_opt_cpu", "cpu", "cpu", "cpu"),
    ],
)
def test_layer_by_layer_baseline_dry_run(
    tmp_path: Path, case: str, parameter_tier: str, activation_tier: str, optimizer_tier: str
) -> None:
    script = BASELINE_DIR / f"{case}.sh"
    results_root = tmp_path / "results with spaces"
    args = ["bash", str(script), "--output-dir", str(results_root)]
    env = {**os.environ, "PYTHON": sys.executable, "NSYS_BIN": "/example/nsys"}
    result = subprocess.run([*args, "--dry-run"], env=env, check=True, capture_output=True, text=True)
    command = shlex.split(result.stderr.strip())
    assert {
        "/example/nsys",
        "--sample=none",
        "--cpuctxsw=process-tree",
        "--gpu-metrics-devices=none",
        "--nproc-per-node=4",
        "model.num_layers=8",
        "model.linear_attention_freq=1",
        "dataset.seq_length=4096",
        "train.micro_batch_size=1",
        "train.global_batch_size=16",
        "train.train_iters=10",
        f"layer_by_layer.parameter_tier={parameter_tier}",
        f"layer_by_layer.activation_tier={activation_tier}",
        f"layer_by_layer.optimizer_tier={optimizer_tier}",
        "profiling.profile_step_start=7",
        "profiling.profile_step_end=8",
    }.issubset(command)
    if optimizer_tier == "cpu":
        assert {
            "--trace=cuda,nvtx",
            "profiling.nvtx_ranges=true",
            "layer_by_layer.cpu_optimizer=grace_adam",
            "OMP_NUM_THREADS=32",
            "layer_by_layer.cpu_threads=32",
        } <= set(command)
    assert all(not arg.startswith(("layer_by_layer.backend=", "layer_by_layer.nvme_dir=")) for arg in command)

    result = subprocess.run([*args, "--print-result-dir"], env=env, check=True, capture_output=True, text=True)
    result_path = Path(result.stdout.strip())
    assert result_path.parent.parent == results_root / "01-baseline" / case.replace("_", "-")
    assert result_path.parent.name.startswith("model_27b-lbl-schedule-")
    report = Path(next(arg.removeprefix("--output=") for arg in command if arg.startswith("--output=")))
    assert report.parent == result_path
    assert report.name.endswith(f"-param_{parameter_tier}-act_{activation_tier}-opt_{optimizer_tier}")
    assert not results_root.exists()


@pytest.mark.unit
def test_layer_by_layer_baseline_grace_adam_cpu_threads() -> None:
    script = BASELINE_DIR / "04_param_cpu_act_cpu_opt_cpu.sh"
    result = subprocess.run(
        ["bash", str(script), "--dry-run", "--cpu-threads", "12"],
        env={**os.environ, "PYTHON": sys.executable},
        check=True,
        capture_output=True,
        text=True,
    )
    assert {"OMP_NUM_THREADS=12", "layer_by_layer.cpu_threads=12"} <= set(shlex.split(result.stderr))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("extra_args", "expected"),
    [
        (
            [],
            {
                "layer_by_layer.parameter_tier=gpu",
                "layer_by_layer.activation_tier=gpu",
                "layer_by_layer.optimizer_tier=gpu",
            },
        ),
        (
            ["--parameter-tier", "cpu"],
            {
                "layer_by_layer.parameter_tier=cpu",
                "layer_by_layer.activation_tier=gpu",
                "layer_by_layer.optimizer_tier=gpu",
            },
        ),
        (
            ["--parameter-tier", "cpu", "--activation-tier", "cpu_act"],
            {
                "layer_by_layer.parameter_tier=cpu",
                "layer_by_layer.activation_tier=cpu",
                "layer_by_layer.optimizer_tier=gpu",
            },
        ),
        (
            ["--optimizer-tier", "cpu"],
            {
                "layer_by_layer.parameter_tier=gpu",
                "layer_by_layer.activation_tier=gpu",
                "layer_by_layer.optimizer_tier=cpu",
                "layer_by_layer.cpu_optimizer=torch",
            },
        ),
    ],
)
def test_layer_by_layer_sweep_placements(extra_args: list[str], expected: set[str]) -> None:
    result = subprocess.run(
        [sys.executable, str(SWEEP), "--dry-run", *extra_args], capture_output=True, text=True, check=True
    )
    command = shlex.split(result.stderr)
    assert expected <= set(command)


@pytest.mark.unit
def test_layer_by_layer_sweep_nvme_uses_case_scratch(tmp_path: Path) -> None:
    nvme = tmp_path / "nvme with spaces"
    nvme.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            str(SWEEP),
            "--dry-run",
            "--parameter-tier",
            "nvme",
            "--optimizer-tier",
            "nvme",
            "--nvme",
            str(nvme),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    command = shlex.split(result.stderr)
    assert {"layer_by_layer.parameter_tier=nvme", "layer_by_layer.optimizer_tier=nvme"} <= set(command)
    scratch = next(
        arg.removeprefix("layer_by_layer.nvme_dir=") for arg in command if arg.startswith("layer_by_layer.nvme_dir=")
    )
    assert Path(json.loads(scratch)).parent.resolve() == nvme.resolve()
    assert not any(nvme.iterdir())


@pytest.mark.unit
@pytest.mark.parametrize(
    "extra_args",
    [
        ["--activation-tier", "nvme"],
        ["--parameter-tier", "nvme"],
        ["--optimizer-tier", "nvme"],
        ["--parameter-tier", "nvme", "--optimizer-tier", "nvme"],
    ],
)
def test_layer_by_layer_sweep_rejects_unsupported_placements(extra_args: list[str]) -> None:
    result = subprocess.run([sys.executable, str(SWEEP), "--dry-run", *extra_args], capture_output=True, text=True)
    assert result.returncode == 2


@pytest.mark.unit
@pytest.mark.parametrize(
    ("parameter_tier", "activation_tier", "fsdp_result", "expected_status"),
    [
        ("gpu", "gpu", {}, "passed"),
        ("cpu", "gpu", {}, "passed"),
        ("cpu", "cpu", {}, "passed"),
        ("cpu", "cpu", {"optimizer_tier": "cpu", "fsdp_layer_by_layer_cpu_optimizer": True}, "passed"),
        ("cpu", "cpu", {"optimizer_tier": "cpu", "optimizer_backend": "torch"}, "passed"),
        ("cpu", "cpu", {"optimizer_tier": "cpu", "fsdp_layer_by_layer_cpu_optimizer": False}, "invalid_result"),
        (
            "cpu",
            "cpu",
            {
                "optimizer_tier": "cpu",
                "fsdp_layer_by_layer_cpu_optimizer": True,
                "fsdp_cpu_grad_accumulation": True,
            },
            "invalid_result",
        ),
        ("cpu", "cpu", {"activation_tier": "gpu"}, "invalid_result"),
        ("cpu", "gpu", {"schedule": "no_pipeline"}, "invalid_result"),
        ("cpu", "gpu", {"parameter_backend": "layer_by_layer"}, "invalid_result"),
        ("cpu", "gpu", {"fsdp_strategy": "optim_grads"}, "invalid_result"),
        ("cpu", "gpu", {"fsdp_cpu_offload": []}, "invalid_result"),
        ("cpu", "gpu", {"steps": 9}, "invalid_result"),
        ("cpu", "gpu", {"world_size": 1}, "invalid_result"),
    ],
)
def test_baseline_result_checks_native_fsdp_metrics(
    tmp_path: Path, parameter_tier: str, activation_tier: str, fsdp_result: dict, expected_status: str
) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    optimizer_tier = fsdp_result.get("optimizer_tier", "gpu")
    cpu_optimizer = fsdp_result.get("optimizer_backend", "grace_adam") if optimizer_tier == "cpu" else None
    expected_offload = ["weight"] if parameter_tier == "cpu" else []
    if cpu_optimizer:
        expected_offload.extend(("gradient", "optimizer"))
    shared = {
        "layers": 8,
        "steps": 10,
        "parameter_tier": parameter_tier,
        "activation_tier": activation_tier,
        "optimizer_tier": optimizer_tier,
    }
    state = {
        **shared,
        "results": str(tmp_path),
        "optimizer_backend": cpu_optimizer,
        "profile": "none",
        "nsys_version": None,
        "worker_overrides": {"GLOO_SOCKET_IFNAME": "eth0", "NCCL_SOCKET_IFNAME": "=eth0"},
    }
    (state_dir / "case.json").write_text(json.dumps(state), encoding="utf-8")
    metrics = {
        **shared,
        "loss": 7.0,
        "peak_allocated_bytes": 1024,
        "peak_reserved_bytes": 2048,
        "schedule": "layer_by_layer",
        "parameter_backend": "megatron_fsdp",
        "fsdp_strategy": "optim_grads_params",
        "fsdp_cpu_offload": expected_offload,
        "fsdp_cpu_grad_accumulation": cpu_optimizer == "torch",
        "optimizer_backend": cpu_optimizer or "FusedAdam",
        "world_size": 4,
        **fsdp_result,
    }
    (tmp_path / "train.log").write_text(f"LAYER_BY_LAYER_RESULT {json.dumps(metrics)}\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(SWEEP), "--finish-case", str(state_dir), "0"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == int(expected_status != "passed"), result.stderr
    with (tmp_path / "summary.csv").open(newline="", encoding="utf-8") as summary:
        row = next(csv.DictReader(summary))
    assert row["status"] == expected_status
    if expected_status == "passed":
        assert row["activation_tier"] == activation_tier
        assert row["optimizer_backend"] == (cpu_optimizer or "FusedAdam")
        assert row["fsdp_cpu_offload"] == ",".join(sorted(expected_offload))
        assert row["peak_allocated_bytes"] == "1024"
