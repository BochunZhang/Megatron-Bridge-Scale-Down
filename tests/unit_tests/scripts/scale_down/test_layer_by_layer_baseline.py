import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest


BASELINE_DIR = Path(__file__).resolve().parents[4] / "examples/scale-down/02-layer-by-layer/01-baseline"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("case", "parameter_tier"),
    [("01_param_gpu_act_gpu_opt_gpu", "gpu"), ("02_param_cpu_act_gpu_opt_gpu", "cpu")],
)
def test_layer_by_layer_baseline_dry_run(tmp_path: Path, case: str, parameter_tier: str) -> None:
    script = BASELINE_DIR / f"{case}.sh"
    results_root = tmp_path / "results with spaces"
    args = ["bash", str(script), "--output-dir", str(results_root), "--nvme", str(tmp_path)]
    env = {**os.environ, "PYTHON": sys.executable, "NSYS_BIN": "/example/nsys"}
    result = subprocess.run([*args, "--dry-run"], env=env, check=True, capture_output=True, text=True)
    command = shlex.split(result.stderr.strip())
    assert {
        "/example/nsys",
        "--gpu-metrics-devices=none",
        "--nproc-per-node=4",
        "model.num_layers=8",
        "model.linear_attention_freq=1",
        "dataset.seq_length=4096",
        "train.micro_batch_size=1",
        "train.global_batch_size=16",
        "train.train_iters=10",
        "layer_by_layer.backend=offload",
        f"layer_by_layer.parameter_tier={parameter_tier}",
        "layer_by_layer.activation_tier=gpu",
        "layer_by_layer.optimizer_tier=gpu",
        "profiling.profile_step_start=7",
        "profiling.profile_step_end=8",
    }.issubset(command)

    result = subprocess.run([*args, "--print-result-dir"], env=env, check=True, capture_output=True, text=True)
    result_path = Path(result.stdout.strip())
    assert result_path.parent.parent == results_root / "01-baseline" / case.replace("_", "-")
    report = Path(next(arg.removeprefix("--output=") for arg in command if arg.startswith("--output=")))
    assert report.parent == result_path
    assert report.name.endswith(f"-param_{parameter_tier}-act_gpu-opt_gpu")
    assert not results_root.exists()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("fsdp_result", "expected_status"),
    [
        ({}, "passed"),
        ({"parameter_backend": "layer_by_layer"}, "invalid_result"),
        ({"parameter_gpu_prefetch_slots": 1}, "invalid_result"),
        ({"parameter_h2d_prefetched_units_by_rank": [190, 190, 190, 189]}, "invalid_result"),
    ],
)
def test_param_cpu_result_requires_fsdp_prefetch(tmp_path: Path, fsdp_result: dict, expected_status: str) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    shared = {
        "layers": 8,
        "steps": 10,
        "parameter_tier": "cpu",
        "activation_tier": "gpu",
        "optimizer_tier": "gpu",
    }
    state = {
        **shared,
        "results": str(tmp_path),
        "microbatches": 4,
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
        "activation_d2h_completed_by_rank": [0] * 4,
        "activation_h2d_completed_by_rank": [0] * 4,
        "parameter_backend": "megatron_fsdp",
        "parameter_gpu_prefetch_slots": 2,
        "parameter_h2d_prefetched_units_by_rank": [190] * 4,
        **fsdp_result,
    }
    (tmp_path / "train.log").write_text(f"LAYER_BY_LAYER_RESULT {json.dumps(metrics)}\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(BASELINE_DIR.parent / "sweep.py"), "--finish-case", str(state_dir), "0"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == int(expected_status != "passed"), result.stderr
    with (tmp_path / "summary.csv").open(newline="", encoding="utf-8") as summary:
        row = next(csv.DictReader(summary))
    assert row["status"] == expected_status
    if expected_status == "passed":
        assert row["parameter_gpu_prefetch_slots"] == "2"
        assert row["parameter_h2d_prefetched_units_per_rank"] == "190"
