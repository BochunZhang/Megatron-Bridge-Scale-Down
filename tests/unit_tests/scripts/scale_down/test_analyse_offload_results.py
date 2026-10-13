# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the selective activation-offload analyser."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
EXAMPLE_DIR = (
    REPO_ROOT / "examples/scale-down/01-analyse/01-fine-grained-offload/02-offload-on-dense-or-expert-model"
)
ANALYSER = EXAMPLE_DIR / "analyse_offload_results.py"


def _write_run(
    results_root: Path,
    *,
    model: str = "qwen35_text_9b",
    gpu_count: int = 4,
    pp: int = 1,
    ep: int = 1,
    mbs: int = 1,
    gbs: int = 32,
    dispatcher: str = "default",
    sharding: str = "fsdp1",
    run_name: str,
    tflops: float,
    run_time: str = "20260101-000000",
    profile: str = "none",
    status: int = 0,
    samples: int = 10,
) -> None:
    run_dir = (
        results_root
        / model
        / f"gpu_{gpu_count}-bf16-pp_{pp}-ep_{ep}-mbs_{mbs}-gbs_{gbs}-{dispatcher}-{run_name}"
        / run_time
    )
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "model": model,
                "dtype": "bf16",
                "gpu_count": gpu_count,
                "pp": pp,
                "ep": ep,
                "gbs": gbs,
                "mbs": mbs,
                "dispatcher": dispatcher,
                "sharding": sharding,
                "run_name": run_name,
                "run_time": run_time,
                "profile": profile,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(json.dumps({"status": status}), encoding="utf-8")
    (run_dir / "gpu_utilization.json").write_text(
        json.dumps({str(step): tflops for step in range(1, samples + 1)}),
        encoding="utf-8",
    )


def _run_analyser(results_root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(ANALYSER),
            "--results-root",
            str(results_root),
            "--output",
            str(results_root / "summary.csv"),
            "--samples-output",
            str(results_root / "samples.csv"),
            *args,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def _read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.unit
def test_analyser_keeps_different_gpu_counts_in_separate_groups(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    for gpu_count, baseline, offload in ((4, 100.0, 95.0), (16, 80.0, 60.0)):
        _write_run(results_root, gpu_count=gpu_count, run_name="baseline", tflops=baseline)
        _write_run(results_root, gpu_count=gpu_count, run_name="offload-mlp", tflops=offload)

    result = _run_analyser(results_root)
    assert result.returncode == 0, result.stderr

    rows = _read_rows(results_root / "summary.csv")
    # Two groups, each with its own baseline; a mixed pairing would show a drop
    # computed across two different launch scales.
    assert len(rows) == 4
    by_scale = {}
    for row in rows:
        by_scale.setdefault(row["gpu_count"], {})[row["run_name"]] = float(row["vs baseline"])
    assert by_scale["4"]["baseline"] == 0.0
    assert by_scale["16"]["baseline"] == 0.0
    assert by_scale["4"]["offload-mlp"] == pytest.approx(0.05)
    assert by_scale["16"]["offload-mlp"] == pytest.approx(0.25)


@pytest.mark.unit
def test_analyser_averages_only_the_final_iterations(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    run_dir = results_root / "qwen35_text_9b/gpu_4-bf16-pp_1-ep_1-mbs_1-gbs_32-default-baseline/20260101-000000"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "model": "qwen35_text_9b",
                "dtype": "bf16",
                "gpu_count": 4,
                "pp": 1,
                "ep": 1,
                "gbs": 32,
                "mbs": 1,
                "dispatcher": "default",
                "sharding": "fsdp1",
                "run_name": "baseline",
                "run_time": "20260101-000000",
                "profile": "none",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(json.dumps({"status": 0}), encoding="utf-8")
    # Warmup steps are deliberately slow so a whole-run average would differ.
    (run_dir / "gpu_utilization.json").write_text(
        json.dumps({str(step): (10.0 if step <= 6 else 100.0) for step in range(1, 11)}),
        encoding="utf-8",
    )
    _write_run(results_root, run_name="offload-mlp", tflops=90.0)

    result = _run_analyser(results_root)
    assert result.returncode == 0, result.stderr

    samples = _read_rows(results_root / "samples.csv")
    baseline = next(row for row in samples if row["run_name"] == "baseline")
    assert float(baseline["mean TFlops/GPU"]) == pytest.approx(100.0)
    assert [float(baseline[f"iteration {index} (TFlops)"]) for index in range(1, 5)] == [100.0] * 4


@pytest.mark.unit
def test_analyser_skips_profiled_and_failed_runs(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    _write_run(results_root, run_name="baseline", tflops=100.0)
    _write_run(results_root, run_name="offload-mlp", tflops=95.0)
    _write_run(results_root, run_name="offload-attn-mlp", tflops=90.0, profile="nsys")
    _write_run(results_root, run_name="offload-mlp", tflops=1.0, run_time="20260102-000000", status=1)

    result = _run_analyser(results_root)
    assert result.returncode == 0, result.stderr

    rows = _read_rows(results_root / "summary.csv")
    assert {row["run_name"] for row in rows} == {"baseline", "offload-mlp"}
    # The failed 20260102 run must not be selected as the newest offload run.
    assert all(row["gpu_count"] == "4" for row in rows)


@pytest.mark.unit
def test_analyser_rejects_a_group_without_a_baseline(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    _write_run(results_root, run_name="offload-mlp", tflops=95.0)

    result = _run_analyser(results_root)
    assert result.returncode != 0
    assert "no baseline run" in result.stderr


@pytest.mark.unit
def test_analyser_requires_at_least_ten_samples(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    _write_run(results_root, run_name="baseline", tflops=100.0, samples=9)
    _write_run(results_root, run_name="offload-mlp", tflops=95.0)

    result = _run_analyser(results_root)
    assert result.returncode != 0
    assert "at least 10 are required" in result.stderr
