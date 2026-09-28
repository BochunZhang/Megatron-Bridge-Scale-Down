import json
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
EXAMPLE_DIR = REPO_ROOT / "examples/scale-down/01-analyse/01-fine-grained-offload/02-offload-on-dense-or-expert-model"
BENCHMARK_SCRIPT = EXAMPLE_DIR / "benchmark_mlp_offload.sh"
COLLECTOR_SCRIPT = EXAMPLE_DIR / "collect_mlp_offload_results.mjs"


@pytest.mark.unit
def test_benchmark_dry_run_covers_dense_and_expert_matrix() -> None:
    result = subprocess.run(
        ["bash", str(BENCHMARK_SCRIPT), "--dry-run"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    matrix_lines = [line for line in result.stdout.splitlines() if line.startswith("matrix ")]
    assert "dtype=bf16" in result.stdout
    assert "train_iters=10 runs_per_case=1 micro_batch_sizes=1,2,4,8" in result.stdout
    assert "results/01-analyse/01-offload-on-dense-and-expert-model" in result.stdout
    assert len(matrix_lines) == 24
    assert sum("model=qwen35_text_9b " in line for line in matrix_lines) == 8
    assert sum("dispatcher=alltoall " in line for line in matrix_lines) == 8
    assert sum("dispatcher=hybridep " in line for line in matrix_lines) == 8
    assert sum("case=offload " in line for line in matrix_lines) == 12
    assert all("dtype=bf16 " in line for line in matrix_lines)
    assert sum("layers=16 experts=64" in line for line in matrix_lines) == 16
    for micro_batch_size in (1, 2, 4, 8):
        assert sum(f"mbs={micro_batch_size} " in line for line in matrix_lines) == 6
    assert all("recompute=null" in line for line in matrix_lines)


@pytest.mark.unit
def test_benchmark_maps_mxfp8_dtype_to_fp8mx_recipe() -> None:
    result = subprocess.run(
        [
            "bash",
            str(BENCHMARK_SCRIPT),
            "--dry-run",
            "--scope",
            "dense",
            "--micro-batch-size",
            "1",
            "--dtype",
            "mxfp8",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    matrix_lines = [line for line in result.stdout.splitlines() if line.startswith("matrix ")]
    assert len(matrix_lines) == 2
    assert all("dtype=mxfp8 " in line for line in matrix_lines)
    assert all("_fp8mx_fsdp1_config " in line for line in matrix_lines)


@pytest.mark.unit
def test_benchmark_dry_run_supports_deepseek_v3_expert_matrix() -> None:
    result = subprocess.run(
        ["bash", str(BENCHMARK_SCRIPT), "--dry-run", "--model", "deepseek"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    matrix_lines = [line for line in result.stdout.splitlines() if line.startswith("matrix ")]
    assert "model=deepseek " in result.stdout
    assert len(matrix_lines) == 16
    assert all("model=deepseek_v3 " in line for line in matrix_lines)
    assert all("recipe=deepseek_v3_pretrain_4gpu_gb200_bf16_fsdp1_config " in line for line in matrix_lines)
    assert all("layers=8 experts=64" in line for line in matrix_lines)
    assert sum("dispatcher=alltoall " in line for line in matrix_lines) == 8
    assert sum("dispatcher=hybridep " in line for line in matrix_lines) == 8
    assert sum("case=baseline " in line for line in matrix_lines) == 8
    assert sum("case=offload " in line for line in matrix_lines) == 8


@pytest.mark.unit
def test_benchmark_rejects_dense_scope_for_deepseek_v3() -> None:
    result = subprocess.run(
        ["bash", str(BENCHMARK_SCRIPT), "--dry-run", "--model", "deepseek", "--scope", "dense"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "DeepSeek-V3 is an expert model" in result.stderr


def _write_run(
    results_root: Path,
    *,
    run_time: str,
    run_name: str,
    step_times_ms: tuple[float, ...],
    micro_batch_size: int,
    profile: str = "none",
) -> None:
    run_dir = results_root / "qwen35_text_35b_a3b" / "bf16" / run_name / run_time
    run_dir.mkdir(parents=True)
    dispatcher = run_name.split("-")[1]
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "model": "qwen35_text_35b_a3b",
                "dtype": "bf16",
                "profile": profile,
                "run_name": run_name,
                "run_time": run_time,
                "dispatcher": dispatcher,
                "global_batch_size": 8,
                "micro_batch_size": micro_batch_size,
                "sequence_length": 4096,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(json.dumps({"status": 0}), encoding="utf-8")
    (run_dir / "train.log").write_text(
        "".join(
            f"iteration {iteration:8d}/      10 | elapsed time per iteration (ms): {step_time_ms:.1f} |\n"
            for iteration, step_time_ms in enumerate(step_times_ms, start=1)
        ),
        encoding="utf-8",
    )


@pytest.mark.unit
@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required by the XLSX collector")
def test_collector_uses_iterations_5_to_9_and_ignores_profiled_runs(tmp_path: Path) -> None:
    run_time = "collector-test"
    results_root = tmp_path / "results"
    _write_run(
        results_root,
        run_time=run_time,
        run_name="expert-alltoall-baseline-mbs1-r01",
        step_times_ms=(1000.0,) * 10,
        micro_batch_size=1,
    )
    _write_run(
        results_root,
        run_time=run_time,
        run_name="expert-alltoall-offload-mbs1-r01",
        step_times_ms=(2000.0,) * 10,
        micro_batch_size=1,
    )
    _write_run(
        results_root,
        run_time=run_time,
        run_name="expert-hybridep-baseline-mbs1-r01",
        step_times_ms=(500.0,) * 10,
        micro_batch_size=1,
        profile="torch",
    )

    result = subprocess.run(
        [
            "node",
            str(COLLECTOR_SCRIPT),
            "--results-root",
            str(results_root),
            "--run-time",
            run_time,
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["runTime"] == run_time
    assert len(payload["runs"]) == 2
    assert {run["caseName"] for run in payload["runs"]} == {"baseline", "offload"}
    baseline = next(run for run in payload["runs"] if run["caseName"] == "baseline")
    assert baseline["dtype"] == "bf16"
    assert baseline["microBatchSize"] == 1
    assert [sample["iteration"] for sample in baseline["samples"]] == [5, 6, 7, 8, 9]
    assert [sample["tokensPerSecond"] for sample in baseline["samples"]] == pytest.approx([32768.0] * 5)
