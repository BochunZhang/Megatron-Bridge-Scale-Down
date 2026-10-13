# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the selective activation-offload matrix entry point."""

import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
EXAMPLE_DIR = (
    REPO_ROOT / "examples/scale-down/01-analyse/01-fine-grained-offload/02-offload-on-dense-or-expert-model"
)
BENCHMARK_SCRIPT = EXAMPLE_DIR / "benchmark_offload.sh"
RUN_SCRIPT = EXAMPLE_DIR / "run_pretrain.sh"


def _dry_run(*args: str) -> list[str]:
    result = subprocess.run(
        ["bash", str(BENCHMARK_SCRIPT), "--dry-run", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line.startswith("matrix ")]


@pytest.mark.unit
def test_group_a_pairs_every_baseline_with_an_offload_case() -> None:
    lines = _dry_run("--group", "a", "--model-family", "qwen", "--mbs", "1")

    # Dense proxy plus two expert dispatchers, three cases each.
    assert len(lines) == 3 + 6 + 2  # +2 for the HybridEP sync A/B runs
    for prefix in (
        "group=a model=qwen35_text_9b gpu=4 pp=1 ep=1 dp=4 mbs=1 gbs=32 dispatcher=default",
        "group=a model=qwen35_text_35b_a3b gpu=4 pp=1 ep=4 dp=1 mbs=1 gbs=32 dispatcher=alltoall",
        "group=a model=qwen35_text_35b_a3b gpu=4 pp=1 ep=4 dp=1 mbs=1 gbs=32 dispatcher=hybridep",
    ):
        matching = [line for line in lines if line.startswith(prefix)]
        # The HybridEP group also carries the sync-sm baseline, hence the extra
        # offload-free run there.
        expected_offload_free = 2 if "dispatcher=hybridep" in prefix else 1
        assert sum("offload=false " in line for line in matching) == expected_offload_free
        assert sum("run=baseline " in line for line in matching) == expected_offload_free
        assert sum("run=offload" in line for line in matching) == 2


@pytest.mark.unit
def test_group_a_sync_sm_cases_only_run_for_hybridep() -> None:
    lines = _dry_run("--group", "a", "--model-family", "qwen", "--mbs", "1")

    sync_lines = [line for line in lines if "sync_sm=true" in line]
    assert len(sync_lines) == 2
    assert all("dispatcher=hybridep" in line for line in sync_lines)
    assert {line.split("run=")[1].split(" ")[0] for line in sync_lines} == {
        "baseline-sync-sm",
        "offload-mlp-sync-sm",
    }


@pytest.mark.unit
def test_group_a_expert_modules_include_the_moe_groups() -> None:
    lines = _dry_run("--group", "a", "--model-family", "qwen", "--mbs", "1")

    expert_mlp = [line for line in lines if "dispatcher=hybridep" in line and "run=offload-mlp " in line]
    assert len(expert_mlp) == 1
    assert "modules=[mlp_norm,expert_fc1,moe_act]" in expert_mlp[0]

    # Qwen3.5 uses GatedDeltaNet, which has no qkv_linear offload group.
    expert_attn = [line for line in lines if "dispatcher=hybridep" in line and "run=offload-attn-mlp " in line]
    assert len(expert_attn) == 1
    assert "modules=[mlp_norm,expert_fc1,moe_act,attn_norm,core_attn,attn_proj]" in expert_attn[0]
    assert "qkv_linear" not in expert_attn[0]

    dense_attn = [line for line in lines if "model=qwen35_text_9b" in line and "run=offload-attn-mlp " in line]
    assert "modules=[mlp_norm,mlp_act,attn_norm,core_attn,attn_proj]" in dense_attn[0]


@pytest.mark.unit
def test_group_a_deepseek_keeps_qkv_linear_and_moe_layer_freq() -> None:
    lines = _dry_run("--group", "a", "--model-family", "deepseek", "--mbs", "1")

    assert all("model=deepseek_v3 " in line for line in lines)
    assert len(lines) == 3 + 3 + 3  # dense proxy + two dispatchers
    assert sum("dispatcher=default " in line for line in lines) == 3
    assert sum("dispatcher=alltoall " in line for line in lines) == 3
    assert sum("dispatcher=hybridep " in line for line in lines) == 3

    deepseek_attn = [line for line in lines if "run=offload-attn-mlp " in line]
    assert all("qkv_linear" in line for line in deepseek_attn)
    # No sync-sm A/B for DeepSeek in group a.
    assert not any("sync_sm=true" in line for line in lines)


@pytest.mark.unit
def test_group_b_runs_full_model_on_fsdp1() -> None:
    lines = _dry_run("--group", "b", "--mbs", "1")

    assert len(lines) == 3
    assert all("model=qwen35_text_35b_a3b " in line for line in lines)
    assert all("gpu=16 " in line and "pp=1 " in line and "ep=16 " in line and "dp=1 " in line for line in lines)
    assert all("sharding=fsdp1 " in line for line in lines)
    assert all("gbs=128 " in line for line in lines)
    # No layer or expert cropping: the 16-GPU group runs the complete model.
    assert not any("--num-layers" in line for line in lines)


@pytest.mark.unit
def test_group_c_runs_pipeline_parallel_without_fsdp() -> None:
    lines = _dry_run("--group", "c", "--mbs", "1")

    assert len(lines) == 3
    assert all("gpu=32 " in line and "pp=2 " in line and "ep=16 " in line for line in lines)
    assert all("sharding=none " in line for line in lines)


@pytest.mark.unit
def test_benchmark_rejects_an_unknown_group() -> None:
    result = subprocess.run(
        ["bash", str(BENCHMARK_SCRIPT), "--dry-run", "--group", "d"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "--group must be a, b, c or all" in result.stderr


@pytest.mark.unit
def test_run_pretrain_rejects_gbs_that_does_not_divide_by_mbs_times_dp() -> None:
    result = subprocess.run(
        [
            "bash",
            str(RUN_SCRIPT),
            "--model",
            "qwen35_text_9b",
            "--run-name",
            "baseline",
            "--offload",
            "false",
            "--gpu",
            "16",
            "--mbs",
            "4",
            "--gbs",
            "32",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "must be divisible by MBS*DP" in result.stderr


@pytest.mark.unit
def test_run_pretrain_rejects_offload_without_a_module_list() -> None:
    result = subprocess.run(
        [
            "bash",
            str(RUN_SCRIPT),
            "--model",
            "qwen35_text_9b",
            "--run-name",
            "offload-mlp",
            "--offload",
            "true",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "--offload-modules is required" in result.stderr


@pytest.mark.unit
def test_run_pretrain_rejects_a_run_name_the_analyser_cannot_pair() -> None:
    result = subprocess.run(
        [
            "bash",
            str(RUN_SCRIPT),
            "--model",
            "qwen35_text_9b",
            "--run-name",
            "offload-off",
            "--offload",
            "false",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    # "offload-off" starts with "offload", so it is accepted; a name that starts
    # with neither prefix must be rejected.
    assert result.returncode == 0

    rejected = subprocess.run(
        [
            "bash",
            str(RUN_SCRIPT),
            "--model",
            "qwen35_text_9b",
            "--run-name",
            "measure-mlp",
            "--offload",
            "false",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode == 2
    assert "must start with baseline or offload" in rejected.stderr


@pytest.mark.unit
def test_run_pretrain_resolves_multi_node_metadata() -> None:
    result = subprocess.run(
        [
            "bash",
            str(RUN_SCRIPT),
            "--model",
            "qwen35_text_35b_a3b",
            "--run-name",
            "baseline",
            "--offload",
            "false",
            "--gpu",
            "32",
            "--pp",
            "2",
            "--ep",
            "16",
            "--sharding",
            "none",
            "--mbs",
            "1",
            "--gbs",
            "32",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "nnodes=8 node_rank=0 pp=2 ep=16 dp=1" in result.stdout
    assert "sharding=none" in result.stdout
    assert "qwen35_text_35b_a3b_pretrain_4gpu_gb200_bf16_fsdp1_config" in result.stdout


@pytest.mark.unit
def test_run_pretrain_rejects_localhost_master_addr_on_multi_node() -> None:
    result = subprocess.run(
        [
            "bash",
            str(RUN_SCRIPT),
            "--model",
            "qwen35_text_9b",
            "--run-name",
            "baseline",
            "--offload",
            "false",
            "--gpu",
            "16",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "MASTER_ADDR": "localhost"},
    )
    assert result.returncode == 2
    assert "MASTER_ADDR must not be localhost" in result.stderr
