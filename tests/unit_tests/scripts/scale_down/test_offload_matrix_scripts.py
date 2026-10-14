# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Dry-run tests for the selective activation-offload wrapper scripts."""

import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
EXAMPLE_DIR = (
    REPO_ROOT / "examples/scale-down/01-analyse/01-fine-grained-offload/02-activation-offload-on-dense-and-expert-model"
)
BLOCKING = EXAMPLE_DIR / "run_4gpu_expert_activation_offload_blocking.sh"
COMPARE_4GPU = EXAMPLE_DIR / "run_4gpu_compare_activation_offload_influence_on_dense_and_expert.sh"
COMPARE_8GPU = EXAMPLE_DIR / "run_8gpu_compare_activation_offload_influence_on_dense_and_expert.sh"
COMPARE_16GPU = EXAMPLE_DIR / "run_16gpu_compare_activation_offload_influence_on_dense_and_expert.sh"
COMPARE_32GPU = EXAMPLE_DIR / "run_32gpu_compare_activation_offload_influence_on_dense_and_expert.sh"
RUN_SCRIPT = EXAMPLE_DIR / "run_pretrain.sh"

# Strip DLC variables from the caller so the tests see a clean single-node shell.
CLEAN_ENV = {
    key: value
    for key, value in os.environ.items()
    if key not in {"RANK", "NODE_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "RUN_TIME", "LOCAL_HF_HOME"}
}


def _run(script: Path, *args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(script), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={**CLEAN_ENV, **(env or {})},
    )


def _matrix(script: Path, *args: str) -> list[str]:
    result = _run(script, "--dry-run", *args)
    assert result.returncode == 0, result.stderr
    return [line for line in result.stdout.splitlines() if line.startswith("matrix ")]


@pytest.mark.unit
def test_blocking_check_runs_the_same_case_with_and_without_sm_sync() -> None:
    lines = _matrix(BLOCKING)

    assert len(lines) == 2
    assert all("model=qwen35_text_35b_a3b type=expert gpu=4 pp=1 ep=4" in line for line in lines)
    assert all("dispatcher=hybridep" in line and "profile=nsys" in line for line in lines)
    assert all("modules=[mlp_norm,expert_fc1,moe_act,attn_norm,core_attn,attn_proj]" in line for line in lines)
    assert "run=offload-attn-mlp-blocking" in lines[0] and "sync_sm=false" in lines[0]
    assert "run=offload-attn-mlp-sync-sm" in lines[1] and "sync_sm=true" in lines[1]
    assert all("--num-layers 16 --num-experts 64" in line for line in lines)


@pytest.mark.unit
def test_blocking_check_supports_the_deepseek_expert_proxy() -> None:
    lines = _matrix(BLOCKING, "--model-family", "deepseek")

    assert len(lines) == 2
    assert all("model=deepseek_v3 type=expert gpu=4 pp=1 ep=4" in line for line in lines)
    assert all("dispatcher=hybridep" in line and "profile=nsys" in line for line in lines)
    assert all(
        "modules=[mlp_norm,expert_fc1,moe_act,attn_norm,qkv_linear,core_attn,attn_proj]" in line for line in lines
    )
    assert "run=offload-attn-mlp-blocking" in lines[0] and "sync_sm=false" in lines[0]
    assert "run=offload-attn-mlp-sync-sm" in lines[1] and "sync_sm=true" in lines[1]
    assert all("extra=[--num-layers 4 --num-experts 32 --moe-layer-freq [1,1,1,1]]" in line for line in lines)


@pytest.mark.unit
def test_4gpu_compare_covers_three_cases_two_mbs_and_both_model_types() -> None:
    lines = _matrix(COMPARE_4GPU)

    assert len(lines) == 2 * 2 * 3
    for mbs in (1, 2):
        for model_type in ("dense", "expert"):
            group = [line for line in lines if f"type={model_type} " in line and f" mbs={mbs} " in line]
            assert {line.split("run=")[1].split(" ")[0] for line in group} == {
                "baseline",
                "offload-mlp",
                "offload-attn-mlp",
            }
    dense = [line for line in lines if "type=dense " in line]
    assert all("model=qwen35_text_9b" in line and "ep=1 " in line and "sync_sm=false" in line for line in dense)
    # The 16-layer crop must override linear_attention_freq too: the Qwen3.5 HF
    # config carries a 32-entry layer_types list, which MCore rejects against a
    # 16-layer model.
    assert all("--num-layers 16 --linear-attention-freq 4 --mtp-num-layers 0" in line for line in dense)
    expert = [line for line in lines if "type=expert " in line]
    assert all("ep=4 " in line and "dispatcher=hybridep" in line and "sync_sm=true" in line for line in expert)


@pytest.mark.unit
def test_4gpu_compare_supports_deepseek_proxies() -> None:
    lines = _matrix(COMPARE_4GPU, "--model-family", "deepseek", "--mbs", "1")

    assert len(lines) == 6
    assert all("model=deepseek_v3 " in line for line in lines)
    assert all("[0,0,0,0]" in line for line in lines if "type=dense " in line)
    assert all("[1,1,1,1]" in line for line in lines if "type=expert " in line)
    assert all("qkv_linear" in line for line in lines if "run=offload-attn-mlp " in line)


@pytest.mark.unit
def test_8gpu_compare_halves_the_16gpu_experts_with_ep8() -> None:
    lines = _matrix(COMPARE_8GPU)

    assert len(lines) == 12
    assert all("gpu=8 pp=1 " in line and "gbs=64 " in line for line in lines)
    dense = [line for line in lines if "type=dense " in line]
    assert all("model=qwen35_text_9b " in line and "ep=1 " in line for line in dense)
    assert not any("--num-layers" in line or "--num-experts" in line for line in dense)
    expert = [line for line in lines if "type=expert " in line]
    assert all("model=qwen35_text_35b_a3b " in line and "ep=8 " in line for line in expert)
    assert all("dispatcher=hybridep" in line and "sync_sm=true" in line for line in expert)
    assert all("extra=[--num-experts 128]" in line for line in expert)


@pytest.mark.unit
def test_8gpu_run_pretrain_resolves_two_nodes_and_128_experts() -> None:
    result = _run(
        COMPARE_8GPU,
        "--dry-run",
        "--mbs",
        "2",
        "--model-type",
        "expert",
        "--case",
        "baseline",
        "--run-time",
        "shared-id",
        env={"RANK": "1", "WORLD_SIZE": "2", "MASTER_ADDR": "dlc-master", "MASTER_PORT": "23456"},
    )
    assert result.returncode == 0, result.stderr
    assert "nnodes=2 node_rank=1 master=dlc-master:23457 pp=1 ep=8 dp=8" in result.stdout
    assert "model.num_moe_experts=128" in result.stdout


@pytest.mark.unit
def test_16gpu_compare_uses_complete_models_and_ep16() -> None:
    lines = _matrix(COMPARE_16GPU)

    assert len(lines) == 12
    assert all("gpu=16 pp=1 " in line and "gbs=128 " in line for line in lines)
    assert all("model=qwen35_text_9b" in line and "ep=1 " in line for line in lines if "type=dense " in line)
    assert all(
        "model=qwen35_text_35b_a3b" in line and "ep=16 " in line for line in lines if "type=expert " in line
    )
    assert not any("--num-layers" in line for line in lines)


@pytest.mark.unit
def test_16gpu_compare_can_be_split_across_jobs() -> None:
    lines = _matrix(COMPARE_16GPU, "--mbs", "1", "--model-type", "expert", "--case", "baseline,offload-mlp")

    assert len(lines) == 2
    assert all(" mbs=1 " in line and "type=expert " in line for line in lines)


@pytest.mark.unit
def test_32gpu_compare_uses_pp4_ep8_and_defaults_to_mbs1() -> None:
    lines = _matrix(COMPARE_32GPU)

    assert len(lines) == 6
    assert all("gpu=32 pp=4 " in line and " mbs=1 " in line for line in lines)
    assert all("model=qwen35_text_27b" in line and "ep=1 " in line for line in lines if "type=dense " in line)
    assert all(
        "model=qwen35_text_122b_a10b" in line and "ep=8 " in line for line in lines if "type=expert " in line
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "script",
    [BLOCKING, COMPARE_4GPU, COMPARE_8GPU, COMPARE_16GPU],
    ids=lambda path: path.stem,
)
def test_wrapper_defaults_to_mxfp8_and_accepts_a_bf16_override(script: Path) -> None:
    default = _run(script, "--dry-run", "--mbs", "1")
    assert default.returncode == 0, default.stderr
    assert " dtype=mxfp8" in default.stdout
    assert "_fp8mx_fsdp1_config" in default.stdout
    assert "_bf16_fsdp1_config" not in default.stdout

    override = _run(script, "--dry-run", "--mbs", "1", "--dtype", "bf16")
    assert override.returncode == 0, override.stderr
    assert " dtype=bf16" in override.stdout
    assert "_bf16_fsdp1_config" in override.stdout
    assert "_fp8mx_fsdp1_config" not in override.stdout


@pytest.mark.unit
def test_32gpu_compare_defaults_to_bf16_for_the_122b_expert() -> None:
    result = _run(COMPARE_32GPU, "--dry-run", "--mbs", "1")
    assert result.returncode == 0, result.stderr
    assert " dtype=bf16" in result.stdout
    assert "_fp8mx_fsdp1_config" not in result.stdout


@pytest.mark.unit
def test_32gpu_compare_rejects_mxfp8_for_the_122b_expert() -> None:
    result = _run(COMPARE_32GPU, "--dry-run", "--dtype", "mxfp8")
    assert result.returncode == 2
    assert "only has a bf16 recipe" in result.stderr


@pytest.mark.unit
def test_multi_node_wrapper_rejects_a_mismatched_dlc_world_size() -> None:
    result = _run(COMPARE_16GPU, "--dry-run", env={"WORLD_SIZE": "8"})
    assert result.returncode == 2
    assert "WORLD_SIZE=8 nodes, but --gpu 16 needs 4 nodes" in result.stderr


@pytest.mark.unit
def test_run_pretrain_uses_dlc_node_rank_for_the_result_directory() -> None:
    result = _run(
        RUN_SCRIPT,
        "--experiment",
        "manual_check",
        "--model",
        "qwen35_text_35b_a3b",
        "--model-type",
        "expert",
        "--run-name",
        "baseline",
        "--offload",
        "false",
        "--gpu",
        "32",
        "--pp",
        "4",
        "--ep",
        "8",
        "--mbs",
        "1",
        "--gbs",
        "128",
        "--run-time",
        "shared-id",
        "--dry-run",
        env={"RANK": "3", "WORLD_SIZE": "8", "MASTER_ADDR": "dlc-master", "MASTER_PORT": "23456"},
    )
    assert result.returncode == 0, result.stderr
    assert "nnodes=8 node_rank=3 master=dlc-master:23457 pp=4 ep=8 dp=8" in result.stdout
    assert "/manual_check/qwen35_text_35b_a3b-expert/" in result.stdout
    # Node 3 writes its node-local files under the master's run directory.
    assert "/shared-id/rank3/rank_logs" in result.stdout


@pytest.mark.unit
def test_run_pretrain_rejects_gbs_that_does_not_divide_by_mbs_times_dp() -> None:
    result = _run(
        RUN_SCRIPT,
        "--experiment",
        "manual_check",
        "--model",
        "qwen35_text_9b",
        "--model-type",
        "dense",
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
    )
    assert result.returncode == 2
    assert "must be divisible by MBS*DP" in result.stderr


@pytest.mark.unit
def test_run_pretrain_rejects_expert_parallelism_on_a_dense_model() -> None:
    result = _run(
        RUN_SCRIPT,
        "--experiment",
        "manual_check",
        "--model",
        "qwen35_text_9b",
        "--model-type",
        "dense",
        "--run-name",
        "baseline",
        "--offload",
        "false",
        "--ep",
        "4",
        "--dry-run",
    )
    assert result.returncode == 2
    assert "dense models have no experts" in result.stderr


@pytest.mark.unit
def test_run_pretrain_rejects_sm_sync_without_hybridep() -> None:
    result = _run(
        RUN_SCRIPT,
        "--experiment",
        "manual_check",
        "--model",
        "qwen35_text_35b_a3b",
        "--model-type",
        "expert",
        "--run-name",
        "baseline",
        "--offload",
        "false",
        "--ep",
        "4",
        "--dispatcher",
        "alltoall",
        "--sync-with-sm",
        "true",
        "--dry-run",
    )
    assert result.returncode == 2
    assert "requires --dispatcher hybridep" in result.stderr


@pytest.mark.unit
@pytest.mark.parametrize(
    "script",
    [BLOCKING, COMPARE_4GPU, COMPARE_8GPU, COMPARE_16GPU, COMPARE_32GPU],
    ids=lambda path: path.stem,
)
def test_wrapper_uses_its_name_as_the_first_result_directory(script: Path) -> None:
    result = _run(script, "--dry-run", "--mbs", "1")
    assert result.returncode == 0, result.stderr

    experiment = script.stem.removeprefix("run_")
    root = "results/01-analyse/01-fine-grained-offload/02-activation-offload-on-dense-and-expert-model"
    assert f"experiment={experiment} " in result.stdout
    results = [line for line in result.stdout.splitlines() if line.startswith("resolved sharding=")]
    assert results
    assert all(f"{root}/{experiment}/" in line for line in results)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("script", "repos"),
    [
        (COMPARE_4GPU, None),
        (COMPARE_8GPU, "Qwen/Qwen3.5-9B-Base Qwen/Qwen3.5-35B-A3B-Base"),
        (COMPARE_16GPU, "Qwen/Qwen3.5-9B-Base Qwen/Qwen3.5-35B-A3B-Base"),
        (COMPARE_32GPU, "Qwen/Qwen3.5-27B Qwen/Qwen3.5-122B-A10B"),
    ],
    ids=lambda value: value.stem if isinstance(value, Path) else None,
)
def test_multi_node_wrappers_stage_a_node_local_hf_home(script: Path, repos: str | None) -> None:
    result = _run(script, "--dry-run", "--mbs", "1")
    assert result.returncode == 0, result.stderr

    staged = [line for line in result.stdout.splitlines() if line.startswith("local hf home: ")]
    if repos is None:
        assert staged == []
    else:
        assert staged == [f"local hf home: /tmp/megatron-bridge-hf-home repos=[{repos}]"]


@pytest.mark.unit
def test_run_pretrain_requires_an_experiment_name() -> None:
    result = _run(
        RUN_SCRIPT, "--model", "qwen35_text_9b", "--model-type", "dense", "--run-name", "baseline",
        "--offload", "false", "--dry-run",
    )
    assert result.returncode == 2
    assert "--experiment is required" in result.stderr


@pytest.mark.unit
def test_run_pretrain_defaults_to_the_mxfp8_recipe() -> None:
    result = _run(
        RUN_SCRIPT, "--experiment", "manual_check", "--model", "qwen35_text_9b", "--model-type", "dense",
        "--run-name", "baseline", "--offload", "false", "--dry-run",
    )
    assert result.returncode == 0, result.stderr
    assert "dtype=mxfp8 recipe=qwen35_text_9b_pretrain_4gpu_gb200_fp8mx_fsdp1_config" in result.stdout
