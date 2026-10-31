# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Dry-run and expert-halving tests for 01-analyse/04-fsdp3-comm."""

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
EXAMPLE = REPO_ROOT / "examples/scale-down/01-analyse/04-fsdp3-comm"
BENCHMARK = EXAMPLE / "benchmark_fsdp3.sh"
RUN_ONE = EXAMPLE / "run_pretrain_fsdp3.sh"
RUN_DATE = "261031-120000"


def _env(**extra: str) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in {"RUN_DATE", "MASTER_PORT", "RANK", "NODE_RANK"}}
    env.update(extra)
    return env


def _run(script: Path, *args: str, **env: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(script), *args], cwd=REPO_ROOT, capture_output=True, text=True, env=_env(**env))


def _commands(stderr: str) -> list[str]:
    return [line for line in stderr.splitlines() if line.startswith("command:")]


def test_default_sweep_runs_three_models_with_mtp_and_both_cuda_graph_modes(tmp_path: Path):
    result = _run(BENCHMARK, "--dry-run", "--run-date", RUN_DATE, "--output-dir", str(tmp_path))
    assert result.returncode == 0, result.stderr
    launches = [line for line in result.stderr.splitlines() if line.startswith("launch ")]
    assert len(launches) == 18
    commands = _commands(result.stderr)
    assert len(commands) == 18
    for command in commands:
        assert "model.num_layers=4" in command
        assert "model.mtp_num_layers=1" in command
        # Megatron-FSDP labels its collectives with layers when NVTX ranges are on.
        assert "profiling.nvtx_ranges=true" in command
        assert "fsdp_comm_nvtx" not in command

    dense_on = next(c for c in commands if "model_27b-" in c and "-cg_on-" in c)
    assert r"model.cuda_graph_modules=\[attn\,mlp\]" in dense_on
    assert "ddp.megatron_fsdp_cuda_graph_mode=true" in dense_on
    moe_on = next(c for c in commands if "model_397b-" in c and "-cg_on-" in c)
    assert r"model.cuda_graph_modules=\[attn\,moe_router\,moe_preprocess\]" in moe_on
    assert "num_moe_experts" not in moe_on  # 397b keeps every routed expert
    moe_off = next(c for c in commands if "model_397b-" in c and "-cg_off-" in c)
    assert "model.cuda_graph_impl=none" in moe_off
    big = [c for c in commands if "model_2p4t-" in c]
    assert big and all("model.num_moe_experts=512" in c for c in big)
    assert all("qwen38_text_2p4t_a95b_pretrain_4gpu_gb200_bf16_fsdp1_config" in c for c in big)


def test_2p4t_halves_experts_after_oom_and_keeps_both_modes_comparable(tmp_path: Path):
    # Stub launcher: fails with a CUDA OOM above 64 experts (or above 128 with CUDA graph off).
    log = tmp_path / "launches.txt"
    stub = tmp_path / "run_one.sh"
    stub.write_text(
        f"""#!/usr/bin/env bash
experts=0; cg=off; print=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-experts) experts="$2"; shift 2;;
        --cuda-graph) cg="$2"; shift 2;;
        --print-result-dir) print=true; shift;;
        *) shift;;
    esac
done
dir="{tmp_path}/res/e${{experts}}-${{cg}}"
if [[ "$print" == true ]]; then echo "$dir"; exit 0; fi
mkdir -p "$dir"; echo "${{experts}} ${{cg}}" >>"{log}"
limit=64; [[ "$cg" == off ]] && limit=128
if (( experts > limit )); then echo "torch.OutOfMemoryError: CUDA out of memory" >"$dir/train.log"; exit 1; fi
exit 0
"""
    )
    stub.chmod(0o755)
    result = _run(
        BENCHMARK,
        *("--model", "2p4t", "--cp", "1", "--seq", "4096", "--mbs", "1", "--profile", "none"),
        *("--run-date", RUN_DATE, "--output-dir", str(tmp_path / "out")),
        RUN_ONE=str(stub),
    )
    assert result.returncode == 0, result.stderr
    # CUDA graph on runs first; one OOM at a count reruns every mode at half the count.
    assert log.read_text().split("\n")[:-1] == ["512 on", "256 on", "128 on", "64 on", "64 off"]
    search = (tmp_path / "out" / f"expert_search-date_{RUN_DATE}-node0.tsv").read_text()
    assert "experts=64\toutcome=ok" in search
    assert "tried=512:oom,256:oom,128:oom,64:ok" in search


def test_multi_node_2p4t_requires_a_fixed_expert_count():
    result = _run(BENCHMARK, "--dry-run", "--model", "2p4t", "--gpus", "8", "--master-addr", "host")
    assert result.returncode == 2
    assert "--experts-2p4t" in result.stderr


def test_run_one_print_result_dir_encodes_shape_and_cuda_graph(tmp_path: Path):
    result = _run(
        RUN_ONE,
        *("--model", "2.4t", "--recipe", "r", "--seq-length", "4096", "--num-layers", "4"),
        *("--mtp-num-layers", "1", "--num-experts", "64", "--cuda-graph", "on"),
        *("--run-date", RUN_DATE, "--results-root", str(tmp_path), "--print-result-dir"),
    )
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()).parent.name.startswith(
        "model_2p4t-fsdp_3-layers_4-mtp_1-experts_64-cg_on-mbs_1-seq_4096-cp_1-ep_4-"
    )


def test_27b_fsdp1_vs_fsdp3_sweep_selects_sharding_and_overlap(tmp_path: Path):
    result = _run(
        EXAMPLE / "benchmark_27b_fsdp1_vs_fsdp3.sh",
        *("--dry-run", "--run-date", RUN_DATE, "--output-dir", str(tmp_path)),
    )
    assert result.returncode == 0, result.stderr
    commands = _commands(result.stderr)
    assert len(commands) == 6
    for command in commands:
        assert "model.num_layers=8" in command
        assert "model.mtp_num_layers=1" in command
        assert "model.cuda_graph_impl=none" in command
        assert "train.global_batch_size=16" in command
        assert "nsys profile" in command
    for seq in (256, 1024, 4096):
        fsdp3 = next(c for c in commands if f"model_27b-fsdp_3-layers_8-mtp_1-experts_0-cg_off-mbs_1-seq_{seq}-" in c)
        assert "ddp.data_parallel_sharding_strategy=optim_grads_params" in fsdp3
        assert "comm_overlap.overlap_param_gather=true" in fsdp3
        fsdp1 = next(c for c in commands if f"model_27b-fsdp_1-layers_8-mtp_1-experts_0-cg_off-mbs_1-seq_{seq}-" in c)
        assert "ddp.data_parallel_sharding_strategy=optim " in fsdp1
        assert "ddp.overlap_grad_reduce=false" in fsdp1
        assert "comm_overlap.overlap_param_gather=false" in fsdp1


def test_run_one_rejects_unknown_fsdp_level():
    result = _run(RUN_ONE, "--model", "27b", "--recipe", "r", "--seq-length", "256", "--fsdp", "2", "--dry-run")
    assert result.returncode == 2
    assert "--fsdp must be 1 or 3" in result.stderr


def test_397b_fsdp1_vs_fsdp3_sweep_crops_experts_without_cp(tmp_path: Path):
    result = _run(
        EXAMPLE / "benchmark_397b_fsdp1_vs_fsdp3.sh",
        *("--dry-run", "--run-date", RUN_DATE, "--output-dir", str(tmp_path)),
    )
    assert result.returncode == 0, result.stderr
    commands = _commands(result.stderr)
    assert len(commands) == 12
    for command in commands:
        assert "model.num_layers=4" in command
        assert "model.mtp_num_layers=1" in command
        assert "model.context_parallel_size=1" in command
        assert "model.expert_model_parallel_size=4" in command
        assert r"model.moe_layer_freq=\[1\,1\,1\,1\]" in command
        assert "model.cuda_graph_impl=none" in command
        assert "nsys profile" in command
    for experts in (64, 32):
        for seq in (256, 1024, 4096):
            for fsdp, strategy in ((3, "optim_grads_params"), (1, "optim")):
                prefix = f"model_397b-fsdp_{fsdp}-layers_4-mtp_1-experts_{experts}-cg_off-mbs_1-seq_{seq}-cp_1-ep_4-"
                command = next(c for c in commands if prefix in c)
                assert f"model.num_moe_experts={experts}" in command
                assert f"ddp.data_parallel_sharding_strategy={strategy} " in command


def test_fsdp1_vs_fsdp3_sweeps_tag_mxfp8_in_result_dirs_and_nsys_reports(tmp_path: Path):
    for script, model_tag, report_tag in (
        ("benchmark_27b_fsdp1_vs_fsdp3.sh", "model_27b", "qwen35_27b"),
        ("benchmark_397b_fsdp1_vs_fsdp3.sh", "model_397b", "qwen35_397b"),
    ):
        bf16 = _commands(_run(EXAMPLE / script, "--dry-run", "--run-date", RUN_DATE, "--output-dir", str(tmp_path)).stderr)
        assert bf16 and all("mxfp8" not in c and "mixed_precision.fp8=" not in c for c in bf16)
        result = _run(EXAMPLE / script, "--dry-run", "--mxfp8", "--run-date", RUN_DATE, "--output-dir", str(tmp_path))
        assert result.returncode == 0, result.stderr
        mxfp8 = _commands(result.stderr)
        assert len(mxfp8) == len(bf16)
        for fsdp in (1, 3):
            command = next(c for c in mxfp8 if f"{model_tag}-fsdp_{fsdp}-mxfp8-layers_" in c)
            assert f"{report_tag}-fsdp_{fsdp}-mxfp8-experts_" in command
            assert "mixed_precision.fp8_recipe=mxfp8" in command
