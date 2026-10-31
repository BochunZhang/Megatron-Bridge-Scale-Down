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
