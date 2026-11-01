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


def test_default_sweep_runs_three_models_without_mtp_and_both_cuda_graph_modes(tmp_path: Path):
    result = _run(BENCHMARK, "--dry-run", "--run-date", RUN_DATE, "--output-dir", str(tmp_path))
    assert result.returncode == 0, result.stderr
    launches = [line for line in result.stderr.splitlines() if line.startswith("launch ")]
    assert len(launches) == 18
    commands = _commands(result.stderr)
    assert len(commands) == 18
    for command in commands:
        assert "model.num_layers=4" in command
        assert "model.mtp_num_layers=0" in command
        # Megatron-FSDP labels its collectives with layers when NVTX ranges are on.
        assert "profiling.nvtx_ranges=true" in command
        assert "fsdp_comm_nvtx" not in command

    # Megatron-FSDP's module hooks rule out per-layer graphs; both models use one
    # full-iteration graph, and MoE adds the static token budget it needs.
    for on in (c for c in commands if "-cg_on-" in c):
        assert "model.cuda_graph_impl=full_iteration" in on
        assert r"model.cuda_graph_modules=\[\]" in on
        assert "ddp.megatron_fsdp_cuda_graph_mode=true" in on
        assert "ddp.fsdp_all_gather_in_start_param_sync=false" in on
        assert "rerun_state_machine.check_for_nan_in_loss=false" in on
        assert "ddp.check_for_nan_in_grad=false" in on
    dense_on = next(c for c in commands if "model_27b-" in c and "-cg_on-" in c)
    assert "moe_expert_rank_capacity_factor" not in dense_on
    assert "moe_paged_stash" not in dense_on
    moe_on = next(c for c in commands if "model_397b-" in c and "-cg_on-" in c)
    assert "model.moe_expert_rank_capacity_factor=1.5" in moe_on
    assert "model.moe_paged_stash=true" in moe_on
    assert "model.use_transformer_engine_op_fuser=true" in moe_on
    assert "num_moe_experts" not in moe_on  # 397b keeps every routed expert
    moe_off = next(c for c in commands if "model_397b-" in c and "-cg_off-" in c)
    assert "model.cuda_graph_impl=none" in moe_off
    big = [c for c in commands if "model_2.4t-" in c]
    assert big and all("model.num_moe_experts=512" in c for c in big)
    assert all("qwen38_text_2p4t_a95b_pretrain_4gpu_gb200_bf16_fsdp1_config" in c for c in big)


def test_2_4t_halves_experts_after_oom_and_keeps_both_modes_comparable(tmp_path: Path):
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


def test_multi_node_2_4t_requires_a_fixed_expert_count():
    result = _run(BENCHMARK, "--dry-run", "--model", "2.4t", "--gpus", "8", "--master-addr", "host")
    assert result.returncode == 2
    assert "--experts-2.4t" in result.stderr


def test_run_one_print_result_dir_encodes_shape_and_cuda_graph(tmp_path: Path):
    result = _run(
        RUN_ONE,
        *("--model", "2.4t", "--recipe", "r", "--seq-length", "4096", "--num-layers", "4"),
        *("--mtp-num-layers", "1", "--num-experts", "64", "--cuda-graph", "on"),
        *("--run-date", RUN_DATE, "--results-root", str(tmp_path), "--print-result-dir"),
    )
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()).parent.name.startswith(
        "model_2.4t-fsdp_3-layers_4-mtp_1-experts_64-cg_on-mbs_1-seq_4096-cp_1-ep_4-"
    )


def test_run_one_cuda_graph_disables_nccl_graph_register_with_expandable_segments(tmp_path: Path):
    # The graph holds NCCL collectives; registering expandable segments can fault.
    result = _run(
        RUN_ONE,
        *("--model", "2.4t", "--recipe", "r", "--seq-length", "1024", "--num-experts", "64", "--cuda-graph", "on"),
        *("--run-date", RUN_DATE, "--results-root", str(tmp_path), "--dry-run"),
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        NCCL_GRAPH_REGISTER="1",
    )
    assert result.returncode == 0, result.stderr
    assert "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in result.stderr.splitlines()
    assert "NCCL_GRAPH_REGISTER=0" in result.stderr.splitlines()
    # The op-fuser grouped MLP MoE needs for the token budget is gated on this variable.
    assert "NVTE_CUTEDSL_FUSED_GROUPED_MLP=1" in result.stderr.splitlines()


def test_cuda_graph_on_rejects_alltoall_for_moe(tmp_path: Path):
    common = ("--run-date", RUN_DATE, "--dry-run", "--dispatcher", "alltoall", "--cuda-graph", "on")
    for script, args in (
        (RUN_ONE, ("--model", "2.4t", "--recipe", "r", "--seq-length", "1024", "--num-experts", "64")),
        (BENCHMARK, ("--model", "397b", "--output-dir", str(tmp_path))),
        (EXAMPLE / "benchmark_2.4t_fsdp1_vs_fsdp3.sh", ("--output-dir", str(tmp_path))),
    ):
        result = _run(script, *args, *common)
        assert result.returncode == 2, result.stderr
        assert "requires --dispatcher hybridep" in result.stderr
    # Dense 27b has no dispatcher, so the sweep still accepts it.
    result = _run(BENCHMARK, "--model", "27b", "--output-dir", str(tmp_path), *common)
    assert result.returncode == 0, result.stderr


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
        assert "model.mtp_num_layers=0" in command
        assert "model.cuda_graph_impl=none" in command
        assert "train.global_batch_size=16" in command
        assert "nsys profile" in command
    for seq in (256, 1024, 4096):
        fsdp3 = next(c for c in commands if f"model_27b-fsdp_3-layers_8-mtp_0-experts_0-cg_off-mbs_1-seq_{seq}-" in c)
        assert "ddp.data_parallel_sharding_strategy=optim_grads_params" in fsdp3
        assert "comm_overlap.overlap_param_gather=true" in fsdp3
        fsdp1 = next(c for c in commands if f"model_27b-fsdp_1-layers_8-mtp_0-experts_0-cg_off-mbs_1-seq_{seq}-" in c)
        assert "ddp.data_parallel_sharding_strategy=optim " in fsdp1
        assert "ddp.overlap_grad_reduce=false" in fsdp1
        assert "comm_overlap.overlap_param_gather=false" in fsdp1


def test_run_one_nsys_wraps_torchrun_with_one_report_per_node(tmp_path: Path):
    result = _run(
        RUN_ONE,
        *("--model", "27b", "--recipe", "r", "--seq-length", "4096", "--profile", "nsys"),
        *("--run-date", RUN_DATE, "--results-root", str(tmp_path), "--dry-run"),
    )
    assert result.returncode == 0, result.stderr
    (command,) = _commands(result.stderr)
    nsys = command.index("nsys profile")
    torchrun = command.index("-m torch.distributed.run")
    assert nsys < torchrun
    assert command.count("nsys profile") == 1
    assert "-seq_4k-node_0 " in command
    assert "%q{RANK}" not in command
    assert r"profiling.profile_ranks=\[0\,1\,2\,3\]" in command


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
        assert "model.mtp_num_layers=0" in command
        assert "model.context_parallel_size=1" in command
        assert "model.expert_model_parallel_size=4" in command
        assert r"model.moe_layer_freq=\[1\,1\,1\,1\]" in command
        assert "model.cuda_graph_impl=none" in command
        assert "nsys profile" in command
    for experts in (64, 32):
        for seq in (256, 1024, 4096):
            for fsdp, strategy in ((3, "optim_grads_params"), (1, "optim")):
                prefix = f"model_397b-fsdp_{fsdp}-layers_4-mtp_0-experts_{experts}-cg_off-mbs_1-seq_{seq}-cp_1-ep_4-"
                command = next(c for c in commands if prefix in c)
                assert f"model.num_moe_experts={experts}" in command
                assert f"ddp.data_parallel_sharding_strategy={strategy} " in command


def test_2_4t_fsdp1_vs_fsdp3_sweep_names_the_model_2_4t(tmp_path: Path):
    result = _run(
        EXAMPLE / "benchmark_2.4t_fsdp1_vs_fsdp3.sh",
        *("--dry-run", "--run-date", RUN_DATE, "--output-dir", str(tmp_path)),
    )
    assert result.returncode == 0, result.stderr
    commands = _commands(result.stderr)
    assert len(commands) == 12
    for command in commands:
        assert "qwen38_text_2p4t_a95b_pretrain_4gpu_gb200_bf16_fsdp1_config" in command
        assert "model.num_layers=4" in command
        assert "model.mtp_num_layers=0" in command
        assert "model.expert_model_parallel_size=4" in command
        assert "model.cuda_graph_impl=none" in command
        assert "/qwen38_2.4t-fsdp_" in command
    for experts in (64, 32):
        for seq in (256, 1024, 4096):
            for fsdp, strategy in ((3, "optim_grads_params"), (1, "optim")):
                prefix = f"model_2.4t-fsdp_{fsdp}-layers_4-mtp_0-experts_{experts}-cg_off-mbs_1-seq_{seq}-cp_1-ep_4-"
                command = next(c for c in commands if prefix in c)
                assert f"model.num_moe_experts={experts}" in command
                assert f"ddp.data_parallel_sharding_strategy={strategy} " in command
    launches = [line for line in result.stderr.splitlines() if line.startswith("launch ")]
    assert launches and all(line.startswith("launch model=2.4t ") for line in launches)


def test_2_4t_fsdp1_vs_fsdp3_sweep_runs_cuda_graph_modes(tmp_path: Path):
    result = _run(
        EXAMPLE / "benchmark_2.4t_fsdp1_vs_fsdp3.sh",
        *("--dry-run", "--experts", "64", "--seq", "1024", "--cuda-graph", "off,on"),
        *("--run-date", RUN_DATE, "--output-dir", str(tmp_path)),
    )
    assert result.returncode == 0, result.stderr
    commands = _commands(result.stderr)
    assert len(commands) == 4
    graphed = [c for c in commands if "-cg_on-" in c]
    assert len(graphed) == 2
    for command in graphed:
        assert "model.cuda_graph_impl=full_iteration" in command
        assert "model.moe_expert_rank_capacity_factor=1.5" in command
        assert "model.moe_paged_stash=true" in command
        assert "ddp.megatron_fsdp_cuda_graph_mode=true" in command


def test_force_balance_tags_moe_cases_and_skips_dense(tmp_path: Path):
    for script, prefix, report in (
        ("benchmark_397b_fsdp1_vs_fsdp3.sh", "model_397b-fsdp_3-fb-layers_4-", "/qwen35_397b-fsdp_3-fb-experts_"),
        ("benchmark_2.4t_fsdp1_vs_fsdp3.sh", "model_2.4t-fsdp_3-fb-layers_4-", "/qwen38_2.4t-fsdp_3-fb-experts_"),
    ):
        result = _run(
            EXAMPLE / script,
            *("--dry-run", "--force-balance", "--experts", "64", "--seq", "256", "--fsdp", "3"),
            *("--run-date", RUN_DATE, "--output-dir", str(tmp_path)),
        )
        assert result.returncode == 0, result.stderr
        (command,) = _commands(result.stderr)
        assert prefix in command and report in command
        assert "model.moe_router_force_load_balancing=true" in command

    result = _run(
        BENCHMARK,
        *("--dry-run", "--force-balance", "--model", "27b,397b", "--seq", "4096", "--cp", "1", "--mbs", "1"),
        *("--cuda-graph", "off", "--run-date", RUN_DATE, "--output-dir", str(tmp_path)),
    )
    assert result.returncode == 0, result.stderr
    dense, moe = _commands(result.stderr)
    assert "model_27b-fsdp_3-layers_4-" in dense and "force_load_balancing" not in dense
    assert "model_397b-fsdp_3-fb-layers_4-" in moe
    assert "model.moe_router_force_load_balancing=true" in moe

    default = _commands(
        _run(
            BENCHMARK,
            "--dry-run",
            "--model",
            "397b",
            "--seq",
            "4096",
            "--cp",
            "1",
            "--mbs",
            "1",
            "--cuda-graph",
            "off",
            "--run-date",
            RUN_DATE,
            "--output-dir",
            str(tmp_path),
        ).stderr
    )
    assert default and all("-fb-" not in c and "force_load_balancing" not in c for c in default)


def test_run_one_rejects_force_balance_for_dense_model():
    result = _run(RUN_ONE, "--model", "27b", "--recipe", "r", "--seq-length", "256", "--force-balance", "--dry-run")
    assert result.returncode == 2
    assert "--force-balance only applies to MoE models" in result.stderr


def test_fsdp1_vs_fsdp3_sweeps_tag_mxfp8_in_result_dirs_and_nsys_reports(tmp_path: Path):
    for script, model_tag, report_tag in (
        ("benchmark_27b_fsdp1_vs_fsdp3.sh", "model_27b", "qwen35_27b"),
        ("benchmark_397b_fsdp1_vs_fsdp3.sh", "model_397b", "qwen35_397b"),
        ("benchmark_2.4t_fsdp1_vs_fsdp3.sh", "model_2.4t", "qwen38_2.4t"),
    ):
        bf16 = _commands(
            _run(EXAMPLE / script, "--dry-run", "--run-date", RUN_DATE, "--output-dir", str(tmp_path)).stderr
        )
        assert bf16 and all("mxfp8" not in c and "mixed_precision.fp8=" not in c for c in bf16)
        result = _run(EXAMPLE / script, "--dry-run", "--mxfp8", "--run-date", RUN_DATE, "--output-dir", str(tmp_path))
        assert result.returncode == 0, result.stderr
        mxfp8 = _commands(result.stderr)
        assert len(mxfp8) == len(bf16)
        for fsdp in (1, 3):
            command = next(c for c in mxfp8 if f"{model_tag}-fsdp_{fsdp}-mxfp8-layers_" in c)
            assert f"{report_tag}-fsdp_{fsdp}-mxfp8-experts_" in command
            assert "mixed_precision.fp8_recipe=mxfp8" in command
