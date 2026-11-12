# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Dry-run tests for 01-analyse/04-fsdp3-comm."""

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest


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


def _report_base(command: str) -> Path:
    return next(Path(arg.removeprefix("--output=")) for arg in shlex.split(command) if arg.startswith("--output="))


def _sweep(tmp_path: Path, *args: str) -> list[str]:
    result = _run(BENCHMARK, "--dry-run", *args, "--run-date", RUN_DATE, "--output-dir", str(tmp_path))
    assert result.returncode == 0, result.stderr
    return _commands(result.stderr)


def test_benchmark_requires_a_model():
    result = _run(BENCHMARK, "--dry-run")
    assert result.returncode == 2
    assert "--model is required" in result.stderr


def test_default_sweep_per_model_crops_layers_and_experts(tmp_path: Path):
    dense = _sweep(tmp_path, "--model", "27b")
    assert len(dense) == 6  # seq 256,1024,4096 x FSDP 3,1
    for command in dense:
        assert "qwen35_text_27b_pretrain_4gpu_gb200_bf16_fsdp1_config" in command
        assert "model.num_layers=8" in command and "model.mtp_num_layers=0" in command
        assert "model.linear_attention_freq=4" in command
        assert "model.tensor_model_parallel_size=1" in command and "model.sequence_parallel=false" in command
        assert "model.cuda_graph_impl=none" in command
        assert "train.global_batch_size=16" in command
        assert "nsys profile" in command and "profiling.nvtx_ranges=true" in command
    for seq in (256, 1024, 4096):
        fsdp3 = next(
            c for c in dense if f"model_27b-fsdp_3-layers_8-mtp_0-experts_0-cg_off-mbs_1-seq_{seq}-cp_1-" in c
        )
        assert "ddp.data_parallel_sharding_strategy=optim_grads_params" in fsdp3
        assert "comm_overlap.overlap_param_gather=true" in fsdp3
        fsdp1 = next(
            c for c in dense if f"model_27b-fsdp_1-layers_8-mtp_0-experts_0-cg_off-mbs_1-seq_{seq}-cp_1-" in c
        )
        assert "ddp.data_parallel_sharding_strategy=optim " in fsdp1
        assert "ddp.overlap_grad_reduce=false" in fsdp1 and "comm_overlap.overlap_param_gather=false" in fsdp1

    for model, recipe, report in (
        ("397b", "qwen35_text_397b_a17b_pretrain_4gpu_gb200_bf16_fsdp1_config", "/qwen35_397b-fsdp_"),
        ("2p4t", "qwen38_text_2p4t_a95b_pretrain_4gpu_gb200_bf16_fsdp1_config", "/qwen38_2.4t-fsdp_"),
    ):
        moe = _sweep(tmp_path, "--model", model)
        assert len(moe) == 12  # experts 64,32 x seq x FSDP 3,1
        name = "2.4t" if model == "2p4t" else model
        for command in moe:
            assert recipe in command and report in command
            assert "model.num_layers=4" in command
            assert r"model.moe_layer_freq=\[1\,1\,1\,1\]" in command
            assert "model.expert_model_parallel_size=4" in command
        for experts in (64, 32):
            prefix = f"model_{name}-fsdp_3-layers_4-mtp_0-experts_{experts}-cg_off-mbs_1-seq_256-cp_1-ep_4-"
            assert f"model.num_moe_experts={experts}" in next(c for c in moe if prefix in c)


def test_experts_full_keeps_the_recipe_expert_count(tmp_path: Path):
    (command,) = _sweep(tmp_path, "--model", "397b", "--experts", "full", "--seq", "256", "--fsdp", "3")
    assert "num_moe_experts" not in command
    assert "-experts_full-" in command


def test_fsdp2_shards_gradients_with_overlapped_reduce_scatter(tmp_path: Path):
    (command,) = _sweep(tmp_path, "--model", "27b", "--fsdp", "2", "--seq", "256")
    assert "model_27b-fsdp_2-layers_8-" in command and "/qwen35_27b-fsdp_2-experts_0-" in command
    assert "ddp.data_parallel_sharding_strategy=optim_grads " in command
    assert "ddp.overlap_grad_reduce=true" in command and "comm_overlap.overlap_grad_reduce=true" in command
    assert "ddp.overlap_param_gather=false" in command and "comm_overlap.overlap_param_gather=false" in command
    assert "dist.use_megatron_fsdp=true" in command and "checkpoint.ckpt_format=fsdp_dtensor" in command


def test_run_one_rejects_unknown_fsdp_level():
    result = _run(RUN_ONE, "--model", "27b", "--seq-length", "256", "--fsdp", "4", "--dry-run")
    assert result.returncode == 2
    assert "--fsdp must be ddp, 1, 2, or 3" in result.stderr


def test_offload_forwards_config_and_names_results(tmp_path: Path):
    common = ("--model", "27b", "--fsdp", "3", "--seq", "1024")
    (command,) = _sweep(
        tmp_path, *common, "--offload", "gradient,optimizer,weight"
    )
    args = shlex.split(command)
    suffix = "-offload_optimizer_weight_gradient"
    assert "ddp.fsdp_cpu_offload=[optimizer,weight,gradient]" in args
    assert "optimizer.use_precision_aware_optimizer=false" in args
    assert any(arg.startswith("--output=") and arg.endswith(suffix) for arg in args)

    result = _run(
        RUN_ONE, *common, "--offload", "weight,gradient,optimizer",
        "--profile", "nsys", "--run-date", RUN_DATE, "--results-root", str(tmp_path),
        "--print-result-dir",
    )
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()).parent.name.endswith(f"-date_{RUN_DATE}{suffix}")


def test_fsdp1_gradient_offload_enables_gradient_overlap(tmp_path: Path):
    (command,) = _sweep(tmp_path, "--model", "27b", "--fsdp", "1", "--seq", "256", "--offload", "gradient")
    args = shlex.split(command)
    assert "ddp.fsdp_cpu_offload=[gradient]" in args
    assert "ddp.overlap_grad_reduce=true" in args
    assert "comm_overlap.overlap_grad_reduce=true" in args


def test_dense_fsdp3_offload_allows_full_iteration_graph(tmp_path: Path):
    (command,) = _sweep(
        tmp_path,
        "--model",
        "27b",
        "--fsdp",
        "3",
        "--seq",
        "256",
        "--offload",
        "weight,gradient,optimizer",
        "--cuda-graph",
        "on",
    )
    args = shlex.split(command)
    assert "ddp.fsdp_cpu_offload=[optimizer,weight,gradient]" in args
    assert "model.cuda_graph_impl=full_iteration" in args
    assert "ddp.megatron_fsdp_cuda_graph_mode=true" in args
    assert "ddp.fsdp_all_gather_in_start_param_sync=false" in args
    assert "optimizer.optimizer_cuda_graph=false" in args


def test_offload_rejects_invalid_launches_before_the_sweep(tmp_path: Path):
    common = ("--model", "27b", "--seq", "256", "--dry-run", "--output-dir", str(tmp_path))
    for options in (
        ("--offload", "weights"),
        ("--offload", "weight,"),
        ("--offload", "weight", "--fsdp", "3,ddp"),
        ("--offload", "weight", "--fsdp", "3,1", "--cuda-graph", "off,on"),
        ("--offload", "weight", "--model", "397b", "--fsdp", "2", "--cuda-graph", "on"),
        ("--offload", "weight", "--symmetric"),
    ):
        result = _run(BENCHMARK, *common, *options)
        assert result.returncode == 2 and "--offload" in result.stderr
        assert not _commands(result.stderr)


def test_tp_turns_on_sequence_parallel_and_shrinks_data_parallel(tmp_path: Path):
    commands = _sweep(tmp_path, "--model", "27b", "--tp", "1,2", "--seq", "4096", "--fsdp", "3")
    assert len(commands) == 2
    tp1 = next(c for c in commands if "-seq_4096-cp_1-" in c)
    assert "model.tensor_model_parallel_size=1" in tp1 and "model.sequence_parallel=false" in tp1
    assert "train.global_batch_size=16" in tp1
    tp2 = next(c for c in commands if "-seq_4096-tp_2-cp_1-" in c)
    assert "model.tensor_model_parallel_size=2" in tp2 and "model.sequence_parallel=true" in tp2
    # 4 microbatches x MBS 1 x DP 2.
    assert "train.global_batch_size=8" in tp2 and "-gbs_8-" in tp2
    assert "/qwen35_27b-fsdp_3-experts_0-cg_off-tp_2-cp_1-" in tp2
    assert "comm_overlap.tp_comm_overlap=true" not in tp2


def test_tp_comm_overlap_applies_only_to_tp_cases(tmp_path: Path):
    tp1, tp2 = _sweep(tmp_path, "--model", "27b", "--tp", "1,2", "--tp-comm-overlap", "--seq", "4096", "--fsdp", "3")
    assert "model.tensor_model_parallel_size=1" in tp1
    assert "comm_overlap.tp_comm_overlap=true" not in tp1 and "-tp-overlap" not in tp1
    assert "model.sequence_parallel=true" in tp2 and "comm_overlap.tp_comm_overlap=true" in tp2
    assert "-seq_4096-tp_2-tp-overlap-cp_1-" in tp2
    assert "/qwen35_27b-fsdp_3-experts_0-cg_off-tp_2-tp-overlap-cp_1-" in tp2
    assert "dist.create_all_gather_group=false" in tp2

    result = _run(RUN_ONE, "--model", "27b", "--seq-length", "4096", "--tp-comm-overlap", "--dry-run")
    assert result.returncode == 2 and "--tp-comm-overlap requires --tp > 1" in result.stderr


def test_cp_and_tp_combine_and_are_validated_before_any_launch(tmp_path: Path):
    command = _sweep(tmp_path, "--model", "397b", "--experts", "64", "--tp", "2", "--cp", "2", "--seq", "4096")[0]
    assert "model.context_parallel_size=2" in command and "model.tensor_model_parallel_size=2" in command
    assert "train.global_batch_size=4" in command  # DP = 4 / (2*2) = 1
    # seq/cp must split over the TP group for sequence parallelism.
    result = _run(RUN_ONE, "--model", "27b", "--seq-length", "6", "--cp", "1", "--tp", "4", "--dry-run")
    assert result.returncode == 2 and "seq-length/cp must be divisible by tp" in result.stderr
    # An invalid case stops the sweep before the first launch.
    result = _run(BENCHMARK, "--dry-run", "--model", "27b", "--gbs", "16,6", "--output-dir", str(tmp_path))
    assert result.returncode == 2
    assert "not divisible by MBS*DP" in result.stderr
    assert not _commands(result.stderr)


def test_mbs_and_gbs_lists_expand_the_matrix(tmp_path: Path):
    commands = _sweep(tmp_path, "--model", "27b", "--fsdp", "3", "--seq", "1024", "--mbs", "1,2", "--gbs", "16,32")
    assert len(commands) == 4
    for mbs in (1, 2):
        for gbs in (16, 32):
            command = next(c for c in commands if f"-mbs_{mbs}-seq_1024-cp_1-ep_1-disp_none-gbs_{gbs}-" in c)
            assert f"train.micro_batch_size={mbs}" in command and f"train.global_batch_size={gbs}" in command
    # Without --gbs the microbatch count derives GBS per MBS.
    derived = _sweep(tmp_path, "--model", "27b", "--fsdp", "3", "--seq", "1024", "--mbs", "2")
    assert "train.global_batch_size=32" in derived[0]


def test_full_and_linear_replace_the_hybrid_attention_layout(tmp_path: Path):
    (full,) = _sweep(tmp_path, "--model", "27b", "--full", "--seq", "256", "--fsdp", "3")
    assert "model.linear_attention_freq=1 " in full
    assert "model_27b-fsdp_3-layers_8-attn_full-mtp_0-" in full and "/qwen35_27b-fsdp_3-attn_full-experts_0-" in full
    (linear,) = _sweep(tmp_path, "--model", "397b", "--linear", "--experts", "64", "--seq", "256", "--fsdp", "3")
    assert r"model.linear_attention_freq=\[1\,1\,1\,1\]" in linear
    assert "model_397b-fsdp_3-layers_4-attn_linear-mtp_0-" in linear
    for script in (BENCHMARK, RUN_ONE):
        result = _run(script, "--model", "27b", "--seq-length", "256", "--full", "--linear", "--dry-run")
        assert result.returncode == 2 and "--full and --linear are exclusive" in result.stderr
    result = _run(RUN_ONE, "--model", "27b", "--seq-length", "256", "--linear", "--dry-run")
    assert result.returncode == 2 and "--linear requires --num-layers" in result.stderr


def test_run_one_print_result_dir_encodes_shape_and_cuda_graph(tmp_path: Path):
    result = _run(
        RUN_ONE,
        *("--model", "2.4t", "--seq-length", "4096", "--num-layers", "4"),
        *("--mtp-num-layers", "1", "--num-experts", "64", "--cuda-graph", "on"),
        *("--run-date", RUN_DATE, "--results-root", str(tmp_path), "--print-result-dir"),
    )
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()).parent.name.startswith(
        "model_2.4t-fsdp_3-layers_4-mtp_1-experts_64-cg_on-partial-moe-mbs_1-seq_4096-cp_1-ep_4-"
    )


def test_nsys_records_the_four_gpus_of_node_0_into_one_report(tmp_path: Path):
    common = ("--model", "27b", "--seq-length", "4096", "--profile", "nsys", "--gpus", "8", "--master-addr", "host")
    common += ("--run-date", RUN_DATE, "--results-root", str(tmp_path), "--dry-run")
    (node0,) = _commands(_run(RUN_ONE, *common, "--node-rank", "0").stderr)
    assert node0.count("nsys profile") == 1
    assert node0.index("nsys profile") < node0.index("-m torch.distributed.run")
    assert "-seq_4k " in node0 and "-node_" not in node0.split("--output=")[1].split()[0]
    assert r"profiling.profile_ranks=\[0\,1\,2\,3\]" in node0
    (node1,) = _commands(_run(RUN_ONE, *common, "--node-rank", "1").stderr)
    assert "nsys profile" not in node1


def test_nsys_traces_kernels_inside_cuda_graphs(tmp_path: Path):
    common = ("--model", "27b", "--seq-length", "4096", "--run-date", RUN_DATE, "--results-root", str(tmp_path))
    common += ("--dry-run",)
    (on,) = _commands(_run(RUN_ONE, *common, "--profile", "nsys", "--cuda-graph", "on").stderr)
    assert on.index("--cuda-graph-trace=node") < on.index("-m torch.distributed.run")
    (off,) = _commands(_run(RUN_ONE, *common, "--profile", "nsys", "--cuda-graph", "off").stderr)
    assert "--cuda-graph-trace" not in off
    (no_profile,) = _commands(_run(RUN_ONE, *common, "--profile", "none", "--cuda-graph", "on").stderr)
    assert "--cuda-graph-trace" not in no_profile


def test_nsys_gpu_metrics_is_opt_in_and_defaults_to_20_khz(tmp_path: Path):
    common = ("--model", "27b", "--seq-length", "4096", "--results-root", str(tmp_path), "--dry-run")
    (plain,) = _commands(_run(RUN_ONE, *common, "--nsys").stderr)
    assert "--gpu-metrics-" not in plain

    (metrics,) = _commands(_run(RUN_ONE, *common, "--metric").stderr)
    assert metrics.startswith("command: nsys profile ")
    assert "--gpu-metrics-devices=all --gpu-metrics-frequency=20000" in metrics
    assert "-metric_20000/node0/" in metrics and "-metric_20000 " in metrics


def test_sweep_forwards_gpu_metrics_frequency(tmp_path: Path):
    common = ("--model", "27b", "--fsdp", "3", "--seq", "256")
    (metrics,) = _sweep(tmp_path, *common, "--metric", "10000")
    assert "--gpu-metrics-devices=all --gpu-metrics-frequency=10000" in metrics
    assert "-metric_10000/node0/" in metrics

    invalid = _run(BENCHMARK, *common, "--dry-run", "--metric=200001")
    assert invalid.returncode == 2
    assert "--metric frequency must be an integer from 10 to 200000 Hz" in invalid.stderr


def test_merge_keeps_cuda_graph_launches_as_one_block(tmp_path: Path):
    commands = _sweep(tmp_path, "--model", "27b", "--fsdp", "3", "--seq", "256", "--cuda-graph", "off,on", "--merge")
    off = next(c for c in commands if "-cg_off-" in c)
    on = next(c for c in commands if "-cg_on-merge-" in c)
    assert "--cuda-graph-trace" not in off
    assert on.index("--cuda-graph-trace=graph") < on.index("-m torch.distributed.run")
    assert "--cuda-graph-trace=node" not in on
    assert "model_27b-fsdp_3-layers_8-mtp_0-experts_0-cg_on-merge-mbs_1-" in on
    assert "/qwen35_27b-fsdp_3-experts_0-cg_on-merge-cp_1-" in on
    common = ("--model", "27b", "--seq-length", "256", "--cuda-graph", "on", "--merge", "--dry-run")
    (no_profile,) = _commands(_run(RUN_ONE, *common, "--results-root", str(tmp_path)).stderr)
    assert "--cuda-graph-trace" not in no_profile and "-cg_on-mbs_" in no_profile


def test_full_iteration_cuda_graph_for_dense_fsdp(tmp_path: Path):
    on = _sweep(tmp_path, "--model", "27b", "--fsdp", "3,2,1", "--seq", "1024", "--cuda-graph", "on")
    assert len(on) == 3
    for command in on:
        assert "model.cuda_graph_impl=full_iteration" in command
        assert r"model.cuda_graph_modules=\[\]" in command
        assert "ddp.megatron_fsdp_cuda_graph_mode=true" in command
        assert "ddp.fsdp_all_gather_in_start_param_sync=false" in command
        assert (
            "rerun_state_machine.check_for_nan_in_loss=false" in command
            and "ddp.check_for_nan_in_grad=false" in command
        )


def test_partial_cuda_graph_for_moe_fsdp(tmp_path: Path):
    for dispatcher in ("hybridep", "alltoall"):
        commands = _sweep(
            tmp_path,
            "--model", "397b,2.4t,122b", "--fsdp", "3", "--experts", "64",
            "--seq", "1024", "--cuda-graph", "on", "--dispatcher", dispatcher,
        )
        assert len(commands) == 3
        for command in commands:
            assert "-cg_on-partial-moe-" in command
            assert "model.cuda_graph_impl=local" in command
            assert "model.cuda_graph_modules=[attn,moe_router,moe_preprocess]" in shlex.split(command)
            assert "model.gradient_accumulation_fusion=false" in command
            assert "ddp.megatron_fsdp_max_pool_double_buffer=true" in command
            assert "ddp.fsdp_all_gather_in_start_param_sync=true" in command
            assert "model.use_te_rng_tracker=true" in command and "rng.te_rng_tracker=true" in command
            assert f"-disp_{dispatcher}-" in command
            for full_graph_only in (
                "moe_expert_rank_capacity_factor",
                "moe_paged_stash",
                "use_transformer_engine_op_fuser",
                "megatron_fsdp_cuda_graph_mode=true",
                "check_for_nan_in_loss=false",
                "check_for_nan_in_grad=false",
            ):
                assert full_graph_only not in command


def test_default_moe_sweep_uses_native_partial_graph_for_fsdp1_and_fsdp3(tmp_path: Path):
    commands = _sweep(tmp_path, "--model", "397b", "--experts", "64", "--seq", "1024", "--cuda-graph", "off,on")
    assert len(commands) == 4
    fsdp1 = next(c for c in commands if "model_397b-fsdp_1-" in c and "-cg_on-partial-moe-" in c)
    assert "model.cuda_graph_impl=local" in fsdp1
    assert "model.cuda_graph_modules=[attn,moe_router,moe_preprocess]" in shlex.split(fsdp1)
    assert "ddp.overlap_param_gather=false" in fsdp1
    assert "ddp.megatron_fsdp_max_pool_double_buffer" not in fsdp1
    assert sum("-cg_on-partial-moe-" in command for command in commands) == 2


def test_default_moe_sweep_keeps_graph_offload_on_fsdp3(tmp_path: Path):
    (command,) = _sweep(
        tmp_path, "--model", "397b", "--experts", "64", "--seq", "1024", "--cuda-graph", "on", "--offload", "weight"
    )
    assert "model_397b-fsdp_3-" in command


def test_moe_fsdp3_offload_allows_native_partial_graph(tmp_path: Path):
    (command,) = _sweep(
        tmp_path,
        "--model", "397b", "--fsdp", "3", "--seq", "1024", "--experts", "64",
        "--offload", "weight,gradient,optimizer", "--cuda-graph", "on",
    )
    args = shlex.split(command)
    assert "ddp.fsdp_cpu_offload=[optimizer,weight,gradient]" in args
    assert "model.cuda_graph_impl=local" in args
    assert "model.cuda_graph_modules=[attn,moe_router,moe_preprocess]" in args
    assert "ddp.fsdp_all_gather_in_start_param_sync=true" in args
    assert "ddp.megatron_fsdp_cuda_graph_mode=true" not in args
    assert "optimizer.optimizer_cuda_graph=false" in args


@pytest.mark.unit
@pytest.mark.parametrize(
    ("script", "result_group", "case_count"),
    [("01_cuda_graph.sh", "01-cuda-graph", 2), ("02_mbs.sh", "02-mbs", 3), ("03_fsdp3.sh", "03-fsdp3", 2)],
)
def test_baseline_saves_stable_case_paths_under_each_experiment(
    tmp_path: Path, script: str, result_group: str, case_count: int
):
    result = _run(EXAMPLE / "01-baseline" / script, "--model", "2.4t", "--dry-run", "--output-dir", str(tmp_path))
    assert result.returncode == 0, result.stderr
    commands = _commands(result.stderr)
    assert len(commands) == case_count
    for command in commands:
        node_dir = _report_base(command).parent
        assert node_dir.name == "node0"
        assert node_dir.parent.parent == tmp_path / "01-baseline" / result_group
        assert node_dir.parent.name.endswith("-profile_nsys")


@pytest.mark.unit
@pytest.mark.parametrize("script", ["01_priority.sh", "02_symmetric_multicast.sh", "03_ag_pause.sh"])
def test_nccl_studies_default_to_the_cta_pause_build(script: str):
    result = _run(EXAMPLE / "05-nccl" / script, "--dry-run", NCCL_LIB="", LD_PRELOAD="")
    assert result.returncode == 0, result.stderr
    case_lines = [line for line in result.stderr.splitlines() if line.startswith("NCCL case ")]
    assert case_lines
    library = REPO_ROOT / "nccl/nccl-2.32.3-cta-with-pause/build/lib/libnccl.so.2"
    assert all(f"LD_PRELOAD={library}" in shlex.split(line) for line in case_lines)


@pytest.mark.unit
@pytest.mark.parametrize("script", sorted(EXAMPLE.glob("[0-9][0-9]-*/[0-9][0-9]_*.sh")), ids=lambda p: p.stem)
def test_numbered_studies_skip_existing_case_directories(tmp_path: Path, script: Path):
    args = ("--model", "27b", "--output-dir", str(tmp_path))
    dry_run = _run(script, *args, "--dry-run", NVME_PATH="", NCCL_LIB="")
    assert dry_run.returncode == 0, dry_run.stderr
    case_dirs = [_report_base(command).parent.parent for command in _commands(dry_run.stderr)]
    assert case_dirs
    case_plan = _run(script, *args, "--print-case-dirs", NVME_PATH="", NCCL_LIB="")
    assert case_plan.returncode == 0, case_plan.stderr
    assert list(map(Path, case_plan.stdout.splitlines())) == case_dirs
    for case_dir in case_dirs:
        assert case_dir.parent == tmp_path / script.parent.name / script.stem.replace("_", "-")
        assert "-date_" not in case_dir.name
        case_dir.mkdir(parents=True)

    result = _run(script, *args, NVME_PATH="", NCCL_LIB="")
    assert result.returncode == 0, result.stderr
    assert result.stderr.count("skip existing case:") == len(case_dirs)
    assert all(not any(case_dir.iterdir()) for case_dir in case_dirs)


@pytest.mark.unit
def test_numbered_study_launches_missing_case_next_to_existing_one(tmp_path: Path):
    script = EXAMPLE / "01-baseline/01_cuda_graph.sh"
    args = ("--model", "27b", "--output-dir", str(tmp_path))
    dry_run = _run(script, *args, "--dry-run")
    assert dry_run.returncode == 0, dry_run.stderr
    existing, missing = map(_report_base, _commands(dry_run.stderr))
    existing.parent.parent.mkdir(parents=True)
    legacy_case_dir = tmp_path / "01-baseline" / f"{missing.parent.parent.name}-baseline"
    legacy_case_dir.mkdir()

    fake_ip = tmp_path / "ip"
    fake_ip.write_text("#!/bin/sh\nprintf '%s\\n' '2: eth0 inet 192.0.2.1/24'\n")
    fake_ip.chmod(0o755)
    fake_nsys = tmp_path / "nsys"
    fake_nsys.write_text(
        '#!/bin/sh\nfor arg; do\n  case "$arg" in\n'
        '    --output=*) printf "report\\n" > "${arg#--output=}.nsys-rep"; exit 0;;\n'
        "  esac\ndone\nexit 1\n"
    )
    fake_nsys.chmod(0o755)

    result = _run(script, *args, PATH=f"{tmp_path}:{os.environ['PATH']}", GLOO_SOCKET_IFNAME="eth0")
    assert result.returncode == 0, result.stderr
    assert f"skip existing case: {existing.parent.parent}" in result.stderr
    assert not existing.parent.exists()
    assert not any(legacy_case_dir.iterdir())
    assert (missing.parent / "run_info.txt").read_text().splitlines()[-1] == "status=0"
    assert missing.with_suffix(".nsys-rep").stat().st_size > 0


@pytest.mark.unit
def test_migrate_legacy_results_preserves_shared_artifacts_and_conflicts(tmp_path: Path):
    args = ("--output-dir", str(tmp_path))
    baseline_plan = _run(EXAMPLE / "01-baseline/01_cuda_graph.sh", *args, "--model", "27b", "--print-case-dirs")
    assert baseline_plan.returncode == 0, baseline_plan.stderr
    baseline_target = next(Path(path) for path in baseline_plan.stdout.splitlines() if "-cg_on-" in path)
    baseline_group = baseline_target.parent.parent
    baseline_source = baseline_group / f"{baseline_target.name}-baseline"
    baseline_targets = [
        baseline_group / study / baseline_target.name for study in ("01-cuda-graph", "02-mbs", "03-fsdp3")
    ]

    nvme_plan = _run(EXAMPLE / "04-nvme/01_dram_vs_nvme.sh", *args, "--model", "27b", "--print-case-dirs")
    assert nvme_plan.returncode == 0, nvme_plan.stderr
    nvme_target = Path(nvme_plan.stdout.splitlines()[0])
    nvme_source = nvme_target.parent.parent / nvme_target.name
    for source in (baseline_source, nvme_source):
        (source / "node0").mkdir(parents=True)
        (source / "node0/train.log").write_text("legacy result\n")
        (source / "node0.attempt-1").mkdir()

    command = [sys.executable, str(EXAMPLE / "migrate_results.py"), *args]
    preview = subprocess.run([*command, "--dry-run"], capture_output=True, text=True)
    assert preview.returncode == 0, preview.stderr
    assert preview.stderr.count("Dry run: Move ") == 2 and preview.stderr.count("Dry run: Copy ") == 2
    assert baseline_source.is_dir() and nvme_source.is_dir()
    assert not any(target.exists() for target in [*baseline_targets, nvme_target])

    nvme_target.mkdir(parents=True)
    conflict = nvme_target / "already-present"
    conflict.write_text("keep\n")
    migration = subprocess.run(command, capture_output=True, text=True)
    assert migration.returncode == 1, migration.stderr
    assert "Destination already exists" in migration.stderr
    assert conflict.read_text() == "keep\n" and nvme_source.is_dir()
    assert not baseline_source.exists()
    for target in baseline_targets:
        assert (target / "node0/train.log").read_text() == "legacy result\n"
        assert (target / "node0.attempt-1").is_dir()

    conflict.unlink()
    nvme_target.rmdir()
    migration = subprocess.run(command, capture_output=True, text=True)
    assert migration.returncode == 0, migration.stderr
    assert not nvme_source.exists()
    assert (nvme_target / "node0/train.log").read_text() == "legacy result\n"
    rerun = subprocess.run(command, capture_output=True, text=True)
    assert rerun.returncode == 0, rerun.stderr
    assert "No legacy result directories matched" in rerun.stderr


def test_weight_offload_study_uses_native_partial_graph_for_moe_cp4(tmp_path: Path):
    result = _run(
        EXAMPLE / "03-cpu-offload/01_weight_offload.sh",
        "--model",
        "2.4t",
        "--dry-run",
        "--output-dir",
        str(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    commands = [shlex.split(command) for command in _commands(result.stderr)]
    assert len(commands) == 8
    cp4_commands = [args for args in commands if "model.context_parallel_size=4" in args]
    assert len(cp4_commands) == 4
    for args in cp4_commands:
        assert "model.cuda_graph_impl=local" in args
        assert "model.cuda_graph_modules=[attn,moe_router,moe_preprocess]" in args
    assert {
        (
            "dist.create_all_gather_group=true" in args,
            "ddp.fsdp_cpu_offload=[weight]" in args,
        )
        for args in cp4_commands
    } == {(False, False), (True, False), (False, True), (True, True)}


def test_moe_attention_only_graph_has_distinct_results(tmp_path: Path):
    result = _run(
        RUN_ONE,
        "--model", "2.4t", "--fsdp", "3", "--seq", "1024", "--cuda-graph", "on", "--local-attention-graph",
        "--results-root", str(tmp_path), "--dry-run",
    )
    assert result.returncode == 0, result.stderr
    (command,) = _commands(result.stderr)
    assert "model.cuda_graph_modules=[attn]" in shlex.split(command)
    assert "-cg_on-partial-mbs_" in command


def test_run_one_cuda_graph_disables_nccl_graph_register_with_expandable_segments(tmp_path: Path):
    # The graph holds NCCL collectives; registering expandable segments can fault.
    result = _run(
        RUN_ONE,
        *("--model", "2.4t", "--seq-length", "1024", "--num-experts", "64", "--cuda-graph", "on"),
        *("--run-date", RUN_DATE, "--results-root", str(tmp_path), "--dry-run"),
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        NCCL_GRAPH_REGISTER="1",
    )
    assert result.returncode == 0, result.stderr
    assert "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in result.stderr.splitlines()
    assert "NCCL_GRAPH_REGISTER=0" in result.stderr.splitlines()


def test_moe_partial_cuda_graph_rejects_unsupported_fsdp_configs(tmp_path: Path):
    common = ("--model", "2.4t", "--seq-length", "1024", "--cuda-graph", "on", "--dry-run")
    for args, message in (
        (("--fsdp", "2"), "require --fsdp 1 or 3"),
        (("--fsdp", "1", "--offload", "weight"), "offload with --cuda-graph on requires --fsdp 3"),
        (("--dtype", "mxfp8"), "require --dtype bf16"),
        (("--tp", "2"), "require --tp 1"),
        (("--symmetric",), "require --symmetric to be off"),
    ):
        result = _run(RUN_ONE, *common, *args, NCCL_ALGO="")
        assert result.returncode == 2, result.stderr
        assert message in result.stderr
    result = _run(
        BENCHMARK,
        "--model", "397b", "--fsdp", "3,2", "--seq", "256", "--cuda-graph", "on",
        "--output-dir", str(tmp_path), "--dry-run",
    )
    assert result.returncode == 2 and "require --fsdp 1 or 3" in result.stderr


def test_ddp_replaces_megatron_fsdp_and_uses_te_per_layer_graphs(tmp_path: Path):
    for model, report_tag, modules in (
        ("27b", "qwen35_27b", r"\[attn\,mlp\]"),
        ("397b", "qwen35_397b", r"\[attn\,moe_router\,moe_preprocess\]"),
        ("2.4t", "qwen38_2.4t", r"\[attn\,moe_router\,moe_preprocess\]"),
    ):
        commands = _sweep(
            tmp_path, "--model", model, "--fsdp", "ddp", "--cuda-graph", "off,on", "--seq", "256", "--experts", "64"
        )
        assert len(commands) == 2
        for command in commands:
            assert f"model_{model}-ddp-sync-ag-layers_" in command and f"/{report_tag}-ddp-sync-ag-experts_" in command
            assert "dist.use_megatron_fsdp=false" in command and "ddp.use_megatron_fsdp=false" in command
            assert "ddp.use_distributed_optimizer=true" in command
            assert "ddp.data_parallel_sharding_strategy=no_shard " in command
            assert "ddp.overlap_grad_reduce=true" in command
            for config in ("ddp", "comm_overlap", "optimizer"):
                assert f"{config}.overlap_param_gather=false" in command
            assert "checkpoint.ckpt_format=torch_dist" in command
        off = next(c for c in commands if "-cg_off-" in c)
        assert "model.cuda_graph_impl=none" in off
        on = next(c for c in commands if "-cg_on-" in c)
        assert "model.cuda_graph_impl=transformer_engine" in on
        assert f"model.cuda_graph_modules={modules}" in on
        assert "model.use_te_rng_tracker=true" in on and "rng.te_rng_tracker=true" in on
        # Dropless MoE stays eager, so none of the FSDP full-iteration graph setup applies.
        for fsdp_only in (
            "moe_expert_rank_capacity_factor",
            "moe_paged_stash",
            "use_transformer_engine_op_fuser",
            "megatron_fsdp_cuda_graph_mode",
            "check_for_nan_in_loss=false",
        ):
            assert fsdp_only not in on


def test_ddp_overlap_enables_forward_parameter_prefetch(tmp_path: Path):
    fsdp2, ddp = _sweep(tmp_path, "--model", "27b", "--fsdp", "2,ddp", "--seq", "256", "--ddp-overlap", "--nsys")
    assert "ddp.overlap_param_gather=false" in fsdp2
    assert "ddp.overlap_param_gather=true" in ddp and "comm_overlap.overlap_param_gather=true" in ddp
    assert "model_27b-ddp-layers_8-" in ddp and "/qwen35_27b-ddp-experts_0-" in ddp

    result = _run(RUN_ONE, "--model", "27b", "--seq-length", "256", "--fsdp", "2", "--ddp-overlap", "--dry-run")
    assert result.returncode == 2 and "--ddp-overlap requires --fsdp ddp" in result.stderr


def test_ddp_cuda_graph_skips_preprocess_with_fp8_padding_on_alltoall(tmp_path: Path):
    common = ("--seq-length", "256", "--num-experts", "64", "--fsdp", "ddp", "--cuda-graph", "on")
    common += ("--dispatcher", "alltoall", "--run-date", RUN_DATE, "--results-root", str(tmp_path), "--dry-run")
    bf16 = _run(RUN_ONE, "--model", "397b", *common, NCCL_GRAPH_REGISTER="1")
    assert bf16.returncode == 0, bf16.stderr
    assert r"model.cuda_graph_modules=\[attn\,moe_router\,moe_preprocess\]" in bf16.stderr
    # TECudaGraphHelper asserts NCCL graph registration is off under expandable segments.
    assert "NCCL_GRAPH_REGISTER=0" in bf16.stderr.splitlines()
    # moe_router_padding_for_fp8 makes the alltoall preprocess sync with the host.
    mxfp8 = _run(RUN_ONE, "--model", "2.4t", "--dtype", "mxfp8", *common)
    assert mxfp8.returncode == 0, mxfp8.stderr
    assert r"model.cuda_graph_modules=\[attn\,moe_router\] " in mxfp8.stderr


def test_overlap_creates_all_gather_group_for_fsdp_cases_only(tmp_path: Path):
    fsdp3, ddp = _sweep(tmp_path, "--model", "27b", "--fsdp", "3,ddp", "--seq", "256", "--overlap", "--nsys")
    assert "model_27b-fsdp_3-overlap-layers_8-" in fsdp3 and "/qwen35_27b-fsdp_3-overlap-experts_0-" in fsdp3
    assert "dist.create_all_gather_group=true" in fsdp3
    assert "model_27b-ddp-sync-ag-layers_8-" in ddp and "dist.create_all_gather_group=false" in ddp
    (default,) = _sweep(tmp_path, "--model", "27b", "--fsdp", "3", "--seq", "256")
    assert "-overlap-" not in default and "dist.create_all_gather_group=false" in default
    result = _run(RUN_ONE, "--model", "27b", "--seq-length", "256", "--fsdp", "ddp", "--overlap", "--dry-run")
    assert result.returncode == 2
    assert "--overlap requires --fsdp 1, 2 or 3" in result.stderr


def test_symmetric_registers_max_pool_double_buffers_for_fsdp_cases_only(tmp_path: Path):
    fsdp3, ddp = _sweep(
        tmp_path, "--model", "2.4t", "--experts", "64", "--fsdp", "3,ddp", "--seq", "256", "--overlap", "--symmetric", "--nsys"
    )
    assert "model_2.4t-fsdp_3-overlap-symm-layers_4-" in fsdp3 and "/qwen38_2.4t-fsdp_3-overlap-symm-experts_64-" in fsdp3
    for override in (
        "ddp.nccl_ub=true",
        "ddp.disable_symmetric_registration=false",
        "ddp.fsdp_double_buffer=true",
        "ddp.megatron_fsdp_max_pool_double_buffer=true",
        "ddp.fsdp_db_use_persist_buf_on_alloc_fail=true",
        "dist.create_all_gather_group=true",
    ):
        assert override in fsdp3
    assert "model_2.4t-ddp-sync-ag-layers_4-" in ddp and "ddp.nccl_ub=false" in ddp
    (default,) = _sweep(tmp_path, "--model", "27b", "--fsdp", "3", "--seq", "256")
    assert "-symm-" not in default and "ddp.nccl_ub=false" in default
    assert "ddp.disable_symmetric_registration" not in default

    common = ("--model", "27b", "--seq-length", "256", "--symmetric", "--dry-run")
    # The multicast symmetric kernels need NVLS even when the shell turned it off.
    result = _run(RUN_ONE, *common, "--results-root", str(tmp_path), NCCL_NVLS_ENABLE="0")
    assert result.returncode == 0, result.stderr
    assert "NCCL_NVLS_ENABLE=1" in result.stderr.splitlines()
    result = _run(RUN_ONE, *common, "--fsdp", "ddp")
    assert result.returncode == 2 and "--symmetric requires --fsdp 1, 2 or 3" in result.stderr
    result = _run(RUN_ONE, *common, NCCL_ALGO="Ring")
    assert result.returncode == 2 and "--symmetric requires NCCL_ALGO to be unset" in result.stderr


def test_force_balance_tags_moe_cases_and_skips_dense(tmp_path: Path):
    dense, moe = _sweep(
        tmp_path, "--model", "27b,397b", "--force-balance", "--experts", "64", "--seq", "256", "--fsdp", "3"
    )
    assert "model_27b-fsdp_3-layers_8-" in dense and "force_load_balancing" not in dense
    assert "model_397b-fsdp_3-fb-layers_4-" in moe and "/qwen35_397b-fsdp_3-fb-experts_" in moe
    assert "model.moe_router_force_load_balancing=true" in moe
    default = _sweep(tmp_path, "--model", "397b", "--experts", "64", "--seq", "256", "--fsdp", "3")
    assert default and all("-fb-" not in c and "force_load_balancing" not in c for c in default)
    result = _run(RUN_ONE, "--model", "27b", "--seq-length", "256", "--force-balance", "--dry-run")
    assert result.returncode == 2
    assert "--force-balance only applies to MoE models" in result.stderr


def test_mxfp8_tags_result_dirs_and_nsys_reports(tmp_path: Path):
    for model, report_tag in (("27b", "qwen35_27b"), ("397b", "qwen35_397b"), ("2.4t", "qwen38_2.4t")):
        bf16 = _sweep(tmp_path, "--model", model)
        assert bf16 and all("-mxfp8-" not in c and "mixed_precision.fp8=" not in c for c in bf16)
        mxfp8 = _sweep(tmp_path, "--model", model, "--mxfp8")
        assert len(mxfp8) == len(bf16)
        for fsdp in (1, 3):
            command = next(c for c in mxfp8 if f"model_{model}-fsdp_{fsdp}-mxfp8-layers_" in c)
            assert f"{report_tag}-fsdp_{fsdp}-mxfp8-experts_" in command
            assert "mixed_precision.fp8_recipe=mxfp8" in command


def test_cpu_grad_accumulation_and_cpu_adamw_launch(tmp_path: Path):
    (grad_command,) = _sweep(tmp_path, "--model", "27b", "--seq", "256", "--fsdp", "2", "--cpu-grad-accumulation")
    grad_args = shlex.split(grad_command.removeprefix("command:"))
    assert "ddp.fsdp_cpu_offload=[gradient]" in grad_args
    assert "ddp.fsdp_cpu_grad_accumulation=true" in grad_args
    assert "ddp.fsdp_cpu_optimizer=true" not in grad_args

    (adam_command,) = _sweep(tmp_path, "--model", "27b", "--seq", "256", "--cpu-optimizer")
    adam_args = shlex.split(adam_command.removeprefix("command:"))
    assert "ddp.fsdp_cpu_offload=[optimizer,gradient]" in adam_args
    assert "ddp.fsdp_cpu_grad_accumulation=true" in adam_args
    assert "ddp.fsdp_cpu_optimizer=true" in adam_args
    assert "ddp.data_parallel_sharding_strategy=optim_grads_params" in adam_args
    assert "-cpu_grad_accum-cpu_adamw" in adam_command


def test_nvme_offload_preserves_the_path_and_enables_nvtx(tmp_path: Path):
    nvme_path = tmp_path / 'swap directory'
    (command,) = _sweep(tmp_path, "--model", "27b", "--seq", "256", "--nvme-path", str(nvme_path))
    args = shlex.split(command.removeprefix("command:"))
    assert f'ddp.fsdp_nvme_path="{nvme_path}"' in args
    assert "ddp.fsdp_cpu_offload=[optimizer,weight,gradient]" in args
    assert "ddp.fsdp_cpu_grad_accumulation=true" in args
    assert "ddp.fsdp_cpu_optimizer=true" in args
    assert "profiling.nvtx_ranges=true" in args
    assert "-cpu_adamw-nvme" in command


def test_nvme_study_defaults_to_target_scratch_and_accepts_overrides(tmp_path: Path):
    script = EXAMPLE / "04-nvme/01_dram_vs_nvme.sh"
    common = ("--model", "27b", "--output-dir", str(tmp_path), "--dry-run")
    for options, env_path, expected_path in (
        ((), "", "/tmp/megatron-fsdp3-nvme"),
        ((), "/scratch/env", "/scratch/env"),
        (("--nvme-path", "/scratch/cli"), "/scratch/env", "/scratch/cli"),
    ):
        result = _run(script, *common, *options, NVME_PATH=env_path)
        assert result.returncode == 0, result.stderr
        dram, nvme = (shlex.split(command) for command in _commands(result.stderr))
        assert not any(arg.startswith("ddp.fsdp_nvme_path=") for arg in dram)
        assert f'ddp.fsdp_nvme_path="{expected_path}"' in nvme


def test_nvme_study_rejects_non_nvme_default_before_nvme_launch(tmp_path: Path):
    script = EXAMPLE / "04-nvme/01_dram_vs_nvme.sh"
    args = ("--model", "27b", "--output-dir", str(tmp_path))
    dry_run = _run(script, *args, "--dry-run", NVME_PATH="")
    assert dry_run.returncode == 0, dry_run.stderr
    dram, nvme = map(_report_base, _commands(dry_run.stderr))
    dram.parent.parent.mkdir(parents=True)
    fake_df = tmp_path / "df"
    fake_df.write_text("#!/bin/sh\nprintf '%s\\n' 'Filesystem 1024-blocks' 'overlay 1000'\n")
    fake_df.chmod(0o755)
    result = _run(
        script, *args,
        PATH=f"{tmp_path}:{os.environ['PATH']}",
        NVME_PATH="",
    )
    assert result.returncode == 2, result.stderr
    assert "Default NVMe path /tmp/megatron-fsdp3-nvme is on overlay" in result.stderr
    assert not nvme.parent.parent.exists()


def test_cpu_offload_rejects_incompatible_fsdp_and_graph_modes():
    for options, expected in (
        (("--fsdp", "1", "--cpu-grad-accumulation"), "requires eager FSDP2/3"),
        (("--fsdp", "2", "--cpu-optimizer"), "require --fsdp 3 and BF16"),
        (("--cuda-graph", "on", "--nvme-path", "/nvme"), "requires eager FSDP2/3"),
    ):
        result = _run(RUN_ONE, "--model", "27b", "--seq", "256", "--dry-run", *options)
        assert result.returncode == 2
        assert expected in result.stderr
