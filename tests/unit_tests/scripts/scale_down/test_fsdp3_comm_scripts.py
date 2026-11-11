# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Dry-run tests for 01-analyse/04-fsdp3-comm."""

import os
import shlex
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
        "model_2.4t-fsdp_3-layers_4-mtp_1-experts_64-cg_on-partial-mbs_1-seq_4096-cp_1-ep_4-"
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
            assert "-cg_on-partial-" in command
            assert "model.cuda_graph_impl=local" in command
            assert r"model.cuda_graph_modules=\[attn\]" in command
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


def test_default_moe_sweep_keeps_fsdp1_for_graph_off_only(tmp_path: Path):
    commands = _sweep(tmp_path, "--model", "397b", "--experts", "64", "--seq", "1024", "--cuda-graph", "off,on")
    assert len(commands) == 3
    assert sum("model_397b-fsdp_1-" in command and "-cg_off-" in command for command in commands) == 1
    assert sum("-cg_on-partial-" in command for command in commands) == 1


def test_moe_fsdp3_offload_allows_attention_graph(tmp_path: Path):
    (command,) = _sweep(
        tmp_path,
        "--model", "397b", "--fsdp", "3", "--seq", "1024", "--experts", "64",
        "--offload", "weight,gradient,optimizer", "--cuda-graph", "on",
    )
    args = shlex.split(command)
    assert "ddp.fsdp_cpu_offload=[optimizer,weight,gradient]" in args
    assert "model.cuda_graph_impl=local" in args
    assert "model.cuda_graph_modules=[attn]" in args
    assert "ddp.fsdp_all_gather_in_start_param_sync=true" in args
    assert "ddp.megatron_fsdp_cuda_graph_mode=true" not in args
    assert "optimizer.optimizer_cuda_graph=false" in args


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
        (("--fsdp", "1"), "require --fsdp 3"),
        (("--fsdp", "2"), "require --fsdp 3"),
        (("--dtype", "mxfp8"), "require --dtype bf16"),
        (("--tp", "2"), "require --tp 1 and --cp 1"),
        (("--cp", "2"), "require --tp 1 and --cp 1"),
        (("--symmetric",), "require --symmetric to be off"),
    ):
        result = _run(RUN_ONE, *common, *args, NCCL_ALGO="")
        assert result.returncode == 2, result.stderr
        assert message in result.stderr
    result = _run(
        BENCHMARK,
        "--model", "397b", "--fsdp", "3,1", "--seq", "256", "--cuda-graph", "on",
        "--output-dir", str(tmp_path), "--dry-run",
    )
    assert result.returncode == 2 and "require --fsdp 3" in result.stderr


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
