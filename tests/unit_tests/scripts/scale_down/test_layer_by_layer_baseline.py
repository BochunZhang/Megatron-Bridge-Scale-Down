import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest


BASELINE_DIR = Path(__file__).resolve().parents[4] / "examples/scale-down/02-layer-by-layer/01-baseline"
SWEEP = BASELINE_DIR.parent / "sweep.py"
CASE_08 = "08_param_cpu_act_nvme_opt_cpu_overlap"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("case", "parameter_tier", "activation_tier", "optimizer_tier"),
    [
        ("01_param_gpu_act_gpu_opt_gpu", "gpu", "gpu", "gpu"),
        ("02_param_cpu_act_gpu_opt_gpu", "cpu", "gpu", "gpu"),
        ("03_param_cpu_act_cpu_opt_gpu", "cpu", "cpu", "gpu"),
        ("04_param_cpu_act_cpu_opt_cpu", "cpu", "cpu", "cpu"),
        ("05_param_cpu_act_cpu_opt_cpu_overlap", "cpu", "cpu", "cpu"),
        ("06_param_nvme_act_cpu_opt_cpu_overlap", "nvme", "cpu", "cpu"),
        ("07_param_nvme_act_nvme_opt_cpu_overlap", "nvme", "nvme", "cpu"),
        (CASE_08, "cpu", "nvme", "cpu"),
    ],
)
def test_layer_by_layer_baseline_dry_run(
    tmp_path: Path, case: str, parameter_tier: str, activation_tier: str, optimizer_tier: str
) -> None:
    script = BASELINE_DIR / f"{case}.sh"
    results_root = tmp_path / "results with spaces"
    args = ["bash", str(script), "--output-dir", str(results_root)]
    env = {**os.environ, "PYTHON": sys.executable, "NSYS_BIN": "/example/nsys", "RANK": "3", "MASTER_ADDR": "tray0"}
    env.pop("NVME_PATH", None)
    result = subprocess.run([*args, "--dry-run"], env=env, check=True, capture_output=True, text=True)
    command = shlex.split(result.stderr.strip())
    assert {
        "/example/nsys",
        "--sample=none",
        "--cpuctxsw=system-wide",
        "--gpu-metrics-devices=none",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=4",
        "model.num_layers=8",
        "model.linear_attention_freq=1",
        "dataset.seq_length=4096",
        "train.micro_batch_size=1",
        "train.global_batch_size=16",
        "train.train_iters=10",
        f"layer_by_layer.parameter_tier={parameter_tier}",
        f"layer_by_layer.activation_tier={activation_tier}",
        f"layer_by_layer.optimizer_tier={optimizer_tier}",
        "profiling.profile_step_start=7",
        "profiling.profile_step_end=8",
    }.issubset(command)
    if optimizer_tier == "cpu":
        assert {
            "--trace=cuda,nvtx",
            "profiling.nvtx_ranges=true",
            "layer_by_layer.cpu_optimizer=grace_adam",
            "OMP_NUM_THREADS=32",
            "OMP_DYNAMIC=FALSE",
            "OMP_PROC_BIND=FALSE",
            "layer_by_layer.cpu_threads=32",
        } <= set(command)
    overlap = case.endswith("_overlap")
    assert ("layer_by_layer.overlap_cpu_optimizer=true" in command) is overlap
    graceadam_threads = [arg for arg in command if arg.startswith("MEGATRON_GRACEADAM_OMP_THREADS=")]
    assert graceadam_threads == (["MEGATRON_GRACEADAM_OMP_THREADS=32"] if overlap else [])
    assert all(not arg.startswith("layer_by_layer.backend=") for arg in command)
    nvme_args = [arg for arg in command if arg.startswith("layer_by_layer.nvme_dir=")]
    if "nvme" in (parameter_tier, activation_tier):
        assert nvme_args == ['layer_by_layer.nvme_dir="/tmp/layer-by-layer-8-<unique>"']
    else:
        assert not nvme_args

    result = subprocess.run([*args, "--print-result-dir"], env=env, check=True, capture_output=True, text=True)
    result_path = Path(result.stdout.strip())
    assert result_path.name == "node0"
    case_dir = case.replace("_", "-")
    if case == CASE_08:
        case_dir += "-4-gpus-1-nodes"
    assert result_path.parent.parent == results_root / "01-baseline" / case_dir
    assert result_path.parent.name.startswith("model_27b-lbl-schedule-")
    report = Path(next(arg.removeprefix("--output=") for arg in command if arg.startswith("--output=")))
    assert report.parent == result_path
    if case == CASE_08:
        assert "-gpus_4-nodes_1-" in result_path.parent.name
        assert "-gpus_4-nodes_1-" in report.name
        assert "CUDA_VISIBLE_DEVICES=0,1,2,3" in command
    overlap_suffix = "-opt_overlap" if overlap else ""
    assert report.name.endswith(f"-param_{parameter_tier}-act_{activation_tier}-opt_{optimizer_tier}{overlap_suffix}")
    assert not results_root.exists()


@pytest.mark.unit
@pytest.mark.parametrize(
    "case", ["06_param_nvme_act_cpu_opt_cpu_overlap", "07_param_nvme_act_nvme_opt_cpu_overlap", CASE_08]
)
def test_layer_by_layer_baseline_nvme_path_override(tmp_path: Path, case: str) -> None:
    script = BASELINE_DIR / f"{case}.sh"
    nvme = tmp_path / "shared NVMe"
    result = subprocess.run(
        ["bash", str(script), "--dry-run", "--nvme", str(nvme)],
        env={**os.environ, "PYTHON": sys.executable, "NVME_PATH": "/example/nvme"},
        check=True,
        capture_output=True,
        text=True,
    )
    scratch = nvme / "layer-by-layer-8-<unique>"
    assert f"layer_by_layer.nvme_dir={json.dumps(str(scratch))}" in shlex.split(result.stderr)
    assert not nvme.exists()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("gpu_per_node", "node_count", "use_gpu_per_node", "node_rank"),
    [
        (4, 4, 1, 0),
        (1, 4, 1, 2),
        (4, 4, 2, 3),
        (4, 4, None, 1),
        (4, 1, 2, 0),
        (4, 3, 1, 2),
    ],
)
def test_case_08_uses_selected_gpus_per_tray(
    tmp_path: Path, gpu_per_node: int, node_count: int, use_gpu_per_node: int | None, node_rank: int
) -> None:
    script = BASELINE_DIR / f"{CASE_08}.sh"
    results = tmp_path / "shared results"
    numarun = tmp_path / "numarun"
    numarun.touch(mode=0o755)
    env = {
        **os.environ,
        "PYTHON": sys.executable,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "CUDA_VISIBLE_DEVICES": "3,2,1,0",
    }
    env.pop("NVME_PATH", None)
    args = [
        "bash",
        str(script),
        "--gpu-per-node",
        str(gpu_per_node),
        "--node",
        str(node_count),
        "--output-dir",
        str(results),
    ]
    if use_gpu_per_node is not None:
        args.extend(("--use-gpu-per-node", str(use_gpu_per_node)))
    if node_count > 1:
        args.extend(("--node-rank", str(node_rank), "--master-addr", "tray0", "--master-port", "29531"))
    result = subprocess.run([*args, "--dry-run"], env=env, capture_output=True, text=True, check=True)
    command = shlex.split(result.stderr)
    nproc_per_node = use_gpu_per_node or gpu_per_node
    world_size = nproc_per_node * node_count
    visible_devices = ",".join(str(gpu) for gpu in range(nproc_per_node))
    assert {
        "CUDA_DEVICE_ORDER=PCI_BUS_ID",
        f"CUDA_VISIBLE_DEVICES={visible_devices}",
        f"--nnodes={node_count}",
        f"--nproc-per-node={nproc_per_node}",
        f"train.global_batch_size={4 * world_size}",
        "layer_by_layer.parameter_tier=cpu",
        "layer_by_layer.activation_tier=nvme",
        "layer_by_layer.optimizer_tier=cpu",
        "layer_by_layer.cpu_optimizer=grace_adam",
        "layer_by_layer.overlap_cpu_optimizer=true",
        'layer_by_layer.nvme_dir="/tmp/layer-by-layer-8-<unique>"',
        "profiling.profile_ranks=[]",
    } <= set(command)
    if node_count > 1:
        assert {f"--node-rank={node_rank}", "--master-addr=tray0", "--master-port=29531"} <= set(command)
        assert "--standalone" not in command
    else:
        assert "--standalone" in command
    assert not {"numarun", "CUDA_VISIBLE_DEVICES=3,2,1,0"} & set(command)

    result = subprocess.run([*args, "--print-result-dir"], env=env, capture_output=True, text=True, check=True)
    result_path = Path(result.stdout.strip())
    assert result_path.name == f"node{node_rank}"
    assert f"-gbs_{4 * world_size}-gpus_{world_size}-nodes_{node_count}-" in result_path.parent.name
    case_dir = f"{CASE_08.replace('_', '-')}-{world_size}-gpus-{node_count}-nodes"
    assert result_path.parent.parent == results / "01-baseline" / case_dir
    report = Path(next(arg.removeprefix("--output=") for arg in command if arg.startswith("--output=")))
    assert report.parent == result_path
    assert f"-dp_{world_size}-gpus_{world_size}-nodes_{node_count}-" in report.name
    assert f"-gbs_{4 * world_size}-" in report.name
    assert not results.exists()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("args", "error"),
    [
        (["--node", "0"], "--node must be a positive integer"),
        (["--gpu-per-node", "2", "--use-gpu-per-node", "3"], "--use-gpu-per-node cannot exceed --gpu-per-node"),
        (["--node", "2", "--node-rank", "2"], "a tray rank from 0 to 1"),
    ],
)
def test_case_08_rejects_invalid_topology(args: list[str], error: str) -> None:
    script = BASELINE_DIR / f"{CASE_08}.sh"
    result = subprocess.run(["bash", str(script), "--dry-run", *args], capture_output=True, text=True)
    assert result.returncode == 2
    assert error in result.stderr


@pytest.mark.unit
@pytest.mark.parametrize("case", ["04_param_cpu_act_cpu_opt_cpu", "05_param_cpu_act_cpu_opt_cpu_overlap"])
def test_layer_by_layer_baseline_grace_adam_cpu_threads(case: str) -> None:
    script = BASELINE_DIR / f"{case}.sh"
    result = subprocess.run(
        ["bash", str(script), "--dry-run", "--cpu-threads", "12"],
        env={**os.environ, "PYTHON": sys.executable},
        check=True,
        capture_output=True,
        text=True,
    )
    command = shlex.split(result.stderr)
    assert {"OMP_NUM_THREADS=12", "layer_by_layer.cpu_threads=12"} <= set(command)
    graceadam_threads = [arg for arg in command if arg.startswith("MEGATRON_GRACEADAM_OMP_THREADS=")]
    assert graceadam_threads == (["MEGATRON_GRACEADAM_OMP_THREADS=12"] if case.endswith("_overlap") else [])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("extra_args", "expected"),
    [
        (
            [],
            {
                "layer_by_layer.parameter_tier=gpu",
                "layer_by_layer.activation_tier=gpu",
                "layer_by_layer.optimizer_tier=gpu",
            },
        ),
        (
            ["--parameter-tier", "cpu"],
            {
                "layer_by_layer.parameter_tier=cpu",
                "layer_by_layer.activation_tier=gpu",
                "layer_by_layer.optimizer_tier=gpu",
            },
        ),
        (
            ["--parameter-tier", "cpu", "--activation-tier", "cpu_act"],
            {
                "layer_by_layer.parameter_tier=cpu",
                "layer_by_layer.activation_tier=cpu",
                "layer_by_layer.optimizer_tier=gpu",
            },
        ),
        (
            ["--optimizer-tier", "cpu"],
            {
                "layer_by_layer.parameter_tier=gpu",
                "layer_by_layer.activation_tier=gpu",
                "layer_by_layer.optimizer_tier=cpu",
                "layer_by_layer.cpu_optimizer=torch",
            },
        ),
    ],
)
def test_layer_by_layer_sweep_placements(extra_args: list[str], expected: set[str]) -> None:
    result = subprocess.run(
        [sys.executable, str(SWEEP), "--dry-run", *extra_args], capture_output=True, text=True, check=True
    )
    command = shlex.split(result.stderr)
    assert expected <= set(command)


@pytest.mark.unit
@pytest.mark.parametrize(
    "tiers",
    [{"parameter": "nvme", "optimizer": "nvme"}, {"parameter": "gpu", "activation": "nvme", "optimizer": "gpu"}],
)
def test_layer_by_layer_sweep_nvme_uses_case_scratch(tmp_path: Path, tiers: dict[str, str]) -> None:
    nvme = tmp_path / "nvme with spaces"
    nvme.mkdir()
    tier_args = [arg for component, tier in tiers.items() for arg in (f"--{component}-tier", tier)]
    result = subprocess.run(
        [sys.executable, str(SWEEP), "--dry-run", *tier_args, "--nvme", str(nvme)],
        capture_output=True,
        text=True,
        check=True,
    )
    command = shlex.split(result.stderr)
    assert {f"layer_by_layer.{component}_tier={tier}" for component, tier in tiers.items()} <= set(command)
    scratch = next(
        arg.removeprefix("layer_by_layer.nvme_dir=") for arg in command if arg.startswith("layer_by_layer.nvme_dir=")
    )
    assert Path(json.loads(scratch)).parent.resolve() == nvme.resolve()
    assert not any(nvme.iterdir())


@pytest.mark.unit
@pytest.mark.parametrize(
    "extra_args",
    [
        ["--activation-tier", "nvme"],
        ["--parameter-tier", "nvme", "--nvme", "/tmp"],
        ["--parameter-tier", "nvme", "--optimizer-tier", "cpu"],
        ["--optimizer-tier", "nvme"],
        ["--parameter-tier", "nvme", "--optimizer-tier", "nvme"],
        ["--overlap-cpu-optimizer"],
        ["--overlap-cpu-optimizer", "--optimizer-tier", "cpu", "--cpu-optimizer", "torch"],
    ],
)
def test_layer_by_layer_sweep_rejects_unsupported_placements(extra_args: list[str]) -> None:
    result = subprocess.run([sys.executable, str(SWEEP), "--dry-run", *extra_args], capture_output=True, text=True)
    assert result.returncode == 2


@pytest.mark.unit
@pytest.mark.parametrize(
    ("launch_overlap", "reported_overlap", "clip_grad", "expected_status"),
    [
        (True, True, 0.0, "passed"),
        (True, False, 0.0, "invalid_result"),
        (True, True, 1.0, "invalid_result"),
        (False, True, 0.0, "invalid_result"),
    ],
)
def test_baseline_result_checks_optimizer_overlap(
    tmp_path: Path, launch_overlap: bool, reported_overlap: bool, clip_grad: float, expected_status: str
) -> None:
    verify_baseline_result(
        tmp_path,
        "cpu",
        "cpu",
        {
            "optimizer_tier": "cpu",
            "fsdp_layer_by_layer_cpu_optimizer": True,
            "fsdp_layer_by_layer_overlap_cpu_optimizer": reported_overlap,
            "clip_grad": clip_grad,
        },
        expected_status,
        launch_overlap=launch_overlap,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("nvme_metrics", "expected_status"),
    [
        ({"fsdp_nvme_weights_only": True}, "passed"),
        ({"fsdp_nvme_weights_only": False}, "invalid_result"),
        ({}, "invalid_result"),
    ],
)
@pytest.mark.parametrize("activation_tier", ["cpu", "nvme"])
def test_baseline_result_checks_nvme_weights_only(
    tmp_path: Path, nvme_metrics: dict, expected_status: str, activation_tier: str
) -> None:
    verify_baseline_result(
        tmp_path,
        "nvme",
        activation_tier,
        {
            "optimizer_tier": "cpu",
            "fsdp_layer_by_layer_cpu_optimizer": True,
            "fsdp_layer_by_layer_overlap_cpu_optimizer": True,
            "clip_grad": 0.0,
            **nvme_metrics,
        },
        expected_status,
        launch_overlap=True,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("parameter_tier", "activation_tier", "fsdp_result", "expected_status"),
    [
        ("gpu", "gpu", {}, "passed"),
        ("cpu", "gpu", {}, "passed"),
        ("cpu", "cpu", {}, "passed"),
        ("cpu", "cpu", {"optimizer_tier": "cpu", "fsdp_layer_by_layer_cpu_optimizer": True}, "passed"),
        ("cpu", "nvme", {"optimizer_tier": "cpu", "fsdp_layer_by_layer_cpu_optimizer": True}, "passed"),
        ("cpu", "cpu", {"optimizer_tier": "cpu", "optimizer_backend": "torch"}, "passed"),
        ("cpu", "cpu", {"optimizer_tier": "cpu", "fsdp_layer_by_layer_cpu_optimizer": False}, "invalid_result"),
        (
            "cpu",
            "cpu",
            {
                "optimizer_tier": "cpu",
                "fsdp_layer_by_layer_cpu_optimizer": True,
                "fsdp_cpu_grad_accumulation": True,
            },
            "invalid_result",
        ),
        ("cpu", "cpu", {"activation_tier": "gpu"}, "invalid_result"),
        ("cpu", "gpu", {"schedule": "no_pipeline"}, "invalid_result"),
        ("cpu", "gpu", {"parameter_backend": "layer_by_layer"}, "invalid_result"),
        ("cpu", "gpu", {"fsdp_strategy": "optim_grads"}, "invalid_result"),
        ("cpu", "gpu", {"fsdp_cpu_offload": []}, "invalid_result"),
        ("cpu", "gpu", {"steps": 9}, "invalid_result"),
        ("cpu", "gpu", {"world_size": 1}, "invalid_result"),
    ],
)
def test_baseline_result_checks_native_fsdp_metrics(
    tmp_path: Path, parameter_tier: str, activation_tier: str, fsdp_result: dict, expected_status: str
) -> None:
    verify_baseline_result(tmp_path, parameter_tier, activation_tier, fsdp_result, expected_status)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("node_rank", "profile", "returncode", "has_report", "case_status"),
    [
        (0, "none", 0, False, "invalid_result"),
        (1, "none", 0, False, "worker_completed"),
        (2, "nsys", 0, False, "invalid_profile"),
        (2, "nsys", 0, True, "worker_completed"),
        (3, "none", 1, False, "failed"),
    ],
)
def test_case_08_validates_training_result_on_tray_zero(
    tmp_path: Path, node_rank: int, profile: str, returncode: int, has_report: bool, case_status: str
) -> None:
    state = {
        "results": str(tmp_path),
        "node_rank": node_rank,
        "nnodes": 4,
        "gpu_per_node": 4,
        "world_size": 8,
        "layers": 8,
        "steps": 10,
        "microbatches": 4,
        "micro_batch_size": 1,
        "seq_length": 4096,
        "tiny": False,
        "parameter_tier": "cpu",
        "activation_tier": "nvme",
        "optimizer_tier": "cpu",
        "optimizer_backend": "grace_adam",
        "overlap_cpu_optimizer": True,
        "profile": profile,
        "worker_overrides": {"GLOO_SOCKET_IFNAME": "bond1", "NCCL_SOCKET_IFNAME": "=bond1"},
        "nsys_version": None,
    }
    (tmp_path / "case.json").write_text(json.dumps(state), encoding="utf-8")
    (tmp_path / "train.log").touch()
    if has_report:
        report = tmp_path / (
            "qwen35_27b-lbl-layers_8-attn_full-bf16-cg_off-tp_1-cp_1-pp_1-dp_8-gpus_8-nodes_4"
            "-mbs_1-gbs_32-seq_4k-iters_10-param_cpu-act_nvme-opt_cpu-opt_overlap.nsys-rep"
        )
        report.write_bytes(b"synthetic report")
    result = subprocess.run(
        [sys.executable, str(SWEEP), "--finish-case", str(tmp_path), str(returncode)],
        capture_output=True,
        text=True,
    )
    expected_exit = int(case_status != "worker_completed")
    assert result.returncode == expected_exit, result.stderr
    run_info = (tmp_path / "run_info.txt").read_text(encoding="utf-8").splitlines()
    assert f"case_status={case_status}" in run_info
    assert f"status={expected_exit}" in run_info


@pytest.mark.unit
@pytest.mark.parametrize(("reported_world_size", "expected_status"), [(8, "passed"), (4, "invalid_result")])
def test_case_08_validates_configured_world_size(
    tmp_path: Path, reported_world_size: int, expected_status: str
) -> None:
    verify_baseline_result(
        tmp_path,
        "cpu",
        "nvme",
        {
            "optimizer_tier": "cpu",
            "fsdp_layer_by_layer_cpu_optimizer": True,
            "fsdp_layer_by_layer_overlap_cpu_optimizer": True,
            "clip_grad": 0.0,
            "world_size": reported_world_size,
        },
        expected_status,
        launch_overlap=True,
        launch_world_size=8,
    )


def verify_baseline_result(
    tmp_path: Path,
    parameter_tier: str,
    activation_tier: str,
    fsdp_result: dict,
    expected_status: str,
    *,
    launch_overlap: bool | None = None,
    launch_world_size: int | None = None,
) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    optimizer_tier = fsdp_result.get("optimizer_tier", "gpu")
    cpu_optimizer = fsdp_result.get("optimizer_backend", "grace_adam") if optimizer_tier == "cpu" else None
    expected_offload = ["weight"] if parameter_tier != "gpu" else []
    if cpu_optimizer:
        expected_offload.extend(("gradient", "optimizer"))
    shared = {
        "layers": 8,
        "steps": 10,
        "parameter_tier": parameter_tier,
        "activation_tier": activation_tier,
        "optimizer_tier": optimizer_tier,
    }
    state = {
        **shared,
        "results": str(tmp_path),
        "optimizer_backend": cpu_optimizer,
        "profile": "none",
        "nsys_version": None,
        "worker_overrides": {"GLOO_SOCKET_IFNAME": "eth0", "NCCL_SOCKET_IFNAME": "=eth0"},
    }
    if launch_overlap is not None:
        state["overlap_cpu_optimizer"] = launch_overlap
    if launch_world_size is not None:
        state["world_size"] = launch_world_size
    (state_dir / "case.json").write_text(json.dumps(state), encoding="utf-8")
    metrics = {
        **shared,
        "loss": 7.0,
        "peak_allocated_bytes": 1024,
        "peak_reserved_bytes": 2048,
        "schedule": "layer_by_layer",
        "parameter_backend": "megatron_fsdp",
        "fsdp_strategy": "optim_grads_params",
        "fsdp_cpu_offload": expected_offload,
        "fsdp_cpu_grad_accumulation": cpu_optimizer == "torch",
        "optimizer_backend": cpu_optimizer or "FusedAdam",
        "world_size": 4,
        **fsdp_result,
    }
    (tmp_path / "train.log").write_text(f"LAYER_BY_LAYER_RESULT {json.dumps(metrics)}\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(SWEEP), "--finish-case", str(state_dir), "0"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == int(expected_status != "passed"), result.stderr
    with (tmp_path / "summary.csv").open(newline="", encoding="utf-8") as summary:
        row = next(csv.DictReader(summary))
    assert row["status"] == expected_status
    if expected_status == "passed":
        assert row["activation_tier"] == activation_tier
        assert row["optimizer_backend"] == (cpu_optimizer or "FusedAdam")
        assert row["overlap_cpu_optimizer"] == str(bool(launch_overlap))
        assert row["fsdp_cpu_offload"] == ",".join(sorted(expected_offload))
        assert row["peak_allocated_bytes"] == "1024"
        assert row["world_size"] == str(launch_world_size or 4)
