from __future__ import annotations

import json
import logging
import os
import platform
import runpy
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


pytestmark = pytest.mark.unit
EXAMPLE = Path(__file__).resolve().parents[3] / "examples/scale-down/01-analyse/02-deepspeed-offload"
SVE_REQUIREMENT = "--require-cpu-adam-sve"
MATRIX_ARGS = [
    "--models",
    "Qwen/Qwen3.5-9B-Base",
    "--micro_batch_sizes",
    "1",
    "--recompute_combos",
    "recompute_none",
    "--param_positions",
    "param_gpu param_cpu",
    "--optimizer_strategies",
    "zero_3 zero_offload super_offload_1.0",
]


@pytest.fixture
def cpu_adam_builder(monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], SimpleNamespace]:
    loads = []
    extension = SimpleNamespace(
        __file__="/fake/cpu_adam_op.so", adam_copy_supported=lambda: False, adam_grad_scale_supported=lambda: True
    )

    def load(*args: object, **kwargs: object) -> SimpleNamespace:
        loads.append("cpu_adam")
        return extension

    deepspeed = ModuleType("deepspeed")
    deepspeed.__version__ = "0.20.1"
    deepspeed.__git_hash__ = "synthetic-development-sha"
    monkeypatch.setitem(sys.modules, deepspeed.__name__, deepspeed)
    builder_module = ModuleType("deepspeed.ops.op_builder")
    builder_module.CPUAdamBuilder = lambda: SimpleNamespace(load=load)
    monkeypatch.setitem(sys.modules, builder_module.__name__, builder_module)
    return loads, extension


def _mock_host(monkeypatch: pytest.MonkeyPatch, *, system: str, machine: str, features: str) -> None:
    real_read_text = Path.read_text

    def read_text(path: Path, *args: object, **kwargs: object) -> str:
        if path == Path("/proc/cpuinfo"):
            return f"processor\t: 0\nFeatures\t: {features}\n"
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(platform, "system", lambda: system)
    monkeypatch.setattr(platform, "machine", lambda: machine)
    monkeypatch.setattr(Path, "read_text", read_text)


def test_accepts_linux_arm_sve_extension(
    monkeypatch: pytest.MonkeyPatch,
    cpu_adam_builder: tuple[list[str], SimpleNamespace],
    caplog: pytest.LogCaptureFixture,
) -> None:
    _mock_host(monkeypatch, system="Linux", machine="aarch64", features="fp asimd sve sve2")
    verify = runpy.run_path(str(EXAMPLE / "cpu_adam_sve.py"))["verify_cpu_adam_sve"]

    with caplog.at_level(logging.INFO):
        verify()

    assert cpu_adam_builder[0] == ["cpu_adam"]
    assert "CPU Adam kernel: ARM SVE" in caplog.text
    assert "cpu_adam_op.so" in caplog.text


@pytest.mark.parametrize(
    ("system", "machine", "features"),
    [("Darwin", "arm64", "sve"), ("Linux", "aarch64", "fp asimd sve2"), ("Linux", "x86_64", "sve")],
)
def test_rejects_non_sve_hosts_before_loading_extension(
    monkeypatch: pytest.MonkeyPatch,
    cpu_adam_builder: tuple[list[str], SimpleNamespace],
    system: str,
    machine: str,
    features: str,
) -> None:
    _mock_host(monkeypatch, system=system, machine=machine, features=features)
    verify = runpy.run_path(str(EXAMPLE / "cpu_adam_sve.py"))["verify_cpu_adam_sve"]

    with pytest.raises(RuntimeError):
        verify()

    assert cpu_adam_builder[0] == []


@pytest.mark.parametrize("extension_kind", ["legacy", "copy_supported", "without_grad_scale"])
def test_rejects_wrong_loaded_extension(
    monkeypatch: pytest.MonkeyPatch,
    cpu_adam_builder: tuple[list[str], SimpleNamespace],
    extension_kind: str,
) -> None:
    _mock_host(monkeypatch, system="Linux", machine="aarch64", features="fp asimd sve")
    loads, extension = cpu_adam_builder
    if extension_kind == "legacy":
        del extension.adam_grad_scale_supported
    elif extension_kind == "copy_supported":
        extension.adam_copy_supported = lambda: True
    else:
        extension.adam_grad_scale_supported = lambda: False
    verify = runpy.run_path(str(EXAMPLE / "cpu_adam_sve.py"))["verify_cpu_adam_sve"]

    with pytest.raises(RuntimeError):
        verify()

    assert loads == ["cpu_adam"]


@pytest.fixture
def launcher_env(tmp_path: Path) -> dict[str, str]:
    python = tmp_path / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "if sys.argv[1] == '-':\n"
        "    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
        "with open(os.environ['TEST_LAUNCH_LOG'], 'a', encoding='utf-8') as log:\n"
        "    log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if sys.argv[1:3] == ['-m', 'deepspeed.launcher.runner']:\n"
        "    sys.exit(0)\n"
        "if os.path.basename(sys.argv[1]) == 'cpu_adam_sve.py':\n"
        "    sys.exit(int(os.environ.get('TEST_SVE_PREFLIGHT_RC', '0')))\n"
        "sys.exit(99)\n"
    )
    python.chmod(0o755)
    return {
        **os.environ,
        "PYTHON_BIN": str(python),
        "REPO_ROOT": str(tmp_path),
        "RESULTS_ROOT": str(tmp_path / "results"),
        "TMP_CONFIG_DIR": str(tmp_path / "configs"),
        "TEST_LAUNCH_LOG": str(tmp_path / "launch.jsonl"),
    }


def _run_matrix(launcher_env: dict[str, str], *options: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(EXAMPLE / "pretrain_experiment.sh"), *MATRIX_ARGS, *options],
        env=launcher_env,
        capture_output=True,
        text=True,
    )


def _python_invocations(launcher_env: dict[str, str]) -> list[list[str]]:
    return [json.loads(line) for line in Path(launcher_env["TEST_LAUNCH_LOG"]).read_text().splitlines()]


def test_dry_run_declares_sve_without_running_preflight(launcher_env: dict[str, str]) -> None:
    result = _run_matrix(launcher_env, "--dry_run")

    assert result.returncode == 0, result.stderr
    assert "CPU Adam kernel: ARM SVE (required)" in result.stdout
    assert not Path(launcher_env["TEST_LAUNCH_LOG"]).exists()
    assert len(list(Path(launcher_env["RESULTS_ROOT"]).glob("**/ds_config.json"))) == 6


def test_matrix_launches_with_the_verified_interpreter_and_requires_sve_for_cpu_offload(
    launcher_env: dict[str, str],
) -> None:
    result = _run_matrix(launcher_env)

    assert result.returncode == 0, result.stderr
    preflight, *launches = _python_invocations(launcher_env)
    assert Path(preflight[0]).name == "cpu_adam_sve.py"
    assert len(launches) == 6
    for args in launches:
        assert args[:2] == ["-m", "deepspeed.launcher.runner"]
        config = json.loads(Path(args[args.index("--deepspeed_config") + 1]).read_text())
        cpu_optimizer = config["zero_optimization"].get("offload_optimizer", {}).get("device") == "cpu"
        assert (SVE_REQUIREMENT in args) == cpu_optimizer
    assert sum(SVE_REQUIREMENT in args for args in launches) == 4


def test_matrix_aborts_before_training_if_sve_preflight_fails(launcher_env: dict[str, str]) -> None:
    result = _run_matrix({**launcher_env, "TEST_SVE_PREFLIGHT_RC": "7"})

    assert result.returncode != 0
    (preflight,) = _python_invocations(launcher_env)
    assert Path(preflight[0]).name == "cpu_adam_sve.py"


def test_gpu_optimizer_only_skips_preflight_and_uses_new_result_root(launcher_env: dict[str, str]) -> None:
    launcher_env.pop("RESULTS_ROOT")
    result = _run_matrix(launcher_env, "--optimizer_strategies", "zero_3", "--param_positions", "param_cpu")

    assert result.returncode == 0, result.stderr
    (launch,) = _python_invocations(launcher_env)
    assert launch[:2] == ["-m", "deepspeed.launcher.runner"]
    assert SVE_REQUIREMENT not in launch
    new_root = Path(launcher_env["REPO_ROOT"]) / "results/01-analyse/02-deepspeed-sve"
    assert len(list(new_root.glob("**/ds_config.json"))) == 1
