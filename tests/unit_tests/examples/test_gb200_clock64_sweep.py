import csv
import importlib.util
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[3] / "examples/gb200/ncu-torch-all-gather"
pytestmark = pytest.mark.unit


def test_clock64_sweep_uses_standalone_ag_in_all_three_tasks(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT_DIR / "sweep_symm_ag_clock64.sh"),
            "--dry-run",
            "--pause-cycles",
            "0,128",
            "--output-dir",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    commands = [
        shlex.split(line)
        for line in result.stdout.splitlines()
        if line.startswith(("env ", "NCCL_SYM_AG_STMC_PAUSE_CYCLES="))
    ]
    assert len(commands) == 6
    for index, command in enumerate(commands):
        pause = (0, 128)[index % 2]
        assert f"NCCL_SYM_AG_STMC_PAUSE_CYCLES={pause}" in command
        assert "--ag-rs" not in command
        assert command[command.index("--algo") + 1] == "symm"
        lib = Path(command[command.index("--nccl-lib") + 1]).parents[2].name
        if index < 4:
            profile = ("store", "split")[index // 2]
            assert "--baseline-only" in command
            assert lib == f"nccl-2.32.3-pause-{profile}-clock64"
        else:
            assert "--baseline-only" not in command
            assert command[command.index("--replay-mode") + 1] == "application"
            assert command[command.index("--communicator") + 1] == "shmem"
            assert lib == "nccl-2.32.3-pause"
    assert not (tmp_path / "out").exists()


def test_clock64_sweep_runs_both_patches_at_each_pause_with_ag_rs(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT_DIR / "sweep_symm_ag_rs_clock64.sh"),
            "--dry-run",
            "--pause-cycles",
            "0,128",
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    commands = [
        shlex.split(line)
        for line in result.stdout.splitlines()
        if line.startswith(("env ", "NCCL_SYM_AG_STMC_PAUSE_CYCLES="))
    ]
    assert len(commands) == 4
    assert "ncu pause-only baseline" not in result.stdout
    for index, command in enumerate(commands):
        profile = ("store", "split")[index // 2]
        pause = (0, 128)[index % 2]
        assert "--ag-rs" in command
        assert "--baseline-only" in command
        assert command[command.index("--warmup") + 1] == "5"
        assert command[command.index("--iters") + 1] == "1"
        assert f"NCCL_SYM_AG_STMC_PAUSE_CYCLES={pause}" in command
        assert f"NCCL_SYM_CLOCK64_CSV_PROFILE={profile}" in command
        lib = Path(command[command.index("--nccl-lib") + 1]).parents[2].name
        assert lib == f"nccl-2.32.3-pause-{profile}-clock64"
        assert Path(command[command.index("--output-dir") + 1]) == output_dir / profile / f"pause-{pause}cyc"
    assert not output_dir.exists()


@pytest.mark.parametrize("index,kernel", ((0, "AllGather_STMC"), (1, "ReduceScatter_LDMC")))
def test_standalone_clock64_capture_exports_only_its_kernel(
    index: int, kernel: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(device=object))
    spec = importlib.util.spec_from_file_location("symm_clock64_csv", SCRIPT_DIR / "symm_clock64_csv.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    batches = [[[[0] * 11]], [[[0] * 11]]]
    batches[index][0][0] = [0, 0, 1, 33, 2, 12, 20, 16, 7, 12, 10]
    capture = object.__new__(module.Clock64Capture)
    capture.profile = "split"
    capture.kernels = (kernel,)
    capture.pause = 0
    capture.stride = 32
    capture.group_size = 8
    capture.output = tmp_path
    capture.buffer = SimpleNamespace(cpu=lambda: SimpleNamespace(tolist=lambda: batches))
    capture.write(rank=0)

    with (tmp_path / "rank-0.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["kernel"] == kernel
    assert rows[0]["ld_wait_sum"] == "20"
