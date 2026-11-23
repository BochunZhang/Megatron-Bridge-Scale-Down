"""Verify adaptive AG sizing and same-round GEMM coverage on CPU."""

import csv
import importlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


_EXPERIMENT = Path(__file__).resolve().parents[3] / "examples" / "gb200" / "hbm-conflict"


@pytest.fixture
def experiment_modules(monkeypatch):
    monkeypatch.syspath_prepend(str(_EXPERIMENT))


@pytest.mark.unit
def test_default_sweep_prints_four_ag_paths_and_one_baseline_per_shape(tmp_path):
    destination = tmp_path / "dry-run"
    result = subprocess.run(
        [
            "bash",
            str(_EXPERIMENT / "run_ce_hbm_kernel_sweep.sh"),
            "--gpus",
            "0,1,2,3",
            "--dry-run",
            "--output-dir",
            str(destination),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    commands = [shlex.split(line) for line in result.stdout.splitlines() if "--mode " in line]
    assert len(commands) == 70
    expected_modes = ["gemm", "gemm-ag-hbm", "gemm-ag-host", "gemm-ag-hbm-unicast", "gemm-ag-host-unicast"]
    for index, m in enumerate(2**power for power in range(14)):
        group = commands[5 * index : 5 * index + 5]
        assert [cmd[cmd.index("--mode") + 1] for cmd in group] == expected_modes
        for cmd in group:
            assert cmd[cmd.index("--gemm") + 1] == f"[{m},8192,7168]"
            assert cmd[cmd.index("--size-mib") + 1] == "1024"
            assert cmd[cmd.index("--gemm-rounds") + 1] == "16"
            assert "--trace=cuda,nvtx" in cmd
            assert "--capture-range=none" in cmd
            assert not any(part.startswith("--gpu-metrics") for part in cmd)
    assert not destination.exists()


@pytest.mark.unit
def test_bash_retries_only_when_trace_has_too_few_gemms_in_a_round(monkeypatch, tmp_path):
    launcher = tmp_path / "run_ce_hbm_kernel_sweep.sh"
    shutil.copy2(_EXPERIMENT / launcher.name, launcher)
    (tmp_path / "run_ce_hbm_benchmark.sh").write_text(
        """#!/usr/bin/env bash
while (( $# )); do
    if [[ "$1" == --output-dir ]]; then mkdir -p "$2"; exit; fi
    shift
done
exit 1
""",
        encoding="utf-8",
    )
    (tmp_path / "analyse_ce_hbm_kernel_trace.py").write_text(
        """import json
import sys
from pathlib import Path

action = sys.argv[1]
opts = dict(zip(sys.argv[2::2], sys.argv[3::2], strict=True))
out = Path(opts["--output-dir"])
if action == "report":
    assert (out / "m8192").is_dir()
    sys.exit(0)
assert action == "attempt"
case, size = opts["--case"], int(opts["--size-mib"])
status, code = "ok", 0
if case == "task3" and size == 1024:
    status, code = "insufficient_overlap", 3
if case == "task7":
    status, code = "error", 1
(out / "attempt.json").write_text(json.dumps({"case": case, "size_mib": size, "status": status}))
sys.exit(code)
""",
        encoding="utf-8",
    )
    uv = tmp_path / "uv"
    uv.write_text(
        """#!/usr/bin/env bash
[[ "$1" == run ]] || exit 2
shift
while [[ "$1" == --* ]]; do shift; done
[[ "$1" == python ]] || exit 2
shift
exec "$SWEEP_TEST_PYTHON" "$@"
""",
        encoding="utf-8",
    )
    uv.chmod(0o755)
    nsys = tmp_path / "nsys"
    nsys.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    nsys.chmod(0o755)
    monkeypatch.setenv("PATH", os.pathsep.join((str(tmp_path), os.environ["PATH"])))
    monkeypatch.setenv("SWEEP_TEST_PYTHON", sys.executable)
    destination = tmp_path / "test-output"
    result = subprocess.run(
        [
            "bash",
            str(launcher),
            "--m-values",
            "8192",
            "--cases",
            "task3,task7",
            "--nsys",
            str(nsys),
            "--output-dir",
            str(destination),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1, result.stderr
    attempts = sorted(
        (attempt["case"], attempt["size_mib"], attempt["status"])
        for path in destination.rglob("attempt.json")
        for attempt in [json.loads(path.read_text(encoding="utf-8"))]
    )
    assert attempts == [
        ("task1", 1024, "ok"),
        ("task3", 1024, "insufficient_overlap"),
        ("task3", 2048, "ok"),
        ("task7", 1024, "error"),
    ]


@pytest.mark.unit
def test_touching_ag_rounds_do_not_count_as_one(experiment_modules, monkeypatch, tmp_path):
    analysis = importlib.import_module("analyse_ce_hbm_kernel_trace")
    case_dir = tmp_path / "task7-gemm-ag-hbm-unicast"
    records = {
        rank: {
            "rank": rank,
            "mode": "gemm-ag-hbm-unicast",
            "physical_gpu": rank,
            "size_bytes": 64,
            "world_size": 2,
            "gemm_mnk": [2, 8192, 7168],
            "gemm_ms": [[0.001, 0.002]] * 3,
            "comm_ms": [[0.001, 0.002]] * 2,
            "comm_source_bytes": 32,
            "comm_rounds_measured": 2,
        }
        for rank in range(2)
    }
    captured = {}
    for rank in records:
        ops = {
            f"gemm{index:02d}": [analysis.Activity(start, end, rank, "gemm", 0, "", "", "")]
            for index, (start, end) in enumerate(((110, 140), (190, 210), (220, 240)))
        }
        for index, (start, end) in enumerate(((100, 200), (200, 300))):
            ops[f"comm{index:02d}"] = [
                analysis.Activity(start, end, rank, "MEMCPY", 32, "Peer-to-Peer", "Device", "Device")
            ]
        captured[rank] = ops, {}
    monkeypatch.setattr(analysis, "_read_records", lambda directory: records)
    monkeypatch.setattr(analysis, "_export_kernel_trace", lambda report, **kwargs: report.with_suffix(".sqlite"))
    monkeypatch.setattr(analysis, "_read_combined", lambda db, **kwargs: captured)

    result = analysis.analyse_kernel_case(case_dir, nsys_bin="nsys")

    assert [gemm["ag_rounds"] for gemm in result["gpu0_gemms"]] == [[0], [], [1]]
    assert result["gpu0_covered_gemm_count"] == 2
    assert result["gpu0_max_gemms_per_ag_round"] == 1


@pytest.mark.unit
def test_ncu_summary_distinguishes_missing_link_samples_from_zero(experiment_modules, tmp_path):
    ncu = importlib.import_module("analyse_ncu_hbm_gemm")
    count = "dram__sectors_read.sum"
    link = "nvlrx__bytes.sum"
    raw = tmp_path / "gemm-raw.csv"
    with raw.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(
            [
                ["Kernel Name", ncu.DURATION, count, link],
                ["", "nsecond", "sector", "byte"],
                ["gemm00", "2000", "4", "0"],
                ["gemm01", "1000", "8", "0"],
            ]
        )
    counters = {
        "dram_read": ncu.Counter(count, 32),
        "nvlink_rx_wire": ncu.Counter(f"pmsampling:{link}", 1),
        "c2c_rx_user": ncu.Counter("pmsampling:ctc__rx_bytes_data_user.sum", 1),
    }

    summary = ncu._summary(raw, counters, rounds=2)

    assert summary["mean_ncu_kernel_us"] == 1.5
    assert summary["traffic"]["dram_read"]["mean_bytes"] == 192
    assert summary["traffic"]["dram_read"]["total_bytes"] == 384
    assert summary["traffic"]["dram_read"]["mean_gb_per_s"] == pytest.approx(0.16)
    assert summary["traffic"]["nvlink_rx_wire"]["total_bytes"] == 0
    assert summary["traffic"]["c2c_rx_user"]["total_bytes"] is None


@pytest.mark.unit
@pytest.mark.parametrize("ncu_source", ["repo", "env", "argument"])
def test_ncu_bash_launches_torch_distributed_worker(tmp_path, monkeypatch, ncu_source):
    destination = tmp_path / "ncu-dry-run"
    ncu_options = []
    ncu_path = _EXPERIMENT.parents[2] / "NVIDIA-Nsight-Compute-2026.4" / "ncu"
    monkeypatch.delenv("NCU_BIN", raising=False)
    if ncu_source in ("env", "argument"):
        ncu_path = tmp_path / "ncu from env"
        monkeypatch.setenv("NCU_BIN", str(ncu_path))
    if ncu_source == "argument":
        ncu_path = tmp_path / "ncu from argument"
        ncu_options.extend(("--ncu", str(ncu_path)))
    result = subprocess.run(
        [
            "bash",
            str(_EXPERIMENT / "run_ncu_hbm_gemm_sweep.sh"),
            "--gpu",
            "3",
            "--m-values",
            "1,8192",
            "--dry-run",
            "--output-dir",
            str(destination),
            *ncu_options,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    commands = [shlex.split(line) for line in result.stdout.splitlines() if "--nvtx-include" in line]
    assert len(commands) == 2
    for command, m in zip(commands, (1, 8192)):
        assert command[command.index("--replay-mode") - 1] == str(ncu_path)
        assert "CUDA_VISIBLE_DEVICES=3" in command
        assert "--ncu-gemm" in command
        assert command[command.index("--pm-sampling-interval") + 1] == "1000"
        assert command[command.index("--gemm") + 1] == f"[{m},8192,7168]"
        assert command[command.index("-m") + 1] == "torch.distributed.run"
        assert "uv" in command
    assert not destination.exists()
