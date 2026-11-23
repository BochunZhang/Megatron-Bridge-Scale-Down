"""Verify adaptive AG sizing and same-round GEMM coverage on CPU."""

import csv
import importlib
import shlex
import subprocess
import sys
from pathlib import Path

import pytest


_EXPERIMENT = Path(__file__).resolve().parents[3] / "examples" / "gb200" / "hbm-conflict"


@pytest.fixture
def sweep(monkeypatch):
    monkeypatch.syspath_prepend(str(_EXPERIMENT))
    return importlib.import_module("run_ce_hbm_kernel_sweep")


@pytest.mark.unit
def test_default_sweep_prints_four_ag_paths_and_one_baseline_per_shape(tmp_path):
    destination = tmp_path / "dry-run"
    result = subprocess.run(
        [
            sys.executable,
            str(_EXPERIMENT / "run_ce_hbm_kernel_sweep.py"),
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
def test_doubles_only_after_valid_trace_has_too_few_gemms_in_a_round(sweep, monkeypatch, tmp_path):
    args = sweep._parse_args(["--m-values", "8192", "--max-size-mib", "4096"])

    def mock_launch(args, *, case, m, size_mib, output_dir):
        (output_dir / f"{case}-gemm-ag-hbm").mkdir(parents=True)

    def mock_trace(case_dir, *, nsys_bin):
        size = int(case_dir.parent.name.removeprefix("size-").removesuffix("mib"))
        return {
            "size_bytes": size << 20,
            "gemm_mnk": [8192, 8192, 7168],
            "gpu0_gemm_count": 16,
            "comm_rounds_measured": 25,
            "gpu0_max_gemms_per_ag_round": 1 if size == 1024 else 2,
        }

    monkeypatch.setattr(sweep, "_launch", mock_launch)
    monkeypatch.setattr(sweep, "analyse_kernel_case", mock_trace)
    result = sweep._run_shape(args, case="task3", m=8192, root=tmp_path)
    assert [(row["size_mib"], row["status"]) for row in result["attempts"]] == [
        (1024, "insufficient_overlap"),
        (2048, "ok"),
    ]

    def missing_trace(case_dir, *, nsys_bin):
        raise ValueError("incomplete CUDA/NVTX capture")

    monkeypatch.setattr(sweep, "analyse_kernel_case", missing_trace)
    result = sweep._run_shape(args, case="task7", m=8192, root=tmp_path)
    assert [(row["size_mib"], row["status"]) for row in result["attempts"]] == [(1024, "error")]


@pytest.mark.unit
def test_touching_ag_rounds_do_not_count_as_one(sweep, monkeypatch, tmp_path):
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
def test_ncu_summary_distinguishes_missing_link_samples_from_zero(sweep, tmp_path):
    ncu = importlib.import_module("run_ncu_hbm_gemm_sweep")
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

    args = ncu._parse_args(["--m-values", "2"])
    command = ncu._command(args, case_dir=tmp_path, m=2, metrics=counters)
    assert "--ncu-gemm" in command
    assert command[command.index("--pm-sampling-interval") + 1] == "1000"
