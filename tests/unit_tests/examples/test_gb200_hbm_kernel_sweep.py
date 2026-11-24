"""Verify CE sweep planning, launch selection, and trace analysis on CPU."""

import csv
import importlib
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest


_EXPERIMENT = Path(__file__).resolve().parents[3] / "examples" / "gb200" / "hbm-conflict"


@pytest.fixture
def experiment_modules(monkeypatch):
    monkeypatch.syspath_prepend(str(_EXPERIMENT))


@pytest.mark.unit
@pytest.mark.parametrize("test_only", [False, True])
def test_sweep_profiles_only_the_selected_shapes_paths_and_gpus(monkeypatch, tmp_path, test_only):
    destination = tmp_path / "dry-run"
    nsys_tmp = tmp_path / "local-nsys"
    monkeypatch.setenv("NSYS_TMPDIR", str(nsys_tmp))
    result = subprocess.run(
        [
            "bash",
            str(_EXPERIMENT / "run_ce_hbm_kernel_sweep.sh"),
            "--gpus",
            "0,1,2,3",
            "--dry-run",
            "--output-dir",
            str(destination),
            *(["--test"] if test_only else []),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    commands = [shlex.split(line) for line in result.stdout.splitlines() if "--mode " in line]
    starts = [shlex.split(line) for line in result.stdout.splitlines() if " --gpu-metrics-frequency=" in line]
    shapes = [8] if test_only else [2**power for power in range(14)]
    assert len(commands) == len(starts) == len(shapes) * 6
    expected_modes = ["gemm", "gemm", "gemm", "gemm-ag-host", "gemm-ag-hbm", "gemm-ag-hbm-unicast"]
    expected_gpus = ["1", "2", "3", "1,2,3", "1,2,3", "1,2,3"]
    for index, m in enumerate(shapes):
        group = commands[6 * index : 6 * index + 6]
        start_group = starts[6 * index : 6 * index + 6]
        assert [cmd[cmd.index("--mode") + 1] for cmd in group] == expected_modes
        for worker_index, (cmd, start) in enumerate(zip(group, start_group, strict=True)):
            assert cmd[cmd.index("--gemm") + 1] == f"[{m},8192,7168]"
            assert cmd[cmd.index("--size-mib") + 1] == ("2048" if m == 8192 and worker_index >= 3 else "1024")
            assert cmd[cmd.index("--gemm-rounds") + 1] == "8"
            assert "--trace=cuda,nvtx" in cmd
            assert "--nvtx-domain-exclude=NCCL" in cmd and "--cuda-trace-all-apis=true" not in cmd
            assert {"--cuda-trace-scope", "--sample", "--cpuctxsw"}.isdisjoint(arg.split("=", 1)[0] for arg in cmd)
            assert {"--sample=none", "--cpuctxsw=none"} <= set(start)
            scratch = next(arg for arg in cmd if arg.startswith("NSYS_TMPDIR="))
            assert Path(scratch.split("=", 1)[1]).parent == nsys_tmp and scratch in start
            assert "--profile-sync-dir" in cmd
            assert f"--gpu-metrics-devices={expected_gpus[worker_index]}" in start
            assert "--gpu-metrics-frequency=200000" in start
    assert not destination.exists()


@pytest.mark.unit
def test_sweep_records_capture_status_without_running_analysis(monkeypatch, tmp_path):
    launcher = tmp_path / "run_ce_hbm_kernel_sweep.sh"
    shutil.copy2(_EXPERIMENT / launcher.name, launcher)
    (tmp_path / "run_ce_hbm_benchmark.sh").write_text(
        """#!/usr/bin/env bash
case_id= output=
while (( $# )); do
    case "$1" in
        --cases) case_id=$2; shift ;;
        --output-dir) output=$2; shift ;;
    esac
    shift
done
[[ -n "$output" ]] || exit 2
mkdir -p "$output"
[[ "$case_id" != task7 ]] || exit 1
touch "$output/mock.nsys-rep"
""",
        encoding="utf-8",
    )
    analysis_called = tmp_path / "analysis-called"
    uv = tmp_path / "uv"
    uv.write_text('#!/usr/bin/env bash\ntouch "$SWEEP_ANALYSIS_CALLED"\nexit 2\n', encoding="utf-8")
    uv.chmod(0o755)
    nsys = tmp_path / "nsys"
    nsys.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    nsys.chmod(0o755)
    monkeypatch.setenv("PATH", os.pathsep.join((str(tmp_path), os.environ["PATH"])))
    monkeypatch.setenv("SWEEP_ANALYSIS_CALLED", str(analysis_called))
    destination = tmp_path / "test-output"
    result = subprocess.run(
        [
            "bash",
            str(launcher),
            "--test",
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
    with (destination / "status.tsv").open(encoding="utf-8") as handle:
        attempts = [
            (row["case"], row["metrics_gpus"], row["size_mib"], row["status"], row["attempt_dir"])
            for row in csv.DictReader(handle, delimiter="\t")
        ]
    assert attempts == [
        ("task1", "1", "1024", "captured", "m0008/task1/gpu1/size-1024mib"),
        ("task1", "2", "1024", "captured", "m0008/task1/gpu2/size-1024mib"),
        ("task1", "3", "1024", "captured", "m0008/task1/gpu3/size-1024mib"),
        ("task3", "1,2,3", "1024", "captured", "m0008/task3/size-1024mib"),
        ("task7", "1,2,3", "1024", "launch_error", "m0008/task7/size-1024mib"),
    ]
    assert len(list(destination.rglob("*.nsys-rep"))) == 4
    assert not analysis_called.exists()
    assert not list(destination.rglob("*.sqlite"))


@pytest.mark.unit
def test_local_overlap_analysis_uses_downloaded_attempt_paths(experiment_modules, monkeypatch, tmp_path):
    analysis = importlib.import_module("analyse_ce_hbm_overlap")
    first = "m0008/task1/gpu1/size-1024mib"
    second = "m0008/task3/size-2048mib"
    failed = "m0008/task7/size-1024mib"
    for path, name in ((first, "task1-gemm"), (second, "task3-gemm-ag-hbm")):
        (tmp_path / path / name).mkdir(parents=True)
    status = tmp_path / "status.tsv"
    capture_status = (
        "m\tcase\tmetrics_gpus\tsize_mib\tstatus\tattempt_dir\n"
        f"8\ttask1\t1\t1024\tcaptured\t{first}\n"
        f"8\ttask3\t1,2,3\t2048\tok\t/remote/results/sweep/{second}\n"
        f"8\ttask7\t1,2,3\t1024\tlaunch_error\t{failed}\n"
    )
    status.write_text(capture_status, encoding="utf-8")
    analyzed = []

    def analyse(*, case_dir, report_gpus, **kwargs):
        analyzed.append((case_dir.parent.relative_to(tmp_path).as_posix(), report_gpus))
        return {"status": "ok" if case_dir.name == "task1-gemm" else "insufficient_overlap"}, []

    monkeypatch.setattr(analysis, "_analyse", analyse)

    assert analysis.main(["--output-dir", str(tmp_path), "--nsys", "nsys"]) == 1
    assert analyzed == [(first, {1}), (second, {1, 2, 3})]
    assert status.read_text(encoding="utf-8") == capture_status
    with (tmp_path / "analysis-status.tsv").open(encoding="utf-8") as handle:
        rows = [(row["attempt_dir"], row["status"]) for row in csv.DictReader(handle, delimiter="\t")]
    assert rows == [(first, "ok"), (second, "insufficient_overlap"), (failed, "launch_error")]


@pytest.mark.unit
def test_local_overlap_export_is_temporary_on_failure(experiment_modules, monkeypatch, tmp_path):
    analysis = importlib.import_module("analyse_ce_hbm_overlap")
    case = tmp_path / "task1-gemm"
    case.mkdir()
    records = {
        0: {
            "mode": "gemm",
            "size_bytes": 64,
            "comm_rounds_measured": 0,
            "gemm_mnk": [8, 8192, 7168],
            "gemm_ms": [[0, 1]],
            "physical_gpu": 1,
        }
    }
    monkeypatch.setattr(analysis, "_read_records", lambda _: records)
    exports = []

    def export(report, *, target, **kwargs):
        exports.append(target)
        target.touch()
        raise subprocess.CalledProcessError(1, "nsys export")

    monkeypatch.setattr(analysis, "_export", export)

    assert analysis.main(["--output-dir", str(tmp_path), "--nsys", "nsys", "--report-gpus", "1"]) == 1
    assert len(exports) == 1 and not exports[0].parent.exists()
    assert not exports[0].resolve().is_relative_to(tmp_path.resolve())
    assert not list(tmp_path.rglob("*.sqlite"))


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
