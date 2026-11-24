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
@pytest.mark.parametrize(("test_only", "allgather_gpus"), [(False, "0,1,2,3"), (True, "4,2,5,6")])
def test_sweep_profiles_only_the_selected_shapes_paths_and_gpus(monkeypatch, tmp_path, test_only, allgather_gpus):
    destination = tmp_path / "dry-run"
    nsys_tmp = tmp_path / "local-nsys"
    monkeypatch.setenv("NSYS_TMPDIR", str(nsys_tmp))
    result = subprocess.run(
        [
            "bash",
            str(_EXPERIMENT / "run_ce_hbm_kernel_sweep.sh"),
            "--gpus",
            allgather_gpus,
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
    expected_modes = ["gemm", "gemm-ag-host", "gemm-ag-hbm", "gemm-ag-hbm-unicast"]
    cases_per_shape = len(expected_modes)
    assert len(commands) == len(starts) == len(shapes) * cases_per_shape
    for index, m in enumerate(shapes):
        offset = index * cases_per_shape
        group = commands[offset : offset + cases_per_shape]
        start_group = starts[offset : offset + cases_per_shape]
        assert [cmd[cmd.index("--mode") + 1] for cmd in group] == expected_modes
        for worker_index, (cmd, start) in enumerate(zip(group, start_group, strict=True)):
            overlap = worker_index != 0
            assert cmd[cmd.index("--gemm") + 1] == f"[{m},8192,7168]"
            assert cmd[cmd.index("--size-mib") + 1] == ("2048" if m == 8192 and overlap else "1024")
            assert f"CUDA_VISIBLE_DEVICES={allgather_gpus if overlap else '2'}" in cmd
            assert f"--nproc_per_node={4 if overlap else 1}" in cmd
            assert cmd[cmd.index("--gemm-rounds") + 1] == "8"
            assert "--trace=cuda,nvtx" in cmd
            assert "--nvtx-domain-exclude=NCCL" in cmd and "--cuda-trace-all-apis=true" not in cmd
            assert {"--cuda-trace-scope", "--sample", "--cpuctxsw"}.isdisjoint(arg.split("=", 1)[0] for arg in cmd)
            assert {"--sample=none", "--cpuctxsw=none"} <= set(start)
            scratch = next(arg for arg in cmd if arg.startswith("NSYS_TMPDIR="))
            assert Path(scratch.split("=", 1)[1]).parent == nsys_tmp and scratch in start
            assert "--profile-sync-dir" in cmd
            assert "--gpu-metrics-devices=2" in start
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
        ("task1", "2", "1024", "captured", "m0008/task1/gpu2/size-1024mib"),
        ("task3", "2", "1024", "captured", "m0008/task3/size-1024mib"),
        ("task7", "2", "1024", "launch_error", "m0008/task7/size-1024mib"),
    ]
    assert len(list(destination.rglob("*.nsys-rep"))) == 2
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
def test_multicast_uses_all_dma_sources_with_one_sampled_gpu(experiment_modules, monkeypatch, tmp_path):
    analysis = importlib.import_module("analyse_ce_hbm_overlap")
    shard_bytes = 256
    source_windows = [(15_000, 30_000), (25_000, 45_000), (50_000, 65_000), (50_000, 65_000)]
    gpu2_samples = {
        2: [
            analysis.Sample(time, metric, 1.0)
            for time in range(15_000, 45_001, 5_000)
            for metric in analysis._RAW_METRICS
        ]
    }
    records = {}
    captures = {}
    channels = {}
    for rank, (start, end) in enumerate(source_windows):
        records[rank] = {
            "rank": rank,
            "physical_gpu": rank,
            "mode": "gemm-ag-hbm",
            "size_bytes": shard_bytes * len(source_windows),
            "comm_rounds_measured": 1,
            "all_gather_out_of_place": True,
            "ce_lead_in_rounds": 1,
            "gemm_mnk": [8, 8192, 7168],
            "gemm_ms": [[0.02, 0.04]],
            "comm_ms": [[start / 1e6, end / 1e6]],
        }
        gemm = analysis.Activity(20_000, 40_000, rank, "GEMM", 0, "", "", "")
        copy = analysis.Activity(start, end, rank, "MEMCPY", shard_bytes, "Device-to-Device", "Device", "Device", rank)
        captures[rank] = {"gemm00": [gemm], "comm00": [copy]}, gpu2_samples if rank == 2 else {}
        channels[(copy.pid, copy.device, copy.start, copy.end, copy.size_bytes)] = {"Async Memcpy"}
    monkeypatch.setattr(analysis, "_read_records", lambda path: records)
    monkeypatch.setattr(analysis, "_export", lambda report, **kwargs: None)
    monkeypatch.setattr(analysis, "_read_combined", lambda database, **kwargs: captures)
    monkeypatch.setattr(analysis, "_read_hardware_channels", lambda report, **kwargs: channels)

    summary, rows = analysis._analyse(
        case_dir=tmp_path, nsys_bin="nsys", report_gpus={2}, dump_timeline="DumpTimeline"
    )

    assert summary["status"] == "ok"
    assert summary["multicast_coverage_source_gpus"] == [0, 1, 2, 3]
    assert summary["hardware_channels_verified"] is True
    assert {gpu["gpu"] for gpu in summary["gpus"]} == {2}
    assert summary["gpus"][0]["gemms"][0]["coverage"] == 1.0
    assert len(rows) == 4 and {row["gpu"] for row in rows} == {2}


@pytest.mark.unit
def test_overlap_reports_traffic_bytes_rates_and_utilization(experiment_modules):
    analysis = importlib.import_module("analyse_ce_hbm_overlap")
    counters = {
        "SM Active": 20,
        "DRAM Read": 62_500,
        "DRAM Write": 31_250,
        "C2C RX User Data": 200_000,
        "C2C TX User Data": 100_000,
        "NVLink RX Requests User Data": 110_000,
        "NVLink RX Responses User Data": 190_000,
        "NVLink TX Requests User Data": 70_000,
        "NVLink TX Responses User Data": 180_000,
    }
    readings = {
        20_000: counters,
        24_000: {name: 2 * value for name, value in counters.items()} | {"SM Active": 80},
        30_000: counters,
    }
    windows = [(20_000, 30_000)]
    rows = analysis._gemm_samples(readings, gpu=2, gemm=0, windows=windows)
    assert [(row["start_ns"], row["end_ns"]) for row in rows] == [(20_000, 24_000), (24_000, 30_000)]
    assert [row["sm_active_pct"] for row in rows] == [20, 80]

    summary = analysis._weighted_metrics(rows)
    assert summary["sm_active_pct"] == pytest.approx(56)
    expected_traffic = {
        "dram_read": (2_000_000, 8_000),
        "dram_write": (1_000_000, 8_000),
        "dram_total": (3_000_000, 8_000),
        "c2c_rx_user": (200_000, 225),
        "c2c_tx_user": (100_000, 225),
        "c2c_total_user": (300_000, 450),
        "nvlink_rx_user": (300_000, 900),
        "nvlink_tx_user": (250_000, 900),
        "nvlink_total_user": (550_000, 1_800),
    }
    for metrics, duration_ns, count_multiplier in [(rows[0], 4_000, 1), (rows[1], 6_000, 2), (summary, 10_000, 3)]:
        for stem, (first_sample_bytes, nominal_gbps) in expected_traffic.items():
            count_bytes = first_sample_bytes * count_multiplier
            expected_gbps = count_bytes / duration_ns
            assert metrics[f"{stem}_bytes"] == count_bytes
            assert metrics[f"{stem}_gbps"] == pytest.approx(expected_gbps)
            assert metrics[f"{stem}_pct_of_nominal"] == pytest.approx(100 * expected_gbps / nominal_gbps)
    assert rows[0].keys() <= set(analysis._CSV_COLUMNS)

    no_samples = analysis._gemm_samples(readings, gpu=2, gemm=0, windows=[(20_001, 29_999)])
    assert no_samples == []
    assert analysis._weighted_metrics(no_samples) == {}

    incomplete = dict(readings)
    incomplete[20_000] = {name: value for name, value in counters.items() if name != "NVLink TX Responses User Data"}
    with pytest.raises(ValueError, match="incomplete GPU metrics"):
        analysis._gemm_samples(incomplete, gpu=2, gemm=0, windows=windows)


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
