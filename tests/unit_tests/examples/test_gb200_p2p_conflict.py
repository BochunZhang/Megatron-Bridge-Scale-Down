# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-only checks for the GB200 P2P conflict benchmark.

The real run needs the four GPUs of a GB200 tray. These tests cover the scenario table, the NCCL
connect-line and environment checks, the result summary, the nsys analysis on a synthetic SQLite
export, and the launcher's commands (one nsys session around torchrun) in --dry-run mode.
"""

import importlib.util
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest


BENCHMARK_DIR = Path(__file__).resolve().parents[3] / "examples" / "gb200" / "p2p-conflict"
WORKER_PATH = BENCHMARK_DIR / "p2p_conflict_benchmark.py"
ANALYSIS_PATH = BENCHMARK_DIR / "analyse_nsys.py"
LAUNCHER_PATH = BENCHMARK_DIR / "run_p2p_conflict_benchmark.sh"
needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
MIB = 1024 * 1024


def _load(path: Path, name: str) -> types.ModuleType:
    """Import a benchmark file by path, stubbing torch if it is not installed."""
    stub_modules: dict[str, types.ModuleType] = {}
    if importlib.util.find_spec("torch") is None:
        torch_stub = types.ModuleType("torch")
        torch_stub.__path__ = []  # type: ignore[attr-defined]
        torch_stub.bfloat16 = "bfloat16"  # type: ignore[attr-defined]
        torch_stub.float32 = "float32"  # type: ignore[attr-defined]
        cuda_stub = types.ModuleType("torch.cuda")
        cuda_stub.__path__ = []  # type: ignore[attr-defined]
        cuda_stub.nvtx = types.ModuleType("torch.cuda.nvtx")  # type: ignore[attr-defined]
        torch_stub.cuda = cuda_stub  # type: ignore[attr-defined]
        torch_stub.distributed = types.ModuleType("torch.distributed")  # type: ignore[attr-defined]
        stub_modules = {
            "torch": torch_stub,
            "torch.distributed": torch_stub.distributed,
            "torch.cuda": cuda_stub,
            "torch.cuda.nvtx": cuda_stub.nvtx,
        }
        sys.modules.update(stub_modules)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    finally:
        for stub in stub_modules:
            del sys.modules[stub]
    return module


worker = _load(WORKER_PATH, "gb200_p2p_conflict")
analysis = _load(ANALYSIS_PATH, "gb200_p2p_conflict_analysis")


@pytest.mark.unit
def test_scenarios_add_background_flows_into_g0_behind_the_test_flow():
    assert worker.SCENARIOS == {
        "alone": ((0, 3),),
        "bg1": ((0, 3), (1, 0)),
        "bg2": ((0, 3), (1, 0), (2, 0)),
        "bg3": ((0, 3), (1, 0), (2, 0), (3, 0)),
    }
    assert worker.all_flows(["bg3", "alone"]) == [(0, 3), (1, 0), (2, 0), (3, 0)]


@pytest.mark.unit
def test_my_ends_lists_every_end_a_rank_runs():
    flows = worker.SCENARIOS["bg3"]
    assert worker.my_ends(flows, 0) == [((0, 3), "send"), ((1, 0), "recv"), ((2, 0), "recv"), ((3, 0), "recv")]
    # G3 receives the test flow and sends into G0 at the same time.
    assert worker.my_ends(flows, 3) == [((0, 3), "recv"), ((3, 0), "send")]
    assert worker.my_ends(worker.SCENARIOS["alone"], 1) == []


@pytest.mark.unit
def test_range_names_parse_back_in_the_analysis():
    match = analysis.FLOW_RANGE.match(worker.flow_range("bg2", 7, (2, 0), "recv"))
    assert match and (match["scenario"], match["trial"], match["src"], match["dst"], match["side"]) == (
        "bg2",
        "7",
        "2",
        "0",
        "recv",
    )
    assert analysis.GATE_RANGE.match(worker.gate_range("alone", 0))


@pytest.mark.unit
def test_nccl_debug_path_expands_host_and_pid():
    path = worker.nccl_debug_path("/r/nccl-%h-%p-%%.log", hostname="node7.cluster", pid=42)
    assert path == Path("/r/nccl-node7-42-%.log")


@pytest.mark.unit
def test_connect_lines_must_be_p2p_in_the_requested_direction():
    write = ["x NCCL INFO Channel 00/1 : 0[0] -> 1[3] via P2P/CUMEM", "Channel 01/1 : 0[0] -> 1[3] via P2P/CUMEM"]
    assert worker.check_connect_lines(write, mode="push") == (["P2P/CUMEM"], [])
    assert "needs P2P read" in worker.check_connect_lines(write, mode="pull")[1][0]
    read = ["Channel 00/1 : 0[0] -> 1[3] via P2P/CUMEM/read"]
    assert worker.check_connect_lines(read, mode="pull") == (["P2P/CUMEM/read"], [])
    assert (
        "not over NVLink P2P" in worker.check_connect_lines(["Channel 00/1 : 0[0] -> 1[3] via SHM"], mode="push")[1][0]
    )
    assert worker.check_connect_lines([], mode="push")[1] == ["no P2P send channel was logged"]


def _good_env(monkeypatch, mode: str) -> None:
    monkeypatch.setenv("NCCL_P2P_READ_ENABLE", "1" if mode == "pull" else "0")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("NCCL_DEBUG", "INFO")
    monkeypatch.setenv("NCCL_DEBUG_FILE", "/tmp/nccl-%h-%p.log")


@pytest.mark.unit
@pytest.mark.parametrize(
    "change, match",
    [
        ({"world_size": 2}, "four ranks"),
        ({"NCCL_P2P_READ_ENABLE": "1"}, "NCCL_P2P_READ_ENABLE=0"),
        ({"CUDA_VISIBLE_DEVICES": "0,1,2,3"}, "unset CUDA_VISIBLE_DEVICES"),
        ({"CUDA_DEVICE_ORDER": "FASTEST_FIRST"}, "PCI_BUS_ID"),
        ({"NCCL_DEBUG": "WARN"}, "NCCL_DEBUG=INFO"),
        ({"NCCL_DEBUG_FILE": "/tmp/nccl.log"}, "%p"),
    ],
)
def test_check_environment(monkeypatch, change, match):
    _good_env(monkeypatch, "push")
    worker.check_environment("push", 4)
    world_size = change.pop("world_size", 4)
    for name, value in change.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match=match):
        worker.check_environment("push", world_size)


@pytest.mark.unit
def test_parse_args_defaults_and_rejections():
    args = worker.parse_args(["--output-dir", "/tmp/x"])
    assert args.scenarios == ["alone", "bg1", "bg2", "bg3"] and args.gpus == [0, 1, 2, 3] and args.size_mib == 1024
    for bad in (["--gpus", "0,1,2"], ["--gpus", "0,1,1,2"], ["--scenarios", "bg9"], ["--trials", "0"]):
        with pytest.raises(SystemExit):
            worker.parse_args(["--output-dir", "/tmp/x", *bad])


def _rank_records(ends_by_rank: list[dict[str, list[float]]], trials: int) -> list[dict[str, object]]:
    return [{"ends": ends, "late": [False] * trials, "ok": True} for ends in ends_by_rank]


@pytest.mark.unit
def test_summarize_scenario_uses_receiver_time_and_counts_covered_trials():
    size = 1024 * MIB
    per_rank = _rank_records(
        [
            {"0>3 send": [1.0, 1.0], "1>0 recv": [3.0, 1.5]},
            {"1>0 send": [2.9, 1.4]},
            {},
            {"0>3 recv": [2.0, 2.0]},
        ],
        trials=2,
    )
    per_rank[2]["late"] = [False, True]
    record = worker.summarize_scenario("bg1", per_rank, size_bytes=size, gpus=[4, 5, 6, 7], alone_gbps=1073.741824)
    test, background = record["flows"]
    assert test["role"] == "test" and (test["src_gpu"], test["dst_gpu"]) == (4, 7)
    assert test["recv_median_ms"] == 2.0 and test["gbps"] == pytest.approx(size / 2e-3 / 1e9)
    assert record["test_vs_alone"] == pytest.approx(0.5)
    # Trial 1: the background flow ended (1.5 ms) before the test flow (2.0 ms).
    assert record["covered_trials"] == 1
    assert record["late_launch_trials"] == 1
    assert background["flow"] == "1>0" and background["recv_median_ms"] == pytest.approx(2.25)

    alone = worker.summarize_scenario(
        "alone",
        _rank_records([{"0>3 send": [1.0]}, {}, {}, {"0>3 recv": [1.0]}], 1),
        size_bytes=size,
        gpus=[0, 1, 2, 3],
        alone_gbps=None,
    )
    assert alone["covered_trials"] is None and alone["test_vs_alone"] is None
    summary = worker.render_summary([alone, record], gpus=[4, 5, 6, 7], mode="push", size_mib=1024)
    assert "| bg1 | 536.9 | 0.500 | 1>0: 477.2 | 1/2 | 1/2 |" in summary


def _report_db(path: Path, *, gpus: list[int], with_metrics: bool = True) -> None:
    """A minimal nsys SQLite export: scenario bg1, one trial, gate + three flow-end kernels."""
    db = sqlite3.connect(path)
    db.executescript(
        """
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE NVTX_EVENTS (start INT, end INT, text TEXT, textId INT, globalTid INT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INT, globalTid INT, correlationId INT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INT, end INT, deviceId INT, streamId INT,
            correlationId INT, globalPid INT, shortName INT, gridX INT, blockX INT);
        CREATE TABLE GPU_METRICS (timestamp INT, typeId INT, metricId INT, value INT);
        CREATE TABLE TARGET_INFO_GPU_METRICS (typeId INT, metricId INT, metricName TEXT);
        CREATE TABLE TARGET_INFO_GPU (id INT, name TEXT, busLocation TEXT);
        """
    )
    db.executemany(
        "INSERT INTO StringIds VALUES (?, ?)",
        [(1, "ncclDevKernel_SendRecv"), (2, "ncclDevKernel_AllReduce_Sum_f32_RING_LL")],
    )
    tid = {rank: (1000 + rank) << 24 for rank in range(4)}
    ranges, kernels, correlation = [], [], 0

    def launch(rank, name, host, kernel_name, start, end, grid):
        nonlocal correlation
        correlation += 1
        ranges.append((host, host + 10, name, None, tid[rank]))
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?)", (host + 1, tid[rank], correlation))
        kernels.append((start, end, gpus[rank], 7, correlation, tid[rank], kernel_name, grid, 640 if grid > 1 else 64))

    for rank in range(4):
        launch(rank, "bg1/trial0/gate", 100 + rank, 2, 10_000, 10_100, 1)
    launch(0, "bg1/trial0/flow 0>3 send", 200, 1, 10_150, 12_150, 32)
    launch(0, "bg1/trial0/flow 1>0 recv", 300, 1, 10_160, 13_160, 32)
    launch(1, "bg1/trial0/flow 1>0 send", 200, 1, 10_150, 13_150, 32)
    launch(3, "bg1/trial0/flow 0>3 recv", 200, 1, 10_140, 12_200, 32)
    db.executemany("INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, ?)", ranges)
    db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", kernels)
    if with_metrics:
        for gpu in gpus:
            type_id = (5 << 8) | gpu
            db.execute("INSERT INTO TARGET_INFO_GPU VALUES (?, 'GB200', ?)", (gpu, f"0000:0{gpu}:00.0"))
            db.execute(
                "INSERT INTO TARGET_INFO_GPU_METRICS VALUES (?, 3, 'NVLink TX Requests User Data [Throughput %]')",
                (type_id,),
            )
            db.execute("INSERT INTO TARGET_INFO_GPU_METRICS VALUES (?, 4, 'SMs Active')", (type_id,))
            for timestamp, value in ((10_000, 99), (11_000, 40 + gpu), (12_000, 60 + gpu), (20_000, 99)):
                db.execute("INSERT INTO GPU_METRICS VALUES (?, ?, 3, ?)", (timestamp, type_id, value))
                db.execute("INSERT INTO GPU_METRICS VALUES (?, ?, 4, 10)", (timestamp, type_id))
    db.commit()
    db.close()


def _run_dir(tmp_path: Path, *, gpus: list[int], with_metrics: bool = True) -> Path:
    run = tmp_path / "run"
    (run / "nsys").mkdir(parents=True)
    metadata = {"gpus": gpus, "scenarios": {"bg1": ["0>3", "1>0"]}, "args": {"trials": 1, "size_mib": 1}}
    (run / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    report = run / "nsys" / "p2p-conflict.nsys-rep"
    report.write_bytes(b"")
    old = time.time() - 60
    os.utime(report, (old, old))  # the export below must count as newer, so nsys is never called
    _report_db(run / "nsys" / "p2p-conflict.sqlite", gpus=gpus, with_metrics=with_metrics)
    return run


@pytest.mark.unit
def test_analysis_maps_kernels_to_flows_and_averages_nvlink_per_gpu(tmp_path):
    gpus = [2, 3, 0, 1]
    run = _run_dir(tmp_path, gpus=gpus)
    summary = analysis.analyse(run, nsys_bin="/nonexistent/nsys").read_text(encoding="utf-8")

    rows = (run / "nsys_flow_kernels.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1 + 4
    # G0 -> G3 recv runs on G3 = GPU1: 140 ns after the gate kernel ended, 2.06 us long.
    assert "| bg1 | G0>3 | recv | GPU1 | 32 | 640 | 0.0 | 2.1 | 2.1 | 509.0 |" in summary
    assert "every flow end of every trial has one kernel on its role's GPU" in summary
    # Window 10140..13160 ns holds the 11000 and 12000 samples only; SMs Active is not reported.
    assert "| bg1 | GPU2 (G0) | 52.0 | - |" in summary
    windows = (run / "nvlink_windows.csv").read_text(encoding="utf-8")
    assert "SMs Active" not in windows and windows.count("NVLink TX Requests User Data") == 4
    samples = (run / "nvlink_samples.csv").read_text(encoding="utf-8").splitlines()
    assert len(samples) == 1 + 4 * 2


@pytest.mark.unit
def test_analysis_reports_missing_flow_ends_wrong_gpus_and_missing_metrics(tmp_path):
    run = _run_dir(tmp_path, gpus=[0, 1, 2, 3], with_metrics=False)
    metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    metadata["args"]["trials"] = 2
    metadata["gpus"] = [0, 1, 3, 2]  # the kernels ran with G3 on GPU3
    (run / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    summary = analysis.analyse(run, nsys_bin="/nonexistent/nsys").read_text(encoding="utf-8")
    assert "bg1 trial 1: no kernel for flow 0>3 send" in summary
    assert "bg1 trial 0 flow 0>3 recv: ran on GPU3, G3 is GPU2" in summary
    assert "recorded no NVLink metric" in summary


def _dry_run(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k not in {"RUN_TIME", "CUDA_VISIBLE_DEVICES"}}
    return subprocess.run(
        ["bash", str(LAUNCHER_PATH), "--dry-run", "--output-dir", str(tmp_path / "run"), *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


@pytest.mark.unit
@needs_bash
def test_launcher_wraps_torchrun_in_one_nsys_session(tmp_path):
    result = _dry_run(tmp_path, "--gpus", "3,2,1,0", "--p2p-mode", "pull")
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    # printf %q escapes commas in some bash versions.
    (run_line,) = [line.replace("\\,", ",") for line in lines if line.startswith("env ")]
    (analyse_line,) = [line for line in lines if "analyse_nsys.py" in line]
    assert run_line.count("nsys profile") == 1
    assert run_line.index("nsys profile") < run_line.index("torch.distributed.run")
    for flag in (
        "--gpu-metrics-devices=all",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        f"--output={tmp_path}/run/nsys/p2p-conflict",
        "--nproc_per_node=4",
        "-u CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER=PCI_BUS_ID",
        "NCCL_P2P_READ_ENABLE=1",
        "--gpus 3,2,1,0",
        "--p2p-mode pull",
        " --nsys",
    ):
        assert flag in run_line, flag
    assert f"--output-dir {tmp_path}/run" in analyse_line


@pytest.mark.unit
@needs_bash
def test_launcher_without_nsys_and_whole_job_capture(tmp_path):
    plain = _dry_run(tmp_path, "--no-nsys")
    assert plain.returncode == 0, plain.stderr
    assert "nsys profile" not in plain.stdout and "analyse_nsys.py" not in plain.stdout
    whole = _dry_run(tmp_path, "--nsys-capture", "all", "--nsys-freq", "high")
    assert "--capture-range" not in whole.stdout and "--gpu-metrics-frequency=200000" in whole.stdout


@pytest.mark.unit
@needs_bash
@pytest.mark.parametrize(
    "args, message",
    [
        (["--gpus", "0,1,2"], "four ids"),
        (["--gpus", "0,1,1,2"], "distinct"),
        (["--gpus", "0,1,2,4"], "not a GPU id"),
        (["--scenarios", "bg9"], "unknown scenario"),
        (["--p2p-mode", "both"], "push or pull"),
        (["--nsys-capture", "some"], "range or all"),
    ],
)
def test_launcher_rejects_bad_options(tmp_path, args, message):
    result = _dry_run(tmp_path, *args)
    assert result.returncode == 2 and message in result.stderr
