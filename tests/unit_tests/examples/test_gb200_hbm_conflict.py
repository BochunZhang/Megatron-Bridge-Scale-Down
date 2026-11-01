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

"""CPU-only checks for the GB200 GEMM / AllGather HBM contention experiment.

The real run needs four GB200 GPUs, nsys and nccl-2.32.3-pause. These tests feed the analysis a
synthetic report in the schema of ``nsys export --type sqlite`` and check the launcher's options
and commands with ``--dry-run``.
"""

import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = REPO_ROOT / "examples" / "gb200" / "hbm-conflict"
LAUNCHER_PATH = EXPERIMENT_DIR / "run_hbm_conflict_benchmark.sh"
PATCH_PATH = REPO_ROOT / "install" / "patches" / "nccl-2.32.3-pause.patch"
PID = 100


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENT_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


analyse = _load("analyse_hbm_conflict")


def _write_report(
    path: Path,
    *,
    case: str,
    ag: list[tuple[int, int]],
    gemm: list[tuple[int, int]],
    metrics: dict[str, list[tuple[int, float]]],
) -> None:
    """A report with one ag<II> / gemm<JJ> NVTX range and one kernel per launch."""
    db = sqlite3.connect(path)
    db.executescript(
        """
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, globalTid INTEGER, text TEXT, textId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER, end INTEGER, correlationId INTEGER, globalTid INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            start INTEGER, end INTEGER, deviceId INTEGER, streamId INTEGER, gridX INTEGER, blockX INTEGER,
            correlationId INTEGER, globalPid INTEGER, shortName INTEGER);
        CREATE TABLE TARGET_INFO_GPU_METRICS (metricId INTEGER, metricName TEXT);
        CREATE TABLE GPU_METRICS (timestamp INTEGER, metricId INTEGER, value REAL);
        """
    )
    tid, gpid = (PID << 24) | 1, PID << 24
    db.execute("INSERT INTO StringIds VALUES (1, 'ncclSymkDevKernel_AllGather_ST'), (2, 'nvjet_tst_gemm')")
    db.execute("INSERT INTO NVTX_EVENTS VALUES (0, 100000, ?, ?, NULL)", (tid, f"hbm-conflict/{case}"))
    correlation = 0
    for prefix, kernels, name_id, stream, grid in (("ag", ag, 1, 7, 16), ("gemm", gemm, 2, 13, 148)):
        for index, (start, end) in enumerate(kernels):
            correlation += 1
            host = 10 * correlation
            db.execute(
                "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, NULL)", (host, host + 5, tid, f"{prefix}{index:02d}")
            )
            db.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?)", (host + 1, host + 2, correlation, tid)
            )
            db.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, 0, ?, ?, 512, ?, ?, ?)",
                (start, end, stream, grid, correlation, gpid, name_id),
            )
    for metric_id, (name, samples) in enumerate(metrics.items(), start=1):
        db.execute("INSERT INTO TARGET_INFO_GPU_METRICS VALUES (?, ?)", (metric_id, name))
        db.executemany("INSERT INTO GPU_METRICS VALUES (?, ?, ?)", [(t, metric_id, v) for t, v in samples])
    db.commit()
    db.close()


def _write_case(run: Path, case: str, mode: str, *, ag, gemm, metrics, pause: str | None = None) -> None:
    case_dir = run / case
    (case_dir / "nsys").mkdir(parents=True)
    (case_dir / "case.json").write_text(json.dumps({"case": case, "mode": mode}))
    events = {
        "case": case,
        "mode": mode,
        "rank": 0,
        "world_size": 4,
        "NCCL_SYM_AG_ST_PAUSE_CYCLES": pause,
        "size_mib": 1024,
        "dtype": "bf16",
        "gemm_mkn": [4096, 7168, 2048],
        "gemm_delay_us": 50.0 if mode == "sweep" else 0.0,
        "ag_warmup": 2,
        "ag_isolated": 2,
        "ag_ms": [[s / 1e6, e / 1e6] for s, e in ag],
        "gemm_ms": [[s / 1e6, e / 1e6] for s, e in gemm],
        "ag_correct": True if mode == "sweep" else None,
    }
    (case_dir / "events.rank_00.json").write_text(json.dumps(events))
    report = case_dir / "nsys" / f"{case}.rank_0.nsys-rep"
    report.write_bytes(b"")
    _write_report(report.with_suffix(".sqlite"), case=case, ag=ag, gemm=gemm, metrics=metrics)
    # The export must be newer than the report, so the analysis reuses it instead of calling nsys.
    os.utime(report, (1, 1))


def _make_run(run: Path, *, sweep_gemm: list[tuple[int, int]]) -> None:
    ag = [(i * 1000 + 10, (i + 1) * 1000) for i in range(6)]
    nvlink = [(t, 80.0 if t <= 4000 else 60.0) for t in range(0, 6000, 50)]
    dram = [(t, 30.0) for t in range(0, 6000, 50)]
    _write_case(
        run,
        "baseline",
        "baseline",
        ag=[],
        gemm=[(100, 250), (300, 450)],
        metrics={"NVLink RX Bytes": [(t, 0.0) for t in range(0, 500, 50)], "DRAM Read Bandwidth": dram[:10]},
    )
    _write_case(
        run,
        "pause1024cyc",
        "sweep",
        ag=ag,
        gemm=sweep_gemm,
        metrics={"NVLink RX Bytes": nvlink, "DRAM Read Bandwidth": dram},
        pause="1024",
    )


def test_report_compares_gemms_under_allgather_with_the_baseline(tmp_path):
    _make_run(tmp_path, sweep_gemm=[(4100, 4300), (4400, 4600)])
    analyse.main(["--output-dir", str(tmp_path), "--nsys", "/nonexistent/nsys"])

    summary = {c["case"]: c for c in json.loads((tmp_path / "summary.json").read_text())}
    sweep, baseline = summary["pause1024cyc"], summary["baseline"]
    assert baseline["gemm"]["mean"] == pytest.approx(150 / 1e6)
    assert sweep["gemm"]["mean"] == pytest.approx(200 / 1e6)
    assert sweep["gemm"]["fully_covered"] == 2 and sweep["gemm"]["min_coverage"] == 1.0
    assert sweep["ag_kernels"] == ["ncclSymkDevKernel_AllGather_ST"] and sweep["ag_grid_x"] == [16]
    # Rounds 2 and 3 are isolated; round 4 holds both GEMMs.
    assert sweep["ag_isolated"]["rounds"] == [2, 3]
    assert sweep["ag_isolated"]["ms"] == pytest.approx(990 / 1e6)
    assert sweep["ag_background_with_gemm"]["rounds"] == [4]
    assert sweep["ag_background_without_gemm"]["rounds"] == [5]
    assert sweep["metrics"]["ag_isolated"]["NVLink RX Bytes"]["mean"] == pytest.approx(80.0)
    assert sweep["metrics"]["gemm"]["NVLink RX Bytes"]["mean"] == pytest.approx(60.0)
    assert sweep["gemm"]["tflops"] == pytest.approx(2 * 4096 * 7168 * 2048 / (200 / 1e6 * 1e9))

    report = (tmp_path / "report.md").read_text()
    assert "| pause1024cyc | 1024 | AllGather_ST | 16 |" in report
    assert "| 1.333 | 2/2 / 100.0% |" in report
    assert "警告" not in report
    rounds = (tmp_path / "ag_rounds.csv").read_text().splitlines()
    assert len(rounds) == 1 + 6


def test_a_gemm_outside_the_allgathers_is_flagged(tmp_path):
    # The second GEMM ends in the gap after round 5 (5010..6000) and runs 100 ns without background.
    _make_run(tmp_path, sweep_gemm=[(4100, 4300), (5900, 6100)])
    analyse.main(["--output-dir", str(tmp_path), "--nsys", "/nonexistent/nsys"])

    sweep = next(c for c in json.loads((tmp_path / "summary.json").read_text()) if c["case"] == "pause1024cyc")
    assert sweep["gemm"]["fully_covered"] == 1
    assert sweep["gemm"]["min_coverage"] == pytest.approx(0.5)
    assert "1 GEMM(s) were not fully inside AllGather kernels" in (tmp_path / "report.md").read_text()


def test_a_report_filed_under_the_wrong_case_fails(tmp_path):
    _make_run(tmp_path, sweep_gemm=[(4100, 4300)])
    (tmp_path / "pause1024cyc").rename(tmp_path / "pause2048cyc")
    with pytest.raises(SystemExit, match="pause2048cyc"):
        analyse.main(["--output-dir", str(tmp_path), "--nsys", "/nonexistent/nsys"])


def test_intervals():
    assert analyse.merge_intervals([(5, 7), (0, 2), (1, 3)]) == [(0, 3), (5, 7)]
    assert analyse.overlap_ns(1, 6, [(0, 3), (5, 7)]) == 3
    assert [analyse.round_phase(i, warmup=5, isolated=5) for i in (0, 4, 5, 9, 10)] == [
        "warmup",
        "warmup",
        "isolated",
        "isolated",
        "background",
    ]


def _launcher(*args: str) -> subprocess.CompletedProcess:
    env = {key: value for key, value in os.environ.items() if not key.startswith("NCCL_")}
    env["RUN_TIME"] = "unit-test"
    return subprocess.run(
        ["bash", str(LAUNCHER_PATH), "--dry-run", "--venv", "none", *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_launcher_runs_the_baseline_and_one_job_per_pause():
    result = _launcher("--pause-cycles", "0,1024", "--no-analyse")
    assert result.returncode == 0, result.stderr
    assert "=== case=baseline" in result.stderr
    commands = [line for line in result.stdout.splitlines() if "hbm_conflict_benchmark.py" in line]
    assert len(commands) == 3
    baseline, pause0, pause1024 = commands
    assert "--mode baseline" in baseline and "NCCL_SYM_AG_ST_PAUSE_CYCLES" not in baseline
    assert "NCCL_SYM_AG_ST_PAUSE_CYCLES=1024" in pause1024 and "--mode sweep" in pause1024
    for command in commands:
        assert "NCCL_NVLS_ENABLE=0" in command
        assert "NCCL_SYM_KERNEL=allgather:AllGather_ST" in command
        assert "NCCL_SYM_TMA_ENABLE=0" in command
        assert "CUDA_VISIBLE_DEVICES=0\\,1\\,2\\,3 " in command
        assert "--capture-range=cudaProfilerApi --capture-range-end=stop" in command
        assert "--gemm-m 4096 --gemm-k 7168 --gemm-n 2048" in command
        assert "--ag-rounds 25 --ag-warmup 5 --ag-isolated 5" in command
    assert "NCCL_MIN_CTAS" not in pause0


def test_launcher_pins_ctas_only_on_request():
    result = _launcher("--pause-cycles", "0", "--skip-baseline", "--no-analyse", "--ctas", "8")
    assert result.returncode == 0, result.stderr
    assert "NCCL_MIN_CTAS=8 NCCL_MAX_CTAS=8" in result.stdout


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("--pause-cycles", "0,0"), "lists 0 twice"),
        (("--ag-rounds", "10"), "must leave background rounds"),
        (("--gemm-mkn", "4096,7168"), "--gemm-mkn must be M,K,N"),
        (("--ctas", "0"), "--ctas must be a positive integer"),
    ],
)
def test_launcher_rejects_bad_options(args, message):
    result = _launcher(*args)
    assert result.returncode != 0
    assert message in result.stderr


def test_the_pause_patch_slows_the_unicast_allgather():
    patch = PATCH_PATH.read_text()
    assert 'NCCL_PARAM(SymAgStPauseCycles, "SYM_AG_ST_PAUSE_CYCLES", 0)' in patch
    assert "args->agStPauseCycles" in patch
    assert "if (pauseCycles != 0) ncclSymkPauseCycles(pauseCycles);" in patch
