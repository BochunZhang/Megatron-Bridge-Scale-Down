"""CPU checks for the adaptive pause launcher and Nsight Systems trace gate."""

import csv
import importlib.util
import os
import shlex
import sqlite3
import subprocess
from contextlib import closing
from pathlib import Path

import pytest


EXAMPLE = Path(__file__).resolve().parents[3] / "examples" / "gb200" / "ncu-torch-all-gather"
SPEC = importlib.util.spec_from_file_location("nsys_torch_adaptive_pause", EXAMPLE / "nsys_torch_adaptive_pause.py")
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def _write_profile(path: Path, gpus: tuple[int, ...], *, rs_end: int) -> None:
    with closing(sqlite3.connect(path)) as db:
        db.executescript("""
CREATE TABLE StringIds (id INTEGER, value TEXT);
CREATE TABLE TARGET_INFO_CUDA_DEVICE (pid INTEGER, cudaId INTEGER, gpuId INTEGER);
CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (globalPid INTEGER, deviceId INTEGER, shortName INTEGER, start INTEGER, end INTEGER);
CREATE TABLE TARGET_INFO_GPU_METRICS (typeId INTEGER, metricId INTEGER, metricName TEXT);
CREATE TABLE GPU_METRICS (typeId INTEGER, metricId INTEGER, timestamp INTEGER, value REAL);
INSERT INTO StringIds VALUES (1, 'ncclSymkDevKernel_AllGather_STMC');
INSERT INTO StringIds VALUES (2, 'ncclSymkDevKernel_ReduceScatter_LDMC_sum_bf16');
""")
        for rank, gpu in enumerate(gpus):
            pid, offset, type_id = 100 + rank, 100 * rank, 256 + gpu
            db.execute("INSERT INTO TARGET_INFO_CUDA_DEVICE VALUES (?, ?, ?)", (pid, rank, gpu))
            db.executemany(
                "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?)",
                [
                    (pid << 24, gpu, 1, 10_000 + offset, 150_000 + offset),
                    (pid << 24, gpu, 2, 45_000 + offset, rs_end + offset),
                ],
            )
            for metric_id, name in enumerate(benchmark.METRICS):
                db.execute(
                    "INSERT INTO TARGET_INFO_GPU_METRICS VALUES (?, ?, ?)", (type_id, metric_id, f"{name} [Bytes]")
                )
                db.executemany(
                    "INSERT INTO GPU_METRICS VALUES (?, ?, ?, ?)",
                    (
                        (type_id, metric_id, timestamp, 250_000 if 45_000 < timestamp <= rs_end else 500_000)
                        for timestamp in range(0, 185_001, 5_000)
                    ),
                )
        db.commit()


@pytest.mark.unit
def test_adaptive_capture_requires_rs_to_finish_before_every_ag(tmp_path):
    gpus = (1, 3, 4, 7)
    _write_profile(tmp_path / "profile.sqlite", gpus, rs_end=90_000)
    benchmark.analyse(tmp_path, gpus, adaptive=True)
    with (tmp_path / "nvlink_by_phase.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4 * 8 * 3
    rx = {
        row["phase"]: float(row["GB/s"]) for row in rows if row["gpu"] == "1" and row["metric"] == benchmark.METRICS[0]
    }
    assert rx == {"ag_before_rs": 100, "ag_rs_overlap": 50, "ag_after_rs": 100}

    control = tmp_path / "control"
    control.mkdir()
    _write_profile(control / "profile.sqlite", gpus, rs_end=170_000)
    with pytest.raises(ValueError, match="last RS must finish"):
        benchmark.analyse(control, gpus, adaptive=True)
    assert (control / "kernel_timing.csv").exists()
    benchmark.analyse(control, gpus, adaptive=False)
    assert "rs_after_ag" in (control / "nvlink_by_phase.csv").read_text()


@pytest.mark.unit
def test_metric_gap_rejects_missing_samples():
    with pytest.raises(ValueError, match="missing NVLink sample interval"):
        benchmark._phase_rate([(0, 0), (5_000, 100), (15_000, 100), (20_000, 100)], start=0, end=20_000)


@pytest.mark.unit
def test_dry_run_profiles_the_same_four_gpus_and_uses_the_requested_poll_interval(tmp_path):
    output = tmp_path / "dry-run"
    result = subprocess.run(
        [
            "bash",
            str(EXAMPLE / "run_nsys_torch_adaptive_pause.sh"),
            "--dry-run",
            "--gpus",
            "1,3,4,7",
            "--rs-delay-ms",
            "0.5",
            "--output-dir",
            str(output),
        ],
        env={**os.environ, "NSYS_BIN": "nsys", "NCCL_SYM_AG_STMC_PAUSE_POLL_CHUNKS": "8"},
        capture_output=True,
        text=True,
        check=True,
    )
    profiles = [shlex.split(line) for line in result.stdout.splitlines() if line.startswith("nsys profile")]
    assert len(profiles) == 2
    for profile, rs_pause in zip(profiles, (3072, 0), strict=True):
        assert "--gpu-metrics-devices=1,3,4,7" in profile
        assert "--gpu-metrics-frequency=200000" in profile
        assert "--nproc_per_node=4" in profile
        assert "NCCL_SYM_AG_STMC_PAUSE_CYCLES=1024" in profile
        assert "NCCL_SYM_AG_STMC_PAUSE_POLL_CHUNKS=8" in profile
        assert f"NCCL_SYM_AG_STMC_RS_PAUSE_CYCLES={rs_pause}" in profile
        assert profile[-2:] == ["--rs-delay-ms", "0.5"]
    assert not output.exists()
