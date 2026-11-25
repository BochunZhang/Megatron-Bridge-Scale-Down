# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the fused AG/RS analyzer with Nsight's GPU identity and sample semantics."""

import importlib.util
import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import pytest


ANALYZER = Path(__file__).resolve().parents[3] / "examples" / "gb200" / "nvls-fused-ag-rs" / "analyse_nsys.py"
SPEC = importlib.util.spec_from_file_location("gb200_fused_ag_rs_nsys", ANALYZER)
assert SPEC is not None and SPEC.loader is not None
analyse = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyse)

SCHEMA = """
CREATE TABLE StringIds(id INTEGER, value TEXT);
CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(
    deviceId INTEGER, start INTEGER, end INTEGER, gridX INTEGER, blockX INTEGER,
    registersPerThread INTEGER, streamId INTEGER, shortName INTEGER, globalPid INTEGER
);
CREATE TABLE TARGET_INFO_CUDA_DEVICE(gpuId INTEGER, cudaId INTEGER, pid INTEGER);
CREATE TABLE TARGET_INFO_GPU_METRICS(typeId INTEGER, metricId INTEGER, metricName TEXT);
CREATE TABLE GPU_METRICS(typeId INTEGER, metricId INTEGER, timestamp INTEGER, value REAL);
"""


def _write_trace(path: Path, *, fused: bool, sample_gap: bool = False) -> None:
    with closing(sqlite3.connect(path)) as db:
        db.executescript(SCHEMA)
        db.executemany("INSERT INTO StringIds VALUES (?, ?)", enumerate(analyse.KERNEL_NAMES.values(), 1))
        # CUDA_VISIBLE_DEVICES=3,1 remaps CUDA ordinals 0,1 to physical GPUs 3,1.
        for cuda_id, gpu in enumerate((3, 1)):
            pid = 100 + cuda_id
            db.executemany("INSERT INTO TARGET_INFO_CUDA_DEVICE VALUES (?, ?, ?)", [(3, 0, pid), (1, 1, pid)])
            type_id = 281479271677952 | gpu
            for start in (10_000, 70_000):
                launches = (
                    [(start, start + 20_000, 1, 1)]
                    if fused
                    else [
                        (start, start + 14_000, 2, 1),
                        (start + 10_000, start + 20_000, 3, 2),
                    ]
                )
                db.executemany(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, 4, 256, 30, ?, ?, ?)",
                    [(cuda_id, left, right, stream, op, pid << 24) for left, right, op, stream in launches],
                )
            for metric_id, (name, (direction, packet, kind)) in enumerate(analyse.NVLINK_BYTES.items()):
                db.execute("INSERT INTO TARGET_INFO_GPU_METRICS VALUES (?, ?, ?)", (type_id, metric_id, name))
                gbps = 100 if (packet, kind) == ("request", "user") else 10
                gbps *= gpu * (2 if direction == "tx" else 1)
                for timestamp in range(0, 100_001, 20_000):
                    if sample_gap and gpu == 3 and metric_id == 0 and timestamp == 20_000:
                        continue
                    # The interval from 40 to 60 µs is between pairs and must contribute no bytes.
                    value = gbps * 20_000 * (10 if timestamp == 60_000 else 1)
                    db.execute("INSERT INTO GPU_METRICS VALUES (?, ?, ?, ?)", (type_id, metric_id, timestamp, value))
        db.commit()


def _args(tmp_path: Path) -> list[str]:
    return [
        str(ANALYZER),
        "--fused",
        str(tmp_path / "fused.sqlite"),
        "--split",
        str(tmp_path / "split.sqlite"),
        "--iters",
        "2",
        "--gpus",
        "3,1",
        "--frequency",
        "50000",
        "--output",
        str(tmp_path / "summary.json"),
    ]


@pytest.mark.unit
def test_nvlink_uses_physical_gpu_ids_and_weights_only_pair_spans(tmp_path, monkeypatch):
    _write_trace(tmp_path / "fused.sqlite", fused=True)
    _write_trace(tmp_path / "split.sqlite", fused=False)
    monkeypatch.setattr(sys, "argv", _args(tmp_path))
    analyse.main()

    report = json.loads((tmp_path / "summary.json").read_text())
    for mode in ("fused", "split"):
        devices = report[mode]["devices"]
        assert set(devices) == {"1", "3"}
        gpu3 = devices["3"]["nvlink"]
        assert gpu3["window_us"] == 40
        assert gpu3["rx"]["user_gb_per_s"] == pytest.approx(330)
        assert gpu3["rx"]["protocol_gb_per_s"] == pytest.approx(60)
        assert gpu3["rx"]["total_gb_per_s"] == pytest.approx(390)
        assert gpu3["rx"]["percent_of_nominal"] == pytest.approx(390 / 900 * 100)
        assert gpu3["tx"]["total_gb_per_s"] == pytest.approx(780)
        assert devices["1"]["nvlink"]["rx"]["total_gb_per_s"] == pytest.approx(130)


@pytest.mark.unit
def test_nvlink_rejects_an_interval_with_dropped_samples(tmp_path, monkeypatch, capsys):
    _write_trace(tmp_path / "fused.sqlite", fused=True, sample_gap=True)
    _write_trace(tmp_path / "split.sqlite", fused=False)
    monkeypatch.setattr(sys, "argv", _args(tmp_path))
    with pytest.raises(SystemExit, match="2"):
        analyse.main()
    assert "missing sample interval of 40000 ns" in capsys.readouterr().err
