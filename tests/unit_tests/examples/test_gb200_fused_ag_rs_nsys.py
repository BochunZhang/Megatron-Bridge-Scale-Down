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


def _write_trace(
    path: Path,
    *,
    fused: bool,
    sample_gap: bool = False,
    fused_suffix: str = "",
    solo: str | None = None,
    grid: int = 4,
    block: int = 256,
) -> None:
    with closing(sqlite3.connect(path)) as db:
        db.executescript(SCHEMA)
        kernel_names = analyse.KERNEL_NAMES | {"fused": analyse.KERNEL_NAMES["fused"] + fused_suffix}
        db.executemany("INSERT INTO StringIds VALUES (?, ?)", enumerate(kernel_names.values(), 1))
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
                if solo is not None:
                    op_id = {"ag": 2, "rs": 3}[solo]
                    launches = [launch for launch in launches if launch[2] == op_id]
                db.executemany(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, 30, ?, ?, ?)",
                    [
                        (cuda_id, left, right, grid, block, stream, op, pid << 24)
                        for left, right, op, stream in launches
                    ],
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
    assert "ag_only" not in report and "rs_only" not in report
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


@pytest.mark.unit
@pytest.mark.parametrize(
    ("variant", "suffix"),
    [
        ("ag-first", "AgFirst"),
        ("rs-first", "RsFirst"),
        ("warp-split", "WarpSplit"),
        ("warp-split-pause", "WarpSplitPause"),
    ],
)
def test_trace_records_verified_fused_variant_and_ctas(tmp_path, monkeypatch, variant, suffix):
    _write_trace(tmp_path / "fused.sqlite", fused=True, fused_suffix=suffix, grid=8)
    _write_trace(tmp_path / "split.sqlite", fused=False)
    monkeypatch.setattr(sys, "argv", [*_args(tmp_path), "--variant", variant, "--ctas", "8"])
    analyse.main()

    fused = json.loads((tmp_path / "summary.json").read_text())["fused"]
    assert fused["variant"] == variant
    assert fused["ctas"] == 8
    for device in fused["devices"].values():
        assert device["kernels"]["fused"]["launch_configs"] == [
            {"grid_x": 8, "block_x": 256, "registers_per_thread": 30}
        ]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("suffix", "grid", "block", "variant_args", "error"),
    [
        ("AgFirst", 4, 256, [], "expected 'ncclDevKernel_FusedAllGatherReduceScatter'"),
        (
            "WarpSplitPause",
            4,
            256,
            ["warp-split", "4"],
            "expected 'ncclDevKernel_FusedAllGatherReduceScatterWarpSplit'",
        ),
        ("AgFirst", 3, 256, ["ag-first", "4"], "gridX=3, blockX=256"),
        ("AgFirst", 4, 128, ["ag-first", "4"], "gridX=4, blockX=128"),
    ],
)
def test_trace_rejects_wrong_fused_kernel_or_geometry(
    tmp_path, monkeypatch, capsys, suffix, grid, block, variant_args, error
):
    _write_trace(tmp_path / "fused.sqlite", fused=True, fused_suffix=suffix, grid=grid, block=block)
    _write_trace(tmp_path / "split.sqlite", fused=False)
    args = _args(tmp_path)
    if variant_args:
        args.extend(("--variant", variant_args[0], "--ctas", variant_args[1]))
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit, match="2"):
        analyse.main()
    assert error in capsys.readouterr().err


@pytest.mark.unit
@pytest.mark.parametrize("solo_ops", [("ag",), ("rs",), ("ag", "rs")])
def test_trace_measures_optional_solo_kernel_windows(tmp_path, monkeypatch, solo_ops):
    _write_trace(tmp_path / "fused.sqlite", fused=True)
    _write_trace(tmp_path / "split.sqlite", fused=False)
    args = _args(tmp_path)
    for op in solo_ops:
        path = tmp_path / f"{op}-only.sqlite"
        _write_trace(path, fused=False, solo=op)
        args.extend((f"--{op}-only", str(path)))
    monkeypatch.setattr(sys, "argv", args)
    analyse.main()

    report = json.loads((tmp_path / "summary.json").read_text())
    assert report["fused"]["median_slowest_device_pair_span_us"] == 20
    assert report["split"]["median_slowest_device_pair_span_us"] == 20
    assert "solo captures" in report["nvlink"]["window"]
    for op, duration_us, min_intervals in (("ag", 14, 2), ("rs", 10, 1)):
        key = f"{op}_only"
        if op not in solo_ops:
            assert key not in report
            continue
        assert report[key]["median_slowest_device_kernel_span_us"] == duration_us
        device = report[key]["devices"]["3"]
        assert device["median_kernel_span_us"] == duration_us
        assert set(device["kernels"]) == {op}
        assert device["kernels"][op]["median_duration_us"] == duration_us
        assert device["nvlink"]["window_us"] == 2 * duration_us
        assert device["nvlink"]["rx"]["total_gb_per_s"] == pytest.approx(390)
        assert device["nvlink"]["min_sample_intervals_per_metric_per_iteration"] == min_intervals


@pytest.mark.unit
@pytest.mark.parametrize(
    ("sql", "error"),
    [
        ("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET shortName = 3", "kernel counts {'rs': 2}, expected {'ag': 2}"),
        ("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start = 10000", "kernel counts {'ag': 1}, expected {'ag': 2}"),
        ("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId = 1", "has kernels on GPUs [3], expected [1, 3]"),
    ],
)
def test_trace_rejects_mismatched_solo_profile(tmp_path, monkeypatch, capsys, sql, error):
    _write_trace(tmp_path / "fused.sqlite", fused=True)
    _write_trace(tmp_path / "split.sqlite", fused=False)
    path = tmp_path / "ag-only.sqlite"
    _write_trace(path, fused=False, solo="ag")
    with closing(sqlite3.connect(path)) as db:
        db.execute(sql)
        db.commit()
    monkeypatch.setattr(sys, "argv", [*_args(tmp_path), "--ag-only", str(path)])
    with pytest.raises(SystemExit, match="2"):
        analyse.main()
    assert error in capsys.readouterr().err
