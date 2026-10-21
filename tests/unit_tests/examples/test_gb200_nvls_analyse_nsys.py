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

"""Run nsys_collective_kernels.sql against synthetic ``nsys export --type sqlite`` databases.

The schema mirrors the subset of the Nsight Systems export the queries use. Fixtures are
built so every expected number can be checked by hand, including the rows the SQL must drop:
warmup kernels launched before the capture window and the ``torch.cuda._sleep`` delay kernel.
Raw reports are named like nsys names them (``rank<R>.<K>.nsys-rep``) and filed through
``nsys/index.tsv``.
"""

import importlib.util
import json
import os
import sqlite3
import sys
from pathlib import Path

import pytest


ANALYSE_PATH = Path(__file__).resolve().parents[3] / "examples" / "gb200" / "nvls-collective" / "analyse_nsys.py"
_SPEC = importlib.util.spec_from_file_location("gb200_nvls_analyse_nsys", ANALYSE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
analyse = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = analyse
_SPEC.loader.exec_module(analyse)

AG = "ncclDevKernel_AllGather_RING_LL"
AR = "ncclDevKernel_AllReduce_Sum_bf16_RING_LL"
CASE = "all_gather+reduce_scatter-64MiB-bf16-single-n2"
IDLE_CASE = "all_gather-64MiB-bf16-single-n2"
RS = "ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL"
SENDRECV = "ncclDevKernel_SendRecv"
# gridX of every launch in the fixtures, i.e. the CTA count NCCL launched it with.
GRID = {AG: 6, RS: 6, AR: 8, SENDRECV: 16}
SCHEMA = """
CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT);
CREATE TABLE NVTX_EVENTS(start INT, end INT, text TEXT, textId INT, globalTid INT, eventType INT);
CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INT, end INT, globalTid INT, correlationId INT, nameId INT);
CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(
    start INT, end INT, deviceId INT, streamId INT, correlationId INT, globalPid INT,
    shortName INT, demangledName INT, gridX INT, blockX INT);
"""


def _make_report(path: Path, *, label: str, trials: list[list[tuple[str, int, int, int]]], pid: int = 4242) -> None:
    """Write one rank's export.

    ``trials[i]`` lists kernels as (name, launch_ns, start_ns, end_ns), with launch times relative
    to the trial's NVTX range. A warmup AG before the case window and a sleep kernel inside every
    trial are added so the tests also prove they are excluded.
    """
    tid, gpid = (pid << 24) | 7, pid << 24
    strings = {AG: 1, RS: 2, "sleep_kernel": 3, label: 4, AR: 5, SENDRECV: 6}
    db = sqlite3.connect(path)
    db.executescript(SCHEMA)
    db.executemany("INSERT INTO StringIds VALUES (?, ?)", [(i, s) for s, i in strings.items()])
    base, width = 10_000, 100_000
    case_end = base + width * len(trials)
    # The case range uses textId and trials use text, covering both NVTX export forms.
    db.execute("INSERT INTO NVTX_EVENTS VALUES (?, ?, NULL, 4, ?, 59)", (base, case_end, tid))
    correlation = 0

    def launch(name: str, launch_ns: int, start_ns: int, end_ns: int, stream: int) -> None:
        nonlocal correlation
        correlation += 1
        db.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?, 0)",
            (launch_ns, launch_ns + 5, tid, correlation),
        )
        db.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, 0, ?, ?, ?, ?, 0, ?, 512)",
            (start_ns, end_ns, stream, correlation, gpid, strings[name], GRID.get(name, 1)),
        )

    launch(AG, base - 500, base - 400, base - 300, 1)  # warmup, outside the window
    for index, kernels in enumerate(trials):
        start = base + width * index
        db.execute(
            "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, NULL, ?, 59)",
            (start + 1, start + width - 1, f"trial{index}", tid),
        )
        launch("sleep_kernel", start + 2, start + 3, start + 10, 9)
        for name, at, k_start, k_end in kernels:
            launch(name, start + at, start + k_start, start + k_end, {AG: 1, RS: 2, AR: 3, SENDRECV: 4}[name])
    db.commit()
    db.close()


def _make_idle_report(path: Path, *, label: str) -> None:
    """A rank that sat the case out: only its NVTX case range, no CUDA activity tables at all."""
    db = sqlite3.connect(path)
    db.executescript(
        "CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT);"
        "CREATE TABLE NVTX_EVENTS(start INT, end INT, text TEXT, textId INT, globalTid INT, eventType INT);"
    )
    db.execute("INSERT INTO NVTX_EVENTS VALUES (10, 20, ?, NULL, 7, 59)", (label,))
    db.commit()
    db.close()


NVL_RX = "NVLink bytes received [Throughput %]"
NVL_TX = "NVLink bytes transmitted [Throughput %]"
SM_ACTIVE = "SMs Active [Throughput %]"


def _add_gpu_metrics(path: Path, samples: list[tuple[int, str, int]]) -> None:
    """Add nsys GPU metric tables with (timestamp_ns, metric, value) samples to an export.

    Every metric is listed twice in TARGET_INFO_GPU_METRICS (two GPU types), as nsys does for a
    report that saw more than one GPU; the query must not double-count it.
    """
    names = {name: index for index, name in enumerate(sorted({metric for _, metric, _ in samples}))}
    db = sqlite3.connect(path)
    db.executescript(
        "CREATE TABLE GPU_METRICS(timestamp INT, typeId INT, metricId INT, value INT);"
        "CREATE TABLE TARGET_INFO_GPU_METRICS(typeId INT, sourceId INT, typeName TEXT, metricId INT, metricName TEXT);"
    )
    db.executemany(
        "INSERT INTO TARGET_INFO_GPU_METRICS VALUES (?, 0, 'gpu', ?, ?)",
        [(type_id, index, name) for name, index in names.items() for type_id in (1, 2)],
    )
    db.executemany(
        "INSERT INTO GPU_METRICS VALUES (?, 1, ?, ?)", [(ts, names[metric], value) for ts, metric, value in samples]
    )
    db.commit()
    db.close()


def _write_index(nsys_dir: Path, labels: list[str]) -> None:
    rows = "".join(f"{i}\t{label}\tsingle\tisolated\t0\t1\n" for i, label in enumerate(labels, start=1))
    (nsys_dir / "index.tsv").write_text(
        "capture_index\tlabel\tscenario\tphase\tstart_unix\tend_unix\n" + rows, encoding="utf-8"
    )


def _stamp_export(report: Path) -> None:
    stamp = report.stat().st_mtime + 10  # newer export -> no nsys needed
    os.utime(report.with_suffix(".sqlite"), (stamp, stamp))


def _make_run(
    tmp_path: Path,
    *,
    per_rank_trials: dict[int, list[list[tuple[str, int, int, int]]]],
    iters: int,
    algo: str = "symm",
) -> Path:
    label = f"{algo}/{CASE}"
    algo_dir = tmp_path / algo
    nsys_dir = algo_dir / "nsys"
    nsys_dir.mkdir(parents=True)
    (algo_dir / "metadata.json").write_text(json.dumps({"world_size": len(per_rank_trials)}), encoding="utf-8")
    _write_index(nsys_dir, [label])
    record = {
        "algo": algo,
        "scenario": "single",
        "phase": "isolated",
        "group_name": "single2",
        "case": CASE,
        "group_size": 2,
        "ranks": sorted(per_rank_trials),
        "workload": "ag+rs",
        "iters": iters,
        "makespan_ms": 0.002,
        "nsys_label": label,
    }
    (algo_dir / "results.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    for rank, trials in per_rank_trials.items():
        report = nsys_dir / f"rank{rank}.1.nsys-rep"
        report.write_bytes(b"placeholder")
        _make_report(report.with_suffix(".sqlite"), label=label, trials=trials, pid=4000 + rank)
        _stamp_export(report)
    return tmp_path


def _query(path: Path, label: str):
    return analyse.query_report(path, analyse.load_queries(), algo="symm", rank=0, label=label)


@pytest.mark.unit
def test_sql_file_defines_every_query():
    assert set(analyse.load_queries()) == {"kernels", "overlap", "label", "gpu_metrics"}


@pytest.mark.unit
def test_query_accepts_the_case_range_it_was_filed_under_and_ignores_trials(tmp_path):
    path = tmp_path / "r.sqlite"
    _make_report(path, label=f"symm/{CASE}", trials=[[(AG, 20, 30, 1030)], [(AG, 20, 30, 1030)]])
    kernels, _ = _query(path, f"symm/{CASE}")
    assert {row.label for row in kernels} == {f"symm/{CASE}"}


@pytest.mark.unit
def test_query_refuses_a_report_filed_under_another_case(tmp_path):
    path = tmp_path / "r.sqlite"
    _make_report(path, label=f"symm/{CASE}", trials=[[(AG, 20, 30, 1030)]])
    with pytest.raises(RuntimeError, match="filed as symm/other"):
        _query(path, "symm/other")


@pytest.mark.unit
def test_query_refuses_a_report_without_a_case(tmp_path):
    path = tmp_path / "r.sqlite"
    db = sqlite3.connect(path)
    db.executescript(SCHEMA)
    db.close()
    with pytest.raises(RuntimeError, match=r"case NVTX ranges are \[\]"):
        _query(path, f"symm/{CASE}")


@pytest.mark.unit
def test_report_of_a_rank_that_sat_the_case_out_has_no_kernel_tables_and_no_rows(tmp_path):
    # Regression: a rank outside every group launches no kernel in its window, so nsys exports
    # no CUPTI_ACTIVITY_KIND_KERNEL table; that used to abort the whole analysis.
    path = tmp_path / "r.sqlite"
    _make_idle_report(path, label=f"symm/{IDLE_CASE}")
    assert _query(path, f"symm/{IDLE_CASE}") == ([], {})


@pytest.mark.unit
def test_kernels_query_keeps_only_collectives_launched_inside_trials(tmp_path):
    path = tmp_path / "r.sqlite"
    label = "ring/all_gather-16MiB-bf16-single-n8"
    _make_report(path, label=label, trials=[[(AG, 20, 30, 1030), (AG, 25, 1030, 2030)]])
    kernels, overlap = _query(path, label)
    assert len(kernels) == 1
    row = kernels[0]
    assert (row.label, row.trial, row.op, row.kernel) == (label, "trial0", "ag", AG)
    assert (row.kernels, row.busy_ns, row.min_ns, row.max_ns, row.streams) == (2, 2000, 1000, 1000, 1)
    assert row.last_end_ns - row.first_start_ns == 2000
    assert overlap == {}  # no RS kernels, so the AG×RS join is empty


@pytest.mark.unit
def test_overlap_query_measures_ag_rs_intersection(tmp_path):
    path = tmp_path / "r.sqlite"
    trials = [
        [(AG, 20, 100, 1100), (RS, 21, 600, 1600)],  # 500 ns overlap
        [(AG, 20, 100, 1100), (RS, 21, 1100, 2100)],  # back to back, no overlap
    ]
    label = "nvls/all_gather+reduce_scatter-64MiB-bf16-single-n8"
    _make_report(path, label=label, trials=trials)
    kernels, overlap = _query(path, label)
    assert {(row.trial, row.op) for row in kernels} == {
        ("trial0", "ag"),
        ("trial0", "rs"),
        ("trial1", "ag"),
        ("trial1", "rs"),
    }
    assert overlap == {(label, "trial0"): 500, (label, "trial1"): 0}


@pytest.mark.unit
def test_schema_check_names_the_missing_table(tmp_path):
    path = tmp_path / "r.sqlite"
    db = sqlite3.connect(path)
    db.executescript("CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT);")
    db.close()
    with pytest.raises(RuntimeError, match="NVTX_EVENTS is missing"):
        _query(path, f"symm/{CASE}")


@pytest.mark.unit
@pytest.mark.parametrize(("name", "expected"), [("rank0.1.nsys-rep", (0, 1)), ("rank12.176.nsys-rep", (12, 176))])
def test_raw_capture_reads_the_launcher_rank_and_the_nsys_capture_number(name, expected):
    assert analyse.raw_capture(Path(name)) == expected


@pytest.mark.unit
@pytest.mark.parametrize("name", ["profile.nsys-rep", "rank12.nsys-rep", "rank3_2.nsys-rep"])
def test_raw_capture_rejects_names_without_a_capture_number(name):
    with pytest.raises(ValueError, match="cannot read a rank and capture index"):
        analyse.raw_capture(Path(name))


@pytest.mark.unit
def test_summary_takes_slowest_rank_then_median_over_trials(tmp_path):
    # Rank 1 is slower in trial0, rank 0 in trial1; iters=2 halves every per-iteration time.
    fast = [(AG, 20, 100, 1100), (RS, 21, 600, 1600)]
    slow = [(AG, 20, 100, 3100), (RS, 21, 600, 1600)]
    run = _make_run(tmp_path, per_rank_trials={0: [fast, slow, fast], 1: [slow, fast, fast]}, iters=2)
    analyse.main(["--output-dir", str(run), "--wait-seconds", "0"])
    report = (run / "nsys_kernels.md").read_text(encoding="utf-8")
    rows = [line for line in report.splitlines() if line.startswith("| symm")]
    assert len(rows) == 1
    cells = [cell.strip() for cell in rows[0].strip("|").split("|")]
    # slowest-rank AG per trial = [3000, 3000, 1000] ns -> median 3000 / 2 iters = 1500 ns
    assert cells[4] == "0.002"  # AG ms/iter (0.0015 rounds to 0.002)
    assert cells[5] == "0.001"  # RS 1000 ns / 2 = 0.0005 ms -> 0.001
    assert cells[-1] == f"{AG};{RS}"
    csv_lines = (run / "symm" / "nsys" / "kernels.csv").read_text(encoding="utf-8").splitlines()
    assert csv_lines[0].startswith("algo,rank,report,label,trial,op,kernel")
    assert len(csv_lines) == 1 + 2 * 3 * 2  # header + ranks × trials × ops
    assert csv_lines[1].split(",")[2] == f"symm-{CASE}.rank_00"


@pytest.mark.unit
def test_reports_are_filed_under_their_case_and_reused_on_a_second_run(tmp_path):
    trial = [(AG, 20, 100, 1100), (RS, 21, 600, 1600)]
    run = _make_run(tmp_path, per_rank_trials={0: [trial], 1: [trial]}, iters=1)
    analyse.main(["--output-dir", str(run), "--wait-seconds", "0"])
    case_dir = run / "symm" / "cases" / f"symm-{CASE}"
    assert sorted(path.name for path in case_dir.iterdir()) == [
        f"symm-{CASE}.rank_00.nsys-rep",
        f"symm-{CASE}.rank_00.sqlite",
        f"symm-{CASE}.rank_01.nsys-rep",
        f"symm-{CASE}.rank_01.sqlite",
    ]
    assert not list((run / "symm" / "nsys").glob("rank*"))
    first = (run / "nsys_kernels.md").read_text(encoding="utf-8")
    # Nothing is left in nsys/, so the second run must work from the filed reports alone.
    analyse.main(["--output-dir", str(run), "--wait-seconds", "0"])
    assert (run / "nsys_kernels.md").read_text(encoding="utf-8") == first


@pytest.mark.unit
def test_filing_refuses_an_index_label_of_another_algorithm(tmp_path):
    run = _make_run(tmp_path, per_rank_trials={0: [[(AG, 20, 100, 1100)]]}, iters=1)
    report = run / "symm" / "nsys" / "rank0.1.nsys-rep"
    with pytest.raises(RuntimeError, match="does not name a symm case"):
        analyse.file_reports(run / "symm", [report], [f"ring/{CASE}"], world_size=1)


@pytest.mark.unit
def test_filing_refuses_a_capture_missing_from_the_index(tmp_path):
    run = _make_run(tmp_path, per_rank_trials={0: [[(AG, 20, 100, 1100)]]}, iters=1)
    report = run / "symm" / "nsys" / "rank0.2.nsys-rep"
    report.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="capture 2 is not in nsys/index.tsv"):
        analyse.file_reports(run / "symm", [report], [f"symm/{CASE}"], world_size=1)


@pytest.mark.unit
def test_filing_maps_the_nsys_capture_number_to_the_index_even_when_files_arrive_late(tmp_path):
    algo_dir = tmp_path / "nvls"
    nsys_dir = algo_dir / "nsys"
    nsys_dir.mkdir(parents=True)
    labels = ["nvls/all_gather-16MiB-bf16-single-n4", "nvls/all_reduce-1024MiB-bf16-single-n4"]
    # Capture 2 of rank 1 is filed first (capture 1 has not landed yet); it must not shift to capture 1.
    late = nsys_dir / "rank1.2.nsys-rep"
    late.write_bytes(b"x")
    (filed,) = analyse.file_reports(algo_dir, [late], labels, world_size=4)
    assert filed == (
        algo_dir
        / "cases"
        / "nvls-all_reduce-1024MiB-bf16-single-n4"
        / "nvls-all_reduce-1024MiB-bf16-single-n4.rank_01.nsys-rep"
    )
    assert analyse.filed_label("nvls", filed) == labels[1]


@pytest.mark.unit
def test_filed_rank_reads_the_padded_suffix():
    assert analyse.filed_rank(Path(f"symm-{CASE}.rank_07.nsys-rep")) == 7
    with pytest.raises(ValueError, match="filed report"):
        analyse.filed_rank(Path("rank0.1.nsys-rep"))


@pytest.mark.unit
def test_filed_label_strips_the_algorithm_prefix_of_hyphenated_impls():
    report = Path("cases/symmetric-tma-all_gather-64MiB-bf16-single-n4/x.rank_00.nsys-rep")
    assert analyse.filed_label("symmetric-tma", report) == "symmetric-tma/all_gather-64MiB-bf16-single-n4"
    with pytest.raises(ValueError, match="does not start with"):
        analyse.filed_label("nvls", report)


@pytest.mark.unit
def test_a_rank_outside_the_case_is_filed_and_skipped_instead_of_failing_the_analysis(tmp_path):
    trial = [(AG, 20, 100, 1100), (RS, 21, 600, 1600)]
    run = _make_run(tmp_path, per_rank_trials={0: [trial], 1: [trial], 2: [trial]}, iters=1)
    # Rank 2 is outside the 2-rank group: its report has the case range but no kernel tables.
    report = run / "symm" / "nsys" / "rank2.1.nsys-rep"
    report.with_suffix(".sqlite").unlink()
    _make_idle_report(report.with_suffix(".sqlite"), label=f"symm/{CASE}")
    _stamp_export(report)
    analyse.main(["--output-dir", str(run), "--wait-seconds", "0"])
    assert (run / "symm" / "cases" / f"symm-{CASE}" / f"symm-{CASE}.rank_02.nsys-rep").exists()
    rows = [
        line for line in (run / "nsys_kernels.md").read_text(encoding="utf-8").splitlines() if line.startswith("|")
    ]
    assert len(rows) == 3  # header, separator, the one case


@pytest.mark.unit
def test_file_only_renames_one_pass_without_exporting(tmp_path):
    trial = [(AG, 20, 100, 1100)]
    run = _make_run(tmp_path, per_rank_trials={0: [trial], 1: [trial]}, iters=1)
    _make_run(tmp_path, per_rank_trials={0: [trial], 1: [trial]}, iters=1, algo="ring")
    for report in (run / "symm" / "nsys").glob("rank*.nsys-rep"):
        report.with_suffix(".sqlite").unlink()  # an export would need nsys; filing must not
    analyse.main(["--output-dir", str(run), "--algo", "symm", "--file-only", "--wait-seconds", "0"])
    assert sorted(p.name for p in (run / "symm" / "cases" / f"symm-{CASE}").iterdir()) == [
        f"symm-{CASE}.rank_00.nsys-rep",
        f"symm-{CASE}.rank_01.nsys-rep",
    ]
    assert len(list((run / "ring" / "nsys").glob("rank*.nsys-rep"))) == 2  # --algo leaves other passes alone
    assert not (run / "nsys_kernels.md").exists()


@pytest.mark.unit
def test_one_failing_pass_does_not_stop_the_others(tmp_path):
    trial = [(AG, 20, 100, 1100), (RS, 21, 600, 1600)]
    run = _make_run(tmp_path, per_rank_trials={0: [trial]}, iters=1)
    _make_run(tmp_path, per_rank_trials={0: [trial]}, iters=1, algo="ring")
    (run / "ring" / "metadata.json").unlink()
    with pytest.raises(RuntimeError, match="nsys analysis failed for ring"):
        analyse.main(["--output-dir", str(run), "--wait-seconds", "0"])
    report = (run / "nsys_kernels.md").read_text(encoding="utf-8")
    assert "| symm" in report and "| ring" not in report


@pytest.mark.unit
def test_summary_values_are_per_iteration_and_overlap_is_relative_to_the_shorter_op():
    kernels = [
        analyse.KernelRow("symm", 0, "r", "L", "trial0", "ag", AG, 0, 4, 4_000_000, 1, 1, 0, 6_000_000, 1, 6, 6, 512),
        analyse.KernelRow("symm", 0, "r", "L", "trial0", "rs", RS, 0, 4, 2_000_000, 1, 1, 0, 5_000_000, 1, 6, 6, 512),
    ]
    results = [
        {
            "nsys_label": "L",
            "ranks": [0],
            "group_name": "g",
            "group_size": 4,
            "workload": "ag+rs",
            "phase": "isolated",
            "iters": 4,
            "makespan_ms": 2.0,
        }
    ]
    (row,) = analyse.summarize(kernels, {(0, "L", "trial0"): 1_000_000}, results)
    assert row["ag_ms"] == pytest.approx(1.0)
    assert row["rs_ms"] == pytest.approx(0.5)
    assert row["overlap_ms"] == pytest.approx(0.25)
    assert row["overlap_fraction"] == pytest.approx(0.5)  # 0.25 of the 0.5 ms RS
    assert row["span_ms"] == pytest.approx(1.5)
    assert row["span_over_event"] == pytest.approx(0.75)


@pytest.mark.unit
def test_summary_ignores_ranks_outside_every_group():
    kernels = [analyse.KernelRow("ring", 5, "r", "L", "trial0", "ag", AG, 0, 1, 10, 10, 10, 0, 10, 1, 6, 6, 512)]
    results = [
        {"nsys_label": "L", "ranks": [0, 1], "group_name": "g", "group_size": 2, "iters": 1, "makespan_ms": 1.0}
    ]
    assert analyse.summarize(kernels, {}, results) == []


@pytest.mark.unit
def test_wait_returns_immediately_when_reports_are_complete_and_stable(tmp_path):
    for rank in range(2):
        (tmp_path / f"rank{rank}.1.nsys-rep").write_bytes(b"x")
    # First pass records sizes, second pass sees them unchanged.
    assert len(analyse.wait_for_reports(tmp_path, expected=2, timeout_s=5, poll_s=0.01)) == 2


@pytest.mark.unit
def test_wait_gives_up_and_returns_what_exists(tmp_path):
    (tmp_path / "rank0.1.nsys-rep").write_bytes(b"x")
    assert len(analyse.wait_for_reports(tmp_path, expected=4, timeout_s=0, poll_s=0.01)) == 1


@pytest.mark.unit
def test_main_refuses_a_directory_without_profiled_runs(tmp_path):
    with pytest.raises(FileNotFoundError, match="nsys/index.tsv"):
        analyse.main(["--output-dir", str(tmp_path), "--wait-seconds", "0"])


def test_algo_order_puts_torch_families_first_then_cpp_impls_by_name():
    names = ["tma", "symm", "ce", "ring", "stmc", "nvls"]
    assert sorted(names, key=analyse.algo_order) == ["ring", "nvls", "symm", "ce", "stmc", "tma"]


@pytest.mark.unit
def test_kernels_query_classifies_all_reduce_and_the_gate_stays_outside_trials(tmp_path):
    path = tmp_path / "r.sqlite"
    label = "nvls/all_reduce-64MiB-bf16-single-n8"
    _make_report(path, label=label, trials=[[(AR, 20, 100, 1100), (AR, 25, 1100, 2100)]])
    # The torch benchmark launches its start gate (an AllReduce) before the trial range opens.
    db = sqlite3.connect(path)
    db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (10000, 10005, ?, 999, 0)", ((4242 << 24) | 7,))
    db.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (10006, 10010, 0, 3, 999, ?, 5, 0, 1, 64)", (4242 << 24,)
    )
    db.commit()
    db.close()
    kernels, overlap = _query(path, label)
    assert [(row.trial, row.op, row.kernel, row.kernels, row.busy_ns) for row in kernels] == [
        ("trial0", "ar", AR, 2, 2000)
    ]
    assert overlap == {}


@pytest.mark.unit
def test_summary_counts_only_the_ops_of_the_case_workload():
    # A trial range that still contains the gate (C++ benchmark) shows an AllReduce in an AG case.
    kernels = [
        analyse.KernelRow("ce", 0, "r", "L", "trial0", "ag", AG, 0, 2, 2_000_000, 1, 1, 0, 2_000_000, 1, 6, 6, 512),
        analyse.KernelRow("ce", 0, "r", "L", "trial0", "ar", AR, 0, 1, 5_000, 5_000, 5_000, 0, 5_000, 1, 6, 6, 512),
    ]
    results = [
        {
            "nsys_label": "L",
            "ranks": [0],
            "group_name": "g",
            "group_size": 4,
            "workload": "ag",
            "phase": "isolated",
            "iters": 2,
            "makespan_ms": 1.0,
        }
    ]
    (row,) = analyse.summarize(kernels, {}, results)
    assert row["ag_ms"] == pytest.approx(1.0)
    assert row["ar_ms"] == 0.0
    assert row["kernels"] == AG
    # In an "ar" case the all-reduce kernels are the measurement.
    results[0]["workload"] = "ar"
    (row,) = analyse.summarize(kernels, {}, results)
    assert row["ar_ms"] == pytest.approx(0.0025)
    assert row["ag_ms"] == 0.0


@pytest.mark.unit
def test_gpu_metrics_query_averages_samples_inside_each_trials_kernel_window(tmp_path):
    path = tmp_path / "r.sqlite"
    label = "symm/all_gather-1024MiB-bf16-single-n4"
    # Trial windows on the GPU: trial0 = [10_100, 12_100], trial1 = [110_100, 110_600].
    _make_report(path, label=label, trials=[[(AG, 20, 100, 1100), (AG, 25, 1100, 2100)], [(AG, 20, 100, 600)]])
    _add_gpu_metrics(
        path,
        [
            (10_050, NVL_TX, 99),  # before the first kernel (start gate, sleep): excluded
            (10_500, NVL_TX, 40),
            (11_500, NVL_TX, 60),
            (11_500, NVL_RX, 90),
            (12_200, NVL_TX, 99),  # after the last kernel: excluded
            (110_300, NVL_TX, 10),
        ],
    )
    rows = analyse.query_gpu_metrics(path, analyse.load_queries(), algo="symm", rank=3)
    got = {(row.trial, row.metric): (row.samples, row.mean, row.peak, row.window_ns) for row in rows}
    assert got == {
        ("trial0", NVL_RX): (1, 90.0, 90, 2000),
        ("trial0", NVL_TX): (2, 50.0, 60, 2000),
        ("trial1", NVL_TX): (1, 10.0, 10, 500),
    }
    assert {(row.algo, row.rank, row.label) for row in rows} == {("symm", 3, label)}


@pytest.mark.unit
def test_gpu_metrics_query_is_empty_for_reports_without_metrics_or_kernels(tmp_path):
    path = tmp_path / "r.sqlite"
    _make_report(path, label=f"symm/{CASE}", trials=[[(AG, 20, 30, 1030)]])
    assert analyse.query_gpu_metrics(path, analyse.load_queries(), algo="symm", rank=0) == []
    idle = tmp_path / "idle.sqlite"
    _make_idle_report(idle, label=f"symm/{IDLE_CASE}")
    _add_gpu_metrics(idle, [(15, NVL_TX, 1)])
    assert analyse.query_gpu_metrics(idle, analyse.load_queries(), algo="symm", rank=0) == []


def _metric(rank: int, trial: str, metric: str, value: float, *, samples: int = 10, label: str = "L"):
    return analyse.MetricRow("symm", rank, "r", label, trial, metric, samples, value, value + 5, 2_000_000)


@pytest.mark.unit
def test_metric_summary_keeps_the_rank_spread_and_takes_the_median_over_trials():
    results = [
        {"nsys_label": "L", "ranks": [0, 1], "group_name": "g", "workload": "ag", "phase": "isolated"},
    ]
    metrics = [
        _metric(0, "trial0", NVL_TX, 40.0),
        _metric(1, "trial0", NVL_TX, 60.0, samples=3),
        _metric(0, "trial1", NVL_TX, 20.0),
        _metric(1, "trial1", NVL_TX, 20.0),
        _metric(0, "trial2", NVL_TX, 30.0),
        _metric(1, "trial2", NVL_TX, 50.0),
        _metric(0, "trial0", SM_ACTIVE, 80.0),  # not an NVLink / DRAM / C2C metric
        _metric(7, "trial0", NVL_TX, 99.0),  # a rank outside every group
    ]
    (row,) = analyse.summarize_metrics(metrics, results)
    assert (row["metric"], row["trials"], row["ranks"], row["min_samples"]) == (NVL_TX, 3, 2, 3)
    assert row["mean"] == pytest.approx(40.0)  # trial means 50, 20, 40
    assert row["rank_min"] == pytest.approx(30.0)  # 40, 20, 30
    assert row["rank_max"] == pytest.approx(50.0)  # 60, 20, 50
    assert row["peak"] == pytest.approx(55.0)  # 65, 25, 55
    assert row["window_ms"] == pytest.approx(2.0)


@pytest.mark.unit
def test_main_writes_the_gpu_metric_report_only_for_runs_with_metrics(tmp_path):
    trial = [(AG, 20, 100, 1100), (RS, 21, 600, 1600)]
    run = _make_run(tmp_path, per_rank_trials={0: [trial], 1: [trial]}, iters=1)
    analyse.main(["--output-dir", str(run), "--wait-seconds", "0"])
    assert not (run / "nsys_gpu_metrics.md").exists()
    assert not (run / "symm" / "nsys" / "gpu_metrics.csv").exists()

    for rank, value in ((0, 30), (1, 50)):
        export = run / "symm" / "cases" / f"symm-{CASE}" / f"symm-{CASE}.rank_{rank:02d}.sqlite"
        _add_gpu_metrics(export, [(10_500, NVL_RX, value), (10_600, SM_ACTIVE, 90)])
        _stamp_export(export.with_suffix(".nsys-rep"))
    analyse.main(["--output-dir", str(run), "--wait-seconds", "0"])
    report = (run / "nsys_gpu_metrics.md").read_text(encoding="utf-8")
    rows = [line for line in report.splitlines() if line.startswith("| symm")]
    assert len(rows) == 1  # SMs Active is in the CSV but not in the report
    cells = [cell.strip() for cell in rows[0].strip("|").split("|")]
    assert cells[4:8] == [NVL_RX, "40.0", "30.0", "50.0"]
    csv_lines = (run / "symm" / "nsys" / "gpu_metrics.csv").read_text(encoding="utf-8").splitlines()
    assert csv_lines[0] == "algo,rank,report,label,trial,metric,samples,mean,peak,window_ns"
    assert len(csv_lines) == 1 + 2 * 2  # header + ranks × metrics


@pytest.mark.unit
def test_kernels_query_counts_send_recv_as_p2p_and_reports_the_grid(tmp_path):
    path = tmp_path / "r.sqlite"
    label = "symm/all_gather+write-1024MiB-bf16-single-n4-cta-p2p32"
    _make_report(path, label=label, trials=[[(AG, 20, 100, 1100), (SENDRECV, 21, 100, 900)]])
    kernels, overlap = _query(path, label)
    got = {row.op: (row.kernel, row.grid_min, row.grid_max, row.block_x) for row in kernels}
    assert got == {"ag": (AG, 6, 6, 512), "p2p": (SENDRECV, 16, 16, 512)}
    assert overlap == {}  # the AG x RS overlap ignores send/recv


@pytest.mark.unit
def test_summary_keeps_p2p_kernels_of_read_write_workloads_and_lists_their_ctas():
    kernels = [
        analyse.KernelRow("symm", 0, "r", "L", "trial0", "ag", AG, 0, 2, 2_000_000, 1, 1, 0, 2_000_000, 1, 6, 6, 512),
        analyse.KernelRow(
            "symm", 0, "r", "L", "trial0", "p2p", SENDRECV, 0, 2, 1_000_000, 1, 1, 0, 1_000_000, 1, 16, 32, 640
        ),
    ]
    results = [
        {
            "nsys_label": "L",
            "ranks": [0],
            "group_name": "g",
            "group_size": 4,
            "workload": "ag+write",
            "phase": "isolated",
            "iters": 2,
            "makespan_ms": 1.0,
        }
    ]
    (row,) = analyse.summarize(kernels, {}, results)
    assert row["p2p_ms"] == pytest.approx(0.5)
    assert row["grid"] == "ag=6;p2p=16|32"
    # In an AG-only case a stray send/recv is not part of the workload.
    results[0]["workload"] = "ag"
    (row,) = analyse.summarize(kernels, {}, results)
    assert row["p2p_ms"] == 0.0 and row["grid"] == "ag=6"
    assert "CTAs (gridX)" in analyse.render([row])
