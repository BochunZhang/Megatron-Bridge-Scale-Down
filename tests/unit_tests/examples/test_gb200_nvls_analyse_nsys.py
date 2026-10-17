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
CASE = "single-n2_ag+rs_bf16_64MiB"
RS = "ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL"
SCHEMA = """
CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT);
CREATE TABLE NVTX_EVENTS(start INT, end INT, text TEXT, textId INT, globalTid INT, eventType INT);
CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INT, end INT, globalTid INT, correlationId INT, nameId INT);
CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(
    start INT, end INT, deviceId INT, streamId INT, correlationId INT, globalPid INT,
    shortName INT, demangledName INT);
"""


def _make_report(path: Path, *, label: str, trials: list[list[tuple[str, int, int, int]]], pid: int = 4242) -> None:
    """Write one rank's export.

    ``trials[i]`` lists kernels as (name, launch_ns, start_ns, end_ns), with launch times relative
    to the trial's NVTX range. A warmup AG before the case window and a sleep kernel inside every
    trial are added so the tests also prove they are excluded.
    """
    tid, gpid = (pid << 24) | 7, pid << 24
    strings = {AG: 1, RS: 2, "sleep_kernel": 3, label: 4}
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
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, 0, ?, ?, ?, ?, 0)",
            (start_ns, end_ns, stream, correlation, gpid, strings[name]),
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
            launch(name, start + at, start + k_start, start + k_end, 1 if name == AG else 2)
    db.commit()
    db.close()


def _make_run(
    tmp_path: Path, *, per_rank_trials: dict[int, list[list[tuple[str, int, int, int]]]], iters: int
) -> Path:
    label = f"symm/{CASE}"
    algo_dir = tmp_path / "symm"
    nsys_dir = algo_dir / "nsys"
    nsys_dir.mkdir(parents=True)
    (algo_dir / "metadata.json").write_text(json.dumps({"world_size": len(per_rank_trials)}), encoding="utf-8")
    (nsys_dir / "index.tsv").write_text(
        f"capture_index\tlabel\tscenario\tphase\tstart_unix\tend_unix\n1\t{label}\tsingle\tisolated\t0\t1\n",
        encoding="utf-8",
    )
    record = {
        "algo": "symm",
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
        sqlite_path = report.with_suffix(".sqlite")
        _make_report(sqlite_path, label=label, trials=trials, pid=4000 + rank)
        stamp = report.stat().st_mtime + 10  # newer export -> no nsys needed
        os.utime(sqlite_path, (stamp, stamp))
    return tmp_path


def _query(path: Path):
    queries = analyse.load_queries()
    return analyse.query_report(path, queries, algo="symm", rank=0)


@pytest.mark.unit
def test_sql_file_defines_every_query():
    assert set(analyse.load_queries()) == {"kernels", "overlap", "label"}


@pytest.mark.unit
def test_label_query_reads_the_case_range_and_ignores_trials(tmp_path):
    path = tmp_path / "r.sqlite"
    _make_report(path, label=f"ring/{CASE}", trials=[[(AG, 20, 30, 1030)], [(AG, 20, 30, 1030)]])
    assert analyse.report_label(path, analyse.load_queries()) == f"ring/{CASE}"


@pytest.mark.unit
def test_label_query_refuses_a_report_without_a_case(tmp_path):
    path = tmp_path / "r.sqlite"
    db = sqlite3.connect(path)
    db.executescript(SCHEMA)
    db.close()
    with pytest.raises(RuntimeError, match="expected one case label"):
        analyse.report_label(path, analyse.load_queries())


@pytest.mark.unit
def test_kernels_query_keeps_only_collectives_launched_inside_trials(tmp_path):
    path = tmp_path / "r.sqlite"
    label = "ring/single-n8_ag_bf16_16MiB"
    _make_report(path, label=label, trials=[[(AG, 20, 30, 1030), (AG, 25, 1030, 2030)]])
    kernels, overlap = _query(path)
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
    label = "nvls/single-n8_ag+rs_bf16_64MiB"
    _make_report(path, label=label, trials=trials)
    kernels, overlap = _query(path)
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
        _query(path)


@pytest.mark.unit
@pytest.mark.parametrize(("name", "rank"), [("rank0.1.nsys-rep", 0), ("rank12.nsys-rep", 12), ("rank3_2.nsys-rep", 3)])
def test_report_rank_reads_the_launcher_prefix(name, rank):
    assert analyse.report_rank(Path(name)) == rank


@pytest.mark.unit
def test_report_rank_rejects_unrelated_files():
    with pytest.raises(ValueError, match="cannot read a rank"):
        analyse.report_rank(Path("profile.nsys-rep"))


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
    assert csv_lines[1].split(",")[2] == f"symm_{CASE}_rank00"


@pytest.mark.unit
def test_reports_are_filed_under_their_case_and_reused_on_a_second_run(tmp_path):
    trial = [(AG, 20, 100, 1100), (RS, 21, 600, 1600)]
    run = _make_run(tmp_path, per_rank_trials={0: [trial], 1: [trial]}, iters=1)
    analyse.main(["--output-dir", str(run), "--wait-seconds", "0"])
    case_dir = run / "symm" / "cases" / CASE
    assert sorted(path.name for path in case_dir.iterdir()) == [
        f"symm_{CASE}_rank00.nsys-rep",
        f"symm_{CASE}_rank00.sqlite",
        f"symm_{CASE}_rank01.nsys-rep",
        f"symm_{CASE}_rank01.sqlite",
    ]
    assert not list((run / "symm" / "nsys").glob("rank*"))
    first = (run / "nsys_kernels.md").read_text(encoding="utf-8")
    # Nothing is left in nsys/, so the second run must work from the filed reports alone.
    analyse.main(["--output-dir", str(run), "--wait-seconds", "0"])
    assert (run / "nsys_kernels.md").read_text(encoding="utf-8") == first


@pytest.mark.unit
def test_filing_refuses_a_report_of_another_algorithm(tmp_path):
    run = _make_run(tmp_path, per_rank_trials={0: [[(AG, 20, 100, 1100)]]}, iters=1)
    nsys_dir = run / "symm" / "nsys"
    report = nsys_dir / "rank0.1.nsys-rep"
    report.with_suffix(".sqlite").unlink()
    _make_report(report.with_suffix(".sqlite"), label=f"ring/{CASE}", trials=[[(AG, 20, 100, 1100)]])
    stamp = report.stat().st_mtime + 20
    os.utime(report.with_suffix(".sqlite"), (stamp, stamp))
    with pytest.raises(RuntimeError, match="does not name a symm case"):
        analyse.file_reports(run / "symm", [report], analyse.load_queries(), nsys_bin="nsys", world_size=1)


@pytest.mark.unit
def test_filed_rank_reads_the_padded_suffix():
    assert analyse.filed_rank(Path(f"symm_{CASE}_rank07.nsys-rep")) == 7
    with pytest.raises(ValueError, match="filed report"):
        analyse.filed_rank(Path("rank0.1.nsys-rep"))


@pytest.mark.unit
def test_summary_values_are_per_iteration_and_overlap_is_relative_to_the_shorter_op():
    kernels = [
        analyse.KernelRow("symm", 0, "r", "L", "trial0", "ag", AG, 0, 4, 4_000_000, 1, 1, 0, 6_000_000, 1),
        analyse.KernelRow("symm", 0, "r", "L", "trial0", "rs", RS, 0, 4, 2_000_000, 1, 1, 0, 5_000_000, 1),
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
    kernels = [analyse.KernelRow("ring", 5, "r", "L", "trial0", "ag", AG, 0, 1, 10, 10, 10, 0, 10, 1)]
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
