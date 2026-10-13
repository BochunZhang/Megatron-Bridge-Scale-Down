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

"""CPU-only checks for the GB200 NVLS collective benchmark and its report.

The real benchmark needs CUDA, NCCL and an NVLink domain. These tests cover the parts that
do not: layout and bandwidth formulas, the ring-vs-NVLS comparison, option handling, and the
requirement that both entry points stay importable with the standard library alone.
"""

import argparse
import ast
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


BENCHMARK_DIR = Path(__file__).resolve().parents[3] / "examples" / "gb200" / "nvls-collective"
BENCHMARK_PATH = BENCHMARK_DIR / "nvls_collective_benchmark.py"
METRICS_PATH = BENCHMARK_DIR / "collective_metrics.py"
STDLIB_ONLY_IMPORTS = {
    "__future__",
    "argparse",
    "contextlib",
    "dataclasses",
    "json",
    "logging",
    "os",
    "socket",
    "time",
    "collections.abc",
    "pathlib",
    "statistics",
}


def _load(name: str, path: Path, *, stub_torch: bool = False) -> types.ModuleType:
    added_path = stub_torch and str(path.parent) not in sys.path
    if added_path:
        sys.path.insert(0, str(path.parent))
    stubbed = stub_torch and importlib.util.find_spec("torch") is None
    if stubbed:
        torch_stub = types.ModuleType("torch")
        torch_stub.__path__ = []  # type: ignore[attr-defined]
        torch_stub.distributed = types.ModuleType("torch.distributed")  # type: ignore[attr-defined]
        torch_stub.bfloat16 = "bfloat16"  # type: ignore[attr-defined]
        torch_stub.float32 = "float32"  # type: ignore[attr-defined]
        cuda_stub = types.ModuleType("torch.cuda")
        cuda_stub.__path__ = []  # type: ignore[attr-defined]
        cuda_stub.profiler = types.ModuleType("torch.cuda.profiler")  # type: ignore[attr-defined]
        cuda_stub.nvtx = types.ModuleType("torch.cuda.nvtx")  # type: ignore[attr-defined]
        torch_stub.cuda = cuda_stub  # type: ignore[attr-defined]
        stub_modules = {
            "torch": torch_stub,
            "torch.distributed": torch_stub.distributed,
            "torch.cuda": cuda_stub,
            "torch.cuda.profiler": cuda_stub.profiler,
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
        if stubbed:
            for name in stub_modules:
                del sys.modules[name]
        if added_path:
            sys.path.remove(str(path.parent))
    return module


metrics = _load("gb200_nvls_metrics", METRICS_PATH)
benchmark = _load("gb200_nvls_benchmark", BENCHMARK_PATH, stub_torch=True)


def _imported_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return names


@pytest.mark.unit
@pytest.mark.parametrize("path", [BENCHMARK_PATH, METRICS_PATH])
def test_both_entry_points_avoid_heavy_dependencies(path):
    # torch is the only third-party import; the sibling module is loaded from the same directory.
    assert _imported_names(path) <= STDLIB_ONLY_IMPORTS | {
        "torch",
        "torch.distributed",
        "torch.cuda",
        "torch.cuda.profiler",
        "collective_metrics",
    }


@pytest.mark.unit
def test_benchmark_module_imports_with_stdlib_only():
    assert callable(benchmark.main)


@pytest.mark.unit
def test_defaults_cover_the_requested_matrix():
    args = benchmark.parse_args(["--algo", "ring", "--output-dir", "/tmp/never-used"])
    assert args.group_sizes == [4, 8, 16, 32]
    assert args.split_totals == [16]
    assert args.split_layouts == ["contiguous"]
    assert args.workloads == ["ag", "rs", "ag+rs"]
    assert args.sizes_mib == [16, 64, 256, 1024]
    assert args.dtype == "bf16"
    assert args.symm_kernel == "default"
    assert args.nsys is False


@pytest.mark.unit
def test_algo_choice_is_the_three_algorithm_families():
    assert sorted(metrics.ALGORITHMS) == ["nvls", "ring", "symm"]
    assert metrics.ALGORITHMS == ("ring", "nvls", "symm")
    assert metrics.NCCL_ALGO_ENV["symm"] is None  # the window path does not set NCCL_ALGO
    for family in metrics.ALGORITHMS:
        benchmark.parse_args(["--algo", family, "--output-dir", "/tmp/never-used"])
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--algo", "tree", "--output-dir", "/tmp/never-used"])


@pytest.mark.unit
def test_split_totals_accepts_none_for_allocations_without_two_groups():
    args = benchmark.parse_args(["--algo", "nvls", "--output-dir", "/tmp/never-used", "--split-totals", "none"])
    assert args.split_totals == []


@pytest.mark.unit
def test_csv_ints_deduplicates_entries_and_ignores_empty_items():
    assert benchmark._csv_ints("4,8,4,16") == [4, 8, 16]
    assert benchmark._csv_ints("4,,8") == [4, 8]


@pytest.mark.unit
def test_csv_ints_rejects_empty_and_non_positive_entries():
    for value in ("", "0,4", "four"):
        with pytest.raises(argparse.ArgumentTypeError, match="^expected|^all entries"):
            benchmark._csv_ints(value)


@pytest.mark.unit
def test_invalid_dtype_and_workload_are_rejected():
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--algo", "ring", "--output-dir", "/tmp/x", "--dtype", "fp16"])
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--algo", "ring", "--output-dir", "/tmp/x", "--workloads", "ag,rs2"])


@pytest.mark.unit
def test_environment_check_requires_matching_algo_and_group_sizes(monkeypatch):
    args = benchmark.parse_args(["--algo", "nvls", "--output-dir", "/tmp/never-used"])
    monkeypatch.delenv("NCCL_ALGO", raising=False)
    with pytest.raises(RuntimeError, match="NCCL_ALGO"):
        benchmark._check_environment("nvls", 8, args)
    monkeypatch.setenv("NCCL_ALGO", "NVLS")
    monkeypatch.delenv("NCCL_NVLS_ENABLE", raising=False)
    with pytest.raises(RuntimeError, match="NCCL_NVLS_ENABLE"):
        benchmark._check_environment("nvls", 32, args)
    monkeypatch.setenv("NCCL_NVLS_ENABLE", "1")
    with pytest.raises(ValueError, match="exceed WORLD_SIZE"):
        benchmark._check_environment("nvls", 8, args)


@pytest.mark.unit
def test_ring_pass_refuses_nvls_channel_allocation(monkeypatch):
    args = benchmark.parse_args(["--algo", "ring", "--output-dir", "/tmp/never-used"])
    monkeypatch.setenv("NCCL_ALGO", "Ring")
    monkeypatch.setenv("NCCL_NVLS_ENABLE", "1")
    with pytest.raises(RuntimeError, match="NCCL_NVLS_ENABLE=0"):
        benchmark._check_environment("ring", 32, args)


@pytest.mark.unit
def test_symm_pass_requires_unset_algo_and_enabled_windows(monkeypatch):
    args = benchmark.parse_args(["--algo", "symm", "--output-dir", "/tmp/never-used"])
    # An algorithm forced by the environment would hide the window-based dispatch.
    monkeypatch.setenv("NCCL_ALGO", "Ring")
    monkeypatch.setenv("NCCL_WIN_ENABLE", "1")
    with pytest.raises(RuntimeError, match="NCCL_ALGO to be unset"):
        benchmark._check_environment("symm", 32, args)
    # An empty NCCL_ALGO counts as unset: the launcher passes the value through as "".
    monkeypatch.setenv("NCCL_ALGO", "")
    benchmark._check_environment("symm", 32, args)
    monkeypatch.delenv("NCCL_ALGO")
    monkeypatch.setenv("NCCL_WIN_ENABLE", "0")
    with pytest.raises(RuntimeError, match="NCCL_WIN_ENABLE=1"):
        benchmark._check_environment("symm", 32, args)


@pytest.mark.unit
def test_symm_pass_accepts_a_pinned_kernel_only_when_it_matches(monkeypatch):
    args = benchmark.parse_args(["--algo", "symm", "--output-dir", "/tmp/x", "--symm-kernel", "ag"])
    monkeypatch.delenv("NCCL_ALGO", raising=False)
    monkeypatch.setenv("NCCL_WIN_ENABLE", "1")
    monkeypatch.setenv("NCCL_SYM_KERNEL", "AllGather_STMC")
    benchmark._check_environment("symm", 32, args)
    monkeypatch.setenv("NCCL_SYM_KERNEL", "ReduceScatter_LDMC")
    with pytest.raises(RuntimeError, match="requires NCCL_SYM_KERNEL in"):
        benchmark._check_environment("symm", 32, args)


@pytest.mark.unit
def test_symm_kernel_pins_are_real_nccl_kernel_names():
    # Names as defined in NCCL's src/sym_kernels.cc kernel table.
    assert benchmark.SYMM_KERNELS["ag"] == ["AllGather_ST", "AllGather_LL", "AllGather_STMC", "AllGather_LLMC"]
    assert benchmark.SYMM_KERNELS["rs"] == ["ReduceScatter_LD", "ReduceScatter_LL", "ReduceScatter_LDMC"]


@pytest.mark.unit
def test_parse_nccl_lines_reports_every_function_and_the_topology():
    lines = [
        "host:1:2 [0] NCCL INFO comm 0x1 rank 0 nRanks 8 nNodes 2 localRanks 4 localRank 0 MNNVL 1",
        "host:1:2 [0] NCCL INFO AllGather: 268435456 Bytes -> Algo NVLS proto Simple channel{Lo..Hi}={0..15}",
        "host:1:2 [0] NCCL INFO ReduceScatter: 268435456 Bytes -> Algo NVLS proto Simple channel{Lo..Hi}={0..15}",
        "host:1:2 [0] NCCL INFO ReduceScatter [Symmetric]: 268435456 Bytes -> Kernel ReduceScatter_LDMC"
        " nchannels 4 nthreads 512 nWorks 1",
    ]
    parsed = benchmark.parse_nccl_lines(lines)
    assert parsed["n_nodes"] == 2
    assert parsed["mnnvl"] is True
    assert parsed["nccl_algo"] == "AllGather=NVLS/Simple;ReduceScatter=NVLS/Simple"
    assert parsed["nccl_kernel"] == "ReduceScatter=ReduceScatter_LDMC"


@pytest.mark.unit
def test_parse_nccl_lines_returns_none_without_evidence():
    assert benchmark.parse_nccl_lines(["NCCL INFO Bootstrap: Using eth0"]) == dict.fromkeys(benchmark.PROBE_FIELDS)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("template", "expected"),
    [
        ("/o/nccl-%h-%p.log", "/o/nccl-node7-42.log"),  # %h stops at the first dot, like getHostName
        ("/o/100%%-%x", "/o/100%-%x"),
    ],
)
def test_nccl_debug_path_matches_nccl_expansion(template, expected):
    assert str(benchmark.nccl_debug_path(template, hostname="node7.cluster.local", pid=42)) == expected


@pytest.mark.unit
def test_nccl_log_returns_only_new_complete_lines(tmp_path, monkeypatch):
    log = tmp_path / "nccl.log"
    log.write_text("first\nsecond\npartial", encoding="utf-8")
    monkeypatch.setenv("NCCL_DEBUG_FILE", str(log))
    reader = benchmark.NcclLog()
    assert reader.consume() == ["first", "second"]
    with log.open("a", encoding="utf-8") as handle:
        handle.write(" line\nthird\n")
    assert reader.consume() == ["partial line", "third"]
    assert reader.consume() == []


@pytest.mark.unit
def test_nccl_log_is_inert_without_a_debug_file(monkeypatch):
    monkeypatch.delenv("NCCL_DEBUG_FILE", raising=False)
    assert benchmark.NcclLog().consume() == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("layout", "expected_a", "expected_b"),
    [
        ("contiguous", [0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]),
        ("interleaved", [0, 2, 4, 6, 8, 10, 12, 14], [1, 3, 5, 7, 9, 11, 13, 15]),
    ],
)
def test_split_groups_are_disjoint_and_cover_every_rank(layout, expected_a, expected_b):
    group_a, group_b = metrics.split_groups(16, layout)
    assert (group_a, group_b) == (expected_a, expected_b)
    assert sorted(group_a + group_b) == list(range(16))
    assert set(group_a).isdisjoint(group_b)


@pytest.mark.unit
def test_split_groups_reject_odd_or_tiny_rank_counts():
    for total_ranks in (0, 1, 7):
        with pytest.raises(ValueError, match="even rank count"):
            metrics.split_groups(total_ranks, "contiguous")
    with pytest.raises(ValueError, match="unknown split layout"):
        metrics.split_groups(8, "blocked")


@pytest.mark.unit
def test_workload_ops_returns_launch_order():
    assert metrics.workload_ops("ag") == ("ag",)
    assert metrics.workload_ops("rs") == ("rs",)
    assert metrics.workload_ops("ag+rs") == ("ag", "rs")
    with pytest.raises(ValueError, match="unknown workload"):
        metrics.workload_ops("both")


@pytest.mark.unit
def test_bus_bandwidth_matches_nccl_tests_definition():
    # 1 GiB in 10 ms over 8 ranks: 107.3741824 * 7/8 GB/s.
    assert metrics.alg_bandwidth_gb_s(total_bytes=1024 * 1024 * 1024, seconds=0.01) == pytest.approx(107.3741824)
    assert metrics.bus_bandwidth_gb_s(total_bytes=1024 * 1024 * 1024, group_size=8, seconds=0.01) == pytest.approx(
        107.3741824 * 7 / 8
    )


@pytest.mark.unit
def test_bandwidth_and_metrics_reject_degenerate_inputs():
    with pytest.raises(ValueError, match="seconds must be positive"):
        metrics.alg_bandwidth_gb_s(total_bytes=1, seconds=0.0)
    with pytest.raises(ValueError, match="group_size must be at least 2"):
        metrics.bus_bandwidth_gb_s(total_bytes=1, group_size=1, seconds=1.0)
    with pytest.raises(ValueError, match="all times must be positive"):
        metrics.overlap_metrics(ag_s=-1.0, rs_s=1.0, both_s=1.0)


@pytest.mark.unit
def test_overlap_metrics_span_full_serialization_to_full_overlap():
    serial = metrics.overlap_metrics(ag_s=0.001, rs_s=0.003, both_s=0.004)
    assert serial["speedup"] == pytest.approx(1.0)
    assert serial["overlap_fraction"] == pytest.approx(0.0)
    perfect = metrics.overlap_metrics(ag_s=0.001, rs_s=0.003, both_s=0.003)
    assert perfect["speedup"] == pytest.approx(4 / 3)
    assert perfect["overlap_fraction"] == pytest.approx(1.0)
    slower = metrics.overlap_metrics(ag_s=0.001, rs_s=0.003, both_s=0.005)
    assert slower["speedup"] < 1.0
    assert slower["overlap_fraction"] < 0.0


def _record(**overrides):
    record = {
        "algo": "ring",
        "scenario": "single",
        "phase": "isolated",
        "total_ranks": 8,
        "layout": None,
        "group_index": 0,
        "group_name": "single8",
        "group_size": 8,
        "ranks": list(range(8)),
        "size_bytes": 256 * 1024 * 1024,
        "size_mib": 256,
        "dtype": "bf16",
        "workload": "ag",
        "makespan_ms": 1.0,
        "ag_ms": 1.0,
        "rs_ms": 1.0,
        "ag_busbw_gb_s": 100.0,
        "rs_busbw_gb_s": 100.0,
        "aggregate_busbw_gb_s": 100.0,
        "check_passed": True,
        "host_bound": False,
        "nccl_algo": None,
        "nccl_kernel": None,
        "n_nodes": 1,
        "mnnvl": False,
        "profiled": False,
    }
    record.update(overrides)
    return record


@pytest.mark.unit
def test_single_rows_report_ag_rs_and_the_concurrent_pair_separately():
    records = [
        _record(workload="ag", makespan_ms=1.0, ag_ms=1.0, aggregate_busbw_gb_s=100.0),
        _record(workload="rs", makespan_ms=2.0, rs_ms=2.0, rs_busbw_gb_s=50.0, aggregate_busbw_gb_s=50.0),
        _record(workload="ag+rs", makespan_ms=3.0, ag_ms=3.0, rs_ms=3.0, aggregate_busbw_gb_s=66.6),
    ]
    rows = metrics.single_rows(records)
    assert len(rows) == 1
    algo, ranks, mib, ag, rs, ag_pair, rs_pair, aggregate, speedup, overlap = rows[0]
    assert (algo, ranks, mib) == ("ring", "8", "256")
    assert (ag, rs) == ("100.0", "50.0")
    assert (ag_pair, rs_pair, aggregate) == ("100.0", "100.0", "66.6")
    assert speedup == "1.00"
    assert overlap == "0.00"


@pytest.mark.unit
def test_single_rows_leave_overlap_columns_empty_without_the_pair():
    rows = metrics.single_rows([_record()])
    assert rows[0][-2:] == ["-", "-"]


@pytest.mark.unit
def test_algo_rows_pair_ring_and_nvls_on_the_same_workload():
    records = [
        _record(algo="ring", aggregate_busbw_gb_s=100.0),
        _record(algo="nvls", aggregate_busbw_gb_s=250.0),
        _record(algo="nvls", size_mib=512, size_bytes=512 * 1024 * 1024, workload="rs"),
    ]
    rows = metrics.algo_rows(records)
    assert len(rows) == 2
    # columns: ranks, MiB, workload, ring, nvls, nvls/ring, symm, symm/ring
    assert rows[0] == ["8", "256", "ag", "100.0", "250.0", "2.50", "-", "-"]
    assert rows[1][-1] == "-"  # no ring record at this size/workload


@pytest.mark.unit
def test_best_of_ranks_every_family_and_names_the_winner():
    records = [
        _record(algo="ring", aggregate_busbw_gb_s=100.0),
        _record(algo="nvls", aggregate_busbw_gb_s=250.0),
        _record(algo="symm", aggregate_busbw_gb_s=310.0),
    ]
    rows = metrics._best_of(records)
    assert rows[0] == ["8", "256", "ag", "100.0", "250.0", "310.0", "symm"]


@pytest.mark.unit
def test_kernel_rows_attribute_evidence_to_its_own_family_and_group():
    records = [
        _record(algo="ring", nccl_algo="AllGather=Ring/Simple"),
        _record(algo="symm", nccl_kernel="AllGather=AllGather_STMC"),
        _record(algo="symm", group_name="single16", group_size=16, n_nodes=4, nccl_kernel=None),
    ]
    assert metrics.kernel_rows(records) == [
        ["ring", "single8", "1", "AllGather=Ring/Simple"],
        ["symm", "single16", "4", "-"],
        ["symm", "single8", "1", "AllGather=AllGather_STMC"],
    ]
    assert metrics.kernel_rows([]) == []


@pytest.mark.unit
def test_report_flags_profiled_runs():
    assert "--nsys" in metrics.render_report([_record(profiled=True)])
    assert "--nsys" not in metrics.render_report([_record(profiled=False)])


@pytest.mark.unit
def test_split_rows_compare_each_group_against_itself():
    common = {"scenario": "split", "total_ranks": 16, "layout": "contiguous", "size_mib": 256}
    records = [
        _record(
            **common,
            workload="ag",
            phase="isolated",
            group_index=0,
            group_size=8,
            makespan_ms=1.0,
            aggregate_busbw_gb_s=100.0,
        ),
        _record(
            **common,
            workload="ag",
            phase="isolated",
            group_index=1,
            group_size=8,
            makespan_ms=1.2,
            aggregate_busbw_gb_s=90.0,
        ),
        _record(
            **common,
            workload="ag",
            phase="concurrent",
            group_index=0,
            group_size=8,
            makespan_ms=1.1,
            aggregate_busbw_gb_s=95.0,
        ),
        _record(
            **common,
            workload="ag",
            phase="concurrent",
            group_index=1,
            group_size=8,
            makespan_ms=1.36,
            aggregate_busbw_gb_s=80.0,
        ),
    ]
    rows = metrics.split_rows(records)
    assert len(rows) == 2
    assert rows[0][:6] == ["ring", "16", "contiguous", "256", "ag", "A"]
    assert rows[0][-1] == "1.100"
    assert rows[1][-1] == "1.133"
    assert all(row[5] in {"A", "B"} for row in rows)


@pytest.mark.unit
def test_render_report_contains_every_question_and_survives_empty_input():
    report = metrics.render_report([_record(), _record(algo="nvls", aggregate_busbw_gb_s=200.0)])
    assert "1. AG、RS 单独与同时发起" in report
    assert "1b. 各实现的绝对吞吐" in report
    assert "2. 各实现相对 ring" in report
    assert "3. 两个半组同时运行" in report
    assert metrics.render_report([]).count("|") > 0


@pytest.mark.unit
def test_load_records_skips_blank_lines(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_text('\n{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    assert [record["a"] for record in metrics.load_records(path)] == [1, 2]


@pytest.mark.unit
def test_compare_merges_both_algorithms_and_rejects_mismatched_files(tmp_path):
    ring_path, nvls_path = tmp_path / "ring.jsonl", tmp_path / "nvls.jsonl"
    ring_path.write_text(json.dumps(_record()) + "\n", encoding="utf-8")
    nvls_path.write_text(json.dumps(_record(algo="nvls")) + "\n", encoding="utf-8")
    output = tmp_path / "compare.md"
    metrics.main(["--input", f"ring={ring_path}", "--input", f"nvls={nvls_path}", "--output", str(output)])
    assert "nvls/ring" in output.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="expected only nvls"):
        metrics.main(["--input", f"nvls={ring_path}", "--output", str(output)])
    with pytest.raises(SystemExit):
        metrics.main(["--input", f"tree={ring_path}", "--output", str(output)])


@pytest.mark.unit
def test_input_argument_requires_a_known_algorithm(tmp_path):
    with pytest.raises(SystemExit):
        metrics.main(["--input", f"tree={tmp_path / 'x.jsonl'}", "--output", str(tmp_path / "o.md")])


@pytest.mark.unit
def test_capture_windows_are_inert_when_profiling_is_off():
    captures = benchmark.CaptureWindows(enabled=False)
    with captures.case("ring/single8/ag/256MiB") as index:
        assert index is None
    assert captures.index == 0
    assert captures.window_unix is None


@pytest.mark.unit
def test_capture_windows_number_cases_and_close_every_range(monkeypatch):
    calls = []
    monkeypatch.setattr(benchmark.torch.cuda, "synchronize", lambda: calls.append("sync"), raising=False)
    monkeypatch.setattr(benchmark.torch.cuda.profiler, "start", lambda: calls.append("start"), raising=False)
    monkeypatch.setattr(benchmark.torch.cuda.profiler, "stop", lambda: calls.append("stop"), raising=False)
    monkeypatch.setattr(benchmark.nvtx, "range_push", lambda name: calls.append(f"push:{name}"), raising=False)
    monkeypatch.setattr(benchmark.nvtx, "range_pop", lambda: calls.append("pop"), raising=False)
    captures = benchmark.CaptureWindows(enabled=True)
    with captures.case("a") as first:
        pass
    with pytest.raises(RuntimeError):
        with captures.case("b"):
            raise RuntimeError("case failed")
    assert first == 1
    assert captures.index == 2
    # Each window is synchronized on both ends, and the profiler stops even when the case raises.
    window = ["sync", "start", "push:{}", "sync", "pop", "stop"]
    assert calls == [step.format("a") for step in window] + [step.format("b") for step in window]
    start, end = captures.window_unix
    assert start <= end
