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
import contextlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest


BENCHMARK_DIR = Path(__file__).resolve().parents[3] / "examples" / "gb200" / "nvls-collective"
BENCHMARK_PATH = BENCHMARK_DIR / "nvls_collective_benchmark.py"
LAUNCHER_PATH = BENCHMARK_DIR / "run_nvls_collective_benchmark.sh"
METRICS_PATH = BENCHMARK_DIR / "collective_metrics.py"
STDLIB_ONLY_IMPORTS = {
    "__future__",
    "argparse",
    "collections",
    "contextlib",
    "dataclasses",
    "functools",
    "json",
    "logging",
    "os",
    "re",
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
    assert args.split_totals == []
    assert args.late_join is False
    assert args.split_layouts == ["contiguous"]
    assert args.workloads == ["ag", "rs", "ar", "ag+rs"]
    assert args.sizes_mib == [16, 32, 64, 128, 256, 512, 1024, 2048]
    assert args.dtype == "bf16"
    assert args.symm_kernel == "default"
    assert args.nsys is False
    assert args.ncu is False


@pytest.mark.unit
def test_nsys_and_ncu_are_exclusive():
    base = ["--algo", "symm", "--output-dir", "/tmp/never-used"]
    assert benchmark.parse_args([*base, "--ncu"]).ncu is True
    with pytest.raises(SystemExit):
        benchmark.parse_args([*base, "--nsys", "--ncu"])


def _launcher_dry_run(*options: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "GLOO_SOCKET_IFNAME": "lo", "RUN_TIME": "dry-run-test"}
    for name in ("NNODES", "NODE_RANK", "RANK", "MASTER_ADDR", "DLC_MASTER_ADDR", "NCCL_ALGO", "NCCL_NVLS_ENABLE"):
        env.pop(name, None)
    return subprocess.run(
        ["bash", str(LAUNCHER_PATH), "--dry-run", "--algos", "symm", "--workloads", "ag", *options],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def _worker_command(stdout: str) -> list[str]:
    (line,) = [line for line in stdout.splitlines() if "torch.distributed.run" in line]
    words = line.replace("\\", "").split()
    return words[words.index("--no-python") + 1 : words.index("--algo")]


@pytest.mark.unit
@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_launcher_nsys_samples_gpu_metrics_on_each_workers_own_gpu():
    result = _launcher_dry_run("--nsys", "--nsys-gpu-metrics-set", "gb10x", "--nsys-freq", "20000")
    assert result.returncode == 0, result.stderr
    worker = _worker_command(result.stdout)
    # nsys_worker.sh resolves --gpu-metrics-devices from LOCAL_RANK, so one session per GPU.
    assert worker[:2] == ["bash", str(BENCHMARK_DIR / "nsys_worker.sh")]
    assert "--gpu-metrics-frequency=20000" in worker
    assert "--gpu-metrics-set=gb10x" in worker
    assert "--capture-range=cudaProfilerApi" in worker
    assert "CUDA_DEVICE_ORDER=PCI_BUS_ID" in result.stdout
    assert " --nsys " in result.stdout


@pytest.mark.unit
@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.parametrize(
    ("value", "expected", "clamped"),
    [
        (None, 50000, False),
        ("high", 200000, False),
        ("300000", 200000, True),
        ("99999999999999999999", 200000, True),
        ("1", 10, True),
        ("-5", 10, True),
        ("010", 10, False),
    ],
)
def test_launcher_nsys_freq_defaults_clamps_and_accepts_high(value, expected, clamped):
    result = _launcher_dry_run("--nsys", *(() if value is None else ("--nsys-freq", value)))
    assert result.returncode == 0, result.stderr
    assert f"--gpu-metrics-frequency={expected}" in _worker_command(result.stdout)
    assert ("is outside [10, 200000] Hz" in result.stderr) == clamped


@pytest.mark.unit
@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_launcher_ncu_profiles_trial0_of_one_full_group_in_lockstep():
    result = _launcher_dry_run("--ncu", "--ncu-sections", "SpeedOfLight,Nvlink_Tables", "--ncu-port", "50000")
    assert result.returncode == 0, result.stderr
    worker = _worker_command(result.stdout)
    assert worker[0] == "ncu"
    for option in (
        "--communicator=tcp",
        "--communicator-tcp-num-peers=4",
        "--communicator-tcp-port=50000",
        "--lockstep-kernel-launch",
        "--lockstep-nvtx-include=trial0/",
        # Lockstep supports kernel and range replay only; app-range hung in the second range.
        "--replay-mode=range",
        "--nvtx-include=trial0/",
        "--clock-control=none",
        "--cache-control=none",
        "--section=SpeedOfLight",
        "--section=Nvlink_Tables",
    ):
        assert option in worker
    # Every rank is in the one group, so lockstep never waits on an idle rank.
    assert "--group-sizes 4 " in result.stdout and "--split-totals none" in result.stdout
    assert " --ncu " in result.stdout
    # Workers read the installed sections instead of racing to copy them into a shared $HOME.
    assert "NV_COMPUTE_PROFILER_DISABLE_STOCK_FILE_DEPLOYMENT=1" in result.stdout


@pytest.mark.unit
@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.parametrize(
    ("options", "message"),
    [
        (("--nsys", "--ncu"), "exclusive"),
        (("--ncu", "--group-sizes", "2"), "--group-sizes must be 4"),
        (("--ncu", "--split-totals", "4"), "split scenarios leave ranks idle"),
        (("--nsys", "--nsys-freq", "fast"), "integer number of Hz or high"),
    ],
)
def test_launcher_rejects_profiler_options_it_cannot_honor(options, message):
    result = _launcher_dry_run(*options)
    assert result.returncode != 0
    assert re.search(message, result.stderr)


@pytest.mark.unit
def test_algo_choice_is_the_five_algorithm_families():
    assert metrics.ALGORITHMS == ("ring", "nvls", "symm", "symm-tma", "copy-engine")
    assert set(metrics.NCCL_ALGO_ENV) == set(metrics.ALGORITHMS)
    for family in ("symm", "symm-tma", "copy-engine"):
        assert metrics.NCCL_ALGO_ENV[family] is None  # the window paths do not set NCCL_ALGO
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


def _debug_env(monkeypatch, *, ce_threshold: str | None = None):
    """The NCCL logging the implementation check depends on, as the launcher sets it."""
    monkeypatch.setenv("NCCL_DEBUG", "INFO")
    monkeypatch.setenv("NCCL_DEBUG_SUBSYS", "INIT,ENV,TUNING")
    monkeypatch.setenv("NCCL_DEBUG_FILE", "/tmp/nccl-raw/nccl-%h-%p.log")
    if ce_threshold is not None:
        monkeypatch.setenv("NCCL_SYM_CE_THRESHOLD", ce_threshold)


@pytest.mark.unit
def test_environment_check_requires_matching_algo_and_group_sizes(monkeypatch):
    args = benchmark.parse_args(["--algo", "nvls", "--output-dir", "/tmp/never-used"])
    _debug_env(monkeypatch)
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
    _debug_env(monkeypatch, ce_threshold="9223372036854775807")
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
    args = benchmark.parse_args(
        ["--algo", "symm", "--output-dir", "/tmp/x", "--symm-kernel", "ag", "--workloads", "ag"]
    )
    _debug_env(monkeypatch, ce_threshold="9223372036854775807")
    monkeypatch.delenv("NCCL_ALGO", raising=False)
    monkeypatch.setenv("NCCL_WIN_ENABLE", "1")
    monkeypatch.setenv("NCCL_SYM_KERNEL", "AllGather_STMC")
    benchmark._check_environment("symm", 32, args)
    monkeypatch.setenv("NCCL_SYM_KERNEL", "ReduceScatter_LDMC")
    with pytest.raises(RuntimeError, match="requires NCCL_SYM_KERNEL in"):
        benchmark._check_environment("symm", 32, args)


@pytest.mark.unit
def test_symm_kernel_mc_pins_both_multicast_kernels_for_every_workload(monkeypatch):
    # Unlike ag / rs / ar, the per-function pin leaves every workload measurable.
    args = benchmark.parse_args(
        ["--algo", "symm", "--output-dir", "/tmp/x", "--symm-kernel", "mc", "--workloads", "ar,ag,rs,ag+rs"]
    )
    _debug_env(monkeypatch, ce_threshold="9223372036854775807")
    monkeypatch.delenv("NCCL_ALGO", raising=False)
    monkeypatch.setenv("NCCL_WIN_ENABLE", "1")
    monkeypatch.setattr(benchmark.torch.cuda.nccl, "version", lambda: (2, 32, 3))
    monkeypatch.setenv("NCCL_SYM_KERNEL", benchmark.SYMM_MC_PIN)
    benchmark._check_environment("symm", 32, args)
    monkeypatch.setenv("NCCL_SYM_KERNEL", "AllGather_STMC")
    with pytest.raises(RuntimeError, match="--symm-kernel mc requires NCCL_SYM_KERNEL="):
        benchmark._check_environment("symm", 32, args)
    # 2.29 reads NCCL_SYM_KERNEL as one kernel name, so the per-function pin would mask everything.
    monkeypatch.setenv("NCCL_SYM_KERNEL", benchmark.SYMM_MC_PIN)
    monkeypatch.setattr(benchmark.torch.cuda.nccl, "version", lambda: (2, 29, 7))
    with pytest.raises(RuntimeError, match="NCCL >= 2.32"):
        benchmark._check_environment("symm", 32, args)


@pytest.mark.unit
def test_check_implementation_holds_symm_mc_to_stmc_and_ldmc_only():
    pin = benchmark.SYMM_MC_PIN
    assert _check([SYMM_AG, SYMM_RS, SYMM_AG, SYMM_RS], ops=("ag", "rs"), pinned_kernel=pin)[1] == []
    unicast = SYMM_AG.replace("AllGather_STMC", "AllGather_ST")
    rs_ld = SYMM_RS.replace("ReduceScatter_LDMC", "ReduceScatter_LD")
    _, errors = _check([unicast, SYMM_AG, rs_ld, SYMM_RS], ops=("ag", "rs"), pinned_kernel=pin)
    assert errors == [
        "AllGather: expected Symmetric/AllGather_STMC, got Symmetric/AllGather_ST x1",
        "ReduceScatter: expected Symmetric/ReduceScatter_LDMC, got Symmetric/ReduceScatter_LD x1",
    ]
    # AllReduce is not in the pin; SYMM_KERNELS["ar"] still holds it to the multicast kernel.
    assert benchmark.pinned_symm_kernel("ar", pin) is None
    assert benchmark.pinned_symm_kernel("ag2", pin) == "AllGather_STMC"
    assert benchmark.implementation_matches("symm", "ar", "Symmetric/AllReduce_RSxLDMC_AGxSTMC", pinned_kernel=pin)


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
def test_symm_pass_auto_sets_ce_threshold_above_every_payload(monkeypatch, caplog):
    args = benchmark.parse_args(["--algo", "symm", "--output-dir", "/tmp/x", "--sizes-mib", "16,1024"])
    monkeypatch.delenv("NCCL_ALGO", raising=False)
    monkeypatch.setenv("NCCL_WIN_ENABLE", "1")
    _debug_env(monkeypatch)
    # A missing or insufficient NCCL_SYM_CE_THRESHOLD is replaced with INT64_MAX so the caller
    # never has to set it by hand; the benchmark warns and keeps every AllGather on the symmetric
    # kernels instead of falling back to the copy-engine path.
    monkeypatch.delenv("NCCL_SYM_CE_THRESHOLD", raising=False)
    with caplog.at_level("WARNING"):
        benchmark._check_environment("symm", 32, args)
    assert os.environ["NCCL_SYM_CE_THRESHOLD"] == "9223372036854775807"
    # NCCL's default (8Mi elements) sends 1 GiB bf16 all-gathers on <= 64 ranks to the copy engines.
    monkeypatch.setenv("NCCL_SYM_CE_THRESHOLD", str(8 * 1024 * 1024))
    with caplog.at_level("WARNING"):
        benchmark._check_environment("symm", 32, args)
    assert os.environ["NCCL_SYM_CE_THRESHOLD"] == "9223372036854775807"
    # An explicit value already large enough is kept as-is.
    monkeypatch.setenv("NCCL_SYM_CE_THRESHOLD", str(1024 * 1024 * 1024))
    benchmark._check_environment("symm", 32, args)
    assert os.environ["NCCL_SYM_CE_THRESHOLD"] == str(1024 * 1024 * 1024)


@pytest.mark.unit
def test_symm_kernel_pin_requires_the_matching_workload_only(monkeypatch):
    # Pinning AllGather_STMC leaves ReduceScatter without any symmetric kernel.
    args = benchmark.parse_args(["--algo", "symm", "--output-dir", "/tmp/x", "--symm-kernel", "ag"])
    monkeypatch.delenv("NCCL_ALGO", raising=False)
    monkeypatch.setenv("NCCL_WIN_ENABLE", "1")
    monkeypatch.setenv("NCCL_SYM_KERNEL", "AllGather_STMC")
    _debug_env(monkeypatch, ce_threshold="9223372036854775807")
    with pytest.raises(ValueError, match="--workloads ag"):
        benchmark._check_environment("symm", 32, args)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("NCCL_DEBUG", "WARN", "NCCL_DEBUG=INFO"),
        ("NCCL_DEBUG_SUBSYS", "INIT,ENV", "TUNING"),
        ("NCCL_DEBUG_FILE", "/tmp/nccl.log", "%p"),
    ],
)
def test_environment_check_requires_the_nccl_log_the_check_reads(monkeypatch, name, value, message):
    args = benchmark.parse_args(["--algo", "ring", "--output-dir", "/tmp/x"])
    monkeypatch.setenv("NCCL_ALGO", "Ring")
    monkeypatch.setenv("NCCL_NVLS_ENABLE", "0")
    _debug_env(monkeypatch)
    benchmark._check_environment("ring", 32, args)
    monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match=message):
        benchmark._check_environment("ring", 32, args)


# The three TUNING launch lines as nccl-2.29.7 prints them (debug.cc prefix + call-site format).
CLASSIC_AG = "h:1:2 [0] NCCL INFO AllGather: 16777216 Bytes -> Algo NVLS proto SIMPLE channel{Lo..Hi}={0..15}"
SYMM_AG = "h:1:2 [0] NCCL INFO AllGather [Symmetric]: 16777216 Bytes -> Kernel AllGather_STMC nchannels 8 nthreads 512 nWorks 1"
SYMM_RS = "h:1:2 [0] NCCL INFO ReduceScatter [Symmetric]: 16777216 Bytes -> Kernel ReduceScatter_LDMC nchannels 8 nthreads 512 nWorks 1"
CE_AG = "h:1:2 [0] NCCL INFO AllGather [Copy Engine]: 16777216 Bytes -> cudaMemcpy; CE synchronization with NVLS"


@pytest.mark.unit
def test_parse_tuning_line_reads_every_launch_format():
    assert benchmark.parse_tuning_line(CLASSIC_AG) == ("AllGather", "NVLS/SIMPLE", 16777216)
    assert benchmark.parse_tuning_line(SYMM_AG) == ("AllGather", "Symmetric/AllGather_STMC", 16777216)
    assert benchmark.parse_tuning_line(CE_AG) == ("AllGather", "CopyEngine+NVLSSync", 16777216)
    unicast_sync = CE_AG.removesuffix("; CE synchronization with NVLS")
    assert benchmark.parse_tuning_line(unicast_sync) == ("AllGather", "CopyEngine", 16777216)
    assert benchmark.parse_tuning_line("h:1:2 [0] NCCL INFO Channel 00/32 : 0 1 2 3") is None


@pytest.mark.unit
def test_parse_nccl_lines_reports_the_copy_engine_path():
    assert benchmark.parse_nccl_lines([CE_AG, SYMM_RS])["nccl_kernel"] == (
        "AllGather=CopyEngine+NVLSSync;ReduceScatter=ReduceScatter_LDMC"
    )


def _check(
    lines,
    *,
    algo="symm",
    ops=("ag",),
    launches=2,
    size_bytes=2 * 16777216,
    group_size=2,
    pinned_kernel=None,
    gate_launches=0,
):
    # Defaults: a 32 MiB tensor on 2 ranks, so AG/RS lines carry a 16 MiB shard.
    return benchmark.check_implementation(
        lines,
        algo=algo,
        ops=ops,
        launches=launches,
        size_bytes=size_bytes,
        group_size=group_size,
        pinned_kernel=pinned_kernel,
        gate_launches=gate_launches,
    )


GATE = "h:1:2 [0] NCCL INFO AllReduce: 4 Bytes -> Algo RING proto LL channel{Lo..Hi}={0..0}"


@pytest.mark.unit
def test_check_implementation_counts_start_gates_only_where_they_are_logged():
    # Gate comm rank 0 logs one AllReduce per trial on top of the case's own launches.
    assert _check([GATE, SYMM_AG, GATE, SYMM_AG], gate_launches=2)[1] == []
    assert _check([GATE, SYMM_AG, SYMM_AG], gate_launches=2)[1] == [
        "start gate: 1 AllReduce launches in the NCCL log, expected 2"
    ]
    # Any other rank must not see gate lines, and a payload-sized all-reduce is never a gate.
    assert _check([GATE, SYMM_AG, SYMM_AG])[1] == ["start gate: 1 AllReduce launches in the NCCL log, expected 0"]
    big = GATE.replace(": 4 Bytes", ": 4096 Bytes")
    assert _check([big, SYMM_AG, SYMM_AG])[1] == ["unexpected collective AllReduce=RING/LL x1"]


@pytest.mark.unit
def test_check_implementation_accepts_symmetric_kernels_on_every_launch():
    assert _check([SYMM_AG, SYMM_RS, SYMM_AG, SYMM_RS], ops=("ag", "rs")) == (
        "AllGather=Symmetric/AllGather_STMC;ReduceScatter=Symmetric/ReduceScatter_LDMC",
        [],
    )


@pytest.mark.unit
def test_check_implementation_rejects_a_single_copy_engine_launch():
    observed, errors = _check([SYMM_AG, CE_AG])
    assert observed == "AllGather=CopyEngine+NVLSSync|Symmetric/AllGather_STMC"
    assert errors == [
        "AllGather: expected Symmetric/AllGather_ST|AllGather_LL|AllGather_STMC|AllGather_LLMC,"
        " got CopyEngine+NVLSSync x1"
    ]


@pytest.mark.unit
def test_check_implementation_rejects_a_classic_fallback_and_a_pin_mismatch():
    _, errors = _check([SYMM_AG, CLASSIC_AG], pinned_kernel="AllGather_ST")
    assert errors == ["AllGather: expected Symmetric/AllGather_ST, got NVLS/SIMPLE x1, Symmetric/AllGather_STMC x1"]


@pytest.mark.unit
@pytest.mark.parametrize(("algo", "accepted"), [("nvls", True), ("ring", False)])
def test_check_implementation_holds_classic_passes_to_their_algorithm(algo, accepted):
    _, errors = _check([CLASSIC_AG, CLASSIC_AG], algo=algo)
    assert (errors == []) is accepted
    if not accepted:
        assert errors == ["AllGather: expected Ring/*, got NVLS/SIMPLE x2"]


@pytest.mark.unit
def test_check_implementation_counts_launches_sizes_and_foreign_collectives():
    _, errors = _check([SYMM_AG, SYMM_RS], launches=3, size_bytes=8192)
    assert errors == [
        "unexpected collective ReduceScatter=Symmetric/ReduceScatter_LDMC x1",
        "AllGather: 1 launches in the NCCL log, expected 3",
        "AllGather: expected 4096 B per rank, got 16777216 B x1",
    ]
    # No TUNING lines at all (e.g. a non-zero comm rank's log) is a failure, not a pass.
    assert _check([]) == (None, ["AllGather: 0 launches in the NCCL log, expected 2"])


@pytest.mark.unit
def test_case_names_carry_scenario_part_workload_dtype_and_size():
    single = metrics.scenario_name("single", total_ranks=16)
    split = metrics.scenario_name("split", total_ranks=16, layout="contiguous")
    assert (single, split) == ("single-n16", "split-n16-contiguous")
    assert metrics.case_name(single, part=None, workload="ag", dtype="bf16", size_mib=1024) == (
        "all_gather-1024MiB-bf16-single-n16"
    )
    assert metrics.case_name(single, part=None, workload="ar", dtype="bf16", size_mib=1024) == (
        "all_reduce-1024MiB-bf16-single-n16"
    )
    assert metrics.case_name(split, part="AB-concurrent", workload="ag+rs", dtype="fp32", size_mib=64) == (
        "all_gather+reduce_scatter-64MiB-fp32-split-n16-contiguous-AB-concurrent"
    )


@pytest.mark.unit
def test_case_directories_and_rank_files_lead_with_the_algorithm():
    case = "all_reduce-1024MiB-bf16-single-n16"
    assert metrics.case_dir_name("nvls", case) == "nvls-all_reduce-1024MiB-bf16-single-n16"
    assert metrics.rank_file("nvls", case, rank=1, world_size=16, suffix=".nsys-rep") == (
        "nvls-all_reduce-1024MiB-bf16-single-n16.rank_01.nsys-rep"
    )
    assert metrics.rank_file("symm", case, rank=3, world_size=16, suffix=".nccl.log") == (
        "symm-all_reduce-1024MiB-bf16-single-n16.rank_03.nccl.log"
    )
    assert metrics.rank_file("ring", "x", rank=7, world_size=128, suffix=".nsys-rep") == "ring-x.rank_007.nsys-rep"


@pytest.mark.unit
def test_cross_case_name_pairs_the_cross_part_with_its_workload():
    assert metrics.case_name(
        "split-n16-contiguous", part="AB-cross", workload="agA-rsB", dtype="bf16", size_mib=1024
    ) == ("all_gather_A+reduce_scatter_B-1024MiB-bf16-split-n16-contiguous-AB-cross")
    for part, workload in (("AB-cross", "ag"), ("AB-concurrent", "agA-rsB"), (None, "agA-rsB")):
        with pytest.raises(ValueError, match="belongs to part"):
            metrics.case_name("split-n16-contiguous", part=part, workload=workload, dtype="bf16", size_mib=16)


@pytest.mark.unit
def test_trial_concurrency_is_the_shorter_share_of_the_longer_group():
    assert metrics.trial_concurrency([[2.0, 1.0], [1.0, 1.0]]) == [0.5, 1.0]
    with pytest.raises(ValueError, match="two groups"):
        metrics.trial_concurrency([[1.0]])
    with pytest.raises(ValueError):
        metrics.trial_concurrency([[1.0, 2.0], [1.0]])


@pytest.mark.unit
def test_case_names_reject_unknown_parts_scenarios_and_workloads():
    with pytest.raises(ValueError, match="unknown split part"):
        metrics.case_name("split-n16-contiguous", part="A", workload="ag", dtype="bf16", size_mib=16)
    with pytest.raises(ValueError, match="unknown workload"):
        metrics.case_name("single-n4", part=None, workload="both", dtype="bf16", size_mib=16)
    with pytest.raises(ValueError, match="unknown scenario"):
        metrics.scenario_name("split", total_ranks=16)


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
        "case": "all_gather-256MiB-bf16-single-n8",
        "iters": 20,
        "trial_makespan_ms": [1.0],
        "concurrency": None,
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
    algo, ranks, mib, ag, rs, ar, ag_pair, rs_pair, aggregate, speedup, overlap, *_kernels = rows[0]
    assert (algo, ranks, mib) == ("ring", "8", "256")
    assert (ag, rs, ar) == ("100.0", "50.0", "-")
    assert (ag_pair, rs_pair, aggregate) == ("100.0", "100.0", "66.6")
    assert speedup == "1.00"
    assert overlap == "0.00"


@pytest.mark.unit
def test_single_rows_leave_overlap_columns_empty_without_the_pair():
    rows = metrics.single_rows([_record()])
    assert rows[0][9:11] == ["-", "-"]


@pytest.mark.unit
def test_algo_rows_pair_ring_and_nvls_on_the_same_workload():
    records = [
        _record(algo="ring", aggregate_busbw_gb_s=100.0),
        _record(algo="nvls", aggregate_busbw_gb_s=250.0),
        _record(algo="nvls", size_mib=512, size_bytes=512 * 1024 * 1024, workload="rs"),
    ]
    rows = metrics.algo_rows(records)
    assert len(rows) == 2
    # columns: ranks, MiB, workload, ring, then <family>, <family>/ring for every other family
    assert rows[0] == ["8", "256", "ag", "100.0", "250.0", "2.50", "-", "-", "-", "-", "-", "-"]
    assert rows[1][-1] == "-"  # no ring record at this size/workload


@pytest.mark.unit
def test_best_of_ranks_every_family_and_names_the_winner():
    records = [
        _record(algo="ring", aggregate_busbw_gb_s=100.0),
        _record(algo="nvls", aggregate_busbw_gb_s=250.0),
        _record(algo="symm", aggregate_busbw_gb_s=310.0),
    ]
    rows = metrics._best_of(records)
    assert rows[0] == ["8", "256", "ag", "100.0", "250.0", "310.0", "-", "-", "symm"]


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
def test_kernel_rows_prefer_the_checked_implementation():
    record = _record(
        algo="symm", nccl_kernel="AllGather=AllGather_STMC", nccl_impl="AllGather=Symmetric/AllGather_STMC"
    )
    assert metrics.kernel_rows([record])[0][-1] == "AllGather=Symmetric/AllGather_STMC"


@pytest.mark.unit
def test_report_flags_profiled_runs():
    assert "--nsys" in metrics.render_report([_record(profiled=True)])
    assert "--nsys" not in metrics.render_report([_record(profiled=False)])


@pytest.mark.unit
def test_split_rows_compare_each_half_with_the_single_group_of_its_size():
    split = {"scenario": "split", "total_ranks": 16, "layout": "contiguous", "phase": "concurrent", "group_size": 8}
    records = [
        # single-n8 is the isolated reference of both halves of the 16-rank split.
        _record(workload="ag", makespan_ms=1.0, aggregate_busbw_gb_s=100.0),
        _record(**split, workload="ag", group_index=0, makespan_ms=1.1, aggregate_busbw_gb_s=95.0),
        _record(**split, workload="ag", group_index=1, makespan_ms=1.36, aggregate_busbw_gb_s=80.0),
        # A single group of another size is not a reference.
        _record(workload="ag", group_size=16, group_name="single16", makespan_ms=9.0),
    ]
    rows = metrics.split_rows(records)
    assert [row[:6] for row in rows] == [
        ["ring", "16", "contiguous", "256", "ag", "A"],
        ["ring", "16", "contiguous", "256", "ag", "B"],
    ]
    assert [row[6:9] for row in rows] == [["100.0", "95.0", "1.100"], ["100.0", "80.0", "1.360"]]
    # Without the single-n8 record the reference cells stay empty instead of being guessed.
    assert metrics.split_rows(records[1:3])[0][6:9] == ["-", "95.0", "-"]


@pytest.mark.unit
def test_render_report_contains_every_question_and_survives_empty_input():
    report = metrics.render_report([_record(), _record(algo="nvls", aggregate_busbw_gb_s=200.0)])
    assert "1. AG、RS、AR 单独与 AG+RS 同时发起" in report
    assert "1b. 各实现的绝对吞吐" in report
    assert "2. 各实现相对 ring" in report
    assert "3. 两个半组同时运行" in report
    assert "0. 每个测试的完成时间" in report
    assert "4. 两组同时：A 组 all-gather，B 组 reduce-scatter" in report
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


class _FakeNcclBackend:
    def __init__(self, calls):
        self.calls = calls

    def register_mem_pool(self, pool, symm=False):
        self.calls.append(("register", self, pool, symm))

    def deregister_mem_pool(self, pool):
        self.calls.append(("deregister", self, pool))


class _FakeProcessGroup:
    """Mirrors ``torch._C._distributed_c10d.ProcessGroup``: no window methods of its own."""

    def __init__(self, backend):
        self.backend = backend

    def _get_backend(self, device):
        assert device == "cuda"
        return self.backend


@pytest.mark.unit
def test_pool_registration_goes_through_the_nccl_backend(monkeypatch):
    monkeypatch.setattr(benchmark.torch, "device", lambda kind: kind, raising=False)
    monkeypatch.setattr(benchmark.torch.cuda, "synchronize", lambda: None, raising=False)
    calls = []
    ag, rs, ar = _FakeNcclBackend(calls), _FakeNcclBackend(calls), _FakeNcclBackend(calls)
    group = benchmark.CollectiveGroup(
        name="single4",
        ranks=[0, 1, 2, 3],
        ag_pg=_FakeProcessGroup(ag),
        rs_pg=_FakeProcessGroup(rs),
        ar_pg=_FakeProcessGroup(ar),
    )
    pool = object()
    benchmark._register_pool(group, pool)
    benchmark._deregister_pool(group, pool)
    assert calls == [
        ("register", ag, pool, True),
        ("register", rs, pool, True),
        ("register", ar, pool, True),
        ("deregister", ag, pool),
        ("deregister", rs, pool),
        ("deregister", ar, pool),
    ]


@pytest.mark.unit
def test_completion_rows_report_mean_median_and_mean_based_bandwidth():
    # 1 GiB on 8 ranks; trials 1, 2, 3 ms -> mean 2 ms, median given by the record.
    record = _record(
        size_bytes=1024**3,
        size_mib=1024,
        workload="ag+rs",
        makespan_ms=2.5,
        trial_makespan_ms=[1.0, 2.0, 3.0],
        mean_makespan_ms=2.0,
        iters=20,
    )
    (row,) = metrics.completion_rows([record])
    assert row[:8] == ["ring", "all_gather-256MiB-bf16-single-n8", "single8", "isolated", "8", "ag+rs", "1024", "3x20"]
    assert row[8:12] == ["2.000", "2.500", "1.000", "3.000"]
    # Two ops of 1 GiB in 2 ms: algbw = 2 * 1.073741824 GB / 0.002 s, busbw = algbw * 7/8.
    assert row[12] == f"{2 * 1.073741824 / 0.002:.1f}"
    assert row[13] == f"{2 * 1.073741824 / 0.002 * 7 / 8:.1f}"
    assert row[14] == "-"
    assert row[15] == "-"  # no nccl_impl evidence, so no kernel is claimed


@pytest.mark.unit
def test_cross_rows_compare_each_group_with_the_same_op_on_the_single_group():
    cross = {"scenario": "split", "total_ranks": 16, "layout": "contiguous", "phase": "cross", "group_size": 8}
    records = [
        # single-n8 ag and rs are the references of A's all-gather and B's reduce-scatter.
        _record(workload="ag", makespan_ms=1.0, aggregate_busbw_gb_s=400.0),
        _record(workload="rs", makespan_ms=2.0, aggregate_busbw_gb_s=200.0),
        _record(**cross, group_index=0, workload="ag", makespan_ms=1.2, aggregate_busbw_gb_s=333.3, concurrency=0.5),
        _record(**cross, group_index=1, workload="rs", makespan_ms=2.4, aggregate_busbw_gb_s=166.7, concurrency=0.5),
        # Another group size is not a reference.
        _record(workload="rs", group_size=16, group_name="single16", makespan_ms=9.0, aggregate_busbw_gb_s=1.0),
    ]
    assert metrics.cross_rows(records) == [
        [
            "ring",
            "16",
            "contiguous",
            "256",
            "400.0",
            "333.3",
            "1.200",
            "1.200",
            "200.0",
            "166.7",
            "2.400",
            "1.200",
            "500.0",
            "0.500",
        ]
    ]
    # Cross records never leak into the same-workload split table.
    assert metrics.split_rows(records) == []


@pytest.mark.unit
def test_split_parts_are_only_the_concurrent_and_cross_cases():
    assert metrics.SPLIT_PARTS == ("AB-concurrent", "AB-cross")
    with pytest.raises(ValueError, match="unknown split part"):
        metrics.case_name("split-n32-contiguous", part="A-isolated", workload="ag", dtype="bf16", size_mib=16)


@pytest.mark.unit
def test_split_needs_its_half_as_a_single_group(monkeypatch):
    monkeypatch.setenv("NCCL_ALGO", "NVLS")
    monkeypatch.setenv("NCCL_NVLS_ENABLE", "1")
    _debug_env(monkeypatch)
    ok = benchmark.parse_args(["--algo", "nvls", "--output-dir", "/tmp/x", "--split-totals", "32"])
    benchmark._check_environment("nvls", 32, ok)  # 16+16, and 16 is a default group size
    missing = benchmark.parse_args(
        ["--algo", "nvls", "--output-dir", "/tmp/x", "--group-sizes", "4,8", "--split-totals", "32"]
    )
    with pytest.raises(ValueError, match=re.escape("--split-totals [32] need their half ([16]) in --group-sizes")):
        benchmark._check_environment("nvls", 32, missing)


class _FakeEvent:
    def __init__(self, enable_timing=False):
        self.at = None

    def record(self, stream=None):
        _FakeEvent.calls.append(("record", self))
        self.at = len(_FakeEvent.calls)

    def elapsed_time(self, other):
        return float(other.at - self.at)

    def synchronize(self):
        pass


class _FakeStream:
    def wait_event(self, event):
        pass

    def synchronize(self):
        _FakeEvent.calls.append(("sync",))


@pytest.mark.unit
@pytest.mark.parametrize("member", [True, False])
def test_timed_trial_passes_the_gate_before_its_clock_starts(monkeypatch, member):
    _FakeEvent.calls = calls = []
    main_stream = _FakeStream()
    monkeypatch.setattr(benchmark.torch.cuda, "Event", _FakeEvent, raising=False)
    monkeypatch.setattr(benchmark.torch.cuda, "current_stream", lambda: main_stream, raising=False)
    monkeypatch.setattr(benchmark.torch.cuda, "_sleep", lambda cycles: calls.append(("sleep", cycles)), raising=False)
    monkeypatch.setattr(
        benchmark.dist, "all_reduce", lambda tensor, group: calls.append(("gate", group)), raising=False
    )
    monkeypatch.setattr(benchmark, "_launch", lambda *args: calls.append(("launch",)))

    @contextlib.contextmanager
    def nvtx_range(name):
        calls.append(("push", name))
        yield
        calls.append(("pop",))

    monkeypatch.setattr(benchmark.nvtx, "range", nvtx_range, raising=False)
    gate = benchmark.StartGate(ranks=[0, 1], pg="gate-pg", tensor="t")
    group = benchmark.CollectiveGroup(name="g", ranks=[0, 1], ag_pg=None, rs_pg=None, ar_pg=None) if member else None
    result = benchmark._timed_trial(
        group, None, ("ag",) if member else (), {"ag": _FakeStream()}, iters=2, delay_cycles=7, gate=gate, trial=3
    )
    kinds = [call[0] for call in calls]
    # The gate is an AllReduce: it must sit outside the trial range, or nsys would count it as "ar".
    assert kinds[:3] == ["sleep", "gate", "push"]
    assert calls[2] == ("push", "trial3")
    assert kinds[-1] == "pop"
    if member:
        # start is recorded right after the gate, then the launches, then the op/end events.
        assert kinds[3:-1] == ["record", "launch", "launch", "record", "record"]
        assert set(result) == {"makespan_ms", "enqueue_ms", "ag_ms"}
    else:
        # The idle half-group only passes the gate, so the gate always has every scenario rank.
        assert kinds[3:-1] == ["sync"]
        assert result is None


@pytest.mark.unit
@pytest.mark.parametrize("name", ["RING", "Ring"])  # 2.29 ncclAlgoToString / older ncclAlgoStr
def test_check_implementation_accepts_the_algorithm_name_in_any_case(name):
    # Reproduces the 32-GPU failure "AllGather: expected Ring/*, got RING/SIMPLE x105".
    line = f"h:1:2 [0] NCCL INFO AllGather: 4194304 Bytes -> Algo {name} proto SIMPLE channel{{Lo..Hi}}={{0..7}}"
    assert _check([line] * 105, algo="ring", launches=105, size_bytes=2 * 4194304) == (f"AllGather={name}/SIMPLE", [])


# All-reduce: workload, payload size, bus bandwidth, kernel names and the implementation check.
CLASSIC_AR = "h:1:2 [0] NCCL INFO AllReduce: 33554432 Bytes -> Algo NVLS proto SIMPLE channel{Lo..Hi}={0..15}"
SYMM_AR = (
    "h:1:2 [0] NCCL INFO AllReduce [Symmetric]: 33554432 Bytes -> Kernel AllReduce_RSxLDMC_AGxSTMC"
    " nchannels 8 nthreads 512 nWorks 1"
)


@pytest.mark.unit
def test_all_reduce_is_a_workload_of_its_own():
    assert metrics.workload_ops("ar") == ("ar",)
    assert metrics.case_name("single-n8", part=None, workload="ar", dtype="bf16", size_mib=64) == (
        "all_reduce-64MiB-bf16-single-n8"
    )
    args = benchmark.parse_args(["--algo", "nvls", "--output-dir", "/tmp/x", "--workloads", "ar"])
    assert args.workloads == ["ar"]


@pytest.mark.unit
def test_all_reduce_moves_the_whole_tensor_per_rank():
    # AG sendcount and RS recvcount are a shard; an all-reduce has no sharding.
    assert metrics.per_rank_bytes("ag", 1024, 8) == 128
    assert metrics.per_rank_bytes("rs", 1024, 8) == 128
    assert metrics.per_rank_bytes("ar", 1024, 8) == 1024
    with pytest.raises(ValueError, match="unknown op"):
        metrics.per_rank_bytes("ag+rs", 1024, 8)


@pytest.mark.unit
def test_all_reduce_bus_bandwidth_uses_the_nccl_tests_factor_two():
    # 1 GiB in 10 ms over 8 ranks: AG/RS busbw = algbw * 7/8, AR busbw = algbw * 2 * 7/8.
    one_way = 107.3741824 * 7 / 8
    kwargs = {"total_bytes": 1024**3, "group_size": 8, "seconds": 0.01}
    assert metrics.op_bus_bandwidth_gb_s(op="ag", **kwargs) == pytest.approx(one_way)
    assert metrics.op_bus_bandwidth_gb_s(op="rs", **kwargs) == pytest.approx(one_way)
    assert metrics.op_bus_bandwidth_gb_s(op="ar", **kwargs) == pytest.approx(2 * one_way)


@pytest.mark.unit
def test_symm_all_reduce_accepts_only_the_multicast_rs_ag_kernel():
    # Name as defined in NCCL 2.29.7 src/sym_kernels.cc kernelName[].
    assert benchmark.SYMM_KERNELS["ar"] == ["AllReduce_RSxLDMC_AGxSTMC"]
    assert benchmark.OP_FUNCS["ar"] == "AllReduce"


@pytest.mark.unit
@pytest.mark.parametrize("kernel", ["AllReduce_AGxLL_R", "AllReduce_AGxLLMC_R", "AllReduce_RSxLD_AGxST"])
def test_symm_all_reduce_fails_on_any_other_symmetric_kernel(kernel):
    # Any of these means NCCL lost the multicast kernel from its mask; the case must not pass.
    line = SYMM_AR.replace("AllReduce_RSxLDMC_AGxSTMC", kernel)
    assert _check([SYMM_AR, line], ops=("ar",))[1] == [
        f"AllReduce: expected Symmetric/AllReduce_RSxLDMC_AGxSTMC, got Symmetric/{kernel} x1"
    ]


@pytest.mark.unit
def test_symm_all_reduce_fails_on_a_classic_fallback():
    assert _check([SYMM_AR, CLASSIC_AR], ops=("ar",))[1] == [
        "AllReduce: expected Symmetric/AllReduce_RSxLDMC_AGxSTMC, got NVLS/SIMPLE x1"
    ]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("extra", "small"),
    [
        (["--group-sizes", "2,4", "--split-totals", "none"], "[2]"),
        # A 4-rank split has 2-rank halves; its reference group of 2 must be listed too.
        (["--group-sizes", "2,4", "--split-totals", "4"], "[2]"),
    ],
)
def test_symm_all_reduce_rejects_groups_without_multicast(monkeypatch, extra, small):
    # hasLsaMultimem needs an LSA team of more than 2 ranks, else RSxLDMC_AGxSTMC is masked out.
    _debug_env(monkeypatch, ce_threshold="9223372036854775807")
    monkeypatch.delenv("NCCL_ALGO", raising=False)
    monkeypatch.setenv("NCCL_WIN_ENABLE", "1")
    args = benchmark.parse_args(["--algo", "symm", "--output-dir", "/tmp/x", "--workloads", "ar", *extra])
    with pytest.raises(ValueError, match=re.escape(f"groups of {small}")):
        benchmark._check_environment("symm", 16, args)
    # Without the all-reduce workload the same groups are fine.
    args = benchmark.parse_args(["--algo", "symm", "--output-dir", "/tmp/x", "--workloads", "ag", *extra])
    benchmark._check_environment("symm", 16, args)


@pytest.mark.unit
def test_symm_kernel_pin_accepts_the_all_reduce_kernel(monkeypatch):
    args = benchmark.parse_args(
        ["--algo", "symm", "--output-dir", "/tmp/x", "--symm-kernel", "ar", "--workloads", "ar"]
    )
    _debug_env(monkeypatch, ce_threshold="9223372036854775807")
    monkeypatch.delenv("NCCL_ALGO", raising=False)
    monkeypatch.setenv("NCCL_WIN_ENABLE", "1")
    monkeypatch.setenv("NCCL_SYM_KERNEL", "AllReduce_RSxLDMC_AGxSTMC")
    benchmark._check_environment("symm", 32, args)
    monkeypatch.setenv("NCCL_SYM_KERNEL", "AllGather_STMC")
    with pytest.raises(RuntimeError, match="requires NCCL_SYM_KERNEL in"):
        benchmark._check_environment("symm", 32, args)


@pytest.mark.unit
def test_check_implementation_accepts_full_tensor_all_reduce_lines():
    # 32 MiB tensor on 2 ranks: the AllReduce line carries all 32 MiB, not a 16 MiB shard.
    assert _check([CLASSIC_AR, CLASSIC_AR], algo="nvls", ops=("ar",)) == ("AllReduce=NVLS/SIMPLE", [])
    assert _check([SYMM_AR, SYMM_AR], ops=("ar",)) == ("AllReduce=Symmetric/AllReduce_RSxLDMC_AGxSTMC", [])
    shard = CLASSIC_AR.replace("33554432 Bytes", "16777216 Bytes")
    assert _check([CLASSIC_AR, shard], algo="nvls", ops=("ar",))[1] == [
        "AllReduce: expected 33554432 B per rank, got 16777216 B x1"
    ]


@pytest.mark.unit
def test_check_implementation_keeps_the_gate_apart_from_a_measured_all_reduce():
    # Same function name; the 4-byte gate is counted as a gate, payload all-reduces as the op.
    assert _check([GATE, CLASSIC_AR, GATE, CLASSIC_AR], algo="ring", ops=("ar",), gate_launches=2)[1] == [
        "AllReduce: expected Ring/*, got NVLS/SIMPLE x2"
    ]
    assert _check([GATE, CLASSIC_AR, CLASSIC_AR], algo="nvls", ops=("ar",), gate_launches=1) == (
        "AllReduce=NVLS/SIMPLE",
        [],
    )
    # A symmetric all-reduce must not satisfy an AG/RS kernel family, nor the other way round.
    assert not benchmark.implementation_matches(
        "symm", "ag", "Symmetric/AllReduce_RSxLDMC_AGxSTMC", pinned_kernel=None
    )
    assert benchmark.implementation_matches("symm", "ar", "Symmetric/AllReduce_RSxLDMC_AGxSTMC", pinned_kernel=None)
    assert not benchmark.implementation_matches("symm", "ar", "Symmetric/AllReduce_AGxLL_R", pinned_kernel=None)


@pytest.mark.unit
def test_all_reduce_records_report_doubled_bus_bandwidth():
    ar = _record(
        workload="ar",
        size_bytes=1024**3,
        size_mib=1024,
        makespan_ms=2.0,
        trial_makespan_ms=[2.0],
        mean_makespan_ms=2.0,
        ar_busbw_gb_s=123.4,
        aggregate_busbw_gb_s=123.4,
    )
    single = metrics.single_rows([ar])
    assert single[0][5] == "123.4"  # AR busbw column
    (row,) = metrics.completion_rows([ar])
    algbw = 1.073741824 / 0.002
    assert row[12] == f"{algbw:.1f}"
    assert row[13] == f"{algbw * 2 * 7 / 8:.1f}"


@pytest.mark.unit
def test_report_names_all_reduce():
    report = metrics.render_report([_record(workload="ar", ar_busbw_gb_s=1.0)])
    assert "AG / RS / AR" in report
    assert "AR busbw" in report


# Kernel columns: names derived from the checked NCCL TUNING evidence, per NCCL 2.29.7 generators.
@pytest.mark.unit
@pytest.mark.parametrize(
    ("func", "implementation", "dtype", "expected"),
    [
        ("AllGather", "Symmetric/AllGather_STMC", "bf16", "ncclSymkDevKernel_AllGather_STMC"),
        ("ReduceScatter", "Symmetric/ReduceScatter_LDMC", "bf16", "ncclSymkDevKernel_ReduceScatter_LDMC_sum_bf16"),
        (
            "AllReduce",
            "Symmetric/AllReduce_RSxLDMC_AGxSTMC",
            "fp32",
            "ncclSymkDevKernel_AllReduce_RSxLDMC_AGxSTMC_sum_f32",
        ),
        ("AllGather", "NVLS/SIMPLE", "bf16", "ncclDevFunc_AllGather_NVLS_SIMPLE"),
        ("ReduceScatter", "RING/LL128", "bf16", "ncclDevFunc_ReduceScatter_Sum_bf16_RING_LL128"),
        ("AllReduce", "Ring/Simple", "fp32", "ncclDevFunc_AllReduce_Sum_f32_RING_SIMPLE"),
        ("AllGather", "CopyEngine", "bf16", "CopyEngine（cudaMemcpy，无 kernel）"),
        ("AllGather", "CopyEngine+NVLSSync", "bf16", "CopyEngine+NVLSSync（cudaMemcpy，无 kernel）"),
        ("AllGather", "Symmetric/AllGather_TmaSTMC", "bf16", "ncclSymkDevKernel_AllGather_TmaSTMC"),
    ],
)
def test_kernel_name_follows_the_nccl_generators(func, implementation, dtype, expected):
    assert metrics.kernel_name(func, implementation, dtype) == expected


@pytest.mark.unit
def test_record_kernels_reads_every_op_and_every_implementation():
    record = _record(
        dtype="bf16",
        nccl_impl="AllGather=Symmetric/AllGather_ST|Symmetric/AllGather_STMC;ReduceScatter=Symmetric/ReduceScatter_LDMC",
    )
    assert metrics.record_kernels(record) == {
        "ag": "ncclSymkDevKernel_AllGather_ST \\| ncclSymkDevKernel_AllGather_STMC",
        "rs": "ncclSymkDevKernel_ReduceScatter_LDMC_sum_bf16",
    }
    assert metrics.record_kernels(_record()) == {}


@pytest.mark.unit
def test_completion_rows_name_the_kernel_of_each_case():
    pair = _record(workload="ag+rs", nccl_impl="AllGather=NVLS/SIMPLE;ReduceScatter=NVLS/SIMPLE")
    (row,) = metrics.completion_rows([pair])
    assert row[-1] == "AG ncclDevFunc_AllGather_NVLS_SIMPLE; RS ncclDevFunc_ReduceScatter_Sum_bf16_NVLS_SIMPLE"


@pytest.mark.unit
def test_single_rows_show_isolated_kernels_and_flag_a_different_one_in_the_pair():
    records = [
        _record(algo="symm", workload="ag", nccl_impl="AllGather=Symmetric/AllGather_STMC"),
        _record(algo="symm", workload="rs", nccl_impl="ReduceScatter=Symmetric/ReduceScatter_LDMC"),
        _record(
            algo="symm", workload="ar", ar_busbw_gb_s=1.0, nccl_impl="AllReduce=Symmetric/AllReduce_RSxLDMC_AGxSTMC"
        ),
        _record(
            algo="symm",
            workload="ag+rs",
            nccl_impl="AllGather=Symmetric/AllGather_ST;ReduceScatter=Symmetric/ReduceScatter_LDMC",
        ),
    ]
    (row,) = metrics.single_rows(records)
    assert row[-3:] == [
        "ncclSymkDevKernel_AllGather_STMC（pair：ncclSymkDevKernel_AllGather_ST）",
        "ncclSymkDevKernel_ReduceScatter_LDMC_sum_bf16",
        "ncclSymkDevKernel_AllReduce_RSxLDMC_AGxSTMC_sum_bf16",
    ]


# symm-tma and copy-engine: a TMA multicast or copy-engine multicast AllGather, paired with
# the multicast symmetric ReduceScatter_LDMC; they measure only ag and ag+rs.
TMA_AG = SYMM_AG.replace("Kernel AllGather_STMC", "Kernel AllGather_TmaSTMC")
PINNED_ENV = {
    "symm-tma": {
        "NCCL_SYM_KERNEL": "allgather:AllGather_TmaSTMC;reducescatter:ReduceScatter_LDMC",
        "NCCL_SYM_CE_THRESHOLD": "9223372036854775807",
    },
    "copy-engine": {
        "NCCL_SYM_KERNEL": "reducescatter:ReduceScatter_LDMC",
        "NCCL_SYM_CE_THRESHOLD": "0",
        "NCCL_CE_COLL_AG_MULTICAST_THRESHOLD": "4611686018427387904",
    },
}


def _pinned_env(monkeypatch, algo):
    """The environment run_nvls_collective_benchmark.sh sets for a symm-tma / copy-engine pass."""
    _debug_env(monkeypatch)
    monkeypatch.delenv("NCCL_ALGO", raising=False)
    monkeypatch.setenv("NCCL_WIN_ENABLE", "1")
    monkeypatch.setenv("NCCL_NVLS_ENABLE", "1")
    for name, value in PINNED_ENV[algo].items():
        monkeypatch.setenv(name, value)


@pytest.mark.unit
def test_pinned_families_measure_only_ag_and_the_pair():
    requested = ["ag", "rs", "ar", "ag+rs"]
    assert metrics.measured_workloads("symm", requested) == requested
    assert metrics.measured_workloads("symm-tma", requested) == ["ag", "ag+rs"]
    assert metrics.measured_workloads("copy-engine", ["ag+rs", "rs", "ag"]) == ["ag+rs", "ag"]


@pytest.mark.unit
@pytest.mark.parametrize("algo", ["symm-tma", "copy-engine"])
def test_pinned_environment_accepts_what_the_launcher_sets(monkeypatch, algo):
    _pinned_env(monkeypatch, algo)
    benchmark._check_environment(algo, 32, benchmark.parse_args(["--algo", algo, "--output-dir", "/tmp/x"]))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("algo", "name", "value", "message"),
    [
        ("symm-tma", "NCCL_SYM_KERNEL", "AllGather_TmaSTMC", "requires NCCL_SYM_KERNEL="),
        ("symm-tma", "NCCL_SYM_CE_THRESHOLD", "8388608", "large AllGathers go to the copy engines"),
        ("symm-tma", "NCCL_ALGO", "NVLS", "requires NCCL_ALGO to be unset"),
        ("symm-tma", "NCCL_NVLS_ENABLE", "0", "NCCL_NVLS_ENABLE=1"),
        ("copy-engine", "NCCL_SYM_CE_THRESHOLD", "9223372036854775807", "NCCL_SYM_CE_THRESHOLD=0"),
        # A shard above the multicast threshold would silently take the N-1 unicast copies.
        ("copy-engine", "NCCL_CE_COLL_AG_MULTICAST_THRESHOLD", "1048576", "multicast address"),
        ("copy-engine", "NCCL_SYM_KERNEL", "", "requires NCCL_SYM_KERNEL="),
    ],
)
def test_pinned_environment_rejects_anything_that_changes_the_implementation(monkeypatch, algo, name, value, message):
    _pinned_env(monkeypatch, algo)
    monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match=re.escape(message)):
        benchmark._check_environment(algo, 32, benchmark.parse_args(["--algo", algo, "--output-dir", "/tmp/x"]))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--workloads", "rs,ar"], "measures only ag and ag+rs"),
        (["--group-sizes", "2,4", "--split-totals", "none"], "groups of >= 3 ranks; got [2]"),
        (["--workloads", "ag", "--symm-kernel", "ag"], "--symm-kernel applies to --algo symm only"),
    ],
)
def test_pinned_environment_rejects_workloads_and_groups_it_cannot_measure(monkeypatch, extra, message):
    _pinned_env(monkeypatch, "copy-engine")
    args = benchmark.parse_args(["--algo", "copy-engine", "--output-dir", "/tmp/x", *extra])
    with pytest.raises(ValueError, match=re.escape(message)):
        benchmark._check_environment("copy-engine", 32, args)


@pytest.mark.unit
def test_check_implementation_holds_symmetric_tma_to_tma_ag_and_ldmc_rs():
    assert _check([TMA_AG, SYMM_RS, TMA_AG, SYMM_RS], algo="symm-tma", ops=("ag", "rs")) == (
        "AllGather=Symmetric/AllGather_TmaSTMC;ReduceScatter=Symmetric/ReduceScatter_LDMC",
        [],
    )
    rs_ld = SYMM_RS.replace("ReduceScatter_LDMC", "ReduceScatter_LD")
    _, errors = _check([TMA_AG, SYMM_AG, rs_ld, SYMM_RS], algo="symm-tma", ops=("ag", "rs"))
    assert errors == [
        "AllGather: expected Symmetric/AllGather_TmaSTMC, got Symmetric/AllGather_STMC x1",
        "ReduceScatter: expected Symmetric/ReduceScatter_LDMC, got Symmetric/ReduceScatter_LD x1",
    ]


@pytest.mark.unit
def test_check_implementation_holds_copy_engine_to_the_multicast_synchronized_ce():
    assert _check([CE_AG, CE_AG], algo="copy-engine") == ("AllGather=CopyEngine+NVLSSync", [])
    # Without "CE synchronization with NVLS" the communicator had no multimem, so no multicast copy.
    unicast_sync = CE_AG.removesuffix("; CE synchronization with NVLS")
    _, errors = _check([CE_AG, unicast_sync, SYMM_AG], algo="copy-engine", launches=3)
    assert errors == ["AllGather: expected CopyEngine+NVLSSync, got CopyEngine x1, Symmetric/AllGather_STMC x1"]


@pytest.mark.unit
def test_symm_variant_rows_compare_tma_and_copy_engine_with_symm():
    records = [
        _record(algo="symm", aggregate_busbw_gb_s=200.0),
        _record(algo="symm-tma", aggregate_busbw_gb_s=300.0),
        _record(algo="copy-engine", aggregate_busbw_gb_s=100.0),
        # symm-only shapes (rs) are not part of this table.
        _record(algo="symm", workload="rs"),
    ]
    assert metrics.symm_variant_rows(records) == [["8", "256", "ag", "200.0", "300.0", "1.50", "100.0", "0.50"]]
    report = metrics.render_report(records)
    assert "### 2a." in report
    assert "| ranks | MiB | workload | symm | symm-tma | symm-tma/symm | copy-engine | copy-engine/symm |" in report


@pytest.mark.unit
def test_connect_runs_pinned_families_on_windows(monkeypatch):
    calls = []
    monkeypatch.setattr(benchmark.torch.cuda, "synchronize", lambda: None, raising=False)
    monkeypatch.setattr(benchmark.torch, "ones", lambda *a, **k: "ones", raising=False)
    monkeypatch.setattr(benchmark.dist, "barrier", lambda: None, raising=False)
    monkeypatch.setattr(benchmark.dist, "all_reduce", lambda t, group: calls.append(("ar", group)), raising=False)
    monkeypatch.setattr(benchmark, "_connect_windows", lambda group, rank: calls.append(("windows", group.name)))
    gate = benchmark.StartGate(ranks=[0, 1], pg=None, tensor=None)
    groups = [
        benchmark.CollectiveGroup(name="A", ranks=[0], ag_pg="ag", rs_pg="rs", ar_pg="ar-A"),
        benchmark.CollectiveGroup(name="B", ranks=[1], ag_pg="ag", rs_pg="rs", ar_pg="ar-B"),
    ]
    benchmark._connect(groups, gate, 0, windows=True)
    # Only this rank's group connects, AG/RS on a registered window and the AR as before.
    assert calls == [("windows", "A"), ("ar", "ar-A")]


@pytest.mark.unit
def test_late_join_replaces_split_and_keeps_both_ops():
    base = ["--algo", "symm", "--output-dir", "/tmp/never-used"]
    assert benchmark.parse_args([*base, "--late-join"]).late_join is True
    with pytest.raises(SystemExit):
        benchmark.parse_args([*base, "--late-join", "--split-totals", "16"])
    with pytest.raises(SystemExit):
        benchmark.parse_args([*base, "--late-join", "--symm-kernel", "ag"])


@pytest.mark.unit
def test_late_join_workload_launches_rs_first_and_names_its_case():
    assert metrics.workload_ops(metrics.LATE_JOIN_WORKLOAD) == ("rs", "ag")
    scenario = metrics.scenario_name("late-join", total_ranks=8)
    assert scenario == "late-join-n8"
    assert metrics.case_name(scenario, part=None, workload=metrics.LATE_JOIN_WORKLOAD, dtype="bf16", size_mib=256) == (
        "reduce_scatter_x4+late_all_gather-256MiB-bf16-late-join-n8"
    )


@pytest.mark.unit
def test_check_implementation_holds_each_op_to_its_own_size():
    # Late join: a 32 MiB AG (16 MiB shard on 2 ranks) next to a 4x RS (64 MiB shard).
    rs_x4 = SYMM_RS.replace("16777216 Bytes", "67108864 Bytes")
    sizes = {"rs": 4 * 2 * 16777216}
    lines = [rs_x4, SYMM_AG, rs_x4, SYMM_AG]
    assert (
        benchmark.check_implementation(
            lines,
            algo="symm",
            ops=("rs", "ag"),
            launches=2,
            size_bytes=2 * 16777216,
            group_size=2,
            pinned_kernel=None,
            op_sizes=sizes,
        )[1]
        == []
    )
    _, errors = benchmark.check_implementation(
        [SYMM_RS, SYMM_AG, SYMM_RS, SYMM_AG],
        algo="symm",
        ops=("rs", "ag"),
        launches=2,
        size_bytes=2 * 16777216,
        group_size=2,
        pinned_kernel=None,
        op_sizes=sizes,
    )
    assert errors == ["ReduceScatter: expected 67108864 B per rank, got 16777216 B x2"]


def _late_join(**times):
    # AG of 1 GB in 1 s alone, RS of 4 GB in 4 s alone; the AG is released 1 s into the RS.
    return metrics.late_join_metrics(ag_bytes=10**9, rs_bytes=4 * 10**9, ag_iso_s=1.0, rs_iso_s=4.0, **times)


@pytest.mark.unit
def test_late_join_metrics_separate_full_speed_from_time_sharing():
    # Both at full speed: the AG takes its isolated 1 s and the RS still ends at 4 s.
    full = _late_join(ag_begin_s=1.0, ag_end_s=2.0, rs_end_s=4.0)
    assert full["join_fraction"] == pytest.approx(0.25)
    assert full["ag_slowdown"] == pytest.approx(1.0)
    assert full["rs_slowdown"] == pytest.approx(1.0)
    assert full["overlap_capacity"] == pytest.approx(2.0)
    assert full["ag_share"] == pytest.approx(0.5)
    assert full["ag_algbw_overlap_gb_s"] == pytest.approx(1.0)
    assert full["ag_inside_rs"] is True
    # Time-shared halves: the AG takes 2 s, the RS is pushed out by 1 s.
    shared = _late_join(ag_begin_s=1.0, ag_end_s=3.0, rs_end_s=5.0)
    assert shared["ag_slowdown"] == pytest.approx(2.0)
    assert shared["rs_rel_speed"] == pytest.approx(0.5)
    assert shared["overlap_capacity"] == pytest.approx(1.0)
    assert shared["ag_share"] == pytest.approx(0.5)


@pytest.mark.unit
def test_late_join_metrics_without_overlap_leave_the_window_empty():
    late = _late_join(ag_begin_s=4.5, ag_end_s=5.5, rs_end_s=4.0)
    assert late["overlap_ms"] == 0.0
    assert late["overlap_capacity"] is None and late["ag_share"] is None
    # The part of an AG that outlives the RS runs at its isolated rate, outside the window.
    tail = _late_join(ag_begin_s=3.5, ag_end_s=4.5, rs_end_s=4.0)
    assert tail["ag_inside_rs"] is False
    assert tail["ag_rel_speed"] == pytest.approx(1.0)


@pytest.mark.unit
def test_report_adds_the_late_join_table_and_per_op_sizes():
    size = 256 * 1024 * 1024
    join = _record(
        scenario="late-join",
        phase="join",
        workload=metrics.LATE_JOIN_WORKLOAD,
        size_bytes=size,
        rs_size_bytes=4 * size,
        makespan_ms=2.0,
        trial_makespan_ms=[2.0],
        ag_begin_ms=0.5,
        ag_ms=1.0,
        rs_ms=2.0,
        ag_iso_ms=0.5,
        rs_iso_ms=2.0,
        **_late_join(ag_begin_s=1.0, ag_end_s=2.0, rs_end_s=4.0),
    )
    (row,) = metrics.late_join_rows([join, _record()])
    assert row[:4] == ["ring", "8", "256", "1024"]
    assert row[15:18] == ["2.00", "0.50", "yes"]
    report = metrics.render_report([join])
    assert "## 5. late join" in report
    # Table 0 counts the RS at its own 4x size: (S + 4S) / 2 ms.
    (completion,) = metrics.completion_rows([join])
    assert completion[12] == f"{5 * size / 2e-3 / 1e9:.1f}"


@pytest.mark.unit
@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_launcher_late_join_defaults_and_split_default_off():
    result = _launcher_dry_run("--late-join")
    assert result.returncode == 0, result.stderr
    for option in ("--late-join", "--sizes-mib 64,256,512", "--iters 5", "--trials 10", "--split-totals none"):
        assert option in result.stdout.replace("\\", "")
    plain = _launcher_dry_run()
    assert "--split-totals none" in plain.stdout and "--late-join" not in plain.stdout
    assert _launcher_dry_run("--late-join", "--split-totals", "16").returncode != 0


# --cta-configs: per-communicator CTA counts (ncclConfig_t minCTAs/maxCTAs).
SYMM_AG_6 = SYMM_AG.replace("nchannels 8", "nchannels 6")
SYMM_RS_3 = SYMM_RS.replace("nchannels 8", "nchannels 3")


@pytest.mark.unit
def test_cta_config_entries_parse_and_name_in_a_fixed_key_order():
    assert metrics.parse_cta_config("default") == {}
    assert metrics.parse_cta_config(" p2p=32, ag=3 ") == {"p2p": 32, "ag": 3}
    assert metrics.cta_config_name({}) == "default"
    assert metrics.cta_config_name({"p2p": 32, "rs": 6, "ag": 3}) == "cta-ag3-rs6-p2p32"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("ag2=3", "must be <op>=<ctas>"),
        ("ag", "must be <op>=<ctas>"),
        ("ag=0", "integer in 1..64"),
        ("ag=65", "integer in 1..64"),
        ("ag=x", "integer in 1..64"),
        ("ag=3,ag=4", "sets ag twice"),
        ("p2p=12", "power-of-two"),
    ],
)
def test_cta_config_entries_reject_what_nccl_cannot_honor(text, message):
    with pytest.raises(ValueError, match=message):
        metrics.parse_cta_config(text)


@pytest.mark.unit
def test_cta_configs_option_keeps_order_drops_duplicates_and_defaults_to_nccl_choice():
    base = ["--algo", "symm", "--output-dir", "/tmp/x"]
    assert benchmark.parse_args(base).cta_configs == [{}]
    args = benchmark.parse_args([*base, "--cta-configs", "default;ag=3,rs=3;p2p=32;rs=3,ag=3;"])
    assert args.cta_configs == [{}, {"ag": 3, "rs": 3}, {"p2p": 32}]
    with pytest.raises(SystemExit):
        benchmark.parse_args([*base, "--cta-configs", "ag=3;rs=100"])


@pytest.mark.unit
def test_cta_config_is_appended_to_the_case_name_only_when_not_default():
    kwargs = {"part": None, "workload": "ag+write", "dtype": "bf16", "size_mib": 1024}
    assert metrics.case_name("single-n4", **kwargs) == "all_gather+write-1024MiB-bf16-single-n4"
    assert metrics.case_name("single-n4", cta_config="default", **kwargs) == "all_gather+write-1024MiB-bf16-single-n4"
    assert metrics.case_name("single-n4", cta_config="cta-ag3-p2p32", **kwargs) == (
        "all_gather+write-1024MiB-bf16-single-n4-cta-ag3-p2p32"
    )
    with pytest.raises(ValueError, match="unknown CTA config name"):
        metrics.case_name("single-n4", cta_config="ag3", **kwargs)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("target", "comm", "ctas_256mib"),
    [
        # 4 GPUs, 1024 MiB: every send is one 256 MiB AG shard on torch's 2-rank pair communicator.
        (None, (None, None), 16),
        (4, (None, 4), 4),
        (8, (None, 8), 8),
        (16, (None, 16), 16),
        # The per-peer limit (32) is only reached with 64 P2P channels on the communicator.
        (32, (64, None), 32),
    ],
)
def test_p2p_cta_targets_map_to_a_communicator_config_that_reaches_them(target, comm, ctas_256mib):
    assert (metrics.p2p_comm_ctas(target) if target else (None, None)) == comm
    min_ctas, max_ctas = comm
    got = metrics.predicted_p2p_ctas(256 << 20, n_ranks=2, multi_node=False, min_ctas=min_ctas, max_ctas=max_ctas)
    assert got == ctas_256mib


@pytest.mark.unit
def test_predicted_p2p_ctas_follow_nccl_size_and_topology_rules():
    def predict(nbytes, **kw):
        return metrics.predicted_p2p_ctas(
            nbytes, **{"n_ranks": 2, "multi_node": False, "min_ctas": None, "max_ctas": None, **kw}
        )

    # Within a tray a channel takes up to 32 x 512 KiB = 16 MiB before NCCL doubles the count.
    assert predict(256 << 20) == 16
    assert predict((256 << 20) + 1) == 32
    assert predict(64 << 10) == 1  # one 64 KiB part
    # Across trays a channel takes one 256 KiB chunk, so even 8 MiB fills all 32 channels.
    assert predict(8 << 20, multi_node=True) == 32
    # An eager 32-rank communicator across trays: 32 / ceil(32/8) = 8 channels per peer.
    assert predict(8 << 20, n_ranks=32, multi_node=True) == 8
    with pytest.raises(ValueError):
        predict(0)


@pytest.mark.unit
def test_comm_ctas_pins_collectives_and_shapes_the_p2p_communicator():
    ctas = {"ag": 3, "rs": 6, "p2p": 32}
    assert benchmark.comm_ctas("ag", ctas) == (3, 3)
    assert benchmark.comm_ctas("ag2", ctas) == (3, 3)  # the second AG communicator follows ag
    assert benchmark.comm_ctas("rs2", ctas) == (6, 6)
    assert benchmark.comm_ctas("ar", ctas) == (None, None)  # not requested: NCCL's choice
    assert benchmark.comm_ctas("read", ctas) == (64, None)
    assert benchmark.comm_ctas("write", {"p2p": 8}) == (None, 8)


class _FakeNcclOptions:
    def __init__(self):
        self.config = types.SimpleNamespace(min_ctas=None, max_ctas=None)


@pytest.mark.unit
def test_new_group_passes_each_communicator_its_own_cta_config(monkeypatch):
    calls = []
    monkeypatch.setattr(
        benchmark.dist, "ProcessGroupNCCL", types.SimpleNamespace(Options=_FakeNcclOptions), raising=False
    )
    monkeypatch.setattr(benchmark.dist, "get_rank", lambda: 0, raising=False)

    def new_group(**kwargs):
        calls.append(kwargs["pg_options"])
        return f"pg{len(calls)}"

    monkeypatch.setattr(benchmark.dist, "new_group", new_group, raising=False)
    group = benchmark._new_group("single4", [0, 1, 2, 3], extended=True, ctas={"ag": 3, "p2p": 32})
    configs = [None if o is None else (o.config.min_ctas, o.config.max_ctas) for o in calls]
    # ag, rs, ar, ag2, rs2, p2p
    assert configs == [(3, 3), None, None, (3, 3), None, (64, None)]
    assert group.ctas == {"ag": 3, "p2p": 32}
    calls.clear()
    benchmark._new_group("single4", [0, 1, 2, 3])
    assert calls == [None, None, None]  # default: torch's own options, as before


@pytest.mark.unit
def test_launch_ctas_reads_symmetric_and_classic_tuning_lines():
    assert benchmark.launch_ctas(SYMM_AG_6) == 6
    assert benchmark.launch_ctas(CLASSIC_AG) == 16  # channel{Lo..Hi}={0..15}
    assert benchmark.launch_ctas(CE_AG) is None


@pytest.mark.unit
def test_check_ctas_reports_every_op_and_holds_requested_ones_to_their_count():
    lines = [GATE, SYMM_AG_6, SYMM_RS_3, SYMM_AG_6, SYMM_RS_3]
    assert benchmark.check_ctas(lines, algo="symm", ops=("ag", "rs"), ctas={}) == ({"ag": "6", "rs": "3"}, [])
    assert benchmark.check_ctas(lines, algo="symm", ops=("ag", "rs"), ctas={"rs": 3}) == ({"ag": "6", "rs": "3"}, [])
    observed, errors = benchmark.check_ctas(lines + [SYMM_AG], algo="symm", ops=("ag", "write"), ctas={"ag": 6})
    assert observed == {"ag": "6|8"}
    assert errors == ["AllGather: requested 6 CTAs, NCCL launched 8 x1"]


@pytest.mark.unit
def test_check_ctas_fails_when_the_cap_moves_symm_off_its_multicast_kernel():
    unicast = SYMM_AG_6.replace("AllGather_STMC", "AllGather_ST")
    _, errors = benchmark.check_ctas([SYMM_AG_6, unicast], algo="symm", ops=("ag",), ctas={"ag": 6})
    assert errors == ["AllGather: 6 CTAs moved NCCL off Symmetric/AllGather_STMC: Symmetric/AllGather_ST x1"]
    # Without a request for that op the kernel choice is check_implementation's business only.
    assert benchmark.check_ctas([unicast], algo="symm", ops=("ag",), ctas={"rs": 3})[1] == []


def _symm_env(monkeypatch):
    _debug_env(monkeypatch, ce_threshold="9223372036854775807")
    monkeypatch.delenv("NCCL_ALGO", raising=False)
    monkeypatch.setenv("NCCL_WIN_ENABLE", "1")
    for name in (*benchmark.CTA_ENV_OVERRIDES, benchmark.SYM_CTAS_ENV):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.unit
def test_cta_configs_refuse_environment_that_overrides_them(monkeypatch):
    _symm_env(monkeypatch)
    args = benchmark.parse_args(
        ["--algo", "symm", "--output-dir", "/tmp/x", "--group-sizes", "4", "--cta-configs", "default;p2p=32"]
    )
    benchmark._check_environment("symm", 4, args)
    monkeypatch.setenv("NCCL_MAX_CTAS", "16")
    with pytest.raises(RuntimeError, match=r"unset \['NCCL_MAX_CTAS'\]"):
        benchmark._check_environment("symm", 4, args)
    # The default config alone leaves the environment as the user set it.
    benchmark._check_environment(
        "symm", 4, benchmark.parse_args(["--algo", "symm", "--output-dir", "/tmp/x", "--group-sizes", "4"])
    )


@pytest.mark.unit
def test_cta_configs_pass_the_multicast_cap_but_not_nccl_sym_ctas(monkeypatch):
    _symm_env(monkeypatch)
    base = ["--algo", "symm", "--output-dir", "/tmp/x", "--group-sizes", "4"]
    benchmark._check_environment("symm", 4, benchmark.parse_args([*base, "--cta-configs", "ag=8,rs=3"]))
    # Above divUp(32, 4) = 8: nccl-2.32.3-cta runs it, stock NCCL clamps it and check_ctas fails the case.
    benchmark._check_environment("symm", 4, benchmark.parse_args([*base, "--cta-configs", "ag=12,rs=64"]))
    benchmark._check_environment("symm", 8, benchmark.parse_args([*base[:-1], "8", "--cta-configs", "ag=1,rs=5"]))
    monkeypatch.setenv("NCCL_SYM_CTAS", "12")
    with pytest.raises(RuntimeError, match="NCCL_SYM_CTAS fixes"):
        benchmark._check_environment("symm", 4, benchmark.parse_args([*base, "--cta-configs", "ag=12"]))
    # NCCL_SYM_CTAS together with a P2P-only config is fine: it leaves the P2P communicator alone.
    benchmark._check_environment("symm", 4, benchmark.parse_args([*base, "--cta-configs", "p2p=32"]))


@pytest.mark.unit
def test_cta_table_pairs_every_config_with_the_default_case():
    default = _record(
        algo="symm",
        workload="ag+write",
        group_size=4,
        ag_ms=2.0,
        write_ms=2.0,
        makespan_ms=2.0,
        observed_ctas={"ag": "6"},
        predicted_p2p_ctas=16,
    )
    doubled = _record(
        algo="symm",
        workload="ag+write",
        group_size=4,
        ag_ms=2.4,
        write_ms=1.5,
        makespan_ms=2.4,
        observed_ctas={"ag": "6"},
        predicted_p2p_ctas=32,
        cta_config="cta-p2p32",
        case="all_gather+write-256MiB-bf16-single-n4-cta-p2p32",
    )
    rows = metrics.cta_rows([default, doubled])
    assert [row[6] for row in rows] == ["cta-p2p32", "default"]
    ops = dict(zip(metrics.CTA_TABLE_OPS, rows[0][7:14]))
    assert ops["ag"] == "6 CTA / 2.400 ms (1.20x)"
    assert ops["write"] == "~32 CTA / 1.500 ms (0.75x)"
    assert ops["rs"] == "-"
    assert rows[0][14:16] == ["2.400", "1.20"]
    assert rows[1][15] == "-"  # the default row is its own reference
    report = metrics.render_report([default, doubled])
    assert "## 6. 不同 CTA 配置" in report
    # Tables 1-5 keep to the default config: one single-group row, not one per config.
    assert len(metrics.single_rows([default])) == 1
    assert "cta-p2p32" in report.split("## 1.")[0]  # table 0 lists every case
    assert metrics.cta_rows([default]) == []
    assert "## 6." not in metrics.render_report([default])


@pytest.mark.unit
@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_launcher_passes_cta_configs_and_unsets_overriding_environment(monkeypatch):
    result = _launcher_dry_run("--cta-configs", "default;ag=3;p2p=32")
    assert result.returncode == 0, result.stderr
    assert "--cta-configs default\\;ag=3\\;p2p=32" in result.stdout
    assert " -u " not in result.stdout
    assert "--cta-configs default " in _launcher_dry_run().stdout
    # GB200 images export NCCL_MIN_CTAS=4 and NCCL_MIN_NCHANNELS=16: unset for the workers, with a warning.
    monkeypatch.setenv("NCCL_MIN_CTAS", "4")
    monkeypatch.setenv("NCCL_MIN_NCHANNELS", "16")
    unset = _launcher_dry_run("--cta-configs", "default;p2p=32")
    assert unset.returncode == 0, unset.stderr
    assert "env -u NCCL_MIN_CTAS -u NCCL_MIN_NCHANNELS " in unset.stdout
    assert "unsetting NCCL_MIN_CTAS=4" in unset.stderr
    # Without a CTA config the environment is passed through untouched.
    assert " -u " not in _launcher_dry_run().stdout
