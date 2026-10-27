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

"""CPU-only checks for the AllGather_STMC multimem.st timing test (nccl-2.32.3-pause-with-debug).

The real run needs GB200 trays and the debug build. These tests cover the parsing of the device printf
lines, the environment checks of the worker, the launcher's dry run with a fake libnccl.so.2 that only
carries the strings it greps for, and the patch's countdown (no division in the store loop).
"""

import dataclasses
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = REPO / "examples" / "gb200" / "nvls-collective"
LAUNCHER = BENCHMARK_DIR / "run_ag_st_debug.sh"
PATCH = REPO / "install" / "patches" / "nccl-2.32.3-pause-with-debug.patch"
DEBUG_ENVS = (
    "NCCL_SYM_AG_STMC_PAUSE_CYCLES",
    "NCCL_SYM_AG_STMC_ST_PRINT_INTERVAL",
    "NCCL_SYM_AG_STMC_ST_PRINT_SKIP",
    "NCCL_SYM_AG_STMC_ST_PRINT_LAUNCHES",
)
needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


@pytest.fixture(scope="module")
def worker() -> types.ModuleType:
    """Import ag_st_debug_benchmark.py, stubbing torch if it is not installed."""
    stubs: dict[str, types.ModuleType] = {}
    if importlib.util.find_spec("torch") is None:
        torch_stub = types.ModuleType("torch")
        torch_stub.__path__ = []  # type: ignore[attr-defined]
        torch_stub.distributed = types.ModuleType("torch.distributed")  # type: ignore[attr-defined]
        stubs = {"torch": torch_stub, "torch.distributed": torch_stub.distributed}
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location(
            "ag_st_debug_benchmark", BENCHMARK_DIR / "ag_st_debug_benchmark.py"
        )
        module = importlib.util.module_from_spec(spec)
        # dataclasses resolves the module's string annotations through sys.modules.
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module


def _line(*, cta: int, warp: int, iter: int, n: int, launch: int = 2, rank: int = 0, cycles: int = 100) -> str:
    return (
        f"[nccl-ag-stmc-st] launch={launch} rank={rank} cta={cta} warp={warp} iter={iter} n={n}"
        f" sum={cycles * n} min={cycles - 10} max={cycles + 10} last={cycles} pause=0 globaltimer={1000 + iter}"
    )


def _warp_lines(*, cta: int, warp: int, chunks: int, interval: int, **kwargs) -> list[str]:
    """The lines one warp prints: one per ``interval`` chunks, then one for the rest."""
    lines = [_line(cta=cta, warp=warp, iter=i, n=interval, **kwargs) for i in range(interval, chunks + 1, interval)]
    if chunks % interval:
        lines.append(_line(cta=cta, warp=warp, iter=chunks, n=chunks % interval, **kwargs))
    return lines


@pytest.mark.unit
def test_lines_per_rank_follow_the_chunks_not_the_ctas(worker):
    gib = 1024 * 1024 * 1024
    # 1 GiB: every rank stores its 1/n shard as 4 KiB chunks.
    assert [worker.chunks_per_rank(gib, n) for n in (4, 8, 16, 32, 64)] == [65536, 32768, 16384, 8192, 4096]
    assert [worker.expected_lines(gib, n, 64) for n in (4, 8, 16, 32, 64)] == [1024, 512, 256, 128, 64]


@pytest.mark.unit
def test_summary_folds_every_window_and_the_tail(worker):
    # Two warps of 100 chunks each at one line per 64: a full window and a 36-chunk tail per warp.
    text = "\n".join(
        ["unrelated stdout"]
        + _warp_lines(cta=0, warp=0, chunks=100, interval=64)
        + _warp_lines(cta=1, warp=15, chunks=100, interval=64, cycles=300)
    )
    lines = worker.parse_lines(text)
    assert len(lines) == 4
    summary = worker.summarize(lines, rank=0, expected_chunks=200)
    assert (summary.lines, summary.warps, summary.chunks) == (4, 2, 200)
    assert (summary.chunks_per_warp_min, summary.chunks_per_warp_max) == (100, 100)
    assert summary.cycles_mean == pytest.approx(200.0)
    assert (summary.cycles_min, summary.cycles_max) == (90, 310)


@pytest.mark.unit
def test_a_dropped_line_fails_the_summary(worker):
    lines = worker.parse_lines("\n".join(_warp_lines(cta=0, warp=0, chunks=256, interval=64)[1:]))
    with pytest.raises(ValueError, match="dropped"):
        worker.summarize(lines, rank=0, expected_chunks=256)


@pytest.mark.unit
def test_missing_chunks_fail_the_summary(worker):
    # A whole warp missing (e.g. the FIFO filled before its first line) leaves the windows consistent but short.
    lines = worker.parse_lines("\n".join(_warp_lines(cta=0, warp=0, chunks=128, interval=64)))
    with pytest.raises(ValueError, match="cover 128 chunks, expected 256"):
        worker.summarize(lines, rank=0, expected_chunks=256)


@pytest.mark.unit
def test_a_malformed_line_raises(worker):
    with pytest.raises(ValueError, match="malformed"):
        worker.parse_lines("[nccl-ag-stmc-st] launch=2 rank=0 cta=0")


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"NCCL_SYM_AG_STMC_ST_PRINT_SKIP": "0"}, "must equal --warmup"),
        ({"NCCL_SYM_AG_STMC_ST_PRINT_LAUNCHES": "0"}, "must equal --warmup"),
        ({"NCCL_SYM_AG_STMC_ST_PRINT_INTERVAL": "0"}, "times nothing"),
        ({"NCCL_SYM_KERNEL": "allgather:AllGather_TmaSTMC"}, "NCCL_SYM_KERNEL"),
        ({"NCCL_NVLS_ENABLE": "0"}, "multicast"),
    ],
)
def test_worker_environment_checks(worker, monkeypatch, overrides, match):
    good = {
        "NCCL_SYM_AG_STMC_PAUSE_CYCLES": "0",
        "NCCL_SYM_AG_STMC_ST_PRINT_INTERVAL": "64",
        "NCCL_SYM_AG_STMC_ST_PRINT_SKIP": "2",
        "NCCL_SYM_AG_STMC_ST_PRINT_LAUNCHES": "1",
        "NCCL_SYM_KERNEL": "allgather:AllGather_STMC",
        "NCCL_WIN_ENABLE": "1",
        "NCCL_NVLS_ENABLE": "1",
    }
    for name, value in good.items():
        monkeypatch.setenv(name, value)
    assert worker.check_environment(2)[worker.INTERVAL_ENV] == 64
    for name, value in overrides.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match=match):
        worker.check_environment(2)


ENV = {
    "NCCL_SYM_AG_STMC_PAUSE_CYCLES": 1024,
    "NCCL_SYM_AG_STMC_ST_PRINT_INTERVAL": 64,
    "NCCL_SYM_AG_STMC_ST_PRINT_SKIP": 2,
    "NCCL_SYM_AG_STMC_ST_PRINT_LAUNCHES": 1,
}


def _nccl_log(*, banner: str = "2.32.3-pause-with-debug+cuda13.0", kernels=("AllGather_STMC",) * 3, extra=""):
    lines = [f"host:1:1 [0] NCCL INFO NCCL version {banner}"]
    lines += [f"host:1:1 [0] NCCL INFO {name} set by environment to {value}." for name, value in ENV.items()]
    lines += [
        f"host:1:1 [0] NCCL INFO AllGather [Symmetric]: 1073741824 Bytes -> Kernel {kernel} nchannels 4"
        " nthreads 512 nWorks 1"
        for kernel in kernels
    ]
    return "\n".join(lines + [extra]) + "\n"


@pytest.mark.unit
def test_nccl_log_of_a_good_run_passes(worker):
    check = worker.check_nccl_log(_nccl_log(), ENV, comm_rank0=True, launches=3)
    assert check.errors == []
    assert check.banner == "2.32.3-pause-with-debug+cuda13.0"
    assert check.tunings[-1] == (1073741824, "AllGather_STMC", 4, 512)


@pytest.mark.unit
@pytest.mark.parametrize(
    "log_kwargs, comm_rank0, match",
    [
        ({"banner": "2.32.3-pause+cuda13.0"}, False, "expected 2.32.3-pause-with-debug"),
        ({"extra": "NCCL INFO NCCL_MIN_CTAS set by environment to 4."}, False, "NCCL read NCCL_MIN_CTAS"),
        ({"kernels": ("AllGather_STMC", "AllGather_TmaSTMC", "AllGather_STMC")}, True, "not AllGather_STMC"),
        ({"kernels": ("AllGather_STMC",) * 2}, True, "2 AllGather launches"),
    ],
)
def test_nccl_log_problems_are_reported(worker, log_kwargs, comm_rank0, match):
    check = worker.check_nccl_log(_nccl_log(**log_kwargs), ENV, comm_rank0=comm_rank0, launches=3)
    assert any(match in error for error in check.errors), check.errors


@pytest.mark.unit
def test_nccl_log_missing_a_value_is_reported(worker):
    check = worker.check_nccl_log(
        _nccl_log(), ENV | {"NCCL_SYM_AG_STMC_PAUSE_CYCLES": 0}, comm_rank0=False, launches=3
    )
    assert check.errors == ["NCCL did not log NCCL_SYM_AG_STMC_PAUSE_CYCLES=0"]


@pytest.mark.unit
def test_output_file_reads_back_into_the_same_lines(worker):
    records = []
    for rank in (0, 1):
        lines = worker.parse_lines(
            "\n".join(
                line for w in range(2) for line in _warp_lines(cta=0, warp=w, chunks=100, interval=64, rank=rank)
            )
        )
        summary = worker.summarize(lines, rank=rank, expected_chunks=200)
        records.append(
            {
                "rank": rank,
                "host": "tray0",
                "elapsed_ms": 1.5,
                "busbw_gbps": 537.0,
                "summary": dataclasses.asdict(summary),
                "lines": [line.render() for line in lines],
            }
        )
    meta = {"nccl": "2.32.3-pause-with-debug+cuda13.0", "ctas": 1, "world_size": 2} | ENV
    text = worker.render_output(meta, records, [])
    header = [line for line in text.splitlines() if line.startswith("#")]
    assert header[0] == "# ag-st-debug status=ok"
    assert "ctas=1" in header[1] and "NCCL_SYM_AG_STMC_PAUSE_CYCLES=1024" in header[1]
    assert header[2].startswith("# rank=0 host=tray0 elapsed_ms=1.5000 busbw_gbps=537.0 lines=4 warps=2 chunks=200")
    assert "st_cycles_mean=100.0" in header[2]
    # The raw lines come back exactly, rank by rank.
    assert [line.render() for line in worker.parse_lines(text)] == records[0]["lines"] + records[1]["lines"]
    failed = worker.render_output(meta, records, ["rank 1: wrong AllGather output"])
    assert failed.startswith("# ag-st-debug status=failed")
    assert "# error: rank 1: wrong AllGather output" in failed


@pytest.mark.unit
def test_a_failed_run_gets_its_own_file_name(worker):
    assert worker.failed_path(Path("/r/pause1024cyc.log")) == Path("/r/pause1024cyc.failed.log")


def _fake_lib(tmp_path: Path, banner: str = "NCCL version 2.32.3-pause-with-debug+cuda13.0", envs=DEBUG_ENVS) -> Path:
    lib = tmp_path / "libnccl.so.2"
    lib.write_bytes(b"\0".join([b"", banner.encode(), *(name.encode() for name in envs), b""]))
    return lib


def _launch(lib: Path, *options: str, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ, "GLOO_SOCKET_IFNAME": "lo", "RUN_TIME": "agdbg-test", **(extra_env or {})}
    for name in DEBUG_ENVS:
        env.pop(name, None)
    command = ["bash", str(LAUNCHER), "--venv", "none", "--nccl-lib", str(lib), "--dry-run", *options]
    return subprocess.run(command, capture_output=True, text=True, env=env, check=False)


def _torchruns(result: subprocess.CompletedProcess) -> list[str]:
    return [line.replace("\\", "") for line in result.stdout.splitlines() if "torch.distributed.run" in line]


@pytest.mark.unit
@needs_bash
def test_dry_run_gives_every_pause_its_own_run_and_file(tmp_path):
    result = _launch(
        _fake_lib(tmp_path), "--warmup", "3", "--pause-cycles", "0,1024", extra_env={"NCCL_MIN_CTAS": "4"}
    )
    assert result.returncode == 0, result.stderr
    runs = _torchruns(result)
    assert len(runs) == 2
    for i, (command, pause) in enumerate(zip(runs, (0, 1024))):
        for setting in (
            f"NCCL_SYM_AG_STMC_PAUSE_CYCLES={pause} ",
            "NCCL_SYM_AG_STMC_ST_PRINT_INTERVAL=64",
            "NCCL_SYM_AG_STMC_ST_PRINT_SKIP=3",
            "NCCL_SYM_AG_STMC_ST_PRINT_LAUNCHES=1",
            "NCCL_SYM_KERNEL=allgather:AllGather_STMC",
            "NCCL_SYM_TMA_ENABLE=0",
            "-u NCCL_DEBUG_FILE",
            f"--master_port={29631 + i}",
            f"--output-file {REPO}/results/gb200/nvls-collective/ag-st-debug/gpu-num-4/agdbg-test/pause{pause}cyc.log",
            f"agdbg-test/.work/pause{pause}cyc.node0",
            "--warmup 3",
            "--size-mib 1024",
        ):
            assert setting in command
        assert "NCCL_MIN_CTAS" not in command
    assert "unsetting NCCL_MIN_CTAS=4" in result.stderr
    assert "65536 chunks of 4 KiB per rank" in result.stderr
    assert "1024 chunks and 16 lines per warp" in result.stderr


@pytest.mark.unit
@needs_bash
@pytest.mark.parametrize(
    "nnodes, pauses",
    [
        ("1", "0,128,256,512,768,1024,1536,2048,2560,3072,4096,6144"),
        ("8", "0,2048,3072,3584,4096,5120,6144,7168,8192,10240,12288,14336"),
        ("16", "0,6144,8192,9728,11264,12288,14336,16384,18432,22528,26624,31744"),
    ],
)
def test_default_pauses_are_those_of_the_pause_sweep(tmp_path, nnodes, pauses):
    options = ["--nnodes", nnodes] + (["--master-addr", "10.0.0.1"] if nnodes != "1" else [])
    result = _launch(_fake_lib(tmp_path), *options)
    assert result.returncode == 0, result.stderr
    assert f"pause cycles for the {int(nnodes) * 4}-GPU group: {pauses}\n" in result.stderr
    assert len(_torchruns(result)) == len(pauses.split(","))


@pytest.mark.unit
@needs_bash
@pytest.mark.parametrize(
    "options, match",
    [
        (["--pause-cycles", "0,abc"], "comma-separated list"),
        (["--pause-cycles", "512,512"], "twice"),
        (["--pause-cycles", "4294967296"], "uint32"),
        (["--gpus-per-node", "6"], "no default --pause-cycles for a 6-GPU group"),
    ],
)
def test_bad_pause_lists_are_rejected(tmp_path, options, match):
    result = _launch(_fake_lib(tmp_path), *options)
    assert result.returncode != 0
    assert match in result.stderr


@pytest.mark.unit
@needs_bash
def test_an_interval_that_overflows_the_printf_fifo_is_refused(tmp_path):
    result = _launch(_fake_lib(tmp_path), "--print-interval", "8")
    assert result.returncode != 0
    assert "pass --print-interval >= 16" in result.stderr


@pytest.mark.unit
@needs_bash
@pytest.mark.parametrize(
    "banner, envs, match",
    [
        ("NCCL version 2.32.3-pause+cuda13.0", DEBUG_ENVS, "not the nccl-2.32.3-pause-with-debug build"),
        ("NCCL version 2.32.3-pause-with-debug+cuda13.0", DEBUG_ENVS[:1], "does not read NCCL_SYM_AG_STMC_ST_PRINT"),
    ],
)
def test_a_library_without_the_timing_is_refused(tmp_path, banner, envs, match):
    result = _launch(_fake_lib(tmp_path, banner=banner, envs=envs))
    assert result.returncode != 0
    assert match in result.stderr


@pytest.mark.unit
def test_patch_spaces_the_lines_without_a_division():
    added = [
        line[1:] for line in PATCH.read_text().splitlines() if line.startswith("+") and not line.startswith("+++")
    ]
    code = [line.split("//")[0] for line in added]
    # The store loop counts down from the interval; neither it nor the countdown is an operand of / or %
    # (the lane test threadIdx.x % WARP_SIZE is a power-of-two constant, i.e. an AND).
    interval_uses = [line for line in code if "printInterval" in line or "stCountdown" in line]
    assert interval_uses
    operand = r"(?:dbg\.printInterval|printInterval|stCountdown)"
    assert not [line for line in code if re.search(rf"[/%]=?\s*{operand}\b|\b{operand}\s*[/%]", line)]
    assert any("--stCountdown == 0" in line for line in code)
    for name in DEBUG_ENVS:
        assert f'"{name.removeprefix("NCCL_")}"' in "\n".join(added)
    assert "NCCL_SUFFIX  := -pause-with-debug" in added
