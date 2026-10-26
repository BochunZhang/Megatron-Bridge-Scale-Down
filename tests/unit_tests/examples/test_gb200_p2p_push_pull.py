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

"""CPU-only checks for the GB200 P2P push / pull benchmark.

The real run needs two NVLink-connected GPUs. These tests cover report naming (which GPU is the
push sender / pull receiver), the environment checks, and the launcher: its torchrun commands and
the renaming of the nsys reports, the latter with fake nvidia-smi / nsys / uv on PATH.
"""

import importlib.util
import os
import shutil
import stat
import subprocess
import sys
import types
from pathlib import Path

import pytest


BENCHMARK_DIR = Path(__file__).resolve().parents[3] / "examples" / "gb200" / "nvls-collective"
WORKER_PATH = BENCHMARK_DIR / "p2p_push_pull_benchmark.py"
LAUNCHER_PATH = BENCHMARK_DIR / "run_p2p_push_pull_benchmark.sh"
needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _load_worker() -> types.ModuleType:
    """Import the worker (and the nvls_collective_benchmark it reuses), stubbing torch if absent."""
    sys.path.insert(0, str(BENCHMARK_DIR))
    stub_modules: dict[str, types.ModuleType] = {}
    if importlib.util.find_spec("torch") is None:
        torch_stub = types.ModuleType("torch")
        torch_stub.__path__ = []  # type: ignore[attr-defined]
        torch_stub.bfloat16 = "bfloat16"  # type: ignore[attr-defined]
        torch_stub.float32 = "float32"  # type: ignore[attr-defined]
        cuda_stub = types.ModuleType("torch.cuda")
        cuda_stub.__path__ = []  # type: ignore[attr-defined]
        cuda_stub.profiler = types.ModuleType("torch.cuda.profiler")  # type: ignore[attr-defined]
        cuda_stub.nvtx = types.ModuleType("torch.cuda.nvtx")  # type: ignore[attr-defined]
        torch_stub.cuda = cuda_stub  # type: ignore[attr-defined]
        torch_stub.distributed = types.ModuleType("torch.distributed")  # type: ignore[attr-defined]
        stub_modules = {
            "torch": torch_stub,
            "torch.distributed": torch_stub.distributed,
            "torch.cuda": cuda_stub,
            "torch.cuda.profiler": cuda_stub.profiler,
            "torch.cuda.nvtx": cuda_stub.nvtx,
        }
        sys.modules.update(stub_modules)
    try:
        spec = importlib.util.spec_from_file_location("gb200_p2p_push_pull", WORKER_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules["gb200_p2p_push_pull"] = module
        spec.loader.exec_module(module)
    finally:
        for name in stub_modules:
            del sys.modules[name]
        sys.path.remove(str(BENCHMARK_DIR))
    return module


worker = _load_worker()


@pytest.mark.unit
def test_report_names_mark_the_push_sender_and_the_pull_receiver_as_initiators():
    assert worker.report_name("push", 1024, rank=0, gpu=0) == "push.1024MiB.gpu0.sender.initiator"
    assert worker.report_name("push", 1024, rank=1, gpu=1) == "push.1024MiB.gpu1.receiver"
    assert worker.report_name("pull", 1024, rank=0, gpu=2) == "pull.1024MiB.gpu2.sender"
    assert worker.report_name("pull", 1024, rank=1, gpu=3) == "pull.1024MiB.gpu3.receiver.initiator"


@pytest.mark.unit
def test_index_rows_list_each_rank_with_its_role_and_report():
    rows = worker.index_rows("pull", 256, capture_index=2, gpus=[0, 1])
    assert [row.rstrip("\n").split("\t") for row in rows] == [
        ["2", "256", "0", "0", "sender", "no", "pull.256MiB.gpu0.sender"],
        ["2", "256", "1", "1", "receiver", "yes", "pull.256MiB.gpu1.receiver.initiator"],
    ]
    assert worker.INDEX_HEADER.rstrip("\n").split("\t")[-1] == "report"


@pytest.mark.unit
def test_physical_gpu_follows_cuda_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    assert [worker.physical_gpu(0), worker.physical_gpu(1)] == [2, 3]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc,GPU-def")
    with pytest.raises(RuntimeError, match="numeric"):
        worker.physical_gpu(0)


def _good_env(monkeypatch, workload: str) -> None:
    monkeypatch.setenv("NCCL_P2P_READ_ENABLE", "1" if workload == "pull" else "0")
    monkeypatch.setenv("NCCL_DEBUG", "INFO")
    monkeypatch.setenv("NCCL_DEBUG_FILE", "/tmp/nccl-%h-%p.log")
    monkeypatch.setenv("NCCL_DEBUG_SUBSYS", "INIT,P2P,ENV")


@pytest.mark.unit
@pytest.mark.parametrize("workload", ["push", "pull"])
def test_check_environment_accepts_the_launcher_settings(monkeypatch, workload):
    _good_env(monkeypatch, workload)
    worker.check_environment(workload, 2)
    monkeypatch.delenv("NCCL_DEBUG_SUBSYS")
    worker.check_environment(workload, 2)


@pytest.mark.unit
@pytest.mark.parametrize(
    "change, match",
    [
        ({"world_size": 4}, "need 2"),
        ({"NCCL_P2P_READ_ENABLE": "1"}, "NCCL_P2P_READ_ENABLE=0"),
        ({"NCCL_P2P_READ_ENABLE": None}, "NCCL_P2P_READ_ENABLE=0"),
        ({"NCCL_DEBUG": "WARN"}, "NCCL_DEBUG=INFO"),
        ({"NCCL_DEBUG_FILE": "/tmp/nccl.log"}, "%p"),
        ({"NCCL_DEBUG_SUBSYS": "COLL,ENV"}, "include INIT"),
        ({"NCCL_DEBUG_SUBSYS": "^INIT"}, "include INIT"),
    ],
)
def test_check_environment_rejects_a_misconfigured_push(monkeypatch, change, match):
    _good_env(monkeypatch, "push")
    world_size = change.pop("world_size", 2)
    for name, value in change.items():
        if value is None:
            monkeypatch.delenv(name)
        else:
            monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match=match):
        worker.check_environment("push", world_size)


@pytest.mark.unit
def test_parse_args_rejects_duplicate_sizes_and_zero_trials(tmp_path):
    args = worker.parse_args(["--workload", "pull", "--output-dir", str(tmp_path), "--sizes-mib", "64,1024"])
    assert args.sizes_mib == [64, 1024] and not args.nsys
    with pytest.raises(SystemExit):
        worker.parse_args(["--workload", "pull", "--output-dir", str(tmp_path), "--sizes-mib", "64,64"])
    with pytest.raises(SystemExit):
        worker.parse_args(["--workload", "push", "--output-dir", str(tmp_path), "--trials", "0"])


def _launcher(*options: str, path: str | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ, "GLOO_SOCKET_IFNAME": "lo", "RUN_TIME": "dry-run-test"}
    env.pop("NCCL_P2P_READ_ENABLE", None)
    if path is not None:
        env["PATH"] = path
    return subprocess.run(["bash", str(LAUNCHER_PATH), *options], capture_output=True, text=True, env=env, check=False)


def _torchrun_lines(stdout: str) -> list[str]:
    # The dry run prints shell-quoted words (printf %q escapes ",").
    return [line.replace("\\", "") for line in stdout.splitlines() if "torch.distributed.run" in line]


@pytest.mark.unit
@needs_bash
def test_launcher_runs_one_two_rank_job_per_workload_with_the_matching_p2p_mode(tmp_path):
    result = _launcher("--dry-run", "--gpus", "2,3", "--output-dir", str(tmp_path / "run"))
    assert result.returncode == 0, result.stderr
    push, pull = _torchrun_lines(result.stdout)
    for line, read_enable, workload in ((push, "0", "push"), (pull, "1", "pull")):
        assert f"NCCL_P2P_READ_ENABLE={read_enable}" in line
        assert "CUDA_VISIBLE_DEVICES=2,3" in line
        assert "--nproc_per_node=2" in line
        assert f"--workload {workload}" in line
        assert f"--output-dir {tmp_path / 'run' / workload}" in line
        # nsys is on by default: one session per GPU, one report per size.
        assert "nsys_worker.sh" in line and "--capture-range-end=repeat" in line and line.endswith("--nsys ")
    assert "--master_port=29711" in push and "--master_port=29712" in pull


@pytest.mark.unit
@needs_bash
def test_launcher_without_nsys_runs_the_worker_directly(tmp_path):
    result = _launcher("--dry-run", "--workloads", "pull", "--no-nsys", "--output-dir", str(tmp_path))
    assert result.returncode == 0, result.stderr
    (line,) = _torchrun_lines(result.stdout)
    assert "nsys_worker.sh" not in line and not line.endswith("--nsys ")
    assert "p2p_push_pull_benchmark.py --workload pull" in line


@pytest.mark.unit
@needs_bash
@pytest.mark.parametrize(
    "options, match",
    [
        (["--gpus", "0,0"], "different GPUs"),
        (["--gpus", "0,4"], "not a GPU id"),
        (["--gpus", "0,1,2"], "exactly two"),
        (["--workloads", "push,read"], "unknown workload"),
        (["--trials", "0"], "must be positive"),
    ],
)
def test_launcher_rejects_bad_options(options, match):
    result = _launcher("--dry-run", *options)
    assert result.returncode != 0
    assert match in result.stderr


def _fake_tool(directory: Path, name: str, body: str) -> None:
    tool = directory / name
    tool.write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8")
    tool.chmod(tool.stat().st_mode | stat.S_IEXEC)


def _fake_tools(tmp_path: Path, *, gpus: int) -> str:
    """PATH with fake nvidia-smi / nsys and a uv that plays the torchrun job.

    The fake job writes what the worker would: nsys/index.tsv (from the worker's own index_rows,
    staged in FAKE_INDEX_DIR) and nsys's raw rank<R>.<K>.nsys-rep files.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake_tool(bin_dir, "nvidia-smi", "".join(f"echo 'GPU {i}: NVIDIA GB200 (UUID: GPU-{i})'\n" for i in range(gpus)))
    _fake_tool(bin_dir, "nsys", "echo 'NVIDIA Nsight Systems version fake'\n")
    _fake_tool(
        bin_dir,
        "uv",
        """args=("$@")
for (( i = 0; i < ${#args[@]}; i++ )); do
    case "${args[i]}" in
        --workload) workload="${args[i + 1]}" ;;
        --output-dir) out="${args[i + 1]}" ;;
    esac
done
mkdir -p "${out}/nsys"
echo "$NCCL_P2P_READ_ENABLE" >"${out}/read-enable"
cp "${FAKE_INDEX_DIR}/${workload}.tsv" "${out}/nsys/index.tsv"
for rank in 0 1; do echo "rank${rank}" >"${out}/nsys/rank${rank}.1.nsys-rep"; done
""",
    )
    return f"{bin_dir}{os.pathsep}{os.environ['PATH']}"


@pytest.mark.unit
@needs_bash
def test_launcher_renames_reports_to_say_who_sent_received_and_initiated(tmp_path, monkeypatch):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    for workload in ("push", "pull"):
        rows = worker.index_rows(workload, 1024, capture_index=1, gpus=[0, 1])
        (index_dir / f"{workload}.tsv").write_text(worker.INDEX_HEADER + "".join(rows), encoding="utf-8")
    monkeypatch.setenv("FAKE_INDEX_DIR", str(index_dir))
    run_dir = tmp_path / "run"

    result = _launcher("--output-dir", str(run_dir), path=_fake_tools(tmp_path, gpus=4))

    assert result.returncode == 0, result.stderr
    assert (run_dir / "push" / "read-enable").read_text().strip() == "0"
    assert (run_dir / "pull" / "read-enable").read_text().strip() == "1"
    assert sorted(p.name for p in (run_dir / "push" / "nsys").glob("*.nsys-rep")) == [
        "push.1024MiB.gpu0.sender.initiator.nsys-rep",
        "push.1024MiB.gpu1.receiver.nsys-rep",
    ]
    assert sorted(p.name for p in (run_dir / "pull" / "nsys").glob("*.nsys-rep")) == [
        "pull.1024MiB.gpu0.sender.nsys-rep",
        "pull.1024MiB.gpu1.receiver.initiator.nsys-rep",
    ]
    # The renamed report keeps rank 0's / rank 1's content.
    assert (run_dir / "push" / "nsys" / "push.1024MiB.gpu0.sender.initiator.nsys-rep").read_text() == "rank0\n"
    summary = (run_dir / "nsys-reports.md").read_text()
    assert (
        "| push | 1024 | GPU0 | push P2P sender | yes | push/nsys/push.1024MiB.gpu0.sender.initiator.nsys-rep |"
        in summary
    )
    assert (
        "| pull | 1024 | GPU1 | pull P2P receiver | yes | pull/nsys/pull.1024MiB.gpu1.receiver.initiator.nsys-rep |"
        in summary
    )


@pytest.mark.unit
@needs_bash
def test_launcher_refuses_a_node_without_four_gpus(tmp_path):
    result = _launcher("--output-dir", str(tmp_path / "run"), path=_fake_tools(tmp_path, gpus=8))
    assert result.returncode != 0
    assert "4-GPU node only" in result.stderr
