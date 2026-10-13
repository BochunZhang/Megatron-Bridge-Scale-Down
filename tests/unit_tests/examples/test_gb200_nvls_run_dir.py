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

"""CPU-only checks for the GB200 NVLS run-directory allocation.

The multi-node handshake is exercised with an in-process stand-in for ``TCPStore``: one
thread per node shares a key/value dict, which is the only behaviour the handshake relies on.
"""

import importlib.util
import sys
import threading
import types
from pathlib import Path

import pytest


RUN_DIR_PATH = Path(__file__).resolve().parents[3] / "examples" / "gb200" / "nvls-collective" / "run_dir.py"
_SPEC = importlib.util.spec_from_file_location("gb200_nvls_run_dir", RUN_DIR_PATH)
assert _SPEC is not None and _SPEC.loader is not None
run_dir = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = run_dir
_SPEC.loader.exec_module(run_dir)


class _FakeStore:
    """Blocking get/wait over a dict shared by every 'node' thread."""

    data: dict[str, bytes] = {}
    cond = threading.Condition()

    def __init__(self, host, port, world_size, is_master, timeout, wait_for_workers=True):
        self.timeout = timeout.total_seconds()

    def set(self, key, value):
        with self.cond:
            self.data[key] = value.encode() if isinstance(value, str) else value
            self.cond.notify_all()

    def get(self, key):
        with self.cond:
            assert self.cond.wait_for(lambda: key in self.data, self.timeout)
            return self.data[key]

    def wait(self, keys, timeout):
        with self.cond:
            assert self.cond.wait_for(lambda: all(k in self.data for k in keys), timeout.total_seconds())


@pytest.fixture
def fake_store(monkeypatch):
    _FakeStore.data = {}
    torch_stub = types.ModuleType("torch")
    distributed_stub = types.ModuleType("torch.distributed")
    distributed_stub.TCPStore = _FakeStore
    torch_stub.distributed = distributed_stub
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "torch.distributed", distributed_stub)


@pytest.mark.unit
def test_first_run_is_run_01_under_gpu_num(tmp_path):
    path = run_dir.claim_run_dir(run_dir.gpu_dir(tmp_path, 16))
    assert path == tmp_path / "gpu-num-16" / "run-01"
    assert (path / run_dir.SHARED_MARKER).is_file()


@pytest.mark.unit
def test_next_run_follows_the_highest_existing_index(tmp_path):
    parent = tmp_path / "gpu-num-4"
    for name in ("run-01", "run-03", "notes", "run-x"):
        (parent / name).mkdir(parents=True)
    (parent / "run-07").write_text("a file, not a run", encoding="utf-8")
    assert run_dir.claim_run_dir(parent).name == "run-04"
    assert run_dir.claim_run_dir(parent).name == "run-05"


@pytest.mark.unit
def test_run_names_keep_two_digits_then_grow():
    assert [run_dir.run_name(i) for i in (1, 9, 10, 99, 100)] == ["run-01", "run-09", "run-10", "run-99", "run-100"]


@pytest.mark.unit
def test_claim_skips_a_directory_created_concurrently(tmp_path, monkeypatch):
    parent = tmp_path / "gpu-num-8"
    parent.mkdir()
    real_mkdir = Path.mkdir

    def racing_mkdir(self, *args, **kwargs):
        if self.name == "run-01" and not self.exists():
            real_mkdir(self)  # another job wins the race
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", racing_mkdir)
    assert run_dir.claim_run_dir(parent).name == "run-02"


@pytest.mark.unit
def test_dry_run_prints_without_creating(tmp_path, capsys):
    run_dir.main(["--root", str(tmp_path), "--world-size", "32", "--dry-run"])
    assert capsys.readouterr().out.strip() == str(tmp_path / "gpu-num-32" / "run-01")
    assert not (tmp_path / "gpu-num-32").exists()


@pytest.mark.unit
def test_single_node_needs_no_store(tmp_path, capsys):
    run_dir.main(["--root", str(tmp_path), "--world-size", "4"])
    assert capsys.readouterr().out.strip() == str(tmp_path / "gpu-num-4" / "run-01")


@pytest.mark.unit
def test_multi_node_requires_a_port(tmp_path):
    with pytest.raises(SystemExit):
        run_dir.main(["--root", str(tmp_path), "--world-size", "8", "--nnodes", "2"])


def _run_nodes(root, nnodes):
    outputs: dict[int, object] = {}

    def node(rank):
        try:
            path = run_dir.claim_run_dir(run_dir.gpu_dir(root, 4 * nnodes)) if rank == 0 else None
            outputs[rank] = run_dir.share_with_nodes(
                path, nnodes=nnodes, node_rank=rank, host="h", port=1, timeout_s=5
            )
        except BaseException as error:  # noqa: BLE001 - surfaced to the test thread
            outputs[rank] = error

    threads = [threading.Thread(target=node, args=(rank,)) for rank in range(nnodes)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    return outputs


@pytest.mark.unit
def test_every_node_gets_node_zeros_directory(tmp_path, fake_store):
    # A previous run exists; late nodes must not pick run-02 independently of node 0.
    (tmp_path / "gpu-num-16" / "run-01").mkdir(parents=True)
    outputs = _run_nodes(tmp_path, 4)
    assert outputs == {rank: tmp_path / "gpu-num-16" / "run-02" for rank in range(4)}


@pytest.mark.unit
def test_node_that_cannot_see_node_zeros_directory_fails(tmp_path, monkeypatch, capsys):
    # Simulates a repo that is not on shared storage: the published path has no marker here.
    unshared = tmp_path / "gpu-num-8" / "run-01"
    unshared.mkdir(parents=True)
    monkeypatch.setattr(run_dir, "share_with_nodes", lambda *a, **k: unshared)
    argv = ["--root", str(tmp_path), "--world-size", "8", "--nnodes", "2", "--node-rank", "1", "--port", "2"]
    with pytest.raises(SystemExit, match="not on storage shared with node 0"):
        run_dir.main(argv)
    (unshared / run_dir.SHARED_MARKER).write_text("x", encoding="utf-8")
    run_dir.main(argv)
    assert capsys.readouterr().out.strip() == str(unshared)
