"""CPU-only checks for direct host-output AllGather addressing and validation."""

import ctypes
import importlib.util
import mmap
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


_EXPERIMENT_DIR = Path(__file__).resolve().parents[3] / "examples" / "gb200" / "hbm-conflict"
_SPEC = importlib.util.spec_from_file_location("ce_host_output", _EXPERIMENT_DIR / "ce_host_output.py")
assert _SPEC is not None and _SPEC.loader is not None
_BACKEND = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BACKEND)


@pytest.mark.unit
def test_direct_host_output_fills_each_receive_buffer(monkeypatch, tmp_path):
    size = 3 * mmap.PAGESIZE
    world_size = 3
    shard = size // world_size
    registered = []
    unregistered = []
    copies = []

    def register(address, count, flags):
        registered.append((address, count, flags))
        return 0

    def unregister(address):
        unregistered.append(address)
        return 0

    def copy(destination, source, count, stream):
        copies.append((destination, count, stream))
        ctypes.memmove(destination, source, count)
        return 0

    def error_string(result, message):
        return 0

    driver = SimpleNamespace(
        cuMemHostRegister_v2=register,
        cuMemHostUnregister=unregister,
        cuMemcpyDtoHAsync_v2=copy,
        cuGetErrorString=error_string,
    )
    monkeypatch.setattr(_BACKEND.ctypes, "CDLL", lambda name: driver)
    # The memfd rendezvous runs on Linux. Here, independent mmap views
    # represent the source ranks' registrations of the same shared file.
    monkeypatch.setattr(_BACKEND.os, "memfd_create", lambda name, flags: -1, raising=False)
    monkeypatch.setattr(_BACKEND.dist, "barrier", lambda: None)
    full = torch.full
    monkeypatch.setattr(torch, "full", lambda shape, value, *, dtype, device: full(shape, value, dtype=dtype))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: SimpleNamespace(cuda_stream=7))
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)

    with (tmp_path / "output").open("w+b") as backing:
        backing.truncate(world_size * size)
        monkeypatch.setattr(
            _BACKEND, "_map_host_outputs", lambda **kwargs: mmap.mmap(backing.fileno(), world_size * size)
        )
        gatherers = [
            _BACKEND.HostOutputCopyEngineAllGather(
                torch.device("cuda", rank), rank=rank, world_size=world_size, size_bytes=size
            )
            for rank in range(world_size)
        ]
        try:
            for rank, gatherer in enumerate(gatherers):
                gatherer.run()
                base, registered_size, _ = registered[rank]
                assert registered_size == world_size * size
                actual = copies[-world_size:]
                expected = [
                    (base + (rank + step) % world_size * size + rank * shard, shard, 7) for step in range(world_size)
                ]
                assert actual == expected
            for gatherer in gatherers:
                gatherer.validate()
            gatherers[1].reset_output()
            with pytest.raises(RuntimeError, match="rank 1 received incorrect bytes for rank 0"):
                gatherers[1].validate()
        finally:
            for gatherer in gatherers:
                gatherer.close()
    assert unregistered == [address for address, _, _ in registered]
