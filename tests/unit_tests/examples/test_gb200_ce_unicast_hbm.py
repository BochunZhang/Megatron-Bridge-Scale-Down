"""CPU-only checks for the GB200 CE unicast sweep and its trace gate."""

import importlib
import shlex
import subprocess
from pathlib import Path

import pytest


_EXPERIMENT = Path(__file__).resolve().parents[3] / "examples" / "gb200" / "hbm-conflict"


def _dry_run(*args: str) -> list[list[str]]:
    result = subprocess.run(
        ["bash", str(_EXPERIMENT / "run_ce_unicast_hbm_sweep.sh"), "--gpus", "0,1,2,3", "--dry-run", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return [shlex.split(line) for line in result.stdout.splitlines() if "--mode " in line]


def _option(command: list[str], key: str) -> str:
    return command[command.index(key) + 1]


@pytest.mark.unit
def test_unicast_sweep_reuses_gemm_shapes_and_configures_unicast():
    commands = _dry_run()
    shapes = [f"[{m},8192,7168]" for m in (2, 4, 8, 16, 32, 64)]
    expected = (
        [("gemm", shape) for shape in shapes]
        + [("ag-hbm-unicast", None), ("ag-host-unicast", None)]
        + [("gemm-ag-hbm-unicast", shape) for shape in shapes]
        + [("gemm-ag-host-unicast", shape) for shape in shapes]
    )
    assert [
        (_option(cmd, "--mode"), _option(cmd, "--gemm") if "--gemm" in cmd else None) for cmd in commands
    ] == expected

    for command in commands:
        if _option(command, "--mode") == "gemm":
            assert {"--nproc_per_node=1", "--capture-range=cudaProfilerApi"}.issubset(command)
        else:
            assert {
                "--nproc_per_node=4",
                "--capture-range=none",
                "NCCL_CTA_POLICY=ZERO",
                "NCCL_WIN_ENABLE=1",
                "NCCL_NVLS_ENABLE=0",
                "NCCL_CE_COLL_AG_MULTICAST_THRESHOLD=0",
            }.issubset(command)


@pytest.mark.unit
def test_unicast_sweep_can_select_a_two_gpu_baseline_and_override_shapes():
    commands = _dry_run("--gpus", "1,3", "--cases", "baseline6,task7", "--gemm", "[4,4096,4096]", "--comm-rounds", "3")
    assert [_option(cmd, "--mode") for cmd in commands] == ["ag-host-unicast", "gemm-ag-hbm-unicast"]
    assert all("--nproc_per_node=2" in cmd and _option(cmd, "--comm-rounds") == "3" for cmd in commands)
    assert _option(commands[1], "--gemm") == "[4,4096,4096]"


@pytest.fixture
def analysis(monkeypatch):
    monkeypatch.syspath_prepend(str(_EXPERIMENT))
    return importlib.import_module("analyse_ce_hbm")


@pytest.mark.unit
def test_dump_timeline_uses_the_source_view_of_peer_copies(analysis):
    rows = [
        {
            "Text": f"Memcpy PtoP ({side})",
            "Tooltip": f"PtoP&nbsp;memcpy&nbsp;1048576&nbsp;bytes<br>Channel&nbsp;type:&nbsp;{channel}<br>",
            "Start": 2000,
            "Duration": 50,
        }
        for side, channel in (("source", "Async&nbsp;Memcpy"), ("destination", "Compute"))
    ]
    document = {
        "GenericHierarchyRows": [
            {
                "Path": "/HWs/0/VMs/0/Processes/100/CudaDevices/0/Contexts/1/GreenContexts/0/Streams/20/Memory/Memcpy/10;",
                "Rows": [{"ViewDataElements": rows}],
            }
        ]
    }

    assert analysis._hardware_channels(document) == {(100, 0, 2000, 2050, 1048576): {"Async Memcpy"}}


@pytest.mark.unit
@pytest.mark.parametrize("host_input", [False, True])
@pytest.mark.parametrize("fault", [None, "compute", "missing channel", "wrong memory", "wrong destination"])
def test_unicast_gate_requires_the_correct_bytes_endpoints_and_hardware_channel(analysis, host_input, fault):
    size_bytes = 2 << 20
    shard_bytes = size_bytes // 2
    payload_bytes = size_bytes if host_input else shard_bytes
    mode = "ag-host-unicast" if host_input else "ag-hbm-unicast"
    records = {
        rank: {
            "rank": rank,
            "world_size": 2,
            "physical_gpu": rank,
            "mode": mode,
            "size_bytes": size_bytes,
            "comm_payload_bytes": shard_bytes,
            "comm_source_bytes": payload_bytes,
            "comm_receive_bytes": payload_bytes,
            "comm_rounds_measured": 1,
            "comm_ms": [[0.001, 0.005]],
        }
        for rank in range(2)
    }
    activities = {}
    channels = {}
    for rank in records:
        copies = []
        for dst in records if host_input else (1 - rank,):
            start = 2000 + 100 * dst
            end = start + 50
            selected = rank == 0 and dst == 1
            src_kind = "Pinned" if host_input else "Device"
            if selected and fault == "wrong memory":
                src_kind = "Device" if host_input else "Pinned"
            copies.append(
                analysis.Activity(
                    start,
                    end,
                    rank,
                    "MEMCPY",
                    shard_bytes,
                    "Host-to-Device" if rank == dst else "Peer-to-Peer",
                    src_kind,
                    "Device",
                    pid=100 + rank,
                    src_device=rank,
                    dst_device=0 if selected and fault == "wrong destination" else dst,
                    copy_count=1,
                )
            )
            if not (selected and fault == "missing channel"):
                channels[100 + rank, rank, start, end, shard_bytes] = {
                    "Compute" if selected and fault == "compute" else "Async Memcpy"
                }
        activities[rank] = {"comm00": copies}

    errors, incoming = analysis._validate_unicast_trace(records, activities, channels)

    if fault is None:
        assert not any(errors.values())
        for rank, copies in incoming.items():
            assert sum(copy.size_bytes for copy in copies) == payload_bytes
            assert all(copy.dst_device == rank for copy in copies)
    else:
        assert errors[0]
