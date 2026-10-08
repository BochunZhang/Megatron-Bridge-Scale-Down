#!/usr/bin/env python3
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

"""Measure GB200 C2C bandwidth while persistent host RDMA traffic runs.

The background stream is supplied by the linux-rdma ``ib_write_bw`` tool.  It
uses host memory and an explicitly selected HCA/IP for each GPU pair, so no
GPU collective or GPU-to-GPU NVLink operation is involved.  Gloo is used only
as a TCP control plane for synchronising the four workers and gathering JSON
results.

Four independent groups are measured: ``d2h_send``, ``d2h_recv``,
``h2d_send``, and ``h2d_recv``.  In each group the selected copy direction is
measured once without background traffic and once while either the perftest
sender or receiver role is active.  The perftest message size is fixed for the
whole run and ``--run_infinitely`` keeps it active until the C2C event has
completed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import signal
import socket
import subprocess
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import torch
import torch.distributed as dist


LOGGER = logging.getLogger(__name__)
BYTES_PER_MIB = 1024 * 1024
BYTES_PER_GB = 1_000_000_000
EXPECTED_WORLD_SIZE = 4
GPU_PAIRS = ((0, 1), (2, 3))
GLOO_TRANSPORT = "TCP"
GLOO_SOCKET_INTERFACE = "lo"
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TOPOLOGY_FILE = DEFAULT_REPO_ROOT / ".cache" / "nvidia-smi_topo.txt"
DEFAULT_IP_ADDR_FILE = DEFAULT_REPO_ROOT / ".cache" / "ip_addr.txt"


@dataclass(frozen=True)
class CopyMeasurement:
    """Result of one host/device copy direction."""

    direction: str
    bandwidth_gb_s: float
    elapsed_seconds: float
    transferred_bytes: int


@dataclass(frozen=True)
class TopologyEndpoint:
    """RDMA endpoint selected for one physical GPU."""

    gpu_index: int
    hca: str
    netdev: str
    ip_address: str
    numa_node: int
    topology_relation: str


@dataclass(frozen=True)
class RankMeasurement:
    """Results and topology metadata for one worker."""

    rank: int
    hostname: str
    gpu_name: str
    gpu_pci_bus_id: str
    gpu_index: int
    direction: str
    rdma_role: str
    peer_gpu_index: int
    endpoint: TopologyEndpoint
    peer_endpoint: TopologyEndpoint
    baseline: CopyMeasurement | None
    with_background: CopyMeasurement | None


@dataclass(frozen=True)
class RdmaProcess:
    """One long-lived perftest process and its output file."""

    process: subprocess.Popen[bytes]
    log_path: Path


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive number, got {value}")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c2c-buffer-mib", type=_positive_int, default=512)
    parser.add_argument("--warmup-iterations", type=_positive_int, default=5)
    parser.add_argument("--copy-iterations", type=_positive_int, default=20)
    parser.add_argument("--ib-write-bw", default="ib_write_bw")
    parser.add_argument("--rdma-size-mib", type=_positive_int, default=256)
    parser.add_argument("--rdma-qp", type=_positive_int, default=1)
    parser.add_argument("--rdma-tx-depth", type=_positive_int, default=128)
    parser.add_argument("--rdma-report-interval", type=_positive_int, default=5)
    parser.add_argument("--rdma-ready-timeout-seconds", type=_positive_float, default=30.0)
    parser.add_argument("--rdma-startup-delay-seconds", type=_positive_float, default=2.0)
    parser.add_argument("--rdma-port", type=_positive_int, default=18515)
    parser.add_argument("--topology-file", type=Path, default=DEFAULT_TOPOLOGY_FILE)
    parser.add_argument("--ip-addr-file", type=Path, default=DEFAULT_IP_ADDR_FILE)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _bandwidth_gb_s(*, transferred_bytes: int, elapsed_seconds: float) -> float:
    if transferred_bytes <= 0:
        raise ValueError("transferred_bytes must be positive")
    if elapsed_seconds <= 0:
        raise ValueError("elapsed_seconds must be positive")
    return transferred_bytes / elapsed_seconds / BYTES_PER_GB


def _drop_percent(*, baseline_gb_s: float, concurrent_gb_s: float) -> float:
    if baseline_gb_s <= 0:
        raise ValueError("baseline_gb_s must be positive")
    return (baseline_gb_s - concurrent_gb_s) / baseline_gb_s * 100.0


@contextmanager
def _nvtx_range(name: str) -> Iterator[None]:
    """Annotate a host scope for Nsight Systems through PyTorch NVTX APIs."""
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def _validate_environment() -> None:
    """Require Gloo TCP for control traffic and avoid Gloo RDMA payloads."""
    transport = os.environ.get("GLOO_DEVICE_TRANSPORT")
    if transport != GLOO_TRANSPORT:
        raise RuntimeError(
            "CPU RDMA perftest benchmark requires Gloo TCP control traffic; "
            f"set GLOO_DEVICE_TRANSPORT={GLOO_TRANSPORT!r}, got {transport!r}"
        )
    socket_interface = os.environ.get("GLOO_SOCKET_IFNAME")
    if socket_interface != GLOO_SOCKET_INTERFACE:
        raise RuntimeError(
            "Gloo control traffic must use loopback; "
            f"set GLOO_SOCKET_IFNAME={GLOO_SOCKET_INTERFACE!r}, got {socket_interface!r}"
        )


def _resolve_gpu_pci_bus_id(local_rank: int) -> str:
    """Resolve the physical PCI bus ID for the rank's visible GPU."""
    bus_ids = [value for value in os.environ.get("BENCHMARK_GPU_PCI_BUS_IDS", "").split(",") if value]
    if local_rank < len(bus_ids):
        return bus_ids[local_rank]
    return os.environ.get("BENCHMARK_GPU_PCI_BUS_ID", "unknown")


def _parse_topology_file(path: Path) -> dict[int, tuple[str, str, int]]:
    """Return ``GPU index -> (HCA, relation, NUMA node)`` from ``nvidia-smi topo -m``."""
    lines = path.read_text(encoding="utf-8").splitlines()
    header_index = next(
        (index for index, line in enumerate(lines) if "GPU0" in line and "NIC0" in line),
        None,
    )
    if header_index is None:
        raise ValueError(f"could not find GPU/NIC header in topology file {path}")
    header = lines[header_index].split()
    nic_columns = {name: index for index, name in enumerate(header) if re.fullmatch(r"NIC\d+", name)}
    if not nic_columns:
        raise ValueError(f"could not find NIC columns in topology file {path}")
    nic_names: dict[str, str] = {}
    for line in lines[header_index + 1 :]:
        match = re.match(r"^\s*(NIC\d+)\s*:\s*(\S+)", line)
        if match:
            nic_names[match.group(1)] = match.group(2)
    result: dict[int, tuple[str, str, int]] = {}
    for line in lines[header_index + 1 :]:
        parts = line.split()
        if not parts or not re.fullmatch(r"GPU\d+", parts[0]):
            continue
        gpu_index = int(parts[0][3:])
        nic_key = f"NIC{gpu_index}"
        if nic_key not in nic_columns or nic_key not in nic_names:
            raise ValueError(f"topology file has no matching {nic_key} for GPU{gpu_index}")
        column = nic_columns[nic_key] + 1  # topology rows include the GPU label
        if column >= len(parts):
            raise ValueError(f"topology row for GPU{gpu_index} is incomplete")
        relation = parts[column]
        if relation.startswith("NV"):
            raise ValueError(f"GPU{gpu_index} to {nic_key} uses an NVLink relation: {relation}")
        if relation not in {"PIX", "PXB", "PHB", "NODE"}:
            raise ValueError(f"GPU{gpu_index} to {nic_key} has unsupported relation {relation!r}")
        numa_node: int | None = None
        for token in parts[column + 1 :]:
            if re.fullmatch(r"\d+", token):
                numa_node = int(token)
                break
        if numa_node is None:
            raise ValueError(f"could not parse NUMA node for GPU{gpu_index}")
        result[gpu_index] = (nic_names[nic_key], relation, numa_node)
    if not result:
        raise ValueError(f"no GPU rows found in topology file {path}")
    return result


def _parse_ip_addr_file(path: Path) -> dict[str, tuple[str, str]]:
    """Return ``HCA -> (bond interface, IPv4 address)`` from ``ip addr`` output."""
    lines = path.read_text(encoding="utf-8").splitlines()
    hca_to_netdev: dict[str, str] = {}
    netdev_to_master: dict[str, str] = {}
    netdev_to_ip: dict[str, str] = {}
    current_interface: str | None = None
    for line in lines:
        rdma_match = re.search(r"\blink\s+(\S+?)/\d+.*\bnetdev\s+(\S+)", line)
        if rdma_match:
            hca_to_netdev[rdma_match.group(1)] = rdma_match.group(2)
        interface_match = re.match(r"^\d+:\s+([^:]+):\s+(.*)$", line)
        if interface_match:
            current_interface = interface_match.group(1).split("@", 1)[0]
            master_match = re.search(r"\bmaster\s+(\S+)", interface_match.group(2))
            if master_match:
                netdev_to_master[current_interface] = master_match.group(1)
            continue
        ip_match = re.search(r"\binet\s+(\d+\.\d+\.\d+\.\d+)/\d+", line)
        if current_interface and ip_match:
            netdev_to_ip[current_interface] = ip_match.group(1)
    result: dict[str, tuple[str, str]] = {}
    for hca, netdev in hca_to_netdev.items():
        master = netdev_to_master.get(netdev, netdev)
        address = netdev_to_ip.get(master) or netdev_to_ip.get(netdev)
        if address is None:
            raise ValueError(f"could not find an IPv4 address for {hca} ({master})")
        result[hca] = (master, address)
    if not result:
        raise ValueError(f"no RDMA HCA/netdev mappings found in IP address file {path}")
    return result


def _physical_gpu_indices() -> tuple[int, ...]:
    """Read the physical GPU order selected by the launcher."""
    value = os.environ.get("BENCHMARK_GPU_INDICES", "0,1,2,3")
    try:
        indices = tuple(int(item) for item in value.split(",") if item != "")
    except ValueError as error:
        raise RuntimeError(f"invalid BENCHMARK_GPU_INDICES={value!r}") from error
    if len(indices) != EXPECTED_WORLD_SIZE or len(set(indices)) != EXPECTED_WORLD_SIZE:
        raise RuntimeError("BENCHMARK_GPU_INDICES must contain four unique GPU indices")
    return indices


def _build_endpoints(topology_file: Path, ip_addr_file: Path) -> dict[int, TopologyEndpoint]:
    """Combine topology and live IP information into per-GPU RDMA endpoints."""
    topology = _parse_topology_file(topology_file)
    addresses = _parse_ip_addr_file(ip_addr_file)
    endpoints: dict[int, TopologyEndpoint] = {}
    for gpu_index in _physical_gpu_indices():
        try:
            hca, relation, numa_node = topology[gpu_index]
        except KeyError as error:
            raise ValueError(f"topology file has no row for GPU{gpu_index}") from error
        try:
            netdev, ip_address = addresses[hca]
        except KeyError as error:
            raise ValueError(f"IP address file has no address for {hca}") from error
        endpoints[gpu_index] = TopologyEndpoint(
            gpu_index=gpu_index,
            hca=hca,
            netdev=netdev,
            ip_address=ip_address,
            numa_node=numa_node,
            topology_relation=relation,
        )
    return endpoints


def _pair_for_gpu(gpu_index: int) -> tuple[int, int]:
    """Return the fixed sender/receiver pair containing ``gpu_index``."""
    for pair in GPU_PAIRS:
        if gpu_index in pair:
            return pair
    raise ValueError(f"GPU{gpu_index} is not in the supported pair mapping {GPU_PAIRS}")


def _build_ib_write_bw_command(
    *,
    executable: str,
    endpoint: TopologyEndpoint,
    peer_endpoint: TopologyEndpoint,
    is_sender: bool,
    port: int,
    size_mib: int,
    qp: int,
    tx_depth: int,
    report_interval: int,
) -> list[str]:
    """Build a fixed-size, persistent unidirectional ``ib_write_bw`` command."""
    command = [
        executable,
        "-R",
        "-d",
        endpoint.hca,
        "-i",
        "1",
        "-s",
        str(size_mib * BYTES_PER_MIB),
        "-q",
        str(qp),
        "-t",
        str(tx_depth),
        "-D",
        str(report_interval),
        "--run_infinitely",
        "--report_gbits",
        "--bind_source_ip",
        endpoint.ip_address,
        "--numa_node",
        str(endpoint.numa_node),
        "-p",
        str(port),
    ]
    if is_sender:
        command.append(peer_endpoint.ip_address)
    return command


def _start_rdma_process(
    *,
    command: Sequence[str],
    log_path: Path,
) -> RdmaProcess:
    """Start one perftest process in its own process group."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("wb")
    try:
        process = subprocess.Popen(
            list(command),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except BaseException:
        log_file.close()
        raise
    log_file.close()
    return RdmaProcess(process=process, log_path=log_path)


def _stop_rdma_process(rdma_process: RdmaProcess) -> None:
    """Stop perftest and its process group after C2C events have completed."""
    process = rdma_process.process
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)


def _wait_for_rdma_startup(
    *,
    rdma_process: RdmaProcess,
    control_group: dist.ProcessGroup,
    timeout_seconds: float,
    startup_delay_seconds: float,
) -> None:
    """Check that all rank-local perftest processes stay alive before C2C."""
    deadline = time.monotonic() + timeout_seconds
    ready_at = time.monotonic() + startup_delay_seconds
    while time.monotonic() < ready_at:
        local_failed = rdma_process.process.poll() is not None
        statuses: list[bool] = [False] * dist.get_world_size(group=control_group)
        dist.all_gather_object(statuses, local_failed, group=control_group)
        if any(statuses):
            raise RuntimeError(f"ib_write_bw exited during startup; see {rdma_process.log_path}")
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for ib_write_bw startup")
        time.sleep(0.25)
    local_failed = rdma_process.process.poll() is not None
    statuses = [False] * dist.get_world_size(group=control_group)
    dist.all_gather_object(statuses, local_failed, group=control_group)
    if any(statuses):
        raise RuntimeError(f"ib_write_bw exited during startup; see {rdma_process.log_path}")


def _measure_copy(
    *,
    direction: str,
    host_buffer: torch.Tensor,
    device_buffer: torch.Tensor,
    stream: torch.cuda.Stream,
    warmup_iterations: int,
    copy_iterations: int,
) -> CopyMeasurement:
    """Measure fixed-size asynchronous H2D or D2H copies."""
    if direction not in {"h2d", "d2h"}:
        raise ValueError(f"unsupported copy direction: {direction}")
    source, destination = (host_buffer, device_buffer) if direction == "h2d" else (device_buffer, host_buffer)
    with _nvtx_range(f"c2c_{direction}_warmup"):
        with torch.cuda.stream(stream):
            for _ in range(warmup_iterations):
                destination.copy_(source, non_blocking=True)
    stream.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    with _nvtx_range(f"c2c_{direction}_timed"):
        with torch.cuda.stream(stream):
            start_event.record(stream)
            for _ in range(copy_iterations):
                destination.copy_(source, non_blocking=True)
            end_event.record(stream)
    end_event.synchronize()
    elapsed_seconds = start_event.elapsed_time(end_event) / 1000.0
    transferred_bytes = host_buffer.numel() * host_buffer.element_size() * copy_iterations
    return CopyMeasurement(
        direction=direction,
        bandwidth_gb_s=_bandwidth_gb_s(transferred_bytes=transferred_bytes, elapsed_seconds=elapsed_seconds),
        elapsed_seconds=elapsed_seconds,
        transferred_bytes=transferred_bytes,
    )


def _summary(records: list[dict[str, Any]], *, direction: str, rdma_role: str) -> dict[str, Any]:
    """Summarize rank measurements for one direction/role group."""
    selected = [record for record in records if record["direction"] == direction and record["rdma_role"] == rdma_role]
    baseline = [record["baseline"]["bandwidth_gb_s"] for record in selected if record["baseline"]]
    concurrent = [record["with_background"]["bandwidth_gb_s"] for record in selected if record["with_background"]]
    if not baseline or not concurrent:
        raise RuntimeError(f"no measurements collected for {direction}_{rdma_role}")
    baseline_mean = fmean(baseline)
    concurrent_mean = fmean(concurrent)
    return {
        "name": f"{direction}_{rdma_role}",
        "direction": direction,
        "rdma_role": rdma_role,
        "baseline_gb_s": baseline_mean,
        "with_background_gb_s": concurrent_mean,
        "drop_percent": _drop_percent(baseline_gb_s=baseline_mean, concurrent_gb_s=concurrent_mean),
        "ranks": selected,
    }


def _serialize(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: _serialize(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    return value


def _initialize_distributed() -> tuple[dist.ProcessGroup, dist.ProcessGroup]:
    """Initialize Gloo TCP and return the control and phase groups."""
    dist.init_process_group(backend="gloo")
    control_group = dist.new_group(backend="gloo")
    phase_group = dist.new_group(backend="gloo")
    return control_group, phase_group


def _run(args: argparse.Namespace) -> None:
    _validate_environment()
    if not shutil.which(args.ib_write_bw):
        raise RuntimeError(f"could not find ib_write_bw executable: {args.ib_write_bw}")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != EXPECTED_WORLD_SIZE:
        raise RuntimeError(f"expected world size {EXPECTED_WORLD_SIZE}, got {world_size}")
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    gpu_indices = _physical_gpu_indices()
    gpu_index = gpu_indices[local_rank]
    endpoints = _build_endpoints(args.topology_file, args.ip_addr_file)
    for pair in GPU_PAIRS:
        if pair[0] not in endpoints or pair[1] not in endpoints:
            raise RuntimeError(f"missing endpoint mapping for GPU pair {pair}")
    endpoint = endpoints[gpu_index]
    pair = _pair_for_gpu(gpu_index)
    peer_gpu_index = pair[1] if gpu_index == pair[0] else pair[0]
    peer_endpoint = endpoints[peer_gpu_index]
    rdma_role = "send" if gpu_index == pair[0] else "recv"
    control_group, phase_group = _initialize_distributed()
    host_buffer = torch.empty(args.c2c_buffer_mib * BYTES_PER_MIB, dtype=torch.uint8, pin_memory=True)
    device_buffer = torch.empty_like(host_buffer, device=f"cuda:{local_rank}")
    stream = torch.cuda.Stream(device=device_buffer.device)
    all_groups: list[dict[str, Any]] = []
    log_dir = args.output.parent / "rdma_logs"
    for group_index, (direction, selected_role) in enumerate(
        (("d2h", "send"), ("d2h", "recv"), ("h2d", "send"), ("h2d", "recv"))
    ):
        is_selected = rdma_role == selected_role
        baseline = (
            _measure_copy(
                direction=direction,
                host_buffer=host_buffer,
                device_buffer=device_buffer,
                stream=stream,
                warmup_iterations=args.warmup_iterations,
                copy_iterations=args.copy_iterations,
            )
            if is_selected
            else None
        )
        dist.barrier(group=phase_group)
        port = args.rdma_port + group_index * 2 + (0 if pair == GPU_PAIRS[0] else 1)
        command = _build_ib_write_bw_command(
            executable=args.ib_write_bw,
            endpoint=endpoint,
            peer_endpoint=peer_endpoint,
            is_sender=rdma_role == "send",
            port=port,
            size_mib=args.rdma_size_mib,
            qp=args.rdma_qp,
            tx_depth=args.rdma_tx_depth,
            report_interval=args.rdma_report_interval,
        )
        log_path = log_dir / f"{direction}_{selected_role}.rank{rank}.{rdma_role}.log"
        rdma_process = _start_rdma_process(command=command, log_path=log_path)
        try:
            dist.barrier(group=phase_group)
            _wait_for_rdma_startup(
                rdma_process=rdma_process,
                control_group=control_group,
                timeout_seconds=args.rdma_ready_timeout_seconds,
                startup_delay_seconds=args.rdma_startup_delay_seconds,
            )
            dist.barrier(group=phase_group)
            with_background = (
                _measure_copy(
                    direction=direction,
                    host_buffer=host_buffer,
                    device_buffer=device_buffer,
                    stream=stream,
                    warmup_iterations=args.warmup_iterations,
                    copy_iterations=args.copy_iterations,
                )
                if is_selected
                else None
            )
            if rdma_process.process.poll() is not None:
                raise RuntimeError(f"ib_write_bw stopped before C2C completed; see {rdma_process.log_path}")
        finally:
            _stop_rdma_process(rdma_process)
            dist.barrier(group=phase_group)
        local_record = RankMeasurement(
            rank=rank,
            hostname=socket.gethostname(),
            gpu_name=torch.cuda.get_device_name(local_rank),
            gpu_pci_bus_id=_resolve_gpu_pci_bus_id(local_rank),
            gpu_index=gpu_index,
            direction=direction,
            rdma_role=rdma_role,
            peer_gpu_index=peer_gpu_index,
            endpoint=endpoint,
            peer_endpoint=peer_endpoint,
            baseline=baseline,
            with_background=with_background,
        )
        gathered: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(gathered, _serialize(local_record), group=control_group)
        if rank == 0:
            all_groups.append(
                _summary([item for item in gathered if item is not None], direction=direction, rdma_role=selected_role)
            )
        dist.barrier(group=phase_group)
    if rank == 0:
        output = {
            "benchmark": "gb200_c2c_with_cpu_rdma_p2p",
            "background_backend": "perftest/ib_write_bw",
            "control_transport": "gloo/tcp",
            "gpu_data_transport": "host-device C2C only; no NCCL or GPU-to-GPU NVLink operation",
            "rdma_config": {
                "size_mib": args.rdma_size_mib,
                "qp": args.rdma_qp,
                "tx_depth": args.rdma_tx_depth,
                "report_interval_seconds": args.rdma_report_interval,
                "run_infinitely": True,
            },
            "topology": {str(index): _serialize(value) for index, value in endpoints.items()},
            "groups": all_groups,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
        LOGGER.info("wrote four-group result to %s", args.output)
        for group in all_groups:
            LOGGER.info(
                "%s: %.3f -> %.3f GB/s (%.2f%% drop)",
                group["name"],
                group["baseline_gb_s"],
                group["with_background_gb_s"],
                group["drop_percent"],
            )


def main() -> None:
    """Parse arguments, run the benchmark, and tear down distributed state."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args()
    try:
        _run(args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
