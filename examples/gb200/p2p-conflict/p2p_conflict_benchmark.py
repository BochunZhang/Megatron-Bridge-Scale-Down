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
"""Does NVLink traffic into G0 slow down a P2P flow out of G0? Four NCCL P2P scenarios on one GB200 tray.

``run_p2p_conflict_benchmark.sh`` starts this once with four ranks; rank ``r`` plays role ``G<r>``
on the GPU given by ``--gpus``. The test flow is ``G0 -> G3``; the scenarios add background flows
into G0 (each flow sends ``--size-mib`` once per trial):

* ``alone``: ``G0 -> G3`` only (the reference)
* ``bg1``:   + ``G1 -> G0``
* ``bg2``:   + ``G1 -> G0``, ``G2 -> G0``
* ``bg3``:   + ``G1 -> G0``, ``G2 -> G0``, ``G3 -> G0`` (G0 <-> G3 is then bidirectional)

Every flow has its own two-rank NCCL communicator and therefore its own NCCL stream and its own
``ncclDevKernel_SendRecv`` launches. Concurrent flows on one GPU are separate kernels; they never
share CTAs, which NCCL would do (warps split between send and recv) if they were one group call.

A trial puts a GPU sleep in front of a four-rank all-reduce (the start gate), then enqueues every
flow of the scenario on its own stream behind the gate. The sleep keeps everything queued until
the host is done, so all flows on all GPUs start within microseconds of the gate. Each flow's time
is measured from the gate to the end of its transfer, on both ends of the flow. Because all flows
start together, a background flow covers the whole test flow exactly when it ends later.

With ``--nsys`` the launcher wraps torchrun in a single ``nsys profile`` session, so the one report
holds all four processes and the GPU metrics of all four GPUs. All trials sit inside one
``cudaProfilerStart``/``Stop`` window opened and closed behind barriers on every rank. The NVTX
ranges name every launch (``<scenario>/trial<k>/flow <src>><dst> <send|recv>`` and
``<scenario>/trial<k>/gate``), which is how ``analyse_nsys.py`` maps kernels back to flows.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import re
import socket
from pathlib import Path
from statistics import median

import torch
import torch.distributed as dist
from torch.cuda import nvtx


LOGGER = logging.getLogger(__name__)

WORLD_SIZE = 4
Flow = tuple[int, int]
TEST_FLOW: Flow = (0, 3)
# Test flow first; scenarios only ever add background flows into G0.
SCENARIOS: dict[str, tuple[Flow, ...]] = {
    "alone": (TEST_FLOW,),
    "bg1": (TEST_FLOW, (1, 0)),
    "bg2": (TEST_FLOW, (1, 0), (2, 0)),
    "bg3": (TEST_FLOW, (1, 0), (2, 0), (3, 0)),
}
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
BYTES_PER_MIB = 1024 * 1024
P2P_MODES = ("push", "pull")
# NCCL decides read vs write per channel when it connects (p2pGetInfo in transport/p2p.cc); on
# Blackwell the topology default is write, the launcher pins it either way.
P2P_READ_ENV = "NCCL_P2P_READ_ENABLE"
# ``Channel 00/1 : 0[0] -> 1[1] via P2P/CUMEM``: one line per send channel connected (p2pSendSetup).
P2P_CONNECT_LINE = re.compile(r"Channel \d+/1 : (\d+)\[\d+\] -> (\d+)\[\d+\] (?:\[send\] )?via (\S+)")
SLEEP_CALIBRATION_CYCLES = 10_000_000


def flow_label(flow: Flow) -> str:
    """``"0>3"`` for G0 -> G3."""
    return f"{flow[0]}>{flow[1]}"


def payload(flow: Flow) -> float:
    """What the sender of ``flow`` sends; distinct per source, so a receiver can tell flows apart (exact in bf16)."""
    return float(flow[0] + 1)


def flow_range(scenario: str, trial: int, flow: Flow, side: str) -> str:
    """NVTX range around the launch of one end of a flow (parsed back by ``analyse_nsys.py``)."""
    return f"{scenario}/trial{trial}/flow {flow_label(flow)} {side}"


def gate_range(scenario: str, trial: int) -> str:
    """NVTX range around the launch of a trial's start gate."""
    return f"{scenario}/trial{trial}/gate"


def my_ends(flows: tuple[Flow, ...], rank: int) -> list[tuple[Flow, str]]:
    """The flow ends ``rank`` runs in a scenario: ``(flow, "send" | "recv")``, in scenario order."""
    ends = []
    for flow in flows:
        if flow[0] == rank:
            ends.append((flow, "send"))
        if flow[1] == rank:
            ends.append((flow, "recv"))
    return ends


def all_flows(scenarios: list[str]) -> list[Flow]:
    """Distinct flows of ``scenarios`` in first-use order; every rank creates their groups in this order."""
    flows: list[Flow] = []
    for name in scenarios:
        for flow in SCENARIOS[name]:
            if flow not in flows:
                flows.append(flow)
    return flows


def nccl_debug_path(template: str, *, hostname: str, pid: int) -> Path:
    """Expand ``NCCL_DEBUG_FILE`` like NCCL's ``debug.cc``: ``%h`` short host name, ``%p`` pid, ``%%`` a ``%``."""
    out: list[str] = []
    chars = iter(template)
    for char in chars:
        if char != "%":
            out.append(char)
            continue
        code = next(chars, "")
        if code == "h":
            out.append(hostname.split(".", 1)[0])
        elif code == "p":
            out.append(str(pid))
        elif code == "%":
            out.append("%")
        else:
            out.append("%" + code)
    return Path("".join(out))


class NcclLog:
    """Read this process's NCCL debug file incrementally, so each flow's connect sees only its own lines."""

    def __init__(self) -> None:
        template = os.environ.get("NCCL_DEBUG_FILE", "")
        self.path = nccl_debug_path(template, hostname=socket.gethostname(), pid=os.getpid()) if template else None
        self.offset = 0

    def consume(self) -> list[str]:
        """Return the complete lines appended since the previous call."""
        if self.path is None or not self.path.exists():
            return []
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            data = handle.read()
        cut = data.rfind(b"\n") + 1
        self.offset += cut
        return data[:cut].decode("utf-8", errors="replace").splitlines()


def check_connect_lines(lines: list[str], *, mode: str) -> tuple[list[str], list[str]]:
    """Transports of the send channels in ``lines`` and the violations of ``mode`` (not P2P, wrong direction)."""
    transports: dict[str, int] = {}
    for line in lines:
        match = P2P_CONNECT_LINE.search(line)
        if match:
            transports[match.group(3)] = transports.get(match.group(3), 0) + 1
    errors = []
    for transport, count in sorted(transports.items()):
        parts = transport.split("/")
        if parts[0] != "P2P":
            errors.append(f"{count} send channel(s) via {transport}, not over NVLink P2P")
        elif ("read" in parts) != (mode == "pull"):
            want = "P2P read (/read)" if mode == "pull" else "P2P write (no /read)"
            errors.append(f"--p2p-mode {mode} needs {want}, {count} send channel(s) via {transport}")
    if not transports:
        errors.append("no P2P send channel was logged")
    return sorted(transports), errors


def check_environment(mode: str, world_size: int) -> None:
    """Refuse to run unless the job has four ranks and NCCL/CUDA are set up the way the analysis assumes."""
    if world_size != WORLD_SIZE:
        raise RuntimeError(f"this benchmark runs G0..G3 as four ranks; got {world_size}")
    want = "1" if mode == "pull" else "0"
    got = os.environ.get(P2P_READ_ENV)
    if got != want:
        raise RuntimeError(f"--p2p-mode {mode} needs {P2P_READ_ENV}={want}, got {got!r}")
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        # Every process sees all GPUs, so CUDA device ids, nvidia-smi ids and nsys GPU ids agree.
        raise RuntimeError("unset CUDA_VISIBLE_DEVICES; --gpus picks the physical GPU of each role")
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("CUDA_DEVICE_ORDER=PCI_BUS_ID is required so CUDA numbers GPUs like nvidia-smi and nsys")
    if os.environ.get("NCCL_DEBUG", "").upper() != "INFO":
        raise RuntimeError("NCCL_DEBUG=INFO is required: the P2P transport is read from NCCL's INIT log")
    if "%p" not in os.environ.get("NCCL_DEBUG_FILE", ""):
        raise RuntimeError("NCCL_DEBUG_FILE must be set and contain %p so every process reads only its own log")


def _csv_ints(value: str) -> list[int]:
    return [int(x) for x in value.split(",")]


def _csv_scenarios(value: str) -> list[str]:
    names = value.split(",")
    unknown = [name for name in names if name not in SCENARIOS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown scenario(s) {unknown}; choose from {sorted(SCENARIOS)}")
    return names


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the worker options (the launcher passes all of them)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenarios", type=_csv_scenarios, default=list(SCENARIOS))
    parser.add_argument("--gpus", type=_csv_ints, default=[0, 1, 2, 3], help="Physical GPU of G0,G1,G2,G3")
    parser.add_argument("--size-mib", type=int, default=1024, help="Bytes each flow sends per trial, in MiB")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--p2p-mode", choices=P2P_MODES, default="push")
    parser.add_argument("--warmup", type=int, default=3, help="Untimed trials per scenario")
    parser.add_argument("--trials", type=int, default=10, help="Timed trials per scenario")
    parser.add_argument("--launch-delay-ms", type=float, default=20.0, help="GPU sleep in front of each start gate")
    parser.add_argument("--nsys", action="store_true", help="Open one cudaProfilerStart/Stop window around all trials")
    args = parser.parse_args(argv)
    if len(args.gpus) != WORLD_SIZE or len(set(args.gpus)) != WORLD_SIZE or min(args.gpus) < 0:
        parser.error("--gpus takes four distinct GPU ids (G0,G1,G2,G3)")
    if len(set(args.scenarios)) != len(args.scenarios):
        parser.error("--scenarios lists a scenario twice")
    if args.size_mib <= 0 or args.trials <= 0 or args.warmup < 0 or args.launch_delay_ms <= 0:
        parser.error("--size-mib, --trials and --launch-delay-ms must be positive, --warmup non-negative")
    return args


def summarize_scenario(
    scenario: str,
    per_rank: list[dict[str, object]],
    *,
    size_bytes: int,
    gpus: list[int],
    alone_gbps: float | None,
) -> dict[str, object]:
    """Rank 0's record of one scenario from every rank's ``{"ends": {"<flow> <side>": [ms...]}, "late": [...]}``.

    A flow's time is its receiver's (the last byte has landed); the sender's is kept for reference.
    ``covered_trials`` counts trials in which every background flow ended no earlier than the test
    flow, i.e. was active during all of it (all flows start at the gate).
    """
    ends: dict[str, list[float]] = {}
    for record in per_rank:
        ends.update(record["ends"])  # type: ignore[arg-type]
    flows = []
    for flow in SCENARIOS[scenario]:
        label = flow_label(flow)
        recv_ms, send_ms = ends[f"{label} recv"], ends[f"{label} send"]
        recv_median = median(recv_ms)
        flows.append(
            {
                "flow": label,
                "role": "test" if flow == TEST_FLOW else "background",
                "src_gpu": gpus[flow[0]],
                "dst_gpu": gpus[flow[1]],
                "recv_ms": recv_ms,
                "send_ms": send_ms,
                "recv_median_ms": recv_median,
                "send_median_ms": median(send_ms),
                "gbps": size_bytes / (recv_median * 1e-3) / 1e9,
            }
        )
    test_ms = ends[f"{flow_label(TEST_FLOW)} recv"]
    backgrounds = [ends[f"{flow_label(flow)} recv"] for flow in SCENARIOS[scenario] if flow != TEST_FLOW]
    covered = sum(all(bg[t] >= test_ms[t] for bg in backgrounds) for t in range(len(test_ms)))
    late = [any(trial) for trial in zip(*(record["late"] for record in per_rank))]  # type: ignore[call-overload]
    test_gbps = flows[0]["gbps"]
    return {
        "scenario": scenario,
        "size_bytes": size_bytes,
        "trials": len(test_ms),
        "flows": flows,
        "test_gbps": test_gbps,
        "test_vs_alone": None if alone_gbps is None else test_gbps / alone_gbps,  # type: ignore[operator]
        "covered_trials": covered if backgrounds else None,
        "late_launch_trials": sum(late),
    }


def render_summary(records: list[dict[str, object]], *, gpus: list[int], mode: str, size_mib: int) -> str:
    """Markdown table of the scenario records (``summary.md``)."""
    lines = [
        "# P2P conflict: G0 -> G3 test flow under background traffic into G0",
        "",
        f"GPUs G0..G3 = {', '.join(f'GPU{g}' for g in gpus)}; P2P {mode}; every flow sends {size_mib} MiB per trial.",
        "Flow time = gate to the receiver's last byte (median over trials); GB/s = 1e9 bytes/s.",
        "",
        "| scenario | test G0>G3 GB/s | vs alone | background flows GB/s | bg covered test | late launches |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for record in records:
        flows = record["flows"]
        backgrounds = ", ".join(f"{f['flow']}: {f['gbps']:.1f}" for f in flows[1:]) or "-"  # type: ignore[index]
        ratio, covered, trials = record["test_vs_alone"], record["covered_trials"], record["trials"]
        ratio_text = "-" if ratio is None else f"{ratio:.3f}"
        covered_text = "-" if covered is None else f"{covered}/{trials}"
        lines.append(
            f"| {record['scenario']} | {record['test_gbps']:.1f} | {ratio_text} | {backgrounds} | {covered_text} | "
            f"{record['late_launch_trials']}/{trials} |"
        )
    lines += [
        "",
        "bg covered test < trials: some background flow finished before the test flow, so those trials",
        "under-load G0; late launches > 0: the GPU sleep ran out before the host finished enqueueing,",
        "so the flows may not have started together (raise --launch-delay-ms).",
        "",
    ]
    return "\n".join(lines)


def _calibrate_sleep_cycles_per_ms() -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda._sleep(SLEEP_CALIBRATION_CYCLES // 10)
    start.record()
    torch.cuda._sleep(SLEEP_CALIBRATION_CYCLES)
    end.record()
    end.synchronize()
    return SLEEP_CALIBRATION_CYCLES / start.elapsed_time(end)


def _launch_end(flow: Flow, side: str, buffer: torch.Tensor, pg: dist.ProcessGroup) -> None:
    """Enqueue one end of ``flow`` on the current stream and make the current stream wait for it."""
    if side == "send":
        work = dist.isend(buffer, dst=flow[1], group=pg)
    else:
        work = dist.irecv(buffer, src=flow[0], group=pg)
    # Orders the current stream behind NCCL's stream for this communicator; the host does not block.
    work.wait()


class Runner:
    """Per-rank state: the gate, one communicator, stream and buffer per flow end, and the launch delay."""

    def __init__(self, args: argparse.Namespace, rank: int, flows: list[Flow]) -> None:
        self.rank = rank
        dtype = DTYPES[args.dtype]
        numel = args.size_mib * BYTES_PER_MIB // torch.empty((), dtype=dtype).element_size()
        # Collective over the default (Gloo) group: every rank creates every group in the same order.
        self.gate_pg = dist.new_group(list(range(WORLD_SIZE)), backend="nccl")
        self.gate = torch.zeros(1, dtype=torch.float32, device="cuda")
        # Create the gate communicator now; created lazily it would block the host inside a slept trial.
        dist.all_reduce(self.gate, group=self.gate_pg)
        torch.cuda.synchronize()
        self.pgs: dict[Flow, dist.ProcessGroup] = {}
        self.streams: dict[Flow, torch.cuda.Stream] = {}
        self.buffers: dict[tuple[Flow, str], torch.Tensor] = {}
        for flow in flows:
            pg = dist.new_group(sorted(flow), backend="nccl")
            if rank not in flow:
                continue
            self.pgs[flow] = pg
            self.streams[flow] = torch.cuda.Stream()
            for side in ("send", "recv"):
                if (side == "send") == (flow[0] == rank):
                    fill = payload(flow) if side == "send" else 0.0
                    self.buffers[(flow, side)] = torch.full((numel,), fill, dtype=dtype, device="cuda")
        self.delay_cycles = int(_calibrate_sleep_cycles_per_ms() * args.launch_delay_ms)

    def connect(self, flows: list[Flow], mode: str, nccl_log: NcclLog) -> dict[str, list[str]]:
        """Connect every flow with one small transfer and check its sender's NCCL connect lines.

        NCCL connects a pair's channels on the first send and keeps them, so this fixes the transport
        (and read vs write) for the whole job. Returns ``{flow: transports}``; raises on every rank if
        any flow is not NVLink P2P in ``mode``.
        """
        transports: dict[str, list[str]] = {}
        for flow in flows:
            nccl_log.consume()
            if flow in self.pgs:
                side = "send" if flow[0] == self.rank else "recv"
                small = torch.zeros(1024, device="cuda")
                _launch_end(flow, side, small, self.pgs[flow])
                torch.cuda.synchronize()
            verdict: object = None
            if flow[0] == self.rank:
                verdict = check_connect_lines(nccl_log.consume(), mode=mode)
            gathered: list[object] = [None] * WORLD_SIZE
            dist.all_gather_object(gathered, verdict)
            found, errors = gathered[flow[0]]  # type: ignore[misc]
            if errors:
                raise RuntimeError(f"flow G{flow_label(flow)}: " + "; ".join(errors))
            transports[flow_label(flow)] = found
        return transports

    def trial(self, scenario: str, trial: int, *, named: bool) -> tuple[dict[str, float], bool]:
        """Run one trial; return ``{"<flow> <side>": ms from the gate to that end's completion}`` and ``late``.

        ``late`` is true when the GPU sleep had already finished once every launch was enqueued, so
        the gate may have run before all flows were queued behind it.
        """
        main = torch.cuda.current_stream()
        torch.cuda._sleep(self.delay_cycles)
        slept = torch.cuda.Event()
        slept.record()
        with nvtx.range(gate_range(scenario, trial)) if named else contextlib.nullcontext():
            dist.all_reduce(self.gate, group=self.gate_pg)
        gate = torch.cuda.Event(enable_timing=True)
        gate.record()
        done: dict[str, torch.cuda.Event] = {}
        for flow, side in my_ends(SCENARIOS[scenario], self.rank):
            stream = self.streams[flow]
            stream.wait_stream(main)
            name = nvtx.range(flow_range(scenario, trial, flow, side)) if named else contextlib.nullcontext()
            with torch.cuda.stream(stream), name:
                _launch_end(flow, side, self.buffers[(flow, side)], self.pgs[flow])
                end = torch.cuda.Event(enable_timing=True)
                end.record()
            done[f"{flow_label(flow)} {side}"] = end
        late = slept.query()
        for flow in {flow for flow, _ in my_ends(SCENARIOS[scenario], self.rank)}:
            main.wait_stream(self.streams[flow])
        torch.cuda.synchronize()
        return {key: gate.elapsed_time(end) for key, end in done.items()}, late

    def received(self, scenario: str) -> bool:
        """Whether every buffer this rank receives into in ``scenario`` holds its sender's payload."""
        return all(
            bool(torch.all(self.buffers[(flow, "recv")] == payload(flow)).item())
            for flow, side in my_ends(SCENARIOS[scenario], self.rank)
            if side == "recv"
        )

    def clear(self) -> None:
        """Zero every receive buffer, so the next check only passes if the next trials deliver."""
        for (_, side), buffer in self.buffers.items():
            if side == "recv":
                buffer.zero_()
        torch.cuda.synchronize()


def main(argv: list[str] | None = None) -> None:
    """Run every scenario under torchrun with four ranks."""
    args = parse_args(argv)
    dist.init_process_group(backend="gloo")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    check_environment(args.p2p_mode, world_size)
    torch.cuda.set_device(args.gpus[rank])
    size_bytes = args.size_mib * BYTES_PER_MIB

    output = args.output_dir / "results.jsonl"
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"{output} already exists; use a fresh --output-dir")

    flows = all_flows(args.scenarios)
    nccl_log = NcclLog()
    runner = Runner(args, rank, flows)
    transports = runner.connect(flows, args.p2p_mode, nccl_log)
    LOGGER.info("rank %d = G%d on GPU%d, transports %s", rank, rank, args.gpus[rank], transports)

    # Warmup and its checks stay outside the profiler window.
    for scenario in args.scenarios:
        for trial in range(args.warmup):
            runner.trial(scenario, trial, named=False)
        ok: list[object] = [None] * WORLD_SIZE
        dist.all_gather_object(ok, args.warmup == 0 or runner.received(scenario))
        if not all(ok):
            raise RuntimeError(f"{scenario} warmup: a receiver did not get its payload ({ok})")
        runner.clear()

    if args.nsys:
        dist.barrier()
        torch.cuda.profiler.start()
        # Second barrier: whichever start call opens the session, every rank's trials come after it.
        dist.barrier()
    results: dict[str, dict[str, object]] = {}
    for scenario in args.scenarios:
        ends: dict[str, list[float]] = {}
        late: list[bool] = []
        with nvtx.range(f"scenario {scenario}: " + ", ".join(f"G{flow_label(f)}" for f in SCENARIOS[scenario])):
            for trial in range(args.trials):
                times, was_late = runner.trial(scenario, trial, named=True)
                for key, ms in times.items():
                    ends.setdefault(key, []).append(ms)
                late.append(was_late)
        # Checked per scenario: later scenarios refill the same receive buffers.
        results[scenario] = {"ends": ends, "late": late, "ok": runner.received(scenario)}
        runner.clear()
    if args.nsys:
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.profiler.stop()

    gathered: list[object] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, results)
    if rank == 0:
        per_scenario = {name: [ranks[name] for ranks in gathered] for name in args.scenarios}  # type: ignore[index]
        for scenario, per_rank in per_scenario.items():
            bad = [r for r, record in enumerate(per_rank) if not record["ok"]]
            if bad:
                raise RuntimeError(f"{scenario}: G{bad} did not receive the payload")
        alone_gbps = None
        if "alone" in per_scenario:
            alone = summarize_scenario(
                "alone", per_scenario["alone"], size_bytes=size_bytes, gpus=args.gpus, alone_gbps=None
            )
            alone_gbps = alone["test_gbps"]
        records = []
        for scenario, per_rank in per_scenario.items():
            record = summarize_scenario(
                scenario, per_rank, size_bytes=size_bytes, gpus=args.gpus, alone_gbps=alone_gbps
            )
            record["transports"] = {flow_label(f): transports[flow_label(f)] for f in SCENARIOS[scenario]}
            records.append(record)
            with output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            LOGGER.info(
                "%-5s test G0>G3 %7.1f GB/s  %s",
                scenario,
                record["test_gbps"],
                "  ".join(f"G{f['flow']} {f['gbps']:.1f}" for f in record["flows"][1:]),  # type: ignore[index]
            )
        metadata = {
            "hostname": socket.gethostname(),
            "torch": torch.__version__,
            "nccl": ".".join(map(str, torch.cuda.nccl.version())),
            "gpus": args.gpus,
            "scenarios": {name: [flow_label(f) for f in SCENARIOS[name]] for name in args.scenarios},
            "transports": transports,
            "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith(("NCCL_", "CUDA_", "TORCH_NCCL"))},
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        }
        (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        summary = render_summary(records, gpus=args.gpus, mode=args.p2p_mode, size_mib=args.size_mib)
        (args.output_dir / "summary.md").write_text(summary, encoding="utf-8")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
