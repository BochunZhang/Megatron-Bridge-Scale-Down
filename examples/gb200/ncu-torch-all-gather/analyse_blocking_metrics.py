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
"""Lay out the back-pressure metrics of run_ncu_torch_all_gather.sh --blocking-metrics per stage.

Reads ``summary.csv`` (``ncu --import <report> --page raw --csv``: header, units, one row per
profiled kernel or range) and writes ``blocking.txt`` with one column per result, stages in the
order the data moves:

* sender SM -> XBAR: ``l1tex2xbar`` stalled / active cycles, the warp stall reasons that wait on
  memory (share of active warps), the average load latency and how busy L2 -> XBAR (the load
  returns) is, peer traffic through SYSLTS
* NVLink: request / response user bytes and wire bytes (PM sampling, the only path that reports
  NVLink on NVSwitch systems so far), the plain profiler counters as a cross-check
* receiver L2: XBAR -> LTS activity, L2 writes not issued by local SMs (= arriving over NVLink,
  derived), LTC fabric (between the L2 partitions) stalled / active
* receiver HBM: read and write activity and bytes

A stage that is busy but not stalled while the stage before it is stalled is the bottleneck. Byte
counts are printed as multiples of one rank's shard S = size / nranks, so they can be checked
against what the algorithm should move (e.g. AllGather_STMC: NVLink rx 4S and tx S on 4 GPUs).
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections.abc import Callable, Sequence
from pathlib import Path


LOGGER = logging.getLogger(__name__)

SECTOR_BYTES = 32
PM = "pmsampling:"

Row = dict[str, str]


def read_raw_csv(path: Path) -> list[Row]:
    """Rows of an ncu raw-page CSV keyed by metric name (line 1 names, line 2 units)."""
    with path.open(newline="") as f:
        lines = list(csv.reader(f))
    if len(lines) < 3:
        raise ValueError(f"{path}: no profiled result (header and units only)")
    header = lines[0]
    return [dict(zip(header, values)) for values in lines[2:] if any(values)]


def value(row: Row, name: str) -> float | None:
    """The metric as a float; None if absent.

    A pmsampling: name falls back to the bare name in case the raw page drops the prefix; a
    profiler name never falls back to its PM-sampled twin, so the NVLink cross-check stays honest.
    """
    keys = (name, name.removeprefix(PM)) if name.startswith(PM) else (name,)
    for key in keys:
        text = row.get(key, "").replace(",", "").strip()
        if text:
            try:
                return float(text)
            except ValueError:
                return None
    return None


def ratio(num: float | None, den: float | None) -> float | None:
    """num / den, None when either is missing or den is 0."""
    if num is None or den is None or den == 0:
        return None
    return num / den


def build_lines(shard_bytes: float) -> list[tuple[str, Callable[[Row], float | None], str]]:
    """(label, getter, format) for every line of the report; a label without getter is a heading."""

    def get(name: str) -> Callable[[Row], float | None]:
        return lambda row: value(row, name)

    def share(num: str, den: str) -> Callable[[Row], float | None]:
        return lambda row: ratio(value(row, num), value(row, den))

    def shards(name: str, scale: float = 1.0) -> Callable[[Row], float | None]:
        return lambda row: ratio(None if (v := value(row, name)) is None else v * scale, shard_bytes)

    def l2_writes_from_outside(row: Row) -> float | None:
        total, local = value(row, "lts__t_sectors_op_write.sum"), value(row, "lts__t_sectors_srcnode_gpc_op_write.sum")
        if total is None or local is None:
            return None
        return (total - local) * SECTOR_BYTES / shard_bytes

    warps = "smsp__warps_active.sum"
    return [
        ("duration [us]", lambda row: ratio(value(row, "gpu__time_duration.sum"), 1e3), "{:.1f}"),
        ("# sender SM -> XBAR", None, ""),
        (
            "l1tex2xbar stalled / active",
            share("l1tex__m_l1tex2xbar_req_cycles_stalled.sum", "l1tex__m_l1tex2xbar_req_cycles_active.sum"),
            "{:.3f}",
        ),
        ("l1tex2xbar write bytes [S]", shards("l1tex__m_l1tex2xbar_write_bytes.sum"), "{:.3f}"),
        ("xbar2l1tex read bytes [S]", shards("l1tex__m_xbar2l1tex_read_bytes.sum"), "{:.3f}"),
        ("warps stalled lg_throttle", share("smsp__warps_issue_stalled_lg_throttle.sum", warps), "{:.1%}"),
        ("warps stalled long_scoreboard", share("smsp__warps_issue_stalled_long_scoreboard.sum", warps), "{:.1%}"),
        ("warps stalled membar", share("smsp__warps_issue_stalled_membar.sum", warps), "{:.1%}"),
        ("warps stalled barrier", share("smsp__warps_issue_stalled_barrier.sum", warps), "{:.1%}"),
        (
            "load latency generic [cycle]",
            get("sm__average_inst_latency_pipe_lsu_ops_generic_cmd_read.ratio"),
            "{:.0f}",
        ),
        ("load latency global [cycle]", get("sm__average_inst_latency_pipe_lsu_mem_lg_cmd_read.ratio"), "{:.0f}"),
        (
            "lts2xbar active % (load returns)",
            get("lts__lts2xbar_cycles_active.avg.pct_of_peak_sustained_elapsed"),
            "{:.1f}",
        ),
        (
            "SYSLTS peer writes [S]",
            shards("syslts__t_sectors_srcnode_gpc_aperture_peer_op_write.sum", SECTOR_BYTES),
            "{:.3f}",
        ),
        (
            "SYSLTS peer reads [S]",
            shards("syslts__t_sectors_srcnode_gpc_aperture_peer_op_read.sum", SECTOR_BYTES),
            "{:.3f}",
        ),
        ("# NVLink (PM sampling)", None, ""),
        ("tx request user [S]", shards(PM + "nvltx__bytes_packet_request_data_user.sum"), "{:.3f}"),
        ("tx response user [S]", shards(PM + "nvltx__bytes_packet_response_data_user.sum"), "{:.3f}"),
        ("rx request user [S]", shards(PM + "nvlrx__bytes_packet_request_data_user.sum"), "{:.3f}"),
        ("rx response user [S]", shards(PM + "nvlrx__bytes_packet_response_data_user.sum"), "{:.3f}"),
        ("tx wire [S]", shards(PM + "nvltx__bytes.sum"), "{:.3f}"),
        ("rx wire [S]", shards(PM + "nvlrx__bytes.sum"), "{:.3f}"),
        ("tx % of peak (kernel avg)", get(PM + "nvltx__bytes.avg.pct_of_peak_sustained_elapsed"), "{:.1f}"),
        ("rx % of peak (kernel avg)", get(PM + "nvlrx__bytes.avg.pct_of_peak_sustained_elapsed"), "{:.1f}"),
        ("# NVLink (profiler counters, cross-check)", None, ""),
        ("tx wire [S]", shards("nvltx__bytes.sum"), "{:.3f}"),
        ("rx wire [S]", shards("nvlrx__bytes.sum"), "{:.3f}"),
        ("tx cycles active %", get("nvltx__cycles_active.avg.pct_of_peak_sustained_elapsed"), "{:.1f}"),
        ("rx cycles active %", get("nvlrx__cycles_active.avg.pct_of_peak_sustained_elapsed"), "{:.1f}"),
        ("# receiver L2", None, ""),
        ("xbar2lts active %", get("lts__xbar2lts_cycles_active.avg.pct_of_peak_sustained_elapsed"), "{:.1f}"),
        ("L2 writes not from local SMs [S]", l2_writes_from_outside, "{:.3f}"),
        (
            "ltcfabric2lts stalled / active",
            share("lts__ltcfabric2lts_cycles_stalled.sum", "lts__ltcfabric2lts_cycles_active.sum"),
            "{:.3f}",
        ),
        ("ltcfabric writes [S]", shards("lts__t_sectors_srcunit_ltcfabric_op_write.sum", SECTOR_BYTES), "{:.3f}"),
        ("# receiver HBM", None, ""),
        ("dram read active %", get("dram__cycles_active_read.avg.pct_of_peak_sustained_elapsed"), "{:.1f}"),
        ("dram write active %", get("dram__cycles_active_write.avg.pct_of_peak_sustained_elapsed"), "{:.1f}"),
        ("dram read [S]", shards("dram__sectors_read.sum", SECTOR_BYTES), "{:.3f}"),
        ("dram write [S]", shards("dram__sectors_write.sum", SECTOR_BYTES), "{:.3f}"),
    ]


def result_name(row: Row) -> str:
    """Column heading of one result: its ID and the process it came from."""
    pid = row.get("Process ID") or row.get("PID") or "?"
    return f"#{row.get('ID', '?')} pid {pid}"


def render(rows: Sequence[Row], *, size_mib: float, nranks: int) -> str:
    """The per-stage table, one column per result."""
    shard_bytes = size_mib * 2**20 / nranks
    heads = [result_name(row) for row in rows]
    lines = build_lines(shard_bytes)
    width = max(len(label) for label, _, _ in lines) + 2
    col = max(12, *(len(h) + 2 for h in heads))
    out = [f"S = {size_mib:g} MiB / {nranks} ranks = {shard_bytes / 2**20:g} MiB per shard", ""]
    out.append("".ljust(width) + "".join(h.rjust(col) for h in heads))
    for label, getter, fmt in lines:
        if getter is None:
            out.append(label)
            continue
        cells = []
        for row in rows:
            v = getter(row)
            cells.append(("n/a" if v is None else fmt.format(v)).rjust(col))
        out.append(f"  {label}".ljust(width) + "".join(cells))
    return "\n".join(out) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    """Write blocking.txt next to summary.csv and log it."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summary", type=Path, help="summary.csv of a --blocking-metrics run")
    parser.add_argument("--size-mib", type=float, required=True, help="all-gather output size per rank in MiB")
    parser.add_argument("--nranks", type=int, required=True, help="number of ranks")
    parser.add_argument("--output", type=Path, help="(default: blocking.txt next to the summary)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    text = render(read_raw_csv(args.summary), size_mib=args.size_mib, nranks=args.nranks)
    output = args.output or args.summary.with_name("blocking.txt")
    output.write_text(text)
    LOGGER.info("%s", text)
    LOGGER.info("written to %s", output)


if __name__ == "__main__":
    main()
