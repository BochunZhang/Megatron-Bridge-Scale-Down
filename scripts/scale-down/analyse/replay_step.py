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

"""Replay device_traces for a specific training step and show source-grouped allocations.

Scale-down variant: additionally parses user-defined phase annotations
(e.g. forward/backward record_function markers) inside each ProfilerStep and
reports per-interval memory behavior. Overlapping intervals are kept as-is
and annotated (strategy A), never clipped or merged.
"""

# Annotations are lazy so the PEP 604 `X | None` syntax below does not need to
# evaluate at import time; these scripts stay runnable on the system python3
# (3.9 on macOS) even though the repo itself targets 3.10+.
from __future__ import annotations

import argparse
import json
import logging
import os
import sys


# Repo root is three levels up from scripts/scale-down/analyse/.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
# Reuse the shared replay engine from the memory-snapshot-analysis skill
# instead of duplicating common.py.
_SKILL_SCRIPTS = os.path.join(_REPO_ROOT, "skills", "nemo-mbridge-memory-snapshot-analysis", "scripts")
sys.path.insert(0, _SKILL_SCRIPTS)
from common import (
    TRUST_EPILOG,
    ReplayResult,
    compute_baseline,
    compute_step_start_deltas,
    find_active_device,
    format_size,
    get_profiler_steps,
    get_step_annotations,
    get_step_events,
    group_by_source,
    live_set_before,
    load_snapshot,
    replay_events,
)


logger = logging.getLogger(__name__)


_PHASE_PREFIXES = ("forward_step", "backward_step", "optimizer_step")


def _parse_mbs(name: str, phase: str) -> int | None:
    """Parse the microbatch index from ``<phase>[<int>]``; None if malformed."""
    rest = name[len(phase) :]
    if not (rest.startswith("[") and rest.endswith("]")):
        return None
    try:
        return int(rest[1:-1])
    except ValueError:
        return None


def get_phase_intervals(
    annotations: list,
    start_us: int,
    end_us: int | None,
) -> tuple[dict, dict, list[int]]:
    """Parse phase annotations inside one step window, keyed by name.

    Mirrors the flat key-based design of :func:`get_profiler_steps`: within a
    single step window each annotation name is expected to appear at most
    once, so a plain dict replaces the LIFO pairing stack and a duplicate
    START raises instead of silently overwriting.

    Rules:
        * Only names starting with ``forward_step``, ``backward_step`` or
          ``optimizer_step`` are considered; all others are skipped.
        * A START is kept only when its ``time_us`` falls inside
          ``[start_us, end_us]``; an END closes the entry with the same name
          (an END with ``time_us <= 0`` is ignored, as in
          :func:`get_profiler_steps`; an END without a kept START is dropped).
        * ``forward_step[N]`` / ``backward_step[N]`` carry a microbatch index
          ``N``; after the scan the forward and backward index sets must be
          equal, otherwise the markers are inconsistent and an error is raised.

    Args:
        annotations: ``external_annotations`` list from the snapshot.
        start_us: step start timestamp (us).
        end_us: step end timestamp (us), or None for an open-ended step.

    Returns:
        Tuple of:
            phases: ``{name: {"phase": str, "mbs": int | None, "start": int,
                "end": int | None}}`` keyed by full annotation name.
            overlaps: ``{name: [{"name": other, "range": (ov_start, ov_end)}]}``
                — ``ov_end`` is None when either side is incomplete.
            start_times: sorted list of all phase start timestamps (us).

    Raises:
        ValueError: on a duplicate phase name inside the window, a malformed
            microbatch index, or a forward/backward microbatch set mismatch.
    """
    phases: dict[str, dict] = {}
    forward_mb: set[int] = set()
    backward_mb: set[int] = set()

    for ann in annotations:
        name = ann.get("name", "")
        phase = next((p for p in _PHASE_PREFIXES if name.startswith(p)), None)
        if phase is None:
            continue
        stage = ann.get("stage", "")
        time_us = ann.get("time_us", 0)

        if time_us < start_us or (end_us is not None and time_us > end_us):
            continue
        if stage == "START":
            if name in phases:
                raise ValueError(f"duplicate phase annotation '{name}' inside step window [{start_us}, {end_us}]")
            mbs = None
            if phase in ("forward_step", "backward_step"):
                mbs = _parse_mbs(name, phase)
                if mbs is None:
                    raise ValueError(f"malformed microbatch index in '{name}' (expected '{phase}[<int>]')")
                (forward_mb if phase == "forward_step" else backward_mb).add(mbs)
            phases[name] = {"phase": phase, "mbs": mbs, "start": time_us, "end": end_us}
        elif stage == "END" and time_us > 0:
            if name not in phases:
                raise ValueError(f"END annotation '{name}' without a START inside step window [{start_us}, {end_us}]")
            assert phases[name]["end"] == end_us, f"duplicate END for phase '{name}'"
            phases[name]["end"] = time_us

    if forward_mb != backward_mb:
        raise ValueError(
            "forward/backward microbatch sets differ: "
            f"forward-only={sorted(forward_mb - backward_mb)}, "
            f"backward-only={sorted(backward_mb - forward_mb)}"
        )

    start_times = sorted(entry["start"] for entry in phases.values())

    # Overlap detection on start-sorted entries: a later entry b overlaps an
    # earlier a while b starts before a ends (or a is still open).
    overlaps: dict[str, list[dict]] = {name: [] for name in phases}
    items = sorted(phases.items(), key=lambda kv: kv[1]["start"])
    for i, (na, a) in enumerate(items):
        for nb, b in items[i + 1 :]:
            if a["end"] is not None and b["start"] >= a["end"]:
                continue
            ov_end = min(a["end"], b["end"]) if a["end"] is not None and b["end"] is not None else None
            overlaps[na].append({"name": nb, "range": (b["start"], ov_end)})
            overlaps[nb].append({"name": na, "range": (b["start"], ov_end)})

    return phases, overlaps, start_times



GIB = 1024**3


def bytes_to_gigabtyes(num_bytes: int) -> float:
    """Convert bytes to GiB rounded to 3 decimals (None passes through)."""
    return round(num_bytes / GIB, 3)



def us_to_ms(time_us: float) -> float:
    """Convert microseconds to milliseconds and round to 3 decimals (None passes through)."""
    return round(time_us / 1000, 3)


def replay_one_step(
    traces: list,
    annotations: list,
    step_num: int,
    step_info: dict,
    top_n: int = 15,
    frame_depth: int = 1,
    as_json: bool = False,
    baseline_at_start: int = 0,
    step_start_delta: int = 0,
) -> dict:
    """Replay a single step and return results."""
    start = step_info["start"]
    end = step_info["end"]
    events = get_step_events(traces, start, end)
    # Seed with what was already live when the step opened, so the source table
    # accounts for memory carried into the step, not only what it allocated.
    result = replay_events(events, initial_live=live_set_before(traces, start))
    sources = group_by_source(result.peak_live_set, depth=frame_depth)
    ann_counts = get_step_annotations(annotations, start, end)

    # Phase-level breakdown: parse forward/backward/optimizer annotations
    # inside this step window, then replay each phase interval separately.
    phases, overlaps, phase_start_times = get_phase_intervals(annotations, start, end)

    # Net delta at each phase start, measured relative to the step start:
    # replay only the step-window events. The phases dict mirrors the
    # ``{key: {"start": us, "end": us | None}}`` shape returned by
    # ``get_profiler_steps`` (keyed by annotation name instead of step number,
    # plus extra "phase"/"mbs" fields that are ignored here), so
    # ``compute_step_start_deltas`` consumes it as-is.
    phase_deltas = compute_step_start_deltas(events, phases)

    phase_results: dict[str, dict] = {}
    for name, entry in sorted(phases.items(), key=lambda kv: kv[1]["start"]):
        p_start = entry["start"]
        p_end = entry["end"]
        p_events = get_step_events(traces, p_start, p_end)
        p_result = replay_events(p_events, initial_live=live_set_before(traces, p_start))
        p_sources = group_by_source(p_result.peak_live_set, depth=frame_depth)
        p_ann_counts = get_step_annotations(annotations, p_start, p_end)
        p_delta_at_start = phase_deltas.get(name, 0)
        # Expand raw overlap entries ({"name", "range": (ov_start, ov_end)}) into
        # explicit records so JSON consumers do not need to know the tuple layout.
        # ov_end is None when either side of the overlap is incomplete.
        p_overlaps = [
            {
                "name": o["name"],
                "overlap_start": o["range"][0],
                "overlap_end": o["range"][1],
                "overlap_ms": (o["range"][1] - o["range"][0]) / 1000 if o["range"][1] is not None else None,
            }
            for o in overlaps.get(name, [])
        ]
        phase_results[name] = {
            "phase": entry["phase"],
            "mbs": entry["mbs"],
            "phase.stt-step.stt[ms]": us_to_ms(p_start - start),
            "phase.end-step.stt[ms]": us_to_ms(p_end - start) if p_end is not None else None,
            "phase.duration[ms]": (p_end - p_start) / 1000 if p_end is not None else None,

            "alloc_count": p_result.alloc_count,
            "free_count": p_result.free_count,
            "unmatched_frees": p_result.unmatched_free_count,
            "unmatched_free_bytes": p_result.unmatched_free_bytes,
            "annotations": p_ann_counts,

            "total_throughput": bytes_to_gigabtyes(p_result.total_alloc_bytes),
            "phase.memory.stt-step.memory.stt[GiB]": bytes_to_gigabtyes(p_delta_at_start),
            "phase.memory.end-step.memory.stt[GiB]": bytes_to_gigabtyes(p_delta_at_start + p_result.end_delta),
            "phase.memory.peak-step.memory.stt[GiB]": bytes_to_gigabtyes(p_delta_at_start + p_result.peak_delta),
            "phase.memory.end-phase.memory.stt[GiB]": bytes_to_gigabtyes(p_result.end_delta),
            "phase.memory.peak-phase.memory.stt[GiB]": bytes_to_gigabtyes(p_result.peak_delta),
            "pahse.memory.stt[GiB]": bytes_to_gigabtyes(baseline_at_start + step_start_delta + p_delta_at_start),
            "pahse.memory.end[GiB]": bytes_to_gigabtyes(baseline_at_start + step_start_delta + p_delta_at_start + p_result.end_delta),
            "phase.memory.peak[GiB]": bytes_to_gigabtyes(baseline_at_start + step_start_delta + p_delta_at_start + p_result.peak_delta),

            "overlaps": p_overlaps,
            "top_sources_at_peak": p_sources[:top_n],
        }

    complete = end is not None
    duration_ms = (end - start) / 1000 if complete else None

    absolute_peak = baseline_at_start + step_start_delta + result.peak_delta

    step_result = {
        "step": step_num,
        "complete": complete,
        "start_ms": us_to_ms(start),
        "end_ms": us_to_ms(end) if end is not None else None,
        "duration_ms": duration_ms,
        "alloc_count": result.alloc_count,
        "free_count": result.free_count,
        "total_throughput_gib": bytes_to_gigabtyes(result.total_alloc_bytes),
        "peak_delta_gib": bytes_to_gigabtyes(result.peak_delta),
        "absolute_peak": absolute_peak,
        "end_delta": result.end_delta,
        "unmatched_frees": result.unmatched_free_count,
        "unmatched_free_bytes": result.unmatched_free_bytes,
        "top_sources_at_peak": sources[:top_n],
        "annotations": ann_counts,
        "phase_start_times": phase_start_times,
        "phase_deltas": phase_deltas,
        "phases": phase_results,
    }

    # if not as_json:
    #     print(f"\n{'=' * 70}")
    #     print(f"  Step {step_num}" + (" (incomplete)" if not complete else ""))
    #     print(f"{'=' * 70}")
    #     if duration_ms:
    #         print(f"  Duration:       {duration_ms:.1f} ms")
    #     print(f"  Allocs:         {result.alloc_count:,}")
    #     print(f"  Frees:          {result.free_count:,}")
    #     print(f"  Throughput:     {format_size(result.total_alloc_bytes)}")
    #     print(f"  Peak delta:     {format_size(result.peak_delta)}")
    #     if baseline_at_start > 0:
    #         print(f"  Absolute peak:  {format_size(absolute_peak)}")
    #     print(f"  End delta:      {format_size(result.end_delta)}")
    #     if result.unmatched_free_count > 0:
    #         print(f"  Pre-existing frees: {result.unmatched_free_count} ({format_size(result.unmatched_free_bytes)})")

    #     if ann_counts:
    #         print("\n  --- Active Annotations ---")
    #         for name, count in sorted(ann_counts.items(), key=lambda x: -x[1]):
    #             print(f"    {count:>5}x  {name}")

    #     if phase_results:
    #         print("\n  --- Phases (deltas relative to step start) ---")
    #         header = (
    #             f"    {'Phase':<24} {'Dur(ms)':>9} {'StartΔ':>11} {'PeakΔ':>11} "
    #             f"{'EndΔ':>11} {'Thruput':>11} {'Allocs':>7} {'Frees':>7} {'Unmatched':>9}"
    #         )
    #         print(header)
    #         print(f"    {'─' * 24} {'─' * 9} {'─' * 11} {'─' * 11} {'─' * 11} {'─' * 11} {'─' * 7} {'─' * 7} {'─' * 9}")
    #         for name, p in phase_results.items():
    #             dur = f"{p['duration_ms']:.1f}" if p["duration_ms"] is not None else "-"
    #             print(
    #                 f"    {name:<24} {dur:>9} {format_size(p['delta_at_start']):>11} "
    #                 f"{format_size(p['peak_delta']):>11} {format_size(p['end_delta']):>11} "
    #                 f"{format_size(p['total_throughput']):>11} "
    #                 f"{p['alloc_count']:>7,} {p['free_count']:>7,} {p['unmatched_frees']:>9,}"
    #             )
    #         for name, p in phase_results.items():
    #             for o in p["overlaps"]:
    #                 ov_end = o["overlap_end"]
    #                 dur = f"{o['overlap_ms']:.1f} ms" if o["overlap_ms"] is not None else "open"
    #                 end_str = str(ov_end) if ov_end is not None else "?"
    #                 print(f"    ! {name} overlaps {o['name']}  [{o['overlap_start']} - {end_str}] ({dur})")

    #     print(f"\n  --- Top {min(top_n, len(sources))} Sources at Peak (by size) ---")
    #     if sources:
    #         print(f"  {'#':>3}  {'Size':>12}  {'Count':>6}  Source")
    #         print(f"  {'─' * 3}  {'─' * 12}  {'─' * 6}  {'─' * 40}")
    #         for i, (key, total, count) in enumerate(sources[:top_n], 1):
    #             print(f"  {i:>3}  {format_size(total):>12}  {count:>6}  {key}")
    #     else:
    #         print("  (no live allocations at peak)")

    return step_result


def main() -> None:
    """Parse arguments and replay the requested training step(s)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Replay device_traces for a training step.",
        epilog=TRUST_EPILOG,
    )
    parser.add_argument("pickle_path", help="Path to the snapshot .pickle file")
    parser.add_argument("--step", type=int, help="Step number to replay")
    parser.add_argument("--all-steps", action="store_true", help="Replay all complete steps")
    parser.add_argument("--top", type=int, default=15, help="Number of top sources to show (default: 15)")
    parser.add_argument(
        "--frame-depth", type=int, default=1, help="Stack frame depth for source grouping (default: 1)"
    )
    parser.add_argument("--device", type=int, help="Device index (default: auto-detect)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if args.step is None and not args.all_steps:
        parser.error("Specify --step N or --all-steps")

    snapshot = load_snapshot(args.pickle_path)
    annotations = snapshot.get("external_annotations", [])
    steps = get_profiler_steps(annotations)

    if not steps:
        logger.error("no ProfilerStep annotations found.")
        sys.exit(1)

    device_idx = args.device if args.device is not None else find_active_device(snapshot)
    if device_idx is None:
        logger.error("no device traces found.")
        sys.exit(1)
    traces = snapshot["device_traces"][device_idx]

    # Compute baseline and per-step starting deltas
    baseline = compute_baseline(snapshot, device_idx)
    step_deltas = compute_step_start_deltas(traces, steps)

    if not args.json:
        print(f"\n  Baseline at profiling start: {format_size(baseline.baseline_at_start)}")

    if args.all_steps:
        step_nums = sorted(steps.keys())
    else:
        if args.step not in steps:
            available = sorted(steps.keys())
            logger.error(f"step {args.step} not found. Available: {available}")
            sys.exit(1)
        step_nums = [args.step]

    all_results = []
    for num in step_nums:
        info = steps[num]
        if not info["start"]:
            continue
        r = replay_one_step(
            traces,
            annotations,
            num,
            info,
            top_n=args.top,
            frame_depth=args.frame_depth,
            as_json=args.json,
            baseline_at_start=baseline.baseline_at_start,
            step_start_delta=step_deltas.get(num, 0),
        )
        all_results.append(r)

    if args.json:
        print(json.dumps(all_results, indent=2, default=str))
    else:
        print()


if __name__ == "__main__":
    main()
