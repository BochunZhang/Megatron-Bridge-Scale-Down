#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Query GPU idle time and communication/compute overlap in an nsys SQLite export."""

import argparse
import json
import logging
import sqlite3
from pathlib import Path


logger = logging.getLogger(__name__)
TRAIN_STEP = "megatron.bridge.training.train.train_step"
SQL_PATH = Path(__file__).with_name("gpu_timeline.sql")


def _columns(db: sqlite3.Connection, table: str) -> set[str]:
    # Table names come only from constants in this module.
    return {row[1] for row in db.execute(f'PRAGMA table_info("{table}")')}


def analyse(
    database: Path, *, window: str = "train-step", start_ns: int | None = None, end_ns: int | None = None
) -> dict:
    """Measure interval unions per captured process/GPU using SQL.

    The default window spans complete train_step NVTX ranges, including gaps
    between steps, and extends to the completion of GPU work launched inside
    them. Explicit timestamps override this window. No source tables are edited.
    """
    if (start_ns is None) != (end_ns is None) or (start_ns is not None and start_ns >= end_ns):
        raise ValueError("Specify both start-ns and end-ns, with start-ns < end-ns")
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        columns = _columns(db, "CUPTI_ACTIVITY_KIND_KERNEL")
        required = {"start", "end", "globalPid", "deviceId", "correlationId"}
        if not required <= columns or not _columns(db, "StringIds"):
            raise ValueError("Missing CUDA kernel activity/schema; export with nsys export --type=sqlite")
        name_col = next((name for name in ("demangledName", "shortName") if name in columns), None)
        if name_col is None:
            raise ValueError("No kernel name column; cannot classify communication")
        db.execute("""CREATE TEMP TABLE activities (
            pid INTEGER, device INTEGER, start INTEGER, end INTEGER, kind TEXT, correlation INTEGER)""")
        # Unknown NCCL collectives (including generic SendRecv kernels) stay communication.
        # AllGather/ReduceScatter naming does not prove the tensor is a parameter/gradient.
        db.execute(f"""
            INSERT INTO activities
            SELECT k.globalPid, k.deviceId, k.start, k.end,
                CASE
                    WHEN lower(s.value) NOT LIKE '%nccl%' THEN 'compute'
                    WHEN replace(replace(lower(s.value), '_', ''), '-', '') LIKE '%allgather%' THEN 'ag'
                    WHEN replace(replace(lower(s.value), '_', ''), '-', '') LIKE '%reducescatter%' THEN 'rs'
                    ELSE 'other_comm'
                END, k.correlationId
            FROM CUPTI_ACTIVITY_KIND_KERNEL k LEFT JOIN StringIds s ON s.id = k.{name_col}
            WHERE k.end > k.start
        """)
        copies = []
        for table in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
            if not _columns(db, table):
                continue  # nsys omits tables for event kinds not present in a capture.
            if not required <= _columns(db, table):
                raise ValueError(f"Unsupported copy/memset schema in {table}")
            copies.append(table)
            db.execute(f"""INSERT INTO activities
                SELECT globalPid, deviceId, start, end, 'copy', correlationId
                FROM {table} WHERE end > start""")
        if not db.execute("SELECT COUNT(*) FROM activities").fetchone()[0]:
            raise ValueError("Trace contains no GPU activity")
        db.execute("CREATE INDEX temp.activity_lookup ON activities(pid, device, correlation)")
        db.execute("CREATE TEMP TABLE windows(pid INTEGER, device INTEGER, start INTEGER, end INTEGER)")
        if start_ns is not None:
            db.execute("INSERT INTO windows SELECT DISTINCT pid, device, ?, ? FROM activities", (start_ns, end_ns))
            window_name = "explicit"
        elif window == "activity":
            db.execute(
                "INSERT INTO windows SELECT pid, device, MIN(start), MAX(end) FROM activities GROUP BY pid, device"
            )
            window_name = "activity-envelope (excludes idle capture edges)"
        else:
            nvtx_cols = _columns(db, "NVTX_EVENTS")
            if not {"start", "end", "globalTid"} <= nvtx_cols or not _columns(db, "CUPTI_ACTIVITY_KIND_RUNTIME"):
                raise ValueError("No NVTX/runtime events; use --window activity or explicit --start-ns/--end-ns")
            direct = "n.text" if "text" in nvtx_cols else "NULL"
            registered = "n.textId" if "textId" in nvtx_cols else "NULL"
            db.execute(
                f"""CREATE TEMP TABLE steps AS
                SELECT n.start, n.end, n.globalTid,
                       (n.globalTid & -16777216) AS pid
                FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id = {registered}
                WHERE COALESCE({direct}, s.value) = ? AND n.end > n.start""",
                (TRAIN_STEP,),
            )
            if not db.execute("SELECT COUNT(*) FROM steps").fetchone()[0]:
                raise ValueError(f"No complete {TRAIN_STEP} NVTX ranges; use --window activity or explicit timestamps")
            # Join by process AND correlation id: correlation ids repeat across ranks.
            db.execute("""INSERT INTO windows
                SELECT a.pid, a.device, MIN(s.start), MAX(s.end)
                FROM (SELECT DISTINCT pid, device FROM activities) a
                JOIN steps s ON s.pid = a.pid GROUP BY a.pid, a.device""")
            db.execute("""UPDATE windows SET end = MAX(end, COALESCE((
                SELECT MAX(a.end) FROM activities a
                JOIN CUPTI_ACTIVITY_KIND_RUNTIME r
                  ON r.correlationId = a.correlation AND (r.globalTid & -16777216) = a.pid
                JOIN steps s ON s.globalTid = r.globalTid AND r.start >= s.start AND r.start < s.end
                WHERE a.pid = windows.pid AND a.device = windows.device
            ), end))""")
            window_name = "train-step-envelope including correlated GPU tail and inter-step gaps"
        rows = [dict(row) for row in db.execute(SQL_PATH.read_text())]
        if not rows:
            raise ValueError("No GPU process matches the requested window")
        for row in rows:
            row["pid"] = (row["global_pid"] >> 24) & 0xFFFFFF
            for name in ("idle", "compute_idle", "ag_rs_exposed"):
                row[f"{name}_pct"] = 100 * row[f"{name}_ns"] / row["window_ns"]
            for name in ("ag", "rs"):
                row[f"{name}_exposed_ns"] = row[f"{name}_ns"] - row[f"{name}_overlap_ns"]
                row[f"{name}_overlap_pct"] = (
                    100 * row[f"{name}_overlap_ns"] / row[f"{name}_ns"] if row[f"{name}_ns"] else None
                )
        names = [
            dict(row)
            for row in db.execute(f"""
            SELECT s.value AS name, COUNT(*) AS launches
            FROM CUPTI_ACTIVITY_KIND_KERNEL k LEFT JOIN StringIds s ON s.id = k.{name_col}
            JOIN windows w ON w.pid = k.globalPid AND w.device = k.deviceId
            WHERE k.start < w.end AND k.end > w.start AND (s.value IS NULL OR lower(s.value) LIKE '%nccl%')
            GROUP BY s.value ORDER BY launches DESC""")
        ]
    return {
        "database": str(database.resolve()),
        "window": window_name,
        "copy_tables": copies,
        "gpus": rows,
        "communication_kernel_names": names,
    }


def main() -> None:
    """Write JSON statistics for one SQLite export without requiring CUDA."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--window", choices=("train-step", "activity"), default="train-step")
    parser.add_argument("--start-ns", type=int)
    parser.add_argument("--end-ns", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyse(args.database, window=args.window, start_ns=args.start_ns, end_ns=args.end_ns)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        logger.info("%s", payload.rstrip())
    for row in result["gpus"]:
        logger.info(
            "pid=%d gpu=%d window=%.3f ms idle=%.2f%% compute_idle=%.2f%% AG_overlap=%s RS_overlap=%s",
            row["pid"],
            row["device_id"],
            row["window_ns"] / 1e6,
            row["idle_pct"],
            row["compute_idle_pct"],
            row["ag_overlap_pct"],
            row["rs_overlap_pct"],
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
