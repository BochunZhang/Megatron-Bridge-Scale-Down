-- Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
-- Input TEMP tables created by analyse_nsys.py:
-- activities(pid, device, start, end, kind), windows(pid, device, start, end).
-- Sweep interval endpoints; concurrent kernels/copies count only once in time.
WITH clipped AS (
    SELECT a.pid, a.device, MAX(a.start, w.start) AS start,
           MIN(a.end, w.end) AS end, a.kind
    FROM activities a JOIN windows w USING (pid, device)
    WHERE a.end > w.start AND a.start < w.end
), endpoints AS (
    SELECT pid, device, start AS t, kind, 1 AS delta FROM clipped
    UNION ALL SELECT pid, device, end, kind, -1 FROM clipped
    UNION ALL SELECT pid, device, start, 'boundary', 0 FROM windows
    UNION ALL SELECT pid, device, end, 'boundary', 0 FROM windows
), changes AS (
    SELECT pid, device, t,
           SUM(delta) AS busy,
           SUM(CASE WHEN kind = 'compute' THEN delta ELSE 0 END) AS compute,
           SUM(CASE WHEN kind = 'ag' THEN delta ELSE 0 END) AS ag,
           SUM(CASE WHEN kind = 'rs' THEN delta ELSE 0 END) AS rs,
           SUM(CASE WHEN kind = 'other_comm' THEN delta ELSE 0 END) AS other_comm
    FROM endpoints GROUP BY pid, device, t
), depths AS (
    SELECT pid, device, t,
           LEAD(t) OVER timeline - t AS dt,
           SUM(busy) OVER timeline AS busy,
           SUM(compute) OVER timeline AS compute,
           SUM(ag) OVER timeline AS ag,
           SUM(rs) OVER timeline AS rs,
           SUM(other_comm) OVER timeline AS other_comm
    FROM changes
    WINDOW timeline AS (PARTITION BY pid, device ORDER BY t ROWS UNBOUNDED PRECEDING)
)
SELECT pid AS global_pid, device AS device_id,
       MIN(t) AS start_ns, MAX(t) AS end_ns, SUM(dt) AS window_ns,
       SUM(CASE WHEN busy = 0 THEN dt ELSE 0 END) AS idle_ns,
       SUM(CASE WHEN compute = 0 THEN dt ELSE 0 END) AS compute_idle_ns,
       SUM(CASE WHEN ag > 0 THEN dt ELSE 0 END) AS ag_ns,
       SUM(CASE WHEN ag > 0 AND compute > 0 THEN dt ELSE 0 END) AS ag_overlap_ns,
       SUM(CASE WHEN rs > 0 THEN dt ELSE 0 END) AS rs_ns,
       SUM(CASE WHEN rs > 0 AND compute > 0 THEN dt ELSE 0 END) AS rs_overlap_ns,
       SUM(CASE WHEN (ag > 0 OR rs > 0) AND compute = 0 THEN dt ELSE 0 END) AS ag_rs_exposed_ns,
       SUM(CASE WHEN other_comm > 0 THEN dt ELSE 0 END) AS other_comm_ns
FROM depths GROUP BY pid, device ORDER BY pid, device;
