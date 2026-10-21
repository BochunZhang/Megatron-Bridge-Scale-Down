-- Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.
--
-- All-gather / reduce-scatter / all-reduce kernel cost from one `nsys export --type sqlite` report.
--
-- Every report written by `run_nvls_collective_benchmark.sh --nsys` holds one case of one rank:
-- an outer NVTX range named like "symm/all_gather+reduce_scatter-256MiB-bf16-single-n4" around NVTX ranges "trial0".."trialN".
-- Warmup, buffer allocation and window registration happen outside the capture window.
--
-- A kernel belongs to a trial when the CUDA runtime call that launched it (joined on
-- correlationId, in the same process) started inside that trial's NVTX range. This is the
-- launch-time attribution used by examples/gb200/benchmark/mbench/build_csv.py; it does not
-- depend on when the GPU actually ran the kernel.
--
-- The start gate is a one-element AllReduce. The torch benchmark opens each trial range only after
-- it, so in an "ar" case every AllReduce kernel inside a trial is the payload all-reduce. The C++
-- benchmark (../nvls-collective-cpp) still opens the range before its gate, so analyse_nsys.py keeps
-- only the ops of each case's own workload; a C++ "ar" case counts one extra tiny gate kernel.
--
-- Kernels are classified by collective only. NCCL 2.28.9 specializes one classic kernel per
-- collective (best_kernel() in src/device/generate.py), so an NVLS all-gather still appears as
-- ncclDevKernel_AllGather_RING_LL and the algorithm must be read from the NCCL TUNING log.
-- Symmetric kernels (ncclSymkDevKernel_AllGather_STMC, ...) do carry the variant in their name.
-- The torch.cuda._sleep launch-delay kernel is not an NCCL kernel and is excluded.
-- ncclDevKernel_SendRecv (the read / write workloads) is counted as op "p2p". gridX is the CTA
-- count of each launch (one CTA per NCCL channel): the only measurement of a send/recv's CTAs,
-- because NCCL logs none for point-to-point operations.
--
-- Run: sqlite3 -header -csv report.sqlite < nsys_collective_kernels.sql
-- The first result set is per (trial, op, kernel); the second is per trial AG/RS overlap; the third
-- is the case label analyse_nsys.py checks against the <algo>/cases/<algo>-<case>/ the report was filed in;
-- the fourth is per (trial, GPU metric) and needs a report recorded with --gpu-metrics-devices.

-- name: kernels
WITH
ranges AS (
    SELECT n.start, n.end, (n.globalTid >> 24) & 16777215 AS pid, COALESCE(n.text, s.value) AS name
    FROM NVTX_EVENTS AS n
    LEFT JOIN StringIds AS s ON s.id = n.textId
    WHERE n.end IS NOT NULL
),
cases AS (
    SELECT start, end, pid, name AS label FROM ranges WHERE name LIKE '%/%MiB-%'
),
trials AS (
    SELECT c.label, t.name AS trial, t.start, t.end, t.pid
    FROM ranges AS t
    JOIN cases AS c ON c.pid = t.pid AND t.start >= c.start AND t.end <= c.end
    WHERE t.name LIKE 'trial%'
),
collective_kernels AS (
    SELECT
        k.start,
        k.end,
        k.deviceId,
        k.streamId,
        k.gridX,
        k.blockX,
        r.start AS launch_start,
        (r.globalTid >> 24) & 16777215 AS pid,
        s.value AS kernel,
        CASE
            WHEN s.value LIKE '%AllGather%' THEN 'ag'
            WHEN s.value LIKE '%ReduceScatter%' THEN 'rs'
            WHEN s.value LIKE '%SendRecv%' THEN 'p2p'
            ELSE 'ar'
        END AS op
    FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
    JOIN CUPTI_ACTIVITY_KIND_RUNTIME AS r
        ON r.correlationId = k.correlationId
        AND ((r.globalTid >> 24) & 16777215) = ((k.globalPid >> 24) & 16777215)
    JOIN StringIds AS s ON s.id = k.shortName
    WHERE s.value LIKE 'nccl%'
        AND (
            s.value LIKE '%AllGather%' OR s.value LIKE '%ReduceScatter%' OR s.value LIKE '%AllReduce%'
            OR s.value LIKE '%SendRecv%'
        )
)
SELECT
    t.label,
    t.trial,
    ck.op,
    ck.kernel,
    ck.deviceId AS device_id,
    COUNT(*) AS kernels,
    SUM(ck.end - ck.start) AS busy_ns,
    MIN(ck.end - ck.start) AS min_ns,
    MAX(ck.end - ck.start) AS max_ns,
    MIN(ck.start) AS first_start_ns,
    MAX(ck.end) AS last_end_ns,
    COUNT(DISTINCT ck.streamId) AS streams,
    MIN(ck.gridX) AS grid_min,
    MAX(ck.gridX) AS grid_max,
    MAX(ck.blockX) AS block_x
FROM trials AS t
JOIN collective_kernels AS ck ON ck.pid = t.pid AND ck.launch_start BETWEEN t.start AND t.end
GROUP BY t.label, t.trial, ck.op, ck.kernel, ck.deviceId
ORDER BY t.label, t.trial, ck.op;

-- name: overlap
-- Time AG and RS kernels of the same trial were resident on the GPU together. Kernels of one
-- collective are serialized on its own stream, so the pairwise intersection sum is exact.
WITH
ranges AS (
    SELECT n.start, n.end, (n.globalTid >> 24) & 16777215 AS pid, COALESCE(n.text, s.value) AS name
    FROM NVTX_EVENTS AS n
    LEFT JOIN StringIds AS s ON s.id = n.textId
    WHERE n.end IS NOT NULL
),
cases AS (
    SELECT start, end, pid, name AS label FROM ranges WHERE name LIKE '%/%MiB-%'
),
trials AS (
    SELECT c.label, t.name AS trial, t.start, t.end, t.pid
    FROM ranges AS t
    JOIN cases AS c ON c.pid = t.pid AND t.start >= c.start AND t.end <= c.end
    WHERE t.name LIKE 'trial%'
),
collective_kernels AS (
    SELECT
        k.start,
        k.end,
        r.start AS launch_start,
        (r.globalTid >> 24) & 16777215 AS pid,
        CASE WHEN s.value LIKE '%AllGather%' THEN 'ag' ELSE 'rs' END AS op
    FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
    JOIN CUPTI_ACTIVITY_KIND_RUNTIME AS r
        ON r.correlationId = k.correlationId
        AND ((r.globalTid >> 24) & 16777215) = ((k.globalPid >> 24) & 16777215)
    JOIN StringIds AS s ON s.id = k.shortName
    WHERE s.value LIKE 'nccl%' AND (s.value LIKE '%AllGather%' OR s.value LIKE '%ReduceScatter%')
),
trial_kernels AS (
    SELECT t.label, t.trial, ck.op, ck.start, ck.end
    FROM trials AS t
    JOIN collective_kernels AS ck ON ck.pid = t.pid AND ck.launch_start BETWEEN t.start AND t.end
)
SELECT
    a.label,
    a.trial,
    SUM(MAX(0, MIN(a.end, r.end) - MAX(a.start, r.start))) AS overlap_ns
FROM trial_kernels AS a
JOIN trial_kernels AS r ON r.label = a.label AND r.trial = a.trial AND r.op = 'rs'
WHERE a.op = 'ag'
GROUP BY a.label, a.trial
ORDER BY a.label, a.trial;

-- name: label
-- The case of one report: the outer NVTX range "<algo>/<case>" that every rank opens around
-- the timed trials, including ranks that sit the case out. Exactly one row is expected.
SELECT DISTINCT COALESCE(n.text, s.value) AS label
FROM NVTX_EVENTS AS n
LEFT JOIN StringIds AS s ON s.id = n.textId
WHERE n.end IS NOT NULL AND COALESCE(n.text, s.value) LIKE '%/%MiB-%';

-- name: gpu_metrics
-- GPU metric samples (nsys --gpu-metrics-devices: NVLink RX/TX, DRAM read/write, ... as integer
-- percent of peak) averaged over each trial's GPU window: first start to last end of the
-- collective kernels launched in the trial. Each report samples only its rank's own GPU, and
-- GPU_METRICS shares the kernels' timebase. The window covers every collective kernel of the
-- trial, send/recv included; in a C++ report it therefore also starts at the start gate's tiny AllReduce.
WITH
ranges AS (
    SELECT n.start, n.end, (n.globalTid >> 24) & 16777215 AS pid, COALESCE(n.text, s.value) AS name
    FROM NVTX_EVENTS AS n
    LEFT JOIN StringIds AS s ON s.id = n.textId
    WHERE n.end IS NOT NULL
),
cases AS (
    SELECT start, end, pid, name AS label FROM ranges WHERE name LIKE '%/%MiB-%'
),
trials AS (
    SELECT c.label, t.name AS trial, t.start, t.end, t.pid
    FROM ranges AS t
    JOIN cases AS c ON c.pid = t.pid AND t.start >= c.start AND t.end <= c.end
    WHERE t.name LIKE 'trial%'
),
collective_kernels AS (
    SELECT k.start, k.end, r.start AS launch_start, (r.globalTid >> 24) & 16777215 AS pid
    FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
    JOIN CUPTI_ACTIVITY_KIND_RUNTIME AS r
        ON r.correlationId = k.correlationId
        AND ((r.globalTid >> 24) & 16777215) = ((k.globalPid >> 24) & 16777215)
    JOIN StringIds AS s ON s.id = k.shortName
    WHERE s.value LIKE 'nccl%'
        AND (
            s.value LIKE '%AllGather%' OR s.value LIKE '%ReduceScatter%' OR s.value LIKE '%AllReduce%'
            OR s.value LIKE '%SendRecv%'
        )
),
windows AS (
    SELECT t.label, t.trial, MIN(ck.start) AS start, MAX(ck.end) AS end
    FROM trials AS t
    JOIN collective_kernels AS ck ON ck.pid = t.pid AND ck.launch_start BETWEEN t.start AND t.end
    GROUP BY t.label, t.trial
),
-- TARGET_INFO_GPU_METRICS can list a metric once per sampled GPU.
metric_names AS (
    SELECT DISTINCT metricId, metricName FROM TARGET_INFO_GPU_METRICS
)
SELECT
    w.label,
    w.trial,
    m.metricName AS metric,
    COUNT(*) AS samples,
    AVG(g.value) AS mean,
    MAX(g.value) AS peak,
    w.end - w.start AS window_ns
FROM windows AS w
JOIN GPU_METRICS AS g ON g.timestamp BETWEEN w.start AND w.end
JOIN metric_names AS m ON m.metricId = g.metricId
GROUP BY w.label, w.trial, m.metricName
ORDER BY w.label, w.trial, m.metricName;
