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
-- All-gather / reduce-scatter kernel cost from one `nsys export --type sqlite` report.
--
-- Every report written by `run_nvls_collective_benchmark.sh --nsys` holds one case of one rank:
-- an outer NVTX range named like "symm/single4/ag+rs/256MiB" around NVTX ranges "trial0".."trialN".
-- Warmup, buffer allocation and window registration happen outside the capture window.
--
-- A kernel belongs to a trial when the CUDA runtime call that launched it (joined on
-- correlationId, in the same process) started inside that trial's NVTX range. This is the
-- launch-time attribution used by examples/gb200/benchmark/mbench/build_csv.py; it does not
-- depend on when the GPU actually ran the kernel.
--
-- Kernels are classified by collective only. NCCL 2.28.9 specializes one classic kernel per
-- collective (best_kernel() in src/device/generate.py), so an NVLS all-gather still appears as
-- ncclDevKernel_AllGather_RING_LL and the algorithm must be read from the NCCL TUNING log.
-- Symmetric kernels (ncclSymkDevKernel_AllGather_STMC, ...) do carry the variant in their name.
-- The torch.cuda._sleep launch-delay kernel is not an NCCL kernel and is excluded.
--
-- Run: sqlite3 -header -csv report.sqlite < nsys_collective_kernels.sql
-- The first result set is per (trial, op, kernel); the second is per trial AG/RS overlap.

-- name: kernels
WITH
ranges AS (
    SELECT n.start, n.end, (n.globalTid >> 24) & 16777215 AS pid, COALESCE(n.text, s.value) AS name
    FROM NVTX_EVENTS AS n
    LEFT JOIN StringIds AS s ON s.id = n.textId
    WHERE n.end IS NOT NULL
),
cases AS (
    SELECT start, end, pid, name AS label FROM ranges WHERE name LIKE '%/%MiB'
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
        r.start AS launch_start,
        (r.globalTid >> 24) & 16777215 AS pid,
        s.value AS kernel,
        CASE WHEN s.value LIKE '%AllGather%' THEN 'ag' ELSE 'rs' END AS op
    FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
    JOIN CUPTI_ACTIVITY_KIND_RUNTIME AS r
        ON r.correlationId = k.correlationId
        AND ((r.globalTid >> 24) & 16777215) = ((k.globalPid >> 24) & 16777215)
    JOIN StringIds AS s ON s.id = k.shortName
    WHERE s.value LIKE 'nccl%' AND (s.value LIKE '%AllGather%' OR s.value LIKE '%ReduceScatter%')
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
    COUNT(DISTINCT ck.streamId) AS streams
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
    SELECT start, end, pid, name AS label FROM ranges WHERE name LIKE '%/%MiB'
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
