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
-- Queries on one `nsys export --type sqlite` report of run_hbm_conflict_benchmark.sh (one case, one rank).
--
-- The capture window holds the case NVTX range "hbm-conflict/<case>" around one range per launch:
-- "ag00".."ag24" (one AllGather each), "gemm-delay" (the torch.cuda._sleep kernel) and
-- "gemm00".."gemm09" (one torch.mm each). A kernel belongs to the range in which the CUDA API call
-- that launched it started (joined on correlationId, in the same process), as in
-- ../nvls-collective/nsys_collective_kernels.sql; when the GPU ran it does not matter.
--
-- Run: sqlite3 -header -csv report.sqlite < hbm_conflict.sql

-- name: label
SELECT DISTINCT COALESCE(n.text, s.value) AS label
FROM NVTX_EVENTS AS n
LEFT JOIN StringIds AS s ON s.id = n.textId
WHERE n.end IS NOT NULL AND COALESCE(n.text, s.value) LIKE 'hbm-conflict/%';

-- name: kernels
-- Every kernel launched inside an ag<II> or gemm<JJ> range, in GPU start order. A torch.mm may
-- launch more than one kernel (e.g. a split-K reduction); they all carry the same range.
WITH
ranges AS (
    SELECT n.start, n.end, (n.globalTid >> 24) & 16777215 AS pid, COALESCE(n.text, s.value) AS name
    FROM NVTX_EVENTS AS n
    LEFT JOIN StringIds AS s ON s.id = n.textId
    WHERE n.end IS NOT NULL
),
launch_ranges AS (
    SELECT * FROM ranges WHERE name GLOB 'ag[0-9]*' OR name GLOB 'gemm[0-9]*'
)
SELECT
    lr.name AS range_name,
    s.value AS kernel,
    k.start,
    k.end,
    k.deviceId AS device_id,
    k.streamId AS stream_id,
    k.gridX AS grid_x,
    k.blockX AS block_x
FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
JOIN CUPTI_ACTIVITY_KIND_RUNTIME AS r
    ON r.correlationId = k.correlationId
    AND ((r.globalTid >> 24) & 16777215) = ((k.globalPid >> 24) & 16777215)
JOIN StringIds AS s ON s.id = k.shortName
JOIN launch_ranges AS lr
    ON lr.pid = (r.globalTid >> 24) & 16777215 AND r.start BETWEEN lr.start AND lr.end
ORDER BY k.start;

-- name: gpu_metrics
-- Every GPU metric sample of the report (nsys --gpu-metrics-devices: NVLink RX/TX, DRAM read/write,
-- SM activity, ... as integer percent of peak). Each report samples only its rank's own GPU, and
-- GPU_METRICS shares the kernels' timebase; analyse_hbm_conflict.py averages them over its windows.
-- TARGET_INFO_GPU_METRICS can list a metric once per sampled GPU.
WITH metric_names AS (
    SELECT DISTINCT metricId, metricName FROM TARGET_INFO_GPU_METRICS
)
SELECT g.timestamp, m.metricName AS metric, g.value
FROM GPU_METRICS AS g
JOIN metric_names AS m ON m.metricId = g.metricId
ORDER BY g.timestamp;
