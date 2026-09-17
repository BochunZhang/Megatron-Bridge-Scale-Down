# GPU memory tracing

`check_gpu.py` polls every GPU returned by `nvidia-smi` and writes one JSON
trace per GPU when monitoring stops. `run_with_gpu_memory_trace.sh` ties the
monitor lifecycle to a training command, similar to a profiler wrapper.

## Trace a training command

```bash
scripts/nvidia-smi/run_with_gpu_memory_trace.sh \
    --interval 1 \
    --output-dir artifacts/gpu-memory/run-001 \
    -- \
    uv run python scripts/training/run_recipe.py <recipe arguments>
```

The wrapper starts the training command, starts the monitor, and waits for the
training command. When training exits, it sends `SIGTERM` to the monitor and
waits for all JSON files to be finalized. The wrapper returns the training
command's exit status. `SIGINT`, `SIGTERM`, and `SIGHUP` terminate both
processes so interrupted runs also finalize any samples already collected. If
the monitor cannot take its first sample, the wrapper stops the training
command and fails instead of silently running without a trace.

If `--output-dir` is omitted, traces are written under a UTC timestamped
directory such as `gpu_memory_traces/20260916T150000Z/`.

## Attach to an existing process

```bash
python3 scripts/nvidia-smi/check_gpu.py \
    --pid "$TRAINING_PID" \
    --interval 1 \
    --output-dir artifacts/gpu-memory/run-001
```

The standalone monitor also supports `--duration SECONDS` and
`--stop-file PATH`. Pressing Ctrl-C finalizes the trace. `--ready-file PATH`
is available for launchers that need confirmation of the first successful
sample; the Bash wrapper manages this automatically.

## Output

Each physical GPU produces `gpu_<index>.json`. The file contains GPU metadata,
the trace start and end times, sample count, peak memory usage and its UTC
timestamp, plus every sample:

```json
{
  "gpu_index": 0,
  "gpu_uuid": "GPU-...",
  "memory_total_mib": 81559,
  "max_memory_used_mib": 74210,
  "max_memory_used_percent": 90.985,
  "max_memory_at": "2026-09-16T15:00:04.123456+00:00",
  "samples": [
    {
      "timestamp": "2026-09-16T15:00:01.123456+00:00",
      "elapsed_seconds": 0.031,
      "memory_used_mib": 1024,
      "memory_total_mib": 81559,
      "memory_used_percent": 1.255
    }
  ]
}
```

`nvidia-smi` reports device-level memory, so the trace includes memory used by
all processes on each GPU. It does not attribute memory to an individual
training process. Sampling is local to the machine where the monitor runs. For
a multi-node job, launch one monitor per node and use distinct output paths,
for example `artifacts/gpu-memory/run-001/$HOSTNAME`.

Short memory spikes between samples can be missed; lower `--interval` when
needed, accounting for the additional `nvidia-smi` overhead. Normal completion,
training failure, Ctrl-C, `SIGTERM`, and `SIGHUP` finalize traces. `SIGKILL` and
node failure cannot run the finalization handler.
