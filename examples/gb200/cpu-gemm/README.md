# GB200 GEMM with CPU-resident operands

Run `C = A @ B` on one GB200 GPU. `A` has shape `[b,M,K]`, `B` has shape `[K,N]`, and `C` has shape `[b,M,N]`. The benchmark flattens `A` to `[b*M,K]` so each launch can reuse the same `B` across all `b*M` rows. All three cases use BF16 inputs and output, FP32 accumulation, and the same `cublasGemmEx` call.

| Case | A | B | C | GPU memory access |
|---|---|---|---|---|
| 1 | GPU | GPU | GPU | GPU-only baseline |
| 2 | GPU | CPU | GPU | Reads B directly from mapped pinned host memory |
| 3 | GPU | GPU | CPU | Writes C directly to mapped pinned host memory |

## Run

On a GB200 host with CUDA Toolkit, cuBLAS, Nsight Systems, and `numactl`, run from the repository root. The account needs permission to lock the selected GPU's clocks and sample its hardware counters. Set `NSYS_BIN` to the path of your Nsight Systems executable if the system `nsys` lacks GB200 metric sets.

```bash
bash examples/gb200/cpu-gemm/run_cpu_gemm.sh
bash examples/gb200/cpu-gemm/run_cpu_gemm.sh --m 4096 --n 2048 --k 8192 --batches 1,4,16 --gpu 0
```

The default sweep uses `M=4096`, `N=2048`, `K=8192` and `b=1,2,4,8,16`. GPU metrics are sampled at 50 kHz, or one sample every 20 µs. Use `--frequency HZ` to change the sampling rate. The launcher locks only the selected GPU to `tdp,tdp` and binds the benchmark process to its nearest Grace CPU NUMA node. Use `--numa-node N` to select a node explicitly or `--numa-node none` to use the current memory policy. For a quick run, add `--batches 1 --iters 20`. `--dry-run` shows the commands without starting the benchmark. To release the clock lock afterwards, run `nvidia-smi -i 0 --reset-gpu-clocks` with the selected GPU index.

The capture contains only the GEMM loop and its final stream synchronization. Initialization, warmup, and sampled CPU-reference checks run outside the capture. The default eight buffer sets reduce reuse across launches. Use `--slots 1` to measure repeated use of the same buffers. Nsight Systems traces catch CUDA memcpy activity inside the capture, and the benchmark reads back samples from the CPU allocation itself in case 3.

## Results

Results go under `results/gb200/cpu-gemm/<timestamp>` or `--output-dir DIR`. The default run saves `.nsys-rep` captures, validation logs, and `config.txt`. Download the entire result directory to analyze it locally with Python 3 and the matching Nsight Systems CLI. On macOS, the CLI is available inside the app bundle.

```bash
python3 examples/gb200/cpu-gemm/analyse_nsys.py /path/to/downloaded/results \
  --nsys '/Applications/NVIDIA Nsight Systems.app/Contents/MacOS/nsys'
```

The analyzer reads the GPU index, sampling frequency, and iteration count from `config.txt`, exports each validated capture to SQLite, and writes `kernels.csv` with individual kernel durations and `summary.csv` with timing and DRAM read/write and C2C RX/TX bandwidth. RX and TX are relative to the GPU. The expected traffic directions are RX in case 2 and TX in case 3. To export and analyze on the GB200 host, pass `--analyse` to the launcher.

The bundled GB200 metric set samples DRAM sectors and C2C user bytes. The DRAM counters use 32 bytes per sector. Bandwidth is in decimal GB/s and reflects device-wide sampling during the profiled kernel sequence. Keep other GPU work off the selected GPU during the run. Short kernels can share a sample interval, so use the case-level bandwidth for comparisons. If the profiler or driver cannot collect a counter, inspect `profile.log` and the metric status in `summary.csv`.

If `nsys` rejects the bundled file, the launcher saves that attempt as `profile-custom.log` and retries with the built-in `gb10x-ct` for the sweep. That set reports DRAM read/write as `% active cycles` and C2C RX/TX in GB/s. Use `--metrics-set gb10x-ct --frequency 10000` to select that path directly. If the chosen set is rejected, the launcher saves the supported metric-set list in `nsys-gpu-metrics.txt`.

References [Nsight Systems GPU Metrics](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#gpu-metrics) and [CUDA mapped host memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#mapped-memory).
