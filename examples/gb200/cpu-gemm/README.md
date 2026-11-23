# GB200 GEMM with CPU-resident operands

Run `C = A @ B` on one GB200 GPU. `A` has shape `[b,M,K]`, `B` has shape `[K,N]`, and `C` has shape `[b,M,N]`. The benchmark flattens `A` to `[b*M,K]` so all `b*M` rows use the same `B`. Both launchers run all three matrix placements with BF16 inputs and final output and FP32 accumulation.

| Case | A | B | C | Output-stationary reference | K-panel |
|---|---|---|---|---|---|
| 1 `all-gpu` | GPU | GPU | GPU | One GPU GEMM | Reads panels of GPU B and writes GPU C |
| 2 `cpu-weight` | GPU | CPU | GPU | GPU GEMM directly reads mapped CPU B | Copies each panel of CPU B to HBM |
| 3 `cpu-output` | GPU | GPU | CPU | GPU GEMM directly writes mapped CPU C | Final BF16 conversion directly writes mapped CPU C |

## Run

On a GB200 host with CUDA Toolkit, cuBLAS, Nsight Systems, and `numactl`, run from the repository root. The account needs permission to lock the selected GPU's clocks and sample its hardware counters. Set `NSYS_BIN` to the path of your Nsight Systems executable if the system `nsys` lacks GB200 metric sets.

```bash
bash examples/gb200/cpu-gemm/run_cpu_gemm_output_stationary.sh
bash examples/gb200/cpu-gemm/run_cpu_gemm_k_panel.sh --k-panel 1024
```

The default sweep uses `M=4096`, `N=2048`, `K=8192` and `b=1,2,4,8,16`. GPU metrics are sampled at 10 kHz, or one sample every 100 µs. Use `--frequency HZ` to change the sampling rate. Each case locks the selected GPU to `tdp,tdp` before starting Nsight Systems, including retries, and records the result in `gpu_clock_lock.log`. The process binds to the GPU's nearest Grace CPU NUMA node. Use `--numa-node N` to select a node explicitly or `--numa-node none` to use the current memory policy. For a quick run, add `--batches 1 --iters 20`. `--dry-run` shows the commands without starting the benchmark. To release the clock lock afterwards, run `nvidia-smi -i 0 --reset-gpu-clocks` with the selected GPU index.

The capture contains the timed GEMMs and final stream synchronization. Initialization, warmup, and sampled CPU-reference checks run outside the capture. The default eight buffer sets reduce reuse across launches. Use `--slots 1` to measure repeated use of the same buffers. The benchmark reads back samples from the CPU allocation itself in case 3.

The output-stationary reference issues one `cublasGemmEx` call per logical GEMM. The K-panel launcher uses a panel size of `min(1024, K)` by default and accepts `--k-panel SIZE`. Each GEMM initializes a GPU FP32 accumulator with `beta=0`, updates it with `beta=1` for later panels, and casts the result to BF16 once. Case 2 uses a reusable GPU staging buffer for B, with a shorter final panel when K is uneven. Copies and kernels run in one CUDA stream and are included in the capture. With the default dimensions and panel size, each case 2 GEMM copies eight 4 MiB panels, for 32 MiB of H2D payload.

## Results

Results go under `results/gb200/cpu-gemm/<algorithm>-<timestamp>` or `--output-dir DIR`. Each report name includes the algorithm, case, batch and effective GEMM dimensions, with `m` equal to `b*M`. For example, `output-stationary-case1-all-gpu-b1-m4096-n2048-k8192.nsys-rep` and `k-panel-case2-cpu-weight-b4-m16384-n2048-k8192-panel1024.nsys-rep`. Each case directory includes its profiler and clock logs. Download the entire result directory, including `config.txt`, to analyze it locally with Python 3 and the matching Nsight Systems CLI. On macOS, the CLI is available inside the app bundle.

```bash
python3 examples/gb200/cpu-gemm/analyse_nsys.py /path/to/downloaded/results \
  --nsys '/Applications/NVIDIA Nsight Systems.app/Contents/MacOS/nsys'
```

The analyzer reads the GPU index, sampling frequency, and iteration count from `config.txt`, exports each validated capture to SQLite, and writes `kernels.csv` with individual kernel durations and `summary.csv` with timing and DRAM read/write and C2C RX/TX bandwidth. RX and TX are relative to the GPU. The expected traffic directions are RX in case 2 and TX in case 3. To export and analyze on the GB200 host, pass `--analyse` to the launcher.

Compare `capture_us_per_gemm` across runs to include all GPU work and gaps between operations. In K-panel case 2, this spans the first H2D copy through the final BF16 conversion. `kernel_us_per_gemm` sums kernel durations, including the conversion. The analyzer checks the H2D count and payload for K-panel case 2, and flags CUDA memcpy in the other cases. It reports exact copy payload in `h2d_bytes_per_gemm` and copy duration in `h2d_us_per_gemm`. The `c2c_rx_bytes_per_gemm` and `c2c_tx_bytes_per_gemm` estimates integrate sampled device-wide C2C rates over the capture window.

The bundled GB200 metric set samples DRAM sectors and C2C user bytes. The DRAM counters use 32 bytes per sector. Bandwidth is in decimal GB/s and reflects device-wide sampling during the profiled kernel sequence. Keep other GPU work off the selected GPU during the run. Short kernels can share a sample interval, so use the case-level bandwidth for comparisons. If the profiler or driver cannot collect a counter, inspect `profile.log` and the metric status in `summary.csv`.

If `nsys` rejects the bundled file, the launcher saves that attempt as `profile-custom.log` and retries with the built-in `gb10x-ct` for the sweep. That set reports DRAM read/write as `% active cycles` and C2C RX/TX in GB/s. Use `--metrics-set gb10x-ct --frequency 10000` to select that path directly. If the chosen set is rejected, the launcher saves the supported metric-set list in `nsys-gpu-metrics.txt`.

References [Nsight Systems GPU Metrics](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#gpu-metrics) and [CUDA mapped host memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#mapped-memory).
