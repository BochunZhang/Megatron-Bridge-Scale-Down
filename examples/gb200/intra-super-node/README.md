# GB200 intra-super-node memory and interconnect benchmarks

| Directory | Contents |
| --- | --- |
| [`gb_benchmark/`](gb_benchmark/) | GB200 port of `gh_benchmark`. The topology (CPU/GPU/HBM NUMA mapping) is discovered at runtime. It adds the CPU-relayed GPU→GPU copy test and keeps the host/device/memcpy/latency/ping-pong microbenchmarks. |
| `gh_benchmark/` | Local reference copy of the original GH200 suite (arXiv 2408.11556). It is not tracked in git because it contains prebuilt binaries, a core dump and site-specific paths. Its NUMA nodes, core IDs and `sm_90` target are hard-coded for GH200. |

The main test measures copies relayed through the CPU, without staging in CPU memory:

```text
GPU0 HBM -C2C-> CPU0 -C2C-> GPU1 HBM                 (both GPUs attached to the same Grace)
GPU0 HBM -C2C-> CPU0 -> CPU1 -C2C-> GPU2 HBM         (GPUs attached to different Graces)
```

Each CPU worker thread loads 64 B from the source HBM mapping into registers and stores it straight to the
destination HBM mapping (`ldp/stp`, `ldnp/stnp` or `memcpy`). The test does not use `cudaMemcpyPeerAsync`,
NCCL, or a GPU peer transport. Direct NVLink copies are measured separately as baselines.

```bash
cd examples/gb200/intra-super-node/gb_benchmark
make                 # on the GB200 node (CUDA 13.0+, libnuma); or ./build_cross_in_docker.sh on x86_64
unset CUDA_VISIBLE_DEVICES
./run_gb200.sh       # results in results/gb200/intra-super-node/YYMMDD-HHMMSS/
```

See [`gb_benchmark/README.md`](gb_benchmark/README.md) for the test matrix, result fields and validation
status.
