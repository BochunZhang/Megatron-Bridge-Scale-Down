# Qwen3.8 2.4T micro-batch benchmark results

Produced by `examples/gb200/micro-batch-size/qwen3.8-2.4T/run_all_benchmarks.sh`
(see that directory's `README.md` for the experiments and measurement method).

```
gemm/ gated_delta_net/ gated_attention/ mlp/ experts/   per-case JSON records
nsys/                                    .nsys-rep and .sqlite (only with --nsys)
csv/<category>.csv                       one CSV per scope, built by build_csv.sh
```
