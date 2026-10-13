# Qwen3.8 2.4T micro-batch benchmark results

Produced by `examples/gb200/micro-batch-size/qwen3.8-2.4T/run_all_benchmarks.sh`
(see that directory's `README.md` for the experiments and measurement method).
Each sequence length (`--seq`, default 4k) has its own directory:

```
seq<len>/                                    e.g. seq4k, seq16k, seq256k
  gemm/ gated_delta_net/ gated_attention/ mlp/ experts/   per-case JSON records
  nsys/                                    .nsys-rep and .sqlite (only with --nsys)
  csv/<category>.csv                       one CSV per scope, built by build_csv.sh
```

`gemm.csv` / `gemm-nsys.csv` at this level predate the `seq<len>/` layout; they are
seq 4k GEMM runs (the second one with `--nsys`).
