# Qwen3.5-27B layer-by-layer capacity test

Run on one GB200 node with four GPUs available to the same CUDA/NCCL environment. The sweep uses four-way full sharding with TP, PP and CP each set to 1. It keeps the Qwen3.5-27B dense text-model widths and varies decoder depth. The full model has 64 decoder layers. Training uses BF16, randomly initialized weights, and synthetic token batches.

## Prepare the training environment

Check out and update the `layer-by-layer` branch in the Megatron-LM source checkout already installed in the Bridge environment. From the Bridge root, this command shows which Megatron Core the environment resolves.

```bash
uv run --no-sync python -c 'from importlib.util import find_spec; print(find_spec("megatron.core").origin)'
```

From the Bridge root, update the scripts on `v0.6.0-scale-down`.

```bash
git fetch origin
git switch v0.6.0-scale-down
git pull --ff-only origin v0.6.0-scale-down
```

The launcher uses Bridge's configured CUDA environment through `uv run --no-sync`. Bridge's layer-by-layer step uses the workload builder and sharded trainer from the installed Megatron-LM checkout. Set `PYTHON` to use an explicit interpreter with that same environment configuration.

## 32-layer all-GPU baseline

Run the baseline from the Bridge repository root. It uses the Qwen3.5-27B text attention mix, 32 decoder layers, sequence length 4096, MBS 1, GBS 16, four micro-batches per rank, and 10 optimizer steps.

```bash
study=examples/scale-down/02-layer-by-layer/01-baseline
bash "$study/01-param-gpu-act-gpu-opt-gpu.sh" --dry-run
bash "$study/01-param-gpu-act-gpu-opt-gpu.sh"
```

This baseline keeps parameter shards, layer-boundary activations, and optimizer state on the GPUs and executes AdamW on the GPUs. The other numbered scripts select CPU parameters and activations with NVMe optimizer state, or NVMe for all three. Set `--nvme DIR` or `NVME_PATH` if those cases use a node-local NVMe mount other than `/tmp`.

The numbered scripts select the experiment settings and call `run_pretrain_layer_by_layer.sh`. That Bash launcher wraps four torchrun workers in Nsight Systems and runs Bridge's `scripts/training/run_recipe.py --recipe qwen35_text_27b_pretrain_4gpu_gb200_bf16_layer_by_layer_config --mode pretrain --dataset mock` on each rank. Bridge runs its pretraining loop and iteration logger, using the Megatron-LM layer-by-layer trainer for each optimizer step. The launcher uses the same NUMA selection order as the FSDP3 study.

The cases use Nsight Systems **2026.6.1** to capture the CUDA and NVTX execution timeline with GPU metrics sampling disabled. The default window is iteration 8 (step 7, zero-based). With `--train-iters 3`, capture covers iteration 3. Forward NVTX ranges label the iteration, layer, and micro-batch. The launcher checks the version at `NsightSystems-2026.6.1/bin/nsys` under the Bridge repository root, or at the executable set through `NSYS_BIN`. Use `--profile none` for an unprofiled run. [Nsight Systems CLI reference](https://docs.nvidia.com/nsight-systems/UserGuide/)

On each node the launcher checks IPv4 addresses on up interfaces. It honors valid `GLOO_SOCKET_IFNAME` and `NCCL_SOCKET_IFNAME` values. Automatic selection tries `eth0`, then `bond1`, then an IPv4 default-route interface. NCCL uses an exact interface name. The selected interfaces and Nsight version are recorded in `run_info.txt`.

The all-GPU baseline writes to `results/02-layer-by-layer/01-baseline/01-param-gpu-act-gpu-opt-gpu/model_.../node0/` in Bridge. The Nsight report is named `qwen35_27b-layer_by_layer-layers_32-attn_hybrid-bf16-cg_off-tp_1-cp_1-pp_1-dp_4-mbs_1-gbs_16-seq_4k-iters_10-param_gpu-act_gpu-opt_gpu.nsys-rep`. `launch.json` records its filename. The same directory contains `command.txt`, `config.yaml`, `run_info.txt`, `train.log`, `summary.csv`, `tensorboard/`, and the source revisions and hashes. Use `--print-result-dir` to resolve the path or `--output-dir` for a fresh repeat.

Bridge prints iteration time and model TFLOP/s per GPU to the terminal and `train.log` as the numbered cases run. TensorBoard records `throughput/tflops/device` and, from the second iteration, `throughput/tokens_per_sec`. Use `--profile none` for throughput measurements without Nsight Systems capture overhead.

The all-GPU baseline passes with `status=0`, a nonempty report when profiling, and zero activation D2H and H2D copies per rank.

## Capacity sweep

```bash
bash examples/scale-down/02-layer-by-layer/run_sweep.sh \
  --nvme-dir /path/to/local/nvme --layers 16 32 64
```

Defaults are sequence length 4096, micro-batch size 1, four micro-batches per rank per step, three optimizer steps, and `nvme` for each tier. Use an already configured Megatron CUDA environment with Transformer Engine and the Gated DeltaNet kernels. Set `PYTHON` to the interpreter in that environment if needed.

Select `--parameter-tier`, `--activation-tier`, and `--optimizer-tier` independently.

| Tier | Parameter shards | Layer-boundary activations | AdamW state and execution |
| --- | --- | --- | --- |
| `gpu` | GPU memory | GPU memory | GPU memory and GPU AdamW |
| `cpu` | Host RAM | Pinned host RAM | Host RAM and CPU AdamW |
| `nvme` | Local NVMe | Local NVMe via pinned host buffers | Local NVMe, staged to host RAM for CPU AdamW |

Each decoder layer processes all micro-batches before advancing. Two GPU weight slots alternate across layers, and all micro-batches of a layer share the same gathered weights. Backward rematerializes each layer and reduce-scatters its FP32 accumulated gradient. The shared communicator issues the gradient reduce-scatter before the next weight all-gather. The prototype uses fixed learning rate, with gradient scaling and gradient clipping disabled.

When the `megatron_nvme_aio` extension from the Megatron-LM checkout's `src/nvme/deepspeed` is installed, the trainer uses direct I/O. Otherwise it uses buffered I/O, with file sync and cache eviction advice where supported. Checkpoint and rollback support are outside the prototype's training entry point.

For a quick smoke test, add `--tiny --layers 4 --seq-length 128 --microbatches 2 --steps 2`. Add `--dry-run` to inspect the commands. To find the limit after 64 layers passes, try more depths with `--layers 64 80 96`, then narrow the interval around a capacity failure. The NVMe filesystem must have room for all four ranks' files. At full depth the FP32 master weights and two Adam moment tensors alone require roughly 324 GB in total. Staging and temporary files add to that footprint.

To verify CPU activation offload, run the smoke test with `--activation-tier cpu`. The log's `LAYER_BY_LAYER_RESULT` records `activation_d2h_completed_by_rank` and `activation_h2d_completed_by_rank` after CUDA synchronization. Expect `[20, 20, 20, 20]` and `[40, 40, 40, 40]` for this smoke test. In general, each rank offloads `(layers + 1) * microbatches * steps` boundary tensors and reloads each twice. The `nvme` activation tier has the same copy counts and additionally reads and writes activation files. The `gpu` activation tier reports zero for both counts.

Each sweep creates a new result directory under `./results`, or at the unused path passed as `--results-dir`. It retains the Bridge revision in `revision.txt`, the Megatron-LM revision in `megatron_lm_revision.txt`, and SHA-256 hashes keyed by `bridge/<file>` and `megatron-lm/<file>` in `source_sha256.json`. It also saves commands, per-depth training logs and TensorBoard directories, and `summary.csv` with the status, final loss, peak GPU memory, and activation copy counts per rank. Each case uses a fresh scratch directory on NVMe. A pass requires every rank to finish the requested steps with finite losses, finite gradient shards, and the expected activation copy counts. CUDA allocation errors, explicit host allocation errors, and NVMe space or quota errors get separate statuses. Other errors require inspecting the corresponding log. The reported depth is the largest case observed to pass in the sweep.
