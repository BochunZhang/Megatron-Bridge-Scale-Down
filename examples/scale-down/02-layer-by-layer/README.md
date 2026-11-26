# Qwen3.5-27B layer-by-layer capacity test

The numbered experiments and capacity sweep run Bridge pretraining with Megatron's layer-by-layer schedule at `PP=1` and four-way Megatron-FSDP parameter, gradient, and optimizer sharding. The model uses BF16 and gated full attention in every decoder layer.

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

The capacity launchers use Bridge's configured CUDA environment through `uv run --no-sync` and the native scheduler and Megatron-FSDP from the installed Megatron-LM checkout. The default is one GB200 node with four CUDA/NCCL ranks. Case 08 also supports four GB200 trays with one rank per tray. Set `PYTHON` to use an explicit interpreter with that same environment configuration.

If the environment resolves the older pinned submodule, put the updated sibling checkout first on the Python import path from the Bridge root.

```bash
export PYTHONPATH="$PWD/../megatron-lm${PYTHONPATH:+:$PYTHONPATH}"
```

The sweep defaults to PyTorch CPU AdamW for CPU and NVMe optimizer tiers. Cases 04 through 08 default to GraceAdam, which requires Megatron-LM's native Megatron-FSDP GraceAdam support and an SVE-enabled build of DeepSpeed `v0.20.1`. The launchers use `--cpu-threads` (default 32) for PyTorch intra-op and OpenMP parallelism, per rank. Overlap cases also pass the thread count to the patched native CPUAdam update. For a direct recipe launch, set `OMP_NUM_THREADS` before startup to match `layer_by_layer.cpu_threads`. When enabling `layer_by_layer.overlap_cpu_optimizer`, also set `MEGATRON_GRACEADAM_OMP_THREADS` to the same value and apply the CPUAdam patches below.

## Native schedule smoke test

Run from the Bridge root with the updated Megatron-LM checkout on the Python import path. The launcher uses the same model and training path as the numbered baseline.

```bash
bash examples/scale-down/02-layer-by-layer/run_sweep.sh \
  --tiny --layers 2 --seq-length 128 --microbatches 2 --steps 2
```

The scheduler runs all microbatches through a decoder layer before advancing, then traverses the layers in reverse during backward. It uses Megatron activation checkpointing across the microbatches of each decoder layer. Bridge runs the standard GPT data and loss path, Megatron gradient finalization, and a standard optimizer step after backward.

## 8-layer all-GPU baseline

Run the baseline from the Bridge repository root. It uses eight full-attention decoder layers, sequence length 4096, MBS 1, GBS 16, four micro-batches per rank, and 10 optimizer steps.

```bash
study=examples/scale-down/02-layer-by-layer/01-baseline
bash "$study/01_param_gpu_act_gpu_opt_gpu.sh" --dry-run
bash "$study/01_param_gpu_act_gpu_opt_gpu.sh"
```

This baseline keeps parameter shards, layer-boundary activations, and optimizer state on the GPUs and executes AdamW on the GPUs.

## 8-layer CPU parameter offload

Run the matched case from the Bridge repository root with the native layer-by-layer scheduler and Megatron-FSDP installed in Megatron-LM.

```bash
study=examples/scale-down/02-layer-by-layer/01-baseline
bash "$study/02_param_cpu_act_gpu_opt_gpu.sh" --dry-run
bash "$study/02_param_cpu_act_gpu_opt_gpu.sh"
```

Megatron-FSDP stores each rank's BF16 decoder compute-weight shards in pinned host memory and stages them to the GPU for NCCL all-gather. Embedding and output weights follow Megatron-FSDP's default handling. AdamW, its FP32 master weights and moments, and layer-boundary activations stay on the GPU. The optimizer publishes updated decoder compute-weight shards back to pinned host memory.

Results go under `results/02-layer-by-layer/01-baseline/02-param-cpu-act-gpu-opt-gpu/`. A pass records `weight` as the only FSDP CPU-offload component, alongside the scheduler, FSDP sharding strategy, optimizer backend, final loss, and peak GPU allocation and reservation across ranks. Use the Nsight trace to inspect actual weight transfers or collectives.

## 8-layer CPU parameter and boundary activation offload

Case 03 uses the same model, batches, FSDP weight offload, and GPU AdamW as case 02. Each decoder microbatch submits its checkpoint input to pinned CPU memory after its layer forward. The last microbatch of the final decoder layer keeps its input on the GPU. Backward reloads checkpoint inputs in descending microbatch order within the current stage and prefetches the preceding input as the current microbatch runs. Each decoder layer accumulates its parameter gradients in a single backward call. Activation gradients and optimizer state stay on the GPU.

```bash
study=examples/scale-down/02-layer-by-layer/01-baseline
bash "$study/03_param_cpu_act_cpu_opt_gpu.sh" --dry-run
bash "$study/03_param_cpu_act_cpu_opt_gpu.sh"
```

Results go under `results/02-layer-by-layer/01-baseline/03-param-cpu-act-cpu-opt-gpu/`. The result records `activation_tier=cpu` and `fsdp_cpu_offload=["weight"]`. In the same capture window, compare case 03 with case 02 using the `activation offloading {stage} microbatch={mb}` and `activation reloading {stage} microbatch={mb}` NVTX ranges. Use `--profile none` for matched training time and peak GPU memory measurements.

## 8-layer CPU parameter, activation, and optimizer offload

```bash
study=examples/scale-down/02-layer-by-layer/01-baseline
bash "$study/04_param_cpu_act_cpu_opt_cpu.sh" --dry-run
bash "$study/04_param_cpu_act_cpu_opt_cpu.sh" --cpu-threads 32
```

Case 04 keeps decoder layer 0's weights, main gradients, and AdamW update on the GPU. Each decoder layer runs one backward for all its microbatches. Megatron-FSDP queues BF16 reduce-scatter for that layer before starting the next layer's backward. It converts each reduced shard for CPU-updated parameters to FP32 on the GPU and copies it directly into the bucket's pinned CPU main gradient. The recipe sets gradient clipping to zero. GraceAdam reads the finalized CPU shards in place and runs once per FSDP-3 parameter bucket after all backward work finishes. Each parameter keeps its configured AdamW hyperparameters.

In Nsight Systems, `fsdp.cpu_grad.d2h bucket=<id>` marks the gradient copy submission. Case 04 uses `<stage>.dense.<parameter count>` for FSDP parameter buckets, with stages named `embedding`, `transformer.0`, and `output`. The count is the number of unsharded parameters in the bucket, expressed in millions. The same label appears in the FSDP collective ranges and the `GraceAdam bucket=<id>` and `fsdp.layer_by_layer.GPUAdamW bucket=<id>` update ranges. Results go under `results/02-layer-by-layer/01-baseline/04-param-cpu-act-cpu-opt-cpu/`.

## 8-layer CPU optimizer overlap

Case 05 uses case 04's model, batches, placement, and AdamW hyperparameters. A worker starts GraceAdam for each CPU-updated FSDP bucket after its reduce-scatter and pinned FP32 main-gradient copy complete. The worker can run alongside later backward computation and collectives. The regular optimizer step waits for outstanding CPU updates and publishes the updated weights before the next forward. The GPU-resident decoder layer 0 still updates in the regular optimizer step.

The overlap path uses BF16 with unit loss scale and per-microbatch loss normalization. It disables gradient clipping, gradient zero counts, gradient-based step skipping, gradient checks, and iteration reruns.

On the GB200 node, run the patch helper from the Bridge root, including when upgrading a build that already releases the GIL. The helper patches the installed DeepSpeed `v0.20.1` CPUAdam C++ sources in place and JIT-compiles the extension. It verifies GIL release and the `graceadam_omp_threads_patch` marker for setting the OpenMP thread count in the native update. The installed package must use JIT for CPUAdam (`DS_BUILD_CPU_ADAM=0` at installation). The command below selects the same Python as the numbered launcher.

```bash
train_python="${PYTHON:-$(uv run --no-sync python -c 'import sys; sys.stdout.write(sys.executable)')}"
patch_helper=examples/scale-down/01-analyse/02-deepspeed-offload/apply_cpu_adam_gil_patch.sh
PYTHON_BIN="$train_python" bash "$patch_helper"
```

If you set `TORCH_EXTENSIONS_DIR`, export the same value for the patch helper and the training launcher. To restore the original CPUAdam sources and build, run `PYTHON_BIN="$train_python" bash "$patch_helper" --recovery`.

```bash
study=examples/scale-down/02-layer-by-layer/01-baseline
bash "$study/05_param_cpu_act_cpu_opt_cpu_overlap.sh" --dry-run
bash "$study/05_param_cpu_act_cpu_opt_cpu_overlap.sh" --cpu-threads 32
```

Results go under `results/02-layer-by-layer/01-baseline/05-param-cpu-act-cpu-opt-cpu-overlap/`. Compare cases 04 and 05 with the same DeepSpeed build and thread count. On the Nsight timeline, check that each worker's `GraceAdam bucket=<id>` update starts after the corresponding GPU-to-CPU gradient transfer ends. Use the same timeline to measure its overlap with subsequent backward and NCCL operations. Use `--profile none` for matching step-time runs.

## 8-layer NVMe parameters with CPU optimizer overlap

Case 06 places all BF16 FSDP compute-weight shards on NVMe and uses case 05's CPU activation and GraceAdam overlap settings. Each rank synchronously stages its local shard from NVMe through CPU memory to GPU before all-gather. When FSDP releases a managed weight bucket, it copies the local shard through CPU memory back to NVMe before freeing the GPU buffer.

Decoder layer 0 keeps its FP32 master weights, main gradients, and Adam moments on the GPU. Its AdamW update casts the updated shard to BF16 on the GPU and writes it through CPU memory to NVMe. GraceAdam updates the remaining buckets' FP32 master weights on the CPU, then casts their updated shards to BF16 on the CPU for NVMe writeback. The next forward reads the updated shards after these writes complete.

Install the Megatron-LM `megatron_nvme_aio` extension in the training environment. It uses the DeepSpeed-derived Linux `libaio` worker and host direct I/O. Building it requires a C++17 compiler, `libaio-dev`, and NVTX3 headers. Set `NVTX_INCLUDE_DIR` if the headers live outside the CUDA include directory.

```bash
uv pip install --no-build-isolation --no-deps --reinstall ../megatron-lm/src/nvme/deepspeed

study=examples/scale-down/02-layer-by-layer/01-baseline
bash "$study/06_param_nvme_act_cpu_opt_cpu_overlap.sh" --dry-run
bash "$study/06_param_nvme_act_cpu_opt_cpu_overlap.sh" --cpu-threads 32
```

All four ranks share the NVMe root `/tmp`. Each launch creates a temporary directory there with separate swap files per rank. To select another local NVMe directory, pass `--nvme DIR` or set `NVME_PATH`. Results go under `results/02-layer-by-layer/01-baseline/06-param-nvme-act-cpu-opt-cpu-overlap/`. The result records `parameter_tier=nvme`, `optimizer_tier=cpu`, and `fsdp_nvme_weights_only=true`.

## 8-layer NVMe parameters and activations with CPU optimizer overlap

Case 07 uses the same BF16 NVMe parameter shards and GraceAdam overlap as case 06. The updated BF16 shards are written back to NVMe before the next forward. The four ranks use one shared `/tmp` scratch root and write separate parameter and activation swap files there. The same `megatron_nvme_aio` extension handles host direct I/O for both, so the case runs without GPUDirect Storage.

Each rank uses two reusable pinned slots for saved layer-boundary activations, one for forward writes and one for backward reads. The host AIO worker uses a reusable page-aligned buffer for direct I/O. Forward waits for the GPU-to-host copy before writing to NVMe and waits for the write before reusing the slot. Backward waits for the NVMe read and the CPU staging copy before submitting the host-to-GPU copy. That copy completes on the requesting CUDA stream before the activation is used and before the read slot is reused. Activation I/O runs synchronously. The overlap in the case name refers to GraceAdam running alongside later backward work.

```bash
study=examples/scale-down/02-layer-by-layer/01-baseline
bash "$study/07_param_nvme_act_nvme_opt_cpu_overlap.sh" --dry-run
bash "$study/07_param_nvme_act_nvme_opt_cpu_overlap.sh" --cpu-threads 32
```

Results go under `results/02-layer-by-layer/01-baseline/07-param-nvme-act-nvme-opt-cpu-overlap/`. The result records `parameter_tier=nvme`, `activation_tier=nvme`, `optimizer_tier=cpu`, and `fsdp_nvme_weights_only=true`.

## 8-layer CPU parameters and NVMe activations

Case 08 uses the CPU parameter and GraceAdam overlap settings from case 05 with the NVMe activation path from case 07. The model still uses eight decoder layers, sequence length 4096, GBS 16, and 10 steps by default. By default it uses all four GPUs on one tray and the shared scratch root `/tmp`. Pass `--nvme DIR` or set `NVME_PATH` to select another local NVMe directory. Activation reads and writes are synchronous. GraceAdam updates can overlap later backward work.

```bash
study=examples/scale-down/02-layer-by-layer/01-baseline
bash "$study/08_param_cpu_act_nvme_opt_cpu_overlap.sh" --dry-run
```

Use `--gpu 4 --node 4` for four trays with four ranks total. The launcher uses only GPU 0 in PCI bus order on each tray. Each rank uses its own tray's NVMe for saved activations.

Prepare the same Megatron-LM layer-by-layer sources, patched DeepSpeed CPUAdam, and `megatron_nvme_aio` environment on all four trays. Before launching, check `findmnt --target "$NVME_PATH"` on each tray and verify the path is backed by that tray's local NVMe. Reserve the device for the test. Pass that directory with `--nvme` or `NVME_PATH`. Each launcher creates and removes its own scratch directory.

Run this command once on each tray, with `NODE_RANK` set to 0, 1, 2, or 3 and the same reachable tray-0 `MASTER_ADDR` and `MASTER_PORT` across trays.

```bash
study=examples/scale-down/02-layer-by-layer/01-baseline
bash "$study/08_param_cpu_act_nvme_opt_cpu_overlap.sh" --gpu 4 --node 4 \
  --node-rank "$NODE_RANK" --master-addr "$MASTER_ADDR" --master-port "${MASTER_PORT:-29500}" \
  --nvme "$NVME_PATH"
```

Add `--dry-run` to inspect the local command. For a repeat, use a fresh `--output-dir` on every tray. The four-tray run adds `-nodes_4` to the model output directory name and keeps per-tray logs and Nsight reports in `node0/` through `node3/`. Check the aggregate loss, optimizer steps, placement, and peak HBM in `node0/summary.csv`. All four `run_info.txt` files should show `status=0`, with `case_status=passed` on node 0 and `case_status=worker_completed` on nodes 1–3. TensorBoard is written on node 3.

## Running and profiling the numbered cases

The numbered scripts select the experiment settings and call `run_pretrain_layer_by_layer.sh`. That Bash launcher wraps each local torchrun agent in Nsight Systems and runs Bridge's `scripts/training/run_recipe.py --recipe qwen35_text_27b_pretrain_4gpu_gb200_bf16_layer_by_layer_config --mode pretrain --dataset mock` on each rank. Bridge calls the layer-by-layer schedule through its normal training step. Single-tray runs use the same NUMA selection order as the FSDP3 study. The four-tray run inherits the CPU affinity on each tray.

The cases use Nsight Systems **2026.6.1** to capture CUDA, NVTX, and system-wide CPU thread scheduling. System-wide scheduling requires Linux perf access. Expand a training process in the timeline to inspect when its threads run on the CPU. CPU call-stack sampling and GPU metrics sampling are disabled. The default window is iteration 8 (step 7, zero-based). With `--train-iters 3`, capture covers iteration 3. The launcher checks `NsightSystems-2026.6.1/bin/nsys` under the Bridge repository root or the executable set through `NSYS_BIN`. Use `--profile none` for an unprofiled run. [Nsight Systems CLI reference](https://docs.nvidia.com/nsight-systems/UserGuide/)

The eight-layer baseline uses outer NVTX ranges `forward.embedding`, `forward.transformer.0` through `forward.transformer.7`, and `forward.output`. Each contains `fwd.microbatch.{mb}.{stage}` ranges, such as `fwd.microbatch.0.transformer.0`. Backward uses `backward.output`, `backward.transformer.7` through `backward.transformer.0`, and `backward.embedding`. Decoder checkpoint replay has separate `recompute.microbatch.{mb}.transformer.{layer}` ranges. The `bwd.microbatch.{mb}.{stage}` ranges track gradient work through autograd hooks. Decoder and output ranges end when their stage-input gradient is ready. Embedding ranges end when the grouped backward completes because token IDs have no gradient and weight-gradient accumulation is shared. Microbatch ranges can overlap. Microbatch and decoder layer indices are zero-based.

On each node the launcher checks IPv4 addresses on up interfaces. It honors valid `GLOO_SOCKET_IFNAME` and `NCCL_SOCKET_IFNAME` values. Automatic selection tries `eth0`, then `bond1`, then an IPv4 default-route interface. The four-tray run tries `bond1` first. NCCL uses an exact interface name. The selected interfaces and Nsight version are recorded in `run_info.txt`.

The all-GPU baseline writes to `results/02-layer-by-layer/01-baseline/01-param-gpu-act-gpu-opt-gpu/model_27b-lbl-schedule-.../node0/` in Bridge. The Nsight report is named `qwen35_27b-lbl-layers_8-attn_full-bf16-cg_off-tp_1-cp_1-pp_1-dp_4-mbs_1-gbs_16-seq_4k-iters_10-param_gpu-act_gpu-opt_gpu.nsys-rep`. `launch.json` records its filename. The same directory contains `command.txt`, `config.yaml`, `run_info.txt`, `train.log`, `summary.csv`, `tensorboard/`, and the source revisions and hashes. Use `--print-result-dir` to resolve the path or `--output-dir` for a fresh repeat.

Bridge prints iteration time and model TFLOP/s per GPU to the terminal and `train.log` as the numbered cases run. TensorBoard records `throughput/tflops/device` and, from the second iteration, `throughput/tokens_per_sec`. Use `--profile none` for throughput measurements without Nsight Systems capture overhead.

The numbered cases require `status=0` and a nonempty report when profiling.

## Capacity sweep

```bash
bash examples/scale-down/02-layer-by-layer/run_sweep.sh \
  --layers 8 16 32 --parameter-tier cpu
```

Defaults are eight decoder layers, sequence length 4096, micro-batch size 1, four microbatches per rank per step, three optimizer steps, and GPU placement for every tier. Choose `gpu` or `cpu` independently for `--parameter-tier`, `--activation-tier`, and `--optimizer-tier`. CPU activation placement covers the saved layer boundaries. CPU-tier PyTorch AdamW uses CPU gradient accumulation. CPU-tier GraceAdam uses the direct per-layer FP32 transfer from case 04. Add `--overlap-cpu-optimizer` with `--parameter-tier cpu --optimizer-tier cpu --cpu-optimizer grace_adam` to enable case 05's optimizer scheduling.

Use `--parameter-tier nvme --optimizer-tier cpu --nvme-dir /tmp` to swap BF16 compute weights and add `--activation-tier nvme` to swap saved layer-boundary activations. Use `--optimizer-tier nvme` with the same parameter tier to swap both weights and optimizer state. Any NVMe tier requires `--nvme-dir` and the `megatron_nvme_aio` extension from the Megatron-LM checkout's `src/nvme/deepspeed`. Each NVMe case uses a fresh scratch directory. Its filesystem holds all four ranks' files. With `--optimizer-tier nvme` at full depth, FP32 master weights and the two Adam moments alone require roughly 324 GB in total, with additional storage for compute weights and staging.

Add `--dry-run` to inspect the commands. To narrow the capacity limit, rerun at depths between the largest passing case and the first allocation failure.

Each sweep creates a new result directory under `./results`, or at the unused path passed as `--results-dir`. It retains the Bridge revision in `revision.txt`, the Megatron-LM revision in `megatron_lm_revision.txt`, and SHA-256 hashes keyed by `bridge/<file>` and `megatron-lm/<file>` in `source_sha256.json`. It also saves commands, per-depth training logs and TensorBoard directories, and `summary.csv`. A pass requires successful worker exit, the expected number of successful optimizer steps, finite final loss, valid measured peak GPU memory, and an exact match for the scheduler and FSDP offload settings in `LAYER_BY_LAYER_RESULT`. CUDA allocation errors, explicit host allocation errors, and NVMe space or quota errors get separate statuses. The reported depth is the largest observed passing case.
