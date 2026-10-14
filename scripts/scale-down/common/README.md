# Scale-down run helpers

Sourceable shell helpers shared by the scale-down experiment scripts. Each file
only defines functions when sourced; running one directly resolves and prints its
values so you can check a node before submitting a job.

```bash
source scripts/scale-down/common/gloo_socket_ifname.sh
source scripts/scale-down/common/node_local_storage.sh
source scripts/scale-down/common/run_timestamp_and_logs.sh

resolve_gloo_socket_ifname "${NNODES}"   # before the run-id handshake: it needs Gloo
setup_runtime_storage
stage_hf_assets Qwen/Qwen3.5-9B-Base
setup_log_dir logs/01-analyse/01-fine-grained-offload
exec > >(tee -a "${RUN_LOG}") 2>&1
```

| File | Provides |
| --- | --- |
| [gloo_socket_ifname.sh](gloo_socket_ifname.sh) | `gloo_available_ifnames`, `resolve_gloo_socket_ifname [NNODES]` |
| [node_local_storage.sh](node_local_storage.sh) | `setup_runtime_storage`, `stage_hf_assets <repo>...` |
| [run_timestamp_and_logs.sh](run_timestamp_and_logs.sh) | `resolve_run_time [DRY_RUN]`, `setup_log_dir <relative-path> [DRY_RUN]` |

## Gloo bootstrap interface

`resolve_gloo_socket_ifname` selects the interface from the IPv4 addresses this
node actually has, then exports `GLOO_SOCKET_IFNAME`:

- one node → `eth0`, falling back to `bond1`
- more than one node → `bond1`, falling back to `eth0`

An inherited `GLOO_SOCKET_IFNAME` (a stale `bond1` from a single-node DSW, for
example) is honored only when that interface has an IPv4 address here. Otherwise
it is replaced and reported, because Gloo resolves the name locally and aborts
process-group creation with `Unable to find address for: <ifname>`. If neither
candidate is usable the helper fails instead of handing a broken value to Gloo.

Requires the `ip` command. `NNODES` defaults to the environment variable of the
same name, so a DLC job only needs `resolve_gloo_socket_ifname "${NNODES}"`.

## Runtime storage

`setup_runtime_storage` splits the paths a run needs by whether the other nodes
must see them.

Node-local, under `/tmp/megatron-bridge` (`NODE_LOCAL_CACHE_ROOT`):

| Variable | Holds |
| --- | --- |
| `HF_HOME`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE` | HF configs, tokenizer, library files |
| `NEMO_HOME`, `NEMO_DATASETS_CACHE`, `NEMO_MODELS_CACHE` | NeMo caches |
| `TRITON_CACHE_DIR` | kernel build output, write-heavy per rank |

Shared, on the repo filesystem:

| Variable | Default |
| --- | --- |
| `RESULTS_ROOT` | `${REPO_ROOT}/results` |
| `UV_CACHE_DIR` | `${REPO_ROOT}/.cache/uv` |

The `uv` cache and the virtualenv stay shared: the venv is installed at a fixed
path and resolves the editable `3rdparty/Megatron-LM` through the common repo
path, so moving it per node breaks the launch for no gain.

Node-local matters for the HF library files specifically. With every rank
reading `tokenizer.json` from the shared `HF_HOME`, a concurrent rewrite of the
cache blob fails `json.load` with `[Errno 116] Stale file handle`; each node
stages the repo once and then reads its own copy. Set `HF_CACHE_ROOT` to a shared
path to opt back into the shared-cache behavior.

The function creates every directory and probes that the node-local root is
writable, so a broken or shared-backed `/tmp` fails at launch rather than 200
lines into a NCCL timeout.

`stage_hf_assets` downloads config and tokenizer files only — weights are
excluded because training loads the checkpoint from its own path — retries
transient hub errors, and then exports `HF_HUB_OFFLINE=1` so a missing file
fails on the node instead of reaching the network mid-step.

## Run id and log location

`resolve_run_time` produces one batch id (`RUN_TIME`, `%Y%m%d-%H%M%S`) shared by
every node, then `setup_log_dir` turns it into this node's log directory:

| Node | Directory |
| --- | --- |
| 0 | `<RESULTS_ROOT>/<relative-path>/<RUN_TIME>/` |
| others | `<RESULTS_ROOT>/<relative-path>/<RUN_TIME>/rank<N>/` |

Each node formatting its own `date` would scatter one logical run across several
timestamps, since a job takes minutes to fan out. Instead node 0 stamps the id
once and publishes it through a TCPStore on `MASTER_ADDR:MASTER_PORT`; the other
nodes block until they read it, and node 0 waits for every acknowledgement before
returning, so the first training step cannot start while a node is still
resolving its path. An exported or `--run-time` value always wins, which is what
pinning a comparison needs.

Non-zero ranks get a `rank<N>` subdirectory because one shared job log written by
several nodes at once is unreadable. The handshake is CPU-only and must run
before any CUDA context exists — it is a separate short-lived process, not the
training process. It bootstraps with Gloo, so `resolve_gloo_socket_ifname` has to
run first; `resolve_run_time` fails with that instruction rather than hanging when
`GLOO_SOCKET_IFNAME` is unset.

Each node also writes a `node.json` (run id, rank, node count, hostname, master
address, Gloo interface, log directory) next to its log, so a batch can be
attributed to nodes after the fact.

`RANK` comes from DLC. It is ambiguous in a job that also uses Knative, where DLC
sets `RANK` to the worker rank as well — translate it before calling this.

If the shared tree is not writable from a worker, non-zero ranks fall back to
`<LOG_DIR>/<relative-path>/<RUN_TIME>/rank<N>/` on node-local disk and say so in
the log; set `LOG_LOCAL=1` to force that on every non-zero rank.

## Environment overrides

| Variable | Effect |
| --- | --- |
| `NNODES` | Default node count for `resolve_gloo_socket_ifname` |
| `NODE_LOCAL_CACHE_ROOT` | Node-local cache root (default `/tmp/megatron-bridge`) |
| `HF_CACHE_ROOT` | Force a shared HF cache instead of staging per node |
| `HF_HOME`, `NEMO_HOME`, `TRITON_CACHE_DIR`, `UV_CACHE_DIR`, `RESULTS_ROOT` | Pinned as-is when already set |
| `RUN_TIME` | Pin the batch id instead of resolving one |
| `MASTER_ADDR`, `MASTER_PORT`, `RANK` | DLC-provided; required to agree on the id across nodes |
| `RUN_TIME_SYNC_TIMEOUT_MINUTES` | Handshake timeout (default 30) |
| `LOG_DIR` | Node-local log root for the fallback path (default `/tmp/megatron-bridge/logs`) |
| `LOG_LOCAL` | `1` forces non-zero ranks onto node-local logs |
| `REPO_ROOT` | Repo root (default: derived from this directory) |

`log`, `warn` and `die` are defined only when the sourcing script has not
already defined them, so these helpers compose with an existing
`lib/common.sh` in an experiment directory.
