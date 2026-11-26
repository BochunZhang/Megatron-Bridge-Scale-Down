#!/usr/bin/env bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../.." && pwd)"
case_name="$1"
parameter_tier="$2"
activation_tier="$3"
optimizer_tier="$4"
shift 4
nvme_path="${NVME_PATH:-}"
output_dir="${OUTPUT_DIR:-${repo_root}/results/02-layer-by-layer}"
layers=8
train_iters=10
profile=nsys
cpu_optimizer=torch
overlap_cpu_optimizer=false
cpu_threads=""
dry_run=false
print_result_dir=false
case_08=false
if [[ "${case_name}" == 08-param-cpu-act-nvme-opt-cpu-overlap ]]; then
    case_08=true
fi
node_count=1
node_rank=""
master_addr=""
master_port=""

fail() { printf '%s\n' "$*" >&2; exit 2; }

while (( $# )); do
    case "$1" in
        --gpu|--node|--node-rank|--master-addr|--master-port)
            [[ "${case_08}" == true ]] || fail "$1 is only supported by case 08"
            [[ -n "${2:-}" && "$2" != --* ]] || fail "Missing value for $1"
            case "$1" in
                --gpu)
                    [[ "$2" == 4 ]] || fail "Case 08 uses 4 GPUs total";;
                --node)
                    [[ "$2" == 1 || "$2" == 4 ]] || fail "--node must be 1 or 4 for case 08"
                    node_count="$2";;
                --node-rank) node_rank="$2";;
                --master-addr) master_addr="$2";;
                --master-port) master_port="$2";;
            esac
            shift 2;;
        --nvme|--nvme-path|--output-dir|--train-iters|--profile|--cpu-optimizer|--cpu-threads)
            [[ -n "${2:-}" && "$2" != --* ]] || fail "Missing value for $1"
            case "$1" in
                --nvme|--nvme-path) nvme_path="$2";;
                --output-dir) output_dir="$2";;
                --cpu-optimizer)
                    [[ "$2" == grace_adam || "$2" == torch ]] || fail "--cpu-optimizer must be grace_adam or torch"
                    cpu_optimizer="$2";;
                --cpu-threads)
                    [[ "$2" =~ ^[1-9][0-9]*$ ]] || fail "--cpu-threads must be a positive integer"
                    cpu_threads="$2";;
                --profile)
                    [[ "$2" == nsys || "$2" == none ]] || fail "--profile must be nsys or none"
                    profile="$2";;
                --train-iters)
                    [[ "$2" =~ ^[1-9][0-9]*$ ]] || fail "--train-iters must be a positive integer"
                    train_iters="$2";;
            esac
            shift 2;;
        --overlap-cpu-optimizer) overlap_cpu_optimizer=true; shift;;
        --dry-run) dry_run=true; shift;;
        --print-result-dir) print_result_dir=true; shift;;
        -h|--help)
            cat <<'HELP'
Run an 8-layer Qwen3.5-27B full-attention case on one four-GPU node by default.

  --nvme DIR            Local scratch directory when an NVMe tier is selected
  --output-dir DIR      Result root (default: results/02-layer-by-layer)
  --train-iters N       Optimizer steps (default: 10)
  --profile nsys|none   Use Nsight Systems 2026.6.1 (default: nsys)
  --cpu-optimizer torch|grace_adam   Optimizer backend when a CPU/NVMe optimizer tier is selected
  --overlap-cpu-optimizer  Start GraceAdam updates as CPU main gradients become ready
  --cpu-threads N       CPU optimizer and PyTorch intra-op threads per rank (default: 32)
  --dry-run             Show the training command
  --print-result-dir    Print this node's result directory
HELP
            if [[ "${case_08}" == true ]]; then
                cat <<'HELP'
  --gpu 4               Total GPU count (default: 4)
  --node 1|4            Compute tray count (default: 1). Use GPU 0 per tray with --node 4.

With --node 4:
  --node-rank N         Tray rank, 0 through 3 (or NODE_RANK / RANK)
  --master-addr HOST    Address of tray 0 (or MASTER_ADDR / DLC_MASTER_ADDR)
  --master-port PORT    Rendezvous port (or MASTER_PORT, default: 29500)
  Set --nvme or NVME_PATH to this tray's local NVMe scratch directory.
HELP
            fi
            exit 0;;
        *) fail "Unknown option $1";;
    esac
done

if [[ "${overlap_cpu_optimizer}" == true ]]; then
    [[ ( "${parameter_tier}" == cpu || "${parameter_tier}" == nvme ) &&
        "${optimizer_tier}" == cpu && "${cpu_optimizer}" == grace_adam ]] ||
        fail "--overlap-cpu-optimizer requires CPU or NVMe weights and the CPU optimizer tier with GraceAdam"
fi

if [[ "${node_count}" == 4 ]]; then
    node_rank="${node_rank:-${NODE_RANK:-${RANK:-}}}"
    master_addr="${master_addr:-${MASTER_ADDR:-${DLC_MASTER_ADDR:-}}}"
    master_port="${master_port:-${MASTER_PORT:-29500}}"
    if [[ "${dry_run}" == true || "${print_result_dir}" == true ]]; then
        node_rank="${node_rank:-0}"
        master_addr="${master_addr:-<rank0-host>}"
        nvme_path="${nvme_path:-/local/nvme}"
    fi
    [[ "${node_rank}" =~ ^[0-3]$ ]] || fail "Set --node-rank or NODE_RANK to a tray rank from 0 to 3"
    if [[ "${print_result_dir}" == false ]]; then
        [[ -n "${master_addr}" ]] || fail "Set --master-addr or MASTER_ADDR to the address of tray 0"
        [[ -n "${nvme_path}" ]] || fail "Set --nvme or NVME_PATH to this tray's local NVMe directory"
    fi
elif [[ "${case_08}" == true ]]; then
    [[ -z "${node_rank}${master_addr}${master_port}" ]] || fail "Rendezvous options require --node 4"
    nvme_path="${nvme_path:-/tmp}"
fi

[[ "${output_dir}" == /* ]] || output_dir="${PWD}/${output_dir}"
model_name="model_27b-lbl-schedule-layers_${layers}-attn_full-cg_off-mbs_1-seq_4096-tp_1-cp_1-gbs_16-gpus_4"
if [[ "${node_count}" == 4 ]]; then
    model_name+="-nodes_4"
fi
model_name+="-param_${parameter_tier}-act_${activation_tier}-opt_${optimizer_tier}-iters_${train_iters}-profile_${profile}"
result_dir="${output_dir%/}/01-baseline/${case_name}/${model_name}/node${node_rank:-0}"
if [[ "${print_result_dir}" == true ]]; then
    printf '%s\n' "${result_dir}"
    exit 0
fi

args=(--single-case-layout --results-dir "${result_dir}"
    --layers "${layers}" --seq-length 4096 --micro-batch-size 1 --microbatches 4
    --steps "${train_iters}" --parameter-tier "${parameter_tier}" --activation-tier "${activation_tier}"
    --optimizer-tier "${optimizer_tier}"
    --cpu-optimizer "${cpu_optimizer}"
    --profile "${profile}")
[[ "${overlap_cpu_optimizer}" == true ]] && args+=(--overlap-cpu-optimizer)
[[ "${dry_run}" == true ]] && args+=(--dry-run)
[[ -z "${nvme_path}" ]] || args+=(--nvme-dir "${nvme_path}")
[[ -z "${cpu_threads}" ]] || args+=(--cpu-threads "${cpu_threads}")
if [[ "${node_count}" == 4 ]]; then
    args+=(--node-rank "${node_rank}" --master-addr "${master_addr}" --master-port "${master_port}")
fi

exec bash "${script_dir}/../run_pretrain_layer_by_layer.sh" "${args[@]}"
