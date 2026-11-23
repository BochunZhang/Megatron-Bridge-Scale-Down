#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

die() { printf 'error: %s\n' "$*" >&2; exit 1; }

case "${1:-}" in
    -h|--help)
        cat <<'EOF'
Usage: bash run_superoffload_nsys.sh [--dry-run]

Profile the SuperOffload 1.0 param_gpu/param_cpu comparison on the first four
CUDA-visible GPUs. Both runs use Qwen3.5-9B, seq 4096, MBS 1, and no recompute.
Set NSYS_BIN to the Nsight Systems 2026.6.1 executable if needed.
Set RESULTS_ROOT to override the parent directory for timestamped results.
--dry-run prints the planned commands without probing or locking GPUs.
EOF
        exit 0
        ;;
    ''|--dry-run) [[ $# -le 1 ]] || die 'expected only --dry-run' ;;
    *) die "unknown argument: $1" ;;
esac

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../../../.." && pwd)
results_parent=${RESULTS_ROOT:-"$repo_root/results/01-analyse/02-deepspeed-nsys"}
nsys_bin=${NSYS_BIN:-"$PWD/NsightSystems-2026.6.1/bin/nsys"}
if [[ -z ${NSYS_BIN:-} && ! -x "$nsys_bin" ]]; then
    nsys_bin=nsys
fi

model=Qwen3.5-9B-Base
experiment=(bash "$script_dir/pretrain_experiment.sh"
    --models "Qwen/$model" --num_gpus 4 --optimizer_strategies super_offload_1.0
    --param_positions 'param_gpu param_cpu' --recompute_combos recompute_none
    --micro_batch_sizes 1 --per_gpu_batch_size 4 --profile nsys)

if [[ ${1:-} == --dry-run ]]; then
    printf '%q --version   # requires 2026.6.1\n' "$nsys_bin"
    printf 'nvidia-smi -i <first-four-visible-GPU-UUIDs> --lock-gpu-clocks=tdp,tdp\n'
    printf 'OVERLAP_COMM=true STEPS=10 SEQ_LEN=4096 '
    printf '%q ' "${experiment[@]}"
    printf '\nResults parent: %s\n' "$results_parent"
    exit 0
fi

nsys_bin=$(command -v "$nsys_bin") || die 'set NSYS_BIN to the Nsight Systems 2026.6.1 executable'
[[ $nsys_bin == /* ]] || nsys_bin="$PWD/$nsys_bin"
nsys_version=$("$nsys_bin" --version) || die "failed to query $nsys_bin"
[[ $nsys_version =~ (^|[^0-9.])2026\.6\.1([.-]|[[:space:]]|$) ]] \
    || die "expected Nsight Systems 2026.6.1, got $nsys_version"
command -v nvidia-smi >/dev/null || die 'nvidia-smi is required'
command -v deepspeed >/dev/null || die 'activate the DeepSpeed training environment first'

cuda_gpu_uuids() {
    "${PYTHON_BIN:-python}" - <<'PY'
from uuid import UUID

import torch

for index in range(torch.cuda.device_count()):
    uuid = UUID(str(torch.cuda.get_device_properties(index).uuid).removeprefix("GPU-"))
    print(f"GPU-{uuid}")
PY
}

# DeepSpeed --include consumes unmasked CUDA ordinals. nvidia-smi uses UUIDs
# so the clock lock targets the same devices regardless of NVML's index order.
visible_uuids=$(cuda_gpu_uuids) || die 'failed to query visible CUDA devices'
all_uuids=$visible_uuids
if [[ ${CUDA_VISIBLE_DEVICES+x} ]]; then
    all_uuids=$(unset CUDA_VISIBLE_DEVICES; cuda_gpu_uuids) || die 'failed to map CUDA devices'
fi
gpu_map=$(awk '
    NR == FNR { slot[$1] = FNR - 1; next }
    FNR <= 4 && ($1 in slot) { print slot[$1], $1 }
' <(printf '%s\n' "$all_uuids") <(printf '%s\n' "$visible_uuids"))
[[ $(awk 'NF == 2 && !seen[$2]++ { count++ } END { print count + 0 }' <<< "$gpu_map") == 4 ]] \
    || die 'four distinct visible CUDA GPUs are required'
gpu_slots=$(awk '{print $1}' <<< "$gpu_map" | paste -sd , -)
gpu_uuids=$(awk '{print $2}' <<< "$gpu_map" | paste -sd , -)

mkdir -p "$results_parent"
run_dir=$(mktemp -d "$(cd "$results_parent" && pwd)/$(date +%Y%m%d_%H%M%S).XXXXXX")
printf '%s\n' "$nsys_version" "NSYS_BIN=$nsys_bin" "CUDA slots=$gpu_slots" "GPU UUIDs=$gpu_uuids" \
    > "$run_dir/profile_env.txt"
printf 'Results: %s\nLocking GPU clocks to TDP on %s\n' "$run_dir" "$gpu_uuids"
nvidia-smi -i "$gpu_uuids" --lock-gpu-clocks=tdp,tdp 2>&1 | tee "$run_dir/gpu_clock_lock.log" \
    || die "TDP lock failed, see $run_dir/gpu_clock_lock.log"

unset CUDA_VISIBLE_DEVICES
NSYS_BIN="$nsys_bin" DEEPSPEED_LOCAL_GPU_IDS="$gpu_slots" OVERLAP_COMM=true STEPS=10 SEQ_LEN=4096 \
    RESULTS_ROOT="$run_dir" TMP_CONFIG_DIR="$run_dir/.config" "${experiment[@]}"

summary=("$run_dir"/experiment_summary_*.txt)
[[ -f ${summary[0]} ]] && awk '$NF == "OK" { ok++ } END { exit !(NR == 2 && ok == 2) }' "${summary[0]}" \
    || die "one or more cases failed, see $run_dir"
for placement in param_gpu param_cpu; do
    reports=("$run_dir/$model/super_offload_1.0-${placement}__recompute_none__mbs1"/*/nsys_trace.nsys-rep)
    [[ -s ${reports[0]} ]] || die "missing Nsight Systems report for $placement under $run_dir"
    printf 'Trace: %s\n' "${reports[0]}"
done
