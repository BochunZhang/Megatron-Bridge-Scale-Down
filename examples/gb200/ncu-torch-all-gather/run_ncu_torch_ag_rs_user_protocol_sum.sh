#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

usage() {
    cat <<'USAGE'
Profile AllGather_STMC and ReduceScatter_LDMC in separate NCU sessions on one GB200 node.
Requires Nsight Compute 2026.4 or newer and a torch/NCCL build with symmetric windows.

Usage: run_ncu_torch_ag_rs_user_protocol_sum.sh [options]
  --workload ag|rs|both    Default: both, one profiled collective per session
  --gpus LIST             One rank per GPU (default: 0,1,2,3)
  --size-mib N            AG output / RS input per rank (default: 1024, BF16)
  --ncu PATH              NCU binary (default: repository 2026.4 first, then PATH)
  --output-dir DIR        Default: results/gb200/ncu-torch-all-gather/ag-rs-protocol-<time>
  --dry-run               Show the commands without starting NCU
  -h, --help

Writes NCU reports and raw CSVs under ag/ and rs/ and a combined nvlink_bytes.csv.
Each row identifies the reported GPU; shmem uses NCU's default primary-device selection.
For each TX/RX request/response pair, collect both User Data and Protocol Data.
The eight traffic fields and both TX/RX totals are direct byte .sum values, as in the P2P test.
USAGE
}

die() { echo "ERROR: $*" >&2; exit 1; }

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
workload=both gpus=0,1,2,3 size_mib=1024 ncu_bin=ncu output_dir=""
dry_run=false
while (( $# )); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --dry-run) dry_run=true; shift ;;
        --workload|--gpus|--size-mib|--ncu|--output-dir)
            [[ -n "${2:-}" && "$2" != --* ]] || die "$1 needs a value"
            case "$1" in
                --workload) workload="$2" ;;
                --gpus) gpus="$2" ;;
                --size-mib) size_mib="$2" ;;
                --ncu) ncu_bin="$2" ;;
                --output-dir) output_dir="$2" ;;
            esac
            shift 2 ;;
        *) die "unknown argument: $1 (see --help)" ;;
    esac
done
case "$workload" in
    both) workloads=(ag rs) ;;
    ag|rs) workloads=("$workload") ;;
    *) die "expected --workload ag, rs or both" ;;
esac
[[ "$gpus" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))+$ ]] || die "--gpus needs at least two comma-separated GPU IDs"
IFS=, read -r -a gpu_list <<<"$gpus"
seen=,
for gpu in "${gpu_list[@]}"; do
    [[ "$seen" != *",${gpu},"* ]] || die "duplicate GPU ${gpu} in --gpus"
    seen+="${gpu},"
done
nranks="${#gpu_list[@]}"
[[ "$size_mib" =~ ^[1-9][0-9]*$ ]] || die "--size-mib must be a positive integer"

output_dir="${output_dir:-${repo_root}/results/gb200/ncu-torch-all-gather/ag-rs-protocol-$(date +%Y%m%d-%H%M%S)}"
[[ "$output_dir" = /* ]] || output_dir="${PWD}/${output_dir}"
qualified_kernel=false
if ! $dry_run; then
    ncu_home="${repo_root}/NVIDIA-Nsight-Compute-2026.4"
    [[ ! -d "$ncu_home" ]] || export PATH="${ncu_home}:${PATH}"
    ncu_bin="$(command -v "$ncu_bin")" || die "ncu not found; pass --ncu PATH"
    ncu_bin="$(cd -- "$(dirname -- "$ncu_bin")" && pwd)/$(basename -- "$ncu_bin")"
    ncu_version="$("$ncu_bin" --version)" || die "could not query the NCU version"
    [[ "$ncu_version" =~ Version[[:space:]]+([0-9]{4})\.([0-9]+) ]] || die "could not parse the NCU version"
    (( 10#${BASH_REMATCH[1]} > 2026 || (10#${BASH_REMATCH[1]} == 2026 && 10#${BASH_REMATCH[2]} >= 4) )) \
        || die "NCU 2026.4 or newer is required for shmem application replay"
    command -v uv >/dev/null || die "uv is required"
    mkdir -p -- "$output_dir"
    [[ -z "$(ls -A "$output_dir")" ]] || die "use an empty output directory: ${output_dir}"
    printf 'NCU executable: %s\n%s\n' "$ncu_bin" "$ncu_version" >"${output_dir}/ncu-version.txt"
fi
cd "$repo_root"

# Use the same symmetric kernels as the standalone launchers. Disable the experimental
# rank-0-only path and clock64 capture so all ranks contribute ordinary NVLink traffic.
unset NCCL_ALGO NCCL_PROTO NCCL_CTA_POLICY NCCL_NVTX_DISABLE NCCL_SYM_NOWIN_ENABLE \
    NCCL_SYM_MULTIMEM_RANK0_ONLY NCCL_SYM_CLOCK64_CSV_PROFILE NCCL_CE_COLL_AG_MULTICAST_THRESHOLD
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpus"
export NCCL_NVLS_ENABLE=1 NCCL_WIN_ENABLE=1 NCCL_SYM_CE_THRESHOLD=9223372036854775807
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV,TUNING
if ! $dry_run; then
    nccl_version="$(uv run --no-sync python -c '
import sys
from torch.distributed import ProcessGroupNCCL

sys.stdout.write(".".join(map(str, ProcessGroupNCCL.get_runtime_nccl_version())))
')" || die "could not query the NCCL library loaded by torch"
    [[ "$nccl_version" =~ ^([0-9]+)\.([0-9]+) ]] || die "could not parse NCCL version: ${nccl_version}"
    if (( 10#${BASH_REMATCH[1]} > 2 || (10#${BASH_REMATCH[1]} == 2 && 10#${BASH_REMATCH[2]} >= 32) )); then
        qualified_kernel=true
    fi
    printf 'Runtime NCCL: %s\n' "$nccl_version" | tee "${output_dir}/nccl-version.txt"
fi

# Match run_ncu_torch_p2p.sh so AG, RS, and P2P use the same NVLink counters.
metrics=()
for direction in nvlrx nvltx; do
    metrics+=("${direction}__bytes.sum")
    for packet in request response; do
        for kind in user protocol; do
            metrics+=("${direction}__bytes_packet_${packet}_data_${kind}.sum")
        done
    done
done
pm_metrics="$(printf 'pmsampling:%s,' "${metrics[@]}")"

for op in "${workloads[@]}"; do
    mode_dir="${output_dir}/${op}"
    common_args=(--size-mib "$size_mib" --dtype bf16 --warmup 5)
    if [[ "$op" == ag ]]; then
        kernel=AllGather_STMC nvtx_range=ncu_all_gather
        worker=("${script_dir}/ncu_torch_all_gather.py" --algo symm --iters 1 "${common_args[@]}")
    else
        kernel=ReduceScatter_LDMC nvtx_range=ncu_reduce_scatter
        worker=("${script_dir}/ncu_torch_reduce_scatter.py" "${common_args[@]}")
    fi
    export NCCL_SYM_KERNEL="$kernel"
    $qualified_kernel && export NCCL_SYM_KERNEL="${kernel%_*}:${kernel}"
    export NCCL_DEBUG_FILE="${mode_dir}/nccl-%h-%p.log"

    cmd=(
        "$ncu_bin" --communicator shmem --communicator-shmem-num-peers "$nranks"
        --replay-mode application --target-processes all --clock-control none
        --nvtx --nvtx-include "${nvtx_range}/" --kernel-name-base function
        --kernel-name "regex:^ncclSymkDevKernel_${kernel}(_[A-Za-z0-9_]+)?$" --launch-count 1
        --metrics "${pm_metrics%,}" --export "${mode_dir}/${op}"
        uv run --no-sync python -m torch.distributed.run --standalone --nnodes 1 --nproc-per-node "$nranks"
        "${worker[@]}"
    )
    printf '[%s] CUDA_VISIBLE_DEVICES=%s\n' "$op" "$gpus"
    if $dry_run; then
        printf 'NCCL_SYM_KERNEL=%s (NCCL < 2.32) or %s (NCCL >= 2.32)\n' "$kernel" "${kernel%_*}:${kernel}"
        printf ' %q' "${cmd[@]}"
        printf '\n'
        continue
    fi
    mkdir -- "$mode_dir"
    printf 'CUDA_VISIBLE_DEVICES=%q NCCL_SYM_KERNEL=%q ' "$CUDA_VISIBLE_DEVICES" "$NCCL_SYM_KERNEL" >"${mode_dir}/command.txt"
    printf '%q ' "${cmd[@]}" >>"${mode_dir}/command.txt"
    printf '\n' >>"${mode_dir}/command.txt"
    "${cmd[@]}" 2>&1 | tee "${mode_dir}/ncu.log"
    report="${mode_dir}/${op}.ncu-repz"
    [[ -s "$report" ]] || report="${mode_dir}/${op}.ncu-rep"
    [[ -s "$report" ]] || die "missing ${op} report under ${mode_dir}"
    "$ncu_bin" --import "$report" --page raw --csv --print-units base >"${mode_dir}/raw.csv"
done
$dry_run && exit 0

uv run --no-sync python - "$output_dir" "$gpus" "${workloads[@]}" -- "${metrics[@]}" <<'PYTHON'
import csv
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

output = Path(sys.argv[1])
gpus = sys.argv[2].split(",")
separator = sys.argv.index("--")
metrics = sys.argv[separator + 1 :]
summary = []
for workload in sys.argv[3:separator]:
    raw = output / workload / "raw.csv"
    seen = set()
    with raw.open(newline="") as handle:
        reader = csv.DictReader(handle)
        units = next(reader, {})
        for row in reader:
            device = row.get("launch__device_id") or ""
            if not device.isdecimal() or int(device) >= len(gpus) or device in seen:
                raise SystemExit(f"{raw}: unexpected or duplicate CUDA device {device!r}")
            seen.add(device)
            counts = []
            for metric in metrics:
                column = f"pmsampling:{metric}"
                if column not in row:
                    column = metric
                try:
                    value = Decimal((row.get(column) or "").replace(",", ""))
                except InvalidOperation:
                    raise SystemExit(f"{raw}: CUDA device {device} is missing {metric}") from None
                if units.get(column) not in ("byte", "bytes") or not value.is_finite() or value < 0:
                    raise SystemExit(f"{raw}: CUDA device {device} has invalid byte counts for {metric}")
                counts.append(value)
            values = dict(zip(metrics, counts))
            if any(values[f"{direction}__bytes.sum"] == 0 for direction in ("nvlrx", "nvltx")):
                raise SystemExit(f"{raw}: CUDA device {device} has zero TX or RX total bytes")
            if not any(count for metric, count in values.items() if metric.endswith("_protocol.sum")):
                raise SystemExit(f"{raw}: CUDA device {device} has zero bytes in all four Protocol counters")
            summary.append([workload, device, gpus[int(device)], *counts])
    if not seen:
        raise SystemExit(f"{raw}: no profiled CUDA device")

with (output / "nvlink_bytes.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["workload", "cuda_device", "gpu", *metrics])
    writer.writerows(summary)
PYTHON

echo "NVLink byte counts: ${output_dir}/nvlink_bytes.csv"
