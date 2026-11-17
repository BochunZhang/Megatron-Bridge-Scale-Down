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
Profile one 1024 MiB NCCL P2P transfer per mode, push then pull, on a GB200 node.
GPU0 is the sender and GPU1 is the receiver in both modes. Requires ncu 2026.4 or newer.

Usage: run_ncu_torch_p2p_backup.sh [options]
  --ncu PATH               Nsight Compute executable (default: ncu)
  --output-dir DIR         Default: results/gb200/ncu-torch-all-gather/p2p-<time>
  --communicator shmem|tcp Default: shmem. Use tcp for one NCU report per rank.
  --dry-run                Show both commands without running them
  -h, --help

Writes raw NCU CSVs and reports under push/ and pull/, plus nvlink_bytes.csv with
one row per GPU and mode. Values use byte .sum metrics, including all eight
TX/RX x request/response x user/protocol categories and the TX/RX totals.
USAGE
}

die() { echo "ERROR: $*" >&2; exit 1; }

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"

ncu_bin=ncu
output_dir=""
communicator=shmem
dry_run=false
while (( $# )); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --dry-run) dry_run=true; shift ;;
        --ncu|--output-dir|--communicator)
            [[ -n "${2:-}" && "$2" != --* ]] || die "$1 needs a value"
            case "$1" in
                --ncu) ncu_bin="$2" ;;
                --output-dir) output_dir="$2" ;;
                --communicator) communicator="$2" ;;
            esac
            shift 2 ;;
        *) die "unknown argument: $1 (see --help)" ;;
    esac
done
[[ "$communicator" == shmem || "$communicator" == tcp ]] || die "expected --communicator shmem or tcp"

output_dir="${output_dir:-${repo_root}/results/gb200/ncu-torch-all-gather/p2p-$(date +%Y%m%d-%H%M%S)}"
[[ "$output_dir" = /* ]] || output_dir="${PWD}/${output_dir}"
if ! $dry_run; then
    ncu_home="${repo_root}/NVIDIA-Nsight-Compute-2026.4"
    [[ ! -d "$ncu_home" ]] || export PATH="${ncu_home}:${PATH}"
    ncu_bin="$(command -v "$ncu_bin")" || die "ncu not found; pass --ncu PATH"
    ncu_bin="$(cd -- "$(dirname -- "$ncu_bin")" && pwd)/$(basename -- "$ncu_bin")"
    command -v uv >/dev/null || die "uv is required"
    mkdir -p -- "$output_dir"
    [[ -z "$(ls -A "$output_dir")" ]] || die "use an empty output directory: ${output_dir}"
fi
cd "$repo_root"

metrics=()
for direction in nvlrx nvltx; do
    metrics+=("${direction}__bytes.sum")
    for packet in request response; do
        for data in user protocol; do
            metrics+=("${direction}__bytes_packet_${packet}_data_${data}.sum")
        done
    done
done
pm_metrics="$(printf 'pmsampling:%s,' "${metrics[@]}")"

export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1
export NCCL_NVLS_ENABLE=0 NCCL_P2P_USE_CUDA_MEMCPY=0 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,P2P,ENV
nvtx_range=ncu_p2p

free_port() {
    uv run --no-sync python -c '
import socket
import sys

with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    sys.stdout.write(str(sock.getsockname()[1]))
'
}

for mode in push pull; do
    mode_dir="${output_dir}/${mode}"
    # Each application replay pass has new worker PIDs, as in the AllGather launcher.
    export NCCL_DEBUG_FILE="${mode_dir}/nccl-%h-%p.log"
    export NCCL_P2P_READ_ENABLE=0
    [[ "$mode" == push ]] || export NCCL_P2P_READ_ENABLE=1

    ncu_args=(
        --replay-mode application --target-processes all --clock-control none
        --nvtx --nvtx-include "${nvtx_range}/"
        --kernel-name-base function --kernel-name 'regex:^ncclDevKernel_SendRecv' --launch-count 1
        --metrics "${pm_metrics%,}"
    )
    worker_cmd=("${script_dir}/ncu_torch_p2p.py"
        --workload "$mode" --size-mib 1024 --dtype bf16 --warmup 1 --iters 1)
    if [[ "$communicator" == shmem ]]; then
        report_names=(p2p)
        cmd=("$ncu_bin" --communicator shmem --communicator-shmem-num-peers 2 --devices 0,1
            "${ncu_args[@]}" --export "${mode_dir}/p2p"
            uv run --no-sync python -m torch.distributed.run --standalone --nnodes 1 --nproc-per-node 2
            "${worker_cmd[@]}")
    else
        export MASTER_ADDR=127.0.0.1 MASTER_PORT='<torch port>'
        ncu_port='<ncu port>'
        if ! $dry_run; then
            MASTER_PORT="$(free_port)"
            ncu_port="$(free_port)"
            while [[ "$ncu_port" == "$MASTER_PORT" ]]; do ncu_port="$(free_port)"; done
        fi
        report_names=(p2p-rank0 p2p-rank1)
        cmd=(bash "${script_dir}/ncu_tcp_ranks.sh" 2 "$ncu_bin"
            --communicator tcp --communicator-tcp-num-peers 2 --communicator-tcp-port "$ncu_port"
            --lockstep-kernel-launch --lockstep-nvtx-include "${nvtx_range}/"
            "${ncu_args[@]}" --export "${mode_dir}/p2p-rank%q{RANK}"
            -- uv run --no-sync python "${worker_cmd[@]}")
    fi
    printf '[%s] CUDA_VISIBLE_DEVICES=%s NCCL_P2P_READ_ENABLE=%s\n' "$mode" "$CUDA_VISIBLE_DEVICES" "$NCCL_P2P_READ_ENABLE"
    printf ' %q' "${cmd[@]}"
    printf '\n'
    $dry_run && continue

    mkdir -- "$mode_dir"
    "${cmd[@]}" 2>&1 | tee "${mode_dir}/ncu.log"
    for name in "${report_names[@]}"; do
        report="${mode_dir}/${name}.ncu-repz"
        [[ -f "$report" ]] || report="${mode_dir}/${name}.ncu-rep"
        [[ -f "$report" ]] || die "missing ${name} report under ${mode_dir}"
        "$ncu_bin" --import "$report" --page raw --csv --print-units base >"${mode_dir}/${name}.csv"
    done
done
$dry_run && exit 0

uv run --no-sync python - "$output_dir" "${report_names[@]}" -- "${metrics[@]}" <<'PYTHON'
import csv
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

output = Path(sys.argv[1])
separator = sys.argv.index("--")
reports = sys.argv[2:separator]
metrics = sys.argv[separator + 1 :]
summary = []

for mode, payload_packet in (("push", "request"), ("pull", "response")):
    results = {}
    for report in reports:
        raw = output / mode / f"{report}.csv"
        with raw.open(newline="") as handle:
            rows = csv.DictReader(handle)
            units = next(rows, {})
            for row in rows:
                gpu = row.get("launch__device_id")
                if gpu not in ("0", "1") or gpu in results:
                    raise SystemExit(f"{raw}: expected exactly one SendRecv result per GPU, got GPU {gpu!r}")
                counts = {}
                for metric in metrics:
                    column = f"pmsampling:{metric}"
                    if column not in row:
                        column = metric
                    try:
                        count = Decimal((row.get(column) or "").replace(",", ""))
                    except InvalidOperation:
                        raise SystemExit(f"{raw}: GPU{gpu} has no numeric value for {metric}") from None
                    if units.get(column) not in ("byte", "bytes") or not count.is_finite() or count < 0:
                        raise SystemExit(f"{raw}: GPU{gpu} has an invalid byte count for {metric}")
                    counts[metric] = count
                results[gpu] = counts

    if set(results) != {"0", "1"}:
        hint = " Retry with --communicator tcp and a fresh --output-dir." if len(reports) == 1 else ""
        raise SystemExit(f"{mode}: expected GPU0 and GPU1 results, got {sorted(results)}.{hint}")
    for gpu, role, direction in (("0", "sender", "nvltx"), ("1", "receiver", "nvlrx")):
        counts = results[gpu]
        payload_metric = f"{direction}__bytes_packet_{payload_packet}_data_user.sum"
        if counts[payload_metric] == 0:
            raise SystemExit(f"{mode}: GPU{gpu} counted zero payload bytes in {payload_metric}")
        summary.append([mode, gpu, role, 1024 * 1024**2, *(counts[metric] for metric in metrics)])

with (output / "nvlink_bytes.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["mode", "gpu", "role", "expected_payload_bytes", *metrics])
    writer.writerows(summary)
PYTHON

echo "NVLink byte counts: ${output_dir}/nvlink_bytes.csv"
