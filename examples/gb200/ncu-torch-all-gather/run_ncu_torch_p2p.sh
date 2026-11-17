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
Profile one 1024 MiB NCCL P2P transfer per selected mode on a GB200 node.
GPU0 is the sender and GPU1 is the receiver in both modes. Requires ncu 2026.4 or newer.
Uses TCP application replay, with one NCU process and report per rank.

Usage: run_ncu_torch_p2p.sh [options]
  --ncu PATH                   Nsight Compute executable (default: ncu)
  --communicator tcp           Explicitly select TCP (default)
  --workload push|pull|both    Select modes to profile (default: both, push then pull)
  --pm-sampling-interval NS    PM sampling interval (default: 10000 ns)
  --pm-sampling-buffer-size B  PM sampling buffer per GPU (default: 33554432 bytes)
  --timeout SEC                Limit per push or pull capture (default: 1800)
  --output-dir DIR             Default: results/gb200/ncu-torch-all-gather/p2p-<time>
  --dry-run                    Show the selected commands without running them
  -h, --help

Writes per-rank NCU reports, raw CSVs, launch commands and logs under each selected mode.
After checking PM sampling coverage, nvlink_bytes.csv gives one row per rank and mode.
Values use byte .sum metrics, including all eight
TX/RX x request/response x user/protocol categories and the TX/RX totals.
USAGE
}

die() { echo "ERROR: $*" >&2; exit 1; }

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"

ncu_bin=ncu output_dir="" timeout_sec=1800 workload=both
pm_interval_ns=10000 pm_buffer_bytes=$((32 * 1024 * 1024))
dry_run=false
while (( $# )); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --dry-run) dry_run=true; shift ;;
        --ncu|--output-dir|--communicator|--workload|--pm-sampling-interval|--pm-sampling-buffer-size|--timeout)
            [[ -n "${2:-}" && "$2" != --* ]] || die "$1 needs a value"
            case "$1" in
                --ncu) ncu_bin="$2" ;;
                --output-dir) output_dir="$2" ;;
                --communicator) [[ "$2" == tcp ]] || die "expected --communicator tcp" ;;
                --workload) workload="$2" ;;
                --pm-sampling-interval) pm_interval_ns="$2" ;;
                --pm-sampling-buffer-size) pm_buffer_bytes="$2" ;;
                --timeout) timeout_sec="$2" ;;
            esac
            shift 2 ;;
        *) die "unknown argument: $1 (see --help)" ;;
    esac
done
case "$workload" in
    both) workloads=(push pull) ;;
    push|pull) workloads=("$workload") ;;
    *) die "expected --workload push, pull or both" ;;
esac
[[ "$pm_interval_ns" =~ ^[1-9][0-9]*$ && "$pm_buffer_bytes" =~ ^[1-9][0-9]*$ && "$timeout_sec" =~ ^[1-9][0-9]*$ ]] \
    || die "PM sampling interval, buffer size and timeout must be positive integers"

output_dir="${output_dir:-${repo_root}/results/gb200/ncu-torch-all-gather/p2p-$(date +%Y%m%d-%H%M%S)}"
[[ "$output_dir" = /* ]] || output_dir="${PWD}/${output_dir}"
if ! $dry_run; then
    ncu_home="${repo_root}/NVIDIA-Nsight-Compute-2026.4"
    [[ ! -d "$ncu_home" ]] || export PATH="${ncu_home}:${PATH}"
    ncu_bin="$(command -v "$ncu_bin")" || die "ncu not found; pass --ncu PATH"
    ncu_bin="$(cd -- "$(dirname -- "$ncu_bin")" && pwd)/$(basename -- "$ncu_bin")"
    command -v uv >/dev/null || die "uv is required"
    timeout_bin="$(command -v timeout || command -v gtimeout)" || die "GNU timeout is required"
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
export MASTER_ADDR=127.0.0.1
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

reports=()
for mode in "${workloads[@]}"; do
    mode_dir="${output_dir}/${mode}"
    # Each application replay pass has new worker PIDs, as in the AllGather launcher.
    export NCCL_DEBUG_FILE="${mode_dir}/nccl-%h-%p.log"
    export NCCL_P2P_READ_ENABLE=0
    [[ "$mode" == push ]] || export NCCL_P2P_READ_ENABLE=1

    export MASTER_PORT='<torch port>'
    ncu_port='<ncu port>'
    if ! $dry_run; then
        MASTER_PORT="$(free_port)"
        ncu_port="$(free_port)"
        while [[ "$ncu_port" == "$MASTER_PORT" ]]; do ncu_port="$(free_port)"; done
    fi

    cmd=(bash "${script_dir}/ncu_tcp_ranks.sh" 2 "$ncu_bin"
        --communicator tcp --communicator-tcp-num-peers 2 --communicator-tcp-port "$ncu_port"
        --replay-mode application --target-processes all --clock-control none
        --nvtx --nvtx-include "${nvtx_range}/"
        --lockstep-kernel-launch --lockstep-nvtx-include "${nvtx_range}/"
        --kernel-name-base function --kernel-name 'regex:^ncclDevKernel_SendRecv' --launch-count 1
        --pm-sampling-interval "$pm_interval_ns" --pm-sampling-buffer-size "$pm_buffer_bytes"
        --metrics "${pm_metrics%,}" --export "${mode_dir}/rank%q{RANK}-report"
        -- uv run --no-sync python "${script_dir}/ncu_torch_p2p.py"
        --workload "$mode" --size-mib 1024 --dtype bf16 --warmup 1 --iters 1)
    printf '[%s] CUDA_VISIBLE_DEVICES=%s NCCL_P2P_READ_ENABLE=%s\n' "$mode" "$CUDA_VISIBLE_DEVICES" "$NCCL_P2P_READ_ENABLE"
    printf ' %q' "${cmd[@]}"
    printf '\n'
    $dry_run && continue

    mkdir -- "$mode_dir"
    {
        printf 'MASTER_ADDR=%q MASTER_PORT=%q CUDA_VISIBLE_DEVICES=%q NCCL_P2P_READ_ENABLE=%q NCCL_DEBUG_FILE=%q ' \
            "$MASTER_ADDR" "$MASTER_PORT" "$CUDA_VISIBLE_DEVICES" "$NCCL_P2P_READ_ENABLE" "$NCCL_DEBUG_FILE"
        printf '%q ' "${cmd[@]}"
        printf '\n'
    } >"${mode_dir}/command.txt"
    "$timeout_bin" --signal=TERM --kill-after=30 "$timeout_sec" "${cmd[@]}" 2>&1 | tee "${mode_dir}/ncu.log" \
        || die "$mode: NCU capture failed; see ${mode_dir}/ncu.log"

    for rank in 0 1; do
        report=""
        for candidate in "$mode_dir/rank${rank}-report"*.ncu-rep{,z}; do
            [[ -f "$candidate" && -s "$candidate" ]] || continue
            [[ -z "$report" ]] || die "$mode: multiple rank $rank NCU reports: $report and $candidate"
            report="$candidate"
        done
        [[ -n "$report" ]] || die "$mode: missing rank $rank NCU report under ${mode_dir}"
        "$ncu_bin" --import "$report" --page raw --csv --print-units base --print-kernel-base function \
            >"${mode_dir}/rank${rank}.csv"
        reports+=("$report")
    done
done
$dry_run && exit 0

uv run --no-sync python - "$output_dir" "$ncu_bin" "${workloads[*]}" "${metrics[*]}" "${reports[@]}" <<'PYTHON'
import csv
import io
import re
import subprocess
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

output = Path(sys.argv[1])
ncu_bin = sys.argv[2]
workloads = sys.argv[3].split()
metrics = sys.argv[4].split()
reports = iter(sys.argv[5:])
payload_bytes = 1024 * 1024**2
summary = []
csv.field_size_limit(sys.maxsize)


def metric_instances(cell):
    # NCU details format: aggregate (correlation ID: value; correlation ID: value).
    _, _, values = cell.partition(" (")
    if values:
        for entry in values.removesuffix(")").split("; "):
            yield entry.split(": ", 1)


def check_sample_coverage(report):
    # Use the workload timestamps from each metric's own PM replay group.
    detailed_csv = subprocess.check_output(
        [ncu_bin, "--import", report, "--page", "raw", "--csv", "--print-units", "base",
         "--print-metric-instances", "details"],
        text=True,
    )
    rows = csv.DictReader(io.StringIO(detailed_csv))
    next(rows, None)  # Units row.
    detail = next(rows, {})
    group_by_metric = {
        column: group
        for group, names in metric_instances(detail.get("profiler__pmsampler_pass_groups", ""))
        for column in re.findall(r"pmsampling:[\w.]+", names)
    }
    interval_by_group = dict(metric_instances(detail.get("profiler__pmsampler_interval_time", "")))
    for metric in metrics:
        column = f"pmsampling:{metric}"
        try:
            group = group_by_metric[column]
            interval = int(interval_by_group[group])
            start, end = (
                int(next(metric_instances(detail[f"profiler__timestamp_workload_{edge}_{group}"]))[0].split()[0])
                for edge in ("start", "end")
            )
            timestamps = [int(key.split()[0]) for key, _ in metric_instances(detail[column])]
            first, last = timestamps[0], timestamps[-1]
        except (KeyError, IndexError, StopIteration, ValueError):
            raise SystemExit(f"{report}: missing PM coverage data for {column}") from None
        if interval <= 0 or end <= start or first > start + interval or last < end - interval:
            raise SystemExit(
                f"{report}: incomplete PM sampling for {column} (first sample {first - start:+,} ns from "
                f"kernel start, last sample {last - end:+,} ns from kernel end). "
                "Increase --pm-sampling-buffer-size or --pm-sampling-interval and rerun."
            )


for mode in workloads:
    payload_packet = "request" if mode == "push" else "response"
    for rank, role, direction in ((0, "sender", "nvltx"), (1, "receiver", "nvlrx")):
        raw = output / mode / f"rank{rank}.csv"
        check_sample_coverage(next(reports))
        with raw.open(newline="") as handle:
            rows = csv.DictReader(handle)
            units = next(rows, {})
            actions = list(rows)
        if len(actions) != 1:
            raise SystemExit(f"{raw}: expected exactly one SendRecv result, got {len(actions)}")
        row = actions[0]
        gpu = row.get("launch__device_id")
        if not row.get("Kernel Name", "").startswith("ncclDevKernel_SendRecv") or gpu != str(rank):
            raise SystemExit(f"{raw}: expected SendRecv on GPU{rank}, got {row.get('Kernel Name')!r} on GPU {gpu!r}")

        counts = {}
        for metric in metrics:
            column = f"pmsampling:{metric}"
            try:
                count = Decimal((row.get(column) or "").replace(",", ""))
            except InvalidOperation:
                raise SystemExit(f"{raw}: missing numeric value for {column}") from None
            if units.get(column) not in ("byte", "bytes") or not count.is_finite() or count < 0 or count % 1:
                raise SystemExit(f"{raw}: invalid byte count for {column}")
            counts[metric] = int(count)
        payload_metric = f"{direction}__bytes_packet_{payload_packet}_data_user.sum"
        if counts[payload_metric] == 0:
            raise SystemExit(f"{raw}: counted zero payload bytes in {payload_metric}")
        summary.append([mode, rank, gpu, role, payload_bytes, *(counts[metric] for metric in metrics)])

with (output / "nvlink_bytes.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["mode", "rank", "gpu", "role", "expected_payload_bytes", *metrics])
    writer.writerows(summary)
PYTHON

echo "NVLink byte counts: ${output_dir}/nvlink_bytes.csv"
