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
Profile isolated NCCL AllGather_STMC and ReduceScatter_LDMC on one GB200 node.
Uses NCU 2026.4+ application replay, with one shmem report per operation by default.
By default, the rank0-only library lets rank 0 issue the multimem work while
every rank still participates in NCCL's barriers and hardware responses.

Usage: run_ncu_torch_ag_rs_rank0_only.sh [options]
  --gpus LIST       At least three GPU IDs (default: CUDA_VISIBLE_DEVICES or 0,1,2,3)
  --size-mib N      AG output / RS input per rank in MiB (default: 1024)
  --dtype NAME      bf16 | fp32 (default: bf16)
  --warmup N        Warmup launches per job (default: 1)
  --full            All ranks issue multimem work, for comparison
  --nccl-lib PATH   Preload this libnccl.so.2 (default in rank0-only mode:
                    nccl/nccl-2.32.3-rank0-only/build/lib/libnccl.so.2)
  --ncu PATH        Nsight Compute executable (default: ncu)
  --communicator shmem|tcp      Default: shmem. Use tcp for one report per rank.
  --pm-sampling-interval NS     PM sampling interval (default: NCU auto)
  --pm-sampling-buffer-size B   PM sampling buffer per GPU (default: NCU auto)
  --timeout SEC     Limit per AG or RS capture (default: 1800)
  --output-dir DIR  Default: results/gb200/ncu-torch-all-gather/ag-rs-rank0-only-<time>
  --dry-run         Show both commands without running them
  -h, --help

Writes reports, byte-unit raw CSVs, launch commands and logs under ag/ and rs/.
nvlink_bytes.csv has one row per operation and rank. The counts include both
TX/RX wire totals and all eight TX/RX x request/response x user/protocol sums.
USAGE
}

die() { echo "ERROR: $*" >&2; exit 1; }

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
gpus="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
size_mib=1024 dtype=bf16 warmup=1 ncu_bin=ncu timeout_sec=1800 nccl_lib="" output_dir=""
communicator=shmem pm_interval_ns="" pm_buffer_bytes=""
rank0_only=1
dry_run=false

while (( $# )); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --full) rank0_only=0; shift ;;
        --dry-run) dry_run=true; shift ;;
        --gpus|--size-mib|--dtype|--warmup|--nccl-lib|--ncu|--timeout|--output-dir|--communicator|--pm-sampling-interval|--pm-sampling-buffer-size)
            [[ -n "${2:-}" && "$2" != --* ]] || die "$1 needs a value"
            case "$1" in
                --gpus) gpus="$2" ;;
                --size-mib) size_mib="$2" ;;
                --dtype) dtype="$2" ;;
                --warmup) warmup="$2" ;;
                --nccl-lib) nccl_lib="$2" ;;
                --ncu) ncu_bin="$2" ;;
                --communicator) communicator="$2" ;;
                --pm-sampling-interval) pm_interval_ns="$2" ;;
                --pm-sampling-buffer-size) pm_buffer_bytes="$2" ;;
                --timeout) timeout_sec="$2" ;;
                --output-dir) output_dir="$2" ;;
            esac
            shift 2 ;;
        *) die "unknown argument: $1 (see --help)" ;;
    esac
done

[[ "$communicator" == shmem || "$communicator" == tcp ]] || die "expected --communicator shmem or tcp"
[[ "$gpus" =~ ^[^,[:space:]]+(,[^,[:space:]]+){2,}$ ]] || die "--gpus needs at least three comma-separated GPU IDs"
IFS=, read -r -a gpu_list <<<"$gpus"
nranks=${#gpu_list[@]}
# NCU sees CUDA ordinals after CUDA_VISIBLE_DEVICES remaps the requested GPUs.
printf -v ncu_devices '%s,' "${!gpu_list[@]}"
ncu_devices=${ncu_devices%,}
[[ "$(printf '%s\n' "${gpu_list[@]}" | sort -u | wc -l)" -eq "$nranks" ]] \
    || die "--gpus must list distinct devices"
[[ "$size_mib" =~ ^[1-9][0-9]*$ && "$warmup" =~ ^[1-9][0-9]*$ && "$timeout_sec" =~ ^[1-9][0-9]*$ ]] \
    || die "--size-mib, --warmup and --timeout must be positive integers"
[[ ( -z "$pm_interval_ns" || "$pm_interval_ns" =~ ^[1-9][0-9]*$ ) &&
   ( -z "$pm_buffer_bytes" || "$pm_buffer_bytes" =~ ^[1-9][0-9]*$ ) ]] \
    || die "--pm-sampling-interval and --pm-sampling-buffer-size must be positive integers"
[[ "$dtype" =~ ^(bf16|fp32)$ ]] || die "--dtype must be bf16 or fp32"
element_bytes=2
[[ "$dtype" == bf16 ]] || element_bytes=4
size_bytes=$(( size_mib * 1024 * 1024 ))
(( size_bytes % (nranks * element_bytes) == 0 )) || die "--size-mib cannot be split into equal $dtype shards"
shard_bytes=$(( size_bytes / nranks ))

issuer_mode=full
patched_banner='NCCL version 2.32.3-rank0-only+cuda'
if (( rank0_only )); then
    issuer_mode=rank0-only
    nccl_lib="${nccl_lib:-${repo_root}/nccl/nccl-2.32.3-rank0-only/build/lib/libnccl.so.2}"
fi
if [[ -n "$nccl_lib" && "$nccl_lib" != /* ]]; then
    nccl_lib="${PWD}/${nccl_lib}"
fi

output_dir="${output_dir:-${repo_root}/results/gb200/ncu-torch-all-gather/ag-rs-rank0-only-$(date +%Y%m%d-%H%M%S)}"
[[ "$output_dir" = /* ]] || output_dir="${PWD}/${output_dir}"
if ! $dry_run; then
    ncu_home="${repo_root}/NVIDIA-Nsight-Compute-2026.4"
    [[ ! -d "$ncu_home" ]] || export PATH="${ncu_home}:${PATH}"
    ncu_bin="$(command -v "$ncu_bin")" || die "ncu not found; pass --ncu PATH"
    ncu_bin="$(cd -- "$(dirname -- "$ncu_bin")" && pwd)/$(basename -- "$ncu_bin")"
    ncu_version="$("$ncu_bin" --version | sed -nE 's/.*Version ([0-9]+\.[0-9]+).*/\1/p')"
    [[ "$ncu_version" =~ ^[0-9]+\.[0-9]+$ ]] || die "could not read ncu version"
    IFS=. read -r ncu_year ncu_minor <<<"$ncu_version"
    (( ncu_year > 2026 || (ncu_year == 2026 && ncu_minor >= 4) )) || die "ncu 2026.4 or newer is required"
    command -v uv >/dev/null || die "uv is required"
    timeout_bin="$(command -v timeout || command -v gtimeout)" || die "GNU timeout is required"

    if [[ -n "$nccl_lib" ]]; then
        [[ -f "$nccl_lib" ]] || die "NCCL library not found: $nccl_lib (see install/install-nccl-2.32.3-rank0-only.sh)"
        if (( rank0_only )); then
            LC_ALL=C grep -aFq "$patched_banner" "$nccl_lib" || die "--nccl-lib must be the 2.32.3 rank0-only build"
        fi
    fi
fi
cd "$repo_root"
if [[ -n "$nccl_lib" ]]; then
    export LD_PRELOAD="${nccl_lib}${LD_PRELOAD:+:${LD_PRELOAD}}"
fi

# Inherit neither P2P/NCCL algorithm selections nor clock64/PAUSE experiments.
unset NCCL_ALGO NCCL_PROTO NCCL_CTA_POLICY NCCL_SYM_CTAS NCCL_NVTX_DISABLE NCCL_SYM_NOWIN_ENABLE
unset NCCL_SYM_CLOCK64_CSV_PROFILE
unset NCCL_SYM_AG_STMC_PAUSE_CYCLES NCCL_SYM_AG_ST_PAUSE_CYCLES
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpus"
export NCCL_NVLS_ENABLE=1 NCCL_WIN_ENABLE=1 NCCL_SYM_TMA_ENABLE=0 NCCL_SYM_CE_THRESHOLD=9223372036854775807
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV,TUNING NCCL_SYM_MULTIMEM_RANK0_ONLY="$rank0_only"

if $dry_run; then
    nccl_version=2.32.3
    echo "Dry run assumes NCCL $nccl_version syntax; preload=${nccl_lib:-<from torch>}"
else
    nccl_version="$(uv run --no-sync python -c '
import sys
from torch.distributed import ProcessGroupNCCL

sys.stdout.write(".".join(map(str, ProcessGroupNCCL.get_runtime_nccl_version())))
')" || die "could not read the NCCL version loaded by torch"
    [[ "$nccl_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "unrecognized runtime NCCL version: $nccl_version"
    if (( rank0_only )) && [[ "$nccl_version" != 2.32.3 ]]; then
        die "rank0-only requires runtime NCCL 2.32.3, got $nccl_version"
    fi
    mkdir -p -- "$output_dir"
    [[ -z "$(ls -A "$output_dir")" ]] || die "use an empty output directory: $output_dir"
    {
        echo "ncu=$ncu_bin $ncu_version NCCL=$nccl_version nccl_lib=${nccl_lib:-<from torch>}"
        echo "gpus=$gpus issuer_mode=$issuer_mode size_mib=$size_mib shard_bytes=$shard_bytes dtype=$dtype warmup=$warmup"
        echo "communicator=$communicator pm_sampling_interval_ns=${pm_interval_ns:-auto} pm_sampling_buffer_size_bytes=${pm_buffer_bytes:-auto}"
        env | grep -E '^(NCCL_|CUDA_VISIBLE_DEVICES=|LD_PRELOAD=)' | sort
    } >"$output_dir/config.txt"
fi
IFS=. read -r nccl_major nccl_minor _ <<<"$nccl_version"

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

free_port() {
    uv run --no-sync python -c '
import socket
import sys

with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    sys.stdout.write(str(sock.getsockname()[1]))
'
}

for mode in ag rs; do
    mode_dir="$output_dir/$mode"
    if [[ "$mode" == ag ]]; then
        operation=AllGather kernel=AllGather_STMC nvtx_range=ncu_all_gather
        worker=("${script_dir}/ncu_torch_all_gather.py" --algo symm --iters 1)
    else
        operation=ReduceScatter kernel=ReduceScatter_LDMC nvtx_range=ncu_reduce_scatter
        worker=("${script_dir}/ncu_torch_reduce_scatter.py")
    fi
    export NCCL_SYM_KERNEL="$kernel"
    if (( nccl_major > 2 || (nccl_major == 2 && nccl_minor >= 32) )); then
        export NCCL_SYM_KERNEL="${operation}:${kernel}"
    fi
    export NCCL_DEBUG_FILE="${mode_dir}/nccl-%h-%p.log"
    ncu_args=(
        --replay-mode application --target-processes all --clock-control none
        --nvtx --nvtx-include "${nvtx_range}/"
        --kernel-name-base function --kernel-name "regex:^ncclSymkDevKernel_${kernel}(_[A-Za-z0-9_]+)?$"
        --metrics "${pm_metrics%,}"
    )
    [[ -z "$pm_interval_ns" ]] || ncu_args+=(--pm-sampling-interval "$pm_interval_ns")
    [[ -z "$pm_buffer_bytes" ]] || ncu_args+=(--pm-sampling-buffer-size "$pm_buffer_bytes")
    worker+=(--size-mib "$size_mib" --dtype "$dtype" --warmup "$warmup")
    if [[ "$communicator" == shmem ]]; then
        # The worker issues one matching launch per rank inside NVTX. A global
        # launch count can limit the shared report to one device.
        report_names=(report)
        cmd=("$ncu_bin" --communicator shmem --communicator-shmem-num-peers "$nranks" --devices "$ncu_devices"
            "${ncu_args[@]}" --export "${mode_dir}/report"
            uv run --no-sync python -m torch.distributed.run --standalone --nnodes 1 --nproc-per-node "$nranks"
            "${worker[@]}")
    else
        export MASTER_ADDR=127.0.0.1 MASTER_PORT='<torch port>'
        ncu_port='<ncu port>'
        if ! $dry_run; then
            MASTER_PORT="$(free_port)"
            ncu_port="$(free_port)"
            while [[ "$ncu_port" == "$MASTER_PORT" ]]; do ncu_port="$(free_port)"; done
        fi
        report_names=()
        for rank in "${!gpu_list[@]}"; do report_names+=("rank${rank}-report"); done
        cmd=(bash "${script_dir}/ncu_tcp_ranks.sh" "$nranks" "$ncu_bin"
            --communicator tcp --communicator-tcp-num-peers "$nranks" --communicator-tcp-port "$ncu_port"
            --lockstep-kernel-launch --lockstep-nvtx-include "${nvtx_range}/"
            "${ncu_args[@]}" --launch-count 1 --export "${mode_dir}/rank%q{RANK}-report"
            -- uv run --no-sync python "${worker[@]}")
    fi
    echo "[$mode] gpus=$gpus issuer_mode=$issuer_mode NCCL_SYM_KERNEL=$NCCL_SYM_KERNEL NCCL_SYM_MULTIMEM_RANK0_ONLY=$rank0_only"
    printf ' %q' "${cmd[@]}"
    printf '\n'
    $dry_run && continue

    mkdir -- "$mode_dir"
    {
        if [[ "$communicator" == tcp ]]; then
            printf 'MASTER_ADDR=%q MASTER_PORT=%q ' "$MASTER_ADDR" "$MASTER_PORT"
        fi
        printf 'NCCL_SYM_KERNEL=%q NCCL_DEBUG_FILE=%q ' "$NCCL_SYM_KERNEL" "$NCCL_DEBUG_FILE"
        printf '%q ' "${cmd[@]}"
        printf '\n'
    } >"$mode_dir/command.txt"
    "$timeout_bin" --signal=TERM --kill-after=30 "$timeout_sec" "${cmd[@]}" 2>&1 | tee "$mode_dir/ncu.log" \
        || die "$mode: NCU capture failed; see $mode_dir/ncu.log"

    if (( rank0_only )); then
        shopt -s nullglob
        logs=("$mode_dir/ncu.log" "$mode_dir"/nccl-*.log)
        shopt -u nullglob
        grep -Fq "$patched_banner" "${logs[@]}" \
            || die "$mode: NCCL did not report the rank0-only build; see $mode_dir"
    fi
    # NCU can append its own number to the export name.
    for name in "${report_names[@]}"; do
        report=""
        for candidate in "$mode_dir/$name"*.ncu-rep{,z}; do
            [[ -f "$candidate" && -s "$candidate" ]] || continue
            [[ -z "$report" ]] || die "$mode: multiple NCU reports for $name: $report and $candidate"
            report="$candidate"
        done
        [[ -n "$report" ]] || die "$mode: missing NCU report under $mode_dir (expected ${name}*.ncu-rep[z])"
        csv_name=${name%-report}
        "$ncu_bin" --import "$report" --page raw --csv --print-units base --print-kernel-base function \
            >"$mode_dir/$csv_name.csv"
    done
done
$dry_run && exit 0

uv run --no-sync python - "$output_dir" "$gpus" "$issuer_mode" "$shard_bytes" "$communicator" "${metrics[@]}" <<'PYTHON'
import csv
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

output = Path(sys.argv[1])
gpus = sys.argv[2].split(",")
issuer_mode = sys.argv[3]
shard_bytes = int(sys.argv[4])
communicator = sys.argv[5]
metrics = sys.argv[6:]
rank_by_device = {str(rank): rank for rank in range(len(gpus))}
summary = []
csv.field_size_limit(sys.maxsize)

for mode, kernel in (("ag", "AllGather_STMC"), ("rs", "ReduceScatter_LDMC")):
    mode_dir = output / mode
    raw_files = [(rank, mode_dir / f"rank{rank}.csv") for rank in range(len(gpus))]
    if communicator == "shmem":
        raw_files = [(None, mode_dir / "report.csv")]

    results = {}
    for file_rank, raw in raw_files:
        with raw.open(newline="") as handle:
            rows = csv.DictReader(handle)
            units = next(rows, {})
            for row in rows:
                device = row.get("launch__device_id")
                rank = rank_by_device.get(device) if file_rank is None else file_rank
                if rank is None or rank in results:
                    raise SystemExit(f"{raw}: expected one {kernel} result per rank, got visible GPU {device!r}")
                name = row.get("Kernel Name", "")
                if not re.fullmatch(rf"ncclSymkDevKernel_{kernel}(?:_[A-Za-z0-9_]+)?", name):
                    raise SystemExit(f"{raw}: expected {kernel}, got {name!r}")

                counts = {}
                for metric in metrics:
                    column = f"pmsampling:{metric}"
                    if column not in row:
                        column = metric
                    try:
                        count = Decimal((row.get(column) or "").replace(",", ""))
                    except InvalidOperation:
                        raise SystemExit(f"{raw}: rank {rank}: missing numeric value for {column}") from None
                    if units.get(column) not in ("byte", "bytes") or not count.is_finite() or count < 0 or count % 1:
                        raise SystemExit(f"{raw}: rank {rank}: invalid byte count for {column}")
                    counts[metric] = int(count)
                results[rank] = counts

    missing = sorted(set(range(len(gpus))) - results.keys())
    if missing:
        hint = " Retry with --communicator tcp and a fresh --output-dir." if communicator == "shmem" else ""
        raise SystemExit(f"{mode}: missing {kernel} results for ranks {missing}.{hint}")
    for rank, gpu in enumerate(gpus):
        counts = results[rank]
        issues_multimem = rank == 0 or issuer_mode == "full"
        if mode == "ag":
            payload_metric = "nvltx__bytes_packet_request_data_user.sum" if issues_multimem else (
                "nvlrx__bytes_packet_request_data_user.sum"
            )
        else:
            payload_metric = "nvlrx__bytes_packet_response_data_user.sum" if issues_multimem else (
                "nvltx__bytes_packet_response_data_user.sum"
            )
        if counts[payload_metric] == 0:
            raise SystemExit(f"{mode}: rank {rank}: counted zero payload bytes in {payload_metric}")
        role = "issuer" if issues_multimem else ("receiver" if mode == "ag" else "contributor")
        summary.append([mode, issuer_mode, rank, gpu, role, shard_bytes, *(counts[metric] for metric in metrics)])

with (output / "nvlink_bytes.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["mode", "issuer_mode", "rank", "gpu", "role", "shard_bytes", *metrics])
    writer.writerows(summary)
PYTHON

echo "NVLink byte counts: $output_dir/nvlink_bytes.csv"
