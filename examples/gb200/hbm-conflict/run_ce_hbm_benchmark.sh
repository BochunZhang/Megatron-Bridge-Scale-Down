#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
    cat <<'EOF'
Profile GB200 BF16 GEMM and copy-engine traffic with Nsight Systems 2026.6.1.

Usage: bash run_ce_hbm_benchmark.sh [options]
  --gemm SHAPES       '[M,N,K]' or '[M,N,K],[M,N,K]' (default: '[8192,8192,7168]')
  --gpus IDS          Ordered GPU IDs (default: 0,1,2,3 or the first four visible GPUs)
  --cases IDS         Comma-separated case IDs (default: task1,task2,task3,task4,baseline1,baseline2,baseline3)
  --size-mib N        H2D size and full AG output size per rank (default: 1024)
  --warmup N          Standalone warmup iterations per active lane (default: 5)
  --gemm-rounds N     GEMMs per rank and shape (default: 20)
  --comm-rounds N     Standalone transfers, or minimum transfers when overlapped (default: 25)
  --nsys PATH         Nsight Systems 2026.6.1 executable (also NSYS_BIN)
  --nsys-freq HZ      GPU metrics sampling rate (default: 50000)
  --output-dir DIR    Results directory (default: results/gb200/hbm-conflict/gpu-num-N/<timestamp>-ce)
  --dry-run           Show commands without starting jobs or changing GPU clocks
  -h, --help          Show this help

task1 GEMM, task2 GEMM+H2D, task3 GEMM+HBM-source CE multicast AG,
task4 GEMM+host-source CE multicast AG. Baselines run H2D, HBM-source AG,
and host-source AG alone, respectively. Multi-GPU cases run on one host.
The launcher saves raw Nsight reports. Run analyse_ce_hbm.py separately for SQLite export and analysis.
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 2; }
show_command() { printf '%q ' "$@"; printf '\n'; }

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../../.." && pwd)"
NSYS_BIN="${NSYS_BIN:-NsightSystems-2026.6.1/bin/nsys}"
gpu_ids="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=, read -r -a default_gpus <<< "$gpu_ids"
gpu_ids="$(IFS=,; printf '%s' "${default_gpus[*]:0:4}")"
case_ids=task1,task2,task3,task4,baseline1,baseline2,baseline3
gemms='[8192,8192,7168]'
size_mib=1024 warmup=5 gemm_rounds=20 comm_rounds=25 nsys_freq=50000
output_dir="" dry_run=false

while (( $# )); do
    case "$1" in
        --gemm|--gpus|--cases|--size-mib|--warmup|--gemm-rounds|--comm-rounds|--nsys|--nsys-freq|--output-dir)
            (( $# >= 2 )) && [[ -n "$2" ]] || die "$1 requires a value"
            case "$1" in
                --gemm) gemms="$2" ;;
                --gpus) gpu_ids="$2" ;;
                --cases) case_ids="$2" ;;
                --size-mib) size_mib="$2" ;;
                --warmup) warmup="$2" ;;
                --gemm-rounds) gemm_rounds="$2" ;;
                --comm-rounds) comm_rounds="$2" ;;
                --nsys) NSYS_BIN="$2" ;;
                --nsys-freq) nsys_freq="$2" ;;
                --output-dir) output_dir="$2" ;;
            esac
            shift 2 ;;
        --dry-run) dry_run=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument $1" ;;
    esac
done

for value in "$size_mib" "$warmup" "$gemm_rounds" "$comm_rounds" "$nsys_freq"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "expected a positive integer, got $value"
done
(( nsys_freq >= 10 && nsys_freq <= 200000 )) || die "--nsys-freq must be in 10..200000"
[[ "$gpu_ids" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ ]] || die "use comma-separated numeric GPU IDs"
IFS=, read -r -a gpus <<< "$gpu_ids"
seen_gpus=,
for gpu in "${gpus[@]}"; do
    [[ "$seen_gpus" != *",$gpu,"* ]] || die "duplicate GPU ID $gpu"
    seen_gpus+="$gpu,"
done

case_re='(task[1-4]|baseline[1-3])'
[[ "$case_ids" =~ ^${case_re}(,${case_re})*$ ]] || die "invalid case list $case_ids"
IFS=, read -r -a cases <<< "$case_ids"
lock_gpus="${gpus[0]}"
seen_cases=,
for case_id in "${cases[@]}"; do
    [[ "$seen_cases" != *",$case_id,"* ]] || die "duplicate case $case_id"
    seen_cases+="$case_id,"
    case "$case_id" in
        task[34]|baseline[23])
            (( ${#gpus[@]} >= 2 )) || die "$case_id needs at least two GPUs"
            lock_gpus="$gpu_ids"
            ;;
    esac
done

remaining="${gemms//[[:space:]]/}"
shape_re='^\[([1-9][0-9]*),([1-9][0-9]*),([1-9][0-9]*)\](,|$)'
shapes=() shape_names=()
while [[ -n "$remaining" ]]; do
    [[ "$remaining" =~ $shape_re ]] || die "invalid --gemm: $gemms"
    shapes+=("[${BASH_REMATCH[1]},${BASH_REMATCH[2]},${BASH_REMATCH[3]}]")
    shape_names+=("m${BASH_REMATCH[1]}-n${BASH_REMATCH[2]}-k${BASH_REMATCH[3]}")
    match="${BASH_REMATCH[0]}"
    remaining="${remaining:${#match}}"
    [[ "$match" != *, || -n "$remaining" ]] || die "trailing comma in --gemm: $gemms"
done
(( ${#shapes[@]} )) || die "--gemm needs at least one shape"

output_dir="${output_dir:-$repo_root/results/gb200/hbm-conflict/gpu-num-${#gpus[@]}/$(date +%Y%m%d-%H%M%S)-ce}"
[[ "$output_dir" = /* ]] || output_dir="$PWD/$output_dir"
[[ "$NSYS_BIN" = /* ]] || NSYS_BIN="$PWD/$NSYS_BIN"
export NSYS_BIN CUDA_DEVICE_ORDER=PCI_BUS_ID
lock_command=(nvidia-smi -i "$lock_gpus" --lock-gpu-clocks=tdp,tdp)
clock_query=(nvidia-smi -i "$lock_gpus" --query-gpu=index,pci.bus_id,name,uuid,clocks.current.sm,clocks.current.memory,power.limit --format=csv)

if $dry_run; then
    show_command "${lock_command[@]}"
else
    [[ -x "$NSYS_BIN" ]] || die "Nsight Systems executable not found at $NSYS_BIN"
    nsys_version="$("$NSYS_BIN" --version)"
    [[ "$nsys_version" =~ (^|[^0-9.])2026\.6\.1([^0-9]|$) ]] || die "expected Nsight Systems 2026.6.1, got $nsys_version"
    command -v uv >/dev/null || die "uv is required"
    source "$repo_root/scripts/scale-down/common/gloo_socket_ifname.sh"
    # Accept the spelling used in launch requests, but PyTorch reads GLOO_SOCKET_IFNAME.
    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${GLOO_SOCKET_NAME:-}}"
    available_ifnames=" $(gloo_available_ifnames) "
    if [[ "$available_ifnames" != *" ${GLOO_SOCKET_IFNAME:-<auto>} "*
        && "$available_ifnames" != *" eth0 "* && "$available_ifnames" != *" bond1 "* ]]; then
        export GLOO_SOCKET_IFNAME=lo
    fi
    resolve_gloo_socket_ifname 1
    mkdir -p -- "$(dirname -- "$output_dir")"
    mkdir -- "$output_dir" || die "use a new results directory"
    printf '%s\n' "$nsys_version" "GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME" "CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER" \
        "gpu_ids=$gpu_ids" "lock_gpus=$lock_gpus" "gemm=$gemms" "cases=$case_ids" \
        "size_mib=$size_mib" "warmup=$warmup" "gemm_rounds=$gemm_rounds" "comm_rounds=$comm_rounds" \
        "nsys_freq=$nsys_freq" "cwd=$PWD" >"$output_dir/run_info.txt"
    show_command "${lock_command[@]}" >>"$output_dir/run_info.txt"
    "${lock_command[@]}" 2>&1 | tee "$output_dir/gpu_clock_lock.log"
    "${clock_query[@]}" >"$output_dir/gpu_clocks.before.csv"
    nvidia-smi topo -m >"$output_dir/gpu_topology.txt"
    "$NSYS_BIN" profile --gpu-metrics-devices="$lock_gpus" --gpu-metrics-set=help \
        >"$output_dir/nsys-gpu-metrics.txt" 2>&1 || true
fi

# The outer uv process places its PyTorch environment on PATH for every worker.
# Each nsys process samples only its rank's physical GPU.
profile_worker='
set -euo pipefail
IFS=, read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
exec "$NSYS_BIN" profile --gpu-metrics-devices="${devices[$LOCAL_RANK]}" "$@"
'

run_case() {
    local name="$1" mode="$2" devices="$3" ranks="$4" shape="${5:-}"
    local case_dir="$output_dir/$name" rank
    local -a command=(uv run --active --no-sync python -m torch.distributed.run --standalone --nnodes=1
        --nproc_per_node="$ranks" --no-python bash -c "$profile_worker" hbm-ce
        --trace=cuda,nvtx --cuda-trace-all-apis=true --sample=none --cpuctxsw=none
        --capture-range=cudaProfilerApi --capture-range-end=stop
        --gpu-metrics-set=gb10x-ct --gpu-metrics-frequency="$nsys_freq" --output="$case_dir/nsys/$name.rank_%q{RANK}"
        python "$script_dir/ce_hbm_benchmark.py" --case "$name" --mode "$mode" --output-dir "$case_dir"
        --size-mib "$size_mib" --warmup "$warmup" --gemm-rounds "$gemm_rounds" --comm-rounds "$comm_rounds")
    if [[ -n "$shape" ]]; then command+=(--gemm "$shape"); fi

    local -a run_env=(env "CUDA_VISIBLE_DEVICES=$devices" "CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER"
        "GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-${GLOO_SOCKET_NAME:-}}"
        "NSYS_BIN=$NSYS_BIN" "NSYS_TMPDIR=$case_dir/nsys/.tmp")
    if $dry_run; then
        show_command "${run_env[@]}" "${command[@]}"
        return
    fi
    mkdir -p -- "$case_dir/nsys/.tmp" || return
    show_command "${run_env[@]}" "${command[@]}" >"$case_dir/command.txt" || return
    printf 'Running %s on GPUs %s\n' "$name" "$devices"
    "${run_env[@]}" "${command[@]}" 2>&1 | tee "$case_dir/torchrun.log" || return
    for ((rank = 0; rank < ranks; rank++)); do
        [[ -s "$case_dir/nsys/$name.rank_$rank.nsys-rep" ]] || {
            printf 'error: %s has no nsys report for rank %d\n' "$name" "$rank" >&2
            return 1
        }
    done
}

failed=0
for case_id in "${cases[@]}"; do
    devices="${gpus[0]}" ranks=1
    case "$case_id" in
        task1) mode=gemm ;;
        task2) mode=gemm-h2d ;;
        task3) mode=gemm-ag-hbm ;;
        task4) mode=gemm-ag-host ;;
        baseline1) mode=h2d ;;
        baseline2) mode=ag-hbm ;;
        baseline3) mode=ag-host ;;
    esac
    case "$mode" in
        *ag-*) devices="$gpu_ids"; ranks="${#gpus[@]}" ;;
    esac
    if [[ "$mode" = gemm* ]]; then
        for index in "${!shapes[@]}"; do
            printf -v case_name '%s-%s-%02d-%s' "$case_id" "$mode" "$((index + 1))" "${shape_names[$index]}"
            run_case "$case_name" "$mode" "$devices" "$ranks" "${shapes[$index]}" || failed=1
        done
    else
        run_case "$case_id-$mode" "$mode" "$devices" "$ranks" || failed=1
    fi
done

if ! $dry_run; then
    "${clock_query[@]}" >"$output_dir/gpu_clocks.after.csv" || failed=1
fi
printf 'Results directory: %s\n' "$output_dir"
exit "$failed"
