#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
    cat <<'USAGE'
Profile a 4 GiB BF16 all-gather and a delayed 1 GiB reduce-scatter on four GB200 GPUs.
Samples NVLink RX/TX bytes on all four GPUs at 200000 Hz and runs a matched control.

  --gpus LIST       Four numeric GPU indices in CUDA PCI bus order (default: CUDA_VISIBLE_DEVICES or 0,1,2,3)
  --rs-delay-ms N   Delay of the RS stream after the common GPU start gate (default: 1)
  --base-pause-cycles N  AG pause in both runs when RS is absent (default: NCCL_SYM_AG_STMC_PAUSE_CYCLES or 1024)
  --rs-pause-cycles N    AG pause during RS in the adaptive run (default: NCCL_SYM_AG_STMC_RS_PAUSE_CYCLES or 3072)
  --nvlink-peak-gb-s N   Per-GPU, one-way denominator for NVLink percentages (default: 900)
  --nccl-lib PATH   NCCL 2.32.3-adaptive-pause libnccl.so.2
  --nsys PATH       Nsight Systems executable (also NSYS_BIN)
  --output-dir DIR  Default: results/gb200/ncu-torch-all-gather/nsys-adaptive-pause-<time>
  --dry-run         Print the commands
  -h, --help
USAGE
}

die() { echo "ERROR: $*" >&2; exit 1; }

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../../.." && pwd)"
worker="$script_dir/nsys_torch_adaptive_pause.py"
gpus="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
rs_delay_ms=1
base_pause="${NCCL_SYM_AG_STMC_PAUSE_CYCLES:-1024}"
adaptive_pause="${NCCL_SYM_AG_STMC_RS_PAUSE_CYCLES:-3072}"
nvlink_peak_gb_s=900
nccl_lib="$repo_root/nccl/nccl-2.32.3-adaptive-pause/build/lib/libnccl.so.2"
nsys_bin="${NSYS_BIN:-$repo_root/NsightSystems-2026.6.1/bin/nsys}"
if [[ -z "${NSYS_BIN:-}" && ! -x "$nsys_bin" ]]; then
    nsys_bin=nsys
fi
output_dir=""
dry_run=false

while (( $# )); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --dry-run) dry_run=true; shift ;;
        --gpus|--rs-delay-ms|--base-pause-cycles|--rs-pause-cycles|--nvlink-peak-gb-s|--nccl-lib|--nsys|--output-dir)
            [[ -n "${2:-}" && "$2" != --* ]] || die "$1 needs a value"
            case "$1" in
                --gpus) gpus="$2" ;;
                --rs-delay-ms) rs_delay_ms="$2" ;;
                --base-pause-cycles) base_pause="$2" ;;
                --rs-pause-cycles) adaptive_pause="$2" ;;
                --nvlink-peak-gb-s) nvlink_peak_gb_s="$2" ;;
                --nccl-lib) nccl_lib="$2" ;;
                --nsys) nsys_bin="$2" ;;
                --output-dir) output_dir="$2" ;;
            esac
            shift 2
            ;;
        *) die "unknown argument: $1 (see --help)" ;;
    esac
done

[[ "$gpus" =~ ^(0|[1-9][0-9]{0,2})(,(0|[1-9][0-9]{0,2})){3}$ ]] || die "--gpus needs four numeric indices"
IFS=, read -r -a gpu_list <<<"$gpus"
[[ "$(printf '%s\n' "${gpu_list[@]}" | sort -u | wc -l)" -eq 4 ]] || die "--gpus must list distinct GPUs"
for gpu in "${gpu_list[@]}"; do
    (( gpu <= 255 )) || die "--gpus indices must be between 0 and 255"
done
[[ "$rs_delay_ms" =~ ^[0-9]+([.][0-9]+)?$ && "$rs_delay_ms" =~ [1-9] ]] || die "--rs-delay-ms must be positive"
[[ "$nvlink_peak_gb_s" =~ ^[0-9]+([.][0-9]+)?$ && "$nvlink_peak_gb_s" =~ [1-9] ]] \
    || die "--nvlink-peak-gb-s must be positive"
# Resolve NCCL's automatic settings for this fixed four-rank test.
[[ "$base_pause" != -1 ]] || base_pause=1024
[[ "$adaptive_pause" != -1 ]] || adaptive_pause=3072
for cycles in "$base_pause" "$adaptive_pause"; do
    [[ "$cycles" =~ ^(0|[1-9][0-9]{0,9})$ ]] && (( cycles <= 4294967295 )) \
        || die "pause cycles must be -1 (automatic) or an integer in 0..4294967295"
done
(( adaptive_pause > base_pause )) || die "--rs-pause-cycles must exceed --base-pause-cycles in the adaptive run"
poll_chunks="${NCCL_SYM_AG_STMC_PAUSE_POLL_CHUNKS:-4}"

output_dir="${output_dir:-$repo_root/results/gb200/ncu-torch-all-gather/nsys-adaptive-pause-$(date +%Y%m%d-%H%M%S)}"
[[ "$output_dir" = /* ]] || output_dir="$PWD/$output_dir"
[[ "$nccl_lib" = /* ]] || nccl_lib="$PWD/$nccl_lib"
config="$output_dir/gb200-nvlink-bytes.config"
if ! $dry_run; then
    nsys_bin="$(command -v "$nsys_bin")" || die "nsys not found (use --nsys)"
    [[ "$nsys_bin" = /* ]] || nsys_bin="$PWD/$nsys_bin"
    command -v uv >/dev/null || die "uv is required"
    [[ -f "$nccl_lib" ]] && LC_ALL=C grep -aFq 'NCCL version 2.32.3-adaptive-pause+cuda' "$nccl_lib" \
        || die "--nccl-lib must point to the build from install/install-nccl-2.32.3-adaptive-pause.sh"
    mkdir -p -- "$output_dir"
    [[ -z "$(ls -A "$output_dir")" ]] || die "use an empty output directory: $output_dir"
    cp -- "$script_dir/gb200-nvlink-bytes.config" "$config"
    "$nsys_bin" --version >"$output_dir/nsys-version.txt"
fi

cd "$repo_root"
venv="$repo_root/.venv/python3.12-torch2.13-nccl2.32.3"
if [[ -z "${UV_PROJECT_ENVIRONMENT:-}" && -d "$venv" ]]; then
    export UV_PROJECT_ENVIRONMENT="$venv"
fi
unset NCCL_ALGO NCCL_PROTO NCCL_CTA_POLICY NCCL_SYM_CTAS NCCL_MIN_CTAS NCCL_MAX_CTAS
unset NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS NCCL_MIN_NRINGS NCCL_MAX_NRINGS NCCL_NVLS_NCHANNELS
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpus"

run() {
    if $dry_run; then
        printf '%q ' "$@"
        printf '\n'
    else
        "$@"
    fi
}

for mode in adaptive control; do
    rs_pause="$adaptive_pause"
    [[ "$mode" != control ]] || rs_pause=0
    mode_dir="$output_dir/$mode"
    cmd=("$nsys_bin" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none
        --capture-range=cudaProfilerApi --capture-range-end=stop --wait=all
        --gpu-metrics-devices="$gpus" --gpu-metrics-frequency=200000 --gpu-metrics-set="file:$config"
        --output="$mode_dir/profile"
        env "GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}" TORCH_NCCL_BLOCKING_WAIT=0
        NCCL_NVLS_ENABLE=1 NCCL_WIN_ENABLE=1 NCCL_CUMEM_ENABLE=1 NCCL_SYM_TMA_ENABLE=0
        NCCL_LAUNCH_ORDER_IMPLICIT=1 NCCL_SYM_CE_THRESHOLD=9223372036854775807
        "NCCL_SYM_KERNEL=AllGather:AllGather_STMC;ReduceScatter:ReduceScatter_LDMC"
        "NCCL_SYM_AG_STMC_PAUSE_CYCLES=$base_pause" "NCCL_SYM_AG_STMC_RS_PAUSE_CYCLES=$rs_pause"
        "NCCL_SYM_AG_STMC_PAUSE_POLL_CHUNKS=$poll_chunks"
        NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV,TUNING "NCCL_DEBUG_FILE=$mode_dir/nccl-%h-%p.log"
        "ADAPTIVE_PAUSE_NCCL_LIB=$nccl_lib" "LD_PRELOAD=$nccl_lib${LD_PRELOAD:+:$LD_PRELOAD}"
        uv run --no-sync python -m torch.distributed.run --standalone --nproc_per_node=4
        "$worker" benchmark --rs-delay-ms "$rs_delay_ms")
    if $dry_run; then
        run "${cmd[@]}"
    else
        mkdir -- "$mode_dir"
        {
            printf '%s\n' "export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpus"
            printf '%q ' "${cmd[@]}"
            printf '\n'
        } >"$mode_dir/command.txt"
        run "${cmd[@]}" 2>&1 | tee "$mode_dir/nsys.log"
        grep -Fq 'Verified 4 GiB AG and 1 GiB RS on all 4 ranks' "$mode_dir/nsys.log" \
            || die "$mode correctness check did not pass (see $mode_dir/nsys.log)"
    fi
    run "$nsys_bin" export --type=sqlite --output="$mode_dir/profile.sqlite" "$mode_dir/profile.nsys-rep"
done

run uv run --no-sync python "$worker" analyse --output-dir "$output_dir" --gpus "$gpus" \
    --nvlink-peak-gb-s "$nvlink_peak_gb_s"
