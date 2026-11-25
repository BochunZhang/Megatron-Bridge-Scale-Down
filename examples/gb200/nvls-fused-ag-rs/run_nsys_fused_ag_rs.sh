#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
    cat <<'USAGE'
Compare fused AG+RS variants with concurrent and isolated symmetric NCCL kernels.
Each case runs with CUDA event timing, then captures CUDA, NVTX and GB100 NVLink
metrics with Nsight Systems 2026.6.1. The split baselines run AG+RS, AG alone and RS alone.
Reports and SQLite exports are saved for analysis on a separate machine.

  --gpus LIST        GPU indices (default: CUDA_VISIBLE_DEVICES or 0,1,2,3)
  --ag-mib N         Full AllGather output per rank (default: 1024)
  --rs-mib N         Full ReduceScatter input per rank (default: 1024)
  --warmup N         Warmup pairs (default: 5)
  --iters N          Measured pairs per run (default: 20)
  --ctas LIST        Comma-separated fused grid sizes to run in order (default: 4,8)
  --variants LIST    Comma-separated variants to run at each grid size
                     (default: ag-first,rs-first,warp-split,warp-split-pause)
  --ag-pause-cycles N  AG pause for warp-split-pause only (default: 3072)
  --nccl-build DIR   NCCL build from install-nccl-2.32.3-fused-ag-rs.sh
  --cuda-home DIR    Toolkit used for the NCCL host shim (default: CUDA_HOME or /usr/local/cuda)
  --nsys PATH        Override NSYS_BIN or the repo default NsightSystems-2026.6.1/bin/nsys
  --nsys-freq HZ     GPU metrics sampling frequency (10..200000, default: 50000)
  --output-dir DIR   Default: results/gb200/nvls-fused-ag-rs/<timestamp>
  --dry-run          Print commands without creating outputs or starting workers
  -h, --help
USAGE
}

die() { echo "ERROR: $*" >&2; exit 1; }

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
nsys_release=2026.6.1
nsys_bin="${NSYS_BIN:-${repo_root}/NsightSystems-${nsys_release}/bin/nsys}"
nsys_metrics_file="${script_dir}/gb200-nvlink-bytes.config"
nsys_freq=50000
gpus="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
ag_mib=1024 rs_mib=1024 warmup=5 iters=20
cta_sizes=4,8
variants=ag-first,rs-first,warp-split,warp-split-pause
ag_pause_cycles=3072
nccl_build="${repo_root}/nccl/nccl-2.32.3-fused-ag-rs/build"
cuda_home="${CUDA_HOME:-/usr/local/cuda}"
output_dir=
dry_run=false

while (( $# )); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --dry-run) dry_run=true; shift ;;
        --gpus|--ag-mib|--rs-mib|--warmup|--iters|--ctas|--variants|--ag-pause-cycles|--nccl-build|--cuda-home|--nsys|--nsys-freq|--output-dir)
            [[ -n "${2:-}" && "$2" != --* ]] || die "$1 needs a value"
            case "$1" in
                --gpus) gpus="$2" ;;
                --ag-mib) ag_mib="$2" ;;
                --rs-mib) rs_mib="$2" ;;
                --warmup) warmup="$2" ;;
                --iters) iters="$2" ;;
                --ctas) cta_sizes="$2" ;;
                --variants) variants="$2" ;;
                --ag-pause-cycles) ag_pause_cycles="$2" ;;
                --nccl-build) nccl_build="$2" ;;
                --cuda-home) cuda_home="$2" ;;
                --nsys) nsys_bin="$2" ;;
                --nsys-freq) nsys_freq="$2" ;;
                --output-dir) output_dir="$2" ;;
            esac
            shift 2
            ;;
        *) die "unknown argument: $1 (see --help)" ;;
    esac
done

[[ "$gpus" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))+$ ]] || die "--gpus needs at least two numeric GPU indices"
IFS=, read -r -a gpu_list <<<"$gpus"
nranks=${#gpu_list[@]}
[[ "$(printf '%s\n' "${gpu_list[@]}" | sort -u | wc -l)" -eq "$nranks" ]] || die "--gpus must list distinct GPUs"
for value in "$ag_mib" "$rs_mib" "$warmup" "$iters"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "message sizes and iteration counts must be positive integers"
done
[[ "$cta_sizes" =~ ^[1-9][0-9]?(,[1-9][0-9]?)*$ ]] || die "--ctas must list values in 1..64"
IFS=, read -r -a cta_list <<<"$cta_sizes"
for cta in "${cta_list[@]}"; do
    (( cta <= 64 )) || die "--ctas must list values in 1..64"
done
[[ "$(printf '%s\n' "${cta_list[@]}" | sort -u | wc -l)" -eq "${#cta_list[@]}" ]] || die "--ctas must list distinct values"
[[ "$variants" =~ ^[a-z-]+(,[a-z-]+)*$ ]] || die "--variants must be a comma-separated list"
IFS=, read -r -a variant_list <<<"$variants"
for variant in "${variant_list[@]}"; do
    case "$variant" in
        ag-first|rs-first|warp-split|warp-split-pause) ;;
        *) die "unknown fused variant: ${variant}" ;;
    esac
done
[[ "$(printf '%s\n' "${variant_list[@]}" | sort -u | wc -l)" -eq "${#variant_list[@]}" ]] \
    || die "--variants must list distinct values"
[[ "$ag_pause_cycles" =~ ^(0|[1-9][0-9]{0,9})$ ]] && (( ag_pause_cycles <= 4294967295 )) \
    || die "--ag-pause-cycles must be in 0..4294967295"
[[ "$nsys_freq" =~ ^[1-9][0-9]{0,5}$ ]] && (( nsys_freq >= 10 && nsys_freq <= 200000 )) \
    || die "--nsys-freq must be in 10..200000 Hz"

output_dir="${output_dir:-${repo_root}/results/gb200/nvls-fused-ag-rs/$(date +%Y%m%d-%H%M%S)}"
[[ "$output_dir" = /* ]] || output_dir="$PWD/$output_dir"
[[ "$nccl_build" = /* ]] || nccl_build="$PWD/$nccl_build"
[[ "$cuda_home" = /* ]] || cuda_home="$PWD/$cuda_home"
[[ "$nsys_bin" != */* || "$nsys_bin" = /* ]] || nsys_bin="$PWD/$nsys_bin"
nccl_lib="${nccl_build}/lib/libnccl.so.2"
shim="${output_dir}/libfused_ag_rs_shim.so"
if $dry_run; then
    gloo_ifname="${GLOO_SOCKET_IFNAME:-<auto>}"
    printf '# GLOO_SOCKET_IFNAME will be resolved and validated against IPv4 interfaces on the target.\n'
else
    nsys_path="$(command -v "$nsys_bin")" || die "Nsight Systems not found at $nsys_bin (use --nsys PATH)"
    nsys_bin="$nsys_path"
    [[ "$nsys_bin" = /* ]] || nsys_bin="$PWD/$nsys_bin"
    nsys_version="$("$nsys_bin" --version)"
    [[ "$nsys_version" =~ (^|[[:space:]])${nsys_release//./\\.}([.-]|[[:space:]]|$) ]] \
        || die "expected Nsight Systems ${nsys_release}, found: ${nsys_version}"
    [[ -r "$nsys_metrics_file" ]] || die "NVLink metric set not found at ${nsys_metrics_file}"
    command -v uv >/dev/null || die "uv is required"
    [[ -x "${cuda_home}/bin/nvcc" ]] || die "CUDA toolkit not found (use --cuda-home DIR)"
    [[ -f "$nccl_lib" && -f "${nccl_build}/include/nccl_device.h" ]] &&
        LC_ALL=C grep -aFq 'NCCL version 2.32.3-fused-ag-rs+cuda' "$nccl_lib" \
        || die "use --nccl-build with the patched NCCL from install/install-nccl-2.32.3-fused-ag-rs.sh"
    # shellcheck source=../../../scripts/scale-down/common/gloo_socket_ifname.sh
    source "${repo_root}/scripts/scale-down/common/gloo_socket_ifname.sh"
    resolve_gloo_socket_ifname 1
    gloo_ifname="$GLOO_SOCKET_IFNAME"
    mkdir -p -- "$output_dir"
    [[ -z "$(ls -A "$output_dir")" ]] || die "use an empty output directory: ${output_dir}"
    {
        echo "$nsys_version"
        echo "gpus=$gpus ag_mib=$ag_mib rs_mib=$rs_mib warmup=$warmup iters=$iters dtype=bf16"
        echo "fused_ctas=$cta_sizes fused_variants=$variants warp_split_pause_ag_cycles=$ag_pause_cycles"
        echo "nccl_lib=$nccl_lib"
        echo "GLOO_SOCKET_IFNAME=$gloo_ifname"
        echo "nsys_gpu_metrics_devices=$gpus nsys_gpu_metrics_alias=gb200-nvlink-bytes nsys_gpu_metrics_frequency_hz=$nsys_freq"
        echo "nsys_gpu_metrics_set=file:$nsys_metrics_file"
    } >"${output_dir}/config.txt"
fi

run() {
    if $dry_run; then
        printf '%q ' "$@"
        printf '\n'
    else
        "$@"
    fi
}

cd -- "$repo_root"
run "${cuda_home}/bin/nvcc" -std=c++17 -shared -Xcompiler=-fPIC \
    -I"${nccl_build}/include" "${script_dir}/fused_ag_rs_shim.cc" -L"${nccl_build}/lib" \
    -Xlinker -rpath -Xlinker "${nccl_build}/lib" -lnccl -o "$shim"

worker_env=(
    env -u NCCL_ALGO -u NCCL_PROTO -u NCCL_CTA_POLICY -u NCCL_SYM_CTAS
    "CUDA_DEVICE_ORDER=PCI_BUS_ID" "CUDA_VISIBLE_DEVICES=$gpus"
    "GLOO_SOCKET_IFNAME=$gloo_ifname"
    "NCCL_NVLS_ENABLE=1" "NCCL_WIN_ENABLE=1" "NCCL_SYM_TMA_ENABLE=0"
    "NCCL_SYM_CE_THRESHOLD=9223372036854775807" "NCCL_LAUNCH_ORDER_IMPLICIT=1"
    "NCCL_SYM_KERNEL=allgather:AllGather_STMC;reducescatter:ReduceScatter_LDMC"
    "TORCH_NCCL_USE_COMM_NONBLOCKING=0"
    "LD_PRELOAD=$nccl_lib${LD_PRELOAD:+:$LD_PRELOAD}"
)

run_case() {
    local mode="$1" mode_dir="$2" cta="$3" variant="${4:-}" pause_cycles="${5:-0}"
    local report_stem="${mode_dir}/${mode}-cta-auto"
    local -a worker timed profiled
    worker=(
        "${worker_env[@]}"
        uv run --active --no-sync python -m torch.distributed.run --standalone --nproc_per_node "$nranks"
        "${script_dir}/fused_ag_rs.py" --mode "$mode" --ag-mib "$ag_mib" --rs-mib "$rs_mib" \
        --warmup "$warmup" --iters "$iters" --ctas "$cta" --nccl-lib "$nccl_lib" --shim "$shim"
    )
    if [[ "$mode" == fused ]]; then
        report_stem="${mode_dir}/${variant}-cta-${cta}"
        worker+=(--variant "$variant" --ag-pause-cycles "$pause_cycles")
    fi
    timed=("${worker[@]}" --output "${mode_dir}/timing.json")
    profiled=(
        "$nsys_bin" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none
        --gpu-metrics-devices="$gpus" --gpu-metrics-set="file:$nsys_metrics_file" --gpu-metrics-frequency="$nsys_freq"
        --capture-range=cudaProfilerApi --capture-range-end=stop --output="$report_stem"
        "${worker[@]}" --profile --profile-frequency "$nsys_freq" --output "${mode_dir}/profile-timing.json"
    )
    if $dry_run; then
        run "${timed[@]}"
        run "${profiled[@]}"
    else
        mkdir -p -- "$mode_dir"
        echo "=== ${mode_dir#"$output_dir"/}"
        printf '%q ' "${timed[@]}" >"${mode_dir}/timing-command.txt"
        printf '\n' >>"${mode_dir}/timing-command.txt"
        printf '%q ' "${profiled[@]}" >"${mode_dir}/profile-command.txt"
        printf '\n' >>"${mode_dir}/profile-command.txt"
        run "${timed[@]}" 2>&1 | tee "${mode_dir}/timing.log"
        run "${profiled[@]}" 2>&1 | tee "${mode_dir}/nsys.log"
        [[ -s "${mode_dir}/timing.json" && -s "${mode_dir}/profile-timing.json" ]] \
            || die "worker did not finish output validation in ${mode_dir}"
    fi
    run "$nsys_bin" export --type=sqlite --output="${mode_dir}/profile.sqlite" "${report_stem}.nsys-rep"
}

for cta in "${cta_list[@]}"; do
    for variant in "${variant_list[@]}"; do
        pause_cycles=0
        if [[ "$variant" == warp-split-pause ]]; then
            pause_cycles="$ag_pause_cycles"
        fi
        run_case fused "${output_dir}/cta-${cta}/${variant}" "$cta" "$variant" "$pause_cycles"
    done
done
# The worker requires a CTA argument, but NCCL selects the grids for these baselines.
run_case split "${output_dir}/split" "${cta_list[0]}"
run_case ag-only "${output_dir}/split/ag-only" "${cta_list[0]}"
run_case rs-only "${output_dir}/split/rs-only" "${cta_list[0]}"

if ! $dry_run; then
    printf 'Reports and SQLite exports saved in %s\n' "$output_dir"
fi
