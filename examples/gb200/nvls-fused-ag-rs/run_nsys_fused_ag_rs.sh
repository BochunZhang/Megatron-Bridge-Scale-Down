#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
    cat <<'USAGE'
Compare one fused AG+RS kernel with two concurrent symmetric NCCL kernels.
Run both paths with CUDA event timing, then capture CUDA and NVTX with Nsight Systems 2026.6.1.

  --gpus LIST        GPU indices (default: CUDA_VISIBLE_DEVICES or 0,1,2,3)
  --ag-mib N         Full AllGather output per rank (default: 256)
  --rs-mib N         Full ReduceScatter input per rank (default: 256)
  --warmup N         Warmup pairs (default: 5)
  --iters N          Measured pairs per run (default: 20)
  --ctas N           Fused grid size (default: 4)
  --nccl-build DIR   NCCL build from install-nccl-2.32.3-fused-ag-rs.sh
  --cuda-home DIR    Toolkit used for the NCCL host shim (default: CUDA_HOME or /usr/local/cuda)
  --nsys PATH        Override NSYS_BIN or the repo default NsightSystems-2026.6.1/bin/nsys
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
gpus="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
ag_mib=256 rs_mib=256 warmup=5 iters=20 ctas=4
nccl_build="${repo_root}/nccl/nccl-2.32.3-fused-ag-rs/build"
cuda_home="${CUDA_HOME:-/usr/local/cuda}"
output_dir=
dry_run=false

while (( $# )); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --dry-run) dry_run=true; shift ;;
        --gpus|--ag-mib|--rs-mib|--warmup|--iters|--ctas|--nccl-build|--cuda-home|--nsys|--output-dir)
            [[ -n "${2:-}" && "$2" != --* ]] || die "$1 needs a value"
            case "$1" in
                --gpus) gpus="$2" ;;
                --ag-mib) ag_mib="$2" ;;
                --rs-mib) rs_mib="$2" ;;
                --warmup) warmup="$2" ;;
                --iters) iters="$2" ;;
                --ctas) ctas="$2" ;;
                --nccl-build) nccl_build="$2" ;;
                --cuda-home) cuda_home="$2" ;;
                --nsys) nsys_bin="$2" ;;
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
for value in "$ag_mib" "$rs_mib" "$warmup" "$iters" "$ctas"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "message sizes, iteration counts and CTA count must be positive integers"
done
(( ctas <= 64 )) || die "--ctas cannot exceed 64"

output_dir="${output_dir:-${repo_root}/results/gb200/nvls-fused-ag-rs/$(date +%Y%m%d-%H%M%S)}"
[[ "$output_dir" = /* ]] || output_dir="$PWD/$output_dir"
[[ "$nccl_build" = /* ]] || nccl_build="$PWD/$nccl_build"
[[ "$cuda_home" = /* ]] || cuda_home="$PWD/$cuda_home"
[[ "$nsys_bin" != */* || "$nsys_bin" = /* ]] || nsys_bin="$PWD/$nsys_bin"
nccl_lib="${nccl_build}/lib/libnccl.so.2"
shim="${output_dir}/libfused_ag_rs_shim.so"
if ! $dry_run; then
    nsys_path="$(command -v "$nsys_bin")" || die "Nsight Systems not found at $nsys_bin (use --nsys PATH)"
    nsys_bin="$nsys_path"
    [[ "$nsys_bin" = /* ]] || nsys_bin="$PWD/$nsys_bin"
    nsys_version="$("$nsys_bin" --version)"
    [[ "$nsys_version" =~ (^|[[:space:]])${nsys_release//./\\.}([.-]|[[:space:]]|$) ]] \
        || die "expected Nsight Systems ${nsys_release}, found: ${nsys_version}"
    command -v uv >/dev/null || die "uv is required"
    [[ -x "${cuda_home}/bin/nvcc" ]] || die "CUDA toolkit not found (use --cuda-home DIR)"
    [[ -f "$nccl_lib" && -f "${nccl_build}/include/nccl_device.h" ]] &&
        LC_ALL=C grep -aFq 'NCCL version 2.32.3-fused-ag-rs+cuda' "$nccl_lib" \
        || die "use --nccl-build with the patched NCCL from install/install-nccl-2.32.3-fused-ag-rs.sh"
    mkdir -p -- "$output_dir"
    [[ -z "$(ls -A "$output_dir")" ]] || die "use an empty output directory: ${output_dir}"
    {
        echo "$nsys_version"
        echo "gpus=$gpus ag_mib=$ag_mib rs_mib=$rs_mib warmup=$warmup iters=$iters fused_ctas=$ctas dtype=bf16"
        echo "nccl_lib=$nccl_lib"
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
    "NCCL_NVLS_ENABLE=1" "NCCL_WIN_ENABLE=1" "NCCL_SYM_TMA_ENABLE=0"
    "NCCL_SYM_CE_THRESHOLD=9223372036854775807" "NCCL_LAUNCH_ORDER_IMPLICIT=1"
    "NCCL_SYM_KERNEL=allgather:AllGather_STMC;reducescatter:ReduceScatter_LDMC"
    "TORCH_NCCL_USE_COMM_NONBLOCKING=0"
    "LD_PRELOAD=$nccl_lib${LD_PRELOAD:+:$LD_PRELOAD}"
)

for mode in fused split; do
    mode_dir="${output_dir}/${mode}"
    worker=(
        "${worker_env[@]}"
        uv run --active --no-sync python -m torch.distributed.run --standalone --nproc_per_node "$nranks"
        "${script_dir}/fused_ag_rs.py" --mode "$mode" --ag-mib "$ag_mib" --rs-mib "$rs_mib" \
        --warmup "$warmup" --iters "$iters" --ctas "$ctas" --nccl-lib "$nccl_lib" --shim "$shim"
    )
    timed=("${worker[@]}" --output "${mode_dir}/timing.json")
    profiled=(
        "$nsys_bin" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none
        --capture-range=cudaProfilerApi --capture-range-end=stop --output="${mode_dir}/profile"
        "${worker[@]}" --profile --output "${mode_dir}/profile-timing.json"
    )
    if $dry_run; then
        run "${timed[@]}"
        run "${profiled[@]}"
    else
        mkdir -- "$mode_dir"
        printf '%q ' "${timed[@]}" >"${mode_dir}/timing-command.txt"
        printf '\n' >>"${mode_dir}/timing-command.txt"
        printf '%q ' "${profiled[@]}" >"${mode_dir}/profile-command.txt"
        printf '\n' >>"${mode_dir}/profile-command.txt"
        run "${timed[@]}" 2>&1 | tee "${mode_dir}/timing.log"
        run "${profiled[@]}" 2>&1 | tee "${mode_dir}/nsys.log"
        [[ -s "${mode_dir}/timing.json" && -s "${mode_dir}/profile-timing.json" ]] \
            || die "${mode} worker did not finish output validation"
    fi
    run "$nsys_bin" export --type=sqlite --output="${mode_dir}/profile.sqlite" "${mode_dir}/profile.nsys-rep"
done

run uv run --active --no-sync python "${script_dir}/analyse_nsys.py" \
    --fused "${output_dir}/fused/profile.sqlite" --split "${output_dir}/split/profile.sqlite" \
    --iters "$iters" --ranks "$nranks" --output "${output_dir}/summary.json"
