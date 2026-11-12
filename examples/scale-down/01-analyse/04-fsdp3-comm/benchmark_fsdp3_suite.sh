#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/01-analyse/04-fsdp3-comm}"
NVME_PATH="${NVME_PATH:-}"
ITEMS=01,02,03,04,05,06,07,08
MODELS=(27b 2.4t)
MODE_ARGS=(--resume)

usage() {
    cat <<'USAGE'
Usage: benchmark_fsdp3_suite.sh [--items 01,02,...] [--model 27b|2.4t]
                               [--nvme-path DIR] [--output-dir DIR] [--dry-run]

Runs on one 4-GPU GB200 tray. Both cropped models use BF16, full attention,
GBS=8, MBS=1,2,4 and sequence lengths 4096,8192,16384. MoE uses 64 experts
and EP=4. Nsight Systems traces CUDA/NVTX at step 7 without GPU metric sampling.

  01-cuda-graph               FSDP3 eager vs CUDA graph
  02-ddp-cp-tp                DDP graphs at baseline, CP=2/4 and TP=2/4
  03-fsdp3-cp-tp              FSDP3 baseline, CP=2 and TP=2
  04-activation-offload      DDP graphs, baseline vs activation offload + recompute
  05-nccl-priority           FSDP3 graphs, default vs high DP communication priority
  06-overlap-symmetric       High-priority FSDP3, four AG/RS overlap x symmetric modes
  07-cpu-optimizer           FSDP3 attention graphs + weight offload + CPU accumulation/AdamW
  08-nvme                    Eager FSDP3 native NVMe offload (requires --nvme-path or NVME_PATH)

The MBS=4, TP=CP=1 points are invalid on 4 GPUs (MBS*DP=16). Where the
comparison permits it, use CP=2 for the MBS=4 pair. Native attention-only
FSDP3 graphs require TP=1 and disallow symmetric registration. Item 03
adds MoE eager topology controls; item 06 adds a matched MoE eager 2x2 sweep.
Items 01/05/06 use full-iteration graphs on 27b and attention graphs on 2.4t.

Each item uses its own directory under results/01-analyse/04-fsdp3-comm.
Rerun the same command to skip cases with status=0 and a nonempty .nsys-rep.
Failed/incomplete attempts are archived. The sweep continues past case failures.
USAGE
}

while (( $# )); do
    case "$1" in
        --items|--model|--nvme-path|--output-dir)
            [[ -n "${2:-}" && "$2" != --* ]] || { echo "Missing value for $1" >&2; exit 2; }
            case "$1" in
                --items) ITEMS="$2";;
                --model)
                    [[ "$2" == 27b || "$2" == 2.4t ]] || { echo "--model must be 27b or 2.4t" >&2; exit 2; }
                    MODELS=("$2");;
                --nvme-path) NVME_PATH="$2";;
                --output-dir) OUTPUT_DIR="$2";;
            esac
            shift 2;;
        --dry-run) MODE_ARGS+=(--dry-run); shift;;
        -h|--help) usage; exit 0;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2;;
    esac
done
[[ "${ITEMS}" =~ ^0[1-8](,0[1-8])*$ ]] || { echo "--items expects a comma-separated list of 01 through 08" >&2; exit 2; }

NORMAL_PRIORITY=(--override 'dist.high_priority_stream_groups=[]')
HIGH_PRIORITY=(--override 'dist.high_priority_stream_groups=[dp_cp,ep_dp]')
ATTEMPTED=0
FAILED=0
SKIPPED=0

run_case() {
    local variant="$1" fsdp="$2" graph="$3" tp="$4" cp="$5"
    shift 5
    local dp=$((4 / (tp * cp)))
    local label="${item}/${variant} model=${model} graph=${graph} tp=${tp} cp=${cp} mbs=${mbs} seq=${seq}"
    if (( 8 % (mbs * dp) != 0 )); then
        printf 'skip %s (GBS=8 is not divisible by MBS*DP=%s)\n' "${label}" "$((mbs * dp))" >&2
        SKIPPED=$((SKIPPED + 1))
        return
    fi
    printf 'case %s\n' "${label}" >&2
    ATTEMPTED=$((ATTEMPTED + 1))
    local launcher=("${SCRIPT_DIR}/run_pretrain_fsdp3.sh")
    if [[ "${index}" == 06 ]]; then
        # A forced algorithm suppresses symmetric kernels. Keep NVLS available
        # in every control and let NCCL choose the algorithm.
        launcher=(env -u NCCL_ALGO NCCL_NVLS_ENABLE=1 "${launcher[@]}")
    fi
    if ! "${launcher[@]}" "${COMMON[@]}" "${MODE_ARGS[@]}" \
        --results-root "${OUTPUT_DIR}/${item}" --stable-result-dir --run-name "${variant}" \
        --fsdp "${fsdp}" --cuda-graph "${graph}" --tp "${tp}" --cp "${cp}" --mbs "${mbs}" --seq "${seq}" "$@"; then
        printf 'failed %s\n' "${label}" >&2
        FAILED=$((FAILED + 1))
    fi
}

overlap_sweep() {
    local graph="$1" cp="$2"
    run_case shared-ag-rs 3 "${graph}" 1 "${cp}" "${HIGH_PRIORITY[@]}"
    run_case split-ag-rs 3 "${graph}" 1 "${cp}" "${HIGH_PRIORITY[@]}" --overlap
    if [[ "${model}" == 27b || "${graph}" == off ]]; then
        run_case shared-ag-rs-symm 3 "${graph}" 1 "${cp}" "${HIGH_PRIORITY[@]}" --symmetric
        run_case split-ag-rs-symm 3 "${graph}" 1 "${cp}" "${HIGH_PRIORITY[@]}" --overlap --symmetric
    fi
}

trap 'exit 130' INT
trap 'exit 143' TERM
for item in 01-cuda-graph 02-ddp-cp-tp 03-fsdp3-cp-tp 04-activation-offload \
    05-nccl-priority 06-overlap-symmetric 07-cpu-optimizer 08-nvme; do
    index="${item%%-*}"
    [[ ",${ITEMS}," == *",${index},"* ]] || continue
    if [[ "${index}" == 08 && -z "${NVME_PATH}" ]]; then
        if [[ "${MODE_ARGS[*]}" == *--dry-run* ]]; then
            NVME_PATH=/path/to/local-nvme
        else
            echo "skip ${item}: supply --nvme-path DIR on the local NVMe drive" >&2
            SKIPPED=$((SKIPPED + 1))
            [[ "${ITEMS}" != 08 ]] || exit 2
            continue
        fi
    fi
    for model in "${MODELS[@]}"; do
        COMMON=(--model "${model}" --full --gbs 8 --dtype bf16 --gpu 4 --nproc-per-node 4 --nnodes 1
            --node-rank 0 --nsys --train-iters 10 --profile-start 7 --profile-end 8)
        activation=mlp_act
        if [[ "${model}" == 27b ]]; then
            COMMON+=(--num-layers 8)
        else
            COMMON+=(--num-layers 4 --num-experts 64 --ep 4 --dispatcher hybridep)
            activation=moe_act
        fi
        if [[ "${model}" == 2.4t && "${index}" == 03 ]]; then
            echo "03-fsdp3-cp-tp: running 2.4t CP/TP topology controls in eager mode" >&2
        fi
        if [[ "${model}" == 2.4t && "${index}" == 06 ]]; then
            echo "06-overlap-symmetric: 2.4t graphs disallow symmetric registration; running the 2x2 in eager mode as well" >&2
        fi
        for mbs in 1 2 4; do
            paired_cp=1
            if (( mbs == 4 )); then paired_cp=2; fi
            fsdp_graph_cp=1
            if [[ "${model}" == 27b ]]; then fsdp_graph_cp="${paired_cp}"; fi
            for seq in 4096 8192 16384; do
                case "${index}" in
                    01)
                        run_case eager 3 off 1 "${fsdp_graph_cp}"
                        run_case graph 3 on 1 "${fsdp_graph_cp}";;
                    02)
                        run_case baseline ddp on 1 1
                        for degree in 2 4; do
                            run_case context-parallel ddp on 1 "${degree}"
                            run_case tensor-parallel ddp on "${degree}" 1
                        done;;
                    03)
                        run_case baseline 3 on 1 1
                        graph=on
                        if [[ "${model}" == 2.4t ]]; then
                            graph=off
                            run_case baseline 3 off 1 1
                        fi
                        run_case context-parallel 3 "${graph}" 1 2
                        run_case tensor-parallel 3 "${graph}" 2 1;;
                    04)
                        # Keep the expert GEMM implementation fixed across the MoE pair.
                        kernel_args=(--override model.use_transformer_engine_op_fuser=false)
                        run_case baseline ddp on 1 "${paired_cp}" "${kernel_args[@]}"
                        run_case offload-recompute ddp on 1 "${paired_cp}" "${kernel_args[@]}" \
                            --override model.fine_grained_activation_offloading=true \
                            --override "model.offload_modules=[${activation}]" \
                            --override model.recompute_granularity=selective \
                            --override "model.recompute_modules=[${activation}]";;
                    05)
                        run_case priority-default 3 on 1 "${fsdp_graph_cp}" "${NORMAL_PRIORITY[@]}"
                        run_case priority-high 3 on 1 "${fsdp_graph_cp}" "${HIGH_PRIORITY[@]}";;
                    06)
                        overlap_sweep on "${fsdp_graph_cp}"
                        if [[ "${model}" == 2.4t ]]; then overlap_sweep off "${paired_cp}"; fi;;
                    07)
                        if [[ "${model}" == 27b ]]; then
                            run_case cpu-adamw 3 on 1 1 --local-attention-graph --offload weight --cpu-optimizer
                        else
                            run_case cpu-adamw 3 on 1 1 --offload weight --cpu-optimizer
                        fi;;
                    08)
                        run_case nvme 3 off 1 "${paired_cp}" --nvme-path "${NVME_PATH}";;
                esac
            done
        done
    done
done

printf 'Checked %s cases, %s failed, %s skipped (batch constraints or missing NVMe path)\n' \
    "${ATTEMPTED}" "${FAILED}" "${SKIPPED}" >&2
(( FAILED == 0 ))
