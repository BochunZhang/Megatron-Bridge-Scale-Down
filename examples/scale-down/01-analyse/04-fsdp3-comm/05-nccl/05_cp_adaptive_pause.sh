#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    cat <<'EOF'
Usage: 05_cp_adaptive_pause.sh [--output-dir DIR] [--nccl-lib FILE] [--dry-run | --print-case-dirs]

On each of eight 4-GPU nodes in the same NVL72 domain, set NODE_RANK (or RANK),
the same MASTER_ADDR (or DLC_MASTER_ADDR), and a shared output directory.
The eight cases use consecutive ports starting at MASTER_PORT (default 29501).

Runs stock and adaptive-pause NCCL 2.32.3 at CP 4/8/16/32 with sequences
16K/32K/64K/128K, 32 FSDP shards, MBS=1, and 30 training steps per case.
GBS defaults to 32/16/8/4 for four microbatches per step at every CP size.
Set CP_SWEEP_GBS to a positive multiple of 8 to use one GBS for all sizes.
Set NCCL_STOCK_LIB or NCCL_ADAPTIVE_PAUSE_LIB to override the library paths.
The node 0 step-8 trace must show only STMC AllGather and LDMC ReduceScatter.
Use a fresh --output-dir on every node when repeating the sweep.
EOF
    exit 0
fi

NCCL_DEFAULT_LIB=nccl/nccl-2.32.3-adaptive-pause/build/lib/libnccl.so.2
NCCL_LIB="${NCCL_ADAPTIVE_PAUSE_LIB:-}"
# shellcheck source=nccl_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/nccl_common.sh" --unlock "$@"

STUDY_GPUS=32
STUDY_GPUS_PER_NODE=4
STUDY_NNODES=8
STUDY_NODE_RANK="${NODE_RANK:-${RANK:-}}"
[[ "${STUDY_NODE_RANK}" =~ ^[0-7]$ ]] || fail "Set NODE_RANK or RANK to the node index (0–7)"
port="${MASTER_PORT:-29501}"
[[ "${port}" =~ ^[1-9][0-9]{0,4}$ ]] && (( port <= 65528 )) || fail "MASTER_PORT must be between 1 and 65528"
if [[ -n "${CP_SWEEP_GBS:-}" ]]; then
    [[ "${CP_SWEEP_GBS}" =~ ^[1-9][0-9]*$ ]] && (( CP_SWEEP_GBS % 8 == 0 )) ||
        fail "CP_SWEEP_GBS must be a positive multiple of 8"
fi

export NSYS_BIN="${NSYS_BIN:-${REPO_ROOT}/NsightSystems-2026.6.1/bin/nsys}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${REPO_ROOT}/.venv/python3.12-torch2.13-nccl2.32.3}"
unset NCCL_CTA_POLICY
NSYS_REPORT_ONLY=true

verify_multimem_trace() (
    local node_dir="$1"
    local reports=("${node_dir}/"*.nsys-rep)
    if (( ${#reports[@]} != 1 )) || [[ ! -s "${reports[0]}" ]]; then
        echo "Expected one nonempty Nsight Systems report under ${node_dir}" >&2
        return 1
    fi
    local sqlite="${node_dir}/.multimem-kernels.sqlite"
    trap 'rm -f -- "${sqlite}"' EXIT
    "${NSYS_BIN}" export --type=sqlite --quiet=true '--tables=^CUPTI_ACTIVITY_KIND_KERNEL$,^StringIds$' \
        --output="${sqlite}" "${reports[0]}" || return $?
    cd "${REPO_ROOT}" || return $?
    uv run --no-sync python - "${sqlite}" "${STUDY_GPUS_PER_NODE}" <<'PY'
import sqlite3
import sys
from pathlib import Path

expected_kernels = {
    "ncclSymkDevKernel_AllGather_STMC",
    "ncclSymkDevKernel_ReduceScatter_LDMC_sum_bf16",
}
database_uri = Path(sys.argv[1]).as_uri() + "?mode=ro"
with sqlite3.connect(database_uri, uri=True) as connection:
    rows = connection.execute("""
        SELECT k.deviceId, s.value, COUNT(*)
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
        JOIN StringIds AS s ON s.id = k.shortName
        WHERE s.value LIKE 'nccl%AllGather%' OR s.value LIKE 'nccl%ReduceScatter%'
        GROUP BY k.deviceId, s.value
        ORDER BY k.deviceId, s.value
    """).fetchall()

kernels_by_device = {}
for device, name, count in rows:
    kernels_by_device.setdefault(device, set()).add(name)
    sys.stdout.write(f"GPU {device}  {name}  {count}\n")

if len(kernels_by_device) != int(sys.argv[2]) or any(
    names != expected_kernels for names in kernels_by_device.values()
):
    sys.exit("Each traced GPU must run STMC AllGather and bf16 LDMC ReduceScatter exclusively.")
PY
)

finish_case() {
    local node_dir="$1" status="$2"
    mkdir -p -- "${node_dir}" || return $?
    node_dir="$(cd "${node_dir}" && pwd)" || return $?
    local verdict="${node_dir}/multimem_check.txt"
    local pending="${verdict}.tmp"
    if (( STUDY_NODE_RANK == 0 && status == 0 )); then
        verify_multimem_trace "${node_dir}" >"${pending}" 2>&1 || status=$?
    fi
    printf 'status=%d\n' "${status}" >>"${pending}" && mv -- "${pending}" "${verdict}" || return 1

    # All nodes see the training and trace verdicts before starting the next rendezvous.
    local rank peer deadline=$((SECONDS + 600))
    for (( rank = 0; rank < STUDY_NNODES; rank++ )); do
        peer="${node_dir%/*}/node${rank}/multimem_check.txt"
        while [[ ! -s "${peer}" ]]; do
            if (( SECONDS >= deadline )); then
                echo "Timed out waiting for ${peer}" >&2
                return 1
            fi
            sleep 1
        done
        if ! grep -qx 'status=0' "${peer}"; then
            printf 'Case validation failed in %s\n' "${peer}" >&2
            cat -- "${peer}" >&2
            return 1
        fi
    done
}

stock_lib="${NCCL_STOCK_LIB:-${REPO_ROOT}/nccl/nccl-2.32.3/build/lib/libnccl.so.2}"
# FSDP registers its symmetric windows after step 1. Validate the steady-state
# kernels instead of globally pinning STMC/LDMC during the unregistered warmup.
case_args=(--overlap --separate-rs --symmetric --metric --train-iters 30
    --override ddp.num_distributed_optimizer_instances=1)

for STUDY_CONTEXT_PARALLEL in 4 8 16 32; do
    STUDY_SEQ_LENGTH=$((4096 * STUDY_CONTEXT_PARALLEL))
    STUDY_GBS="${CP_SWEEP_GBS:-$((4 * STUDY_GPUS / STUDY_CONTEXT_PARALLEL))}"
    NCCL_BUILD=stock NCCL_LIB="${stock_lib}" run_nccl_case stock high 0 "${case_args[@]}" --master-port "${port}"
    (( FAILED == 0 )) || exit 1
    port=$((port + 1))
    NCCL_BUILD=adaptive-pause run_nccl_case adaptive-pause high -1 "${case_args[@]}" --master-port "${port}"
    (( FAILED == 0 )) || exit 1
    port=$((port + 1))
done
