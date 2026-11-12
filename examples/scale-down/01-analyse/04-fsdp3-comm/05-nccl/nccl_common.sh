#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

# shellcheck source=../offload_study_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/../offload_study_common.sh" "$@"

NCCL_LIB="${NCCL_LIB:-${REPO_ROOT}/nccl/nccl-2.32.3-cta-with-pause/build/lib/libnccl.so.2}"
[[ "${NCCL_LIB}" == /* ]] || NCCL_LIB="${PWD}/${NCCL_LIB}"
NCCL_PRELOAD="${NCCL_LIB}"
NCCL_CHECKED=false

# Replace any NCCL selected by nccl_use and preserve other preloaded libraries.
IFS=': ' read -r -a inherited_preloads <<< "${LD_PRELOAD:-}"
for lib in ${inherited_preloads[@]+"${inherited_preloads[@]}"}; do
    case "${lib##*/}" in
        ''|libnccl.so|libnccl.so.*) ;;
        *) NCCL_PRELOAD+=":${lib}" ;;
    esac
done

prepare_case() {
    if [[ "${NCCL_CHECKED}" == false ]]; then
        [[ -f "${NCCL_LIB}" && -r "${NCCL_LIB}" ]] || fail "NCCL library is missing or unreadable: ${NCCL_LIB}"
        grep -aEq 'NCCL version 2\.32\.3-(cta-)?pause\+cuda' "${NCCL_LIB}" ||
            fail "Expected an NCCL 2.32.3 build with the AG pause patch: ${NCCL_LIB}"
        grep -aq 'SYM_AG_STMC_PAUSE_CYCLES' "${NCCL_LIB}" ||
            fail "The NCCL build does not expose SYM_AG_STMC_PAUSE_CYCLES"

        # A bundled NCCL in the PyTorch wheel can be loaded alongside LD_PRELOAD.
        (
            cd "${REPO_ROOT}"
            LD_PRELOAD="${NCCL_PRELOAD}" uv run --no-sync python - "${NCCL_LIB}" <<'PY'
import os
import sys

import torch  # Loads the NCCL dependencies of libtorch_cuda.

expected = os.path.realpath(sys.argv[1])
with open("/proc/self/maps") as maps:
    libraries = {line.split(None, 5)[-1].strip() for line in maps}
loaded = sorted({os.path.realpath(path) for path in libraries if os.path.basename(path).startswith("libnccl.so")})
if loaded != [expected]:
    sys.exit(f"Expected only {expected} for torch {torch.__version__}, but found {loaded}")
PY
        )
        NCCL_CHECKED=true
    fi
    LAUNCHER=(env "LD_PRELOAD=${NCCL_PRELOAD}" "${LAUNCHER[@]}")
}

# run_nccl_case NAME PRIORITY PAUSE [launcher flags...]
run_nccl_case() {
    local name="$1" priority="$2" pause="$3"
    shift 3
    local groups='[]'
    if [[ "${priority}" == high ]]; then groups='[dp_cp,ep_dp]'; fi
    local launcher_env=(env -u NCCL_ALGO -u NCCL_SYM_KERNEL
        NCCL_NVLS_ENABLE=1 NCCL_COMM_SPLIT_SHARE_RESOURCES=0
        "NCCL_SYM_AG_STMC_PAUSE_CYCLES=${pause}")
    if [[ " $* " == *" --symmetric "* ]]; then
        launcher_env+=(
            'NCCL_SYM_KERNEL=allgather:AllGather_STMC;reducescatter:ReduceScatter_LDMC'
            NCCL_SYM_TMA_ENABLE=0 NCCL_SYM_CE_THRESHOLD=9223372036854775807
        )
    fi
    printf 'NCCL case %s:' "${name}" >&2
    printf ' %q' "${launcher_env[@]}" "LD_PRELOAD=${NCCL_PRELOAD}" >&2
    printf '\n' >&2
    local LAUNCHER=("${launcher_env[@]}" "${BENCHMARK_DIR}/run_pretrain_fsdp3.sh")
    run_case 27b "${name}" 3 on 1 1 1 --override "dist.high_priority_stream_groups=${groups}" "$@"
}
