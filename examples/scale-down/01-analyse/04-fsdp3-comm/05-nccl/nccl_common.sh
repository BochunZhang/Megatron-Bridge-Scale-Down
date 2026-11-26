#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

NCCL_DEFAULT_LIB="${NCCL_DEFAULT_LIB:-nccl/nccl-2.32.3-cta-with-pause/build/lib/libnccl.so.2}"
# shellcheck source=../offload_study_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/../offload_study_common.sh" "$@"

NCCL_LIB="${NCCL_LIB:-${REPO_ROOT}/${NCCL_DEFAULT_LIB}}"
NCCL_BUILD=static-pause
NCCL_CHECKED=""
NCCL_PRELOAD_SUFFIX=""

# Replace any NCCL selected by nccl_use and preserve other preloaded libraries.
IFS=': ' read -r -a inherited_preloads <<< "${LD_PRELOAD:-}"
for lib in ${inherited_preloads[@]+"${inherited_preloads[@]}"}; do
    case "${lib##*/}" in
        ''|libnccl.so|libnccl.so.*) ;;
        *) NCCL_PRELOAD_SUFFIX+=":${lib}" ;;
    esac
done

prepare_case() {
    local check_key="${NCCL_BUILD}:${nccl_lib}"
    if [[ "${NCCL_CHECKED}" != "${check_key}" ]]; then
        local version pause_knob
        case "${NCCL_BUILD}" in
            stock) version='2\.32\.3'; pause_knob="";;
            adaptive-pause) version='2\.32\.3-adaptive-pause'; pause_knob=SYM_AG_STMC_RS_PAUSE_CYCLES;;
            static-pause) version='2\.32\.3-(cta-)?pause'; pause_knob=SYM_AG_STMC_PAUSE_CYCLES;;
        esac
        [[ -f "${nccl_lib}" && -r "${nccl_lib}" ]] || fail "NCCL library is missing or unreadable: ${nccl_lib}"
        grep -aEq "NCCL version ${version}\\+cuda" "${nccl_lib}" ||
            fail "Expected NCCL 2.32.3 (${NCCL_BUILD}) at ${nccl_lib}"
        if [[ -n "${pause_knob}" ]] && ! grep -aFq "${pause_knob}" "${nccl_lib}"; then
            fail "The NCCL build does not expose ${pause_knob}"
        fi

        # A bundled NCCL in the PyTorch wheel can be loaded alongside LD_PRELOAD.
        (
            cd "${REPO_ROOT}"
            LD_PRELOAD="${nccl_preload}" uv run --no-sync python - "${nccl_lib}" <<'PY'
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
        NCCL_CHECKED="${check_key}"
    fi
    LAUNCHER=(env "LD_PRELOAD=${nccl_preload}" "${LAUNCHER[@]}")
}

# run_nccl_case NAME PRIORITY PAUSE [launcher flags...]
run_nccl_case() {
    local name="$1" priority="$2" pause="$3"
    shift 3
    local nccl_lib="${NCCL_LIB}"
    [[ "${nccl_lib}" == /* ]] || nccl_lib="${PWD}/${nccl_lib}"
    local nccl_preload="${nccl_lib}${NCCL_PRELOAD_SUFFIX}"
    local groups='[]'
    if [[ "${priority}" == high ]]; then groups='[dp_cp,ep_dp]'; fi
    local launcher_env=(env -u NCCL_ALGO -u NCCL_PROTO -u NCCL_SYM_KERNEL
        NCCL_NVLS_ENABLE=1 NCCL_COMM_SPLIT_SHARE_RESOURCES=0
        "NCCL_SYM_AG_STMC_PAUSE_CYCLES=${pause}" NCCL_SYM_AG_ST_PAUSE_CYCLES=0)
    if [[ " $* " == *" --symmetric "* ]]; then
        launcher_env+=(
            NCCL_SYM_TMA_ENABLE=0 NCCL_SYM_CE_THRESHOLD=9223372036854775807
        )
    fi
    if [[ "${NCCL_BUILD}" != static-pause ]]; then
        launcher_env+=(NCCL_WIN_ENABLE=1)
    fi
    if (( STUDY_NNODES > 1 )); then
        launcher_env+=(NCCL_MNNVL_ENABLE=1)
    fi
    if [[ "${NCCL_BUILD}" == adaptive-pause ]]; then
        launcher_env+=(NCCL_SYM_AG_STMC_RS_PAUSE_CYCLES=-1 NCCL_SYM_AG_STMC_PAUSE_POLL_CHUNKS=4)
    fi
    printf 'NCCL case %s:' "${name}" >&2
    printf ' %q' "${launcher_env[@]}" "LD_PRELOAD=${nccl_preload}" >&2
    printf '\n' >&2
    local LAUNCHER=("${launcher_env[@]}" "${BENCHMARK_DIR}/run_pretrain_fsdp3.sh")
    run_case 27b "${name}" 3 on 1 1 1 --override "dist.high_priority_stream_groups=${groups}" "$@"
}
