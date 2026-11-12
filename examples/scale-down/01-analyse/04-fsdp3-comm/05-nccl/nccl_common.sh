#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

# shellcheck source=../offload_study_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/../offload_study_common.sh" "$@"
STUDY_SCRIPT="${BASH_SOURCE[1]##*/}"

NCCL_LIB="${NCCL_LIB:-${REPO_ROOT}/nccl/nccl-2.32.3-fsdp-multicast/build/lib/libnccl.so.2}"
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
        grep -aEq 'NCCL version 2\.32\.3-cta-pause-fsdp\+cuda' "${NCCL_LIB}" ||
            fail "Build the FSDP multicast NCCL with install/install-nccl-2.32.3-pause.sh --fsdp-multicast"

        # A bundled NCCL in the PyTorch wheel can be loaded alongside LD_PRELOAD.
        (
            if [[ -n "${MEGATRON_LM_DIR:-}" ]]; then
                mlm_source="$(cd "${MEGATRON_LM_DIR}" && pwd)"
                export PYTHONPATH="${mlm_source}${PYTHONPATH:+:${PYTHONPATH}}"
            fi
            cd "${REPO_ROOT}"
            LD_PRELOAD="${NCCL_PRELOAD}" uv run --no-sync python - "${NCCL_LIB}" <<'PY'
import ctypes
import os
import sys

import torch  # Loads the NCCL dependencies of libtorch_cuda.
from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import ParamAndGradBuffer

expected = os.path.realpath(sys.argv[1])
with open("/proc/self/maps") as maps:
    libraries = {line.split(None, 5)[-1].strip() for line in maps}
loaded = sorted({os.path.realpath(path) for path in libraries if os.path.basename(path).startswith("libnccl.so")})
if loaded != [expected]:
    sys.exit(f"Expected only {expected} for torch {torch.__version__}, but found {loaded}")
if not hasattr(ctypes.CDLL(None), "ncclSetFsdpMulticast"):
    sys.exit(f"{expected} does not export ncclSetFsdpMulticast")
if not hasattr(ParamAndGradBuffer, "_fsdp_multicast_scope"):
    sys.exit("Update the active Megatron-LM checkout for FSDP-scoped multicast")
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
    local launcher_env=(env -u NCCL_ALGO -u NCCL_PROTO -u NCCL_SYM_KERNEL
        NCCL_NVLS_ENABLE=1 NCCL_COMM_SPLIT_SHARE_RESOURCES=0
        NCCL_ENQUEUE_REARCH_ENABLE=0 NCCL_SYM_AG_STMC_PAUSE_CYCLES=0 NCCL_SYM_AG_ST_PAUSE_CYCLES=0
        "NCCL_FSDP_AG_STMC_PAUSE_CYCLES=${pause}")
    if [[ " $* " == *" --symmetric "* ]]; then
        launcher_env+=(MEGATRON_FSDP_MULTICAST=1)
    else
        launcher_env+=(MEGATRON_FSDP_MULTICAST=0)
    fi
    printf 'NCCL case %s:' "${name}" >&2
    printf ' %q' "${launcher_env[@]}" "LD_PRELOAD=${NCCL_PRELOAD}" >&2
    printf '\n' >&2
    local LAUNCHER=("${launcher_env[@]}" "${BENCHMARK_DIR}/run_pretrain_fsdp3.sh")
    run_case 27b "${name}" 3 on 1 1 1 --override "dist.high_priority_stream_groups=${groups}" "$@"
}
