#!/usr/bin/env bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
python_command=(uv run --project "${repo_root}" --no-sync python)
if [[ -n "${PYTHON:-}" ]]; then
    python_command=("${PYTHON}")
fi

state_dir="$(mktemp -d "${TMPDIR:-/tmp}/layer-by-layer-state.XXXXXX")"
case_pid=""

cleanup() {
    local scratch
    if [[ -f "${state_dir}/scratch" ]] && IFS= read -r scratch < "${state_dir}/scratch"; then
        if [[ "${scratch}" == /* && "${scratch##*/}" == layer-by-layer-[0-9]*-* ]]; then
            rm -rf -- "${scratch}" || true
        fi
    fi
    rm -rf -- "${state_dir}"
}

finish_case() {
    "${python_command[@]}" "${script_dir}/sweep.py" --finish-case "${state_dir}" "$1"
}

terminate() {
    local signal_status="$1" child_status=0
    trap '' HUP INT TERM
    if [[ -n "${case_pid}" ]]; then
        kill -TERM -- "-${case_pid}" 2>/dev/null || kill -TERM "${case_pid}" 2>/dev/null || true
        wait "${case_pid}" || child_status=$?
        finish_case "${child_status}" || true
    fi
    exit "${signal_status}"
}

trap cleanup EXIT
trap 'terminate 129' HUP
trap 'terminate 130' INT
trap 'terminate 143' TERM

"${python_command[@]}" "${script_dir}/sweep.py" --prepare-case "${state_dir}" "$@"
if [[ ! -f "${state_dir}/command" ]]; then
    for arg in "$@"; do
        case "${arg}" in -h|--help|--dry-run) exit 0;; esac
    done
    printf '%s\n' 'Missing prepared layer-by-layer launch command.' >&2
    exit 1
fi

{
    IFS= read -r -d '' train_log && IFS= read -r -d '' executable &&
        [[ "${train_log}" == /* && -n "${executable}" ]] || {
        printf '%s\n' 'Invalid prepared layer-by-layer launch command.' >&2
        exit 1
    }
    command=("${executable}")
    while IFS= read -r -d '' arg; do
        command+=("${arg}")
    done
} < "${state_dir}/command"

# Give the launcher and its ranks a process group for signal delivery.
set -m
(
    cd "${repo_root}"
    exec "${command[@]}"
) > "${train_log}" 2>&1 &
case_pid=$!
set +m

child_status=0
wait "${case_pid}" || child_status=$?
case_pid=""
finish_case "${child_status}"
