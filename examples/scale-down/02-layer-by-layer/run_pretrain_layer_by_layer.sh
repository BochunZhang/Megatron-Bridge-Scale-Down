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
tee_pid=""

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
        wait "${tee_pid}" || true
        finish_case "${child_status}" || true
    elif [[ -n "${tee_pid}" ]]; then
        kill -TERM "${tee_pid}" 2>/dev/null || true
        wait "${tee_pid}" || true
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

# Keep tee outside the launcher's process group so it drains the log on termination.
output_pipe="${state_dir}/train-output"
mkfifo "${output_pipe}"
set -m
tee "${train_log}" < "${output_pipe}" &
tee_pid=$!
(
    cd "${repo_root}"
    exec "${command[@]}"
) > "${output_pipe}" 2>&1 &
case_pid=$!
set +m

child_status=0
wait "${case_pid}" || child_status=$?
tee_status=0
wait "${tee_pid}" || tee_status=$?
if (( child_status == 0 )); then
    child_status="${tee_status}"
fi
case_pid=""
tee_pid=""
finish_case "${child_status}"
