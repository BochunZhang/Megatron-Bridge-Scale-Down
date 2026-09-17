#!/usr/bin/env bash

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -uo pipefail

usage() {
    echo "Usage: $0 [-i SECONDS] [-o OUTPUT_DIR] -- TRAINING_COMMAND [ARGS...]" >&2
}

interval="1.0"
output_dir=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--interval)
            if [[ $# -lt 2 ]]; then
                usage
                exit 2
            fi
            interval="$2"
            shift 2
            ;;
        -o|--output-dir)
            if [[ $# -lt 2 ]]; then
                usage
                exit 2
            fi
            output_dir="$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 2
            ;;
    esac
done

if [[ $# -eq 0 ]]; then
    echo "A training command is required." >&2
    usage
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python3}"
if [[ -z "$output_dir" ]]; then
    output_dir="gpu_memory_traces/$(date -u +%Y%m%dT%H%M%SZ)"
fi
ready_file="$output_dir/.monitor-ready-$$"

training_pid=""
monitor_pid=""

stop_children() {
    if [[ -n "$training_pid" ]] && kill -0 "$training_pid" 2>/dev/null; then
        kill -TERM "$training_pid" 2>/dev/null || true
    fi
    if [[ -n "$monitor_pid" ]] && kill -0 "$monitor_pid" 2>/dev/null; then
        kill -TERM "$monitor_pid" 2>/dev/null || true
    fi
}

trap stop_children INT TERM HUP

"$@" &
training_pid=$!

"$python_bin" "$script_dir/check_gpu.py" \
    --interval "$interval" \
    --ready-file "$ready_file" \
    --output-dir "$output_dir" &
monitor_pid=$!

while [[ ! -e "$ready_file" ]]; do
    if ! kill -0 "$monitor_pid" 2>/dev/null; then
        wait "$monitor_pid"
        monitor_status=$?
        monitor_pid=""
        stop_children
        wait "$training_pid" 2>/dev/null || true
        rm -f "$ready_file"
        echo "GPU memory monitor failed before its first sample." >&2
        exit "$monitor_status"
    fi
    sleep 0.05
done

wait "$training_pid"
training_status=$?
training_pid=""

kill -TERM "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid"
monitor_status=$?
monitor_pid=""
rm -f "$ready_file"

echo "GPU memory traces: $output_dir" >&2

if [[ $training_status -eq 0 && $monitor_status -ne 0 ]]; then
    exit "$monitor_status"
fi
exit "$training_status"
