#!/usr/bin/env bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${PYTHON:-}" ]]; then
    exec "${PYTHON}" "${script_dir}/sweep.py" "$@"
fi

repo_root="$(cd "${script_dir}/../../.." && pwd)"
exec uv run --project "${repo_root}" --no-sync python "${script_dir}/sweep.py" "$@"
