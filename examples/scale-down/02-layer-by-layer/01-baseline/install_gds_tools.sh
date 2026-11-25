#!/usr/bin/env bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
set -euo pipefail

fail() { printf 'error: %s\n' "$*" >&2; exit 2; }

if [[ $# == 1 && ( "$1" == -h || "$1" == --help ) ]]; then
    printf 'Usage: %s\nInstall CUDA 13.0 GDS tools on Ubuntu 24.04 ARM64.\n' "${0##*/}"
    exit 0
fi
[[ $# == 0 ]] || fail "Usage: ${0##*/}"

[[ -r /etc/os-release ]] || fail 'This installer requires Ubuntu 24.04 ARM64'
. /etc/os-release
[[ "${ID:-}" == ubuntu && "${VERSION_ID:-}" == 24.04 && "$(dpkg --print-architecture)" == arm64 ]] \
    || fail 'This installer requires Ubuntu 24.04 ARM64'

as_root=()
if (( EUID != 0 )); then
    as_root=(sudo)
fi

repo_path=compute/cuda/repos/ubuntu2404/sbsa
repo_url="https://developer.download.nvidia.com/$repo_path"
package=gds-tools-13-0
tools_dir=/usr/local/cuda-13.0/gds/tools

# Include configured sources whose package indexes have not been downloaded yet.
index_urls="$(apt-get indextargets --no-release-info --format '$(URI)' 'Identifier: Packages')"
if [[ "$index_urls" != *"//developer.download.nvidia.com/$repo_path/"* &&
      "$index_urls" != *"//developer.download.nvidia.cn/$repo_path/"* ]]; then
    keyring="$(mktemp --suffix=.deb)"
    trap 'rm -f -- "$keyring"' EXIT
    curl -fsSL "$repo_url/cuda-keyring_1.1-1_all.deb" -o "$keyring"
    "${as_root[@]}" dpkg -i "$keyring"
fi

"${as_root[@]}" apt-get update
"${as_root[@]}" apt-get install -y --no-install-recommends "$package"

for tool in gdscheck.py gdsio gds_stats; do
    [[ -x "$tools_dir/$tool" ]] || fail "Expected installed executable at $tools_dir/$tool"
done
printf 'Installed %s in %s\n' "$package" "$tools_dir"
printf 'Set GDS_TOOLS_DIR=%s when running check_gds.sh.\n' "$tools_dir"
