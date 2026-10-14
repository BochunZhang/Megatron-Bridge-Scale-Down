#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
#
# Pick the interface Gloo bootstraps on, from the interfaces this node actually
# has an IPv4 address on.
#
# Sourcing this file only defines functions; run it to print what it resolves.
#
#   source scripts/scale-down/common/gloo_socket_ifname.sh
#   resolve_gloo_socket_ifname      # single node  -> eth0
#   resolve_gloo_socket_ifname 4    # four nodes   -> bond1
#
# Why not trust the environment: a GLOO_SOCKET_IFNAME inherited from a single-node
# DSW (bond1) aborts process-group creation on a DLC worker with "Unable to find
# address for: bond1", because Gloo resolves the name against the local
# interfaces. An exported value is therefore honored only when the interface
# carries an IPv4 address here; anything else is replaced and reported.
#
# Requires: the `ip` command (iproute2).

# log/warn/die are shared with the caller; define only what is missing so the
# helpers survive being sourced alongside another script's versions.
declare -F log >/dev/null || log() { printf '%s\n' "$*"; }
declare -F warn >/dev/null || warn() { printf 'warning: %s\n' "$*" >&2; }
declare -F die >/dev/null || die() { printf 'error: %s\n' "$*" >&2; exit 2; }

# Interfaces that currently have an IPv4 address, space separated.
gloo_available_ifnames() {
    command -v ip >/dev/null \
        || die "the ip command is required to select GLOO_SOCKET_IFNAME; export it explicitly"
    ip -o -4 addr show up 2>/dev/null | awk '{print $2}' | sort -u | tr '\n' ' '
}

# resolve_gloo_socket_ifname [NNODES]
#
# Exports GLOO_SOCKET_IFNAME. With more than one node the fabric is bond1 and
# eth0 is the fallback; on a single node it is the other way round.
resolve_gloo_socket_ifname() {
    local nnodes="${1:-${NNODES:-1}}"
    [[ "${nnodes}" =~ ^[1-9][0-9]*$ ]] || die "NNODES must be a positive integer: ${nnodes}"
    local preferred=eth0 fallback=bond1
    if (( nnodes > 1 )); then
        preferred=bond1
        fallback=eth0
    fi
    local available
    available="$(gloo_available_ifnames)"
    if [[ -n "${GLOO_SOCKET_IFNAME:-}" ]]; then
        if [[ " ${available} " == *" ${GLOO_SOCKET_IFNAME} "* ]]; then
            log "gloo interface: ${GLOO_SOCKET_IFNAME} (from the environment)"
            return 0
        fi
        warn "GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME} has no IPv4 address here; overriding it"
    fi
    if [[ " ${available} " == *" ${preferred} "* ]]; then
        export GLOO_SOCKET_IFNAME="${preferred}"
    elif [[ " ${available} " == *" ${fallback} "* ]]; then
        warn "${preferred} has no IPv4 address; using ${fallback} (up: ${available:-none})"
        export GLOO_SOCKET_IFNAME="${fallback}"
    else
        die "neither ${preferred} nor ${fallback} has an IPv4 address (up: ${available:-none}); \
set GLOO_SOCKET_IFNAME to the interface that should carry Gloo traffic"
    fi
    log "gloo interface: ${GLOO_SOCKET_IFNAME}"
}

# shellcheck disable=SC2317  # reachable when this file is executed, not sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -euo pipefail
    resolve_gloo_socket_ifname "${1:-${NNODES:-1}}"
    printf 'export GLOO_SOCKET_IFNAME=%s\n' "${GLOO_SOCKET_IFNAME}"
fi
