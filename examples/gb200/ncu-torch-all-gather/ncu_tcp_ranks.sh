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

# ncu_tcp_ranks.sh <nranks> <ncu cmd...> -- <worker cmd...>
#
# run_ncu_torch_all_gather.sh --communicator tcp: start one ncu per rank, each wrapping one worker,
# with the per-rank variables torchrun would set (RANK, LOCAL_RANK, WORLD_SIZE, LOCAL_WORLD_SIZE;
# MASTER_ADDR / MASTER_PORT come from the caller). Every output line gets a "[rank <r>] " prefix.
# Exits 0 when every rank does, else with the status of the last failing rank.

set -euo pipefail

(( $# >= 3 )) || { echo "usage: $0 <nranks> <ncu cmd...> -- <worker cmd...>" >&2; exit 2; }
nranks="$1"
shift
ncu_cmd=()
while [[ $# -gt 0 && "$1" != -- ]]; do
    ncu_cmd+=("$1")
    shift
done
[[ $# -gt 1 ]] || { echo "$0: missing -- <worker cmd...>" >&2; exit 2; }
shift

pids=()
for (( r = 0; r < nranks; r++ )); do
    (
        RANK=$r LOCAL_RANK=$r WORLD_SIZE=$nranks LOCAL_WORLD_SIZE=$nranks "${ncu_cmd[@]}" "$@" 2>&1 \
            | awk -v prefix="[rank ${r}] " '{ print prefix $0; fflush() }'
    ) &
    pids+=($!)
done
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=$?
done
exit "$status"
