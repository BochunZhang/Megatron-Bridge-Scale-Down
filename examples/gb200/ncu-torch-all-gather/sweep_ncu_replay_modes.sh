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

set -euo pipefail

usage() {
    cat <<'USAGE'
Find which ncu replay mode + communicator combinations can profile the NCCL kernels: run
run_ncu_torch_all_gather.sh once per combination, put each one's results in its own folder and
report which finish, hang or fail. By default: the all-gather and a reduce-scatter running
together (--ag-rs), --algo symm, 1 GiB per rank, the SpeedOfLight section.

Why: ncu 2026.4 + shmem + app-range fails at the start of the NVTX range ("Failed to profile
"range""); the 2026.4 CLI docs list only kernel and range replay for shmem, the release notes add
application replay. Only range and app-range keep the AG and RS concurrent.

A combination is <replay>:<communicator>[:nolockstep], e.g. app-range:tcp:nolockstep:
  replay        kernel | application | range | app-range
  communicator  shmem | tcp | none
  nolockstep    tcp without --lockstep-kernel-launch / --lockstep-nvtx-include
Default list (application:shmem first as the control that worked in #13):
  application:shmem,range:shmem,app-range:shmem,range:tcp,app-range:tcp,app-range:tcp:nolockstep,app-range:none

Each combination is one launcher run:

  run_ncu_torch_all_gather.sh --skip-baseline --replay-mode <replay> --communicator <comm> \
      [--no-lockstep] --timeout <sec> --algo <algo> [--ag-rs] --size-mib <size> \
      --output-dir <dir>/<replay>_<comm>[_nolockstep] --section <s> ... <launcher args>

A hang is ncu giving up on a pass ("Failed to wait for profiling") or the launcher's --timeout;
leftover workers and ncu instances are killed before the next combination starts.

Usage:
  bash examples/gb200/ncu-torch-all-gather/sweep_ncu_replay_modes.sh [options] [-- launcher args]

Options:
  --combos LIST      Comma-separated combinations (default: the list above)
  --algo NAME        ring | nvls | symm | p2p (default: symm)
  --no-ag-rs         Profile the all-gather alone
  --size-mib N       All-gather output size per rank in MiB (default: 1024)
  --section NAME     Collect this ncu section; repeat for several (default: SpeedOfLight)
  --timeout SEC      Launcher --timeout per combination (default: 900)
  --output-dir DIR   (default: results/gb200/ncu-torch-all-gather/<algo>-replay-sweep-<time>)
  --dry-run          Print the launcher commands without running them
  -h, --help

Launcher args go to run_ncu_torch_all_gather.sh after the defaults above, so they can override
them, e.g. -- --ncu /path/to/ncu --warmup 1 -- --devices 0.

Outputs in --output-dir:
  <combo>/       the launcher's output (config.txt, ncu.log, nccl-logs/, reports, summary.csv)
                 plus launcher.log, its console output
  summary.txt    one line per combination: <combo> ok|hang|fail <passes> <detail>
USAGE
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
launcher="${script_dir}/run_ncu_torch_all_gather.sh"

combos="application:shmem,range:shmem,app-range:shmem,range:tcp,app-range:tcp,app-range:tcp:nolockstep,app-range:none"
algo="symm"
ag_rs=true
size_mib=1024
sections=()
timeout_sec=900
output_dir=""
dry_run=false
launcher_args=()

die() {
    echo "ERROR: $*" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --combos) combos="$2"; shift 2 ;;
        --algo) algo="$2"; shift 2 ;;
        --no-ag-rs) ag_rs=false; shift ;;
        --size-mib) size_mib="$2"; shift 2 ;;
        --section) sections+=("$2"); shift 2 ;;
        --timeout) timeout_sec="$2"; shift 2 ;;
        --output-dir) output_dir="$2"; shift 2 ;;
        --dry-run) dry_run=true; shift ;;
        --) shift; launcher_args=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1 (see --help)" ;;
    esac
done

[[ "$algo" =~ ^(ring|nvls|symm|p2p)$ ]] || die "--algo must be ring, nvls, symm or p2p"
(( ${#sections[@]} > 0 )) || sections=(SpeedOfLight)
case_args=()
$ag_rs && case_args=(--ag-rs)
IFS=, read -r -a combo_list <<<"$combos"
(( ${#combo_list[@]} > 0 )) || die "--combos is empty"
for combo in "${combo_list[@]}"; do
    [[ "$combo" =~ ^(kernel|application|range|app-range):(shmem|tcp|none)(:nolockstep)?$ ]] \
        || die "bad combination ${combo}: expected <replay>:<communicator>[:nolockstep]"
    [[ "$combo" != *:nolockstep || "$combo" == *:tcp:nolockstep ]] || die "${combo}: nolockstep needs tcp"
done
output_dir="${output_dir:-${repo_root}/results/gb200/ncu-torch-all-gather/${algo}-replay-sweep-$(date +%Y%m%d-%H%M%S)}"

# combo_cmd <combo>: the launcher command for one combination, in the array "cmd"; its folder in "dir".
combo_cmd() {
    local replay comm lockstep s
    IFS=: read -r replay comm lockstep <<<"$1"
    dir="${output_dir}/${1//:/_}"
    cmd=(
        bash "$launcher" --skip-baseline --replay-mode "$replay" --communicator "$comm"
        --timeout "$timeout_sec" --algo "$algo" ${case_args[@]+"${case_args[@]}"} --size-mib "$size_mib"
        --output-dir "$dir"
    )
    [[ "$lockstep" == nolockstep ]] && cmd+=(--no-lockstep)
    for s in "${sections[@]}"; do cmd+=(--section "$s"); done
    cmd+=(${launcher_args[@]+"${launcher_args[@]}"})
}

if $dry_run; then
    for combo in "${combo_list[@]}"; do
        combo_cmd "$combo"
        printf '%q ' "${cmd[@]}"
        echo
    done
    exit 0
fi

mkdir -p "$output_dir"
[[ -z "$(ls -A "$output_dir")" ]] || die "${output_dir} is not empty"
summary="${output_dir}/summary.txt"

# A hung NCCL kernel can outlive the launcher; the next combination must start on idle GPUs. The
# workers carry the worker script in their command line, the tcp ncu instances the output folder.
kill_leftovers() {
    local pattern="ncu_torch_(all_gather|ag_rs_overlap)\\.py|${1}/"
    pgrep -f "$pattern" >/dev/null || return 0
    echo "killing leftover workers / ncu of ${1}"
    pkill -KILL -f "$pattern" || true
    for _ in $(seq 60); do
        pgrep -f "$pattern" >/dev/null || return 0
        sleep 1
    done
    die "processes matching ${pattern} still alive 60s after SIGKILL; check the GPUs before going on"
}

for combo in "${combo_list[@]}"; do
    combo_cmd "$combo"
    log="${output_dir}/${combo//:/_}.log"
    echo "=== ${combo}: ${cmd[*]}"
    rc=0
    "${cmd[@]}" >"$log" 2>&1 || rc=$?
    if (( rc == 0 )); then
        status=ok
    elif grep -qE "did not finish in .* \(hang\)|ncu gave up waiting" "$log"; then
        status=hang
    else
        status=fail
    fi
    # The highest replay pass ncu reported, if any.
    passes="$(grep -o 'replay pass [0-9]*' "$log" | awk '{print $3}' | sort -n | tail -n 1 || true)"
    passes="${passes:--}"
    # The launcher's die message, else the first ncu error.
    detail="$(grep -m1 "^ERROR: " "$log" || grep -m1 "==ERROR==" "$log" || true)"
    # The launcher needs an empty --output-dir, so its console log is moved in afterwards.
    if [[ -d "$dir" ]]; then
        mv "$log" "${dir}/launcher.log"
        log="${dir}/launcher.log"
    fi
    detail="${detail:-exit ${rc}}; log ${log}"
    [[ "$status" == ok ]] || kill_leftovers "$dir"
    printf '%-28s %-6s %-6s %s\n' "$combo" "$status" "$passes" "$detail" | tee -a "$summary"
    # A wrong algorithm fails every combination the same way.
    if [[ "$detail" == *"NCCL did not run --algo"* ]]; then
        die "NCCL did not run --algo ${algo}; fix the NCCL environment first (${log})"
    fi
done

echo "=== summary: ${summary} (combination, status, replay passes, detail)"
cat "$summary"
