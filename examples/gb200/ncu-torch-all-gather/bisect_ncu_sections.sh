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
Find the ncu section that hangs the NCCL AllGather kernel: run run_ncu_torch_all_gather.sh once
per section and report which ones finish, hang or fail.

With ncu 2026.4, --replay-mode application --set full got through 38 application replay passes
and then hung in pass 39 (the kernel never completed on any rank). Each pass collects a different
group of sections, so one section's collection method stops the kernel. This script profiles one
section at a time:

  run_ncu_torch_all_gather.sh --skip-baseline --replay-mode application --timeout 300 \
      --metrics gpu__time_duration.sum --output-dir <dir>/<section> <launcher args> -- --section <section>

--metrics replaces --set with one single-pass metric; ncu collects the metric and the --section
together. A hung section is killed by the launcher's --timeout; leftover workers are killed before
the next section starts.

Usage:
  bash examples/gb200/ncu-torch-all-gather/bisect_ncu_sections.sh [options] [-- launcher args]

Options:
  --sections LIST    Comma-separated ncu section identifiers
                     (default: Nvlink_Topology,Nvlink_Tables,Nvlink,SourceCounters,InstructionStats,
                     PmSampling,PmSampling_WarpStates,WarpStateStats). `ncu --list-sets` lists the
                     sections of --set full. Identifiers missing from `ncu --list-sections` are
                     reported as unknown and skipped.
  --output-dir DIR   (default: results/gb200/ncu-torch-all-gather/sections-<time>)
  --dry-run          Print the launcher commands without running them
  -h, --help

Launcher args go to run_ncu_torch_all_gather.sh after the defaults above, so they can override
them, e.g. -- --ncu /path/to/ncu --timeout 600 --algo ring. They must not contain "--"; the
script passes --section through it.

Outputs in --output-dir: <section>.log (launcher output), <section>/ (launcher --output-dir) and
summary.txt (one line per section: ok | hang | fail | unknown).
USAGE
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
launcher="${script_dir}/run_ncu_torch_all_gather.sh"
worker="${script_dir}/ncu_torch_all_gather.py"

sections="Nvlink_Topology,Nvlink_Tables,Nvlink,SourceCounters,InstructionStats,PmSampling,PmSampling_WarpStates,WarpStateStats"
output_dir=""
dry_run=false
launcher_args=()

die() {
    echo "ERROR: $*" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sections) sections="$2"; shift 2 ;;
        --output-dir) output_dir="$2"; shift 2 ;;
        --dry-run) dry_run=true; shift ;;
        --) shift; launcher_args=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1 (see --help)" ;;
    esac
done

IFS=, read -r -a section_list <<<"$sections"
(( ${#section_list[@]} > 0 )) || die "--sections is empty"
# The launcher's own --ncu decides which ncu lists the sections.
ncu_bin="ncu"
for (( i = 0; i < ${#launcher_args[@]}; i++ )); do
    [[ "${launcher_args[i]}" != "--" ]] || die "launcher args must not contain \"--\"; the script passes --section through it"
    if [[ "${launcher_args[i]}" == "--ncu" ]] && (( i + 1 < ${#launcher_args[@]} )); then
        ncu_bin="${launcher_args[i + 1]}"
    fi
done
output_dir="${output_dir:-${repo_root}/results/gb200/ncu-torch-all-gather/sections-$(date +%Y%m%d-%H%M%S)}"

# section_cmd <section>: the launcher command for one section, in the array "cmd".
section_cmd() {
    cmd=(
        bash "$launcher" --skip-baseline --replay-mode application --timeout 300
        --metrics gpu__time_duration.sum --output-dir "${output_dir}/$1"
        ${launcher_args[@]+"${launcher_args[@]}"} -- --section "$1"
    )
}

if $dry_run; then
    for s in "${section_list[@]}"; do
        section_cmd "$s"
        printf '%q ' "${cmd[@]}"
        echo
    done
    exit 0
fi

command -v "$ncu_bin" >/dev/null || die "${ncu_bin} not found; pass -- --ncu PATH"
known_sections="$("$ncu_bin" --list-sections | awk '{print $1}')" || die "${ncu_bin} --list-sections failed"
mkdir -p "$output_dir"
[[ -z "$(ls -A "$output_dir")" ]] || die "${output_dir} is not empty"
summary="${output_dir}/summary.txt"

# A hung NCCL kernel can outlive the launcher's timeout; the next section must start on idle GPUs.
kill_leftover_workers() {
    pgrep -f "$worker" >/dev/null || return 0
    echo "killing leftover workers of ${worker}"
    pkill -KILL -f "$worker" || true
    for _ in $(seq 60); do
        pgrep -f "$worker" >/dev/null || return 0
        sleep 1
    done
    die "workers of ${worker} still alive 60s after SIGKILL; check the GPUs before going on"
}

for s in "${section_list[@]}"; do
    if ! grep -qxF "$s" <<<"$known_sections"; then
        status=unknown
        detail="not in ${ncu_bin} --list-sections"
    else
        section_cmd "$s"
        log="${output_dir}/${s}.log"
        echo "=== section ${s}: ${cmd[*]}"
        rc=0
        "${cmd[@]}" >"$log" 2>&1 || rc=$?
        if (( rc == 0 )); then
            status=ok
        elif grep -q "did not finish in .* (hang)" "$log"; then
            status=hang
        else
            status=fail
        fi
        # The launcher's die message, else the first ncu error.
        detail="$(grep -m1 "^ERROR: " "$log" || grep -m1 "==ERROR==" "$log" || true)"
        detail="${detail:-exit ${rc}}; log ${log}"
        [[ "$status" == ok ]] || kill_leftover_workers
    fi
    printf '%-24s %-8s %s\n' "$s" "$status" "$detail" | tee -a "$summary"
done

echo "=== summary: ${summary}"
cat "$summary"
