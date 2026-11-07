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
Profile the NCCL AllGather kernel one ncu section (or group of sections) at a time: run
run_ncu_torch_all_gather.sh once per item with application replay, put every item's results in
its own folder, and report which items finish, hang or fail and how many replay passes each one
needed. By default: every section of --set full, --algo symm, 1 GiB per rank.

With ncu 2026.4, --replay-mode application --set full hangs in pass 39 for nvls and ring, at
256 MiB and at 1 MiB: all ranks enter the kernel and it never completes. Two explanations are
told apart here:
  - a section's collection method stops the kernel: that section hangs on its own;
  - something breaks after 38 relaunches: every section finishes on its own, but a group of
    sections that needs 39 or more passes hangs.

Each item of the list is one launcher run:

  run_ncu_torch_all_gather.sh --skip-baseline --replay-mode application --timeout 1800 \
      --algo <algo> --size-mib <size> --output-dir <dir>/<item> \
      --section <s1> [--section <s2> ...] <launcher args>

An item is a section identifier, or several joined with "+" (e.g. SourceCounters+InstructionStats)
to collect them in one run. A hang is ncu giving up on a pass ("Failed to wait for profiling") or
the launcher's --timeout; leftover workers are killed before the next item starts.

Usage:
  bash examples/gb200/ncu-torch-all-gather/bisect_ncu_sections.sh [options] [-- launcher args]

Options:
  --algo NAME        ring | nvls | symm | p2p (default: symm)
  --size-mib N       All-gather output size per rank in MiB (default: 1024)
  --set NAME         Test every section of the ncu set NAME, one at a time (default: full). The
                     members are read from the *.section files in --section-dir.
  --sections LIST    Comma-separated items instead of --set. Sections missing from
                     `ncu --list-sections` are reported as unknown and skipped.
  --section-dir DIR  Section files for --set (default: the sections/ directory next to ncu)
  --output-dir DIR   (default: results/gb200/ncu-torch-all-gather/<algo>-all-gather-<time>)
  --dry-run          Print the launcher commands without running them
  -h, --help

ncu: <repo>/NVIDIA-Nsight-Compute-2026.4 is put first in PATH when it exists, as the launcher
does; the launcher refuses ncu < 2026.4.

Launcher args go to run_ncu_torch_all_gather.sh after the defaults above, so they can override
them, e.g. -- --ncu /path/to/ncu --warmup 1 --timeout 3600.

Outputs in --output-dir:
  <item>/        the launcher's output for that item (config.txt, ncu.log, nccl-logs/,
                 all_gather.ncu-repz, summary.csv) plus launcher.log, its console output
  summary.txt    one line per item: <item> ok|hang|fail|unknown <passes> <detail>
USAGE
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
launcher="${script_dir}/run_ncu_torch_all_gather.sh"
worker="${script_dir}/ncu_torch_all_gather.py"

algo="symm"
size_mib=1024
sections=""
set_name="full"
section_dir=""
output_dir=""
dry_run=false
launcher_args=()

die() {
    echo "ERROR: $*" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --algo) algo="$2"; shift 2 ;;
        --size-mib) size_mib="$2"; shift 2 ;;
        --set) set_name="$2"; sections=""; shift 2 ;;
        --sections) sections="$2"; set_name=""; shift 2 ;;
        --section-dir) section_dir="$2"; shift 2 ;;
        --output-dir) output_dir="$2"; shift 2 ;;
        --dry-run) dry_run=true; shift ;;
        --) shift; launcher_args=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1 (see --help)" ;;
    esac
done

[[ "$algo" =~ ^(ring|nvls|symm|p2p)$ ]] || die "--algo must be ring, nvls, symm or p2p"
ncu_home="${repo_root}/NVIDIA-Nsight-Compute-2026.4"
[[ -d "$ncu_home" ]] && export PATH="${ncu_home}:${PATH}"
# The launcher's own --ncu decides which ncu lists the sections.
ncu_bin="ncu"
for (( i = 0; i + 1 < ${#launcher_args[@]}; i++ )); do
    [[ "${launcher_args[i]}" == "--ncu" ]] && ncu_bin="${launcher_args[i + 1]}"
done

# set_members <set>: section identifiers whose file has a Sets { Identifier: "<set>" } entry.
set_members() {
    local dir="$section_dir"
    if [[ -z "$dir" ]]; then
        local ncu_path
        ncu_path="$(command -v "$ncu_bin")" || die "${ncu_bin} not found; pass -- --ncu PATH or --section-dir"
        dir="$(dirname -- "$(readlink -f -- "$ncu_path")")/sections"
    fi
    compgen -G "${dir}/*.section" >/dev/null || die "no *.section files in ${dir}; pass --section-dir"
    # Plain POSIX awk (no gawk ENDFILE): mawk is the default awk on Ubuntu.
    awk -v pat="^[[:space:]]+Identifier:[[:space:]]*\"$1\"" '
        function flush() { if (member && id != "") print id }
        FNR == 1 { flush(); id = ""; member = 0 }
        /^Identifier:/ && id == "" { id = $2; gsub(/"/, "", id) }
        $0 ~ pat { member = 1 }
        END { flush() }
    ' "${dir}"/*.section | sort -u
}

if [[ -n "$set_name" ]]; then
    sections="$(set_members "$set_name" | paste -sd, -)"
    [[ -n "$sections" ]] || die "no section of set ${set_name} found"
    echo "set ${set_name}: ${sections}"
fi
IFS=, read -r -a item_list <<<"$sections"
(( ${#item_list[@]} > 0 )) || die "--sections is empty"
output_dir="${output_dir:-${repo_root}/results/gb200/ncu-torch-all-gather/${algo}-all-gather-$(date +%Y%m%d-%H%M%S)}"

# item_cmd <item>: the launcher command for one item, in the array "cmd".
item_cmd() {
    local s
    cmd=(
        bash "$launcher" --skip-baseline --replay-mode application --timeout 1800
        --algo "$algo" --size-mib "$size_mib" --output-dir "${output_dir}/$1"
    )
    IFS=+ read -r -a parts <<<"$1"
    for s in "${parts[@]}"; do cmd+=(--section "$s"); done
    cmd+=(${launcher_args[@]+"${launcher_args[@]}"})
}

if $dry_run; then
    for item in "${item_list[@]}"; do
        item_cmd "$item"
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

# A hung NCCL kernel can outlive the launcher; the next item must start on idle GPUs.
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

for item in "${item_list[@]}"; do
    IFS=+ read -r -a parts <<<"$item"
    missing=""
    for s in "${parts[@]}"; do
        grep -qxF "$s" <<<"$known_sections" || missing+=" ${s}"
    done
    passes="-"
    if [[ -n "$missing" ]]; then
        status=unknown
        detail="not in ${ncu_bin} --list-sections:${missing}"
    else
        item_cmd "$item"
        log="${output_dir}/${item}.log"
        echo "=== ${item}: ${cmd[*]}"
        rc=0
        "${cmd[@]}" >"$log" 2>&1 || rc=$?
        if (( rc == 0 )); then
            status=ok
        elif grep -qE "did not finish in .* \(hang\)|ncu gave up waiting" "$log"; then
            status=hang
        else
            status=fail
        fi
        # The highest replay pass ncu reached; application replay prints one line per rank and pass.
        passes="$(grep -o 'replay pass [0-9]*' "$log" | awk '{print $3}' | sort -n | tail -n 1 || true)"
        passes="${passes:-1}"
        # The launcher's die message, else the first ncu error.
        detail="$(grep -m1 "^ERROR: " "$log" || grep -m1 "==ERROR==" "$log" || true)"
        # The launcher needs an empty --output-dir, so its console log is moved in afterwards.
        if [[ -d "${output_dir}/${item}" ]]; then
            mv "$log" "${output_dir}/${item}/launcher.log"
            log="${output_dir}/${item}/launcher.log"
        fi
        detail="${detail:-exit ${rc}}; log ${log}"
        [[ "$status" == ok ]] || kill_leftover_workers
    fi
    printf '%-40s %-8s %-6s %s\n' "$item" "$status" "$passes" "$detail" | tee -a "$summary"
    # A wrong algorithm fails every item the same way; stop instead of profiling the wrong kernel.
    if [[ "$detail" == *"NCCL did not run --algo"* ]]; then
        die "NCCL did not run --algo ${algo}; fix the NCCL environment first (${log})"
    fi
done

echo "=== summary: ${summary} (item, status, replay passes, detail)"
cat "$summary"
