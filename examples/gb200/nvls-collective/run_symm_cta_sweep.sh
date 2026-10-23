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
Sweep the symmetric-kernel AG / RS CTA counts with nccl-2.32.3-cta preloaded.

Runs run_nvls_collective_benchmark.sh --algos symm --workloads ar,ag,rs,ag+rs --sizes-mib 1024 --nsys
with --cta-configs 'default;ag=A,rs=R;...' for every 1 <= A <= R <= --max-ctas. Stock NCCL clamps
the multicast symmetric kernels to divUp(32, group size) CTAs (1 on 32 GPUs), so the benchmark's
libnccl.so.2 is replaced through LD_PRELOAD by the nccl-2.32.3-cta build
(install/install-nccl-2.32.3-cta.sh), which lets a minCTAs above that cap through, up to 64.
NCCL_SYM_TMA_ENABLE=0 keeps NCCL off the 2.32 TMA symmetric kernels (e.g. AllGather_TmaST), and
--symm-kernel mc pins AG / RS to AllGather_STMC / ReduceScatter_LDMC, so the default config is the
multicast kernels at NCCL's own CTA count too (otherwise 4 GPUs at 1 GiB pick the unicast
AllGather_ST). Pass -- --symm-kernel default to let NCCL choose the kernel again.

LD_PRELOAD only swaps the library cleanly for a torch that loads one libnccl.so.2 through its
soname, i.e. the torch built by install/install-torch-2.13.0-nccl-2.32.3.sh: the official wheel
also loads nvidia-nccl-cu13 by path at import. So before the benchmark, this script checks that
the env's python maps exactly one libnccl, the preloaded one; after it, that every NCCL process
logged the "NCCL version 2.32.3-cta+cuda..." banner (the workers run under nsys).

Run the same command on every node, like run_nvls_collective_benchmark.sh.

Usage:
  run_symm_cta_sweep.sh [options] [-- run_nvls_collective_benchmark.sh options]

Options:
  --nnodes N          Nodes in the job (default: 8)
  --group-sizes LIST  Single-group sizes (default: 32)
  --max-ctas N        Largest ag / rs CTA count of the sweep (default: 8)
  --venv DIR          Virtualenv to activate, or 'none' to keep the current one
                      (default: <repo>/.venv/python3.12-torch2.13-nccl2.32.3)
  --nccl-lib PATH     libnccl.so.2 to preload (default: <repo>/nccl/nccl-2.32.3-cta/build/lib/libnccl.so.2)
  -h, --help          Show this help

Anything after the options (or after --) goes to run_nvls_collective_benchmark.sh and, coming
last, overrides the defaults above, e.g. --sizes-mib 256 or --dry-run.

Examples:
  # Eight trays, one 32-GPU group (run on every node).
  bash examples/gb200/nvls-collective/run_symm_cta_sweep.sh

  # One tray, AG / RS up to 8 CTAs each on 4 GPUs, without nsys reports.
  bash examples/gb200/nvls-collective/run_symm_cta_sweep.sh --nnodes 1 --group-sizes 4 -- --sizes-mib 256
USAGE
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
nnodes=8
group_sizes=32
max_ctas=8
venv="${repo_root}/.venv/python3.12-torch2.13-nccl2.32.3"
nccl_lib="${repo_root}/nccl/nccl-2.32.3-cta/build/lib/libnccl.so.2"
nccl_suffix="-cta"
passthrough=()

die() {
    echo "ERROR: $*" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nnodes) nnodes="$2"; shift 2 ;;
        --group-sizes) group_sizes="$2"; shift 2 ;;
        --max-ctas) max_ctas="$2"; shift 2 ;;
        --venv) venv="$2"; shift 2 ;;
        --nccl-lib) nccl_lib="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; passthrough+=("$@"); break ;;
        *) passthrough+=("$1"); shift ;;
    esac
done

[[ "$max_ctas" =~ ^[0-9]+$ ]] && (( max_ctas >= 1 && max_ctas <= 64 )) || die "--max-ctas must be in 1..64, got ${max_ctas}"
dry_run=false
for arg in ${passthrough[@]+"${passthrough[@]}"}; do
    [[ "$arg" == --dry-run ]] && dry_run=true
done

cta_configs="default"
for (( ag = 1; ag <= max_ctas; ag++ )); do
    for (( rs = ag; rs <= max_ctas; rs++ )); do
        cta_configs+=";ag=${ag},rs=${rs}"
    done
done

if [[ "$venv" != none ]]; then
    [[ -f "${venv}/bin/activate" ]] || die "no virtualenv at ${venv}; pass --venv DIR or --venv none"
    # shellcheck source=/dev/null
    source "${venv}/bin/activate"
fi
# Nsight Compute unpacked into the repo, for --ncu passed through.
[[ -d "${repo_root}/NVIDIA-Nsight-Compute-2026.3" ]] && export PATH="${repo_root}/NVIDIA-Nsight-Compute-2026.3:${PATH}"
# GB200 images export these; they would override every communicator's minCTAs (the launcher unsets them too).
unset NCCL_MIN_CTAS NCCL_MIN_NCHANNELS

[[ -f "$nccl_lib" ]] || die "${nccl_lib} not found; build it with install/install-nccl-2.32.3-cta.sh or pass --nccl-lib"
nccl_lib="$(cd -- "$(dirname -- "$nccl_lib")" && pwd -P)/$(basename -- "$nccl_lib")"
grep -aq "NCCL version 2.32.3${nccl_suffix}+cuda" "$nccl_lib" \
    || die "${nccl_lib} is not the nccl-2.32.3${nccl_suffix} build (no \"NCCL version 2.32.3${nccl_suffix}+cuda\" banner)"
preload="${nccl_lib}${LD_PRELOAD:+:${LD_PRELOAD}}"

cd "$repo_root"
if ! $dry_run; then
    echo "=== Checking that torch maps only ${nccl_lib}" >&2
    LD_PRELOAD="$preload" uv run --no-sync python - "$nccl_lib" <<'PY'
import os
import sys

import torch  # noqa: F401  (libtorch_cuda.so needs libnccl.so.2; the official wheel also preloads its own)


want = os.path.realpath(sys.argv[1])
with open("/proc/self/maps") as maps:
    paths = {line.split(None, 5)[5].strip() for line in maps if len(line.split(None, 5)) == 6}
loaded = sorted({os.path.realpath(p) for p in paths if os.path.basename(p).startswith("libnccl.so")})
if loaded != [want]:
    sys.exit(f"ERROR: torch {torch.__version__} maps libnccl {loaded}, expected only {want}; LD_PRELOAD needs a torch "
             "without nvidia-nccl-cu13 (install/install-torch-2.13.0-nccl-2.32.3.sh)")
print(f"  [ok]   torch {torch.__version__} maps only {want}", file=sys.stderr)
PY
fi

log="$(mktemp -t symm-cta-sweep.XXXXXX)"
set +e
# NCCL_SYM_TMA_ENABLE=0 masks the 2.32 TMA symmetric kernels (sym_kernels.cc kernelMask_Tma:
# AllGather_TmaST/TmaSTMC, ReduceScatter_TmaLD, AllReduce_RSxTmaLD_AGxTmaST). Otherwise the cost
# model picks AllGather_TmaST at 1 GiB, which is outside the symm pass's SYMM_KERNELS, and the
# sweep would leave the multicast kernels (AllGather_STMC, ReduceScatter_LDMC) it is meant to size.
# --symm-kernel mc does the same for the default config, which the CTA checks do not cover.
NCCL_SYM_TMA_ENABLE=0 LD_PRELOAD="$preload" bash "${script_dir}/run_nvls_collective_benchmark.sh" \
    --nnodes "$nnodes" \
    --group-sizes "$group_sizes" \
    --sizes-mib 1024 \
    --algos symm \
    --workloads ar,ag,rs,ag+rs \
    --nsys \
    --cta-configs "$cta_configs" \
    --symm-kernel mc \
    ${passthrough[@]+"${passthrough[@]}"} 2>&1 | tee "$log"
status="${PIPESTATUS[0]}"
set -e
$dry_run && { rm -f "$log"; exit "$status"; }

# Every NCCL process logs its banner once at NCCL_DEBUG=INFO (init.cc showVersion); the launcher
# writes those logs to <run>/<algo>/nccl-raw/. A banner without the suffix is a stock libnccl.
output_dir="$(sed -n 's/.* output_dir=\([^ ]*\).*/\1/p' "$log" | tail -1)"
rm -f "$log"
if [[ -z "$output_dir" ]]; then
    echo "WARNING: no output_dir in the launcher output; skipped the libnccl banner check" >&2
    exit "$status"
fi
banners="$(cat "${output_dir}"/*/nccl-raw/*.log 2>/dev/null | grep -o "NCCL version [^ ]*" | sort | uniq -c || true)"
echo "=== libnccl banners in ${output_dir}/*/nccl-raw" >&2
echo "${banners:-  none}" >&2
if [[ -z "$banners" ]] || grep -qv "NCCL version 2.32.3${nccl_suffix}+" <<<"$banners"; then
    echo "ERROR: not every NCCL process ran nccl-2.32.3${nccl_suffix}; results above the multicast cap are not valid" >&2
    exit 1
fi
exit "$status"
