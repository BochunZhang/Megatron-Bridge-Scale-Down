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
AllGather only, NCCL 2.32.3 only: ring vs symm-p2p at the same CTA count.

Two run_nvls_collective_benchmark.sh passes into one run directory, --workloads ag, one group of
every GPU (4 x nnodes ranks):

  1. ring      NCCL_ALGO=Ring at NCCL's own channel count. The CTAs of every size's AllGather are
               read from ring/results.jsonl (observed_ctas, comm rank 0's TUNING lines) on node 0
               and handed to the other nodes over a TCPStore.
  2. symm-p2p  The same sizes on registered windows with multicast off (AllGather_ST, or
               AllGather_TmaST with --p2p-kernel tma), every communicator at minCTAs = maxCTAs = a
               ring count. --cta-configs takes one entry per distinct ring count, and each entry
               reruns every size, so with k distinct counts symm-p2p runs every size k times; the
               table pairs each ring case only with the symm-p2p case at its own count.

Stock NCCL 2.32.3 already lets minCTAs/maxCTAs reach the unicast symmetric kernels up to 64
(tuning/sym_model/lsa.cc maxBlocksLsa caps only the multicast kernels), and ring never uses more
than MAXCHANNELS = 64, so no CTA patch is needed. The symmetric scheduler picks among symmetric
kernels only (scheduler/symmetric_sched.cc), so a large forced CTA count cannot push symm-p2p
onto the classic ring kernel; the benchmark still fails a case that leaves the unicast kernels.

Process-wide CTA variables (NCCL_MIN_CTAS, NCCL_MIN_NCHANNELS, ...) are unset for both passes, so
ring runs at NCCL's own count. NCCL_SYM_TMA_ENABLE is set explicitly from --p2p-kernel.

Afterwards node 0 writes ring_vs_symm_p2p_all_gather.md and exits non-zero if a size has
no symm-p2p case at ring's count, ran another kernel or CTA count, or failed its output check;
every node checks that its NCCL processes logged an "NCCL version 2.32.3" banner.

The environment is always <repo>/.venv/python3.12-torch2.13-nccl2.32.3 (activated here).

Output: <repo>/results/gb200/nvls-collective/gpu-num-<4*nnodes>/<yyyymmdd-hhmmss>/
  ring/, symm-p2p/                 each pass: results.jsonl, metadata.json, cases/, nccl-raw/, nsys/
  compare.md                       the launcher's tables over both passes
  ring_ag_ctas.node<N>.json        ring's CTAs per [ranks, MiB, CTAs] as node N received them
  ring_vs_symm_p2p_all_gather.md   the paired table (node 0)

Run the same command on every node.

Usage:
  run_all_gather_nccl_2_32_3_ring_vs_symm_p2p.sh [options]

Options:
  --nnodes N           Nodes in the job, 4 GPUs each (default: NNODES or 1)
  --sizes-mib LIST     Full AllGather output sizes in MiB (default: 1024)
  --nsys               One Nsight Systems report per rank per case, with NVLink GPU metrics
  --p2p-kernel st|tma  st: AllGather_ST, the kernel of the NCCL 2.29.7 runs (NCCL_SYM_TMA_ENABLE=0);
                       tma: AllGather_TmaST, the 2.32.3 default (NCCL_SYM_TMA_ENABLE=1) (default: st)
  --node-rank N        This node's rank (default: NODE_RANK, RANK or 0)
  --master-addr ADDR   Rendezvous host (default: MASTER_ADDR, DLC_MASTER_ADDR or 127.0.0.1)
  --master-port PORT   ring uses PORT, symm-p2p PORT+10, the CTA handshake PORT+20 (default: 29611)
  --nccl-lib PATH      libnccl.so.2 to LD_PRELOAD, e.g. a nccl-2.32.3-cta build (default: torch's own)
  --dry-run            Print the launcher commands only; symm-p2p shows a placeholder CTA config
  -h, --help           Show this help

Examples:
  # Sixteen trays: all 64 GPUs in one AllGather, 1 GiB; run on every node.
  bash examples/gb200/nvls-collective/run_all_gather_nccl_2_32_3_ring_vs_symm_p2p.sh --nnodes 16

  # Eight trays: all 32 GPUs, 1 GiB and 256 MiB.
  bash examples/gb200/nvls-collective/run_all_gather_nccl_2_32_3_ring_vs_symm_p2p.sh \
    --nnodes 8 --sizes-mib 256,1024

  # Same with nsys reports and the TMA kernel.
  bash examples/gb200/nvls-collective/run_all_gather_nccl_2_32_3_ring_vs_symm_p2p.sh \
    --nnodes 8 --sizes-mib 1024 --nsys --p2p-kernel tma
USAGE
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
launcher="${script_dir}/run_nvls_collective_benchmark.sh"
helper="${script_dir}/all_gather_nccl_2_32_3_ring_vs_symm_p2p.py"

nnodes="${NNODES:-1}"
node_rank="${NODE_RANK:-${RANK:-0}}"
master_addr="${MASTER_ADDR:-${DLC_MASTER_ADDR:-127.0.0.1}}"
master_port=29611
sizes_mib=1024
nsys=false
p2p_kernel=st
venv="${repo_root}/.venv/python3.12-torch2.13-nccl2.32.3"
nccl_lib=""
dry_run=false
# Minutes node 0 waits for the other nodes to read ring's CTA counts.
handshake_timeout_min=30

die() {
    echo "ERROR: $*" >&2
    exit 1
}
need_value() {
    [[ $# -ge 2 && -n "$2" ]] || die "$1 requires a value"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nnodes) need_value "$@"; nnodes="$2"; shift 2 ;;
        --sizes-mib) need_value "$@"; sizes_mib="$2"; shift 2 ;;
        --nsys) nsys=true; shift ;;
        --p2p-kernel) need_value "$@"; p2p_kernel="$2"; shift 2 ;;
        --node-rank) need_value "$@"; node_rank="$2"; shift 2 ;;
        --master-addr) need_value "$@"; master_addr="$2"; shift 2 ;;
        --master-port) need_value "$@"; master_port="$2"; shift 2 ;;
        --nccl-lib) need_value "$@"; nccl_lib="$2"; shift 2 ;;
        --dry-run) dry_run=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1 (see --help)" ;;
    esac
done

[[ "$nnodes" =~ ^[1-9][0-9]*$ ]] || die "--nnodes must be a positive integer"
[[ "$node_rank" =~ ^[0-9]+$ ]] && (( node_rank < nnodes )) || die "--node-rank must be in [0, nnodes)"
[[ "$master_port" =~ ^[0-9]+$ ]] && (( master_port > 1 && master_port + 20 < 65536 )) \
    || die "--master-port must be in 2..65515"
[[ "$sizes_mib" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]] || die "--sizes-mib must be a comma-separated list of MiB"
# Every GPU of the job takes part in each AllGather: one group of 4 x nnodes ranks.
world_size=$(( nnodes * 4 ))
case "$p2p_kernel" in
    st) sym_tma_enable=0 ;;
    tma) sym_tma_enable=1 ;;
    *) die "--p2p-kernel accepts st or tma; got ${p2p_kernel}" ;;
esac
if (( nnodes > 1 )) && [[ "$master_addr" == 127.0.0.1 ]]; then
    die "multi-node runs require --master-addr or MASTER_ADDR"
fi

[[ -f "${venv}/bin/activate" ]] || die "no virtualenv at ${venv}; build it with install/install-torch-2.13.0-nccl-2.32.3.sh"
# shellcheck source=/dev/null
source "${venv}/bin/activate"
# GB200 images export some of these; they would decide ring's channel count instead of NCCL.
unset NCCL_MIN_CTAS NCCL_MAX_CTAS NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS NCCL_MIN_NRINGS NCCL_MAX_NRINGS \
    NCCL_SYM_CTAS NCCL_SYM_KERNEL
export NCCL_SYM_TMA_ENABLE="$sym_tma_enable"
if [[ -n "$nccl_lib" ]]; then
    [[ -f "$nccl_lib" ]] || die "${nccl_lib} not found"
    nccl_lib="$(cd -- "$(dirname -- "$nccl_lib")" && pwd -P)/$(basename -- "$nccl_lib")"
    grep -aq "NCCL version 2.32.3" "$nccl_lib" || die "${nccl_lib} is not an NCCL 2.32.3 build"
    export LD_PRELOAD="${nccl_lib}${LD_PRELOAD:+:${LD_PRELOAD}}"
fi

cd "$repo_root"
if ! $dry_run; then
    # torch.cuda.nccl.version() is ncclGetVersion() of the libnccl torch loaded.
    uv run --no-sync python -c '
import sys
import torch
version = torch.cuda.nccl.version()
if tuple(version) != (2, 32, 3):
    sys.exit(f"ERROR: torch {torch.__version__} loads NCCL {version}, this script needs 2.32.3")
' || die "torch in ${venv} does not load NCCL 2.32.3; pass --nccl-lib"
fi

common=(
    --nnodes "$nnodes"
    --node-rank "$node_rank"
    --master-addr "$master_addr"
    --workloads ag
    --group-sizes "$world_size"
    --sizes-mib "$sizes_mib"
)
$nsys && common+=(--nsys)
$dry_run && common+=(--dry-run)

# run_pass <log> <launcher options>: one launcher pass, its output also in <log>.
run_pass() {
    local log="$1" status
    shift
    set +e
    bash "$launcher" "${common[@]}" "$@" 2>&1 | tee "$log"
    status="${PIPESTATUS[0]}"
    set -e
    (( status == 0 )) || die "launcher failed (exit ${status}): $*"
}

log="$(mktemp -t ag-ring-vs-p2p.XXXXXX)"
trap 'rm -f "$log"' EXIT

echo "=== 1/2 ring, NCCL's own channel count" >&2
run_pass "$log" --algos ring --master-port "$master_port"
output_dir="$(sed -n 's/.* output_dir=\([^ ]*\).*/\1/p' "$log" | tail -1)"
[[ -n "$output_dir" ]] || die "no output_dir in the ring pass output"

if $dry_run; then
    cta_configs="ag=<ring CTAs>"
else
    ctas_file="${output_dir}/ring_ag_ctas.node${node_rank}.json"
    uv run --no-sync python "$helper" ring-ctas --output-dir "$output_dir" \
        --group-sizes "$world_size" --sizes-mib "$sizes_mib" \
        --nnodes "$nnodes" --node-rank "$node_rank" --master-addr "$master_addr" \
        --port "$(( master_port + 20 ))" --timeout-min "$handshake_timeout_min" --out "$ctas_file" \
        || die "could not agree on ring's CTA counts"
    # One --cta-configs entry per distinct ring count, e.g. 'ag=16;ag=32'.
    cta_configs="$(uv run --no-sync python -c '
import json, sys
print(";".join(f"ag={n}" for n in sorted({ctas for _, _, ctas in json.load(open(sys.argv[1]))})))
' "$ctas_file")"
    [[ -n "$cta_configs" ]] || die "no ring CTA count in ${ctas_file}"
fi

echo "=== 2/2 symm-p2p, ${cta_configs} (NCCL_SYM_TMA_ENABLE=${sym_tma_enable})" >&2
# The same run directory: a set RUN_TIME skips the launcher's run-id handshake.
export RUN_TIME="$(basename -- "$output_dir")"
run_pass "$log" --algos symm-p2p \
    --master-port "$(( master_port + 10 ))" --cta-configs "$cta_configs"
$dry_run && exit 0

status=0
# Every NCCL process logs its banner once at NCCL_DEBUG=INFO; the launcher keeps those logs here.
banners="$(cat "${output_dir}"/{ring,symm-p2p}/nccl-raw/*.log 2>/dev/null | grep -o "NCCL version [^ ]*" | sort | uniq -c || true)"
echo "=== libnccl banners in ${output_dir}/{ring,symm-p2p}/nccl-raw" >&2
echo "${banners:-  none}" >&2
if [[ -z "$banners" ]] || grep -qv "NCCL version 2\.32\.3[-+]" <<<"$banners"; then
    echo "ERROR: not every NCCL process ran NCCL 2.32.3" >&2
    status=1
fi
if (( node_rank == 0 )); then
    uv run --no-sync python "$helper" summary --output-dir "$output_dir" \
        --group-sizes "$world_size" --sizes-mib "$sizes_mib" \
        --p2p-kernel "$p2p_kernel" || status=1
fi
exit "$status"
