#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Checks whether CPU threads on a GB200 node can load/store GPU HBM in place, for every allocation method,
# and records the driver/NUMA state that decides it. Output: results/gb200/intra-super-node/hbm_probe-<ts>/
#   env.txt      driver memory mode, NUMA nodes, kernel/driver parameters
#   probe.log    hbm_probe output (one verdict per method)
#   probe.json   same, machine readable
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
BIN="${HBM_PROBE_BIN:-$SCRIPT_DIR/build/hbm_probe}"
OUT_ROOT="$REPO_ROOT/results/gb200/intra-super-node"
RUN_DIR=""
GPUS=""
PROBE_ARGS=()

usage() {
  echo "usage: $0 [--gpus 0,2] [--out-root DIR | --out-dir DIR] [-- hbm_probe options, e.g. --bytes 1G --methods numa,vmm]"
  echo "  --out-dir writes into DIR itself instead of a new OUT_ROOT/hbm_probe-<ts> (used by run_gb200.sh)"
}
while (($#)); do
  case "$1" in
    --gpus) GPUS=$2; shift 2 ;;
    --out-root) OUT_ROOT=$2; shift 2 ;;
    --out-dir) RUN_DIR=$2; shift 2 ;;
    --) shift; PROBE_ARGS=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
if [[ ! -x "$BIN" ]]; then
  echo "building $BIN" >&2
  make -C "$SCRIPT_DIR" probe >&2 || { echo "build failed; run 'make probe' in $SCRIPT_DIR" >&2; exit 1; }
fi
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && echo "[warn] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2

[[ -n "$RUN_DIR" ]] || RUN_DIR="$OUT_ROOT/hbm_probe-$(date +%y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"

section() { printf '\n##### %s\n' "$*"; }
{
  echo "# $(date -Iseconds) $(hostname) $(uname -r)"
  section "nvidia-smi"
  nvidia-smi
  section "nvidia-smi -q: addressing mode / NUMA / C2C"
  nvidia-smi -q | grep -iE "Addressing Mode|NUMA|C2C|Driver Version|CUDA Version|Product Name|Bus Id|Persistence" || true
  section "nvidia-smi topo -m"
  nvidia-smi topo -m
  section "nvidia-smi c2c -s"
  nvidia-smi c2c -s 2>&1 || true
  section "numactl -H"
  numactl -H 2>&1 || echo "numactl not installed"
  section "/sys/devices/system/node"
  for f in online possible has_cpu has_memory has_normal_memory; do
    printf '%-18s %s\n' "$f" "$(cat /sys/devices/system/node/$f 2>/dev/null)"
  done
  for n in /sys/devices/system/node/node*; do
    printf '%s cpus=%s %s\n' "$(basename "$n")" "$(cat "$n/cpulist")" "$(grep -m1 MemTotal "$n/meminfo" | awk '{print $4 $5}')"
  done
  section "nvidia driver parameters (NVreg_*)"
  grep -H . /proc/driver/nvidia/params 2>/dev/null | grep -iE "numa|cdmm|coherent|c2c" || true
  cat /proc/driver/nvidia/params 2>/dev/null
  grep -rhE "options nvidia" /etc/modprobe.d/ 2>/dev/null
  section "kernel cmdline"
  cat /proc/cmdline
  section "process limits"
  echo "Cpus_allowed_list: $(grep Cpus_allowed_list /proc/self/status | cut -f2)"
  echo "Mems_allowed_list: $(grep Mems_allowed_list /proc/self/status | cut -f2)"
  grep -E "^CapEff" /proc/self/status
  grep -E "^Seccomp:" /proc/self/status
} >"$RUN_DIR/env.txt" 2>&1

echo "[env] nvidia-smi: $(grep -v '^#' "$RUN_DIR/env.txt" | grep -iE 'Addressing Mode' | sort -u | xargs)"
echo "[env] NUMA nodes with memory: $(cat /sys/devices/system/node/has_memory 2>/dev/null), with CPUs: $(cat /sys/devices/system/node/has_cpu 2>/dev/null)"

rc=0
if [[ -z "$GPUS" ]]; then
  "$BIN" ${PROBE_ARGS[@]+"${PROBE_ARGS[@]}"} --out "$RUN_DIR/probe.json" 2>&1 | tee "$RUN_DIR/probe.log"
  rc=${PIPESTATUS[0]}
else
  for g in ${GPUS//,/ }; do
    "$BIN" --gpu "$g" ${PROBE_ARGS[@]+"${PROBE_ARGS[@]}"} --out "$RUN_DIR/probe-gpu$g.json" 2>&1 | tee "$RUN_DIR/probe-gpu$g.log"
    ((PIPESTATUS[0] != 0)) && rc=1
  done
fi

echo
echo "verdicts:"
grep -h -E "^=== |verdict:|unavailable:" "$RUN_DIR"/probe*.log
echo "results: $RUN_DIR"
exit "$rc"
