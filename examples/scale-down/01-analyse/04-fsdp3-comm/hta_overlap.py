# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Measure HTA communication/computation overlap with HybridEP and optional TE Userbuffers recognition."""

import argparse
import re
import sys
from dataclasses import replace

import pandas as pd
from hta.trace_analysis import TraceAnalysis
from hta.utils import utils as hta_utils


_HTA_COMMUNICATION = hta_utils.CommunicateKernelGroupingPattern


def configure_hybridep_communication(*, include_sync: bool = False, include_tp_userbuffers: bool = False) -> None:
    """Extend HTA's process-wide communication matcher before starting analysis.

    Args:
        include_sync: Count HybridEP's inter-rank device barrier as communication.
        include_tp_userbuffers: Count TE Userbuffers transfers, send/receive signals and explicit peer copies.
    """
    ep_kernels = "dispatch_kernel|combine_kernel"
    if include_sync:
        ep_kernels += "|device_sync_kernel"
    # Match the function itself, including demangled names with a return type.
    # ag_nvl_kernel is HybridEP's routing-map gather in the global namespace.
    hybrid_ep = rf"^(?:void\s+)?(?:hybrid_ep::(?:{ep_kernels})|ag_nvl_kernel)(?=\s*(?:[<(]|$))"
    native = _HTA_COMMUNICATION.pattern
    patterns = [native.pattern, hybrid_ep]
    if include_tp_userbuffers:
        # TP4 traces contain both multicast data kernels and single-thread peer signals.
        ub_kernels = r"kuserbuffers_push(?:send|recv)|userbuffers_fp16_sum_inplace_gpu_mc_(?:ag|rs(?:_oop)?)"
        patterns.extend([rf"^(?:void\s+)?(?:{ub_kernels})(?=\s*(?:[<(]|$))", r"^Memcpy PtoP\b"])
    hta_utils.CommunicateKernelGroupingPattern = replace(
        _HTA_COMMUNICATION, pattern=re.compile("|".join(patterns), native.flags)
    )


def _union_ms(events: pd.DataFrame) -> float:
    intervals = hta_utils.merge_kernel_intervals(events[["ts", "dur"]].copy())
    return (intervals["end"] - intervals["ts"]).sum() / 1000


def main() -> None:
    """Write one CSV row per rank for the supplied torch_profile directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", help="Directory containing per-rank PyTorch profiler traces")
    parser.add_argument("--include-ep-sync", action="store_true", help="Include HybridEP's device-side barrier")
    parser.add_argument(
        "--include-tp-userbuffers",
        action="store_true",
        help="Include TE Userbuffers data and signals, and PtoP copies",
    )
    args = parser.parse_args()

    configure_hybridep_communication(
        include_sync=args.include_ep_sync, include_tp_userbuffers=args.include_tp_userbuffers
    )
    analysis = TraceAnalysis(trace_dir=args.trace_dir, include_last_profiler_step=True)
    overlap = analysis.get_comm_comp_overlap(visualize=False).set_index("rank")
    types = [hta_utils.get_kernel_type(name) for name in analysis.t.symbol_table.get_sym_table()]
    for rank, trace in analysis.t.traces.items():
        gpu = trace.loc[trace["stream"].ne(-1)]
        kind = gpu["name"].map(types.__getitem__)
        comm = kind.eq(hta_utils.KernelType.COMMUNICATION.name)
        compute = kind.eq(hta_utils.KernelType.COMPUTATION.name)
        overlap.loc[rank, "communication_ms"] = _union_ms(gpu[comm])
        # |communication \ computation| = |communication U computation| - |computation|.
        overlap.loc[rank, "exposed_communication_ms"] = _union_ms(gpu[comm | compute]) - _union_ms(gpu[compute])
    overlap.sort_index().to_csv(sys.stdout, float_format="%.3f")


if __name__ == "__main__":
    main()
