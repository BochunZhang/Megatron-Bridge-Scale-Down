#!/usr/bin/env python3
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

"""Capture per-warp NCCL clock64 sum/max groups and export them as CSV."""

import csv
import ctypes
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import torch


_FIELDS = (
    "profile",
    "pause_cycles",
    "kernel",
    "rank",
    "launch",
    "cta",
    "warp",
    "tile_first",
    "tile_last",
    "n",
    "store_window_sum",
    "store_window_max",
    "ld_issue_sum",
    "ld_issue_max",
    "ld_wait_sum",
    "ld_wait_max",
    "st_issue_sum",
    "st_issue_max",
)
_KERNELS = ("AllGather_STMC", "ReduceScatter_LDMC")
_ROW_WORDS = 11  # ncclSymClock64Row in the NCCL patch
_MAX_WARPS = 64 * (512 // 32)  # ncclSymkMaxBlocks * ncclSymkMaxThreads / WARP_SIZE in 2.32.3


def _unsigned(value: int) -> int:
    return value % (1 << 64)


class Clock64Capture:
    """Own the zeroed GPU table for the measured launches on one rank."""

    def __init__(self, *, profile: str, device: torch.device, shard_bytes: int, launches: int) -> None:
        if profile not in ("store", "split"):
            raise ValueError(f"Unknown clock64 profile {profile!r}")
        self.profile = profile
        self.output = Path(os.environ["NCCL_SYM_CLOCK64_CSV_DIR"])
        self.pause = int(os.environ["NCCL_SYM_AG_STMC_PAUSE_CYCLES"])
        self.stride = int(os.environ["NCCL_SYM_CLOCK64_STRIDE"])
        self.group_size = int(os.environ["NCCL_SYM_CLOCK64_GROUP"])
        self.launches = launches
        path = Path(os.environ["NCCL_SYM_CLOCK64_CSV_LIB"]).resolve(strict=True)
        # Only bind the already-preloaded library. Loading a second copy would
        # register the buffer with a different scheduler from PyTorch's.
        library = ctypes.CDLL(str(path), mode=os.RTLD_NOLOAD)
        self.register = library.ncclSymClock64Register
        self.register.argtypes = (ctypes.c_void_p,) + (ctypes.c_uint32,) * 4
        self.register.restype = ctypes.c_int
        if not all(0 < x < 1 << 32 for x in (self.stride, self.group_size, launches)):
            raise ValueError("Clock64 stride, group size, and launch count must be positive uint32 values")

        tiles = shard_bytes // 4096
        # Group-major device rows may have holes in the last group. At most
        # ceil(tiles / (stride * group_size)) + nWarps slots are needed.
        tiles_per_group = self.stride * self.group_size
        self.capacity = (tiles + tiles_per_group - 1) // tiles_per_group + _MAX_WARPS
        if not 0 < self.capacity < 1 << 32:
            raise ValueError("Clock64 row capacity does not fit uint32")
        self.buffer = torch.zeros(
            (len(_KERNELS), launches, self.capacity, _ROW_WORDS), dtype=torch.int64, device=device
        )

    @contextmanager
    def arm(self) -> Iterator[None]:
        """Register the table for the timed launches after warmup."""
        status = self.register(self.buffer.data_ptr(), self.capacity, self.launches, self.stride, self.group_size)
        if status:
            raise RuntimeError(f"ncclSymClock64Register failed with NCCL error {status}")
        try:
            yield
        finally:
            self.register(None, 0, 0, 0, 0)

    def write(self, *, rank: int) -> None:
        """Write the completed per-warp groups for both collectives."""
        batches = self.buffer.cpu().tolist()
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / f"rank-{rank}.csv").open("x", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(_FIELDS)
            for kernel, launches in zip(_KERNELS, batches, strict=True):
                for launch, rows in enumerate(launches):
                    valid = sorted(row for row in rows if row[4])
                    if not valid:
                        raise RuntimeError(f"Missing clock64 data for {kernel}, rank {rank}, timed launch {launch}")
                    for cta, warp, first, last, n, *metrics in valid:
                        # A reserved split-probe digest can leave a gap in the group.
                        span = last - first
                        if (
                            first < 1
                            or not 1 <= n <= self.group_size
                            or span < self.stride * (n - 1)
                            or span % self.stride
                        ):
                            raise RuntimeError(f"Invalid clock64 group for {kernel}, CTA {cta}, warp {warp}")
                        sums = [_unsigned(value) for value in metrics[:3]]
                        maxima = [_unsigned(value) for value in metrics[3:]]
                        if self.profile == "store":
                            fields = [sums[0], maxima[0], "", "", "", "", "", ""]
                        else:
                            fields = ["", "", *(x for pair in zip(sums, maxima, strict=True) for x in pair)]
                        writer.writerow(
                            [self.profile, self.pause, kernel, rank, launch, cta, warp, first, last, n, *fields]
                        )
