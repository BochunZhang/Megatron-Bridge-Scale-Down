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
"""Watchdog process for :class:`mbench.gate.LaunchGate`.

Reads ``arm <generation> <timeout_s>`` / ``disarm <generation>`` lines from
stdin. If an armed generation is not disarmed in time, it writes the generation
to the fired slot and to the release slot of the shared flag page, which
unblocks the gated CUDA stream. Exits on stdin EOF. Deliberately imports
nothing CUDA-related so it starts fast and never touches the device.
"""

import mmap
import os
import select
import struct
import sys
import time


_PAGE_BYTES = 4096
_RELEASE_OFFSET = 0
_FIRED_OFFSET = 4


def main() -> None:
    """Serve arm/disarm commands until the parent closes stdin."""

    with open(sys.argv[1], "r+b") as stream:
        flag = mmap.mmap(stream.fileno(), _PAGE_BYTES)
    stdin_fd = sys.stdin.fileno()
    pending = b""
    armed: int | None = None
    deadline = 0.0
    while True:
        timeout = None if armed is None else max(0.0, deadline - time.monotonic())
        readable, _, _ = select.select([stdin_fd], [], [], timeout)
        if not readable:
            struct.pack_into("<I", flag, _FIRED_OFFSET, armed)
            struct.pack_into("<I", flag, _RELEASE_OFFSET, armed)
            armed = None
            continue
        chunk = os.read(stdin_fd, 4096)
        if not chunk:
            break
        pending += chunk
        *lines, pending = pending.split(b"\n")
        for line in lines:
            fields = line.decode().split()
            if not fields:
                continue
            generation = int(fields[1])
            if fields[0] == "arm":
                armed = generation
                deadline = time.monotonic() + float(fields[2])
            elif fields[0] == "disarm" and armed == generation:
                armed = None
    flag.close()


if __name__ == "__main__":
    main()
