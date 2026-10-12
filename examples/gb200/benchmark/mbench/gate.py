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
"""Host-released CUDA stream gate built on ``cuStreamWaitValue32``.

``LaunchGate.arm`` enqueues a stream-wait on a host-mapped 32-bit flag. The GPU
stalls at that point while the host enqueues all timed work; ``release`` writes
the flag and the GPU then executes the queued work back to back, so no host
launch latency can appear between timed CUDA events.

If the host blocks while the stream is gated (a full launch queue, or an
implicit host synchronization inside a library), the process would deadlock.
A watchdog *process* (not a thread, so it also works when the blocked call
holds the GIL) releases the flag after a timeout and marks the round as fired;
callers must discard fired rounds.
"""

from __future__ import annotations

import ctypes
import logging
import mmap
import os
import struct
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import torch


LOGGER = logging.getLogger(__name__)

_PAGE_BYTES = 4096
_RELEASE_OFFSET = 0
_FIRED_OFFSET = 4
_CU_MEMHOSTREGISTER_PORTABLE = 0x01
_CU_MEMHOSTREGISTER_DEVICEMAP = 0x02
_CU_STREAM_WAIT_VALUE_GEQ = 0x0
_WATCHDOG_SCRIPT = Path(__file__).with_name("gate_watchdog.py")


def _load_driver() -> ctypes.CDLL:
    driver = ctypes.CDLL("libcuda.so.1")
    driver.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    return driver


def _driver_symbol(driver: ctypes.CDLL, *names: str) -> Callable[..., int]:
    """Return the first exported symbol; CUDA 12+ exposes the ``_v2`` ABI names."""

    for name in names:
        try:
            return getattr(driver, name)
        except AttributeError:
            continue
    raise RuntimeError(f"libcuda does not export any of {names}")


class LaunchGate:
    """Stream gate that holds GPU execution until the host releases it.

    Args:
        timeout_s: Watchdog timeout for one armed round. It must exceed the
            host time needed to enqueue one round.
    """

    def __init__(self, *, timeout_s: float) -> None:
        self._timeout_s = timeout_s
        self._generation = 0
        self._driver = _load_driver()
        self._wait_value32 = _driver_symbol(self._driver, "cuStreamWaitValue32_v2", "cuStreamWaitValue32")
        self._wait_value32.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint]
        host_register = _driver_symbol(self._driver, "cuMemHostRegister_v2", "cuMemHostRegister")
        host_register.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
        get_device_pointer = _driver_symbol(self._driver, "cuMemHostGetDevicePointer_v2", "cuMemHostGetDevicePointer")
        get_device_pointer.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_void_p, ctypes.c_uint]
        self._host_unregister = _driver_symbol(self._driver, "cuMemHostUnregister")
        self._host_unregister.argtypes = [ctypes.c_void_p]

        # A shared file mapping lets the watchdog process write the same flag.
        shm_dir = "/dev/shm" if os.path.isdir("/dev/shm") else None
        handle, self._path = tempfile.mkstemp(prefix="mbench-gate-", dir=shm_dir)
        os.write(handle, b"\0" * _PAGE_BYTES)
        self._mmap = mmap.mmap(handle, _PAGE_BYTES)
        os.close(handle)
        self._anchor = ctypes.c_char.from_buffer(self._mmap)
        self._host_ptr = ctypes.addressof(self._anchor)

        # Registration needs a current CUDA context; touching the device creates it.
        torch.empty(1, device="cuda")
        self._check(
            host_register(self._host_ptr, _PAGE_BYTES, _CU_MEMHOSTREGISTER_PORTABLE | _CU_MEMHOSTREGISTER_DEVICEMAP),
            "cuMemHostRegister",
        )
        device_ptr = ctypes.c_uint64()
        self._check(get_device_pointer(ctypes.byref(device_ptr), self._host_ptr, 0), "cuMemHostGetDevicePointer")
        self._device_ptr = device_ptr.value
        self._watchdog = subprocess.Popen(
            [sys.executable, str(_WATCHDOG_SCRIPT), self._path],
            stdin=subprocess.PIPE,
            text=True,
        )
        LOGGER.info("launch gate ready: flag=%s timeout=%.1fs", self._path, timeout_s)

    def _check(self, result: int, what: str) -> None:
        if result == 0:
            return
        message = ctypes.c_char_p()
        self._driver.cuGetErrorString(result, ctypes.byref(message))
        detail = message.value.decode() if message.value else "unknown error"
        raise RuntimeError(f"{what} failed with CUresult {result}: {detail}")

    def _send(self, line: str) -> None:
        assert self._watchdog.stdin is not None
        self._watchdog.stdin.write(line + "\n")
        self._watchdog.stdin.flush()

    def arm(self, stream: torch.cuda.Stream) -> None:
        """Block ``stream`` on the GPU until the next :meth:`release`."""

        self._generation += 1
        self._send(f"arm {self._generation} {self._timeout_s}")
        self._check(
            self._wait_value32(
                ctypes.c_void_p(stream.cuda_stream), self._device_ptr, self._generation, _CU_STREAM_WAIT_VALUE_GEQ
            ),
            "cuStreamWaitValue32",
        )

    def release(self) -> None:
        """Let the GPU run the queued work."""

        self._send(f"disarm {self._generation}")
        struct.pack_into("<I", self._mmap, _RELEASE_OFFSET, self._generation)

    def watchdog_fired(self) -> bool:
        """Whether the watchdog released the current round.

        Call after synchronizing. A fired round means the host blocked while the
        stream was gated, so its timings may contain host-induced bubbles.
        """

        (fired,) = struct.unpack_from("<I", self._mmap, _FIRED_OFFSET)
        return fired == self._generation

    def close(self) -> None:
        """Stop the watchdog and unregister the host flag."""

        torch.cuda.synchronize()
        if self._watchdog.stdin is not None:
            self._watchdog.stdin.close()
        self._watchdog.wait(timeout=10)
        self._check(self._host_unregister(self._host_ptr), "cuMemHostUnregister")
        del self._anchor
        self._mmap.close()
        os.unlink(self._path)
