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
"""Native NCCL copy-engine AllGather for GPU and pinned CPU sources."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import torch
import torch.distributed as dist


_NCCL_UINT8 = 1
_MIN_NCCL_VERSION = (2, 32, 2)


def _loaded_nccl() -> ctypes.CDLL:
    # An LD_PRELOAD and a bundled Torch library can coexist. A communicator
    # pointer may only be used with the library that created it.
    libraries = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and os.path.basename(fields[5]).startswith("libnccl.so"):
            libraries.add(os.path.realpath(fields[5]))
    if len(libraries) != 1:
        raise RuntimeError(f"Expected exactly one loaded libnccl, found {sorted(libraries)}")
    library = libraries.pop()
    nccl = ctypes.CDLL(library, mode=os.RTLD_NOW | os.RTLD_NOLOAD)
    nccl.ncclGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    nccl.ncclGetErrorString.argtypes = [ctypes.c_int]
    nccl.ncclGetErrorString.restype = ctypes.c_char_p
    nccl.ncclAllGather.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    return nccl


def _check_nccl(nccl: ctypes.CDLL, result: int, operation: str) -> None:
    if result != 0:
        message = nccl.ncclGetErrorString(result)
        detail = message.decode() if message else "unknown error"
        raise RuntimeError(f"{operation} failed with NCCL result {result}: {detail}")


def _nccl_version(nccl: ctypes.CDLL) -> str:
    number = ctypes.c_int()
    _check_nccl(nccl, nccl.ncclGetVersion(ctypes.byref(number)), "ncclGetVersion")
    if number.value < 10000:
        major, remainder = divmod(number.value, 1000)
    else:
        major, remainder = divmod(number.value, 10000)
    minor, patch = divmod(remainder, 100)
    version = f"{major}.{minor}.{patch}"
    if (major, minor, patch) < _MIN_NCCL_VERSION:
        raise RuntimeError(f"Native CE multicast needs NCCL 2.32.2 or newer, found {version}")
    return version


def _mapped_host_pointer(tensor: torch.Tensor) -> int:
    if not tensor.is_pinned():
        raise RuntimeError("Native NCCL host input must be CUDA-pinned")
    driver = ctypes.CDLL("libcuda.so.1")
    address = ctypes.c_uint64()
    get_device_pointer = driver.cuMemHostGetDevicePointer_v2
    get_device_pointer.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_void_p, ctypes.c_uint]
    result = get_device_pointer(ctypes.byref(address), tensor.data_ptr(), 0)
    if result != 0:
        message = ctypes.c_char_p()
        driver.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        driver.cuGetErrorString(result, ctypes.byref(message))
        detail = message.value.decode() if message.value else "unknown error"
        raise RuntimeError(f"cuMemHostGetDevicePointer failed with CUresult {result}: {detail}")
    return address.value


class NcclCopyEngineAllGather:
    """Launch native AllGather on the current CUDA stream with symmetric receive storage.

    All ranks construct and close the NCCL subgroup in the same order. The
    launcher selects the NCCL CE path and the profiler validates which path
    ran. The caller retains its Gloo group for control collectives.

    Args:
        device: This rank's CUDA device.
        rank: This rank's index in the distributed group.
        world_size: Number of ranks participating in multicast.
        size_bytes: Total bytes of the gathered output on each GPU.
        host_input: Read directly from a pinned CPU tensor instead of GPU memory.
    """

    def __init__(self, device: torch.device, rank: int, world_size: int, size_bytes: int, host_input: bool) -> None:
        if not 3 <= world_size <= 255 or not 0 <= rank < world_size:
            raise ValueError(f"CE multicast requires 3–255 ranks, got rank {rank} of {world_size}")
        if size_bytes <= 0 or size_bytes % world_size:
            raise ValueError(f"Cannot split {size_bytes} bytes evenly across {world_size} ranks")

        self._device = device
        self._rank = rank
        self._shard_bytes = size_bytes // world_size

        options = dist.ProcessGroupNCCL.Options()
        # Raw calls have no ProcessGroupNCCL Work object to poll ncclInProgress.
        options.config.blocking = 1
        self._group = dist.new_group(backend="nccl", device_id=device, pg_options=options)
        self._registered = False
        try:
            self._backend = self._group._get_backend(device)
            self._comm = self._backend._comm_ptr()
            if not self._comm:
                raise RuntimeError("PyTorch did not initialize the NCCL communicator for this CUDA device")
            self._nccl = _loaded_nccl()
            self.version = _nccl_version(self._nccl)

            self._pool = torch.cuda.MemPool(self._backend.mem_allocator)
            with torch.cuda.use_mem_pool(self._pool):
                self._output = torch.empty(size_bytes, dtype=torch.uint8, device=device)
                if not host_input:
                    self._input = torch.full((self._shard_bytes,), rank + 1, dtype=torch.uint8, device=device)
            if host_input:
                # Leave the host buffer outside NCCL's window registration.
                self._input = torch.empty(self._shard_bytes, dtype=torch.uint8, device="cpu", pin_memory=True)
                self._input.fill_(rank + 1)
                self._source = _mapped_host_pointer(self._input)
            else:
                self._source = self._input.data_ptr()
            self._destination = self._output.data_ptr()

            torch.cuda.synchronize(device)
            self._backend.register_mem_pool(self._pool, symm=True)
            self._registered = True
        except Exception:
            self.close()
            raise

    def run(self) -> None:
        """Enqueue a native NCCL collective on the current CUDA stream."""
        if self._group is None:
            raise RuntimeError("Cannot launch AllGather on a closed communicator")
        stream = torch.cuda.current_stream(self._device).cuda_stream
        if stream == 0:
            raise RuntimeError("CE multicast requires a non-default CUDA stream")
        _check_nccl(
            self._nccl,
            self._nccl.ncclAllGather(
                self._source, self._destination, self._shard_bytes, _NCCL_UINT8, self._comm, stream
            ),
            "ncclAllGather",
        )

    def reset_output(self) -> None:
        """Poison this rank's local output before measuring a fresh collective."""
        self._output.zero_()

    def validate(self) -> None:
        """Check the full local receive buffer after communication completes."""
        for peer, shard in enumerate(self._output.split(self._shard_bytes)):
            if not torch.all(shard.eq(peer + 1)).item():
                raise RuntimeError(f"CE multicast rank {self._rank} received incorrect bytes for rank {peer}")

    def close(self) -> None:
        """Drain the current device and release the registered NCCL group."""
        if self._group is None:
            return
        group = self._group
        self._group = None
        try:
            torch.cuda.synchronize(self._device)
            if self._registered:
                self._backend.deregister_mem_pool(self._pool)
        finally:
            dist.destroy_process_group(group)
