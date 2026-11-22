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
"""Copy GPU or pinned host shards into a symmetric CUDA multicast buffer."""

from __future__ import annotations

import ctypes
from collections.abc import Callable

import torch
import torch.distributed as dist


_CUDA_13_0 = 13000
_CUDA_13_1 = 13010
_CU_MEMCPY_SRC_ACCESS_ORDER_STREAM = 1
_CU_MEMCPY_FLAG_PREFER_OVERLAP_WITH_COMPUTE = 1
_DevicePointer = ctypes.c_uint64


class _MemoryLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _CopyAttributes(ctypes.Structure):
    _fields_ = [
        ("src_access_order", ctypes.c_int),
        ("src_location_hint", _MemoryLocation),
        ("dst_location_hint", _MemoryLocation),
        ("flags", ctypes.c_uint),
    ]


def _check_cuda(driver: ctypes.CDLL, result: int, operation: str) -> None:
    if result != 0:
        message = ctypes.c_char_p()
        driver.cuGetErrorString(result, ctypes.byref(message))
        detail = message.value.decode() if message.value else "unknown error"
        raise RuntimeError(f"{operation} failed with CUresult {result}: {detail}")


def _load_memcpy_batch(driver: ctypes.CDLL) -> tuple[Callable[..., int], int]:
    driver.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    driver.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    version = ctypes.c_int()
    _check_cuda(driver, driver.cuDriverGetVersion(ctypes.byref(version)), "cuDriverGetVersion")
    if version.value < _CUDA_13_0:
        raise RuntimeError(f"CE multicast uses the CUDA 13.0 driver ABI, found {version.value}")

    # CUDA 12.x includes a failIdx parameter. Request the CUDA 13 ABI explicitly.
    get_proc_address = driver.cuGetProcAddress_v2
    get_proc_address.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_int),
    ]
    pointer = ctypes.c_void_p()
    status = ctypes.c_int()
    _check_cuda(
        driver,
        get_proc_address(b"cuMemcpyBatchAsync", ctypes.byref(pointer), _CUDA_13_0, 0, ctypes.byref(status)),
        "cuGetProcAddress(cuMemcpyBatchAsync)",
    )
    if pointer.value is None:
        raise RuntimeError(f"CUDA driver did not provide cuMemcpyBatchAsync for CUDA 13.0, status {status.value}")
    prototype = ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.POINTER(_DevicePointer),
        ctypes.POINTER(_DevicePointer),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.POINTER(_CopyAttributes),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.c_void_p,
    )
    # Driver 13.1 added support for this hint on non-Tegra platforms.
    flags = _CU_MEMCPY_FLAG_PREFER_OVERLAP_WITH_COMPUTE if version.value >= _CUDA_13_1 else 0
    return prototype(pointer.value), flags


class MulticastAllGather:
    """Enqueue one copy per rank into disjoint slots of an NVLink multicast mapping.

    The caller initializes the distributed group and synchronizes ranks before
    writing or validating the receive buffer. The benchmark reuses an immutable
    source and identical destination slots each round.

    Args:
        device: This rank's CUDA device.
        rank: This rank's index in the distributed group.
        world_size: Number of ranks participating in multicast.
        size_bytes: Total bytes of the gathered output on each GPU.
        host_input: Read directly from a pinned CPU tensor instead of GPU memory.
    """

    def __init__(self, device: torch.device, rank: int, world_size: int, size_bytes: int, host_input: bool) -> None:
        if not 2 <= world_size <= 255 or not 0 <= rank < world_size:
            raise ValueError(f"CE multicast requires 2–255 ranks, got rank {rank} of {world_size}")
        if size_bytes <= 0 or size_bytes % world_size:
            raise ValueError(f"Cannot split {size_bytes} bytes evenly across {world_size} ranks")

        from torch.distributed import _symmetric_memory as symm_mem

        self._device = device
        self._rank = rank
        self._shard_bytes = size_bytes // world_size
        self._source_kind = "pinned host" if host_input else "HBM"
        self._driver = ctypes.CDLL("libcuda.so.1")
        self._memcpy_batch, copy_flags = _load_memcpy_batch(self._driver)

        symm_mem.set_backend("CUDA")
        self._output = symm_mem.empty(size_bytes, dtype=torch.uint8, device=device)
        self._symmetric_output = symm_mem.rendezvous(self._output, dist.group.WORLD)
        if self._symmetric_output is None:
            raise RuntimeError("PyTorch did not establish a CUDA symmetric memory mapping")
        multicast_ptr = self._symmetric_output.multicast_ptr
        if not multicast_ptr:
            raise RuntimeError("The CUDA symmetric memory group has no NVLink multicast mapping")

        source_device = torch.device("cpu") if host_input else device
        self._input = torch.empty(self._shard_bytes, dtype=torch.uint8, device=source_device, pin_memory=host_input)
        self._input.fill_(rank + 1)

        # multicast_ptr already includes the symmetric tensor's storage offset.
        self._dst = _DevicePointer(multicast_ptr + rank * self._shard_bytes)
        self._src = _DevicePointer(self._input.data_ptr())
        if host_input:
            if not self._input.is_pinned():
                raise RuntimeError("CE multicast host input must be CUDA-pinned")
            # cudaHostAlloc uses the same UVA address. A cudaHostRegister-based
            # PyTorch pinned allocator can require a distinct device alias.
            get_device_pointer = self._driver.cuMemHostGetDevicePointer_v2
            get_device_pointer.argtypes = [ctypes.POINTER(_DevicePointer), ctypes.c_void_p, ctypes.c_uint]
            _check_cuda(
                self._driver,
                get_device_pointer(ctypes.byref(self._src), self._input.data_ptr(), 0),
                "cuMemHostGetDevicePointer",
            )
        self._size = ctypes.c_size_t(self._shard_bytes)
        self._attributes_index = ctypes.c_size_t(0)
        self._attributes = _CopyAttributes(
            src_access_order=_CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
            flags=copy_flags,
        )

    def run(self) -> None:
        """Enqueue one multicast copy of this rank's shard on the current CUDA stream."""
        stream = torch.cuda.current_stream(self._device).cuda_stream
        if stream == 0:
            raise RuntimeError("CE multicast requires a non-default CUDA stream")
        _check_cuda(
            self._driver,
            self._memcpy_batch(
                ctypes.byref(self._dst),
                ctypes.byref(self._src),
                ctypes.byref(self._size),
                1,
                ctypes.byref(self._attributes),
                ctypes.byref(self._attributes_index),
                1,
                stream,
            ),
            f"cuMemcpyBatchAsync({self._source_kind} to multicast HBM)",
        )

    def reset_output(self) -> None:
        """Poison this rank's local output before measuring a fresh transfer."""
        self._output.zero_()

    def validate(self) -> None:
        """Check the full local receive buffer after all rank streams complete."""
        for peer, shard in enumerate(self._output.split(self._shard_bytes)):
            if not torch.all(shard.eq(peer + 1)).item():
                raise RuntimeError(f"CE multicast rank {self._rank} received incorrect bytes for rank {peer}")
