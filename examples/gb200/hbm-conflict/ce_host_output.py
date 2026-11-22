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

"""Gather GPU shards directly into per-rank pinned host buffers."""

from __future__ import annotations

import ctypes
import mmap
import os
import socket
import uuid
from contextlib import ExitStack

import torch
import torch.distributed as dist


_CU_MEMHOSTREGISTER_PORTABLE = 1
_SOCKET_TIMEOUT_SECONDS = 30
_VALIDATION_CHUNK_BYTES = 16 << 20


def _map_host_outputs(*, rank: int, world_size: int, size_bytes: int) -> mmap.mmap:
    total_bytes = world_size * size_bytes
    with ExitStack() as cleanup:
        connection = cleanup.enter_context(socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET))
        connection.settimeout(_SOCKET_TIMEOUT_SECONDS)
        address = [b""]
        if rank == 0:
            address[0] = b"\0ce-hbm-" + uuid.uuid4().hex.encode("ascii")
            connection.bind(address[0])
            connection.listen(world_size - 1)
            fd = os.memfd_create("ce-hbm-host-output", os.MFD_CLOEXEC)
            cleanup.callback(os.close, fd)
            os.ftruncate(fd, total_bytes)
        dist.broadcast_object_list(address, src=0)
        if rank == 0:
            for _ in range(world_size - 1):
                peer, _ = connection.accept()
                with peer:
                    peer.settimeout(_SOCKET_TIMEOUT_SECONDS)
                    socket.send_fds(peer, [b"\0"], [fd])
        else:
            connection.connect(address[0])
            _, fds, flags, _ = socket.recv_fds(connection, 1, 1)
            for received_fd in fds:
                cleanup.callback(os.close, received_fd)
            if len(fds) != 1 or flags & socket.MSG_CTRUNC:
                raise RuntimeError("Expected one host-output memfd from rank 0")
            fd = fds[0]
            os.set_inheritable(fd, False)
        if os.fstat(fd).st_size != total_bytes:
            raise RuntimeError("Shared host-output memfd has an unexpected size")
        # Back each output with pages allocated under the receiving rank's host
        # memory policy before another rank registers the full mapping.
        os.posix_fallocate(fd, rank * size_bytes, size_bytes)
        return mmap.mmap(fd, total_bytes, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)


class HostOutputCopyEngineAllGather:
    """Enqueue one direct D2H copy per destination from the rank-local GPU shard.

    A single-node Gloo group shares the host mapping and controls the
    measurement. The caller drains all ranks before validating the outputs.

    Args:
        device: This rank's CUDA device.
        rank: This rank's index in the Gloo group.
        world_size: Number of ranks on the node.
        size_bytes: Total gathered output bytes per host receive buffer.
    """

    def __init__(self, device: torch.device, *, rank: int, world_size: int, size_bytes: int) -> None:
        if not 2 <= world_size <= 255 or not 0 <= rank < world_size:
            raise ValueError(f"Host-output AllGather requires 2–255 ranks, got rank {rank} of {world_size}")
        if size_bytes <= 0 or size_bytes % world_size or size_bytes % mmap.PAGESIZE:
            raise ValueError("Host output size must be positive and divisible by both rank count and host page size")
        if not hasattr(os, "memfd_create"):
            raise RuntimeError("Host-output AllGather requires Linux memfd support")

        self._device = device
        self._rank = rank
        self._size_bytes = size_bytes
        self._shard_bytes = size_bytes // world_size
        self._mapping: mmap.mmap | None = None
        self._registered = False

        self._driver = ctypes.CDLL("libcuda.so.1")
        self._driver.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        self._driver.cuMemHostRegister_v2.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
        self._driver.cuMemHostUnregister.argtypes = [ctypes.c_void_p]
        self._driver.cuMemcpyDtoHAsync_v2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]
        self._input = torch.full((self._shard_bytes,), rank + 1, dtype=torch.uint8, device=device)
        self._mapping = _map_host_outputs(rank=rank, world_size=world_size, size_bytes=size_bytes)
        try:
            self._base = ctypes.addressof(ctypes.c_char.from_buffer(self._mapping))
            self._local_output = self._base + rank * size_bytes
            # Begin with the rank-local output, then rotate across peers so that
            # the senders do not all target the same receiver first.
            self._write_addresses = [
                self._base + ((rank + peer) % world_size) * size_bytes + rank * self._shard_bytes
                for peer in range(world_size)
            ]
            self.reset_output()
            dist.barrier()
            with torch.cuda.device(device):
                self._check(
                    self._driver.cuMemHostRegister_v2(self._base, len(self._mapping), _CU_MEMHOSTREGISTER_PORTABLE),
                    "cuMemHostRegister",
                )
            self._registered = True
        except BaseException:
            self.close()
            raise

    def _check(self, result: int, operation: str) -> None:
        if result != 0:
            message = ctypes.c_char_p()
            self._driver.cuGetErrorString(result, ctypes.byref(message))
            detail = message.value.decode() if message.value else "unknown error"
            raise RuntimeError(f"{operation} failed with CUDA result {result}: {detail}")

    def run(self) -> None:
        """Enqueue direct writes to every receiver on the current CUDA stream."""
        if not self._registered:
            raise RuntimeError("Cannot launch AllGather on a closed host mapping")
        stream = torch.cuda.current_stream(self._device).cuda_stream
        if stream == 0:
            raise RuntimeError("Host-output AllGather requires a non-default CUDA stream")
        source = self._input.data_ptr()
        with torch.cuda.device(self._device):
            for destination in self._write_addresses:
                self._check(
                    self._driver.cuMemcpyDtoHAsync_v2(destination, source, self._shard_bytes, stream),
                    "cuMemcpyDtoHAsync",
                )

    def reset_output(self) -> None:
        """Zero this rank's output after all previous writers have finished."""
        if self._mapping is None:
            raise RuntimeError("Cannot reset a closed host mapping")
        ctypes.memset(self._local_output, 0, self._size_bytes)

    def validate(self) -> None:
        """Check every byte received by this rank after all writers have finished."""
        if self._mapping is None:
            raise RuntimeError("Cannot validate a closed host mapping")
        for peer in range(len(self._write_addresses)):
            shard = torch.frombuffer(
                self._mapping,
                dtype=torch.uint8,
                count=self._shard_bytes,
                offset=self._rank * self._size_bytes + peer * self._shard_bytes,
            )
            for chunk in shard.split(_VALIDATION_CHUNK_BYTES):
                if not chunk.eq(peer + 1).all().item():
                    raise RuntimeError(
                        f"Host-output AllGather rank {self._rank} received incorrect bytes for rank {peer}"
                    )

    def close(self) -> None:
        """Drain local GPU writes, unregister the buffer, and release the mapping."""
        if self._mapping is None:
            return
        try:
            if self._registered:
                with torch.cuda.device(self._device):
                    try:
                        torch.cuda.synchronize(self._device)
                    finally:
                        self._check(self._driver.cuMemHostUnregister(self._base), "cuMemHostUnregister")
                        self._registered = False
        finally:
            if not self._registered:
                self._mapping.close()
                self._mapping = None
