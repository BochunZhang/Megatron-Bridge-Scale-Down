#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

set -euo pipefail
export LC_ALL=C

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

if [[ ${1:-} == -h || ${1:-} == --help ]]; then
    cat <<'EOF'
Usage: run_native_nvme_io_benchmark.sh [NVME_DIR]

Measure sequential read, sequential write, and concurrent read/write with
Megatron-LM's native libaio extension. QP is the queue depth per worker.
Concurrent read/write uses two workers and two files of the specified size.

File sizes are 40 MiB, 160 MiB, 320 MiB, and 1 GiB. QP ranges from 1 through
64 by powers of two. Block sizes are 64 KiB, 128 KiB, and 256 KiB.

NVME_DIR defaults to $NVME_PATH or /tmp. Use a directory on the local NVMe.
Requires megatron-nvme-aio 0.1.1 or newer in the active Python environment.
The default interpreter is python from PATH. Set PYTHON to override it.
The 1 GiB cases use two 1 GiB files and two 1 GiB host buffers.

Results go to results/gb200/nvme-io/native-*/summary.csv and stdout in GiB/s.
EOF
    exit 0
fi

(( $# <= 1 )) || die 'Expected one NVMe directory at most. See --help.'
[[ $(uname -s) == Linux ]] || die 'Run this benchmark on the GB200 Linux node.'
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)
nvme_dir=$(cd -- "${1:-${NVME_PATH:-/tmp}}" && pwd -P)
python_bin=${PYTHON:-python}
command -v "$python_bin" >/dev/null || die 'Activate the Python environment with megatron-nvme-aio or set PYTHON.'

scratch_dir=$(mktemp -d -- "$nvme_dir/.gb200-native-aio.XXXXXX")
trap 'rm -rf -- "$scratch_dir"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

results_root="$repo_root/results/gb200/nvme-io"
mkdir -p -- "$results_root"
results_dir=$(mktemp -d -- "$results_root/native-$(date -u +%Y%m%dT%H%M%SZ).XXXXXX")
printf 'Results directory: %s\n' "$results_dir" >&2
{
    uname -snrm
    findmnt --target "$nvme_dir" --output SOURCE,TARGET,FSTYPE,MAJ:MIN
    cat <<'EOF'
Engine: megatron_nvme_aio, libaio with O_DIRECT
QP: queue depth per worker; concurrent read/write uses one worker per direction
Files: two fully allocated, initialized files of the specified size on the same mount
Timing: one pass per direction after warmup; writes overwrite existing data without fsync
Bandwidth: read/write use their own elapsed time; total uses the full interval
EOF
} >"$results_dir/host_info.txt"

partial="$results_dir/partial.csv"
summary="$results_dir/summary.csv"
printf '%s\n' 'file_size,block_kib,qp,mode,read_gib_per_s,write_gib_per_s,total_gib_per_s' >"$partial"

"$python_bin" -u - "$scratch_dir" "$results_dir/host_info.txt" >>"$partial" <<'PYTHON'
import csv
import mmap
import os
import sys
from concurrent.futures import ThreadPoolExecutor, wait
from hashlib import sha256
from pathlib import Path
from threading import Barrier
from time import perf_counter_ns
from types import ModuleType


MIB = 1024**2
GIB = 1024**3
FILE_SIZES = (("40MiB", 40 * MIB), ("160MiB", 160 * MIB), ("320MiB", 320 * MIB), ("1GiB", GIB))
BLOCK_SIZES_KIB = (64, 128, 256)
QUEUE_DEPTHS = (1, 2, 4, 8, 16, 32, 64)


def _load_aio() -> ModuleType:
    try:
        import megatron_nvme_aio
    except ImportError as error:
        raise SystemExit(
            f"Cannot import megatron_nvme_aio in {sys.executable}: {error}. Install megatron-nvme-aio 0.1.1 or newer."
        ) from None

    try:
        megatron_nvme_aio.create_worker(max(QUEUE_DEPTHS), max(BLOCK_SIZES_KIB) * 1024)
    except (TypeError, ValueError):
        raise SystemExit(
            f"Upgrade megatron-nvme-aio in {sys.executable} to 0.1.1 or newer. "
            "The installed extension does not support both QP 64 and configurable block sizes."
        ) from None
    return megatron_nvme_aio


def _transfer(
    aio: ModuleType,
    worker: object,
    buffer: mmap.mmap | memoryview,
    path: Path,
    *,
    write: bool,
    barrier: Barrier | None = None,
) -> tuple[int, int]:
    if barrier is not None:
        barrier.wait()
    start = perf_counter_ns()
    aio.wait(aio.submit(worker, buffer, str(path), write, 0))
    return start, perf_counter_ns()


def _prepare(aio: ModuleType, worker: object, buffer: mmap.mmap, path: Path) -> None:
    with path.open("wb") as file:
        os.posix_fallocate(file.fileno(), 0, len(buffer))
        # Materialize the extents so direct reads fetch data from the drive.
        _transfer(aio, worker, buffer, path, write=True)
        os.fsync(file.fileno())


def _bandwidth(size: int, interval: tuple[int, int]) -> str:
    start, end = interval
    return f"{size / GIB * 1e9 / (end - start):.6f}"


def _benchmark(aio: ModuleType, scratch: Path, block_kib: int) -> None:
    output = csv.writer(sys.stdout, lineterminator="\n")
    read_path = scratch / "read.bin"
    write_path = scratch / "write.bin"
    block_bytes = block_kib * 1024
    initializer = aio.create_worker(16, block_bytes)
    data = os.urandom(MIB)
    zero = bytes(MIB)

    with ThreadPoolExecutor(max_workers=2) as pool:
        for file_size, size in FILE_SIZES:
            with mmap.mmap(-1, size) as read_buffer, mmap.mmap(-1, size) as write_buffer:
                # Fault in both aligned host buffers before timing any transfer.
                for offset in range(0, size, MIB):
                    read_buffer[offset : offset + MIB] = zero
                    write_buffer[offset : offset + MIB] = data
                _prepare(aio, initializer, write_buffer, read_path)
                _prepare(aio, initializer, write_buffer, write_path)

                for qp in QUEUE_DEPTHS:
                    reader = aio.create_worker(qp, block_bytes)
                    writer = aio.create_worker(qp, block_bytes)
                    # The workers open their libaio contexts on their first I/O.
                    with (
                        memoryview(read_buffer)[:block_bytes] as read_block,
                        memoryview(write_buffer)[:block_bytes] as write_block,
                    ):
                        _transfer(aio, reader, read_block, read_path, write=False)
                        _transfer(aio, writer, write_block, write_path, write=True)

                    key = (file_size, block_kib, qp)
                    read_rate = _bandwidth(size, _transfer(aio, reader, read_buffer, read_path, write=False))
                    output.writerow((*key, "sequential_read", read_rate, "", read_rate))
                    write_rate = _bandwidth(size, _transfer(aio, writer, write_buffer, write_path, write=True))
                    output.writerow((*key, "sequential_write", "", write_rate, write_rate))

                    barrier = Barrier(2)
                    read = pool.submit(_transfer, aio, reader, read_buffer, read_path, write=False, barrier=barrier)
                    write = pool.submit(_transfer, aio, writer, write_buffer, write_path, write=True, barrier=barrier)
                    wait((read, write))
                    read_interval, write_interval = read.result(), write.result()
                    if max(read_interval[0], write_interval[0]) >= min(read_interval[1], write_interval[1]):
                        raise RuntimeError(
                            f"Concurrent transfers did not overlap for {file_size}, {block_kib} KiB, QP {qp}"
                        )
                    start = min(read_interval[0], write_interval[0])
                    end = max(read_interval[1], write_interval[1])
                    output.writerow(
                        (
                            *key,
                            "concurrent_read_write",
                            _bandwidth(size, read_interval),
                            _bandwidth(size, write_interval),
                            _bandwidth(2 * size, (start, end)),
                        )
                    )


aio = _load_aio()
extension_path = Path(aio.__file__)
with Path(sys.argv[2]).open("a", encoding="utf-8") as host_info:
    host_info.write(f"Python: {sys.executable}\n")
    host_info.write(f"Extension: {extension_path}\n")
    host_info.write(f"Extension SHA256: {sha256(extension_path.read_bytes()).hexdigest()}\n")

scratch = Path(sys.argv[1])
for block_kib in BLOCK_SIZES_KIB:
    sys.stderr.write(f"Running native AIO with {block_kib} KiB blocks\n")
    _benchmark(aio, scratch, block_kib)
PYTHON

mv -- "$partial" "$summary"
cat -- "$summary"
