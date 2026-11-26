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
MEGATRON_LM_DIR defaults to the sibling megatron-lm checkout with async AIO.
PYTHON can select the training interpreter. The default uses uv run --no-sync.
The temporary build needs setuptools, Python headers, a C++17 compiler, libaio
headers, and NVTX3 headers (see NVTX_INCLUDE_DIR in the native extension).
The 1 GiB cases use two 1 GiB files and two 1 GiB host buffers.

Results go to results/gb200/nvme-io/native-*/summary.csv and stdout in GiB/s.
EOF
    exit 0
fi

(( $# <= 1 )) || die 'Expected one NVMe directory at most. See --help.'
[[ $(uname -s) == Linux ]] || die 'Run this benchmark on the GB200 Linux node.'
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)
nvme_dir=$(cd -- "${1:-${NVME_PATH:-/tmp}}" && pwd -P)
native_source="${MEGATRON_LM_DIR:-$repo_root/../megatron-lm}/src/nvme/deepspeed"
[[ -r $native_source/aio.cpp ]] || die 'Set MEGATRON_LM_DIR to the Megatron-LM checkout with native AIO.'

if [[ -n ${PYTHON:-} ]]; then
    python=("$PYTHON")
else
    python=(uv run --project "$repo_root" --no-sync python)
fi

scratch_dir=$(mktemp -d -- "$nvme_dir/.gb200-native-aio.XXXXXX")
trap 'rm -rf -- "$scratch_dir"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
cp -- "$native_source/"{aio.cpp,setup.py,LICENSE} "$scratch_dir/"

results_root="$repo_root/results/gb200/nvme-io"
mkdir -p -- "$results_root"
results_dir=$(mktemp -d -- "$results_root/native-$(date -u +%Y%m%dT%H%M%SZ).XXXXXX")
printf 'Results directory: %s\n' "$results_dir" >&2
{
    uname -snrm
    findmnt --target "$nvme_dir" --output SOURCE,TARGET,FSTYPE,MAJ:MIN
    (cd -- "$scratch_dir" && sha256sum aio.cpp setup.py)
    cat <<'EOF'
Engine: megatron_nvme_aio, libaio with O_DIRECT
Build: one temporary extension per block size with a queue depth limit of 64
QP: queue depth per worker; concurrent read/write uses one worker per direction
Files: two fully allocated, initialized files of the specified size on the same mount
Timing: one pass per direction after warmup; writes overwrite existing data without fsync
Bandwidth: read/write use their own elapsed time; total uses the full interval
EOF
} >"$results_dir/host_info.txt"

partial="$results_dir/partial.csv"
summary="$results_dir/summary.csv"
printf '%s\n' 'file_size,block_kib,qp,mode,read_gib_per_s,write_gib_per_s,total_gib_per_s' >"$partial"

# Each process imports exactly one build of the native extension.
for block_kib in 64 128 256; do
    printf 'Running native AIO with %s KiB blocks\n' "$block_kib" >&2
    "${python[@]}" -u - "$scratch_dir" "$results_dir/build-${block_kib}k.log" "$block_kib" >>"$partial" <<'PYTHON'
import csv
import importlib.util
import mmap
import os
import re
import shutil
import subprocess
import sys
import sysconfig
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from threading import Barrier
from time import perf_counter_ns
from types import ModuleType


MIB = 1024**2
GIB = 1024**3
FILE_SIZES = (("40MiB", 40 * MIB), ("160MiB", 160 * MIB), ("320MiB", 320 * MIB), ("1GiB", GIB))
QUEUE_DEPTHS = (1, 2, 4, 8, 16, 32, 64)


def _build_aio(scratch: Path, *, block_kib: int, build_log: Path) -> ModuleType:
    build = scratch / f"build-{block_kib}k"
    build.mkdir()
    source = (scratch / "aio.cpp").read_text(encoding="utf-8")
    max_depth = max(QUEUE_DEPTHS)
    for constant, value in (("size_t kBlockSize", block_kib * 1024), ("int kMaxQueueDepth", max_depth)):
        source, count = re.subn(
            rf"^constexpr {constant} = [^;\n]+;", f"constexpr {constant} = {value};", source, flags=re.M
        )
        if count != 1:
            raise SystemExit(f"Expected one definition of {constant} in the native AIO source. Check MEGATRON_LM_DIR.")
    source = source.replace("AIO queue depth must be in [1, 16]", f"AIO queue depth must be in [1, {max_depth}]")
    (build / "aio.cpp").write_text(source, encoding="utf-8")
    for name in ("setup.py", "LICENSE"):
        shutil.copyfile(scratch / name, build / name)

    with build_log.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            [sys.executable, "setup.py", "build_ext", "--inplace"], cwd=build, stdout=log, stderr=subprocess.STDOUT
        )
    if result.returncode:
        raise SystemExit(f"Could not build the {block_kib} KiB native AIO extension. See {build_log}.")

    name = "megatron_nvme_aio"
    library = build / (name + sysconfig.get_config_var("EXT_SUFFIX"))
    spec = importlib.util.spec_from_file_location(name, library)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load the native AIO extension at {library}.")
    aio = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(aio)
    return aio


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
    initializer = aio.create_worker(16)
    data = os.urandom(MIB)
    zero = bytes(MIB)
    block_bytes = block_kib * 1024

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
                    reader = aio.create_worker(qp)
                    writer = aio.create_worker(qp)
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


scratch = Path(sys.argv[1])
build_log = Path(sys.argv[2])
block_kib = int(sys.argv[3])
_benchmark(_build_aio(scratch, block_kib=block_kib, build_log=build_log), scratch, block_kib)
PYTHON
done

mv -- "$partial" "$summary"
cat -- "$summary"
