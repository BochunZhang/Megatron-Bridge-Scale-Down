#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Report and save NVMe, RDMA, and GPUDirect Storage diagnostics from a Linux container."""

import argparse
import ctypes as ct
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from contextlib import ExitStack
from ctypes.util import find_library
from datetime import datetime
from pathlib import Path


logger = logging.getLogger(__name__)
SYSFS = Path("/sys")
DEV = Path("/dev")
PROC = Path("/proc")
MOUNTINFO = PROC / "self/mountinfo"
NVME_NAMESPACE = re.compile(r"(nvme[0-9]+)(?:c[0-9]+)?(n[0-9]+)(?:p[0-9]+)?")
GDS_PROBE_BYTES = 64 * 1024


def _nvme_backing(block: Path) -> set[str]:
    try:
        block = block.resolve(strict=True)
    except OSError:
        return set()
    if match := NVME_NAMESPACE.fullmatch(block.name):
        return {"".join(match.groups())}
    if (block / "partition").is_file():
        return _nvme_backing(block.parent)
    # Device-mapper and MD devices expose their backing devices as slaves.
    return {namespace for slave in (block / "slaves").glob("*") for namespace in _nvme_backing(slave)}


def _show_nvme_mounts(mountinfo: str) -> dict[Path, str]:
    logger.info("NVMe mounts (target  source  filesystem  NVMe namespaces)")
    targets: set[str] = set()
    namespaces: set[str] = set()
    local_mounts: dict[Path, str] = {}
    for line in mountinfo.rstrip("\n").split("\n"):
        fields, filesystem_info = line.split(" - ", 1)
        _, _, device, _, target, *_ = fields.split(" ")
        filesystem, source, _ = filesystem_info.split(" ", 2)
        backing = _nvme_backing(SYSFS / "dev/block" / device)
        # A bind mount can expose an NVMe filesystem without exposing its block node.
        if not backing and source.startswith("/dev/") and (match := NVME_NAMESPACE.fullmatch(source[5:])):
            backing = {"".join(match.groups())}
        if backing:
            targets.add(target)
            namespaces.update(backing)
            logger.info("  %s  %s  %s  %s", target, source, filesystem, ", ".join(sorted(backing)))
            if filesystem in {"ext4", "xfs"}:
                path = re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), target)
                local_mounts[Path(path)] = device
    logger.info("nvme_mountpoints=%d", len(targets))
    logger.info("mounted_nvme_namespaces=%d", len(namespaces))
    return local_mounts


def _show_filesystem_usage() -> None:
    logger.info("\nFilesystems (df -h)")
    try:
        result = subprocess.run(
            ["df", "-h"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", check=False
        )
    except OSError as error:
        logger.warning("Cannot run df -h: %s", error)
        return
    if result.stdout:
        logger.info("%s", result.stdout.rstrip("\n"))
    if result.returncode:
        logger.warning("df -h exited with status %d", result.returncode)


def _sysfs_port_links(ib_class: Path) -> list[dict[str, str]]:
    links = []
    for port in sorted(ib_class.glob("*/ports/[0-9]*")):
        try:
            state = (port / "state").read_text().strip()
        except OSError:
            state = "unknown"
        links.append({"ifname": port.parent.parent.name, "port": port.name, "state": state})
    return links


def _show_rdma_links() -> None:
    ib_class = SYSFS / "class/infiniband"
    logger.info("\nRDMA ports (HCA/port  state  mapped network interfaces)")
    try:
        result = subprocess.run(["rdma", "-j", "link", "show"], capture_output=True, text=True, check=True, timeout=5)
        links = json.loads(result.stdout)
        method = "rdma link show"
    except (OSError, subprocess.SubprocessError, ValueError):
        if not ib_class.is_dir():
            logger.info("RDMA port inventory unavailable (rdma command and InfiniBand sysfs)")
            logger.info("rdma_netdevs=unavailable")
            return
        links = _sysfs_port_links(ib_class)
        method = "InfiniBand sysfs"
    logger.info("RDMA port inventory source: %s", method)

    try:
        visible_names = {name for _, name in socket.if_nameindex()}
    except OSError:
        visible_names = None

    hcas: set[str] = set()
    ports: set[tuple[str, str]] = set()
    netdevs: set[str] = set()
    for link in links:
        hca, port = link["ifname"], str(link["port"])
        hcas.add(hca)
        ports.add((hca, port))
        names: set[str] = set()
        if visible_names is not None:
            netdev = link.get("netdev")
            if netdev and netdev in visible_names:
                names.add(netdev)
            else:
                for path in (ib_class / hca / "ports" / port / "gid_attrs/ndevs").glob("*"):
                    try:
                        name = path.read_text().strip()
                    except OSError:  # Unpopulated GID slots can return ENODEV.
                        continue
                    if name in visible_names:
                        names.add(name)
        netdevs.update(names)
        logger.info("  %s/%s  %s  %s", hca, port, link.get("state", "unknown"), ", ".join(sorted(names)) or "-")
    logger.info("rdma_hcas_with_ports=%d", len(hcas))
    logger.info("rdma_ports=%d", len(ports))
    logger.info("rdma_netdevs=%s", len(netdevs) if visible_names is not None else "unavailable")


def _find_gdscheck() -> str | None:
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    cuda_dirs = sorted(
        Path("/usr/local").glob("cuda-[0-9]*"),
        key=lambda path: tuple(map(int, re.findall(r"[0-9]+", path.name))),
        reverse=True,
    )
    search_dirs = [
        *os.get_exec_path(),
        "/usr/local/cuda/gds/tools",
        "/usr/libexec/gds/tools",
        *(str(path / "gds/tools") for path in cuda_dirs),
    ]
    if cuda_home:
        search_dirs.insert(0, str(Path(cuda_home) / "gds/tools"))
    for directory in search_dirs:
        if checker := shutil.which("gdscheck", path=directory) or shutil.which("gdscheck.py", path=directory):
            return checker
    return None


def _run_probe(command: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            env=env,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        if error.output:
            logger.info("%s", error.output.decode(errors="replace").rstrip("\n"))
        logger.warning("%s timed out after %s seconds", command[0], error.timeout)
    except OSError as error:
        logger.warning("Cannot run %s: %s", command[0], error)
    return None


def _show_gds_packages() -> None:
    logger.info("\nGDS packages (local APT index)")
    if not (apt_cache := shutil.which("apt-cache")):
        logger.info("apt-cache unavailable")
        return
    result = _run_probe(
        [apt_cache, "--names-only", "search", r"^(gds-tools|nvidia-gds|nvidia-fs|libcufile)(-|[0-9]|$)"]
    )
    if result is None:
        return
    if result.returncode:
        logger.info("apt-cache search exited with status %d\n%s", result.returncode, result.stdout.rstrip("\n"))
        return
    packages = sorted({line.split(" - ", 1)[0] for line in result.stdout.splitlines() if " - " in line})
    if not packages:
        logger.info("No matching packages in the local APT index")
        return
    if policy := _run_probe([apt_cache, "policy", *packages]):
        if policy.stdout:
            logger.info("%s", policy.stdout.rstrip("\n"))
        if policy.returncode:
            logger.warning("apt-cache policy exited with status %d", policy.returncode)


def _show_gds(log_path: Path) -> None:
    logger.info("\nGPUDirect Storage (GDS) platform diagnostics")
    try:
        loaded = {line.split()[0] for line in (PROC / "modules").read_text().splitlines() if line}
    except OSError as error:
        logger.info("/proc/modules unavailable: %s", error)
        loaded = set()

    logger.info("GDS and related kernel modules (container view)")
    for name in ("nvidia_fs", "nvidia", "nvidia_uvm", "nvme", "nvme_core", "nvidia_peermem", "ib_core", "mlx5_ib"):
        if name in loaded:
            status = "loaded (/proc/modules)"
        elif (SYSFS / "module" / name).is_dir():
            status = "visible (/sys/module)"
        else:
            status = "not visible"
        logger.info("  %s  %s", name, status)

    for name in ("version", "devcount"):
        path = PROC / "driver/nvidia-fs" / name
        try:
            value = path.read_text().strip()
        except OSError as error:
            value = f"unavailable ({error.strerror})"
        logger.info("%s: %s", path, value)
    nodes = sorted(str(path) for path in DEV.glob("nvidia[_-]fs*") if path.is_char_device())
    logger.info("nvidia_fs_device_nodes=%d (%s)", len(nodes), ", ".join(nodes) or "none visible")
    if modinfo := shutil.which("modinfo"):
        if result := _run_probe([modinfo, "-n", "nvidia_fs"]):
            logger.info("modinfo -n nvidia_fs (exit %d): %s", result.returncode, result.stdout.strip())

    try:
        with (PROC / "kallsyms").open() as symbols:
            observed = any(len(fields) > 2 and fields[2] == "p2pdma_pgmap_ops" for fields in map(str.split, symbols))
    except OSError as error:
        logger.info("p2pdma_pgmap_ops: unavailable (%s)", error.strerror)
    else:
        logger.info("p2pdma_pgmap_ops: %s", "visible" if observed else "not observed")

    checker = _find_gdscheck()
    if checker is None:
        logger.info("gdscheck unavailable in PATH or CUDA toolkit directories")
        return

    cufile_log = log_path.with_suffix(".cufile.log")
    logger.info("Running %s -p", checker)
    if result := _run_probe([checker, "-p"], env={**os.environ, "CUFILE_LOGFILE_PATH": str(cufile_log)}):
        if result.stdout:
            logger.info("%s", result.stdout.rstrip("\n"))
        logger.info("gdscheck_exit_code=%d", result.returncode)
    if cufile_log.is_file():
        logger.info("cuFile log: %s", cufile_log)


def _gds_probe_dir(mounts: dict[Path, str], requested_dir: Path | None) -> Path | None:
    candidates = [requested_dir] if requested_dir else [Path("/tmp"), *sorted(mounts)]
    device_ids = set(mounts.values())
    for directory in candidates:
        if not directory.is_dir() or not os.access(directory, os.W_OK | os.X_OK):
            continue
        try:
            device = directory.stat().st_dev
        except OSError:
            continue
        if f"{os.major(device)}:{os.minor(device)}" in device_ids:
            return directory.resolve()
    return None


def _show_cufile_probe(log_path: Path, directory: Path | None) -> None:
    logger.info("\ncuFile runtime probe (compatibility fallback disabled)")
    library = find_library("cufile")
    logger.info("libcufile lookup: %s", library or "no match")
    cufile_log = log_path.with_suffix(".cufile-probe.log")
    command = [sys.executable, str(Path(__file__).resolve()), "--_cufile-probe", library or "libcufile.so.0"]
    try:
        with ExitStack() as stack:
            config_dir = Path(
                stack.enter_context(tempfile.TemporaryDirectory(prefix=".gds-config-", dir=log_path.parent))
            )
            config = config_dir / "cufile.json"
            config.write_text(
                json.dumps(
                    {
                        "properties": {
                            "allow_compat_mode": False,
                            "force_compat_mode": False,
                            "use_pci_p2pdma": True,
                        },
                        "block": {"nvme": {"use_pci_p2pdma": True}},
                    }
                )
            )
            if directory is None:
                logger.info("cuFile read skipped (no writable, verified ext4/XFS NVMe directory)")
            else:
                try:
                    scratch = Path(
                        stack.enter_context(tempfile.TemporaryDirectory(prefix=".gds-read-", dir=directory))
                    )
                    probe_file = scratch / "probe.bin"
                    with probe_file.open("xb") as stream:
                        stream.write(os.urandom(GDS_PROBE_BYTES))
                        stream.flush()
                        os.fsync(stream.fileno())
                except OSError as error:
                    logger.warning("cuFile read setup failed for %s: %s", directory, error)
                else:
                    logger.info("cuFile read target: %s (%d bytes)", directory, GDS_PROBE_BYTES)
                    command.append(str(probe_file))
            env = {
                **os.environ,
                "CUFILE_ENV_PATH_JSON": str(config),
                "CUFILE_ALLOW_COMPAT_MODE": "false",
                "CUFILE_FORCE_COMPAT_MODE": "false",
                "CUFILE_USE_PCIP2PDMA": "true",
                "CUFILE_LOGFILE_PATH": str(cufile_log),
            }
            if result := _run_probe(command, env=env):
                if result.stdout:
                    logger.info("%s", result.stdout.rstrip("\n"))
                logger.info("cufile_probe_exit_code=%d", result.returncode)
    except OSError as error:
        logger.warning("Cannot prepare cuFile probe: %s", error)
    if cufile_log.is_file():
        logger.info("cuFile probe log: %s", cufile_log)


def main() -> int:
    """Report device visibility and test cuFile on a temporary NVMe file."""
    parser = argparse.ArgumentParser(
        description=__doc__, epilog="Logs are saved to results/gb200/nvme-io/ under the repository root."
    )
    parser.add_argument(
        "--gds-test-dir", type=Path, help="Writable ext4/XFS NVMe directory for the temporary cuFile read test."
    )
    args = parser.parse_args()
    hostname = socket.gethostname()
    results_dir = Path(__file__).resolve().parents[3] / "results/gb200/nvme-io"
    log_path = results_dir / f"{hostname}-{datetime.now():%Y%m%d-%H%M%S-%f}.log"
    try:
        results_dir.mkdir(parents=True, exist_ok=True)
        log_file = logging.FileHandler(log_path, mode="x", encoding="utf-8", errors="backslashreplace")
    except OSError as error:
        logger.error("Cannot create log %s: %s", log_path, error)
        return 1
    logging.basicConfig(
        level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler(sys.stdout), log_file]
    )
    logger.info("Log: %s", log_path)
    try:
        mountinfo = MOUNTINFO.read_bytes().decode("utf-8", errors="surrogateescape")
    except OSError as error:
        logger.error("Cannot read mount metadata (Linux required): %s", error)
        return 1

    logger.info("Host: %s", hostname)
    nvme_mounts = _show_nvme_mounts(mountinfo)
    _show_rdma_links()
    uverbs = sorted(path.name for path in (DEV / "infiniband").glob("uverbs[0-9]*") if path.is_char_device())
    logger.info("uverbs_device_nodes=%d (%s)", len(uverbs), ", ".join(uverbs) or "none")
    _show_gds(log_path)
    _show_gds_packages()
    probe_dir = _gds_probe_dir(nvme_mounts, args.gds_test_dir)
    if args.gds_test_dir is not None and probe_dir is None:
        logger.warning("Requested cuFile target is outside the writable ext4/XFS NVMe mounts: %s", args.gds_test_dir)
    _show_cufile_probe(log_path, probe_dir)
    _show_filesystem_usage()
    return 0


class _CuFileError(ct.Structure):
    _fields_ = [("err", ct.c_int), ("cu_err", ct.c_int)]


class _CuFileHandle(ct.Union):
    _fields_ = [("fd", ct.c_int), ("handle", ct.c_void_p)]


class _CuFileDescriptor(ct.Structure):
    _fields_ = [("type", ct.c_int), ("handle", _CuFileHandle), ("fs_ops", ct.c_void_p)]


def _bind(library: ct.CDLL, name: str, result: type | None, *arguments: type) -> ct._CFuncPtr:
    function = getattr(library, name)
    function.argtypes = list(arguments)
    function.restype = result
    return function


def _check_cuda(status: int, operation: str) -> None:
    if status:
        raise RuntimeError(f"{operation} failed with CUDA status {status}")


def _check_cufile(status: _CuFileError, operation: str) -> None:
    if status.err:
        raise RuntimeError(f"{operation} failed with cuFile status {status.err}, CUDA status {status.cu_err}")


def _read_cufile(cufile: ct.CDLL, path: Path, cleanup: ExitStack) -> None:
    with path.open("rb") as probe_file:
        expected = probe_file.read(GDS_PROBE_BYTES)
    if len(expected) != GDS_PROBE_BYTES:
        raise RuntimeError(f"Probe file must contain at least {GDS_PROBE_BYTES} bytes")

    cuda = ct.CDLL("libcuda.so.1")
    _check_cuda(_bind(cuda, "cuInit", ct.c_int, ct.c_uint)(0), "cuInit")
    device = ct.c_int()
    get_device = _bind(cuda, "cuDeviceGet", ct.c_int, ct.POINTER(ct.c_int), ct.c_int)
    _check_cuda(get_device(ct.byref(device), 0), "cuDeviceGet(0)")
    context = ct.c_void_p()
    retain = _bind(cuda, "cuDevicePrimaryCtxRetain", ct.c_int, ct.POINTER(ct.c_void_p), ct.c_int)
    release = _bind(cuda, "cuDevicePrimaryCtxRelease_v2", ct.c_int, ct.c_int)
    _check_cuda(retain(ct.byref(context), device), "cuDevicePrimaryCtxRetain")
    cleanup.callback(release, device)
    set_current = _bind(cuda, "cuCtxSetCurrent", ct.c_int, ct.c_void_p)
    _check_cuda(set_current(context), "cuCtxSetCurrent")
    cleanup.callback(set_current, None)

    allocate = _bind(cuda, "cuMemAlloc_v2", ct.c_int, ct.POINTER(ct.c_uint64), ct.c_size_t)
    free = _bind(cuda, "cuMemFree_v2", ct.c_int, ct.c_uint64)
    address = ct.c_uint64()
    _check_cuda(allocate(ct.byref(address), 2 * GDS_PROBE_BYTES), "cuMemAlloc")
    cleanup.callback(free, address)
    # Leave room to align the registered address even if cuMemAlloc's base is unaligned.
    aligned_address = ((address.value + GDS_PROBE_BYTES - 1) // GDS_PROBE_BYTES) * GDS_PROBE_BYTES
    gpu_pointer = ct.c_void_p(aligned_address)

    file_descriptor = os.open(path, os.O_RDONLY | os.O_DIRECT)
    cleanup.callback(os.close, file_descriptor)
    descriptor = _CuFileDescriptor(1, _CuFileHandle(fd=file_descriptor), None)  # CU_FILE_HANDLE_TYPE_OPAQUE_FD
    file_handle = ct.c_void_p()
    register_file = _bind(
        cufile, "cuFileHandleRegister", _CuFileError, ct.POINTER(ct.c_void_p), ct.POINTER(_CuFileDescriptor)
    )
    deregister_file = _bind(cufile, "cuFileHandleDeregister", None, ct.c_void_p)
    _check_cufile(register_file(ct.byref(file_handle), ct.byref(descriptor)), "cuFileHandleRegister")
    cleanup.callback(deregister_file, file_handle)
    register_buffer = _bind(cufile, "cuFileBufRegister", _CuFileError, ct.c_void_p, ct.c_size_t, ct.c_int)
    deregister_buffer = _bind(cufile, "cuFileBufDeregister", _CuFileError, ct.c_void_p)
    _check_cufile(register_buffer(gpu_pointer, GDS_PROBE_BYTES, 0), "cuFileBufRegister")
    cleanup.callback(deregister_buffer, gpu_pointer)

    # Linux aarch64 and x86_64 both have a signed 64-bit off_t.
    read = _bind(cufile, "cuFileRead", ct.c_ssize_t, ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_int64, ct.c_int64)
    ct.set_errno(0)
    count = read(file_handle, gpu_pointer, GDS_PROBE_BYTES, 0, 0)
    error_number = ct.get_errno()
    logger.info("cufile_read_bytes=%d", count)
    if count == -1 and error_number:
        raise OSError(error_number, os.strerror(error_number), path)
    if count != GDS_PROBE_BYTES:
        raise RuntimeError(f"cuFileRead returned {count}, expected {GDS_PROBE_BYTES} bytes")

    received = ct.create_string_buffer(GDS_PROBE_BYTES)
    copy_to_host = _bind(cuda, "cuMemcpyDtoH_v2", ct.c_int, ct.c_void_p, ct.c_uint64, ct.c_size_t)
    _check_cuda(copy_to_host(received, aligned_address, GDS_PROBE_BYTES), "cuMemcpyDtoH")
    if received.raw != expected:
        logger.info("cufile_data_match=no")
        raise RuntimeError("cuFileRead data integrity check failed")
    logger.info("cufile_data_match=yes")


def _cufile_worker(arguments: list[str]) -> int:
    """Check library loading and initialization, then optionally read a test file."""
    parser = argparse.ArgumentParser(description="Internal cuFile worker")
    parser.add_argument("library")
    parser.add_argument("probe_file", type=Path, nargs="?")
    args = parser.parse_args(arguments)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        cufile = ct.CDLL(args.library, use_errno=True)
    except OSError as error:
        logger.error("cufile_library_load=failed (%s)", error)
        return 1
    logger.info("cufile_library_load=ok (%s)", args.library)

    try:
        with ExitStack() as cleanup:
            driver_open = _bind(cufile, "cuFileDriverOpen", _CuFileError)
            driver_close = _bind(cufile, "cuFileDriverClose", _CuFileError)
            status = driver_open()
            logger.info("cufile_driver_open=cuFile:%d CUDA:%d", status.err, status.cu_err)
            _check_cufile(status, "cuFileDriverOpen")
            cleanup.callback(driver_close)
            if args.probe_file is not None:
                _read_cufile(cufile, args.probe_file, cleanup)
    except (AttributeError, OSError, RuntimeError) as error:
        logger.error("cufile_probe_error=%s", error)
        return 1
    return 0


if __name__ == "__main__":
    # Keep native library faults and timeouts isolated from the rest of the report.
    if sys.argv[1:2] == ["--_cufile-probe"]:
        sys.exit(_cufile_worker(sys.argv[2:]))
    sys.exit(main())
