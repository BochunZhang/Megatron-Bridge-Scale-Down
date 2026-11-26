#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Report and save NVMe, RDMA, and GPUDirect Storage diagnostics from a Linux container."""

import argparse
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
from ctypes.util import find_library
from datetime import datetime
from pathlib import Path


logger = logging.getLogger(__name__)
SYSFS = Path("/sys")
DEV = Path("/dev")
PROC = Path("/proc")
MOUNTINFO = PROC / "self/mountinfo"
NVME_NAMESPACE = re.compile(r"(nvme[0-9]+)(?:c[0-9]+)?(n[0-9]+)(?:p[0-9]+)?")


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


def _show_nvme_mounts(mountinfo: str) -> None:
    logger.info("NVMe mounts (target  source  filesystem  NVMe namespaces)")
    targets: set[str] = set()
    namespaces: set[str] = set()
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
    logger.info("nvme_mountpoints=%d", len(targets))
    logger.info("mounted_nvme_namespaces=%d", len(namespaces))


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
    search_dirs = [*os.get_exec_path(), "/usr/local/cuda/gds/tools", *(str(path / "gds/tools") for path in cuda_dirs)]
    if cuda_home:
        search_dirs.insert(0, str(Path(cuda_home) / "gds/tools"))
    for directory in search_dirs:
        if checker := shutil.which("gdscheck", path=directory) or shutil.which("gdscheck.py", path=directory):
            return checker
    return None


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

    checker = _find_gdscheck()
    if checker is None:
        logger.info("gdscheck unavailable in PATH or CUDA toolkit directories")
        logger.info("libcufile lookup: %s", find_library("cufile") or "no match")
        return

    cufile_log = log_path.with_suffix(".cufile.log")
    logger.info("Running %s -p", checker)
    try:
        result = subprocess.run(
            [checker, "-p"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            env={**os.environ, "CUFILE_LOGFILE_PATH": str(cufile_log)},
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        if error.output:
            logger.info("%s", error.output.decode(errors="replace").rstrip("\n"))
        logger.warning("gdscheck -p timed out after %s seconds", error.timeout)
    except OSError as error:
        logger.warning("Cannot run gdscheck -p: %s", error)
    else:
        if result.stdout:
            logger.info("%s", result.stdout.rstrip("\n"))
        logger.info("gdscheck_exit_code=%d", result.returncode)
    if cufile_log.is_file():
        logger.info("cuFile log: %s", cufile_log)


def main() -> int:
    """Report device visibility using read-only Linux kernel interfaces."""
    argparse.ArgumentParser(
        description=__doc__, epilog="Logs are saved to results/gb200/nvme-io/ under the repository root."
    ).parse_args()
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
    _show_nvme_mounts(mountinfo)
    _show_rdma_links()
    uverbs = sorted(path.name for path in (DEV / "infiniband").glob("uverbs[0-9]*") if path.is_char_device())
    logger.info("uverbs_device_nodes=%d (%s)", len(uverbs), ", ".join(uverbs) or "none")
    _show_gds(log_path)
    _show_filesystem_usage()
    return 0


if __name__ == "__main__":
    sys.exit(main())
