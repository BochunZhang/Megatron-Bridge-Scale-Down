# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Select local socket interfaces for the single-node layer-by-layer launcher."""

import logging
import subprocess
from collections.abc import Mapping

_LOG = logging.getLogger(__name__)


def _ip_lines(env: Mapping[str, str], *args: str) -> list[str]:
    command = ["ip", "-o", "-4", *args]
    try:
        result = subprocess.run(command, env=env, check=True, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise RuntimeError(
            "Install iproute2 so the launcher can select a local IPv4 interface"
        ) from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or f"exit status {error.returncode}"
        raise RuntimeError(f"{' '.join(command)} failed with {detail}") from error
    except OSError as error:
        raise RuntimeError(f"Cannot run {' '.join(command)}: {error}") from error
    return result.stdout.splitlines()


def _default_interface(env: Mapping[str, str], interfaces: set[str]) -> str:
    for name in ("eth0", "bond1"):
        if name in interfaces:
            return name
    for line in _ip_lines(env, "route", "show", "default"):
        fields = line.split()
        for field, name in zip(fields, fields[1:]):
            if field == "dev" and name != "lo" and name in interfaces:
                return name
    available = ", ".join(sorted(interfaces)) or "none"
    raise RuntimeError(
        f"No active IPv4 uplink found (up with IPv4: {available}). "
        "Set GLOO_SOCKET_IFNAME to the local interface to use"
    )


def resolve_network(env: Mapping[str, str]) -> dict[str, str]:
    """Return validated Gloo and exact-match NCCL interfaces for one node."""
    interfaces = set()
    for line in _ip_lines(env, "addr", "show", "up"):
        fields = line.split()
        if len(fields) > 3 and fields[2] == "inet":
            interfaces.add(fields[1].split("@", 1)[0])

    gloo = env.get("GLOO_SOCKET_IFNAME", "")
    if gloo not in interfaces:
        if gloo:
            _LOG.warning("Ignoring GLOO_SOCKET_IFNAME=%r. Select an active IPv4 interface.", gloo)
        gloo = _default_interface(env, interfaces)

    nccl_override = env.get("NCCL_SOCKET_IFNAME", "")
    nccl = nccl_override.removeprefix("=")
    if nccl not in interfaces:
        if nccl_override:
            _LOG.warning(
                "Ignoring NCCL_SOCKET_IFNAME=%r. Select one active IPv4 interface.", nccl_override
            )
        nccl = gloo

    return {"GLOO_SOCKET_IFNAME": gloo, "NCCL_SOCKET_IFNAME": f"={nccl}"}
