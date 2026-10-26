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

"""CPU-only checks for run_symm_cta_pause_sweep.sh.

The real sweep needs GB200 trays and the nccl-2.32.3-cta-pause build. These tests dry-run it with a
fake libnccl.so.2 that only carries the strings the script greps for: one launcher run per pause,
each in its own ``<id>-pause<cycles>cyc`` directory, and the option / library checks.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[3] / "examples" / "gb200" / "nvls-collective" / "run_symm_cta_pause_sweep.sh"
needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _fake_lib(tmp_path: Path, banner: str = "NCCL version 2.32.3-cta-pause+cuda13.0") -> Path:
    lib = tmp_path / "libnccl.so.2"
    lib.write_bytes(f"\0{banner}\0NCCL_SYM_AG_STMC_PAUSE_CYCLES\0".encode())
    return lib


def _sweep(lib: Path, *options: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "GLOO_SOCKET_IFNAME": "lo", "RUN_TIME": "sweep-test"}
    env.pop("NCCL_SYM_AG_STMC_PAUSE_CYCLES", None)
    command = ["bash", str(SCRIPT), "--venv", "none", "--nccl-lib", str(lib), "--nnodes", "1", "--group-sizes", "4"]
    return subprocess.run([*command, *options], capture_output=True, text=True, env=env, check=False)


@pytest.mark.unit
@needs_bash
def test_dry_run_gives_every_pause_its_own_tagged_run(tmp_path):
    result = _sweep(_fake_lib(tmp_path), "--max-ctas", "2", "--pause-cycles", "0,1000", "--dry-run")
    assert result.returncode == 0, result.stderr
    output = result.stdout + result.stderr
    assert "NCCL_SYM_AG_STMC_PAUSE_CYCLES=0: run id sweep-test-pause0cyc" in output
    assert "NCCL_SYM_AG_STMC_PAUSE_CYCLES=1000: run id sweep-test-pause1000cyc" in output
    torchruns = [line.replace("\\", "") for line in result.stdout.splitlines() if "torch.distributed.run" in line]
    assert len(torchruns) == 2
    for line, pause in zip(torchruns, (0, 1000)):
        assert f"gpu-num-4/sweep-test-pause{pause}cyc/symm" in line
        assert "--workloads ag,rs,ag+rs" in line and "--symm-kernel mc" in line
        assert "--cta-configs default;ag=1,rs=1;ag=1,rs=2;ag=2,rs=2" in line


@pytest.mark.unit
@needs_bash
@pytest.mark.parametrize(
    "options, match",
    [
        (["--pause-cycles", "0,abc"], "comma-separated list"),
        (["--pause-cycles", "500,500"], "twice"),
        (["--pause-cycles", "4294967296"], "uint32"),
        (["--max-ctas", "0"], "1..64"),
    ],
)
def test_bad_options_are_rejected(tmp_path, options, match):
    result = _sweep(_fake_lib(tmp_path), *options, "--dry-run")
    assert result.returncode != 0
    assert match in result.stderr


@pytest.mark.unit
@needs_bash
def test_a_library_without_the_pause_is_refused(tmp_path):
    result = _sweep(_fake_lib(tmp_path, banner="NCCL version 2.32.3-cta+cuda13.0"), "--dry-run")
    assert result.returncode != 0
    assert "not the nccl-2.32.3-cta-pause build" in result.stderr


@pytest.mark.unit
@needs_bash
def test_a_library_with_the_globaltimer_pause_is_refused(tmp_path):
    lib = tmp_path / "libnccl.so.2"
    lib.write_bytes(b"\0NCCL version 2.32.3-cta-pause+cuda13.0\0NCCL_SYM_AG_STMC_PAUSE_NS\0")
    result = _sweep(lib, "--dry-run")
    assert result.returncode != 0
    assert "does not read NCCL_SYM_AG_STMC_PAUSE_CYCLES" in result.stderr
