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
    result = _sweep(_fake_lib(tmp_path), "--pause-cycles", "0,1000", "--dry-run")
    assert result.returncode == 0, result.stderr
    output = result.stdout + result.stderr
    assert "NCCL_SYM_AG_STMC_PAUSE_CYCLES=0: run id sweep-test-pause0cyc" in output
    assert "NCCL_SYM_AG_STMC_PAUSE_CYCLES=1000: run id sweep-test-pause1000cyc" in output
    torchruns = [line.replace("\\", "") for line in result.stdout.splitlines() if "torch.distributed.run" in line]
    assert len(torchruns) == 2
    for line, pause in zip(torchruns, (0, 1000)):
        assert f"gpu-num-4/sweep-test-pause{pause}cyc/symm" in line
        assert "--workloads ag,rs,ag+rs" in line and "--symm-kernel mc" in line
        args = line.split()
        # Only the pause changes: NCCL's own CTA counts, on the multicast symmetric kernels.
        assert args[args.index("--cta-configs") + 1] == "default"
        assert args[args.index("--algo") + 1] == "symm"


@pytest.mark.unit
@needs_bash
@pytest.mark.parametrize(
    "group_size, pauses",
    [
        # 16 paths per GPU at NCCL's own AG CTA count (4 x 4, 8 x 2, 16 x 1): knee at about 360 cycles.
        ("4", "0,128,256,512,768,1024,1536,2048,2560,3072,4096,6144"),
        ("8", "0,128,256,512,768,1024,1536,2048,2560,3072,4096,6144"),
        ("16", "0,128,256,512,768,1024,1536,2048,2560,3072,4096,6144"),
        # 32 paths per GPU (32 x 1 CTA): knee at about 3400 cycles.
        ("32", "0,2048,3072,3584,4096,5120,6144,7168,8192,10240,12288,14336"),
        # 64 paths per GPU (64 x 1 CTA): knee at about 9500 cycles.
        ("64", "0,6144,8192,9728,11264,12288,14336,16384,18432,22528,26624,31744"),
    ],
)
def test_default_pauses_follow_the_paths_per_gpu(tmp_path, group_size, pauses):
    # The launcher may reject a group larger than the fake job; the default is picked before it runs.
    result = _sweep(_fake_lib(tmp_path), "--group-sizes", group_size, "--dry-run")
    assert f"pause cycles for the {group_size}-GPU group: {pauses}\n" in result.stderr


@pytest.mark.unit
@needs_bash
def test_cta_variables_do_not_reach_the_runs(tmp_path):
    # A copy of the sweep next to a fake launcher that prints the NCCL variables it inherits.
    sweep_dir = tmp_path / "repo" / "examples" / "gb200" / "nvls-collective"
    sweep_dir.mkdir(parents=True)
    shutil.copy(SCRIPT, sweep_dir / SCRIPT.name)
    (sweep_dir / "run_nvls_collective_benchmark.sh").write_text('env | grep "^NCCL_" | sort\n')
    common = tmp_path / "repo" / "scripts" / "scale-down"
    common.mkdir(parents=True)
    (common / "common").symlink_to(SCRIPT.parents[3] / "scripts" / "scale-down" / "common")
    # GB200 images export NCCL_MIN_CTAS=4; with the cta patch it would lift the multicast kernels' CTA counts.
    cta_env = {"NCCL_MIN_CTAS": "4", "NCCL_MIN_NCHANNELS": "16", "NCCL_SYM_CTAS": "2"}
    env = {**os.environ, **cta_env, "GLOO_SOCKET_IFNAME": "lo", "RUN_TIME": "sweep-test"}
    command = ["bash", str(sweep_dir / SCRIPT.name), "--venv", "none", "--nccl-lib", str(_fake_lib(tmp_path))]
    result = subprocess.run(
        [*command, "--pause-cycles", "0", "--dry-run"], capture_output=True, text=True, env=env, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "NCCL_SYM_AG_STMC_PAUSE_CYCLES=0" in result.stdout
    for name, value in cta_env.items():
        assert f"unsetting {name}={value}" in result.stderr
        assert f"{name}=" not in result.stdout


@pytest.mark.unit
@needs_bash
@pytest.mark.parametrize(
    "options, match",
    [
        (["--pause-cycles", "0,abc"], "comma-separated list"),
        (["--pause-cycles", "500,500"], "twice"),
        (["--pause-cycles", "4294967296"], "uint32"),
        (["--", "--cta-configs", "ag=2,rs=2"], "fixed by this sweep"),
        (["--", "--symm-kernel", "default"], "fixed by this sweep"),
        (["--", "--algos", "symm-p2p"], "fixed by this sweep"),
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
