# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU coverage for examples/gb200/ncu-torch-all-gather: section bisection and the report summary."""

import os
import shlex
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = ROOT / "examples/gb200/ncu-torch-all-gather"
BISECT = SCRIPT_DIR / "bisect_ncu_sections.sh"
pytestmark = pytest.mark.unit


def _run(script: Path, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(script), *args], capture_output=True, text=True, env=env or dict(os.environ), timeout=60
    )


def test_dry_run_builds_one_application_replay_launch_per_section(tmp_path: Path):
    result = _run(
        BISECT,
        "--dry-run",
        "--sections",
        "Nvlink,SourceCounters",
        "--output-dir",
        str(tmp_path),
        "--",
        "--timeout",
        "600",
    )
    assert result.returncode == 0, result.stderr
    commands = [shlex.split(line) for line in result.stdout.splitlines()]
    assert len(commands) == 2
    for section, command in zip(("Nvlink", "SourceCounters"), commands):
        assert command[1] == str(SCRIPT_DIR / "run_ncu_torch_all_gather.sh")
        assert command[command.index("--replay-mode") + 1] == "application"
        assert command[command.index("--metrics") + 1] == "gpu__time_duration.sum"
        assert command[command.index("--output-dir") + 1] == str(tmp_path / section)
        assert "--skip-baseline" in command
        # Launcher args come after the defaults, so the launcher's last --timeout wins.
        assert command.index("600") > command.index("300")
        assert command[-3:] == ["--", "--section", section]


def test_launcher_args_must_not_contain_a_separator():
    result = _run(BISECT, "--dry-run", "--", "--algo", "ring", "--", "--devices", "0")
    assert result.returncode != 0
    assert "must not contain" in result.stderr


@pytest.fixture
def fake_tree(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """A copy of the script next to a fake launcher, and a fake ncu on PATH."""
    script_dir = tmp_path / "repo/examples/gb200/ncu-torch-all-gather"
    script_dir.mkdir(parents=True)
    shutil.copy(BISECT, script_dir)
    (script_dir / "run_ncu_torch_all_gather.sh").write_text(
        textwrap.dedent("""\
        section="${@: -1}"
        echo "$@" >>"$(dirname "$0")/calls.txt"
        case "$section" in
            Good) echo "ok" ;;
            Hangs) echo "ERROR: ncu did not finish in 300s (hang); see ncu.log" >&2; exit 1 ;;
            *) echo "==ERROR== Failed to profile"; exit 1 ;;
        esac
    """)
    )
    tools = tmp_path / "bin"
    tools.mkdir()
    ncu = tools / "ncu"
    ncu.write_text(
        textwrap.dedent("""\
        #!/usr/bin/env bash
        [[ "$1" == --list-sections ]] || exit 2
        echo "Identifier Display-Name Enabled Filename"
        for s in Good Hangs Breaks; do echo "$s $s no $s.section"; done
    """)
    )
    ncu.chmod(0o755)
    return script_dir, {**os.environ, "PATH": f"{tools}:{os.environ['PATH']}"}


def test_sections_are_classified_and_unknown_ones_skipped(fake_tree: tuple[Path, dict[str, str]], tmp_path: Path):
    script_dir, env = fake_tree
    out = tmp_path / "out"
    result = _run(
        script_dir / "bisect_ncu_sections.sh",
        "--sections",
        "Good,Hangs,Breaks,Missing",
        "--output-dir",
        str(out),
        env=env,
    )
    assert result.returncode == 0, result.stderr
    status = {line.split()[0]: line.split()[1] for line in (out / "summary.txt").read_text().splitlines()}
    assert status == {"Good": "ok", "Hangs": "hang", "Breaks": "fail", "Missing": "unknown"}
    assert "Failed to profile" in (out / "summary.txt").read_text()
    # Missing never reaches the launcher.
    calls = (script_dir / "calls.txt").read_text().splitlines()
    assert [call.split()[-1] for call in calls] == ["Good", "Hangs", "Breaks"]


LAUNCHER = SCRIPT_DIR / "run_ncu_torch_all_gather.sh"


@pytest.fixture
def fake_ncu(tmp_path: Path) -> dict[str, str]:
    """A fake ncu 2026.4 that writes the report as $FAKE_NCU_EXT plus one NVLS TUNING log, and a fake nvidia-smi."""
    tools = tmp_path / "fake-bin"
    tools.mkdir()
    ncu = tools / "ncu"
    ncu.write_text(
        textwrap.dedent("""\
        #!/usr/bin/env bash
        if [[ "$1" == --version ]]; then echo "Version 2026.4.0.0 (build 1)"; exit 0; fi
        if [[ "$1" == --import ]]; then
            echo "$2" >"$FAKE_NCU_IMPORTED"
            printf '"ID","gpu__time_duration.sum"\\n"","usecond"\\n"0","10"\\n"1","11"\\n'
            exit 0
        fi
        while [[ $# -gt 0 ]]; do
            [[ "$1" == --export ]] && report="$2"
            shift
        done
        [[ -n "$FAKE_NCU_EXT" ]] && echo report >"${report}.${FAKE_NCU_EXT}"
        log="${NCCL_DEBUG_FILE//%h/host}"
        echo "host:1:1 NCCL INFO AllGather: 268435456 Bytes -> Algo NVLS proto Simple" >"${log//%p/1}"
    """)
    )
    smi = tools / "nvidia-smi"
    smi.write_text("#!/usr/bin/env bash\nfor g in 0 1 2 3; do echo GPU $g; done\n")
    for tool in (ncu, smi):
        tool.chmod(0o755)
    return {**os.environ, "PATH": f"{tools}:{os.environ['PATH']}", "FAKE_NCU_IMPORTED": str(tmp_path / "imported")}


def _launch(out: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return _run(
        LAUNCHER,
        "--skip-baseline",
        "--gpus",
        "0,1",
        "--warmup",
        "0",
        "--iters",
        "1",
        "--output-dir",
        str(out),
        env=env,
    )


@pytest.mark.parametrize("ext", ["ncu-repz", "ncu-rep"])
def test_launcher_summarizes_the_report_in_either_format(fake_ncu: dict[str, str], tmp_path: Path, ext: str):
    out = tmp_path / "out"
    result = _launch(out, {**fake_ncu, "FAKE_NCU_EXT": ext})
    assert result.returncode == 0, result.stdout + result.stderr
    assert Path(fake_ncu["FAKE_NCU_IMPORTED"]).read_text().strip() == str(out / f"all_gather.{ext}")
    assert "2 result(s)" in result.stdout
    assert (out / "summary.csv").exists()


def test_launcher_fails_when_ncu_writes_no_report(fake_ncu: dict[str, str], tmp_path: Path):
    result = _launch(tmp_path / "out", {**fake_ncu, "FAKE_NCU_EXT": ""})
    assert result.returncode != 0
    assert "wrote neither" in result.stderr
