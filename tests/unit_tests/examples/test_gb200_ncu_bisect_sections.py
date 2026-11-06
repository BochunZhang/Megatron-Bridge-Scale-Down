# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU coverage for bisect_ncu_sections.sh: launcher commands and per-section status."""

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
