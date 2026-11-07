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


def test_dry_run_builds_one_application_replay_launch_per_item(tmp_path: Path):
    result = _run(
        BISECT,
        "--dry-run",
        "--sections",
        "Nvlink,SourceCounters+InstructionStats",
        "--output-dir",
        str(tmp_path),
        "--",
        "--timeout",
        "900",
        "--",
        "--devices",
        "0",
    )
    assert result.returncode == 0, result.stderr
    commands = [shlex.split(line) for line in result.stdout.splitlines()]
    assert len(commands) == 2
    for item, sections, command in zip(
        ("Nvlink", "SourceCounters+InstructionStats"), (["Nvlink"], ["SourceCounters", "InstructionStats"]), commands
    ):
        assert command[1] == str(SCRIPT_DIR / "run_ncu_torch_all_gather.sh")
        assert command[command.index("--replay-mode") + 1] == "application"
        assert command[command.index("--output-dir") + 1] == str(tmp_path / item)
        assert "--skip-baseline" in command and "--metrics" not in command
        assert [command[i + 1] for i, arg in enumerate(command) if arg == "--section"] == sections
        # Launcher args come last: the launcher's last --timeout wins, ncu args stay after "--".
        assert command.index("900") > command.index("600")
        assert command[-3:] == ["--", "--devices", "0"]


def _section_file(directory: Path, identifier: str, sets: tuple[str, ...]) -> None:
    body = f'Identifier: "{identifier}"\nDisplayName: "{identifier}"\n'
    body += "".join(f'Sets {{\n  Identifier: "{name}"\n}}\n' for name in sets)
    body += 'Header {\n  Metrics {\n    Label: "Duration"\n    Name: "gpu__time_duration.sum"\n  }\n}\n'
    (directory / f"{identifier}.section").write_text(body)


def test_set_expands_to_its_sections_from_the_section_files(tmp_path: Path):
    section_dir = tmp_path / "sections"
    section_dir.mkdir()
    _section_file(section_dir, "SpeedOfLight", ("basic", "detailed", "full"))
    _section_file(section_dir, "SourceCounters", ("full",))
    _section_file(section_dir, "Nvlink", ("nvlink",))
    result = _run(
        BISECT, "--dry-run", "--set", "full", "--section-dir", str(section_dir), "--output-dir", str(tmp_path / "o")
    )
    assert result.returncode == 0, result.stderr
    commands = [shlex.split(line) for line in result.stdout.splitlines() if not line.startswith("set ")]
    assert [command[command.index("--section") + 1] for command in commands] == ["SourceCounters", "SpeedOfLight"]


@pytest.fixture
def fake_tree(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """A copy of the script next to a fake launcher, and a fake ncu on PATH."""
    script_dir = tmp_path / "repo/examples/gb200/ncu-torch-all-gather"
    script_dir.mkdir(parents=True)
    shutil.copy(BISECT, script_dir)
    (script_dir / "run_ncu_torch_all_gather.sh").write_text(
        textwrap.dedent("""\
        sections=()
        while [[ $# -gt 0 ]]; do
            [[ "$1" == --section ]] && sections+=("$2")
            shift
        done
        item="$(IFS=+; echo "${sections[*]}")"
        echo "$item" >>"$(dirname "$0")/calls.txt"
        for p in 1 2 3; do echo "==PROF== Profiling kernel: Application replay pass $p"; done
        case "$item" in
            Good|Good+Breaks2) echo "ok" ;;
            Hangs) echo "ERROR: ncu gave up waiting for the profiled kernel (hang in replay pass 3)" >&2; exit 1 ;;
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
        for s in Good Hangs Breaks Breaks2; do echo "$s $s no $s.section"; done
    """)
    )
    ncu.chmod(0o755)
    return script_dir, {**os.environ, "PATH": f"{tools}:{os.environ['PATH']}"}


def test_items_are_classified_with_their_pass_counts(fake_tree: tuple[Path, dict[str, str]], tmp_path: Path):
    script_dir, env = fake_tree
    out = tmp_path / "out"
    result = _run(
        script_dir / "bisect_ncu_sections.sh",
        "--sections",
        "Good,Hangs,Breaks,Good+Breaks2,Missing",
        "--output-dir",
        str(out),
        env=env,
    )
    assert result.returncode == 0, result.stderr
    rows = {line.split()[0]: line.split()[1:3] for line in (out / "summary.txt").read_text().splitlines()}
    assert rows == {
        "Good": ["ok", "3"],
        "Hangs": ["hang", "3"],
        "Breaks": ["fail", "3"],
        "Good+Breaks2": ["ok", "3"],
        "Missing": ["unknown", "-"],
    }
    assert "Failed to profile" in (out / "summary.txt").read_text()
    # Missing never reaches the launcher; a group is one launcher run with all its sections.
    assert (script_dir / "calls.txt").read_text().splitlines() == ["Good", "Hangs", "Breaks", "Good+Breaks2"]


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
        echo "$@" >"$FAKE_NCU_ARGS"
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
    return {
        **os.environ,
        "PATH": f"{tools}:{os.environ['PATH']}",
        "FAKE_NCU_IMPORTED": str(tmp_path / "imported"),
        "FAKE_NCU_ARGS": str(tmp_path / "ncu-args"),
    }


def _launch(out: Path, env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
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
        *args,
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


def test_launcher_sections_replace_the_set(fake_ncu: dict[str, str], tmp_path: Path):
    result = _launch(tmp_path / "out", {**fake_ncu, "FAKE_NCU_EXT": "ncu-repz"}, "--section", "A", "--section", "B")
    assert result.returncode == 0, result.stdout + result.stderr
    args = shlex.split(Path(fake_ncu["FAKE_NCU_ARGS"]).read_text())
    assert "--set" not in args
    assert [args[i + 1] for i, arg in enumerate(args) if arg == "--section"] == ["A", "B"]
    # The summary metric is collected next to the sections.
    assert args[args.index("--metrics") + 1] == "gpu__time_duration.sum"


def test_launcher_names_the_pass_ncu_gave_up_on(fake_ncu: dict[str, str], tmp_path: Path):
    tools = Path(fake_ncu["PATH"].split(":")[0])
    ncu = tools / "ncu"
    ncu.write_text(
        ncu.read_text().replace(
            'echo "$@" >"$FAKE_NCU_ARGS"',
            'echo "==PROF== Profiling k: Application replay pass 38"; echo "==PROF== Profiling k: Application replay pass 39"\n'
            'echo "==ERROR== Failed to wait for profiling"; exit 1',
        )
    )
    result = _launch(tmp_path / "out", fake_ncu)
    assert result.returncode != 0
    assert "ncu gave up waiting for the profiled kernel (hang in replay pass 39)" in result.stderr
