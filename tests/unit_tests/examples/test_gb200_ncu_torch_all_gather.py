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
        assert command[command.index("--algo") + 1] == "symm"
        assert command[command.index("--size-mib") + 1] == "1024"
        assert [command[i + 1] for i, arg in enumerate(command) if arg == "--section"] == sections
        # Launcher args come last: the launcher's last --timeout wins, ncu args stay after "--".
        assert command.index("900") > command.index("1800")
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
            [[ "$1" == --output-dir ]] && OUT="$(dirname "$2")"
            shift
        done
        item="$(IFS=+; echo "${sections[*]}")"
        [[ "$item" == Missing ]] || mkdir -p "$OUT/$item"
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
    # Every item that ran keeps its console log in its own folder.
    assert all((out / item / "launcher.log").is_file() for item in ("Good", "Hangs", "Breaks", "Good+Breaks2"))
    assert not list(out.glob("*.log"))
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
        if [[ "$1" == --version ]]; then echo "Version ${FAKE_NCU_VERSION:-2026.4}.0.0 (build 1)"; exit 0; fi
        if [[ "$1" == --import ]]; then
            echo "$2" >"$FAKE_NCU_IMPORTED"
            echo "$@" >"${FAKE_NCU_IMPORTED}.args"
            if [[ -n "${FAKE_NCU_RAW:-}" ]]; then cat "$FAKE_NCU_RAW"; exit 0; fi
            printf '"ID","gpu__time_duration.sum"\\n"","usecond"\\n"0","10"\\n"1","11"\\n'
            exit 0
        fi
        echo "$@" >"${FAKE_NCU_ARGS}${RANK:+.rank${RANK}}"
        [[ -n "${RANK:-}" ]] && echo "${RANK} ${LOCAL_RANK} ${WORLD_SIZE} ${MASTER_ADDR} ${MASTER_PORT}" >"${FAKE_NCU_ARGS}.env${RANK}"
        while [[ $# -gt 0 ]]; do
            [[ "$1" == --export ]] && report="$(sed "s/%q{RANK}/${RANK:-}/" <<<"$2")"
            shift
        done
        [[ -n "$FAKE_NCU_EXT" ]] && echo report >"${report}.${FAKE_NCU_EXT}"
        log="${NCCL_DEBUG_FILE//%h/host}"
        IFS='|' read -r -a tuning <<<"${FAKE_TUNING:-AllGather: 268435456 Bytes -> Algo NVLS proto Simple}"
        for t in "${tuning[@]}"; do echo "host:1:1 NCCL INFO $t"; done >"${log//%p/1}"
    """)
    )
    uv = tools / "uv"
    uv.write_text('#!/usr/bin/env bash\necho "${FAKE_NCCL_RUNTIME:-2.29.7}"\n')
    smi = tools / "nvidia-smi"
    smi.write_text("#!/usr/bin/env bash\nfor g in 0 1 2 3; do echo GPU $g; done\n")
    for tool in (ncu, uv, smi):
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
            'echo "$@" >"${FAKE_NCU_ARGS}${RANK:+.rank${RANK}}"',
            'echo "==PROF== Profiling k: Application replay pass 38"; echo "==PROF== Profiling k: Application replay pass 39"\n'
            'echo "==ERROR== Failed to wait for profiling"; exit 1',
        )
    )
    result = _launch(tmp_path / "out", fake_ncu)
    assert result.returncode != 0
    assert "ncu gave up waiting for the profiled kernel (hang in replay pass 39)" in result.stderr


@pytest.mark.parametrize(
    ("algo", "runtime", "sym_kernel", "nvls"),
    [
        ("symm", "2.29.7", "AllGather_STMC", "1"),
        ("p2p", "2.29.7", "AllGather_ST", "0"),
        ("symm", "2.32.3", "AllGather:AllGather_STMC", "1"),
        ("p2p", "2.32.3", "AllGather:AllGather_ST", "0"),
    ],
)
def test_symmetric_algorithms_pin_the_kernel_for_the_runtime_nccl(
    fake_ncu: dict[str, str], tmp_path: Path, algo: str, runtime: str, sym_kernel: str, nvls: str
):
    kernel = sym_kernel.split(":")[-1]
    env = {
        **fake_ncu,
        "FAKE_NCU_EXT": "ncu-repz",
        "FAKE_NCCL_RUNTIME": runtime,
        "FAKE_TUNING": f"AllGather [Symmetric]: 1073741824 Bytes -> Kernel {kernel} nchannels 32",
    }
    out = tmp_path / "out"
    result = _launch(out, env, "--algo", algo)
    assert result.returncode == 0, result.stdout + result.stderr
    config = (out / "config.txt").read_text().splitlines()
    assert f"NCCL_SYM_KERNEL={sym_kernel}" in config
    assert f"NCCL_NVLS_ENABLE={nvls}" in config
    assert "NCCL_WIN_ENABLE=1" in config and "NCCL_SYM_CE_THRESHOLD=9223372036854775807" in config
    assert not any(line.startswith("NCCL_ALGO=") for line in config)


def test_p2p_rejects_a_multicast_kernel(fake_ncu: dict[str, str], tmp_path: Path):
    env = {
        **fake_ncu,
        "FAKE_NCU_EXT": "ncu-repz",
        "FAKE_TUNING": "AllGather [Symmetric]: 1073741824 Bytes -> Kernel AllGather_STMC nchannels 32",
    }
    result = _launch(tmp_path / "out", env, "--algo", "p2p")
    assert result.returncode != 0
    assert "NCCL did not run --algo p2p" in result.stderr


def test_launcher_refuses_ncu_older_than_2026_4(fake_ncu: dict[str, str], tmp_path: Path):
    result = _launch(tmp_path / "out", {**fake_ncu, "FAKE_NCU_VERSION": "2026.3"})
    assert result.returncode != 0
    assert "need >= 2026.4" in result.stderr


def test_launcher_prefers_the_repo_nsight_compute_2026_4(fake_ncu: dict[str, str], tmp_path: Path):
    repo = tmp_path / "repo"
    script_dir = repo / "examples/gb200/ncu-torch-all-gather"
    script_dir.mkdir(parents=True)
    shutil.copy(LAUNCHER, script_dir)
    # Only the repo copy of ncu exists; the fake on PATH is renamed away.
    tools = Path(fake_ncu["PATH"].split(":")[0])
    ncu_home = repo / "NVIDIA-Nsight-Compute-2026.4"
    ncu_home.mkdir()
    (tools / "ncu").rename(ncu_home / "ncu")
    out = tmp_path / "out"
    result = _run(
        script_dir / "run_ncu_torch_all_gather.sh",
        "--skip-baseline",
        "--gpus",
        "0,1",
        "--warmup",
        "0",
        "--output-dir",
        str(out),
        env={**fake_ncu, "FAKE_NCU_EXT": "ncu-repz"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"ncu: {ncu_home / 'ncu'} (2026.4)" in result.stdout


AG_STMC = "AllGather [Symmetric]: 268435456 Bytes -> Kernel AllGather_STMC nchannels 6"
RS_LDMC = "ReduceScatter [Symmetric]: 268435456 Bytes -> Kernel ReduceScatter_LDMC nchannels 6"


def _overlap_env(fake_ncu: dict[str, str], runtime: str, lines: list[str]) -> dict[str, str]:
    return {**fake_ncu, "FAKE_NCU_EXT": "ncu-repz", "FAKE_NCCL_RUNTIME": runtime, "FAKE_TUNING": "|".join(lines)}


def _ncu_args(fake_ncu: dict[str, str]) -> list[str]:
    return shlex.split(Path(fake_ncu["FAKE_NCU_ARGS"]).read_text())


@pytest.mark.parametrize(
    ("runtime", "sym_kernel"),
    [("2.29.7", None), ("2.32.3", "AllGather:AllGather_STMC;ReduceScatter:ReduceScatter_LDMC")],
)
def test_ag_rs_app_range_runs_the_overlap_worker_and_pins_what_nccl_allows(
    fake_ncu: dict[str, str], tmp_path: Path, runtime: str, sym_kernel: str | None
):
    out = tmp_path / "out"
    # --warmup 1 --iters 1: two launches of each op per run.
    env = _overlap_env(fake_ncu, runtime, [AG_STMC, AG_STMC, RS_LDMC, RS_LDMC])
    result = _launch(out, env, "--algo", "symm", "--ag-rs", "--warmup", "1", "--replay-mode", "app-range")
    assert result.returncode == 0, result.stdout + result.stderr
    config = (out / "config.txt").read_text().splitlines()
    pinned = [line for line in config if line.startswith("NCCL_SYM_KERNEL=")]
    assert pinned == ([f"NCCL_SYM_KERNEL={sym_kernel}"] if sym_kernel else [])
    args = _ncu_args(fake_ncu)
    assert args[args.index("--replay-mode") + 1] == "app-range"
    assert args[args.index("--nvtx-include") + 1] == "ncu_collectives/"
    assert args[args.index("--launch-count") + 1] == "1" and "--kernel-name" not in args
    worker = args.index(str(SCRIPT_DIR / "ncu_torch_ag_rs_overlap.py"))
    assert args[worker + 1 : worker + 3] == ["--algo", "symm"]
    assert (out / "ag_rs.ncu-repz").is_file()
    assert "2/2 AllGather launches" in result.stdout and "2/2 ReduceScatter launches" in result.stdout


def test_ag_rs_kernel_replay_selects_both_kernels(fake_ncu: dict[str, str], tmp_path: Path):
    env = _overlap_env(fake_ncu, "2.32.3", [AG_STMC, AG_STMC, RS_LDMC, RS_LDMC])
    result = _launch(tmp_path / "out", env, "--algo", "symm", "--ag-rs", "--warmup", "1")
    assert result.returncode == 0, result.stdout + result.stderr
    args = _ncu_args(fake_ncu)
    assert args[args.index("--kernel-name") + 1] == "regex:^nccl(Symk)?DevKernel_(AllGather|ReduceScatter)"
    assert args[args.index("--launch-count") + 1] == "2"


@pytest.mark.parametrize(
    ("algo", "runtime", "sym_kernel", "lines"),
    [
        (
            "p2p",
            "2.32.3",
            "AllGather:AllGather_ST;ReduceScatter:ReduceScatter_LD",
            [AG_STMC.replace("STMC", "ST")] * 2 + [RS_LDMC.replace("LDMC", "LD")] * 2,
        ),
        (
            "nvls",
            "2.29.7",
            None,
            ["AllGather: 268435456 Bytes -> Algo NVLS proto Simple"] * 2
            + ["ReduceScatter: 268435456 Bytes -> Algo NVLS proto Simple"] * 2,
        ),
    ],
)
def test_reduce_scatter_setups(
    fake_ncu: dict[str, str],
    tmp_path: Path,
    algo: str,
    runtime: str,
    sym_kernel: str | None,
    lines: list[str],
):
    out = tmp_path / "out"
    result = _launch(out, _overlap_env(fake_ncu, runtime, lines), "--algo", algo, "--ag-rs", "--warmup", "1")
    assert result.returncode == 0, result.stdout + result.stderr
    config = (out / "config.txt").read_text().splitlines()
    assert [line for line in config if line.startswith("NCCL_SYM_KERNEL=")] == (
        [f"NCCL_SYM_KERNEL={sym_kernel}"] if sym_kernel else []
    )


def test_ag_rs_fails_when_nccl_picks_another_reduce_scatter_kernel(fake_ncu: dict[str, str], tmp_path: Path):
    # NCCL 2.29 leaves NCCL_SYM_KERNEL unset for ag+rs, so its own choice must be checked.
    rs_ld = RS_LDMC.replace("LDMC", "LD")
    env = _overlap_env(fake_ncu, "2.29.7", [AG_STMC, AG_STMC, rs_ld, rs_ld])
    result = _launch(tmp_path / "out", env, "--algo", "symm", "--ag-rs", "--warmup", "1")
    assert result.returncode != 0
    assert "NCCL did not run --algo symm for every ReduceScatter" in result.stderr


def test_ag_rs_needs_a_warmup_launch(fake_ncu: dict[str, str], tmp_path: Path):
    result = _launch(tmp_path / "out", fake_ncu, "--ag-rs")
    assert result.returncode != 0
    assert "--ag-rs needs --warmup >= 1" in result.stderr


def test_range_replay_profiles_the_nvtx_range(fake_ncu: dict[str, str], tmp_path: Path):
    env = _overlap_env(fake_ncu, "2.29.7", [AG_STMC, AG_STMC, RS_LDMC, RS_LDMC])
    result = _launch(tmp_path / "out", env, "--algo", "symm", "--ag-rs", "--warmup", "1", "--replay-mode", "range")
    assert result.returncode == 0, result.stdout + result.stderr
    args = _ncu_args(fake_ncu)
    assert args[args.index("--replay-mode") + 1] == "range"
    assert args[args.index("--nvtx-include") + 1] == "ncu_collectives/"
    assert args[args.index("--launch-count") + 1] == "1" and "--kernel-name" not in args
    assert args[args.index("--communicator") + 1] == "shmem"


def test_launcher_names_a_range_ncu_could_not_profile(fake_ncu: dict[str, str], tmp_path: Path):
    tools = Path(fake_ncu["PATH"].split(":")[0])
    ncu = tools / "ncu"
    ncu.write_text(
        ncu.read_text().replace(
            'echo "$@" >"${FAKE_NCU_ARGS}${RANK:+.rank${RANK}}"',
            "echo '==ERROR== Failed to profile \"range\" in client process 7'; exit 9",
        )
    )
    env = _overlap_env(fake_ncu, "2.29.7", [])
    result = _launch(tmp_path / "out", env, "--algo", "symm", "--ag-rs", "--warmup", "1", "--replay-mode", "app-range")
    assert result.returncode != 0
    assert "ncu could not profile the NVTX range (communicator=shmem, replay=app-range)" in result.stderr


@pytest.mark.parametrize("lockstep", [True, False])
def test_tcp_runs_one_ncu_per_rank_and_summarizes_every_report(
    fake_ncu: dict[str, str], tmp_path: Path, lockstep: bool
):
    out = tmp_path / "out"
    tools = Path(fake_ncu["PATH"].split(":")[0])
    ncu = tools / "ncu"
    ncu.write_text(
        ncu.read_text().replace('echo "$@" >"${FAKE_NCU_ARGS}', 'echo "==PROF== fake"; echo "$@" >"${FAKE_NCU_ARGS}')
    )
    env = _overlap_env(fake_ncu, "2.29.7", [AG_STMC, AG_STMC, RS_LDMC, RS_LDMC])
    extra = () if lockstep else ("--no-lockstep",)
    result = _launch(
        out,
        env,
        "--algo",
        "symm",
        "--ag-rs",
        "--warmup",
        "1",
        "--replay-mode",
        "app-range",
        "--communicator",
        "tcp",
        *extra,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    ranks = [shlex.split(Path(f"{fake_ncu['FAKE_NCU_ARGS']}.rank{r}").read_text()) for r in (0, 1)]
    assert ranks[0] == ranks[1]
    args = ranks[0]
    assert args[args.index("--communicator") + 1] == "tcp"
    assert args[args.index("--communicator-tcp-num-peers") + 1] == "2"
    assert int(args[args.index("--communicator-tcp-port") + 1]) > 0
    assert args[args.index("--export") + 1] == str(out / "ag_rs-rank%q{RANK}")
    if lockstep:
        assert "--lockstep-kernel-launch" in args
        assert args[args.index("--lockstep-nvtx-include") + 1] == "ncu_collectives/"
    else:
        assert "--lockstep-kernel-launch" not in args and "--lockstep-nvtx-include" not in args
    # No torchrun: each ncu wraps one worker, with the variables torchrun would have set.
    assert "torch.distributed.run" not in args
    worker = args.index(str(SCRIPT_DIR / "ncu_torch_ag_rs_overlap.py"))
    assert args[worker - 4 : worker] == ["uv", "run", "--no-sync", "python"]
    envs = [Path(f"{fake_ncu['FAKE_NCU_ARGS']}.env{r}").read_text().split() for r in (0, 1)]
    assert [e[:3] for e in envs] == [["0", "0", "2"], ["1", "1", "2"]]
    assert envs[0][3:] == envs[1][3:] and envs[0][3] == "127.0.0.1"
    # Both reports are summarized, each with the fake's two rows, under one header.
    assert (out / "ag_rs-rank0.ncu-repz").is_file() and (out / "ag_rs-rank1.ncu-repz").is_file()
    assert "4 result(s)" in result.stdout
    summary = (out / "summary.csv").read_text().splitlines()
    assert len(summary) == 2 + 4 and summary[0].startswith('"ID"')
    log = (out / "ncu.log").read_text().splitlines()
    assert sorted(log) == ["[rank 0] ==PROF== fake", "[rank 1] ==PROF== fake"]


def test_tcp_fails_when_a_rank_wrote_no_report(fake_ncu: dict[str, str], tmp_path: Path):
    tools = Path(fake_ncu["PATH"].split(":")[0])
    ncu = tools / "ncu"
    ncu.write_text(ncu.read_text().replace('[[ -n "$FAKE_NCU_EXT" ]]', '[[ -n "$FAKE_NCU_EXT" && "$RANK" == 0 ]]'))
    out = tmp_path / "out"
    result = _launch(out, {**fake_ncu, "FAKE_NCU_EXT": "ncu-repz"}, "--communicator", "tcp")
    assert result.returncode != 0
    assert f"wrote neither {out / 'all_gather-rank1'}.ncu-rep" in result.stderr


def test_launcher_rejects_an_unknown_communicator(fake_ncu: dict[str, str], tmp_path: Path):
    result = _launch(tmp_path / "out", fake_ncu, "--communicator", "mpi")
    assert result.returncode != 0
    assert "--communicator must be shmem, tcp or none" in result.stderr


SWEEP = SCRIPT_DIR / "sweep_ncu_replay_modes.sh"


def test_sweep_dry_run_builds_one_launch_per_combination(tmp_path: Path):
    result = _run(
        SWEEP,
        "--dry-run",
        "--combos",
        "range:shmem,app-range:tcp:nolockstep",
        "--output-dir",
        str(tmp_path),
        "--",
        "--warmup",
        "2",
    )
    assert result.returncode == 0, result.stderr
    commands = [shlex.split(line) for line in result.stdout.splitlines()]
    assert len(commands) == 2
    for (replay, comm, lockstep, folder), command in zip(
        (("range", "shmem", True, "range_shmem"), ("app-range", "tcp", False, "app-range_tcp_nolockstep")), commands
    ):
        assert command[1] == str(LAUNCHER)
        assert command[command.index("--replay-mode") + 1] == replay
        assert command[command.index("--communicator") + 1] == comm
        assert ("--no-lockstep" not in command) == lockstep
        assert command[command.index("--output-dir") + 1] == str(tmp_path / folder)
        assert "--ag-rs" in command and "--skip-baseline" in command
        assert command[command.index("--algo") + 1] == "symm"
        assert command[command.index("--section") + 1] == "SpeedOfLight"
        assert command[command.index("--timeout") + 1] == "900"
        assert command[-2:] == ["--warmup", "2"]


def test_sweep_default_starts_with_the_application_replay_control():
    result = _run(SWEEP, "--dry-run", "--no-ag-rs")
    assert result.returncode == 0, result.stderr
    commands = [shlex.split(line) for line in result.stdout.splitlines()]
    pairs = [(c[c.index("--replay-mode") + 1], c[c.index("--communicator") + 1]) for c in commands]
    assert pairs[0] == ("application", "shmem")
    assert ("app-range", "tcp") in pairs and ("range", "tcp") in pairs and ("app-range", "none") in pairs
    assert all("--ag-rs" not in c for c in commands)


@pytest.mark.parametrize("combo", ["app-range", "range:mpi", "app-range:shmem:nolockstep", "graph:tcp"])
def test_sweep_rejects_bad_combinations(combo: str):
    result = _run(SWEEP, "--dry-run", "--combos", combo)
    assert result.returncode != 0
    assert combo in result.stderr


def test_bisect_passes_ag_rs_and_replay_mode_and_names_the_output_after_them(tmp_path: Path):
    result = _run(
        BISECT,
        "--dry-run",
        "--sections",
        "PmSampling",
        "--ag-rs",
        "--replay-mode",
        "app-range",
    )
    assert result.returncode == 0, result.stderr
    (command,) = [shlex.split(line) for line in result.stdout.splitlines()]
    assert "--ag-rs" in command
    assert command[command.index("--replay-mode") + 1] == "app-range"
    out = Path(command[command.index("--output-dir") + 1])
    assert out.parent.name.startswith("symm-all-gather-reduce-scatter-") and out.name == "PmSampling"


ANALYSE = SCRIPT_DIR / "analyse_blocking_metrics.py"
GIB = 2**30

# What decides the bottleneck and is in no section of --set full (checked against the #13
# reports), plus the NVLink totals that only the PM-sampled .sum aggregates give correctly.
BLOCKING_KEY_METRICS = (
    # sender SM -> XBAR
    "l1tex__m_l1tex2xbar_req_cycles_stalled.sum",
    "l1tex__m_l1tex2xbar_req_cycles_active.sum",
    "smsp__warps_active.sum",
    "smsp__warps_issue_stalled_lg_throttle.sum",
    "smsp__warps_issue_stalled_long_scoreboard.sum",
    "smsp__warps_issue_stalled_membar.sum",
    "smsp__warps_issue_stalled_barrier.sum",
    "sm__average_inst_latency_pipe_lsu_ops_generic_cmd_read.ratio",
    "sm__average_inst_latency_pipe_lsu_mem_lg_cmd_read.ratio",
    "lts__lts2xbar_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "syslts__t_sectors_srcnode_gpc_aperture_peer_op_write.sum",
    "syslts__t_sectors_srcnode_gpc_aperture_peer_op_read.sum",
    # NVLink
    "nvlrx__cycles_active.avg.pct_of_peak_sustained_elapsed",
    "nvltx__cycles_active.avg.pct_of_peak_sustained_elapsed",
    "pmsampling:nvlrx__bytes_packet_request_data_user.sum",
    "pmsampling:nvlrx__bytes_packet_response_data_user.sum",
    "pmsampling:nvltx__bytes_packet_request_data_user.sum",
    "pmsampling:nvltx__bytes_packet_response_data_user.sum",
    "pmsampling:nvlrx__bytes.sum",
    "pmsampling:nvltx__bytes.sum",
    # receiver L2 and HBM
    "lts__t_sectors_op_write.sum",
    "lts__t_sectors_srcnode_gpc_op_write.sum",
    "lts__ltcfabric2lts_cycles_stalled.sum",
    "lts__ltcfabric2lts_cycles_active.sum",
    "lts__t_sectors_srcunit_ltcfabric_op_write.sum",
    "dram__cycles_active_read.avg.pct_of_peak_sustained_elapsed",
    "dram__cycles_active_write.avg.pct_of_peak_sustained_elapsed",
    # timelines of the sender pressure
    "pmsampling:smsp__warps_issue_stalled_lg_throttle.avg.pct_of_peak_sustained_elapsed",
    "pmsampling:l1tex__m_l1tex2xbar_req_cycles_active.avg.pct_of_peak_sustained_elapsed",
    # GB200: clocks under --clock-control none, the two-die LTC fabric, C2C / sysmem to Grace,
    # the L2 read coalescer and the synchronization / tail warp stalls
    "sm__cycles_elapsed.avg.per_second",
    "lts__cycles_elapsed.avg.per_second",
    "dram__cycles_elapsed.avg.per_second",
    "nvlrx__cycles_elapsed.avg.per_second",
    "pmsampling:sm__cycles_elapsed.avg.per_second",
    "lts__t_sectors_srcunit_ltcfabric_op_read.sum",
    "pmsampling:ctc__tx_bytes_data_user.sum",
    "pmsampling:ctc__rx_bytes_data_user.sum",
    "syslts__t_sectors_aperture_sysmem.sum",
    "lrc__lrc2xbar_cycles_stalled.sum",
    "lrc__lrc2xbar_cycles_active.sum",
    "smsp__warps_issue_stalled_wait.sum",
    "smsp__warps_issue_stalled_drain.sum",
)

# One STMC result of a 1 GiB all-gather on 4 GPUs (S = 256 MiB), shaped like #13: rx 4S, tx S.
STMC_ROW = {
    "ID": "0",
    "Process ID": "4242",
    "sm__cycles_elapsed.avg.per_second": "1965000000",
    "pmsampling:ctc__tx_bytes_data_user.sum": str(45 * 2**20),
    "gpu__time_duration.sum": "1766000",
    "l1tex__m_l1tex2xbar_req_cycles_stalled.sum": "300",
    "l1tex__m_l1tex2xbar_req_cycles_active.sum": "1000",
    "smsp__warps_active.sum": "1000",
    "smsp__warps_issue_stalled_lg_throttle.sum": "300",
    "pmsampling:nvlrx__bytes_packet_request_data_user.sum": str(GIB),
    "pmsampling:nvltx__bytes_packet_request_data_user.sum": str(GIB // 4),
    "pmsampling:nvlrx__bytes.sum": str(int(1.234 * GIB)),
    "lts__t_sectors_op_write.sum": str((GIB + GIB // 4) // 32),
    "lts__t_sectors_srcnode_gpc_op_write.sum": str(GIB // 4 // 32),
    "dram__sectors_write.sum": str(GIB // 32),
}


def _raw_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    header = sorted({key for row in rows for key in row})
    with path.open("w") as f:
        f.write(",".join(f'"{h}"' for h in header) + "\n")
        f.write(",".join('""' for _ in header) + "\n")
        for row in rows:
            f.write(",".join(f'"{row.get(h, "")}"' for h in header) + "\n")
    return path


def _blocking_line(text: str, label: str) -> list[str]:
    (line,) = [line for line in text.splitlines() if line.strip().startswith(label)]
    return line[len(line) - len(line.lstrip()) + len(label) :].split()


def test_blocking_report_counts_bytes_in_shards_and_keeps_the_nvlink_cross_check_separate(tmp_path: Path):
    summary = _raw_csv(tmp_path / "summary.csv", [STMC_ROW])
    result = subprocess.run(
        ["python3", str(ANALYSE), str(summary), "--size-mib", "1024", "--nranks", "4"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    text = (tmp_path / "blocking.txt").read_text()
    assert "256 MiB per shard" in text
    assert _blocking_line(text, "duration [us]") == ["1766.0"]
    assert _blocking_line(text, "SM") == ["1965"]
    assert _blocking_line(text, "C2C tx user [S]") == ["0.176"]
    assert _blocking_line(text, "l1tex2xbar stalled / active") == ["0.300"]
    assert _blocking_line(text, "warps stalled lg_throttle") == ["30.0%"]
    assert _blocking_line(text, "rx request user [S]") == ["4.000"]
    assert _blocking_line(text, "tx request user [S]") == ["1.000"]
    assert _blocking_line(text, "L2 writes not from local SMs [S]") == ["4.000"]
    assert _blocking_line(text, "dram write [S]") == ["4.000"]
    # Both NVLink blocks have an "rx wire [S]" line: PM sampling first, then the profiler
    # counter, which must not borrow the PM-sampled value.
    rx_wire = [line.split()[-1] for line in text.splitlines() if line.strip().startswith("rx wire [S]")]
    assert rx_wire == ["4.936", "n/a"]


def test_launcher_blocking_metrics_adds_both_collections_and_writes_the_report(
    fake_ncu: dict[str, str], tmp_path: Path
):
    out = tmp_path / "out"
    raw = _raw_csv(tmp_path / "raw.csv", [STMC_ROW, {**STMC_ROW, "ID": "1", "Process ID": "4243"}])
    env = {
        **fake_ncu,
        "FAKE_NCU_EXT": "ncu-repz",
        "FAKE_NCU_RAW": str(raw),
        "FAKE_TUNING": "AllGather [Symmetric]: 1073741824 Bytes -> Kernel AllGather_STMC nchannels 6",
    }
    # Four ranks (the later --gpus wins over _launch's two): S = 1024 MiB / 4.
    result = _launch(
        out,
        env,
        "--gpus",
        "0,1,2,3",
        "--algo",
        "symm",
        "--size-mib",
        "1024",
        "--replay-mode",
        "application",
        "--blocking-metrics",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    args = _ncu_args(fake_ncu)
    assert "--set" not in args
    collected = args[args.index("--metrics") + 1].split(",")
    assert set(BLOCKING_KEY_METRICS) <= set(collected), set(BLOCKING_KEY_METRICS) - set(collected)
    assert len(collected) == len(set(collected))
    assert args[args.index("--clock-control") + 1] == "none"
    # The summary is the whole raw page, so the PM-sampled aggregates are not filtered away.
    assert "--metrics" not in Path(fake_ncu["FAKE_NCU_IMPORTED"] + ".args").read_text().split()
    text = (out / "blocking.txt").read_text()
    assert "#0 pid 4242" in text and "#1 pid 4243" in text
    assert _blocking_line(text, "rx request user [S]") == ["4.000", "4.000"]
    assert "blocking_metrics=true" in (out / "config.txt").read_text()


def test_launcher_blocking_metrics_keeps_a_clock_control_given_after_the_separator(
    fake_ncu: dict[str, str], tmp_path: Path
):
    env = {**fake_ncu, "FAKE_NCU_EXT": "ncu-repz", "FAKE_NCU_RAW": str(_raw_csv(tmp_path / "raw.csv", [STMC_ROW]))}
    result = _launch(
        tmp_path / "out", env, "--blocking-metrics", "--replay-mode", "application", "--", "--clock-control", "base"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    args = _ncu_args(fake_ncu)
    assert [args[i + 1] for i, arg in enumerate(args) if arg == "--clock-control"] == ["base"]


def test_launcher_blocking_metrics_refuses_kernel_replay_with_multicast(fake_ncu: dict[str, str], tmp_path: Path):
    result = _launch(tmp_path / "out", fake_ncu, "--algo", "symm", "--blocking-metrics")
    assert result.returncode != 0
    assert "use --replay-mode application or app-range" in result.stderr
