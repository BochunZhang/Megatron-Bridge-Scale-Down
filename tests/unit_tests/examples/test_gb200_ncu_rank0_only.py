# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU regression tests for per-rank AG/RS NCU report discovery."""

import csv
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = ROOT / "examples/gb200/ncu-torch-all-gather/run_ncu_torch_ag_rs_rank0_only.sh"
pytestmark = pytest.mark.unit


def _write_tool(path: Path, shebang: str, body: str) -> None:
    path.write_text(shebang + "\n" + textwrap.dedent(body), encoding="utf-8")
    path.chmod(0o755)


@pytest.fixture
def fake_ncu(tmp_path: Path) -> dict[str, str]:
    """Produce distinct report contents per rank without starting any GPU workers."""
    tools = tmp_path / "bin"
    tools.mkdir()
    python = f"#!{sys.executable}"
    _write_tool(
        tools / "ncu",
        python,
        """\
        import csv
        import os
        import sys
        from pathlib import Path

        if sys.argv[1] == "--version":
            sys.stdout.write("Version 2026.4.0\\n")
        elif sys.argv[1] == "--import":
            report = Path(sys.argv[2])
            kernel = {
                "ag": "ncclSymkDevKernel_AllGather_STMC",
                "rs": "ncclSymkDevKernel_ReduceScatter_LDMC_sum_bf16",
            }[report.parent.name]
            rank = int(report.read_text())
            metrics = []
            for direction in ("nvlrx", "nvltx"):
                metrics.append(f"pmsampling:{direction}__bytes.sum")
                for packet in ("request", "response"):
                    for data in ("user", "protocol"):
                        metrics.append(f"pmsampling:{direction}__bytes_packet_{packet}_data_{data}.sum")
            writer = csv.writer(sys.stdout)
            writer.writerow(["Kernel Name", *metrics])
            writer.writerow(["", *["byte"] * len(metrics)])
            writer.writerow([kernel, *[rank + 1] * len(metrics)])
        else:
            rank = os.environ["RANK"]
            prefix = sys.argv[sys.argv.index("--export") + 1].replace("%q{RANK}", rank)
            scenario = os.environ["FAKE_NCU_SCENARIO"]
            suffix = "" if scenario == "unsuffixed" else ("1", "0", "2", "0")[int(rank)]
            extension = ".ncu-repz" if int(rank) % 2 == 0 else ".ncu-rep"
            Path(prefix + suffix + extension).write_text(rank)
            if scenario == "duplicate" and rank == "1":
                Path(prefix + "8.ncu-repz").write_text(rank)
        """,
    )
    _write_tool(
        tools / "uv",
        python,
        """\
        import os
        import sys
        from pathlib import Path

        args = sys.argv[4:]
        if args[0] != "-c":
            os.execv(sys.executable, [sys.executable, *args])
        if "ProcessGroupNCCL" in args[1]:
            sys.stdout.write("2.32.3")
        else:
            counter = Path(os.environ["FAKE_NCU_PORT_COUNTER"])
            port = int(counter.read_text()) + 1 if counter.exists() else 20000
            counter.write_text(str(port))
            sys.stdout.write(str(port))
        """,
    )
    _write_tool(tools / "timeout", "#!/usr/bin/env bash", 'shift 3\nexec "$@"\n')
    return {
        **os.environ,
        "PATH": f"{tools}{os.pathsep}{os.environ.get('PATH', os.defpath)}",
        "LD_PRELOAD": "",
        "FAKE_NCU_PORT_COUNTER": str(tmp_path / "port"),
        "FAKE_NCU_BIN": str(tools / "ncu"),
    }


@pytest.mark.parametrize("scenario", ["numbered", "unsuffixed", "duplicate"])
def test_launcher_resolves_reports_by_rank(tmp_path: Path, fake_ncu: dict[str, str], scenario: str) -> None:
    # The full-mode launcher uses the same report naming and exports without a real patched library.
    output = tmp_path / "out"
    result = subprocess.run(
        [
            "bash",
            str(LAUNCHER),
            "--full",
            "--gpus",
            "2,0,3,1",
            "--ncu",
            fake_ncu["FAKE_NCU_BIN"],
            "--output-dir",
            str(output),
        ],
        env={**fake_ncu, "FAKE_NCU_SCENARIO": scenario},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    summary = output / "nvlink_bytes.csv"
    if scenario == "duplicate":
        assert result.returncode != 0
        assert "rank 1" in result.stderr
        assert not summary.exists()
        return

    assert result.returncode == 0, result.stderr
    for mode in ("ag", "rs"):
        assert sorted(path.name for path in (output / mode).glob("rank*.csv")) == [
            f"rank{rank}.csv" for rank in range(4)
        ]
    with summary.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    actual = [(row["mode"], row["rank"], row["gpu"], row["nvlrx__bytes.sum"]) for row in records]
    expected = [(mode, str(rank), gpu, str(rank + 1)) for mode in ("ag", "rs") for rank, gpu in enumerate("2031")]
    assert actual == expected
