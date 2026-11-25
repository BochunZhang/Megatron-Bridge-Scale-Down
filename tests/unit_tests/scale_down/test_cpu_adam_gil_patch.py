from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit
_EXAMPLE = Path(__file__).resolve().parents[3] / "examples/scale-down/01-analyse/02-deepspeed-offload"
_SCRIPT = _EXAMPLE / "apply_cpu_adam_gil_patch.sh"
_PATCH = _EXAMPLE / "patches/v0.20.1/deepspeed-cpu-adam-release-gil.patch"


def _write_sources(ops_dir: Path) -> dict[Path, str]:
    """Build the unpatched fixture from the old lines and context in each hunk."""
    sources = {}
    for diff in _PATCH.read_text(encoding="utf-8").split("diff --git ")[1:]:
        old_path = diff.splitlines()[1].removeprefix("--- a/")
        lines: list[str] = []
        for hunk in diff.split("\n@@ ")[1:]:
            header, body = hunk.split("\n", 1)
            old_start, old_count = map(int, header.split()[0].removeprefix("-").split(","))
            first_line = old_start - 1
            end_line = first_line + old_count
            old_lines = [line[1:] for line in body.split("\n") if not line.startswith("+")]
            lines.extend(["// fixture padding"] * (end_line - len(lines)))
            lines[first_line:end_line] = old_lines[:old_count]
        path = ops_dir / old_path
        path.parent.mkdir(parents=True, exist_ok=True)
        sources[path] = "\n".join(lines) + "\n"
        path.write_text(sources[path], encoding="utf-8")
    return sources


@pytest.fixture
def installed_deepspeed(tmp_path: Path) -> tuple[dict[str, str], dict[Path, str]]:
    site_packages = tmp_path / "site packages"
    package = site_packages / "deepspeed"
    ops_dir = package / "ops"
    sources = _write_sources(ops_dir)
    (package / "__init__.py").touch()
    (package / "git_version_info.py").write_text(
        'import os\ninstalled_ops = {"cpu_adam": os.environ.get("TEST_PREBUILT") == "1"}\naccelerator_name = "cuda"\n',
        encoding="utf-8",
    )
    (package / "accelerator.py").write_text(
        'from types import SimpleNamespace\ndef get_accelerator():\n    return SimpleNamespace(_name="cuda")\n',
        encoding="utf-8",
    )
    (ops_dir / "op_builder.py").write_text(
        """\
import os
from pathlib import Path
from types import SimpleNamespace

class CPUAdamBuilder:
    def load(self):
        if os.environ.get("TEST_BUILD_FAILURE") == "1":
            raise RuntimeError("Fixture CPUAdam build failed")
        binding = Path(__file__).parent / "csrc/adam/cpu_adam.cpp"
        op = SimpleNamespace(__file__="fixture_cpu_adam.so")
        if 'm.attr("adam_update_releases_gil") = true;' in binding.read_text():
            op.adam_update_releases_gil = True
        return op
""",
        encoding="utf-8",
    )
    env = {"PATH": os.environ["PATH"], "PYTHON_BIN": sys.executable, "PYTHONPATH": str(site_packages)}
    return env, sources


def _run(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(_SCRIPT), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


def test_apply_and_recover_cpu_adam_gil_patch(
    installed_deepspeed: tuple[dict[str, str], dict[Path, str]],
) -> None:
    env, original_sources = installed_deepspeed
    result = _run(env)
    assert result.returncode == 0, result.stderr
    patched_sources = {path: path.read_text(encoding="utf-8") for path in original_sources}
    assert all(contents != original_sources[path] for path, contents in patched_sources.items())
    assert any("adam_update_releases_gil" in contents for contents in patched_sources.values())

    result = _run(env)
    assert result.returncode == 0, result.stderr
    assert {path: path.read_text(encoding="utf-8") for path in patched_sources} == patched_sources

    result = _run(env, "--recovery")
    assert result.returncode == 0, result.stderr
    assert {path: path.read_text(encoding="utf-8") for path in original_sources} == original_sources

    result = _run(env, "--recovery")
    assert result.returncode == 0, result.stderr


def test_rejects_prebuilt_cpu_adam_without_changing_sources(
    installed_deepspeed: tuple[dict[str, str], dict[Path, str]],
) -> None:
    env, original_sources = installed_deepspeed
    result = _run({**env, "TEST_PREBUILT": "1"})
    assert result.returncode != 0
    assert (
        "prebuilt" in (result.stdout + result.stderr).lower()
        or "precompiled" in (result.stdout + result.stderr).lower()
    )
    assert {path: path.read_text(encoding="utf-8") for path in original_sources} == original_sources


def test_mismatched_patch_leaves_both_sources_untouched(
    installed_deepspeed: tuple[dict[str, str], dict[Path, str]],
) -> None:
    env, sources = installed_deepspeed
    for path in sources:
        if path.name == "cpu_adam_impl.cpp":
            sources[path] = "// incompatible CPUAdam source\n"
            path.write_text(sources[path], encoding="utf-8")

    result = _run(env)
    assert result.returncode != 0
    assert {path: path.read_text(encoding="utf-8") for path in sources} == sources


def test_failed_cpu_adam_build_restores_original_sources(
    installed_deepspeed: tuple[dict[str, str], dict[Path, str]],
) -> None:
    env, original_sources = installed_deepspeed
    result = _run({**env, "TEST_BUILD_FAILURE": "1"})
    assert result.returncode != 0
    assert "Fixture CPUAdam build failed" in result.stderr
    assert {path: path.read_text(encoding="utf-8") for path in original_sources} == original_sources
