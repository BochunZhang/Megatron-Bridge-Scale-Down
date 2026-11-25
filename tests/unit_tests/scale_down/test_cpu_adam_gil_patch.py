from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit
_EXAMPLE = Path(__file__).resolve().parents[3] / "examples/scale-down/01-analyse/02-deepspeed-offload"
_SCRIPT = _EXAMPLE / "apply_cpu_adam_gil_patch.sh"
_PATCH_DIR = _EXAMPLE / "patches/v0.20.1"
_GIL_PATCH = _PATCH_DIR / "deepspeed-cpu-adam-release-gil.patch"
_OMP_PATCH = _PATCH_DIR / "deepspeed-cpu-adam-graceadam-omp.patch"


def _write_sources(ops_dir: Path) -> dict[Path, str]:
    """Build the unpatched fixture from the old lines and context in each hunk."""
    source_lines: dict[Path, list[str]] = {}
    for patch in (_GIL_PATCH, _OMP_PATCH):
        for diff in patch.read_text(encoding="utf-8").split("diff --git ")[1:]:
            old_path = next(line.removeprefix("--- a/") for line in diff.splitlines() if line.startswith("--- a/"))
            lines = source_lines.setdefault(ops_dir / old_path, [])
            for hunk in diff.split("\n@@ ")[1:]:
                header, body = hunk.split("\n", 1)
                old_start, _, old_count = header.split()[0].removeprefix("-").partition(",")
                first_line = int(old_start) - 1
                line_count = int(old_count or "1")
                end_line = first_line + line_count
                old_lines = [line[1:] for line in body.split("\n") if not line or line.startswith((" ", "-"))]
                lines.extend(["// fixture padding"] * (end_line - len(lines)))
                lines[first_line:end_line] = old_lines[:line_count]
    sources = {}
    for path, lines in source_lines.items():
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
        binding = (Path(__file__).parent / "csrc/adam/cpu_adam.cpp").read_text()
        op = SimpleNamespace(__file__="fixture_cpu_adam.so")
        if 'm.attr("adam_update_releases_gil") = true;' in binding:
            op.adam_update_releases_gil = True
        if 'm.attr("graceadam_omp_threads_patch") = true;' in binding and os.environ.get("TEST_STALE_BUILD") != "1":
            op.graceadam_omp_threads_patch = True
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


def _apply_legacy_gil_patch(env: dict[str, str]) -> None:
    subprocess.run(
        ["patch", "--force", "--silent", "--fuzz=0", "-p1"],
        cwd=Path(env["PYTHONPATH"]) / "deepspeed" / "ops",
        input=_GIL_PATCH.read_text(encoding="utf-8"),
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )


@pytest.mark.parametrize("legacy", [False, True], ids=["pristine", "gil-only"])
def test_apply_and_recover_cpu_adam_gil_patch(
    installed_deepspeed: tuple[dict[str, str], dict[Path, str]],
    legacy: bool,
) -> None:
    env, original_sources = installed_deepspeed
    if legacy:
        _apply_legacy_gil_patch(env)
    result = _run(env)
    assert result.returncode == 0, result.stderr
    patched_sources = {path: path.read_text(encoding="utf-8") for path in original_sources}
    assert all(contents != original_sources[path] for path, contents in patched_sources.items())
    assert any("adam_update_releases_gil" in contents for contents in patched_sources.values())
    assert any("graceadam_omp_threads_patch" in contents for contents in patched_sources.values())

    result = _run(env)
    assert result.returncode == 0, result.stderr
    assert {path: path.read_text(encoding="utf-8") for path in patched_sources} == patched_sources

    result = _run(env, "--recovery")
    assert result.returncode == 0, result.stderr
    assert {path: path.read_text(encoding="utf-8") for path in original_sources} == original_sources

    result = _run(env, "--recovery")
    assert result.returncode == 0, result.stderr


def test_recover_legacy_gil_only_patch(
    installed_deepspeed: tuple[dict[str, str], dict[Path, str]],
) -> None:
    env, original_sources = installed_deepspeed
    _apply_legacy_gil_patch(env)
    result = _run(env, "--recovery")
    assert result.returncode == 0, result.stderr
    assert {path: path.read_text(encoding="utf-8") for path in original_sources} == original_sources


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


@pytest.mark.parametrize("source_name", ["cpu_adam_impl.cpp", "sve.h"])
def test_mismatched_patch_leaves_sources_untouched(
    installed_deepspeed: tuple[dict[str, str], dict[Path, str]],
    source_name: str,
) -> None:
    env, sources = installed_deepspeed
    path = next(path for path in sources if path.name == source_name)
    sources[path] = "// incompatible CPUAdam source\n"
    path.write_text(sources[path], encoding="utf-8")

    result = _run(env)
    assert result.returncode != 0
    assert {path: path.read_text(encoding="utf-8") for path in sources} == sources


@pytest.mark.parametrize("legacy", [False, True], ids=["pristine", "gil-only"])
def test_failed_cpu_adam_build_restores_previous_sources(
    installed_deepspeed: tuple[dict[str, str], dict[Path, str]],
    legacy: bool,
) -> None:
    env, original_sources = installed_deepspeed
    if legacy:
        _apply_legacy_gil_patch(env)
    previous_sources = {path: path.read_text(encoding="utf-8") for path in original_sources}
    result = _run({**env, "TEST_BUILD_FAILURE": "1"})
    assert result.returncode != 0
    assert "Fixture CPUAdam build failed" in result.stderr
    assert {path: path.read_text(encoding="utf-8") for path in original_sources} == previous_sources


def test_missing_omp_marker_restores_original_sources(
    installed_deepspeed: tuple[dict[str, str], dict[Path, str]],
) -> None:
    env, original_sources = installed_deepspeed
    result = _run({**env, "TEST_STALE_BUILD": "1"})
    assert result.returncode != 0
    assert {path: path.read_text(encoding="utf-8") for path in original_sources} == original_sources
