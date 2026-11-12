# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Move legacy FSDP3 study results into the numbered experiment directories."""

import argparse
import logging
import os
import shutil
import subprocess
from pathlib import Path


logger = logging.getLogger(__name__)
STUDY_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = STUDY_DIR.parents[3] / "results/01-analyse/04-fsdp3-comm"


def _migration_plan(output_dir: Path) -> dict[Path, list[Path]]:
    plan: dict[Path, list[Path]] = {}
    for script in sorted(STUDY_DIR.glob("[0-9][0-9]-*/[0-9][0-9]_*.sh")):
        result = subprocess.run(
            ["bash", str(script), "--output-dir", str(output_dir), "--print-case-dirs"],
            check=True,
            capture_output=True,
            text=True,
        )
        group = script.parent.name
        for path in result.stdout.splitlines():
            target = Path(path)
            old_name = f"{target.name}-baseline" if group == "01-baseline" else target.name
            plan.setdefault(output_dir / group / old_name, []).append(target)
    return plan


def _move_case(source: Path, targets: list[Path]) -> None:
    claimed = []
    try:
        for target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            # The numbered launchers also claim a case with mkdir before training.
            target.mkdir()
            claimed.append(target)

        for target in targets[1:]:
            shutil.copytree(source, target, symlinks=True, dirs_exist_ok=True)
        source.rename(targets[0])
    except (OSError, KeyboardInterrupt):
        if source.is_dir():
            for target in reversed(claimed):
                shutil.rmtree(target)
        raise


def main() -> int:
    """Migrate result trees that match the numbered experiment matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("OUTPUT_DIR") or DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dry-run", action="store_true", help="Show moves and copies without changing files.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        plan = _migration_plan(args.output_dir.resolve())
    except subprocess.CalledProcessError as error:
        logger.error("Could not resolve experiment directories: %s\n%s", error, error.stderr.strip())
        return 1

    failed = False
    matched = False
    for source, targets in plan.items():
        if source.is_symlink():
            logger.warning("Skipping legacy symlink: %s", source)
            failed = True
            continue
        if not source.is_dir():
            continue
        matched = True

        conflicts = [target for target in targets if target.exists() or target.is_symlink()]
        if conflicts:
            logger.error("Destination already exists for %s: %s", source, ", ".join(map(str, conflicts)))
            failed = True
            continue
        if not args.dry_run:
            try:
                _move_case(source, targets)
            except OSError as error:
                logger.error("Could not migrate %s: %s", source, error)
                failed = True
                continue
        for index, target in enumerate(targets):
            action = "Copy" if index else "Move"
            logger.info("%s%s %s -> %s", "Dry run: " if args.dry_run else "", action, source, target)

    for group_dir in sorted({source.parent for source in plan}):
        for source in sorted(group_dir.glob("model_*")):
            if source.is_dir() and source not in plan:
                logger.warning("Unmapped result directory: %s", source)
    if not matched and not failed:
        logger.info("No legacy result directories matched the current experiment matrix.")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
