import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
EXPORT_SCRIPT = REPO_ROOT / "scripts/scale-down/analyse/export_gpu_utilization.py"


@pytest.mark.unit
def test_export_gpu_utilization_cli(tmp_path: Path) -> None:
    log_path = tmp_path / "train.log"
    output_path = tmp_path / "results" / "gpu_utilization.json"
    log_path.write_text(
        "Step Time : 2.50s GPU utilization: 617.1MODEL_TFLOP/s/GPU\n"
        " [2026-09-17 12:00:00] iteration        1/       2 | lm loss: 1.0 |\n"
        "Step Time : 2.00s GPU utilization: 725.4MODEL_TFLOP/s/GPU\n"
        " [2026-09-17 12:00:02] iteration        2/       2 | lm loss: 0.9 |\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(EXPORT_SCRIPT),
            "--log-file",
            str(log_path),
            "--output",
            str(output_path),
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == {"0": 617.1, "1": 725.4}
