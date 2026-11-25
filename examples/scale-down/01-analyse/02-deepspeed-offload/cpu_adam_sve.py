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

"""Verify the CPU Adam extension used by the ARM SVE offload experiment."""

import logging
import platform
from pathlib import Path


logger = logging.getLogger(__name__)


def verify_cpu_adam_sve() -> None:
    """Require Linux AArch64 SVE hardware and a DeepSpeed CPU Adam SVE build."""
    if platform.system() != "Linux" or platform.machine().lower() not in {"aarch64", "arm64"}:
        raise RuntimeError("ARM SVE CPU Adam requires Linux AArch64 with SVE.")
    try:
        cpu_features = Path("/proc/cpuinfo").read_text().splitlines()
    except OSError as exc:
        raise RuntimeError("Could not check ARM SVE support in /proc/cpuinfo.") from exc
    if not any(line.startswith("Features") and "sve" in line.split() for line in cpu_features):
        raise RuntimeError("ARM SVE CPU Adam requires the Linux CPU feature 'sve'.")

    error = "CPU Adam must be built with ARM SVE. Install the DeepSpeed pin in requirements.txt on the Grace host."
    try:
        import deepspeed
        from deepspeed.ops.op_builder import CPUAdamBuilder

        cpu_adam = CPUAdamBuilder().load()
        # On AArch64 this combination identifies SVE in DeepSpeed v0.20.1.
        # The scalar and NEON builds both support the fused BF16 copy.
        is_sve = not cpu_adam.adam_copy_supported() and cpu_adam.adam_grad_scale_supported()
    except (ImportError, OSError, RuntimeError, AttributeError) as exc:
        raise RuntimeError(error) from exc
    if not is_sve:
        raise RuntimeError(error)

    logger.info(
        "CPU Adam kernel: ARM SVE (DeepSpeed %s, git %s, extension %s)",
        deepspeed.__version__,
        deepspeed.__git_hash__,
        cpu_adam.__file__,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        verify_cpu_adam_sve()
    except RuntimeError as exc:
        logger.error("%s", exc)
        raise SystemExit(2) from exc
