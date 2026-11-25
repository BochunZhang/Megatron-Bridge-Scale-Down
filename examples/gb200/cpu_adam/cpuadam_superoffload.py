# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run DeepSpeed's SuperOffload optimizer process for the CPUAdam benchmark."""

import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from torch import Tensor

_POLL_SECONDS = 0.001  # Match the polling interval of SuperOffload Stage 3.
_RESULT_TIMEOUT_SECONDS = 180


@contextmanager
def superoffload_step(
    *, parameter: "Tensor", gradient: "Tensor", config: dict[str, Any], threads: int, rank_cpus: list[int]
) -> Iterator[tuple[Callable[[], None], dict[str, Any]]]:
    """Yield a complete submit-and-wait step and the process binding metadata.

    Args:
        parameter: Contiguous FP32 CPU weights.
        gradient: Contiguous FP32 CPU gradients, with the same shape as the weights.
        config: DeepSpeed optimizer group configuration.
        threads: Number of CPUs assigned to the worker, with the same OpenMP request.
        rank_cpus: CPU affinity of the rank before the SuperOffload split.

    Yields:
        The step callback and worker metadata. The callback includes IPC and
        the worker's gradient copy.
    """
    from deepspeed.runtime.superoffload.superoffload_utils import EventTypes, SuperOffloadCPUOptimizer, TaskKeys

    parameter.share_memory_()
    gradient.share_memory_()
    optimizer = None
    try:
        # DeepSpeed splits the CPU list at floor((1 - fraction) * count).
        optimizer = SuperOffloadCPUOptimizer(
            optimizer_config=config,
            cpuadam_cores_perc=(threads - 0.5) / len(rank_cpus),
            max_grad_numel=gradient.numel(),
            pin_memory=True,
        )
        worker = optimizer.cpuadam_process
        controller_cpus = sorted(os.sched_getaffinity(0))
        worker_cpus = sorted(os.sched_getaffinity(worker.pid))
        if controller_cpus != rank_cpus[:-threads] or worker_cpus != rank_cpus[-threads:]:
            raise RuntimeError(
                f"Unexpected SuperOffload CPU split. Controller={controller_cpus}, worker={worker_cpus}."
            )

        def step() -> None:
            optimizer.async_step(0, 0, parameter, gradient)
            deadline = time.monotonic() + _RESULT_TIMEOUT_SECONDS
            while (result := optimizer.get_result(expected_event_type=EventTypes.ADAM_STEP)) is None:
                if not worker.is_alive():
                    raise RuntimeError(f"SuperOffload worker exited with status {worker.exitcode}.")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Timed out waiting for the DeepSpeed CPUAdam worker.")
                time.sleep(_POLL_SECONDS)
            if result[TaskKeys.SUB_GROUP_ID] != 0:
                raise RuntimeError("SuperOffload returned an unexpected parameter subgroup.")
            # The worker updates the already-shared parameter storage in place.

        yield (
            step,
            {
                "controller_cpus": controller_cpus,
                "worker_cpus": worker_cpus,
                "worker_pid": worker.pid,
                "gradient_scratch_pinned": True,
                "timing_scope": "submit_copy_adam_result",
            },
        )
    finally:
        try:
            if optimizer is not None:
                optimizer.close()
        finally:
            os.sched_setaffinity(0, rank_cpus)
