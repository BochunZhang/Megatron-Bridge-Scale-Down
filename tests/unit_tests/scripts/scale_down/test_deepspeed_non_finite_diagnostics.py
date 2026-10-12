"""Exercise the training script's diagnostics without importing CUDA/DeepSpeed."""

from __future__ import annotations

import ast
import contextlib
import json
import logging
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch


pytestmark = pytest.mark.unit
TRAIN_SCRIPT = Path(__file__).resolve().parents[4] / "examples/scale-down/01-analyse/02-deepspeed-offload/train.py"


@pytest.fixture
def diagnostics_module() -> dict[str, Any]:
    # Import just these definitions: importing train.py also loads distributed
    # model implementations and DeepSpeed's CUDA extensions.
    tree = ast.parse(TRAIN_SCRIPT.read_text())
    names = {"sync_cuda", "_non_finite_tensor_report", "_NonFiniteDiagnostics"}
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    namespace = {
        "torch": torch,
        "contextlib": contextlib,
        "json": json,
        "os": os,
        "sys": sys,
        "Any": Any,
        "Iterator": Iterator,
        "logger": logging.getLogger(__name__),
    }
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, str(TRAIN_SCRIPT), "exec"), namespace)
    return namespace


class _Optimizer:
    def __init__(self, param: torch.Tensor) -> None:
        self.fp16_groups = [[param]]
        self.fp32_partitioned_groups_flat = [torch.nn.Parameter(torch.ones(2))]

    def _get_param_partition_rank(self, param: torch.Tensor) -> int:
        return 1  # Last shard: the second element is padding for a size-3 weight.

    def _release_sub_group(self, sub_group_id: int, timer_names: Any) -> None:
        self.fp32_partitioned_groups_flat[sub_group_id].grad = None


@pytest.fixture
def case(diagnostics_module: dict[str, Any], tmp_path: Path) -> SimpleNamespace:
    model = torch.nn.Module()
    param = torch.nn.Parameter(torch.ones(3, dtype=torch.bfloat16))
    param.ds_id = 7
    param.ds_numel = 3
    param.ds_shape = torch.Size([3])
    param.ds_status = SimpleNamespace(name="AVAILABLE")
    param.ds_tensor = torch.ones(2, dtype=torch.bfloat16)
    param.ds_tensor.ds_numel = 2
    param.ds_tensor.status = SimpleNamespace(name="AVAILABLE")
    model.register_parameter("weight", param)
    optimizer = _Optimizer(param)
    engine = SimpleNamespace(optimizer=optimizer, module=model, device=torch.device("cpu"), global_steps=0)
    state = SimpleNamespace(engine=engine, rank=0)
    diagnostics = diagnostics_module["_NonFiniteDiagnostics"](state, str(tmp_path / "metrics.csv"))
    return SimpleNamespace(
        diagnostics=diagnostics,
        optimizer=optimizer,
        param=param,
        master=optimizer.fp32_partitioned_groups_flat[0],
        output_dir=tmp_path,
    )


def _report(case: SimpleNamespace, kind: str, *, summary: bool = False) -> dict[str, Any]:
    suffix = "" if summary else ".rank0"
    return json.loads((case.output_dir / f"non_finite_{kind}{suffix}.json").read_text())


def test_scanner_records_every_position_across_chunk_boundaries(diagnostics_module: dict[str, Any]) -> None:
    values = torch.tensor([0, float("nan"), float("nan"), float("nan"), float("inf"), -float("inf"), 1])
    result = diagnostics_module["_non_finite_tensor_report"](values, chunk_size=2)
    assert result["counts"] == {"NaN": 3, "+Inf": 1, "-Inf": 1}
    assert result["non_finite_count"] == 5
    assert result["ranges"] == [
        {"start": 1, "end": 4, "value": "NaN"},
        {"start": 4, "end": 5, "value": "+Inf"},
        {"start": 5, "end": 6, "value": "-Inf"},
    ]
    json.dumps(result, allow_nan=False)


def test_initial_failure_writes_both_reports_and_continues(case: SimpleNamespace) -> None:
    case.master.data[1] = float("nan")
    case.param.ds_tensor[0] = float("inf")
    case.diagnostics.check_initial()  # Must not raise SystemExit.
    initial = _report(case, "init_weight")
    assert initial["non_finite_count"] == 1
    assert _report(case, "init_param")["non_finite_count"] == 1
    layout = initial["tensors"][0]["parameter_layout"][0]
    assert layout["name"] == "weight"
    assert layout["subgroup_offset"] == 0
    assert layout["parameter_offset"] == 2
    assert layout["valid_numel"] == 1
    assert layout["padding_numel"] == 1
    assert _report(case, "init_weight", summary=True)["counts"]["NaN"] == 1


def test_adam_gradient_survives_release_in_report_and_all_checks_finish(case: SimpleNamespace) -> None:
    case.master.data[1] = float("nan")
    case.diagnostics.check_initial()
    initial_bytes = (case.output_dir / "non_finite_init_weight.rank0.json").read_bytes()
    case.diagnostics.begin_step(0)
    case.diagnostics.losses = [torch.tensor(float("nan")), torch.tensor(1.0)]
    case.master.grad = torch.tensor([0.1, float("nan")])
    adam = torch.optim.Adam([case.master], lr=0.01)
    with case.diagnostics.capture_optimizer_gradients():
        adam.step()
        case.param.ds_tensor.copy_(case.master.detach())
        case.optimizer._release_sub_group(0, set())
    assert case.master.grad is None
    assert "_release_sub_group" not in vars(case.optimizer)
    with pytest.raises(SystemExit) as exc:
        case.diagnostics.finish_step(1.0)  # Earlier micro-batch NaN must still be found.
    assert exc.value.code == 3
    for kind in ("loss", "gradient", "main_weight", "param"):
        assert _report(case, kind)["non_finite_count"] == 1
        assert _report(case, kind, summary=True)["non_finite_count"] == 1
    assert _report(case, "gradient")["tensors"][0]["source"] == "fp32_partitioned_groups_flat.grad"
    assert _report(case, "main_weight")["tensors"][0]["ranges"] == _report(case, "init_weight")["tensors"][0]["ranges"]
    assert (case.output_dir / "non_finite_init_weight.rank0.json").read_bytes() == initial_bytes


def test_finite_master_gradients_ignore_forward_parameter_grad(case: SimpleNamespace) -> None:
    case.param.grad = torch.full_like(case.param, float("nan"))
    case.master.grad = torch.ones_like(case.master)
    case.diagnostics.begin_step(0)
    case.diagnostics.losses = [torch.tensor(1.0)]
    with case.diagnostics.capture_optimizer_gradients():
        case.optimizer._release_sub_group(0, set())
    case.diagnostics.finish_step(1.0)
    assert _report(case, "gradient")["checked_numel"] == 2
    assert _report(case, "gradient")["non_finite_count"] == 0


@pytest.mark.parametrize("status", ["NOT_AVAILABLE", "INFLIGHT"])
def test_nvme_placeholder_is_skipped_but_resident_full_parameter_is_checked(
    case: SimpleNamespace, status: str
) -> None:
    case.param.ds_tensor.fill_(float("nan"))
    case.param.ds_tensor.status.name = status
    case.param.data[0] = -float("inf")
    case.diagnostics.check_initial()
    report = _report(case, "init_param")
    assert report["counts"] == {"NaN": 0, "+Inf": 0, "-Inf": 1}
    assert report["checked_numel"] == 3
    assert report["skipped"][0]["reason"] == status
    assert report["tensors"][0]["source"] == "param.data"


def test_reduced_loss_failure_writes_remaining_categories(case: SimpleNamespace) -> None:
    case.diagnostics.begin_step(0)
    case.diagnostics.losses = [torch.tensor(1.0)]
    with pytest.raises(SystemExit):
        case.diagnostics.finish_step(float("inf"))
    loss = _report(case, "loss")
    assert loss["tensors"][0]["non_finite_count"] == 0
    assert loss["tensors"][1]["counts"]["+Inf"] == 1
    assert _report(case, "gradient")["skipped"]  # Absent gradients never count as checked.
    assert _report(case, "main_weight")["non_finite_count"] == 0
    assert _report(case, "param")["non_finite_count"] == 0


def test_other_rank_failure_exits_only_after_all_reports(
    case: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    case.diagnostics.begin_step(0)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    barriers = []
    monkeypatch.setattr(torch.distributed, "barrier", lambda: barriers.append(True))

    def gather(ranks: list, local: dict) -> None:
        remote = json.loads(json.dumps(local))
        for summary in remote.values():
            summary["rank"] = 1
        remote["gradient"]["counts"]["NaN"] = 2
        remote["gradient"]["non_finite_count"] = 2
        ranks[:] = [local, remote]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    with pytest.raises(SystemExit):
        case.diagnostics.finish_step(1.0)
    assert _report(case, "gradient", summary=True)["non_finite_count"] == 2
    assert _report(case, "param", summary=True)["non_finite_count"] == 0
    assert barriers == [True]


def test_gradient_hook_restores_existing_override_on_exception(case: SimpleNamespace) -> None:
    override = lambda *args: None
    case.optimizer._release_sub_group = override
    with pytest.raises(RuntimeError), case.diagnostics.capture_optimizer_gradients():
        raise RuntimeError("optimizer failed")
    assert case.optimizer._release_sub_group is override
