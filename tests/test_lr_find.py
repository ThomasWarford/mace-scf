"""--lr_find: fastai-style lr range test (mace_scf/utils/lr_find.py).

The unit tests drive lr_find through the real take_step with a toy linear model, so the
lr ramp, per-group ratios, single-group ramps and divergence stop are pinned without
building a MACE. The run_train tests check the flag end to end: it range-tests one stage,
writes the csv/json/png, and exits before training (no checkpoint, no model file).
"""

import csv
import json
import math

import numpy as np
import pytest
import torch

from mace_scf.utils.lr_find import lr_find, suggest_lr, write_lr_find_outputs
from tests.run_train_pipeline.harness import (
    RunTrainCase,
    make_run_train_config,
    model_config_overrides,
    run_train_case,
)


class _Batch:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def to(self, device):
        return self

    def to_dict(self):
        return {"x": self.x}


class _Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch_dict, training):
        return self.model(batch_dict["x"])


def _mse(pred, ref):
    return ((pred - ref.y) ** 2).mean()


def _setup(group_lrs=(1e-3, 1e-3)):
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
    model = model.double()
    x = torch.randn(64, 4, dtype=torch.float64)
    y = x.sum(dim=1, keepdim=True)
    loader = [_Batch(x[i : i + 8], y[i : i + 8]) for i in range(0, 64, 8)]
    optimizer = torch.optim.Adam(
        [
            {"name": "first", "params": model[0].parameters(), "lr": group_lrs[0]},
            {"name": "second", "params": model[2].parameters(), "lr": group_lrs[1]},
        ]
    )
    return model, _Wrapper(model), loader, optimizer


def _run(model, wrapper, loader, optimizer, **kwargs):
    kwargs.setdefault("ref_lr", 1e-3)
    return lr_find(
        model, wrapper, _mse, loader, optimizer, torch.device("cpu"), **kwargs
    )


def test_lr_ramps_geometrically_and_cycles_the_loader():
    model, wrapper, loader, optimizer = _setup()
    records = _run(
        model, wrapper, loader, optimizer,
        lr_min=1e-5, lr_max=1e-3, num_steps=20, diverge_factor=math.inf,
    )
    assert len(records) == 20  # more steps than the 8 batches
    lrs = np.array([r["lr"] for r in records])
    assert lrs[0] == pytest.approx(1e-5)
    assert lrs[-1] == pytest.approx(1e-3)
    assert np.allclose(np.diff(np.log(lrs)), np.log(100) / 19)
    for group in optimizer.param_groups:
        assert group["lr"] == pytest.approx(1e-3)


def test_group_lr_ratios_follow_the_global_ramp():
    # stage lr 1e-3 with second_lr 1e-2: the second group stays 10x the swept lr
    model, wrapper, loader, optimizer = _setup(group_lrs=(1e-3, 1e-2))
    _run(model, wrapper, loader, optimizer, lr_min=1e-6, lr_max=1e-4, num_steps=5)
    lrs = [g["lr"] for g in optimizer.param_groups]
    assert lrs == pytest.approx([1e-4, 1e-3])


def test_named_group_ramps_alone():
    model, wrapper, loader, optimizer = _setup(group_lrs=(1e-3, 2e-3))
    _run(
        model, wrapper, loader, optimizer,
        lr_min=1e-6, lr_max=1e-4, num_steps=5, group="second",
    )
    lrs = [g["lr"] for g in optimizer.param_groups]
    assert lrs == pytest.approx([1e-3, 1e-4])


def test_unknown_group_is_rejected():
    model, wrapper, loader, optimizer = _setup()
    with pytest.raises(ValueError, match="no param group named"):
        _run(model, wrapper, loader, optimizer, num_steps=5, group="nope")


def test_stops_when_the_loss_diverges():
    model, wrapper, loader, optimizer = _setup()
    records = _run(
        model, wrapper, loader, optimizer, lr_min=1e-4, lr_max=1e4, num_steps=400,
    )
    assert 10 < len(records) < 400
    smoothed = [r["smoothed_loss"] for r in records]
    assert smoothed[-1] > 4 * min(smoothed) or not math.isfinite(records[-1]["loss"])


def test_stops_on_non_finite_loss():
    model, wrapper, loader, optimizer = _setup()
    loader[3] = _Batch(loader[3].x, torch.full_like(loader[3].y, float("nan")))
    records = _run(model, wrapper, loader, optimizer, num_steps=50)
    assert len(records) == 3
    assert all(math.isfinite(r["loss"]) for r in records)


def test_clipping_is_applied_and_recorded():
    model, wrapper, loader, optimizer = _setup()
    records = _run(model, wrapper, loader, optimizer, num_steps=5, max_grad_norm=1e-3)
    assert all(r["grad_clip_applied"] for r in records)
    assert all(r["grad_norm_before_clip"] > 1e-3 for r in records)


def _u_curve(n=200):
    lrs = np.logspace(-6, -1, n)
    x = np.log10(lrs)
    losses = 1.0 + (x + 3.0) ** 2 * (x > -3) * 4 + 1.0 / (1 + np.exp(4 * (x + 4.5)))
    return [{"lr": lr, "smoothed_loss": loss} for lr, loss in zip(lrs, losses)]


def test_suggestions_sit_on_the_descending_side_of_the_minimum():
    s = suggest_lr(_u_curve())
    assert 1e-6 < s["valley"] < 1e-3
    assert 1e-5 < s["steepest"] < 10**-3.5
    assert s["min_div10"] == pytest.approx(1e-4, rel=0.2)


def test_suggestions_are_nan_for_short_curves():
    s = suggest_lr(_u_curve()[:12])
    assert all(math.isnan(v) for v in s.values())


def test_outputs_are_written(tmp_path):
    records = [
        {**r, "step": i, "loss": r["smoothed_loss"], "grad_norm_before_clip": 1.0,
         "grad_clip_applied": False}
        for i, r in enumerate(_u_curve(50))
    ]
    suggestions = suggest_lr(records)
    write_lr_find_outputs(records, suggestions, tmp_path / "sub" / "run", max_grad_norm=10.0)
    with open(tmp_path / "sub" / "run.csv") as f:
        assert len(list(csv.DictReader(f))) == 50
    assert json.loads((tmp_path / "sub" / "run.json").read_text()) == suggestions
    assert (tmp_path / "sub" / "run.png").stat().st_size > 0


@pytest.mark.parametrize("model", ["MACE", "FixedPoint"])
def test_run_train_lr_find_writes_outputs_and_does_not_train(tmp_path, model):
    case = RunTrainCase(
        name=f"lr_find_{model}",
        config=make_run_train_config(
            model=model, name=f"lr_find_{model}", overrides=model_config_overrides(model)
        ),
        extra_args=[
            "--lr_find", "--lr_find_steps", "12", "--lr_find_repeats", "2",
            "--lr_find_min", "1e-5", "--lr_find_max", "1e-2",
        ],
        timeout_s=600,
    )
    result = run_train_case(tmp_path / model, case)
    result.assert_success()
    results = result.run_dir / "results"
    for repeat in (0, 1):
        prefix = f"lr_find_{model}_run-5_stage1_lrfind_r{repeat}"
        with open(results / f"{prefix}.csv") as f:
            rows = list(csv.DictReader(f))
        assert 0 < len(rows) <= 12
        assert float(rows[0]["lr"]) == pytest.approx(1e-5)
        assert (results / f"{prefix}.png").exists()
        assert set(json.loads((results / f"{prefix}.json").read_text())) == {
            "valley", "steepest", "min_div10",
        }
    assert "running the range test with AdamW" in result.log_text()
    assert result.checkpoints() == []
    assert result.models() == []
