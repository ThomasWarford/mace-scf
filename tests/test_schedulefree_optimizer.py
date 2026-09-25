"""--optimizer schedulefree: the MACE-Polar recipe's knobs must reach AdamWScheduleFree.

Polar trains with betas (0.9, 0.98) and a 2000-step linear warmup. The betas come from
--beta / --beta_two and the warmup from --warmup_steps_schedulefree; before that flag
existed the warmup was silently 0. Also pinned: per-stage lr overrides still act
(run_train.py writes param_group["lr"] at each stage start), and the optimizer state
survives a checkpoint round trip, which --restart_latest depends on.
"""

import pytest
import torch

pytest.importorskip("schedulefree")

from mace_scf.utils.run_train_utils import build_optimizer, get_param_options
from tests.test_polar_backbone_options import _args, _build


def _optimizer(*extra):
    args = _args(
        "MACE", "--optimizer", "schedulefree",
        "--beta", "0.9", "--beta_two", "0.98", "--lr", "0.01", *extra,
    )
    model = _build(args)
    return model, build_optimizer(get_param_options(model, args), args)


def _step(model, optimizer):
    optimizer.zero_grad()
    loss = sum((p ** 2).sum() for p in model.parameters())
    loss.backward()
    optimizer.step()


def test_warmup_and_betas_reach_the_optimizer():
    _, optimizer = _optimizer("--warmup_steps_schedulefree", "2000")
    assert type(optimizer).__name__ == "AdamWScheduleFree"
    for group in optimizer.param_groups:
        assert group["warmup_steps"] == 2000
        assert group["betas"] == (0.9, 0.98)
        assert "amsgrad" not in group


def test_warmup_defaults_to_off():
    _, optimizer = _optimizer()
    assert all(g["warmup_steps"] == 0 for g in optimizer.param_groups)


def test_warmup_ramps_the_scheduled_lr():
    model, optimizer = _optimizer("--warmup_steps_schedulefree", "10")
    optimizer.train()
    _step(model, optimizer)
    lr_first = optimizer.param_groups[0]["scheduled_lr"]
    for _ in range(19):
        _step(model, optimizer)
    lr_after = optimizer.param_groups[0]["scheduled_lr"]
    assert lr_first == pytest.approx(0.01 / 10)
    assert lr_after == pytest.approx(0.01)


def test_stage_lr_override_acts():
    """run_train sets param_group['lr'] per stage; schedulefree must read it each step."""
    model, optimizer = _optimizer()
    optimizer.train()
    _step(model, optimizer)
    for group in optimizer.param_groups:
        group["lr"] = 0.001
    _step(model, optimizer)
    assert optimizer.param_groups[0]["scheduled_lr"] == pytest.approx(0.001)


def test_state_survives_a_checkpoint_round_trip():
    model, optimizer = _optimizer("--warmup_steps_schedulefree", "10")
    optimizer.train()
    for _ in range(3):
        _step(model, optimizer)
    optimizer.eval()  # train.py leaves it in eval mode when it checkpoints
    saved = optimizer.state_dict()

    _, restored = _optimizer("--warmup_steps_schedulefree", "10")
    restored.load_state_dict(saved)
    for a, b in zip(optimizer.param_groups, restored.param_groups):
        for key in ("k", "weight_sum", "lr_max", "train_mode", "warmup_steps"):
            if key in a:
                assert a[key] == b[key], key
    for sa, sb in zip(saved["state"].values(), restored.state_dict()["state"].values()):
        for key in sa:
            torch.testing.assert_close(sb[key], sa[key])


@pytest.mark.parametrize(
    "flag, expected", [((), True), (("--amsgrad",), True), (("--no-amsgrad",), False)]
)
def test_amsgrad_can_be_switched_off(flag, expected):
    """Upstream's --amsgrad is store_true with default True: it could never be disabled."""
    args = _args("MACE", "--optimizer", "adam", *flag)
    assert args.amsgrad is expected
    model = _build(args)
    optimizer = build_optimizer(get_param_options(model, args), args)
    assert type(optimizer) is torch.optim.Adam
    assert all(g["amsgrad"] is expected for g in optimizer.param_groups)
