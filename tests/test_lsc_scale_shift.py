"""--local_scale_shift: LocalSplitCharges with ScaleShiftMACE's per-atom scale/shift.

The R2 backbone is a ScaleShiftMACE, so its readouts learn (E_local - shift) / scale per
atom. LocalSplitCharges gets the same target by scaling ZBL and every readout and adding
the shift once per atom; E0 and the electrostatics are left alone. Without the flag the
model must be exactly the one trained before the flag existed.
"""

import numpy as np
import pytest
import torch

import mace.tools
from tests.test_polar_backbone_options import _args, _build
from tests.utils import dataset_from_atoms, seed_torch, water_configs

SCALE, SHIFT = 0.7, -1.5
LSC_ARGV = ("--pair_repulsion", "--distance_transform", "Agnesi")
SCALE_SHIFT_ARGV = ("--local_scale_shift", "True", "--mean", str(SHIFT), "--std", str(SCALE))


def _model(*extra):
    seed_torch(0)
    return _build(_args("LocalSplitCharges", *LSC_ARGV, *extra))


def _outputs(model):
    torch.set_default_dtype(torch.float64)
    dataset = dataset_from_atoms(water_configs()[:2], cutoff=3.0, atomic_multipoles_max_l=1)
    loader = mace.tools.torch_geometric.dataloader.DataLoader(
        dataset=dataset, batch_size=2, shuffle=False
    )
    batch = next(iter(loader))
    out = model(batch.to_dict(), training=False, compute_force=True, compute_stress=True)
    return out, batch


def _electrostatic(out):
    """Everything in the total energy that is not a local contribution column."""
    return out["energy"] - out["contributions"].sum(dim=-1)


def test_off_by_default_and_unchanged():
    model = _model()
    assert not hasattr(model, "scale_shift")
    # E0, ZBL, one readout per layer: no shift column
    out, _ = _outputs(model)
    assert out["contributions"].shape[-1] == 1 + 1 + 2


def test_scale_shift_block_is_built_from_mean_and_std():
    model = _model(*SCALE_SHIFT_ARGV)
    assert model.scale_shift.scale.item() == pytest.approx(SCALE)
    assert model.scale_shift.shift.item() == pytest.approx(SHIFT)
    extra = set(model.state_dict()) - set(_model().state_dict())
    assert extra == {"scale_shift.scale", "scale_shift.shift"}


def test_energy_decomposes_as_scale_shift_mace():
    plain = _model()
    scaled = _model(*SCALE_SHIFT_ARGV)
    scaled.load_state_dict(plain.state_dict(), strict=False)

    out_plain, batch = _outputs(plain)
    out_scaled, _ = _outputs(scaled)
    c_plain, c_scaled = out_plain["contributions"], out_scaled["contributions"]
    num_atoms = (batch.ptr[1:] - batch.ptr[:-1]).to(c_plain.dtype)

    torch.testing.assert_close(c_scaled[:, 0], c_plain[:, 0])  # E0 untouched
    torch.testing.assert_close(c_scaled[:, 1:-1], SCALE * c_plain[:, 1:])  # ZBL + readouts
    torch.testing.assert_close(c_scaled[:, -1], SHIFT * num_atoms)  # shift, once per atom
    # the charges do not depend on the energy readouts, so neither does the Coulomb term
    torch.testing.assert_close(_electrostatic(out_scaled), _electrostatic(out_plain))
    torch.testing.assert_close(
        out_scaled["density_coefficients"], out_plain["density_coefficients"]
    )


def test_shift_alone_leaves_forces_and_stress_alone():
    plain = _model()
    shifted = _model("--local_scale_shift", "True", "--mean", str(SHIFT), "--std", "1.0")
    shifted.load_state_dict(plain.state_dict(), strict=False)
    out_plain, _ = _outputs(plain)
    out_shifted, _ = _outputs(shifted)
    torch.testing.assert_close(out_shifted["forces"], out_plain["forces"])
    torch.testing.assert_close(out_shifted["stress"], out_plain["stress"])


def test_scale_scales_the_local_forces():
    """F = F_electrostatic + F_local, and only F_local is scaled. F_electrostatic is read
    off a model whose readouts and ZBL are scaled to zero."""
    plain = _model()
    local_off = _model("--local_scale_shift", "True", "--mean", "0.0", "--std", "0.0")
    scaled = _model(*SCALE_SHIFT_ARGV)
    for model in (local_off, scaled):
        model.load_state_dict(plain.state_dict(), strict=False)

    f_plain = _outputs(plain)[0]["forces"]
    f_elec = _outputs(local_off)[0]["forces"]
    f_scaled = _outputs(scaled)[0]["forces"]
    torch.testing.assert_close(f_scaled - f_elec, SCALE * (f_plain - f_elec))
    assert not torch.allclose(f_plain, f_elec)  # the local part is not trivially zero


@pytest.mark.parametrize("model", ["LocalCharges", "FixedChargeBaselinedMACE", "MACE"])
def test_refused_where_not_implemented(model):
    with pytest.raises(NotImplementedError, match="local_scale_shift"):
        _build(_args(model, *SCALE_SHIFT_ARGV))


def test_needs_mean_and_std():
    with pytest.raises(ValueError, match="mean and std"):
        _build(_args("LocalSplitCharges", "--local_scale_shift", "True"))


def test_scale_shift_survives_save_and_load(tmp_path):
    """Checkpoints and .model files are torch.save'd whole; the block and its buffers
    must come back and give the same energies."""
    model = _model(*SCALE_SHIFT_ARGV)
    path = tmp_path / "lsc_scale_shift.model"
    torch.save(model, path)
    loaded = torch.load(path, weights_only=False)

    assert loaded.scale_shift.scale.item() == pytest.approx(SCALE)
    assert loaded.scale_shift.shift.item() == pytest.approx(SHIFT)
    out, _ = _outputs(model)
    out_loaded, _ = _outputs(loaded)
    torch.testing.assert_close(out_loaded["energy"], out["energy"])
    torch.testing.assert_close(out_loaded["forces"], out["forces"])
