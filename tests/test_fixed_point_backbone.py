"""The R2 (MACE-Polar-M) backbone options on FixedPoint.

--edge_irreps / --use_agnostic_product are covered with the other models in
test_polar_backbone_options.py. Here: --local_scale_shift (ScaleShiftMACE's per-atom
scale/shift of ZBL + backbone readouts, as for LocalSplitCharges) and
--field_hidden_irreps (a narrower width for the SCF-side blocks under a wide backbone).
Without either flag the model must be exactly the one built before they existed, and the
compiled SCF path must agree with the eager one with all options on.
"""

import numpy as np
import pytest
import torch
from e3nn import o3

from mace_scf.calculators.fixedpoint_scf import MACEFixedPointSCF
from mace_scf.electrostatics.compiled_fixed_point import (
    build_compiled_fixed_point_evaluator,
)
from tests.test_polar_backbone_options import NONLINEAR, _args, _build
from tests.utils import seed_torch, water_configs

SCALE, SHIFT = 0.7, -1.5
FP_ARGV = ("--pair_repulsion", "--distance_transform", "Agnesi")
SCALE_SHIFT_ARGV = ("--local_scale_shift", "True", "--mean", str(SHIFT), "--std", str(SCALE))
WIDE = "16x0e + 16x1o"
NARROW = "8x0e + 8x1o"
FIELD_WIDTH_ARGV = ("--hidden_irreps", WIDE, "--field_hidden_irreps", NARROW)
R2_ARGV = (
    *SCALE_SHIFT_ARGV,
    *FIELD_WIDTH_ARGV,
    "--edge_irreps", "4x0e + 4x1o",
    "--use_agnostic_product", "True",
)
SCF_OPTIONS = {
    "constant_charge": False,
    "num_scf_steps": 3,
    "mixing_parameter": 0.2,
    "scf_tolerance": 1e-300,
    "initial_density": "local_guess",
    "initial_fermi_level": "from_data",
}


def _model(*extra):
    seed_torch(0)
    return _build(_args("FixedPoint", *FP_ARGV, *extra))


def _atoms():
    atoms = water_configs()[0].copy()
    atoms.info["total_charge"] = 0.0
    atoms.info["initial_fermi_level"] = -5.0
    atoms.arrays["initial_density_coefficients"] = np.zeros((len(atoms), 4))
    return atoms


def _calculator(model, tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "model.model"
    torch.save(model, path)
    return MACEFixedPointSCF(
        model_path=str(path),
        device="cpu",
        pbc_handling="pbc",
        scf_options=SCF_OPTIONS,
    )


def _local(model, tmp_path):
    calc = _calculator(model, tmp_path)
    batch = calc._build_batch(_atoms())
    return calc.model.local_part(batch.to_dict()), batch


def test_off_by_default_and_unchanged():
    model = _model()
    assert not hasattr(model, "scale_shift")
    assert not hasattr(model, "field_node_projection")
    # passing the backbone width explicitly is the default too
    same = _model("--field_hidden_irreps", NARROW)
    assert not hasattr(same, "field_node_projection")
    sd, sd_same = model.state_dict(), same.state_dict()
    assert sd.keys() == sd_same.keys()
    assert all(sd[k].shape == sd_same[k].shape for k in sd)


def test_scale_shift_block_is_built_from_mean_and_std():
    model = _model(*SCALE_SHIFT_ARGV)
    assert model.scale_shift.scale.item() == pytest.approx(SCALE)
    assert model.scale_shift.shift.item() == pytest.approx(SHIFT)
    extra = set(model.state_dict()) - set(_model().state_dict())
    assert extra == {"scale_shift.scale", "scale_shift.shift"}


def test_local_energy_decomposes_as_scale_shift_mace(tmp_path):
    plain = _model()
    scaled = _model(*SCALE_SHIFT_ARGV)
    scaled.load_state_dict(plain.state_dict(), strict=False)

    state_plain, batch = _local(plain, tmp_path / "plain")
    state_scaled, _ = _local(scaled, tmp_path / "scaled")
    c_plain, c_scaled = state_plain.energies, state_scaled.energies
    num_atoms = (batch.ptr[1:] - batch.ptr[:-1]).to(c_plain.dtype)

    # E0, ZBL, one readout per layer, then the shift
    assert c_plain.shape[-1] == 1 + 1 + 2
    assert c_scaled.shape[-1] == 1 + 1 + 2 + 1
    torch.testing.assert_close(c_scaled[:, 0], c_plain[:, 0])  # E0 untouched
    torch.testing.assert_close(c_scaled[:, 1:-1], SCALE * c_plain[:, 1:])  # ZBL + readouts
    torch.testing.assert_close(c_scaled[:, -1], SHIFT * num_atoms)  # shift, once per atom
    # the charges and SCF-side features do not depend on the energy readouts
    torch.testing.assert_close(
        state_scaled.field_independent_charge_density,
        state_plain.field_independent_charge_density,
    )
    torch.testing.assert_close(state_scaled.all_layer_feats, state_plain.all_layer_feats)


def test_field_hidden_irreps_is_refused_for_other_models():
    with pytest.raises(ValueError, match="FixedPoint option"):
        _args("LocalSplitCharges", "--field_hidden_irreps", NARROW)


def test_field_blocks_take_the_field_width():
    model = _model(*FIELD_WIDTH_ARGV)
    wide, narrow = o3.Irreps(WIDE), o3.Irreps(NARROW)
    assert model.products[-1].linear.irreps_out == wide
    assert model.field_node_projection.irreps_in == wide
    assert model.field_node_projection.irreps_out == narrow
    assert all(lin.irreps_out == narrow for lin in model.layer_feature_mixer.linears)
    assert model.field_dependent_charges_map.node_feats_irreps == narrow
    assert model.local_electron_energy.node_feats_irreps == narrow
    # the SCF blocks are the default-width ones: same parameter count as a narrow model's
    narrow_model = _model("--hidden_irreps", NARROW)
    for name in ("field_dependent_charges_map", "local_electron_energy"):
        n = sum(p.numel() for p in getattr(model, name).parameters())
        n_ref = sum(p.numel() for p in getattr(narrow_model, name).parameters())
        assert n == n_ref, name


def test_field_width_forward_and_backward(tmp_path):
    model = _model(*FIELD_WIDTH_ARGV)
    calc = _calculator(model, tmp_path)
    for p in calc.model.parameters():
        p.requires_grad = True
    batch = calc._build_batch(_atoms())
    out = calc.runner.eval(calc.model, batch.to_dict(), training=True, compute_force=True)
    loss = out["energy"].sum() + out["forces"].pow(2).sum()
    loss.backward()
    assert torch.isfinite(loss)
    grads = [p.grad for p in calc.model.field_node_projection.parameters()]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)


def test_nonlinear_r2_backbone_builds():
    model = _model(*R2_ARGV)
    assert type(model.interactions[0]).__name__ == NONLINEAR
    assert all(p.use_agnostic_product for p in model.products)
    assert model.interactions[1].edge_irreps == o3.Irreps("4x0e + 4x1o")


def test_compiled_matches_eager_with_all_options(tmp_path):
    calc = _calculator(_model(*R2_ARGV), tmp_path)
    batch = calc._build_batch(_atoms())
    restart_state = {"initial_charge_density": None, "fermi_level": None}
    reference = calc._run_model(batch, restart_state)
    evaluator = build_compiled_fixed_point_evaluator(
        calc.model,
        pbc_handling="pbc",
        num_scf_steps=calc.scf_options.num_scf_steps,
        mixing_parameter=calc.scf_options.mixing_parameter,
        enabled=False,
    )
    actual = evaluator.evaluate(batch.to_dict())
    for key in ("energy", "forces", "density_coefficients", "fermi_level", "electron_energy"):
        np.testing.assert_allclose(
            actual[key].detach().cpu().numpy(),
            reference[key].detach().cpu().numpy(),
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"{key} mismatch",
        )
