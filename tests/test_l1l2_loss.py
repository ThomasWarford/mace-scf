"""The L1/L2 loss terms must reproduce MACE-Polar's `l1l2energyforces`, plus stress.

energy_per_atom_l1 + forces_l2norm decompose upstream's WeightedEnergyForcesL1L2Loss
the same way the *_huber terms decompose UniversalLoss. stress_l1 has no upstream
counterpart (Polar trains on molecules), so it is pinned against a hand computation.
"""

import numpy as np
import pytest
import torch
from mace.modules.loss import WeightedEnergyForcesL1L2Loss

from mace_scf.electrostatics.loss import WeightedLoss
from tests.test_universal_loss import NUM_STRUCTS, ATOMS_PER_STRUCT, _synthetic_batch


def test_matches_upstream_l1l2_with_uniform_weights():
    ref, pred = _synthetic_batch()
    actual = WeightedLoss(
        {"energy_per_atom_l1": {"weight": 10.0}, "forces_l2norm": {"weight": 10.0}}
    )(ref, pred)
    expected = WeightedEnergyForcesL1L2Loss(energy_weight=10.0, forces_weight=10.0)(
        ref, pred, ddp=False
    )
    np.testing.assert_allclose(actual.item(), expected.item(), rtol=0, atol=1e-12)


def test_stress_l1_is_the_weighted_mean_absolute_error():
    ref, pred = _synthetic_batch()
    ref.stress_weight = torch.tensor([0.0, 1.0, 2.0])
    actual = WeightedLoss({"stress_l1": {"weight": 100.0}})(ref, pred)
    expected = 100.0 * (
        ref.stress_weight.view(-1, 1, 1) * (ref["stress"] - pred["stress"]).abs()
    ).mean()
    np.testing.assert_allclose(actual.item(), expected.item(), rtol=0, atol=1e-12)


def test_energy_l1_is_per_atom_and_weighted():
    ref, pred = _synthetic_batch()
    ref.weight = torch.tensor([0.5, 1.0, 2.0])
    ref.energy_weight = torch.tensor([1.0, 0.0, 1.0])
    actual = WeightedLoss({"energy_per_atom_l1": {"weight": 1.0}})(ref, pred)
    expected = (
        ref.weight * ref.energy_weight
        * (ref["energy"] - pred["energy"]).abs() / ATOMS_PER_STRUCT
    ).mean()
    np.testing.assert_allclose(actual.item(), expected.item(), rtol=0, atol=1e-12)


def test_forces_l2norm_honours_config_weights():
    """Deliberate deviation from upstream's mean_normed_error_forces, which ignores
    ref.weight and ref.forces_weight: a zero-weighted config must drop out."""
    ref, pred = _synthetic_batch()
    ref.forces_weight = torch.tensor([1.0, 0.0, 1.0])
    actual = WeightedLoss({"forces_l2norm": {"weight": 1.0}})(ref, pred)

    norms = torch.linalg.vector_norm(ref["forces"] - pred["forces"], dim=-1)
    mask = torch.repeat_interleave(ref.forces_weight, ATOMS_PER_STRUCT)
    expected = (mask * norms).sum() / (NUM_STRUCTS * ATOMS_PER_STRUCT)
    np.testing.assert_allclose(actual.item(), expected.item(), rtol=0, atol=1e-12)

    # the norm, not the squared norm: scaling the error scales the loss linearly
    pred2 = dict(pred, forces=ref["forces"] + 2.0 * (pred["forces"] - ref["forces"]))
    doubled = WeightedLoss({"forces_l2norm": {"weight": 1.0}})(ref, pred2)
    np.testing.assert_allclose(doubled.item(), 2.0 * actual.item(), rtol=1e-12)


@pytest.mark.parametrize("name", ["energy_per_atom_l1", "forces_l2norm", "stress_l1"])
def test_bare_weight_form_works(name):
    """Plain functions, so check_args' bare `name: 10.0` -> {weight: 10.0} is fine."""
    ref, pred = _synthetic_batch()
    loss = WeightedLoss({name: {"weight": 10.0}})(ref, pred)
    assert torch.isfinite(loss) and loss.item() > 0


def _with_multipoles(scale: float, seed: int = 5):
    """The synthetic batch plus [n_atoms, 4] multipoles (q and a dipole), errors ~ scale."""
    ref, pred = _synthetic_batch()
    g = torch.Generator().manual_seed(seed)
    n = NUM_STRUCTS * ATOMS_PER_STRUCT
    ref.density_coefficients = torch.randn(n, 4, generator=g, dtype=torch.float64)
    ref.density_coefficients_weight = torch.ones(NUM_STRUCTS, dtype=torch.float64)
    pred = dict(
        pred,
        density_coefficients=ref.density_coefficients
        + scale * torch.randn(n, 4, generator=g, dtype=torch.float64),
    )
    return ref, pred


def test_multipole_huber_is_half_the_mse_for_small_errors():
    ref, pred = _with_multipoles(scale=1e-6)
    huber = WeightedLoss({"atomic_multipoles_huber": {"weight": 1.0, "huber_delta": 0.01}})
    mse = WeightedLoss({"atomic_multipoles": {"weight": 1.0}})
    np.testing.assert_allclose(huber(ref, pred).item(), 0.5 * mse(ref, pred).item(), rtol=1e-12)


def test_multipole_huber_is_linear_for_large_errors():
    delta = 0.01
    ref, pred = _with_multipoles(scale=10.0)
    err = (pred["density_coefficients"] - ref.density_coefficients).abs()
    assert bool((err > delta).all())
    expected = (delta * (err - 0.5 * delta)).mean()
    actual = WeightedLoss({"atomic_multipoles_huber": {"weight": 1.0, "huber_delta": delta}})(
        ref, pred
    )
    np.testing.assert_allclose(actual.item(), expected.item(), rtol=1e-12)


def test_multipole_huber_drops_masked_frames():
    """density_coefficients_weight = 0 marks NaN-DDEC6 frames: they must add nothing."""
    ref, pred = _with_multipoles(scale=0.1)
    ref.density_coefficients_weight = torch.tensor([1.0, 0.0, 1.0])
    loss = WeightedLoss({"atomic_multipoles_huber": {"weight": 1.0, "huber_delta": 0.01}})
    before = loss(ref, pred).item()

    garbage = pred["density_coefficients"].clone()
    garbage[ATOMS_PER_STRUCT : 2 * ATOMS_PER_STRUCT] = 1e6
    after = loss(ref, dict(pred, density_coefficients=garbage)).item()
    np.testing.assert_allclose(after, before, rtol=1e-12)
    assert before > 0
