"""The Huber loss terms must reproduce MACE's `universal` loss (the 0b3 recipe).

mace_scf composes losses per term from _LOSS_FUNCTIONS rather than selecting a monolithic
`--loss`, so UniversalLoss is decomposed into energy_per_atom_huber / forces_huber /
stress_huber. These tests pin that the decomposition is exact on ordinary data, and pin
the one deliberate deviation: the terms also carry ref.weight, which UniversalLoss omits.
"""

import numpy as np
import pytest
import torch
from mace.modules.loss import UniversalLoss

from mace_scf.electrostatics.loss import WeightedLoss

class _RefBatch:
    """Stands in for a Batch: the loss terms use both ref.energy and ref["energy"]."""

    def __init__(self, **fields):
        self.__dict__.update(fields)

    def __getitem__(self, key):
        return self.__dict__[key]


ATOMS_PER_STRUCT = 4
NUM_STRUCTS = 3

ENERGY_WEIGHT = 1.0
FORCES_WEIGHT = 10.0
STRESS_WEIGHT = 10.0
HUBER_DELTA = 0.01


def _synthetic_batch(seed: int = 3, config_weights=None):
    """A reference batch and a prediction dict, with forces spanning the conditional
    huber bins (the 100/200/300 eV/A thresholds), so the binning is actually exercised."""
    torch.set_default_dtype(torch.float64)
    g = torch.Generator().manual_seed(seed)
    n = ATOMS_PER_STRUCT * NUM_STRUCTS

    def r(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float64)

    ptr = torch.arange(0, n + 1, ATOMS_PER_STRUCT)
    batch_index = torch.repeat_interleave(
        torch.arange(NUM_STRUCTS), ATOMS_PER_STRUCT
    )
    if config_weights is None:
        config_weights = torch.ones(NUM_STRUCTS, dtype=torch.float64)

    ref_forces = r(n, 3)
    # push a few atoms into each of the higher bins
    ref_forces[1] *= 150.0
    ref_forces[5] *= 250.0
    ref_forces[9] *= 400.0

    ref = _RefBatch(
        energy=r(NUM_STRUCTS),
        forces=ref_forces,
        stress=r(NUM_STRUCTS, 3, 3),
        weight=config_weights,
        energy_weight=torch.ones(NUM_STRUCTS, dtype=torch.float64),
        forces_weight=torch.ones(NUM_STRUCTS, dtype=torch.float64),
        stress_weight=torch.ones(NUM_STRUCTS, dtype=torch.float64),
        ptr=ptr,
        batch=batch_index,
    )
    pred = {
        "energy": r(NUM_STRUCTS),
        "forces": r(n, 3),
        "stress": r(NUM_STRUCTS, 3, 3),
    }
    return ref, pred


def _weighted_loss():
    return WeightedLoss(
        {
            "energy_per_atom_huber": {
                "weight": ENERGY_WEIGHT,
                "huber_delta": HUBER_DELTA,
            },
            "forces_huber": {"weight": FORCES_WEIGHT, "huber_delta": HUBER_DELTA},
            "stress_huber": {"weight": STRESS_WEIGHT, "huber_delta": HUBER_DELTA},
        }
    )


def test_matches_universal_loss_with_uniform_config_weights():
    ref, pred = _synthetic_batch()
    actual = _weighted_loss()(ref, pred)
    expected = UniversalLoss(
        energy_weight=ENERGY_WEIGHT,
        forces_weight=FORCES_WEIGHT,
        stress_weight=STRESS_WEIGHT,
        huber_delta=HUBER_DELTA,
    )(ref, pred, ddp=False)

    np.testing.assert_allclose(
        actual.item(), expected.item(), rtol=0, atol=1e-12
    )


def test_deviates_from_universal_loss_with_non_uniform_config_weights():
    """Deliberate: the terms carry ref.weight so WeightedLoss's loss_weight_modifier,
    which discounts non-converged SCF configs, is not silently a no-op. UniversalLoss
    ignores ref.weight, so the two must differ once it is not all ones -- and because
    huber is non-linear, not merely by a rescaling."""
    weights = torch.tensor([0.25, 1.0, 4.0], dtype=torch.float64)
    ref, pred = _synthetic_batch(config_weights=weights)

    actual = _weighted_loss()(ref, pred)
    expected = UniversalLoss(
        energy_weight=ENERGY_WEIGHT,
        forces_weight=FORCES_WEIGHT,
        stress_weight=STRESS_WEIGHT,
        huber_delta=HUBER_DELTA,
    )(ref, pred, ddp=False)

    assert abs(actual.item() - expected.item()) > 1e-6


def test_huber_energy_differs_from_mse_on_large_errors():
    """Sanity that the huber term is not just the MSE one under a different name."""
    ref, pred = _synthetic_batch()
    pred["energy"] = ref["energy"] + 100.0  # far outside huber_delta

    huber = WeightedLoss(
        {"energy_per_atom_huber": {"weight": 1.0, "huber_delta": HUBER_DELTA}}
    )(ref, pred)
    mse = WeightedLoss({"energy_per_atom": {"weight": 1.0}})(ref, pred)

    assert huber.item() < mse.item()


def test_bare_weight_form_is_rejected_for_option_taking_terms():
    """A weight-only entry takes WeightedLoss's `len(options) == 1` branch, which stores
    the registry value uncalled. For a class-valued term that means forward() would build
    an instance instead of returning a tensor -- silently, at the first training step.
    check_args.fill_default_train_settings normalises a bare `forces_huber: 10.0` to
    exactly this shape, so this is the form that actually reaches WeightedLoss."""
    with pytest.raises(ValueError, match="needs the dict form"):
        WeightedLoss({"forces_huber": {"weight": 10.0}})
