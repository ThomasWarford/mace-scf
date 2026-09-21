"""Configs whose reference multipoles are NaN must not poison the multipole loss."""

import numpy as np
import pytest
import torch
from mace.data.utils import Configuration
from mace.tools import AtomicNumberTable

from mace_scf.data.new_atomic_data import ExtAtomicData


def _config(multipoles):
    return Configuration(
        atomic_numbers=np.array([1, 8]),
        positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        cell=np.eye(3) * 10.0,
        pbc=(True, True, True),
        properties={"energy": 0.0, "forces": np.zeros((2, 3)), "atomic_multipoles": multipoles},
        property_weights={"energy": 1.0, "forces": 1.0, "atomic_multipoles": 1.0},
    )


def _build(multipoles):
    return ExtAtomicData.from_config(
        _config(multipoles),
        z_table=AtomicNumberTable([1, 8]),
        cutoff=5.0,
        heads=["default"],
        atomic_multipoles_max_l=1,
    )


def test_clean_multipoles_keep_their_weight():
    data = _build(np.arange(8, dtype=float).reshape(2, 4))

    assert data.density_coefficients_weight.item() == pytest.approx(1.0)
    assert torch.allclose(
        data.density_coefficients, torch.arange(8, dtype=torch.get_default_dtype()).view(2, 4)
    )


def test_nan_multipoles_are_zeroed_and_dropped_from_the_loss():
    multipoles = np.arange(8, dtype=float).reshape(2, 4)
    multipoles[1, 2] = np.nan
    data = _build(multipoles)

    assert data.density_coefficients_weight.item() == pytest.approx(0.0)
    assert torch.isfinite(data.density_coefficients).all()
    assert data.density_coefficients[1, 2].item() == pytest.approx(0.0)


def test_all_nan_multipoles_are_dropped():
    data = _build(np.full((2, 4), np.nan))

    assert data.density_coefficients_weight.item() == pytest.approx(0.0)
    assert torch.isfinite(data.density_coefficients).all()


def test_masked_config_contributes_no_multipole_loss():
    from mace_scf.electrostatics.loss import weighted_mean_squared_error_dma
    from mace.tools.torch_geometric.batch import Batch

    multipoles = np.arange(8, dtype=float).reshape(2, 4)
    multipoles[0, 0] = np.nan
    batch = Batch.from_data_list([_build(multipoles)])
    pred = {"density_coefficients": torch.ones(2, 4, dtype=torch.get_default_dtype())}

    assert weighted_mean_squared_error_dma(batch, pred).item() == pytest.approx(0.0)
