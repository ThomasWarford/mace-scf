"""Tests for kspace_fields against brute-force enumeration and analytic Gaussians."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from matpes_pipeline.kspace_fields import (
    enumerate_k_triplets,
    fourier_coefficients,
    grid_electron_moment,
    reciprocal_cell,
)

CUBIC = 5.0 * np.eye(3)
TRICLINIC = np.array([[6.0, 0.0, 0.0], [1.2, 5.5, 0.0], [-0.8, 2.1, 7.3]])


def brute_force_half_space(cell, k_cutoff):
    """All triplets with |k| <= k_cutoff, keeping one of each +/-n pair."""
    rcell = reciprocal_cell(cell)
    kept = set()
    m = 30  # generous fixed search box for these small test cells
    for n1 in range(-m, m + 1):
        for n2 in range(-m, m + 1):
            for n3 in range(-m, m + 1):
                if np.linalg.norm(np.array([n1, n2, n3]) @ rcell) > k_cutoff:
                    continue
                is_canonical = (n1, n2, n3) >= (0, 0, 0) and not (
                    n1 == 0 and (n2, n3) < (0, 0)
                ) and not (n1 == 0 and n2 == 0 and n3 < 0)
                if is_canonical:
                    kept.add((n1, n2, n3))
    return kept


@pytest.mark.parametrize("cell", [CUBIC, TRICLINIC], ids=["cubic", "triclinic"])
def test_triplet_enumeration(cell):
    k_cutoff = 6.0
    triplets = enumerate_k_triplets(cell, k_cutoff)
    as_set = {tuple(t) for t in triplets}

    assert len(as_set) == len(triplets), "duplicate triplets"
    assert (0, 0, 0) in as_set
    for t in as_set:
        neg = tuple(-x for x in t)
        assert t == neg or neg not in as_set, f"both +/-{t} present"
    norms = np.linalg.norm(triplets @ reciprocal_cell(cell), axis=1)
    assert np.all(norms <= k_cutoff)

    reference = brute_force_half_space(cell, k_cutoff)
    # Same k-content: either n or -n present for every reference triplet.
    ours_canonical = {t if t in reference else tuple(-x for x in t) for t in as_set}
    assert ours_canonical == reference


def gaussian_on_grid(cell, center, amplitude, sigma, dims):
    """Periodic Gaussian sampled on a grid, returned as a rho*V grid (VASP style)."""
    fracs = [np.arange(n) / n for n in dims]
    f1, f2, f3 = np.meshgrid(*fracs, indexing="ij")
    rho = np.zeros(dims)
    for i1 in (-1, 0, 1):  # nearest periodic images
        for i2 in (-1, 0, 1):
            for i3 in (-1, 0, 1):
                shifted = np.stack(
                    [f1 + i1, f2 + i2, f3 + i3], axis=-1
                ) @ cell - center
                r2 = np.sum(shifted**2, axis=-1)
                rho += np.exp(-r2 / (2 * sigma**2))
    rho *= amplitude / ((2 * np.pi) ** 1.5 * sigma**3)
    volume = abs(np.linalg.det(cell))
    return rho * volume


@pytest.mark.parametrize("cell", [CUBIC, TRICLINIC], ids=["cubic", "triclinic"])
def test_fourier_coefficients_analytic_gaussian(cell):
    amplitude, sigma = 2.5, 0.5
    center = np.array([0.3, 0.45, 0.6]) @ cell
    grid = gaussian_on_grid(cell, center, amplitude, sigma, dims=(48, 54, 60))

    triplets = enumerate_k_triplets(cell, k_cutoff=6.0)
    coeffs = fourier_coefficients(grid, triplets)

    k_vectors = triplets @ reciprocal_cell(cell)
    k2 = np.sum(k_vectors**2, axis=1)
    expected = amplitude * np.exp(-(sigma**2) * k2 / 2) * np.exp(-1j * k_vectors @ center)

    np.testing.assert_allclose(coeffs[:, 0], expected.real, atol=1e-6 * amplitude)
    np.testing.assert_allclose(coeffs[:, 1], expected.imag, atol=1e-6 * amplitude)
    assert abs(coeffs[0, 0] - amplitude) < 1e-6  # k=0 is the total integral


def test_nyquist_check_raises():
    grid = np.ones((8, 8, 8))
    too_large = np.array([[0, 0, 4]])
    with pytest.raises(ValueError, match="Nyquist"):
        fourier_coefficients(grid, too_large)


def test_grid_electron_moment_two_gaussians():
    cell = CUBIC
    amplitude, sigma = 1.7, 0.35
    r_plus = np.array([0.4, 0.5, 0.5]) @ cell
    r_minus = np.array([0.6, 0.5, 0.5]) @ cell
    dims = (50, 50, 50)
    grid = gaussian_on_grid(cell, r_plus, amplitude, sigma, dims) - gaussian_on_grid(
        cell, r_minus, amplitude, sigma, dims
    )
    moment = grid_electron_moment(grid, cell)
    np.testing.assert_allclose(moment, amplitude * (r_plus - r_minus), atol=1e-5)
