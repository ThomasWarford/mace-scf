"""Fourier coefficients and dipoles of VASP charge-density grids.

Conventions
-----------
Reciprocal lattice: ``rcell = 2*pi * inv(cell.T)`` (rows are b1, b2, b3), so a
Miller triplet ``n = (n1, n2, n3)`` maps to the k-vector ``k = n @ rcell``.
This matches ``rcell`` as computed in ``mace_scf/data/new_atomic_data.py``.

Because the density is real, rho(-k) = conj(rho(k)) and we only store a
half-space of triplets, enumerated the same way as
``graph_longrange.kspace.compute_k_vectors_flat``:
the origin (0,0,0), then (0,0,n3>0), then (0,n2>0,any n3), then
(n1>0, any n2, n3), each filtered to |k| <= k_cutoff. Triplets are stored
explicitly, so consumers should always match by triplet, never by position.

Stored coefficient: ``rho_k = integral over cell of rho(r) exp(-i k.r) d3r``,
in units of electrons (the k=0 coefficient of a CHGCAR 'total' grid is NELECT).
VASP grid files hold rho(r) * V_cell, so this is simply ``fftn(grid) / grid.size``.
The repo-internal convention used by
``graph_longrange.kspace.evaluate_fourier_series_at_points_flat`` is
``rho_k * (2*pi)**3 / V_cell``.
"""

import numpy as np

TWO_PI = 2.0 * np.pi


def reciprocal_cell(cell: np.ndarray) -> np.ndarray:
    """Rows are the reciprocal lattice vectors b1, b2, b3 (with 2*pi factor)."""
    return TWO_PI * np.linalg.inv(np.asarray(cell, dtype=float).T)


def enumerate_k_triplets(cell: np.ndarray, k_cutoff: float) -> np.ndarray:
    """Half-space Miller triplets (n_k, 3) int32 with |n @ rcell| <= k_cutoff.

    Order: (0,0,0) first, then (0,0,n3>0), (0,n2>0,any n3), (n1>0,any n2,n3).
    """
    rcell = reciprocal_cell(cell)

    # Upper bound per axis: the projection of b_i onto the direction normal to
    # the other two reciprocal vectors is 2*pi/|a_i|, so n_i can be at most
    # k_cutoff * |a_i| / (2*pi).
    cell = np.asarray(cell, dtype=float)
    nmax = np.ceil(k_cutoff * np.linalg.norm(cell, axis=1) / TWO_PI).astype(int)

    origin = [(0, 0, 0)]
    half_line = [(0, 0, n3) for n3 in range(1, nmax[2] + 1)]
    half_plane = [
        (0, n2, n3)
        for n2 in range(1, nmax[1] + 1)
        for n3 in range(-nmax[2], nmax[2] + 1)
    ]
    half_sphere = [
        (n1, n2, n3)
        for n1 in range(1, nmax[0] + 1)
        for n2 in range(-nmax[1], nmax[1] + 1)
        for n3 in range(-nmax[2], nmax[2] + 1)
    ]
    triplets = np.array(origin + half_line + half_plane + half_sphere, dtype=np.int32)

    k_norms = np.linalg.norm(triplets @ rcell, axis=1)
    return triplets[k_norms <= k_cutoff]


def fourier_coefficients(grid: np.ndarray, triplets: np.ndarray) -> np.ndarray:
    """Fourier coefficients (n_k, 2) [Re, Im] of a VASP rho*V grid.

    Returns rho_k = integral of rho(r) exp(-i k.r) d3r, in electrons.
    Raises ValueError if any triplet exceeds the grid's Nyquist limit.
    """
    triplets = np.asarray(triplets)
    dims = np.array(grid.shape)
    if np.any(np.abs(triplets) >= dims // 2):
        raise ValueError(
            f"k-vectors exceed Nyquist limit of grid {grid.shape}: "
            f"max |n| = {np.abs(triplets).max(axis=0)}, need < {dims // 2}"
        )
    ft = np.fft.fftn(grid) / grid.size
    values = ft[triplets[:, 0], triplets[:, 1], triplets[:, 2]]  # negative n wraps
    return np.stack([values.real, values.imag], axis=1)


def density_on_grid(triplets: np.ndarray, coeffs: np.ndarray, dims) -> np.ndarray:
    """Reconstruct the (band-limited) rho*V grid from stored Fourier data.

    Inverse of ``fourier_coefficients``: places each stored half-space
    coefficient and its conjugate at -n, then inverse-FFTs. The result is a
    real grid in the same rho*V convention VASP files use, containing exactly
    the Fourier content below the storage cutoff.
    """
    triplets = np.asarray(triplets)
    values = np.asarray(coeffs[:, 0] + 1j * coeffs[:, 1], dtype=complex)
    n_grid = int(np.prod(dims))
    ft = np.zeros(tuple(dims), dtype=complex)
    ft[triplets[:, 0], triplets[:, 1], triplets[:, 2]] = n_grid * values
    nonzero = ~np.all(triplets == 0, axis=1)  # k=0 has no distinct conjugate
    neg = -triplets[nonzero]
    ft[neg[:, 0], neg[:, 1], neg[:, 2]] = n_grid * np.conj(values[nonzero])
    return np.fft.ifftn(ft).real


def evaluate_density(
    triplets: np.ndarray, coeffs: np.ndarray, cell: np.ndarray, points: np.ndarray
) -> np.ndarray:
    """Evaluate the (band-limited) density at Cartesian points, in e/Angstrom^3.

    rho(r) = (1/V) * [rho_0 + sum_{k!=0} 2*(Re cos(k.r) - Im sin(k.r))],
    the real half-space form also used by
    ``graph_longrange.kspace.evaluate_fourier_series_at_points_flat``.
    """
    k_vectors = np.asarray(triplets) @ reciprocal_cell(cell)
    phases = points @ k_vectors.T  # [n_points, n_k]
    weights = np.full(len(k_vectors), 2.0)
    weights[np.all(np.asarray(triplets) == 0, axis=1)] = 1.0
    volume = abs(np.linalg.det(np.asarray(cell, dtype=float)))
    summand = np.cos(phases) * (weights * coeffs[:, 0]) - np.sin(phases) * (
        weights * coeffs[:, 1]
    )
    return summand.sum(axis=1) / volume


def grid_electron_moment(grid: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """First moment integral of r * rho(r) d3r over the cell, from a rho*V grid.

    Origin is the cell origin (0, 0, 0). Units: electrons * Angstrom.
    Computed axis-by-axis from fractional grid coordinates, never building a
    full (N, 3) mesh.
    """
    cell = np.asarray(cell, dtype=float)
    n1, n2, n3 = grid.shape
    # Mean of frac_i * rho * V over grid points = integral of frac_i * rho d3r.
    frac_means = np.array(
        [
            np.tensordot(np.arange(n1) / n1, grid.sum(axis=(1, 2)), axes=1),
            np.tensordot(np.arange(n2) / n2, grid.sum(axis=(0, 2)), axes=1),
            np.tensordot(np.arange(n3) / n3, grid.sum(axis=(0, 1)), axes=1),
        ]
    ) / grid.size
    return frac_means @ cell
