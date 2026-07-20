"""Round-trip tests: rebuild the charge density and multipoles from an extxyz
frame using only what is saved in it.

Three levels:
- synthetic: an analytic Gaussian density pushed through the real production
  path (fourier_coefficients -> build_atoms -> extxyz write/read -> the README
  reshape recipe -> density_on_grid / evaluate_density) and compared to the
  analytic answer. Runs anywhere with ase installed.
- real data: the stage-1 npz of one sample launcher through build_atoms and
  extxyz, reconstructed and compared to the actual CHGCAR / AECCAR grids.
  Tolerances reflect what a k_cutoff of 12 1/Angstrom can represent: the
  smooth valence density is reproduced to ~0.3% relative L2, the spin density
  to ~3%, while AECCAR2-AECCAR1 keeps sharp near-core features above the
  cutoff (~43% relative L2 on this sample) -- its low moments are still
  accurate. Skipped when the CFS sample data is not reachable.
- production xyz: DDEC6 partitioned charges/multipoles read back from the
  assembled MatPES-PBE.xyz. Skipped when the file is not reachable.

Run with:  conda run -n dft python -m pytest tests/test_xyz_roundtrip.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

ase = pytest.importorskip("ase", reason="round-trip tests need ase for extxyz io")
from ase.io import iread, read as ase_read, write as ase_write

from matpes_pipeline.assemble_xyz import build_atoms
from matpes_pipeline.kspace_fields import (
    density_on_grid,
    enumerate_k_triplets,
    evaluate_density,
    fourier_coefficients,
    grid_electron_moment,
)
from test_kspace_fields import TRICLINIC, gaussian_on_grid

K_CUTOFF = 12.0

SAMPLE_LAUNCHER = Path(
    "/global/cfs/cdirs/matgen/esoteric/matpes_chg_density_restore/"
    "block_2024-02-21-01-03-25-062171/launcher_2024-02-22-00-14-36-014143/"
    "launcher_2024-02-22-01-06-43-954475"
)
SAMPLE_NPZ = Path(
    "/global/cfs/cdirs/matgen/esoteric/matpes_processed/npz/"
    "block_2024-02-21-01-03-25-062171/"
    "block_2024-02-21-01-03-25-062171__launcher_2024-02-22-00-14-36-014143"
    "__launcher_2024-02-22-01-06-43-954475.npz"
)
PRODUCTION_XYZ = Path("/global/cfs/cdirs/matgen/esoteric/matpes_processed/xyz/MatPES-PBE.xyz")


def synthetic_npz_record(cell, dims):
    """A fake stage-1 npz dict whose density fields are analytic Gaussians.

    chg_total = two Gaussians integrating to nelect, chg_diff = their
    difference, aeccar_diff = a displaced +/- Gaussian pair (zero total).
    """
    a1, a2, sigma = 8.0, 4.0, 0.5
    r1 = np.array([0.30, 0.45, 0.60]) @ cell
    r2 = np.array([0.65, 0.55, 0.35]) @ cell
    g1 = gaussian_on_grid(cell, r1, a1, sigma, dims)
    g2 = gaussian_on_grid(cell, r2, a2, sigma, dims)
    grids = {
        "fc_chg_total": g1 + g2,
        "fc_chg_diff": g1 - g2,
        "fc_aeccar_diff": g1 / a1 - gaussian_on_grid(cell, r2, 1.0, sigma, dims),
    }
    triplets = enumerate_k_triplets(cell, K_CUTOFF)
    record = {
        "numbers": np.array([1, 1]),
        "positions": np.array([r1, r2]),
        "cell": cell,
        "forces": np.zeros((2, 3)),
        "energy": -1.0,
        "stress_kbar": np.zeros((3, 3)),
        "nelect": a1 + a2,
        "magnetization": a1 - a2,
        "k_triplets": triplets,
        "grid_dims": np.array(dims),
        "dipole": np.zeros(3),
        "dipole_diff_field": -grid_electron_moment(grids["fc_aeccar_diff"], cell),
        "matpes_id": "matpes-synthetic",
        "functional": "pbe",
        "rel_path": "synthetic",
    }
    for key in (
        "efermi", "fermi_plus_bet", "fermi_plus_alpha_bet", "fermi_plus_xc_g0",
        "fermi_plus_alpha_bet_xc_g0", "alpha_bet", "alpha", "bet", "pscenc",
        "xc_g0", "bandgap", "vbm", "cbm", "nkpts", "encut", "kspacing",
    ):
        record[key] = 0.0
    for field, grid in grids.items():
        record[field] = fourier_coefficients(grid, triplets)
    return record, grids


def roundtrip_through_xyz(record, tmp_path):
    """build_atoms -> extxyz write -> read back, as production does."""
    atoms, _ = build_atoms(record, meta=None, float32_fourier=True)
    path = tmp_path / "frame.xyz"
    ase_write(path, atoms, format="extxyz")
    return ase_read(path, format="extxyz")


def reshape_fourier(atoms, key):
    """The README reshape recipe: (triplets, [Re, Im] coefficients)."""
    n_k = int(atoms.info["n_k"])
    triplets = np.asarray(atoms.info["k_triplets"]).reshape(n_k, 3).astype(int)
    coeffs = np.asarray(atoms.info[key], dtype=np.float64).reshape(n_k, 2)
    return triplets, coeffs


def origin_coefficient(triplets, coeffs):
    """rho(k=0), located by triplet lookup, never by array position."""
    (idx,) = np.where(np.all(triplets == 0, axis=1))
    assert len(idx) == 1
    return coeffs[idx[0], 0]


def test_synthetic_roundtrip(tmp_path):
    cell, dims = TRICLINIC, (48, 54, 60)
    record, grids = synthetic_npz_record(cell, dims)
    atoms = roundtrip_through_xyz(record, tmp_path)

    triplets, coeffs = reshape_fourier(atoms, "REF_fourier_chg_total")
    np.testing.assert_array_equal(triplets, record["k_triplets"])

    # Monopole: total electron count is the k=0 coefficient.
    assert abs(origin_coefficient(triplets, coeffs) - atoms.info["nelect"]) < 1e-4

    # The reconstructed grid matches the original (analytic, band-limited)
    # density; the only losses are float32 storage and the Gaussian tail
    # beyond the cutoff.
    grid = grids["fc_chg_total"]
    recon = density_on_grid(triplets, coeffs, atoms.info["fourier_grid_dims"])
    assert np.linalg.norm(recon - grid) / np.linalg.norm(grid) < 1e-5

    # Point evaluation agrees with the same density sampled on the grid.
    rng = np.random.default_rng(0)
    idx = (rng.random((200, 3)) * dims).astype(int)
    points = (idx / dims) @ atoms.cell[:]
    values = evaluate_density(triplets, coeffs, atoms.cell[:], points)
    volume = atoms.get_volume()
    expected = grid[idx[:, 0], idx[:, 1], idx[:, 2]] / volume
    np.testing.assert_allclose(values, expected, atol=1e-5 * expected.max())

    # Dipole (first moment) of the reconstruction matches the original grid.
    np.testing.assert_allclose(
        grid_electron_moment(recon, cell), grid_electron_moment(grid, cell), atol=1e-4
    )

    # Spin monopole: rho_diff(0) is the total magnetization analogue.
    trip_d, coeffs_d = reshape_fourier(atoms, "REF_fourier_chg_diff")
    assert abs(origin_coefficient(trip_d, coeffs_d) - atoms.info["magnetization"]) < 1e-4

    # aeccar-like field: its dipole is recovered from the coefficients alone
    # and matches the stored REF_dipole_diff_field.
    trip_a, coeffs_a = reshape_fourier(atoms, "REF_fourier_aeccar_diff")
    recon_a = density_on_grid(trip_a, coeffs_a, atoms.info["fourier_grid_dims"])
    np.testing.assert_allclose(
        -grid_electron_moment(recon_a, cell), atoms.info["REF_dipole_diff_field"], atol=1e-4
    )


@pytest.mark.skipif(
    not (SAMPLE_NPZ.exists() and SAMPLE_LAUNCHER.exists()),
    reason="CFS sample launcher / npz not reachable",
)
def test_real_data_roundtrip(tmp_path):
    Chgcar = pytest.importorskip("pymatgen.io.vasp").Chgcar

    record = dict(np.load(SAMPLE_NPZ))
    atoms = roundtrip_through_xyz(record, tmp_path)
    cell = atoms.cell[:]
    dims = atoms.info["fourier_grid_dims"]

    chgcar = Chgcar.from_file(SAMPLE_LAUNCHER / "CHGCAR.gz")
    aec1 = Chgcar.from_file(SAMPLE_LAUNCHER / "AECCAR1.gz")
    aec2 = Chgcar.from_file(SAMPLE_LAUNCHER / "AECCAR2.gz")
    originals = {
        "REF_fourier_chg_total": chgcar.data["total"],
        "REF_fourier_chg_diff": chgcar.data["diff"],
        "REF_fourier_aeccar_diff": aec2.data["total"] - aec1.data["total"],
    }
    # Measured on this sample at k_cutoff 12: 3.0e-3, 2.9e-2, 4.3e-1.
    max_rel_l2 = {
        "REF_fourier_chg_total": 0.01,
        "REF_fourier_chg_diff": 0.10,
        "REF_fourier_aeccar_diff": 0.60,
    }
    # Measured moment errors: 1.6e-2, 9.5e-3, 5.0e-3 e*Angstrom.
    max_moment_err = {
        "REF_fourier_chg_total": 0.05,
        "REF_fourier_chg_diff": 0.03,
        "REF_fourier_aeccar_diff": 0.02,
    }

    for key, grid in originals.items():
        triplets, coeffs = reshape_fourier(atoms, key)
        recon = density_on_grid(triplets, coeffs, dims)
        rel_l2 = np.linalg.norm(recon - grid) / np.linalg.norm(grid)
        assert rel_l2 < max_rel_l2[key], f"{key}: rel L2 {rel_l2:.3e}"
        moment_err = np.abs(
            grid_electron_moment(recon, cell) - grid_electron_moment(grid, cell)
        ).max()
        assert moment_err < max_moment_err[key], f"{key}: moment err {moment_err:.3e}"

    # Monopoles from the saved coefficients alone.
    trip_t, coeffs_t = reshape_fourier(atoms, "REF_fourier_chg_total")
    assert abs(origin_coefficient(trip_t, coeffs_t) - atoms.info["nelect"]) < 1e-2
    trip_d, coeffs_d = reshape_fourier(atoms, "REF_fourier_chg_diff")
    assert abs(origin_coefficient(trip_d, coeffs_d) - atoms.info["magnetization"]) < 0.05
    trip_a, coeffs_a = reshape_fourier(atoms, "REF_fourier_aeccar_diff")
    assert abs(origin_coefficient(trip_a, coeffs_a)) < 0.05

    # The stored REF_dipole_diff_field is recovered from the coefficients.
    recon_a = density_on_grid(trip_a, coeffs_a, dims)
    np.testing.assert_allclose(
        -grid_electron_moment(recon_a, cell),
        atoms.info["REF_dipole_diff_field"],
        atol=0.02,
    )


@pytest.mark.skipif(not PRODUCTION_XYZ.exists(), reason="production xyz not reachable")
def test_production_xyz_ddec6():
    """Partitioned (DDEC6) charges and multipoles read back from the product."""
    checked = 0
    for atoms in iread(PRODUCTION_XYZ, index=":20", format="extxyz"):
        charges = atoms.arrays["REF_ddec6_charges"]
        dipoles = atoms.arrays["REF_ddec6_dipoles"]
        assert charges.shape == (len(atoms),)
        assert dipoles.shape == (len(atoms), 3)
        if not atoms.info["has_ddec6"]:
            assert np.isnan(charges).all()
            continue
        assert np.isfinite(charges).all() and np.isfinite(dipoles).all()
        # DDEC6 partial charges partition a neutral cell.
        assert abs(charges.sum()) < 0.05
        checked += 1
    assert checked > 0, "no frame with DDEC6 data in the first 20"
