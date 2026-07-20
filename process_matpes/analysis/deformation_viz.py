"""Visualize deformation densities (AECCAR2-AECCAR1), especially frames where
the integral rho_tilde(0) is far from zero.

Figures:
- deformation_distribution.png: histogram of rho_tilde(0) + correlations with
  heaviest element and grid spacing.
- deformation_<n>_<name>.png: per-structure band-limited reconstruction
  (mid-slice through the strongest feature + planar averages), for the worst
  violators and one typical frame. For the single worst case the raw AECCAR
  grids are also read to show where the missing charge actually sits.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms

sys.path.insert(0, "/global/u1/t/twarford/dev/mace-scf/process_matpes")
from matpes_pipeline.kspace_fields import density_on_grid

SP = Path(__file__).parent
NPZ_ROOT = Path("/global/cfs/cdirs/matgen/esoteric/matpes_processed/npz")
DATA_ROOT = Path("/global/cfs/cdirs/matgen/esoteric/matpes_chg_density_restore")

data = json.load(open(SP / "deformation_stats.json"))
sample = data["sample"]
violators = data["violators"]


def npz_path(name):
    return NPZ_ROOT / name.split("__")[0] / f"{name}.npz"


def launcher_dir(name):
    return DATA_ROOT / Path(*name.split("__"))


def load_record(name):
    with np.load(npz_path(name)) as npz:
        return {k: npz[k] for k in npz.files if k != "matpes_id"}


# ---------------------------------------------------------------- figure 1
rho0 = np.array([r["rho0"] for r in sample])
zmax = np.array([r["zmax"] for r in sample])
spacing = np.array([r["spacing"] for r in sample])

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

ax = axes[0]
bins = np.linspace(-0.8, 0.8, 81)
ax.hist(np.clip(rho0, -0.8, 0.8), bins=bins, color="steelblue", log=True)
viol_vals = np.clip([v for _, v in violators], -0.8, 0.8)
ax.hist(viol_vals, bins=bins, color="crimson", alpha=0.6, log=True,
        label=f"report violators (n={len(violators)})")
for x in (-0.05, 0.05):
    ax.axvline(x, color="k", ls="--", lw=0.8)
ax.set_xlabel(r"$\tilde\rho(0)=\int(\rho_{AEC2}-\rho_{AEC1})\,d^3r$  [e]")
ax.set_ylabel("count")
ax.set_title(f"random sample of {len(sample)} npz (blue)\nclipped to ±0.8 e; dashed = ±0.05 e check")
ax.legend(fontsize=8)

ax = axes[1]
jitter = (np.random.default_rng(1).random(len(zmax)) - 0.5) * 0.6
ax.scatter(zmax + jitter, np.abs(rho0) + 1e-6, s=6, alpha=0.4)
ax.set_yscale("log")
ax.axhline(0.05, color="k", ls="--", lw=0.8)
ax.set_xlabel("heaviest element Z in structure")
ax.set_ylabel(r"$|\tilde\rho(0)|$  [e]")
ax.set_title("vs heaviest element")

ax = axes[2]
ax.scatter(spacing, np.abs(rho0) + 1e-6, s=6, alpha=0.4)
ax.set_yscale("log")
ax.axhline(0.05, color="k", ls="--", lw=0.8)
ax.set_xlabel("coarsest grid spacing  [Å]")
ax.set_ylabel(r"$|\tilde\rho(0)|$  [e]")
ax.set_title("vs FFT grid spacing")

fig.tight_layout()
fig.savefig(SP / "deformation_distribution.png", dpi=150)
plt.close(fig)
print("wrote deformation_distribution.png")

# ------------------------------------------------------- per-structure figures
worst = sorted(violators, key=lambda t: -abs(t[1]))[:3]
typical = min(sample, key=lambda r: abs(abs(r["rho0"]) - data["summary"]["median_abs"]))
cases = [(name, val, "worst violator") for name, val in worst]
cases.append((typical["name"], typical["rho0"], "typical frame"))

for i, (name, rho0_val, label) in enumerate(cases, 1):
    rec = load_record(name)
    cell = rec["cell"]
    dims = tuple(int(d) for d in rec["grid_dims"])
    volume = abs(np.linalg.det(cell))
    grid = density_on_grid(rec["k_triplets"], rec["fc_aeccar_diff"], dims) / volume
    atoms = Atoms(numbers=rec["numbers"], positions=rec["positions"], cell=cell, pbc=True)
    formula = atoms.get_chemical_formula()

    # slice through the voxel with the largest |rho| (band-limited)
    i0, j0, k0 = np.unravel_index(np.abs(grid).argmax(), grid.shape)
    slice_2d = grid[:, :, k0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    lim = np.abs(slice_2d).max()
    im = ax.imshow(slice_2d.T, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim,
                   extent=[0, 1, 0, 1], aspect="auto")
    fig.colorbar(im, ax=ax, label=r"$\rho$  [e/Å$^3$]")
    frac = atoms.get_scaled_positions(wrap=True)
    near = np.abs((frac[:, 2] - k0 / dims[2] + 0.5) % 1.0 - 0.5) < 0.05
    ax.scatter(frac[near, 0], frac[near, 1], c="k", s=30, marker="o",
               facecolors="none", label="atoms near slice")
    ax.set_xlabel("frac a")
    ax.set_ylabel("frac b")
    ax.set_title(f"band-limited slice at frac c = {k0 / dims[2]:.2f}")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    for axis, axis_label in enumerate("abc"):
        planes = tuple(j for j in range(3) if j != axis)
        profile = grid.mean(axis=planes)
        x = np.arange(dims[axis]) / dims[axis]
        ax.plot(x, profile, label=f"along {axis_label}")
    ax.axhline(rho0_val / volume, color="k", ls="--", lw=0.8,
               label=r"$\tilde\rho(0)/V$")
    ax.set_xlabel("fractional coordinate")
    ax.set_ylabel(r"planar-mean $\rho$  [e/Å$^3$]")
    ax.legend(fontsize=8)
    ax.set_title("planar averages")

    fig.suptitle(
        f"{label}: {formula}  ({rec['functional']})   "
        rf"$\tilde\rho(0)$ = {rho0_val:+.3f} e,  NELECT = {rec['nelect']:.0f},  "
        f"grid {dims}, natoms {len(atoms)}"
    )
    fig.tight_layout()
    out = SP / f"deformation_{i}_{formula}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out.name}: {label} {formula} rho0={rho0_val:+.4f}")

# ----------------------------------------- raw vs band-limited for the worst
name, rho0_val = worst[0]
rec = load_record(name)
try:
    from pymatgen.io.vasp import Chgcar

    ld = launcher_dir(name)
    a1 = Chgcar.from_file(ld / "AECCAR1.gz").data["total"]
    a2 = Chgcar.from_file(ld / "AECCAR2.gz").data["total"]
    cell = rec["cell"]
    volume = abs(np.linalg.det(cell))
    raw = (a2 - a1) / volume  # e/A^3
    dims = raw.shape
    bl = density_on_grid(rec["k_triplets"], rec["fc_aeccar_diff"], dims) / volume
    atoms = Atoms(numbers=rec["numbers"], positions=rec["positions"], cell=cell, pbc=True)
    frac = atoms.get_scaled_positions(wrap=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for axis, (ax, axis_label) in enumerate(zip(axes, "abc")):
        planes = tuple(j for j in range(3) if j != axis)
        x = np.arange(dims[axis]) / dims[axis]
        ax.plot(x, raw.mean(axis=planes), lw=0.8, label="raw AECCAR2−AECCAR1")
        ax.plot(x, bl.mean(axis=planes), lw=1.2, label="band-limited (k ≤ 12 Å⁻¹)")
        for f in frac[:, axis]:
            ax.axvline(f, color="gray", lw=0.5, alpha=0.5)
        ax.set_xlabel(f"frac {axis_label}")
        ax.set_ylabel(r"planar-mean $\rho$  [e/Å$^3$]")
        if axis == 0:
            ax.legend(fontsize=8)
    fig.suptitle(
        f"worst violator {atoms.get_chemical_formula()}: raw grids vs stored coefficients "
        rf"($\tilde\rho(0)$ = {rho0_val:+.3f} e; gray lines = atom planes)"
    )
    fig.tight_layout()
    fig.savefig(SP / "deformation_raw_vs_bandlimited.png", dpi=150)
    print("wrote deformation_raw_vs_bandlimited.png")
except Exception as exc:
    print(f"raw comparison skipped: {type(exc).__name__}: {exc}")
