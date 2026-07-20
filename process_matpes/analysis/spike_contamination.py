"""How much of each stored Fourier coefficient is spike contamination?

Method: take the raw AECCAR2-AECCAR1 grid, replace the spike voxels (detected
as |value| above a robust threshold, all sitting at nuclei) with the mean of
their 6 grid neighbours, and re-FT. The difference between raw and cleaned
coefficients is the spike's contribution, per k. If cleaning also restores
rho(0) ~ 0, the spikes account for the whole integral defect.

Run for the worst violator (Cl8U24) and the typical frame (CCeRu3).
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pymatgen.io.vasp import Chgcar

sys.path.insert(0, "/global/u1/t/twarford/dev/mace-scf/process_matpes")
from matpes_pipeline.kspace_fields import fourier_coefficients, reciprocal_cell

SP = Path(__file__).parent
NPZ_ROOT = Path("/global/cfs/cdirs/matgen/esoteric/matpes_processed/npz")
DATA_ROOT = Path("/global/cfs/cdirs/matgen/esoteric/matpes_chg_density_restore")

d = json.load(open(SP / "deformation_stats.json"))
worst = d["summary"]["worst_violators"][0][0]
typical = min(d["sample"], key=lambda r: abs(abs(r["rho0"]) - d["summary"]["median_abs"]))["name"]


def neighbour_mean(grid, mask):
    """Mean of the 6 axis neighbours for each masked voxel."""
    total = np.zeros_like(grid)
    for axis in range(3):
        for shift in (1, -1):
            total += np.roll(grid, shift, axis=axis)
    return np.where(mask, total / 6.0, grid)


def analyse(name):
    rec = dict(np.load(NPZ_ROOT / name.split("__")[0] / f"{name}.npz"))
    ld = DATA_ROOT / Path(*name.split("__"))
    a1 = Chgcar.from_file(ld / "AECCAR1.gz").data["total"]
    a2 = Chgcar.from_file(ld / "AECCAR2.gz").data["total"]
    raw = a2 - a1  # rho*V grid

    # spike voxels: |value| beyond 50x the 99.9th percentile of |raw|
    thresh = 50 * np.percentile(np.abs(raw), 99.9)
    mask = np.abs(raw) > thresh
    clean = neighbour_mean(raw, mask)

    triplets = rec["k_triplets"]
    c_raw = fourier_coefficients(raw, triplets)
    c_clean = fourier_coefficients(clean, triplets)
    contamination = c_raw - c_clean
    k_norm = np.linalg.norm(triplets @ reciprocal_cell(rec["cell"]), axis=1)

    mag = lambda c: np.hypot(c[:, 0], c[:, 1])
    print(f"{name.split('__')[-1]}: {mask.sum()} spike voxels "
          f"(natoms {len(rec['numbers'])}), "
          f"rho(0) raw {c_raw[0, 0]:+.3f} -> cleaned {c_clean[0, 0]:+.3f} e")
    return {
        "k": k_norm, "raw": mag(c_raw), "clean": mag(c_clean),
        "contam": mag(contamination), "n_spikes": int(mask.sum()),
        "rho0_raw": c_raw[0, 0], "rho0_clean": c_clean[0, 0],
    }


res_w = analyse(worst)
res_t = analyse(typical)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
bins = np.linspace(0, 12, 25)
for ax, res, title in [(axes[0], res_w, "Cl$_8$U$_{24}$ (worst violator)"),
                       (axes[1], res_t, "CCeRu$_3$ (typical frame)")]:
    mids = 0.5 * (bins[1:] + bins[:-1])
    for key, label, color in [("raw", "stored coefficient |ρ̃(k)|", "#3D6D9E"),
                              ("clean", "after removing spike voxels", "#4E9E7A"),
                              ("contam", "spike contribution", "#B23A42")]:
        med = [np.median(res[key][(res["k"] >= lo) & (res["k"] < hi)])
               for lo, hi in zip(bins[:-1], bins[1:])]
        ax.semilogy(mids, med, "o-", ms=3.5, lw=1.2, color=color, label=label)
    ax.set_xlabel(r"$|k|$  [Å$^{-1}$]")
    ax.set_title(title)
    ax.grid(alpha=0.25)
axes[0].set_ylabel("median coefficient magnitude  [e]")
axes[0].legend(fontsize=9)
fig.suptitle("Spike contamination of the stored Fourier coefficients, binned by |k|")
fig.tight_layout()
fig.savefig(SP / "deformation_spike_contamination.png", dpi=150)

# summary ratios in the low-k band the loss would upweight
for res, label in [(res_w, "worst"), (res_t, "typical")]:
    low = (res["k"] > 0) & (res["k"] < 4)
    ratio = np.median(res["contam"][low] / res["raw"][low])
    print(f"{label}: median contamination/|coefficient| for 0<|k|<4: {ratio:.2%}")
print("wrote deformation_spike_contamination.png")
