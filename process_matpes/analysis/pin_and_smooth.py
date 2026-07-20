"""Does pinning rho(0)=0 + removing high-k components give a sane deformation
density? Test on the worst violator (Cl8U24) vs a typical frame (CCeRu3).

Treatments applied to the stored k <= 12 1/A coefficients:
- pin: set the k=0 coefficient to zero
- hard truncation: keep |k| <= kc for kc in {12, 8, 6, 4}
- Gaussian smoothing: multiply by exp(-sigma^2 k^2 / 2), sigma in {0.25, 0.5, 1.0}
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/global/u1/t/twarford/dev/mace-scf/process_matpes")
from matpes_pipeline.kspace_fields import density_on_grid, reciprocal_cell

SP = Path(__file__).parent
NPZ_ROOT = Path("/global/cfs/cdirs/matgen/esoteric/matpes_processed/npz")

d = json.load(open(SP / "deformation_stats.json"))
worst_name = d["summary"]["worst_violators"][0][0]
typical_name = min(d["sample"], key=lambda r: abs(abs(r["rho0"]) - d["summary"]["median_abs"]))["name"]


def load(name):
    rec = dict(np.load(NPZ_ROOT / name.split("__")[0] / f"{name}.npz"))
    rec["volume"] = abs(np.linalg.det(rec["cell"]))
    rec["k_norm"] = np.linalg.norm(rec["k_triplets"] @ reciprocal_cell(rec["cell"]), axis=1)
    return rec


def treated_grid(rec, kc=None, sigma=None):
    """Pinned + filtered density in e/A^3."""
    coeffs = rec["fc_aeccar_diff"].copy()
    coeffs[0] = 0.0  # pin the monopole (origin triplet is stored first)
    if kc is not None:
        coeffs[rec["k_norm"] > kc] = 0.0
    if sigma is not None:
        coeffs *= np.exp(-0.5 * sigma**2 * rec["k_norm"] ** 2)[:, None]
    dims = tuple(int(x) for x in rec["grid_dims"])
    return density_on_grid(rec["k_triplets"], coeffs, dims) / rec["volume"]


worst, typical = load(worst_name), load(typical_name)

# line through the deepest well of the unfiltered reconstruction (a U nucleus)
g0 = treated_grid(worst)
i0, j0, k0 = np.unravel_index(g0.argmin(), g0.shape)
axis_len = np.linalg.norm(worst["cell"], axis=1)
x = np.arange(g0.shape[0]) / g0.shape[0] * axis_len[0]

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))

ax = axes[0]
for kc, color in [(12, "#B23A42"), (8, "#C77E4E"), (6, "#3D6D9E"), (4, "#4E9E7A")]:
    g = treated_grid(worst, kc=kc)
    ax.plot(x, g[:, j0, k0], lw=1.2, color=color,
            label=f"k ≤ {kc} Å⁻¹  (min {g.min():+.2f})")
ax.set_xlabel("distance along a through the U site  [Å]")
ax.set_ylabel(r"$\rho$  [e/Å$^3$]")
ax.legend(fontsize=8)
ax.set_title("Cl$_8$U$_{24}$, pinned: hard truncation")

ax = axes[1]
for sigma, color in [(0.25, "#B23A42"), (0.5, "#3D6D9E"), (1.0, "#4E9E7A")]:
    g = treated_grid(worst, sigma=sigma)
    ax.plot(x, g[:, j0, k0], lw=1.2, color=color,
            label=f"σ = {sigma} Å  (min {g.min():+.3f})")
ax.set_xlabel("distance along a through the U site  [Å]")
ax.set_ylabel(r"$\rho$  [e/Å$^3$]")
ax.legend(fontsize=8)
ax.set_title("Cl$_8$U$_{24}$, pinned: Gaussian smoothing")

# amplitude comparison: worst vs typical under each treatment
ax = axes[2]
treatments = [("raw k≤12", {}), ("k≤6", {"kc": 6}), ("σ=0.5", {"sigma": 0.5}), ("σ=1.0", {"sigma": 1.0})]
width = 0.35
for offset, (rec, label, color) in [(-width / 2, (worst, "Cl$_8$U$_{24}$ (worst)", "#B23A42")),
                                    (width / 2, (typical, "CCeRu$_3$ (typical)", "#3D6D9E"))]:
    amps = [np.abs(treated_grid(rec, **kw)).max() for _, kw in treatments]
    ax.bar(np.arange(len(treatments)) + offset, amps, width, color=color, label=label)
ax.set_xticks(range(len(treatments)))
ax.set_xticklabels([t for t, _ in treatments])
ax.set_yscale("log")
ax.set_ylabel(r"max $|\rho|$  [e/Å$^3$]")
ax.legend(fontsize=8)
ax.set_title("peak amplitude: worst vs typical")

fig.suptitle("Pinning $\\tilde\\rho(0)=0$ + removing high-k components")
fig.tight_layout()
fig.savefig(SP / "deformation_pinned_smoothed.png", dpi=150)

# numbers
print(f"{'treatment':>12s} {'worst min/max':>22s} {'typical min/max':>22s} {'ratio of max|rho|':>18s}")
for label, kw in treatments:
    gw = treated_grid(worst, **kw)
    gt = treated_grid(typical, **kw)
    print(f"{label:>12s} {gw.min():+10.3f}/{gw.max():+8.3f}  {gt.min():+10.3f}/{gt.max():+8.3f}  "
          f"{np.abs(gw).max()/np.abs(gt).max():14.1f}x")
print("wrote deformation_pinned_smoothed.png")
