"""Show the raw deformation-density spikes at the nuclei of the worst violator:
line cuts through a U atom (no averaging), raw vs band-limited, plus the raw
AECCAR1/AECCAR2 fields themselves along the same line."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pymatgen.io.vasp import Chgcar

sys.path.insert(0, "/global/u1/t/twarford/dev/mace-scf/process_matpes")
from matpes_pipeline.kspace_fields import density_on_grid

SP = Path(__file__).parent
NPZ_ROOT = Path("/global/cfs/cdirs/matgen/esoteric/matpes_processed/npz")
DATA_ROOT = Path("/global/cfs/cdirs/matgen/esoteric/matpes_chg_density_restore")

name = json.load(open(SP / "deformation_stats.json"))["summary"]["worst_violators"][0][0]
rec = dict(np.load(NPZ_ROOT / name.split("__")[0] / f"{name}.npz"))
ld = DATA_ROOT / Path(*name.split("__"))

a1 = Chgcar.from_file(ld / "AECCAR1.gz").data["total"]
a2 = Chgcar.from_file(ld / "AECCAR2.gz").data["total"]
cell = rec["cell"]
volume = abs(np.linalg.det(cell))
dims = a1.shape
raw = (a2 - a1) / volume  # e/A^3
bl = density_on_grid(rec["k_triplets"], rec["fc_aeccar_diff"], dims) / volume

# voxel of the most negative raw value = a U nucleus
i0, j0, k0 = np.unravel_index(raw.argmin(), raw.shape)
numbers = rec["numbers"]
frac_atoms = rec["positions"] @ np.linalg.inv(cell) % 1.0
voxel_frac = np.array([i0 / dims[0], j0 / dims[1], k0 / dims[2]])
delta = np.abs((frac_atoms - voxel_frac + 0.5) % 1.0 - 0.5)
nearest = np.argmin(np.linalg.norm(delta @ cell, axis=1))
print(f"worst voxel at frac {voxel_frac.round(3)}, nearest atom Z={numbers[nearest]}")
print(f"raw   min/max: {raw.min():.1f} / {raw.max():.1f} e/A^3")
print(f"band-limited min/max: {bl.min():.2f} / {bl.max():.2f} e/A^3")
print(f"AECCAR2/V at spike voxel: {a2[i0, j0, k0]/volume:.1f} e/A^3")
print(f"AECCAR1/V at spike voxel: {a1[i0, j0, k0]/volume:.1f} e/A^3")

axis_len = np.linalg.norm(cell, axis=1)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

# line cut along a through the U nucleus
x = np.arange(dims[0]) / dims[0] * axis_len[0]
ax = axes[0]
ax.plot(x, raw[:, j0, k0], lw=1.0, label="raw AECCAR2−AECCAR1")
ax.plot(x, bl[:, j0, k0], lw=1.4, label="band-limited (k ≤ 12 Å⁻¹)")
ax.set_xlabel("distance along a through the U site  [Å]")
ax.set_ylabel(r"$\rho$  [e/Å$^3$]")
ax.legend(fontsize=8)
ax.set_title("line cut through the nucleus (no averaging)")

# zoom near the nucleus, +/- 1 A
ax = axes[1]
x0 = i0 / dims[0] * axis_len[0]
m = np.abs(x - x0) < 1.0
ax.plot(x[m] - x0, raw[m, j0, k0], "o-", ms=3, lw=1.0, label="raw")
ax.plot(x[m] - x0, bl[m, j0, k0], lw=1.4, label="band-limited")
ax.set_xlabel("distance from U nucleus  [Å]")
ax.set_ylabel(r"$\rho$  [e/Å$^3$]")
ax.legend(fontsize=8)
ax.set_title("zoom: ±1 Å (markers = actual grid points)")

# the two AE fields themselves, log scale
ax = axes[2]
ax.semilogy(x[m] - x0, a2[m, j0, k0] / volume, "o-", ms=3, lw=1.0,
            label="AECCAR2 / V (SCF valence)")
ax.semilogy(x[m] - x0, a1[m, j0, k0] / volume, "s-", ms=3, lw=1.0,
            label="AECCAR1 / V (atomic superposition)")
ax.set_xlabel("distance from U nucleus  [Å]")
ax.set_ylabel(r"$\rho$  [e/Å$^3$]")
ax.legend(fontsize=8)
ax.set_title("the two all-electron valence fields at the core")

fig.suptitle(
    f"Cl$_8$U$_{{24}}$ raw deformation density at a uranium nucleus "
    f"(raw min {raw.min():.0f} e/Å$^3$, band-limited min {bl.min():.1f} e/Å$^3$)"
)
fig.tight_layout()
fig.savefig(SP / "deformation_raw_spikes.png", dpi=150)
print("wrote deformation_raw_spikes.png")
