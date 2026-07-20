# MatPES → mace-scf training data pipeline

Converts the restored MatPES VASP static calculations (with charge densities)
at `/global/cfs/cdirs/matgen/esoteric/matpes_chg_density_restore` into extxyz
training files, one per functional (PBE, r2SCAN). ~782k calculations.

## Pipeline

Stage 1 — one npz per launcher directory (parallel, resumable):

```bash
# spot test on a login node
conda run -n dft python -m matpes_pipeline.run_extraction --limit 20 --workers 4
# full run as a SLURM array job (8 nodes)
sbatch submit_extraction.sbatch
# after the array finishes: reprocess anything that failed
conda run -n dft python -m matpes_pipeline.run_extraction --retry-failures --workers 128
```

Each launcher produces `npz/<block>/<name>.npz` or, on error,
`<name>.fail.json` with the exception. Existing npz files are skipped, so
re-running any shard layout resumes cleanly.

Stage 2 — join with MatPES jsonl metadata, write extxyz:

```bash
sbatch submit_assembly.sbatch    # or, for small tests:
conda run -n dft python -m matpes_pipeline.assemble_xyz --limit 100 --workers 4
```

Writes `xyz/MatPES-PBE.xyz`, `xyz/MatPES-R2SCAN.xyz` and
`xyz/assembly_report.json` (frame counts, sanity-check violations, duplicate
matpes_ids, npz read errors). Frames with a duplicate `(functional,
matpes_id)` are all written; the report lists them for downstream filtering.

The MatPES jsonl files are expected in `data/` (see `process_matpes.ipynb`
for the download).

## What each frame contains

Naming rule: **potential training targets are prefixed `REF_`**; unprefixed
keys are metadata/inputs.

Per-atom `arrays`: `REF_forces` (eV/Å); DDEC6 data from the MatPES jsonls
(`REF_ddec6_charges`, `REF_ddec6_spin_moments`, `REF_ddec6_bond_order_sums`,
`REF_ddec6_rsquared_moments`, `REF_ddec6_rcubed_moments`,
`REF_ddec6_rfourth_moments`, `REF_ddec6_dipoles`) — NaN-filled with
`info["has_ddec6"]=False` where MatPES has no DDEC6 entry or the site order
could not be matched.

`info` scalars:

- `REF_energy` (eV, `e_0_energy`), `REF_stress` (eV/Å³, positive = tensile),
  `stress_vasp_kbar` (raw VASP sign/units), `REF_total_charge` (always 0.0).
- Fermi-level reference ingredients, all in eV, from the OUTCAR line
  `E-fermi : … XC(G=0): … alpha+bet : …` and `alpha Z  PSCENC = …`:
  `REF_vasp_fermi_level` (raw E-fermi) and precombined variants
  `REF_vasp_fermi_level_plus_bet`, `REF_vasp_fermi_level_plus_alpha_bet`,
  `REF_vasp_fermi_level_plus_xc_g0`,
  `REF_vasp_fermi_level_plus_alpha_bet_xc_g0`; raw ingredients (metadata)
  `alpha_bet`, `pscenc`, `alpha` (= PSCENC/NELECT),
  `bet` (= alpha_bet − alpha), `xc_g0`.
- `nelect`, `bandgap`, `vbm`, `cbm` (from vasprun eigenvalues),
  `magnetization` (μB, OUTCAR), `nkpts`, `encut`, `kspacing`.
- Provenance: `matpes_id`, `functional`, `provenance_path` (relative to the
  restore root), and from the jsonl join: `matpes_bandgap`,
  `formation_energy_per_atom`, `cohesive_energy_per_atom`, `original_mp_id`.
  For matpes-mp-structures jobs (FW.json has `spec.matpes_id = null`) the
  Materials Project id (`spec.mp_id`, e.g. `mp-8634`) is used as `matpes_id`,
  matching how the jsonls key those entries. Several restored calculations can
  share one id (volume-scaled reruns), so the jsonl metadata and DDEC6 arrays
  are only attached when the vasprun and jsonl total energies agree to
  1e-3 eV; mismatched frames are recorded as `energy_mismatch` violations and
  get NaN DDEC6 arrays with `has_ddec6=False`.

`info` dipoles (e·Å): `REF_dipole` = Σ_i ZVAL_i·R_i − ∫ r ρ_valence(r) d³r
using the CHGCAR total density, and `REF_dipole_diff_field` = −∫ r (ρ_AECCAR2
− ρ_AECCAR1) d³r. Convention: cell origin at (0,0,0), atoms wrapped into the
cell; in periodic boundary conditions these are origin/branch dependent.

## Fourier coefficients of the charge density

Stored for three fields: `REF_fourier_chg_total` (CHGCAR valence
pseudo-density), `REF_fourier_chg_diff` (spin density ρ↑−ρ↓),
`REF_fourier_aeccar_diff` (AECCAR2−AECCAR1, i.e. SCF valence minus superposed
atomic valence).

Convention (see `matpes_pipeline/kspace_fields.py`):

- Reciprocal basis `rcell = 2π·inv(cellᵀ)`; a Miller triplet `n` maps to
  `k = n @ rcell`.
- Half-space enumeration with `|k| ≤ k_cutoff` (default 12 Å⁻¹,
  `--kspace-cutoff`): the density is real, so only one of each ±k pair is
  stored. **Always match coefficients by triplet, never by array position** —
  the enumeration order depends on the cutoff.
- Stored value: `ρ̃(k) = ∫_cell ρ(r) e^(−ik·r) d³r` in electrons, so
  `ρ̃(0)` of `fourier_chg_total` = NELECT and `ρ̃(0)` of
  `fourier_aeccar_diff` = 0 in the continuum (not always in practice — see
  below). The repo-internal convention of
  `graph_longrange.kspace.evaluate_fourier_series_at_points_flat` is this
  value × (2π)³/V.

### `aeccar_diff` core-region spikes

AECCAR1/2 are VASP's *all-electron* reconstructed densities (unlike CHGCAR,
the smooth pseudo/valence density), so they carry the sharply-peaked density
near heavy-element nuclei. A finite FFT grid can't integrate that peak
exactly, so `ρ̃(0)` of `fourier_aeccar_diff` — exactly 0 in the continuum,
since AECCAR2 and AECCAR1 both integrate to NELECT — comes out nonzero for a
minority of frames, up to several electrons in the worst cases (uranium,
lanthanide, actinide compounds).

Investigated in `analysis/`: the defect isn't smooth grid noise, it's
single-voxel spikes sitting exactly at heavy nuclei — confirmed one grid
point wide across every severity level sampled (`raw_spikes.py`,
`spike_contamination.py`). A single-voxel spike has a flat Fourier spectrum,
so it contaminates *every* stored k roughly equally, not just the monopole —
to an atom-centered multipole model this looks like a spurious point charge
sitting on that nucleus, not just a bad total. That ruled out fixing the grid
directly (`pin_and_smooth.py`): neighbour-voxel averaging can't undo a defect
spread across the whole spectrum, and a fixed voxel threshold can spuriously
trigger on frames that were already fine.

Frames are instead *flagged*, per atom, via `assemble_xyz.py`'s
`CHECK_AECCAR_TOL_PER_ATOM` (`|ρ̃(0)| / n_atoms > 0.01 e/atom`, recorded as an
`aeccar_diff_nonzero` violation in `assembly_report.json`). Per atom, not per
volume or as an absolute cell total: it's a per-nucleus quadrature error that
accumulates with atom count (Spearman ρ≈0.5 vs NELECT/natoms in a random
sample, ρ≈0.3 vs volume and only because volume tracks natoms in bulk
structures) — padding a cell with vacuum can't dilute an error that lives at
the nuclei, so a per-volume or flat per-cell threshold is physically wrong in
both directions: it misses small contaminated cells (the newly-caught
population under the per-atom rule has a median of 3 atoms) and over-flags
large ones whose absolute error is just ordinary per-atom noise summed over
many atoms (~20% of the frames flagged under the old flat 0.05 e cutoff were
released at 0.01 e/atom — their per-atom rate sits inside background noise).
Currently flags 19,040/774,818 frames (2.46%).

**`REF_fourier_aeccar_diff` is stored raw** — pinning or masking the monopole
(or the whole channel) is left to the consumer. Check the
`aeccar_diff_nonzero` violations in `assembly_report.json` before trusting
this field for a given frame; `chg_total` and `chg_diff` come from CHGCAR,
the smooth pseudo-density, and aren't affected by this failure mode.

extxyz `info` only handles 1-D arrays reliably, so everything is flattened
row-major. Reshape recipe:

```python
n_k = atoms.info["n_k"]
triplets = atoms.info["k_triplets"].reshape(n_k, 3)              # int Miller indices
coeffs = atoms.info["REF_fourier_aeccar_diff"].reshape(n_k, 2)   # [Re, Im]
k_vectors = triplets @ (2 * np.pi * np.linalg.inv(atoms.cell[:].T))
```

`fourier_grid_dims` records the FFT grid the coefficients came from.
Coefficients are float32 in the xyz (`--float64-fourier` to change).

## Tests

```bash
conda run -n dft python -m pytest tests/
```

`test_kspace_fields.py` checks the k-space conventions against brute-force
enumeration and analytic Gaussians. `test_xyz_roundtrip.py` checks that the
density and its multipoles are reproducible **from the saved xyz alone**:
a synthetic Gaussian density through the full production path
(coefficients → extxyz → reshape recipe → reconstruction, error < 1e-5),
plus integration tests against one real launcher (skipped off CFS). Measured
reconstruction accuracy at the k-cutoff of 12 Å⁻¹, relative L2 on the grids:
`chg_total` ~0.3%, `chg_diff` ~3%, `aeccar_diff` ~43% (near-core features lie
above the cutoff; its monopole/dipole are still accurate to <0.01 e·Å).
