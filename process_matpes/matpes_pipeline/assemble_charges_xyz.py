"""Stage 2b: lightweight "-charges" extxyz for the electrostatics fits.

Same join as `assemble_xyz` (stage-1 npz + MatPES jsonl -> per-frame Atoms,
with REF_forces/REF_stress/REF_energy/REF_total_charge/REF_dipole and DDEC6
per-atom data), plus two things the electrostatics models need that plain
`assemble_xyz` doesn't provide:

- Formal (integer) oxidation states per atom, guessed with pymatgen: try
  `BVAnalyzer.get_valences` (bond-valence, site-resolved) first, then fall
  back to `Composition.oxi_state_guesses` (composition-only, so uniform per
  element) if bond-valence can't find a charge-balanced assignment, then to
  all-zero if neither succeeds. Both attempts are recorded independently
  (`oxi_state_bv`, `oxi_state_composition`, NaN where that particular method
  didn't succeed) plus which one was actually used as `REF_formal_charges`
  (`oxi_state_method`: "bv_analyzer" / "composition_guess" / "fail_all_zero").
- `REF_multipoles`: per-atom l<=1 multipoles (4 components) built from the
  DDEC6 partitioning -- monopole = REF_ddec6_charges, l=1 block =
  REF_ddec6_dipoles reordered from Cartesian (x, y, z) to the e3nn/MACE real
  spherical harmonic convention (y, z, x) -- matching `hidden_irreps`-style
  1o ordering and `atomic_multipoles_max_l: 1` in the fit configs.

Only the Fourier/k-space fields (`REF_fourier_*`, `k_triplets`,
`fourier_grid_dims`, `n_k`) are dropped, the same set `strip_kspace_info.py`
removes for `-nofourier.xyz` -- they're multi-MB per frame and irrelevant to
the electrostatics fits. Everything else `assemble_xyz` writes (Fermi-level
ingredients, nelect/bandgap/vbm/cbm/magnetization/nkpts/encut/kspacing,
stress_vasp_kbar, REF_dipole_diff_field, ...) is kept as-is.

Typical use (from the process_matpes directory):
    conda run -n dft python -m matpes_pipeline.assemble_charges_xyz --workers 32
"""

import argparse
import glob
import json
import multiprocessing as mp
import shutil
import time
from functools import partial
from pathlib import Path

import numpy as np
from ase.io import write as ase_write
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.core import Structure

from matpes_pipeline.assemble_xyz import (
    DEFAULT_JSONL_DIR,
    DEFAULT_NPZ_DIR,
    DEFAULT_OUT_DIR,
    build_atoms,
    load_jsonl_index,
)

# The Fourier/k-space fields only -- same set strip_kspace_info.py removes
# for -nofourier.xyz. Everything else `build_atoms` sets is kept.
STRIP_INFO_KEYS = {
    "n_k",
    "k_triplets",
    "fourier_grid_dims",
    "REF_fourier_chg_total",
    "REF_fourier_chg_diff",
    "REF_fourier_aeccar_diff",
}


def guess_oxidation_states(atoms) -> tuple:
    """(REF_formal_charges, oxi_state_bv, oxi_state_composition, method).

    Bond-valence analysis (site-resolved) is tried first; composition-based
    guessing (uniform per element) is always also attempted so both columns
    are populated whenever they succeed, independent of which one is used
    for REF_formal_charges.
    """
    n = len(atoms)
    symbols = atoms.get_chemical_symbols()
    structure = Structure(
        lattice=atoms.cell.array,
        species=symbols,
        coords=atoms.get_positions(),
        coords_are_cartesian=True,
    )

    oxi_bv = np.full(n, np.nan)
    bv_ok = False
    try:
        oxi_bv = np.asarray(BVAnalyzer().get_valences(structure), dtype=np.float64)
        bv_ok = True
    except Exception:
        pass

    oxi_comp = np.full(n, np.nan)
    comp_ok = False
    try:
        guesses = structure.composition.oxi_state_guesses()
        if guesses:
            best = guesses[0]
            oxi_comp = np.array([best[sym] for sym in symbols], dtype=np.float64)
            comp_ok = True
    except Exception:
        pass

    if bv_ok:
        return oxi_bv, oxi_bv, oxi_comp, "bv_analyzer"
    if comp_ok:
        return oxi_comp, oxi_bv, oxi_comp, "composition_guess"
    return np.zeros(n), oxi_bv, oxi_comp, "fail_all_zero"


def add_multipoles(atoms) -> None:
    """REF_multipoles (n, 4): DDEC6 monopole + dipole in e3nn (y, z, x) order."""
    charges = atoms.arrays["REF_ddec6_charges"]
    dipoles = atoms.arrays["REF_ddec6_dipoles"]
    atoms.arrays["REF_multipoles"] = np.column_stack(
        [charges, dipoles[:, 1], dipoles[:, 2], dipoles[:, 0]]
    )


def build_charges_atoms(npz: dict, meta) -> tuple:
    atoms, violations = build_atoms(npz, meta, float32_fourier=True)

    formal, oxi_bv, oxi_comp, method = guess_oxidation_states(atoms)
    atoms.arrays["REF_formal_charges"] = formal
    atoms.arrays["oxi_state_bv"] = oxi_bv
    atoms.arrays["oxi_state_composition"] = oxi_comp
    atoms.info["oxi_state_method"] = method

    add_multipoles(atoms)

    for key in STRIP_INFO_KEYS:
        atoms.info.pop(key, None)

    return atoms, violations


# Inherited from assemble_xyz: the jsonl index lives here before the worker
# pool forks so workers share it without per-task pickling.
_INDEX = None


def process_chunk(chunk_id_and_files, parts_dir):
    chunk_id, files = chunk_id_and_files
    stats = {
        "frames": {"pbe": 0, "r2scan": 0},
        "violations": [],
        "ids": [],
        "npz_errors": [],
    }
    for path in files:
        try:
            with np.load(path) as npz:
                record = {k: npz[k] for k in npz.files}
        except Exception as exc:
            stats["npz_errors"].append(f"{Path(path).name}: {type(exc).__name__}: {exc}")
            continue

        functional = str(record["functional"])
        matpes_id = str(record["matpes_id"])
        meta = _INDEX.get((functional, matpes_id))
        atoms, violations = build_charges_atoms(record, meta)

        name = Path(path).stem
        stats["ids"].append((functional, matpes_id, name))
        stats["violations"].extend((name, v) for v in violations)

        part = parts_dir / f"part_{chunk_id:05d}_{functional}.xyz"
        ase_write(part, atoms, format="extxyz", append=True)
        stats["frames"][functional] += 1
    return stats


def concatenate_parts(parts_dir: Path, out_dir: Path) -> dict:
    outputs = {
        "pbe": out_dir / "MatPES-PBE-charges.xyz",
        "r2scan": out_dir / "MatPES-R2SCAN-charges.xyz",
    }
    for functional, out_path in outputs.items():
        parts = sorted(parts_dir.glob(f"part_*_{functional}.xyz"))
        with open(out_path, "wb") as out:
            for part in parts:
                with open(part, "rb") as f:
                    shutil.copyfileobj(f, out)
        print(f"{out_path}: {len(parts)} parts concatenated", flush=True)
    return {k: str(v) for k, v in outputs.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-dir", type=Path, default=Path(DEFAULT_NPZ_DIR))
    parser.add_argument("--jsonl-dir", type=Path, default=Path(DEFAULT_JSONL_DIR))
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--limit", type=int, default=None, help="only process the first N npz files")
    args = parser.parse_args()

    start = time.time()
    npz_files = sorted(glob.glob(f"{args.npz_dir}/block_*/*.npz"))
    print(f"found {len(npz_files)} npz files")
    if args.limit is not None:
        npz_files = npz_files[: args.limit]

    parts_dir = args.out_dir / "charges_parts"
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True)

    global _INDEX
    _INDEX = load_jsonl_index(args.jsonl_dir)

    chunks = [
        (i, npz_files[start : start + args.chunk_size])
        for i, start in enumerate(range(0, len(npz_files), args.chunk_size))
    ]
    worker = partial(process_chunk, parts_dir=parts_dir)

    totals = {"pbe": 0, "r2scan": 0}
    all_violations, all_ids, npz_errors = [], [], []
    with mp.Pool(args.workers) as pool:
        for i, stats in enumerate(pool.imap_unordered(worker, chunks), 1):
            for functional in totals:
                totals[functional] += stats["frames"][functional]
            all_violations.extend(stats["violations"])
            all_ids.extend(stats["ids"])
            npz_errors.extend(stats["npz_errors"])
            print(f"[chunk {i}/{len(chunks)}] frames={totals}", flush=True)

    outputs = concatenate_parts(parts_dir, args.out_dir)
    shutil.rmtree(parts_dir)

    seen, duplicates = {}, []
    for functional, matpes_id, name in sorted(all_ids, key=lambda t: t[2]):
        key = (functional, matpes_id)
        if key in seen:
            duplicates.append({"functional": functional, "matpes_id": matpes_id,
                               "npz": name, "first_seen_in": seen[key]})
        else:
            seen[key] = name

    violation_counts = {}
    for _, violation in all_violations:
        kind = violation.split(":")[0]
        violation_counts[kind] = violation_counts.get(kind, 0) + 1

    report = {
        "n_npz": len(npz_files),
        "frames_written": totals,
        "outputs": outputs,
        "violation_counts": violation_counts,
        "n_duplicates": len(duplicates),
        "n_npz_errors": len(npz_errors),
        "elapsed_s": round(time.time() - start),
        "violations": [f"{name}: {v}" for name, v in all_violations[:20000]],
        "duplicates": duplicates[:20000],
        "npz_errors": npz_errors[:2000],
    }
    report_path = args.out_dir / "assembly_charges_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"report: {report_path}")
    print(json.dumps({k: report[k] for k in
                      ("n_npz", "frames_written", "violation_counts", "n_duplicates",
                       "n_npz_errors", "elapsed_s")}, indent=2))


if __name__ == "__main__":
    main()
