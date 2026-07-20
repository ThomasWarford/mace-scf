"""Stage 2: join stage-1 npz files with MatPES jsonl metadata, write extxyz.

Reads every ``<npz-dir>/block_*/*.npz`` produced by run_extraction, attaches
MatPES jsonl metadata (energies, band gap, DDEC6 per-atom data) by
(functional, matpes_id), and writes one extxyz per functional:
``MatPES-PBE.xyz`` and ``MatPES-R2SCAN.xyz``.

Work is parallelised over chunks of npz files; each worker writes partial xyz
files which are concatenated in deterministic (sorted) order at the end.
A JSON report with counts and sanity-check violations is written alongside.

Typical use (from the process_matpes directory):
    conda run -n dft python -m matpes_pipeline.assemble_xyz --workers 32
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
from ase import Atoms
from ase.io import write as ase_write

DEFAULT_NPZ_DIR = "/global/cfs/cdirs/matgen/esoteric/matpes_processed/npz"
DEFAULT_JSONL_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_OUT_DIR = "/global/cfs/cdirs/matgen/esoteric/matpes_processed/xyz"

JSONL_FILES = {
    "pbe": "MatPES-PBE-2025.2.jsonl",
    "r2scan": "MatPES-R2SCAN-2025.2.jsonl",
}
DDEC6_FIELDS = [
    "partial_charges",
    "spin_moments",
    "bond_order_sums",
    "rsquared_moments",
    "rcubed_moments",
    "rfourth_moments",
    "dipoles",  # (n_atoms, 3); kept last so scalars come first in the file
]
# kBar -> eV/Angstrom^3, and flip VASP's sign so positive = tensile (ASE convention).
KBAR_TO_EV_PER_A3 = -0.1 / 160.21766208

CHECK_NELECT_TOL = 1e-2  # |rho(0) of CHGCAR - NELECT|, electrons
# Per-atom, not absolute: the aeccar_diff defect is grid-quadrature error
# localized at each nucleus, so it accumulates with atom count (Spearman
# rho=0.44-0.53 vs natoms/nelect in a random sample) rather than with cell
# volume. 0.01 e/atom ~ separates the known violator population (median
# 0.019 e/atom) from typical background noise (median 0.0004 e/atom).
CHECK_AECCAR_TOL_PER_ATOM = 0.01  # |rho(0) of AECCAR2-AECCAR1| / natoms, electrons
CHECK_ENERGY_TOL = 1e-3  # |E_vasprun - E_jsonl|, eV
CHECK_FRAC_TOL = 1e-3  # site matching, fractional coordinates


def load_jsonl_index(jsonl_dir: Path) -> dict:
    """(functional, matpes_id) -> compact metadata dict, streaming the jsonls."""
    index = {}
    for functional, filename in JSONL_FILES.items():
        path = jsonl_dir / filename
        print(f"indexing {path} ...", flush=True)
        with open(path) as f:
            for line in f:
                entry = json.loads(line)
                sites = entry["structure"]["sites"]
                compact = {
                    "energy": entry["energy"],
                    "bandgap": entry["bandgap"],
                    "formation_energy_per_atom": entry["formation_energy_per_atom"],
                    "cohesive_energy_per_atom": entry["cohesive_energy_per_atom"],
                    "original_mp_id": entry["provenance"]["original_mp_id"],
                    "symbols": tuple(s["species"][0]["element"] for s in sites),
                    "frac_coords": np.array([s["abc"] for s in sites], dtype=np.float32),
                }
                ddec6 = entry.get("ddec6")
                if ddec6 is not None:
                    for field in DDEC6_FIELDS:
                        compact[f"ddec6_{field}"] = np.array(ddec6[field], dtype=np.float32)
                index[(functional, entry["matpes_id"])] = compact
    print(f"indexed {len(index)} jsonl entries", flush=True)
    return index


def sites_match(atoms: Atoms, meta: dict) -> bool:
    """Same species sequence and fractional coordinates (mod 1) as the jsonl entry."""
    if tuple(atoms.get_chemical_symbols()) != meta["symbols"]:
        return False
    delta = atoms.get_scaled_positions(wrap=False) - meta["frac_coords"]
    delta -= np.round(delta)  # nearest periodic image
    return bool(np.abs(delta).max() < CHECK_FRAC_TOL)


def build_atoms(npz: dict, meta, float32_fourier: bool) -> tuple:
    """One npz record (+ jsonl metadata or None) -> (Atoms, violations list)."""
    violations = []
    atoms = Atoms(
        numbers=npz["numbers"], positions=npz["positions"], cell=npz["cell"], pbc=True
    )
    atoms.arrays["REF_forces"] = np.asarray(npz["forces"])

    # Potential training targets are prefixed REF_; unprefixed keys are metadata.
    info = atoms.info
    info["REF_energy"] = float(npz["energy"])
    info["REF_stress"] = np.asarray(npz["stress_kbar"]) * KBAR_TO_EV_PER_A3
    info["stress_vasp_kbar"] = np.asarray(npz["stress_kbar"])
    info["REF_total_charge"] = 0.0

    # Fermi-level reference ingredients and precombined variants.
    info["REF_vasp_fermi_level"] = float(npz["efermi"])
    info["REF_vasp_fermi_level_plus_bet"] = float(npz["fermi_plus_bet"])
    info["REF_vasp_fermi_level_plus_alpha_bet"] = float(npz["fermi_plus_alpha_bet"])
    info["REF_vasp_fermi_level_plus_xc_g0"] = float(npz["fermi_plus_xc_g0"])
    info["REF_vasp_fermi_level_plus_alpha_bet_xc_g0"] = float(npz["fermi_plus_alpha_bet_xc_g0"])
    for key in ("alpha_bet", "alpha", "bet", "pscenc", "xc_g0"):
        info[key] = float(npz[key])

    for key in ("nelect", "bandgap", "vbm", "cbm", "magnetization", "nkpts", "encut", "kspacing"):
        info[key] = float(npz[key])

    # Density fourier coefficients (flattened; reshape recipe in the README).
    fourier_dtype = np.float32 if float32_fourier else np.float64
    info["n_k"] = int(len(npz["k_triplets"]))
    info["k_triplets"] = np.asarray(npz["k_triplets"]).ravel()
    for field in ("fc_chg_total", "fc_chg_diff", "fc_aeccar_diff"):
        key = field.replace("fc_", "REF_fourier_")
        info[key] = np.asarray(npz[field], dtype=fourier_dtype).ravel()
    info["fourier_grid_dims"] = np.asarray(npz["grid_dims"])

    info["REF_dipole"] = np.asarray(npz["dipole"])
    info["REF_dipole_diff_field"] = np.asarray(npz["dipole_diff_field"])

    matpes_id = str(npz["matpes_id"])
    functional = str(npz["functional"])
    info["matpes_id"] = matpes_id
    info["functional"] = functional
    info["provenance_path"] = str(npz["rel_path"])

    # Sanity checks on the fourier data.
    if abs(npz["fc_chg_total"][0, 0] - npz["nelect"]) > CHECK_NELECT_TOL:
        violations.append(f"nelect_mismatch: rho(0)={npz['fc_chg_total'][0, 0]:.4f}")
    aeccar_rho0_per_atom = abs(npz["fc_aeccar_diff"][0, 0]) / len(npz["numbers"])
    if aeccar_rho0_per_atom > CHECK_AECCAR_TOL_PER_ATOM:
        violations.append(
            f"aeccar_diff_nonzero: rho(0)={npz['fc_aeccar_diff'][0, 0]:.4f} "
            f"({aeccar_rho0_per_atom:.4f} e/atom)"
        )

    # jsonl metadata join.
    if meta is None:
        violations.append("no_jsonl_match")
        info["has_ddec6"] = False
        fill_ddec6_nan(atoms)
        return atoms, violations

    # Several restored calculations can share one matpes_id (e.g. the
    # volume-scaled mp-structure runs), and fractional coordinates are
    # invariant under volume scaling, so a site match alone is not enough:
    # only trust the jsonl join when the total energies agree.
    if abs(info["REF_energy"] - meta["energy"]) > CHECK_ENERGY_TOL:
        violations.append(
            f"energy_mismatch: vasprun={info['REF_energy']:.6f} jsonl={meta['energy']:.6f}"
        )
        info["has_ddec6"] = False
        fill_ddec6_nan(atoms)
        return atoms, violations

    info["matpes_bandgap"] = none_to_nan(meta["bandgap"])
    info["formation_energy_per_atom"] = none_to_nan(meta["formation_energy_per_atom"])
    info["cohesive_energy_per_atom"] = none_to_nan(meta["cohesive_energy_per_atom"])
    info["original_mp_id"] = str(meta["original_mp_id"])

    has_ddec6 = "ddec6_partial_charges" in meta
    if has_ddec6 and not sites_match(atoms, meta):
        violations.append("site_order_mismatch")
        has_ddec6 = False
    info["has_ddec6"] = has_ddec6
    if has_ddec6:
        atoms.arrays["REF_ddec6_charges"] = meta["ddec6_partial_charges"].astype(np.float64)
        for field in DDEC6_FIELDS[1:]:
            atoms.arrays[f"REF_ddec6_{field}"] = meta[f"ddec6_{field}"].astype(np.float64)
    else:
        fill_ddec6_nan(atoms)

    return atoms, violations


def fill_ddec6_nan(atoms: Atoms) -> None:
    """NaN placeholders so every frame has the same per-atom arrays."""
    n = len(atoms)
    atoms.arrays["REF_ddec6_charges"] = np.full(n, np.nan)
    for field in DDEC6_FIELDS[1:]:
        shape = (n, 3) if field == "dipoles" else (n,)
        atoms.arrays[f"REF_ddec6_{field}"] = np.full(shape, np.nan)


def none_to_nan(value) -> float:
    return float("nan") if value is None else float(value)


# The jsonl index is large (~GB). It is stored in this module-level global
# before the worker pool is created, so forked workers inherit it without any
# per-task pickling.
_INDEX = None


def process_chunk(chunk_id_and_files, parts_dir, float32_fourier):
    """Worker: one chunk of npz files -> partial xyz per functional + stats."""
    chunk_id, files = chunk_id_and_files
    stats = {
        "frames": {"pbe": 0, "r2scan": 0},
        "violations": [],  # (npz name, violation string)
        "ids": [],  # (functional, matpes_id, npz name), for duplicate detection
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
        atoms, violations = build_atoms(record, meta, float32_fourier)

        name = Path(path).stem
        stats["ids"].append((functional, matpes_id, name))
        stats["violations"].extend((name, v) for v in violations)

        part = parts_dir / f"part_{chunk_id:05d}_{functional}.xyz"
        ase_write(part, atoms, format="extxyz", append=True)
        stats["frames"][functional] += 1
    return stats


def concatenate_parts(parts_dir: Path, out_dir: Path) -> dict:
    """Join the partial xyz files in chunk order into one file per functional."""
    outputs = {"pbe": out_dir / "MatPES-PBE.xyz", "r2scan": out_dir / "MatPES-R2SCAN.xyz"}
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
    parser.add_argument(
        "--float64-fourier",
        action="store_true",
        help="write fourier coefficients in float64 instead of the default float32",
    )
    args = parser.parse_args()

    start = time.time()
    npz_files = sorted(glob.glob(f"{args.npz_dir}/block_*/*.npz"))
    print(f"found {len(npz_files)} npz files")
    if args.limit is not None:
        npz_files = npz_files[: args.limit]

    parts_dir = args.out_dir / "parts"
    if parts_dir.exists():
        shutil.rmtree(parts_dir)  # parts are append-mode; never reuse stale ones
    parts_dir.mkdir(parents=True)

    global _INDEX
    _INDEX = load_jsonl_index(args.jsonl_dir)

    chunks = [
        (i, npz_files[start : start + args.chunk_size])
        for i, start in enumerate(range(0, len(npz_files), args.chunk_size))
    ]
    worker = partial(
        process_chunk,
        parts_dir=parts_dir,
        float32_fourier=not args.float64_fourier,
    )

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

    # Duplicate matpes_ids (e.g. reruns of the same calculation).
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
    report_path = args.out_dir / "assembly_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"report: {report_path}")
    print(json.dumps({k: report[k] for k in
                      ("n_npz", "frames_written", "violation_counts", "n_duplicates",
                       "n_npz_errors", "elapsed_s")}, indent=2))


if __name__ == "__main__":
    main()
