"""Stage 1 worker: extract one MatPES VASP launcher directory into one .npz.

Reads the gzipped VASP outputs in a launcher directory (FW.json, INCAR,
OUTCAR, vasprun.xml, POTCAR, CHGCAR, AECCAR1, AECCAR2) and writes a single
compressed npz with the structure, energetics, Fermi-level reference
ingredients, grid-derived dipoles and Fourier coefficients of three density
fields (CHGCAR total, CHGCAR spin, AECCAR2 - AECCAR1).

Writes are atomic (tmp file + os.replace). Failures produce a
``<name>.fail.json`` next to the npz so a re-run can skip or retry cleanly.

Run on a single directory for debugging (from the process_matpes directory):
    conda run -n dft python -m matpes_pipeline.extract_launcher <launcher_dir> --out-dir /tmp/npz
"""

import argparse
import gzip
import json
import os
import re
import time
from pathlib import Path

import numpy as np
from pymatgen.io.vasp.inputs import Incar, Potcar
from pymatgen.io.vasp.outputs import Chgcar, Vasprun

from matpes_pipeline.kspace_fields import (
    enumerate_k_triplets,
    fourier_coefficients,
    grid_electron_moment,
)

# " E-fermi :   3.7644     XC(G=0): -10.0804     alpha+bet :-10.6577"
# (note: no space before a negative alpha+bet value)
FERMI_LINE = re.compile(
    r"E-fermi\s*:\s*([-\d.]+)\s+XC\(G=0\)\s*:\s*([-\d.]+)\s+alpha\+bet\s*:\s*([-\d.]+)"
)
PSCENC_LINE = re.compile(r"alpha Z\s+PSCENC\s*=\s*([-\d.]+)")
MAGNETIZATION_LINE = re.compile(r"number of electron\s+[-\d.]+\s+magnetization\s+([-\d.]+)")


def output_name(launcher_dir: Path) -> str:
    """Deterministic npz stem: path components from the block_* dir onward.

    Handles both layouts found in the restore:
    deep    block_*/launcher_*/launcher_*  ->  block__outer__inner
    shallow block_*/launcher_*             ->  block__launcher
    """
    parts = launcher_dir.parts
    block_index = max(i for i, p in enumerate(parts) if p.startswith("block_"))
    return "__".join(parts[block_index:])


def block_name(launcher_dir: Path) -> str:
    """The block_* path component, used as an output subdirectory."""
    return output_name(launcher_dir).split("__", 1)[0]


def parse_outcar(path: Path) -> dict:
    """Fermi-level reference ingredients and magnetization from OUTCAR."""
    with gzip.open(path, "rt") as f:
        text = f.read()

    fermi_match = FERMI_LINE.search(text)
    if fermi_match is None:
        raise ValueError("no 'E-fermi ... XC(G=0) ... alpha+bet' line in OUTCAR")
    efermi, xc_g0, alpha_bet = map(float, fermi_match.groups())

    pscenc_matches = PSCENC_LINE.findall(text)
    if not pscenc_matches:
        raise ValueError("no 'alpha Z  PSCENC' line in OUTCAR")
    pscenc = float(pscenc_matches[-1])

    mag_matches = MAGNETIZATION_LINE.findall(text)
    magnetization = float(mag_matches[-1]) if mag_matches else np.nan

    return {
        "efermi": efermi,
        "xc_g0": xc_g0,
        "alpha_bet": alpha_bet,
        "pscenc": pscenc,
        "magnetization": magnetization,
    }


def parse_functional(incar: Incar, fw_job_name: str) -> str:
    if str(incar.get("METAGGA", "")).lower() == "r2scan":
        return "r2scan"
    if str(incar.get("GGA", "")).lower() == "pe":
        return "pbe"
    # Some INCARs carry neither tag; fall back to the fireworks job name,
    # e.g. "... MatPES GGA static" / "... MatPES meta-GGA static".
    if "meta-GGA static" in fw_job_name:
        return "r2scan"
    if "GGA static" in fw_job_name:
        return "pbe"
    raise ValueError(
        f"unrecognised functional: GGA={incar.get('GGA')} METAGGA={incar.get('METAGGA')} "
        f"fw_job_name={fw_job_name!r}"
    )


def per_atom_zval(potcar: Potcar, structure) -> np.ndarray:
    """Valence charge of the pseudopotential ion for each site, in order."""
    zval_by_element = {p.element: p.zval for p in potcar}
    return np.array([zval_by_element[site.specie.symbol] for site in structure])


def extract_one(launcher_dir, out_dir, k_cutoff: float, overwrite: bool = False) -> dict:
    """Extract one launcher directory to <out_dir>/<name>.npz.

    Returns {"name", "status": ok|skipped|failed, "reason", "elapsed"}.
    """
    launcher_dir, out_dir = Path(launcher_dir), Path(out_dir)
    name = output_name(launcher_dir)
    # One subdirectory per block: ~780k npz files in a single directory would
    # strain the filesystem and everything that lists it.
    block_dir = out_dir / block_name(launcher_dir)
    npz_path = block_dir / f"{name}.npz"
    start = time.time()

    if npz_path.exists() and not overwrite:
        return {"name": name, "status": "skipped", "reason": "exists", "elapsed": 0.0}

    try:
        block_dir.mkdir(parents=True, exist_ok=True)
        record = build_record(launcher_dir, k_cutoff)
        atomic_savez(npz_path, record)
        return {"name": name, "status": "ok", "reason": "", "elapsed": time.time() - start}
    except Exception as exc:
        fail = {"path": str(launcher_dir), "exc_type": type(exc).__name__, "message": str(exc)}
        atomic_write_json(block_dir / f"{name}.fail.json", fail)
        return {
            "name": name,
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
            "elapsed": time.time() - start,
        }


def build_record(launcher_dir: Path, k_cutoff: float) -> dict:
    """Parse all files of one launcher into a flat dict of npz-ready arrays."""
    with gzip.open(launcher_dir / "FW.json.gz", "rt") as f:
        fw_spec = json.load(f)["spec"]
    # matpes-mp-structures jobs have spec.matpes_id = None; for those the
    # jsonl uses the Materials Project id (spec.mp_id) as the matpes_id.
    matpes_id = fw_spec.get("matpes_id") or fw_spec.get("mp_id") or ""
    try:
        fw_job_name = fw_spec["_tasks"][0]["job"]["function"]["@bound"]["name"]
    except (KeyError, IndexError, TypeError):
        fw_job_name = ""

    incar = Incar.from_file(launcher_dir / "INCAR.gz")
    functional = parse_functional(incar, fw_job_name)

    outcar = parse_outcar(launcher_dir / "OUTCAR.gz")

    vasprun = Vasprun(
        launcher_dir / "vasprun.xml.gz",
        parse_dos=False,
        parse_projected_eigen=False,
        parse_potcar_file=False,
    )
    structure = vasprun.final_structure
    last_step = vasprun.ionic_steps[-1]
    bandgap, cbm, vbm, _ = vasprun.eigenvalue_band_properties
    nelect = float(vasprun.parameters["NELECT"])

    potcar = Potcar.from_file(launcher_dir / "POTCAR.gz")
    zvals = per_atom_zval(potcar, structure)

    chgcar = Chgcar.from_file(launcher_dir / "CHGCAR.gz")
    aeccar1 = Chgcar.from_file(launcher_dir / "AECCAR1.gz")
    aeccar2 = Chgcar.from_file(launcher_dir / "AECCAR2.gz")
    if not tuple(chgcar.dim) == tuple(aeccar1.dim) == tuple(aeccar2.dim):
        raise ValueError(
            f"grid_mismatch: CHGCAR {chgcar.dim} AECCAR1 {aeccar1.dim} AECCAR2 {aeccar2.dim}"
        )

    cell = structure.lattice.matrix
    chg_total = chgcar.data["total"]
    chg_diff = chgcar.data["diff"]
    aeccar_diff = aeccar2.data["total"] - aeccar1.data["total"]

    triplets = enumerate_k_triplets(cell, k_cutoff)

    # Dipole convention: cell origin, atoms wrapped into the cell,
    # p = sum_i ZVAL_i * R_i - integral of r * rho_valence(r) d3r  (e * Angstrom).
    wrapped_cart = (structure.frac_coords % 1.0) @ cell
    dipole = zvals @ wrapped_cart - grid_electron_moment(chg_total, cell)
    dipole_diff_field = -grid_electron_moment(aeccar_diff, cell)

    efermi, alpha_bet, pscenc = outcar["efermi"], outcar["alpha_bet"], outcar["pscenc"]
    alpha = pscenc / nelect
    bet = alpha_bet - alpha

    return {
        # structure
        "numbers": np.array(structure.atomic_numbers, dtype=np.int32),
        "positions": structure.cart_coords,
        "cell": np.asarray(cell),
        # energetics
        "energy": last_step["e_0_energy"],
        "forces": np.asarray(last_step["forces"]),
        "stress_kbar": np.asarray(last_step["stress"]),
        # fermi-level reference ingredients and precombined variants
        "efermi": efermi,
        "xc_g0": outcar["xc_g0"],
        "alpha_bet": alpha_bet,
        "pscenc": pscenc,
        "alpha": alpha,
        "bet": bet,
        "fermi_plus_bet": efermi + bet,
        "fermi_plus_alpha_bet": efermi + alpha_bet,
        "fermi_plus_xc_g0": efermi + outcar["xc_g0"],
        "fermi_plus_alpha_bet_xc_g0": efermi + alpha_bet + outcar["xc_g0"],
        # electronic structure extras
        "nelect": nelect,
        "bandgap": bandgap,
        "vbm": vbm,
        "cbm": cbm,
        "magnetization": outcar["magnetization"],
        "nkpts": len(vasprun.actual_kpoints),
        "encut": float(incar.get("ENCUT", np.nan)),
        "kspacing": float(incar.get("KSPACING", np.nan)),
        # density fourier coefficients
        "k_triplets": triplets,
        "fc_chg_total": fourier_coefficients(chg_total, triplets),
        "fc_chg_diff": fourier_coefficients(chg_diff, triplets),
        "fc_aeccar_diff": fourier_coefficients(aeccar_diff, triplets),
        "grid_dims": np.asarray(chgcar.dim, dtype=np.int32),
        # dipoles
        "dipole": dipole,
        "dipole_diff_field": dipole_diff_field,
        # provenance
        "matpes_id": matpes_id,
        "functional": functional,
        "rel_path": output_name(launcher_dir).replace("__", "/"),
    }


def atomic_savez(path: Path, record: dict) -> None:
    tmp = path.with_suffix(".npz.tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **record)
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("launcher_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--kspace-cutoff", type=float, default=12.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = extract_one(args.launcher_dir, args.out_dir, args.kspace_cutoff, args.overwrite)
    print(json.dumps(result, indent=2))
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
