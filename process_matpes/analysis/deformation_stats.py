"""Stats on the integrated deformation charge rho_tilde(0) of AECCAR2-AECCAR1.

Sources:
- assembly_report.json violations (|rho(0)| > 0.05 e already flagged there)
- a random sample of npz files, for the overall distribution and correlations
  with heaviest element and grid resolution.
Writes a json summary + the sampled values for plotting.
"""

import json
import multiprocessing as mp
import re
import sys
from pathlib import Path

import numpy as np

SP = Path(__file__).parent
REPORT = Path("/global/cfs/cdirs/matgen/esoteric/matpes_processed/xyz/assembly_report.json")
SAMPLE_SIZE = 2000


def read_one(path):
    try:
        with np.load(path) as npz:
            rho0 = float(npz["fc_aeccar_diff"][0, 0])
            numbers = npz["numbers"]
            cell = npz["cell"]
            dims = npz["grid_dims"]
            # coarsest linear grid spacing across the three axes, in Angstrom
            spacing = float(np.max(np.linalg.norm(cell, axis=1) / dims))
            return {
                "name": Path(path).stem,
                "rho0": rho0,
                "natoms": int(len(numbers)),
                "zmax": int(numbers.max()),
                "nelect": float(npz["nelect"]),
                "spacing": spacing,
                "functional": str(npz["functional"]),
            }
    except Exception as exc:
        return {"name": Path(path).stem, "error": f"{type(exc).__name__}: {exc}"}


def main():
    # 1. Violators from the assembly report (message holds the rho(0) value).
    report = json.load(open(REPORT))
    pat = re.compile(r"^(\S+): aeccar_diff_nonzero: rho\(0\)=(-?[\d.]+)")
    violators = []
    for line in report["violations"]:
        m = pat.match(line)
        if m:
            violators.append((m.group(1), float(m.group(2))))
    print(f"report: {report['violation_counts'].get('aeccar_diff_nonzero')} violations total, "
          f"{len(violators)} present in (possibly truncated) list")

    # 2. Random npz sample.
    all_npz = Path(SP / "npz_list.txt").read_text().splitlines()
    rng = np.random.default_rng(42)
    sample = [all_npz[i] for i in rng.choice(len(all_npz), SAMPLE_SIZE, replace=False)]
    with mp.Pool(8) as pool:
        rows = pool.map(read_one, sample, chunksize=20)
    ok = [r for r in rows if "error" not in r]
    print(f"sampled {len(ok)}/{len(sample)} npz")

    rho0 = np.array([r["rho0"] for r in ok])
    abs0 = np.abs(rho0)
    summary = {
        "n_sampled": len(ok),
        "mean": float(rho0.mean()),
        "median_abs": float(np.median(abs0)),
        "p90_abs": float(np.percentile(abs0, 90)),
        "p99_abs": float(np.percentile(abs0, 99)),
        "max_abs": float(abs0.max()),
        "frac_above_0.05": float((abs0 > 0.05).mean()),
        "frac_above_0.01": float((abs0 > 0.01).mean()),
        "worst_violators": sorted(violators, key=lambda t: -abs(t[1]))[:15],
        "n_violators_in_report_list": len(violators),
    }
    json.dump({"summary": summary, "sample": ok, "violators": violators},
              open(SP / "deformation_stats.json", "w"))
    print(json.dumps(summary, indent=2)[:2000])


if __name__ == "__main__":
    main()
