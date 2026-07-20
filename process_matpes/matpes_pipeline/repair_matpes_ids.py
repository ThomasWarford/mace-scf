"""One-off repair: fix npz files whose matpes_id is None or empty.

matpes-mp-structures jobs have spec.matpes_id = None in FW.json, so stage 1
(before the fix in extract_launcher.py) stored a pickled None — which
np.load(allow_pickle=False) then refuses to read — or an empty string. The
jsonl identifies those entries by the Materials Project id (spec.mp_id), so
this script rewrites each affected npz with matpes_id taken from
spec.matpes_id or spec.mp_id, leaving every other array untouched.

Scanning only opens the small matpes_id member of each npz zip; rewrites are
atomic (tmp + os.replace). Safe to re-run.

Typical use:
    conda run -n dft python -m matpes_pipeline.repair_matpes_ids --workers 64
"""

import os

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import argparse
import glob
import gzip
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

DEFAULT_DATA_ROOT = "/global/cfs/cdirs/matgen/esoteric/matpes_chg_density_restore"
DEFAULT_NPZ_DIR = "/global/cfs/cdirs/matgen/esoteric/matpes_processed/npz"

_DATA_ROOT = Path(DEFAULT_DATA_ROOT)


def matpes_id_from_fw_json(launcher_dir: Path) -> str:
    with gzip.open(launcher_dir / "FW.json.gz", "rt") as f:
        fw_spec = json.load(f)["spec"]
    return fw_spec.get("matpes_id") or fw_spec.get("mp_id") or ""


def repair_one(path: str) -> str:
    """Returns 'ok' (untouched), 'repaired', 'unfixable', or 'error: ...'."""
    try:
        with np.load(path, allow_pickle=True) as npz:
            record = {k: npz[k] for k in npz.files}
        current = record["matpes_id"]
        if current.dtype != object and str(current) != "":
            return "ok"

        new_id = matpes_id_from_fw_json(_DATA_ROOT / str(record["rel_path"]))
        if new_id == "":
            return "unfixable"  # FW.json really has no id; leave the npz alone

        record["matpes_id"] = new_id
        tmp = f"{path}.tmp"
        # savez appends ".npz" to bare paths, so hand it an open file instead.
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **record)
        os.replace(tmp, path)
        return "repaired"
    except Exception as exc:
        return f"error: {Path(path).name}: {type(exc).__name__}: {exc}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-dir", default=DEFAULT_NPZ_DIR)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    global _DATA_ROOT
    _DATA_ROOT = Path(args.data_root)

    npz_files = sorted(glob.glob(f"{args.npz_dir}/block_*/*.npz"))
    print(f"found {len(npz_files)} npz files")
    if args.limit is not None:
        npz_files = npz_files[: args.limit]

    counts = {"ok": 0, "repaired": 0, "unfixable": 0, "error": 0}
    errors = []
    start = time.time()
    with mp.Pool(args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(repair_one, npz_files, chunksize=16), 1):
            if result.startswith("error"):
                counts["error"] += 1
                errors.append(result)
            else:
                counts[result] += 1
            if i % 20000 == 0 or i == len(npz_files):
                rate = i / (time.time() - start)
                print(f"[{i}/{len(npz_files)}] {counts} ({rate:.0f}/s)", flush=True)

    for line in errors[:50]:
        print(line)
    print(f"done in {time.time() - start:.0f}s: {counts}")


if __name__ == "__main__":
    main()
