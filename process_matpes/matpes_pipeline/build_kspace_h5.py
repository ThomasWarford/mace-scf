"""Build an HDF5 sidecar of the k-point/Fourier density data stripped out of
the extxyz files by `strip_kspace_info.py`.

Rationale: those arrays vary in length per frame (n_k depends on cell
volume) and dominate the extxyz file size (~98%), while being needed only
for a future density-fitting loss, not the energy/forces training that
`mace_scf` does today. Storing them CSR-style -- one flat, concatenated
dataset per field plus a `frame_offsets` row-pointer array -- avoids both
the per-frame text-parsing cost that made ase.io slow on the original
combined files, and the metadata overhead of one HDF5 group per frame
(774,818 frames total).

Frame i's arrays are `k_triplets[frame_offsets[i]:frame_offsets[i+1]]`,
etc. `provenance_path` gives the same join key stored in the extxyz
`info["provenance_path"]`, so alignment can be verified independently of
processing order (join by string, not by assuming row order).

Iterates the npz files with the exact same `sorted(glob(...))` as
assemble_xyz.py, split by functional -- every npz becomes exactly one row
here, same as in the *-nofourier.xyz files (assemble_xyz.py writes a frame
for every npz regardless of jsonl/violation status), so in practice row i
here lines up with row i of the matching functional's nofourier.xyz. The
stored provenance_path lets a consumer confirm that rather than assume it.
"""

import argparse
import glob
import multiprocessing as mp
import time
from pathlib import Path

import h5py
import numpy as np

DEFAULT_NPZ_DIR = "/global/cfs/cdirs/matgen/esoteric/matpes_processed/npz"
DEFAULT_OUT_DIR = "/global/cfs/cdirs/matgen/esoteric/matpes_processed/xyz"

FIELDS = ("k_triplets", "fc_chg_total", "fc_chg_diff", "fc_aeccar_diff")
FIELD_WIDTH = {"k_triplets": 3, "fc_chg_total": 2, "fc_chg_diff": 2, "fc_aeccar_diff": 2}
FIELD_DTYPE = {
    "k_triplets": np.int32,
    "fc_chg_total": np.float32,
    "fc_chg_diff": np.float32,
    "fc_aeccar_diff": np.float32,
}


def read_one(path: str):
    try:
        with np.load(path) as npz:
            return {
                "ok": True,
                "path": path,
                "functional": str(npz["functional"]),
                "provenance_path": str(npz["rel_path"]),
                "grid_dims": np.asarray(npz["grid_dims"], dtype=np.int32),
                **{f: np.asarray(npz[f], dtype=FIELD_DTYPE[f]) for f in FIELDS},
            }
    except Exception as exc:
        return {"ok": False, "path": path, "error": f"{type(exc).__name__}: {exc}"}


class CsrWriter:
    """Buffers frames and flushes into growing, chunked HDF5 datasets."""

    def __init__(self, h5file: h5py.File, flush_every: int = 5000, compression="gzip"):
        self.f = h5file
        self.flush_every = flush_every
        self.compression = compression
        self.buf_provenance = []
        self.buf_grid_dims = []
        self.buf_fields = {f: [] for f in FIELDS}
        self.n_frames = 0
        self.n_k_total = 0
        self.offsets = [0]
        self._init_datasets()

    def _init_datasets(self):
        f = self.f
        self.ds_provenance = f.create_dataset(
            "provenance_path", shape=(0,), maxshape=(None,),
            dtype=h5py.string_dtype("utf-8"), chunks=True,
        )
        self.ds_grid_dims = f.create_dataset(
            "grid_dims", shape=(0, 3), maxshape=(None, 3), dtype=np.int32, chunks=True,
        )
        self.ds_fields = {}
        for field in FIELDS:
            w = FIELD_WIDTH[field]
            self.ds_fields[field] = f.create_dataset(
                field, shape=(0, w), maxshape=(None, w), dtype=FIELD_DTYPE[field],
                chunks=(65536, w), compression=self.compression,
            )

    def add(self, row: dict):
        self.buf_provenance.append(row["provenance_path"])
        self.buf_grid_dims.append(row["grid_dims"])
        for field in FIELDS:
            self.buf_fields[field].append(row[field])
        self.n_k_total += len(row["k_triplets"])
        self.offsets.append(self.n_k_total)
        self.n_frames += 1
        if len(self.buf_provenance) >= self.flush_every:
            self.flush()

    def flush(self):
        if not self.buf_provenance:
            return
        n_new = len(self.buf_provenance)
        prev_n = self.ds_provenance.shape[0]
        self.ds_provenance.resize((prev_n + n_new,))
        self.ds_provenance[prev_n:] = self.buf_provenance
        self.ds_grid_dims.resize((prev_n + n_new, 3))
        self.ds_grid_dims[prev_n:] = np.stack(self.buf_grid_dims)

        for field in FIELDS:
            chunk = np.concatenate(self.buf_fields[field], axis=0)
            prev_k = self.ds_fields[field].shape[0]
            self.ds_fields[field].resize((prev_k + len(chunk), FIELD_WIDTH[field]))
            self.ds_fields[field][prev_k:] = chunk
            self.buf_fields[field] = []
        self.buf_provenance = []
        self.buf_grid_dims = []

    def close(self):
        self.flush()
        self.f.create_dataset("frame_offsets", data=np.asarray(self.offsets, dtype=np.int64))
        self.f.attrs["n_frames"] = self.n_frames
        self.f.attrs["n_k_total"] = self.n_k_total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz-dir", type=Path, default=Path(DEFAULT_NPZ_DIR))
    ap.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--flush-every", type=int, default=5000)
    args = ap.parse_args()

    start = time.time()
    npz_files = sorted(glob.glob(f"{args.npz_dir}/block_*/*.npz"))
    print(f"found {len(npz_files)} npz files", flush=True)
    if args.limit is not None:
        npz_files = npz_files[: args.limit]

    tmp_paths = {
        "pbe": args.out_dir / "MatPES-PBE-kspace.h5.tmp",
        "r2scan": args.out_dir / "MatPES-R2SCAN-kspace.h5.tmp",
    }
    final_paths = {
        "pbe": args.out_dir / "MatPES-PBE-kspace.h5",
        "r2scan": args.out_dir / "MatPES-R2SCAN-kspace.h5",
    }
    files = {k: h5py.File(v, "w") for k, v in tmp_paths.items()}
    writers = {k: CsrWriter(f, flush_every=args.flush_every) for k, f in files.items()}

    n_ok, n_err = 0, 0
    errors = []
    with mp.Pool(args.workers) as pool:
        for result in pool.imap(read_one, npz_files, chunksize=50):
            if not result["ok"]:
                n_err += 1
                errors.append(result)
                continue
            writers[result["functional"]].add(result)
            n_ok += 1
            if n_ok % 200_000 == 0:
                elapsed = time.time() - start
                print(f"  {n_ok} ok, {n_err} errors, {elapsed:.0f}s elapsed", flush=True)

    for functional in writers:
        writers[functional].close()
        files[functional].close()
        tmp_paths[functional].replace(final_paths[functional])
        print(f"wrote {final_paths[functional]} ({writers[functional].n_frames} frames, "
              f"{writers[functional].n_k_total} total k-points)", flush=True)

    print(f"done: {n_ok} ok, {n_err} errors, {time.time()-start:.0f}s total", flush=True)
    if errors:
        err_path = args.out_dir / "kspace_h5_errors.txt"
        err_path.write_text("\n".join(f"{e['path']}: {e['error']}" for e in errors))
        print(f"wrote {len(errors)} errors to {err_path}", flush=True)


if __name__ == "__main__":
    main()
