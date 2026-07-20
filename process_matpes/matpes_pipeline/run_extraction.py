"""Stage 1 driver: extract all MatPES launcher directories in parallel.

Discovers launcher directories under the data root (~780k of them),
optionally takes an interleaved shard (for SLURM array jobs), and runs
``extract_launcher.extract_one`` over a multiprocessing pool. Every launcher
produces either a ``.npz`` or a ``.fail.json`` under ``<out-dir>/<block>/``,
so re-running (with any shard layout) resumes where it left off.

Typical use (from the process_matpes directory):
    conda run -n dft python -m matpes_pipeline.run_extraction --limit 20 --workers 4
or as one task of a SLURM array job (see submit_extraction.sbatch):
    ... run_extraction --num-shards 8 --shard-id $SLURM_ARRAY_TASK_ID
"""

import os

# One process per launcher; keep numpy/FFT single-threaded to avoid
# oversubscription. Must happen before numpy is first imported.
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import argparse
import glob
import json
import multiprocessing as mp
import time
from functools import partial
from pathlib import Path

from matpes_pipeline.extract_launcher import block_name, extract_one, output_name

DEFAULT_DATA_ROOT = "/global/cfs/cdirs/matgen/esoteric/matpes_chg_density_restore"
DEFAULT_OUT_DIR = "/global/cfs/cdirs/matgen/esoteric/matpes_processed/npz"


def discover_launchers(data_root: str, cache_file: Path) -> list:
    """All launcher dirs, sorted so shard membership is deterministic.

    Two layouts exist in the restore:
    deep    block_*/launcher_*/launcher_*/<VASP files>   (the vast majority)
    shallow block_*/launcher_*/<VASP files>
    A shallow launcher is recognised by containing FW.json.gz itself.

    Globbing ~800k paths on CFS takes minutes, so the result is cached to a
    text file shared by all array tasks. Delete the cache to re-discover.
    """
    if cache_file.exists():
        launchers = cache_file.read_text().splitlines()
        print(f"using cached launcher list {cache_file}")
        return launchers
    deep = glob.glob(f"{data_root}/block_*/launcher_*/launcher_*/FW.json.gz")
    shallow = glob.glob(f"{data_root}/block_*/launcher_*/FW.json.gz")
    launchers = sorted(str(Path(p).parent) for p in deep + shallow)
    tmp = cache_file.with_suffix(".txt.tmp")
    tmp.write_text("\n".join(launchers) + "\n")
    os.replace(tmp, cache_file)
    return launchers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None, help="only process the first N launchers")
    parser.add_argument("--kspace-cutoff", type=float, default=12.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        help="delete existing .fail.json records first so those launchers are reprocessed",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--shard-id",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)),
        help="which shard to process (defaults to $SLURM_ARRAY_TASK_ID or 0)",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("discovering launcher directories ...", flush=True)
    launchers = discover_launchers(args.data_root, args.out_dir / "launchers.txt")
    print(f"found {len(launchers)} launcher directories")
    launchers = launchers[args.shard_id :: args.num_shards]
    print(f"shard {args.shard_id}/{args.num_shards}: {len(launchers)} launchers")
    if args.limit is not None:
        launchers = launchers[: args.limit]

    if args.retry_failures:
        removed = 0
        for fail_file in args.out_dir.glob("*/*.fail.json"):
            fail_file.unlink()
            removed += 1
        print(f"removed {removed} .fail.json records for retry")

    def has_fail_record(launcher: str) -> bool:
        launcher = Path(launcher)
        return (args.out_dir / block_name(launcher) / f"{output_name(launcher)}.fail.json").exists()

    launchers = [l for l in launchers if not has_fail_record(l)]
    print(f"{len(launchers)} launchers to process after skipping failure records", flush=True)

    worker = partial(
        extract_one, out_dir=args.out_dir, k_cutoff=args.kspace_cutoff, overwrite=args.overwrite
    )
    counts = {"ok": 0, "skipped": 0, "failed": 0}
    failures = []
    start = time.time()

    with mp.Pool(args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(worker, launchers, chunksize=4), 1):
            counts[result["status"]] += 1
            if result["status"] == "failed":
                failures.append(result)
            if i % 200 == 0 or i == len(launchers):
                rate = i / (time.time() - start)
                print(f"[{i}/{len(launchers)}] {counts} ({rate:.1f}/s)", flush=True)

    summary_path = args.out_dir / f"failures_shard{args.shard_id}.jsonl"
    with open(summary_path, "w") as f:
        for failure in failures:
            f.write(json.dumps(failure) + "\n")
    print(f"done in {time.time() - start:.0f}s: {counts}; failures listed in {summary_path}")


if __name__ == "__main__":
    main()
