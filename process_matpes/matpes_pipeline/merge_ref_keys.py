"""Copy REF_-prefixed keys from one extxyz file into another, frame by frame.

Stopgap for regenerating a `-charges.xyz` the slow way (re-running
assemble_charges_xyz.py, which repeats an expensive per-frame pymatgen
bond-valence analysis just to also get REF_forces/REF_stress/REF_total_charge
that a `-nofourier.xyz` from the same npz/jsonl join already has). Both files
come from the same deterministic (sorted-glob, chunk-ordered) assembly, so
frame N in one lines up with frame N in the other -- verified per frame via
matching `provenance_path`/`matpes_id`, not assumed.

Pure text streaming (see strip_kspace_info.py for the same rationale): reads
both files in lockstep and, for every REF_-prefixed field in `--extra` that
`--base` doesn't already have, appends it -- as an info token if it's a
comment-line scalar, or as new Properties columns if it's a per-atom array.
"""

import argparse
import re
from pathlib import Path

TOKEN_RE = re.compile(r'(\S+)=("[^"]*"|\S+)')
PROP_FIELD_RE = re.compile(r'([A-Za-z0-9_]+):([A-Za-z]):(\d+)')


def parse_properties(comment: str):
    """Properties=... -> [(name, dtype, ncols, start_col), ...]."""
    m = re.search(r'Properties=(\S+)', comment)
    if not m:
        raise ValueError(f"no Properties= in comment line: {comment[:200]!r}")
    fields = []
    col = 0
    for name, dtype, ncols in PROP_FIELD_RE.findall(m.group(1)):
        ncols = int(ncols)
        fields.append((name, dtype, ncols, col))
        col += ncols
    return fields


def frame_key(comment: str) -> str:
    tokens = dict(TOKEN_RE.findall(comment))
    return tokens.get("provenance_path") or tokens.get("matpes_id", "")


def read_frame(fh):
    natoms_line = fh.readline()
    if not natoms_line:
        return None
    natoms = int(natoms_line)
    comment = fh.readline().rstrip("\n")
    atom_lines = [fh.readline() for _ in range(natoms)]
    return natoms, comment, atom_lines


def merge_frame(base_comment, base_atoms, extra_comment, extra_atoms):
    base_tokens = dict(TOKEN_RE.findall(base_comment))
    added_info = [
        f"{k}={v}"
        for k, v in TOKEN_RE.findall(extra_comment)
        if k.startswith("REF_") and k not in base_tokens
    ]
    new_comment = f"{base_comment} {' '.join(added_info)}" if added_info else base_comment

    base_field_names = {name for name, _, _, _ in parse_properties(base_comment)}
    copy_specs = [
        (name, dtype, ncols, start)
        for name, dtype, ncols, start in parse_properties(extra_comment)
        if name.startswith("REF_") and name not in base_field_names
    ]
    if not copy_specs:
        return new_comment, base_atoms

    prop_suffix = "".join(f":{name}:{dtype}:{ncols}" for name, dtype, ncols, _ in copy_specs)
    new_comment = re.sub(r"Properties=\S+", lambda m: m.group(0) + prop_suffix, new_comment)

    new_atoms = []
    for base_line, extra_line in zip(base_atoms, extra_atoms):
        extra_cols = extra_line.split()
        extra_extra = [
            col for _, _, ncols, start in copy_specs for col in extra_cols[start : start + ncols]
        ]
        new_atoms.append(f"{base_line.rstrip(chr(10))} {' '.join(extra_extra)}\n")
    return new_comment, new_atoms


def merge_files(base_path: Path, extra_path: Path, out_path: Path, limit=None) -> int:
    n_frames = 0
    mismatches = 0
    with open(base_path) as fb, open(extra_path) as fe, open(out_path, "w") as fout:
        while limit is None or n_frames < limit:
            base = read_frame(fb)
            extra = read_frame(fe)
            if base is None and extra is None:
                break
            if base is None or extra is None:
                raise ValueError(f"frame count mismatch at frame {n_frames}: one file ran out first")
            b_natoms, b_comment, b_atoms = base
            e_natoms, e_comment, e_atoms = extra
            if b_natoms != e_natoms:
                raise ValueError(f"natoms mismatch at frame {n_frames}: {b_natoms} vs {e_natoms}")
            if frame_key(b_comment) != frame_key(e_comment):
                mismatches += 1
                if mismatches <= 5:
                    print(
                        f"WARNING: key mismatch at frame {n_frames}: "
                        f"{frame_key(b_comment)!r} vs {frame_key(e_comment)!r}"
                    )
            comment, atoms = merge_frame(b_comment, b_atoms, e_comment, e_atoms)
            fout.write(f"{b_natoms}\n{comment}\n")
            fout.writelines(atoms)
            n_frames += 1
    if mismatches:
        print(
            f"WARNING: {mismatches}/{n_frames} frames had a provenance_path/matpes_id "
            "mismatch -- base and extra are not in the same order; do not trust this output"
        )
    return n_frames


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("base", type=Path, help="file to enrich, e.g. MatPES-PBE-charges.xyz")
    ap.add_argument("extra", type=Path, help="file to copy REF_ keys from, e.g. MatPES-PBE-nofourier.xyz")
    ap.add_argument("out", type=Path)
    ap.add_argument("--limit", type=int, default=None, help="stop after N frames (for testing)")
    args = ap.parse_args()

    tmp = args.out.with_name(args.out.name + ".tmp")
    n = merge_files(args.base, args.extra, tmp, limit=args.limit)
    tmp.replace(args.out)
    print(f"wrote {args.out} ({n} frames)")


if __name__ == "__main__":
    main()
