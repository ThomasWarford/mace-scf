"""Split an extxyz into -train/-valid by structure, not by frame.

MatPES has many frames per structure (ionic steps of one relaxation,
volume-scaled reruns of one mp-id), so a per-frame random split (what
mace's own random_train_valid_split does) leaks near-duplicate structures
across train/valid. This groups frames by structure identity first, then
splits whole groups, so every frame of a given structure lands on the same
side.

Group key: `original_mp_id` when present (the common case -- present
whenever a frame was matched to its MatPES jsonl entry); otherwise the
structure number embedded in `matpes_id` (matpes-<date>_<number>_<suffix>),
or the bare matpes_id itself when it's already an mp-id
(matpes-mp-structures jobs use the MP id directly, with no per-frame
suffix), or the raw matpes_id as a last-resort singleton group.

Pure text streaming (see strip_kspace_info.py for why): two passes over the
file -- first to read each frame's comment line and compute its group key,
second to re-stream frames straight to -train.xyz/-valid.xyz by that
frame's group assignment. Atom lines are never parsed.

Typical use (from the process_matpes directory):
    conda run -n dft python -m matpes_pipeline.split_by_structure \
        /path/to/MatPES-PBE-charges-quick.xyz --valid_fraction 0.03
"""

import argparse
import json
import random
import re
from pathlib import Path

TOKEN_RE = re.compile(r'(\S+)=("[^"]*"|\S+)')
MP_DIRECT_RE = re.compile(r"^mp-\d+$")
MATPES_DATED_RE = re.compile(r"^matpes-\d+_(\d+)_\d+$")


def group_key(tokens: dict) -> str:
    mp_id = tokens.get("original_mp_id")
    if mp_id:
        return mp_id
    matpes_id = tokens.get("matpes_id", "")
    if MP_DIRECT_RE.match(matpes_id):
        return matpes_id
    m = MATPES_DATED_RE.match(matpes_id)
    if m:
        return f"matpes-{m.group(1)}"
    return matpes_id


def iter_frame_spans(path: Path):
    """Yield (natoms, comment_line) for each frame without reading atom lines."""
    with open(path) as f:
        while True:
            natoms_line = f.readline()
            if not natoms_line:
                return
            natoms = int(natoms_line)
            comment = f.readline().rstrip("\n")
            for _ in range(natoms):
                f.readline()
            yield natoms, comment


def assign_splits(group_sizes: dict, valid_fraction: float, seed: int):
    """Shuffle groups, greedily fill valid until its frame share hits valid_fraction."""
    total_frames = sum(group_sizes.values())
    target = valid_fraction * total_frames

    groups = list(group_sizes.keys())
    random.Random(seed).shuffle(groups)

    valid_groups, valid_frames = set(), 0
    for g in groups:
        if valid_frames >= target:
            break
        valid_groups.add(g)
        valid_frames += group_sizes[g]

    return valid_groups, valid_frames, total_frames


def split_file(src: Path, out_dir: Path, valid_fraction: float, seed: int):
    stem = src.name[:-4] if src.name.endswith(".xyz") else src.name
    train_path = out_dir / f"{stem}-train.xyz"
    valid_path = out_dir / f"{stem}-valid.xyz"
    report_path = out_dir / f"{stem}-split_report.json"

    # Pass 1: group key per frame, in order.
    group_sizes: dict = {}
    frame_groups = []
    for _, comment in iter_frame_spans(src):
        tokens = dict(TOKEN_RE.findall(comment))
        key = group_key(tokens)
        frame_groups.append(key)
        group_sizes[key] = group_sizes.get(key, 0) + 1

    valid_groups, valid_frames, total_frames = assign_splits(group_sizes, valid_fraction, seed)
    print(
        f"{src.name}: {len(group_sizes)} structures, {total_frames} frames -- "
        f"valid: {len(valid_groups)} structures, {valid_frames} frames "
        f"({100 * valid_frames / total_frames:.2f}%)"
    )

    # Pass 2: re-stream frames straight to train/valid by precomputed assignment.
    train_tmp = train_path.with_name(train_path.name + ".tmp")
    valid_tmp = valid_path.with_name(valid_path.name + ".tmp")
    with open(src) as fin, open(train_tmp, "w") as ftrain, open(valid_tmp, "w") as fvalid:
        for key in frame_groups:
            natoms_line = fin.readline()
            natoms = int(natoms_line)
            comment_line = fin.readline()
            atom_lines = [fin.readline() for _ in range(natoms)]
            out = fvalid if key in valid_groups else ftrain
            out.write(natoms_line)
            out.write(comment_line)
            out.writelines(atom_lines)
    train_tmp.replace(train_path)
    valid_tmp.replace(valid_path)

    report = {
        "source": str(src),
        "valid_fraction_requested": valid_fraction,
        "seed": seed,
        "n_structures": len(group_sizes),
        "n_frames": total_frames,
        "n_valid_structures": len(valid_groups),
        "n_valid_frames": valid_frames,
        "valid_frame_fraction_actual": valid_frames / total_frames,
        "valid_group_keys": sorted(valid_groups),
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {train_path} ({total_frames - valid_frames} frames)")
    print(f"wrote {valid_path} ({valid_frames} frames)")
    print(f"wrote {report_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("xyz", type=Path, help="extxyz file to split")
    ap.add_argument("--valid_fraction", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out-dir", type=Path, default=None, help="default: same directory as the input file")
    args = ap.parse_args()

    out_dir = args.out_dir or args.xyz.parent
    split_file(args.xyz, out_dir, args.valid_fraction, args.seed)


if __name__ == "__main__":
    main()
