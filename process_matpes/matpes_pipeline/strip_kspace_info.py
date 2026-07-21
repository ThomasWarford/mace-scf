"""Strip k-point/Fourier-density info keys from an extxyz file.

Pure text streaming, not `ase.io`: the `REF_fourier_*` and `k_triplets`
arrays make the per-frame comment line multiple MB long, and ASE's extxyz
info-dict parser (regex + literal_eval per array) is too slow at ~800k-frame
scale. A standard extxyz frame is exactly

    <natoms>
    <comment line: space-separated key=value / key="quoted value" tokens>
    <natoms atom lines>

repeated to EOF, so atom lines never need to be touched -- only the
comment line's tokens are filtered.
"""

import argparse
import re
from pathlib import Path

REMOVE_KEYS = {
    "n_k",
    "k_triplets",
    "fourier_grid_dims",
    "REF_fourier_chg_total",
    "REF_fourier_chg_diff",
    "REF_fourier_aeccar_diff",
}

# key=value or key="quoted value" tokens, in order.
TOKEN_RE = re.compile(r'(\S+)=("[^"]*"|\S+)')


def strip_comment_line(line: str) -> str:
    kept = (
        f"{key}={value}"
        for key, value in TOKEN_RE.findall(line)
        if key not in REMOVE_KEYS
    )
    return " ".join(kept)


def strip_file(src: Path, dst: Path, progress_every: int = 200_000) -> int:
    """Stream src -> dst with k-point/Fourier info keys removed. Returns n frames."""
    n_frames = 0
    with open(src) as fin, open(dst, "w") as fout:
        while True:
            natoms_line = fin.readline()
            if not natoms_line:
                break
            natoms = int(natoms_line)
            comment_line = fin.readline()
            fout.write(natoms_line)
            fout.write(strip_comment_line(comment_line.rstrip("\n")))
            fout.write("\n")
            for _ in range(natoms):
                fout.write(fin.readline())
            n_frames += 1
            if progress_every and n_frames % progress_every == 0:
                print(f"  {src.name}: {n_frames} frames", flush=True)
    return n_frames


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    args = ap.parse_args()

    tmp = args.dst.with_name(args.dst.name + ".tmp")
    n = strip_file(args.src, tmp)
    tmp.replace(args.dst)
    print(f"wrote {args.dst} ({n} frames)")


if __name__ == "__main__":
    main()
