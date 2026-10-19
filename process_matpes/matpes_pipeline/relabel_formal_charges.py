"""Relabel implausible formal charges in preprocessed (.h5) shards, writing new copies.

The pymatgen oxidation-state guesses behind REF_formal_charges (see assemble_charges_xyz)
occasionally put large same-sign charges on bonded atoms -- azide N-N both -5/3, sulfate-like
S at -2 next to O at -2, H2 as two hydrides -- and the Coulomb energy of those formal charges
reaches hundreds of eV/atom. LocalSplitCharges cannot cancel that, so ~1% of structures carry
almost all of the energy RMSE.

A structure is flagged when the Coulomb energy of its formal charges alone exceeds --threshold
eV/atom, evaluated with the model's own electrostatics (GTO density, max_l 1, smearing and
k-space cutoff as in the fits, self-interaction included, pbc). Flagged structures get new
formal charges in two variants, each written to its own copy of the dataset:

- zero: all formal charges set to 0.
- ddec: integer charges fitted to the DDEC6 charges (monopole of atomic_multipoles). Per
  element, a DDEC6/formal scale and a set of allowed states are fitted on the unflagged
  training structures; each flagged structure then gets, by a small MILP, the allowed states
  minimising sum_i (scale_Z * q_i - ddec_i)^2 subject to sum_i q_i = 0 exactly (MatPES is
  all neutral). If that assignment is still above the threshold, it falls back to zeros.

Unflagged structures, and every other field, are left untouched. The source directory is only
read; the destinations must not exist yet. statistics.json is copied unchanged, and
relabel_report.json (fitted scales/states, per-structure changes) goes next to it.

    conda run -n mace_scf python -m matpes_pipeline.relabel_formal_charges \\
        --src $PSCRATCH/mace-scf/matpes_fit/processed_pbe_random_frame_split \\
        --dst-zero $PSCRATCH/mace-scf/matpes_fit/processed_pbe_random_frame_split_fqzero \\
        --dst-ddec $PSCRATCH/mace-scf/matpes_fit/processed_pbe_random_frame_split_fqddec

(run from process_matpes with graph_electrostatics on PYTHONPATH).
"""

import argparse
import collections
import json
import shutil
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.optimize import Bounds, LinearConstraint, milp

from graph_longrange.energy import GTOElectrostaticEnergy
from graph_longrange.gto_utils import gto_basis_kspace_cutoff
from graph_longrange.kspace import compute_k_vectors_flat


class FormalCoulomb:
    """Coulomb energy per atom of monopoles only, batched over structures."""

    def __init__(self, smearing, max_l, kspace_cutoff_factor, device):
        self.device = device
        self.kc = kspace_cutoff_factor * gto_basis_kspace_cutoff([smearing], max_l)
        self.max_l = max_l
        self.energy = GTOElectrostaticEnergy(
            density_max_l=max_l,
            density_smearing_width=smearing,
            kspace_cutoff=self.kc,
            include_self_interaction=True,
            pbc_handling="pbc",
        ).to(device)

    @torch.no_grad()
    def __call__(self, structs):
        """structs: list of (positions [n,3], cell [3,3], pbc [3], charges [n]) -> eV/atom."""
        dev = self.device
        ns = [len(s[3]) for s in structs]
        pos = torch.tensor(np.concatenate([s[0] for s in structs]), device=dev)
        cell = torch.tensor(np.stack([s[1] for s in structs]), device=dev)
        pbc = torch.tensor(np.stack([s[2] for s in structs]), device=dev)
        src = torch.zeros(len(pos), (self.max_l + 1) ** 2, device=dev)
        src[:, 0] = torch.tensor(np.concatenate([s[3] for s in structs]), device=dev)
        batch = torch.repeat_interleave(torch.arange(len(structs), device=dev), torch.tensor(ns, device=dev))
        rcell = 2 * np.pi * torch.linalg.inv(cell.mT)
        kv, kn2, kb, k0 = compute_k_vectors_flat(torch.tensor(self.kc, device=dev), cell, rcell)
        e = self.energy(
            k_vectors=kv, k_norm2=kn2, k_vector_batch=kb, k0_mask=k0, source_feats=src,
            node_positions=pos, batch=batch, volume=torch.linalg.det(cell).abs(), pbc=pbc,
        )
        return (e.cpu().numpy() / np.array(ns)).tolist()


def iter_configs(path):
    with h5py.File(path, "r") as f:
        for name, grp in f.items():
            for key, c in grp.items():
                p = c["properties"]
                yield (name, key), dict(
                    Z=c["atomic_numbers"][()],
                    positions=c["positions"][()],
                    cell=c["cell"][()],
                    pbc=c["pbc"][()],
                    charges=p["charges"][()],
                    ddec=p["atomic_multipoles"][()][:, 0],
                )


def scan_split(files, coulomb, batch_size, threshold, fit_stats):
    """Formal-charge Coulomb energy of every structure; accumulates fit statistics if asked."""
    rows, buf = [], []

    def flush():
        for (meta, c), e in zip(buf, coulomb([(c["positions"], c["cell"], c["pbc"], c["charges"]) for _, c in buf])):
            rows.append((*meta, e))
            if fit_stats is not None and e <= threshold:
                q = c["charges"]
                if np.all(np.isfinite(c["ddec"])) and np.allclose(q, np.round(q), atol=1e-6):
                    num, den, states = fit_stats
                    for z, qi, di in zip(c["Z"], np.round(q).astype(int), c["ddec"]):
                        states[int(z)][int(qi)] += 1
                        if qi != 0:
                            num[int(z)] += qi * di
                            den[int(z)] += qi * qi
        buf.clear()

    for fi, path in enumerate(files):
        for key, c in iter_configs(path):
            buf.append(((fi, key), c))
            if len(buf) == batch_size:
                flush()
        print(f"  scanned {path.name}: {len(rows)} structures so far", flush=True)
    if buf:
        flush()
    return rows


def fit_assignment(fit_stats, min_count_frac, min_count, min_scale_atoms):
    num, den, states = fit_stats
    glob = sum(num.values()) / sum(den.values())
    scale = {z: (num[z] / den[z] if den[z] > min_scale_atoms else glob) for z in states}
    allowed = {
        z: sorted({k for k, n in cnt.items() if n >= max(min_count, min_count_frac * sum(cnt.values()))} | {0})
        for z, cnt in states.items()
    }
    return glob, scale, allowed


def milp_assign(Z, ddec, scale, allowed, glob):
    opts = [allowed.get(int(z), [0]) for z in Z]
    idx = np.cumsum([0] + [len(o) for o in opts])
    n, nv = len(Z), idx[-1]
    cost = np.concatenate([[(scale.get(int(z), glob) * k - d) ** 2 for k in o] for z, d, o in zip(Z, ddec, opts)])
    A = np.zeros((n + 1, nv))
    for a in range(n):
        A[a, idx[a]:idx[a + 1]] = 1
        A[n, idx[a]:idx[a + 1]] = opts[a]
    lo = np.r_[np.ones(n), 0.0]
    res = milp(cost, constraints=LinearConstraint(A, lo, lo), integrality=np.ones(nv), bounds=Bounds(0, 1))
    if not res.success:
        return None
    x = np.round(res.x)
    q = np.array([opts[a][int(np.argmax(x[idx[a]:idx[a + 1]]))] for a in range(n)], dtype=float)
    assert q.sum() == 0
    return q


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst-zero", type=Path, required=True)
    ap.add_argument("--dst-ddec", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--threshold", type=float, default=2.0, help="eV/atom of formal-charge Coulomb energy")
    ap.add_argument("--smearing", type=float, default=1.5)
    ap.add_argument("--max-l", type=int, default=1)
    ap.add_argument("--kspace-cutoff-factor", type=float, default=1.25)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    for d in (args.dst_zero, args.dst_ddec):
        if d.exists():
            raise SystemExit(f"{d} exists; refusing to overwrite")
        if d.resolve() == args.src.resolve():
            raise SystemExit("destination equals source")
    torch.set_default_dtype(torch.float64)
    coulomb = FormalCoulomb(args.smearing, args.max_l, args.kspace_cutoff_factor, args.device)
    files = {s: sorted((args.src / s).glob("*.h5")) for s in args.splits}
    t0 = time.time()

    # 1) scan: training first, fitting the DDEC6 assignment on its unflagged structures
    fit_stats = (collections.defaultdict(float), collections.defaultdict(float), collections.defaultdict(collections.Counter))
    scans = {}
    for s in args.splits:
        print(f"scan {s}: {len(files[s])} shards", flush=True)
        scans[s] = scan_split(files[s], coulomb, args.batch_size, args.threshold, fit_stats if s == "train" else None)
    glob, scale, allowed = fit_assignment(fit_stats, min_count_frac=0.005, min_count=3, min_scale_atoms=50)
    print(f"fit: global scale {glob:.3f}, {len(scale)} elements ({time.time() - t0:.0f} s)", flush=True)

    # 2) new charges for flagged structures
    report = dict(threshold=args.threshold, smearing=args.smearing, max_l=args.max_l,
                  kspace_cutoff_factor=args.kspace_cutoff_factor, global_scale=glob,
                  scale={str(z): v for z, v in sorted(scale.items())},
                  allowed={str(z): v for z, v in sorted(allowed.items())}, splits={})
    for s in args.splits:
        flagged = collections.defaultdict(dict)  # file index -> key -> row
        for fi, key, e in scans[s]:
            if e > args.threshold:
                flagged[fi][key] = dict(file=files[s][fi].name, config="/".join(key), ecoul_old=e)
        cand = []
        for fi, keys in flagged.items():
            for key, c in iter_configs(files[s][fi]):
                if key in keys:
                    q = milp_assign(c["Z"], c["ddec"], scale, allowed, glob) if np.all(np.isfinite(c["ddec"])) else None
                    keys[key]["n"] = len(c["Z"])
                    keys[key]["Z"] = c["Z"].tolist()
                    keys[key]["q_old"] = c["charges"].tolist()
                    if q is not None:
                        cand.append((fi, key, (c["positions"], c["cell"], c["pbc"], q)))
                        keys[key]["q_ddec_fit"] = q.tolist()
        for i in range(0, len(cand), args.batch_size):
            chunk = cand[i:i + args.batch_size]
            for (fi, key, _), e in zip(chunk, coulomb([x[2] for x in chunk])):
                flagged[fi][key]["ecoul_ddec_fit"] = e
        for keys in flagged.values():
            for r in keys.values():
                ok = r.get("ecoul_ddec_fit") is not None and r["ecoul_ddec_fit"] <= args.threshold
                r["ddec_variant"] = "ddec_fit" if ok else "zero"
        n_all = len(scans[s]); n_flag = sum(len(k) for k in flagged.values())
        n_fb = sum(r["ddec_variant"] == "zero" for k in flagged.values() for r in k.values())
        print(f"{s}: {n_flag}/{n_all} flagged ({100 * n_flag / n_all:.2f}%), ddec variant falls back to zeros for {n_fb}", flush=True)
        report["splits"][s] = dict(n_structures=n_all, n_flagged=n_flag, n_ddec_fallback_zero=n_fb,
                                   ecoul_old=[e for _, _, e in scans[s]],
                                   flagged=[r for k in flagged.values() for r in k.values()])

        # 3) write both copies, then rewrite only the flagged structures' charges
        for variant, dst in (("zero", args.dst_zero), ("ddec", args.dst_ddec)):
            (dst / s).mkdir(parents=True)
            for fi, path in enumerate(files[s]):
                out = dst / s / path.name
                shutil.copy2(path, out)
                if fi not in flagged:
                    continue
                with h5py.File(out, "r+") as f:
                    for key, r in flagged[fi].items():
                        ds = f[key[0]][key[1]]["properties"]["charges"]
                        new = np.zeros(ds.shape) if (variant == "zero" or r["ddec_variant"] == "zero") else np.array(r["q_ddec_fit"])
                        assert new.shape == ds.shape and abs(new.sum()) < 1e-9
                        ds[...] = new
            print(f"  wrote {variant} {s} ({time.time() - t0:.0f} s)", flush=True)

    for dst, variant in ((args.dst_zero, "zero"), (args.dst_ddec, "ddec")):
        shutil.copy2(args.src / "statistics.json", dst / "statistics.json")
        json.dump(dict(report, variant=variant, source=str(args.src)), open(dst / "relabel_report.json", "w"))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
