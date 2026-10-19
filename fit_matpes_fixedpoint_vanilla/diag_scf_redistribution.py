"""How much charge does the self-consistent field move? One GPU, through train.sh with an e3nn run's
flags (no CUEQ), on an e3nn state dict (convert_and_probe_scf.py's <stem>_e3nn.pt):

    SLURM_NTASKS=1 PROBE=1 PROBE_SCRIPT=diag_scf_redistribution.py CASE=<e3nn config> BACKBONE=r2 \
        UPDATE=monotone TAG=wrand E3NN_STATE=<..._e3nn.pt> OUT_JSON=<path> fit_matpes_fixedpoint_vanilla/train.sh

Per validation frame: constant-charge SCF from the local guess (100 steps, mixing 0.3, tol 1e-6,
mu from data, as the runner's eval does) gives q* and mu*. At mu* the density is then built at
four levels, each followed by observables with that density frozen (forces at fixed density):
  q0    local guess (field-independent charges; no field, no Fermi level)
  qmu   q0 + update with the Fermi-level features only (Coulomb field of the density off)
  q1    q0 + one update with the Coulomb field of q0 (non-self-consistent, first order)
  q*    converged.
qmu -> q* is the Coulomb-driven (self-consistent) redistribution; q1 -> q* is what iterating adds
beyond first order. Also the spectral radius of J = d(q0 + update(q))/dq at q*, fixed mu*
(power iteration on J^T per frame): the strength of the charge-charge coupling; 0 = no
self-consistency, near 1 = slow, strongly coupled SCF.
"""

import json
import logging
import os

import numpy as np
import torch

from mace import tools
from mace.tools import torch_geometric
from mace.tools.scatter import scatter_sum

import mace_scf.utils
import mace_scf.utils.run_train_utils as rtu
from mace_scf.electrostatics.fixed_point_runner import FixedPointSCFRunner
from mace_scf.electrostatics.fixed_point_state import FixedPointSCFOptions
from mace_scf.utils.check_args import check_config_conflicts
from mace_scf.utils.load_data import load_train_valid_sets_from_preprocessed


def per_graph(x, batch, G):
    return scatter_sum(x, batch, dim=-1, dim_size=G)


def update(model, data, ls, density_in, mu_feats):
    dep, feats = model.scf_step(data, ls, charge_density_in=density_in, total_charges=density_in,
                                fermi_level_features=mu_feats)
    return ls.field_independent_charge_density + dep, feats


def spectral_radius(model, data, ls, q_star, mu_feats, n_iter):
    """|lambda|_max of d(update)/dq at q_star per graph, by power iteration on J^T."""
    batch, G = data["batch"], data["ptr"].numel() - 1
    q = q_star.detach().clone().requires_grad_(True)
    out, _ = update(model, data, ls, q, mu_feats)
    v = torch.randn_like(q)
    lam = torch.zeros(G, device=q.device, dtype=q.dtype)
    for _ in range(n_iter):
        nrm = per_graph((v ** 2).sum(-1), batch, G).sqrt()
        v = v / nrm[batch].unsqueeze(-1)
        w = torch.autograd.grad(out, q, grad_outputs=v, retain_graph=True)[0]
        lam = per_graph((w ** 2).sum(-1), batch, G).sqrt()
        v = w
    return lam


def observables(model, data, density, mu, density_in_for_feats, mu_feats_fn):
    """Energy and forces at a frozen density; field features as the update that made it saw."""
    ls = model.local_part(data, compute_force=True)
    src = density_in_for_feats.detach()
    _, feats = model.scf_step(data, ls, charge_density_in=src, total_charges=src,
                              fermi_level_features=mu_feats_fn(ls))
    out = model.build_observables(data=data, local_state=ls, density=density.detach(), fermi_level=mu,
                                  field_feats=feats, training=False, compute_force=True)
    return out["energy"].detach(), out["forces"].detach(), out["electrostatic_energy"].detach()


def main():
    args = mace_scf.utils.extended_arg_parser().parse_args()
    args.distributed = False
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    check_config_conflicts(args)
    tools.set_default_dtype(args.default_dtype)
    device = torch.device("cuda")
    n_frames = int(os.environ.get("N_FRAMES", "512"))
    bsz = int(os.environ.get("DIAG_BATCH", "8"))
    n_iter = int(os.environ.get("POWER_ITERS", "40"))
    out_json = os.environ["OUT_JSON"]
    assert not os.path.exists(out_json), f"{out_json} exists; not overwriting"

    _, valid_set, z_table, atomic_energies, _ = load_train_valid_sets_from_preprocessed(args)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(valid_set), size=min(n_frames, len(valid_set)), replace=False)
    loader = torch_geometric.dataloader.DataLoader([valid_set[int(i)] for i in idx], batch_size=bsz, shuffle=False)
    args.fermi_level_offset = rtu.get_fermi_level_offset(loader, args, device)
    args.field_feature_norms = rtu.get_field_feature_norms(loader, args, device,
                                                           fermi_level_offset=args.fermi_level_offset)
    args.atom_density_scaling = rtu.get_atom_density_scaling(loader, args, device, z_table=z_table)
    model = rtu.build_model(args=args, z_table=z_table, atomic_energies=atomic_energies,
                            atomic_charges=None, train_loader=loader).to(device)
    model.load_state_dict(torch.load(os.environ["E3NN_STATE"], map_location=device, weights_only=False)["model"],
                          strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    runner = FixedPointSCFRunner(FixedPointSCFOptions(
        num_scf_steps=100, scf_tolerance=1e-6, mixing_parameter=0.3, constant_charge=True,
        initial_density="local_guess", initial_fermi_level="from_data"))
    frames, atoms = [], {k: [] for k in ("q_ref", "q0", "qmu", "q1", "qs", "d_ref", "d0", "dmu", "d1", "ds", "z")}
    for bi, batch in enumerate(loader):
        batch = batch.to(device)
        data = batch.to_dict()
        G = data["ptr"].numel() - 1
        b = data["batch"]
        n = (data["ptr"][1:] - data["ptr"][:-1])

        ls = model.local_part(data, compute_force=False)
        res = runner.converge(model, data, ls, runner.get_initial_density(ls, data),
                              runner.get_initial_fermi(model, ls, data), compute_force=False)
        q_star, mu = res.density.detach(), res.fermi_level.detach()
        steps = int(res.terminated_step) + 2

        def mu_feats_fn(l):
            return model.features_from_fermi_level(b, l.positions, mu)

        mf = mu_feats_fn(ls)
        q0 = ls.field_independent_charge_density.detach()
        zero = torch.zeros_like(q0)
        with torch.no_grad():
            qmu, _ = update(model, data, ls, zero, mf)
            q1, _ = update(model, data, ls, q0, mf)
            qfix, _ = update(model, data, ls, q_star, mf)
        lam = spectral_radius(model, data, ls, q_star, mf, n_iter)

        levels = {"q0": (q0, None), "qmu": (qmu, zero), "q1": (q1, q0), "qs": (q_star, q_star)}
        E, F, Ec = {}, {}, {}
        for k, (dens, src) in levels.items():
            if k == "q0":
                # local guess: no update was made; give the energy terms the zero-field features
                src = zero
            E[k], F[k], Ec[k] = observables(model, data, dens, mu, src, mu_feats_fn)

        ref = data["density_coefficients"]
        for g in range(G):
            m = b == g
            row = dict(n=int(n[g]), status=res.status, steps=steps, spectral_radius=float(lam[g]),
                       fixed_point_residual=float((qfix[m] - q_star[m]).abs().max()),
                       mu_drift=float((mu[g] - data["fermi_level"][g]).abs()),
                       ref_E=float(data["energy"][g] / n[g]))
            for k, (dens, _) in levels.items():
                row[f"qmae_{k}_vs_qs"] = float((dens[m, 0] - q_star[m, 0]).abs().mean())
                row[f"qmae_{k}_vs_ref"] = float((dens[m, 0] - ref[m, 0]).abs().mean())
                row[f"Q_{k}"] = float(dens[m, 0].sum() - data["total_charge"][g])
                if dens.shape[1] >= 4:
                    row[f"dmae_{k}_vs_qs"] = float((dens[m, 1:4] - q_star[m, 1:4]).norm(dim=-1).mean())
                    row[f"dmae_{k}_vs_ref"] = float((dens[m, 1:4] - ref[m, 1:4]).norm(dim=-1).mean())
                row[f"dE_{k}_vs_qs"] = float((E[k][g] - E["qs"][g]) / n[g])
                row[f"Eerr_{k}"] = float((E[k][g] - data["energy"][g]).abs() / n[g])
                row[f"Ecoul_{k}"] = float(Ec[k][g] / n[g])
                row[f"dF_{k}_vs_qs"] = float((F[k][m] - F["qs"][m]).abs().mean())
                row[f"Ferr_{k}"] = float((F[k][m] - data["forces"][m]).abs().mean())
            row["q_ref_mean_abs"] = float(ref[m, 0].abs().mean())
            frames.append(row)
        for k, t in (("q_ref", ref), ("q0", q0), ("qmu", qmu), ("q1", q1), ("qs", q_star)):
            atoms[k].append(t[:, 0].cpu().numpy())
            if t.shape[1] >= 4:
                atoms["d" + k[1:]].append(t[:, 1:4].norm(dim=-1).cpu().numpy())
        atoms["z"].append(data["node_attrs"].argmax(-1).cpu().numpy())
        if bi % 8 == 0:
            logging.info("batch %d/%d: %s, %d steps, rho max %.3f", bi, len(loader), res.status, steps, float(lam.max()))

    keys = [k for k in frames[0] if k not in ("status",)]
    summary = {}
    for k in keys:
        v = np.array([r[k] for r in frames], dtype=float)
        summary[k] = {"mean": float(v.mean()), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
                      "max": float(v.max()), "min": float(v.min())}
    summary["status"] = {s: sum(r["status"] == s for r in frames) for s in set(r["status"] for r in frames)}
    a = {k: np.concatenate(v) for k, v in atoms.items() if v}
    dq = a["qs"] - a["qmu"]
    summary["atoms"] = {
        "n_atoms": int(a["qs"].size),
        "rms_q_ref": float(np.sqrt((a["q_ref"] ** 2).mean())),
        "rms_qs": float(np.sqrt((a["qs"] ** 2).mean())),
        "rms_dq_coulomb (qs-qmu)": float(np.sqrt((dq ** 2).mean())),
        "rms_dq_beyond_first_order (qs-q1)": float(np.sqrt(((a["qs"] - a["q1"]) ** 2).mean())),
        "rms_dq_total (qs-q0)": float(np.sqrt(((a["qs"] - a["q0"]) ** 2).mean())),
        "frac_atoms_|dq_coulomb|>0.05e": float((np.abs(dq) > 0.05).mean()),
        "frac_atoms_|dq_coulomb|>0.1e": float((np.abs(dq) > 0.1).mean()),
        "max_|dq_coulomb|": float(np.abs(dq).max()),
        # does the Coulomb redistribution move charges towards DDEC? regression slope of
        # (q_ref - qmu) on (qs - qmu): 1 = right amount, 0 = uncorrelated
        "corr(qs-qmu, qref-qmu)": float(np.corrcoef(dq, a["q_ref"] - a["qmu"])[0, 1]),
        "slope(qref-qmu on qs-qmu)": float((dq * (a["q_ref"] - a["qmu"])).sum() / (dq ** 2).sum()),
    }
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump({"checkpoint": os.environ["E3NN_STATE"], "summary": summary, "frames": frames}, fh, indent=1)
    np.savez(os.path.splitext(out_json)[0] + "_atoms.npz", **a)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
