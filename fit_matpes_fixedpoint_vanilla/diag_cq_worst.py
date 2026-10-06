"""Which frames break constant-charge SCF, and how? Run through train.sh (one GPU) on the model's flags:

    CASE=direct_docs_lr0.028 CUEQ=1 BATCH=16 PROBE_SCRIPT=diag_cq_worst.py DIAG_CKPT=<..._epoch-49.pt> \
        fit_matpes_fixedpoint_vanilla/interactive_probe.sh            # inside an interactive salloc

1. screen: every validation frame (DIAG_MAX_FRAMES caps it), constant-charge SCF from the local
   guess (100 steps, mixing 0.3, tol 1e-6) without forces, batched. Per frame: atoms, formula,
   reference mu, dQ/dmu and total-charge error at the reference (one SCF update at the reference
   density, as converge_constant_charge computes dq/dmu), and after the SCF: converged, final
   density change, |mu - mu_ref|, |Q - Q_ref|, energy error per atom. Also constant-Fermi SCF on
   the same batches, for the energy error with mu held at the reference.
2. deep dive on the DIAG_TOP worst frames by mu drift (and the worst by energy error), one frame
   per batch:
   - per-step trajectory (mu, total response dQ/dmu, Q, density change) parsed from the SCF loop's
     own debug records, so it is the production code path;
   - at the final state: per-atom dq_i/dmu (sign mix, |sum| vs sum|.|), Fukui weights f_i,
     charges vs reference;
   - energy / force errors with forces through the SCF, constant charge vs constant Fermi.
Writes diag_cq_worst.json (all screened frames + deep dives) to the results dir.
"""

import json
import logging
import os
import re
from collections import Counter

import numpy as np
import torch
from ase.data import chemical_symbols

from mace import tools
from mace.tools import torch_geometric
from mace.tools.scatter import scatter_sum

import mace_scf.utils
import mace_scf.utils.run_train_utils as rtu
from mace_scf.electrostatics.fixed_point_runner import FixedPointSCFRunner
from mace_scf.electrostatics.fixed_point_state import FixedPointSCFOptions
from mace_scf.utils.check_args import check_config_conflicts
from mace_scf.utils.load_data import load_train_valid_sets_from_preprocessed

STEP_RE = re.compile(
    r"step (\d+), fermi_levels=tensor\(\[([^\]]*)\].*?gradient=tensor\(\[([^\]]*)\].*?"
    r"total_qs=tensor\(\[([^\]]*)\].*?abs_change=([-\d.e+]+)", re.S)


class Capture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.msgs = []

    def emit(self, record):
        self.msgs.append(record.getMessage())


def opts(cc, steps=100):
    return FixedPointSCFOptions(num_scf_steps=steps, scf_tolerance=1e-6, mixing_parameter=0.3,
                                constant_charge=cc, initial_density="local_guess",
                                initial_fermi_level="from_data")


def formula(zs):
    c = Counter(int(z) for z in zs)
    return "".join(f"{chemical_symbols[z]}{n if n > 1 else ''}" for z, n in sorted(c.items()))


def dq_at(model, data, ls, density, mu):
    """per-atom dq/dmu and the new density for one SCF update at (density, mu)."""
    node_mu = torch.index_select(mu, 0, data["batch"]).detach().clone().requires_grad_(True)
    feats = model.features_from_fermi_level_nodewise(data["batch"], ls.positions, node_mu)
    dep, _ = model.scf_step(data, ls, charge_density_in=density, total_charges=density,
                            fermi_level_features=feats)
    dq = torch.autograd.grad(dep[:, 0].sum(), node_mu)[0]
    return dq.detach(), (ls.field_independent_charge_density + dep).detach()


def screen(model, loader, device, zs_of):
    rows = []
    run_cc, run_cf = FixedPointSCFRunner(opts(True)), FixedPointSCFRunner(opts(False))
    k = 0
    for batch in loader:
        batch = batch.to(device)
        data = batch.to_dict()
        n = (data["ptr"][1:] - data["ptr"][:-1])
        G = n.numel()
        ls = model.local_part(data, compute_force=False)
        dq, dens = dq_at(model, data, ls, data["density_coefficients"], data["fermi_level"])
        dQ0 = scatter_sum(dq, data["batch"], dim=-1, dim_size=G)
        Q0 = scatter_sum(dens[:, 0], data["batch"], dim=-1, dim_size=G) - data["total_charge"]
        out = {}
        for tag, run in (("cc", run_cc), ("cf", run_cf)):
            # no torch.no_grad(): converge_constant_charge takes dq/dmu by autograd
            res = run.converge(model, data, ls, run.get_initial_density(ls, data),
                               run.get_initial_fermi(model, ls, data), compute_force=False)
            obs = model.build_observables(data=data, local_state=ls, density=res.density.detach(),
                                          fermi_level=res.fermi_level.detach(),
                                          field_feats=res.field_feats.detach(),
                                          training=False, compute_force=False)
            Q = scatter_sum(res.density[:, 0], data["batch"], dim=-1, dim_size=G)
            out[tag] = (res, obs, Q)
        for g in range(G):
            rc, oc, Qc = out["cc"]
            rf, of, _ = out["cf"]
            zs = zs_of[k]
            rows.append(dict(
                idx=k, n=int(n[g]), formula=formula(zs), mu_ref=float(data["fermi_level"][g]),
                dQdmu_ref=float(dQ0[g]), Qerr_ref=float(Q0[g]),
                cc_conv=bool(rc.final_avg_abs_change[g] <= 1e-6 and abs(float(Qc[g] - data["total_charge"][g])) < 1e-4),
                cc_final_change=float(rc.final_avg_abs_change[g]),
                cc_mu_drift=float(rc.fermi_level[g] - data["fermi_level"][g]),
                cc_Qerr=float(Qc[g] - data["total_charge"][g]),
                cc_Eerr=float((oc["energy"][g] - data["energy"][g]) / n[g]),
                cf_conv=bool(rf.final_avg_abs_change[g] <= 1e-6),
                cf_Eerr=float((of["energy"][g] - data["energy"][g]) / n[g]),
            ))
            k += 1
        if k % 1024 < G:
            logging.info("screened %d frames", k)
    return rows


def deep_dive(model, frame, device, cap):
    loader = torch_geometric.dataloader.DataLoader([frame], batch_size=1, shuffle=False)
    batch = next(iter(loader)).to(device)
    data = batch.to_dict()
    n = int(batch.num_nodes)
    res_out = {}
    for tag, cc in (("cc", True), ("cf", False)):
        run = FixedPointSCFRunner(opts(cc))
        ls = model.local_part(data, compute_force=True)
        cap.msgs.clear()
        res = run.converge(model, data, ls, run.get_initial_density(ls, data),
                           run.get_initial_fermi(model, ls, data), compute_force=True)
        obs = model.build_observables(data=data, local_state=ls, density=res.density,
                                      fermi_level=res.fermi_level, field_feats=res.field_feats,
                                      training=False, compute_force=True)
        traj = []
        for m in cap.msgs:
            mt = STEP_RE.search(m)
            if mt:
                traj.append(dict(step=int(mt.group(1)), mu=float(mt.group(2).split(",")[0]),
                                 dQdmu=float(mt.group(3).split(",")[0]),
                                 Q=float(mt.group(4).split(",")[0]), change=float(mt.group(5))))
        ferr = (obs["forces"].detach() - data["forces"]).abs()
        d = dict(status=res.status, steps=int(res.terminated_step) + 2,
                 mu_final=float(res.fermi_level[0]), Q_final=float(res.density[:, 0].sum()),
                 E_err_per_atom=float((obs["energy"][0].detach() - data["energy"][0]) / n),
                 F_mae=float(ferr.mean()), F_max=float(ferr.max()),
                 q_mae=float((res.density[:, 0].detach() - data["density_coefficients"][:, 0]).abs().mean()),
                 q_max_abs=float(res.density[:, 0].detach().abs().max()),
                 traj=traj)
        if cc:
            ls0 = model.local_part(data, compute_force=False)
            dq, _ = dq_at(model, data, ls0, res.density.detach(), res.fermi_level.detach())
            tot = float(dq.sum())
            d["final_dq"] = dict(
                per_atom=dq.tolist(), sum=tot, sum_abs=float(dq.abs().sum()),
                frac_positive=float((dq > 0).float().mean()),
                fukui_max_abs=float((dq / tot).abs().max()) if tot != 0 else float("inf"),
                q_local=ls0.field_independent_charge_density[:, 0].detach().tolist(),
                q_final=res.density[:, 0].detach().tolist(),
                q_ref=data["density_coefficients"][:, 0].tolist(),
                Z=[int(z) for z in batch.node_attrs.argmax(-1).tolist()],
            )
        res_out[tag] = d
    return res_out


def main():
    args = mace_scf.utils.extended_arg_parser().parse_args()
    args.distributed = False
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    check_config_conflicts(args)
    tools.set_default_dtype(args.default_dtype)
    device = torch.device("cuda")

    _, valid_set, z_table, atomic_energies, _ = load_train_valid_sets_from_preprocessed(args)
    n_max = int(os.environ.get("DIAG_MAX_FRAMES", "0")) or len(valid_set)
    frames = [valid_set[i] for i in range(min(n_max, len(valid_set)))]
    zs_of = [[z_table.zs[int(a)] for a in f.node_attrs.argmax(-1)] for f in frames]
    loader = torch_geometric.dataloader.DataLoader(frames, batch_size=args.batch_size, shuffle=False)
    args.fermi_level_offset = rtu.get_fermi_level_offset(loader, args, device)
    args.field_feature_norms = rtu.get_field_feature_norms(loader, args, device,
                                                           fermi_level_offset=args.fermi_level_offset)
    args.atom_density_scaling = rtu.get_atom_density_scaling(loader, args, device, z_table=z_table)
    model = rtu.build_model(args=args, z_table=z_table, atomic_energies=atomic_energies,
                            atomic_charges=None, train_loader=loader).to(device)
    model.load_state_dict(torch.load(os.environ["DIAG_CKPT"], map_location=device, weights_only=False)["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    logging.info("screening %d validation frames", len(frames))
    rows = screen(model, loader, device, zs_of)

    top = int(os.environ.get("DIAG_TOP", "30"))
    by_mu = sorted(rows, key=lambda r: -abs(r["cc_mu_drift"]))[:top]
    by_E = sorted(rows, key=lambda r: -abs(r["cc_Eerr"]))[:10]
    pick = {r["idx"]: r for r in by_mu + by_E}
    scf_log = logging.getLogger("mace_scf.electrostatics.fixed_point_scf")
    cap = Capture()
    scf_log.addHandler(cap)
    scf_log.setLevel(logging.DEBUG)
    scf_log.propagate = False
    dives = {}
    for i, r in pick.items():
        dives[i] = deep_dive(model, frames[i], device, cap)
        dc, df = dives[i]["cc"], dives[i]["cf"]
        logging.info("frame %d %s n=%d: cc %s mu %.1f->%.1f E %.3f F %.3f | cf E %.3f F %.3f | final dq sum %.3g sum|.| %.3g",
                     i, r["formula"], r["n"], dc["status"], r["mu_ref"], dc["mu_final"], dc["E_err_per_atom"],
                     dc["F_mae"], df["E_err_per_atom"], df["F_mae"], dc["final_dq"]["sum"], dc["final_dq"]["sum_abs"])

    out = os.path.join(args.results_dir, "diag_cq_worst.json")
    os.makedirs(args.results_dir, exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"frames": rows, "deep_dives": {str(k): v for k, v in dives.items()}}, fh)
    logging.info("wrote %s", out)


if __name__ == "__main__":
    main()
