"""Convert a cueq-trained FixedPoint checkpoint to e3nn, check parity, and probe its SCF behaviour.

Run through train.sh (one GPU) so the model is built from exactly the training flags, with CUEQ=1
and the run's config/batch so the cueq twin matches the checkpoint:

    sbatch -N1 -n1 --qos=debug --time=00:30:00 --export=ALL,CASE=direct_docs_lr0.028,CUEQ=1,BATCH=16,\
PROBE=1,PROBE_SCRIPT=convert_and_probe_scf.py,CONVERT_CKPT=<..._direct_epoch-49.pt> \
        fit_matpes_fixedpoint_vanilla/train.sh

1. conversion: builds the cueq model (the checkpoint's) and an e3nn twin (same flags,
   --enable_cueq off) and moves the weights with MACE's convert_cueq_e3nn.transfer_weights.
   kmax per layer is read off the e3nn model's products, as tests/test_cueq_equivalence.py does
   for the other direction. Saves <ckpt stem>_e3nn.model (whole module) and
   <ckpt stem>_e3nn.pt ({"model": state_dict}, for seeding e3nn runs, e.g. linearize_solve);
2. parity on PROBE_BATCHES validation batches: direct mode, and unroll_scf (10 steps, from data);
3. SCF probe on the e3nn model, same batches, no gradients needed beyond dq/dmu:
   - dQ/dmu per frame (one SCF update at the reference density, autograd w.r.t. the per-atom
     Fermi level, as converge_constant_charge computes it) and the total-charge error of the
     direct-mode density at the reference mu; their ratio is the Newton jump the
     constant-charge loop would ask for at step 0;
   - full SCF from the local guess (100 steps, mixing 0.3, tol 1e-6) in constant-charge and in
     constant-Fermi (mu from data) mode: per frame convergence, steps, |mu - mu_ref|, |Q - Q_ref|,
     and energy / force / charge errors against the reference.
"""

import json
import logging
import os

import numpy as np
import torch

from mace import tools
from mace.tools import torch_geometric
from mace.tools.scatter import scatter_sum
import mace.cli.convert_cueq_e3nn as cueq_e3nn

import mace_scf.utils
import mace_scf.utils.run_train_utils as rtu
from mace_scf.electrostatics.fixed_point_runner import FixedPointSCFRunner
from mace_scf.electrostatics.fixed_point_state import FixedPointSCFOptions, FixedPointTrainingOptions
from mace_scf.utils.check_args import check_config_conflicts
from mace_scf.utils.load_data import load_train_valid_sets_from_preprocessed


def build(args, z_table, atomic_energies, loader, cueq, device):
    args.enable_cueq = cueq
    return rtu.build_model(args=args, z_table=z_table, atomic_energies=atomic_energies,
                           atomic_charges=None, train_loader=loader).to(device)


def to_e3nn(cueq_model, e3nn_model, correlation):
    """cueq -> e3nn, the inverse of tests/test_cueq_equivalence.py's transfer_e3nn_to_cueq."""
    src = cueq_model.state_dict()
    if not any(k.endswith("symmetric_contractions.weight") for k in src):
        # symmetric contraction stayed e3nn: names line up, copy directly
        tgt = e3nn_model.state_dict()
        for k, v in src.items():
            if k in tgt:
                tgt[k] = v if v.shape == tgt[k].shape else cueq_e3nn.reshape_like(v, tgt[k].shape)
        e3nn_model.load_state_dict(tgt, strict=True)
        return
    kmax_pairs = [[i, len(p.symmetric_contractions.contractions) - 1]
                  for i, p in enumerate(e3nn_model.products)]
    original = cueq_e3nn.get_kmax_pairs
    cueq_e3nn.get_kmax_pairs = lambda *_a, **_k: kmax_pairs
    try:
        cueq_e3nn.transfer_weights(
            cueq_model, e3nn_model,
            num_product_irreps=max(k for _, k in kmax_pairs),
            correlation=correlation, num_layers=len(e3nn_model.interactions),
            use_reduced_cg=False,
        )
    finally:
        cueq_e3nn.get_kmax_pairs = original


def wrapper_for(model, mode, scf=None):
    opts = FixedPointTrainingOptions(mode=mode, scf=scf)
    return mace_scf.utils.make_model_wrapper(
        model=model, optimizer=None,
        output_args={"energy": True, "forces": True, "virials": False, "stress": False,
                     "polarizability": False},
        fixed_point_training_options=opts,
    )


def parity(a, b, batches, mode, scf=None):
    wa, wb = wrapper_for(a, mode, scf), wrapper_for(b, mode, scf)
    worst = {}
    for batch in batches:
        oa = wa(batch.to_dict(), training=False)
        ob = wb(batch.to_dict(), training=False)
        for k in ("energy", "forces", "density_coefficients"):
            d = (oa[k] - ob[k]).abs().max().item()
            scale = oa[k].abs().max().item()
            worst[k] = max(worst.get(k, (0, 0)), (d, d / max(scale, 1e-30)))
    return {k: {"max_abs": v[0], "max_rel": v[1]} for k, v in worst.items()}


def dq_dmu_probe(model, batches):
    rows = []
    for batch in batches:
        data = batch.to_dict()
        ls = model.local_part(data, compute_force=False)
        node_mu = torch.index_select(data["fermi_level"], 0, data["batch"]).clone().requires_grad_(True)
        feats = model.features_from_fermi_level_nodewise(data["batch"], ls.positions, node_mu)
        dep, _ = model.scf_step(data, ls, charge_density_in=data["density_coefficients"],
                                total_charges=data["density_coefficients"], fermi_level_features=feats)
        dq = torch.autograd.grad(dep[:, 0].sum(), node_mu)[0]
        n = data["ptr"][1:] - data["ptr"][:-1]
        G = n.numel()
        dQ = scatter_sum(dq, data["batch"], dim=-1, dim_size=G)
        Q = scatter_sum((ls.field_independent_charge_density + dep)[:, 0].detach(), data["batch"], dim=-1, dim_size=G)
        err = Q - data["total_charge"]
        for g in range(G):
            rows.append(dict(n=int(n[g]), dQdmu=float(dQ[g]), Q_err=float(err[g]),
                             jump=float(-err[g] / dQ[g]) if dQ[g] != 0 else float("inf")))
    return rows


def scf_probe(model, batches, constant_charge):
    opts = FixedPointSCFOptions(num_scf_steps=100, scf_tolerance=1e-6, mixing_parameter=0.3,
                                constant_charge=constant_charge, initial_density="local_guess",
                                initial_fermi_level="from_data")
    runner = FixedPointSCFRunner(opts)
    rows = []
    for batch in batches:
        data = batch.to_dict()
        ls = model.local_part(data, compute_force=True)
        d0 = runner.get_initial_density(ls, data)
        mu0 = runner.get_initial_fermi(model, ls, data)
        # compute_force=True keeps the SCF graph so forces include d(density)/dR, as the
        # runner's eval does; detaching the converged density (an earlier version) drops it
        res = runner.converge(model, data, ls, d0, mu0, compute_force=True)
        out = model.build_observables(data=data, local_state=ls, density=res.density,
                                      fermi_level=res.fermi_level, field_feats=res.field_feats,
                                      training=False, compute_force=True)
        n = data["ptr"][1:] - data["ptr"][:-1]
        G = n.numel()
        Q = scatter_sum(res.density[:, 0].detach(), data["batch"], dim=-1, dim_size=G)
        ferr = (out["forces"].detach() - data["forces"]).abs()
        ferr_g = scatter_sum(ferr.sum(-1), data["batch"], dim=-1, dim_size=G) / (3 * n)
        qerr_g = scatter_sum((res.density[:, 0].detach() - data["density_coefficients"][:, 0]).abs(),
                             data["batch"], dim=-1, dim_size=G) / n
        for g in range(G):
            rows.append(dict(
                n=int(n[g]),
                converged=bool(res.final_avg_abs_change[g] <= opts.scf_tolerance)
                and (not constant_charge or abs(float(Q[g] - data["total_charge"][g])) < 1e-4),
                final_change=float(res.final_avg_abs_change[g]),
                batch_status=res.status, batch_steps=int(res.terminated_step) + 2,
                mu_drift=float((res.fermi_level[g] - data["fermi_level"][g]).abs()),
                Q_err=float((Q[g] - data["total_charge"][g]).abs()),
                E_err_per_atom=float((out["energy"][g].detach() - data["energy"][g]).abs() / n[g]),
                F_mae=float(ferr_g[g]), q_mae=float(qerr_g[g]),
            ))
    return rows


def summarise(rows, keys):
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=float)
        v = v[np.isfinite(v)]
        out[k] = {"median": float(np.median(v)), "p10": float(np.percentile(v, 10)),
                  "p90": float(np.percentile(v, 90)), "max": float(v.max()), "mean": float(v.mean())} if v.size else None
    return out


def main():
    args = mace_scf.utils.extended_arg_parser().parse_args()
    args.distributed = False
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    check_config_conflicts(args)
    tools.set_default_dtype(args.default_dtype)
    device = torch.device("cuda")
    ckpt = os.environ["CONVERT_CKPT"]
    n_batches = int(os.environ.get("PROBE_BATCHES", "16"))

    _, valid_set, z_table, atomic_energies, _ = load_train_valid_sets_from_preprocessed(args)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(valid_set), size=min(n_batches * args.batch_size, len(valid_set)), replace=False)
    loader = torch_geometric.dataloader.DataLoader([valid_set[int(i)] for i in idx],
                                                   batch_size=args.batch_size, shuffle=False)
    batches = [b.to(device) for b in loader]
    args.fermi_level_offset = rtu.get_fermi_level_offset(loader, args, device)
    args.field_feature_norms = rtu.get_field_feature_norms(loader, args, device,
                                                           fermi_level_offset=args.fermi_level_offset)
    args.atom_density_scaling = rtu.get_atom_density_scaling(loader, args, device, z_table=z_table)

    cueq_model = build(args, z_table, atomic_energies, loader, True, device)
    state = torch.load(ckpt, map_location=device, weights_only=False)["model"]
    cueq_model.load_state_dict(state)
    e3nn_model = build(args, z_table, atomic_energies, loader, False, device)
    to_e3nn(cueq_model, e3nn_model, args.correlation)
    for m in (cueq_model, e3nn_model):
        m.eval()

    results = {"checkpoint": ckpt}
    results["parity_direct"] = parity(cueq_model, e3nn_model, batches, "direct")
    results["parity_unroll10"] = parity(
        cueq_model, e3nn_model, batches, "unroll_scf",
        FixedPointSCFOptions(num_scf_steps=10, mixing_parameter=0.3, constant_charge=True,
                             initial_density="from_data", initial_fermi_level="from_data"))
    logging.info("parity direct: %s", results["parity_direct"])
    logging.info("parity unroll10: %s", results["parity_unroll10"])

    stem = os.path.splitext(ckpt)[0]
    torch.save(e3nn_model, stem + "_e3nn.model")
    torch.save({"model": e3nn_model.state_dict()}, stem + "_e3nn.pt")
    logging.info("saved %s_e3nn.model / .pt", stem)

    rows = dq_dmu_probe(e3nn_model, batches)
    results["dq_dmu"] = summarise(rows, ["dQdmu", "Q_err", "jump"])
    results["dq_dmu"]["abs"] = summarise([{k: abs(v) for k, v in r.items()} for r in rows], ["dQdmu", "Q_err", "jump"])
    for cc in (True, False):
        r = scf_probe(e3nn_model, batches, cc)
        key = "scf_constant_charge" if cc else "scf_constant_fermi"
        results[key] = summarise(r, ["final_change", "mu_drift", "Q_err", "E_err_per_atom", "F_mae", "q_mae"])
        results[key]["frac_converged"] = float(np.mean([x["converged"] for x in r]))
        results[key]["batch_status"] = {s: sum(1 for x in r if x["batch_status"] == s) for s in set(x["batch_status"] for x in r)}
    # direct-mode errors on the same frames, for reference
    w = wrapper_for(e3nn_model, "direct")
    e, f = [], []
    for batch in batches:
        o = w(batch.to_dict(), training=False)
        n = batch.ptr[1:] - batch.ptr[:-1]
        e += ((o["energy"].detach() - batch.energy).abs() / n).tolist()
        f.append((o["forces"].detach() - batch.forces).abs().mean().item())
    results["direct_ref"] = {"E_err_per_atom_median": float(np.median(e)), "E_mae": float(np.mean(e)),
                             "F_mae_mean_over_batches": float(np.mean(f))}

    out = os.path.join(args.results_dir, "convert_and_probe_scf.json")
    os.makedirs(args.results_dir, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(results, fh, indent=1)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
