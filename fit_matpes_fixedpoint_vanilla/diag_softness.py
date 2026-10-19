"""How does OneBodyMonotoneChargeUpdate's softness S move through training? Run through train.sh
(one GPU) on the model's flags, e.g.

    CASE=mono_unroll_cq UPDATE=monotone CUEQ=1 BATCH=16 PROBE_SCRIPT=diag_softness.py \
        DIAG_CHECKPOINTS="<ckpt> <ckpt> ..." fit_matpes_fixedpoint_vanilla/interactive_probe.sh

For each checkpoint in DIAG_CHECKPOINTS, on the first DIAG_MAX_FRAMES validation frames (default
4096): quantiles of S per Gaussian width over all atoms, the median S per element, and the
total dQ/dmu per frame from one SCF update at the reference density and Fermi level (as
converge_constant_charge computes it). Writes diag_softness.json to the results dir.
"""

import json
import logging
import os

import torch
from ase.data import chemical_symbols

from mace import tools
from mace.tools import torch_geometric
from mace.tools.scatter import scatter_sum

import mace_scf.utils
import mace_scf.utils.run_train_utils as rtu
from mace_scf.utils.check_args import check_config_conflicts
from mace_scf.utils.load_data import load_train_valid_sets_from_preprocessed

QS = [0.01, 0.1, 0.5, 0.9, 0.99, 1.0]


def total_dq_dmu(model, data, ls):
    node_mu = torch.index_select(data["fermi_level"], 0, data["batch"]).detach().clone().requires_grad_(True)
    feats = model.features_from_fermi_level_nodewise(data["batch"], ls.positions, node_mu)
    dep, _ = model.scf_step(data, ls, charge_density_in=data["density_coefficients"],
                            total_charges=data["density_coefficients"], fermi_level_features=feats)
    dq = torch.autograd.grad(dep[:, 0].sum(), node_mu)[0]
    return scatter_sum(dq, data["batch"], dim=-1, dim_size=data["ptr"].numel() - 1).detach()


def main():
    args = mace_scf.utils.extended_arg_parser().parse_args()
    args.distributed = False
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    check_config_conflicts(args)
    tools.set_default_dtype(args.default_dtype)
    device = torch.device("cuda")

    _, valid_set, z_table, atomic_energies, _ = load_train_valid_sets_from_preprocessed(args)
    n = min(int(os.environ.get("DIAG_MAX_FRAMES", "4096")), len(valid_set))
    frames = [valid_set[i] for i in range(n)]
    loader = torch_geometric.dataloader.DataLoader(frames, batch_size=args.batch_size, shuffle=False)
    args.fermi_level_offset = rtu.get_fermi_level_offset(loader, args, device)
    args.field_feature_norms = rtu.get_field_feature_norms(loader, args, device,
                                                           fermi_level_offset=args.fermi_level_offset)
    args.atom_density_scaling = rtu.get_atom_density_scaling(loader, args, device, z_table=z_table)
    model = rtu.build_model(args=args, z_table=z_table, atomic_energies=atomic_energies,
                            atomic_charges=None, train_loader=loader).to(device)
    block = model.field_dependent_charges_map
    assert hasattr(block, "softness"), f"{type(block).__name__} has no softness"
    batches = [b.to(device).to_dict() for b in loader]

    results = {}
    qs = torch.tensor(QS, dtype=torch.float64)
    for ckpt in os.environ["DIAG_CHECKPOINTS"].split():
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"])
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        S_all, Z_all, dQ = [], [], []
        for data in batches:
            ls = model.local_part(data, compute_force=False)
            S_all.append(block.softness(data["node_attrs"], ls.all_layer_feats).detach().cpu())
            Z_all.append(data["node_attrs"].argmax(-1).cpu())
            dQ.append(total_dq_dmu(model, data, ls).cpu())
        S, Z, dQ = torch.cat(S_all), torch.cat(Z_all), torch.cat(dQ)
        by_el = {chemical_symbols[z_table.zs[z]]: [float(x) for x in S[Z == z].median(0).values]
                 for z in Z.unique().tolist()}
        r = dict(S_quantiles=torch.quantile(S.double(), qs, dim=0).T.tolist(),
                 dQdmu_quantiles=torch.quantile(dQ.double(), qs).tolist(),
                 S_median_by_element=by_el)
        results[os.path.basename(ckpt)] = r
        logging.info("%s: S median %s p99 %s max %s | dQ/dmu median %.4f",
                     os.path.basename(ckpt), [round(w[2], 4) for w in r["S_quantiles"]],
                     [round(w[4], 4) for w in r["S_quantiles"]], [round(w[5], 4) for w in r["S_quantiles"]],
                     r["dQdmu_quantiles"][2])

    out = os.path.join(args.results_dir, "diag_softness.json")
    os.makedirs(args.results_dir, exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"frames": n, "quantiles": QS, "checkpoints": results}, fh)
    logging.info("wrote %s", out)


if __name__ == "__main__":
    main()
