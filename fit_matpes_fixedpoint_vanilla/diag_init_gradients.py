"""Where does the density come from, and which loss terms move which parameters, in direct mode?

Run through train.sh so it sees exactly the training flags (one GPU, no DDP):

    sbatch -N1 -n1 --qos=debug --time=00:30:00 \
        --export=ALL,CASE=vanilla,PROBE=1,PROBE_SCRIPT=diag_init_gradients.py \
        fit_matpes_fixedpoint_vanilla/train.sh

For the freshly initialised model, and for each checkpoint in DIAG_CHECKPOINTS (space-separated
.pt paths, e.g. the smoke run's direct epoch), on DIAG_BATCHES random train batches of
--batch_size frames, in the first train stage's mode (direct for vanilla.yaml):

- density decomposition: RMS of the field-independent (local) part, the field update, their sum
  and the DDEC reference, per component (q, atomic dipole);
- gradient norm of each weighted loss term with respect to each parameter group (backbone,
  local charge head, field-update head, energy readout), and of the total loss;
- the same restricted to the two charge-head output layers that --fixedpoint-initial-charge-head-
  scale multiplies by 0.01.
"""

import json
import logging
import os
from collections import defaultdict

import numpy as np
import torch

from mace import tools
from mace.tools import torch_geometric

import mace_scf.utils
import mace_scf.utils.run_train_utils as rtu
from mace_scf.electrostatics.loss import WeightedLoss
from mace_scf.utils.check_args import check_config_conflicts
from mace_scf.utils.load_data import load_train_valid_sets_from_preprocessed

GROUPS = {
    "backbone": ("node_embedding", "radial_embedding", "interactions", "products", "readouts",
                 "layer_feature_mixer", "atomic_energies_fn", "pair_repulsion_fn"),
    "local_charges": ("lr_source_maps",),
    "field_update": ("field_dependent_charges_map",),
    "energy_readout": ("local_electron_energy",),
}
HEAD_LAYERS = {
    "local_charges.linear_2": "lr_source_maps.",          # + ".linear_2." inside
    "field_update.element_select_out": "field_dependent_charges_map.element_select_out.",
}


def group_of(name):
    for group, prefixes in GROUPS.items():
        if name.startswith(prefixes):
            return group
    return "other"


def rms(x):
    return float(torch.sqrt(torch.mean(x.detach() ** 2)).item()) if x.numel() else 0.0


def analyse(model, wrapper, loss_cfg, batches, label):
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    head_masks = {
        k: [n.startswith(pref) and (k != "local_charges.linear_2" or ".linear_2." in n) for n in names]
        for k, pref in HEAD_LAYERS.items()
    }
    groups = [group_of(n) for n in names]
    terms = list(loss_cfg) + ["TOTAL"]

    gnorm = defaultdict(list)    # (term, group) -> [norm per batch]
    dens = defaultdict(list)
    for batch in batches:
        batch_dict = batch.to_dict()
        # the pieces of the direct-mode density, as _forward_direct builds them
        local_state = model.local_part(batch_dict, compute_force=True)
        p_local = local_state.field_independent_charge_density
        fermi_feats = model.features_from_fermi_level(
            batch_dict["batch"], local_state.positions, batch_dict["fermi_level"]
        )
        field_dep, field_feats = model.scf_step(
            batch_dict, local_state,
            charge_density_in=batch_dict["density_coefficients"],
            total_charges=batch_dict["density_coefficients"],
            fermi_level_features=fermi_feats,
        )
        ref = batch_dict["density_coefficients"]
        for comp, sl in (("q", slice(0, 1)), ("dipole", slice(1, 4))):
            dens[f"{comp}:local"].append(rms(p_local[:, sl]))
            dens[f"{comp}:field_update"].append(rms(field_dep[:, sl]))
            dens[f"{comp}:sum"].append(rms((p_local + field_dep)[:, sl]))
            dens[f"{comp}:reference"].append(rms(ref[:, sl]))
        dens["field_feats_rms(normalised)"].append(rms(field_feats))

        out = wrapper(batch_dict, training=True)
        for term in terms:
            cfg = loss_cfg if term == "TOTAL" else {term: loss_cfg[term]}
            loss = WeightedLoss(cfg)(pred=out, ref=batch)
            grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            sq = defaultdict(float)
            for g, grp, n in zip(grads, groups, names):
                if g is not None:
                    sq[grp] += float(torch.sum(g.detach() ** 2))
            for k, mask in head_masks.items():
                sq[k] = sum(float(torch.sum(g.detach() ** 2)) for g, m in zip(grads, mask) if m and g is not None)
            for grp in list(GROUPS) + ["other"] + list(HEAD_LAYERS):
                gnorm[(term, grp)].append(sq[grp] ** 0.5)
            gnorm[(term, "_loss")].append(float(loss.detach()))
        del out

    print(f"\n===== {label} =====")
    print("density RMS (e, e*A), median over batches:")
    for k, v in dens.items():
        print(f"   {k:28s} {np.median(v):.3e}")
    cols = list(GROUPS) + list(HEAD_LAYERS)
    print("gradient norm per loss term (rows) and parameter group (cols), median over batches:")
    print(f"   {'term':26s} {'loss':>10s} " + " ".join(f"{c[:22]:>22s}" for c in cols))
    for term in terms:
        print(f"   {term:26s} {np.median(gnorm[(term, '_loss')]):10.3e} "
              + " ".join(f"{np.median(gnorm[(term, c)]):22.3e}" for c in cols))
    pn = defaultdict(float)
    for n, p in model.named_parameters():
        pn[group_of(n)] += float(torch.sum(p.detach() ** 2))
        for k, pref in HEAD_LAYERS.items():
            if n.startswith(pref) and (k != "local_charges.linear_2" or ".linear_2." in n):
                pn[k] += float(torch.sum(p.detach() ** 2))
    print("   parameter norms:          " + " ".join(f"{pn[c] ** 0.5:22.3e}" for c in cols))
    return {"density": {k: float(np.median(v)) for k, v in dens.items()},
            "grad": {f"{t}|{g}": float(np.median(v)) for (t, g), v in gnorm.items()}}


def main():
    args = mace_scf.utils.extended_arg_parser().parse_args()
    args.distributed = False
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    check_config_conflicts(args)
    tools.set_default_dtype(args.default_dtype)
    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    train_set, _, z_table, atomic_energies, _ = load_train_valid_sets_from_preprocessed(args)
    rng = np.random.default_rng(args.seed)
    n_batches = int(os.environ.get("DIAG_BATCHES", "16"))
    idx = rng.choice(len(train_set), size=n_batches * args.batch_size, replace=False)
    loader = torch_geometric.dataloader.DataLoader(
        [train_set[int(i)] for i in idx], batch_size=args.batch_size, shuffle=False
    )
    batches = [b.to(device) for b in loader]

    args.fermi_level_offset = rtu.get_fermi_level_offset(loader, args, device)
    args.field_feature_norms = rtu.get_field_feature_norms(
        loader, args, device, fermi_level_offset=args.fermi_level_offset
    )
    args.atom_density_scaling = rtu.get_atom_density_scaling(loader, args, device, z_table=z_table)
    model = rtu.build_model(
        args=args, z_table=z_table, atomic_energies=atomic_energies,
        atomic_charges=None, train_loader=loader,
    ).to(device)
    stage = args.train_schedule[0]
    output_args = {"energy": True, "forces": True, "virials": False,
                   "stress": args.compute_stress, "polarizability": False}
    optimizer = rtu.build_optimizer(rtu.get_param_options(model, args), args)
    wrapper = mace_scf.utils.make_model_wrapper(
        model=model, optimizer=optimizer, output_args=output_args,
        fixed_point_training_options=stage["fixed_point_training_options"],
    )
    loss_cfg = stage["loss"]
    logging.info("stage %s, mode %s, loss %s", stage["name"],
                 stage["fixed_point_training_options"].mode, loss_cfg)
    for n, p in model.named_parameters():
        if n.startswith("lr_source_maps.0.") or n.startswith("field_dependent_charges_map.element_select_out"):
            logging.info("param %s shape %s norm %.3e", n, tuple(p.shape), p.detach().norm().item())

    results = {"init": analyse(model, wrapper, loss_cfg, batches, "initialisation")}
    for ckpt in os.environ.get("DIAG_CHECKPOINTS", "").split():
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        results[ckpt] = analyse(model, wrapper, loss_cfg, batches, os.path.basename(ckpt))

    out = os.path.join(args.results_dir, "diag_init_gradients.json")
    os.makedirs(args.results_dir, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    logging.info("wrote %s", out)


if __name__ == "__main__":
    main()
