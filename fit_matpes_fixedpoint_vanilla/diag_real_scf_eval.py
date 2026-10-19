"""Real-SCF validation errors of a FixedPoint checkpoint: SCF from the local guess, no reference input.

Training metrics of the SCF stages start from the reference DDEC density and Fermi level, so they
do not say how a model does in use. This runs, per validation batch: the local guess, a
constant-charge SCF (100 steps, tol 1e-6, mixing 0.3 -- the training settings; Fermi level
initialised from data, as scf_convergence_summary does), then energy / forces / stress at the SCF
result (the commented-out error path of mace_scf/utils/scf_convergence_summary.py), and writes one
line per frame: per-graph SCF status (scf_convergence_summary's criterion), steps, E error per atom,
force abs-error sum and max, stress error, and charge abs-error sum against DDEC.

Run through train.sh (no CUEQ, e3nn checkpoint), one task per GPU:

    SLURM_NTASKS=1 PROBE=1 PROBE_SCRIPT=diag_real_scf_eval.py CASE=linsolve_cq_huber_stress_lr0.0014_20ep \
        BACKBONE=r2 UPDATE=monotone TAG=wrand BATCH=4 REAL_CKPT=<.pt> REAL_SHARD=0 REAL_NSHARDS=4 \
        REAL_OUT=<dir> fit_matpes_fixedpoint_vanilla/train.sh

Writes <REAL_OUT>/<ckpt stem>_shard<k>.jsonl.
"""

import json
import logging
import os

import torch

from mace import tools
from mace.tools import torch_geometric

import mace_scf.utils
import mace_scf.utils.run_train_utils as rtu
from mace_scf.electrostatics.fixed_point_runner import FixedPointSCFRunner
from mace_scf.electrostatics.fixed_point_state import FixedPointSCFOptions
from mace_scf.utils.check_args import check_config_conflicts
from mace_scf.utils.load_data import load_train_valid_sets_from_preprocessed
from mace_scf.utils.scf_convergence_summary import _per_graph_statuses


def main():
    args = mace_scf.utils.extended_arg_parser().parse_args()
    args.distributed = False
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    check_config_conflicts(args)
    tools.set_default_dtype(args.default_dtype)
    device = torch.device("cuda")
    ckpt = os.environ["REAL_CKPT"]
    shard, nshards = int(os.environ.get("REAL_SHARD", "0")), int(os.environ.get("REAL_NSHARDS", "1"))
    out_dir = os.environ["REAL_OUT"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{os.path.basename(ckpt).removesuffix('.pt')}_shard{shard}.jsonl")

    _, valid_set, z_table, atomic_energies, _ = load_train_valid_sets_from_preprocessed(args)
    indices = list(range(shard, len(valid_set), nshards))
    frames = [valid_set[i] for i in indices]
    loader = torch_geometric.dataloader.DataLoader(frames, batch_size=args.batch_size, shuffle=False)
    norm_loader = torch_geometric.dataloader.DataLoader(frames[:2048], batch_size=args.batch_size)
    args.fermi_level_offset = rtu.get_fermi_level_offset(norm_loader, args, device)
    args.field_feature_norms = rtu.get_field_feature_norms(
        norm_loader, args, device, fermi_level_offset=args.fermi_level_offset
    )
    args.atom_density_scaling = rtu.get_atom_density_scaling(norm_loader, args, device, z_table=z_table)
    model = rtu.build_model(args=args, z_table=z_table, atomic_energies=atomic_energies,
                            atomic_charges=None, train_loader=norm_loader).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    options = FixedPointSCFOptions(
        num_scf_steps=100, scf_tolerance=1e-6, mixing_parameter=0.3, constant_charge=True,
        use_autograd_forces=True, initial_density="local_guess", initial_fermi_level="from_data",
    )
    runner = FixedPointSCFRunner(options)
    logging.info("shard %d/%d: %d frames -> %s", shard, nshards, len(frames), out_path)
    position = 0
    with open(out_path, "w") as out:
        for batch in loader:
            batch = batch.to(device)
            batch_dict = batch.to_dict()
            n_graphs = batch.num_graphs
            frame_ids = indices[position:position + n_graphs]
            position += n_graphs
            local_state = model.local_part(batch_dict, compute_force=True, compute_displacement=True)
            scf_result = runner.converge(
                model, batch_dict, local_state,
                runner.get_initial_density(local_state, batch_dict),
                runner.get_initial_fermi(model, local_state, batch_dict),
                compute_force=True,
            )
            statuses = _per_graph_statuses(batch_dict=batch_dict, scf_result=scf_result, scf_options=options)
            pred = model.build_observables(
                data=batch_dict, local_state=local_state, density=scf_result.density,
                fermi_level=scf_result.fermi_level, field_feats=scf_result.field_feats,
                training=False, compute_force=True, compute_virials=False, compute_stress=True,
            )
            with torch.no_grad():
                ptr = batch.ptr
                num_atoms = (ptr[1:] - ptr[:-1]).double()
                e_err = ((pred["energy"] - batch.energy) / num_atoms).tolist()
                f_abs = (pred["forces"] - batch.forces).abs()
                s_err = (pred["stress"] - batch.stress.view(-1, 3, 3)).abs()
                q_abs = (scf_result.density[:, 0] - batch.density_coefficients[:, 0]).abs()
                q_w = batch.density_coefficients_weight.view(-1)
                fin = scf_result.final_avg_abs_change.detach().reshape(-1).tolist()
                for g in range(n_graphs):
                    a, b = int(ptr[g]), int(ptr[g + 1])
                    out.write(json.dumps({
                        "frame": frame_ids[g], "n_atoms": b - a, "status": statuses[g],
                        "steps": int(scf_result.terminated_step), "final_avg_abs_change": fin[g],
                        "E_err_per_atom": e_err[g],
                        "F_abs_sum": float(f_abs[a:b].sum()), "F_err_max": float(f_abs[a:b].max()),
                        "S_sq_sum": float((s_err[g] ** 2).sum()), "S_err_max": float(s_err[g].max()),
                        "q_abs_sum": float(q_abs[a:b].sum()) if float(q_w[g]) > 0 else None,
                    }) + "\n")
            if hasattr(model, "batch_positions"):
                del model.batch_positions
            del pred, scf_result, local_state, batch_dict, batch
    logging.info("done %s", out_path)


if __name__ == "__main__":
    main()
