"""Find the validation frames on which linearize_solve validation blows up, and why.

Validation of the stress linsolve run (epoch 61) had MAE_F 378 but RMSE_F 57 eV/A: a few frames with
forces ~1e4 eV/A. This evaluates a checkpoint in the run's linearize_solve validation mode over a
shard of the validation set and records, per batch: energy / force / stress errors, SCF status and
steps, linearize fallbacks, and from inside linearize_and_solve_density the SCF residual at the
solved state, the two chord-step sizes, and the Jacobian's smallest singular value and condition
number; and the starting SCF's status, steps and per-frame final mean |dq|.

Run through train.sh (no CUEQ) with the linsolve run's flags, one task per GPU:

    SLURM_NTASKS=1 PROBE=1 PROBE_SCRIPT=diag_linsolve_outliers.py CASE=linsolve_cq_huber_stress_lr0.0014_20ep \
        BACKBONE=r2 UPDATE=monotone TAG=wrand BATCH=4 DIAG_CKPT=<.pt> DIAG_SHARD=0 DIAG_NSHARDS=4 \
        DIAG_OUT=<dir> fit_matpes_fixedpoint_vanilla/train.sh

Writes <DIAG_OUT>/<ckpt stem>_shard<k>.jsonl (one line per batch).
"""

import json
import logging
import os

import torch

from mace import tools
from mace.tools import torch_geometric

import mace_scf.utils
import mace_scf.utils.linearize_solve as linsolve
import mace_scf.utils.run_train_utils as rtu
from mace_scf.utils.check_args import check_config_conflicts
from mace_scf.utils.load_data import load_train_valid_sets_from_preprocessed


RECORD = {}


def patch_linearize_solve():
    """Record the Jacobian's spectrum and the chord-step sizes of each linearize_and_solve_density."""
    jacobian_fn = linsolve._jacobian_from_vmapped_vjps

    def recording_jacobian(outputs, inputs, **kwargs):
        jacobian = jacobian_fn(outputs, inputs, **kwargs)
        with torch.no_grad():
            singular_values = torch.linalg.svdvals(jacobian.detach())
            RECORD["dim"] = int(jacobian.shape[0])
            RECORD["smin"] = float(singular_values[-1])
            RECORD["smax"] = float(singular_values[0])
            RECORD["residual_norm"] = float(outputs.detach().norm())
        return jacobian

    linsolve._jacobian_from_vmapped_vjps = recording_jacobian

    solve_fn = torch.linalg.solve

    def recording_solve(A, B, *a, **k):
        x = solve_fn(A, B, *a, **k)
        RECORD.setdefault("delta_norms", []).append(float(x.detach().norm()))
        RECORD.setdefault("delta_max_abs", []).append(float(x.detach().abs().max()))
        RECORD.setdefault("rhs_norms", []).append(float(B.detach().norm()))
        return x

    torch.linalg.solve = recording_solve

    # the SCF that linearize_solve starts from: status, steps, and per-frame final mean |dq|
    # (implicit mode's unconverged gate is final_avg_abs_change > SCF_FALLBACK_ABS_CHANGE_THRESHOLD)
    from mace_scf.electrostatics.fixed_point_runner import FixedPointSCFRunner
    converge_fn = FixedPointSCFRunner.converge

    def recording_converge(self, *a, **k):
        result = converge_fn(self, *a, **k)
        if "scf_status" not in RECORD:
            RECORD["scf_status"] = result.status
            RECORD["scf_terminated_step"] = int(result.terminated_step)
            RECORD["scf_final_avg_abs_change"] = result.final_avg_abs_change.detach().reshape(-1).tolist()
        return result

    FixedPointSCFRunner.converge = recording_converge


class Count(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def main():
    args = mace_scf.utils.extended_arg_parser().parse_args()
    args.distributed = False
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    check_config_conflicts(args)
    tools.set_default_dtype(args.default_dtype)
    device = torch.device("cuda")
    ckpt = os.environ["DIAG_CKPT"]
    shard, nshards = int(os.environ.get("DIAG_SHARD", "0")), int(os.environ.get("DIAG_NSHARDS", "1"))
    out_dir = os.environ["DIAG_OUT"]
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.basename(ckpt).removesuffix(".pt")
    # DIAG_LINSOLVE_MODE=newton1: the original one-step attached-Jacobian scheme; it and
    # DIAG_MAX_NEWTON_STEP need the (uncommitted) LINEARIZE_STEP_MODE / MAX_NEWTON_STEP switches
    linsolve.LINEARIZE_STEP_MODE = os.environ.get(
        "DIAG_LINSOLVE_MODE", getattr(linsolve, "LINEARIZE_STEP_MODE", "chord2")
    )
    mode_tag = "" if linsolve.LINEARIZE_STEP_MODE == "chord2" else f"_{linsolve.LINEARIZE_STEP_MODE}"
    # DIAG_MAX_NEWTON_STEP=x: unroll fallback when the first Newton step's max |delta| > x
    if os.environ.get("DIAG_MAX_NEWTON_STEP"):
        linsolve.MAX_NEWTON_STEP = float(os.environ["DIAG_MAX_NEWTON_STEP"])
        mode_tag += f"_maxstep{linsolve.MAX_NEWTON_STEP:g}"
    out_path = os.path.join(out_dir, f"{stem}{mode_tag}_shard{shard}.jsonl")

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
    optimizer = rtu.build_optimizer(rtu.get_param_options(model, args), args)
    stage = args.train_schedule[-1]
    assert stage["fixed_point_training_options"].mode == "linearize_solve"
    wrapper = mace_scf.utils.make_model_wrapper(
        model=model, optimizer=optimizer,
        output_args={"energy": True, "forces": True, "virials": False,
                     "stress": args.compute_stress, "polarizability": False},
        fixed_point_training_options=stage["fixed_point_training_options"],
    )
    patch_linearize_solve()
    counter = Count()
    logging.getLogger().addHandler(counter)

    logging.info("shard %d/%d: %d frames, %d batches -> %s", shard, nshards, len(frames),
                 len(loader), out_path)
    position = 0
    with open(out_path, "w") as out:
        for batch_number, batch in enumerate(loader):
            batch = batch.to(device)
            n_graphs = batch.num_graphs
            frame_ids = indices[position:position + n_graphs]
            position += n_graphs
            RECORD.clear()
            before = len(counter.messages)
            for p in model.parameters():
                p.requires_grad_(False)
            pred = wrapper(batch.to_dict(), training=False)
            num_atoms = (batch.ptr[1:] - batch.ptr[:-1]).double()
            row = {"batch": batch_number, "frames": frame_ids, "n_atoms": num_atoms.long().tolist()}
            with torch.no_grad():
                row["E_err_per_atom"] = ((pred["energy"] - batch.energy) / num_atoms).abs().tolist()
                force_err = (pred["forces"] - batch.forces).abs().amax(dim=-1)
                row["F_err_max_per_frame"] = [
                    float(force_err[batch.ptr[g]:batch.ptr[g + 1]].max()) for g in range(n_graphs)
                ]
                if pred.get("stress") is not None:
                    row["S_err_max"] = (
                        (pred["stress"] - batch.stress.view(-1, 3, 3)).abs().amax(dim=(-1, -2)).tolist()
                    )
            row.update(RECORD)
            row["warnings"] = [m[:300] for m in counter.messages[before:]]
            out.write(json.dumps(row) + "\n")
            out.flush()
            if max(row["F_err_max_per_frame"]) > 10.0:
                logging.info("OUTLIER %s", json.dumps(row))
            del pred
    logging.info("done %s", out_path)


if __name__ == "__main__":
    main()
