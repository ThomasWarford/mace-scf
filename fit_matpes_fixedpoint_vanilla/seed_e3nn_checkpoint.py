"""Turn a converted e3nn state dict (convert_and_probe_scf.py's <stem>_e3nn.pt) into a full
checkpoint an e3nn run can --restart_latest from, and time a few training steps of its last stage.

Run through train.sh (one GPU) with the e3nn run's flags (no CUEQ), so model and optimizer are
built exactly as run_train builds them:

    SLURM_NTASKS=1 PROBE=1 PROBE_SCRIPT=seed_e3nn_checkpoint.py CASE=<config> BACKBONE=r2 \
        UPDATE=monotone TAG=wrand E3NN_STATE=<..._e3nn.pt> SEED_DEST=<run ckpt dir>/<name>_run-1_direct_epoch-51.pt \
        fit_matpes_fixedpoint_vanilla/train.sh

The state dict is loaded strictly (run_train's own checkpoint load is not strict, so a key
mismatch there would silently keep initial weights). The optimizer state is empty, with the e3nn
model's group layout, so the seeded stage must set reset_optimizer. Then TIME_STEPS
forward/backward steps (no optimizer step) of the last stage's mode and loss on validation
batches, with the wrapper's fallback warnings counted.
"""

import logging
import os
import time

import numpy as np
import torch

from mace import tools
from mace.tools import torch_geometric

import mace_scf.utils
import mace_scf.utils.run_train_utils as rtu
from mace_scf.electrostatics.loss import WeightedLoss
from mace_scf.utils.check_args import check_config_conflicts
from mace_scf.utils.load_data import load_train_valid_sets_from_preprocessed


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
    src, dst = os.environ["E3NN_STATE"], os.environ["SEED_DEST"]
    assert not os.path.exists(dst), f"{dst} exists; not overwriting"
    n_steps = int(os.environ.get("TIME_STEPS", "20"))

    _, valid_set, z_table, atomic_energies, _ = load_train_valid_sets_from_preprocessed(args)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(valid_set), size=min(n_steps * args.batch_size, len(valid_set)), replace=False)
    loader = torch_geometric.dataloader.DataLoader([valid_set[int(i)] for i in idx],
                                                   batch_size=args.batch_size, shuffle=False)
    args.fermi_level_offset = rtu.get_fermi_level_offset(loader, args, device)
    args.field_feature_norms = rtu.get_field_feature_norms(loader, args, device,
                                                           fermi_level_offset=args.fermi_level_offset)
    args.atom_density_scaling = rtu.get_atom_density_scaling(loader, args, device, z_table=z_table)

    model = rtu.build_model(args=args, z_table=z_table, atomic_energies=atomic_energies,
                            atomic_charges=None, train_loader=loader).to(device)
    model.load_state_dict(torch.load(src, map_location=device, weights_only=False)["model"], strict=True)
    optimizer = rtu.build_optimizer(rtu.get_param_options(model, args), args)
    assert args.optimizer == "schedulefree"  # its NoOpLRScheduler's state is {}
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "lr_scheduler": {}}, dst)
    logging.info("wrote %s", dst)

    stage = args.train_schedule[-1]
    wrapper = mace_scf.utils.make_model_wrapper(
        model=model, optimizer=optimizer,
        output_args={"energy": True, "forces": True, "virials": False, "stress": args.compute_stress,
                     "polarizability": False},
        fixed_point_training_options=stage["fixed_point_training_options"],
    )
    loss_fn = WeightedLoss(stage["loss"])
    counter = Count()
    logging.getLogger().addHandler(counter)
    model.train()
    times, losses, natoms = [], [], []
    for batch in loader:
        batch = batch.to(device)
        torch.cuda.synchronize()
        t0 = time.time()
        out = wrapper(batch.to_dict(), training=True)
        loss = loss_fn(pred=out, ref=batch)
        loss.backward()
        torch.cuda.synchronize()
        times.append(time.time() - t0)
        losses.append(float(loss))
        natoms.append(int(batch.num_nodes))
        model.zero_grad(set_to_none=True)
        del out, loss
    logging.getLogger().removeHandler(counter)
    t = np.array(times[1:])  # first step includes warm-up
    logging.info("stage %s, %d steps of B=%d: s/step median %.2f mean %.2f max %.2f; loss median %.3g max %.3g; "
                 "atoms/batch median %d max %d; peak mem %.1f GiB",
                 stage["name"], len(times), args.batch_size, np.median(t), t.mean(), t.max(),
                 np.median(losses), max(losses), np.median(natoms), max(natoms),
                 torch.cuda.max_memory_allocated() / 2**30)
    logging.info("%d warnings during timing: %s", len(counter.messages),
                 sorted(set(m[:120] for m in counter.messages))[:10])


if __name__ == "__main__":
    main()
