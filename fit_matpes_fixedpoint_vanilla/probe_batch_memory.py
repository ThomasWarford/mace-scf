"""Worst-case GPU memory of one training step, per stage and batch size, for the FixedPoint recipe.

Run through train.sh so it sees exactly the training flags (one GPU, no DDP):

    sbatch -N1 -n1 --qos=debug --time=00:30:00 --export=ALL,CASE=vanilla,PROBE=1 \
        fit_matpes_fixedpoint_vanilla/train.sh

The batch size has to work in every stage, and in linearize_solve's fallback: the wrapper turns
any exception in linearize_and_solve_density (an OOM included) into a 15-step unroll_scf, and a
diverged SCF into an unroll silently, so the step that must fit is the larger of the dense
Jacobian solve and that unroll. The worst batch the DistributedSampler can draw is the B largest
frames, so each batch size B is probed on those (and on B random frames, for a typical figure).

Per (stage, path, B, batch kind) it reports peak allocated memory of a train step
(forward, loss, backward, clip, optimizer step) and of a validation forward, wall time, and
linearize_solve fallback warnings. Batch sizes come from PROBE_BATCH_SIZES (default "16 8 4").

PROBE_ATOMS="50 100 ..." instead sweeps atoms per batch (for a variable-batch-size / atom-budget
sampler): per budget A, the largest frames that fit in A and random frames up to A. Results go to
probe_atom_sweep.json.
"""

import glob
import json
import logging
import os
import time
from multiprocessing import Pool

import h5py
import numpy as np
import torch

from mace import tools
from mace.data import HDF5Dataset
from mace.tools import torch_geometric

import mace_scf.utils
import mace_scf.utils.run_train_utils as rtu
from mace_scf.data import ExtAtomicData
from mace_scf.electrostatics.loss import WeightedLoss
from mace_scf.utils.check_args import check_config_conflicts
from mace_scf.utils.load_data import load_train_valid_sets_from_preprocessed
from mace_scf.utils.model_training_wrappers import LINEARIZE_FALLBACK_UNROLLED_STEPS


class FallbackCounter(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages = []

    def emit(self, record):
        msg = record.getMessage()
        if "linearize_solve" in msg:
            self.messages.append(msg)


def count_atoms(path):
    out = []
    with h5py.File(path, "r") as f:
        batch_size = len(f["config_batch_0"].keys())
        for b in range(len(f.keys())):
            grp = f[f"config_batch_{b}"]
            for c in range(batch_size):
                out.append((b * batch_size + c, grp[f"config_{c}"]["atomic_numbers"].shape[0]))
    return path, out


def atom_counts(train_dir):
    files = sorted(glob.glob(os.path.join(train_dir, "*")))
    with Pool(min(32, len(files))) as pool:
        results = pool.map(count_atoms, files)
    return [(path, idx, n) for path, items in results for idx, n in items]


def make_batch(frames, args, z_table, device):
    datasets = {}
    items = []
    for path, idx, _ in frames:
        if path not in datasets:
            datasets[path] = HDF5Dataset(
                path, r_max=args.r_max, z_table=z_table,
                atomic_dataclass=ExtAtomicData,
                atomic_multipoles_max_l=args.atomic_multipoles_max_l,
            )
        items.append(datasets[path][idx])
    loader = torch_geometric.dataloader.DataLoader(items, batch_size=len(items), shuffle=False)
    return next(iter(loader)).to(device)


def measure(fn, device):
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    try:
        fn()
        status = "ok"
    except torch.OutOfMemoryError:
        status = "OOM"
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device) / 2**30
    torch.cuda.empty_cache()
    return status, peak, time.time() - t0


def main():
    args = mace_scf.utils.extended_arg_parser().parse_args()
    args.distributed = False
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    check_config_conflicts(args)
    tools.set_default_dtype(args.default_dtype)
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    total_gib = torch.cuda.get_device_properties(device).total_memory / 2**30
    batch_sizes = [int(b) for b in os.environ.get("PROBE_BATCH_SIZES", "16 8 4").split()]
    atom_budgets = [int(a) for a in os.environ.get("PROBE_ATOMS", "").split()]

    counter = FallbackCounter()
    logging.getLogger().addHandler(counter)

    frames = atom_counts(args.train_file)
    natoms = np.array([n for _, _, n in frames])
    logging.info(
        "train frames %d, atoms mean %.1f median %d p99 %d p99.9 %d max %d",
        len(natoms), natoms.mean(), np.median(natoms),
        np.percentile(natoms, 99), np.percentile(natoms, 99.9), natoms.max(),
    )
    order = np.argsort(natoms)[::-1]
    rng = np.random.default_rng(args.seed)

    # model built as run_train.py builds it; data only for z_table, E0s and the field-feature norms
    train_set, _, z_table, atomic_energies, _ = load_train_valid_sets_from_preprocessed(args)
    norm_idx = rng.choice(len(train_set), size=min(2048, len(train_set)), replace=False)
    norm_loader = torch_geometric.dataloader.DataLoader(
        [train_set[int(i)] for i in norm_idx], batch_size=args.batch_size, shuffle=False,
    )
    args.fermi_level_offset = rtu.get_fermi_level_offset(norm_loader, args, device)
    args.field_feature_norms = rtu.get_field_feature_norms(
        norm_loader, args, device, fermi_level_offset=args.fermi_level_offset
    )
    args.atom_density_scaling = rtu.get_atom_density_scaling(norm_loader, args, device, z_table=z_table)
    model = rtu.build_model(
        args=args, z_table=z_table, atomic_energies=atomic_energies,
        atomic_charges=None, train_loader=norm_loader,
    ).to(device)
    param_options = rtu.get_param_options(model, args)
    output_args = {"energy": True, "forces": True, "virials": False,
                   "stress": args.compute_stress, "polarizability": False}

    cases = []  # (B label, kind, frames)
    if atom_budgets:
        # memory vs atoms per batch, for an atom-budget (variable batch size) sampler. Two
        # compositions per budget A: "large" = the largest frames that fit in A (few big cells,
        # the worst case), "small" = random frames until A is reached (many small cells, typical)
        for A in atom_budgets:
            large, total = [], 0
            for i in order:
                if total + natoms[i] <= A:
                    large.append(frames[i]); total += natoms[i]
                if total >= 0.97 * A:
                    break
            small, total = [], 0
            for i in rng.permutation(len(frames)):
                small.append(frames[i]); total += natoms[i]
                if total >= A:
                    break
            cases += [(len(large), "large", large), (len(small), "small", small)]
    else:
        for B in batch_sizes:
            cases += [
                (B, "largest", [frames[i] for i in order[:B]]),
                (B, "random", [frames[i] for i in rng.choice(len(frames), size=B, replace=False)]),
            ]

    rows = []
    for B, kind, chosen in cases:
        batch = make_batch(chosen, args, z_table, device)
        n_atoms = int(batch.num_nodes)
        for stage in args.train_schedule:
            options = stage["fixed_point_training_options"]
            loss_fn = WeightedLoss(stage["loss"])
            optimizer = rtu.build_optimizer(param_options, args)
            if hasattr(optimizer, "train"):
                optimizer.train()
            wrapper = mace_scf.utils.make_model_wrapper(
                model=model, optimizer=optimizer, output_args=output_args,
                fixed_point_training_options=options,
            )
            paths = {options.mode: None}
            if options.mode == "linearize_solve":
                # what a linearize_solve fallback runs instead
                paths["linsolve_fallback_unroll"] = LINEARIZE_FALLBACK_UNROLLED_STEPS

            for path, unroll_steps in paths.items():
                def forward(training):
                    batch_dict = batch.to_dict()
                    if unroll_steps is None:
                        return wrapper(batch_dict, training=training)
                    return wrapper._forward_unroll_scf(
                        batch_dict, training, num_scf_steps=unroll_steps
                    )

                def train_step():
                    optimizer.zero_grad(set_to_none=True)
                    out = forward(True)
                    loss = loss_fn(pred=out, ref=batch)
                    loss.backward()
                    del out
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    optimizer.step()
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"non-finite loss {loss.item()}")

                def valid_step():
                    for p in model.parameters():
                        p.requires_grad_(False)
                        p.grad = None
                    forward(False)

                for phase, fn in (("train", train_step), ("valid", valid_step)):
                    before = len(counter.messages)
                    status, peak, seconds = measure(fn, device)
                    fallbacks = counter.messages[before:]
                    if any("OutOfMemory" in m for m in fallbacks):
                        status += "+fallback(OOM)"
                    elif fallbacks:
                        status += f"+fallback({len(fallbacks)})"
                    row = dict(stage=stage["name"], path=path, phase=phase, B=B, kind=kind,
                               n_atoms=n_atoms, status=status, peak_gib=round(peak, 2),
                               frac=round(peak / total_gib, 3), seconds=round(seconds, 2))
                    rows.append(row)
                    logging.info("PROBE %s", json.dumps(row))
                    for m in fallbacks:
                        logging.info("  fallback: %s", m[:300])
                for p in model.parameters():
                    p.requires_grad_(True)

    logging.info("GPU total %.1f GiB", total_gib)
    print(f"\n{'stage':9} {'path':25} {'phase':5} {'B':>3} {'kind':7} {'atoms':>5} {'GiB':>6} {'frac':>5} {'s':>6}  status")
    for r in rows:
        print(f"{r['stage']:9} {r['path']:25} {r['phase']:5} {r['B']:>3} {r['kind']:7} {r['n_atoms']:>5} "
              f"{r['peak_gib']:>6} {r['frac']:>5} {r['seconds']:>6}  {r['status']}")
    out = os.path.join(
        args.results_dir, "probe_atom_sweep.json" if atom_budgets else "probe_batch_memory.json"
    )
    os.makedirs(args.results_dir, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"gpu_total_gib": total_gib, "rows": rows}, f, indent=1)
    logging.info("wrote %s", out)


if __name__ == "__main__":
    main()
