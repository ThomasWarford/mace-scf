"""Seed a UPDATE=monotone run from a vanilla (OneBodyVariableUpdate) checkpoint.

    python seed_monotone_checkpoint.py <vanilla checkpoint .pt> <output .pt>

The output is named for the monotone run's tag and stage (e.g.
<name>_run-1_direct_epoch-49.pt in its checkpoint dir), so --restart_latest loads it. Its model
weights are the vanilla ones: the checkpoint loader is not strict, so the monotone block's
softness_linear / softness_bias keep their initial values (S = softness_init everywhere). The
optimizer state cannot carry over (the monotone model has one more parameter in the
field_dependent_charges_map group and an extra undecayed group for softness_bias), so it is an
empty state with the new model's group layout; the next stage must set reset_optimizer.
"""

import sys

import torch

src, dst = sys.argv[1:3]
ckpt = torch.load(src, map_location="cpu", weights_only=False)
assert not any("softness" in k for k in ckpt["model"]), f"{src} is already a monotone checkpoint"

groups = []
for group in ckpt["optimizer"]["param_groups"]:
    groups.append(dict(group))
    if group["name"] == "field_dependent_charges_map":
        groups[-1]["params"] = list(group["params"]) + [None]  # + softness_linear.weight
        groups.append(dict(group, weight_decay=0.0, params=[None]))  # softness_bias
assert sum(g["name"] == "field_dependent_charges_map" for g in groups) == 2
i = 0
for group in groups:
    group["params"] = list(range(i, i + len(group["params"])))
    i += len(group["params"])
ckpt["optimizer"] = {"state": {}, "param_groups": groups}
torch.save(ckpt, dst)
print(f"wrote {dst}: {i} optimizer params in {len(groups)} groups")
