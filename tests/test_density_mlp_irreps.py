"""FixedPointCore's field-independent density head (GeneralNonLinearBiasReadoutBlock) outputs
atomic dipoles only if its hidden irreps contain an l=1 irrep. density_MLP_irreps sets them
separately from MLP_irreps, which MACE's energy readout shares and which must stay scalar."""

import torch
from e3nn import o3

from tests.models_for_radial_options import build_batch, build_fixed_point_core
from tests.utils import seed_torch


def _local_density(model, data):
    return model.local_part(data).field_independent_charge_density.detach()


def test_default_density_head_has_no_dipoles():
    model = build_fixed_point_core()
    for block in model.lr_source_maps:
        assert block.hidden_irreps == o3.Irreps("16x0e")
    density = _local_density(model, build_batch().to_dict())
    assert torch.all(density[:, 1:] == 0.0)
    assert torch.any(density[:, 0] != 0.0)


def test_density_mlp_irreps_with_l1_gives_equivariant_dipoles():
    irreps = o3.Irreps("8x0e+8x1o")
    # two interactions, so the last energy readout is MACE's NonLinearReadoutBlock
    model = build_fixed_point_core(density_MLP_irreps=irreps, num_interactions=2)
    for block in model.lr_source_maps:
        assert block.hidden_irreps == irreps
    # MACE's energy readout keeps the scalar MLP_irreps
    assert model.readouts[-1].hidden_irreps == o3.Irreps("16x0e")
    density = _local_density(model, build_batch().to_dict())
    assert torch.any(density[:, 1:].abs() > 0.0)

    # each block maps rotated node features to rotated multipoles
    seed_torch(0)
    block = model.lr_source_maps[0]
    irreps_in = block.linear_1.irreps_in
    feats = irreps_in.randn(5, -1, dtype=torch.get_default_dtype())
    rotation = o3.rand_matrix(dtype=torch.get_default_dtype())
    out = block(None, feats)
    out_rot = block(None, feats @ irreps_in.D_from_matrix(rotation).T)
    expected = out @ block.irreps_out.D_from_matrix(rotation).T
    assert torch.allclose(out_rot, expected, atol=1e-10)
