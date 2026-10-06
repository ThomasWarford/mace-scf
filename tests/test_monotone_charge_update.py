"""Charge response to the Fermi level in the FixedPoint field-update block.

In constant-charge SCF (fixed_point_scf.converge_constant_charge) the Fermi level mu enters
the update block only through the l=0 field features, and dq_i/dmu drives both the Fukui
redistribution and the Newton step on mu. These tests pin down how q_i depends on mu.
"""

import copy
import functools
import os
from pathlib import Path

import pytest
import torch

import mace.tools
import mace.tools.torch_geometric
from mace.tools import torch_tools
from mace.tools.scatter import scatter_sum

from mace_scf.electrostatics import field_blocks
from mace_scf.electrostatics.fixed_point_runner import FixedPointSCFRunner
from mace_scf.electrostatics.fixed_point_state import FixedPointSCFOptions
from tests.models_for_radial_options import (
    build_batch,
    build_fixed_point_core,
    water_pair_atoms,
)
from tests.paths import reference_model, require_file
from tests.utils import dataset_from_atoms, disable_e3nn_codegen, seed_torch


# A trained FixedPoint model for the checks on real weights. Defaults to the regression
# fixture; point MACE_SCF_FIXEDPOINT_MODEL at any other FixedPoint .model to use that.
TRAINED_MODEL_PATH = Path(
    os.environ.get(
        "MACE_SCF_FIXEDPOINT_MODEL", str(reference_model("fixedpoint_onebodylinear"))
    )
)
# cuequivariance models need a GPU: run with MACE_DEVICE=cuda.
DEVICE = os.environ.get("MACE_DEVICE", "cpu")
MU_STEPS = (-2, -1, 0, 1, 2)
FIELD_FEATURE_WIDTHS = [1.5, 3.0]
# l=0 then l=1 norms for FIELD_FEATURE_WIDTHS, as used by fit_matpes_fixedpoint_vanilla.
MATPES_FIELD_FEATURE_NORMS = [16.85128938, 11.29358665, 0.45036201, 0.25651795]


def _update_config(cls, **options):
    return {
        "type": cls,
        "potential_embedding_cls": field_blocks.BiasedLinearPotentialEmbedding,
        "nonlinearity_cls": field_blocks.NoNonLinearity,
        **options,
    }


@functools.lru_cache(maxsize=None)
def _base_model():
    """A small FixedPointCore with the default update block. Building one takes minutes on
    CPU, so it is built once; other variants are copies with a different update block."""
    return build_fixed_point_core(field_feature_widths=FIELD_FEATURE_WIDTHS)


def _with_update_block(model, block):
    model = copy.deepcopy(model)
    model.field_dependent_charges_map = block
    return model.to(DEVICE)


def _trained_model():
    require_file(TRAINED_MODEL_PATH, "Trained FixedPoint model")
    torch_tools.set_default_dtype("float64")
    model = torch.load(TRAINED_MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = model.to(torch.float64)
    block = model.field_dependent_charges_map
    if not (
        type(block) is field_blocks.OneBodyVariableUpdate
        and type(block.nonlinearity) is field_blocks.NoNonLinearity
    ):
        pytest.skip(
            f"{TRAINED_MODEL_PATH.name} uses {type(block).__name__} with "
            f"{type(getattr(block, 'nonlinearity', None)).__name__}, not the linear default"
        )
    dataset = dataset_from_atoms(
        [water_pair_atoms()] * 2,
        cutoff=float(model.r_max),
        z_table=mace.tools.AtomicNumberTable([int(z) for z in model.atomic_numbers]),
        atomic_multipoles_max_l=int(model.coulomb_energy.density_max_l),
    )
    loader = mace.tools.torch_geometric.dataloader.DataLoader(
        dataset=dataset, batch_size=len(dataset), shuffle=False
    )
    return model, next(iter(loader)).to(DEVICE).to_dict()


def _charges_at_fermi_level(model, data, local_state, density, fermi_level):
    """q_i(mu) for one update at fixed input density, as converge_constant_charge does it."""
    node_fermi = torch.index_select(fermi_level, 0, data["batch"])
    fermi_features = model.features_from_fermi_level_nodewise(
        data["batch"], local_state.positions, node_fermi
    )
    field_dep, _ = model.scf_step(
        data,
        local_state,
        charge_density_in=density,
        total_charges=density,
        fermi_level_features=fermi_features,
    )
    return field_dep[:, 0]


def _assert_charge_linear_in_fermi_level(model, data):
    """q_i(mu) is affine (zero second differences) with the same dq_i/dmu at every mu.
    Returns dq_i/dmu per atom."""
    model.eval()
    local_state = model.local_part(data)
    density = local_state.field_independent_charge_density.detach()
    num_graphs = data["ptr"].numel() - 1
    mu0 = model.fermi_level_offset.expand(num_graphs).clone()

    def charges(mu):
        return _charges_at_fermi_level(model, data, local_state, density, mu)

    for delta in (0.1, 1.0):
        q = torch.stack([charges(mu0 + t * delta).detach() for t in MU_STEPS])
        scale = 1.0 + q.abs().max()
        second_differences = q[2:] - 2 * q[1:-1] + q[:-2]
        assert torch.allclose(
            second_differences, torch.zeros_like(second_differences), atol=1e-10 * scale
        ), f"q(mu) not affine for step {delta}: max |d2q| = {second_differences.abs().max()}"

    atoms = torch.arange(data["batch"].numel(), device=data["batch"].device)
    # Each q_i depends only on its own graph's mu: dq_i/dmu is that column of the Jacobian.
    slopes = torch.stack(
        [torch.autograd.functional.jacobian(charges, mu0 + t)[atoms, data["batch"]] for t in MU_STEPS]
    )
    assert torch.allclose(slopes, slopes[:1].expand_as(slopes), atol=1e-10, rtol=1e-10)

    dq_dmu = slopes[0]
    per_graph = scatter_sum(src=dq_dmu, index=data["batch"], dim=0, dim_size=num_graphs)
    print(
        f"\ndq_i/dmu per atom: {dq_dmu.tolist()}\n"
        f"fraction > 0: {(dq_dmu > 0).double().mean().item():.2f}, "
        f"per-graph sum: {per_graph.tolist()}"
    )
    return dq_dmu


def test_existing_block_charge_is_linear_in_mu_random_weights():
    model = _base_model().to(DEVICE)
    _assert_charge_linear_in_fermi_level(model, build_batch().to(DEVICE).to_dict())


def test_existing_block_charge_is_linear_in_mu_trained_model():
    # The default device matters for cuequivariance's naive kernels, which create
    # index tensors at call time.
    with torch.device(DEVICE):
        model, data = _trained_model()
        _assert_charge_linear_in_fermi_level(model, data)


# ---------------------------------------------------------------------------------------
# OneBodyMonotoneChargeUpdate
# ---------------------------------------------------------------------------------------

ACTIVATIONS = tuple(field_blocks.SOFTNESS_ACTIVATIONS)
SOFTNESS_INIT = {"softplus": 0.3, "sigmoid": 0.5, "2sigmoid": 1.0}


def _block(cls, seed=4, **options):
    """An update block with the irreps of _base_model's, built directly (fast)."""
    ref = _base_model().field_dependent_charges_map
    seed_torch(seed)
    with disable_e3nn_codegen():
        block = cls(
            node_attrs_irreps=ref.node_attrs_irreps,
            node_feats_irreps=ref.node_feats_irreps,
            edge_attrs_irreps=ref.edge_attrs_irreps,
            edge_feats_irreps=ref.edge_feats_irreps,
            target_irreps=ref.target_irreps,
            hidden_irreps=ref.hidden_irreps,
            avg_num_neighbors=ref.avg_num_neighbors,
            potential_irreps=ref.potential_irreps,
            charges_irreps=ref.charges_irreps,
            field_norm_factor=float(ref.field_norm_factor),
            potential_embedding_cls=field_blocks.BiasedLinearPotentialEmbedding,
            nonlinearity_cls=field_blocks.NoNonLinearity,
            **options,
        )
    return block.to(DEVICE)


def _monotone_block(activation, seed=4):
    """A monotone block away from its zero-initialised weights, so S depends on the
    environment."""
    block = _block(
        field_blocks.OneBodyMonotoneChargeUpdate,
        seed,
        softness_activation=activation,
        softness_init=SOFTNESS_INIT[activation],
    )
    with torch.no_grad():
        block.softness_linear.weight.normal_()
        block.softness_bias.normal_()
    return block


def _block_inputs(block, num_nodes=7, seed=0):
    generator = torch.Generator().manual_seed(seed)
    elements = torch.randint(0, block.node_attrs_irreps.dim, (num_nodes,), generator=generator)
    node_attrs = torch.nn.functional.one_hot(elements, block.node_attrs_irreps.dim).double()
    node_feats = torch.randn(num_nodes, block.node_feats_irreps.dim, generator=generator).double()
    potentials = torch.randn(num_nodes, block.potential_irreps.dim, generator=generator).double()
    return node_attrs.to(DEVICE), node_feats.to(DEVICE), potentials.to(DEVICE)


def _update(block, node_attrs, node_feats, potentials):
    unused = torch.empty(0, device=potentials.device)
    return block(node_attrs, node_feats, unused, unused, unused, potentials, unused, unused)


@pytest.mark.parametrize("activation", ACTIVATIONS)
def test_monotone_block_is_linear_in_v0_with_slope_minus_softness(activation):
    """Moving only the l=0 features moves q0 linearly, by exactly -sum S dv, whatever the
    l>=1 features are."""
    _, (low, high) = field_blocks.SOFTNESS_ACTIVATIONS[activation][1:]
    for seed in (4, 5, 6):
        block = _monotone_block(activation, seed=seed)
        node_attrs, node_feats, potentials = _block_inputs(block, seed=seed)
        n0 = block.num_scalar_potentials
        softness = block.softness(node_attrs, node_feats).detach()
        assert torch.all(softness > low) and torch.all(softness < high)

        direction = torch.zeros_like(potentials)
        direction[:, :n0] = torch.randn_like(potentials[:, :n0])
        expected_step = -(softness * direction[:, :n0]).sum(-1)
        for higher in (potentials[:, n0:], 10.0 * torch.randn_like(potentials[:, n0:])):
            base = torch.cat([potentials[:, :n0], higher], dim=-1)
            q = torch.stack(
                [_update(block, node_attrs, node_feats, base + t * direction)[:, 0] for t in MU_STEPS]
            ).detach()
            second_differences = q[2:] - 2 * q[1:-1] + q[:-2]
            assert torch.allclose(second_differences, torch.zeros_like(second_differences), atol=1e-10)
            assert torch.allclose(q[1:] - q[:-1], expected_step.expand_as(q[1:]), atol=1e-10)


@pytest.mark.parametrize("activation", ACTIVATIONS)
def test_monotone_block_initial_softness(activation):
    block = _block(
        field_blocks.OneBodyMonotoneChargeUpdate,
        softness_activation=activation,
        softness_init=SOFTNESS_INIT[activation],
    )
    node_attrs, node_feats, _ = _block_inputs(block)
    softness = block.softness(node_attrs, node_feats)
    assert torch.allclose(softness, torch.full_like(softness, SOFTNESS_INIT[activation]))


@pytest.mark.parametrize(
    "activation, softness_init",
    [("tanh", 0.1), ("softplus", 0.0), ("softplus", -1.0), ("sigmoid", 1.0), ("2sigmoid", 2.5)],
)
def test_monotone_block_rejects_bad_options(activation, softness_init):
    with pytest.raises(ValueError):
        _block(
            field_blocks.OneBodyMonotoneChargeUpdate,
            softness_activation=activation,
            softness_init=softness_init,
        )


@pytest.mark.parametrize("activation", ACTIVATIONS)
def test_monotone_block_keeps_parent_dipoles(activation):
    block = _monotone_block(activation)
    parent = _block(field_blocks.OneBodyVariableUpdate, seed=99)
    parent.load_state_dict(
        {k: v for k, v in block.state_dict().items() if not k.startswith("softness")}
    )
    node_attrs, node_feats, potentials = _block_inputs(block)
    ours = _update(block, node_attrs, node_feats, potentials)
    theirs = _update(parent, node_attrs, node_feats, potentials)
    assert torch.allclose(ours[:, 1:], theirs[:, 1:], atol=1e-12)
    assert not torch.allclose(ours[:, 0], theirs[:, 0])


@pytest.mark.parametrize("activation", ACTIVATIONS)
def test_monotone_model_charge_decreases_with_mu(activation):
    """End to end through features_from_fermi_level_nodewise + scf_step, as in
    converge_constant_charge: q_i is affine in mu with dq_i/dmu < 0 on every atom."""
    model = _with_update_block(_base_model(), _monotone_block(activation))
    dq_dmu = _assert_charge_linear_in_fermi_level(model, build_batch().to(DEVICE).to_dict())
    assert torch.all(dq_dmu < 0), dq_dmu


@functools.lru_cache(maxsize=None)
def _normalised_monotone_model():
    """Built through fixedpoint_update_config, with the MatPES fits' feature norms."""
    return build_fixed_point_core(
        field_feature_widths=FIELD_FEATURE_WIDTHS,
        field_feature_norms=MATPES_FIELD_FEATURE_NORMS,
        fixedpoint_update_config=_update_config(field_blocks.OneBodyMonotoneChargeUpdate),
    )


@pytest.mark.parametrize("activation", ACTIVATIONS)
def test_monotone_model_constant_charge_scf_runs(activation):
    """Constant-charge SCF converges from the default softness_init on the default
    features (own Coulomb potential excluded), with the l=0 feature norms of the MatPES
    FixedPoint fits. The norms matter: with norms of 1 the own-site l=0 feature per unit
    charge is the 7 A cell's Madelung term (about -16.5), and the SCF diverges for S above
    about 0.035."""
    model = _with_update_block(
        _normalised_monotone_model(),
        _block(field_blocks.OneBodyMonotoneChargeUpdate, softness_activation=activation),
    )
    model.eval()
    data = build_batch().to(DEVICE).to_dict()
    runner = FixedPointSCFRunner(
        FixedPointSCFOptions(
            num_scf_steps=100,
            scf_tolerance=1e-8,
            mixing_parameter=0.3,
            constant_charge=True,
            initial_density="local_guess",
            initial_fermi_level="zero",
        )
    )
    local_state = model.local_part(data)
    result = runner.converge(
        model=model,
        data=data,
        local_state=local_state,
        initial_density=runner.get_initial_density(local_state, data),
        initial_fermi_level=runner.get_initial_fermi(model, local_state, data),
    )
    total_charge = scatter_sum(
        src=result.density[:, 0],
        index=data["batch"],
        dim=0,
        dim_size=data["total_charge"].numel(),
    )
    assert torch.all(torch.isfinite(result.density))
    assert result.status == "converged", (result.status, result.terminated_step)
    assert torch.allclose(total_charge, data["total_charge"], atol=1e-3), total_charge
