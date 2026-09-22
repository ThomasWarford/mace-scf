"""--pair_repulsion must add exactly the ZBL term, in the right place, for every model.

`--pair_repulsion` was honoured only for the upstream MACE/ScaleShiftMACE branches; for
mace_scf's own models the flag parsed and did nothing. These tests pin the energy delta
against the standalone ZBLBasis, which in one assertion catches a wrong `p`, a wrong sign,
a double count, and an insertion on the wrong side of the electrostatic term.
"""

import numpy as np
import pytest
import torch
from mace.modules import ZBLBasis
from mace.modules.utils import get_edge_vectors_and_lengths
from mace.tools.scatter import scatter_sum

from mace_scf.electrostatics.compiled_localsources import (
    build_compiled_local_source_evaluator,
)
from tests.models_for_radial_options import (
    ALL_MODELS,
    LOCAL_SOURCE_MODELS,
    MODELS_WITH_NODE_ENERGY,
    build_any,
    build_batch,
    build_model,
    graph_energy,
)

# float64 throughout, so the ZBL delta should reproduce to round-off.
ZBL_ABSOLUTE_TOLERANCE = 1e-12
# Guards against a trivially zero ZBL term making the comparison vacuous.
MINIMUM_ZBL_SIGNAL = 1e-6

COMPILED_PARITY_RTOL = 1e-6
COMPILED_PARITY_ATOL = 1e-8

NUM_POLYNOMIAL_CUTOFF = 6  # matches build_model's backbone


def _standalone_zbl(model, data, num_polynomial_cutoff=NUM_POLYNOMIAL_CUTOFF):
    """The ZBL term computed independently of the model's own forward."""
    _, lengths = get_edge_vectors_and_lengths(
        positions=data["positions"],
        edge_index=data["edge_index"],
        shifts=data["shifts"],
    )
    zbl = ZBLBasis(p=num_polynomial_cutoff)
    pair_node_energy = zbl(
        lengths, data["node_attrs"], data["edge_index"], model.atomic_numbers
    )
    pair_energy = scatter_sum(
        src=pair_node_energy,
        index=data["batch"],
        dim=-1,
        dim_size=int(data["ptr"].numel() - 1),
    )
    return pair_node_energy, pair_energy


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_pair_repulsion_adds_exactly_the_zbl_energy(model_name):
    data = build_batch().to_dict()
    off, num_poly = build_any(model_name)
    on, _ = build_any(model_name, pair_repulsion=True)

    assert not hasattr(off, "pair_repulsion")
    assert hasattr(on, "pair_repulsion")

    e_off = graph_energy(off, model_name, data).detach()
    e_on = graph_energy(on, model_name, data).detach()
    _, expected = _standalone_zbl(on, data, num_poly)
    expected = expected.detach()

    assert torch.max(torch.abs(expected)).item() > MINIMUM_ZBL_SIGNAL
    np.testing.assert_allclose(
        (e_on - e_off).numpy(), expected.numpy(), rtol=0, atol=ZBL_ABSOLUTE_TOLERANCE
    )


@pytest.mark.parametrize("model_name", MODELS_WITH_NODE_ENERGY)
def test_pair_repulsion_lands_in_node_energy(model_name):
    """node_energy is the local part only, computed before the Coulomb term is added."""
    data = build_batch().to_dict()
    off = build_model(model_name)
    on = build_model(model_name, pair_repulsion=True)

    n_off = off(data, compute_force=False, compute_stress=False)["node_energy"].detach()
    n_on = on(data, compute_force=False, compute_stress=False)["node_energy"].detach()
    expected, _ = _standalone_zbl(on, data)

    np.testing.assert_allclose(
        (n_on - n_off).numpy(),
        expected.detach().numpy(),
        rtol=0,
        atol=ZBL_ABSOLUTE_TOLERANCE,
    )


@pytest.mark.parametrize("model_name", LOCAL_SOURCE_MODELS)
def test_contributions_width_unchanged_when_disabled(model_name):
    """`contributions` is written out as BO_contributions by scripts/eval_local_charges.py.

    The ZBL column is appended only when enabled, unlike upstream MACE, precisely so this
    width stays put for every model trained before pair repulsion existed.
    """
    data = build_batch().to_dict()
    off = build_model(model_name)
    on = build_model(model_name, pair_repulsion=True)

    width_off = off(data, compute_force=False, compute_stress=False)[
        "contributions"
    ].shape[-1]
    width_on = on(data, compute_force=False, compute_stress=False)[
        "contributions"
    ].shape[-1]

    assert width_on == width_off + 1


@pytest.mark.parametrize("model_name", LOCAL_SOURCE_MODELS)
def test_zbl_is_independent_of_the_distance_transform(model_name):
    """ZBL reads the raw lengths, so Agnesi must not change the delta it contributes."""
    data = build_batch().to_dict()
    plain_off = build_model(model_name)
    plain_on = build_model(model_name, pair_repulsion=True)
    agnesi_off = build_model(model_name, distance_transform="Agnesi")
    agnesi_on = build_model(
        model_name, distance_transform="Agnesi", pair_repulsion=True
    )

    def energy(model):
        return model(data, compute_force=False, compute_stress=False)["energy"].detach()

    plain_delta = energy(plain_on) - energy(plain_off)
    agnesi_delta = energy(agnesi_on) - energy(agnesi_off)

    np.testing.assert_allclose(
        agnesi_delta.numpy(),
        plain_delta.numpy(),
        rtol=0,
        atol=ZBL_ABSOLUTE_TOLERANCE,
    )


@pytest.mark.parametrize("model_name", ["LocalSplitCharges", "LocalCharges"])
@pytest.mark.parametrize("pair_repulsion", [False, True])
def test_compiled_core_matches_eager(model_name, pair_repulsion):
    """The compiled cores copy a fixed submodule list, so ZBL has to be copied too.

    The False case is the regression guard for pre-ZBL checkpoints: the copy is `hasattr`-
    guarded, and an unguarded one would raise AttributeError here.
    """
    model = build_model(model_name, pair_repulsion=pair_repulsion)
    data = build_batch(n_graphs=1).to_dict()

    reference = model(data, compute_force=True, compute_stress=False)
    evaluator = build_compiled_local_source_evaluator(
        model, pbc_handling="pbc", enabled=False
    )
    actual = evaluator.evaluate(data)

    for key in ("energy", "forces"):
        np.testing.assert_allclose(
            actual[key].detach().cpu().numpy(),
            reference[key].detach().cpu().numpy(),
            rtol=COMPILED_PARITY_RTOL,
            atol=COMPILED_PARITY_ATOL,
        )
