"""--distance_transform must actually reach the radial basis of every mace_scf model.

The flag comes free from MACE's argument parser, so it has always parsed; until the
transform was threaded into the three RadialEmbeddingBlock construction sites it simply
did nothing, silently. These tests pin that it bites, and pin the upstream contract that
is easiest to invert by accident: the polynomial cutoff is evaluated on the raw distance
while only the Bessel argument is transformed.
"""

import numpy as np
import pytest
import torch
from mace.modules import RadialEmbeddingBlock
from mace.modules.radial import AgnesiTransform, SoftTransform

from tests.models_for_radial_options import (
    ALL_MODELS,
    build_any,
    build_batch,
    graph_energy,
)

TRANSFORM_CLASSES = {"Agnesi": AgnesiTransform, "Soft": SoftTransform}


@pytest.mark.parametrize("model_name", ALL_MODELS)
@pytest.mark.parametrize("transform", sorted(TRANSFORM_CLASSES))
def test_distance_transform_is_registered(model_name, transform):
    model, _ = build_any(model_name, distance_transform=transform)
    assert isinstance(
        model.radial_embedding.distance_transform, TRANSFORM_CLASSES[transform]
    )


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_no_distance_transform_registers_no_attribute(model_name):
    """"None" must leave the attribute absent, which is what keeps old models loadable."""
    model, _ = build_any(model_name)
    assert not hasattr(model.radial_embedding, "distance_transform")


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_agnesi_changes_the_energy(model_name):
    """Same weights, same batch: only the radial transform differs."""
    data = build_batch().to_dict()
    plain, _ = build_any(model_name)
    agnesi, _ = build_any(model_name, distance_transform="Agnesi")

    e_plain = graph_energy(plain, model_name, data)
    e_agnesi = graph_energy(agnesi, model_name, data)

    assert torch.max(torch.abs(e_agnesi - e_plain)).item() > 1e-6


def test_cutoff_uses_raw_distance_and_bessel_uses_transformed():
    """The ordering inside RadialEmbeddingBlock, pinned by hand.

    Upstream computes the cutoff envelope on the untransformed length and feeds only the
    Bessel basis the transformed one. A refactor that transforms first would still look
    plausible and would silently change every Agnesi model.
    """
    torch.set_default_dtype(torch.float64)
    block = RadialEmbeddingBlock(
        r_max=6.0, num_bessel=8, num_polynomial_cutoff=6, distance_transform="Agnesi"
    )
    lengths = torch.tensor([[0.9], [1.7], [3.1]], dtype=torch.float64)
    # Two elements, one-hot; edges all run 0 -> 1 so the covalent radii are well defined.
    node_attrs = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64
    ).repeat(2, 1)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    atomic_numbers = torch.tensor([1, 8], dtype=torch.int64)

    actual, _ = block(lengths, node_attrs, edge_index, atomic_numbers)

    expected_cutoff = block.cutoff_fn(lengths)  # raw
    transformed = block.distance_transform(
        lengths, node_attrs, edge_index, atomic_numbers
    )
    expected = block.bessel_fn(transformed) * expected_cutoff

    np.testing.assert_allclose(
        actual.detach().numpy(), expected.detach().numpy(), rtol=0, atol=1e-14
    )
    # and the wrong ordering really is different, so the assertion above has teeth
    wrong = block.bessel_fn(transformed) * block.cutoff_fn(transformed)
    assert np.abs(wrong.detach().numpy() - expected.detach().numpy()).max() > 1e-6
