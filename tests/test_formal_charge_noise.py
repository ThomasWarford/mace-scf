import argparse

import pytest
import torch

from mace.tools.torch_geometric import Data

from mace_scf.data.augmentation import (
    FormalChargeNoiseTransform,
    TransformedDataset,
    add_formal_charge_noise,
)
from mace_scf.utils.check_args import check_formal_charge_noise


def make_data(charges, batch=None):
    kwargs = {"charges": torch.tensor(charges, dtype=torch.get_default_dtype())}
    if batch is not None:
        kwargs["batch"] = torch.tensor(batch, dtype=torch.long)
    return Data(**kwargs)


def test_noise_preserves_total_charge():
    data = make_data([-2.0, 1.0, 1.0])
    noised = FormalChargeNoiseTransform(0.3)(data)
    assert noised.charges.sum().item() == pytest.approx(0.0, abs=1e-6)
    assert not torch.allclose(noised.charges, data.charges)


def test_noise_is_zero_sum_per_configuration_in_a_batch():
    data = make_data([-2.0, 1.0, 1.0, -1.0, 1.0], batch=[0, 0, 0, 1, 1])
    noised = FormalChargeNoiseTransform(0.3)(data)
    per_config = torch.zeros(2).scatter_add_(0, data.batch, noised.charges - data.charges)
    assert torch.allclose(per_config, torch.zeros(2), atol=1e-6)


def test_noise_has_the_requested_standard_deviation():
    torch.manual_seed(0)
    transform = FormalChargeNoiseTransform(0.25)
    data = make_data([0.0] * 8)
    samples = torch.stack([transform(data).charges for _ in range(4000)])
    assert samples.std().item() == pytest.approx(0.25, rel=0.05)


def test_single_atom_configuration_gets_no_noise():
    data = make_data([1.0])
    noised = FormalChargeNoiseTransform(0.5)(data)
    assert torch.allclose(noised.charges, data.charges)


def test_zero_sigma_returns_input_raises_error():
    data = make_data([-2.0, 1.0, 1.0])
    assert FormalChargeNoiseTransform(0.0)(data) is data

    try: 
        add_formal_charge_noise([data], 0.0)
        raise AssertionError("Expected ValueError for sigma=0.0")
    except ValueError as e:
        pass
        



def test_negative_sigma_is_rejected():
    with pytest.raises(ValueError):
        FormalChargeNoiseTransform(-0.1)


def test_noise_does_not_accumulate_across_epochs():
    original = make_data([-2.0, 1.0, 1.0])
    dataset = add_formal_charge_noise([original], 0.3)
    assert isinstance(dataset, TransformedDataset) and len(dataset) == 1

    first, second = dataset[0], dataset[0]
    assert torch.allclose(original.charges, torch.tensor([-2.0, 1.0, 1.0]))
    assert not torch.allclose(first.charges, second.charges)
    for noised in (first, second):
        assert (noised.charges - original.charges).abs().max().item() < 3.0


def noise_args(**overrides):
    args = argparse.Namespace(
        model="LocalSplitCharges",
        formal_charges_from_data=True,
        formal_charge_noise_sigma=0.1,
    )
    vars(args).update(overrides)
    return args


def test_check_args_accepts_supported_configuration():
    check_formal_charge_noise(noise_args())
    check_formal_charge_noise(noise_args(model="MACE", formal_charge_noise_sigma=None))


@pytest.mark.parametrize(
    "overrides",
    [
        {"formal_charge_noise_sigma": -0.1},
        {"formal_charge_noise_sigma": 0.0},
        {"model": "LocalCharges"},
        {"formal_charges_from_data": False},
    ],
)
def test_check_args_rejects_unusable_configurations(overrides):
    with pytest.raises(ValueError):
        check_formal_charge_noise(noise_args(**overrides))
