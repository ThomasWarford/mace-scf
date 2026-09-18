"""The Fermi level offset must never cost a training-set scan for a model that ignores it."""

import argparse
import json

import numpy as np
import pytest

from mace_scf.utils.check_args import load_statistics_file
from mace_scf.utils.load_data import fermi_level_offset_from_configurations
from mace_scf.utils.run_train_utils import get_fermi_level_offset


class _ExplodingDataset:
    """Stands in for the training set; iterating it at all is the bug under test."""

    def __iter__(self):
        raise AssertionError("the training set was scanned for the Fermi level offset")


class _Loader:
    dataset = _ExplodingDataset()


@pytest.mark.parametrize(
    "model", ["LocalSplitCharges", "MACE", "ScaleShiftMACE", "LocalCharges", "MACEQEq"]
)
def test_scan_is_skipped_for_models_that_ignore_the_offset(model):
    args = argparse.Namespace(fermi_level_offset=None, model=model)
    assert get_fermi_level_offset(_Loader(), args, device="cpu") == 0.0


def test_explicit_value_wins_for_every_model():
    for model in ("LocalSplitCharges", "FixedPoint"):
        args = argparse.Namespace(fermi_level_offset=1.25, model=model)
        assert get_fermi_level_offset(_Loader(), args, device="cpu") == 1.25


def test_fixed_point_still_scans():
    """The guard must not silence the model that genuinely needs the measurement."""
    args = argparse.Namespace(fermi_level_offset=None, model="FixedPoint")
    with pytest.raises(AssertionError, match="was scanned"):
        get_fermi_level_offset(_Loader(), args, device="cpu")


class _Config:
    def __init__(self, fermi_level, weight):
        self.properties = {} if fermi_level is None else {"fermi_level": fermi_level}
        self.property_weights = {} if weight is None else {"fermi_level": weight}


def test_offset_from_configurations_averages_weighted_entries():
    configs = [_Config(1.0, 1.0), _Config(3.0, 1.0), _Config(99.0, 0.0), _Config(None, None)]
    assert fermi_level_offset_from_configurations(configs) == pytest.approx(2.0)


def test_offset_from_configurations_is_none_without_data():
    assert fermi_level_offset_from_configurations([_Config(None, None)]) is None
    assert fermi_level_offset_from_configurations([]) is None


def test_offset_from_configurations_matches_numpy_scalars():
    configs = [_Config(np.float64(2.0), np.float64(1.0))]
    assert fermi_level_offset_from_configurations(configs) == pytest.approx(2.0)


def _statistics(tmp_path, **extra):
    stats = {
        "atomic_energies": "{1: -1.0}",
        "avg_num_neighbors": 10.0,
        "mean": 0.0,
        "std": 1.0,
        "atomic_numbers": "[1]",
        "r_max": 6.0,
        **extra,
    }
    path = tmp_path / "statistics.json"
    path.write_text(json.dumps(stats))
    return str(path)


def _args(statistics_file, **extra):
    return argparse.Namespace(
        statistics_file=statistics_file, atomic_numbers=None, E0s=None, r_max=6.0,
        avg_num_neighbors=1.0, compute_avg_num_neighbors=True, mean=None, std=None,
        fermi_level_offset=None, **extra,
    )


def test_statistics_file_supplies_the_offset(tmp_path):
    args = _args(_statistics(tmp_path, fermi_level_offset=4.5))
    load_statistics_file(args)
    assert args.fermi_level_offset == 4.5


def test_command_line_offset_beats_the_statistics_file(tmp_path):
    args = _args(_statistics(tmp_path, fermi_level_offset=4.5))
    args.fermi_level_offset = 0.0
    load_statistics_file(args)
    assert args.fermi_level_offset == 0.0


def test_older_statistics_files_without_the_key_still_load(tmp_path):
    """The key postdates the shards in matpes_fit/, which must keep working untouched."""
    args = _args(_statistics(tmp_path))
    load_statistics_file(args)
    assert args.fermi_level_offset is None
    assert args.avg_num_neighbors == 10.0


def test_null_offset_in_statistics_is_treated_as_absent(tmp_path):
    """preprocess_data writes null when the data carries no Fermi level."""
    args = _args(_statistics(tmp_path, fermi_level_offset=None))
    load_statistics_file(args)
    assert args.fermi_level_offset is None
