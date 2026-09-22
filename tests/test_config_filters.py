"""Preprocess-time filtering of unusable configurations."""

from types import SimpleNamespace

import numpy as np

from mace_scf.utils.load_data import filter_xyz_collections


def _config(*, forces=((0.0, 0.0, 0.0),), multipoles=((0.0, 0.0, 0.0, 0.0),), omit=()):
    properties = {"forces": np.asarray(forces), "atomic_multipoles": np.asarray(multipoles)}
    for key in omit:
        properties.pop(key)
    return SimpleNamespace(properties=properties)


def _collections(train=(), valid=(), tests=()):
    return SimpleNamespace(train=list(train), valid=list(valid), tests=list(tests))


def _args(max_force=None, require_finite_multipoles=False):
    return SimpleNamespace(
        max_force=max_force, require_finite_multipoles=require_finite_multipoles
    )


def test_no_options_leaves_every_split_untouched():
    collections = _collections(train=[_config(forces=[[500.0, 0, 0]])], valid=[_config()])

    filter_xyz_collections(collections, _args())

    assert len(collections.train) == 1 and len(collections.valid) == 1


def test_max_force_is_inclusive_at_the_threshold():
    collections = _collections(
        train=[
            _config(forces=[[19.9, 0, 0]]),
            _config(forces=[[20.0, 0, 0]]),
            _config(forces=[[20.1, 0, 0]]),
            _config(forces=[[0, 0, -493.0]]),
        ]
    )

    filter_xyz_collections(collections, _args(max_force=20.0))

    kept = [float(np.abs(c.properties["forces"]).max()) for c in collections.train]
    assert kept == [19.9, 20.0]


def test_nan_multipoles_are_dropped_and_finite_ones_kept():
    collections = _collections(
        train=[
            _config(multipoles=[[0.1, 0.2, 0.3, 0.4]]),
            _config(multipoles=[[np.nan, np.nan, np.nan, np.nan]]),
            _config(multipoles=[[0.1, np.nan, 0.3, 0.4]]),
            _config(multipoles=[[np.inf, 0.0, 0.0, 0.0]]),
        ]
    )

    filter_xyz_collections(collections, _args(require_finite_multipoles=True))

    assert len(collections.train) == 1


def test_config_missing_the_judged_property_is_kept():
    collections = _collections(
        train=[_config(omit=("forces",)), _config(omit=("atomic_multipoles",))]
    )

    filter_xyz_collections(collections, _args(max_force=1.0, require_finite_multipoles=True))

    assert len(collections.train) == 2


def test_valid_is_filtered_but_tests_are_not():
    bad = lambda: _config(forces=[[99.0, 0, 0]])
    collections = _collections(
        train=[bad()], valid=[bad()], tests=[("holdout", [bad(), bad()])]
    )

    filter_xyz_collections(collections, _args(max_force=20.0))

    assert collections.train == [] and collections.valid == []
    assert len(collections.tests[0][1]) == 2


def test_both_filters_compose():
    collections = _collections(
        train=[
            _config(),
            _config(forces=[[99.0, 0, 0]]),
            _config(multipoles=[[np.nan, 0, 0, 0]]),
        ]
    )

    filter_xyz_collections(collections, _args(max_force=20.0, require_finite_multipoles=True))

    assert len(collections.train) == 1
