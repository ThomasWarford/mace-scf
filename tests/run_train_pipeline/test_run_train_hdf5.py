"""Training from preprocessed .h5 shards matches training from the equivalent .xyz.

run_train.py used to accept .xyz only, so the shards written by scripts/preprocess_data.py
could not be trained on. These tests pin the equivalence of the two input paths: same
configurations, same split, same order, so every logged metric must agree exactly.

That equivalence is what catches the plumbing that differs between the two loaders --
HDF5Dataset supplies its own ``heads`` default, and per-atom properties (formal charges,
atomic multipoles) are re-read from the shards rather than parsed from the xyz.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.run_train_pipeline.harness import (
    DEFAULT_REFERENCE_DATA,
    REPO_ROOT,
    RunTrainCase,
    _clean_metrics,
    _parse_pretty_tables,
    make_run_train_config,
    model_config_overrides,
    run_train_case,
    script_env,
    summary_differences,
)

PREPROCESS_SCRIPT = REPO_ROOT / "scripts" / "preprocess_data.py"

NUM_TRAIN = 16
NUM_VALID = 4

# Taken from the harness config so the reference data's key schema is stated once.
# HDF5Dataset reads back under the "Default" head name.
_BASE_CONFIG = make_run_train_config()
HEADS = json.dumps({"Default": _BASE_CONFIG["heads"]["DFT"]})
E0S = _BASE_CONFIG["E0s"]
R_MAX = _BASE_CONFIG["r_max"]


@pytest.fixture(scope="module")
def preprocessed_dataset(tmp_path_factory):
    """A small xyz train/valid pair and the .h5 shards preprocessed from it.

    ``--num_process=1`` and ``--shuffle=False`` keep the shard contents in file order:
    dataset_from_sharded_hdf5 globs the directory without sorting, so more than one shard
    would make the concatenation order filesystem-dependent and the comparison meaningless.
    """
    import ase.io

    if not DEFAULT_REFERENCE_DATA.exists():
        pytest.skip(f"reference data not found: {DEFAULT_REFERENCE_DATA}")
    tmp_path = tmp_path_factory.mktemp("hdf5_equivalence")

    configs = ase.io.read(DEFAULT_REFERENCE_DATA, index=":")[: NUM_TRAIN + NUM_VALID]
    if len(configs) < NUM_TRAIN + NUM_VALID:
        pytest.skip("reference data has too few configurations")

    train_xyz = tmp_path / "train.xyz"
    valid_xyz = tmp_path / "valid.xyz"
    ase.io.write(train_xyz, configs[:NUM_TRAIN])
    ase.io.write(valid_xyz, configs[NUM_TRAIN:])

    h5_prefix = f"{tmp_path / 'h5'}/"
    process = subprocess.run(
        [
            sys.executable,
            str(PREPROCESS_SCRIPT),
            f"--train_file={train_xyz}",
            f"--valid_file={valid_xyz}",
            f"--h5_prefix={h5_prefix}",
            f"--r_max={R_MAX}",
            f"--E0s={E0S}",
            f"--heads={HEADS}",
            "--num_process=1",
            "--shuffle=False",
            "--seed=1",
        ],
        cwd=REPO_ROOT,
        env=script_env(),
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )
    assert process.returncode == 0, process.stderr[-4000:]

    statistics = Path(h5_prefix + "statistics.json")
    assert statistics.exists()
    return {
        "statistics": statistics,
        "xyz": {"train_file": str(train_xyz), "valid_file": str(valid_xyz)},
        "h5": {
            "train_file": h5_prefix + "train",
            "valid_file": h5_prefix + "val",
        },
    }


def _case(model, name, statistics):
    config = make_run_train_config(
        model=model,
        name=name,
        overrides={
            **model_config_overrides(model),
            # valid_fraction is dropped so both runs use the same explicit split, and the
            # statistics file pins avg_num_neighbors so neither run recomputes it.
            "valid_fraction": None,
            "statistics_file": str(statistics),
            # adam rather than the harness default: schedulefree is an optional
            # dependency and the optimizer is irrelevant to an input-format comparison.
            "optimizer": "adam",
        },
    )
    return RunTrainCase(name=name, config=config, timeout_s=900)


@pytest.mark.parametrize("model", ["MACE", "LocalSplitCharges"])
def test_hdf5_training_matches_xyz_training(model, preprocessed_dataset, tmp_path):
    data = preprocessed_dataset
    statistics = json.loads(data["statistics"].read_text())

    xyz_result = run_train_case(
        tmp_path / f"{model}_xyz",
        _case(model, f"{model.lower()}_xyz", data["statistics"]),
        config_overrides=data["xyz"],
        overwrite=True,
        check=True,
    )
    h5_result = run_train_case(
        tmp_path / f"{model}_h5",
        _case(model, f"{model.lower()}_h5", data["statistics"]),
        config_overrides=data["h5"],
        overwrite=True,
        check=True,
    )

    for result in (xyz_result, h5_result):
        log_text = result.log_text()
        assert (
            f"Total number of configurations: train={NUM_TRAIN}, valid={NUM_VALID}"
            in log_text
        )
        # --statistics_file used to be rejected outright; it now supplies the z-table,
        # E0s and the avg_num_neighbors that must not be recomputed per split.
        assert f"Using statistics file {data['statistics']}" in log_text
        assert (
            f"Average number of neighbors: {statistics['avg_num_neighbors']}" in log_text
        )
    assert "Loading preprocessed .h5 input" in h5_result.log_text()

    xyz_metrics = _clean_metrics(xyz_result.metrics())
    assert xyz_metrics, "no metrics were logged"
    assert not summary_differences(xyz_metrics, _clean_metrics(h5_result.metrics()))

    xyz_tables = _parse_pretty_tables(xyz_result.log_text())
    assert xyz_tables
    assert _parse_pretty_tables(h5_result.log_text()) == xyz_tables
