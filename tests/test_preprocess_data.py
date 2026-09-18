"""preprocess_data.py validates its input and writes a machine-readable statistics.json.

The pbc / low-density / dipole-weight checks used to run only in the .xyz training path.
Data that becomes .h5 shards never passes through there, so preprocess_data.py runs them
instead -- these tests pin that it actually does.
"""

import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import write

from tests.paths import REPO_ROOT, script_env

PREPROCESS_SCRIPT = REPO_ROOT / "scripts" / "preprocess_data.py"

E0S = "{1: -12.0}"
HEADS = json.dumps(
    {
        "Default": {
            "info_keys": {"energy": "REF_energy", "total_charge": "total_charge"},
            "arrays_keys": {"forces": "REF_forces"},
        }
    }
)


def write_configs(path, *, pbc, cell_size=5.0, num_images=4):
    images = []
    for index in range(num_images):
        atoms = Atoms(
            "H2",
            positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.8 + 0.01 * index]],
            cell=[cell_size] * 3,
            pbc=pbc,
        )
        atoms.info["REF_energy"] = -24.0 - 0.1 * index
        atoms.info["total_charge"] = 0.0
        atoms.arrays["REF_forces"] = np.zeros((2, 3))
        images.append(atoms)
    write(path, images, format="extxyz")
    return path


def run_preprocess(tmp_path, *, pbc=(True, True, True), cell_size=5.0, extra_args=()):
    train_file = write_configs(tmp_path / "train.xyz", pbc=pbc, cell_size=cell_size)
    valid_file = write_configs(tmp_path / "valid.xyz", pbc=pbc, cell_size=cell_size)
    h5_prefix = f"{tmp_path / 'h5'}/"
    return subprocess.run(
        [
            sys.executable,
            str(PREPROCESS_SCRIPT),
            f"--train_file={train_file}",
            f"--valid_file={valid_file}",
            f"--h5_prefix={h5_prefix}",
            "--r_max=4.0",
            f"--E0s={E0S}",
            f"--heads={HEADS}",
            "--num_process=1",
            "--shuffle=False",
            "--seed=1",
            *extra_args,
        ],
        cwd=REPO_ROOT,
        env=script_env(),
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    ), h5_prefix


@pytest.fixture(scope="module")
def default_run(tmp_path_factory):
    """One successful preprocess run, shared by the tests that only read its output."""
    process, h5_prefix = run_preprocess(tmp_path_factory.mktemp("default"))
    assert process.returncode == 0, process.stderr[-4000:]
    return h5_prefix


def test_statistics_json_is_machine_readable(default_run):
    """Every consumer of statistics.json parses it with ast.literal_eval, so a numpy
    repr ("np.int64(1)") in any field makes the file unreadable."""
    statistics = json.loads(Path(default_run + "statistics.json").read_text())

    atomic_numbers = ast.literal_eval(statistics["atomic_numbers"])
    assert atomic_numbers == [1]
    assert all(type(z) is int for z in atomic_numbers)

    atomic_energies = ast.literal_eval(statistics["atomic_energies"])
    assert sorted(atomic_energies) == atomic_numbers

    for key in ("avg_num_neighbors", "mean", "std", "r_max"):
        assert isinstance(statistics[key], float)


def test_shards_are_written(default_run):
    assert list(Path(default_run + "train").glob("*.h5"))
    assert list(Path(default_run + "val").glob("*.h5"))


def test_rejects_pbc_incompatible_with_electrostatic_method(tmp_path):
    process, _ = run_preprocess(
        tmp_path,
        pbc=(False, False, False),
        extra_args=["--electrostatic_pbc_method=pbc"],
    )

    assert process.returncode != 0
    assert "--electrostatic_pbc_method=pbc" in process.stderr


def test_override_pbc_checks_permits_incompatible_pbc(tmp_path):
    process, _ = run_preprocess(
        tmp_path,
        pbc=(False, False, False),
        extra_args=["--electrostatic_pbc_method=pbc", "--override_pbc_checks"],
    )

    assert process.returncode == 0, process.stderr[-4000:]


def test_rejects_low_density_periodic_config(tmp_path):
    process, _ = run_preprocess(
        tmp_path,
        cell_size=40.0,
        extra_args=["--low_density_pbc_max_volume_per_atom=100.0"],
    )

    assert process.returncode != 0
    assert "Suspicious low-density fully periodic config" in process.stderr
    assert "--allow_low_density_pbc" in process.stderr


def test_allow_low_density_pbc_permits_sparse_cell(tmp_path):
    process, _ = run_preprocess(
        tmp_path,
        cell_size=40.0,
        extra_args=[
            "--low_density_pbc_max_volume_per_atom=100.0",
            "--allow_low_density_pbc",
        ],
    )

    assert process.returncode == 0, process.stderr[-4000:]
