
from typing import Dict, Iterable, Tuple

import ase.io
from mace_scf.data.new_atomic_data import ExtAtomicData
import mace_scf
import mace
from mace.tools.scripts_utils import get_dataset_from_xyz, get_atomic_energies
import argparse
import ast
import logging
import os
import numpy as np
from mace import tools


def _config_type(config) -> str:
    properties = getattr(config, "properties", {}) or {}
    config_type = properties.get("config_type", None)
    if config_type is None:
        config_type = getattr(config, "config_type", None)
    return "unknown" if config_type is None else str(config_type)


def _is_fully_periodic(pbc) -> bool:
    if pbc is None:
        return False
    pbc_array = np.asarray(pbc, dtype=bool).reshape(-1)
    return pbc_array.shape == (3,) and bool(np.all(pbc_array))


def _volume_from_cell(cell) -> float:
    if cell is None:
        raise ValueError("fully periodic config has no cell")
    cell_array = np.asarray(cell, dtype=float)
    if cell_array.shape != (3, 3):
        raise ValueError(f"fully periodic config has invalid cell shape {cell_array.shape}")
    return float(abs(np.linalg.det(cell_array)))


def _named_config_splits(collections) -> Iterable[Tuple[str, Iterable]]:
    yield "train", collections.train
    yield "valid", collections.valid
    for test_name, test_configs in collections.tests:
        yield f"test:{test_name}", test_configs


def _as_path_list(path_or_paths):
    if path_or_paths is None:
        return []
    if isinstance(path_or_paths, (str, os.PathLike)):
        return [path_or_paths]
    return list(path_or_paths)


def _run_config_validators(args, validators):
    """Apply per-config validators in a single pass over each input file.

    Every validator sees the same configurations, so they share one ase.io.iread:
    a pass per check would parse a multi-GB training file once per check.
    """
    validators = [validate for validate in validators if validate is not None]
    if not validators:
        return
    for split_name in ("train", "valid", "test"):
        for path in _as_path_list(getattr(args, f"{split_name}_file", None)):
            for config_index, atoms in enumerate(ase.io.iread(path, index=":")):
                for validate in validators:
                    validate(atoms, config_index, path, split_name)


def _dipole_weight_validator(key_specification):
    dipole_key = key_specification.info_keys.get("dipole")
    if dipole_key is None or dipole_key == "none":
        return None

    def validate(atoms, config_index, file_path, split_name):
        if dipole_key not in atoms.info:
            return
        if "config_dipole_weight" not in atoms.info:
            raise ValueError(
                "Dipole data found without explicit config_dipole_weight. "
                f"split={split_name}, file={file_path}, index={config_index}, "
                f"dipole_key={dipole_key!r}. "
                "Set config_dipole_weight to a 3-vector such as [1, 1, 1], "
                "[0, 0, 1], or [0, 0, 0]."
            )

        dipole_weight = np.asarray(
            atoms.info["config_dipole_weight"], dtype=float
        ).reshape(-1)
        if dipole_weight.shape != (3,):
            raise ValueError(
                "config_dipole_weight must be a 3-vector whenever dipole data is "
                "present. "
                f"split={split_name}, file={file_path}, index={config_index}, "
                f"dipole_key={dipole_key!r}, shape={dipole_weight.shape}."
            )

    return validate


def check_explicit_dipole_component_weights(file_path, key_specification, split_name):
    validate = _dipole_weight_validator(key_specification)
    if validate is None:
        return
    for config_index, atoms in enumerate(ase.io.iread(file_path, index=":")):
        validate(atoms, config_index, file_path, split_name)


_ALLOWED_PBC_BY_METHOD = {
    "realspace": {(False, False, False)},
    "pbc": {(True, True, True)},
    "slab": {(True, True, False)},
    "molecule_in_box": {(False, False, False)},
    "mixed_periodic": {
        (True, True, True),
        (True, True, False),
        (False, False, False),
    },
}


def _pbc_tuple(atoms) -> Tuple[bool, bool, bool]:
    if not hasattr(atoms, "pbc"):
        raise ValueError("config has no pbc attribute")
    pbc = np.asarray(atoms.pbc, dtype=bool).reshape(-1)
    return tuple(bool(x) for x in pbc)


def _pbc_validator(electrostatic_pbc_method):
    allowed = _ALLOWED_PBC_BY_METHOD.get(electrostatic_pbc_method)
    if allowed is None:
        return None

    def validate(atoms, config_index, file_path, split_name):
        pbc = _pbc_tuple(atoms)
        if pbc not in allowed:
            allowed_str = ", ".join(
                "".join("T" if x else "F" for x in p) for p in sorted(allowed)
            )
            pbc_str = "".join("T" if x else "F" for x in pbc)
            raise ValueError(
                f"Found pbc={pbc_str} in {split_name} file {file_path}, "
                f"which is incompatible with "
                f"--electrostatic_pbc_method={electrostatic_pbc_method} "
                f"(allowed: {allowed_str})."
            )

    return validate


def check_pbc_consistent_with_electrostatic_method(
    file_path, electrostatic_pbc_method, split_name
):
    validate = _pbc_validator(electrostatic_pbc_method)
    if validate is None:
        return
    for config_index, atoms in enumerate(ase.io.iread(file_path, index=":")):
        validate(atoms, config_index, file_path, split_name)


def _enabled_pbc_validator(args):
    if getattr(args, "override_pbc_checks", False):
        return None
    return _pbc_validator(getattr(args, "electrostatic_pbc_method", None))


def check_pbc_consistent_with_electrostatic_method_for_paths(args):
    _run_config_validators(args, [_enabled_pbc_validator(args)])


def check_explicit_dipole_component_weights_for_paths(args):
    _run_config_validators(args, [_dipole_weight_validator(args.key_specification)])


def validate_xyz_paths(args):
    """Every check that needs the raw xyz, in one pass per file.

    This is the single definition of which path-level checks constitute validation;
    run_train (for .xyz input) and preprocess_data (for input that becomes .h5 shards)
    both call it rather than each listing the checks themselves.
    """
    _run_config_validators(
        args,
        [_dipole_weight_validator(args.key_specification), _enabled_pbc_validator(args)],
    )


def validate_xyz_collections(collections, args):
    """The checks that need parsed configurations rather than the raw file."""
    check_low_density_periodic_configs(
        collections,
        max_volume_per_atom=args.low_density_pbc_max_volume_per_atom,
        allow_low_density_pbc=args.allow_low_density_pbc,
    )


def check_low_density_periodic_configs(
    collections,
    *,
    max_volume_per_atom: float,
    allow_low_density_pbc: bool,
) -> None:
    if allow_low_density_pbc:
        logging.info("Skipping low-density pbc=TTT config check.")
        return
    if max_volume_per_atom <= 0.0:
        raise ValueError("low_density_pbc_max_volume_per_atom must be positive")

    for split_name, configs in _named_config_splits(collections):
        for config_index, config in enumerate(configs):
            if not _is_fully_periodic(getattr(config, "pbc", None)):
                continue
            volume = _volume_from_cell(getattr(config, "cell", None))
            num_atoms = len(config.atomic_numbers)
            volume_per_atom = volume / num_atoms
            if volume_per_atom <= max_volume_per_atom:
                continue
            raise ValueError(
                "Suspicious low-density fully periodic config detected. "
                f"split={split_name}, index={config_index}, "
                f"config_type={_config_type(config)}, pbc={getattr(config, 'pbc', None)}, "
                f"volume={volume:.6g}, num_atoms={num_atoms}, "
                f"volume_per_atom={volume_per_atom:.6g}, "
                f"max_volume_per_atom={max_volume_per_atom:.6g}. "
                "If this is intentional, rerun with --allow_low_density_pbc."
            )


def get_atomic_number_table_from_zs(zs) -> tools.AtomicNumberTable:
    """An AtomicNumberTable of plain ints.

    config.atomic_numbers is a numpy array, so feeding it to the upstream helper builds a
    table of np.int64, whose repr is "np.int64(1)" -- not int()-able and not
    literal_eval-able. That leaks into everything that stringifies the table: the log line
    parsed by the test harness, and statistics.json.
    """
    return tools.get_atomic_number_table_from_zs(int(z) for z in zs)


def log_dataset_summary(z_table, train_set, valid_set, tests=()) -> None:
    """The one definition of this line; the test harness parses it."""
    logging.info(z_table)
    test_summary = ", ".join(f"{name}: {len(configs)}" for name, configs in tests)
    logging.info(
        f"Total number of configurations: train={len(train_set)}, "
        f"valid={len(valid_set)}, tests=[{test_summary}]"
    )


def load_train_valid_sets_from_xyz(args: argparse.Namespace,  config_type_weights: Dict):
    # data
    validate_xyz_paths(args)
    collections, atomic_energies_dict = get_dataset_from_xyz(
        work_dir=args.work_dir,
        train_path=args.train_file,
        valid_path=args.valid_file,
        valid_fraction=args.valid_fraction,
        config_type_weights=config_type_weights,
        test_path=args.test_file,
        seed=args.valid_set_seed,
        key_specification=args.key_specification,
    )
    validate_xyz_collections(collections, args)

    # Atomic number table
    z_table = get_atomic_number_table_from_zs(
        z
        for configs in (collections.train, collections.valid)
        for config in configs
        for z in config.atomic_numbers
    )
    log_dataset_summary(
        z_table, collections.train, collections.valid, collections.tests
    )

    # energies
    if atomic_energies_dict is None or len(atomic_energies_dict) == 0:
        # if args.train_file.endswith(".xyz"):
        atomic_energies_dict = get_atomic_energies(
            args.E0s, collections.train, z_table
        )
    atomic_energies: np.ndarray = np.array(
        [atomic_energies_dict[z] for z in z_table.zs]
    )

    train_set = [
        mace_scf.data.ExtAtomicData.from_config(
            config, 
            z_table=z_table, 
            cutoff=args.r_max, 
            atomic_multipoles_max_l=args.atomic_multipoles_max_l
        )
        for config in collections.train
    ]
   
    valid_set = [
        mace_scf.data.ExtAtomicData.from_config(
            config, 
            z_table=z_table, 
            cutoff=args.r_max, 
            atomic_multipoles_max_l=args.atomic_multipoles_max_l
        )
        for config in collections.valid
    ]

    return train_set, valid_set, z_table, atomic_energies, collections.tests


def load_train_valid_sets_from_preprocessed(args: argparse.Namespace):
    """Load preprocessed HDF5: a directory of shards, or a single .h5 file.

    Returns the same 5-tuple as the .xyz loader. Shards carry no config_type grouping,
    so there are never test collections.
    """
    # The z-table cannot be derived from the shards, so it has to come from the command
    # line (or, via load_statistics_file, from the statistics.json written beside them).
    z_table = get_atomic_number_table_from_zs(ast.literal_eval(args.atomic_numbers))
    atomic_energies_dict = get_atomic_energies(args.E0s, None, z_table)

    def load(path):
        read = (
            mace.data.dataset_from_sharded_hdf5
            if os.path.isdir(path)
            else mace.data.HDF5Dataset
        )
        return read(
            path,
            r_max=args.r_max,
            z_table=z_table,
            atomic_dataclass=ExtAtomicData,
            atomic_multipoles_max_l=args.atomic_multipoles_max_l,
        )

    train_set = load(args.train_file)
    valid_set = load(args.valid_file)
    atomic_energies: np.ndarray = np.array(
        [atomic_energies_dict[z] for z in z_table.zs]
    )
    log_dataset_summary(z_table, train_set, valid_set)
    return train_set, valid_set, z_table, atomic_energies, []
