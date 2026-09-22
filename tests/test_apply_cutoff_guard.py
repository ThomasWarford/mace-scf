"""--apply_cutoff must be refused where it cannot take effect, not silently ignored.

apply_cutoff=False makes RadialEmbeddingBlock hand the cutoff back for the interaction
blocks to apply. Only the upstream MACE branches pass it on; every mace_scf forward
discards it, so honouring the flag there would train a model with no radial cutoff at all.
"""

import numpy as np
import pytest

import mace.tools
import mace_scf.utils
from mace_scf.utils.check_args import check_config_conflicts
from mace_scf.utils.run_train_utils import (
    APPLY_CUTOFF_SUPPORTED_MODELS,
    build_model,
    get_formal_charges,
)

HEADS = (
    '{"default": {"info_keys": {"energy": "REF_energy", "total_charge": "total_charge"},'
    ' "arrays_keys": {"forces": "REF_forces"}}}'
)
SCHEDULE = (
    '{0: {"name": "stage1", "start": 0, "end": 1,'
    ' "loss": {"energy_per_atom": 1.0, "forces": 10.0}, "lr": 0.01}}'
)


ELECTROSTATIC_ARGV = (
    "--electrostatic_pbc_method", "pbc",
    "--atomic_multipoles_max_l", "1",
    "--atomic_multipoles_smearing_width", "1.5",
    "--kspace_cutoff_factor", "0.75",
)
# H/O, so every charged model needs formal charges for both
FORMAL_CHARGES_ARGV = ("--atomic_formal_charges", "{1: 1.0, 8: -2.0}")

# The models that need formal charges declared up front.
EXTRA_ARGV = {
    "LocalSplitCharges": FORMAL_CHARGES_ARGV + ELECTROSTATIC_ARGV,
    "LocalCharges": ELECTROSTATIC_ARGV,
    "FixedChargeBaselinedMACE": FORMAL_CHARGES_ARGV + ELECTROSTATIC_ARGV,
}

MACE_SCF_MODELS = sorted(EXTRA_ARGV)


def _args(model: str, apply_cutoff: bool):
    argv = [
        "--name", "apply_cutoff_test",
        "--train_file", "unused.xyz",
        "--heads", HEADS,
        "--train_schedule", SCHEDULE,
        "--error_table", "PerAtomRMSE",
        "--model", model,
        "--hidden_irreps", "4x0e + 4x1o",
        "--MLP_irreps", "4x0e",
        "--r_max", "3.0",
        "--max_ell", "3",
        "--correlation", "3",
        "--num_interactions", "2",
        "--compute_avg_num_neighbors", "False",
        "--avg_num_neighbors", "10.0",
        "--compute_polarizability", "False",
        "--default_dtype", "float64",
        "--device", "cpu",
        "--apply_cutoff", str(apply_cutoff),
        *EXTRA_ARGV[model],
    ]
    args = mace_scf.utils.extended_arg_parser().parse_args(argv)
    check_config_conflicts(args)
    return args


def _build(args):
    z_table = mace.tools.get_atomic_number_table_from_zs([1, 8])
    charges = get_formal_charges(
        args.model,
        args.formal_charges_from_data,
        args.atomic_formal_charges,
        z_table,
    )
    return build_model(args, z_table, np.zeros(len(z_table)), charges, None)


@pytest.mark.parametrize("model", MACE_SCF_MODELS)
def test_apply_cutoff_false_is_refused_for_mace_scf_models(model):
    with pytest.raises(NotImplementedError, match="apply_cutoff"):
        _build(_args(model, apply_cutoff=False))


@pytest.mark.parametrize("model", MACE_SCF_MODELS)
def test_apply_cutoff_true_is_accepted(model):
    """The default must not be caught by the guard."""
    _build(_args(model, apply_cutoff=True))


def test_upstream_models_are_the_supported_set():
    assert APPLY_CUTOFF_SUPPORTED_MODELS == frozenset({"MACE", "ScaleShiftMACE"})
