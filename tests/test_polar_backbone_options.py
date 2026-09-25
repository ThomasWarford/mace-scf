"""--edge_irreps and --use_agnostic_product must reach the model, or be refused.

Both flags parse upstream, but build_model used to drop them, so a run asking for the
MACE-Polar backbone silently trained the default one. They are wired into the upstream
MACE / ScaleShiftMACE branches and the models built on _LocalSourceModelBase
(LocalSplitCharges, LocalCharges, FixedChargeBaselinedMACE). FixedPoint and MACEQEq
re-implement the backbone without either, so there they must raise rather than be ignored.
"""

import numpy as np
import pytest
import torch
from e3nn import o3

import mace.tools
import mace_scf.utils
from mace.modules.blocks import RealAgnosticResidualNonLinearInteractionBlock
from mace_scf.utils.check_args import check_config_conflicts
from mace_scf.utils.run_train_utils import (
    BACKBONE_OPTION_MODELS,
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
NONLINEAR = "RealAgnosticResidualNonLinearInteractionBlock"

ELECTROSTATIC_ARGV = (
    "--electrostatic_pbc_method", "pbc",
    "--atomic_multipoles_max_l", "1",
    "--atomic_multipoles_smearing_width", "1.5",
    "--kspace_cutoff_factor", "0.75",
)
FORMAL_CHARGES_ARGV = ("--atomic_formal_charges", "{1: 1.0, 8: -2.0}")
EXTRA_ARGV = {
    "MACE": (),
    "ScaleShiftMACE": ("--mean", "0.0", "--std", "1.0"),
    "LocalSplitCharges": FORMAL_CHARGES_ARGV + ELECTROSTATIC_ARGV,
    "LocalCharges": ELECTROSTATIC_ARGV,
    "FixedChargeBaselinedMACE": FORMAL_CHARGES_ARGV + ELECTROSTATIC_ARGV,
    "FixedPoint": ELECTROSTATIC_ARGV,
    "MACEQEq": ELECTROSTATIC_ARGV,
}
REFUSING_MODELS = sorted(set(EXTRA_ARGV) - BACKBONE_OPTION_MODELS)


def _args(model: str, *extra, check: bool = True):
    argv = [
        "--name", "backbone_options_test",
        "--train_file", "unused.xyz",
        "--heads", HEADS,
        "--train_schedule", SCHEDULE,
        "--error_table", "PerAtomRMSE",
        "--model", model,
        "--interaction_first", NONLINEAR,
        "--interaction", NONLINEAR,
        "--hidden_irreps", "8x0e + 8x1o",
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
        *EXTRA_ARGV[model],
        *extra,
    ]
    args = mace_scf.utils.extended_arg_parser().parse_args(argv)
    if check:
        check_config_conflicts(args)
    return args


def _build(args):
    torch.set_default_dtype(torch.float64)
    z_table = mace.tools.get_atomic_number_table_from_zs([1, 8])
    charges = get_formal_charges(
        args.model, args.formal_charges_from_data, args.atomic_formal_charges, z_table
    )
    return build_model(args, z_table, np.zeros(len(z_table)), charges, train_loader=None)


def _n_params(model):
    return sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize("model", sorted(BACKBONE_OPTION_MODELS))
def test_edge_irreps_reach_the_interaction_blocks(model):
    default = _build(_args(model))
    narrow = _build(_args(model, "--edge_irreps", "4x0e + 4x1o"))

    # upstream MACE hands edge_irreps to layers 2+ only; layer 1 falls back to its input
    assert narrow.interactions[1].edge_irreps == o3.Irreps("4x0e + 4x1o")
    assert default.interactions[1].edge_irreps == o3.Irreps("8x0e + 8x1o")
    assert narrow.interactions[1].linear_up.irreps_out == o3.Irreps("4x0e + 4x1o")
    assert _n_params(narrow) < _n_params(default)


@pytest.mark.parametrize("model", sorted(BACKBONE_OPTION_MODELS))
def test_agnostic_product_reaches_the_product_blocks(model):
    default = _build(_args(model))
    agnostic = _build(_args(model, "--use_agnostic_product", "True"))

    assert all(p.use_agnostic_product for p in agnostic.products)
    assert not any(p.use_agnostic_product for p in default.products)
    # one set of contraction weights instead of one per element (H and O here)
    assert _n_params(agnostic) < _n_params(default)


def test_refusing_models_are_the_ones_without_the_shared_backbone():
    assert REFUSING_MODELS == ["FixedPoint", "MACEQEq"]


@pytest.mark.parametrize("model", REFUSING_MODELS)
@pytest.mark.parametrize(
    "flag", [("--edge_irreps", "4x0e + 4x1o"), ("--use_agnostic_product", "True")]
)
def test_other_models_refuse_the_flags(model, flag):
    # unchecked: FixedPoint's config check wants training options that build_model never
    # reaches, since the refusal comes before any model is constructed
    with pytest.raises(NotImplementedError, match="not wired up"):
        _build(_args(model, *flag, check=False))


def test_nonlinear_transposes_are_kept_without_cueq():
    """The mul_ir fix must only fire under cueq; the e3nn model keeps upstream's
    (identity) transposes untouched."""
    model = _build(_args("MACE"))
    blocks = [
        m for m in model.modules()
        if isinstance(m, RealAgnosticResidualNonLinearInteractionBlock)
    ]
    assert len(blocks) == 2
    # without cueq upstream's wrapper already returns None, and build_model changes nothing
    assert all(b.transpose_mul_ir is None and b.transpose_ir_mul is None for b in blocks)
