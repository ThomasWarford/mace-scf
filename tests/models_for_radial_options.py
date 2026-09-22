"""In-process model builders for the distance-transform and pair-repulsion tests.

The compiled and calculator suites drive on-disk reference checkpoints, which will never
carry a ZBL term or a distance transform. These builders make small random-weight models
instead, so the tests need no new fixture files.
"""

import numpy as np
from ase.atoms import Atoms
from e3nn import o3

import mace.modules
import mace.tools
from mace.tools import torch_tools

import mace_scf.electrostatics as electrostatics
from tests.utils import disable_e3nn_codegen, seed_torch

FORMAL_CHARGES_KEY = "formal_oxidation_states"

# Two water molecules far enough apart to keep the cell sparse, close enough within each
# molecule that the ZBL envelope (r_cov(Z_u) + r_cov(Z_v)) is open on the O-H edges.
ATOMIC_NUMBERS = [1, 8]


def water_pair_atoms() -> Atoms:
    atoms = Atoms(
        "OH2OH2",
        positions=[
            [1.0, 1.0, 1.0],
            [1.9, 1.0, 1.0],
            [0.7, 1.9, 1.0],
            [4.0, 4.0, 4.0],
            [4.9, 4.0, 4.0],
            [3.7, 4.9, 4.0],
        ],
        cell=np.diag([7.0, 7.0, 7.0]),
        pbc=True,
    )
    atoms.arrays[FORMAL_CHARGES_KEY] = np.array([-2.0, 1.0, 1.0, -2.0, 1.0, 1.0])
    return atoms


def _shared_kwargs(z_table):
    interaction_cls = mace.modules.interaction_classes[
        "RealAgnosticResidualInteractionBlock"
    ]
    return dict(
        r_max=3.0,
        num_bessel=8,
        num_polynomial_cutoff=6,
        max_ell=2,
        interaction_cls=interaction_cls,
        interaction_cls_first=interaction_cls,
        num_interactions=2,
        num_elements=len(z_table),
        hidden_irreps=o3.Irreps("8x0e+8x1o"),
        MLP_irreps=o3.Irreps("16x0e"),
        atomic_energies=np.array([1.0] * len(z_table)),
        avg_num_neighbors=10.0,
        atomic_numbers=z_table.zs,
        correlation=2,
        gate=mace.modules.gate_dict["silu"],
    )


# Per-model kwargs beyond the shared MACE backbone ones.
_MODEL_SPECS = {
    "LocalSplitCharges": (
        lambda: electrostatics.LocalSplitCharges,
        dict(
            formal_charges_from_data=True,
            atomic_multipoles_max_l=1,
            atomic_multipoles_smearing_width=1.5,
            include_electrostatic_self_interaction=True,
            pbc_handling="pbc",
        ),
    ),
    "LocalCharges": (
        lambda: electrostatics.LocalCharges,
        dict(
            atomic_multipoles_max_l=1,
            atomic_multipoles_smearing_width=1.5,
            include_electrostatic_self_interaction=True,
            pbc_handling="pbc",
        ),
    ),
    "FixedChargeBaselinedMACE": (
        lambda: electrostatics.FixedChargeBaselinedMACE,
        dict(
            formal_charges_from_data=True,
            atomic_multipoles_smearing_width=1.5,
            include_electrostatic_self_interaction=True,
            pbc_handling="pbc",
        ),
    ),
}

LOCAL_SOURCE_MODELS = tuple(_MODEL_SPECS)

# Models whose node_energy output carries the per-node ZBL term. FixedPointCore and
# MACEQEq accumulate graph-level energies only.
MODELS_WITH_NODE_ENERGY = LOCAL_SOURCE_MODELS

# FixedPointCore and MACEQEq need bespoke construction, and their own suites all skip on
# missing fixture checkpoints, so without these builders the two forwards would have no
# coverage of the radial options at all.
GRAPH_LEVEL_MODELS = ("FixedPointCore", "MACEQEq")

ALL_MODELS = LOCAL_SOURCE_MODELS + GRAPH_LEVEL_MODELS


def build_model(model_name: str, seed: int = 11, **overrides):
    """Build a random-weight model. `overrides` carries distance_transform/pair_repulsion."""
    torch_tools.set_default_dtype("float64")
    seed_torch(seed)
    z_table = mace.tools.get_atomic_number_table_from_zs(ATOMIC_NUMBERS)
    cls_fn, extra = _MODEL_SPECS[model_name]
    with disable_e3nn_codegen():
        return cls_fn()(**_shared_kwargs(z_table), **extra, **overrides)


def build_batch(atoms=None, n_graphs: int = 2, cutoff: float = 3.0):
    """A batched data dict for the models above, with `formal_oxidation_states` present."""
    import mace.tools.torch_geometric
    from tests.utils import dataset_from_atoms

    # The dataset tensors take the current default dtype, so pin it here as well as in
    # build_model: otherwise whichever of the two a test calls first decides the dtype,
    # and calling build_batch first yields float32 data against a float64 model.
    torch_tools.set_default_dtype("float64")

    if atoms is None:
        atoms = water_pair_atoms()
    dataset = dataset_from_atoms(
        [atoms] * n_graphs,
        cutoff=cutoff,
        charges_key=FORMAL_CHARGES_KEY,
        atomic_multipoles_max_l=1,
    )
    loader = mace.tools.torch_geometric.dataloader.DataLoader(
        dataset=dataset, batch_size=n_graphs, shuffle=False
    )
    return next(iter(loader))


def build_fixed_point_core(seed: int = 4, num_polynomial_cutoff: int = 5, **overrides):
    """A small FixedPointCore. Mirrors tests/distributed/fixtures.build_model."""
    from mace_scf.electrostatics import field_blocks

    torch_tools.set_default_dtype("float64")
    seed_torch(seed)
    z_table = mace.tools.get_atomic_number_table_from_zs(ATOMIC_NUMBERS)
    interaction_cls = mace.modules.interaction_classes[
        "RealAgnosticResidualInteractionBlock"
    ]
    with disable_e3nn_codegen():
        return electrostatics.FixedPointCore(
            r_max=3.0,
            num_bessel=8,
            num_polynomial_cutoff=num_polynomial_cutoff,
            max_ell=3,
            interaction_cls=interaction_cls,
            interaction_cls_first=interaction_cls,
            num_interactions=1,
            num_elements=len(z_table),
            hidden_irreps=o3.Irreps("4x0e+4x1o"),
            atomic_energies=np.array([-12.674624, -2041.039790]),
            avg_num_neighbors=10.0,
            atomic_numbers=z_table.zs,
            correlation=3,
            gate=mace.modules.gate_dict["silu"],
            MLP_irreps=o3.Irreps("16x0e"),
            radial_MLP=[64, 64, 64],
            radial_type="bessel",
            atom_density_scaling=np.ones(len(z_table)),
            kspace_cutoff_factor=1.0,
            atomic_multipoles_max_l=1,
            atomic_multipoles_smearing_width=1.5,
            field_feature_max_l=1,
            field_feature_widths=[1.5],
            include_electrostatic_self_interaction=True,
            add_local_electron_energy=True,
            fixedpoint_update_config={
                "type": field_blocks.OneBodyVariableUpdate,
                "potential_embedding_cls": field_blocks.BiasedLinearPotentialEmbedding,
                "nonlinearity_cls": field_blocks.NoNonLinearity,
            },
            field_readout_config={
                "type": field_blocks.StrictQuadraticFieldEnergyReadout
            },
            pbc_handling="pbc",
            **overrides,
        )


def build_qeq(seed: int = 4, num_polynomial_cutoff: int = 6, **overrides):
    """A small MACEQEq."""
    torch_tools.set_default_dtype("float64")
    seed_torch(seed)
    z_table = mace.tools.get_atomic_number_table_from_zs(ATOMIC_NUMBERS)
    interaction_cls = mace.modules.interaction_classes[
        "RealAgnosticResidualInteractionBlock"
    ]
    with disable_e3nn_codegen():
        return electrostatics.MACEQEq(
            r_max=3.0,
            num_bessel=8,
            num_polynomial_cutoff=num_polynomial_cutoff,
            max_ell=3,
            interaction_cls=interaction_cls,
            interaction_cls_first=interaction_cls,
            num_interactions=2,
            num_elements=len(z_table),
            hidden_irreps=o3.Irreps("4x0e+4x1o"),
            MLP_irreps=o3.Irreps("4x0e"),
            atomic_energies=np.array([1.0] * len(z_table)),
            avg_num_neighbors=10.0,
            atomic_numbers=z_table.zs,
            correlation=3,
            gate=mace.modules.gate_dict["silu"],
            atomic_multipoles_max_l=1,
            atomic_multipoles_smearing_width=1.5,
            kspace_cutoff_factor=0.75,
            include_electrostatic_self_interaction=True,
            pbc_handling="pbc",
            **overrides,
        )


# num_polynomial_cutoff differs per builder, and ZBLBasis takes p=num_polynomial_cutoff,
# so the parity tests need it alongside the model.
_GRAPH_LEVEL_BUILDERS = {
    "FixedPointCore": (build_fixed_point_core, 5),
    "MACEQEq": (build_qeq, 6),
}

LOCAL_SOURCE_NUM_POLYNOMIAL_CUTOFF = 6


def build_any(model_name: str, **overrides):
    """Build any model by name, and report its num_polynomial_cutoff."""
    if model_name in _GRAPH_LEVEL_BUILDERS:
        builder, p = _GRAPH_LEVEL_BUILDERS[model_name]
        return builder(**overrides), p
    return build_model(model_name, **overrides), LOCAL_SOURCE_NUM_POLYNOMIAL_CUTOFF


def graph_energy(model, model_name, data):
    """Total graph-level energy. FixedPointCore exposes it via local_part, which is the
    SCF-independent pass the ZBL term belongs to."""
    if model_name == "FixedPointCore":
        return model.local_part(data).energies.sum(dim=-1)
    return model(data, compute_force=False, compute_stress=False)["energy"]
