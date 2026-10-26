"""Energies, forces and dipoles must be continuous when a pair crosses r_max.

RealAgnosticResidualNonLinearInteractionBlock (and the Att block) feed element embeddings into the edge MLP, so their
edge weights are not zero at r_max unless the block applies the cutoff itself. The models therefore build their radial
embedding with apply_cutoff=False and pass the cutoff to the interactions; before, a trained LSC model jumped by meV
and 1e-2 polarization quanta whenever an atom pair crossed r_max.
"""

import numpy as np
import pytest
import torch
from ase import Atoms
from e3nn import o3

import mace.modules
from mace_scf.calculators.localsources import MACELocalCharges, MACELocalSplitCharges
from mace_scf.electrostatics import LocalCharges, LocalSplitCharges
from mace_scf.electrostatics.bonded_blocks import static_bond_transfer_blocks
from mace_scf.electrostatics.compiled_localsources import build_compiled_local_source_evaluator

R_MAX = 4.0
NONLINEAR = "RealAgnosticResidualNonLinearInteractionBlock"
EPS = 1e-9  # small enough that the smooth change across the crossing is ~1e-9


def _model(model_cls, interaction_name, source_block="OxidationDependentSymmetricPredictionSourceBlock"):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    interaction = mace.modules.interaction_classes[interaction_name]
    kwargs = dict(
        r_max=R_MAX,
        num_bessel=8,
        num_polynomial_cutoff=5,
        max_ell=2,
        interaction_cls=interaction,
        interaction_cls_first=interaction,
        num_interactions=2,
        num_elements=2,
        hidden_irreps=o3.Irreps("8x0e + 8x1o"),
        MLP_irreps=o3.Irreps("4x0e"),
        atomic_energies=np.zeros(2),
        avg_num_neighbors=8.0,
        atomic_numbers=[1, 8],
        correlation=2,
        gate=torch.nn.functional.silu,
        atomic_multipoles_max_l=1,
    )
    if model_cls is LocalSplitCharges:
        kwargs.update(
            formal_charges_from_data=True,
            static_bond_transfer_block=source_block,
            oxidation_state_mixer="SumOxidationStateMixer",
            oxidation_state_range=(-2.0, 2.0),
        )
    return model_cls(**kwargs)


def _atoms(distance):
    """O-H plus a second H at `distance` from the first H, out of range of the O."""
    atoms = Atoms(
        "OHH",
        positions=[[5.0, 5.0, 5.0], [6.0, 5.0, 5.0], [6.0 + distance, 5.0, 5.0]],
        cell=[20.0, 20.0, 20.0],
        pbc=True,
    )
    atoms.arrays["charges"] = np.zeros(3)
    atoms.info["total_charge"] = 0.0
    return atoms


def _evaluate(model_path, calc_cls, atoms, compiled):
    calc = calc_cls(model_path=str(model_path), device="cpu", pbc_handling="pbc")
    data = next(iter(calc._build_data_loader(atoms))).to(calc.device).to_dict()
    if compiled:
        evaluator = build_compiled_local_source_evaluator(calc.model, pbc_handling="pbc", enabled=False)
        out = evaluator.evaluate(data)
    else:
        out = calc.model(data, compute_force=True)
    return {k: out[k].detach().cpu().numpy() for k in ("energy", "forces", "dipole")}


def _across_r_max(tmp_path, model, calc_cls, compiled):
    path = tmp_path / "model.model"
    torch.save(model, path)
    inside = _evaluate(path, calc_cls, _atoms(R_MAX - EPS), compiled)
    outside = _evaluate(path, calc_cls, _atoms(R_MAX + EPS), compiled)
    return {k: np.abs(inside[k] - outside[k]).max() for k in inside}


MODELS = [(LocalSplitCharges, MACELocalSplitCharges), (LocalCharges, MACELocalCharges)]


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("interaction", list(mace.modules.interaction_classes))
@pytest.mark.parametrize("model_cls,calc_cls", MODELS)
def test_continuous_at_r_max(tmp_path, model_cls, calc_cls, interaction, compiled):
    jumps = _across_r_max(tmp_path, _model(model_cls, interaction), calc_cls, compiled)
    for key, jump in jumps.items():
        assert jump < 1e-7, (key, jump)


@pytest.mark.parametrize("source_block", list(static_bond_transfer_blocks))
def test_lsc_source_blocks_continuous_at_r_max(tmp_path, source_block):
    model = _model(LocalSplitCharges, NONLINEAR, source_block=source_block)
    jumps = _across_r_max(tmp_path, model, MACELocalSplitCharges, compiled=False)
    for key, jump in jumps.items():
        assert jump < 1e-7, (key, jump)


@pytest.mark.parametrize("model_cls,calc_cls", MODELS)
def test_models_saved_with_the_cutoff_folded_in_keep_it(tmp_path, model_cls, calc_cls):
    """Models saved before the fix carry apply_cutoff=True and must run as before, discontinuities included."""
    model = _model(model_cls, NONLINEAR)
    model.radial_embedding.apply_cutoff = True
    jumps = _across_r_max(tmp_path, model, calc_cls, compiled=False)
    assert jumps["energy"] > 1e-4, jumps
