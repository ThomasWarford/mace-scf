"""The torch-sim wrapper must reproduce the ASE calculator on a batch of systems.

A random-weight LocalSplitCharges model is enough: the test checks that the batched
input dict matches the one the ASE calculator builds, not the physics.

torch-sim wraps positions into the cell and the ASE calculator does not. Energy, forces
and stress do not depend on the periodic image of each atom, but the dipole of a
periodic system does (it is defined up to the polarization quantum), so the dipole is
compared with the ASE result for the wrapped configuration.
"""

import numpy as np
import pytest
import torch

ts = pytest.importorskip("torch_sim")

from mace_scf.calculators.localsources import MACELocalSplitCharges
from mace_scf.calculators.torchsim import MACELocalSourceTorchSimModel
from tests.test_lsc_stress_finite_difference import (
    CUBIC_CELL,
    FORMAL_CHARGES_KEY,
    PERIODIC_IMAGE_SHIFTS,
    TRICLINIC_CELL,
    build_random_local_split_charges_model,
    periodic_water_dimer,
)

ENERGY_TOLERANCE = 1e-10  # eV
FORCE_TOLERANCE = 1e-10  # eV/A
STRESS_TOLERANCE = 1e-12  # eV/A^3
DEVICE = "cpu"


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    model = build_random_local_split_charges_model(seed=42, atomic_numbers=[1, 8])
    path = tmp_path_factory.mktemp("models") / "random_local_split_charges.model"
    torch.save(model, path)
    return str(path)


def batch_of_systems():
    shifted = periodic_water_dimer(TRICLINIC_CELL)
    shifted.positions = shifted.positions + PERIODIC_IMAGE_SHIFTS @ TRICLINIC_CELL
    return [
        periodic_water_dimer(CUBIC_CELL),
        periodic_water_dimer(TRICLINIC_CELL),
        shifted,
    ]


def state_with_formal_charges(atoms_list):
    state = ts.io.atoms_to_state(atoms_list, device=torch.device(DEVICE), dtype=torch.float64)
    state.atom_extras["formal_charges"] = torch.tensor(
        np.concatenate([atoms.arrays[FORMAL_CHARGES_KEY] for atoms in atoms_list]),
        dtype=torch.float64,
    )
    return state


def test_batched_outputs_match_ase_calculator(model_path):
    calc = MACELocalSplitCharges(
        model_path=model_path,
        device=DEVICE,
        formal_charges_key=FORMAL_CHARGES_KEY,
        pbc_handling="pbc",
    )
    atoms_list = batch_of_systems()
    reference = []
    for atoms in atoms_list:
        atoms.calc = calc
        reference.append(
            (
                atoms.get_potential_energy(),
                atoms.get_forces(),
                atoms.get_stress(voigt=False),
                calc.results["partial_charges"],
            )
        )
        atoms.calc = None
    reference_dipoles = []
    for atoms in atoms_list:
        wrapped = atoms.copy()
        wrapped.wrap()
        wrapped.calc = calc
        reference_dipoles.append(wrapped.get_dipole_moment())

    model = MACELocalSourceTorchSimModel(model_path, device=DEVICE, pbc_handling="pbc")
    state = state_with_formal_charges(atoms_list)
    out = model(state)

    for i, ((energy, forces, stress, charges), dipole) in enumerate(
        zip(reference, reference_dipoles)
    ):
        mask = (state.system_idx == i).numpy()
        np.testing.assert_allclose(out["energy"][i].item(), energy, rtol=0, atol=ENERGY_TOLERANCE)
        np.testing.assert_allclose(out["forces"].numpy()[mask], forces, rtol=0, atol=FORCE_TOLERANCE)
        np.testing.assert_allclose(out["stress"][i].numpy(), stress, rtol=0, atol=STRESS_TOLERANCE)
        np.testing.assert_allclose(out["partial_charges"].numpy()[mask], charges, rtol=0, atol=1e-12)
        np.testing.assert_allclose(out["dipole"][i].numpy(), dipole, rtol=0, atol=1e-12)


def test_missing_formal_charges_raise(model_path):
    model = MACELocalSourceTorchSimModel(model_path, device=DEVICE, pbc_handling="pbc")
    state = ts.io.atoms_to_state(batch_of_systems(), device=torch.device(DEVICE), dtype=torch.float64)
    with pytest.raises(ValueError, match="formal charges"):
        model(state)


def test_formal_charges_survive_integrator(model_path):
    model = MACELocalSourceTorchSimModel(model_path, device=DEVICE, pbc_handling="pbc")
    state = state_with_formal_charges(batch_of_systems())
    kT = torch.tensor(300 * ts.units.MetalUnits.temperature, dtype=torch.float64)
    dt = torch.tensor(0.0005 * ts.units.MetalUnits.time, dtype=torch.float64)  # 0.5 fs
    md_state = ts.nvt_langevin_init(state, model, kT=kT)
    for _ in range(3):
        md_state = ts.nvt_langevin_step(md_state, model, dt=dt, kT=kT)
    torch.testing.assert_close(md_state.formal_charges, state.formal_charges)
    assert torch.isfinite(md_state.energy).all()


def test_plain_scale_shift_mace_matches_mace_calculator(tmp_path):
    """Plain MACE models need no formal charges and return energy/forces/stress only."""
    from e3nn import o3

    import mace.modules
    from mace.calculators import MACECalculator
    from mace.tools import torch_tools
    from tests.utils import disable_e3nn_codegen, seed_torch

    torch_tools.set_default_dtype("float64")
    seed_torch(7)
    interaction_cls = mace.modules.interaction_classes["RealAgnosticResidualInteractionBlock"]
    with disable_e3nn_codegen():
        model = mace.modules.ScaleShiftMACE(
            r_max=3.0,
            num_bessel=8,
            num_polynomial_cutoff=6,
            max_ell=2,
            interaction_cls=interaction_cls,
            interaction_cls_first=interaction_cls,
            num_interactions=2,
            num_elements=2,
            hidden_irreps=o3.Irreps("8x0e+8x1o"),
            MLP_irreps=o3.Irreps("16x0e"),
            atomic_energies=np.array([1.0, 2.0]),
            avg_num_neighbors=10.0,
            atomic_numbers=[1, 8],
            correlation=2,
            gate=mace.modules.gate_dict["silu"],
            atomic_inter_scale=1.5,
            atomic_inter_shift=0.1,
        )
    model_path = tmp_path / "random_scale_shift_mace.model"
    torch.save(model, model_path)

    calc = MACECalculator(model_paths=str(model_path), device=DEVICE, default_dtype="float64")
    atoms_list = batch_of_systems()
    reference = []
    for atoms in atoms_list:
        atoms.calc = calc
        reference.append((atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress(voigt=False)))
        atoms.calc = None

    ts_model = MACELocalSourceTorchSimModel(str(model_path), device=DEVICE)
    assert not ts_model.requires_formal_charges
    state = ts.io.atoms_to_state(atoms_list, device=torch.device(DEVICE), dtype=torch.float64)
    out = ts_model(state)
    assert set(out) == {"energy", "forces", "stress"}
    for i, (energy, forces, stress) in enumerate(reference):
        mask = (state.system_idx == i).numpy()
        np.testing.assert_allclose(out["energy"][i].item(), energy, rtol=0, atol=ENERGY_TOLERANCE)
        np.testing.assert_allclose(out["forces"].numpy()[mask], forces, rtol=0, atol=FORCE_TOLERANCE)
        np.testing.assert_allclose(out["stress"][i].numpy(), stress, rtol=0, atol=STRESS_TOLERANCE)
