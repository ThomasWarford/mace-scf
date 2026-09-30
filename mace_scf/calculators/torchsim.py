"""TorchSim interface for the local-source models (LocalSplitCharges, LocalCharges,
FixedChargeBaselinedMACE).

MACE's own torch-sim wrapper (mace.calculators.mace_torchsim) builds the extra inputs
only for PolarMACE, so it cannot run these models: they also read per-atom formal
charges (data["charges"]) and a per-system external field (data["external_field"]).
Both are read from SimState extras here:

    state = ts.io.atoms_to_state(atoms_list, device=device, dtype=torch.float64)
    state.atom_extras["formal_charges"] = charges  # [n_atoms]; needed if the model
                                                   # was fitted with formal charges from data
    state.system_extras["external_field"] = field  # [n_systems, 3]; optional, default 0

Extras must go in these dicts: an attribute set after construction
(state.formal_charges = ...) is not registered as an extra.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from mace_scf.electrostatics.bonded_blocks import PerAtomFormalChargesBlock

try:
    import torch_sim as ts
    from torch_sim.models.interface import ModelInterface
    from torch_sim.neighbors import torchsim_nl

    _TORCHSIM_IMPORT_ERROR: Optional[ImportError] = None
except ImportError as exc:
    ts = None
    torchsim_nl = None
    _TORCHSIM_IMPORT_ERROR = exc

    class ModelInterface(torch.nn.Module):  # type: ignore[no-redef]
        """Fallback base class when torch-sim is not installed."""


class MACELocalSourceTorchSimModel(ModelInterface):
    """Batched energies, forces and stresses of a local-source model for torch-sim.

    Besides energy/forces/stress, the output holds "partial_charges" [n_atoms],
    "density_coefficients" [n_atoms, (l_max+1)^2] and "dipole" [n_systems, 3].
    Plain MACE models (e.g. ScaleShiftMACE) also work; they give energy/forces/stress only.

    Positions are wrapped into the cell before evaluation, so for periodic systems the
    dipole is that of the wrapped configuration; it can differ from the ASE
    calculator's for unwrapped atoms by a polarization quantum. Energies, forces and
    stresses do not depend on the wrapping.
    """

    def __init__(
        self,
        model: Union[str, Path, torch.nn.Module],
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float64,
        compute_forces: bool = True,
        compute_stress: bool = True,
        pbc_handling: Optional[str] = None,
        formal_charges_key: str = "formal_charges",
        external_field_key: str = "external_field",
    ) -> None:
        if _TORCHSIM_IMPORT_ERROR is not None:
            raise ImportError(
                "MACELocalSourceTorchSimModel requires torch-sim "
                "(pip install torch-sim-atomistic)."
            ) from _TORCHSIM_IMPORT_ERROR
        super().__init__()
        self._device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._dtype = dtype
        self._compute_forces = compute_forces
        self._compute_stress = compute_stress
        self._memory_scales_with = "n_atoms_x_density"
        self.formal_charges_key = formal_charges_key
        self.external_field_key = external_field_key

        # The models allocate internal buffers with torch.get_default_dtype(), so it
        # must match the model dtype, as the ASE calculators also ensure.
        torch.set_default_dtype(dtype)

        if isinstance(model, (str, Path)):
            model = torch.load(str(model), map_location=self._device, weights_only=False)
        self.model = model.to(device=self._device, dtype=dtype).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        if pbc_handling is not None:
            self.model.coulomb_energy.set_pbc_handling(pbc_handling)

        self.r_max = float(self.model.r_max)
        self.requires_formal_charges = isinstance(
            getattr(self.model, "formal_charges", None), PerAtomFormalChargesBlock
        )
        atomic_numbers = [int(z) for z in self.model.atomic_numbers]
        # Z -> one-hot index lookup on device; -1 marks elements the model lacks.
        z_to_index = torch.full(
            (max(atomic_numbers) + 1,), -1, dtype=torch.long, device=self._device
        )
        z_to_index[atomic_numbers] = torch.arange(len(atomic_numbers), device=self._device)
        self.register_buffer("z_to_index", z_to_index, persistent=False)
        self.num_elements = len(atomic_numbers)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def compute_forces(self) -> bool:
        return self._compute_forces

    @property
    def compute_stress(self) -> bool:
        return self._compute_stress

    def _node_attrs(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        if int(atomic_numbers.max()) >= self.z_to_index.numel():
            raise ValueError("state contains elements the model was not trained on")
        indices = self.z_to_index[atomic_numbers]
        if bool((indices < 0).any()):
            raise ValueError("state contains elements the model was not trained on")
        return torch.nn.functional.one_hot(indices, self.num_elements).to(self._dtype)

    def _formal_charges(self, state: Any) -> torch.Tensor:
        if state.has_extras(self.formal_charges_key):
            return getattr(state, self.formal_charges_key).to(self._dtype)
        if self.requires_formal_charges:
            raise ValueError(
                "model uses per-atom formal charges; set "
                f"state.atom_extras[{self.formal_charges_key!r}]"
            )
        return torch.zeros(state.n_atoms, device=self._device, dtype=self._dtype)

    def _external_field(self, state: Any) -> torch.Tensor:
        if state.has_extras(self.external_field_key):
            field = getattr(state, self.external_field_key).to(self._dtype)
            return field.view(state.n_systems, 3)
        return torch.zeros(state.n_systems, 3, device=self._device, dtype=self._dtype)

    def forward(self, state: Any, **_kwargs: Any) -> Dict[str, torch.Tensor]:
        if not isinstance(state, ts.SimState):
            state = ts.SimState(**dict(state))
        state = state.to(self._device, self._dtype)
        n_systems = state.n_systems

        positions = (
            ts.transforms.pbc_wrap_batched(
                state.positions, state.cell, state.system_idx, state.pbc
            )
            if state.pbc.any()
            else state.positions
        )
        edge_index, mapping_system, unit_shifts = torchsim_nl(
            positions,
            state.row_vector_cell,
            state.pbc,
            torch.tensor(self.r_max, device=self._device, dtype=self._dtype),
            state.system_idx,
        )
        shifts = ts.transforms.compute_cell_shifts(
            state.row_vector_cell, unit_shifts, mapping_system
        )
        ptr = torch.zeros(n_systems + 1, dtype=torch.long, device=self._device)
        ptr[1:] = torch.cumsum(torch.bincount(state.system_idx, minlength=n_systems), 0)

        data = {
            "positions": positions.requires_grad_(True),
            "node_attrs": self._node_attrs(state.atomic_numbers),
            "batch": state.system_idx,
            "ptr": ptr,
            "head": torch.zeros(n_systems, dtype=torch.long, device=self._device),
            "pbc": state.pbc,
            "cell": state.row_vector_cell,
            "edge_index": edge_index,
            "shifts": shifts,
            "unit_shifts": unit_shifts,
            "charges": self._formal_charges(state),
            "external_field": self._external_field(state),
        }
        out = self.model(
            data,
            compute_force=self._compute_forces,
            compute_stress=self._compute_stress,
        )

        results = {"energy": out["energy"].detach()}
        if self._compute_forces:
            results["forces"] = out["forces"].detach()
        if self._compute_stress:
            results["stress"] = out["stress"].detach()
        # Charge outputs exist only for the local-source models; a plain
        # (ScaleShift)MACE returns energy/forces/stress alone.
        if out.get("density_coefficients") is not None:
            density_coefficients = out["density_coefficients"].detach()
            results["density_coefficients"] = density_coefficients
            results["partial_charges"] = density_coefficients[:, 0]
        if out.get("dipole") is not None:
            results["dipole"] = out["dipole"].detach()
        return results
