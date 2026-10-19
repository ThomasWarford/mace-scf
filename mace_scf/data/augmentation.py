import copy
import logging
from typing import Optional, Tuple

import torch

from mace.tools.scatter import scatter_sum


class FormalChargeNoiseTransform:
    """Add zero-sum Gaussian noise to the per-atom formal charges of a configuration."""

    def __init__(self, sigma: float, charge_range: Optional[Tuple[float, float]] = None):
        if sigma < 0.0:
            raise ValueError(f"formal charge noise sigma must be non-negative, got {sigma}")
        self.sigma = float(sigma)
        # if given, a configuration's noise is shrunk just enough to keep every noised charge in
        # [lo, hi] (the model's oxidation_state_range); shrinking the whole vector keeps it zero-sum
        self.charge_range = None if charge_range is None else (float(charge_range[0]), float(charge_range[1]))

    def __call__(self, data):
        if self.sigma == 0.0:
            return data
        charges = data["charges"]
        if charges is None:
            raise ValueError("FormalChargeNoiseTransform requires per-atom `charges`")
        data = copy.copy(data)  # dataset samples are reused every epoch; never noise in place
        noise = self._noise(charges, data["batch"])
        if self.charge_range is not None:
            noise = self._keep_in_range(charges, noise, data["batch"])
        data.charges = charges + noise
        return data

    def _keep_in_range(
        self, charges: torch.Tensor, noise: torch.Tensor, batch: Optional[torch.Tensor]
    ) -> torch.Tensor:
        lo, hi = self.charge_range
        if batch is None:
            batch = torch.zeros_like(charges, dtype=torch.long)
        # largest per-atom factor t <= 1 with lo <= q + t * eps <= hi (charges already outside
        # the range get t = 0, i.e. the configuration is left un-noised)
        room = torch.where(noise > 0, hi - charges, lo - charges)
        t = torch.where(noise != 0, room / noise, torch.ones_like(noise)).clamp(min=0.0, max=1.0)
        n_graphs = int(batch.max()) + 1 if batch.numel() else 0
        scale = torch.ones(n_graphs, dtype=noise.dtype, device=noise.device)
        scale = scale.scatter_reduce(0, batch, t, reduce="amin")
        return noise * scale[batch]

    def _noise(self, charges: torch.Tensor, batch: Optional[torch.Tensor]) -> torch.Tensor:
        z = torch.randn_like(charges)
        if batch is None:
            batch = torch.zeros_like(charges, dtype=torch.long)
        n_atoms = scatter_sum(torch.ones_like(z), batch, dim=0)
        mean = scatter_sum(z, batch, dim=0, dim_size=n_atoms.shape[0]) / n_atoms
        # sqrt(N / (N - 1)) makes Var(eps_i) exactly sigma^2; single-atom configs get no noise
        scale = torch.where(
            n_atoms > 1.0, (n_atoms / (n_atoms - 1.0).clamp(min=1.0)).sqrt(), torch.zeros_like(n_atoms)
        )
        return self.sigma * scale[batch] * (z - mean[batch])

    def __repr__(self):
        return f"{self.__class__.__name__}(sigma={self.sigma}, charge_range={self.charge_range})"


class TransformedDataset(torch.utils.data.Dataset):
    """Apply a transform to each sample of a dataset, preserving its length."""

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.transform(self.dataset[index])


def add_formal_charge_noise(dataset, sigma: float, charge_range: Optional[Tuple[float, float]] = None):
    """Wrap `dataset` in formal charge noise augmentation; sigma must be positive. With
    `charge_range`, noised charges are kept inside it (see FormalChargeNoiseTransform)."""
    if sigma <= 0.0:
        raise ValueError(f"formal charge noise sigma must be positive, got {sigma}")
    transform = FormalChargeNoiseTransform(sigma, charge_range)
    logging.info(f"Augmenting training formal charges with {transform}")
    return TransformedDataset(dataset, transform)
