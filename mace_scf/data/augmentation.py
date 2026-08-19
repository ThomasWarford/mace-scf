import copy
import logging
from typing import Optional

import torch

from mace.tools.scatter import scatter_sum


class FormalChargeNoiseTransform:
    """Add zero-sum Gaussian noise to the per-atom formal charges of a configuration."""

    def __init__(self, sigma: float):
        if sigma < 0.0:
            raise ValueError(f"formal charge noise sigma must be non-negative, got {sigma}")
        self.sigma = float(sigma)

    def __call__(self, data):
        if self.sigma == 0.0:
            return data
        charges = data["charges"]
        if charges is None:
            raise ValueError("FormalChargeNoiseTransform requires per-atom `charges`")
        data = copy.copy(data)  # dataset samples are reused every epoch; never noise in place
        data.charges = charges + self._noise(charges, data["batch"])
        return data

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
        return f"{self.__class__.__name__}(sigma={self.sigma})"


class TransformedDataset(torch.utils.data.Dataset):
    """Apply a transform to each sample of a dataset, preserving its length."""

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.transform(self.dataset[index])


def add_formal_charge_noise(dataset, sigma: float):
    """Wrap `dataset` in formal charge noise augmentation; sigma must be positive."""
    if sigma <= 0.0:
        raise ValueError(f"formal charge noise sigma must be positive, got {sigma}")
    transform = FormalChargeNoiseTransform(sigma)
    logging.info(f"Augmenting training formal charges with {transform}")
    return TransformedDataset(dataset, transform)
