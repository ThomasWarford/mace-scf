"""lr_find under DDP: every rank must record the same (rank-averaged) curve and stop at the
same step, or the ranks that keep stepping hang in DDP's gradient all-reduce. Exhausting the
loader advances the DistributedSampler's epoch, as training does each epoch."""

import pytest
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from mace_scf.utils.lr_find import lr_find

from .gloo import run_gloo


class _Batch:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def to(self, device):
        return self

    def to_dict(self):
        return {"x": self.x}


class _Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch_dict, training):
        return self.model(batch_dict["x"])


class _Sampler:
    def __init__(self):
        self.epochs = []

    def set_epoch(self, epoch):
        self.epochs.append(epoch)


def _mse(pred, ref):
    return ((pred - ref.y) ** 2).mean()


def _run_rank(rank, world_size, nan_on_rank1_at):
    # each worker otherwise starts a full set of intra-op threads; on a shared login node
    # (or a small allocation) the oversubscription stalls the workers past run_gloo's timeout
    torch.set_num_threads(1)
    torch.manual_seed(0)  # same init on every rank
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
    model = model.double()
    torch.manual_seed(1 + rank)  # different data per rank
    x = torch.randn(32, 4, dtype=torch.float64)
    y = x.sum(dim=1, keepdim=True) * (1 + rank)
    loader = [_Batch(x[i : i + 8], y[i : i + 8]) for i in range(0, 32, 8)]
    if rank == 1 and nan_on_rank1_at is not None:
        loader[nan_on_rank1_at] = _Batch(x[:8], torch.full_like(y[:8], float("nan")))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    sampler = _Sampler()
    records = lr_find(
        model, DDP(_Wrapper(model)), _mse, loader, optimizer, torch.device("cpu"),
        ref_lr=1e-3, lr_min=1e-5, lr_max=1e-2, num_steps=10,
        train_sampler=sampler, sampler_epoch=1000,
    )
    return {"records": records, "epochs": sampler.epochs}


@pytest.mark.timeout(240)
def test_ranks_record_the_same_averaged_curve():
    results = run_gloo(_run_rank, world_size=2, args=(None,), timeout_s=180.0)
    assert len(results[0]["records"]) == 10
    for key in ("lr", "loss", "smoothed_loss"):
        assert [r[key] for r in results[0]["records"]] == [r[key] for r in results[1]["records"]]
    # 10 steps over 4 batches: two restarts of the loader
    assert results[0]["epochs"] == [1000, 1001, 1002]


@pytest.mark.timeout(240)
def test_nan_on_one_rank_stops_every_rank():
    results = run_gloo(_run_rank, world_size=2, args=(2,), timeout_s=180.0)
    assert len(results[0]["records"]) == len(results[1]["records"]) == 2
