"""fastai-style learning-rate range test.

Ramps the lr geometrically over a fixed number of optimizer steps on the real model, data
and loss, records an EMA-smoothed loss, and stops once it diverges. Each step goes through
take_step, so the finder sees the same gradients as training (fixed-point implicit
gradients, clipping, per-group lrs). The model and optimizer are left in the state reached
at the end of the ramp: callers that train afterwards must restore them.
"""

import csv
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .train import take_step


def lr_find(
    model: torch.nn.Module,
    model_wrapper,
    loss_fn: torch.nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    ref_lr: float,
    lr_min: float = 1e-6,
    lr_max: float = 1e-1,
    num_steps: int = 500,
    beta: float = 0.98,
    diverge_factor: float = 4.0,
    min_steps: int = 10,
    max_grad_norm: Optional[float] = None,
    group: Optional[str] = None,
    train_sampler=None,
    sampler_epoch: int = 0,
) -> List[Dict[str, float]]:
    """Run the range test and return one record per step.

    With group=None the global lr sweeps lr_min -> lr_max and every param group keeps its
    ratio to ref_lr (the stage's global lr), so per-group <name>_lr overrides are honoured.
    With a group name, only the groups of that name sweep lr_min -> lr_max; the others
    stay at their current lr. Each record's "lr" is the swept lr.

    Under torch.distributed the recorded loss is the mean over ranks, so every rank sees the
    same curve and stops at the same step. A DistributedSampler is set to sampler_epoch,
    then advanced by one each time the loader is exhausted.
    """
    if not 0 < lr_min < lr_max:
        raise ValueError(f"need 0 < lr_min < lr_max, got {lr_min}, {lr_max}")
    if num_steps < 2:
        raise ValueError(f"num_steps must be at least 2, got {num_steps}")
    if group is not None:
        names = [g.get("name") for g in optimizer.param_groups]
        if group not in names:
            raise ValueError(f"no param group named {group!r}; groups are {names}")

    base_lrs = [g["lr"] for g in optimizer.param_groups]

    def set_lr(lr: float) -> None:
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if group is None:
                param_group["lr"] = lr * base_lr / ref_lr
            elif param_group.get("name") == group:
                param_group["lr"] = lr

    if hasattr(optimizer, "train"):  # schedule-free
        optimizer.train()

    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    if train_sampler is not None:
        train_sampler.set_epoch(sampler_epoch)

    records = []
    avg_loss, best_smoothed = 0.0, math.inf
    batches = iter(train_loader)
    for step in range(num_steps):
        lr = lr_min * (lr_max / lr_min) ** (step / (num_steps - 1))
        set_lr(lr)
        try:
            batch = next(batches)
        except StopIteration:
            sampler_epoch += 1
            if train_sampler is not None:
                train_sampler.set_epoch(sampler_epoch)
            batches = iter(train_loader)
            batch = next(batches)

        _, metrics = take_step(
            model=model,
            model_wrapper=model_wrapper,
            loss_fn=loss_fn,
            batch=batch,
            optimizer=optimizer,
            ema=None,
            max_grad_norm=max_grad_norm,
            device=device,
        )
        loss = float(metrics["loss"])
        if distributed:
            loss_t = torch.tensor(loss, dtype=torch.float64, device=device)
            torch.distributed.all_reduce(loss_t)
            loss = loss_t.item() / torch.distributed.get_world_size()
        if not math.isfinite(loss):
            logging.info(f"lr_find: non-finite loss at step {step}, lr={lr:.3g}; stopping")
            break
        avg_loss = beta * avg_loss + (1 - beta) * loss
        smoothed = avg_loss / (1 - beta ** (step + 1))
        records.append(
            {
                "step": step,
                "lr": lr,
                "loss": loss,
                "smoothed_loss": smoothed,
                "grad_norm_before_clip": metrics.get("grad_norm_before_clip", float("nan")),
                "grad_clip_applied": metrics.get("grad_clip_applied", False),
            }
        )
        best_smoothed = min(best_smoothed, smoothed)
        if step >= min_steps and smoothed > diverge_factor * best_smoothed:
            logging.info(f"lr_find: loss diverged at step {step}, lr={lr:.3g}; stopping")
            break
    return records


def suggest_lr(
    records: List[Dict[str, float]], skip_start: int = 10, skip_end: int = 5
) -> Dict[str, float]:
    """Suggested lrs from the smoothed curve: fastai's valley and steepest rules, and the
    lr at the minimum / 10. Values are nan if the trimmed curve is too short."""
    nan = float("nan")
    end = len(records) - skip_end if skip_end > 0 else len(records)
    kept = records[skip_start:end]
    if len(kept) < 3:
        return {"valley": nan, "steepest": nan, "min_div10": nan}
    lrs = np.array([r["lr"] for r in kept])
    losses = np.array([r["smoothed_loss"] for r in kept])

    # valley: the longest decreasing subsequence of the loss, then a point 2/3 of the way
    # down it (fastai's `valley`)
    n = len(losses)
    lds = [1] * n
    max_start, max_end = 0, 0
    for i in range(1, n):
        for j in range(i):
            if losses[i] < losses[j] and lds[i] < lds[j] + 1:
                lds[i] = lds[j] + 1
            if lds[max_end] < lds[i]:
                max_end = i
                max_start = max_end - lds[max_end]
    sections = (max_end - max_start) / 3
    valley_idx = max_start + int(sections) + int(sections / 2)

    steepest_idx = int(np.argmin(np.gradient(losses, np.log10(lrs))))
    return {
        "valley": float(lrs[valley_idx]),
        "steepest": float(lrs[steepest_idx]),
        "min_div10": float(lrs[int(np.argmin(losses))] / 10),
    }


def write_lr_find_outputs(
    records: List[Dict[str, float]],
    suggestions: Dict[str, float],
    out_prefix: str,
    max_grad_norm: Optional[float] = None,
) -> None:
    """Write <out_prefix>.csv (records), .json (suggestions) and .png (plot)."""
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fields = ["step", "lr", "loss", "smoothed_loss", "grad_norm_before_clip", "grad_clip_applied"]
    with open(f"{out_prefix}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    with open(f"{out_prefix}.json", "w") as f:
        json.dump(suggestions, f, indent=2)
    plot_lr_find(records, suggestions, f"{out_prefix}.png", max_grad_norm)


def read_lr_find_records(csv_path: str) -> List[Dict[str, float]]:
    """Records from a csv written by write_lr_find_outputs."""
    with open(csv_path) as f:
        return [
            {
                "step": int(row["step"]),
                "lr": float(row["lr"]),
                "loss": float(row["loss"]),
                "smoothed_loss": float(row["smoothed_loss"]),
                "grad_norm_before_clip": float(row["grad_norm_before_clip"]),
                "grad_clip_applied": row["grad_clip_applied"] == "True",
            }
            for row in csv.DictReader(f)
        ]


def plot_lr_find(
    records: List[Dict[str, float]],
    suggestions: Dict[str, float],
    png_path: str,
    max_grad_norm: Optional[float] = None,
) -> None:
    """Loss (raw and smoothed) and grad norm against lr, with the suggestions marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lrs = [r["lr"] for r in records]
    fig, (ax_loss, ax_grad) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    ax_loss.plot(lrs, [r["loss"] for r in records], color="0.75", lw=0.8, label="loss")
    ax_loss.plot(lrs, [r["smoothed_loss"] for r in records], color="C0", label="smoothed")
    for (name, lr), color in zip(suggestions.items(), ("C1", "C2", "C3")):
        if math.isfinite(lr):
            ax_loss.axvline(lr, color=color, ls="--", label=f"{name} = {lr:.2g}")
    if records:
        # fix both limits to the smoothed curve up to its minimum: autoscaling to the raw
        # loss lets the divergence spike (and its margin) flatten the curve
        smoothed = np.array([r["smoothed_loss"] for r in records])
        best = int(np.argmin(smoothed))
        ax_loss.set_yscale("log")
        ax_loss.set_ylim(0.9 * smoothed[best], 1.1 * smoothed[: best + 1].max())
    ax_loss.set_ylabel("loss")
    ax_loss.legend(fontsize="small")

    ax_grad.plot(lrs, [r["grad_norm_before_clip"] for r in records], color="C0")
    if max_grad_norm is not None:
        ax_grad.axhline(max_grad_norm, color="C3", ls="--", label=f"clip = {max_grad_norm:g}")
        ax_grad.legend(fontsize="small")
    ax_grad.set_yscale("log")
    ax_grad.set_ylabel("grad norm before clip")
    ax_grad.set_xscale("log")
    ax_grad.set_xlabel("learning rate")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
