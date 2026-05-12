"""Training and evaluation loop for RecVAEModel."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable

import torch

from .config import Config
from .model import RecVAEModel

logger = logging.getLogger(__name__)

EpochCallback = Callable[[int, dict], None]


def fit(
    model: RecVAEModel,
    train_loader: Iterable,
    h0: torch.Tensor,
    cfg: Config | None = None,
    *,
    epochs: int | None = None,
    lr: float | None = None,
    rho: float | None = None,
    opt_func=torch.optim.SGD,
    log_every: int = 1,
    callbacks: list[EpochCallback] | None = None,
) -> dict:
    """Alternating optimization loop.

    Each epoch:

    1. SGD step on encoder/decoder/inference parameters and ``z_vectors``.
       (``z_vectors`` is registered as a Parameter on the model, so a single
       optimizer over ``model.parameters()`` covers everything — the original
       notebook listed it in a second param group, which was equivalent.)
    2. Closed-form ridge update on ``F`` using the per-subject posterior
       trajectory accumulated during the SGD pass.

    Progress is emitted through ``logging`` rather than ``print``; each
    epoch logs one INFO-level line of the form
    ``[epoch k/N] mean_loss=... (recon=..., temporal=..., z_l1=...)`` where
    each value is the mean across batches in that epoch (not the noisy
    last-batch value). If no handler is configured for the ``recvae.train``
    logger (or any ancestor short of root), ``fit()`` attaches a stderr
    handler to the ``recvae.train`` logger only; never reconfigures root.
    This keeps importing applications in full control of their own logging.

    TODO(research): SGD@1e-6 over 500 epochs likely under-trains a model of
    this size. Consider AdamW with lr ~1e-4 and a cosine scheduler.

    Parameters
    ----------
    model : the RecVAE
    train_loader : iterable yielding ``(batch_volumes, batch_indices)``
    h0 : ``(1, latent_dim)`` initial latent state
    cfg : configuration; defaults to ``model.cfg``
    epochs, lr, rho : override config values
    opt_func : optimizer constructor
    log_every : log progress every N epochs
    callbacks : list of callables fired after each epoch as
                ``callback(epoch, metrics_dict)``. The metrics dict contains
                ``loss``, ``loss1``, ``loss2``, ``loss_z``, ``loss_F`` (means
                across batches in the epoch) plus ``epoch``.

    Returns
    -------
    dict with keys ``train_loss_history``, ``all_h_history``, ``last_index``.
    """
    cfg = cfg or model.cfg
    epochs = epochs if epochs is not None else cfg.epochs
    lr = lr if lr is not None else cfg.learning_rate
    rho = rho if rho is not None else cfg.rho
    callbacks = list(callbacks) if callbacks else []

    # Library-friendly logging: only attach a handler if the caller has
    # not configured one. This keeps tutorials / examples streaming
    # per-epoch lines without forcing a root-logger config on importers.
    if not logger.handlers and not logger.parent.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)

    optimizer = opt_func(model.parameters(), lr=lr)

    h_history_history = torch.zeros(
        model.train_size,
        cfg.tol_time,
        cfg.latent_dim,
        device=h0.device,
        dtype=h0.dtype,
    )

    train_loss_history = []
    last_index = None

    for epoch in range(epochs):
        model.train()
        # Accumulate per-batch metrics so we can report an epoch mean rather
        # than a noisy last-batch value.
        epoch_totals = {"loss": 0.0, "loss1": 0.0, "loss2": 0.0, "loss_z": 0.0, "loss_F": 0.0}
        n_batches = 0

        for batch, batch_index in train_loader:
            which_ones = batch_index.long()
            h_batch = h0.expand(batch.size(0), -1)

            loss, loss_dic, h_history = model.training_step(batch, h_batch, which_ones)

            # Store posterior trajectory for the F update (detached — no grad).
            # training_step also exposes h_history as a list; either form
            # produces the same (B, T, D) buffer slot.
            h_stacked = torch.stack(h_history, dim=1)  # (B, T, D)
            h_history_history[which_ones] = h_stacked.detach()
            last_index = which_ones

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_totals["loss"] += loss.detach().item()
            epoch_totals["loss1"] += loss_dic["loss1"].item()
            epoch_totals["loss2"] += loss_dic["loss2"].item()
            epoch_totals["loss_z"] += loss_dic["loss_z"].item()
            epoch_totals["loss_F"] += loss_dic["loss_F"].item()
            n_batches += 1

        # Closed-form F update (no gradient).
        model.updating_F(h_history_history, h0, rho)

        if n_batches == 0:
            # Empty loader — skip logging and callbacks for this epoch.
            continue

        metrics = {k: v / n_batches for k, v in epoch_totals.items()}
        metrics["epoch"] = epoch

        if epoch % log_every == 0:
            train_loss_history.append(metrics["loss"])
            logger.info(
                "[epoch %d/%d] mean_loss=%.4f (recon=%.4f, temporal=%.4f, z_l1=%.4f)",
                epoch + 1,
                epochs,
                metrics["loss"],
                metrics["loss1"],
                metrics["loss2"],
                metrics["loss_z"],
            )

        for cb in callbacks:
            cb(epoch, metrics)

    return {
        "train_loss_history": train_loss_history,
        "all_h_history": h_history_history,
        "last_index": last_index,
    }


@torch.no_grad()
def evaluate(
    model: RecVAEModel,
    x: torch.Tensor,
    h0: torch.Tensor,
    which_ones: torch.Tensor,
):
    """Run the model in eval mode (BatchNorm uses running stats)."""
    model.eval()
    return model(x, h0, which_ones)
