"""Training and evaluation loop for RecVAEModel."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch

from .config import Config
from .model import RecVAEModel


def fit(
    model: RecVAEModel,
    train_loader: Iterable,
    h0: torch.Tensor,
    cfg: Optional[Config] = None,
    *,
    epochs: Optional[int] = None,
    lr: Optional[float] = None,
    rho: Optional[float] = None,
    opt_func=torch.optim.SGD,
    log_every: int = 1,
) -> Dict:
    """Alternating optimization loop.

    Each epoch:

    1. SGD step on encoder/decoder/inference parameters and ``z_vectors``.
       (``z_vectors`` is registered as a Parameter on the model, so a single
       optimizer over ``model.parameters()`` covers everything — the original
       notebook listed it in a second param group, which was equivalent.)
    2. Closed-form ridge update on ``F`` using the per-subject posterior
       trajectory accumulated during the SGD pass.

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
    log_every : print progress every N epochs

    Returns
    -------
    dict with keys ``train_loss_history``, ``all_h_history``, ``last_index``.
    """
    cfg = cfg or model.cfg
    epochs = epochs if epochs is not None else cfg.epochs
    lr = lr if lr is not None else cfg.learning_rate
    rho = rho if rho is not None else cfg.rho

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
    last_loss = None
    last_loss_dic = None

    for epoch in range(epochs):
        model.train()
        for batch, batch_index in train_loader:
            which_ones = batch_index.long()
            h_batch = h0.expand(batch.size(0), -1)

            loss, loss_dic, h_history = model.training_step(batch, h_batch, which_ones)

            # Store posterior trajectory for the F update (detached — no grad).
            h_stacked = torch.stack(h_history).transpose(0, 1)  # (B, T, D)
            h_history_history[which_ones] = h_stacked.detach()
            last_index = which_ones
            last_loss = loss
            last_loss_dic = loss_dic

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % log_every == 0 and last_loss is not None:
            train_loss_history.append(last_loss.detach().item())
            print(
                f"Epoch [{epoch}]: train loss: {last_loss.item():.4f} "
                f"(loss1: {last_loss_dic['loss1'].item():.4f}, "
                f"loss2: {last_loss_dic['loss2'].item():.4f}, "
                f"loss_z: {last_loss_dic['loss_z'].item():.4f}, "
                f"loss_F: {last_loss_dic['loss_F'].item():.4f})",
            )

        # Closed-form F update (no gradient).
        model.updating_F(h_history_history, h0, rho)

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
