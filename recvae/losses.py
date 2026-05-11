"""Composable loss callables for RecVAE training.

The default :class:`RecVAELoss` reproduces the historical scaling used by
``RecVAEModel.training_step``: a per-volume reconstruction MSE, a temporal
proxy MSE between ``h_t`` and ``g(h_{t-1})``, plus an L1 penalty on the
subject-specific noise. ``loss_F`` is reported but not added to the gradient
loss — F is updated in closed form by :meth:`RecVAEModel.updating_F`.

:class:`KLRecVAELoss` swaps the temporal MSE for the closed-form KL between
two diagonal Gaussians, which is the proper variational lower bound term.
Both losses share the same call signature, so they are drop-in replacements:

    loss_fn = RecVAELoss(sig_x=1.0, sig_h=1.0, lambda_z=10.0, rho=0.1)
    loss, loss_dic = loss_fn(model, batch, h_history, gh_history, mu_history, x_list)

Importantly, callers pass in the per-timestep trajectories rather than the
loss object running the rollout itself. This keeps the loss free of any
opinion about how the rollout is sliced (the model handles that) and makes
unit testing trivial.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RecVAELoss:
    """Default composite loss matching the canonical training_step.

    Parameters
    ----------
    sig_x : observation noise std for the reconstruction term.
    sig_h : noise std for the temporal-prior proxy term.
    lambda_z : weight on the L1 sparsity penalty over ``z_vectors``.
    rho : ridge penalty coefficient. Reported only; F is updated separately
          via :meth:`RecVAEModel.updating_F`.

    The composite training loss is

        loss = loss1 + loss2 + loss_z

    where
        loss1 = (1 / (sig_x^2 * 2 B T)) * sum_t sum_b ||x_t - mu_t||^2
        loss2 = (1 / (sig_h^2 * 2 B T)) * sum_t sum_b ||h_t - g(h_{t-1})||^2
        loss_z = lambda_z * ||z||_1

    ``loss_F = rho * ||F||_F^2`` is computed for logging only.
    """

    sig_x: float = 1.0
    sig_h: float = 1.0
    lambda_z: float = 10.0
    rho: float = 0.1

    def __call__(
        self,
        model,
        batch: torch.Tensor,
        h_history: list[torch.Tensor],
        gh_history: list[torch.Tensor],
        mu_history: list[torch.Tensor],
        x_list: list[torch.Tensor],
    ) -> tuple[torch.Tensor, dict]:
        T = len(h_history)
        denom = 2 * batch.size(0) * T

        loss1 = sum((x - mu).pow(2).sum() for x, mu in zip(x_list, mu_history))
        loss1 = loss1 / (self.sig_x**2) / denom

        loss2 = sum((h - gh).pow(2).sum() for h, gh in zip(h_history, gh_history))
        loss2 = loss2 / (self.sig_h**2) / denom

        loss_F = self.rho * (model.F_mat**2).sum()
        loss_z = self.lambda_z * model.z_vectors.abs().sum()

        loss = loss1 + loss2 + loss_z

        loss_dic = {
            "loss1": loss1.detach(),
            "loss2": loss2.detach(),
            "loss_F": loss_F.detach(),
            "loss_z": loss_z.detach(),
        }
        return loss, loss_dic


@dataclass
class KLRecVAELoss:
    """Variational loss using the closed-form KL for the temporal term.

    Replaces ``loss2`` of :class:`RecVAELoss` with

        KL(q(h_t | x_t, h_{t-1}) || N(g(h_{t-1}), sig_h^2 I))

    summed over latent dimensions and averaged over batch+time. The KL term
    requires the per-step posterior mean and log-variance, so the model
    forward must additionally yield ``mu_h_history`` and ``log_var_h_history``;
    pass them in via the ``mu_h_history``/``log_var_h_history`` keyword
    arguments. If those are not supplied, the call falls back to the MSE
    proxy so that this class remains call-compatible with the rest of the
    package.

    This is provided as an opt-in alternative; the default training loop
    still uses :class:`RecVAELoss`.
    """

    sig_x: float = 1.0
    sig_h: float = 1.0
    lambda_z: float = 10.0
    rho: float = 0.1

    def __call__(
        self,
        model,
        batch: torch.Tensor,
        h_history: list[torch.Tensor],
        gh_history: list[torch.Tensor],
        mu_history: list[torch.Tensor],
        x_list: list[torch.Tensor],
        *,
        mu_h_history: list[torch.Tensor] | None = None,
        log_var_h_history: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict]:
        T = len(h_history)
        denom = 2 * batch.size(0) * T

        loss1 = sum((x - mu).pow(2).sum() for x, mu in zip(x_list, mu_history))
        loss1 = loss1 / (self.sig_x**2) / denom

        if mu_h_history is None or log_var_h_history is None:
            # Fall back to the MSE proxy when posterior moments aren't passed.
            loss2 = sum((h - gh).pow(2).sum() for h, gh in zip(h_history, gh_history))
            loss2 = loss2 / (self.sig_h**2) / denom
        else:
            sig_h = self.sig_h
            kl_sum = batch.new_zeros(())
            for mu_h, log_var_h, gh in zip(mu_h_history, log_var_h_history, gh_history):
                var_q = torch.exp(log_var_h)
                log_sig_q = log_var_h / 2
                per_dim = (
                    torch.log(torch.tensor(sig_h, device=mu_h.device, dtype=mu_h.dtype))
                    - log_sig_q
                    + (var_q + (mu_h - gh) ** 2) / (2 * sig_h**2)
                    - 0.5
                )
                kl_sum = kl_sum + per_dim.sum(dim=-1).mean()
            loss2 = kl_sum / T

        loss_F = self.rho * (model.F_mat**2).sum()
        loss_z = self.lambda_z * model.z_vectors.abs().sum()

        loss = loss1 + loss2 + loss_z

        loss_dic = {
            "loss1": loss1.detach(),
            "loss2": loss2.detach(),
            "loss_F": loss_F.detach(),
            "loss_z": loss_z.detach(),
        }
        return loss, loss_dic
