"""Recurrent 3D-convolutional VAE for temporal fMRI volumes.

Tensor shape conventions
------------------------
- batch x:            ``(B, 1, X=91, Y=109, Z=91, T=120)``
- single timestep:    ``(B, 1, 91, 109, 91)``
- encoded feature:    ``(B, enc_out_dim=100)``
- latent state h:     ``(B, latent_dim=10)``
- F matrix:           ``(latent_dim, latent_dim)``
- z_vectors:          ``(N_train, z_dim)``
- decoded mu:         ``(B, 1, 91, 109, 91)``

Encoder/decoder spatial dimension chain (verified against PyTorch 2.x):

    91 -> 45 -> 22 -> 11 -> 5      (Conv3d k=4, s=2, p=1)
     5 -> 11 -> 22 -> 45 -> 91     (ConvTranspose3d with non-trivial output_padding)

The non-trivial ``output_padding`` values are required because the spatial
extents (91, 109, 91) are odd; choosing wrong here will silently produce a
mismatched reconstruction shape.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import nn

from .config import Config


class RecVAEModel(nn.Module):
    """Recurrent VAE with linear temporal prior ``g(h) = h @ F^T``.

    Parameters
    ----------
    train_size : number of subjects in the training set (sets ``z_vectors`` size)
    cfg : configuration; falls back to ``Config()`` defaults
    """

    def __init__(self, train_size: int, cfg: Optional[Config] = None):
        super().__init__()
        cfg = cfg or Config()
        self.cfg = cfg
        self.latent_dim = cfg.latent_dim
        self.tol_time = cfg.tol_time
        self.train_size = train_size

        # ---- Encoder: (1, 91, 109, 91) -> enc_out_dim ----
        self.encoder1 = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(4),
            nn.LeakyReLU(0.2, inplace=True),
        )  # (4, 45, 54, 45)
        self.encoder2 = nn.Sequential(
            nn.Conv3d(4, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(8),
            nn.LeakyReLU(0.2, inplace=True),
        )  # (8, 22, 27, 22)
        self.encoder3 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.2, inplace=True),
        )  # (16, 11, 13, 11)
        self.encoder4 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )  # (32, 5, 6, 5)
        self.encoder5 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 5 * 6 * 5, cfg.enc_out_dim),
            nn.Tanh(),
        )  # (enc_out_dim,)

        # ---- Inference head: (enc_x, h_{t-1}) -> mu_h, log_var_h ----
        h_in = cfg.enc_out_dim + cfg.latent_dim
        self.hidden2mu = nn.Linear(h_in, cfg.latent_dim)
        self.hidden2log_var = nn.Linear(h_in, cfg.latent_dim)

        # ---- Decoder: latent -> (1, 91, 109, 91) ----
        self.decoder1 = nn.Sequential(
            nn.Linear(cfg.latent_dim, 32 * 5 * 6 * 5),
            nn.Unflatten(1, (32, 5, 6, 5)),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.2, inplace=True),
        )  # (16, 11, 13, 11)
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=4, stride=2, padding=1, output_padding=(0, 1, 0), bias=False),
            nn.BatchNorm3d(8),
            nn.LeakyReLU(0.2, inplace=True),
        )  # (8, 22, 27, 22)
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose3d(8, 4, kernel_size=4, stride=2, padding=1, output_padding=(1, 0, 1), bias=False),
            nn.BatchNorm3d(4),
            nn.LeakyReLU(0.2, inplace=True),
        )  # (4, 45, 54, 45)
        self.decoder5 = nn.Sequential(
            nn.ConvTranspose3d(4, 1, kernel_size=4, stride=2, padding=1, output_padding=1, bias=False),
            nn.Tanh(),
        )  # (1, 91, 109, 91)

        # ---- Subject-specific noise z_s (Parameter; updated via SGD) ----
        z_init = torch.randn(train_size, cfg.z_dim) * cfg.sig_z
        self.z_vectors = nn.Parameter(z_init)

        # ---- Transition matrix F (Buffer; updated via closed-form ridge) ----
        # Buffer (not Parameter) because it has no gradient — we solve for it
        # in closed form. Buffers move with .to(device) like parameters.
        self.register_buffer("F_mat", torch.rand(cfg.latent_dim, cfg.latent_dim))

    # --------------------- core ops ---------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, 1, X, Y, Z) -> (B, enc_out_dim)``."""
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.encoder4(x)
        return self.encoder5(x)

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """``(B, latent_dim) -> (B, 1, X, Y, Z)``."""
        h = self.decoder1(h)
        h = self.decoder2(h)
        h = self.decoder3(h)
        h = self.decoder4(h)
        return self.decoder5(h)

    def reparametrize(self, mu_h: torch.Tensor, log_var_h: torch.Tensor) -> torch.Tensor:
        """Standard Gaussian reparameterization. Device- and dtype-agnostic.

        The original notebook called ``torch.randn(size=...)`` without a
        device argument and relied on ``torch.set_default_tensor_type
        ('torch.cuda.FloatTensor')`` to land on GPU. Here we pass device and
        dtype explicitly so the model runs on CPU/MPS/CUDA without global
        side effects.
        """
        sigma_h = torch.exp(log_var_h / 2)
        eps = torch.randn(mu_h.shape, device=mu_h.device, dtype=mu_h.dtype)
        return mu_h + sigma_h * eps

    def g_transform(self, h_old: torch.Tensor) -> torch.Tensor:
        """Linear temporal prior. ``h_old: (B, D) -> (B, D)``."""
        return h_old @ self.F_mat.T

    def vae_step(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        which_ones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One timestep of inference + decode.

        Returns ``(mu_t, h_t)`` where ``mu_t`` is the reconstruction and
        ``h_t`` the sampled posterior latent state.
        """
        enc_x = self.encode(x_t)
        combined = torch.cat([enc_x, h_prev], dim=1)
        mu_h = self.hidden2mu(combined)
        log_var_h = self.hidden2log_var(combined)
        h = self.reparametrize(mu_h, log_var_h)
        h_tilde = h + self.z_vectors[which_ones]
        mu_x = self.decode(h_tilde)
        return mu_x, h

    def forward(
        self,
        x: torch.Tensor,
        h_0: torch.Tensor,
        which_ones: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Roll out the latent recurrence over ``self.tol_time`` timesteps.

        Parameters
        ----------
        x : ``(B, 1, X, Y, Z, T)``
        h_0 : ``(B, latent_dim)``
        which_ones : ``(B,)`` long indices into ``z_vectors``

        Returns
        -------
        x_list, mu_history, h_history, gh_history : each a list of length ``tol_time``
        """
        T = x.shape[-1]
        if T < self.tol_time:
            raise ValueError(
                f"input has only {T} timepoints, need >= {self.tol_time}",
            )

        x_list = [x[..., t] for t in range(self.tol_time)]
        h = h_0
        mu_history: List[torch.Tensor] = []
        h_history: List[torch.Tensor] = []
        gh_history: List[torch.Tensor] = []
        for t in range(self.tol_time):
            gh_history.append(self.g_transform(h))
            mu, h = self.vae_step(x_list[t], h, which_ones)
            mu_history.append(mu)
            h_history.append(h)
        return x_list, mu_history, h_history, gh_history

    # --------------------- closed-form F update ---------------------
    @torch.no_grad()
    def updating_F(
        self,
        h_history_history: torch.Tensor,
        h0: torch.Tensor,
        rho: float,
    ) -> None:
        """Closed-form ridge update for the transition matrix F.

        Solves ``(X^T X + 2*sig_h^2*rho*I) F^T = X^T Y`` for F where:

        - ``Y`` stacks the posterior states ``h_t`` across all subjects/times
        - ``X`` is the same sequence shifted by one (``h_{t-1}``, with ``h_0``
          prepended) so that ``F`` predicts ``h_t`` from ``h_{t-1}``.

        This is MAP estimation of F under a zero-mean Gaussian prior with
        precision ``rho``; the factor ``2*sig_h^2`` matches the loss-2 scaling
        in :meth:`training_step`.

        Parameters
        ----------
        h_history_history : ``(N, T, latent_dim)`` accumulated posterior states
        h0 : ``(1, latent_dim)`` initial state shared across subjects
        rho : ridge penalty coefficient
        """
        N, T, D = h_history_history.shape
        if D != self.latent_dim:
            raise ValueError(f"last dim {D} != latent_dim {self.latent_dim}")

        rho_I = (
            2 * (self.cfg.sig_h ** 2) * rho
            * torch.eye(D, device=self.F_mat.device, dtype=self.F_mat.dtype)
        )

        Y_tilde = h_history_history.reshape(-1, D)
        x_shifted = torch.empty_like(h_history_history)
        x_shifted[:, 0] = h0
        x_shifted[:, 1:] = h_history_history[:, :-1]
        X_tilde = x_shifted.reshape(-1, D)

        XX = X_tilde.T @ X_tilde
        XY = X_tilde.T @ Y_tilde
        # torch.linalg.solve(A, B) returns X such that A X = B.
        new_F_T = torch.linalg.solve(XX + rho_I, XY)  # (D, D)
        # in-place buffer assignment to keep device/dtype/registration intact
        self.F_mat.copy_(new_F_T.T)

    # --------------------- training step ---------------------
    def training_step(
        self,
        batch: torch.Tensor,
        h_0: torch.Tensor,
        which_ones: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict, list]:
        """Compute composite training loss for one batch.

        Loss components (matches the canonical notebook):

        - ``loss1``: per-volume reconstruction MSE / ``sig_x^2``
        - ``loss2``: temporal-prior MSE / ``sig_h^2`` between ``h_t`` and ``g(h_{t-1})``
        - ``loss_z``: ``lambda_z * ||z||_1`` sparsity penalty
        - ``loss_F``: ``rho * ||F||_F^2``  (reported only — F is updated by closed
          form in :meth:`updating_F`, not by gradient).

        TODO(research): The canonical VAE ELBO has a KL term, not a temporal
        MSE. This implementation does MAP-style point estimation. To get a
        proper variational lower bound, replace ``loss2`` with
        ``KL(q(h_t|·) || N(g(h_{t-1}), sig_h^2 I))``.
        """
        cfg = self.cfg
        x_list, mu_history, h_history, gh_history = self(batch, h_0, which_ones)

        denom = 2 * batch.size(0) * len(h_history)
        loss1 = sum((x - mu).pow(2).sum() for x, mu in zip(x_list, mu_history))
        loss1 = loss1 / (cfg.sig_x ** 2) / denom

        loss2 = sum((h - gh).pow(2).sum() for h, gh in zip(h_history, gh_history))
        loss2 = loss2 / (cfg.sig_h ** 2) / denom

        loss_F = cfg.rho * (self.F_mat ** 2).sum()
        loss_z = cfg.lambda_z * self.z_vectors.abs().sum()

        loss = loss1 + loss2 + loss_z

        return (
            loss,
            {
                "loss1": loss1.detach(),
                "loss2": loss2.detach(),
                "loss_F": loss_F.detach(),
                "loss_z": loss_z.detach(),
            },
            h_history,
        )
