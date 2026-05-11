"""Recurrent 3D-convolutional VAE for temporal fMRI volumes.

Tensor shape conventions
------------------------
- batch x:            ``(B, 1, X=91, Y=109, Z=91, T=120)``
- single timestep:    ``(B, 1, 91, 109, 91)``
- encoded feature:    ``(B, enc_out_dim=100)``
- latent state h:     ``(B, latent_dim=10)``
- F matrix:           ``(latent_dim, latent_dim)``
- z_vectors:          ``(N_train, latent_dim)``
- decoded mu:         ``(B, 1, 91, 109, 91)``

Encoder/decoder spatial dimension chain (verified against PyTorch 2.x):

    91 -> 45 -> 22 -> 11 -> 5      (Conv3d k=4, s=2, p=1)
     5 -> 11 -> 22 -> 45 -> 91     (ConvTranspose3d with non-trivial output_padding)

The non-trivial ``output_padding`` values are required because the spatial
extents (91, 109, 91) are odd; choosing wrong here will silently produce a
mismatched reconstruction shape.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from .config import Config
from .losses import RecVAELoss


class RolloutOutput(NamedTuple):
    """Stacked output of one :meth:`RecVAEModel.forward` rollout.

    Each field has a leading batch dimension followed by time:

    - ``x`` : ``(B, T, 1, X, Y, Z)``  — the input volume sliced per timestep
    - ``mu``: ``(B, T, 1, X, Y, Z)``  — decoder reconstructions
    - ``h`` : ``(B, T, latent_dim)``  — sampled posterior latent states
    - ``gh``: ``(B, T, latent_dim)``  — temporal prior means ``h_{t-1} @ F^T``

    Implements ``__iter__`` returning the legacy 4-tuple-of-lists view
    ``(x_list, mu_history, h_history, gh_history)`` so that existing
    callers ``xs, mus, hs, ghs = model(...)`` continue to work unchanged.
    """

    x: torch.Tensor
    mu: torch.Tensor
    h: torch.Tensor
    gh: torch.Tensor

    def __iter__(self):
        """Backward-compat 4-list view.

        The original ``forward`` returned four Python lists of per-timestep
        tensors. Yielding the equivalent lists here lets callers continue
        to use tuple unpacking without code changes; new callers should
        use the named attributes directly.
        """
        x_list = [self.x[:, t] for t in range(self.x.shape[1])]
        mu_list = [self.mu[:, t] for t in range(self.mu.shape[1])]
        h_list = [self.h[:, t] for t in range(self.h.shape[1])]
        gh_list = [self.gh[:, t] for t in range(self.gh.shape[1])]
        yield x_list
        yield mu_list
        yield h_list
        yield gh_list


class RecVAEModel(nn.Module):
    """Recurrent VAE with linear temporal prior ``g(h) = h @ F^T``.

    Parameters
    ----------
    train_size : number of subjects in the training set (sets ``z_vectors`` size)
    cfg : configuration; falls back to ``Config()`` defaults
    """

    def __init__(self, train_size: int, cfg: Config | None = None):
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
            nn.ConvTranspose3d(
                32, 16, kernel_size=4, stride=2, padding=1, output_padding=1, bias=False
            ),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.2, inplace=True),
        )  # (16, 11, 13, 11)
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose3d(
                16, 8, kernel_size=4, stride=2, padding=1, output_padding=(0, 1, 0), bias=False
            ),
            nn.BatchNorm3d(8),
            nn.LeakyReLU(0.2, inplace=True),
        )  # (8, 22, 27, 22)
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose3d(
                8, 4, kernel_size=4, stride=2, padding=1, output_padding=(1, 0, 1), bias=False
            ),
            nn.BatchNorm3d(4),
            nn.LeakyReLU(0.2, inplace=True),
        )  # (4, 45, 54, 45)
        self.decoder5 = nn.Sequential(
            nn.ConvTranspose3d(
                4, 1, kernel_size=4, stride=2, padding=1, output_padding=1, bias=False
            ),
            nn.Tanh(),
        )  # (1, 91, 109, 91)

        # ---- Subject-specific noise z_s (Parameter; updated via SGD) ----
        z_init = torch.randn(train_size, cfg.latent_dim) * cfg.sig_z
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
    ) -> RolloutOutput:
        """Roll out the latent recurrence over ``self.tol_time`` timesteps.

        Parameters
        ----------
        x : ``(B, 1, X, Y, Z, T)``
        h_0 : ``(B, latent_dim)``
        which_ones : ``(B,)`` long indices into ``z_vectors``

        Returns
        -------
        RolloutOutput with stacked tensors of shape ``(B, T, ...)``.

        The returned NamedTuple still supports the legacy
        ``xs, mus, hs, ghs = model(...)`` unpacking via ``__iter__``; it
        yields four lists of per-timestep tensors in that case.
        """
        T = x.shape[-1]
        if self.tol_time > T:
            raise ValueError(
                f"input has only {T} timepoints, need >= {self.tol_time}",
            )

        x_per_t = [x[..., t] for t in range(self.tol_time)]
        h = h_0
        mu_per_t = []
        h_per_t = []
        gh_per_t = []
        for t in range(self.tol_time):
            gh_per_t.append(self.g_transform(h))
            mu, h = self.vae_step(x_per_t[t], h, which_ones)
            mu_per_t.append(mu)
            h_per_t.append(h)

        # Stack lists into (B, T, ...) tensors. torch.stack on dim=1 places
        # the time axis right after the batch axis, which keeps slicing of
        # the form out.h[:, t] cheap and matches RolloutOutput's docstring.
        x_stack = torch.stack(x_per_t, dim=1)
        mu_stack = torch.stack(mu_per_t, dim=1)
        h_stack = torch.stack(h_per_t, dim=1)
        gh_stack = torch.stack(gh_per_t, dim=1)

        return RolloutOutput(x=x_stack, mu=mu_stack, h=h_stack, gh=gh_stack)

    # --------------------- closed-form F update ---------------------
    @torch.no_grad()
    def updating_F(
        self,
        h_history_history: torch.Tensor,
        h0: torch.Tensor,
        rho: float,
    ) -> None:
        """Closed-form ridge update for the transition matrix F.

        Objective minimized
        -------------------
        This is the F minimizer of the same composite loss optimized by SGD
        in :meth:`training_step`. The SGD ``loss2`` is

            loss2(F) = (1 / (sig_h^2 * 2 N T)) * sum_{n,t} ||h_{n,t} - g(h_{n,t-1})||^2

        plus the F-only regularizer

            loss_F(F) = rho * ||F||_F^2.

        Setting ``d/dF (loss2 + loss_F) = 0`` gives the normal equation

            (X^T X + 2 N T sig_h^2 rho I) F^T = X^T Y

        where ``Y`` stacks the posterior states ``h_t`` over all subjects
        and time, and ``X`` is the same sequence shifted by one (``h_{t-1}``
        with ``h_0`` prepended at t=0). The previous implementation used
        only ``2 sig_h^2 rho I`` and so was solving a different objective
        than the SGD loss claimed to optimize; this method fixes that.

        Parameters
        ----------
        h_history_history : ``(N, T, latent_dim)`` accumulated posterior states
        h0 : ``(1, latent_dim)`` initial state shared across subjects
        rho : ridge penalty coefficient
        """
        N, T, D = h_history_history.shape
        if self.latent_dim != D:
            raise ValueError(f"last dim {D} != latent_dim {self.latent_dim}")

        # Ridge term must include the N*T factor so the closed form matches
        # the per-volume averaging used inside training_step / RecVAELoss.
        rho_I = (
            2
            * N
            * T
            * (self.cfg.sig_h**2)
            * rho
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
        loss_fn: RecVAELoss | None = None,
    ) -> tuple[torch.Tensor, dict, list]:
        """Compute composite training loss for one batch.

        Thin wrapper over a :class:`recvae.losses.RecVAELoss` callable; the
        default loss reproduces the canonical notebook scaling. Pass a
        custom ``loss_fn`` (e.g. :class:`recvae.losses.KLRecVAELoss`) to
        swap loss formulations without touching this method.

        Returns ``(loss, loss_dic, h_history)`` where ``h_history`` is the
        list of per-timestep posterior tensors. The list form is preserved
        for callers (e.g. :func:`recvae.train.fit`) that downstream stack
        the trajectory into the F-update buffer.

        Loss components reported in ``loss_dic``:

        - ``loss1``: per-volume reconstruction MSE / ``sig_x^2``
        - ``loss2``: temporal-prior MSE / ``sig_h^2`` between ``h_t`` and
          ``g(h_{t-1})``
        - ``loss_z``: ``lambda_z * ||z||_1`` sparsity penalty
        - ``loss_F``: ``rho * ||F||_F^2`` (reported only — F is updated by
          closed form in :meth:`updating_F`, not by gradient).
        """
        cfg = self.cfg
        if loss_fn is None:
            loss_fn = RecVAELoss(
                sig_x=cfg.sig_x,
                sig_h=cfg.sig_h,
                lambda_z=cfg.lambda_z,
                rho=cfg.rho,
            )

        out = self(batch, h_0, which_ones)
        # Recover per-timestep lists via the legacy iteration interface so
        # the loss callable can consume the same data shape regardless of
        # whether forward returns a RolloutOutput or a 4-list tuple.
        x_list, mu_history, h_history, gh_history = out

        loss, loss_dic = loss_fn(
            self,
            batch,
            h_history,
            gh_history,
            mu_history,
            x_list,
        )

        return loss, loss_dic, h_history
