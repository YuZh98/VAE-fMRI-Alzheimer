"""Hyperparameter configuration for RecVAE.

The defaults match the experimental settings in the canonical
RecVAE_on_fMRI.ipynb / Version4.ipynb notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    """All hyperparameters for the RecVAE model and training loop.

    Notes
    -----
    The legacy notebooks carried two separate dimension fields, ``latent_dim``
    and ``z_dim``. In V4 they were always set equal, and the package collapses
    them into a single ``latent_dim`` to remove the duplicate-source-of-truth
    pitfall. The subject-specific noise tensor ``z_vectors`` is shaped
    ``(train_size, latent_dim)``.
    """

    # ---- Model dimensions ----
    enc_out_dim: int = 100
    latent_dim: int = 10
    tol_time: int = 120
    spatial_shape: tuple[int, int, int, int] = field(default_factory=lambda: (1, 91, 109, 91))

    # ---- Loss noise scales (treated as fixed weights, NOT learned) ----
    # TODO(research): make these learnable nn.Parameter (with softplus to keep
    # positive) if you want the model to calibrate its own observation noise.
    sig_x: float = 1.0
    sig_h: float = 1.0
    sig_z: float = 1.0

    # ---- Regularization ----
    rho: float = 0.1  # ridge penalty for closed-form F update
    lambda_z: float = 10.0  # L1 weight on subject-specific noise z_s

    # ---- Training ----
    batch_size: int = 4
    learning_rate: float = 1e-6
    epochs: int = 500

    # ---- Reproducibility ----
    seed: int = 2022

    def __post_init__(self) -> None:
        """Validate that dimensions and noise scales are positive.

        Catching these at construction time prevents downstream
        ``torch.zeros((-1, 10))`` style crashes whose tracebacks point at
        somewhere unrelated to the misconfiguration.
        """
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be > 0, got {self.latent_dim}")
        if self.enc_out_dim <= 0:
            raise ValueError(f"enc_out_dim must be > 0, got {self.enc_out_dim}")
        if self.tol_time <= 0:
            raise ValueError(f"tol_time must be > 0, got {self.tol_time}")
        if self.sig_x <= 0:
            raise ValueError(f"sig_x must be > 0, got {self.sig_x}")
        if self.sig_h <= 0:
            raise ValueError(f"sig_h must be > 0, got {self.sig_h}")
        if self.sig_z <= 0:
            raise ValueError(f"sig_z must be > 0, got {self.sig_z}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be > 0, got {self.epochs}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
