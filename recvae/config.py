"""Hyperparameter configuration for RecVAE.

The defaults match the experimental settings in the canonical
RecVAE_on_fMRI.ipynb / Version4.ipynb notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Config:
    """All hyperparameters for the RecVAE model and training loop."""

    # ---- Model dimensions ----
    enc_out_dim: int = 100
    latent_dim: int = 10
    z_dim: int = 10
    tol_time: int = 120
    spatial_shape: Tuple[int, int, int, int] = field(default_factory=lambda: (1, 91, 109, 91))

    # ---- Loss noise scales (treated as fixed weights, NOT learned) ----
    # TODO(research): make these learnable nn.Parameter (with softplus to keep
    # positive) if you want the model to calibrate its own observation noise.
    sig_x: float = 1.0
    sig_h: float = 1.0
    sig_z: float = 1.0

    # ---- Regularization ----
    rho: float = 0.1        # ridge penalty for closed-form F update
    lambda_z: float = 10.0  # L1 weight on subject-specific noise z_s

    # ---- Training ----
    batch_size: int = 4
    learning_rate: float = 1e-6
    epochs: int = 500

    # ---- Reproducibility ----
    seed: int = 2022
