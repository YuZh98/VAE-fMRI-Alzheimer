"""Shared pytest fixtures.

All fixtures use synthetic tensors so the suite needs no real fMRI data
and no nibabel install.
"""

from __future__ import annotations

import pytest
import torch

from recvae import Config, RecVAEModel, set_seed


@pytest.fixture(autouse=True)
def _pin_seed():
    """Every test runs with the same RNG state."""
    set_seed(0)


@pytest.fixture
def small_cfg() -> Config:
    """Config sized for fast tests: 3 timepoints instead of 120."""
    return Config(tol_time=3, batch_size=2, epochs=1, learning_rate=1e-3)


@pytest.fixture
def synthetic_volumes() -> torch.Tensor:
    """4 subjects, 3 timepoints; matches Config.spatial_shape (1,91,109,91)."""
    return torch.randn(4, 1, 91, 109, 91, 3)


@pytest.fixture
def model(small_cfg) -> RecVAEModel:
    return RecVAEModel(train_size=4, cfg=small_cfg)
