"""Shape and behavior tests for RecVAEModel."""

from __future__ import annotations

import pytest
import torch

from recvae import RecVAEModel


def test_encoder_decoder_shape_chain(model: RecVAEModel):
    """Round-trip a single timepoint through encoder + decoder."""
    x = torch.randn(2, 1, 91, 109, 91)
    enc = model.encode(x)
    assert enc.shape == (2, model.cfg.enc_out_dim), enc.shape
    h = torch.randn(2, model.latent_dim)
    dec = model.decode(h)
    assert dec.shape == (2, 1, 91, 109, 91), dec.shape


def test_forward_returns_lists_of_correct_length(model: RecVAEModel, synthetic_volumes):
    x = synthetic_volumes[:2]
    h0 = torch.randn(2, model.latent_dim)
    which = torch.tensor([0, 1])
    xs, mus, hs, ghs = model(x, h0, which)
    T = model.tol_time
    assert len(xs) == len(mus) == len(hs) == len(ghs) == T
    assert mus[0].shape == (2, 1, 91, 109, 91)
    assert hs[0].shape == (2, model.latent_dim)
    assert ghs[0].shape == (2, model.latent_dim)


def test_forward_raises_on_short_input(model: RecVAEModel):
    """tol_time=3 input fails when given only 2 timepoints."""
    x = torch.randn(2, 1, 91, 109, 91, 2)
    h0 = torch.randn(2, model.latent_dim)
    which = torch.tensor([0, 1])
    with pytest.raises(ValueError, match="timepoints"):
        model(x, h0, which)


def test_reparametrize_is_device_agnostic(model: RecVAEModel):
    """Sample epsilon should land on the same device as inputs without
    relying on a global default tensor type.
    """
    mu = torch.zeros(2, model.latent_dim)
    log_var = torch.zeros(2, model.latent_dim)
    out = model.reparametrize(mu, log_var)
    assert out.device == mu.device
    assert out.dtype == mu.dtype
    assert out.shape == mu.shape


def test_z_vectors_is_parameter_with_correct_size(model: RecVAEModel):
    assert isinstance(model.z_vectors, torch.nn.Parameter)
    assert model.z_vectors.shape == (model.train_size, model.cfg.latent_dim)
    assert model.z_vectors.requires_grad


def test_F_mat_is_buffer_no_grad(model: RecVAEModel):
    """F_mat is a buffer — it's updated in closed form, not by gradient."""
    assert "F_mat" in dict(model.named_buffers())
    assert model.F_mat.requires_grad is False
    assert model.F_mat.shape == (model.latent_dim, model.latent_dim)


def test_g_transform_uses_F_transpose(model: RecVAEModel):
    """g(h) should equal h @ F^T."""
    h = torch.randn(3, model.latent_dim)
    expected = h @ model.F_mat.T
    got = model.g_transform(h)
    assert torch.allclose(got, expected)


def test_updating_F_mutates_buffer_with_finite_values(model: RecVAEModel):
    N, T, D = 4, model.tol_time, model.latent_dim
    h_hist = torch.randn(N, T, D)
    h0 = torch.randn(1, D)
    F_before = model.F_mat.clone()
    model.updating_F(h_hist, h0, rho=0.1)
    assert not torch.allclose(F_before, model.F_mat)
    assert torch.isfinite(model.F_mat).all()


def test_updating_F_solves_ridge_system(model: RecVAEModel):
    """Manually solve the ridge system and compare against updating_F.

    The ridge factor must include the N*T sample-count term so the closed
    form matches the same objective the SGD loss minimizes (per-volume
    averaged ``loss2`` plus ``rho * ||F||_F^2``).
    """
    N, T, D = 2, model.tol_time, model.latent_dim
    torch.manual_seed(42)
    h_hist = torch.randn(N, T, D)
    h0 = torch.zeros(1, D)
    rho = 0.5

    # Reference computation
    Y = h_hist.reshape(-1, D)
    x_shifted = torch.empty_like(h_hist)
    x_shifted[:, 0] = h0
    x_shifted[:, 1:] = h_hist[:, :-1]
    X = x_shifted.reshape(-1, D)
    rho_I = 2 * N * T * (model.cfg.sig_h**2) * rho * torch.eye(D)
    F_expected = torch.linalg.solve(X.T @ X + rho_I, X.T @ Y).T

    model.updating_F(h_hist, h0, rho=rho)
    assert torch.allclose(model.F_mat, F_expected, atol=1e-5)


def test_updating_F_is_critical_point_of_combined_loss(model: RecVAEModel):
    """At the closed-form F, the gradient of (loss2 + loss_F) wrt F is ~0.

    This is the strongest possible check that the closed form solves the
    objective claimed by training_step: differentiating the same expression
    that goes into the SGD loss and evaluating at ``model.F_mat`` must
    yield a near-zero gradient.
    """
    N, T, D = 3, model.tol_time, model.latent_dim
    torch.manual_seed(7)
    h_hist = torch.randn(N, T, D)
    h0 = torch.zeros(1, D)
    rho = 0.4
    sig_h = model.cfg.sig_h

    model.updating_F(h_hist, h0, rho=rho)

    # Build (X, Y) from the same h_history the closed form consumed.
    Y = h_hist.reshape(-1, D)
    x_shifted = torch.empty_like(h_hist)
    x_shifted[:, 0] = h0
    x_shifted[:, 1:] = h_hist[:, :-1]
    X = x_shifted.reshape(-1, D)

    F_mat = model.F_mat.detach().clone().requires_grad_(True)
    # gh = X @ F^T; loss2 matches training_step's averaging (/ (2 * N * T * sig_h^2)).
    gh = X @ F_mat.T
    loss2 = (Y - gh).pow(2).sum() / (sig_h**2) / (2 * N * T)
    loss_F = rho * (F_mat**2).sum()
    loss_combined = loss2 + loss_F

    (grad,) = torch.autograd.grad(loss_combined, F_mat)
    assert grad.abs().max().item() < 1e-4, grad.abs().max().item()


def test_updating_F_rejects_wrong_latent_dim(model: RecVAEModel):
    h_hist = torch.randn(2, model.tol_time, model.latent_dim + 1)
    h0 = torch.zeros(1, model.latent_dim + 1)
    with pytest.raises(ValueError, match="latent_dim"):
        model.updating_F(h_hist, h0, rho=0.1)
