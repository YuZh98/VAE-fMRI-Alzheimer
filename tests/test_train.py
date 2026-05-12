"""End-to-end smoke tests for fit() and evaluate()."""

from __future__ import annotations

import logging

import torch

from recvae import (
    Config,
    FMRIDataset,
    RecVAEModel,
    build_dataloader,
    evaluate,
    fit,
)


def test_one_step_loss_is_finite_and_grads_flow(model: RecVAEModel, synthetic_volumes):
    """One forward+backward step on a tiny batch — no NaN, every parameter
    that should learn actually receives a gradient.
    """
    x = synthetic_volumes[:2]
    h0 = torch.zeros(2, model.latent_dim)
    which = torch.tensor([0, 1])

    loss, dic, _ = model.training_step(x, h0, which)
    assert torch.isfinite(loss), loss
    for k, v in dic.items():
        assert torch.isfinite(v), (k, v)

    loss.backward()
    # at least one encoder/decoder/inference weight got a non-None grad
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert all(torch.isfinite(g).all() for g in grads), "non-finite grad"


def test_z_vectors_receive_gradient(model: RecVAEModel, synthetic_volumes):
    """The L1 sparsity term should produce a gradient on z_vectors."""
    x = synthetic_volumes[:2]
    h0 = torch.zeros(2, model.latent_dim)
    which = torch.tensor([0, 1])
    loss, _, _ = model.training_step(x, h0, which)
    loss.backward()
    assert model.z_vectors.grad is not None
    # subjects 0 and 1 were in the batch + L1 fires on all rows -> all rows get grad
    assert torch.isfinite(model.z_vectors.grad).all()


def test_fit_runs_one_epoch_and_updates_F(small_cfg: Config, synthetic_volumes):
    """Smoke test: fit() runs 1 epoch on 4 synthetic subjects and updates F."""
    model = RecVAEModel(train_size=4, cfg=small_cfg)
    F_before = model.F_mat.clone()

    ds = FMRIDataset(synthetic_volumes)
    dl = build_dataloader(ds, batch_size=2, shuffle=False, seed=0)
    h0 = torch.zeros(1, model.latent_dim)

    out = fit(model, dl, h0, cfg=small_cfg, epochs=1, log_every=1)

    assert "train_loss_history" in out
    assert "all_h_history" in out
    assert out["all_h_history"].shape == (4, small_cfg.tol_time, model.latent_dim)
    assert torch.isfinite(out["all_h_history"]).all()
    assert not torch.allclose(F_before, model.F_mat), "F should have been updated"


def test_evaluate_runs_in_no_grad_mode(model: RecVAEModel, synthetic_volumes):
    x = synthetic_volumes[:2]
    h0 = torch.zeros(2, model.latent_dim)
    which = torch.tensor([0, 1])
    xs, mus, hs, ghs = evaluate(model, x, h0, which)
    assert mus[0].requires_grad is False
    assert model.training is False


def test_fit_logs_one_info_line_per_epoch(small_cfg: Config, synthetic_volumes, caplog):
    """fit() must emit exactly one INFO line containing '[epoch' per epoch."""
    model = RecVAEModel(train_size=4, cfg=small_cfg)
    ds = FMRIDataset(synthetic_volumes)
    dl = build_dataloader(ds, batch_size=2, shuffle=False, seed=0)
    h0 = torch.zeros(1, model.latent_dim)

    # Capture from the recvae.train logger at INFO. propagate=True ensures
    # the logger forwards records to caplog's root handler.
    with caplog.at_level(logging.INFO, logger="recvae.train"):
        fit(model, dl, h0, cfg=small_cfg, epochs=2, log_every=1)

    epoch_lines = [
        r for r in caplog.records if r.levelno == logging.INFO and "[epoch" in r.getMessage()
    ]
    assert len(epoch_lines) == 2, (
        f"expected 2 epoch log lines, got {len(epoch_lines)}: "
        f"{[r.getMessage() for r in epoch_lines]}"
    )
