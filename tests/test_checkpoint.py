"""Round-trip tests for save_run / load_run."""

from __future__ import annotations

import torch

from recvae import RecVAEModel
from recvae.checkpoint import load_run, save_run


def test_save_load_round_trip_matches_state(tmp_path, small_cfg):
    model = RecVAEModel(train_size=4, cfg=small_cfg)
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)
    # Take a step so the optimizer has non-trivial state to round-trip.
    x = torch.randn(2, 1, 91, 109, 91, small_cfg.tol_time)
    h0 = torch.zeros(2, model.latent_dim)
    which = torch.tensor([0, 1])
    loss, _, _ = model.training_step(x, h0, which)
    loss.backward()
    optim.step()

    history = {"train_loss_history": [1.0, 0.9, 0.8]}

    run_dir = save_run(
        tmp_path / "run0",
        model,
        optimizer=optim,
        epoch=2,
        cfg=small_cfg,
        history=history,
    )
    assert (run_dir / "model_state.pt").is_file()
    assert (run_dir / "optimizer_state.pt").is_file()
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "history.json").is_file()
    assert (run_dir / "rng_states.pt").is_file()

    # Build a fresh model + optimizer with different weights, load, compare.
    model2 = RecVAEModel(train_size=4, cfg=small_cfg)
    optim2 = torch.optim.SGD(model2.parameters(), lr=1e-3)
    # Ensure they actually differ before load.
    assert not torch.equal(model.F_mat, model2.F_mat) or not torch.equal(
        model.z_vectors.detach(), model2.z_vectors.detach()
    )

    loaded = load_run(run_dir, model2, optimizer=optim2, restore_rng=False)

    # Parameter equality after load.
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters(), strict=True):
        assert n1 == n2
        assert torch.equal(p1.detach(), p2.detach()), n1
    for (n1, b1), (n2, b2) in zip(model.named_buffers(), model2.named_buffers(), strict=True):
        assert n1 == n2
        assert torch.equal(b1, b2), n1

    assert loaded.get("epoch") == 2
    assert loaded.get("history") == history
    assert "cfg" in loaded
    assert loaded["cfg"]["latent_dim"] == small_cfg.latent_dim


def test_save_load_without_optional_args(tmp_path, small_cfg):
    """Minimal save: only model. load_run should still return a dict."""
    model = RecVAEModel(train_size=2, cfg=small_cfg)
    run_dir = save_run(tmp_path / "run1", model)
    assert (run_dir / "model_state.pt").is_file()
    assert not (run_dir / "optimizer_state.pt").exists()

    model2 = RecVAEModel(train_size=2, cfg=small_cfg)
    out = load_run(run_dir, model2, restore_rng=False)
    assert isinstance(out, dict)
    # No optimizer/cfg/history were saved, so the dict only contains
    # whatever optional metadata happens to be present (e.g. git_sha).
    assert "optimizer_state" not in out
    assert "cfg" not in out
    assert "history" not in out
