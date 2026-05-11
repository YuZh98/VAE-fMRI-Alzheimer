"""Tests for subject-level held-out evaluation."""

from __future__ import annotations

import torch

from recvae import RecVAEModel
from recvae.evaluation import evaluate_held_out, split_subjects


def test_split_subjects_partitions_indices():
    """Each subject appears in exactly one test set, in no train set of that fold."""
    splits = split_subjects(n_subjects=10, n_folds=5, seed=0)
    assert len(splits) == 5
    # Union of test sets covers every subject exactly once.
    all_test = []
    for train, test in splits:
        assert set(train).isdisjoint(test)
        assert set(train) | set(test) == set(range(10))
        all_test.extend(test)
    assert sorted(all_test) == list(range(10))


def test_split_subjects_is_deterministic_in_seed():
    a = split_subjects(10, 3, seed=42)
    b = split_subjects(10, 3, seed=42)
    c = split_subjects(10, 3, seed=43)
    assert a == b
    assert a != c


def test_evaluate_held_out_shapes_and_recon_mse(small_cfg, synthetic_volumes):
    model = RecVAEModel(train_size=4, cfg=small_cfg)
    # 2 held-out subjects; reuse synthetic_volumes which is shape (4, 1, 91, 109, 91, 3)
    volumes_test = synthetic_volumes[:2]
    h0 = torch.zeros(1, model.latent_dim)
    out = evaluate_held_out(
        model,
        volumes_test,
        h0,
        inner_steps=3,
        inner_lr=1e-3,
        inner_sig_z=0.5,
    )
    assert set(out.keys()) == {"recon_mse", "z_test", "h_test"}
    assert isinstance(out["recon_mse"], float)
    assert torch.isfinite(torch.tensor(out["recon_mse"]))
    assert out["z_test"].shape == (2, model.latent_dim)
    assert out["h_test"].shape == (2, small_cfg.tol_time, model.latent_dim)


def test_evaluate_held_out_changes_z(small_cfg, synthetic_volumes):
    """z_test must move during the inner loop — otherwise we are not learning."""
    torch.manual_seed(0)
    model = RecVAEModel(train_size=4, cfg=small_cfg)
    volumes_test = synthetic_volumes[:2]
    h0 = torch.zeros(1, model.latent_dim)

    # Snapshot z_test after zero inner steps via a tiny patch: we re-run with
    # 0 steps to get the initial value, then with >0 to confirm motion.
    out0 = evaluate_held_out(
        model,
        volumes_test,
        h0,
        inner_steps=0,
        inner_lr=1e-2,
        inner_sig_z=0.5,
    )
    torch.manual_seed(0)  # same init RNG so the random init lands on the same value
    out1 = evaluate_held_out(
        model,
        volumes_test,
        h0,
        inner_steps=5,
        inner_lr=1e-2,
        inner_sig_z=0.5,
    )
    assert not torch.allclose(out0["z_test"], out1["z_test"]), (
        "z_test did not change after 5 inner SGD steps",
    )


def test_evaluate_held_out_does_not_mutate_model_params(small_cfg, synthetic_volumes):
    """Model parameters and buffers must be identical before and after the call."""
    model = RecVAEModel(train_size=4, cfg=small_cfg)
    volumes_test = synthetic_volumes[:2]
    h0 = torch.zeros(1, model.latent_dim)

    before_params = {name: p.detach().clone() for name, p in model.named_parameters()}
    before_buffers = {name: b.detach().clone() for name, b in model.named_buffers()}
    before_flags = {name: p.requires_grad for name, p in model.named_parameters()}

    _ = evaluate_held_out(
        model,
        volumes_test,
        h0,
        inner_steps=3,
        inner_lr=1e-3,
        inner_sig_z=0.5,
    )

    for name, p in model.named_parameters():
        assert torch.equal(before_params[name], p.detach()), f"{name} mutated"
        assert p.requires_grad == before_flags[name], f"{name} requires_grad changed"
    for name, b in model.named_buffers():
        assert torch.equal(before_buffers[name], b.detach()), f"buffer {name} mutated"
