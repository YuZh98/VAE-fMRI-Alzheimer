"""Subject-level held-out evaluation for RecVAE.

The training loop fits one ``z_s`` per training subject and shares
encoder/decoder/inference parameters across subjects. To score on a
held-out subject, those shared parameters stay frozen while a fresh
subject-specific ``z`` is fit by a few SGD steps minimizing only the
reconstruction term. The reconstruction MSE after that inner loop is the
out-of-sample metric we report.

This is essentially the inference-time analogue of the training
optimization, with the F-update and L1 sparsity penalty dropped — those
are training-time regularizers and don't influence the held-out
reconstruction score.
"""

from __future__ import annotations

import random

import torch
from torch import nn

from .model import RecVAEModel


def split_subjects(n_subjects: int, n_folds: int, seed: int) -> list[tuple[list[int], list[int]]]:
    """Return k-fold (train_idx, test_idx) splits over subject IDs.

    Subjects are shuffled with the given seed, then partitioned into
    ``n_folds`` contiguous groups. Each fold takes one group as test and
    the rest as train. With ``n_subjects`` not divisible by ``n_folds``,
    the first few folds get one extra subject in the test set.

    Parameters
    ----------
    n_subjects : total number of subjects (indices 0 .. n_subjects-1).
    n_folds : number of folds. Must satisfy ``1 <= n_folds <= n_subjects``.
    seed : RNG seed for the shuffle.

    Returns
    -------
    list of ``(train_idx, test_idx)`` tuples, length ``n_folds``. Each
    element is a pair of Python lists of integers.
    """
    if n_subjects <= 0:
        raise ValueError(f"n_subjects must be > 0, got {n_subjects}")
    if not (1 <= n_folds <= n_subjects):
        raise ValueError(
            f"n_folds must be in [1, n_subjects={n_subjects}], got {n_folds}",
        )

    rng = random.Random(seed)
    indices = list(range(n_subjects))
    rng.shuffle(indices)

    # Even split with the remainder distributed to the first folds.
    base = n_subjects // n_folds
    rem = n_subjects % n_folds

    splits = []
    cursor = 0
    for fold in range(n_folds):
        size = base + (1 if fold < rem else 0)
        test = indices[cursor : cursor + size]
        train = indices[:cursor] + indices[cursor + size :]
        splits.append((train, test))
        cursor += size
    return splits


def _forward_with_external_z(
    model: RecVAEModel,
    x: torch.Tensor,
    h0: torch.Tensor,
    z_test: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mimic :meth:`RecVAEModel.forward` but use ``z_test`` as subject noise.

    Returns ``(mu_stack, h_stack)`` of shape ``(B, T, ...)``. We re-implement
    the rollout instead of calling ``model.forward`` because that method
    indexes into ``model.z_vectors`` via ``which_ones``; here we want the
    gradient to flow into our own ``z_test`` tensor.
    """
    T = x.shape[-1]
    if model.tol_time > T:
        raise ValueError(
            f"input has only {T} timepoints, need >= {model.tol_time}",
        )

    h = h0
    mu_per_t = []
    h_per_t = []
    for t in range(model.tol_time):
        x_t = x[..., t]
        enc_x = model.encode(x_t)
        combined = torch.cat([enc_x, h], dim=1)
        mu_h = model.hidden2mu(combined)
        log_var_h = model.hidden2log_var(combined)
        h = model.reparametrize(mu_h, log_var_h)
        h_tilde = h + z_test
        mu_x = model.decode(h_tilde)
        mu_per_t.append(mu_x)
        h_per_t.append(h)

    return torch.stack(mu_per_t, dim=1), torch.stack(h_per_t, dim=1)


def evaluate_held_out(
    model: RecVAEModel,
    volumes_test: torch.Tensor,
    h0: torch.Tensor,
    *,
    inner_steps: int = 100,
    inner_lr: float = 1e-3,
    inner_sig_z: float = 1.0,
) -> dict:
    """Score a frozen model on held-out subjects by fitting their z_s only.

    Freezes the encoder, decoder, inference head, F buffer, and training
    ``z_vectors``. A fresh ``z_test`` of shape ``(N_test, latent_dim)`` is
    optimized for ``inner_steps`` plain SGD steps on the reconstruction
    MSE (the ``loss1`` term only). Returns the final reconstruction MSE,
    the optimized ``z_test``, and the per-step latent trajectory ``h_test``.

    Model parameters are guaranteed to be unchanged after the call: the
    function tracks the original ``requires_grad`` flags and restores them
    on exit, and no in-place mutation of any tensor in ``model.parameters()``
    or ``model.buffers()`` is performed.

    Parameters
    ----------
    model : trained RecVAEModel.
    volumes_test : ``(N_test, 1, X, Y, Z, T)`` held-out volumes.
    h0 : ``(1, latent_dim)`` or ``(N_test, latent_dim)`` initial latent state.
    inner_steps : number of inner SGD iterations.
    inner_lr : learning rate for the inner SGD on z_test.
    inner_sig_z : initialization std for ``z_test``.

    Returns
    -------
    dict with keys ``recon_mse`` (scalar), ``z_test`` (``(N_test, D)``),
    and ``h_test`` (``(N_test, T, D)``).
    """
    if volumes_test.dim() != 6:
        raise ValueError(
            f"expected 6D (N,1,X,Y,Z,T), got shape {tuple(volumes_test.shape)}",
        )
    n_test = volumes_test.shape[0]
    device = volumes_test.device
    dtype = volumes_test.dtype
    latent_dim = model.latent_dim

    # Eval mode pins BatchNorm to running stats so the inner-loop forward
    # passes are deterministic given the same RNG. Reparam still draws eps
    # at each step; that randomness is the only nondeterminism left.
    was_training = model.training
    model.eval()

    # Snapshot and freeze every parameter on the model.
    saved_requires_grad = {name: p.requires_grad for name, p in model.named_parameters()}
    for p in model.parameters():
        p.requires_grad_(False)

    # Broadcast h0 if a single (1, D) was passed.
    if h0.dim() == 2 and h0.shape[0] == 1 and n_test > 1:
        h_init = h0.expand(n_test, -1).contiguous()
    else:
        h_init = h0

    z_test = nn.Parameter(torch.randn(n_test, latent_dim, device=device, dtype=dtype) * inner_sig_z)

    opt = torch.optim.SGD([z_test], lr=inner_lr)

    try:
        for _ in range(inner_steps):
            opt.zero_grad()
            mu_stack, _ = _forward_with_external_z(model, volumes_test, h_init, z_test)
            # x_stack: (N, 1, X, Y, Z, T) -> (N, T, 1, X, Y, Z) to match mu_stack.
            x_stack = volumes_test.permute(0, 5, 1, 2, 3, 4).contiguous()
            recon = (x_stack - mu_stack).pow(2).mean()
            recon.backward()
            opt.step()

        # Final forward to record h_test and the post-fit MSE.
        with torch.no_grad():
            mu_stack, h_stack = _forward_with_external_z(
                model,
                volumes_test,
                h_init,
                z_test,
            )
            x_stack = volumes_test.permute(0, 5, 1, 2, 3, 4).contiguous()
            final_mse = (x_stack - mu_stack).pow(2).mean().item()
    finally:
        # Restore parameter grad flags whether the inner loop succeeded or not.
        for name, p in model.named_parameters():
            p.requires_grad_(saved_requires_grad.get(name, True))
        if was_training:
            model.train()

    return {
        "recon_mse": final_mse,
        "z_test": z_test.detach(),
        "h_test": h_stack.detach(),
    }
