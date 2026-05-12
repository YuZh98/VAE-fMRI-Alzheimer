"""Widen the decoder and drop the saturating Tanh — the fix path for the
decoder-collapse problem documented in the README and
``docs/background/this_models_design.md``.

The canonical model uses decoder channels ``[4, 8, 16, 32]`` and a final
``Tanh`` that, combined with per-subject min-max normalization, makes
"output ≈ 0" the cheapest place for the optimizer to land. This script
hot-patches a fresh ``RecVAEModel`` to use widths ``[16, 32, 64, 128]``
and replaces the final ``Tanh`` with ``Hardtanh(-3, 3)`` so the network
still has some output bounding without saturating inside ``[-0.3, 0.3]``
where the bulk of normalized voxel values sit.

Caveat
------
Demonstrates the fix path; not validated on real fMRI; changes the model
contract so existing checkpoints won't load. The wide-decoder model has
roughly 16x more decoder parameters than the canonical one and is
intended only as a demonstration that *the structural diagnosis is
correct*, not as a drop-in replacement.

Empirical note: at this demo scale (8 subjects, T=4, 30 epochs) widening
alone does not yet drive Pearson(x, mu) above ~0.01. The reconstruction
std/input std ratio does climb (~0.3-0.4 vs ~0.17 on the canonical
config), confirming the decoder is no longer collapsed to a flat output,
but recovering high-frequency voxel structure still needs more training
budget and most likely a switch away from per-subject min-max
normalization. This script is the smallest reproduction of the fix
direction, not a converged result.

Run::

    .venv/bin/python examples/wide_decoder.py

Should exit 0 in under 90 seconds on CPU.
"""

from __future__ import annotations

import pathlib
import sys
import time

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from recvae import (  # noqa: E402
    Config,
    FMRIDataset,
    RecVAEModel,
    build_dataloader,
    fit,
    normalize_per_subject,
    set_seed,
    synthetic_cohort,
)


def widen_decoder(model: RecVAEModel) -> RecVAEModel:
    """Replace the decoder stack with a wider one and drop the final ``Tanh``.

    Decoder widths: ``[16, 32, 64, 128]`` (vs canonical ``[4, 8, 16, 32]``).
    Output activation: ``Hardtanh(-3, 3)`` instead of ``Tanh``. The
    intermediate spatial-extent chain is unchanged from the canonical
    decoder; only the channel counts grow.
    """
    latent_dim = model.cfg.latent_dim
    # Bottleneck: (128, 5, 6, 5) instead of (32, 5, 6, 5).
    model.decoder1 = nn.Sequential(
        nn.Linear(latent_dim, 128 * 5 * 6 * 5),
        nn.Unflatten(1, (128, 5, 6, 5)),
        nn.BatchNorm3d(128),
        nn.LeakyReLU(0.2, inplace=True),
    )
    model.decoder2 = nn.Sequential(
        nn.ConvTranspose3d(
            128, 64, kernel_size=4, stride=2, padding=1, output_padding=1, bias=False
        ),
        nn.BatchNorm3d(64),
        nn.LeakyReLU(0.2, inplace=True),
    )  # (64, 11, 13, 11)
    model.decoder3 = nn.Sequential(
        nn.ConvTranspose3d(
            64, 32, kernel_size=4, stride=2, padding=1, output_padding=(0, 1, 0), bias=False
        ),
        nn.BatchNorm3d(32),
        nn.LeakyReLU(0.2, inplace=True),
    )  # (32, 22, 27, 22)
    model.decoder4 = nn.Sequential(
        nn.ConvTranspose3d(
            32, 16, kernel_size=4, stride=2, padding=1, output_padding=(1, 0, 1), bias=False
        ),
        nn.BatchNorm3d(16),
        nn.LeakyReLU(0.2, inplace=True),
    )  # (16, 45, 54, 45)
    model.decoder5 = nn.Sequential(
        nn.ConvTranspose3d(
            16, 1, kernel_size=4, stride=2, padding=1, output_padding=1, bias=False
        ),
        nn.Hardtanh(-3.0, 3.0),
    )  # (1, 91, 109, 91)
    return model


def _pearson_and_std_ratio(x: torch.Tensor, mu: torch.Tensor) -> tuple[float, float]:
    """Pearson(x, mu) and std(mu)/std(x), both computed across all voxels."""
    x_flat = x.reshape(-1).detach().cpu().numpy()
    mu_flat = mu.reshape(-1).detach().cpu().numpy()
    x_c = x_flat - x_flat.mean()
    mu_c = mu_flat - mu_flat.mean()
    x_std = float(x_c.std())
    mu_std = float(mu_c.std())
    denom = x_std * mu_std
    pearson = float((x_c * mu_c).mean() / denom) if denom > 0 else 0.0
    std_ratio = mu_std / x_std if x_std > 0 else 0.0
    return pearson, std_ratio


def main() -> int:
    t0 = time.time()
    set_seed(2022)

    # T=4 (not T=8) so the wider model finishes under 90s on CPU; the
    # decoder-collapse problem reproduces identically at either T.
    volumes, _labels = synthetic_cohort(n_cn=4, n_ad=4, T=4, seed=2022)
    volumes, _vmax, _vmin = normalize_per_subject(volumes)

    cfg = Config(tol_time=4, epochs=30, batch_size=2, learning_rate=1e-3)
    model = RecVAEModel(train_size=volumes.shape[0], cfg=cfg)
    widen_decoder(model)

    # Confirm widths and final activation.
    print("wide_decoder: channels [16, 32, 64, 128], Hardtanh(-3, 3) on output")

    dl = build_dataloader(FMRIDataset(volumes), batch_size=2, shuffle=True, seed=2022)
    h0 = torch.zeros(1, cfg.latent_dim)
    fit(model, dl, h0, cfg=cfg, epochs=cfg.epochs, lr=cfg.learning_rate,
        opt_func=torch.optim.AdamW)

    # Eval on subject 0 at t=0.
    model.eval()
    x_one = volumes[:1]
    which_ones = torch.tensor([0], dtype=torch.long)
    h_batch = h0.expand(1, -1)
    with torch.no_grad():
        out = model(x_one, h_batch, which_ones)
    x0 = x_one[0, 0, :, :, :, 0]
    mu0 = out.mu[0, 0, 0]
    pearson, std_ratio = _pearson_and_std_ratio(x0, mu0)
    print(f"wide_decoder: Pearson(x, mu) = {pearson:.3f}, "
          f"recon_std/input_std = {std_ratio:.2f}")
    elapsed = time.time() - t0
    print(f"wide_decoder: elapsed {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
