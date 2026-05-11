"""Compare default SGD@1e-6 vs AdamW@1e-4 + cosine annealing.

Variation from canonical
------------------------
- Run A: ``fit(..., opt_func=torch.optim.SGD, lr=1e-6)`` — the canonical
  default. The ``opt_func`` kwarg is the clean swap point.
- Run B: a small custom loop that builds AdamW + a CosineAnnealingLR
  scheduler externally. This is the pattern to reach for when you need a
  scheduler or non-trivial optimizer wiring, since fit() owns its own
  optimizer instance and does not expose a scheduler hook.
- Print both loss histories side by side for direct comparison.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from recvae import (  # noqa: E402
    Config,
    FMRIDataset,
    RecVAEModel,
    build_dataloader,
    fit,
    set_seed,
    synthetic_cohort,
)


def build_setup(epochs: int, lr: float):
    """Fresh model + loader + cfg + h0 with deterministic seeding."""
    set_seed(2022)
    vols, _ = synthetic_cohort(n_cn=2, n_ad=2, T=4, seed=2022)
    cfg = Config(tol_time=4, epochs=epochs, batch_size=2, learning_rate=lr)
    model = RecVAEModel(train_size=vols.shape[0], cfg=cfg)
    dl = build_dataloader(FMRIDataset(vols), batch_size=2, shuffle=True, seed=2022)
    h0 = torch.zeros(1, cfg.latent_dim)
    return model, dl, cfg, h0


def run_sgd(epochs: int):
    """Canonical path: hand the optimizer constructor to fit()."""
    model, dl, cfg, h0 = build_setup(epochs, lr=1e-6)
    result = fit(model, dl, h0, cfg=cfg, epochs=epochs, opt_func=torch.optim.SGD)
    return result["train_loss_history"]


def run_adamw_cosine(epochs: int):
    """Custom loop: AdamW + cosine annealing, with closed-form F still applied."""
    model, dl, cfg, h0 = build_setup(epochs, lr=1e-4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    h_buf = torch.zeros(model.train_size, cfg.tol_time, cfg.latent_dim)
    history = []
    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch, idx in dl:
            which = idx.long()
            h_batch = h0.expand(batch.size(0), -1)
            loss, _, h_history = model.training_step(batch, h_batch, which)
            opt.zero_grad()
            loss.backward()
            opt.step()
            h_buf[which] = torch.stack(h_history, dim=1).detach()
            epoch_loss += loss.item()
            n_batches += 1
        model.updating_F(h_buf, h0, cfg.rho)
        sched.step()
        history.append(epoch_loss / max(n_batches, 1))
    return history


def main() -> int:
    epochs = 5

    sgd_hist = run_sgd(epochs)
    adamw_hist = run_adamw_cosine(epochs)

    print(f"SGD @ 1e-6 loss history (len={len(sgd_hist)}):")
    for i, v in enumerate(sgd_hist):
        print(f"  epoch {i}: {v:.4f}")
    print(f"  delta(first -> last): {sgd_hist[-1] - sgd_hist[0]:+.4f}")

    print(f"AdamW @ 1e-4 + cosine loss history (len={len(adamw_hist)}):")
    for i, v in enumerate(adamw_hist):
        print(f"  epoch {i}: {v:.4f}")
    print(f"  delta(first -> last): {adamw_hist[-1] - adamw_hist[0]:+.4f}")

    # Rule of thumb: AdamW with a sane lr should drop the loss far more in
    # the same number of epochs than SGD@1e-6, which is intentionally tiny
    # in the canonical config.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
