"""Generate the three README visuals from a deterministic synthetic run.

Run from the repo root::

    .venv/bin/python tools/make_figures.py

Produces three PNGs in ``docs/assets/``:

- ``loss_curve.png``         per-epoch loss components over 5 epochs.
- ``recon_slice.png``        input vs. reconstruction mid-axial slice at t=0.
- ``latent_trajectory.png``  each latent dim over T=8 timesteps, batch-averaged.

The script is deterministic: it pins the global RNG via :func:`recvae.set_seed`
and uses ``seed=2022`` everywhere a draw is requested. Re-running regenerates
identical PNGs (modulo backend-specific floating-point quirks), so future
contributors can refresh the README artifacts with one command.

Implementation notes
--------------------
- The cohort size and ``tol_time`` are tiny (8 subjects, T=8) so the figure
  pipeline finishes in roughly a minute on CPU and the PNGs stay well under
  the 300 KB budget at ``dpi=100``.
- Per-epoch loss components are captured via a ``fit`` callback rather than
  by mutating the train loop, so the figure script is decoupled from
  internals of :func:`recvae.fit`.
- All three figures use ``bbox_inches='tight'`` and small ``figsize`` values
  to keep the rendered files small without resorting to JPEG.
"""

from __future__ import annotations

import pathlib
import sys

# Allow running from the repo root without installing the package.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

# Use a non-interactive backend so the script never tries to open a window
# (e.g. on CI runners and headless dev boxes).
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

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

ASSETS_DIR = REPO_ROOT / "docs" / "assets"
SEED = 2022
N_CN = 4
N_AD = 4
T = 8
EPOCHS = 5
LR = 1e-5


def _build_run():
    """Set up cohort, dataloader, model, h0 with deterministic seeding."""
    set_seed(SEED)

    volumes, _labels = synthetic_cohort(n_cn=N_CN, n_ad=N_AD, T=T, seed=SEED)
    volumes, _vmax, _vmin = normalize_per_subject(volumes)

    cfg = Config(tol_time=T, epochs=EPOCHS, batch_size=2, learning_rate=LR)
    dataset = FMRIDataset(volumes)
    loader = build_dataloader(dataset, batch_size=cfg.batch_size, shuffle=True, seed=SEED)

    train_size = N_CN + N_AD
    model = RecVAEModel(train_size=train_size, cfg=cfg)
    h0 = torch.zeros(1, cfg.latent_dim)
    return cfg, volumes, loader, model, h0


def _train_capturing_losses(model, loader, h0, cfg):
    """Run ``fit`` and record per-epoch loss components from the callback."""
    history = {"loss1": [], "loss2": [], "loss_z": []}

    def _cb(_epoch: int, metrics: dict) -> None:
        # The callback fires once per epoch with epoch-mean metrics; storing
        # the three components keeps the figure script independent of any
        # private fit() return shape.
        history["loss1"].append(float(metrics["loss1"]))
        history["loss2"].append(float(metrics["loss2"]))
        history["loss_z"].append(float(metrics["loss_z"]))

    fit(model, loader, h0, cfg=cfg, epochs=cfg.epochs, lr=cfg.learning_rate, callbacks=[_cb])
    return history


def _plot_loss_curve(history: dict, out_path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4), dpi=100)
    epochs = list(range(1, len(history["loss1"]) + 1))
    ax.plot(epochs, history["loss1"], marker="o", label="loss1 (recon MSE)")
    ax.plot(epochs, history["loss2"], marker="s", label="loss2 (temporal)")
    ax.plot(epochs, history["loss_z"], marker="^", label="loss_z (L1 on z)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss component (epoch mean)")
    ax.set_title("Training on synthetic cohort (8 subjects, 5 epochs)")
    ax.set_xticks(epochs)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _plot_recon_slice(volumes: torch.Tensor, model: RecVAEModel, h0: torch.Tensor, out_path: pathlib.Path) -> None:
    # Take the first subject and run a fresh rollout in eval mode so the
    # reconstruction reflects the post-training state with BatchNorm running
    # stats applied.
    model.eval()
    x_one = volumes[:1]  # (1, 1, X, Y, Z, T)
    which_ones = torch.tensor([0], dtype=torch.long)
    h_batch = h0.expand(1, -1)
    with torch.no_grad():
        out = model(x_one, h_batch, which_ones)

    # Mid-axial slice (Z midpoint) at t=0 for both input and reconstruction.
    z_mid = x_one.shape[4] // 2
    inp_slice = x_one[0, 0, :, :, z_mid, 0].cpu().numpy()
    rec_slice = out.mu[0, 0, 0, :, :, z_mid].cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=100)
    im0 = axes[0].imshow(inp_slice, cmap="gray")
    axes[0].set_title("input (t=0)")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(rec_slice, cmap="gray")
    axes[1].set_title("reconstruction (t=0)")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle("Input vs reconstruction (synthetic, untrained-quality, after 5 epochs)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_latent_trajectory(rollout, out_path: pathlib.Path) -> None:
    # out.h has shape (B, T, latent_dim); average over the batch dim so the
    # figure shows a single trajectory per latent coordinate.
    h_mean = rollout.h.mean(dim=0).detach().cpu().numpy()  # (T, latent_dim)
    timesteps = list(range(h_mean.shape[0]))

    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    for d in range(h_mean.shape[1]):
        ax.plot(timesteps, h_mean[:, d], marker="o", label=f"h[{d}]")
    ax.set_xlabel("timestep")
    ax.set_ylabel("latent value (mean over batch)")
    ax.set_title("Latent state trajectory across timesteps")
    ax.set_xticks(timesteps)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, ncol=1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    cfg, volumes, loader, model, h0 = _build_run()
    history = _train_capturing_losses(model, loader, h0, cfg)

    loss_path = ASSETS_DIR / "loss_curve.png"
    recon_path = ASSETS_DIR / "recon_slice.png"
    latent_path = ASSETS_DIR / "latent_trajectory.png"

    _plot_loss_curve(history, loss_path)
    # Reuse the rollout from the reconstruction figure for the latent plot
    # so both are guaranteed to reflect identical post-training state.
    rollout = _plot_recon_slice(volumes, model, h0, recon_path)
    _plot_latent_trajectory(rollout, latent_path)

    for path in (loss_path, recon_path, latent_path):
        size_kb = path.stat().st_size / 1024
        print(f"wrote {path.relative_to(REPO_ROOT)} ({size_kb:.1f} KB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
