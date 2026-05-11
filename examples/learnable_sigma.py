"""Make ``sig_x``, ``sig_h``, ``sig_z`` learnable instead of fixed scalars.

Variation from canonical
------------------------
- In the canonical model the three noise scales are fixed floats on Config.
- Here we subclass ``RecVAEModel`` and register raw learnable parameters,
  applying ``softplus`` so the effective sigmas stay strictly positive.
- We re-instantiate ``RecVAELoss`` per training step pointing at the
  current sigma values, then call training_step with ``loss_fn=``.
- Print the sigmas before and after a short training run.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from recvae import (  # noqa: E402
    Config,
    FMRIDataset,
    RecVAELoss,
    RecVAEModel,
    build_dataloader,
    set_seed,
    synthetic_cohort,
)


def _inv_softplus(y: float) -> float:
    """Inverse-softplus initializer so softplus(raw) ~= y at start."""
    y_t = torch.tensor(float(y))
    # log(exp(y) - 1) is numerically unstable for large y; use the expm1 form.
    return torch.log(torch.expm1(y_t)).item()


class LearnableSigmaRecVAE(RecVAEModel):
    """RecVAE with sig_x, sig_h, sig_z as softplus-reparameterized parameters."""

    def __init__(self, train_size: int, cfg: Config):
        super().__init__(train_size=train_size, cfg=cfg)
        self.raw_sig_x = nn.Parameter(torch.tensor(_inv_softplus(cfg.sig_x)))
        self.raw_sig_h = nn.Parameter(torch.tensor(_inv_softplus(cfg.sig_h)))
        self.raw_sig_z = nn.Parameter(torch.tensor(_inv_softplus(cfg.sig_z)))

    def sigmas(self):
        return (
            F.softplus(self.raw_sig_x),
            F.softplus(self.raw_sig_h),
            F.softplus(self.raw_sig_z),
        )


def main() -> int:
    set_seed(2022)
    vols, _ = synthetic_cohort(n_cn=2, n_ad=2, T=4, seed=2022)
    cfg = Config(tol_time=4, epochs=5, batch_size=2, learning_rate=1e-4)
    model = LearnableSigmaRecVAE(train_size=vols.shape[0], cfg=cfg)
    dl = build_dataloader(FMRIDataset(vols), batch_size=2, shuffle=True, seed=2022)
    h0 = torch.zeros(1, cfg.latent_dim)

    sx0, sh0, sz0 = (s.item() for s in model.sigmas())
    print(f"initial sig_x = {sx0:.4f}, sig_h = {sh0:.4f}, sig_z = {sz0:.4f}")

    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    h_buf = torch.zeros(model.train_size, cfg.tol_time, cfg.latent_dim)
    history = []
    for _ in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        n = 0
        for batch, idx in dl:
            which = idx.long()
            h_batch = h0.expand(batch.size(0), -1)
            sx, sh, _sz = model.sigmas()
            # Build a fresh loss with current sigma values; ``.item()`` is
            # avoided so gradients flow back to raw_sig_x / raw_sig_h.
            loss_fn = RecVAELoss(
                sig_x=sx, sig_h=sh, lambda_z=cfg.lambda_z, rho=cfg.rho,
            )
            loss, _, h_history = model.training_step(
                batch, h_batch, which, loss_fn=loss_fn,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            h_buf[which] = torch.stack(h_history, dim=1).detach()
            epoch_loss += loss.item()
            n += 1
        model.updating_F(h_buf, h0, cfg.rho)
        history.append(epoch_loss / max(n, 1))

    sxN, shN, szN = (s.item() for s in model.sigmas())
    print(f"final   sig_x = {sxN:.4f}, sig_h = {shN:.4f}, sig_z = {szN:.4f}")
    print(f"loss history: {[round(x, 3) for x in history]}")
    # Caveat: sig_z does not appear in the default RecVAELoss expression
    # (only in z_vectors initialization), so its parameter receives no
    # gradient from the loss above and should not move much.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
