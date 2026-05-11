"""Amortize the subject offset z_s with a small 3D-conv encoder.

Variation from canonical
------------------------
- Canonical: ``z_vectors`` is an ``nn.Parameter`` of shape ``(N_train, D)``.
  Held-out subjects require an inner-loop SGD pass to recover their z_s
  (see ``recvae.evaluation.evaluate_held_out``).
- Amortized: a tiny ``Z_Enc`` net maps ``(B, 1, X, Y, Z, T) -> (B, D)`` in
  a single forward pass. New subjects do NOT require re-optimization at
  inference time; they go through ``Z_Enc`` once.
- Demo: 2 epochs of training; then we time the amortized inference for a
  new subject against ``evaluate_held_out`` with inner_steps=20. We only
  report timing — relative accuracy is not the point here.
"""

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from recvae import (  # noqa: E402
    Config,
    FMRIDataset,
    RecVAEModel,
    build_dataloader,
    evaluate_held_out,
    set_seed,
    synthetic_cohort,
)


class ZEnc(nn.Module):
    """Tiny conv encoder over the time-mean volume -> subject offset z_s."""

    def __init__(self, z_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=8, stride=8),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(4, z_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, X, Y, Z, T) -> time-mean -> (B, 1, X, Y, Z)
        x_mean = x.mean(dim=-1)
        return self.net(x_mean)


def rollout_with_z(model: RecVAEModel, x, h0, z_s):
    """Run the rollout using an externally supplied z_s rather than indexing."""
    h = h0.expand(x.size(0), -1)
    mu_list = []
    for t in range(model.tol_time):
        x_t = x[..., t]
        enc_x = model.encode(x_t)
        comb = torch.cat([enc_x, h], dim=1)
        mu_h = model.hidden2mu(comb)
        log_var_h = model.hidden2log_var(comb)
        h = model.reparametrize(mu_h, log_var_h)
        mu_list.append(model.decode(h + z_s))
    return torch.stack(mu_list, dim=1)


def main() -> int:
    set_seed(2022)
    vols, _ = synthetic_cohort(n_cn=2, n_ad=2, T=4, seed=2022)
    cfg = Config(tol_time=4, epochs=2, batch_size=2, learning_rate=1e-4)
    model = RecVAEModel(train_size=vols.shape[0], cfg=cfg)
    z_enc = ZEnc(z_dim=cfg.latent_dim)
    dl = build_dataloader(FMRIDataset(vols), batch_size=2, shuffle=True, seed=2022)
    h0 = torch.zeros(1, cfg.latent_dim)

    # Train both jointly. We bypass model.z_vectors entirely.
    params = [p for n, p in model.named_parameters() if n != "z_vectors"]
    opt = torch.optim.Adam(params + list(z_enc.parameters()), lr=cfg.learning_rate)

    for _ in range(cfg.epochs):
        model.train()
        z_enc.train()
        for batch, _ in dl:
            z_s = z_enc(batch)
            mu_stack = rollout_with_z(model, batch, h0, z_s)
            x_stack = batch.permute(0, 5, 1, 2, 3, 4).contiguous()
            loss = (x_stack - mu_stack).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    # --- timing comparison: amortized vs per-subject inner optimization ---
    new_vol, _ = synthetic_cohort(n_cn=1, n_ad=0, T=4, seed=9999)

    model.eval()
    z_enc.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        z_amort = z_enc(new_vol)
    amort_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    out = evaluate_held_out(model, new_vol, h0, inner_steps=20, inner_lr=1e-3)
    eval_time = time.perf_counter() - t0

    print(f"amortized z_s shape           : {tuple(z_amort.shape)}")
    print(f"amortized inference (1 fwd)   : {amort_time*1000:.2f} ms")
    print(f"evaluate_held_out (20 steps)  : {eval_time*1000:.2f} ms")
    print(f"speedup factor (rough)        : {eval_time / max(amort_time, 1e-9):.1f}x")
    print("note: accuracy is not claimed — the pattern is what matters. A "
          "fair comparison would tune both paths and use more subjects.")
    _ = out  # keep returned dict from being garbage-collected before print
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
