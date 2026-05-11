"""Train with the variational KL temporal term vs the default MSE proxy.

Variation from canonical
------------------------
- The default ``RecVAELoss`` uses an MSE proxy for loss2 (between h_t and
  g(h_{t-1})). ``KLRecVAELoss`` swaps that for the closed-form KL between
  two diagonal Gaussians — the proper variational lower bound term.
- ``KLRecVAELoss`` needs the per-step posterior moments (mu_h, log_var_h),
  which ``model.forward`` does not expose. We use a small custom loop that
  manually steps the rollout and accumulates those tensors, then calls the
  loss callable directly.
- Two runs on the same seeded data, side-by-side loss history.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from recvae import (  # noqa: E402
    Config,
    FMRIDataset,
    KLRecVAELoss,
    RecVAELoss,
    RecVAEModel,
    build_dataloader,
    set_seed,
    synthetic_cohort,
)


def rollout_with_moments(model, x, h0, which):
    """Manual rollout that also captures mu_h and log_var_h at each step."""
    h = h0.expand(x.size(0), -1)
    x_list, mu_list, h_list, gh_list = [], [], [], []
    mu_h_list, lv_h_list = [], []
    for t in range(model.tol_time):
        x_t = x[..., t]
        x_list.append(x_t)
        gh_list.append(h @ model.F_mat.T)
        enc_x = model.encode(x_t)
        comb = torch.cat([enc_x, h], dim=1)
        mu_h = model.hidden2mu(comb)
        log_var_h = model.hidden2log_var(comb)
        mu_h_list.append(mu_h)
        lv_h_list.append(log_var_h)
        h = model.reparametrize(mu_h, log_var_h)
        h_list.append(h)
        mu_list.append(model.decode(h + model.z_vectors[which]))
    return x_list, mu_list, h_list, gh_list, mu_h_list, lv_h_list


def train_with_loss(loss_fn, epochs: int):
    """Custom loop: training_step is fine for MSE, but KL needs moments."""
    set_seed(2022)
    vols, _ = synthetic_cohort(n_cn=2, n_ad=2, T=4, seed=2022)
    cfg = Config(tol_time=4, epochs=epochs, batch_size=2, learning_rate=1e-5)
    model = RecVAEModel(train_size=vols.shape[0], cfg=cfg)
    dl = build_dataloader(FMRIDataset(vols), batch_size=2, shuffle=True, seed=2022)
    h0 = torch.zeros(1, cfg.latent_dim)
    opt = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate)
    history = []
    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        n = 0
        for batch, idx in dl:
            which = idx.long()
            xs, mus, hs, ghs, mu_hs, lv_hs = rollout_with_moments(model, batch, h0, which)
            if isinstance(loss_fn, KLRecVAELoss):
                loss, _ = loss_fn(
                    model, batch, hs, ghs, mus, xs,
                    mu_h_history=mu_hs, log_var_h_history=lv_hs,
                )
            else:
                loss, _ = loss_fn(model, batch, hs, ghs, mus, xs)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n += 1
        history.append(epoch_loss / max(n, 1))
    return history


def main() -> int:
    epochs = 3
    mse_hist = train_with_loss(RecVAELoss(), epochs)
    kl_hist = train_with_loss(KLRecVAELoss(), epochs)

    print(f"{'epoch':>6} | {'RecVAELoss (MSE)':>18} | {'KLRecVAELoss':>14}")
    print("-" * 46)
    for i in range(epochs):
        print(f"{i:>6} | {mse_hist[i]:>18.4f} | {kl_hist[i]:>14.4f}")
    print()
    print("Note: the absolute losses are not directly comparable — KL has a "
          "log term that MSE lacks — but both should decrease monotonically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
