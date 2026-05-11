"""Maintain two F matrices (one per cohort) and switch between them per subject.

Variation from canonical
------------------------
- The canonical model has a single transition matrix ``F_mat`` shared
  across all subjects. Here we keep ``F_cn`` and ``F_ad`` buffers and
  pick which one to apply based on a per-batch label tensor.
- We do NOT modify the encoder/decoder/inference parameters. The change
  is purely at the temporal-prior step where ``h_old`` is multiplied by
  the transition matrix.
- This is intended as a *pattern demonstrator* — only loss1 is used as
  the backward target so we don't need to also re-derive a per-cohort
  closed-form F update. Extending that is left as a reader exercise.
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
    set_seed,
    synthetic_cohort,
)


def cohort_rollout(model: RecVAEModel, x, h0, which, labels, F_cn, F_ad):
    """Mirror RecVAEModel.forward but pick F per-subject from labels."""
    B = x.size(0)
    h = h0.expand(B, -1)
    # Build a per-row F by indexing on labels: shape (B, D, D)
    F_per = torch.where(labels.view(B, 1, 1).bool(),
                        F_ad.unsqueeze(0).expand(B, -1, -1),
                        F_cn.unsqueeze(0).expand(B, -1, -1))
    mu_list, h_list, gh_list = [], [], []
    for t in range(model.tol_time):
        x_t = x[..., t]
        # g(h) for each row uses that row's F.
        gh = torch.bmm(h.unsqueeze(1), F_per.transpose(1, 2)).squeeze(1)
        gh_list.append(gh)
        enc_x = model.encode(x_t)
        comb = torch.cat([enc_x, h], dim=1)
        mu_h = model.hidden2mu(comb)
        log_var_h = model.hidden2log_var(comb)
        h = model.reparametrize(mu_h, log_var_h)
        h_tilde = h + model.z_vectors[which]
        mu_list.append(model.decode(h_tilde))
        h_list.append(h)
    return mu_list, h_list, gh_list


def main() -> int:
    set_seed(2022)
    vols, labels = synthetic_cohort(n_cn=2, n_ad=2, T=4, seed=2022)
    cfg = Config(tol_time=4, epochs=3, batch_size=2, learning_rate=1e-5)
    model = RecVAEModel(train_size=vols.shape[0], cfg=cfg)

    # Two F buffers; initialize from the random F that the model already has.
    F_cn = torch.rand(cfg.latent_dim, cfg.latent_dim) * 0.1
    F_ad = torch.rand(cfg.latent_dim, cfg.latent_dim) * 0.1
    # Register so device moves carry them along; they are not module
    # attributes here only because we want to keep the recvae package
    # untouched. In a fork you would ``model.register_buffer(...)`` these.

    ds = FMRIDataset(vols)
    dl = build_dataloader(ds, batch_size=2, shuffle=True, seed=2022)
    h0 = torch.zeros(1, cfg.latent_dim)
    opt = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate)
    history = []
    for _ in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        n = 0
        for batch, idx in dl:
            which = idx.long()
            batch_labels = labels[which]
            mu_list, h_list, gh_list = cohort_rollout(
                model, batch, h0, which, batch_labels, F_cn, F_ad,
            )
            # Reconstruction loss only (pattern demo, see module docstring).
            x_list = [batch[..., t] for t in range(model.tol_time)]
            denom = 2 * batch.size(0) * model.tol_time
            loss1 = sum((x - mu).pow(2).sum() for x, mu in zip(x_list, mu_list))
            loss1 = loss1 / (cfg.sig_x ** 2) / denom
            loss2 = sum((h - g).pow(2).sum() for h, g in zip(h_list, gh_list))
            loss2 = loss2 / (cfg.sig_h ** 2) / denom
            loss = loss1 + loss2
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n += 1
        history.append(epoch_loss / max(n, 1))

    print(f"loss history: {[round(x, 3) for x in history]}")
    print(f"F_cn norm: {F_cn.norm().item():.4f}  F_ad norm: {F_ad.norm().item():.4f}")
    print("note: gradient does not yet flow into F_cn / F_ad because they "
          "are detached tensors used only in the rollout; treat this as the "
          "structural pattern. To learn them, register as Parameters and "
          "include them in the optimizer's param list, or fork updating_F "
          "to solve a per-cohort closed form.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
